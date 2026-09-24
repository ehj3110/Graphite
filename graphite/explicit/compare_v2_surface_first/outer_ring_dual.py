"""Outer ring dual: faces between deleted (intersecting) hexes and the next layer.

Algorithm (user-specified)
-------------------------
1. Build an SC background grid over CAD bounds expanded by ``pad_cells``
   (default 2) times the cell size on every side.
2. Mark every hex that **intersects** the solid (any corner inside, or any
   hex edge/face that crosses the surface — practical test: not all eight
   corners strictly outside).
3. Delete that set ``D``.
4. Keep the **next layer** ``L``: hexes not in ``D`` that share a face with
   at least one hex in ``D``.
5. Emit only the faces of ``L`` that were shared with a deleted neighbour
   in ``D`` (complete quads — no per-edge chopping, no morph).

Those faces form a connected stair-step shell just outside the intersecting
voxelization of the part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from graphite.explicit.conformal_core import safe_signed_distance
from graphite.explicit.hex_rules import _HEX_FACES
from graphite.explicit.proven_topologies import generate_background_grid


@dataclass
class OuterRingReport:
    n_background_hex: int = 0
    n_deleted_intersecting: int = 0
    n_next_layer: int = 0
    n_interface_faces: int = 0
    n_nodes: int = 0
    n_struts: int = 0
    pad_cells: int = 2
    min_corner_sdf: float = 0.0
    n_corners_inside: int = 0


def _as_cell_xyz(
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    cs = np.asarray(cell_size, dtype=np.float64).ravel()
    if cs.size == 1:
        v = float(cs[0])
        return (v, v, v)
    if cs.size != 3:
        raise ValueError(f"cell_size must be scalar or length-3; got {cs}")
    return (float(cs[0]), float(cs[1]), float(cs[2]))


def _expanded_bounds(
    bounds: np.ndarray,
    cell_xyz: tuple[float, float, float],
    pad_cells: int,
) -> np.ndarray:
    b = np.asarray(bounds, dtype=np.float64).reshape(2, 3).copy()
    pad = np.asarray(cell_xyz, dtype=np.float64) * float(pad_cells)
    b[0] -= pad
    b[1] += pad
    return b


def _hex_intersects_solid(
    corners: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    outside_eps: float = 1e-3,
) -> bool:
    """True unless all eight corners are strictly outside the CAD."""
    sd = safe_signed_distance(cad_mesh, np.asarray(corners, dtype=np.float64))
    return bool(np.any(sd <= float(outside_eps)))


def _face_key(node_ids: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in node_ids))


def build_outer_ring_interface_faces(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    *,
    pad_cells: int = 2,
    outside_eps: float = 1e-3,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, OuterRingReport]:
    """
    Returns unique nodes and undirected struts for the outer-ring interface faces.
    """
    report = OuterRingReport(pad_cells=int(pad_cells))
    cell_xyz = _as_cell_xyz(cell_size)
    bounds = _expanded_bounds(cad_mesh.bounds, cell_xyz, int(pad_cells))

    grid_nodes, cells = generate_background_grid("SC", bounds, cell_xyz)
    report.n_background_hex = int(len(cells))

    # --- classify intersecting vs free ---
    intersecting: list[bool] = []
    for cell in cells:
        corners = grid_nodes[np.asarray(cell, dtype=np.int64)]
        intersecting.append(
            _hex_intersects_solid(corners, cad_mesh, outside_eps=outside_eps)
        )
    intersecting_arr = np.asarray(intersecting, dtype=bool)
    report.n_deleted_intersecting = int(np.count_nonzero(intersecting_arr))

    # --- face adjacency between hexes (shared 4-corner key) ---
    # Map sorted face corner key -> list of hex indices that own that face.
    face_to_hexes: dict[tuple[int, ...], list[int]] = {}
    hex_faces: list[list[tuple[int, ...]]] = []
    for hi, cell in enumerate(cells):
        corners = np.asarray(cell, dtype=np.int64)
        keys: list[tuple[int, ...]] = []
        for fv in _HEX_FACES:
            key = _face_key(corners[list(fv)])
            keys.append(key)
            face_to_hexes.setdefault(key, []).append(hi)
        hex_faces.append(keys)

    # Neighbours: hexes that share a face.
    neighbors: list[set[int]] = [set() for _ in range(len(cells))]
    for key, owners in face_to_hexes.items():
        if len(owners) != 2:
            continue
        a, b = int(owners[0]), int(owners[1])
        neighbors[a].add(b)
        neighbors[b].add(a)

    # --- next layer: free hexes adjacent to at least one deleted hex ---
    next_layer: list[int] = []
    for hi in range(len(cells)):
        if intersecting_arr[hi]:
            continue
        if any(intersecting_arr[n] for n in neighbors[hi]):
            next_layer.append(hi)
    report.n_next_layer = int(len(next_layer))
    if not next_layer:
        raise ValueError(
            "No next-layer hexes found; increase pad_cells or check CAD/cell size"
        )

    # --- interface faces: faces of next-layer hexes shared with a deleted hex ---
    interface_faces: list[np.ndarray] = []  # 4 grid-node indices each
    seen_face: set[tuple[int, ...]] = set()
    for hi in next_layer:
        cell = np.asarray(cells[hi], dtype=np.int64)
        for fv in _HEX_FACES:
            local = cell[list(fv)]
            key = _face_key(local)
            owners = face_to_hexes.get(key, [])
            # Must be shared with exactly one deleted neighbour.
            if len(owners) != 2:
                continue
            a, b = int(owners[0]), int(owners[1])
            other = b if a == hi else a
            if other == hi or not intersecting_arr[other]:
                continue
            if key in seen_face:
                continue
            seen_face.add(key)
            interface_faces.append(local.copy())

    report.n_interface_faces = int(len(interface_faces))
    if not interface_faces:
        raise ValueError("Next layer found but no interface faces with deleted hexes")

    # Build strut graph from complete face perimeters.
    edge_set: set[tuple[int, int]] = set()
    used_nodes: set[int] = set()
    for face in interface_faces:
        corners = [int(c) for c in face]
        used_nodes.update(corners)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if a != b:
                edge_set.add((min(a, b), max(a, b)))

    used = sorted(used_nodes)
    old_to_new = {old: new for new, old in enumerate(used)}
    nodes = np.asarray(grid_nodes[np.asarray(used, dtype=np.int64)], dtype=np.float64)
    # Round lightly for stable export identity.
    nodes = np.round(nodes, round_decimals)
    struts = np.asarray(
        [(old_to_new[a], old_to_new[b]) for a, b in sorted(edge_set)],
        dtype=np.int64,
    )
    report.n_nodes = int(len(nodes))
    report.n_struts = int(len(struts))

    sd = safe_signed_distance(cad_mesh, nodes)
    report.min_corner_sdf = float(sd.min()) if len(sd) else 0.0
    report.n_corners_inside = int(np.count_nonzero(sd <= float(outside_eps)))

    print(
        f"  Outer-ring dual: background={report.n_background_hex}, "
        f"deleted_intersecting={report.n_deleted_intersecting}, "
        f"next_layer={report.n_next_layer}, "
        f"interface_faces={report.n_interface_faces}, "
        f"nodes={report.n_nodes}, struts={report.n_struts}, "
        f"min_corner_sdf={report.min_corner_sdf:.3f}, "
        f"corners_inside={report.n_corners_inside}"
    )
    return nodes, struts, report
