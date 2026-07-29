"""
Hexahedral surface dual — boundary quad centroid graph.

Standalone module (not wired into ``hex_topology_module`` yet). Builds struts
between centroids of adjacent **boundary** quad faces (shared quad edge), analogous
to tet ``generate_surface_dual_cage`` on triangle skins.

Volume lattice rules (octahedral, grid, etc.) are separate: they define how interior
nodes connect; this module only closes the **skin** graph on the conformal hex scaffold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Local corner order matches ``hex_rules`` / ``hex_scaffold_module``.
HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)

_QUAD_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))


def hex_node_ids_from_elements(
    hex_elements: np.ndarray,
    round_decimals: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Collapse ``(N, 8, 3)`` hex coordinates to unique nodes + global corner IDs.

  Returns:
      unique_nodes: (V, 3)
      hex_node_ids: (N, 8) int32
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must be (N, 8, 3); got {elems.shape}.")
    flat = elems.reshape(-1, 3)
    keys = np.round(flat, round_decimals)
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    return unique_keys.astype(np.float64), inv.reshape(-1, 8).astype(np.int32)


def extract_ordered_boundary_quads(
    hex_node_ids: np.ndarray,
    *,
    hex_faces: tuple[tuple[int, int, int, int], ...] = HEX_FACES,
) -> np.ndarray:
    """
    Boundary quads as ordered global node IDs, shape ``(F, 4)``.

    A face owned by exactly one hex is on the exterior; shared faces are interior.
    """
    ids = np.asarray(hex_node_ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[1] != 8:
        raise ValueError(f"hex_node_ids must be (N, 8); got {ids.shape}.")

    face_records: dict[tuple[int, int, int, int], tuple[int, int] | None] = {}
    for hi, elem in enumerate(ids):
        for fi, face in enumerate(hex_faces):
            ordered = tuple(int(elem[i]) for i in face)
            key = tuple(sorted(ordered))
            if key not in face_records:
                face_records[key] = (int(hi), int(fi))
            else:
                face_records[key] = None

    out: list[np.ndarray] = []
    for rec in face_records.values():
        if rec is None:
            continue
        hi, fi = rec
        out.append(ids[hi][list(hex_faces[fi])].astype(np.int32))
    if not out:
        return np.empty((0, 4), dtype=np.int32)
    return np.vstack(out).astype(np.int32)


@dataclass(frozen=True)
class BoundaryQuadTopology:
    """Persistent mapping from boundary quad index to owning hex and local face."""

    quads: np.ndarray
    hex_owner: np.ndarray
    face_index: np.ndarray

    @property
    def n_quads(self) -> int:
        return int(self.quads.shape[0])

    def quad_to_hex_face(self, quad_id: int) -> tuple[int, int]:
        return int(self.hex_owner[quad_id]), int(self.face_index[quad_id])


def extract_ordered_boundary_quads_with_owners(
    hex_node_ids: np.ndarray,
    *,
    hex_faces: tuple[tuple[int, int, int, int], ...] = HEX_FACES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boundary quads and owning hex / local face index per quad.

    Returns:
        boundary_quads: (F, 4) ordered global corner IDs
        quad_hex_owner: (F,) hex index in ``hex_node_ids``
        quad_hex_face: (F,) local face index 0..5 in ``HEX_FACES``
    """
    ids = np.asarray(hex_node_ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[1] != 8:
        raise ValueError(f"hex_node_ids must be (N, 8); got {ids.shape}.")

    face_records: dict[tuple[int, int, int, int], tuple[int, int] | None] = {}
    for hi, elem in enumerate(ids):
        for fi, face in enumerate(hex_faces):
            ordered = tuple(int(elem[i]) for i in face)
            key = tuple(sorted(ordered))
            if key not in face_records:
                face_records[key] = (int(hi), int(fi))
            else:
                face_records[key] = None

    quads_out: list[np.ndarray] = []
    owners_out: list[int] = []
    faces_out: list[int] = []
    for rec in face_records.values():
        if rec is None:
            continue
        hi, fi = rec
        quads_out.append(ids[hi][list(hex_faces[fi])].astype(np.int32))
        owners_out.append(int(hi))
        faces_out.append(int(fi))

    if not quads_out:
        empty_i = np.empty((0,), dtype=np.int32)
        return np.empty((0, 4), dtype=np.int32), empty_i, empty_i
    return (
        np.vstack(quads_out).astype(np.int32),
        np.asarray(owners_out, dtype=np.int32),
        np.asarray(faces_out, dtype=np.int32),
    )


def build_boundary_quad_topology(
    hex_node_ids: np.ndarray,
    *,
    hex_faces: tuple[tuple[int, int, int, int], ...] = HEX_FACES,
) -> BoundaryQuadTopology:
    quads, owners, face_idx = extract_ordered_boundary_quads_with_owners(
        hex_node_ids, hex_faces=hex_faces
    )
    return BoundaryQuadTopology(quads=quads, hex_owner=owners, face_index=face_idx)


def boundary_quad_centroids(
    nodes: np.ndarray,
    boundary_quads: np.ndarray,
) -> np.ndarray:
    """Centroid coordinates for each boundary quad, shape ``(F, 3)``."""
    pts = np.asarray(nodes, dtype=np.float64)
    quads = np.asarray(boundary_quads, dtype=np.int64)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError(f"boundary_quads must be (F, 4); got {quads.shape}.")
    return np.mean(pts[quads], axis=1)


def get_boundary_quad_adjacency(
    boundary_quads: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pairs of boundary-quad indices that share a mesh edge (two vertex IDs).

    Returns:
        pairs: (E, 2) with ``pairs[k, 0] < pairs[k, 1]``
        shared_edges: (E, 2) canonical vertex IDs for each pair
    """
    quads = np.asarray(boundary_quads, dtype=np.int64)
    if quads.size == 0:
        empty = np.empty((0, 2), dtype=np.int64)
        return empty, empty

    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(quads.shape[0]):
        q = quads[fi]
        for a, b in _QUAD_EDGE_PAIRS:
            va, vb = int(q[a]), int(q[b])
            edge = (min(va, vb), max(va, vb))
            edge_to_faces[edge].append(int(fi))

    pairs_list: list[tuple[int, int]] = []
    edges_list: list[tuple[int, int]] = []
    for edge, faces in edge_to_faces.items():
        if len(faces) != 2:
            continue
        i, j = min(faces), max(faces)
        pairs_list.append((i, j))
        edges_list.append(edge)

    if not pairs_list:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    return (
        np.asarray(pairs_list, dtype=np.int64),
        np.asarray(edges_list, dtype=np.int64),
    )


def boundary_vertex_quad_incidence(
    boundary_quads: np.ndarray,
) -> dict[int, list[int]]:
    """Map global corner vertex ID -> list of incident boundary quad indices."""
    quads = np.asarray(boundary_quads, dtype=np.int64)
    out: dict[int, list[int]] = defaultdict(list)
    for fi in range(quads.shape[0]):
        for vid in quads[fi]:
            out[int(vid)].append(int(fi))
    return out


def _quad_unit_normals(nodes: np.ndarray, boundary_quads: np.ndarray) -> np.ndarray:
    """Unit normal per boundary quad from corner ordering (F, 3)."""
    corners = nodes[boundary_quads]  # (F, 4, 3)
    p0, p1, p2, p3 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    n = np.cross(p1 - p0, p3 - p0) + np.cross(p3 - p0, p2 - p1)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norms, 1e-12)


def generate_hex_surface_dual_cage(
    nodes: np.ndarray,
    boundary_quads: np.ndarray,
    *,
    target_element_size: float | None = None,
    coplanar_cos_threshold: float = 0.99,
    include_isolated_closure: bool = True,
    include_corner_closure: bool = True,
    corner_valence_min: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Surface dual on a conformal hex skin: struts between centroids of adjacent
    boundary quads (quad-edge neighbors).

    Args:
        nodes: Global mesh nodes ``(V, 3)``.
        boundary_quads: ``(F, 4)`` global corner indices per exterior quad.
        target_element_size: If set, skip struts longer than ``1.5 * size``.
        coplanar_cos_threshold: For curved skins, only link quads with
            ``|dot(n_i, n_j)| >= threshold``. Set ``0`` to disable (production
            tet ``generate_surface_dual_cage`` does not use this filter).
        include_isolated_closure: Tet-style: quads with no dual neighbor get a
            strut from centroid to one corner vertex (see tet
            ``_surface_dual_face_centroid_struts``).
        include_corner_closure: At boundary vertices where ``>= corner_valence_min``
            quads meet (cube corners), add strut from each incident quad centroid
            to that corner vertex.
        corner_valence_min: Minimum incident boundary quads to treat as a corner.

    Returns:
        cage_nodes: ``(F + C, 3)`` — quad centroids then optional corner vertices.
        struts: ``(S, 2)`` indices into ``cage_nodes``.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    quads = np.asarray(boundary_quads, dtype=np.int64)
    if quads.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    centroids = boundary_quad_centroids(pts, quads)
    pairs, _ = get_boundary_quad_adjacency(quads)
    normals = _quad_unit_normals(pts, quads)

    max_dist = None
    if target_element_size is not None:
        max_dist = 1.5 * float(target_element_size)

    strut_set: set[tuple[int, int]] = set()
    degree = np.zeros(quads.shape[0], dtype=np.int64)

    for i, j in pairs:
        if coplanar_cos_threshold > 0.0:
            dot = float(np.abs(np.dot(normals[i], normals[j])))
            if dot < coplanar_cos_threshold:
                continue
        if max_dist is not None:
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d > max_dist:
                continue
        a, b = int(min(i, j)), int(max(i, j))
        strut_set.add((a, b))
        degree[a] += 1
        degree[b] += 1

    corner_coords: list[np.ndarray] = []
    corner_vid_to_cage: dict[int, int] = {}
    n_centroids = int(centroids.shape[0])

    def _corner_cage_index(vid: int) -> int:
        if vid not in corner_vid_to_cage:
            corner_vid_to_cage[vid] = n_centroids + len(corner_coords)
            corner_coords.append(pts[int(vid)].copy())
        return corner_vid_to_cage[vid]

    if include_isolated_closure:
        for fi in np.where(degree == 0)[0]:
            vid = int(quads[int(fi), 0])
            cn = _corner_cage_index(vid)
            strut_set.add((int(min(fi, cn)), int(max(fi, cn))))

    if include_corner_closure and corner_valence_min >= 2:
        vtx_inc = boundary_vertex_quad_incidence(quads)
        for vid, qlist in vtx_inc.items():
            if len(qlist) < int(corner_valence_min):
                continue
            cn = _corner_cage_index(int(vid))
            for fi in qlist:
                a, b = int(min(fi, cn)), int(max(fi, cn))
                strut_set.add((a, b))

    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    if corner_coords:
        cage_nodes = np.vstack((centroids, np.vstack(corner_coords)))
    else:
        cage_nodes = centroids
    return cage_nodes, struts


def _coplanar_filter_applies(
    owner_i: int,
    owner_j: int,
    hex_has_internal_lattice: np.ndarray,
) -> bool:
    """Coplanar gate is for core skin links; sliver patches meet neighbors at corners/edges."""
    return bool(hex_has_internal_lattice[int(owner_i)]) and bool(
        hex_has_internal_lattice[int(owner_j)]
    )


def generate_hex_surface_dual_cage_volume_gated(
    nodes: np.ndarray,
    boundary_quads: np.ndarray,
    quad_hex_owner: np.ndarray,
    hex_has_internal_lattice: np.ndarray,
    *,
    target_element_size: float | None = None,
    coplanar_cos_threshold: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Surface dual with per-hex volume-fraction gating.

    All boundary quads use shared-edge centroid links (same as standard surface
    dual). Surface-only hexes (no internal lattice) also get a local cage linking
    every exterior face centroid on that hex so corner/edge slivers stay closed
  when coplanar or topology would otherwise leave a gap.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    quads = np.asarray(boundary_quads, dtype=np.int64)
    owners = np.asarray(quad_hex_owner, dtype=np.int32).ravel()
    has_internal = np.asarray(hex_has_internal_lattice, dtype=bool).ravel()
    n_hex = has_internal.shape[0]
    if quads.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {
                "n_boundary_quads": 0,
                "n_adjacent_struts": 0,
                "n_element_cage_struts": 0,
                "n_hexes_without_internal": 0,
            },
        )
    if owners.shape[0] != quads.shape[0]:
        raise ValueError("quad_hex_owner length must match boundary_quads rows.")
    if owners.max(initial=0) >= n_hex or owners.min(initial=0) < 0:
        raise ValueError("quad_hex_owner indices out of range for hex_has_internal_lattice.")

    centroids = boundary_quad_centroids(pts, quads)
    pairs, _ = get_boundary_quad_adjacency(quads)
    normals = _quad_unit_normals(pts, quads)

    max_dist = None
    if target_element_size is not None:
        max_dist = 1.5 * float(target_element_size)

    strut_set: set[tuple[int, int]] = set()
    degree = np.zeros(quads.shape[0], dtype=np.int64)
    n_adjacent = 0
    n_element_cage = 0

    for i, j in pairs:
        if coplanar_cos_threshold > 0.0 and _coplanar_filter_applies(
            int(owners[i]), int(owners[j]), has_internal
        ):
            dot = float(np.abs(np.dot(normals[i], normals[j])))
            if dot < coplanar_cos_threshold:
                continue
        if max_dist is not None:
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d > max_dist:
                continue
        a, b = int(min(i, j)), int(max(i, j))
        strut_set.add((a, b))
        degree[a] += 1
        degree[b] += 1
        n_adjacent += 1

    hex_to_faces: dict[int, list[int]] = defaultdict(list)
    for fi, hi in enumerate(owners.tolist()):
        if not has_internal[int(hi)]:
            hex_to_faces[int(hi)].append(int(fi))

    for face_ids in hex_to_faces.values():
        if len(face_ids) < 2:
            continue
        for a in range(len(face_ids)):
            for b in range(a + 1, len(face_ids)):
                i, j = face_ids[a], face_ids[b]
                if max_dist is not None:
                    d = float(np.linalg.norm(centroids[i] - centroids[j]))
                    if d > max_dist:
                        continue
                sa, sb = int(min(i, j)), int(max(i, j))
                if (sa, sb) not in strut_set:
                    n_element_cage += 1
                strut_set.add((sa, sb))
                degree[sa] += 1
                degree[sb] += 1

    strut_set = {s for s in strut_set if s[0] != s[1]}

    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    report = {
        "n_boundary_quads": int(quads.shape[0]),
        "n_adjacent_struts": int(n_adjacent),
        "n_element_cage_struts": int(n_element_cage),
        "n_hexes_without_internal": int(np.sum(~has_internal)),
        "n_hexes_with_internal": int(np.sum(has_internal)),
    }
    return centroids, struts, report


def generate_hex_surface_dual_on_surface_paths(
    nodes: np.ndarray,
    boundary_quads: np.ndarray,
    quad_hex_owner: np.ndarray | None = None,
    hex_has_internal_lattice: np.ndarray | None = None,
    *,
    round_decimals: int = 6,
    include_sliver_element_cage: bool = True,
    max_centroid_span_factor: float | None = None,
    target_element_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Surface dual on the hex skin mesh without normal-angle gates.

    For each pair of boundary quads sharing a mesh edge, build a path on the
    surface: centroid_i -> midpoint(shared edge) -> centroid_j. Optional
    element cages on surface-only hexes (VF-gated slivers) link all exterior
    face centroids on that cell.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    quads = np.asarray(boundary_quads, dtype=np.int64)
    n_faces = int(quads.shape[0])
    if n_faces == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {"n_boundary_quads": 0, "n_path_struts": 0, "n_element_cage_struts": 0},
        )

    centroids = boundary_quad_centroids(pts, quads)
    pairs, shared_edges = get_boundary_quad_adjacency(quads)

    max_span = None
    if max_centroid_span_factor is not None and target_element_size is not None:
        max_span = float(max_centroid_span_factor) * float(target_element_size)
    elif target_element_size is not None:
        max_span = 1.5 * float(target_element_size)

    nodes_list = [centroids[i].copy() for i in range(n_faces)]
    strut_set: set[tuple[int, int]] = set()
    edge_mid_index: dict[tuple[int, int], int] = {}
    n_path = 0

    def _mid_index(va: int, vb: int) -> int:
        key = (min(va, vb), max(va, vb))
        idx = edge_mid_index.get(key)
        if idx is None:
            m = 0.5 * (pts[va] + pts[vb])
            idx = len(nodes_list)
            nodes_list.append(m)
            edge_mid_index[key] = idx
        return idx

    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if max_span is not None:
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d > max_span:
                continue
        va, vb = int(shared_edges[k, 0]), int(shared_edges[k, 1])
        mi = _mid_index(va, vb)
        a, b = int(min(i, mi)), int(max(i, mi))
        c, d = int(min(mi, j)), int(max(mi, j))
        strut_set.add((a, b))
        strut_set.add((c, d))
        n_path += 2

    n_cage = 0
    if (
        include_sliver_element_cage
        and quad_hex_owner is not None
        and hex_has_internal_lattice is not None
    ):
        owners = np.asarray(quad_hex_owner, dtype=np.int32).ravel()
        has_internal = np.asarray(hex_has_internal_lattice, dtype=bool).ravel()
        hex_to_faces: dict[int, list[int]] = defaultdict(list)
        for fi, hi in enumerate(owners.tolist()):
            if not has_internal[int(hi)]:
                hex_to_faces[int(hi)].append(int(fi))
        for face_ids in hex_to_faces.values():
            if len(face_ids) < 2:
                continue
            for a in range(len(face_ids)):
                for b in range(a + 1, len(face_ids)):
                    i, j = face_ids[a], face_ids[b]
                    sa, sb = int(min(i, j)), int(max(i, j))
                    if (sa, sb) not in strut_set:
                        n_cage += 1
                    strut_set.add((sa, sb))

    cage_nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    report = {
        "n_boundary_quads": n_faces,
        "n_path_struts": int(n_path),
        "n_element_cage_struts": int(n_cage),
        "n_skin_nodes": int(len(cage_nodes)),
        "n_skin_struts": int(len(struts)),
        "surface_dual_mode": "on_surface_paths",
    }
    return cage_nodes, struts, report


def boundary_quad_wire_segments(
    nodes: np.ndarray,
    boundary_quads: np.ndarray,
) -> np.ndarray:
    """
    Line segments for boundary quad edges only, shape ``(S, 2, 3)``.

    For visualization (Vedo ``Lines``).
    """
    pts = np.asarray(nodes, dtype=np.float64)
    quads = np.asarray(boundary_quads, dtype=np.int64)
    segs: list[np.ndarray] = []
    for q in quads:
        loop = [int(q[i]) for i in range(4)] + [int(q[0])]
        for k in range(4):
            segs.append(np.stack((pts[loop[k]], pts[loop[k + 1]]), axis=0))
    if not segs:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.asarray(segs, dtype=np.float64)


def surface_dual_wire_segments(
    centroid_nodes: np.ndarray,
    struts: np.ndarray,
) -> np.ndarray:
    """Segments for surface-dual struts, shape ``(S, 2, 3)``."""
    c = np.asarray(centroid_nodes, dtype=np.float64)
    s = np.asarray(struts, dtype=np.int64)
    if s.size == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack((c[s[:, 0]], c[s[:, 1]]), axis=1)
