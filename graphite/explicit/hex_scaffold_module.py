"""
Graphite Explicit Engine - Hexahedral Scaffold Generation

This module is responsible for generating regular Cartesian hexahedral (8-node) 
grids and conformal hexahedral scaffolds from bounding meshes. It interfaces 
with Gmsh to generate high-quality unstructured block meshes and provides 
utilities for snapping, cropping, and managing the boundary connectivity graph.
"""
from __future__ import annotations

import os
import tempfile
from collections import defaultdict

import gmsh
import numpy as np
import trimesh

from .boundary_policy import (
    apply_gated_node_targets,
    apply_laplacian_smoothing,
    apply_tiered_boundary_policy,
    build_edt_sdf_sampler,
    calculate_hex_volume_fractions,
    closest_points_with_fallback,
    closest_points_with_normals,
    compress_graph_to_kept_struts,
)
from .hex_surface_dual import (
    BoundaryQuadTopology,
    build_boundary_quad_topology,
    extract_ordered_boundary_quads_with_owners,
    generate_hex_surface_dual_on_surface_paths,
    get_boundary_quad_adjacency,
    hex_node_ids_from_elements,
)
from .hex_topology_module import (
    filter_struts_drop_exterior_shell_pairs,
    generate_hex_octahedral_volume_with_boundary_face_map,
    generate_hex_topology,
)
from .scaffold_module import _gmsh_discrete_surface_to_cad


def _extract_gmsh_hex_elements() -> np.ndarray:
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if node_tags.size == 0:
        return np.empty((0, 8, 3), dtype=np.float64)
    coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
    tag_to_xyz = {int(tag): coords[i] for i, tag in enumerate(np.asarray(node_tags, dtype=np.int64))}

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(3)
    out: list[np.ndarray] = []
    for et, en in zip(elem_types, elem_nodes):
        if int(et) != 5:
            continue
        en = np.asarray(en, dtype=np.int64)
        if en.size == 0:
            continue
        mat = en.reshape(-1, 8)
        for row in mat:
            out.append(np.array([tag_to_xyz[int(t)] for t in row], dtype=np.float64))
    return np.array(out, dtype=np.float64) if out else np.empty((0, 8, 3), dtype=np.float64)


def generate_conformal_hex_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    algorithm_3d: int = 1,
) -> np.ndarray:
    """
    Generate a boundary-conforming hexahedral mesh using Gmsh.

    This function exports the target mesh to a temporary STL, classifies the 
    surface, and instructs Gmsh to generate an unstructured, recombined 
    hexahedral volume mesh using the specified 3D algorithm.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The target watertight boundary mesh.
    target_element_size : float
        Target edge length for the hexahedral elements.
    algorithm_3d : int, optional
        Gmsh 3D algorithm to use, by default 1.

    Returns
    -------
    ndarray
        (N, 8, 3) array of coordinates for N hexahedral elements.

    Raises
    ------
    TypeError
        If the mesh is not a trimesh.Trimesh object.
    ValueError
        If target_element_size is <= 0.
    RuntimeError
        If Gmsh fails to generate type-5 hexahedral elements.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"`mesh` must be trimesh.Trimesh, got {type(mesh)}")
    if target_element_size <= 0:
        raise ValueError("`target_element_size` must be > 0.")

    temp_stl_path = None
    gmsh_initialized = False
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            temp_stl_path = tmp.name
        mesh.export(temp_stl_path)

        gmsh.initialize()
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(target_element_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_element_size))
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm_3d))
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 1)

        gmsh.merge(temp_stl_path)
        _gmsh_discrete_surface_to_cad()

        surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surface_tags:
            raise RuntimeError("No surface entities found after STL classification.")
        gmsh.model.geo.addSurfaceLoop(surface_tags, 1)
        gmsh.model.geo.addVolume([1], 1)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        hexes = _extract_gmsh_hex_elements()
        if len(hexes) == 0:
            raise RuntimeError(
                "Gmsh did not produce type-5 hexahedral elements from this conformal surface."
            )
        return hexes
    finally:
        if gmsh_initialized:
            gmsh.finalize()
        if temp_stl_path is not None and os.path.exists(temp_stl_path):
            os.remove(temp_stl_path)


def _mesh_grid_anchor(mesh: trimesh.Trimesh, grid_anchor: str | np.ndarray | list | tuple) -> np.ndarray:
    """
    Origin for symmetric Cartesian hex grids.

    ``bbox_min`` is not symmetric on centered parts; prefer ``bbox_center`` or
    ``center_mass`` for organic / centered STLs.
    """
    if isinstance(grid_anchor, (list, tuple, np.ndarray)):
        return np.asarray(grid_anchor, dtype=np.float64)
    key = str(grid_anchor).strip().lower()
    if key in ("bbox_min", "min", "corner"):
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        return bounds[0].copy()
    if key in ("bbox_center", "center", "centroid_bbox"):
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        return 0.5 * (bounds[0] + bounds[1])
    if key in ("centroid", "mesh_centroid"):
        return np.asarray(mesh.centroid, dtype=np.float64)
    if key in ("center_mass", "com", "center_of_mass"):
        try:
            return np.asarray(mesh.center_mass, dtype=np.float64)
        except Exception:
            return np.asarray(mesh.centroid, dtype=np.float64)
    raise ValueError(
        f"Unknown grid_anchor {grid_anchor!r}. "
        "Use bbox_min, bbox_center, centroid, or center_mass."
    )


def _symmetric_axis_knots(lo: float, hi: float, step: float, center: float) -> np.ndarray:
    """Knots along one axis: ``n_cells = ceil(extent/step)``, grid centered on ``center``."""
    extent = float(hi) - float(lo)
    if extent <= 0.0:
        return np.array([center], dtype=np.float64)
    n_cells = max(1, int(np.ceil(extent / step)))
    half_span = 0.5 * n_cells * float(step)
    return np.linspace(center - half_span, center + half_span, n_cells + 1)


def generate_cropped_hex_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    *,
    grid_anchor: str = "bbox_center",
) -> np.ndarray:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"`mesh` must be trimesh.Trimesh, got {type(mesh)}")
    if target_element_size <= 0:
        raise ValueError("`target_element_size` must be > 0.")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    step = float(target_element_size)
    anchor = _mesh_grid_anchor(mesh, grid_anchor)
    xs = _symmetric_axis_knots(lo[0], hi[0], step, anchor[0])
    ys = _symmetric_axis_knots(lo[1], hi[1], step, anchor[1])
    zs = _symmetric_axis_knots(lo[2], hi[2], step, anchor[2])
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        return np.empty((0, 8, 3), dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    ny, nz = len(ys), len(zs)

    def idx(i: int, j: int, k: int) -> int:
        return i * (ny * nz) + j * nz + k

    corners = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    out: list[np.ndarray] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                ids = [idx(i + di, j + dj, k + dk) for di, dj, dk in corners]
                cell = points[ids]
                c = cell.mean(axis=0)
                if bool(mesh.contains(c.reshape(1, 3))[0]):
                    out.append(cell)
    return np.array(out, dtype=np.float64) if out else np.empty((0, 8, 3), dtype=np.float64)


def _generate_bbox_hex_grid(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    *,
    grid_anchor: str = "bbox_center",
) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    step = float(target_element_size)
    anchor = _mesh_grid_anchor(mesh, grid_anchor)
    xs = _symmetric_axis_knots(lo[0], hi[0], step, anchor[0])
    ys = _symmetric_axis_knots(lo[1], hi[1], step, anchor[1])
    zs = _symmetric_axis_knots(lo[2], hi[2], step, anchor[2])
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        return np.empty((0, 8, 3), dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    ny, nz = len(ys), len(zs)

    def idx(i: int, j: int, k: int) -> int:
        return i * (ny * nz) + j * nz + k

    corners = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]
    out: list[np.ndarray] = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                ids = [idx(i + di, j + dj, k + dk) for di, dj, dk in corners]
                out.append(points[ids])
    return np.array(out, dtype=np.float64) if out else np.empty((0, 8, 3), dtype=np.float64)


_HEX_EDGES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)
_HEX_FACES = np.array(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
    ],
    dtype=np.int32,
)
_CORNER_NEIGHBORS = {
    0: (1, 3, 4),
    1: (0, 2, 5),
    2: (1, 3, 6),
    3: (0, 2, 7),
    4: (0, 5, 7),
    5: (1, 4, 6),
    6: (2, 5, 7),
    7: (3, 4, 6),
}


def _build_unique_node_representation(
    hex_elements: np.ndarray, round_decimals: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(hex_elements, dtype=np.float64).reshape(-1, 3)
    keys = np.round(flat, round_decimals)
    unique_keys, inv = np.unique(keys, axis=0, return_inverse=True)
    return unique_keys.astype(np.float64), inv.reshape(-1, 8).astype(np.int32)


def _compute_boundary_node_mask(hex_node_ids: np.ndarray, n_nodes: int) -> np.ndarray:
    face_count: dict[tuple[int, int, int, int], int] = defaultdict(int)
    face_owner: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for hi, elem in enumerate(hex_node_ids):
        for fi, face in enumerate(_HEX_FACES):
            gids = tuple(sorted(int(elem[i]) for i in face))
            face_count[gids] += 1
            face_owner[gids] = (hi, fi)
    mask = np.zeros(n_nodes, dtype=bool)
    for key, cnt in face_count.items():
        if cnt == 1:
            for gid in key:
                mask[int(gid)] = True
    return mask


def _hex_cell_adjacency(hex_node_ids: np.ndarray) -> list[list[int]]:
    """Undirected adjacency between hex cells that share a face."""
    n_hex = len(hex_node_ids)
    face_to_hex: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for hi, elem in enumerate(hex_node_ids):
        for face in _HEX_FACES:
            key = tuple(sorted(int(elem[i]) for i in face))
            face_to_hex[key].append(int(hi))
    adj: list[list[int]] = [[] for _ in range(n_hex)]
    for owners in face_to_hex.values():
        if len(owners) != 2:
            continue
        a, b = int(owners[0]), int(owners[1])
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _hex_layer_distance_from_skin(hex_node_ids: np.ndarray) -> np.ndarray:
    """
    BFS layer index from any hex with an exterior boundary face (dist 0 = skin hex).
    """
    from collections import deque

    n_hex = len(hex_node_ids)
    face_count: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for elem in hex_node_ids:
        for face in _HEX_FACES:
            key = tuple(sorted(int(elem[i]) for i in face))
            face_count[key] += 1

    is_skin = np.zeros(n_hex, dtype=bool)
    for hi, elem in enumerate(hex_node_ids):
        for face in _HEX_FACES:
            key = tuple(sorted(int(elem[i]) for i in face))
            if face_count[key] == 1:
                is_skin[hi] = True
                break

    dist = np.full(n_hex, -1, dtype=np.int32)
    q: deque[int] = deque()
    for hi in range(n_hex):
        if is_skin[hi]:
            dist[hi] = 0
            q.append(hi)
    adj = _hex_cell_adjacency(hex_node_ids)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _laplacian_frozen_mask_layer_limited(
    n_nodes: int,
    hex_node_ids: np.ndarray,
    boundary_node_mask: np.ndarray,
    hex_layer_dist: np.ndarray,
    *,
    smooth_layers_inward: int,
    freeze_boundary_nodes: bool = True,
) -> np.ndarray:
    """
    Freeze nodes not in the first ``smooth_layers_inward`` layers inward from skin
    hexes (layer 1 .. N). Boundary nodes are frozen only when ``freeze_boundary_nodes``.
    """
    layers = int(smooth_layers_inward)
    if layers <= 0:
        if freeze_boundary_nodes:
            return np.asarray(boundary_node_mask, dtype=bool).copy()
        return np.zeros(n_nodes, dtype=bool)

    node_min_layer = np.full(n_nodes, np.iinfo(np.int32).max, dtype=np.int32)
    for hi, elem in enumerate(hex_node_ids):
        d = int(hex_layer_dist[hi])
        if d < 0:
            continue
        for gid in elem:
            gid = int(gid)
            node_min_layer[gid] = min(node_min_layer[gid], d)

    can_smooth = (node_min_layer >= 1) & (node_min_layer <= layers) & (
        node_min_layer < np.iinfo(np.int32).max
    )
    if freeze_boundary_nodes:
        can_smooth &= ~np.asarray(boundary_node_mask, dtype=bool)
    return ~can_smooth


def _compute_boundary_faces(
    hex_node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return boundary faces as global node IDs and owning hex indices.
    """
    face_count: dict[tuple[int, int, int, int], int] = defaultdict(int)
    face_owner: dict[tuple[int, int, int, int], int] = {}
    for hi, elem in enumerate(hex_node_ids):
        for face in _HEX_FACES:
            key = tuple(sorted(int(elem[i]) for i in face))
            face_count[key] += 1
            face_owner[key] = int(hi)
    faces = []
    owners = []
    for key, cnt in face_count.items():
        if cnt == 1:
            faces.append(np.array(key, dtype=np.int32))
            owners.append(face_owner[key])
    if not faces:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return np.vstack(faces).astype(np.int32), np.asarray(owners, dtype=np.int32)


def _compute_ordered_boundary_faces(hex_node_ids: np.ndarray) -> np.ndarray:
    """
    Return boundary faces as ordered global node IDs (N, 4).
    """
    face_records: dict[tuple[int, int, int, int], tuple[int, int] | None] = {}
    for hi, elem in enumerate(hex_node_ids):
        for fi, face in enumerate(_HEX_FACES):
            ordered = tuple(int(elem[i]) for i in face)
            key = tuple(sorted(ordered))
            prev = face_records.get(key)
            if prev is None:
                face_records[key] = (int(hi), int(fi))
            else:
                # shared internal face
                face_records[key] = None
    out: list[np.ndarray] = []
    for rec in face_records.values():
        if rec is None:
            continue
        hi, fi = rec
        out.append(hex_node_ids[hi][_HEX_FACES[fi]].astype(np.int32))
    if not out:
        return np.empty((0, 4), dtype=np.int32)
    return np.vstack(out).astype(np.int32)


def _node_incident_hexes(hex_node_ids: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for hi, elem in enumerate(hex_node_ids):
        for local_idx, gid in enumerate(elem):
            out[int(gid)].append((hi, int(local_idx)))
    return out


def _node_incident_hex_ids(hex_node_ids: np.ndarray) -> dict[int, set[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for hi, elem in enumerate(hex_node_ids):
        for gid in elem:
            out[int(gid)].add(int(hi))
    return out


def _node_local_edge_scale(
    unique_nodes: np.ndarray, hex_node_ids: np.ndarray
) -> np.ndarray:
    n = len(unique_nodes)
    neighbors: dict[int, set[int]] = defaultdict(set)
    for elem in hex_node_ids:
        for a, b in _HEX_EDGES:
            ga = int(elem[a])
            gb = int(elem[b])
            neighbors[ga].add(gb)
            neighbors[gb].add(ga)
    scales = np.zeros(n, dtype=np.float64)
    for i in range(n):
        neigh = sorted(neighbors.get(i, set()))
        if not neigh:
            scales[i] = 0.0
            continue
        d = np.linalg.norm(unique_nodes[np.asarray(neigh, dtype=np.int32)] - unique_nodes[i], axis=1)
        scales[i] = float(np.median(d))
    return scales


def _corner_jacobian_proxy(hex_coords: np.ndarray, corner: int) -> float:
    n1, n2, n3 = _CORNER_NEIGHBORS[int(corner)]
    v0 = hex_coords[int(corner)]
    e1 = hex_coords[n1] - v0
    e2 = hex_coords[n2] - v0
    e3 = hex_coords[n3] - v0
    return float(np.linalg.det(np.column_stack((e1, e2, e3))))


def validate_hex_inversion(
    original_hex_coords: np.ndarray,
    proposed_hex_coords: np.ndarray,
    local_corner_idx: int,
    min_det_ratio: float = 0.2,
    min_abs_det: float = 1e-10,
) -> bool:
    base_det = _corner_jacobian_proxy(original_hex_coords, local_corner_idx)
    prop_det = _corner_jacobian_proxy(proposed_hex_coords, local_corner_idx)
    if abs(base_det) < min_abs_det:
        return False
    if abs(prop_det) < max(min_abs_det, min_det_ratio * abs(base_det)):
        return False
    if np.sign(prop_det) != np.sign(base_det):
        return False
    return True


def _project_to_surface_with_sdf(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
    sdf_sampler,
    max_iters: int = 2,
) -> np.ndarray:
    """
    Project points to CAD boundary and refine using SDF sign.
    """
    proj, _ = closest_points_with_fallback(mesh, points)
    out = np.asarray(proj, dtype=np.float64).copy()
    for _ in range(max_iters):
        sdf = np.asarray(sdf_sampler(out), dtype=np.float64)
        if np.all(np.abs(sdf) < 1e-4):
            break
        reproj, _ = closest_points_with_fallback(mesh, out)
        out = np.asarray(reproj, dtype=np.float64)
    return out


def _hex_cell_centroids(nodes: np.ndarray, hex_node_ids: np.ndarray) -> np.ndarray:
    coords = nodes[np.asarray(hex_node_ids, dtype=np.int32)]
    return coords.mean(axis=1)


def _stretch_neighbor_nodes_into_dropped_gaps(
    nodes: np.ndarray,
    hex_node_ids: np.ndarray,
    dropped_hex_ids: np.ndarray,
    reference_centroids: np.ndarray,
    mesh: trimesh.Trimesh,
    sdf_sampler,
    *,
    fill_fraction: float = 0.65,
    surface_blend: float = 0.35,
    max_edge_fraction: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move nodes shared by a kept hex and a dropped hex outward into the vacated cell.

    Uses pre-drop centroids for a stable outward direction (kept cell -> dropped cell).
    Returns updated nodes and a boolean mask of nodes that were moved.
    """
    dropped = {int(h) for h in np.asarray(dropped_hex_ids, dtype=np.int64).ravel()}
    if not dropped:
        return nodes, np.zeros(len(nodes), dtype=bool)

    n_nodes = len(nodes)
    incident = _node_incident_hex_ids(hex_node_ids)
    edge_scale = _node_local_edge_scale(nodes, hex_node_ids)
    out = nodes.copy()
    moved = np.zeros(n_nodes, dtype=bool)

    for gid in range(n_nodes):
        inc = incident.get(gid)
        if not inc:
            continue
        dropped_touch = inc & dropped
        kept_touch = inc - dropped
        if not dropped_touch or not kept_touch:
            continue

        pos = out[gid]
        hint = np.zeros(3, dtype=np.float64)
        nh = 0
        for kh in kept_touch:
            c_k = reference_centroids[kh]
            for dh in dropped_touch:
                c_d = reference_centroids[dh]
                v = c_d - c_k
                ln = float(np.linalg.norm(v))
                if ln > 1e-9:
                    hint += v / ln
                    nh += 1
        if nh == 0:
            continue
        hint /= float(nh)
        hint_len = float(np.linalg.norm(hint))
        if hint_len < 1e-9:
            continue
        direction = hint / hint_len

        best_gap = 0.0
        for dh in dropped_touch:
            best_gap = max(best_gap, float(np.linalg.norm(reference_centroids[dh] - pos)))

        es = float(edge_scale[gid])
        if es <= 0.0:
            es = best_gap
        step_dist = min(best_gap * float(fill_fraction), es * float(max_edge_fraction))
        if step_dist < 1e-9:
            continue

        target_gap = pos + direction * step_dist
        sdf_val = float(
            np.asarray(sdf_sampler(target_gap.reshape(1, 3)), dtype=np.float64).ravel()[0]
        )
        cpt, _ = closest_points_with_fallback(mesh, target_gap.reshape(1, 3))
        cpt = np.asarray(cpt[0], dtype=np.float64)
        if sdf_val <= 0.0:
            target = target_gap * (1.0 - surface_blend) + cpt * surface_blend
        else:
            target = cpt

        out[gid] = target
        moved[gid] = True

    return out, moved


def build_hex_surface_skin(
    nodes: np.ndarray,
    kept_hex_node_ids: np.ndarray,
    *,
    round_decimals: int = 8,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Build a triangulated boundary skin graph for kept conformed hexes.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    kept = np.asarray(kept_hex_node_ids, dtype=np.int32)
    if kept.ndim != 2 or kept.shape[1] != 8:
        raise ValueError(f"kept_hex_node_ids must have shape (N, 8), got {kept.shape}.")

    ordered_boundary_faces = _compute_ordered_boundary_faces(kept)
    if len(ordered_boundary_faces) == 0:
        return np.empty((0, 2), dtype=np.int64), {
            "n_boundary_faces": 0,
            "n_quad_perimeter_edges": 0,
            "n_quad_diagonals": 0,
            "n_skin_struts_total": 0,
        }

    strut_set: set[tuple[int, int]] = set()
    n_quad_perimeter_edges = 0
    n_quad_diagonals = 0

    def _add_edge(a: int, b: int) -> bool:
        if a == b:
            return False
        if a > b:
            a, b = b, a
        old = len(strut_set)
        strut_set.add((int(a), int(b)))
        return len(strut_set) > old

    # A3/A4: perimeter + shortest diagonal per boundary quad.
    for face in ordered_boundary_faces:
        a, b, c, d = [int(v) for v in face]
        for u, v in ((a, b), (b, c), (c, d), (d, a)):
            if _add_edge(u, v):
                n_quad_perimeter_edges += 1
        d0 = float(np.linalg.norm(pts[a] - pts[c]))
        d1 = float(np.linalg.norm(pts[b] - pts[d]))
        if d0 <= d1:
            if _add_edge(a, c):
                n_quad_diagonals += 1
        else:
            if _add_edge(b, d):
                n_quad_diagonals += 1

    # Keep strut indices in `nodes` global ID space (no remap) so volume graph can merge.
    skin_struts = np.array(sorted(strut_set), dtype=np.int64) if strut_set else np.empty((0, 2), dtype=np.int64)

    return skin_struts, {
        "n_boundary_faces": int(len(ordered_boundary_faces)),
        "n_quad_perimeter_edges": int(n_quad_perimeter_edges),
        "n_quad_diagonals": int(n_quad_diagonals),
        "n_skin_struts_total": int(len(skin_struts)),
    }


def _merge_hex_volume_skin_explicit(
    vol_nodes: np.ndarray,
    vol_struts: np.ndarray,
    boundary_face_to_vol_node: dict[tuple[int, int], int],
    skin_nodes: np.ndarray,
    skin_struts: np.ndarray,
    quad_topology: BoundaryQuadTopology,
    skin_quad_centroid_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Merge volume and skin graphs using integer (hex, face) bridges — no coordinate dedup.
    """
    nodes_list = [np.asarray(p, dtype=np.float64) for p in vol_nodes]
    skin_to_global: dict[int, int] = {}
    n_bridges = 0
    n_mapped_quads = 0

    for qi in range(quad_topology.n_quads):
        hi, fi = quad_topology.quad_to_hex_face(qi)
        vol_id = boundary_face_to_vol_node.get((hi, fi))
        skin_local = int(skin_quad_centroid_indices[qi])
        if vol_id is not None:
            skin_to_global[skin_local] = int(vol_id)
            n_mapped_quads += 1
        elif skin_local not in skin_to_global:
            skin_to_global[skin_local] = len(nodes_list)
            nodes_list.append(np.asarray(skin_nodes[skin_local], dtype=np.float64))

    for si in range(len(skin_nodes)):
        if si in skin_to_global:
            continue
        skin_to_global[si] = len(nodes_list)
        nodes_list.append(np.asarray(skin_nodes[si], dtype=np.float64))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    strut_set: set[tuple[int, int]] = set()

    def _add(a: int, b: int) -> None:
        if a == b:
            return
        if a > b:
            a, b = b, a
        strut_set.add((int(a), int(b)))

    for a, b in vol_struts:
        _add(int(a), int(b))
    for a, b in skin_struts:
        _add(skin_to_global[int(a)], skin_to_global[int(b)])

    for qi in range(quad_topology.n_quads):
        hi, fi = quad_topology.quad_to_hex_face(qi)
        vol_id = boundary_face_to_vol_node.get((hi, fi))
        if vol_id is None:
            continue
        skin_local = int(skin_quad_centroid_indices[qi])
        skin_id = skin_to_global[skin_local]
        if vol_id != skin_id:
            _add(int(vol_id), int(skin_id))
            n_bridges += 1

    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts, {
        "n_volume_skin_bridges": int(n_bridges),
        "n_skin_quads_mapped_to_volume": int(n_mapped_quads),
        "merge_mode": "explicit_hex_face_topology",
    }


def synthesize_vf_gated_hex_volume_and_surface_dual(
    hex_elements: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    hex_elements_for_vf: np.ndarray | None = None,
    volume_fraction_threshold: float = 0.5,
    volume_on_all_hexes: bool = False,
    hex_rule: str = "octahedral",
    target_element_size: float | None = None,
    topology_round_decimals: int = 6,
    edt_resolution: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Build volume + surface-dual graphs with per-hex inside-volume gating.

    All hex elements are retained in the conformal scaffold.     Internal volume lattice (``hex_rule``) is emitted only for hexes with inside
    fraction > threshold unless ``volume_on_all_hexes`` is True. Surface dual uses
    VF gating for sliver cages; exterior face centroids merge with octahedral face
    nodes when coordinates coincide.
    """
    from .boundary_policy import build_edt_sdf_sampler, calculate_hex_volume_fractions
    from .hex_surface_dual import (
        generate_hex_surface_dual_cage_volume_gated,
        hex_node_ids_from_elements,
    )

    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must be (N, 8, 3); got {elems.shape}.")
    n_hex = elems.shape[0]
    if n_hex == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        z2 = np.empty((0, 2), dtype=np.int64)
        return empty, z2, {"n_hexes": 0}

    if edt_resolution is None:
        edt_resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, float(edt_resolution))
    vf_source = (
        np.asarray(hex_elements_for_vf, dtype=np.float64)
        if hex_elements_for_vf is not None
        else elems
    )
    if vf_source.shape != elems.shape:
        raise ValueError(
            "hex_elements_for_vf must match hex_elements shape "
            f"{elems.shape}; got {vf_source.shape}."
        )
    vf = calculate_hex_volume_fractions(vf_source, sample_sdf)
    has_internal_skin = vf > float(volume_fraction_threshold)
    has_internal_volume = (
        np.ones(n_hex, dtype=bool)
        if volume_on_all_hexes
        else has_internal_skin.copy()
    )

    nodes, hex_node_ids = hex_node_ids_from_elements(
        elems, round_decimals=int(topology_round_decimals)
    )
    quad_topology = build_boundary_quad_topology(hex_node_ids)
    rule_key = str(hex_rule).strip().lower()
    vol_nodes = np.empty((0, 3), dtype=np.float64)
    vol_struts = np.empty((0, 2), dtype=np.int64)
    boundary_face_to_vol: dict[tuple[int, int], int] = {}
    if rule_key == "octahedral":
        vol_nodes, vol_struts, boundary_face_to_vol = (
            generate_hex_octahedral_volume_with_boundary_face_map(
                elems,
                volume_emit_mask=has_internal_volume,
                round_decimals=int(topology_round_decimals),
            )
        )
    elif np.any(has_internal_volume):
        vol_nodes, vol_struts = generate_hex_topology(
            elems[has_internal_volume],
            rule_name=str(hex_rule),
            round_decimals=int(topology_round_decimals),
        )

    skin_centroids, skin_struts, skin_report = generate_hex_surface_dual_cage_volume_gated(
        nodes,
        quad_topology.quads,
        quad_topology.hex_owner,
        has_internal_skin,
        target_element_size=target_element_size,
    )
    skin_quad_ids = np.arange(quad_topology.n_quads, dtype=np.int64)

    if rule_key == "octahedral" and quad_topology.n_quads > 0:
        merged_nodes, merged_struts, merge_report = _merge_hex_volume_skin_explicit(
            vol_nodes,
            vol_struts,
            boundary_face_to_vol,
            skin_centroids,
            skin_struts,
            quad_topology,
            skin_quad_ids,
        )
    else:
        node_map: dict[tuple[float, float, float], int] = {}
        nodes_list: list[np.ndarray] = []
        strut_set: set[tuple[int, int]] = set()
        rd = int(topology_round_decimals)

        def _global_index(pt: np.ndarray) -> int:
            key = tuple(np.round(pt, rd).tolist())
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(np.asarray(pt, dtype=np.float64))
            return idx

        for pt in vol_nodes:
            _global_index(pt)
        for pt in skin_centroids:
            _global_index(pt)
        for a, b in vol_struts:
            ga = _global_index(vol_nodes[int(a)])
            gb = _global_index(vol_nodes[int(b)])
            if ga != gb:
                if ga > gb:
                    ga, gb = gb, ga
                strut_set.add((ga, gb))
        for a, b in skin_struts:
            ga = _global_index(skin_centroids[int(a)])
            gb = _global_index(skin_centroids[int(b)])
            if ga != gb:
                if ga > gb:
                    ga, gb = gb, ga
                strut_set.add((ga, gb))
        merged_nodes = (
            np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
        )
        merged_struts = (
            np.array(sorted(strut_set), dtype=np.int64)
            if strut_set
            else np.empty((0, 2), dtype=np.int64)
        )
        merge_report = {"merge_mode": "coordinate_round", "n_volume_skin_bridges": 0}

    report = {
        "n_hexes": int(n_hex),
        "volume_on_all_hexes": bool(volume_on_all_hexes),
        "n_hexes_with_internal_lattice": int(np.sum(has_internal_volume)),
        "n_hexes_surface_only": int(np.sum(~has_internal_skin)),
        "n_hexes_with_volume_lattice": int(np.sum(has_internal_volume)),
        "volume_fraction_threshold": float(volume_fraction_threshold),
        "volume_nodes": int(len(vol_nodes)),
        "volume_struts": int(len(vol_struts)),
        "skin_centroids": int(len(skin_centroids)),
        "skin_struts": int(len(skin_struts)),
        "merged_nodes": int(len(merged_nodes)),
        "merged_struts": int(len(merged_struts)),
        **merge_report,
        **skin_report,
    }
    return merged_nodes, merged_struts, report


# Layers 1..N inward from skin hexes may move during Conformal Dual compliance relax.
CONFORMAL_DUAL_COMPLIANCE_LAYERS = 2


def synthesize_conformal_dual_lattice(
    conformed_hex_elements: np.ndarray,
    *,
    topology_round_decimals: int = 6,
    volume_emit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Conformal Dual synthesis: octahedral volume + integer-mapped surface dual skin.

    Volume face-center nodes are built with
    ``generate_hex_octahedral_volume_with_boundary_face_map``. Skin struts connect
    adjacent boundary-quad face centers via ``boundary_face_to_node[(hex, face)]``.

    Parameters
    ----------
    conformed_hex_elements : ndarray
        ``(N, 8, 3)`` conformed hex brick coordinates.
    topology_round_decimals : int
        Node deduplication precision for topology merge.
    volume_emit_mask : ndarray, optional
        Per-hex mask for interior octahedral strut emission. Default: all True.

    Returns
    -------
    nodes : ndarray
        ``(V, 3)`` unified node coordinates (volume + face centers).
    volume_struts : ndarray
        ``(E_vol, 2)`` interior octahedral struts (global node indices).
    dual_skin_struts : ndarray
        ``(E_skin, 2)`` integer-mapped surface dual struts (global node indices).
    report : dict
        Strut counts and merge diagnostics.
    """
    from .hex_topology_module import generate_hex_octahedral_volume_with_boundary_face_map

    elems = np.asarray(conformed_hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"conformed_hex_elements must be (N, 8, 3); got {elems.shape}.")
    n_hex = elems.shape[0]
    if n_hex == 0:
        empty_n = np.empty((0, 3), dtype=np.float64)
        empty_s = np.empty((0, 2), dtype=np.int64)
        return empty_n, empty_s, empty_s, {"n_hexes": 0}

    emit = (
        np.ones(n_hex, dtype=bool)
        if volume_emit_mask is None
        else np.asarray(volume_emit_mask, dtype=bool).ravel()
    )
    if emit.shape[0] != n_hex:
        raise ValueError("volume_emit_mask length must match hex count.")

    vol_nodes, volume_struts, boundary_face_to_node = (
        generate_hex_octahedral_volume_with_boundary_face_map(
            elems,
            volume_emit_mask=emit,
            round_decimals=int(topology_round_decimals),
        )
    )

    _, hex_node_ids = hex_node_ids_from_elements(
        elems, round_decimals=int(topology_round_decimals)
    )
    quad_topology = build_boundary_quad_topology(hex_node_ids)

    dual_skin_set: set[tuple[int, int]] = set()
    missing_face_lookup = 0
    pairs, _shared_edges = get_boundary_quad_adjacency(quad_topology.quads)

    for i, j in pairs:
        hi_a, fi_a = quad_topology.quad_to_hex_face(int(i))
        hi_b, fi_b = quad_topology.quad_to_hex_face(int(j))
        key_a = (int(hi_a), int(fi_a))
        key_b = (int(hi_b), int(fi_b))
        if key_a not in boundary_face_to_node or key_b not in boundary_face_to_node:
            missing_face_lookup += 1
            continue
        ga = int(boundary_face_to_node[key_a])
        gb = int(boundary_face_to_node[key_b])
        if ga == gb:
            continue
        a, b = (ga, gb) if ga < gb else (gb, ga)
        dual_skin_set.add((a, b))

    dual_skin_struts = (
        np.array(sorted(dual_skin_set), dtype=np.int64)
        if dual_skin_set
        else np.empty((0, 2), dtype=np.int64)
    )

    merged_set: set[tuple[int, int]] = set()
    for a, b in np.asarray(volume_struts, dtype=np.int64):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        merged_set.add((ia, ib) if ia < ib else (ib, ia))
    merged_set |= dual_skin_set

    merged_struts = (
        np.array(sorted(merged_set), dtype=np.int64)
        if merged_set
        else np.empty((0, 2), dtype=np.int64)
    )

    report = {
        "n_hexes": int(n_hex),
        "n_boundary_quads": int(quad_topology.n_quads),
        "n_volume_struts": int(len(volume_struts)),
        "n_dual_skin_struts": int(len(dual_skin_struts)),
        "n_merged_struts": int(len(merged_struts)),
        "n_dual_pairs": int(len(pairs)),
        "missing_face_lookup": int(missing_face_lookup),
        "merge_mode": "conformal_dual_integer_face_map",
    }
    return vol_nodes, volume_struts, dual_skin_struts, report


def synthesize_vf_gated_hex_volume_and_surface_dual_on_surface_paths(
    hex_elements: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    hex_elements_for_vf: np.ndarray | None = None,
    volume_fraction_threshold: float = 0.5,
    volume_on_all_hexes: bool = False,
    hex_rule: str = "octahedral",
    target_element_size: float | None = None,
    topology_round_decimals: int = 6,
    edt_resolution: float | None = None,
    include_sliver_element_cage: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Like ``synthesize_vf_gated_hex_volume_and_surface_dual`` but skin links use
    centroid -> shared-edge midpoint -> centroid paths (no coplanar filter).
    """
    from .boundary_policy import build_edt_sdf_sampler, calculate_hex_volume_fractions
    from .hex_surface_dual import (
        generate_hex_surface_dual_on_surface_paths,
        hex_node_ids_from_elements,
    )

    elems = np.asarray(hex_elements, dtype=np.float64)
    n_hex = elems.shape[0]
    if n_hex == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, np.empty((0, 2), dtype=np.int64), {"n_hexes": 0}

    if edt_resolution is None:
        edt_resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, float(edt_resolution))
    vf_source = (
        np.asarray(hex_elements_for_vf, dtype=np.float64)
        if hex_elements_for_vf is not None
        else elems
    )
    vf = calculate_hex_volume_fractions(vf_source, sample_sdf)
    has_internal_skin = vf > float(volume_fraction_threshold)
    has_internal_volume = (
        np.ones(n_hex, dtype=bool)
        if volume_on_all_hexes
        else has_internal_skin.copy()
    )

    nodes, hex_node_ids = hex_node_ids_from_elements(
        elems, round_decimals=int(topology_round_decimals)
    )
    quad_topology = build_boundary_quad_topology(hex_node_ids)
    rule_key = str(hex_rule).strip().lower()
    vol_nodes = np.empty((0, 3), dtype=np.float64)
    vol_struts = np.empty((0, 2), dtype=np.int64)
    boundary_face_to_vol: dict[tuple[int, int], int] = {}
    if rule_key == "octahedral":
        vol_nodes, vol_struts, boundary_face_to_vol = (
            generate_hex_octahedral_volume_with_boundary_face_map(
                elems,
                volume_emit_mask=has_internal_volume,
                round_decimals=int(topology_round_decimals),
            )
        )
    elif np.any(has_internal_volume):
        vol_nodes, vol_struts = generate_hex_topology(
            elems[has_internal_volume],
            rule_name=str(hex_rule),
            round_decimals=int(topology_round_decimals),
        )

    skin_nodes, skin_struts, skin_report = generate_hex_surface_dual_on_surface_paths(
        nodes,
        quad_topology.quads,
        quad_topology.hex_owner,
        has_internal_skin,
        include_sliver_element_cage=include_sliver_element_cage,
        target_element_size=target_element_size,
    )
    skin_quad_ids = np.arange(quad_topology.n_quads, dtype=np.int64)

    if rule_key == "octahedral" and quad_topology.n_quads > 0:
        merged_nodes, merged_struts, merge_report = _merge_hex_volume_skin_explicit(
            vol_nodes,
            vol_struts,
            boundary_face_to_vol,
            skin_nodes,
            skin_struts,
            quad_topology,
            skin_quad_ids,
        )
    else:
        node_map: dict[tuple[float, float, float], int] = {}
        nodes_list: list[np.ndarray] = []
        strut_set: set[tuple[int, int]] = set()
        rd = int(topology_round_decimals)

        def _global_index(pt: np.ndarray) -> int:
            key = tuple(np.round(pt, rd).tolist())
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(np.asarray(pt, dtype=np.float64))
            return idx

        for pt in vol_nodes:
            _global_index(pt)
        for pt in skin_nodes:
            _global_index(pt)
        for a, b in vol_struts:
            ga = _global_index(vol_nodes[int(a)])
            gb = _global_index(vol_nodes[int(b)])
            if ga != gb:
                if ga > gb:
                    ga, gb = gb, ga
                strut_set.add((ga, gb))
        for a, b in skin_struts:
            ga = _global_index(skin_nodes[int(a)])
            gb = _global_index(skin_nodes[int(b)])
            if ga != gb:
                if ga > gb:
                    ga, gb = gb, ga
                strut_set.add((ga, gb))
        merged_nodes = (
            np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
        )
        merged_struts = (
            np.array(sorted(strut_set), dtype=np.int64)
            if strut_set
            else np.empty((0, 2), dtype=np.int64)
        )
        merge_report = {"merge_mode": "coordinate_round", "n_volume_skin_bridges": 0}

    report = {
        "n_hexes": int(n_hex),
        "volume_on_all_hexes": bool(volume_on_all_hexes),
        "n_hexes_with_volume_lattice": int(np.sum(has_internal_volume)),
        "n_hexes_surface_only": int(np.sum(~has_internal_skin)),
        "volume_fraction_threshold": float(volume_fraction_threshold),
        "volume_nodes": int(len(vol_nodes)),
        "volume_struts": int(len(vol_struts)),
        "skin_nodes": int(len(skin_nodes)),
        "skin_struts": int(len(skin_struts)),
        "merged_nodes": int(len(merged_nodes)),
        "merged_struts": int(len(merged_struts)),
        **merge_report,
        **skin_report,
    }
    return merged_nodes, merged_struts, report


def synthesize_hex_volume_with_stl_surface_skin(
    hex_elements: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    hex_elements_for_vf: np.ndarray | None = None,
    volume_fraction_threshold: float = 0.5,
    volume_on_all_hexes: bool = False,
    target_element_size: float | None = None,
    topology_round_decimals: int = 6,
    edt_resolution: float | None = None,
    stl_skin_max_span_factor: float | None = 1.5,
    bridge_merge_tolerance: float | None = None,
    skin_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Octahedral volume lattice + STL-native surface skin + explicit hex-face bridges.

    Returns:
        merged_nodes, merged_struts, stl_skin_nodes, stl_skin_struts, report
    """
    from .boundary_policy import build_edt_sdf_sampler, calculate_hex_volume_fractions
    from .hex_surface_dual import hex_node_ids_from_elements
    from .stl_surface_skin import (
        generate_stl_surface_path_skin,
        merge_volume_with_stl_surface_skin,
    )

    elems = np.asarray(hex_elements, dtype=np.float64)
    n_hex = elems.shape[0]
    if n_hex == 0:
        z = np.empty((0, 3), dtype=np.float64)
        z2 = np.empty((0, 2), dtype=np.int64)
        return z, z2, z, z2, {"n_hexes": 0}

    if edt_resolution is None:
        edt_resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, float(edt_resolution))
    vf_source = (
        np.asarray(hex_elements_for_vf, dtype=np.float64)
        if hex_elements_for_vf is not None
        else elems
    )
    vf = calculate_hex_volume_fractions(vf_source, sample_sdf)
    has_internal_volume = (
        np.ones(n_hex, dtype=bool)
        if volume_on_all_hexes
        else (vf > float(volume_fraction_threshold))
    )

    vol_nodes, vol_struts, boundary_face_to_vol = (
        generate_hex_octahedral_volume_with_boundary_face_map(
            elems,
            volume_emit_mask=has_internal_volume,
            round_decimals=int(topology_round_decimals),
        )
    )

    skin_source = skin_mesh if skin_mesh is not None else mesh
    max_span = None
    if (
        target_element_size is not None
        and stl_skin_max_span_factor is not None
    ):
        max_span = float(stl_skin_max_span_factor) * float(target_element_size)

    stl_nodes, stl_struts, stl_report = generate_stl_surface_path_skin(
        skin_source,
        project_to_surface=True,
        max_centroid_span=max_span,
    )

    _, hex_node_ids = hex_node_ids_from_elements(
        elems, round_decimals=int(topology_round_decimals)
    )
    quad_topology = build_boundary_quad_topology(hex_node_ids)

    merged_nodes, merged_struts, merge_report = merge_volume_with_stl_surface_skin(
        vol_nodes,
        vol_struts,
        boundary_face_to_vol,
        quad_topology,
        elems,
        stl_nodes,
        stl_struts,
        mesh,
        bridge_merge_tolerance=bridge_merge_tolerance,
    )
    if skin_mesh is not None:
        stl_report = {
            **stl_report,
            "skin_mesh_faces": int(len(skin_mesh.faces)),
        }

    report: dict[str, int | float | str] = {
        "n_hexes": int(n_hex),
        "volume_on_all_hexes": bool(volume_on_all_hexes),
        "n_hexes_with_volume_lattice": int(np.sum(has_internal_volume)),
        "volume_fraction_threshold": float(volume_fraction_threshold),
        "volume_nodes": int(len(vol_nodes)),
        "volume_struts": int(len(vol_struts)),
        "merged_nodes": int(len(merged_nodes)),
        "merged_struts": int(len(merged_struts)),
        **stl_report,
        **merge_report,
    }
    return merged_nodes, merged_struts, stl_nodes, stl_struts, report


def synthesize_hex_volume_with_stl_surface_skin_decoupled(
    hex_elements: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    hex_elements_for_vf: np.ndarray | None = None,
    volume_fraction_threshold: float = 0.5,
    volume_on_all_hexes: bool = False,
    target_element_size: float | None = None,
    topology_round_decimals: int = 6,
    edt_resolution: float | None = None,
    stl_skin_max_span_factor: float | None = 1.5,
    bridge_merge_tolerance: float | None = None,
    skin_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Decoupled octahedral core + STL-native skin (surface-hugging) + explicit bridges.

    - **Skin:** triangle centroid → edge-mid → centroid on ``skin_mesh`` (projected).
    - **Core:** octahedral volume struts only (boolean-crop later).
    - **Bridges:** each hex boundary face center → nearest STL skin node (cKDTree).

    Returns ``(nodes_core, struts_core, nodes_skin, skin_struts, report)`` for
    ``generate_decoupled_core_and_ribbed_skin``. ``nodes_core`` is
    ``vstack(volume_nodes, stl_skin_nodes)``; ``struts_core`` is volume + bridges;
    ``skin_struts`` index ``nodes_skin`` locally (0 .. n_skin-1).
    """
    from .boundary_policy import build_edt_sdf_sampler, calculate_hex_volume_fractions
    from .hex_surface_dual import build_boundary_quad_topology, hex_node_ids_from_elements
    from .stl_surface_skin import (
        build_volume_stl_skin_bridge_struts,
        generate_stl_surface_path_skin,
    )

    elems = np.asarray(hex_elements, dtype=np.float64)
    n_hex = elems.shape[0]
    if n_hex == 0:
        z = np.empty((0, 3), dtype=np.float64)
        z2 = np.empty((0, 2), dtype=np.int64)
        return z, z2, z, z2, {"n_hexes": 0}

    if edt_resolution is None:
        edt_resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, float(edt_resolution))
    vf_source = (
        np.asarray(hex_elements_for_vf, dtype=np.float64)
        if hex_elements_for_vf is not None
        else elems
    )
    vf = calculate_hex_volume_fractions(vf_source, sample_sdf)
    has_internal_volume = (
        np.ones(n_hex, dtype=bool)
        if volume_on_all_hexes
        else (vf > float(volume_fraction_threshold))
    )

    vol_nodes, vol_struts, boundary_face_to_vol = (
        generate_hex_octahedral_volume_with_boundary_face_map(
            elems,
            volume_emit_mask=has_internal_volume,
            round_decimals=int(topology_round_decimals),
        )
    )

    skin_source = skin_mesh if skin_mesh is not None else mesh
    max_span = None
    if target_element_size is not None and stl_skin_max_span_factor is not None:
        max_span = float(stl_skin_max_span_factor) * float(target_element_size)

    stl_nodes, stl_struts, stl_report = generate_stl_surface_path_skin(
        skin_source,
        project_to_surface=True,
        max_centroid_span=max_span,
    )

    _, hex_node_ids = hex_node_ids_from_elements(
        elems, round_decimals=int(topology_round_decimals)
    )
    quad_topology = build_boundary_quad_topology(hex_node_ids)

    bridge_arr, bridge_meta = build_volume_stl_skin_bridge_struts(
        vol_nodes,
        boundary_face_to_vol,
        quad_topology,
        elems,
        stl_nodes,
        mesh,
        bridge_merge_tolerance=bridge_merge_tolerance,
    )

    n_vol = int(len(vol_nodes))
    nodes_core = (
        np.vstack((vol_nodes, stl_nodes))
        if len(stl_nodes)
        else vol_nodes.copy()
    )

    core_set: set[tuple[int, int]] = set()
    for a, b in np.asarray(vol_struts, dtype=np.int64):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        core_set.add((ia, ib) if ia < ib else (ib, ia))
    for a, b in bridge_arr:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        core_set.add((ia, ib) if ia < ib else (ib, ia))

    struts_core = (
        np.array(sorted(core_set), dtype=np.int64)
        if core_set
        else np.empty((0, 2), dtype=np.int64)
    )

    report: dict[str, int | float | str] = {
        "n_hexes": int(n_hex),
        "merge_mode": "decoupled_volume_stl_skin_bridges",
        "volume_on_all_hexes": bool(volume_on_all_hexes),
        "n_hexes_with_volume_lattice": int(np.sum(has_internal_volume)),
        "volume_struts": int(len(vol_struts)),
        "struts_core": int(len(struts_core)),
        "skin_struts": int(len(stl_struts)),
        "bridge_struts": int(len(bridge_arr)),
        **stl_report,
        **bridge_meta,
    }
    if skin_mesh is not None:
        report["skin_mesh_faces"] = int(len(skin_mesh.faces))
    return nodes_core, struts_core, stl_nodes, stl_struts, report


def synthesize_conformed_hex_volume_and_skin(
    nodes_full: np.ndarray,
    kept_hex_node_ids: np.ndarray,
    boundary_node_mask: np.ndarray,
    hex_elements: np.ndarray,
    hex_rule: str,
    *,
    topology_round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Split **core** (volume + bridges, cropped later) from **skin** (A3+A4, uncropped ribbons).

    Returns ``(nodes_core, struts_core, nodes_skin, skin_struts, report)`` where
    ``nodes_skin`` are hex corner coordinates (same as ``nodes_full``) and
    ``skin_struts`` index only those corners. Core graph is compressed to endpoints
    used by volume + bridge struts only.
    """
    nf = np.asarray(nodes_full, dtype=np.float64)
    bmask = np.asarray(boundary_node_mask, dtype=bool)
    kept = np.asarray(kept_hex_node_ids, dtype=np.int32)
    elems = np.asarray(hex_elements, dtype=np.float64)
    rd = int(topology_round_decimals)
    rule_key = str(hex_rule).strip().lower()

    vol_nodes, vol_struts = generate_hex_topology(
        elems, rule_name=hex_rule, round_decimals=rd
    )
    volume_struts_before_filter = int(len(vol_struts))

    skin_struts, skin_meta = build_hex_surface_skin(nf, kept)

    if rule_key == "hex_face_dual":
        n_full = int(len(nf))
        unified_nodes = np.vstack([nf, vol_nodes]) if len(vol_nodes) else nf.copy()

        ordered_faces = _compute_ordered_boundary_faces(kept)
        face_centroids = (
            np.mean(nf[ordered_faces.astype(np.int64)], axis=1)
            if len(ordered_faces)
            else np.empty((0, 3), dtype=np.float64)
        )
        tol = 1e-3
        vol_on_boundary = np.zeros(len(vol_nodes), dtype=bool)
        if len(face_centroids) and len(vol_nodes):
            dmat = np.linalg.norm(
                vol_nodes[:, None, :] - face_centroids[None, :, :], axis=2
            )
            vol_on_boundary = np.any(dmat < tol, axis=1)

        filtered_vol: list[tuple[int, int]] = []
        for a, b in np.asarray(vol_struts, dtype=np.int64):
            va, vb = int(a), int(b)
            if vol_on_boundary[va] and vol_on_boundary[vb]:
                continue
            ga, gb = n_full + va, n_full + vb
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            filtered_vol.append((ga, gb))
        volume_struts_after_filter = int(len(filtered_vol))

        bridge_list: list[tuple[int, int]] = []
        if len(ordered_faces) and len(vol_nodes):
            for face in ordered_faces:
                g = [int(x) for x in face]
                c = np.mean(nf[g], axis=0)
                d = np.linalg.norm(vol_nodes - c, axis=1)
                k = int(np.argmin(d))
                if float(d[k]) > tol:
                    continue
                vk = n_full + k
                for gi in g:
                    u, v = (vk, gi) if vk < gi else (gi, vk)
                    bridge_list.append((u, v))

        core_set = set(filtered_vol) | set(bridge_list)
        struts_core_arr = (
            np.array(sorted(core_set), dtype=np.int64)
            if core_set
            else np.empty((0, 2), dtype=np.int64)
        )
        keep = np.ones(len(struts_core_arr), dtype=bool) if len(struts_core_arr) else np.zeros(0, dtype=bool)
        nodes_c, struts_c = compress_graph_to_kept_struts(unified_nodes, struts_core_arr, keep)
        skin_arr = np.asarray(skin_struts, dtype=np.int64)

        report = {
            "skin_struts": int(len(skin_arr)),
            "volume_struts_before_filter": volume_struts_before_filter,
            "volume_struts_after_filter": volume_struts_after_filter,
            "bridge_struts": int(len(bridge_list)),
            "struts_core": int(len(struts_c)),
            "final_merged_struts": int(len(struts_c) + len(skin_arr)),
            "struts_after_compress": int(len(struts_c)),
            "n_boundary_faces": int(skin_meta.get("n_boundary_faces", 0)),
            "n_quad_diagonals": int(skin_meta.get("n_quad_diagonals", 0)),
        }
        return nodes_c, struts_c, nf.copy(), skin_arr.copy(), report

    # Corner-based volume rules (e.g. grid): map volume nodes onto nodes_full.
    key_to_gid: dict[tuple[float, ...], int] = {}
    for gid in range(len(nf)):
        k = tuple(np.round(nf[gid], rd).tolist())
        key_to_gid[k] = int(gid)

    vol_to_full: list[int] = []
    for i in range(len(vol_nodes)):
        k = tuple(np.round(vol_nodes[i], rd).tolist())
        vol_to_full.append(int(key_to_gid.get(k, -1)))

    mapped: list[tuple[int, int]] = []
    for a, b in np.asarray(vol_struts, dtype=np.int64):
        ga, gb = vol_to_full[int(a)], vol_to_full[int(b)]
        if ga < 0 or gb < 0:
            continue
        if ga == gb:
            continue
        if ga > gb:
            ga, gb = gb, ga
        mapped.append((ga, gb))

    filtered_vol: list[tuple[int, int]] = []
    for ga, gb in mapped:
        if bool(bmask[ga]) and bool(bmask[gb]):
            continue
        filtered_vol.append((ga, gb))
    volume_struts_after_filter = int(len(filtered_vol))

    core_set = set(filtered_vol)
    struts_core_arr = (
        np.array(sorted(core_set), dtype=np.int64)
        if core_set
        else np.empty((0, 2), dtype=np.int64)
    )
    keep = np.ones(len(struts_core_arr), dtype=bool) if len(struts_core_arr) else np.zeros(0, dtype=bool)
    nodes_c, struts_c = compress_graph_to_kept_struts(nf, struts_core_arr, keep)
    skin_arr = np.asarray(skin_struts, dtype=np.int64)

    report = {
        "skin_struts": int(len(skin_arr)),
        "volume_struts_before_filter": volume_struts_before_filter,
        "volume_struts_after_filter": volume_struts_after_filter,
        "bridge_struts": 0,
        "struts_core": int(len(struts_c)),
        "final_merged_struts": int(len(struts_c) + len(skin_arr)),
        "struts_after_compress": int(len(struts_c)),
        "n_boundary_faces": int(skin_meta.get("n_boundary_faces", 0)),
        "n_quad_diagonals": int(skin_meta.get("n_quad_diagonals", 0)),
    }
    return nodes_c, struts_c, nf.copy(), skin_arr.copy(), report


def generate_conformed_hex_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    *,
    grid_anchor: str = "bbox_center",
    neighbor_stretch: bool = True,
    boundary_stretch_out: bool = True,
    cull_mostly_external_hexes: bool = True,
    vf_cull_threshold: float = 0.5,
    laplacian_iterations: int = 5,
    laplacian_alpha: float = 0.5,
    laplacian_smooth_layers_inward: int | None = None,
    shrink_wrap_relax: bool = True,
    conformal_dual_mode: bool = False,
    conformal_dual_compliance_layers: int = CONFORMAL_DUAL_COMPLIANCE_LAYERS,
    return_debug_payload: bool = False,
    eps_pull: float | None = None,
    min_jacobian_proxy: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Route 3: Conformed hexahedral scaffold (SDF/EDT field snapping).

    Workflow:
    1. Symmetric Cartesian hex grid over the part bounding box.
    2. Loose centroid SDF pre-cull (keeps a boundary band for volume-fraction tests).
    3. Optionally on boundary hexes: drop cells with <= 50% inside volume.
    4. Optionally stretch nodes on kept hexes that bordered a dropped cell into the gap,
       then tiered boundary conform on the new skin (pull-in always; stretch-out optional).
    5. Relaxation: ``shrink_wrap_relax`` freezes the full boundary and smooths the
       interior; ``conformal_dual_mode`` freezes boundary + layers deeper than
       ``conformal_dual_compliance_layers`` and smooths only layers 1..N inward.

    Args:
        neighbor_stretch: If True, expand shared nodes into cells removed by VF culling.
        boundary_stretch_out: If True, blend inside-near-boundary nodes toward the surface.
        cull_mostly_external_hexes: If True, remove boundary hexes with <= 50% inside volume.
        laplacian_iterations: Interior relaxation passes (0 disables Laplacian).
        laplacian_smooth_layers_inward: Legacy layer-limited smoothing (ignored if
            ``shrink_wrap_relax`` or ``conformal_dual_mode`` is True).
        shrink_wrap_relax: If True, full boundary snap then frozen-boundary Laplacian.
        conformal_dual_mode: If True, forceful boundary pull then compliance-layer
            Laplacian (layers 1..``conformal_dual_compliance_layers`` only).
        conformal_dual_compliance_layers: Inward layer count that may move when
            ``conformal_dual_mode`` is True (default 2).
    """
    # Route 3 uses a fatter base set than Route 2: start from full bbox grid,
    # then apply a loose centroid SDF filter to avoid the strict centroid paradox.
    all_hexes = _generate_bbox_hex_grid(
        mesh, target_element_size, grid_anchor=grid_anchor
    )
    if len(all_hexes) == 0:
        base_report = {
            "boundary_nodes": 0,
            "proposed_moves": 0,
            "accepted_moves": 0,
            "rejected_moves": 0,
            "rejected_cap": 0,
            "rejected_inversion": 0,
            "shrunk_cells": 0,
            "grown_cells": 0,
            "shrink_hexes": 0,
            "grow_hexes": 0,
        }
        if return_debug_payload:
            return (
                np.empty((0, 8, 3), dtype=np.float64),
                base_report,
                {
                    "nodes": np.empty((0, 3), dtype=np.float64),
                    "kept_hex_node_ids": np.empty((0, 8), dtype=np.int32),
                    "boundary_faces": np.empty((0, 4), dtype=np.int32),
                    "boundary_node_mask": np.empty((0,), dtype=bool),
                },
            )
        return np.empty((0, 8, 3), dtype=np.float64), base_report
    resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, resolution)
    all_centroids = np.mean(all_hexes, axis=1)
    sdf_centroids = sample_sdf(all_centroids)
    loose_keep = sdf_centroids < float(target_element_size)
    hexes = all_hexes[loose_keep]
    if len(hexes) == 0:
        base_report = {
            "boundary_nodes": 0,
            "proposed_moves": 0,
            "accepted_moves": 0,
            "rejected_moves": 0,
            "rejected_cap": 0,
            "rejected_inversion": 0,
            "shrunk_cells": 0,
            "grown_cells": 0,
            "shrink_hexes": 0,
            "grow_hexes": 0,
            "kept_hexes_after_shrink": 0,
            "loose_prefilter_hexes": 0,
        }
        if return_debug_payload:
            return (
                np.empty((0, 8, 3), dtype=np.float64),
                base_report,
                {
                    "nodes": np.empty((0, 3), dtype=np.float64),
                    "kept_hex_node_ids": np.empty((0, 8), dtype=np.int32),
                    "boundary_faces": np.empty((0, 4), dtype=np.int32),
                    "boundary_node_mask": np.empty((0,), dtype=bool),
                },
            )
        return np.empty((0, 8, 3), dtype=np.float64), base_report

    unique_nodes, hex_node_ids = _build_unique_node_representation(hexes)
    original_nodes = unique_nodes.copy()
    pre_conform_hex_elements = original_nodes[hex_node_ids].copy()
    current_nodes = unique_nodes.copy()
    n_nodes = len(unique_nodes)

    _, boundary_face_owner = _compute_boundary_faces(hex_node_ids)
    boundary_hex_ids = np.unique(boundary_face_owner) if len(boundary_face_owner) else np.empty((0,), dtype=np.int32)
    node_sdf = sample_sdf(current_nodes)

    # Volume-fraction classify boundary hexes (inside fraction in [0, 1]).
    boundary_hex_nodes = (
        current_nodes[hex_node_ids[boundary_hex_ids]]
        if len(boundary_hex_ids)
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    boundary_vf = (
        calculate_hex_volume_fractions(boundary_hex_nodes, sample_sdf)
        if len(boundary_hex_nodes)
        else np.empty((0,), dtype=np.float64)
    )
    # Mostly outside (<= threshold inside): optional drop. Mostly inside (> threshold inside): keep.
    would_drop_mask = boundary_vf <= vf_cull_threshold if len(boundary_vf) else np.empty((0,), dtype=bool)
    dropped_hex_ids = (
        boundary_hex_ids[would_drop_mask]
        if cull_mostly_external_hexes and len(boundary_hex_ids)
        else np.empty((0,), dtype=np.int32)
    )
    kept_partial_boundary_hex_ids = (
        boundary_hex_ids[boundary_vf > vf_cull_threshold]
        if len(boundary_hex_ids)
        else np.empty((0,), dtype=np.int32)
    )

    keep_hex_mask = np.ones(len(hex_node_ids), dtype=bool)
    if cull_mostly_external_hexes and len(dropped_hex_ids) > 0:
        keep_hex_mask[dropped_hex_ids] = False

    kept_hex_ids = np.where(keep_hex_mask)[0]
    kept_hex_node_ids = hex_node_ids[kept_hex_ids]

    neighbor_stretch_mask = np.zeros(n_nodes, dtype=bool)
    stretch_gate_counts: dict[str, int] = {}
    if neighbor_stretch and len(dropped_hex_ids) > 0:
        reference_centroids = _hex_cell_centroids(original_nodes, hex_node_ids)
        nodes_before_stretch = current_nodes.copy()
        stretch_targets, neighbor_stretch_mask = _stretch_neighbor_nodes_into_dropped_gaps(
            current_nodes,
            hex_node_ids,
            dropped_hex_ids,
            reference_centroids,
            mesh,
            sample_sdf,
        )
        current_nodes = nodes_before_stretch
        if np.any(neighbor_stretch_mask):
            stretch_gate_counts = apply_gated_node_targets(
                current_nodes,
                stretch_targets,
                neighbor_stretch_mask,
                hex_node_ids,
                original_nodes[hex_node_ids],
                keep_hex_mask=keep_hex_mask,
            )
    neighbor_stretch_nodes = int(np.sum(neighbor_stretch_mask))

    # Recompute boundary of the surviving set.
    boundary_node_mask = np.zeros(n_nodes, dtype=bool)
    if len(kept_hex_node_ids) > 0:
        boundary_node_mask = _compute_boundary_node_mask(kept_hex_node_ids, n_nodes)

    # Tiered conform on the new skin (shared with supercell hybrid path).
    node_sdf = sample_sdf(current_nodes)
    closest_pts, _ = closest_points_with_fallback(mesh, current_nodes)
    local_eps_pull = eps_pull if eps_pull is not None else (0.15 * float(target_element_size))
    local_min_jacobian_proxy = min_jacobian_proxy if min_jacobian_proxy is not None else 0.05
    band_inner = 0.5 * float(target_element_size) if boundary_stretch_out else 0.0
    alpha_stretch = 0.28 if boundary_stretch_out else 0.0
    if conformal_dual_mode:
        # Forceful pull-only snap on the skin (no inward stretch-out blend).
        tiered_nodes, tier_counts = apply_tiered_boundary_policy(
            current_nodes,
            closest_pts,
            node_sdf,
            boundary_node_mask,
            eps_pull=local_eps_pull,
            band_inner=0.0,
            alpha_stretch=0.0,
            hex_node_ids=hex_node_ids,
            keep_hex_mask=keep_hex_mask,
            reference_hex_coords=original_nodes[hex_node_ids],
            min_jacobian_proxy=local_min_jacobian_proxy,
        )
    elif shrink_wrap_relax:
        tiered_nodes, tier_counts = apply_tiered_boundary_policy(
            current_nodes,
            closest_pts,
            node_sdf,
            boundary_node_mask,
            eps_pull=local_eps_pull,
            band_inner=band_inner,
            alpha_stretch=alpha_stretch,
            min_jacobian_proxy=local_min_jacobian_proxy,
        )
    else:
        tiered_nodes, tier_counts = apply_tiered_boundary_policy(
            current_nodes,
            closest_pts,
            node_sdf,
            boundary_node_mask,
            eps_pull=local_eps_pull,
            band_inner=band_inner,
            alpha_stretch=alpha_stretch,
            hex_node_ids=hex_node_ids,
            keep_hex_mask=keep_hex_mask,
            reference_hex_coords=original_nodes[hex_node_ids],
            min_jacobian_proxy=local_min_jacobian_proxy,
        )
    current_nodes = tiered_nodes

    stretch_out_count = int(tier_counts.get("conform_stretch_out", 0))
    accepted = int(neighbor_stretch_nodes) + int(
        tier_counts.get("conform_pull_in_full", 0)
        + tier_counts.get("conform_pull_in_soft", 0)
        + stretch_out_count
    )
    proposed_shrink_moves = int(tier_counts.get("conform_pull_in_full", 0))
    proposed_grow_moves = int(
        tier_counts.get("conform_pull_in_soft", 0)
        + stretch_out_count
        + neighbor_stretch_nodes
    )

    if len(kept_hex_node_ids) == 0:
        conformed_hexes = np.empty((0, 8, 3), dtype=np.float64)
    else:
        if int(laplacian_iterations) > 0:
            if conformal_dual_mode:
                compliance_layers = max(1, int(conformal_dual_compliance_layers))
                hex_layer_dist = _hex_layer_distance_from_skin(kept_hex_node_ids)
                laplacian_frozen = _laplacian_frozen_mask_layer_limited(
                    n_nodes,
                    kept_hex_node_ids,
                    boundary_node_mask,
                    hex_layer_dist,
                    smooth_layers_inward=compliance_layers,
                    freeze_boundary_nodes=True,
                )
                laplacian_frozen = np.asarray(laplacian_frozen, dtype=bool) | np.asarray(
                    boundary_node_mask, dtype=bool
                )
                current_nodes = apply_laplacian_smoothing(
                    current_nodes,
                    kept_hex_node_ids,
                    laplacian_frozen,
                    iterations=int(laplacian_iterations),
                    alpha=float(laplacian_alpha),
                    tangential_boundary=False,
                )
            elif shrink_wrap_relax:
                laplacian_frozen = np.asarray(boundary_node_mask, dtype=bool).copy()
                current_nodes = apply_laplacian_smoothing(
                    current_nodes,
                    kept_hex_node_ids,
                    laplacian_frozen,
                    iterations=int(laplacian_iterations),
                    alpha=float(laplacian_alpha),
                    tangential_boundary=False,
                )
            elif laplacian_smooth_layers_inward is not None:
                hex_layers = _hex_layer_distance_from_skin(kept_hex_node_ids)
                laplacian_frozen = _laplacian_frozen_mask_layer_limited(
                    n_nodes,
                    kept_hex_node_ids,
                    boundary_node_mask,
                    hex_layers,
                    smooth_layers_inward=int(laplacian_smooth_layers_inward),
                    freeze_boundary_nodes=False,
                )
                _, surface_normals = closest_points_with_normals(mesh, current_nodes)
                current_nodes = apply_laplacian_smoothing(
                    current_nodes,
                    kept_hex_node_ids,
                    laplacian_frozen,
                    iterations=int(laplacian_iterations),
                    alpha=float(laplacian_alpha),
                    boundary_mask=boundary_node_mask,
                    surface_normals=surface_normals,
                    tangential_boundary=True,
                )
            else:
                laplacian_frozen = np.zeros(n_nodes, dtype=bool)
                _, surface_normals = closest_points_with_normals(mesh, current_nodes)
                current_nodes = apply_laplacian_smoothing(
                    current_nodes,
                    kept_hex_node_ids,
                    laplacian_frozen,
                    iterations=int(laplacian_iterations),
                    alpha=float(laplacian_alpha),
                    boundary_mask=boundary_node_mask,
                    surface_normals=surface_normals,
                    tangential_boundary=True,
                )
        conformed_hexes = current_nodes[kept_hex_node_ids]

    inversion_warning_hexes = 0
    ref_kept = original_nodes[hex_node_ids[kept_hex_ids]]
    prop_kept = current_nodes[hex_node_ids[kept_hex_ids]]
    for ki in range(len(kept_hex_ids)):
        orig_hex = ref_kept[ki]
        prop_hex = prop_kept[ki]
        bad = False
        for c in range(8):
            if not validate_hex_inversion(orig_hex, prop_hex, local_corner_idx=c):
                bad = True
                break
        if bad:
            inversion_warning_hexes += 1
    proposed_moves = int(proposed_shrink_moves + proposed_grow_moves)
    report = {
        "boundary_nodes": int(np.sum(boundary_node_mask)),
        "proposed_moves": proposed_moves,
        "accepted_moves": int(accepted),
        "rejected_moves": int(
            tier_counts.get("conform_line_search_rejected", 0)
            + stretch_gate_counts.get("conform_line_search_rejected", 0)
        ),
        "rejected_cap": 0,
        "rejected_inversion": int(
            tier_counts.get("conform_line_search_rejected", 0)
            + stretch_gate_counts.get("conform_line_search_rejected", 0)
        ),
        "conform_line_search_partial": int(
            tier_counts.get("conform_line_search_partial", 0)
            + stretch_gate_counts.get("conform_line_search_partial", 0)
        ),
        "proposed_shrink_moves": int(proposed_shrink_moves),
        "proposed_grow_moves": int(proposed_grow_moves),
        "rejected_shrink_cap": 0,
        "rejected_shrink_inversion": 0,
        "rejected_grow_cap": 0,
        "rejected_grow_inversion": 0,
        "inversion_warning_hexes": int(inversion_warning_hexes),
        "dropped_hexes": int(len(dropped_hex_ids)),
        "kept_partial_boundary_hexes": int(len(kept_partial_boundary_hex_ids)),
        "neighbor_stretch_nodes": int(neighbor_stretch_nodes),
        # Legacy report keys (grow = dropped mostly-outside, shrink = kept partial boundary).
        "grown_cells": int(len(dropped_hex_ids)),
        "shrunk_cells": int(len(kept_partial_boundary_hex_ids)),
        "grow_hexes": int(len(dropped_hex_ids)),
        "shrink_hexes": int(len(kept_partial_boundary_hex_ids)),
        "handled_inversion_to_shrink_hexes": 0,
        "kept_hexes_after_shrink": int(len(kept_hex_ids)),
        "loose_prefilter_hexes": int(len(hexes)),
        "eps_pull_mm": float(eps_pull),
        "band_inner_mm": float(band_inner),
        "neighbor_stretch_enabled": bool(neighbor_stretch),
        "boundary_stretch_out_enabled": bool(boundary_stretch_out),
        "cull_mostly_external_hexes": bool(cull_mostly_external_hexes),
        "would_drop_hexes": int(np.sum(would_drop_mask)) if len(would_drop_mask) else 0,
        "laplacian_iterations": int(laplacian_iterations),
        "laplacian_smooth_layers_inward": (
            int(laplacian_smooth_layers_inward)
            if laplacian_smooth_layers_inward is not None
            else -1
        ),
        "shrink_wrap_relax": bool(shrink_wrap_relax),
        "conformal_dual_mode": bool(conformal_dual_mode),
        "conformal_dual_compliance_layers": int(conformal_dual_compliance_layers),
    }
    if not return_debug_payload:
        return conformed_hexes, report

    ordered_boundary_faces = _compute_ordered_boundary_faces(kept_hex_node_ids)
    debug_payload = {
        "nodes": current_nodes.copy(),
        "kept_hex_node_ids": kept_hex_node_ids.copy(),
        "boundary_faces": ordered_boundary_faces.copy(),
        "boundary_node_mask": boundary_node_mask.copy(),
        "pre_conform_hex_elements": pre_conform_hex_elements.copy(),
    }
    return conformed_hexes, report, debug_payload


def _structured_hex_elements_from_node_grid(
    nodes: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """Build ``(nx*ny*nz, 8, 3)`` bricks from a ``(nx+1, ny+1, nz+1, 3)`` node lattice."""
    corners = (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )
    out: list[np.ndarray] = []
    for i in range(int(nx)):
        for j in range(int(ny)):
            for k in range(int(nz)):
                brick = np.array(
                    [nodes[i + di, j + dj, k + dk] for di, dj, dk in corners],
                    dtype=np.float64,
                )
                out.append(brick)
    return np.array(out, dtype=np.float64) if out else np.empty((0, 8, 3), dtype=np.float64)


def _fixed_grid_shell_and_core_hex_ids(
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Global hex indices for exterior shell (skin) and interior (volume)."""
    skin: list[int] = []
    core: list[int] = []
    gid = 0
    for i in range(int(nx)):
        for j in range(int(ny)):
            for k in range(int(nz)):
                on_shell = (
                    i == 0
                    or i == nx - 1
                    or j == 0
                    or j == ny - 1
                    or k == 0
                    or k == nz - 1
                )
                if on_shell:
                    skin.append(gid)
                else:
                    core.append(gid)
                gid += 1
    return (
        np.asarray(core, dtype=np.int32),
        np.asarray(skin, dtype=np.int32),
    )


def _mesh_section_extents_on_plane(
    mesh: trimesh.Trimesh,
    axis: int,
    value: float,
    out_axes: tuple[int, int],
    *,
    fallback_lo: np.ndarray,
    fallback_hi: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Intersect mesh triangles with a plane and return extents along ``out_axes``.

    ``axis`` is 0=X, 1=Y, 2=Z fixed at ``value``.
    """
    ax = int(axis)
    oa, ob = int(out_axes[0]), int(out_axes[1])
    coord = float(value)
    pts: list[np.ndarray] = []
    for tri in np.asarray(mesh.triangles, dtype=np.float64):
        vals = tri[:, ax]
        if vals.min() <= coord <= vals.max():
            for e0, e1 in ((0, 1), (1, 2), (2, 0)):
                a, b = tri[e0], tri[e1]
                da, db = float(a[ax] - coord), float(b[ax] - coord)
                if da * db <= 0.0 and abs(float(a[ax] - b[ax])) > 1e-12:
                    t = (coord - float(a[ax])) / (float(b[ax]) - float(a[ax]))
                    pts.append(a + t * (b - a))
    if len(pts) < 2:
        return (
            float(fallback_lo[oa]),
            float(fallback_hi[oa]),
            float(fallback_lo[ob]),
            float(fallback_hi[ob]),
        )
    p = np.vstack(pts)
    return (
        float(p[:, oa].min()),
        float(p[:, oa].max()),
        float(p[:, ob].min()),
        float(p[:, ob].max()),
    )


def generate_brute_force_fixed_grid_hex_scaffold(
    mesh: trimesh.Trimesh,
    *,
    nx: int = 8,
    ny: int = 4,
    nz: int = 4,
    taper_along: str = "z",
    snap_surface_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | bool | np.ndarray]]:
    """
    Trophy-style brute-force scaffold: fixed ``nx x ny x nz`` topology (no VF cull).

    - **128 bricks always** — cells deform; none are dropped.
    - Uniform **Z** layer spacing ``dz = (z_max - z_min) / nz`` on every column.
    - Lateral spacing shrinks with taper:
      - ``taper_along='z'``: at each Z layer, X/Y extents from mesh plane slice.
      - ``taper_along='y'``: uniform Y pitch; X and Z extents from mesh slice at each
        depth (Trophy_base_thin on its side — ramp shrinks X and Z, not Y).

    Returns ``(volume_hex_elements, skin_hex_elements, report)`` for
    ``synthesize_two_branch_hex_lattice`` (octahedral in all cells).
    """
    nx_i, ny_i, nz_i = int(nx), int(ny), int(nz)
    if min(nx_i, ny_i, nz_i) < 1:
        raise ValueError("nx, ny, nz must each be >= 1.")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    dz = float(hi[2] - lo[2]) / float(nz_i)
    dy = float(hi[1] - lo[1]) / float(ny_i)

    nodes = np.zeros((nx_i + 1, ny_i + 1, nz_i + 1, 3), dtype=np.float64)
    taper = str(taper_along).strip().lower()
    if taper not in ("z", "y"):
        raise ValueError("taper_along must be 'z' or 'y'.")

    if taper == "z":
        for k in range(nz_i + 1):
            z = float(lo[2]) + float(k) * dz
            x0, x1, y0, y1 = _mesh_section_extents_on_plane(
                mesh, 2, z, (0, 1), fallback_lo=lo, fallback_hi=hi
            )
            for j in range(ny_i + 1):
                yv = float(lo[1]) + float(j) * dy
                for i in range(nx_i + 1):
                    nodes[i, j, k, 0] = x0 + float(i) * (x1 - x0) / float(nx_i)
                    nodes[i, j, k, 1] = yv
                    nodes[i, j, k, 2] = z
    else:
        for j in range(ny_i + 1):
            yv = float(lo[1]) + float(j) * dy
            x0, x1, z0, z1 = _mesh_section_extents_on_plane(
                mesh, 1, yv, (0, 2), fallback_lo=lo, fallback_hi=hi
            )
            z_span = max(float(z1) - float(z0), 1e-9)
            for k in range(nz_i + 1):
                z = float(z0) + float(k) * z_span / float(nz_i)
                for i in range(nx_i + 1):
                    nodes[i, j, k, 0] = x0 + float(i) * (x1 - x0) / float(nx_i)
                    nodes[i, j, k, 1] = yv
                    nodes[i, j, k, 2] = z

    if snap_surface_nodes:
        for i in range(nx_i + 1):
            for j in range(ny_i + 1):
                for k in range(nz_i + 1):
                    if i not in (0, nx_i) and j not in (0, ny_i) and k not in (0, nz_i):
                        continue
                    pt = nodes[i, j, k].reshape(1, 3)
                    snapped, _d, _ = trimesh.proximity.closest_point(mesh, pt)
                    s = np.asarray(snapped[0], dtype=np.float64)
                    nodes[i, j, k, 0] = float(s[0])
                    nodes[i, j, k, 1] = float(s[1])
                    if taper == "z":
                        if k == nz_i:
                            nodes[i, j, k, 2] = float(s[2])
                        elif k == 0:
                            nodes[i, j, k, 2] = float(lo[2])
                        else:
                            nodes[i, j, k, 2] = float(lo[2]) + float(k) * dz
                    else:
                        yv = float(lo[1]) + float(j) * dy
                        _, _, z0, z1 = _mesh_section_extents_on_plane(
                            mesh, 1, yv, (0, 2), fallback_lo=lo, fallback_hi=hi
                        )
                        z_span = max(float(z1) - float(z0), 1e-9)
                        nodes[i, j, k, 2] = float(z0) + float(k) * z_span / float(nz_i)

    all_hexes = _structured_hex_elements_from_node_grid(nodes, nx_i, ny_i, nz_i)
    _core_ids, shell_ids = _fixed_grid_shell_and_core_hex_ids(nx_i, ny_i, nz_i)
    n_hex = int(len(all_hexes))
    all_gids = np.arange(n_hex, dtype=np.int32)
    # Octahedral core in every grid cell (including the one-cell-thick "shell").
    # Surface-path dual still owns the exterior envelope; shell octahedral struts
    # on the part boundary are suppressed at synthesis (no duplicate skin cage).
    volume_hexes = all_hexes.copy()
    skin_hexes = all_hexes.copy()

    report: dict[str, int | float | bool | np.ndarray] = {
        "brute_force_fixed_grid": True,
        "grid_nx": nx_i,
        "grid_ny": ny_i,
        "grid_nz": nz_i,
        "n_hexes_total": n_hex,
        "volume_hexes": n_hex,
        "skin_hexes": n_hex,
        "volume_interior_only_hexes": int(len(_core_ids)),
        "skin_shell_hexes": int(len(shell_ids)),
        "dz_mm": float(dz),
        "dx_bottom_mm": float((hi[0] - lo[0]) / nx_i),
        "dy_bottom_mm": float((hi[1] - lo[1]) / ny_i),
        "taper_along": taper,
        "snap_surface_nodes": bool(snap_surface_nodes),
        "volume_global_hex_ids": all_gids,
        "skin_global_hex_ids": all_gids,
    }
    return volume_hexes, skin_hexes, report


def synthesize_brute_force_kelvin14_lattice(
    hex_elements: np.ndarray,
    *,
    rule_name: str = "kelvin14",
    topology_round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """
    Volume-only Kelvin lattice on a fixed conformal hex scaffold.

    Uses the 24-node / 36-strut truncated octahedron (``kelvin_cell.generate_kelvin_cell``)
    mapped into each hex brick via trilinear shape functions. No separate surface-path dual.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    nodes, struts = generate_hex_topology(
        elems,
        rule_name=str(rule_name),
        round_decimals=int(topology_round_decimals),
    )
    report: dict[str, int | str] = {
        "merge_mode": "kelvin14_volume_only",
        "hex_rule": str(rule_name),
        "n_hexes": int(elems.shape[0]),
        "n_nodes": int(len(nodes)),
        "n_struts": int(len(struts)),
        "surface_dual": "skipped",
    }
    return nodes, struts, report


def synthesize_brute_force_tesseract_lattice(
    hex_elements: np.ndarray,
    *,
    rule_name: str = "tesseract",
    topology_round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str]]:
    """
    Volume-only tesseract lattice on a fixed conformal hex scaffold.

    Uses the 16-node / 32-strut nested hypercube projection
    (``tesseract_cell.generate_tesseract_cell``) mapped into each hex brick.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    nodes, struts = generate_hex_topology(
        elems,
        rule_name=str(rule_name),
        round_decimals=int(topology_round_decimals),
    )
    report: dict[str, int | str] = {
        "merge_mode": "tesseract_volume_only",
        "hex_rule": str(rule_name),
        "n_hexes": int(elems.shape[0]),
        "n_nodes": int(len(nodes)),
        "n_struts": int(len(struts)),
        "surface_dual": "skipped",
    }
    return nodes, struts, report


def generate_conformed_hex_scaffold_two_branch(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    *,
    grid_anchor: str = "bbox_center",
    cull_mostly_external_hexes: bool = True,
    neighbor_stretch: bool = True,
    laplacian_iterations: int = 5,
    laplacian_alpha: float = 0.4,
    conformal_dual_compliance_layers: int = CONFORMAL_DUAL_COMPLIANCE_LAYERS,
    skin_tangential_fairing: bool = True,
    skin_fairing_iterations: int = 3,
    skin_fairing_alpha: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | np.ndarray]]:
    """
    Two-branch conformal hex scaffold (Phase 18).

    **Volume branch:** hexes with >50% interior volume (VF cull) — for octahedral core.
    **Skin branch:** all boundary-band hexes (including would-drop), fully snapped /
    stretched onto the CAD — for surface-path dual only (no volume struts inside
    dropped cells).

    Returns:
        volume_hex_elements: ``(N_vol, 8, 3)``
        skin_hex_elements: ``(N_skin, 8, 3)`` (includes VF-dropped boundary cells)
        report: includes ``volume_global_hex_ids`` and ``skin_global_hex_ids`` for synthesis
    """
    empty = np.empty((0, 8, 3), dtype=np.float64)
    base: dict[str, int | np.ndarray] = {
        "volume_hexes": 0,
        "skin_hexes": 0,
        "dropped_hexes": 0,
    }

    all_hexes = _generate_bbox_hex_grid(mesh, target_element_size, grid_anchor=grid_anchor)
    if len(all_hexes) == 0:
        return empty, empty, dict(base)

    resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, resolution)
    loose_keep = sample_sdf(np.mean(all_hexes, axis=1)) < float(target_element_size)
    hexes = all_hexes[loose_keep]
    if len(hexes) == 0:
        return empty, empty, dict(base)

    unique_nodes, hex_node_ids = _build_unique_node_representation(hexes)
    original_nodes = unique_nodes.copy()
    current_nodes = unique_nodes.copy()
    n_nodes = len(unique_nodes)

    _, boundary_face_owner = _compute_boundary_faces(hex_node_ids)
    boundary_hex_ids = (
        np.unique(boundary_face_owner) if len(boundary_face_owner) else np.empty((0,), dtype=np.int32)
    )
    boundary_hex_nodes = (
        current_nodes[hex_node_ids[boundary_hex_ids]]
        if len(boundary_hex_ids)
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    boundary_vf = (
        calculate_hex_volume_fractions(boundary_hex_nodes, sample_sdf)
        if len(boundary_hex_nodes)
        else np.empty((0,), dtype=np.float64)
    )
    would_drop_mask = boundary_vf <= 0.5 if len(boundary_vf) else np.empty((0,), dtype=bool)
    dropped_hex_ids = (
        boundary_hex_ids[would_drop_mask]
        if cull_mostly_external_hexes and len(boundary_hex_ids)
        else np.empty((0,), dtype=np.int32)
    )

    keep_hex_mask = np.ones(len(hex_node_ids), dtype=bool)
    if cull_mostly_external_hexes and len(dropped_hex_ids) > 0:
        keep_hex_mask[dropped_hex_ids] = False

    kept_hex_ids = np.where(keep_hex_mask)[0]
    skin_hex_ids = np.asarray(boundary_hex_ids, dtype=np.int32)
    skin_hex_node_ids = hex_node_ids[skin_hex_ids]

    stretch_gate_counts: dict[str, int] = {}
    neighbor_stretch_nodes = 0
    if neighbor_stretch and len(dropped_hex_ids) > 0:
        reference_centroids = _hex_cell_centroids(original_nodes, hex_node_ids)
        nodes_before_stretch = current_nodes.copy()
        stretch_targets, neighbor_stretch_mask = _stretch_neighbor_nodes_into_dropped_gaps(
            current_nodes,
            hex_node_ids,
            dropped_hex_ids,
            reference_centroids,
            mesh,
            sample_sdf,
        )
        current_nodes = nodes_before_stretch
        if np.any(neighbor_stretch_mask):
            stretch_gate_counts = apply_gated_node_targets(
                current_nodes,
                stretch_targets,
                neighbor_stretch_mask,
                hex_node_ids,
                original_nodes[hex_node_ids],
                keep_hex_mask=keep_hex_mask,
            )
        neighbor_stretch_nodes = int(np.sum(neighbor_stretch_mask))

    skin_boundary_node_mask = _compute_boundary_node_mask(skin_hex_node_ids, n_nodes)
    closest_pts, _ = closest_points_with_fallback(mesh, current_nodes)
    node_sdf = sample_sdf(current_nodes)
    eps_pull = 0.15 * float(target_element_size)
    tiered_nodes, tier_counts = apply_tiered_boundary_policy(
        current_nodes,
        closest_pts,
        node_sdf,
        skin_boundary_node_mask,
        eps_pull=eps_pull,
        band_inner=0.0,
        alpha_stretch=0.0,
    )
    current_nodes = tiered_nodes

    if skin_tangential_fairing and int(skin_fairing_iterations) > 0:
        _, surface_normals = closest_points_with_normals(mesh, current_nodes)
        skin_frozen = ~np.asarray(skin_boundary_node_mask, dtype=bool)
        current_nodes = apply_laplacian_smoothing(
            current_nodes,
            skin_hex_node_ids,
            skin_frozen,
            iterations=int(skin_fairing_iterations),
            alpha=float(skin_fairing_alpha),
            boundary_mask=skin_boundary_node_mask,
            surface_normals=surface_normals,
            tangential_boundary=True,
        )

    kept_hex_node_ids = hex_node_ids[kept_hex_ids]
    if int(laplacian_iterations) > 0 and len(kept_hex_node_ids) > 0:
        compliance_layers = max(1, int(conformal_dual_compliance_layers))
        hex_layer_dist = _hex_layer_distance_from_skin(kept_hex_node_ids)
        laplacian_frozen = _laplacian_frozen_mask_layer_limited(
            n_nodes,
            kept_hex_node_ids,
            skin_boundary_node_mask,
            hex_layer_dist,
            smooth_layers_inward=compliance_layers,
            freeze_boundary_nodes=True,
        )
        laplacian_frozen = np.asarray(laplacian_frozen, dtype=bool) | np.asarray(
            skin_boundary_node_mask, dtype=bool
        )
        current_nodes = apply_laplacian_smoothing(
            current_nodes,
            kept_hex_node_ids,
            laplacian_frozen,
            iterations=int(laplacian_iterations),
            alpha=float(laplacian_alpha),
            tangential_boundary=False,
        )

    volume_hex_elements = (
        current_nodes[kept_hex_node_ids].copy()
        if len(kept_hex_ids)
        else empty
    )
    skin_hex_elements = (
        current_nodes[hex_node_ids[skin_hex_ids]].copy()
        if len(skin_hex_ids)
        else empty
    )

    inversion_volume = 0
    inversion_skin = 0
    if len(kept_hex_ids):
        ref_v = original_nodes[hex_node_ids[kept_hex_ids]]
        prop_v = current_nodes[hex_node_ids[kept_hex_ids]]
        for ki in range(len(kept_hex_ids)):
            if any(
                not validate_hex_inversion(ref_v[ki], prop_v[ki], local_corner_idx=c)
                for c in range(8)
            ):
                inversion_volume += 1
    if len(skin_hex_ids):
        ref_s = original_nodes[hex_node_ids[skin_hex_ids]]
        prop_s = current_nodes[hex_node_ids[skin_hex_ids]]
        for si in range(len(skin_hex_ids)):
            if any(
                not validate_hex_inversion(ref_s[si], prop_s[si], local_corner_idx=c)
                for c in range(8)
            ):
                inversion_skin += 1

    report: dict[str, int | np.ndarray] = {
        "two_branch_mode": True,
        "volume_hexes": int(len(kept_hex_ids)),
        "skin_hexes": int(len(skin_hex_ids)),
        "dropped_hexes": int(len(dropped_hex_ids)),
        "neighbor_stretch_nodes": int(neighbor_stretch_nodes),
        "inversion_warning_hexes_volume": int(inversion_volume),
        "inversion_warning_hexes_skin": int(inversion_skin),
        "conform_pull_in_full": int(tier_counts.get("conform_pull_in_full", 0)),
        "volume_global_hex_ids": np.asarray(kept_hex_ids, dtype=np.int32),
        "skin_global_hex_ids": np.asarray(skin_hex_ids, dtype=np.int32),
        "target_element_size_mm": float(target_element_size),
    }
    return volume_hex_elements, skin_hex_elements, report


def _merge_two_branch_volume_and_skin_paths(
    vol_nodes: np.ndarray,
    vol_struts: np.ndarray,
    boundary_face_to_vol_node: dict[tuple[int, int], int],
    skin_path_nodes: np.ndarray,
    skin_path_struts: np.ndarray,
    quad_topology: BoundaryQuadTopology,
    skin_global_hex_ids: np.ndarray,
    volume_global_hex_ids: np.ndarray,
    n_path_face_centroids: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Decoupled graphs: core (volume + bridges) and skin (path dual only).

    ``skin_path_nodes`` first ``n_path_face_centroids`` entries are quad centroids.
    """
    global_to_vol_row = {int(g): i for i, g in enumerate(volume_global_hex_ids)}
    skin_gid = np.asarray(skin_global_hex_ids, dtype=np.int32)

    nodes_list = [np.asarray(p, dtype=np.float64) for p in vol_nodes]
    skin_to_global: dict[int, int] = {}
    n_bridges = 0
    n_mapped = 0

    for qi in range(quad_topology.n_quads):
        skin_hi, fi = quad_topology.quad_to_hex_face(qi)
        skin_global = int(skin_gid[int(skin_hi)])
        vol_row = global_to_vol_row.get(skin_global)
        vol_id = (
            boundary_face_to_vol_node.get((int(vol_row), int(fi)))
            if vol_row is not None
            else None
        )
        if qi >= n_path_face_centroids:
            continue
        skin_local = int(qi)
        if vol_id is not None:
            skin_to_global[skin_local] = int(vol_id)
            n_mapped += 1
        elif skin_local not in skin_to_global:
            skin_to_global[skin_local] = len(nodes_list)
            nodes_list.append(np.asarray(skin_path_nodes[skin_local], dtype=np.float64))

    for si in range(min(n_path_face_centroids, len(skin_path_nodes))):
        if si in skin_to_global:
            continue
        skin_to_global[si] = len(nodes_list)
        nodes_list.append(np.asarray(skin_path_nodes[si], dtype=np.float64))

    nodes_core = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    core_set: set[tuple[int, int]] = set()
    for a, b in np.asarray(vol_struts, dtype=np.int64):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        core_set.add((ia, ib) if ia < ib else (ib, ia))
    for qi in range(quad_topology.n_quads):
        if qi >= n_path_face_centroids:
            continue
        skin_hi, fi = quad_topology.quad_to_hex_face(qi)
        vol_row = global_to_vol_row.get(int(skin_gid[int(skin_hi)]))
        vol_id = (
            boundary_face_to_vol_node.get((int(vol_row), int(fi)))
            if vol_row is not None
            else None
        )
        if vol_id is None:
            continue
        skin_id = skin_to_global.get(int(qi))
        if skin_id is None or int(vol_id) == int(skin_id):
            continue
        a, b = int(vol_id), int(skin_id)
        core_set.add((a, b) if a < b else (b, a))
        n_bridges += 1

    struts_core = (
        np.array(sorted(core_set), dtype=np.int64)
        if core_set
        else np.empty((0, 2), dtype=np.int64)
    )

    nodes_skin = np.asarray(skin_path_nodes, dtype=np.float64).copy()
    struts_skin = np.asarray(skin_path_struts, dtype=np.int64).copy()

    return nodes_core, struts_core, nodes_skin, struts_skin, {
        "bridge_struts": int(n_bridges),
        "n_skin_quads_mapped_to_volume": int(n_mapped),
        "struts_core": int(len(struts_core)),
        "skin_struts": int(len(struts_skin)),
    }


def synthesize_two_branch_hex_lattice(
    mesh: trimesh.Trimesh,
    volume_hex_elements: np.ndarray,
    skin_hex_elements: np.ndarray,
    *,
    volume_global_hex_ids: np.ndarray,
    skin_global_hex_ids: np.ndarray,
    target_element_size: float | None = None,
    topology_round_decimals: int = 6,
    include_sliver_cages: bool = False,
    suppress_volume_exterior_shell_struts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Phase 18 synthesis: VF-gated octahedral core + surface-path skin on deformed envelope.

    Returns ``(nodes_core, struts_core, nodes_skin, struts_skin, report)`` for
    decoupled export (cropped core + curved/ribbon skin).

    Skin uses surface-path dual only (no integer face-center dual). Volume struts
    on the exterior shell are dropped by default so core octahedral spokes do not
    stack on the path dual.
    """
    vol_elems = np.asarray(volume_hex_elements, dtype=np.float64)
    skin_elems = np.asarray(skin_hex_elements, dtype=np.float64)
    vol_gids = np.asarray(volume_global_hex_ids, dtype=np.int32).ravel()
    skin_gids = np.asarray(skin_global_hex_ids, dtype=np.int32).ravel()

    if vol_elems.shape[0] != vol_gids.shape[0]:
        raise ValueError("volume_global_hex_ids must match volume_hex_elements rows.")
    if skin_elems.shape[0] != skin_gids.shape[0]:
        raise ValueError("skin_global_hex_ids must match skin_hex_elements rows.")

    volume_global_set = set(int(g) for g in vol_gids)

    corner_nodes, skin_hex_node_ids = hex_node_ids_from_elements(
        skin_elems, round_decimals=int(topology_round_decimals)
    )
    quads, quad_owners, _quad_face = extract_ordered_boundary_quads_with_owners(
        skin_hex_node_ids
    )
    hex_has_internal = np.array(
        [int(skin_gids[int(hi)]) in volume_global_set for hi in quad_owners],
        dtype=bool,
    )

    path_nodes, path_struts, path_rep = generate_hex_surface_dual_on_surface_paths(
        corner_nodes,
        quads,
        quad_owners,
        hex_has_internal,
        target_element_size=target_element_size,
        include_sliver_element_cage=bool(include_sliver_cages),
        max_centroid_span_factor=None,
    )
    n_face_centroids = int(path_rep.get("n_boundary_quads", quads.shape[0]))

    vol_nodes, vol_struts, boundary_face_to_vol = (
        generate_hex_octahedral_volume_with_boundary_face_map(
            vol_elems,
            volume_emit_mask=np.ones(vol_elems.shape[0], dtype=bool),
            round_decimals=int(topology_round_decimals),
        )
    )
    n_vol_struts_raw = int(len(vol_struts))
    n_shell_suppressed = 0
    if suppress_volume_exterior_shell_struts and len(boundary_face_to_vol) > 0:
        shell_nodes = frozenset(int(v) for v in boundary_face_to_vol.values())
        vol_struts, n_shell_suppressed = filter_struts_drop_exterior_shell_pairs(
            vol_struts, shell_nodes
        )

    quad_topology = build_boundary_quad_topology(skin_hex_node_ids)
    nodes_c, struts_c, nodes_s, struts_s, merge_rep = _merge_two_branch_volume_and_skin_paths(
        vol_nodes,
        vol_struts,
        boundary_face_to_vol,
        path_nodes,
        path_struts,
        quad_topology,
        skin_gids,
        vol_gids,
        n_face_centroids,
    )

    report: dict[str, int | float | str] = {
        "merge_mode": "two_branch_volume_surface_paths",
        "n_volume_hexes": int(vol_elems.shape[0]),
        "n_skin_hexes": int(skin_elems.shape[0]),
        "volume_struts": int(len(vol_struts)),
        "n_volume_struts_before_shell_filter": int(n_vol_struts_raw),
        "n_volume_exterior_shell_struts_suppressed": int(n_shell_suppressed),
        "suppress_volume_exterior_shell_struts": bool(suppress_volume_exterior_shell_struts),
        "include_sliver_element_cage": bool(include_sliver_cages),
        "integer_surface_dual_on_skin": False,
        **path_rep,
        **merge_rep,
    }
    return nodes_c, struts_c, nodes_s, struts_s, report


def generate_conformed_hex_scaffold_deformation_stages(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    *,
    grid_anchor: str = "bbox_center",
    cull_mostly_external_hexes: bool = True,
    laplacian_iterations: int = 5,
    laplacian_alpha: float = 0.4,
    conformal_dual_compliance_layers: int = CONFORMAL_DUAL_COMPLIANCE_LAYERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """
    Conformal-dual deformation checkpoints for hex wireframe diagnostics.

    Returns hex element bricks ``(N, 8, 3)`` at three pipeline stops:

    1. **Pre-deformation** — after bbox grid + optional 50% VF cull; no snap/smooth.
    2. **Post-snap** — after forceful conformal-dual boundary pull; no Laplacian.
    3. **Post-smooth** — after compliance-layer Laplacian (layers 1..N inward).

    Neighbor stretch and boundary stretch-out are disabled to isolate snap + smooth.
    """
    empty = np.empty((0, 8, 3), dtype=np.float64)
    base_meta: dict[str, int] = {"kept_hexes": 0, "dropped_hexes": 0}

    all_hexes = _generate_bbox_hex_grid(mesh, target_element_size, grid_anchor=grid_anchor)
    if len(all_hexes) == 0:
        return empty, empty, empty, base_meta

    resolution = min(2.0, max(0.5, np.max(mesh.extents) / 100.0))
    sample_sdf = build_edt_sdf_sampler(mesh, resolution)
    all_centroids = np.mean(all_hexes, axis=1)
    loose_keep = sample_sdf(all_centroids) < float(target_element_size)
    hexes = all_hexes[loose_keep]
    if len(hexes) == 0:
        return empty, empty, empty, base_meta

    unique_nodes, hex_node_ids = _build_unique_node_representation(hexes)
    original_nodes = unique_nodes.copy()
    current_nodes = unique_nodes.copy()
    n_nodes = len(unique_nodes)

    _, boundary_face_owner = _compute_boundary_faces(hex_node_ids)
    boundary_hex_ids = (
        np.unique(boundary_face_owner) if len(boundary_face_owner) else np.empty((0,), dtype=np.int32)
    )
    boundary_hex_nodes = (
        current_nodes[hex_node_ids[boundary_hex_ids]]
        if len(boundary_hex_ids)
        else np.empty((0, 8, 3), dtype=np.float64)
    )
    boundary_vf = (
        calculate_hex_volume_fractions(boundary_hex_nodes, sample_sdf)
        if len(boundary_hex_nodes)
        else np.empty((0,), dtype=np.float64)
    )
    would_drop_mask = boundary_vf <= 0.5 if len(boundary_vf) else np.empty((0,), dtype=bool)
    dropped_hex_ids = (
        boundary_hex_ids[would_drop_mask]
        if cull_mostly_external_hexes and len(boundary_hex_ids)
        else np.empty((0,), dtype=np.int32)
    )

    keep_hex_mask = np.ones(len(hex_node_ids), dtype=bool)
    if cull_mostly_external_hexes and len(dropped_hex_ids) > 0:
        keep_hex_mask[dropped_hex_ids] = False

    kept_hex_ids = np.where(keep_hex_mask)[0]
    kept_hex_node_ids = hex_node_ids[kept_hex_ids]
    if len(kept_hex_node_ids) == 0:
        meta = {**base_meta, "dropped_hexes": int(len(dropped_hex_ids))}
        return empty, empty, empty, meta

    stage1_hexes = original_nodes[kept_hex_node_ids].copy()

    boundary_node_mask = _compute_boundary_node_mask(kept_hex_node_ids, n_nodes)
    node_sdf = sample_sdf(current_nodes)
    closest_pts, _ = closest_points_with_fallback(mesh, current_nodes)
    eps_pull = 0.15 * float(target_element_size)
    tiered_nodes, tier_counts = apply_tiered_boundary_policy(
        current_nodes,
        closest_pts,
        node_sdf,
        boundary_node_mask,
        eps_pull=eps_pull,
        band_inner=0.0,
        alpha_stretch=0.0,
    )
    current_nodes = tiered_nodes
    stage2_hexes = current_nodes[kept_hex_node_ids].copy()

    if int(laplacian_iterations) > 0:
        compliance_layers = max(1, int(conformal_dual_compliance_layers))
        hex_layer_dist = _hex_layer_distance_from_skin(kept_hex_node_ids)
        laplacian_frozen = _laplacian_frozen_mask_layer_limited(
            n_nodes,
            kept_hex_node_ids,
            boundary_node_mask,
            hex_layer_dist,
            smooth_layers_inward=compliance_layers,
            freeze_boundary_nodes=True,
        )
        laplacian_frozen = np.asarray(laplacian_frozen, dtype=bool) | np.asarray(
            boundary_node_mask, dtype=bool
        )
        current_nodes = apply_laplacian_smoothing(
            current_nodes,
            kept_hex_node_ids,
            laplacian_frozen,
            iterations=int(laplacian_iterations),
            alpha=float(laplacian_alpha),
            tangential_boundary=False,
        )

    stage3_hexes = current_nodes[kept_hex_node_ids].copy()

    inversion_warning_hexes = 0
    ref_kept = original_nodes[hex_node_ids[kept_hex_ids]]
    prop_kept = current_nodes[hex_node_ids[kept_hex_ids]]
    for ki in range(len(kept_hex_ids)):
        orig_hex = ref_kept[ki]
        prop_hex = prop_kept[ki]
        for c in range(8):
            if not validate_hex_inversion(orig_hex, prop_hex, local_corner_idx=c):
                inversion_warning_hexes += 1
                break

    meta = {
        "kept_hexes": int(len(kept_hex_ids)),
        "dropped_hexes": int(len(dropped_hex_ids)),
        "boundary_nodes": int(np.sum(boundary_node_mask)),
        "conform_pull_in_full": int(tier_counts.get("conform_pull_in_full", 0)),
        "inversion_warning_hexes": int(inversion_warning_hexes),
        "laplacian_iterations": int(laplacian_iterations),
        "conformal_dual_compliance_layers": int(conformal_dual_compliance_layers),
    }
    return stage1_hexes, stage2_hexes, stage3_hexes, meta
