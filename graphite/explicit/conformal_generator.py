"""
Graphite Conformal Lattice Engine V2 — Unified Conformal Generator

Supports both:
- A15: Conformed Tetrahedral template grid + Kagome dual routing.
- SC: Conformed Simple Cubic template grid + Octahedral dual routing.

No GMSH dependency. 100% GMSH-free watertight meshing.
"""

from __future__ import annotations
import os
import time
import tempfile
import warnings
from pathlib import Path
from collections import defaultdict, deque
from itertools import combinations
from typing import NamedTuple, Callable

import numpy as np
import trimesh

from .mesh_repair import repair_cad_mesh
from .sizing_solver import solve_sizing
from .geometry_module import union_lattice_with_spherical_joints, boolean_intersect_with_cad


class ScaffoldResult(NamedTuple):
    """
    Structured return type for conformed background grid scaffolds.
    """
    nodes: np.ndarray
    elements: np.ndarray
    surface_faces: np.ndarray
    element_order: int = 1


# Face definitions
TET_FACE_TRIPLETS = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
HEX_FACE_QUADS = np.array([
    [0, 1, 2, 3],  # Bottom (Z=0)
    [4, 5, 6, 7],  # Top (Z=1)
    [0, 1, 5, 4],  # Front (Y=0)
    [1, 2, 6, 5],  # Right (X=1)
    [2, 3, 7, 6],  # Back (Y=1)
    [3, 0, 4, 7],  # Left (X=0)
], dtype=np.int64)

# Adjacent face pairs (sharing an edge)
TET_ADJACENT_FACES = list(combinations(range(4), 2))
HEX_ADJACENT_FACES = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 5),
    (4, 3), (4, 5)
]

LATTICE_CONFIGS = {
    "A15": {
        "faces": TET_FACE_TRIPLETS,
        "adjacent_faces": TET_ADJACENT_FACES,
        "unit_cell_multiplier": 4.0,
        "boundary_valency_cutoff": 3,
        "clique_size": 4,
    },
    "SC": {
        "faces": HEX_FACE_QUADS,
        "adjacent_faces": HEX_ADJACENT_FACES,
        "unit_cell_multiplier": 1.0,
        "boundary_valency_cutoff": 4,
        "clique_size": 8,
    }
}


def _pt_key(pt: np.ndarray) -> tuple[int, int, int]:
    return (int(np.round(pt[0] * 2000)), int(np.round(pt[1] * 2000)), int(np.round(pt[2] * 2000)))


def _coord_face_key(nodes: np.ndarray, cell: np.ndarray, face_indices: np.ndarray) -> frozenset[tuple[int, int, int]]:
    face_nodes = cell[face_indices]
    coords = nodes[face_nodes]
    return frozenset(_pt_key(c) for c in coords)


def safe_signed_distance(cad_mesh: trimesh.Trimesh, points: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """
    Computes signed distance of points to the CAD mesh using ray-casting.
    Negative means inside, positive means outside.
    """
    n_pts = len(points)
    s_dists = np.zeros(n_pts, dtype=np.float64)
    proximity = trimesh.proximity.ProximityQuery(cad_mesh)

    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        chunk = points[start:end]
        dists = proximity.signed_distance(chunk)
        # trimesh signed_distance returns positive for inside, negative for outside.
        # We invert it: negative inside, positive outside.
        s_dists[start:end] = -dists

    return s_dists


def project_to_cad_surface(nodes: np.ndarray, cad_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects nodes to the closest points on the CAD mesh surface.
    Returns (closest_points, is_inside).
    """
    proximity = trimesh.proximity.ProximityQuery(cad_mesh)
    closest_pts, _, _ = proximity.on_surface(nodes)
    # Check inside status
    inside = proximity.signed_distance(nodes) >= -1e-5
    return closest_pts, inside


def apply_sdf_ironing(
    nodes_3d: np.ndarray,
    struts: np.ndarray,
    boundary_node_ids: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    valency_cutoff: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Applies the valency-based boundary snapping (SDF Ironing) rule.
    """
    # Count node valencies
    degrees = np.zeros(len(nodes_3d), dtype=np.int32)
    for u, v in struts:
        degrees[u] += 1
        degrees[v] += 1

    # Project boundary nodes to CAD surface
    boundary_coords = nodes_3d[boundary_node_ids]
    closest_pts, is_inside = project_to_cad_surface(boundary_coords, cad_mesh)

    nodes_ironed = nodes_3d.copy()
    conformed_mask = np.zeros(len(nodes_3d), dtype=bool)
    conformed_nodes_map = {}

    for local_i, idx in enumerate(boundary_node_ids):
        inside = is_inside[local_i]
        valency = degrees[idx]

        conformed = False
        if not inside:
            # Outside: always snap
            conformed = True
        else:
            # Inside: snap only if valency <= cutoff
            conformed = (valency <= valency_cutoff)

        if conformed:
            snap_pt = closest_pts[local_i]
            nodes_ironed[idx] = snap_pt
            conformed_mask[idx] = True
            conformed_nodes_map[idx] = snap_pt

    return nodes_ironed, conformed_mask, conformed_nodes_map


def apply_depth_gated_relaxation(
    nodes_ironed: np.ndarray,
    struts: np.ndarray,
    node_depths: np.ndarray,
    iterations: int = 15,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Applies depth-gated Laplacian smoothing to prevent interior collapse.
    """
    nodes_relaxed = nodes_ironed.copy()
    adj = defaultdict(list)
    for u, v in struts:
        adj[u].append(v)
        adj[v].append(u)

    # Compute relaxation weight based on depth
    # Depth = 0 (boundary conformed): weight = 0.0 (fixed)
    # Depth = 1: weight = 0.25
    # Depth >= 2: weight = 1.0 (full relaxation)
    weights = np.zeros(len(nodes_relaxed), dtype=np.float64)
    for i in range(len(nodes_relaxed)):
        d = node_depths[i]
        if d == 0:
            weights[i] = 0.0
        elif d == 1:
            weights[i] = 0.25
        else:
            weights[i] = 1.0

    for _ in range(iterations):
        next_nodes = nodes_relaxed.copy()
        for idx in range(len(nodes_relaxed)):
            w = weights[idx]
            if w == 0.0 or len(adj[idx]) == 0:
                continue
            neighbors = adj[idx]
            avg = np.mean(nodes_relaxed[neighbors], axis=0)
            next_nodes[idx] = (1.0 - alpha * w) * nodes_relaxed[idx] + (alpha * w) * avg
        nodes_relaxed = next_nodes

    return nodes_relaxed


def sweep_to_manifold(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """
    Helper to generate watertight solid cylinders using trimesh.
    """
    cylinders = []
    for u, v in struts:
        p0 = nodes[u]
        p1 = nodes[v]
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        # Align cylinder to connection vector
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        direction = vec / length
        rotation_matrix = _rotation_matrix_from_z(direction)
        translation = 0.5 * (p0 + p1)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        cyl.apply_transform(transform)
        cylinders.append(cyl)

    if not cylinders:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(cylinders)


def sweep_struts_concat(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """
    Helper to sweep lines to solid cylinders.
    """
    return sweep_to_manifold(nodes, struts, radius)


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    v = np.cross(z_axis, vec)
    c = np.dot(z_axis, vec)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c < 0:
            # Opposite direction
            return -np.eye(3)
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def generate_conformal_lattice(
    cad_filepath: str | trimesh.Trimesh,
    cell_size: float,
    strut_radius: float,
    lattice_type: str = "A15",
    export_dir: str = "output",
    export_debug_stls: bool = False,
    skin_only: bool = False,
    skin_output_name: str | None = None,
    skip_sweep: bool = False,
    signed_distance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    mode: str = "conformal",
) -> dict[str, any]:
    """
    Unified entry point to generate conformed lattices (A15 Kagome or SC Octahedral)
    without using GMSH.
    """
    start_time = time.time()
    config = LATTICE_CONFIGS.get(lattice_type)
    if config is None:
        raise ValueError(f"Unknown lattice type: {lattice_type}. Supported: 'A15', 'SC'")

    CELL_FACES = config["faces"]
    CELL_ADJACENT_FACES = config["adjacent_faces"]
    UNIT_CELL_MULTIPLIER = config["unit_cell_multiplier"]
    valency_cutoff = config["boundary_valency_cutoff"]

    # 1. Load and repair CAD Mesh
    if isinstance(cad_filepath, trimesh.Trimesh):
        cad_mesh = cad_filepath
        part_name = "mesh"
    else:
        raw_mesh = trimesh.load(cad_filepath)
        print(f"\nRepairing part geometry: {cad_filepath}")
        cad_mesh = repair_cad_mesh(raw_mesh)
        part_name = Path(cad_filepath).stem

    print(f"Processing conformed lattice ({lattice_type}) for part: {part_name}")
    print(f"  CAD bounds: {cad_mesh.bounds}")

    # 2. Bounding Box & Grid Auto-Scaling
    min_bound, max_bound = cad_mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # Generate background grid nodes and elements
    from graphite.explicit.proven_topologies import generate_background_grid
    grid_nodes, cells = generate_background_grid(lattice_type, cad_mesh.bounds, cell_size)

    print(f"  Background grid: {len(grid_nodes)} nodes, {len(cells)} cells")

    # 3. Exact Face-Centroid Culling
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in CELL_FACES:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    if signed_distance_fn is not None:
        s_dists = np.asarray(signed_distance_fn(all_centroids), dtype=np.float64)
    else:
        s_dists = safe_signed_distance(cad_mesh, all_centroids)

    kept_cells = []
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        if mode == "boolean":
            if np.any(c_dists >= -1e-5):
                kept_cells.append(cell)
        else:
            if np.all(c_dists >= -1e-5):
                kept_cells.append(cell)

    surviving_cells = np.array(kept_cells)
    print(f"  Surviving cells: {len(surviving_cells)} / {len(cells)}")
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # 4. Extract boundary faces
    boundary_faces = []
    boundary_faces_set = set()
    if mode == "conformal":
        face_counts = defaultdict(int)
        for cell in surviving_cells:
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_counts[fkey] += 1
        boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]
        boundary_faces_set = set(boundary_faces)

    # 5. Generate dual graph and map boundary nodes
    dual_coords = []
    dual_coord_to_idx = {}
    strut_set = set()
    face_to_centroid = {}

    def get_or_add_dual(coord):
        key = _pt_key(coord)
        if key not in dual_coord_to_idx:
            dual_coord_to_idx[key] = len(dual_coords)
            dual_coords.append(coord.copy())
        return dual_coord_to_idx[key]

    for cell in surviving_cells:
        face_node_ids = []
        for fv in CELL_FACES:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            n_idx = get_or_add_dual(centroid)
            face_node_ids.append(n_idx)

            if mode == "conformal":
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_to_centroid[fkey] = centroid

        for a, b in CELL_ADJACENT_FACES:
            strut_set.add((min(face_node_ids[a], face_node_ids[b]), max(face_node_ids[a], face_node_ids[b])))

    nodes_3d = np.array(dual_coords)
    struts = np.array(sorted(strut_set))

    boundary_node_ids = np.empty(0, dtype=np.int64)
    if mode == "conformal":
        boundary_nodes_list = []
        for bf in boundary_faces:
            centroid = face_to_centroid[bf]
            n_idx = dual_coord_to_idx[_pt_key(centroid)]
            boundary_nodes_list.append(n_idx)
        boundary_node_ids = np.unique(boundary_nodes_list)

    if mode == "conformal":
        # 6. Topological BFS Depth Tagging
        M_t = len(surviving_cells)
        face_to_cells = defaultdict(list)
        for hi, cell in enumerate(surviving_cells):
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_to_cells[fkey].append(hi)

        cell_depth = np.full(M_t, -1, dtype=np.int32)
        queue = deque()
        for bf in boundary_faces:
            if bf in face_to_cells:
                for hi in face_to_cells[bf]:
                    if cell_depth[hi] == -1:
                        cell_depth[hi] = 0
                        queue.append(hi)

        cell_neighbors = defaultdict(list)
        for fkey, owning_cells in face_to_cells.items():
            if len(owning_cells) == 2:
                u, v = owning_cells[0], owning_cells[1]
                cell_neighbors[u].append(v)
                cell_neighbors[v].append(u)

        while queue:
            curr = queue.popleft()
            d = cell_depth[curr]
            for nb in cell_neighbors[curr]:
                if cell_depth[nb] == -1:
                    cell_depth[nb] = d + 1
                    queue.append(nb)

        node_depths = np.full(len(nodes_3d), 999999, dtype=np.int32)
        for hi, cell in enumerate(surviving_cells):
            d = cell_depth[hi]
            if d == -1:
                continue
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                centroid = face_to_centroid[fkey]
                idx = dual_coord_to_idx[_pt_key(centroid)]
                node_depths[idx] = min(node_depths[idx], d)

        node_depths[node_depths == 999999] = int(np.max(node_depths[node_depths != 999999])) + 1 if np.any(node_depths != 999999) else 0

        # 7. SDF Ironing
        nodes_ironed, conformed_mask, conformed_nodes_map = apply_sdf_ironing(
            nodes_3d, struts, boundary_node_ids, cad_mesh, valency_cutoff
        )

        # 8. Depth-Gated Relaxation
        nodes_relaxed = apply_depth_gated_relaxation(
            nodes_ironed, struts, node_depths, iterations=15, alpha=0.5
        )

        # 9. Topological Wiring
        edge_to_faces = defaultdict(list)
        for bf in boundary_faces:
            verts_list = sorted(bf)
            for a, b in combinations(verts_list, 2):
                edge_key = (a, b) if a < b else (b, a)
                edge_to_faces[edge_key].append(bf)

        cyan_struts_set = set()
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                fa, fb = faces[0], faces[1]
                c_a = face_to_centroid[fa]
                c_b = face_to_centroid[fb]
                idx_a = dual_coord_to_idx[_pt_key(c_a)]
                idx_b = dual_coord_to_idx[_pt_key(c_b)]
                cyan_struts_set.add((min(idx_a, idx_b), max(idx_a, idx_b)))

        cyan_struts = np.array(list(cyan_struts_set), dtype=np.int64) if cyan_struts_set else np.empty((0, 2), dtype=np.int64)

        active_nodes = set(range(len(nodes_3d))) - set(boundary_node_ids) | set(conformed_nodes_map.keys())
        red_struts_list = []
        for u, v in struts:
            if u in active_nodes and v in active_nodes:
                red_struts_list.append((u, v))
        red_struts = np.array(red_struts_list, dtype=np.int64) if red_struts_list else np.empty((0, 2), dtype=np.int64)
    else:
        # Boolean mode: no skin, no relaxation, all struts are core (red) struts
        nodes_relaxed = nodes_3d.copy()
        cyan_struts = np.empty((0, 2), dtype=np.int64)
        red_struts = struts.copy()

    if skip_sweep:
        elapsed_time = time.time() - start_time
        return {
            "nodes_count": len(nodes_3d),
            "struts_count": len(struts),
            "boundary_faces_count": len(boundary_faces),
            "cyan_struts_count": len(cyan_struts),
            "red_struts_count": len(red_struts),
            "elapsed_time": elapsed_time,
            "nodes_relaxed": nodes_relaxed,
            "cyan_struts": cyan_struts,
            "red_struts": red_struts,
        }

    # 10. Sweep and Export STLs
    core_manifold = None
    skin_manifold = None

    if not skin_only and len(red_struts) > 0:
        core_manifold = sweep_to_manifold(nodes_relaxed, red_struts, radius=strut_radius)
    if len(cyan_struts) > 0:
        skin_manifold = sweep_to_manifold(nodes_relaxed, cyan_struts, radius=1.5 * strut_radius)

    os.makedirs(export_dir, exist_ok=True)

    if skin_only:
        skin_fname = skin_output_name or f"{part_name}_surface_dual.stl"
        skin_path = os.path.join(export_dir, skin_fname)
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)
    else:
        combined_lattice = None
        if core_manifold is not None and skin_manifold is not None:
            # watertight union of struts
            combined_lattice, _ = union_lattice_with_spherical_joints(
                nodes_relaxed,
                np.vstack([red_struts, cyan_struts]),
                np.hstack([np.full(len(red_struts), strut_radius), np.full(len(cyan_struts), 1.5 * strut_radius)]),
            )
        elif core_manifold is not None:
            combined_lattice = core_manifold
        elif skin_manifold is not None:
            combined_lattice = skin_manifold

        if combined_lattice is not None:
            combined_lattice, _ = boolean_intersect_with_cad(combined_lattice, cad_mesh)

        lattice_path = os.path.join(export_dir, f"{part_name}_conformal_lattice.stl")
        if combined_lattice is not None:
            combined_lattice.export(lattice_path)
        else:
            trimesh.Trimesh().export(lattice_path)

        skin_path = os.path.join(export_dir, f"{part_name}_boundary_skin.stl")
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)

    elapsed_time = time.time() - start_time
    result = {
        "nodes_count": len(nodes_3d),
        "struts_count": len(struts),
        "boundary_faces_count": len(boundary_faces),
        "cyan_struts_count": len(cyan_struts),
        "red_struts_count": len(red_struts),
        "elapsed_time": elapsed_time,
    }
    return result


def generate_conformal_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    **kwargs
) -> ScaffoldResult:
    """
    GMSH-free conformed background grid generator for tetrahedral meshes (A15).
    Conforms the grid nodes to the boundary mesh and culls elements.
    """
    # 1. Bounding box & bounds scaling
    cell_size = target_element_size
    min_bound, max_bound = mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # A15 Basis & Grid Setup
    A15_BASIS = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        [0.25, 0.5, 0.0], [0.75, 0.5, 0.0],
        [0.0, 0.25, 0.5], [0.0, 0.75, 0.5],
        [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
    ], dtype=np.float64)
    BOND_CUTOFF = 0.62

    pts_list = []
    for ix_n in range(-1, 3):
        for iy_n in range(-1, 3):
            for iz_n in range(-1, 3):
                for b in A15_BASIS:
                    pts_list.append(b + np.array([ix_n, iy_n, iz_n], dtype=np.float64))
    pts = np.unique(np.round(np.vstack(pts_list), 9), axis=0)

    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    adj = {i: set() for i in range(len(pts))}
    for u, v in zip(*np.where((dist > 1e-9) & (dist <= BOND_CUTOFF))):
        if u < v:
            adj[u].add(v); adj[v].add(u)

    cliques = []
    for u in range(len(pts)):
        for v in adj[u]:
            if v <= u: continue
            uv = adj[u] & adj[v]
            for w in uv:
                if w <= v: continue
                for x in (uv & adj[w]):
                    if x > w:
                        cliques.append((u, v, w, x))

    arr = np.array(cliques, dtype=np.int64)
    centroids_frac = pts[arr].mean(axis=1)
    inside = np.all((centroids_frac >= -1e-9) & (centroids_frac <= 1.0 + 1e-9), axis=1)
    cliques_inside = arr[inside]

    node_coords_int = []
    node_coord_to_idx = {}

    def get_or_add(coord_int):
        key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
        if key not in node_coord_to_idx:
            node_coord_to_idx[key] = len(node_coords_int)
            node_coords_int.append(coord_int)
        return node_coord_to_idx[key]

    seen_cells = set()
    cells_out = []
    pts_int = np.round(pts * 4.0).astype(np.int64)

    for ix in range(min_ix, max_ix + 1):
        for iy in range(min_iy, max_iy + 1):
            for iz in range(min_iz, max_iz + 1):
                offset = np.array([ix, iy, iz], dtype=np.int64) * 4
                for cell in cliques_inside:
                    cell_int = pts_int[cell] + offset
                    ck = frozenset(tuple(p) for p in cell_int)
                    if ck in seen_cells:
                        continue
                    seen_cells.add(ck)
                    c_indices = [get_or_add(p) for p in cell_int]
                    cells_out.append(c_indices)

    grid_nodes = np.array(node_coords_int, dtype=np.float64) * (cell_size / 4.0)
    cells = np.array(cells_out, dtype=np.int64)

    # Trimming using face-centroid checks
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in TET_FACE_TRIPLETS:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    s_dists = safe_signed_distance(mesh, all_centroids)

    kept_cells = []
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        if np.all(c_dists >= -1e-5):
            kept_cells.append(cell)

    surviving_cells = np.array(kept_cells)
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # Boundary faces culling
    face_counts = defaultdict(int)
    for cell in surviving_cells:
        for fv in TET_FACE_TRIPLETS:
            fkey = _coord_face_key(grid_nodes, cell, fv)
            face_counts[fkey] += 1
    boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]

    # Map face keys back to node indices
    # Each face key is a frozenset of point rounded keys. Let's find node indices matching these.
    boundary_node_ids = set()
    for bf in boundary_faces:
        # Resolve face points from frozenset of rounded coords
        for cell in surviving_cells:
            for fv in TET_FACE_TRIPLETS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    for n in cell[fv]:
                        boundary_node_ids.add(n)

    boundary_node_ids = np.array(list(boundary_node_ids), dtype=np.int64)

    # Snap boundary nodes to CAD surface
    grid_nodes_conformed = grid_nodes.copy()
    if len(boundary_node_ids) > 0:
        boundary_coords = grid_nodes[boundary_node_ids]
        closest_pts, _ = project_to_cad_surface(boundary_coords, mesh)
        for i, idx in enumerate(boundary_node_ids):
            grid_nodes_conformed[idx] = closest_pts[i]

    # Convert boundary faces back to node IDs
    surface_faces_list = []
    for bf in boundary_faces:
        face_nodes = []
        for cell in surviving_cells:
            for fv in TET_FACE_TRIPLETS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    face_nodes = list(cell[fv])
                    break
            if face_nodes:
                break
        surface_faces_list.append(face_nodes)

    surface_faces = np.array(surface_faces_list, dtype=np.int64)

    return ScaffoldResult(
        nodes=grid_nodes_conformed,
        elements=surviving_cells,
        surface_faces=surface_faces,
        element_order=1
    )


def generate_conformed_hex_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    **kwargs
) -> ScaffoldResult:
    """
    GMSH-free conformed background grid generator for hexahedral meshes (SC).
    Conforms the grid nodes to the boundary mesh and culls elements.
    """
    cell_size = target_element_size
    min_bound, max_bound = mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # Grid Setup (SC)
    node_coords_int = []
    node_coord_to_idx = {}

    def get_or_add(coord_int):
        key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
        if key not in node_coord_to_idx:
            node_coord_to_idx[key] = len(node_coords_int)
            node_coords_int.append(coord_int)
        return node_coord_to_idx[key]

    cells_out = []
    for ix in range(min_ix, max_ix + 1):
        for iy in range(min_iy, max_iy + 1):
            for iz in range(min_iz, max_iz + 1):
                voxel_corners = np.array([
                    [ix, iy, iz],
                    [ix + 1, iy, iz],
                    [ix + 1, iy + 1, iz],
                    [ix, iy + 1, iz],
                    [ix, iy, iz + 1],
                    [ix + 1, iy, iz + 1],
                    [ix + 1, iy + 1, iz + 1],
                    [ix, iy + 1, iz + 1]
                ], dtype=np.int64)
                c_indices = [get_or_add(p) for p in voxel_corners]
                cells_out.append(c_indices)

    grid_nodes = np.array(node_coords_int, dtype=np.float64) * cell_size
    cells = np.array(cells_out, dtype=np.int64)

    # Trimming using face-centroid checks
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in HEX_FACE_QUADS:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    s_dists = safe_signed_distance(mesh, all_centroids)

    kept_cells = []
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        if np.all(c_dists >= -1e-5):
            kept_cells.append(cell)

    surviving_cells = np.array(kept_cells)
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # Boundary faces culling
    face_counts = defaultdict(int)
    for cell in surviving_cells:
        for fv in HEX_FACE_QUADS:
            fkey = _coord_face_key(grid_nodes, cell, fv)
            face_counts[fkey] += 1
    boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]

    # Map face keys back to node indices
    boundary_node_ids = set()
    for bf in boundary_faces:
        for cell in surviving_cells:
            for fv in HEX_FACE_QUADS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    for n in cell[fv]:
                        boundary_node_ids.add(n)

    boundary_node_ids = np.array(list(boundary_node_ids), dtype=np.int64)

    # Snap boundary nodes to CAD surface
    grid_nodes_conformed = grid_nodes.copy()
    if len(boundary_node_ids) > 0:
        boundary_coords = grid_nodes[boundary_node_ids]
        closest_pts, _ = project_to_cad_surface(boundary_coords, mesh)
        for i, idx in enumerate(boundary_node_ids):
            grid_nodes_conformed[idx] = closest_pts[i]

    # Convert boundary faces back to node IDs
    surface_faces_list = []
    for bf in boundary_faces:
        face_nodes = []
        for cell in surviving_cells:
            for fv in HEX_FACE_QUADS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    face_nodes = list(cell[fv])
                    break
            if face_nodes:
                break
        surface_faces_list.append(face_nodes)

    surface_faces = np.array(surface_faces_list, dtype=np.int64)

    return ScaffoldResult(
        nodes=grid_nodes_conformed,
        elements=surviving_cells,
        surface_faces=surface_faces,
        element_order=1
    )


# Backwards compatibility wrappers
def generate_a15_conformal_lattice(*args, **kwargs) -> dict[str, any]:
    """Wrapper to maintain compatibility with conformed A15 tests."""
    return generate_conformal_lattice(*args, **kwargs, lattice_type="A15")
