"""
Graphite Conformal Lattice Engine V2 — A15 Conformal Lattice Generation

Topological depth engine, SDF projection, and depth-gated relaxation utilities
for A15 conformal trimming. Uses integer coordinate space background grids to
prevent floating-point tiling seam gaps.
"""

from __future__ import annotations
import os
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path
import numpy as np
import trimesh

from graphite.explicit.mesh_repair import repair_cad_mesh
from graphite.explicit.geometry_module import (
    union_lattice_with_spherical_joints,
    boolean_intersect_with_cad,
)

FACE_TRIPLETS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def _pt_key(p, decimals: int = 5) -> tuple:
    return tuple(np.round(p, decimals))


def _coord_face_key(nodes: np.ndarray, tet: np.ndarray, fv: tuple) -> frozenset:
    return frozenset(int(tet[i]) for i in fv)


def _unique_tet_edges(tets: np.ndarray) -> np.ndarray:
    """Extract unique edges from tetrahedral connectivity (6 edges per tet)."""
    edges: set[tuple[int, int]] = set()
    for tet in tets:
        for u, v in combinations(tet, 2):
            ui, vi = int(u), int(v)
            edges.add((min(ui, vi), max(ui, vi)))
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(edges), dtype=np.int64)


def safe_signed_distance(cad_mesh, points, chunk_size=5000) -> np.ndarray:
    """Query signed distance in chunks to avoid memory errors with large meshes."""
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N == 0:
        return np.empty(0, dtype=np.float64)

    dists = np.empty(N, dtype=np.float64)
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        dists[start_idx:end_idx] = trimesh.proximity.signed_distance(
            cad_mesh, points[start_idx:end_idx]
        )
    return dists


def project_to_cad_surface(nodes: np.ndarray, cad_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Finds closest point on the CAD mesh surface for each node, and determines if it is inside."""
    nodes = np.asarray(nodes, dtype=np.float64)
    query = trimesh.proximity.ProximityQuery(cad_mesh)
    closest_pts = query.on_surface(nodes)[0]

    s_dists = safe_signed_distance(cad_mesh, nodes)
    is_inside = s_dists >= -1e-5
    return closest_pts, is_inside


def apply_sdf_ironing(
    nodes_3d: np.ndarray,
    struts: np.ndarray,
    boundary_node_ids: np.ndarray,
    cad_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Iron boundary nodes onto the CAD surface using valency-gated snapping."""
    nodes_3d = np.asarray(nodes_3d, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    boundary_node_ids = np.asarray(boundary_node_ids, dtype=np.int64)

    # Calculate 3D valency (degrees) of Kagome nodes in the surviving mesh
    degrees = np.zeros(len(nodes_3d), dtype=np.int64)
    for u, v in struts:
        degrees[u] += 1
        degrees[v] += 1

    # Project boundary nodes to CAD surface
    boundary_coords = nodes_3d[boundary_node_ids]
    closest_pts, is_inside = project_to_cad_surface(boundary_coords, cad_mesh)

    # Apply valency snap rule
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
            # Inside: snap only if valency <= 3
            conformed = (valency <= 3)

        if conformed:
            snap_pt = closest_pts[local_i]
            nodes_ironed[idx] = snap_pt
            conformed_mask[idx] = True
            conformed_nodes_map[idx] = snap_pt

    return nodes_ironed, conformed_mask, conformed_nodes_map


def apply_depth_gated_relaxation(
    nodes: np.ndarray,
    struts: np.ndarray,
    node_depths: np.ndarray,
    iterations: int = 15,
    alpha: float = 0.5,
) -> np.ndarray:
    """Applies Laplacian smoothing to the 'Relaxation Zone' (nodes at topological Depth 1 and 2)."""
    nodes_relaxed = np.asarray(nodes, dtype=np.float64).copy()
    struts = np.asarray(struts, dtype=np.int64)
    node_depths = np.asarray(node_depths, dtype=np.int32)

    N = len(nodes_relaxed)

    # Build connectivity graph
    neighbors = defaultdict(list)
    for u, v in struts:
        neighbors[u].append(v)
        neighbors[v].append(u)

    # Frozen Mask:
    # A node is frozen (True) if node_depths[i] == 0 (skin) OR node_depths[i] >= 3 (rigid core)
    frozen_mask = (node_depths == 0) | (node_depths >= 3)

    for _ in range(iterations):
        next_nodes = nodes_relaxed.copy()
        for i in range(N):
            if frozen_mask[i]:
                continue
            nb_list = neighbors[i]
            if len(nb_list) == 0:
                continue
            neighbor_mean = np.mean(nodes_relaxed[nb_list], axis=0)
            next_nodes[i] = nodes_relaxed[i] + alpha * (neighbor_mean - nodes_relaxed[i])
        nodes_relaxed = next_nodes

    return nodes_relaxed


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vec)
    if length <= 0:
        return np.eye(4)
    v = vec / length
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if axis_norm < 1e-12:
        if dot < 0:
            return trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        return np.eye(4)
    axis /= axis_norm
    angle = np.arccos(dot)
    return trimesh.transformations.rotation_matrix(angle, axis)


def sweep_to_manifold(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """Sweep struts to cylinders and boolean-union them using Manifold3D."""
    import manifold3d
    manifolds = []
    for a, b in struts:
        p1, p2 = nodes[a], nodes[b]
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        mat = _rotation_matrix_from_z(vec)
        mat[:3, 3] = (p1 + p2) * 0.5
        cyl.apply_transform(mat)

        try:
            m = manifold3d.Manifold(
                manifold3d.Mesh(
                    vert_properties=np.asarray(cyl.vertices, dtype=np.float32),
                    tri_verts=np.asarray(cyl.faces, dtype=np.uint32),
                )
            )
            manifolds.append(m)
        except Exception:
            continue

    if not manifolds:
        return trimesh.Trimesh()

    while len(manifolds) > 1:
        next_level = []
        for i in range(0, len(manifolds), 2):
            if i + 1 < len(manifolds):
                try:
                    next_level.append(manifolds[i] + manifolds[i + 1])
                except Exception:
                    next_level.append(manifolds[i])
            else:
                next_level.append(manifolds[i])
        manifolds = next_level

    combined = manifolds[0]
    mesh_raw = combined.to_mesh()
    verts = np.asarray(mesh_raw.vert_properties).reshape(-1, 3)
    faces = np.asarray(mesh_raw.tri_verts).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def sweep_struts_concat(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """Lightweight cylinder sweep for debug exports (concatenate, no boolean union)."""
    if len(struts) == 0:
        return trimesh.Trimesh()

    chunk_size = 5000
    combined: trimesh.Trimesh | None = None
    for start in range(0, len(struts), chunk_size):
        chunk = struts[start : start + chunk_size]
        meshes: list[trimesh.Trimesh] = []
        for a, b in chunk:
            p1, p2 = nodes[a], nodes[b]
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1e-6:
                continue
            cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=6)
            mat = _rotation_matrix_from_z(vec)
            mat[:3, 3] = (p1 + p2) * 0.5
            cyl.apply_transform(mat)
            meshes.append(cyl)
        if not meshes:
            continue
        chunk_mesh = trimesh.util.concatenate(meshes)
        combined = chunk_mesh if combined is None else trimesh.util.concatenate([combined, chunk_mesh])
    return combined if combined is not None else trimesh.Trimesh()


def generate_a15_conformal_lattice(
    cad_filepath: str | trimesh.Trimesh,
    cell_size: float,
    strut_radius: float,
    export_dir: str,
    *,
    export_debug_stls: bool = False,
    skin_only: bool = False,
    skin_output_name: str | None = None,
    skip_sweep: bool = False,
    signed_distance_fn=None,
    mode: str = "conformal",
) -> dict:
    """
    Production core executing the A15 conformal trimming and relaxation pipeline.
    Ensures input mesh is repaired first, and runs scale-invariant integer-space background grids.
    """
    start_time = time.time()

    # 1. Load and Repair CAD Mesh
    if isinstance(cad_filepath, trimesh.Trimesh):
        cad_mesh = cad_filepath
        part_name = "mesh"
    else:
        if not os.path.exists(cad_filepath):
            raise FileNotFoundError(f"CAD mesh file not found: {cad_filepath}")
        raw_mesh = trimesh.load(cad_filepath)
        print(f"\nRepairing part geometry: {cad_filepath}")
        cad_mesh = repair_cad_mesh(raw_mesh)
        part_name = Path(cad_filepath).stem

    print(f"Processing conformed lattice for part: {part_name}")
    print(f"  CAD bounds: {cad_mesh.bounds}")

    # 2. Bounding Box & Grid Auto-Scaling
    min_bound, max_bound = cad_mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    # Determine unit cell grid bounds
    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # Generate background grid of tets
    from graphite.explicit.proven_topologies import generate_background_grid
    tet_nodes, tets = generate_background_grid("A15", cad_mesh.bounds, cell_size)

    print(f"  Background grid: {len(tet_nodes)} nodes, {len(tets)} tetrahedra")

    # Debug Stage 1: raw tetrahedral grid (pre-cull)
    if export_debug_stls:
        raw_tet_edges = _unique_tet_edges(tets)
        debug_01 = sweep_struts_concat(tet_nodes, raw_tet_edges, radius=strut_radius)
        debug_01_path = os.path.join(export_dir, "debug_01_raw_tet_grid.stl")
        debug_01.export(debug_01_path)

    # 3. Exact Face-Centroid Culling
    print("  Trimming tetrahedra using exact face-centroid culling...")
    all_centroids = []
    tet_to_centroids_indices = []
    for tet in tets:
        verts_coords = tet_nodes[tet]
        indices = []
        for fv in FACE_TRIPLETS:
            centroid = verts_coords[list(fv)].mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        tet_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    if signed_distance_fn is not None:
        s_dists = np.asarray(signed_distance_fn(all_centroids), dtype=np.float64)
    else:
        s_dists = safe_signed_distance(cad_mesh, all_centroids)

    kept_tets = []
    for i, tet in enumerate(tets):
        indices = tet_to_centroids_indices[i]
        t_dists = s_dists[indices]
        if mode == "boolean":
            if np.any(t_dists >= -1e-5):
                kept_tets.append(tet)
        else:
            if np.all(t_dists >= -1e-5):
                kept_tets.append(tet)

    surviving_tets = np.array(kept_tets)
    print(f"  Surviving tets: {len(surviving_tets)} / {len(tets)}")
    if len(surviving_tets) == 0:
        raise ValueError("No tets survived trimming!")

    # Debug Stage 2: culled tetrahedral grid
    if export_debug_stls:
        culled_tet_edges = _unique_tet_edges(surviving_tets)
        debug_02 = sweep_struts_concat(tet_nodes, culled_tet_edges, radius=strut_radius)
        debug_02_path = os.path.join(export_dir, "debug_02_culled_tet_grid.stl")
        debug_02.export(debug_02_path)

    # 4. Extract boundary faces
    boundary_faces = []
    boundary_faces_set = set()
    if mode == "conformal":
        face_counts: dict[frozenset, int] = defaultdict(int)
        for tet in surviving_tets:
            for fv in FACE_TRIPLETS:
                fkey = _coord_face_key(tet_nodes, tet, fv)
                face_counts[fkey] += 1
        boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]
        boundary_faces_set = set(boundary_faces)

    # 5. Generate Kagome graph and map boundary nodes
    kagome_coords = []
    kagome_coord_to_idx = {}
    strut_set = set()
    face_to_centroid = {}

    def get_or_add_kagome(coord):
        key = _pt_key(coord)
        if key not in kagome_coord_to_idx:
            kagome_coord_to_idx[key] = len(kagome_coords)
            kagome_coords.append(coord.copy())
        return kagome_coord_to_idx[key]

    for tet in surviving_tets:
        face_node_ids = []
        for fv in FACE_TRIPLETS:
            verts = tet[list(fv)]
            verts_coords = tet_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            n_idx = get_or_add_kagome(centroid)
            face_node_ids.append(n_idx)

            if mode == "conformal":
                fkey = _coord_face_key(tet_nodes, tet, fv)
                face_to_centroid[fkey] = centroid

        for a, b in combinations(face_node_ids, 2):
            strut_set.add((min(a, b), max(a, b)))

    nodes_3d = np.array(kagome_coords)
    struts = np.array(sorted(strut_set))

    boundary_node_ids = np.empty(0, dtype=np.int64)
    if mode == "conformal":
        boundary_nodes_list = []
        for bf in boundary_faces:
            centroid = face_to_centroid[bf]
            n_idx = kagome_coord_to_idx[_pt_key(centroid)]
            boundary_nodes_list.append(n_idx)
        boundary_node_ids = np.unique(boundary_nodes_list)

        # 6. Topological BFS Depth Tagging
        M_t = len(surviving_tets)
        face_to_tets: dict[frozenset, list] = defaultdict(list)
        for hi, tet in enumerate(surviving_tets):
            for fv in FACE_TRIPLETS:
                fkey = _coord_face_key(tet_nodes, tet, fv)
                face_to_tets[fkey].append(hi)

        tet_depth = np.full(M_t, -1, dtype=np.int32)
        queue = deque()
        for bf in boundary_faces:
            if bf in face_to_tets:
                for hi in face_to_tets[bf]:
                    if tet_depth[hi] == -1:
                        tet_depth[hi] = 0
                        queue.append(hi)

        tet_neighbors = defaultdict(list)
        for fkey, owning_tets in face_to_tets.items():
            if len(owning_tets) == 2:
                u, v = owning_tets[0], owning_tets[1]
                tet_neighbors[u].append(v)
                tet_neighbors[v].append(u)

        while queue:
            curr = queue.popleft()
            d = tet_depth[curr]
            for nb in tet_neighbors[curr]:
                if tet_depth[nb] == -1:
                    tet_depth[nb] = d + 1
                    queue.append(nb)

        node_depths = np.full(len(nodes_3d), 999999, dtype=np.int32)
        for hi, tet in enumerate(surviving_tets):
            d = tet_depth[hi]
            if d == -1:
                continue
            for fv in FACE_TRIPLETS:
                fkey = _coord_face_key(tet_nodes, tet, fv)
                centroid = face_to_centroid[fkey]
                idx = kagome_coord_to_idx[_pt_key(centroid)]
                node_depths[idx] = min(node_depths[idx], d)

        node_depths[node_depths == 999999] = int(np.max(node_depths[node_depths != 999999])) + 1 if np.any(node_depths != 999999) else 0

        # 7. SDF Ironing
        nodes_ironed, conformed_mask, conformed_nodes_map = apply_sdf_ironing(
            nodes_3d, struts, boundary_node_ids, cad_mesh
        )

        # 8. Depth-Gated Relaxation
        nodes_relaxed = apply_depth_gated_relaxation(
            nodes_ironed, struts, node_depths, iterations=15, alpha=0.5
        )

        # 9. Topological Wiring
        edge_to_faces: dict[tuple, list] = defaultdict(list)
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
                idx_a = kagome_coord_to_idx[_pt_key(c_a)]
                idx_b = kagome_coord_to_idx[_pt_key(c_b)]
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
            # Perform boolean intersection with CAD to clean up protruding edges
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
