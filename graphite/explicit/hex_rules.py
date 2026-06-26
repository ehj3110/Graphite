"""
Graphite Explicit Engine - Hexahedral Topology Rules

This module defines local lattice topology rules for single hexahedral 
elements. These rules generate node coordinates and strut connectivity 
graphs for different lattice types (e.g., grid, octahedral, star, octet)
given the eight corner coordinates of a single conformal hex.
"""
from __future__ import annotations

import numpy as np

_HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)

_ADJACENT_FACE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
)

_HEX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _validate(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (8, 3):
        raise ValueError(f"coords must have shape (8, 3), got {coords.shape}.")
    return coords


def _hex_face_centers(coords: np.ndarray) -> np.ndarray:
    return np.array([coords[list(f)].mean(axis=0) for f in _HEX_FACES], dtype=np.float64)


def _hex_centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)


def apply_hex_grid(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a surface-conforming grid topology for a hexahedral cell.

    This rule simply connects the 8 corners of the hex along its 12 edges.

    Parameters
    ----------
    coords : ndarray
        (8, 3) array of the hex corner coordinates.

    Returns
    -------
    nodes : ndarray
        (8, 3) array of node coordinates.
    struts : ndarray
        (12, 2) array of strut connectivity edges.
    """
    coords = _validate(coords)
    return coords.copy(), np.array(_HEX_EDGES, dtype=np.int64)


def apply_hex_octahedral(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = _validate(coords)
    return _hex_face_centers(coords), np.array(_ADJACENT_FACE_PAIRS, dtype=np.int64)


def apply_hex_star(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = _validate(coords)
    centroid = _hex_centroid(coords)
    nodes = np.vstack((centroid[None, :], coords))
    struts = np.array([(0, i) for i in range(1, 9)], dtype=np.int64)
    return nodes, struts


def apply_hex_octet_truss(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = _validate(coords)
    face_ctrs = _hex_face_centers(coords)
    nodes = np.vstack((coords, face_ctrs))
    strut_list: list[tuple[int, int]] = []
    for fi, face in enumerate(_HEX_FACES):
        fc_idx = 8 + fi
        for corner in face:
            strut_list.append((corner, fc_idx))
    for a, b in _ADJACENT_FACE_PAIRS:
        strut_list.append((8 + a, 8 + b))
    return nodes, np.array(strut_list, dtype=np.int64)


def apply_hex_dual(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Local dual primitive for a single hex:
      - node: the hex centroid
      - struts: none (adjacency is resolved globally across neighboring cells)
    """
    coords = _validate(coords)
    centroid = _hex_centroid(coords)
    return centroid.reshape(1, 3), np.empty((0, 2), dtype=np.int64)


def apply_hex_face_dual(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Local face-dual primitive for a single hex:
      - nodes: centroids of the 6 quad faces
      - struts: octahedral connectivity between adjacent face-centroids
    """
    coords = _validate(coords)
    face_centers = _hex_face_centers(coords)
    struts = np.array(_ADJACENT_FACE_PAIRS, dtype=np.int64)
    return face_centers, struts


def apply_hex_kelvin14(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Kelvin (truncated octahedron) unit mapped into one conformal hex brick.

    24 nodes / 36 struts per reference cell (see ``kelvin_cell.generate_kelvin_cell``).
    Legacy name ``kelvin14`` retained for rule registry; this is **not** the old
    14-node oct-tet corner+face-center graph.
    """
    from .kelvin_cell import map_kelvin_cell_to_hex_brick

    coords = _validate(coords)
    nodes, struts = map_kelvin_cell_to_hex_brick(coords, L=1.0)
    return nodes, struts


def apply_hex_kelvin(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alias for the 24-node truncated-octahedron Kelvin rule."""
    return apply_hex_kelvin14(coords)


def apply_hex_tesseract(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Tesseract (nested hypercube projection) mapped into one conformal hex brick.

    16 nodes / 32 struts per reference cell (see ``tesseract_cell.generate_tesseract_cell``):
    outer cube at +/- L/2, inner cube at +/- L/4, plus 8 radial corner struts.
    """
    from .tesseract_cell import map_tesseract_cell_to_hex_brick

    coords = _validate(coords)
    nodes, struts = map_tesseract_cell_to_hex_brick(coords, L=1.0)
    return nodes, struts


def apply_hex_a15_kagome(
    coords: np.ndarray,
    sdf_sampler=None,
    boundary_mesh=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate an A15 Kagome lattice topology for a single hexahedral cell.

    This rule maps the canonical A15 atomic basis into a deformed hex cell using
    trilinear interpolation, extracts the tetrahedral elements using deterministic
    4-clique reconstruction over a padded domain to capture boundary-straddling tets,
    applies sub-cell gating if an sdf_sampler or boundary_mesh is provided,
    crops the resulting Kagome nodes to the [0, 1]^3 bounding box, and warps them.

    Parameters
    ----------
    coords : ndarray
        (8, 3) array of the hex corner coordinates.
    sdf_sampler : callable, optional
        A function mapping (N, 3) points to SDF values (negative inside, positive outside).
    boundary_mesh : trimesh.Trimesh, optional
        Mesh boundary if sdf_sampler is not pre-built.

    Returns
    -------
    nodes : ndarray
        (V, 3) array of Kagome node coordinates.
    struts : ndarray
        (E, 2) array of Kagome strut connectivity edges.
    """
    coords = _validate(coords)

    # 1. Define canonical A15 basis in unit coordinates [0, 1]^3
    basis = np.array([
        # B atoms (BCC sites)
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        # A atoms (face-offset sites)
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75]
    ], dtype=np.float64)

    # 2. Tile the basis in a padded domain [-0.3, 1.3]^3 to get all neighbor nodes
    # We use i, j, k in -1, 0, 1, 2 to cover all direct boundary-straddling connections
    canonical_pts_list = []
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            for k in (-1, 0, 1, 2):
                offset = np.array([i, j, k], dtype=np.float64)
                for pt in basis:
                    t_pt = pt + offset
                    if (t_pt[0] >= -0.3 - 1e-5 and t_pt[0] <= 1.3 + 1e-5 and
                        t_pt[1] >= -0.3 - 1e-5 and t_pt[1] <= 1.3 + 1e-5 and
                        t_pt[2] >= -0.3 - 1e-5 and t_pt[2] <= 1.3 + 1e-5):
                        canonical_pts_list.append(t_pt)

    canonical_pts = np.vstack(canonical_pts_list)
    canonical_pts = np.unique(np.round(canonical_pts, 8), axis=0)

    # 3. Find primary bonds in canonical space (cutoff = 0.62)
    diffs = canonical_pts[:, None, :] - canonical_pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    pairs = np.argwhere((dists > 1e-5) & (dists <= 0.62))
    edges = [tuple(p) for p in pairs if p[0] < p[1]]

    # 4. Find all 4-cliques (tetrahedra) in the canonical graph
    adj = {i: set() for i in range(len(canonical_pts))}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    cliques = []
    num_pts = len(canonical_pts)
    for u in range(num_pts):
        for v in adj[u]:
            if v > u:
                common_uv = adj[u].intersection(adj[v])
                for w in common_uv:
                    if w > v:
                        common_uvw = common_uv.intersection(adj[w])
                        for x in common_uvw:
                            if x > w:
                                cliques.append((u, v, w, x))

    cliques_arr = np.array(cliques, dtype=np.int64)

    # Helper function for trilinear warping
    def trilinear_warp(pts: np.ndarray) -> np.ndarray:
        u = pts[:, 0]
        v = pts[:, 1]
        w = pts[:, 2]
        weights = np.column_stack([
            (1.0 - u) * (1.0 - v) * (1.0 - w),
            u * (1.0 - v) * (1.0 - w),
            u * v * (1.0 - w),
            (1.0 - u) * v * (1.0 - w),
            (1.0 - u) * (1.0 - v) * w,
            u * (1.0 - v) * w,
            u * v * w,
            (1.0 - u) * v * w
        ])
        return np.dot(weights, coords)

    # 5. NEW SUB-CELL GATING
    # If sdf_sampler is not provided but boundary_mesh is, build it.
    if sdf_sampler is None and boundary_mesh is not None:
        from graphite.explicit.boundary_policy import build_edt_sdf_sampler
        sdf_sampler = build_edt_sdf_sampler(boundary_mesh, resolution=0.5)

    if sdf_sampler is not None and len(cliques_arr) > 0:
        # Warp the canonical points to physical coordinates
        physical_pts = trilinear_warp(canonical_pts)
        # Compute cell size from corner coords
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        edge_lens = [np.linalg.norm(coords[u] - coords[v]) for u, v in edge_indices]
        cell_size = float(np.mean(edge_lens))
        
        # Evaluate SDF at all vertices of all tetrahedra
        clique_physical_pts = physical_pts[cliques_arr]
        M = len(cliques_arr)
        all_verts = clique_physical_pts.reshape(-1, 3)
        all_sdf = sdf_sampler(all_verts).reshape(M, 4)
        all_verts_inside = np.all(all_sdf <= 0.0, axis=1)
        
        # Evaluate SDF at centroids
        centroids = np.mean(clique_physical_pts, axis=1)
        centroids_sdf = sdf_sampler(centroids)
        centroid_deeply_inside = centroids_sdf <= (-0.15 * cell_size)
        
        keep_mask = all_verts_inside | centroid_deeply_inside
        surviving_cliques = cliques_arr[keep_mask]
    else:
        surviving_cliques = cliques_arr

    # 6. Apply face-centroid Kagome mapping on surviving tetrahedra
    face_to_k_node = {}
    kagome_nodes_list = []

    face_triplets = [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3)
    ]

    for tet in surviving_cliques:
        for f_idx in face_triplets:
            face = tuple(sorted([tet[f_idx[0]], tet[f_idx[1]], tet[f_idx[2]]]))
            if face not in face_to_k_node:
                idx = len(kagome_nodes_list)
                face_to_k_node[face] = idx
                centroid = (canonical_pts[face[0]] + canonical_pts[face[1]] + canonical_pts[face[2]]) / 3.0
                kagome_nodes_list.append(centroid)

    if not kagome_nodes_list:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    kagome_nodes = np.array(kagome_nodes_list, dtype=np.float64)

    kagome_edges_set = set()
    for tet in surviving_cliques:
        k_indices = []
        for f_idx in face_triplets:
            face = tuple(sorted([tet[f_idx[0]], tet[f_idx[1]], tet[f_idx[2]]]))
            k_indices.append(face_to_k_node[face])

        for i in range(4):
            for j in range(i + 1, 4):
                u_k = k_indices[i]
                v_k = k_indices[j]
                kagome_edges_set.add((min(u_k, v_k), max(u_k, v_k)))

    kagome_edges = np.array(list(kagome_edges_set), dtype=np.int64)

    # 7. Crop the Kagome nodes to the unit cell boundary [0, 1]^3
    mask = (
        (kagome_nodes[:, 0] >= -1e-5) & (kagome_nodes[:, 0] <= 1.0 + 1e-5) &
        (kagome_nodes[:, 1] >= -1e-5) & (kagome_nodes[:, 1] <= 1.0 + 1e-5) &
        (kagome_nodes[:, 2] >= -1e-5) & (kagome_nodes[:, 2] <= 1.0 + 1e-5)
    )
    cropped_kagome_nodes = kagome_nodes[mask]

    if len(cropped_kagome_nodes) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    # Map old index to new cropped index
    old_to_new = np.full(len(kagome_nodes), -1, dtype=np.int64)
    old_to_new[mask] = np.arange(np.sum(mask))

    # Filter struts: keep only those where both endpoints survived the crop
    struts_mask = (old_to_new[kagome_edges[:, 0]] >= 0) & (old_to_new[kagome_edges[:, 1]] >= 0)
    cropped_kagome_edges = kagome_edges[struts_mask]

    # Reindex the struts
    cropped_kagome_edges = np.column_stack([
        old_to_new[cropped_kagome_edges[:, 0]],
        old_to_new[cropped_kagome_edges[:, 1]]
    ])

    # 8. Warp cropped canonical points to the deformed hex cell using trilinear interpolation
    warped_pts = trilinear_warp(cropped_kagome_nodes)

    return warped_pts, cropped_kagome_edges

