"""
Graphite Explicit Engine - Hexahedral Topology Rules

This module defines local lattice topology rules for single hexahedral 
elements. These rules generate node coordinates and strut connectivity 
graphs for different lattice types (e.g., grid, octahedral, star, octet)
given the eight corner coordinates of a single conformal hex.
"""
from __future__ import annotations

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Quantized octahedral half-cells (6 polarities — face centers only)
# ---------------------------------------------------------------------------
#
# SC corner index (unit cube) — used only to locate face centroids:
#   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
#   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
#
# Octahedral nodes = 6 face centers (local face index):
#   0: −Z (0,1,2,3)    1: +Z (4,5,6,7)
#   2: −Y (0,1,5,4)    3: +Y (3,2,6,7)
#   4: −X (0,3,7,4)    5: +X (1,2,6,5)
#
# Full octahedral connects only adjacent face pairs (share an edge) — never
# parallel opposite faces (0–1, 2–3, 4–5).
#
# Each Half_* is the square-pyramid leaf of that octahedron:
#   - keep one face-center apex (solid side)
#   - keep four equatorial face centers (mid-plane diamond)
#   - omit the opposite (empty) face center
#   - NO midplane hub / NO braces through opposite diamond corners
# Adjacent cells share face-center coordinates and weld — no transitions.

# Local connectivity (apex=0, diamond=1..4) — octahedral adjacent pairs only:
#   - 4 spokes apex→diamond          (INTERNAL) — apex shares an edge with each
#   - 4 perimeter edges of diamond   (NATIVE SURFACE DUAL) — adjacent sides only
# Diagonals 1–3 / 2–4 (opposite faces) are intentionally absent.
_HALF_SPOKES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
)
_HALF_DIAMOND_CYCLE: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)
_HALF_STRUTS: tuple[tuple[int, int], ...] = _HALF_SPOKES + _HALF_DIAMOND_CYCLE
assert len(_HALF_STRUTS) == 8

# Local node i ← global face-center index (apex first, then diamond cycle)
# Z cut: equatorial diamond faces 2,5,3,4
_HALF_NEG_Z_FACE_INDICES: tuple[int, ...] = (0, 2, 5, 3, 4)  # apex −Z; omit +Z
_HALF_POS_Z_FACE_INDICES: tuple[int, ...] = (1, 2, 5, 3, 4)  # apex +Z; omit −Z
# X cut: equatorial diamond faces 0,2,1,3
_HALF_NEG_X_FACE_INDICES: tuple[int, ...] = (4, 0, 2, 1, 3)  # apex −X; omit +X
_HALF_POS_X_FACE_INDICES: tuple[int, ...] = (5, 0, 2, 1, 3)  # apex +X; omit −X
# Y cut: equatorial diamond faces 0,5,1,4
_HALF_NEG_Y_FACE_INDICES: tuple[int, ...] = (2, 0, 5, 1, 4)  # apex −Y; omit +Y
_HALF_POS_Y_FACE_INDICES: tuple[int, ...] = (3, 0, 5, 1, 4)  # apex +Y; omit −Y

# Back-compat aliases (legacy unsigned names = negative polarity)
_HALF_Z_FACE_INDICES = _HALF_NEG_Z_FACE_INDICES
_HALF_X_FACE_INDICES = _HALF_NEG_X_FACE_INDICES
_HALF_Y_FACE_INDICES = _HALF_NEG_Y_FACE_INDICES
_HALF_Z_SPOKES = _HALF_SPOKES
_HALF_Z_DIAMOND_CYCLE = _HALF_DIAMOND_CYCLE


# ---------------------------------------------------------------------------
# Explicit surface-dual tags (Task 15) — no coincidence extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafSurfaceSpec:
    """
    Explicit surface membership for one octahedral leaf rule.

    Half:
      - surface nodes = mid-plane diamond corners only (locals 1–4)
      - native surface struts = diamond perimeter only (not apex spokes)
    Full:
      - surface nodes = exposed face centers only (candidates 0–5, filtered later)
      - native surface struts = none (manifold stitcher supplies dual)
    """

    surface_node_locals: tuple[int, ...]
    native_surface_strut_locals: tuple[tuple[int, int], ...]
    exposed_candidate_locals: tuple[int, ...] = ()
    local_to_face_index: tuple[int, ...] = ()


_HALF_SURFACE_NODE_LOCALS: tuple[int, ...] = (1, 2, 3, 4)


def _half_surface_spec(face_indices: tuple[int, ...]) -> LeafSurfaceSpec:
    # face_indices[0]=apex, [1:5]=diamond (5 nodes total; no midplane hub)
    local_to_face = tuple(int(fi) for fi in face_indices)
    return LeafSurfaceSpec(
        surface_node_locals=_HALF_SURFACE_NODE_LOCALS,
        native_surface_strut_locals=_HALF_DIAMOND_CYCLE,
        exposed_candidate_locals=(),
        local_to_face_index=local_to_face,
    )


_FULL_SURFACE_SPEC = LeafSurfaceSpec(
    surface_node_locals=(),
    native_surface_strut_locals=(),
    exposed_candidate_locals=(0, 1, 2, 3, 4, 5),
    local_to_face_index=(0, 1, 2, 3, 4, 5),
)

_HALF_SURFACE_SPECS: dict[str, LeafSurfaceSpec] = {
    "octahedral_half_neg_z": _half_surface_spec(_HALF_NEG_Z_FACE_INDICES),
    "octahedral_half_pos_z": _half_surface_spec(_HALF_POS_Z_FACE_INDICES),
    "octahedral_half_neg_x": _half_surface_spec(_HALF_NEG_X_FACE_INDICES),
    "octahedral_half_pos_x": _half_surface_spec(_HALF_POS_X_FACE_INDICES),
    "octahedral_half_neg_y": _half_surface_spec(_HALF_NEG_Y_FACE_INDICES),
    "octahedral_half_pos_y": _half_surface_spec(_HALF_POS_Y_FACE_INDICES),
    "octahedral_half_z": _half_surface_spec(_HALF_NEG_Z_FACE_INDICES),
    "octahedral_half_x": _half_surface_spec(_HALF_NEG_X_FACE_INDICES),
    "octahedral_half_y": _half_surface_spec(_HALF_NEG_Y_FACE_INDICES),
}


def leaf_surface_spec_for_rule(rule_name: str) -> LeafSurfaceSpec | None:
    """Return explicit surface tags for a registered octahedral leaf rule."""
    n = str(rule_name).strip().lower()
    if n == "octahedral":
        return _FULL_SURFACE_SPEC
    return _HALF_SURFACE_SPECS.get(n)


def _apply_octahedral_half(
    coords: np.ndarray,
    face_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Half octahedral leaf = square pyramid (adjacent face pairs only).

    Node layout (template for every polarity, e.g. half_z)::

        0     apex (solid-side face center)
        1..4  mid-plane diamond corners (equatorial face centers)

    Struts: 4 apex spokes + 4 diamond perimeter edges.
    Never connects parallel opposite faces (no diamond diagonals, no hub).
    """
    coords = _validate(coords)
    face_ctrs = _hex_face_centers(coords)
    nodes = face_ctrs[list(face_indices)]
    struts = np.array(_HALF_STRUTS, dtype=np.int64)
    return nodes, struts


def apply_hex_octahedral_half_neg_z(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep −Z apex (face 0); omit +Z. Diamond: faces 2,5,3,4."""
    return _apply_octahedral_half(coords, _HALF_NEG_Z_FACE_INDICES)


def apply_hex_octahedral_half_pos_z(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep +Z apex (face 1); omit −Z. Diamond: faces 2,5,3,4."""
    return _apply_octahedral_half(coords, _HALF_POS_Z_FACE_INDICES)


def apply_hex_octahedral_half_neg_x(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep −X apex (face 4); omit +X. Diamond: faces 0,2,1,3."""
    return _apply_octahedral_half(coords, _HALF_NEG_X_FACE_INDICES)


def apply_hex_octahedral_half_pos_x(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep +X apex (face 5); omit −X. Diamond: faces 0,2,1,3."""
    return _apply_octahedral_half(coords, _HALF_POS_X_FACE_INDICES)


def apply_hex_octahedral_half_neg_y(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep −Y apex (face 2); omit +Y. Diamond: faces 0,5,1,4."""
    return _apply_octahedral_half(coords, _HALF_NEG_Y_FACE_INDICES)


def apply_hex_octahedral_half_pos_y(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep +Y apex (face 3); omit −Y. Diamond: faces 0,5,1,4."""
    return _apply_octahedral_half(coords, _HALF_POS_Y_FACE_INDICES)


# Legacy unsigned names → negative polarity (historical default).
apply_hex_octahedral_half_z = apply_hex_octahedral_half_neg_z
apply_hex_octahedral_half_x = apply_hex_octahedral_half_neg_x
apply_hex_octahedral_half_y = apply_hex_octahedral_half_neg_y


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


def apply_hex_cross(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross unit cell: 14 nodes (8 corners + 6 face centers).
    24 face-center-to-corner spokes (C-F 'X' crosses on 6 faces)
    plus 12 box edge struts (C-C), with no central octahedral diamond.
    """
    coords = _validate(coords)
    face_ctrs = _hex_face_centers(coords)
    nodes = np.vstack((coords, face_ctrs))
    strut_list: list[tuple[int, int]] = []
    # 24 C-F spokes (face diagonals / cross on 6 faces)
    for fi, face in enumerate(_HEX_FACES):
        fc_idx = 8 + fi
        for corner in face:
            strut_list.append((corner, fc_idx))
    # 12 C-C edge struts
    for a, b in (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ):
        strut_list.append((a, b))
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

    # Filter cliques by canonical centroid inside [0, 1]^3
    if len(cliques_arr) > 0:
        cliques_centroids = np.mean(canonical_pts[cliques_arr], axis=1)
        canonical_inside = np.all((cliques_centroids >= -1e-5) & (cliques_centroids <= 1.0 + 1e-5), axis=1)
        cliques_arr = cliques_arr[canonical_inside]

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

    # 7. Warp canonical points to the deformed hex cell using trilinear interpolation (no cropping)
    warped_pts = trilinear_warp(kagome_nodes)

    return warped_pts, kagome_edges


