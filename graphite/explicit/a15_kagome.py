"""
A15 Frank-Kasper → Kagome lattice on an undeformed Cartesian hex grid.

Supports fractional box extents on a quarter-cell (0.25 * cell_size) grid,
valency-gated planar snapping, and shared-edge surface dual (no proximity,
no Delaunay).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Crystallographic constants
# ---------------------------------------------------------------------------

A15_BASIS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
    ],
    dtype=np.float64,
)

BOND_CUTOFF = 0.62
FACE_TRIPLETS = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))

# Quarter-cell slice increment (fraction of cell_size).
SLICE_FRACTION = 0.25

# Band around each cut plane used for cull / snap (fraction of cell_size).
SNAP_BAND_FRACTION = 0.15

# Outward stretch allowed only when node valency is at or below this.
MAX_STRETCH_VALENCY = 3

# Tet VF gate: keep if all verts inside, or centroid at/inside this SDF.
SDF_EPS = 1e-4


def assert_quarter_cell_extent(extent_mm: float, cell_size: float, *, name: str = "extent") -> None:
    """Require extent / cell_size to land on a k * 0.25 grid (k integer)."""
    n_cells = float(extent_mm) / float(cell_size)
    steps = n_cells / SLICE_FRACTION
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            f"{name}={extent_mm} mm is not a {SLICE_FRACTION} * cell_size "
            f"({cell_size} mm) increment (got {n_cells} cells)."
        )


def build_canonical_cliques() -> tuple[np.ndarray, np.ndarray]:
    """Padded A15 basis + 4-cliques whose centroids lie in [0, 1]^3."""
    pts_list = []
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            for k in (-1, 0, 1, 2):
                for b in A15_BASIS:
                    pts_list.append(b + np.array([i, j, k], dtype=np.float64))
    pts = np.unique(np.round(np.vstack(pts_list), 9), axis=0)

    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    adj: dict[int, set[int]] = {i: set() for i in range(len(pts))}
    for u, v in zip(*np.where((dist > 1e-9) & (dist <= BOND_CUTOFF))):
        if u < v:
            adj[u].add(v)
            adj[v].add(u)

    cliques: list[tuple[int, int, int, int]] = []
    for u in range(len(pts)):
        for v in adj[u]:
            if v <= u:
                continue
            uv = adj[u] & adj[v]
            for w in uv:
                if w <= v:
                    continue
                for x in uv & adj[w]:
                    if x > w:
                        cliques.append((u, v, w, x))

    arr = np.array(cliques, dtype=np.int64)
    centroids = pts[arr].mean(axis=1)
    inside = np.all((centroids >= -1e-9) & (centroids <= 1.0 + 1e-9), axis=1)
    return pts, arr[inside]


_CANONICAL_PTS: np.ndarray | None = None
_CANONICAL_CLIQUES: np.ndarray | None = None


def get_canonical_cliques() -> tuple[np.ndarray, np.ndarray]:
    global _CANONICAL_PTS, _CANONICAL_CLIQUES
    if _CANONICAL_PTS is None:
        _CANONICAL_PTS, _CANONICAL_CLIQUES = build_canonical_cliques()
    return _CANONICAL_PTS, _CANONICAL_CLIQUES


def trilinear_warp(frac: np.ndarray, corners: np.ndarray) -> np.ndarray:
    u, v, w = frac[:, 0], frac[:, 1], frac[:, 2]
    weights = np.column_stack(
        [
            (1 - u) * (1 - v) * (1 - w),
            u * (1 - v) * (1 - w),
            u * v * (1 - w),
            (1 - u) * v * (1 - w),
            (1 - u) * (1 - v) * w,
            u * (1 - v) * w,
            u * v * w,
            (1 - u) * v * w,
        ]
    )
    return weights @ corners


def build_undeformed_hex_grid(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """Shared-vertex Cartesian hex grid covering [0, nx*cs] × [0, ny*cs] × [0, nz*cs]."""
    xs = np.linspace(0.0, nx * cell_size, nx + 1)
    ys = np.linspace(0.0, ny * cell_size, ny + 1)
    zs = np.linspace(0.0, nz * cell_size, nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    global_nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    stride_y = nz + 1
    stride_x = (ny + 1) * (nz + 1)
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    c0 = (ii.ravel() * stride_x) + (jj.ravel() * stride_y) + kk.ravel()

    h0 = c0
    h1 = c0 + stride_x
    h2 = c0 + stride_x + stride_y
    h3 = c0 + stride_y
    h4 = c0 + 1
    h5 = c0 + stride_x + 1
    h6 = c0 + stride_x + stride_y + 1
    h7 = c0 + stride_y + 1
    return global_nodes[np.column_stack([h0, h1, h2, h3, h4, h5, h6, h7])]


def box_sdf(pts: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Signed distance to axis-aligned box [0, Lx] × [0, Ly] × [0, Lz] (negative inside)."""
    half = 0.5 * np.asarray(box_size, dtype=np.float64)
    center = half
    d = np.abs(pts - center) - half
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(np.max(d, axis=-1), 0.0)


def collect_surviving_tets(
    hex_elements: np.ndarray,
    box_size: Sequence[float],
    *,
    deep_thresh: float | None = None,
    cell_size: float | None = None,
) -> np.ndarray:
    """
    Warp A15 cliques into each hex and keep tets that pass the VF gate:

      keep T  iff  (all 4 verts have SDF <= SDF_EPS)
                   or (centroid SDF <= deep_thresh)

    Default deep_thresh = 0.0 (centroid must be inside or on the box).
    """
    if deep_thresh is None:
        deep_thresh = 0.0
    box = np.asarray(box_size, dtype=np.float64)
    pts, cliques = get_canonical_cliques()
    kept: list[np.ndarray] = []
    for corners in hex_elements:
        phys = trilinear_warp(pts, corners)
        for clique in cliques:
            tet = phys[clique]
            sdf_v = box_sdf(tet, box)
            sdf_c = float(box_sdf(tet.mean(axis=0, keepdims=True), box)[0])
            if np.all(sdf_v <= SDF_EPS) or sdf_c <= deep_thresh:
                kept.append(tet)
    if not kept:
        return np.empty((0, 4, 3), dtype=np.float64)
    
    seen = set()
    unique_kept = []
    for tet in kept:
        key = tuple(sorted(tuple(np.round(v, 4)) for v in tet))
        if key not in seen:
            seen.add(key)
            unique_kept.append(tet)
    return np.asarray(unique_kept, dtype=np.float64)


def generate_pristine_kagome(
    surviving_tets: np.ndarray,
    *,
    round_decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int, int]], np.ndarray]:
    """
    Face-centroid Kagome on surviving tets.

    Returns
    -------
    nodes, struts, kagome_to_face, unique_tet_verts
        kagome_to_face maps each Kagome node index to its parent face as a
        sorted triplet of global tet-vertex indices into unique_tet_verts.
    """
    if surviving_tets.size == 0:
        empty_nodes = np.empty((0, 3), dtype=np.float64)
        empty_struts = np.empty((0, 2), dtype=np.int64)
        return empty_nodes, empty_struts, {}, empty_nodes

    faces = surviving_tets[:, FACE_TRIPLETS]
    centroids = faces.mean(axis=2)
    flat = np.round(centroids.reshape(-1, 3), decimals=round_decimals)
    nodes, inverse = np.unique(flat, axis=0, return_inverse=True)
    tet_node_indices = inverse.reshape(-1, 4)

    strut_set: set[tuple[int, int]] = set()
    for tet_n in tet_node_indices:
        for a, b in combinations(range(4), 2):
            u, v = int(tet_n[a]), int(tet_n[b])
            strut_set.add((min(u, v), max(u, v)))
    struts = np.array(sorted(strut_set), dtype=np.int64)

    # Global tet-vertex IDs for hanging-face tests during snap.
    flat_verts = np.round(surviving_tets.reshape(-1, 3), decimals=3)
    unique_verts, inv_verts = np.unique(flat_verts, axis=0, return_inverse=True)
    tet_ids = inv_verts.reshape(-1, 4)

    kagome_to_face: dict[int, tuple[int, int, int]] = {}
    for i in range(len(surviving_tets)):
        for j, trip in enumerate(FACE_TRIPLETS):
            g_idx = int(inverse[i * 4 + j])
            face_verts = tuple(sorted(int(v) for v in tet_ids[i, list(trip)]))
            kagome_to_face[g_idx] = face_verts  # type: ignore[assignment]

    return nodes, struts, kagome_to_face, unique_verts


def node_degrees(n_nodes: int, struts: np.ndarray) -> np.ndarray:
    degrees = np.zeros(n_nodes, dtype=np.int64)
    if struts.size == 0:
        return degrees
    for u, v in struts:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def apply_valency_planar_snap(
    nodes: np.ndarray,
    struts: np.ndarray,
    box_size: Sequence[float],
    kagome_to_face: dict[int, tuple[int, int, int]],
    unique_tet_verts: np.ndarray,
    *,
    cell_size: float,
    snap_band_fraction: float = SNAP_BAND_FRACTION,
    max_stretch_valency: int = MAX_STRETCH_VALENCY,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Snap nodes near the six box faces using the quarter-plane valency rules.

    For each coordinate within ``snap_band`` of a face plane C:

    * **Hanging** parent face (straddles C): always allow inward pull; outward
      stretch only if valency <= max_stretch_valency.
    * **Non-hanging**: always allow inward pull (outside → boundary); outward
      stretch only if valency <= max_stretch_valency.
    * Otherwise leave the coordinate recessed (protected).
    """
    tol = snap_band_fraction * float(cell_size)
    lx, ly, lz = (float(v) for v in box_size)
    planes = [
        (0, 0.0, True),
        (0, lx, False),
        (1, 0.0, True),
        (1, ly, False),
        (2, 0.0, True),
        (2, lz, False),
    ]
    degrees = node_degrees(len(nodes), struts)
    snapped = nodes.copy()
    counts = {"hanging": 0, "inward_or_low_valency": 0, "protected": 0, "shifts": 0}

    for idx in range(len(snapped)):
        valency = int(degrees[idx])
        face_verts = kagome_to_face.get(idx)
        for coord_idx, plane_c, is_min in planes:
            val = float(snapped[idx, coord_idx])
            if abs(val - plane_c) > tol:
                continue

            is_hanging = False
            if face_verts is not None and len(unique_tet_verts):
                face_coords = unique_tet_verts[list(face_verts)]
                face_vals = face_coords[:, coord_idx]
                is_hanging = bool(
                    (np.max(face_vals) > plane_c) and (np.min(face_vals) < plane_c)
                )

            # Outward stretch: node is inside the box and would move toward the face.
            is_stretch = (val > plane_c + 1e-5) if is_min else (val < plane_c - 1e-5)

            allow = False
            bucket = "protected"
            if is_hanging:
                if is_stretch:
                    if valency <= max_stretch_valency:
                        allow = True
                        bucket = "hanging"
                else:
                    allow = True
                    bucket = "hanging"
            else:
                if not is_stretch:
                    allow = True
                    bucket = "inward_or_low_valency"
                elif valency <= max_stretch_valency:
                    allow = True
                    bucket = "inward_or_low_valency"

            if allow:
                if abs(val - plane_c) > 1e-5:
                    counts["shifts"] += 1
                snapped[idx, coord_idx] = plane_c
                counts[bucket] += 1
            else:
                counts["protected"] += 1

    return snapped, counts


def shared_edge_surface_dual(
    surviving_tets: np.ndarray,
    lattice_nodes: np.ndarray,
    lattice_struts: np.ndarray,
    *,
    match_tol: float = 1e-4,
    vert_round_decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Shared-edge surface dual (Track C).

    Exposed tet faces (global count == 1) that share an edge contribute a cage
    strut between their face-centroid Kagome nodes. Native Kagome edges are
    excluded. No proximity stitching.
    """
    report = {
        "exposed_faces": 0,
        "edge_adjacencies": 0,
        "map_failures": 0,
        "cage_struts": 0,
    }
    empty = np.empty((0, 2), dtype=np.int64)
    empty_nodes = np.empty(0, dtype=np.int64)
    if surviving_tets.size == 0 or lattice_nodes.size == 0:
        return empty_nodes, empty, report

    n_tets = surviving_tets.shape[0]
    flat_verts = np.round(surviving_tets.reshape(-1, 3), decimals=vert_round_decimals)
    unique_verts, inv = np.unique(flat_verts, axis=0, return_inverse=True)
    tet_ids = inv.reshape(n_tets, 4)

    all_faces = tet_ids[:, FACE_TRIPLETS].reshape(-1, 3)
    all_faces_sorted = np.sort(all_faces, axis=1)
    unique_faces, counts = np.unique(all_faces_sorted, axis=0, return_counts=True)
    exposed_faces = unique_faces[counts == 1]
    report["exposed_faces"] = int(len(exposed_faces))
    if len(exposed_faces) == 0:
        return empty_nodes, empty, report

    exposed_centroids = unique_verts[exposed_faces].mean(axis=1)

    e0 = np.sort(exposed_faces[:, [0, 1]], axis=1)
    e1 = np.sort(exposed_faces[:, [0, 2]], axis=1)
    e2 = np.sort(exposed_faces[:, [1, 2]], axis=1)
    all_edges = np.vstack([e0, e1, e2])
    face_indices = np.tile(np.arange(len(exposed_faces)), 3)
    _, inv_edges = np.unique(all_edges, axis=0, return_inverse=True)
    order = np.argsort(inv_edges)
    sorted_inv = inv_edges[order]
    sorted_faces = face_indices[order]
    match = sorted_inv[:-1] == sorted_inv[1:]
    adjacency = np.column_stack([sorted_faces[:-1][match], sorted_faces[1:][match]])
    report["edge_adjacencies"] = int(len(adjacency))

    tree = cKDTree(lattice_nodes)
    dists, matched = tree.query(exposed_centroids, distance_upper_bound=match_tol)
    invalid = (dists > match_tol) | (matched == len(lattice_nodes))
    report["map_failures"] = int(np.sum(invalid))
    valid = set(np.where(~invalid)[0].tolist())

    native = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in lattice_struts}
    cage_set: set[tuple[int, int]] = set()
    for a, b in adjacency:
        if int(a) not in valid or int(b) not in valid:
            continue
        ga = int(matched[a])
        gb = int(matched[b])
        pair = (min(ga, gb), max(ga, gb))
        if pair not in native:
            cage_set.add(pair)

    cage_struts = (
        np.array(sorted(cage_set), dtype=np.int64) if cage_set else empty
    )
    report["cage_struts"] = int(len(cage_struts))
    surf_nodes = (
        np.unique(matched[list(valid)]) if valid else empty_nodes
    )
    return surf_nodes, cage_struts, report


def compact_subgraph(
    nodes: np.ndarray,
    struts: np.ndarray,
    keep_node_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes[keep] and struts remapped into that index space."""
    if keep_node_ids.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)
    keep = np.asarray(keep_node_ids, dtype=np.int64)
    idx_map = {int(old): new for new, old in enumerate(keep.tolist())}
    keep_set = set(idx_map)
    compact_struts = [
        (idx_map[int(u)], idx_map[int(v)])
        for u, v in struts
        if int(u) in keep_set and int(v) in keep_set
    ]
    out_struts = (
        np.array(compact_struts, dtype=np.int64)
        if compact_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes[keep], out_struts


def build_a15_kagome_box(
    box_size: Sequence[float],
    cell_size: float = 5.0,
    *,
    apply_snap: bool = True,
    deep_thresh: float = 0.0,
) -> dict:
    """
    Full undeformed A15 Kagome pipeline for an axis-aligned box.

    ``box_size`` extents must lie on the quarter-cell grid
    (k * 0.25 * cell_size).

    Returns a dict with nodes/struts (pre- and post-snap), cage, reports.
    """
    box = np.asarray(box_size, dtype=np.float64)
    for name, extent in zip(("Lx", "Ly", "Lz"), box):
        assert_quarter_cell_extent(float(extent), cell_size, name=name)

    nx = int(np.ceil(box[0] / cell_size - 1e-12))
    ny = int(np.ceil(box[1] / cell_size - 1e-12))
    nz = int(np.ceil(box[2] / cell_size - 1e-12))

    hex_elements = build_undeformed_hex_grid(nx, ny, nz, cell_size)
    surviving_tets = collect_surviving_tets(
        hex_elements, box, deep_thresh=deep_thresh, cell_size=cell_size
    )
    nodes, struts, kagome_to_face, unique_verts = generate_pristine_kagome(surviving_tets)

    snap_report: dict[str, int] = {}
    nodes_snap = nodes
    if apply_snap and len(nodes):
        nodes_snap, snap_report = apply_valency_planar_snap(
            nodes,
            struts,
            box,
            kagome_to_face,
            unique_verts,
            cell_size=cell_size,
        )

    # Shared-edge dual uses unsnapped centroids for exact face→node matching,
    # then applies the same strut indices on the snapped node table.
    surf_nodes, cage_struts, cage_report = shared_edge_surface_dual(
        surviving_tets, nodes, struts
    )

    return {
        "box_size": box,
        "cell_size": float(cell_size),
        "grid_cells": (nx, ny, nz),
        "hex_elements": hex_elements,
        "surviving_tets": surviving_tets,
        "nodes": nodes,
        "struts": struts,
        "nodes_snap": nodes_snap,
        "snap_report": snap_report,
        "surf_nodes": surf_nodes,
        "cage_struts": cage_struts,
        "cage_report": cage_report,
    }
