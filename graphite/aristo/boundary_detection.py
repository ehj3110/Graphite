"""
Aristo Boundary Detection — Robust BC Heuristics

Identifies fixed and loaded faces for Aristo FEA runs.
Phase 1 uses uniaxial heuristics combining:
  1. Principal component (projection along the load direction)
  2. Surface normal alignment (ensuring force is applied to logical top/bottom)

Author: Graphite / Aristo Project
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from graphite.geometry.surface_picking import compute_face_surface_ids

def detect_boundary_masks(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_direction: np.ndarray,
    fixed_quantile: float = 0.05,
    load_quantile: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristically identifies which surface faces are 'Fixed' and which are 'Loaded'.

    Fixed Faces:
        Faces on the 'back' of the part relative to load direction.
        (e.g., if loading -Z, these are faces at the bottom).

    Loaded Faces:
        Faces on the 'front' of the part relative to load direction.
        (e.g., if loading -Z, these are faces at the top).

    Logic:
      - Uses projection of centroids onto the load axis.
      - Filters by face normals (Loaded faces typically have normals pointing
        opposite to the load vector if the load is compressive pressure).

    Parameters
    ----------
    nodes : (N, 3)
    surface_faces : (K, 3)
    load_direction : (3,) unit vector
    fixed_quantile : float
    load_quantile : float

    Returns
    -------
    fixed_mask : (K,) bool
    load_mask  : (K,) bool
    """
    if surface_faces.shape[0] == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)

    # Calculate centroids
    verts = nodes[surface_faces]
    centroids = verts.mean(axis=1) # (K, 3)

    # Calculate face normals
    v0 = verts[:, 0, :]
    v1 = verts[:, 1, :]
    v2 = verts[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_mags = np.linalg.norm(normals, axis=1, keepdims=True)
    unit_normals = normals / (norm_mags + 1e-15)

    # 1. Spatial Projection
    # Project centroids onto load direction
    proj = centroids @ load_direction # (K,)

    # 2. Normal Alignment
    # Loaded faces usually point 'against' the load (cos theta < 0)
    # for a compressive load (like a top plate pushing down on a top face).
    normal_alignment = unit_normals @ load_direction

    # Candidates for FIXED (back of part)
    # Highest projection values
    fixed_thresh = np.quantile(proj, 1.0 - fixed_quantile)
    fixed_mask = proj >= fixed_thresh

    # Candidates for LOADED (front of part)
    # Lowest projection values + pointing against load
    load_proj_thresh = np.quantile(proj, load_quantile)
    load_mask = (proj <= load_proj_thresh) & (normal_alignment < 0.1)

    # Safety: ensure at least one face
    if not fixed_mask.any():
        fixed_mask[np.argmax(proj)] = True
    if not load_mask.any():
        # Fall back to pure spatial if normal filter is too strict
        load_mask[np.argmin(proj)] = True

    return fixed_mask, load_mask


def detect_boundary_masks_z_band(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_direction: np.ndarray,
    band_fraction: float = 0.01,
    *,
    normal_filter_load: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed/load faces as thin bands at the extremes along the load axis.

    ``band_fraction`` is the half-band thickness as a fraction of total height
    (e.g. 0.01 → top/bottom 1% of span). Loaded faces sit at the end the load
    pushes toward (max axis position for compression along -Z).
    """
    if surface_faces.shape[0] == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if not (0.0 < band_fraction < 0.5):
        raise ValueError(f"band_fraction must be in (0, 0.5), got {band_fraction}.")

    d_hat = np.asarray(load_direction, dtype=np.float64)
    d_norm = np.linalg.norm(d_hat)
    if d_norm < 1e-12:
        raise ValueError("load_direction must be non-zero.")
    d_hat = d_hat / d_norm

    verts = nodes[surface_faces]
    centroids = verts.mean(axis=1)

    # Axis position increases toward the loaded end (opposite load arrow).
    axis_pos = -centroids @ d_hat
    span = float(axis_pos.max() - axis_pos.min())
    if span <= 0.0:
        raise ValueError("Mesh has zero extent along the load axis.")

    tol = band_fraction * span
    fixed_mask = axis_pos <= axis_pos.min() + tol
    load_mask = axis_pos >= axis_pos.max() - tol

    if normal_filter_load:
        v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
        normals = np.cross(v1 - v0, v2 - v0)
        unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)
        load_mask &= unit_normals @ d_hat < 0.1

    if not fixed_mask.any():
        fixed_mask[np.argmin(axis_pos)] = True
    if not load_mask.any():
        load_mask[np.argmax(axis_pos)] = True

    return fixed_mask, load_mask


def _find_z_plane_from_surface_vertices(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    *,
    min_vertices_at_plane: int = 100,
    pick: str = "max",
) -> tuple[float, int]:
    """
    Dominant top or bottom Z among surface nodes (exact ``float64`` equality).

    ``pick='max'``: highest Z with at least ``min_vertices_at_plane`` surface nodes.
    ``pick='min'``: lowest Z with the same threshold.
    """
    if min_vertices_at_plane < 1:
        raise ValueError(f"min_vertices_at_plane must be >= 1, got {min_vertices_at_plane}.")
    if pick not in ("max", "min"):
        raise ValueError(f"pick must be 'max' or 'min', got {pick!r}.")
    if surface_faces.shape[0] == 0:
        raise ValueError("No surface faces for Z-plane detection.")

    surf_ids = np.unique(surface_faces.ravel())
    z_vals = nodes[surf_ids, 2]
    unique_z, counts = np.unique(z_vals, return_counts=True)
    order = np.argsort(-unique_z) if pick == "max" else np.argsort(unique_z)
    for idx in order:
        if int(counts[idx]) >= min_vertices_at_plane:
            return float(unique_z[idx]), int(counts[idx])

    best_idx = int(order[0])
    which = "top" if pick == "max" else "bottom"
    raise ValueError(
        f"No {which} Z level with >= {min_vertices_at_plane} surface vertices "
        f"(best: z={unique_z[best_idx]:.9f} mm, n={counts[best_idx]})."
    )


def find_cap_z_from_surface_vertices(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    *,
    min_vertices_at_plane: int = 100,
) -> tuple[float, int]:
    """Highest Z among surface nodes with at least ``min_vertices_at_plane`` matches."""
    return _find_z_plane_from_surface_vertices(
        nodes,
        surface_faces,
        min_vertices_at_plane=min_vertices_at_plane,
        pick="max",
    )


def find_floor_z_from_surface_vertices(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    *,
    min_vertices_at_plane: int = 100,
) -> tuple[float, int]:
    """Lowest Z among surface nodes with at least ``min_vertices_at_plane`` matches."""
    return _find_z_plane_from_surface_vertices(
        nodes,
        surface_faces,
        min_vertices_at_plane=min_vertices_at_plane,
        pick="min",
    )


def detect_boundary_masks_flat_top_vertex_plane(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_direction: np.ndarray,
    fixed_band_fraction: float = 0.01,
    *,
    min_vertices_at_plane: int = 100,
    top_normal_z_min: float = 0.99,
    bottom_normal_z_min: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fixed: faces whose **all three vertices** lie on the dominant bottom Z plane
    (exact coordinate match) and whose outward normal has
    ``normal_z < -bottom_normal_z_min`` (downward-facing for compression along -Z).

    Load: faces whose **all three vertices** lie on the dominant top Z plane
    (exact coordinate match) and whose outward normal has ``normal_z > top_normal_z_min``.

    Cap/floor Z levels are the highest/lowest discrete Z with at least
    ``min_vertices_at_plane`` surface nodes — no centroid tolerance band.
    """
    if surface_faces.shape[0] == 0:
        return (
            np.array([], dtype=bool),
            np.array([], dtype=bool),
            {},
        )
    if not (0.0 < fixed_band_fraction < 0.5):
        raise ValueError(
            f"fixed_band_fraction must be in (0, 0.5), got {fixed_band_fraction}."
        )

    d_hat = np.asarray(load_direction, dtype=np.float64)
    d_norm = np.linalg.norm(d_hat)
    if d_norm < 1e-12:
        raise ValueError("load_direction must be non-zero.")
    d_hat = d_hat / d_norm

    verts = nodes[surface_faces]
    centroids = verts.mean(axis=1)

    v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)

    axis_pos = -centroids @ d_hat
    span = float(axis_pos.max() - axis_pos.min())
    if span <= 0.0:
        raise ValueError("Mesh has zero extent along the load axis.")

    z_floor, n_vertices_at_floor = find_floor_z_from_surface_vertices(
        nodes,
        surface_faces,
        min_vertices_at_plane=min_vertices_at_plane,
    )
    z_cap, n_vertices_at_cap = find_cap_z_from_surface_vertices(
        nodes,
        surface_faces,
        min_vertices_at_plane=min_vertices_at_plane,
    )

    z_tri = verts[:, :, 2]
    all_vertices_on_cap = np.all(z_tri == z_cap, axis=1)
    all_vertices_on_floor = np.all(z_tri == z_floor, axis=1)
    upward = unit_normals[:, 2] > float(top_normal_z_min)
    downward = unit_normals[:, 2] < -float(bottom_normal_z_min)
    load_mask = all_vertices_on_cap & upward
    fixed_mask = all_vertices_on_floor & downward

    n_on_cap = int(all_vertices_on_cap.sum())
    n_on_cap_downward = int((all_vertices_on_cap & ~upward).sum())
    n_on_floor = int(all_vertices_on_floor.sum())
    n_on_floor_upward = int((all_vertices_on_floor & ~downward).sum())
    n_old_style_centroid = int(
        (
            (centroids[:, 2] >= z_cap)
            & (unit_normals[:, 2] > float(top_normal_z_min))
        ).sum()
    )
    n_old_style_floor_band = int(
        (axis_pos <= axis_pos.min() + fixed_band_fraction * span).sum()
    )

    meta = {
        "bc_load_mode": "flat_top_vertex_plane",
        "z_cap_mm": z_cap,
        "z_floor_mm": z_floor,
        "n_surface_vertices_at_z_cap": n_vertices_at_cap,
        "n_surface_vertices_at_z_floor": n_vertices_at_floor,
        "min_vertices_at_plane": int(min_vertices_at_plane),
        "top_normal_z_min": float(top_normal_z_min),
        "bottom_normal_z_min": float(bottom_normal_z_min),
        "n_load_faces": int(load_mask.sum()),
        "n_fixed_faces": int(fixed_mask.sum()),
        "n_faces_all_vertices_on_cap": n_on_cap,
        "n_faces_on_cap_excluded_downward_normal": n_on_cap_downward,
        "n_faces_all_vertices_on_floor": n_on_floor,
        "n_faces_on_floor_excluded_upward_normal": n_on_floor_upward,
        "n_faces_old_centroid_style_at_z_cap": n_old_style_centroid,
        "n_faces_old_z_band_style_fixed": n_old_style_floor_band,
    }

    if not fixed_mask.any():
        raise ValueError(
            "Vertex-plane fixed BC found zero faces. "
            f"z_floor={z_floor:.9f} mm, vertices_at_floor={n_vertices_at_floor}, "
            f"on_floor_faces={n_on_floor}, excluded_upward={n_on_floor_upward}, "
            f"normal_z_min={bottom_normal_z_min}."
        )
    if not load_mask.any():
        raise ValueError(
            "Vertex-plane load BC found zero faces. "
            f"z_cap={z_cap:.9f} mm, vertices_at_cap={n_vertices_at_cap}, "
            f"on_cap_faces={n_on_cap}, excluded_downward={n_on_cap_downward}, "
            f"normal_z_min={top_normal_z_min}."
        )

    return fixed_mask, load_mask, meta


def detect_boundary_masks_flat_top_load(
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_direction: np.ndarray,
    fixed_band_fraction: float = 0.01,
    *,
    z_tolerance_mm: float = 1e-4,
    top_normal_z_min: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed: bottom band along the load axis (same Z-band paradigm as z_band mode).
    Load: flat top cap only — tight Z tolerance plus upward-facing normal gate.

    For compression along -Z, loaded faces must satisfy:
      centroid_z >= z_max - z_tolerance_mm
      normal_z > top_normal_z_min
    """
    if surface_faces.shape[0] == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    if not (0.0 < fixed_band_fraction < 0.5):
        raise ValueError(
            f"fixed_band_fraction must be in (0, 0.5), got {fixed_band_fraction}."
        )

    d_hat = np.asarray(load_direction, dtype=np.float64)
    d_norm = np.linalg.norm(d_hat)
    if d_norm < 1e-12:
        raise ValueError("load_direction must be non-zero.")
    d_hat = d_hat / d_norm

    verts = nodes[surface_faces]
    centroids = verts.mean(axis=1)

    v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0)
    unit_normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-15)

    axis_pos = -centroids @ d_hat
    span = float(axis_pos.max() - axis_pos.min())
    if span <= 0.0:
        raise ValueError("Mesh has zero extent along the load axis.")

    fixed_tol = fixed_band_fraction * span
    fixed_mask = axis_pos <= axis_pos.min() + fixed_tol

    z_max = float(nodes[:, 2].max())
    load_mask = (centroids[:, 2] >= z_max - float(z_tolerance_mm)) & (
        unit_normals[:, 2] > float(top_normal_z_min)
    )

    if not fixed_mask.any():
        fixed_mask[np.argmin(axis_pos)] = True
    if not load_mask.any():
        raise ValueError(
            "Flat-top load BC found zero faces. "
            f"z_max={z_max:.6f} mm, tolerance={z_tolerance_mm} mm, "
            f"normal_z_min={top_normal_z_min}."
        )

    return fixed_mask, load_mask


def map_surfaces_to_fea(
    mesh: trimesh.Trimesh,
    fea_nodes: np.ndarray,
    fea_surface_faces: np.ndarray,
    feature_angle: float = 45.0
) -> np.ndarray:
    """
    Groups FEA surface faces into logical Surface IDs using the input trimesh topology.

    Since the FEA mesh surface is a re-meshing of the input trimesh, we compute
    segmentation on the trimesh and transfer the segment IDs to the FEA mesh
    via nearest-centroid lookup.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Original input mesh used as the source of segments.
    fea_nodes : (N, 3)
    fea_surface_faces : (K, 3)
    feature_angle : float
        Dihedral angle threshold for segmenting the trimesh.

    Returns
    -------
    fea_surface_ids : (K,) np.ndarray (int64)
        The logical Surface ID for each surface face in the FEA mesh.
    """
    # 1. Segment original trimesh
    trimesh_surface_ids = compute_face_surface_ids(mesh, feature_angle)

    # 2. Compute centroids of trimesh faces
    trimesh_centroids = mesh.triangles_center # (F, 3)
    
    # 3. Build KDTree for trimesh centroids
    tree = cKDTree(trimesh_centroids)

    # 4. Compute centroids of FEA surface faces
    fea_surface_verts = fea_nodes[fea_surface_faces]
    fea_centroids = fea_surface_verts.mean(axis=1) # (K, 3)

    # 5. Map FEA face -> nearest Trimesh face -> Surface ID
    _dists, nearest_indices = tree.query(fea_centroids, k=1)
    fea_surface_ids = trimesh_surface_ids[nearest_indices]

    return fea_surface_ids
