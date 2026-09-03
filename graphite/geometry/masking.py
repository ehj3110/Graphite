"""
Voxelization and signed-distance field (CAD SDF) from a watertight mesh.

Shared by conformal, osteochondral, boundary-graded, and related implicit engines.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt

# Multi-axis probe directions for robust inside/outside voting.
_CONTAIN_RAY_DIRS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


def contains_points_multi_ray(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    *,
    n_dirs: int = 7,
) -> np.ndarray:
    """
    Robust solid containment via multi-ray majority vote.

    Prefer generalized winding numbers when trimesh exposes them; otherwise
    cast several rays and take a majority odd-crossing vote so a single
    bad normal / grazing hit cannot flip an interior sample.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), dtype=bool)
    if len(pts) == 0:
        return out

    # Prefer winding numbers when available (libigl / newer trimesh).
    try:
        wn = np.asarray(mesh.winding_number(pts), dtype=np.float64).reshape(-1)
        return np.abs(wn) > 0.5
    except Exception:
        pass

    dirs = _CONTAIN_RAY_DIRS[: max(1, min(int(n_dirs), len(_CONTAIN_RAY_DIRS)))].copy()
    for i in range(len(dirs)):
        nrm = float(np.linalg.norm(dirs[i]))
        if nrm > 1e-14:
            dirs[i] /= nrm

    # Batched ray cast per direction; odd hit count => inside for that ray.
    votes = np.zeros(len(pts), dtype=np.int32)
    for d in dirs:
        try:
            locs, ray_id, _tri = mesh.ray.intersects_location(
                ray_origins=pts,
                ray_directions=np.tile(d, (len(pts), 1)),
                multiple_hits=True,
            )
        except Exception:
            continue
        counts = np.zeros(len(pts), dtype=np.int32)
        if locs is not None and len(locs) and ray_id is not None:
            for rid in np.asarray(ray_id, dtype=np.int64).reshape(-1):
                if 0 <= int(rid) < len(pts):
                    counts[int(rid)] += 1
        votes += (counts % 2 == 1).astype(np.int32)

    # Fallback: trimesh.contains for any unresolved (zero votes cast).
    majority = votes * 2 >= len(dirs)
    try:
        need = votes == 0
        if np.any(need):
            majority[need] = np.asarray(mesh.contains(pts[need]), dtype=bool)
    except Exception:
        pass
    return majority


def voxelize_mesh_and_edt(
    mesh: trimesh.Trimesh,
    resolution: float,
    pad_width: int = 4,
):
    """
    Voxelize mesh, pad the inside mask, build world-aligned grids, and compute CAD SDF.

    Sanitizes the CAD (normals / holes) before voxelization and flips the SDF
    if the solid centroid would otherwise land outside (hollow-core inversion).

    Returns
    -------
    X, Y, Z : ndarray
        Meshgrid coordinates (world mm).
    cad_sdf : ndarray
        Signed distance field: negative inside, positive outside (scaled by resolution).
    padded_min_bound : ndarray, shape (3,)
    padded_max_bound : ndarray, shape (3,)
    nx, ny, nz : int
        Voxel grid dimensions.
    """
    from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf

    mesh = sanitize_cad_mesh_for_sdf(mesh)
    vox = mesh.voxelized(pitch=resolution).fill()
    inside_mask = np.pad(vox.matrix, pad_width, mode="constant", constant_values=False)

    padded_min_bound = np.asarray(vox.translation, dtype=float) - (pad_width * resolution)
    nx, ny, nz = inside_mask.shape
    padded_max_bound = padded_min_bound + (np.array([nx, ny, nz], dtype=float) - 1.0) * resolution

    x_axis = np.arange(nx, dtype=float) * resolution + padded_min_bound[0]
    y_axis = np.arange(ny, dtype=float) * resolution + padded_min_bound[1]
    z_axis = np.arange(nz, dtype=float) * resolution + padded_min_bound[2]

    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    inside_dist = edt(inside_mask)
    outside_dist = edt(~inside_mask)
    cad_sdf = (outside_dist - inside_dist) * resolution

    # Guard against inverted fill: solid centroid must be inside (SDF < 0).
    try:
        centroid = np.asarray(mesh.centroid, dtype=np.float64).reshape(3)
        # Trilinear sample at centroid via nearest voxel
        ijk = np.round((centroid - padded_min_bound) / float(resolution)).astype(int)
        ijk = np.clip(ijk, 0, np.asarray(cad_sdf.shape) - 1)
        sdf_c = float(cad_sdf[tuple(ijk)])
        robust_inside = bool(contains_points_multi_ray(mesh, centroid.reshape(1, 3))[0])
        if robust_inside and sdf_c > 0.0:
            cad_sdf = -cad_sdf
        elif (not robust_inside) and sdf_c < 0.0 and float(getattr(mesh, "volume", 0.0) or 0.0) > 0.0:
            # Winding/ray say outside but volume positive — trust volume + flip
            cad_sdf = -cad_sdf
    except Exception:
        pass

    return X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, nx, ny, nz


def axis_aligned_box_grid(
    width_x_mm: float,
    depth_y_mm: float,
    height_z_mm: float,
    resolution_mm: float,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
    pad_voxels: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    """
    World-aligned sampling grid on an axis-aligned box domain.

    ``pad_voxels`` extends the grid outside the box (same role as EDT padding) so
    marching-cubes can close the exterior skin.

    Returns ``X, Y, Z, grid_origin, spacing`` where ``grid_origin`` is the world
    coordinate of ``X[0,0,0]`` and ``spacing`` is ``(dx, dy, dz)`` between samples.
    """
    ox, oy, oz = float(origin_x), float(origin_y), float(origin_z)
    wx, wy, hz = float(width_x_mm), float(depth_y_mm), float(height_z_mm)
    res = float(resolution_mm)
    pad = int(pad_voxels) * res
    xmin, xmax = ox - pad, ox + wx + pad
    ymin, ymax = oy - pad, oy + wy + pad
    zmin, zmax = oz - pad, oz + hz + pad
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    nz = int(np.ceil((zmax - zmin) / res)) + 1
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    z = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_origin = np.array([x[0], y[0], z[0]], dtype=np.float64)
    spacing = (float(x[1] - x[0]), float(y[1] - y[0]), float(z[1] - z[0]))
    return X, Y, Z, grid_origin, spacing


def axis_aligned_box_sdf(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    width_x_mm: float,
    depth_y_mm: float,
    height_z_mm: float,
) -> np.ndarray:
    """
    Analytic axis-aligned box SDF: negative inside, positive outside (mm).
    """
    ox, oy, oz = float(origin_x), float(origin_y), float(origin_z)
    xmax = ox + float(width_x_mm)
    ymax = oy + float(depth_y_mm)
    zmax = oz + float(height_z_mm)
    x_cap = np.maximum(ox - X, X - xmax)
    y_cap = np.maximum(oy - Y, Y - ymax)
    z_cap = np.maximum(oz - Z, Z - zmax)
    return np.maximum(np.maximum(x_cap, y_cap), z_cap)


def voxelize_cylinder_slab_and_edt(
    radius_mm: float,
    z0_mm: float,
    z1_mm: float,
    resolution: float,
    pad_width: int = 4,
):
    """
    Analytic short-cylinder mask + EDT (avoids trimesh subdivide voxelization).

    Cylinder axis is Z, centered on XY origin, spanning ``[z0_mm, z1_mm]``.
    """
    r = float(radius_mm)
    z0 = float(z0_mm)
    z1 = float(z1_mm)
    res = float(resolution)
    if z1 <= z0:
        raise ValueError(f"Non-positive slab height: z0={z0}, z1={z1}")

    pad = int(pad_width) * res
    xmin, xmax = -r - pad, r + pad
    ymin, ymax = -r - pad, r + pad
    zmin, zmax = z0 - pad, z1 + pad
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    nz = int(np.ceil((zmax - zmin) / res)) + 1

    padded_min_bound = np.array([xmin, ymin, zmin], dtype=np.float64)
    padded_max_bound = padded_min_bound + (np.array([nx, ny, nz], dtype=float) - 1.0) * res

    x_axis = np.arange(nx, dtype=float) * res + xmin
    y_axis = np.arange(ny, dtype=float) * res + ymin
    z_axis = np.arange(nz, dtype=float) * res + zmin
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    radial = np.sqrt(X**2 + Y**2)
    inside_mask = (radial <= r + 1e-9) & (Z >= z0 - 1e-9) & (Z <= z1 + 1e-9)

    inside_dist = edt(inside_mask)
    outside_dist = edt(~inside_mask)
    cad_sdf = (outside_dist - inside_dist) * res

    return X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, nx, ny, nz
