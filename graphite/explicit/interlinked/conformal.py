"""
Graphite Explicit Interlinked — Conformal Seeding & Inset Culling Module

Provides:
    - Planar grid seeding (nx, ny, nz) with orthonormal tangent frames.
    - Surface conformal seeding (normal n and tangent frame t1, t2).
    - Inset culling: drops candidate rings whose center point does not satisfy
      SDF(center) <= -(R_outer + margin), guaranteeing zero broken or cut rings.
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING
import numpy as np
import trimesh

if TYPE_CHECKING:
    from .patterns import Ring


def seed_planar_grid(
    nx: int,
    ny: int,
    nz: int = 1,
    pitch: float = 10.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed regular grid points with orthonormal frames (normal, t1, t2).

    Args:
        nx, ny, nz: Number of points along grid axes.
        pitch: Center-to-center distance L in mm.
        origin: (x0, y0, z0) origin offset.
        plane_normal: Base orientation normal vector.

    Returns:
        tuple: (centers, normals, t1_vectors, t2_vectors)
            centers: (N, 3) grid coordinates.
            normals: (N, 3) unit normals.
            t1_vectors: (N, 3) primary tangent vectors.
            t2_vectors: (N, 3) secondary tangent vectors.
    """
    orig = np.asarray(origin, dtype=np.float64).reshape(3)
    n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    n = n / float(np.linalg.norm(n))

    # Construct tangent frame
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, up))) > 0.90:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t1 = np.cross(up, n)
    t1 = t1 / float(np.linalg.norm(t1))
    t2 = np.cross(n, t1)
    t2 = t2 / float(np.linalg.norm(t2))

    xs = np.arange(nx, dtype=np.float64) * pitch
    ys = np.arange(ny, dtype=np.float64) * pitch
    zs = np.arange(nz, dtype=np.float64) * pitch

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    local_pts = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    # Transform to frame: orig + x * t1 + y * t2 + z * n
    centers = orig + local_pts[:, 0:1] * t1 + local_pts[:, 1:2] * t2 + local_pts[:, 2:3] * n
    num_pts = len(centers)

    normals = np.tile(n, (num_pts, 1))
    t1_arr = np.tile(t1, (num_pts, 1))
    t2_arr = np.tile(t2, (num_pts, 1))

    return centers, normals, t1_arr, t2_arr


def seed_surface_conformal_frames(
    boundary_mesh: trimesh.Trimesh,
    target_spacing: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points conforming to the surface manifold of a boundary mesh.

    Computes local surface normals and orthonormal tangent bases (t1, t2)
    aligned with principal curvature directions or planar projections.

    Args:
        boundary_mesh: Input trimesh solid mesh.
        target_spacing: Desired average distance between adjacent seeds in mm.

    Returns:
        tuple: (centers, normals, t1_vectors, t2_vectors)
    """
    area = float(boundary_mesh.area)
    # Approximate number of samples from triangular area: area / (spacing^2 * sqrt(3)/4)
    cell_area = 0.866 * (target_spacing ** 2)
    n_samples = max(4, int(round(area / cell_area)))

    samples, face_indices = trimesh.sample.sample_surface(boundary_mesh, n_samples)
    samples = np.asarray(samples, dtype=np.float64)
    normals = np.asarray(boundary_mesh.face_normals[face_indices], dtype=np.float64)

    # Normalize normals
    n_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(n_len, 1e-12)

    # Tangent vectors
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot_up = np.abs(np.dot(normals, up))
    alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    ref_up = np.where(dot_up[:, None] > 0.90, alt_up, up)
    t1 = np.cross(ref_up, normals)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=-1, keepdims=True), 1e-12)
    t2 = np.cross(normals, t1)
    t2 = t2 / np.maximum(np.linalg.norm(t2, axis=-1, keepdims=True), 1e-12)

    return samples, normals, t1, t2


def cull_rings_by_sdf(
    rings: list[Ring],
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    margin: float = 0.5,
) -> tuple[list[Ring], list[Ring]]:
    """
    Apply Inset Culling to candidate rings against a Signed Distance Function.

    Condition:
        A ring of outer radius R_outer is kept if and only if:
            SDF(center) <= -(R_outer + margin)
    where SDF < 0 inside the solid boundary and SDF > 0 outside.

    This ensures that 100% of surviving rings are whole and intact, completely
    contained within the boundary volume with at least `margin` clearance to the skin.
    No rings are ever cut, clipped, or broken.

    Args:
        rings: List of candidate Ring objects.
        sdf_fn: Callable accepting (N, 3) coordinates and returning (N,) signed distances.
        margin: Additional inset buffer distance in mm.

    Returns:
        tuple: (surviving_rings, culled_rings)
    """
    if not rings:
        return [], []

    centers = np.array([r.center for r in rings], dtype=np.float64)
    sdf_vals = np.asarray(sdf_fn(centers), dtype=np.float64).reshape(-1)

    surviving: list[Ring] = []
    culled: list[Ring] = []

    for i, ring in enumerate(rings):
        threshold = -(ring.outer_radius + float(margin))
        if sdf_vals[i] <= threshold:
            surviving.append(ring)
        else:
            culled.append(ring)

    return surviving, culled


def cull_rings_by_mesh(
    rings: list[Ring],
    boundary_mesh: trimesh.Trimesh,
    margin: float = 0.5,
) -> tuple[list[Ring], list[Ring]]:
    """
    Convenience wrapper: Inset-cull candidate rings against a watertight boundary mesh.

    Uses `trimesh.proximity.signed_distance` where inside points have positive distance
    in trimesh convention, converted here to canonical negative-inside convention:
        SDF(x) = -mesh_signed_distance(x)
    """
    if not rings:
        return [], []

    centers = np.array([r.center for r in rings], dtype=np.float64)
    # trimesh signed_distance: positive inside, negative outside
    trimesh_sd = trimesh.proximity.signed_distance(boundary_mesh, centers)
    # Convert to canonical convention: negative inside, positive outside
    canonical_sdf = -np.asarray(trimesh_sd, dtype=np.float64)

    def _mesh_sdf(pts: np.ndarray) -> np.ndarray:
        return canonical_sdf

    return cull_rings_by_sdf(rings, _mesh_sdf, margin=margin)
