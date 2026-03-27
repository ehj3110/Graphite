"""
Voxelization and signed-distance field (CAD SDF) from a watertight mesh.

Shared by conformal, osteochondral, boundary-graded, and related implicit engines.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt


def voxelize_mesh_and_edt(
    mesh: trimesh.Trimesh,
    resolution: float,
    pad_width: int = 4,
):
    """
    Voxelize mesh, pad the inside mask, build world-aligned grids, and compute CAD SDF.

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

    return X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, nx, ny, nz
