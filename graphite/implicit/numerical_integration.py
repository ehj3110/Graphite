"""
Numerically integrated radial phase for implicit Gyroid fields.

Builds a spherically symmetric phase Φ(r) = ∫ ω(s) ds with ω = 2π/L(r) and
linear L(r), then distributes phase along Cartesian axes as U = x̂ Φ/r (etc.)
to avoid cell shearing from naive k(r)·x substitution.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid


def generate_radial_integrated_gyroid(
    grid_shape: tuple[int, int, int],
    physical_size: tuple[float, float, float],
    center: tuple[float, float, float],
    L_center: float,
    L_edge: float,
    r_samples: int = 10_000,
) -> np.ndarray:
    """
    Gyroid scalar field with radially varying pore period via numerical phase integration.

    The voxel grid spans [-sx/2, sx/2] × [-sy/2, sy/2] × [-sz/2, sz/2] (mm).

    Parameters
    ----------
    grid_shape :
        (nx, ny, nz) sample counts.
    physical_size :
        Full width along each axis (mm).
    center :
        (cx, cy, cz) reference for radial distance (mm).
    L_center, L_edge :
        Linear pore period L(r) from r = 0 to r = max(R) on the grid (mm).
    r_samples :
        Number of points in the 1D radial quadrature (default 10000).

    Returns
    -------
    field : ndarray, shape (nx, ny, nz)
    """
    nx, ny, nz = grid_shape
    sx, sy, sz = physical_size
    cx, cy, cz = center

    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("grid_shape components must be >= 1")
    if L_center <= 0 or L_edge <= 0:
        raise ValueError("L_center and L_edge must be positive")
    if r_samples < 2:
        raise ValueError("r_samples must be at least 2")

    x = np.linspace(-sx / 2.0, sx / 2.0, nx)
    y = np.linspace(-sy / 2.0, sy / 2.0, ny)
    z = np.linspace(-sz / 2.0, sz / 2.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    Xc = X - cx
    Yc = Y - cy
    Zc = Z - cz
    R = np.sqrt(Xc**2 + Yc**2 + Zc**2)

    r_max = float(np.max(R))
    r_1d = np.linspace(0.0, r_max, r_samples)

    if r_max < 1e-12:
        L_1d = np.full(r_samples, L_center)
    else:
        L_1d = L_center + (L_edge - L_center) * (r_1d / r_max)

    omega_1d = 2.0 * np.pi / L_1d
    Phi_1d = cumulative_trapezoid(omega_1d, r_1d, initial=0.0)

    Phi_3d = np.interp(R.ravel(), r_1d, Phi_1d).reshape(R.shape)

    R_safe = np.maximum(R, 1e-9)
    scale = Phi_3d / R_safe
    U = Xc * scale
    V = Yc * scale
    W = Zc * scale

    return (
        np.sin(U) * np.cos(V)
        + np.sin(V) * np.cos(W)
        + np.sin(W) * np.cos(U)
    )
