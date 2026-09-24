"""
Phase-integrated implicit Gyroid for a 1D linear pore-size gradient along Z.

Avoids ellipsoidal stretching from substituting a spatially varying wavenumber k(z)
directly into sin(k(z)*z) by integrating the local wavenumber to obtain a true
phase field W(z) = ∫ k_z(ζ) dζ.
"""

from __future__ import annotations

import numpy as np


def generate_1d_integrated_gyroid(
    grid_shape: tuple[int, int, int],
    physical_size: tuple[float, float, float],
    L_start: float,
    L_end: float,
) -> np.ndarray:
    """
    Build a Gyroid scalar field with linear pore period L(z) along Z and
    constant in-plane period L_start for X and Y.

    Parameters
    ----------
    grid_shape :
        (nx, ny, nz) sample counts along X, Y, Z.
    physical_size :
        (sx, sy, sz) physical extent in mm; z runs from 0 to sz (bottom to top).
    L_start, L_end :
        Pore scale L(z) = a*z + b at z=0 and z=H respectively (mm), with H = sz.

    Returns
    -------
    field : ndarray, shape (nx, ny, nz)
        Gyroid combination of integrated phases U, V, W.
    """
    nx, ny, nz = grid_shape
    sx, sy, sz = physical_size
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("grid_shape components must be >= 1")
    if sz <= 0:
        raise ValueError("physical_size[2] (height H) must be positive")

    H = sz
    a = (L_end - L_start) / H
    b = L_start

    x = np.linspace(0.0, sx, nx)
    y = np.linspace(0.0, sy, ny)
    z = np.linspace(0.0, sz, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    U = (2.0 * np.pi / L_start) * X
    V = (2.0 * np.pi / L_start) * Y

    if np.isclose(a, 0.0):
        W = (2.0 * np.pi / L_start) * Z
    else:
        ratio = (a * Z + b) / b
        if np.any(ratio <= 0.0):
            raise ValueError(
                "Phase log domain invalid: require (a*Z + b) / b > 0 everywhere "
                f"(check L_start, L_end, and z in [0, {H}])"
            )
        W = (2.0 * np.pi / a) * np.log(ratio)

    field = (
        np.sin(U) * np.cos(V)
        + np.sin(V) * np.cos(W)
        + np.sin(W) * np.cos(U)
    )
    return field
