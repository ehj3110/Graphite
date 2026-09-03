"""
Graphite Math - Procedural Surface Textures

This module provides analytical mathematical definitions and evaluation functions
for surface micro-textures (microgrooves, micropillars/bumps, knurling, and
spinodal textures) as well as triplanar projection mapping for conformal
surface texturing on 3D meshes and implicit fields.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def microgroove_field(
    X: np.ndarray | float,
    Y: np.ndarray | float,
    Z: np.ndarray | float,
    wavelength_mm: float,
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    profile: str = "sine",
) -> np.ndarray:
    """
    Evaluate directional microgrooves in 3D space.

    Grooves modulate along the specified unit direction vector. For example,
    direction=(0, 0, 1) means groove peaks and valleys alternate along Z,
    forming parallel striation lines in the XY plane.

    Parameters
    ----------
    X, Y, Z : ndarray or float
        Spatial coordinates (mm).
    wavelength_mm : float
        Spatial period (peak-to-peak distance) of the grooves in millimeters.
    direction : tuple of float, optional
        3D direction vector along which grooves modulate. Defaults to (0, 0, 1).
    profile : str, optional
        Groove wave profile: "sine", "triangle", or "square". Defaults to "sine".

    Returns
    -------
    ndarray
        Evaluated field with values normalized in [-1.0, 1.0].
    """
    if wavelength_mm <= 0.0:
        raise ValueError(f"wavelength_mm must be > 0, got {wavelength_mm}")

    d = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-12:
        raise ValueError("direction vector cannot be zero-length")
    d = d / norm

    # Project coordinates along direction axis
    s = d[0] * np.asarray(X, dtype=np.float64) + d[1] * np.asarray(Y, dtype=np.float64) + d[2] * np.asarray(Z, dtype=np.float64)
    k = (2.0 * np.pi) / float(wavelength_mm)
    phase = k * s

    p_type = profile.strip().lower()
    if p_type in ("sine", "sinusoidal"):
        return np.cos(phase)
    elif p_type in ("triangle", "triangular", "v-groove", "v_groove"):
        # Normalized smooth triangular / V-groove series (peaks at 1.0, valley at -1.0)
        norm_factor = 1.0 + 1.0 / 9.0 + 1.0 / 25.0
        return (np.cos(phase) + (1.0 / 9.0) * np.cos(3.0 * phase) + (1.0 / 25.0) * np.cos(5.0 * phase)) / norm_factor
    elif p_type in ("square", "rectangular", "rect"):
        # Smooth square wave approximation peaking at 0, 2pi (value near +1.0) with flat troughs (-1.0)
        return np.tanh(4.0 * np.cos(phase))
    else:
        raise ValueError(f"Unknown profile: '{profile}'. Must be 'rectangular', 'triangular', or 'sine'.")


def bump_field(
    X: np.ndarray | float,
    Y: np.ndarray | float,
    Z: np.ndarray | float,
    wavelength_mm: float,
    profile: str = "nodule",
) -> np.ndarray:
    """
    Evaluate 3D periodic bump / nodular field.

    Parameters
    ----------
    X, Y, Z : ndarray or float
        Spatial coordinates (mm).
    wavelength_mm : float
        Spatial period (center-to-center spacing) of bumps in millimeters.
    profile : str, optional
        Bump profile:
        - "nodule": smooth continuous nodular field in [-1.0, 1.0].
        - "isolated": positive-only hemispherical bumps in [0.0, 1.0].

    Returns
    -------
    ndarray
        Evaluated scalar field.
    """
    if wavelength_mm <= 0.0:
        raise ValueError(f"wavelength_mm must be > 0, got {wavelength_mm}")

    k = (2.0 * np.pi) / float(wavelength_mm)
    kx = k * np.asarray(X, dtype=np.float64)
    ky = k * np.asarray(Y, dtype=np.float64)
    kz = k * np.asarray(Z, dtype=np.float64)

    raw = (np.cos(kx) + np.cos(ky) + np.cos(kz)) / 3.0

    p_type = profile.strip().lower()
    if p_type == "nodule":
        return raw
    elif p_type == "isolated":
        # Clamped positive nodules: zero along valleys and between nodes, peaks at 1.0
        # When X shifts by half wavelength, raw drops to 1/3 (< 0.4), so bumps isolate at grid nodes
        shifted = np.maximum(0.0, raw - 0.4) / 0.6
        return shifted**2
    else:
        raise ValueError(f"Unknown bump profile: {profile}. Must be 'nodule' or 'isolated'.")


def knurl_field(
    X: np.ndarray | float,
    Y: np.ndarray | float,
    Z: np.ndarray | float,
    wavelength_mm: float,
    axis: str = "z",
) -> np.ndarray:
    """
    Evaluate diamond knurling pattern (cross-hatched diagonal ridges).

    Parameters
    ----------
    X, Y, Z : ndarray or float
        Spatial coordinates (mm).
    wavelength_mm : float
        Spatial period (ridge-to-ridge spacing) in millimeters.
    axis : str, optional
        Axis around which diamond knurl is oriented ("x", "y", or "z"). Defaults to "z".

    Returns
    -------
    ndarray
        Evaluated field with values in [-1.0, 1.0].
    """
    if wavelength_mm <= 0.0:
        raise ValueError(f"wavelength_mm must be > 0, got {wavelength_mm}")

    ax = axis.strip().lower()
    if ax == "z":
        u_coord, v_coord = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    elif ax == "y":
        u_coord, v_coord = np.asarray(X, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    elif ax == "x":
        u_coord, v_coord = np.asarray(Y, dtype=np.float64), np.asarray(Z, dtype=np.float64)
    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'x', 'y', or 'z'.")

    k = (2.0 * np.pi) / float(wavelength_mm)
    # Diagonal coordinates rotated 45 degrees
    diag1 = (u_coord + v_coord) / np.sqrt(2.0)
    diag2 = (u_coord - v_coord) / np.sqrt(2.0)

    return np.cos(k * diag1) * np.cos(k * diag2)


def spinodal_spectral_field(
    X: np.ndarray | float,
    Y: np.ndarray | float,
    Z: np.ndarray | float,
    wavelength_mm: float,
    num_waves: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """
    Evaluate isotropic spinodal / Cahn-Hilliard stochastic texture via
    Gaussian random wave superposition.

    Parameters
    ----------
    X, Y, Z : ndarray or float
        Spatial coordinates (mm).
    wavelength_mm : float
        Characteristic spinodal wavelength in millimeters.
    num_waves : int, optional
        Number of spectral wave components, by default 32.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    ndarray
        Evaluated stochastic field approximately distributed in [-1.0, 1.0].
    """
    if wavelength_mm <= 0.0:
        raise ValueError(f"wavelength_mm must be > 0, got {wavelength_mm}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be > 0, got {num_waves}")

    rng = np.random.default_rng(seed)
    # Generate isotropic random unit wave vectors on sphere
    z_dirs = rng.uniform(-1.0, 1.0, num_waves)
    theta = rng.uniform(0.0, 2.0 * np.pi, num_waves)
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - z_dirs**2))
    x_dirs = r_xy * np.cos(theta)
    y_dirs = r_xy * np.sin(theta)
    phases = rng.uniform(0.0, 2.0 * np.pi, num_waves)

    k = (2.0 * np.pi) / float(wavelength_mm)
    kx = k * x_dirs
    ky = k * y_dirs
    kz = k * z_dirs

    x_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(Y, dtype=np.float64)
    z_arr = np.asarray(Z, dtype=np.float64)

    total = np.zeros_like(x_arr, dtype=np.float64)
    for i in range(num_waves):
        dot = kx[i] * x_arr + ky[i] * y_arr + kz[i] * z_arr + phases[i]
        total += np.cos(dot)

    return total * np.sqrt(2.0 / num_waves)


def triplanar_weights(
    normals: np.ndarray,
    sharpness: float = 2.0,
) -> np.ndarray:
    """
    Compute normalized triplanar blending weights from 3D surface normal vectors.

    Parameters
    ----------
    normals : ndarray, shape (..., 3)
        Unit normal vectors.
    sharpness : float, optional
        Blending exponent controlling transition sharpness between projection planes.
        Higher values produce sharper transitions; lower values produce smoother blends.
        Defaults to 2.0.

    Returns
    -------
    ndarray, shape (..., 3)
        Weights (wx, wy, wz) normalized such that wx + wy + wz = 1.0 for each vector.
    """
    n = np.asarray(normals, dtype=np.float64)
    if n.shape[-1] != 3:
        raise ValueError(f"normals must have last dimension 3, got shape {n.shape}")

    # Exponentiate absolute values of normal components
    w = np.abs(n) ** float(sharpness)
    w_sum = np.sum(w, axis=-1, keepdims=True)

    # Where normal is degenerate or zero, assign equal 1/3 weights
    degenerate = w_sum < 1e-12
    w_sum_safe = np.where(degenerate, 1.0, w_sum)
    weights = w / w_sum_safe

    if np.any(degenerate):
        equal = np.full_like(weights, 1.0 / 3.0)
        weights = np.where(degenerate, equal, weights)

    return weights


def triplanar_map(
    points: np.ndarray,
    normals: np.ndarray,
    wavelength_mm: float,
    texture_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    sharpness: float = 2.0,
) -> np.ndarray:
    """
    Apply a 2D texture function to 3D surface points via triplanar projection.

    Projects points onto three orthogonal planes (YZ, XZ, XY), evaluates the 2D
    texture on each plane, and blends the three results using the normal-derived
    triplanar weights.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        3D spatial coordinates of surface vertices or sample points (mm).
    normals : ndarray, shape (N, 3)
        Surface normal vectors at each point.
    wavelength_mm : float
        Spatial period for the texture.
    texture_fn : Callable[[u, v, wavelength_mm], ndarray]
        A 2D texture function taking (u, v, wavelength_mm) and returning
        scalar displacement values.
    sharpness : float, optional
        Triplanar blending exponent, by default 2.0.

    Returns
    -------
    ndarray, shape (N,)
        Blended scalar displacement values at each point.
    """
    pts = np.asarray(points, dtype=np.float64)
    norms = np.asarray(normals, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if norms.ndim != 2 or norms.shape[1] != 3:
        raise ValueError(f"normals must have shape (N, 3), got {norms.shape}")
    if len(pts) != len(norms):
        raise ValueError(f"points and normals must have matching length ({len(pts)} vs {len(norms)})")

    if len(pts) == 0:
        return np.empty((0,), dtype=np.float64)

    weights = triplanar_weights(norms, sharpness=sharpness)

    # Projection onto X-normal plane: (u, v) = (Y, Z)
    val_x = texture_fn(pts[:, 1], pts[:, 2], wavelength_mm)

    # Projection onto Y-normal plane: (u, v) = (X, Z)
    val_y = texture_fn(pts[:, 0], pts[:, 2], wavelength_mm)

    # Projection onto Z-normal plane: (u, v) = (X, Y)
    val_z = texture_fn(pts[:, 0], pts[:, 1], wavelength_mm)

    # Blend using weights
    blended = weights[:, 0] * val_x + weights[:, 1] * val_y + weights[:, 2] * val_z
    return blended
