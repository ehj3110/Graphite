"""
Graphite Math - Gaussian Random Field (GRF) Spinodal Decomposition Engine

This module implements the mathematical foundations for generating Gaussian
Random Field (GRF) spinodal metamaterial architectures via superpositions
of standing cosine waves with random wavevectors and phases.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfinv


def _sample_marsaglia_unit_sphere(
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample uniformly distributed unit vectors on S^2 using Marsaglia's (1972) method.

    Parameters
    ----------
    num_samples : int
        Number of unit vectors to sample.
    rng : np.random.Generator
        NumPy random number generator instance.

    Returns
    -------
    np.ndarray, shape (num_samples, 3), dtype float64
        Uniformly distributed unit vectors on the 2-sphere.
    """
    pts: list[np.ndarray] = []
    total_accepted = 0
    # Batch size: acceptance rate of unit disc in square [-1, 1]^2 is pi/4 (~0.785)
    batch_size = max(int(num_samples * 1.5), 32)

    while total_accepted < num_samples:
        u = rng.uniform(-1.0, 1.0, size=batch_size)
        v = rng.uniform(-1.0, 1.0, size=batch_size)
        s = u * u + v * v
        mask = (s > 0.0) & (s < 1.0)
        u_acc = u[mask]
        v_acc = v[mask]
        s_acc = s[mask]
        factor = 2.0 * np.sqrt(1.0 - s_acc)
        x = u_acc * factor
        y = v_acc * factor
        z = 1.0 - 2.0 * s_acc
        accepted = np.column_stack([x, y, z])
        pts.append(accepted)
        total_accepted += len(accepted)

    all_pts = np.vstack(pts)
    return all_pts[:num_samples]


def generate_spinodal_wavevectors(
    num_waves: int,
    wavelength: float,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate wavevectors and random phases for Gaussian Random Field spinodal decomposition.

    Parameters
    ----------
    num_waves : int
        Number of standing cosine wave components N.
    wavelength : float
        Characteristic pore / feature wavelength lambda in millimeters.
    anisotropy : tuple of 3 floats, optional
        Scaling factors (ax, ay, az) for directional anisotropy, by default (1.0, 1.0, 1.0).
    seed : int or None, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    wavevectors : np.ndarray, shape (num_waves, 3), dtype float64
        Wavevectors k_i scaled by k0 and normalized after anisotropy scaling.
    phases : np.ndarray, shape (num_waves,), dtype float64
        Uniform random phases phi_i in [0, 2*pi).

    Raises
    ------
    ValueError
        If num_waves <= 0, wavelength <= 0, or any anisotropy component <= 0.
    """
    if num_waves <= 0:
        raise ValueError(f"num_waves must be > 0, got {num_waves}")
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")

    aniso = np.asarray(anisotropy, dtype=np.float64)
    if aniso.shape != (3,):
        raise ValueError(f"anisotropy must be a 3-element tuple or array, got {anisotropy}")
    if np.any(aniso <= 0.0):
        raise ValueError(f"anisotropy components must be strictly positive, got {anisotropy}")

    rng = np.random.default_rng(seed)

    # Base wavenumber k0 = 2*pi / lambda
    k0 = (2.0 * np.pi) / float(wavelength)

    # Marsaglia uniform unit vectors on S^2
    n_hat = _sample_marsaglia_unit_sphere(num_waves, rng)

    # Anisotropic scaling: A * n_hat / ||A * n_hat||
    scaled_n = n_hat * aniso
    scaled_norms = np.linalg.norm(scaled_n, axis=1, keepdims=True)
    wavevectors = k0 * (scaled_n / scaled_norms)

    # Random phases phi_i ~ U(0, 2*pi)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=num_waves)

    return wavevectors, phases


def evaluate_spinodal_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    wavelength: float,
    num_waves: int = 120,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    seed: int | None = 42,
) -> np.ndarray:
    """
    Evaluate the Gaussian Random Field (GRF) scalar field on a 3D coordinate grid.

    The scalar field is evaluated via the superposition of N standing cosine waves:
        F(x) = sqrt(2 / N) * sum_{i=1}^N cos(k_i . x + phi_i)

    By the Central Limit Theorem, F(x) is asymptotically standard normal N(0, 1).

    Memory Efficiency:
    Calculations avoid constructing 4D intermediate tensors (N, nx, ny, nz).
    Instead, wave evaluations are accumulated in-place into a single preallocated
    float32 array.

    Parameters
    ----------
    X, Y, Z : np.ndarray
        Meshgrid coordinate arrays in millimeters (e.g. from indexing="ij").
    wavelength : float
        Characteristic pore / feature wavelength in millimeters.
    num_waves : int, optional
        Number of wave components, by default 120.
    anisotropy : tuple of 3 floats, optional
        Directional anisotropy scaling vector (ax, ay, az), by default (1.0, 1.0, 1.0).
    seed : int or None, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    np.ndarray, dtype float32
        Evaluated GRF scalar field F(x) with shape matching X, Y, Z.

    Raises
    ------
    ValueError
        If coordinate array shapes do not match, or if parameters are invalid.
    """
    if X.shape != Y.shape or X.shape != Z.shape:
        raise ValueError(
            f"Coordinate arrays X, Y, Z must have identical shapes, got {X.shape}, {Y.shape}, {Z.shape}"
        )

    wavevectors, phases = generate_spinodal_wavevectors(
        num_waves=num_waves,
        wavelength=wavelength,
        anisotropy=anisotropy,
        seed=seed,
    )

    # Preallocate output field and single temporary argument buffer
    field = np.zeros(X.shape, dtype=np.float32)
    arg_buf = np.empty(X.shape, dtype=np.float32)

    X_f = np.asarray(X, dtype=np.float32)
    Y_f = np.asarray(Y, dtype=np.float32)
    Z_f = np.asarray(Z, dtype=np.float32)

    # In-place accumulation across N modes without 4D allocation
    for i in range(num_waves):
        kx = np.float32(wavevectors[i, 0])
        ky = np.float32(wavevectors[i, 1])
        kz = np.float32(wavevectors[i, 2])
        phi = np.float32(phases[i])

        np.multiply(X_f, kx, out=arg_buf)
        arg_buf += ky * Y_f
        arg_buf += kz * Z_f
        arg_buf += phi
        np.cos(arg_buf, out=arg_buf)
        field += arg_buf

    scale = np.float32(np.sqrt(2.0 / num_waves))
    field *= scale

    return field


def threshold_spinodal_field(
    F: np.ndarray,
    solid_fraction: float,
    is_sheet: bool = False,
) -> np.ndarray:
    """
    Threshold a standard normal Gaussian Random Field to achieve a target solid fraction.

    In Graphite's level-set convention, solid regions are defined by:
        solid_field(x) <= 0.

    1. Skeletal / Network Topology (is_sheet=False):
       Threshold t is derived from standard normal CDF Phi(t) = phi:
           t = sqrt(2) * erfinv(2 * phi - 1)
           solid_field(x) = F(x) - t <= 0  ==>  F(x) <= t  (solid)
       This yields P(solid) = Phi(t) = phi.

    2. Sheet / Lamellar Topology (is_sheet=True):
       Threshold t_sheet is derived from P(|F| <= t_sheet) = erf(t_sheet / sqrt(2)) = phi:
           t_sheet = sqrt(2) * erfinv(phi)
           solid_field(x) = |F(x)| - t_sheet <= 0  ==>  |F(x)| <= t_sheet  (solid)
       This yields P(solid) = phi.

    Parameters
    ----------
    F : np.ndarray
        Evaluated GRF field F(x) ~ N(0, 1).
    solid_fraction : float
        Target solid volume fraction phi in the open interval (0, 1).
    is_sheet : bool, optional
        If True, generates a sheet/lamellar topology. If False, generates a
        skeletal/network topology. By default False.

    Returns
    -------
    np.ndarray, dtype float32
        Implicit solid field where values <= 0 define solid material.

    Raises
    ------
    ValueError
        If solid_fraction is not in (0, 1).
    """
    if not (0.0 < solid_fraction < 1.0):
        raise ValueError(
            f"solid_fraction must be in the open interval (0, 1), got {solid_fraction}"
        )

    F_arr = np.asarray(F, dtype=np.float32)

    if is_sheet:
        t_sheet = np.float32(np.sqrt(2.0) * erfinv(solid_fraction))
        return np.abs(F_arr) - t_sheet
    else:
        t = np.float32(np.sqrt(2.0) * erfinv(2.0 * solid_fraction - 1.0))
        return F_arr - t
