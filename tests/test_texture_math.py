from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.math.textures import (
    bump_field,
    knurl_field,
    microgroove_field,
    spinodal_spectral_field,
    triplanar_map,
    triplanar_weights,
)


def test_microgroove_field_periodicity() -> None:
    wavelength = 0.050  # 50 um
    z_samples = np.array([0.0, 0.025, 0.050, 0.075, 0.100])
    x = np.zeros_like(z_samples)
    y = np.zeros_like(z_samples)

    # Sine profile: cos(k * z)
    val_sine = microgroove_field(x, y, z_samples, wavelength_mm=wavelength, direction=(0, 0, 1), profile="sine")
    assert np.isclose(val_sine[0], 1.0, atol=1e-6)
    assert np.isclose(val_sine[1], -1.0, atol=1e-6)  # half wavelength = valley
    assert np.isclose(val_sine[2], 1.0, atol=1e-6)   # full wavelength = peak
    assert np.isclose(val_sine[4], 1.0, atol=1e-6)   # two wavelengths = peak

    # Triangle profile
    val_tri = microgroove_field(x, y, z_samples, wavelength_mm=wavelength, direction=(0, 0, 1), profile="triangle")
    assert np.all(val_tri >= -1.0 - 1e-6) and np.all(val_tri <= 1.0 + 1e-6)
    assert np.isclose(val_tri[0], 1.0, atol=1e-6)
    assert np.isclose(val_tri[1], -1.0, atol=1e-6)

    # Square profile
    val_sq = microgroove_field(x, y, z_samples, wavelength_mm=wavelength, direction=(0, 0, 1), profile="square")
    assert np.all(val_sq >= -1.0 - 1e-6) and np.all(val_sq <= 1.0 + 1e-6)
    assert val_sq[0] > 0.9  # near +1 at peak
    assert val_sq[1] < -0.9  # near -1 at valley


def test_microgroove_field_arbitrary_direction() -> None:
    wavelength = 0.100
    # Modulate along X with exact spacing 0.01 mm
    x = np.linspace(0.0, 0.5, 51)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    val_x = microgroove_field(x, y, z, wavelength_mm=wavelength, direction=(1, 0, 0))
    # Period should be exactly 0.1 mm -> peaks at 0, 0.1, 0.2, etc.
    peak_idx = np.where(np.isclose(x, 0.1, atol=1e-5))[0]
    assert len(peak_idx) > 0
    assert np.isclose(val_x[peak_idx[0]], 1.0, atol=1e-5)


def test_microgroove_field_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="wavelength_mm must be > 0"):
        microgroove_field(0, 0, 0, wavelength_mm=-0.1)

    with pytest.raises(ValueError, match="direction vector cannot be zero-length"):
        microgroove_field(0, 0, 0, wavelength_mm=0.1, direction=(0, 0, 0))

    with pytest.raises(ValueError, match="Unknown profile"):
        microgroove_field(0, 0, 0, wavelength_mm=0.1, profile="nonexistent")


def test_bump_field_profiles_and_periodicity() -> None:
    wavelength = 0.200
    x = np.array([0.0, 0.100, 0.200])
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    val_nodule = bump_field(x, y, z, wavelength_mm=wavelength, profile="nodule")
    assert np.all(val_nodule >= -1.0 - 1e-6) and np.all(val_nodule <= 1.0 + 1e-6)
    assert np.isclose(val_nodule[0], 1.0, atol=1e-6)  # at origin (cos(0)+cos(0)+cos(0))/3 = 1
    assert np.isclose(val_nodule[2], 1.0, atol=1e-6)  # at 1 full wavelength

    val_isolated = bump_field(x, y, z, wavelength_mm=wavelength, profile="isolated")
    assert np.all(val_isolated >= 0.0 - 1e-6) and np.all(val_isolated <= 1.0 + 1e-6)
    assert np.isclose(val_isolated[0], 1.0, atol=1e-6)
    assert np.isclose(val_isolated[1], 0.0, atol=1e-6)  # valley clamped to 0


def test_knurl_field_symmetry() -> None:
    wavelength = 0.100
    grid = np.linspace(-0.2, 0.2, 21)
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    Z = np.zeros_like(X)

    k_field = knurl_field(X, Y, Z, wavelength_mm=wavelength, axis="z")
    assert np.all(k_field >= -1.0 - 1e-6) and np.all(k_field <= 1.0 + 1e-6)
    # Origin should be maximum
    assert np.isclose(k_field[10, 10], 1.0, atol=1e-6)
    # Diagonal symmetry: field(x, y) == field(y, x)
    assert np.allclose(k_field, k_field.T, atol=1e-6)
    # Inversion symmetry: field(x, y) == field(-x, -y)
    assert np.allclose(k_field, k_field[::-1, ::-1], atol=1e-6)


def test_spinodal_spectral_field_reproducibility() -> None:
    wavelength = 0.150
    pts = np.linspace(0.0, 1.0, 30)
    X, Y, Z = np.meshgrid(pts, pts, pts, indexing="ij")

    s1 = spinodal_spectral_field(X, Y, Z, wavelength_mm=wavelength, num_waves=16, seed=42)
    s2 = spinodal_spectral_field(X, Y, Z, wavelength_mm=wavelength, num_waves=16, seed=42)
    s_diff = spinodal_spectral_field(X, Y, Z, wavelength_mm=wavelength, num_waves=16, seed=99)

    assert np.allclose(s1, s2)
    assert not np.allclose(s1, s_diff)
    # Mean of Gaussian random wave superposition should be near 0
    assert abs(float(np.mean(s1))) < 0.15
    # Standard deviation should be close to 1.0
    assert 0.5 < float(np.std(s1)) < 1.5


def test_triplanar_weights() -> None:
    # Test cardinal directions
    normals = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    w = triplanar_weights(normals, sharpness=2.0)
    assert np.allclose(w[0], [1.0, 0.0, 0.0])
    assert np.allclose(w[1], [1.0, 0.0, 0.0])
    assert np.allclose(w[2], [0.0, 1.0, 0.0])
    assert np.allclose(w[3], [0.0, 1.0, 0.0])
    assert np.allclose(w[4], [0.0, 0.0, 1.0])
    assert np.allclose(w[5], [0.0, 0.0, 1.0])

    # Diagonal normal (1, 1, 1) / sqrt(3) -> equal weights
    diag = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    w_diag = triplanar_weights(diag, sharpness=2.0)
    assert np.allclose(w_diag, [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])

    # All weights sum to 1
    random_normals = np.random.randn(50, 3)
    random_normals /= np.linalg.norm(random_normals, axis=1, keepdims=True)
    w_rand = triplanar_weights(random_normals, sharpness=3.0)
    assert np.allclose(np.sum(w_rand, axis=1), 1.0)

    # Degenerate zero normal handled safely without NaN
    zero_normal = np.array([[0.0, 0.0, 0.0]])
    w_zero = triplanar_weights(zero_normal)
    assert not np.any(np.isnan(w_zero))
    assert np.allclose(w_zero, [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])


def test_triplanar_map() -> None:
    # 2D texture: vertical grooves along v
    def simple_2d_groove(u: np.ndarray, v: np.ndarray, wl: float) -> np.ndarray:
        return np.cos(2.0 * np.pi * v / wl)

    # Points on cube faces
    points = np.array([
        [1.0, 0.0, 0.0],  # X face -> projection uses (Y, Z) -> v = Z
        [0.0, 1.0, 0.0],  # Y face -> projection uses (X, Z) -> v = Z
        [0.0, 0.0, 1.0],  # Z face -> projection uses (X, Y) -> v = Y
    ])
    normals = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    wl = 0.100
    mapped = triplanar_map(points, normals, wavelength_mm=wl, texture_fn=simple_2d_groove)
    assert len(mapped) == 3
    assert not np.any(np.isnan(mapped))
    # At (1, 0, 0) with normal (1, 0, 0), v=Z=0 -> cos(0) = 1.0
    assert np.isclose(mapped[0], 1.0)

    # Empty input handling
    empty_pts = np.empty((0, 3))
    empty_norms = np.empty((0, 3))
    empty_res = triplanar_map(empty_pts, empty_norms, wavelength_mm=wl, texture_fn=simple_2d_groove)
    assert len(empty_res) == 0
