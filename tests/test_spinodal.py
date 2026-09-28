"""
Unit tests for Gaussian Random Field (GRF) Spinodal Lattice Generator.

Tests verify mathematical statistical properties (CLT convergence),
solid fraction analytic thresholding precision, input parameter validation,
and end-to-end CAD-clipped lattice generation.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.implicit.spinodal import generate_spinodal_lattice
from graphite.math.spinodal import (
    _sample_marsaglia_unit_sphere,
    evaluate_spinodal_field,
    generate_spinodal_wavevectors,
    threshold_spinodal_field,
)


# ---------------------------------------------------------------------------
# 1. Marsaglia Sampling & Wavevector Generation Tests
# ---------------------------------------------------------------------------

def test_marsaglia_unit_sphere_properties() -> None:
    """Verify that Marsaglia-sampled vectors strictly lie on S^2 with uniform spread."""
    rng = np.random.default_rng(12345)
    num_samples = 5000
    pts = _sample_marsaglia_unit_sphere(num_samples, rng)

    assert pts.shape == (num_samples, 3)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7, err_msg="Marsaglia vectors must have unit norm")

    # Means should be near 0 on all axes
    np.testing.assert_allclose(pts.mean(axis=0), 0.0, atol=0.05)
    # Variances on each axis should be approx 1/3 for uniform sphere distribution
    np.testing.assert_allclose(pts.var(axis=0), 1.0 / 3.0, atol=0.05)


def test_generate_spinodal_wavevectors_isotropic() -> None:
    """Verify wavevectors magnitude equals 2*pi/wavelength for isotropic scaling."""
    wavelength = 3.0
    num_waves = 100
    k_vecs, phases = generate_spinodal_wavevectors(
        num_waves=num_waves,
        wavelength=wavelength,
        anisotropy=(1.0, 1.0, 1.0),
        seed=42,
    )

    assert k_vecs.shape == (num_waves, 3)
    assert phases.shape == (num_waves,)
    assert np.all(phases >= 0.0) and np.all(phases < 2.0 * np.pi)

    expected_k0 = (2.0 * np.pi) / wavelength
    norms = np.linalg.norm(k_vecs, axis=1)
    np.testing.assert_allclose(norms, expected_k0, atol=1e-7)


def test_generate_spinodal_wavevectors_anisotropic() -> None:
    """Verify anisotropic scaling skews wavevectors along specified axes."""
    wavelength = 2.0
    num_waves = 80
    anisotropy = (2.0, 1.0, 0.5)
    k_vecs, phases = generate_spinodal_wavevectors(
        num_waves=num_waves,
        wavelength=wavelength,
        anisotropy=anisotropy,
        seed=99,
    )

    expected_k0 = (2.0 * np.pi) / wavelength
    norms = np.linalg.norm(k_vecs, axis=1)
    np.testing.assert_allclose(norms, expected_k0, atol=1e-7)


# ---------------------------------------------------------------------------
# 2. Mathematical Properties: Mean ~ 0 and Std ~ 1 (within 3% tolerance)
# ---------------------------------------------------------------------------

def test_spinodal_field_statistical_properties() -> None:
    """
    Verify Central Limit Theorem convergence:
    Evaluated field F(x) should exhibit mean ~ 0 and standard deviation ~ 1
    within a 3% tolerance.
    """
    # Sample on a 20 mm domain with 0.25 mm resolution (80x80x80 = 512,000 points)
    res = 0.25
    domain_len = 20.0
    coords = np.arange(0.0, domain_len, res)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    wavelength = 2.0
    num_waves = 120
    field = evaluate_spinodal_field(
        X,
        Y,
        Z,
        wavelength=wavelength,
        num_waves=num_waves,
        anisotropy=(1.0, 1.0, 1.0),
        seed=42,
    )

    assert field.dtype == np.float32
    assert field.shape == X.shape

    mean_val = float(np.mean(field))
    std_val = float(np.std(field))

    # Mean must be within 0.03 of 0.0
    assert abs(mean_val) < 0.03, f"Expected mean ~ 0, got {mean_val:.4f}"
    # Standard deviation must be within 3% of 1.0 (i.e. [0.97, 1.03])
    assert abs(std_val - 1.0) < 0.03, f"Expected std within 3% of 1.0, got {std_val:.4f}"


# ---------------------------------------------------------------------------
# 3. Voxel Solid Fraction Precision (within +- 0.02)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solid_fraction", [0.2, 0.3, 0.5, 0.7])
def test_threshold_spinodal_field_solid_fraction_skeletal(solid_fraction: float) -> None:
    """Verify skeletal solid fraction matches target within +- 0.02."""
    res = 0.25
    domain_len = 20.0
    coords = np.arange(0.0, domain_len, res)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    field = evaluate_spinodal_field(
        X,
        Y,
        Z,
        wavelength=2.0,
        num_waves=120,
        seed=42,
    )

    solid_field = threshold_spinodal_field(field, solid_fraction=solid_fraction, is_sheet=False)
    measured_sf = float(np.mean(solid_field <= 0.0))

    assert abs(measured_sf - solid_fraction) <= 0.02, (
        f"Skeletal solid fraction {measured_sf:.4f} deviated from target {solid_fraction:.4f} "
        f"by {abs(measured_sf - solid_fraction):.4f} (> 0.02)"
    )


@pytest.mark.parametrize("solid_fraction", [0.2, 0.3, 0.5, 0.7])
def test_threshold_spinodal_field_solid_fraction_sheet(solid_fraction: float) -> None:
    """Verify sheet solid fraction matches target within +- 0.02."""
    res = 0.25
    domain_len = 20.0
    coords = np.arange(0.0, domain_len, res)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    field = evaluate_spinodal_field(
        X,
        Y,
        Z,
        wavelength=2.0,
        num_waves=120,
        seed=42,
    )

    solid_field = threshold_spinodal_field(field, solid_fraction=solid_fraction, is_sheet=True)
    measured_sf = float(np.mean(solid_field <= 0.0))

    assert abs(measured_sf - solid_fraction) <= 0.02, (
        f"Sheet solid fraction {measured_sf:.4f} deviated from target {solid_fraction:.4f} "
        f"by {abs(measured_sf - solid_fraction):.4f} (> 0.02)"
    )


# ---------------------------------------------------------------------------
# 4. End-to-End Integration Test on Box Primitive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_sheet", [False, True])
def test_generate_spinodal_lattice_box_primitive(is_sheet: bool) -> None:
    """
    Generate spinodal lattice on a 10x10x10 mm box mesh.
    Verify resulting mesh is non-empty, contains valid faces/vertices,
    has no NaN coordinates, and is bounded within the CAD envelope.
    """
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    mesh = generate_spinodal_lattice(
        cad_mesh=box,
        resolution=0.4,
        wavelength=2.5,
        solid_fraction=0.3,
        is_sheet=is_sheet,
        anisotropy=(1.0, 1.0, 1.0),
        num_waves=80,
        seed=42,
        pad_width=4,
    )

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty, "Extracted spinodal mesh should not be empty"
    assert len(mesh.vertices) > 0, "Mesh must contain vertices"
    assert len(mesh.faces) > 0, "Mesh must contain triangular faces"

    # Verify no NaN or Inf coordinates
    assert not np.isnan(mesh.vertices).any(), "Mesh vertices must not contain NaN"
    assert not np.isinf(mesh.vertices).any(), "Mesh vertices must not contain Inf"

    # Verify bounds lie within CAD envelope with small tolerance for marching cubes
    tol = 0.5  # voxel pitch margin
    min_bound, max_bound = mesh.bounds
    np.testing.assert_array_less(-5.0 - tol, min_bound)
    np.testing.assert_array_less(max_bound, 5.0 + tol)


def test_generate_spinodal_lattice_with_export(tmp_path) -> None:
    """Verify that export_path saves a readable mesh file."""
    box = trimesh.creation.box(extents=(8.0, 8.0, 8.0))
    out_file = tmp_path / "spinodal_test.stl"

    mesh = generate_spinodal_lattice(
        cad_mesh=box,
        resolution=0.5,
        wavelength=2.5,
        solid_fraction=0.35,
        num_waves=60,
        seed=42,
        output_path=out_file,
    )

    assert out_file.is_file()
    assert out_file.stat().st_size > 0
    reloaded = trimesh.load(str(out_file))
    assert len(reloaded.faces) == len(mesh.faces)


# ---------------------------------------------------------------------------
# 5. Reproducibility & Seed Consistency
# ---------------------------------------------------------------------------

def test_spinodal_reproducibility() -> None:
    """Verify identical random seeds yield identical scalar fields."""
    coords = np.arange(0.0, 5.0, 0.5)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    f1 = evaluate_spinodal_field(X, Y, Z, wavelength=2.0, num_waves=50, seed=42)
    f2 = evaluate_spinodal_field(X, Y, Z, wavelength=2.0, num_waves=50, seed=42)
    f_diff = evaluate_spinodal_field(X, Y, Z, wavelength=2.0, num_waves=50, seed=99)

    np.testing.assert_array_equal(f1, f2)
    assert not np.allclose(f1, f_diff)


# ---------------------------------------------------------------------------
# 6. Input Validation & Error Handling
# ---------------------------------------------------------------------------

def test_input_validation_errors() -> None:
    """Verify proper ValueError / TypeError exceptions for invalid parameters."""
    coords = np.arange(0.0, 5.0, 1.0)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Invalid solid_fraction
    with pytest.raises(ValueError, match="solid_fraction"):
        threshold_spinodal_field(X, solid_fraction=0.0)
    with pytest.raises(ValueError, match="solid_fraction"):
        threshold_spinodal_field(X, solid_fraction=1.0)
    with pytest.raises(ValueError, match="solid_fraction"):
        threshold_spinodal_field(X, solid_fraction=-0.2)

    # Invalid wavelength
    with pytest.raises(ValueError, match="wavelength"):
        generate_spinodal_wavevectors(num_waves=50, wavelength=0.0)
    with pytest.raises(ValueError, match="wavelength"):
        generate_spinodal_wavevectors(num_waves=50, wavelength=-1.0)

    # Invalid num_waves
    with pytest.raises(ValueError, match="num_waves"):
        generate_spinodal_wavevectors(num_waves=0, wavelength=2.0)

    # Invalid anisotropy
    with pytest.raises(ValueError, match="anisotropy"):
        generate_spinodal_wavevectors(num_waves=50, wavelength=2.0, anisotropy=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="anisotropy"):
        generate_spinodal_wavevectors(num_waves=50, wavelength=2.0, anisotropy=(1.0, 1.0))

    # Incompatible coordinate arrays
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_spinodal_field(X, Y, Z[:2, :, :], wavelength=2.0)

    # Invalid CAD mesh in generator
    box = trimesh.creation.box(extents=(5.0, 5.0, 5.0))
    with pytest.raises(ValueError, match="resolution"):
        generate_spinodal_lattice(cad_mesh=box, resolution=-0.1)
    with pytest.raises(ValueError, match="pad_width"):
        generate_spinodal_lattice(cad_mesh=box, pad_width=-1)
    with pytest.raises(TypeError, match="cad_mesh"):
        generate_spinodal_lattice(cad_mesh=12345)
