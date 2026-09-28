"""
Unit tests for Flying Edges Isosurface Extraction & Volume-Preserving Taubin Smoothing.

Tests verify:
1. Geometric fidelity (volume and surface area within 2% of analytic sphere formulas).
2. Volume conservation under Taubin curvature smoothing (|ΔV/V0| < 0.5%) and curvature relaxation.
3. No-regression verification ensuring PyVista/VTK flying_edges is invoked without fallback.
4. Robust error handling and fallback mechanisms.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from graphite.implicit.spinodal import generate_spinodal_lattice
from graphite.mesh.extraction import extract_isosurface_flying_edges
from graphite.mesh.smoothing import compute_mean_curvature, smooth_mesh_taubin


# ---------------------------------------------------------------------------
# 1. Extraction Validation: Known Analytic Sphere Field (R = 5 mm)
# ---------------------------------------------------------------------------

def test_flying_edges_sphere_extraction_accuracy() -> None:
    """
    Generate an implicit sphere SDF with R=5 mm, extract with Flying Edges,
    and assert surface area and volume match analytic formulas within 2%.
    """
    R = 5.0
    res = 0.2
    pad = 2.0
    bounds = [-R - pad, R + pad]
    x_axis = np.arange(bounds[0], bounds[1] + res, res)
    X, Y, Z = np.meshgrid(x_axis, x_axis, x_axis, indexing="ij")

    # SDF: negative inside, positive outside
    sdf = np.sqrt(X**2 + Y**2 + Z**2) - R
    origin = (float(x_axis[0]), float(x_axis[0]), float(x_axis[0]))
    spacing = (res, res, res)

    mesh = extract_isosurface_flying_edges(
        field=sdf,
        origin=origin,
        spacing=spacing,
        level=0.0,
    )

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    assert mesh.is_watertight

    analytic_volume = (4.0 / 3.0) * np.pi * (R**3)
    analytic_area = 4.0 * np.pi * (R**2)

    extracted_volume = abs(float(mesh.volume))
    extracted_area = float(mesh.area)

    vol_error = abs(extracted_volume - analytic_volume) / analytic_volume
    area_error = abs(extracted_area - analytic_area) / analytic_area

    # Assert volume and surface area match within 2%
    assert vol_error < 0.02, (
        f"Volume error {vol_error * 100:.3f}% exceeds 2% tolerance "
        f"(extracted={extracted_volume:.3f}, analytic={analytic_volume:.3f})"
    )
    assert area_error < 0.02, (
        f"Surface area error {area_error * 100:.3f}% exceeds 2% tolerance "
        f"(extracted={extracted_area:.3f}, analytic={analytic_area:.3f})"
    )

    # Centroid should be very close to origin (0, 0, 0)
    np.testing.assert_allclose(mesh.centroid, [0.0, 0.0, 0.0], atol=0.05)


# ---------------------------------------------------------------------------
# 2. Volume Conservation & Curvature Relaxation Test
# ---------------------------------------------------------------------------

def test_taubin_smoothing_volume_conservation_and_curvature() -> None:
    """
    Run Taubin smoothing on a generated spinodal mesh for 20 iterations.
    Assert:
    - Volume change |ΔV / V0| < 0.005 (less than 0.5% drift).
    - Maximum mean curvature variation σ(H) decreases compared to raw mesh.
    """
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    # Generate raw un-smoothed spinodal mesh
    raw_mesh = generate_spinodal_lattice(
        cad_mesh=box,
        resolution=0.35,
        wavelength=2.5,
        solid_fraction=0.35,
        seed=42,
        taubin_iterations=0,  # disable smoothing to get raw mesh
    )

    assert not raw_mesh.is_empty
    v0 = abs(float(raw_mesh.volume))
    assert v0 > 0.0

    # Smooth for 20 iterations
    smoothed_mesh = smooth_mesh_taubin(
        raw_mesh,
        iterations=20,
        lamb=0.5,
        nu=-0.53,
        inplace=False,
    )

    v1 = abs(float(smoothed_mesh.volume))
    drift = abs(v1 - v0) / v0

    # 1. Volume conservation: drift must be < 0.5%
    assert drift < 0.005, (
        f"Volume drift {drift * 100:.4f}% exceeded 0.5% threshold "
        f"(V0={v0:.3f}, V1={v1:.3f})"
    )

    # 2. Curvature relaxation: σ(H) must decrease
    h_raw = compute_mean_curvature(raw_mesh)
    h_smoothed = compute_mean_curvature(smoothed_mesh)

    sigma_h_raw = float(np.std(h_raw))
    sigma_h_smoothed = float(np.std(h_smoothed))

    assert sigma_h_smoothed < sigma_h_raw, (
        f"Expected curvature variation reduction, got σ(H_raw)={sigma_h_raw:.4f} "
        f"<= σ(H_smooth)={sigma_h_smoothed:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. No Regressions: Flying Edges is Primary Backend (No Marching Cubes Calls)
# ---------------------------------------------------------------------------

def test_no_marching_cubes_called_when_flying_edges_available() -> None:
    """
    Ensure no calls to skimage.measure.marching_cubes are made when
    Flying Edges / PyVista dependencies are satisfied.
    """
    nx, ny, nz = 15, 15, 15
    field = np.ones((nx, ny, nz), dtype=np.float32)
    field[4:11, 4:11, 4:11] = -1.0

    with patch("skimage.measure.marching_cubes") as mock_mc:
        mesh = extract_isosurface_flying_edges(
            field=field,
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            level=0.0,
        )
        mock_mc.assert_not_called()

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty


# ---------------------------------------------------------------------------
# 4. Fallback Mechanism when Flying Edges Backend is Unavailable
# ---------------------------------------------------------------------------

def test_fallback_to_marching_cubes_on_pyvista_failure() -> None:
    """Verify clean fallback to marching_cubes when PyVista fails."""
    nx, ny, nz = 15, 15, 15
    field = np.ones((nx, ny, nz), dtype=np.float32)
    field[4:11, 4:11, 4:11] = -1.0

    # Simulate pyvista failure
    with patch.dict("sys.modules", {"pyvista": None}):
        with pytest.warns(RuntimeWarning, match="Flying Edges extraction backend unavailable"):
            mesh = extract_isosurface_flying_edges(
                field=field,
                origin=(10.0, 20.0, 30.0),
                spacing=(0.5, 0.5, 0.5),
                level=0.0,
            )

    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty
    # Bounds should reflect origin offset (10, 20, 30)
    assert mesh.bounds[0, 0] >= 10.0 - 0.5


# ---------------------------------------------------------------------------
# 5. Parameter Validation & Edge Cases
# ---------------------------------------------------------------------------

def test_extraction_parameter_validation() -> None:
    """Verify input validation in extract_isosurface_flying_edges."""
    field_2d = np.zeros((10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="3D array"):
        extract_isosurface_flying_edges(field_2d, origin=(0, 0, 0), spacing=(1, 1, 1))

    field_3d = np.zeros((10, 10, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="spacing components must be positive"):
        extract_isosurface_flying_edges(field_3d, origin=(0, 0, 0), spacing=(1, 0, 1))
    with pytest.raises(ValueError, match="spacing components must be positive"):
        extract_isosurface_flying_edges(field_3d, origin=(0, 0, 0), spacing=(-1, 1, 1))


def test_taubin_smoothing_parameter_validation() -> None:
    """Verify input validation in smooth_mesh_taubin."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=2.0)

    with pytest.raises(ValueError, match="iterations"):
        smooth_mesh_taubin(sphere, iterations=0)
    with pytest.raises(ValueError, match="lamb"):
        smooth_mesh_taubin(sphere, lamb=0.0)
    with pytest.raises(ValueError, match="lamb"):
        smooth_mesh_taubin(sphere, lamb=1.0)


def test_taubin_smoothing_inplace_behavior() -> None:
    """Verify inplace=True modifies the original mesh and inplace=False returns a copy."""
    sphere1 = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    orig_v0 = sphere1.vertices.copy()

    # inplace=False
    out_copy = smooth_mesh_taubin(sphere1, iterations=4, inplace=False)
    assert not np.array_equal(out_copy.vertices, orig_v0)
    np.testing.assert_array_equal(sphere1.vertices, orig_v0)

    # inplace=True
    out_inplace = smooth_mesh_taubin(sphere1, iterations=4, inplace=True)
    assert out_inplace is sphere1
    assert not np.array_equal(sphere1.vertices, orig_v0)
