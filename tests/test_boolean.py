"""
Unit tests for smooth Boolean operators (R-functions) and implicit skin blending.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.implicit.blending import blend_lattice_with_skin
from graphite.implicit.spinodal import generate_spinodal_lattice
from graphite.math.boolean import smooth_difference, smooth_max, smooth_min
from graphite.mesh.extraction import extract_isosurface_flying_edges


class TestSmoothBooleanOperators:
    """Test mathematical accuracy, continuity, and convergence of R-functions."""

    def test_asymptotic_convergence_to_hard_boolean(self):
        """As r -> 0, smooth operators must converge to exact min/max."""
        rng = np.random.default_rng(1234)
        a = rng.uniform(-10.0, 10.0, size=(20, 20))
        b = rng.uniform(-10.0, 10.0, size=(20, 20))

        # Test at r=0.0 exact threshold guard
        np.testing.assert_allclose(smooth_min(a, b, r=0.0), np.minimum(a, b), atol=1e-12)
        np.testing.assert_allclose(smooth_max(a, b, r=0.0), np.maximum(a, b), atol=1e-12)
        np.testing.assert_allclose(smooth_difference(a, b, r=0.0), np.maximum(a, -b), atol=1e-12)

        # Test convergence for each method as r becomes small
        for method in ("polynomial", "circular", "exponential"):
            smin_val = smooth_min(a, b, r=1e-4, method=method)
            smax_val = smooth_max(a, b, r=1e-4, method=method)
            np.testing.assert_allclose(smin_val, np.minimum(a, b), atol=1e-3)
            np.testing.assert_allclose(smax_val, np.maximum(a, b), atol=1e-3)

    def test_diagonal_fillet_offsets(self):
        """At a = b = 0, fillet offset must match analytical formula for each method."""
        r = 2.0
        # Polynomial: smax(0, 0, r) = +r/4, smin(0, 0, r) = -r/4
        assert pytest.approx(smooth_max(0.0, 0.0, r=r, method="polynomial")) == r / 4.0
        assert pytest.approx(smooth_min(0.0, 0.0, r=r, method="polynomial")) == -r / 4.0

        # Exponential: smax(0, 0, r) = +r * ln(2), smin(0, 0, r) = -r * ln(2)
        assert pytest.approx(smooth_max(0.0, 0.0, r=r, method="exponential")) == r * np.log(2.0)
        assert pytest.approx(smooth_min(0.0, 0.0, r=r, method="exponential")) == -r * np.log(2.0)

        # Circular: smax(0, 0, r) = +0.5 * r, smin(0, 0, r) = -0.5 * r
        assert pytest.approx(smooth_max(0.0, 0.0, r=r, method="circular")) == 0.5 * r
        assert pytest.approx(smooth_min(0.0, 0.0, r=r, method="circular")) == -0.5 * r

    def test_demorgan_duality(self):
        """Verify De Morgan's duality: smax(a, b, r) == -smin(-a, -b, r)."""
        rng = np.random.default_rng(42)
        a = rng.uniform(-5.0, 5.0, size=(10, 10))
        b = rng.uniform(-5.0, 5.0, size=(10, 10))
        r = 1.5

        for method in ("polynomial", "circular", "exponential"):
            smax_direct = smooth_max(a, b, r=r, method=method)
            smax_dual = -smooth_min(-a, -b, r=r, method=method)
            np.testing.assert_allclose(smax_direct, smax_dual, atol=1e-12)

    def test_smooth_difference(self):
        """Smooth difference A \\ B must equal smax(a, -b, r)."""
        a = np.array([0.0, -1.0, 2.0])
        b = np.array([0.0, 1.0, -2.0])
        r = 0.8
        for method in ("polynomial", "circular", "exponential"):
            diff = smooth_difference(a, b, r=r, method=method)
            expected = smooth_max(a, -b, r=r, method=method)
            np.testing.assert_allclose(diff, expected, atol=1e-12)

    def test_numerical_stability_extremes(self):
        """Extreme large and small values must not overflow or return NaN/Inf."""
        a = np.array([-1000.0, 1000.0, -500.0, 500.0])
        b = np.array([1000.0, -1000.0, 500.0, -500.0])
        r = 1.0

        for method in ("polynomial", "circular", "exponential"):
            smin_val = smooth_min(a, b, r=r, method=method)
            smax_val = smooth_max(a, b, r=r, method=method)
            assert np.all(np.isfinite(smin_val))
            assert np.all(np.isfinite(smax_val))

    def test_input_validation(self):
        """Check exceptions on invalid blend_radius or unknown method."""
        with pytest.raises(ValueError, match="blend_radius must be non-negative"):
            smooth_min(0.0, 0.0, r=-0.5)

        with pytest.raises(ValueError, match="blend_radius must be non-negative"):
            smooth_max(0.0, 0.0, r=-1.0)

        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth_min(0.0, 0.0, r=1.0, method="spline")

        with pytest.raises(ValueError, match="Unknown smoothing method"):
            smooth_max(0.0, 0.0, r=1.0, method="nonexistent")


class TestLatticeSkinBlending:
    """Test high-level blend_lattice_with_skin pipeline."""

    def test_blend_lattice_with_skin_validation(self):
        """Test input checks on array shape, thickness, and radius."""
        f1 = np.zeros((10, 10, 10))
        f2 = np.zeros((10, 10, 12))
        with pytest.raises(ValueError, match="Shape mismatch"):
            blend_lattice_with_skin(f1, f2, skin_thickness=1.0)

        with pytest.raises(ValueError, match="skin_thickness must be non-negative"):
            blend_lattice_with_skin(f1, f1, skin_thickness=-0.5)

        with pytest.raises(ValueError, match="blend_radius must be non-negative"):
            blend_lattice_with_skin(f1, f1, skin_thickness=1.0, blend_radius=-1.0)

    def test_blend_lattice_with_skin_execution(self):
        """Verify blending generates a valid, smoothed scalar field."""
        # Create a 3D box domain with SDF
        grid = np.linspace(-5, 5, 31)
        X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
        # Sphere SDF: center at origin, radius 4.0
        cad_sdf = np.sqrt(X**2 + Y**2 + Z**2) - 4.0
        # Simple slab lattice: solid where |X| <= 1.0 (X^2 - 1 <= 0)
        lattice_field = np.abs(X) - 1.0

        # Hard boolean
        field_hard = blend_lattice_with_skin(
            lattice_field, cad_sdf, skin_thickness=0.5, blend_radius=0.0
        )
        # Smooth boolean
        field_smooth = blend_lattice_with_skin(
            lattice_field, cad_sdf, skin_thickness=0.5, blend_radius=0.8, method="polynomial"
        )

        assert field_hard.shape == cad_sdf.shape
        assert field_smooth.shape == cad_sdf.shape

        # Extract mesh of the smooth blended geometry
        mesh = extract_isosurface_flying_edges(
            field_smooth,
            origin=(-5.0, -5.0, -5.0),
            spacing=(10.0 / 30.0, 10.0 / 30.0, 10.0 / 30.0),
            level=0.0,
        )
        assert not mesh.is_empty
        assert len(mesh.faces) > 0
        assert mesh.is_watertight


class TestConformalSpinodalBlending:
    """Test smooth Boolean integration in conformal spinodal lattice generation."""

    def test_spinodal_with_blend_radius(self):
        """Verify generate_spinodal_lattice executes with blend_radius > 0 and produces valid mesh."""
        box = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
        mesh = generate_spinodal_lattice(
            cad_mesh=box,
            resolution=0.5,
            wavelength=3.0,
            solid_fraction=0.35,
            num_waves=40,
            blend_radius=0.4,
            blend_method="polynomial",
            taubin_iterations=5,
            seed=42,
        )
        assert not mesh.is_empty
        assert len(mesh.faces) > 0
        assert mesh.is_watertight
