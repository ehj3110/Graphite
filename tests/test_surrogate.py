"""
Unit tests for offline parameter sweep and constitutive tensor surrogate modeling.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from graphite.fea.homogenization import (
    EngineeringConstants,
    HomogenizationResult,
    RVEGridConfig,
    build_isotropic_material_matrix,
    homogenize_tpms_cell,
)
from graphite.fea.surrogate import (
    MaterialTensorSurrogate,
    SurrogateCalibrationPoint,
    build_tpms_homogenization_surrogate,
)


def _make_dummy_cubic_point(val: float, E_base: float = 2000.0, nu_base: float = 0.3) -> SurrogateCalibrationPoint:
    """Create a synthetic cubic calibration point scaling with val."""
    # Scale modulus with val^2 (typical cellular bending behavior)
    E_eff = E_base * (val ** 1.8)
    C_iso = build_isotropic_material_matrix(E_eff, nu_base)
    # Introduce small cubic anisotropy: scale shear
    C_iso[3, 3] *= 1.2
    C_iso[4, 4] *= 1.2
    C_iso[5, 5] *= 1.2

    # Invert for engineering constants
    S = np.linalg.inv(C_iso)
    constants = EngineeringConstants(
        E_x=float(1.0 / S[0, 0]),
        E_y=float(1.0 / S[1, 1]),
        E_z=float(1.0 / S[2, 2]),
        G_xy=float(1.0 / S[3, 3]),
        G_yz=float(1.0 / S[4, 4]),
        G_zx=float(1.0 / S[5, 5]),
        nu_xy=float(-S[1, 0] / S[0, 0]),
        nu_yx=float(-S[0, 1] / S[1, 1]),
        nu_xz=float(-S[2, 0] / S[0, 0]),
        nu_zx=float(-S[0, 2] / S[2, 2]),
        nu_yz=float(-S[2, 1] / S[1, 1]),
        nu_zy=float(-S[1, 2] / S[2, 2]),
        bulk_modulus=float(np.sum(C_iso[:3, :3]) / 9.0),
        zener_anisotropy=float(2.0 * C_iso[3, 3] / (C_iso[0, 0] - C_iso[0, 1])),
    )

    return SurrogateCalibrationPoint(
        param_value=val,
        solid_fraction=val,
        C_tensor=C_iso,
        constants=constants,
    )


class TestMaterialTensorSurrogate:
    """Test suite for tensor surrogate interpolation and calibration."""

    @pytest.fixture
    def cubic_surrogate(self) -> MaterialTensorSurrogate:
        vals = [0.10, 0.20, 0.35, 0.50, 0.65]
        pts = [_make_dummy_cubic_point(v) for v in vals]
        return MaterialTensorSurrogate(
            param_name="solid_fraction",
            param_range=(0.10, 0.65),
            sample_points=pts,
            symmetry_type="cubic",
            fitting_method="pchip",
        )

    def test_exact_interpolation_at_knots(self, cubic_surrogate: MaterialTensorSurrogate):
        """PCHIP and Cubic splines must interpolate calibration points with zero residual."""
        for pt in cubic_surrogate.sample_points:
            C_eval = cubic_surrogate.evaluate_material_tensor(pt.param_value)
            # In cubic symmetry, diagonal normal components are averaged, which match pt.C_tensor
            np.testing.assert_allclose(C_eval, pt.C_tensor, rtol=1e-10, atol=1e-10)

    def test_monotonic_positive_definiteness(self, cubic_surrogate: MaterialTensorSurrogate):
        """Eigenvalues must remain strictly positive across 100 query points, with monotonic C11."""
        test_pts = np.linspace(0.10, 0.65, 100)
        C_stack = cubic_surrogate.evaluate_material_tensor(test_pts)

        assert C_stack.shape == (100, 6, 6)

        c11_vals = C_stack[:, 0, 0]
        # Strictly monotonically increasing
        assert np.all(np.diff(c11_vals) > 0.0)

        # Check positive definiteness of every tensor
        for i in range(100):
            eigvals = np.linalg.eigvalsh(C_stack[i])
            assert np.all(eigvals > 0.0), f"Non-positive eigenvalue at index {i}: {eigvals}"

    def test_symmetry_modes(self):
        """Verify cubic, orthotropic, and anisotropic symmetry invariants."""
        vals = [0.2, 0.4, 0.6]
        pts = [_make_dummy_cubic_point(v) for v in vals]

        # 1. Cubic
        surr_cubic = MaterialTensorSurrogate(
            param_name="phi",
            param_range=(0.2, 0.6),
            sample_points=pts,
            symmetry_type="cubic",
        )
        C_c = surr_cubic.evaluate_material_tensor(0.35)
        # C00 == C11 == C22
        assert pytest.approx(C_c[0, 0]) == C_c[1, 1]
        assert pytest.approx(C_c[0, 0]) == C_c[2, 2]
        # C33 == C44 == C55
        assert pytest.approx(C_c[3, 3]) == C_c[4, 4]
        assert pytest.approx(C_c[3, 3]) == C_c[5, 5]
        # Couplings
        assert pytest.approx(C_c[0, 1]) == C_c[0, 2]

        # 2. Orthotropic
        surr_ortho = MaterialTensorSurrogate(
            param_name="phi",
            param_range=(0.2, 0.6),
            sample_points=pts,
            symmetry_type="orthotropic",
        )
        C_o = surr_ortho.evaluate_material_tensor(0.35)
        # Diagonal and off-diagonal symmetry
        np.testing.assert_allclose(C_o, C_o.T, atol=1e-12)
        # Shear-normal couplings must be zero in orthotropic axes
        assert np.allclose(C_o[:3, 3:], 0.0)

        # 3. Anisotropic
        surr_aniso = MaterialTensorSurrogate(
            param_name="phi",
            param_range=(0.2, 0.6),
            sample_points=pts,
            symmetry_type="anisotropic",
        )
        C_a = surr_aniso.evaluate_material_tensor(0.35)
        np.testing.assert_allclose(C_a, C_a.T, atol=1e-12)

    def test_fitting_methods(self):
        """Compare pchip, cubic, and power_law interpolation strategies."""
        vals = [0.15, 0.30, 0.45, 0.60]
        pts = [_make_dummy_cubic_point(v) for v in vals]

        for method in ["pchip", "cubic", "power_law"]:
            surr = MaterialTensorSurrogate(
                param_name="solid_fraction",
                param_range=(0.15, 0.60),
                sample_points=pts,
                fitting_method=method,
            )
            C_mid = surr.evaluate_material_tensor(0.37)
            eigvals = np.linalg.eigvalsh(C_mid)
            assert np.all(eigvals > 0.0)

    def test_vectorized_batch_and_shapes(self, cubic_surrogate: MaterialTensorSurrogate):
        """Verify scalar, 1D, 2D, and 3D query shapes and evaluate vectorized speed."""
        # Scalar
        C_scalar = cubic_surrogate.evaluate_material_tensor(0.3)
        assert C_scalar.shape == (6, 6)

        # 1D array
        arr_1d = np.array([0.2, 0.3, 0.4])
        C_1d = cubic_surrogate.evaluate_material_tensor(arr_1d)
        assert C_1d.shape == (3, 6, 6)

        # 3D volume grid of macro-element centroids
        nx, ny, nz = 10, 8, 6
        arr_3d = np.linspace(0.15, 0.55, nx * ny * nz).reshape((nx, ny, nz))
        C_3d = cubic_surrogate.evaluate_material_tensor(arr_3d)
        assert C_3d.shape == (nx, ny, nz, 6, 6)

        # Speed test: 100,000 macro element queries
        big_query = np.linspace(0.1, 0.65, 100_000)
        t0 = time.perf_counter()
        C_large = cubic_surrogate.evaluate_material_tensor(big_query)
        dt = time.perf_counter() - t0
        assert C_large.shape == (100_000, 6, 6)
        # Vectorized evaluation of 100k elements must take < 100ms
        assert dt < 0.10, f"Vectorized evaluation took {dt:.4f}s, expected < 0.10s"

    def test_engineering_constants_evaluation(self, cubic_surrogate: MaterialTensorSurrogate):
        """Verify extraction of directional moduli, Poisson ratios, and Zener anisotropy."""
        # Scalar query
        res_scalar = cubic_surrogate.evaluate_engineering_constants(0.35)
        assert isinstance(res_scalar["E_x"], float)
        assert res_scalar["E_x"] > 0.0
        assert res_scalar["G_xy"] > 0.0
        assert res_scalar["bulk_modulus"] > 0.0
        assert res_scalar["zener_anisotropy"] > 0.0

        # Vector query
        arr_1d = np.array([0.2, 0.35, 0.5])
        res_vec = cubic_surrogate.evaluate_engineering_constants(arr_1d)
        assert isinstance(res_vec["E_x"], np.ndarray)
        assert res_vec["E_x"].shape == (3,)
        assert np.all(res_vec["E_x"] > 0.0)
        assert np.all(res_vec["bulk_modulus"] > 0.0)

    def test_json_save_load_roundtrip(self, cubic_surrogate: MaterialTensorSurrogate, tmp_path: Path):
        """Surrogate model must serialize to JSON and deserialize with zero loss of accuracy."""
        save_file = tmp_path / "test_surrogate.json"
        cubic_surrogate.save(save_file)
        assert save_file.is_file()

        loaded_surrogate = MaterialTensorSurrogate.load(save_file)
        assert loaded_surrogate.param_name == cubic_surrogate.param_name
        assert loaded_surrogate.param_range == cubic_surrogate.param_range
        assert loaded_surrogate.symmetry_type == cubic_surrogate.symmetry_type
        assert loaded_surrogate.fitting_method == cubic_surrogate.fitting_method
        assert len(loaded_surrogate.sample_points) == len(cubic_surrogate.sample_points)

        # Check numerical agreement across queries
        query = np.linspace(0.12, 0.60, 20)
        C_orig = cubic_surrogate.evaluate_material_tensor(query)
        C_load = loaded_surrogate.evaluate_material_tensor(query)
        np.testing.assert_allclose(C_orig, C_load, atol=1e-12)

    def test_tpms_parameter_sweep_driver(self):
        """End-to-end integration: run automated sweep on Gyroid RVE and construct surrogate."""
        # Use small grid resolution=12 for fast test execution (<2s)
        cfg = RVEGridConfig(resolution=12, base_E=2000.0, base_nu=0.35, solver_backend="cg")
        surrogate = build_tpms_homogenization_surrogate(
            lattice_type="Gyroid",
            solid_fractions=(0.20, 0.40, 0.60),
            is_sheet=True,
            rve_config=cfg,
            fitting_method="pchip",
        )

        assert surrogate.param_name == "solid_fraction"
        assert surrogate.param_range == (0.20, 0.60)
        assert len(surrogate.sample_points) == 3
        assert surrogate.symmetry_type == "cubic"

        # Interpolate at an intermediate solid fraction
        C_interp = surrogate.evaluate_material_tensor(0.30)
        assert C_interp.shape == (6, 6)
        eigvals = np.linalg.eigvalsh(C_interp)
        assert np.all(eigvals > 0.0)

        consts = surrogate.evaluate_engineering_constants(0.30)
        assert consts["E_x"] > 0.0
        assert consts["bulk_modulus"] > 0.0
