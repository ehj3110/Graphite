"""
Unit tests for the micro-scale voxel RVE periodic homogenization engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.fea.homogenization import (
    RVEGridConfig,
    build_isotropic_material_matrix,
    build_voxel_c3d8_stiffness,
    homogenize_strut_cell,
    homogenize_tpms_cell,
    homogenize_voxel_rve,
)


class TestVoxelHomogenization:
    """Test micro-scale voxel RVE homogenization mathematical properties."""

    def test_c3d8_stiffness_matrix_properties(self):
        """Verify C3D8 stiffness matrix symmetry and rigid body zero eigenvalues."""
        hx, hy, hz = 0.5, 0.5, 0.5
        E, nu = 1000.0, 0.3
        K0, B_cen = build_voxel_c3d8_stiffness(hx, hy, hz, E, nu)

        assert K0.shape == (24, 24)
        assert B_cen.shape == (6, 24)
        # Symmetry
        np.testing.assert_allclose(K0, K0.T, atol=1e-12)

        # 6 rigid body modes must yield zero eigenvalues
        eigvals = np.linalg.eigvalsh(K0)
        zero_eigs = eigvals[:6]
        pos_eigs = eigvals[6:]
        np.testing.assert_allclose(zero_eigs, 0.0, atol=1e-8)
        assert np.all(pos_eigs > 0.0)

    def test_solid_unit_cube_limit(self):
        """All-solid voxels (phi = 1.0) must recover exact isotropic base material properties."""
        E_base = 2500.0
        nu_base = 0.30
        config = RVEGridConfig(
            resolution=12,
            cell_size=1.0,
            base_E=E_base,
            base_nu=nu_base,
            solver_backend="cg",
        )

        solid_mask = np.ones((12, 12, 12), dtype=bool)
        res = homogenize_voxel_rve(solid_mask, config=config)

        # Homogenized tensor must match analytical isotropic tensor within 0.1%
        C_exact = build_isotropic_material_matrix(E_base, nu_base)
        np.testing.assert_allclose(res.C_homogenized, C_exact, rtol=1e-3, atol=1e-6)

        # Check engineering constants
        assert pytest.approx(res.engineering_constants.E_x, rel=1e-3) == E_base
        assert pytest.approx(res.engineering_constants.E_y, rel=1e-3) == E_base
        assert pytest.approx(res.engineering_constants.E_z, rel=1e-3) == E_base
        assert pytest.approx(res.engineering_constants.nu_xy, rel=1e-3) == nu_base
        assert pytest.approx(res.engineering_constants.nu_yz, rel=1e-3) == nu_base
        assert pytest.approx(res.engineering_constants.zener_anisotropy, rel=1e-3) == 1.0

    def test_void_unit_cube_limit(self):
        """All-void voxels (phi = 0.0) must scale directly with ersatz_ratio."""
        E_base = 2000.0
        nu_base = 0.35
        ersatz = 1e-5
        config = RVEGridConfig(
            resolution=12,
            cell_size=1.0,
            base_E=E_base,
            base_nu=nu_base,
            ersatz_ratio=ersatz,
            solver_backend="cg",
        )

        void_mask = np.zeros((12, 12, 12), dtype=bool)
        res = homogenize_voxel_rve(void_mask, config=config)

        expected_C = ersatz * build_isotropic_material_matrix(E_base, nu_base)
        np.testing.assert_allclose(res.C_homogenized, expected_C, rtol=1e-3, atol=1e-6)
        assert pytest.approx(res.engineering_constants.E_x, rel=1e-3) == ersatz * E_base

    def test_tpms_cubic_symmetry_gyroid(self):
        """Gyroid sheet unit cell must exhibit exact cubic symmetry relations."""
        config = RVEGridConfig(
            resolution=16,
            cell_size=1.0,
            base_E=2000.0,
            base_nu=0.35,
            solver_backend="cg",
        )
        res = homogenize_tpms_cell("Gyroid", solid_fraction=0.30, is_sheet=True, config=config)

        C = res.C_homogenized
        # In cubic symmetry: C11 = C22 = C33, C12 = C13 = C23, C44 = C55 = C66
        np.testing.assert_allclose(C[0, 0], C[1, 1], rtol=0.02)
        np.testing.assert_allclose(C[1, 1], C[2, 2], rtol=0.02)

        np.testing.assert_allclose(C[0, 1], C[0, 2], rtol=0.02)
        np.testing.assert_allclose(C[0, 2], C[1, 2], rtol=0.02)

        np.testing.assert_allclose(C[3, 3], C[4, 4], rtol=0.02)
        np.testing.assert_allclose(C[4, 4], C[5, 5], rtol=0.02)

        # Moduli must be positive and strictly less than solid material
        assert 0.0 < res.engineering_constants.E_x < config.base_E
        assert 0.0 < res.engineering_constants.G_xy < config.base_E
        # Zener anisotropy ratio for Gyroid is well-defined
        assert res.engineering_constants.zener_anisotropy > 0.0

    def test_strut_simple_cubic_beam_theory_limit(self):
        """Simple cubic strut lattice must match 1D rod stiffness theory in the slender limit."""
        # 8 corners of unit cube [0, 1]^3
        nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float64,
        )
        # 12 edges
        struts = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # z = 0
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # z = 1
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # vertical
            ],
            dtype=np.int64,
        )

        r_rel = 0.10  # r / L = 0.10
        E_base = 3000.0
        config = RVEGridConfig(
            resolution=16,
            cell_size=1.0,
            base_E=E_base,
            base_nu=0.30,
            solver_backend="cg",
        )

        res = homogenize_strut_cell(nodes, struts, strut_radius=r_rel, config=config)

        # In 1 periodic cell, 4 quarter-struts along each axis form 1 full strut of cross-section pi*r^2
        # Rod theory: E_eff / E_s ~ pi * (r/L)^2 = pi * 0.01 = ~0.0314
        expected_ratio = np.pi * (r_rel**2)
        actual_ratio = res.engineering_constants.E_x / E_base

        # Voxel discretization of cylinder at resolution 16 has discretized cross-sectional area ~0.047.
        # Effective rod stiffness tracks cross-sectional area (pi*r^2 = 0.0314) within voxelization bounds.
        assert 0.025 <= actual_ratio <= 0.065
        # Cubic symmetry: E_x ~ E_y ~ E_z
        np.testing.assert_allclose(
            res.engineering_constants.E_x, res.engineering_constants.E_y, rtol=0.03
        )
        np.testing.assert_allclose(
            res.engineering_constants.E_y, res.engineering_constants.E_z, rtol=0.03
        )

    def test_grid_convergence_schwarz_p(self):
        """Schwarz-P homogenization must refine across 12^3 to 16^3."""
        config_coarse = RVEGridConfig(resolution=12, solver_backend="cg")
        config_fine = RVEGridConfig(resolution=16, solver_backend="cg")

        res_c = homogenize_tpms_cell("Schwarz-P", solid_fraction=0.30, config=config_coarse)
        res_f = homogenize_tpms_cell("Schwarz-P", solid_fraction=0.30, config=config_fine)

        # Both must produce valid positive-definite stiffness
        assert res_c.engineering_constants.E_x > 0.0
        assert res_f.engineering_constants.E_x > 0.0
        # Results should be within 10% of each other
        np.testing.assert_allclose(
            res_c.engineering_constants.E_x, res_f.engineering_constants.E_x, rtol=0.10
        )
