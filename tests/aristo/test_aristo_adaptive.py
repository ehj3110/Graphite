"""
Integration and Unit Tests for Aristo Adapt (Two-Scale Stress-Adaptive Lattice Optimization).
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.aristo import (
    AristoAdaptConfig,
    AristoAdaptResult,
    MacroMesh,
    MaterialTensorSurrogate,
    build_octet_homogenization_surrogate,
    create_box_continuum_mesh,
    homogenize_octet_cell,
    homogenize_tpms_cell,
    homogenize_voxel_rve,
    run_aristo_adaptive,
    run_two_scale_macro_fea,
)


class TestAristoAdaptive:
    """Test suite for Aristo Adapt API and workflow."""

    def test_imports_and_public_api(self):
        """Verify all Aristo Adapt entry points are importable and functional."""
        assert AristoAdaptConfig is not None
        assert run_aristo_adaptive is not None
        assert build_octet_homogenization_surrogate is not None
        assert create_box_continuum_mesh is not None

    def test_run_aristo_adaptive_box_workflow(self, tmp_path):
        """Run end-to-end two-scale adaptive optimization on a box part."""
        bounds = ((0.0, 0.0, 0.0), (12.0, 6.0, 4.0))
        config = AristoAdaptConfig(
            target_stress=30.0,
            target_volume_fraction=0.15,
            move_limit=0.10,
            max_iterations=5,
            convergence_tol=1e-3,
        )

        out_vtu = tmp_path / "box_opt.vtu"
        out_stl = tmp_path / "box_lattice.stl"

        result = run_aristo_adaptive(
            part=bounds,
            config=config,
            lattice_type="octet",
            cell_size_mm=2.0,
            fixed_face="-x",
            load_face="+x",
            total_force_N=500.0,
            load_direction=(1.0, 0.0, 0.0),
            realize_lattice=True,
            clean_miter=True,
            output_stl=out_stl,
            output_vtu=out_vtu,
        )

        assert isinstance(result, AristoAdaptResult)
        assert len(result.history) > 0
        assert result.optimal_densities.shape == (result.mesh.elements.shape[0],)
        assert np.all(result.optimal_densities >= config.min_density - 1e-4)
        assert np.all(result.optimal_densities <= config.max_density + 1e-4)

        # Check volume fraction conservation
        elem_vols = result.mesh.element_volumes
        total_vol = np.sum(elem_vols)
        achieved_vf = float(np.sum(result.optimal_densities * elem_vols) / total_vol)
        np.testing.assert_allclose(achieved_vf, 0.15, rtol=1e-2)

        # Check export files exist
        assert out_vtu.exists()
        assert out_stl.exists()
        assert out_stl.stat().st_size > 0
