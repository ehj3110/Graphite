"""
Unit tests for the Universal Analytical PAM Dimensioning Engine.
Tests all 11 PAM unit cell architectures across all 4 solving modes:
  1. (clearance, solid_fraction) -> (pitch, wire_diameter)
  2. (clearance, wire_diameter) -> (pitch, solid_fraction)
  3. (pitch, solid_fraction) -> (wire_diameter, clearance)
  4. (pitch, wire_diameter) -> (clearance, solid_fraction)
Also tests jamming bounds, configurable warning clearance, and generator integration.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from graphite.explicit.interlinked import (
    InterlinkedRegistry,
    InterlinkedConfig,
    generate_interlinked_lattice,
    PAMDimensioner,
    PAMDimensionResult,
    JammingLimitExceededError,
    DEFAULT_WARNING_CLEARANCE,
    dimension_pam,
)

ALL_PAM_CELLS = [
    "d4tet",
    "c6tt",
    "j4oct",
    "j4ring",
    "d4cube",
    "d4ct",
    "d4ico",
    "t6ring",
    "t6hex",
    "c6co",
    "b8co",
]


class TestPAMDimensionerCore:
    """Test core analytical methods of PAMDimensioner."""

    @pytest.mark.parametrize("cell_tag", ALL_PAM_CELLS)
    def test_c_geom_and_kappa_positive(self, cell_tag: str):
        inst = PAMDimensioner.resolve_cell(cell_tag)
        kappa = PAMDimensioner.get_kappa(inst)
        c_geom = PAMDimensioner.compute_c_geom(inst)
        vf_max = PAMDimensioner.get_jamming_limit(inst)

        assert kappa > 0.0, f"kappa must be positive for {cell_tag}"
        assert c_geom > 0.0, f"C_geom must be positive for {cell_tag}"
        assert 0.0 < vf_max < 1.0, f"Jamming limit Vf_max must be in (0, 1) for {cell_tag}"

    @pytest.mark.parametrize("cell_tag", ALL_PAM_CELLS)
    def test_round_trip_all_four_modes(self, cell_tag: str):
        """
        Verify mathematical consistency and bijection across all 4 solving modes:
          (a0, d0) -> (Delta, Vf) -> (a1, d1)
        """
        inst = PAMDimensioner.resolve_cell(cell_tag)
        kappa = PAMDimensioner.get_kappa(inst)
        c_geom = PAMDimensioner.compute_c_geom(inst)

        # Pick a safe forward state well below jamming limit
        a0 = 12.0
        # Choose d such that Delta > 0 and Vf < Vf_max
        xi_target = 0.5 * kappa  # half of jamming aspect ratio
        d0 = a0 * xi_target

        # Mode 4: (pitch, wire_diameter) -> (clearance, solid_fraction)
        fwd = PAMDimensioner.solve_from_pitch_and_wire(inst, pitch=a0, wire_diameter=d0)
        assert fwd.is_feasible
        assert fwd.clearance > 0.0
        assert fwd.solid_fraction > 0.0

        # Mode 1: (clearance, solid_fraction) -> (pitch, wire_diameter)
        inv = PAMDimensioner.solve_from_clearance_and_vf(
            inst,
            clearance=fwd.clearance,
            solid_fraction=fwd.solid_fraction,
        )
        assert math.isclose(inv.pitch, a0, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(inv.wire_diameter, d0, rel_tol=1e-5, abs_tol=1e-5)

        # Mode 2: (clearance, wire_diameter) -> (pitch, solid_fraction)
        mode2 = PAMDimensioner.solve_from_clearance_and_wire(
            inst,
            clearance=fwd.clearance,
            wire_diameter=d0,
        )
        assert math.isclose(mode2.pitch, a0, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(mode2.solid_fraction, fwd.solid_fraction, rel_tol=1e-5, abs_tol=1e-5)

        # Mode 3: (pitch, solid_fraction) -> (wire_diameter, clearance)
        mode3 = PAMDimensioner.solve_from_pitch_and_vf(
            inst,
            pitch=a0,
            solid_fraction=fwd.solid_fraction,
        )
        assert math.isclose(mode3.wire_diameter, d0, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(mode3.clearance, fwd.clearance, rel_tol=1e-5, abs_tol=1e-5)

    def test_universal_solve_dispatcher(self):
        """Test universal solve() function dispatching across various parameter pairs."""
        # a and Vf specified
        res1 = PAMDimensioner.solve("c6tt", pitch=10.0, solid_fraction=0.08)
        assert res1.is_feasible
        assert res1.pitch == 10.0
        assert math.isclose(res1.solid_fraction, 0.08)

        # Delta and d specified
        res2 = PAMDimensioner.solve("c6tt", clearance=0.30, wire_diameter=0.80)
        assert res2.is_feasible
        assert res2.clearance == 0.30
        assert res2.wire_diameter == 0.80

        # Exactly 2 parameters required
        with pytest.raises(ValueError, match="Exact 2-parameter specification required"):
            PAMDimensioner.solve("c6tt", pitch=10.0)

        with pytest.raises(ValueError, match="Exact 2-parameter specification required"):
            PAMDimensioner.solve("c6tt", pitch=10.0, wire_diameter=0.5, clearance=0.2)

    def test_convenience_function(self):
        """Test dimension_pam convenience function."""
        res = dimension_pam("j4ring", clearance=0.20, solid_fraction=0.10)
        assert isinstance(res, PAMDimensionResult)
        assert res.cell_name == "J-4-RING"
        assert res.clearance == 0.20
        assert res.is_feasible


class TestJammingLimitsAndWarnings:
    """Test jamming limit guards and configurable warning clearance threshold."""

    def test_jamming_limit_rejection(self):
        """Requesting Vf >= Vf_max must raise JammingLimitExceededError when requested."""
        vf_max = PAMDimensioner.get_jamming_limit("c6tt")
        excess_vf = vf_max + 0.05  # exceed jamming limit

        with pytest.raises(JammingLimitExceededError, match="Jamming Limit Exceeded"):
            PAMDimensioner.solve_from_clearance_and_vf(
                "c6tt",
                clearance=0.30,
                solid_fraction=excess_vf,
                raise_on_jamming=True,
            )

    def test_jamming_limit_non_raising(self):
        """When raise_on_jamming=False, returns is_feasible=False with warning message."""
        vf_max = PAMDimensioner.get_jamming_limit("c6tt")
        res = PAMDimensioner.solve_from_clearance_and_vf(
            "c6tt",
            clearance=0.30,
            solid_fraction=vf_max + 0.05,
            raise_on_jamming=False,
        )
        assert not res.is_feasible
        assert any("Jamming Limit Exceeded" in w for w in res.warnings)

    def test_warning_clearance_threshold(self):
        """Configurable warning clearance threshold triggers warnings when Delta < warning_clearance."""
        # 1. Delta = 0.20 mm < default 0.250 mm -> warning triggered
        res1 = PAMDimensioner.solve_from_clearance_and_vf(
            "c6tt",
            clearance=0.20,
            solid_fraction=0.06,
            warning_clearance=0.250,
        )
        assert any("below the recommended manufacturing warning threshold" in w for w in res1.warnings)

        # 2. Lower threshold to 0.150 mm -> Delta = 0.20 mm > 0.150 mm -> no warning
        res2 = PAMDimensioner.solve_from_clearance_and_vf(
            "c6tt",
            clearance=0.20,
            solid_fraction=0.06,
            warning_clearance=0.150,
        )
        assert len(res2.warnings) == 0

    def test_percentage_auto_normalization(self):
        """Entering 8.0 for 8% solid fraction is auto-normalized to 0.08."""
        res_fraction = PAMDimensioner.solve_from_clearance_and_vf("d4tet", clearance=0.30, solid_fraction=0.08)
        res_percent = PAMDimensioner.solve_from_clearance_and_vf("d4tet", clearance=0.30, solid_fraction=8.0)

        assert math.isclose(res_fraction.pitch, res_percent.pitch)
        assert math.isclose(res_fraction.wire_diameter, res_percent.wire_diameter)
        assert math.isclose(res_percent.solid_fraction, 0.08)


class TestCellProtocolResolveDimensions:
    """Test resolve_dimensions on individual cell instances."""

    @pytest.mark.parametrize("cell_tag", ALL_PAM_CELLS)
    def test_cell_resolve_dimensions_method(self, cell_tag: str):
        inst = PAMDimensioner.resolve_cell(cell_tag)
        assert hasattr(inst, "resolve_dimensions")

        # Use 0.04 which is below the jamming limit of all 11 PAM cells (b8co max is ~5.27%)
        pitch, strut_d = inst.resolve_dimensions(target_clearance=0.25, target_solid_fraction=0.04)
        assert pitch > 0.0
        assert strut_d > 0.0

        # Confirm clearance matches target
        pred_clr = inst.forward_clearance(pitch, strut_d)
        assert math.isclose(pred_clr, 0.25, rel_tol=1e-4)


class TestEngineIntegration:
    """Test integration into InterlinkedConfig and generate_interlinked_lattice."""

    def test_auto_dimensioning_in_generator(self):
        """generate_interlinked_lattice automatically dimensions pitch and wire_radius from target_solid_fraction."""
        config = InterlinkedConfig(
            cell="c6tt",
            grid_size=(2, 2, 2),
            target_solid_fraction=0.08,
            min_clearance=0.20,
            warning_clearance=0.250,
        )
        result = generate_interlinked_lattice(config)

        dim_res = result.metadata.get("dimensioning")
        assert dim_res is not None
        assert isinstance(dim_res, PAMDimensionResult)
        assert math.isclose(result.metadata["pitch"], dim_res.pitch)
        assert math.isclose(result.metadata["wire_radius"], dim_res.wire_radius)
        assert result.metadata["clearance_valid"] is True
