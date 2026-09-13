"""
Unit tests for Hashin-Shtrikman Optimal Plate-Lattice Generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.generators.plate_lattice import (
    generate_plate_lattice,
    calibrate_plate_thickness,
)


class TestPlateLattices:
    """Test SC, BCC, FCC, and hybrid plate-lattices and thickness calibration."""

    def test_sc_plate_lattice_watertight(self):
        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (20.0, 20.0, 20.0)),
            unit_cell_size=10.0,
            plate_thickness=0.5,
            topology="sc",
            crop_to_bounds=True,
        )
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume > 0.0
        # Extents should match bounds
        np.testing.assert_allclose(mesh.extents, [20.0, 20.0, 20.0], atol=0.1)

    def test_bcc_plate_lattice_watertight(self):
        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            unit_cell_size=10.0,
            plate_thickness=0.3,
            topology="bcc",
            crop_to_bounds=True,
        )
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_fcc_plate_lattice_watertight(self):
        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            unit_cell_size=10.0,
            plate_thickness=0.3,
            topology="fcc",
            crop_to_bounds=True,
        )
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_sc_bcc_hybrid_watertight(self):
        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
            unit_cell_size=10.0,
            plate_thickness=0.2,
            topology="sc_bcc",
            crop_to_bounds=True,
        )
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_plate_thickness_calibration(self):
        a = 10.0
        target_sf = 0.10
        t_opt = calibrate_plate_thickness(a, target_sf, topology="sc")
        assert t_opt > 0.0

        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (a, a, a)),
            unit_cell_size=a,
            plate_thickness=t_opt,
            topology="sc",
            crop_to_bounds=True,
        )
        measured_sf = mesh.volume / (a**3)
        assert abs(measured_sf - target_sf) < 0.005

    def test_direct_target_solid_fraction_generation(self):
        mesh = generate_plate_lattice(
            bounds=((0.0, 0.0, 0.0), (20.0, 20.0, 10.0)),
            unit_cell_size=10.0,
            target_solid_fraction=0.12,
            topology="sc",
            crop_to_bounds=True,
        )
        assert mesh.is_watertight
        measured_sf = mesh.volume / (20.0 * 20.0 * 10.0)
        assert abs(measured_sf - 0.12) < 0.01

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            generate_plate_lattice(
                bounds=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)),
                unit_cell_size=10.0,
                plate_thickness=0.5,
                topology="invalid_topology",
            )
