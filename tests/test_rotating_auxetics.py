"""
Unit tests for Rotating Rigid Squares Auxetic Metamaterial Generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.generators.rotating_auxetics import generate_rotating_squares_lattice


class TestRotatingSquaresAuxetics:
    """Test Grima & Evans rotating squares auxetic kinematics and watertightness."""

    def test_rotating_squares_mesh_watertight(self):
        mesh = generate_rotating_squares_lattice(
            dimensions=(3, 3),
            square_side=10.0,
            plate_thickness=2.0,
            hinge_radius=0.5,
            rotation_angle_deg=25.0,
        )
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_deployment_angle_range(self):
        # 0 degrees (fully closed, zero void size)
        m_closed = generate_rotating_squares_lattice(
            dimensions=(2, 2),
            square_side=10.0,
            plate_thickness=1.5,
            hinge_radius=0.4,
            rotation_angle_deg=0.0,
        )
        assert m_closed.is_watertight
        assert m_closed.volume > 0.0

        # 25 degrees (partially deployed with open voids)
        m_deployed = generate_rotating_squares_lattice(
            dimensions=(2, 2),
            square_side=10.0,
            plate_thickness=1.5,
            hinge_radius=0.4,
            rotation_angle_deg=25.0,
        )
        assert m_deployed.is_watertight
        assert m_deployed.volume > 0.0

        # 45 degrees (maximum void opening)
        m_open = generate_rotating_squares_lattice(
            dimensions=(2, 2),
            square_side=10.0,
            plate_thickness=1.5,
            hinge_radius=0.4,
            rotation_angle_deg=45.0,
        )
        assert m_open.is_watertight
        assert m_open.volume > 0.0


    def test_3d_volumetric_stacking(self):
        mesh_3d = generate_rotating_squares_lattice(
            dimensions=(2, 2, 2),
            square_side=10.0,
            plate_thickness=2.0,
            hinge_radius=0.5,
            rotation_angle_deg=20.0,
            layer_spacing=4.0,
        )
        assert mesh_3d.is_watertight
        assert mesh_3d.extents[2] >= 4.0

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="rotation_angle_deg"):
            generate_rotating_squares_lattice(
                dimensions=(2, 2),
                rotation_angle_deg=75.0,
            )

        with pytest.raises(ValueError, match="dimensions"):
            generate_rotating_squares_lattice(
                dimensions=(0, 2),
            )
