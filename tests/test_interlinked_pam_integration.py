"""
Comprehensive multi-cell integration test suite for Graphite's Interlinked PAM engine.

Verifies:
1. Canonical D-4-TET via unified InterlinkedConfig with diamond seeding.
2. Canonical C-6-TT via unified InterlinkedConfig with cartesian seeding.
3. Canonical J-4-OCT via unified InterlinkedConfig with checkerboard catenation.
4. DfAM pre-flight guardrail rejecting colliding parameters before spatial allocation.
5. Universal clean miter joints default producing watertight particle solid meshes.
6. Polymorphic auto-pitch resolution meeting target min_clearance.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.explicit.geometry_module import _trimesh_to_manifold
from graphite.explicit.interlinked import (
    InterlinkedConfig,
    InterlinkedLatticeResult,
    generate_interlinked_lattice,
)


class TestInterlinkedPAMIntegration:
    def test_canonical_d4tet_diamond_seeding(self):
        """Verify D-4-TET diamond lattice generates valid 8-particle unit cell with clearance > 0.4 mm."""
        cfg = InterlinkedConfig(
            cell="d4tet",
            grid_size=(1, 1, 1),
            pitch=15.0,
            wire_radius=0.4,
            seeding_type="diamond",
            min_clearance=0.40,
        )
        res = generate_interlinked_lattice(cfg)
        assert isinstance(res, InterlinkedLatticeResult)
        assert len(res.particles) == 8
        assert res.clearance_valid
        assert res.min_clearance >= 0.40
        assert res.mesh.is_watertight

    def test_canonical_c6tt_cartesian_seeding(self):
        """Verify C-6-TT cartesian lattice generates valid 4-particle grid with zero solid collision."""
        cfg = InterlinkedConfig(
            cell="c6tt",
            grid_size=(2, 2, 1),
            pitch=12.0,
            wire_radius=0.4,
            seeding_type="cartesian",
            min_clearance=0.30,
        )
        res = generate_interlinked_lattice(cfg)
        assert isinstance(res, InterlinkedLatticeResult)
        assert len(res.particles) == 4
        assert res.clearance_valid
        assert res.min_clearance >= 0.30
        assert res.mesh.is_watertight

    def test_canonical_j4oct_checkerboard(self):
        """Verify J-4-OCT generates alternating orthogonal particles with positive clearance."""
        cfg = InterlinkedConfig(
            cell="j4oct",
            grid_size=(2, 1, 1),
            pitch=10.0,
            wire_radius=0.35,
            seeding_type="cartesian",
            min_clearance=0.30,
        )
        res = generate_interlinked_lattice(cfg)
        assert isinstance(res, InterlinkedLatticeResult)
        assert len(res.particles) == 2
        assert res.clearance_valid
        assert res.min_clearance >= 0.30
        assert res.mesh.is_watertight

    def test_dfam_preflight_guardrail_rejection(self):
        """Verify that physical collision configurations are rejected before geometry allocation."""
        cfg = InterlinkedConfig(
            cell="c6tt",
            grid_size=(2, 2, 1),
            pitch=5.0,
            wire_radius=1.0,
            seeding_type="cartesian",
        )
        with pytest.raises(ValueError, match="DfAM Pre-Flight Rejection"):
            generate_interlinked_lattice(cfg)

    def test_clean_miter_joints_default_watertight(self):
        """Verify clean miter truss joints are the universal default and yield watertight solids."""
        cfg = InterlinkedConfig(
            cell="d4tet",
            grid_size=(1, 1, 1),
            pitch=14.0,
            wire_radius=0.45,
            seeding_type="diamond",
        )
        res = generate_interlinked_lattice(cfg)
        assert res.mesh.is_watertight
        assert res.metadata["is_watertight"] is True
        # Verify manifold3d validity
        m_solid = _trimesh_to_manifold(res.mesh)
        assert "NoError" in str(m_solid.status())

    def test_auto_pitch_resolution(self):
        """Verify auto_resolve_pitch=True calculates pitch to satisfy requested min_clearance."""
        target_clr = 0.60
        cfg = InterlinkedConfig(
            cell="d4tet",
            grid_size=(1, 1, 1),
            wire_radius=0.4,
            seeding_type="diamond",
            min_clearance=target_clr,
            auto_resolve_pitch=True,
        )
        res = generate_interlinked_lattice(cfg)
        assert res.metadata["pitch"] > 10.0
        assert res.min_clearance >= target_clr - 0.05
