"""
Unit tests for C-6-TT (Cubic Truncated Tetrahedral) 3D bulk PAM (Zhou et al., Science 2025).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphite.explicit.geometry_module import _trimesh_to_manifold
from graphite.explicit.interlinked import (
    generate_truncated_tetrahedron_particle,
    generate_c6tt_cubic_tiling,
    generate_pam_lattice,
    export_pam_multibody_stl,
    recalibrate_pam_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_C6TT = ROOT / "outputs" / "c6tt_3d_bulk.stl"


class TestC6TTParticle:
    def test_truncated_tetrahedron_topology(self):
        p = generate_truncated_tetrahedron_particle(size=10.0, center=(1.0, 2.0, 3.0), particle_id=5)
        assert p.geometry_type == "TT"
        assert p.nodes.shape == (12, 3)
        assert p.struts.shape == (18, 2)
        assert p.particle_id == 5
        np.testing.assert_allclose(p.center, (1.0, 2.0, 3.0))

        # Check tuple unpacking support: nodes, struts = p
        nodes, struts = p
        assert nodes.shape == (12, 3)
        assert struts.shape == (18, 2)

        # Edge lengths should all be identical: 2/3 * size
        expected_edge = (2.0 / 3.0) * 10.0
        for i, j in p.struts:
            edge_dist = float(np.linalg.norm(p.nodes[i] - p.nodes[j]))
            assert np.isclose(edge_dist, expected_edge, atol=1e-4)


class TestC6TTBulkLattice:
    @pytest.fixture(scope="class")
    def tiling_result(self):
        return generate_c6tt_cubic_tiling(
            repeats=(2, 2, 2),
            size=10.0,
            strut_radius=0.50,
            min_clearance=0.30,
            build_meshes=True,
        )

    def test_tiling_particle_count_and_code(self, tiling_result):
        assert tiling_result.tripartite_code == "C-6-TT"
        assert len(tiling_result.particles) == 8
        assert len(tiling_result.meshes) == 8
        assert tiling_result.clearance_valid

    def test_surface_clearance_dfam_criterion(self, tiling_result):
        # Minimum surface clearance must exceed 0.30 mm
        assert tiling_result.min_clearance_mm >= 0.30
        assert tiling_result.min_clearance_mm > 0.60

    def test_zero_solid_manifold_collision(self, tiling_result):
        # Convert meshes to Manifold3D and compute all pairwise Boolean intersections
        manifolds = [_trimesh_to_manifold(m) for m in tiling_result.meshes]
        for m in tiling_result.meshes:
            assert m.is_watertight

        for i in range(len(manifolds)):
            for j in range(i + 1, len(manifolds)):
                intersection = manifolds[i] ^ manifolds[j]
                inter_vol = float(intersection.volume())
                assert inter_vol <= 1e-4, f"Pair ({i}, {j}) collided with volume {inter_vol}"

    def test_multibody_stl_export(self, tiling_result):
        out_path = export_pam_multibody_stl(tiling_result, OUT_C6TT)
        assert out_path.is_file()
        assert out_path.stat().st_size > 1000

    def test_generate_pam_lattice_router(self):
        res = generate_pam_lattice("C-6-TT", unit_cell_size=10.0, repeats=(2, 2, 2), strut_radius=0.50, min_clearance=0.30)
        assert res.tripartite_code == "C-6-TT"
        assert res.clearance_valid
        assert len(res.particles) == 8

    def test_recalibrate_pam_lattice_c6tt(self):
        best_a0 = recalibrate_pam_lattice("C-6-TT", target_d=10.0, r=0.50, repeats=(2, 2, 2))
        assert best_a0 > 10.0
        assert 11.5 <= best_a0 <= 14.0
