"""
Unit tests for Phase-1 D-4-TET polycatenated tetrahedral particle pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from graphite.explicit.geometry_module import _trimesh_to_manifold
from graphite.explicit.interlinked.pams import (
    generate_d4tet_interlocked_pair,
    generate_d4tet_diamond_tiling,
    generate_pam_lattice,
    generate_tetrahedral_particle,
    particle_pair_clearance,
    particles_are_corner_catenated,
    export_pam_multibody_stl,
    diamond_network_sites,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_STL = ROOT / "outputs" / "d4tet_interlocked_pair.stl"


class TestPAMTetrahedralParticle:
    def test_tet_topology(self):
        p = generate_tetrahedral_particle(edge_length=10.0, center=(1.0, 2.0, 3.0), particle_id=7)
        assert p.geometry_type == "TET"
        assert p.particle_id == 7
        assert p.nodes.shape == (4, 3)
        assert p.struts.shape == (6, 2)
        assert p.struts.dtype == np.int64
        # Local indices only
        assert int(p.struts.min()) >= 0
        assert int(p.struts.max()) <= 3
        np.testing.assert_allclose(p.center, (1.0, 2.0, 3.0))
        # Edge lengths ~10
        for i, j in p.struts:
            assert abs(float(np.linalg.norm(p.nodes[i] - p.nodes[j])) - 10.0) < 1e-9


class TestD4TETInterlockedPair:
    @pytest.fixture(scope="class")
    def pair(self):
        return generate_d4tet_interlocked_pair(
            edge_length=12.0,
            strut_radius=0.55,
            min_clearance=0.40,
        )

    def test_two_independent_particles(self, pair):
        assert pair.tripartite_code == "D-4-TET"
        assert len(pair.particles) == 2
        a, b = pair.particles
        assert a.particle_id != b.particle_id
        # Decoupled local index spaces (both use 0..3; no shared global merge)
        assert set(map(int, a.struts.ravel())) <= {0, 1, 2, 3}
        assert set(map(int, b.struts.ravel())) <= {0, 1, 2, 3}
        # Centers distinct
        assert float(np.linalg.norm(a.center - b.center)) > 1.0

    def test_corner_catenation_and_clearance(self, pair):
        a, b = pair.particles
        assert particles_are_corner_catenated(a, b)
        clr = particle_pair_clearance(a, b, pair.strut_radius)
        assert clr >= 0.40
        assert pair.min_clearance_mm >= 0.40
        assert pair.clearance_valid is True

    def test_zero_solid_intersection(self, pair):
        assert len(pair.meshes) == 2
        for m in pair.meshes:
            assert isinstance(m, trimesh.Trimesh)
            assert m.is_watertight
            assert m.volume > 0.0
        m0 = _trimesh_to_manifold(pair.meshes[0])
        m1 = _trimesh_to_manifold(pair.meshes[1])
        inter_vol = float((m0 ^ m1).volume())
        assert inter_vol <= 1e-4
        gap = float(m0.min_gap(m1, 40.0))
        assert gap >= 0.40

    def test_generate_pam_lattice_api(self):
        res = generate_pam_lattice(
            "D-4-TET",
            unit_cell_size=12.0,
            repeats=(1, 1, 1),
            strut_radius=0.55,
            min_clearance=0.40,
        )
        assert res.clearance_valid
        assert len(res.particles) == 2

    def test_export_review_stl(self, pair):
        path = export_pam_multibody_stl(pair, OUT_STL)
        assert path.is_file()
        loaded = trimesh.load_mesh(path, force="mesh")
        assert isinstance(loaded, trimesh.Trimesh)
        assert loaded.faces.shape[0] > 0


class TestD4TETDiamondTiling:
    def test_conventional_cell_site_count(self):
        sites = diamond_network_sites((1, 1, 1), conventional_cell_size=20.0)
        assert len(sites) == 8
        assert sum(1 for _, s in sites if s == "A") == 4
        assert sum(1 for _, s in sites if s == "B") == 4

    def test_small_tiling_clearance(self):
        res = generate_d4tet_diamond_tiling(
            repeats=(1, 1, 1),
            edge_length=12.0,
            strut_radius=0.55,
            min_clearance=0.40,
            build_meshes=False,
        )
        assert res.metadata["num_particles"] == 8
        assert res.clearance_valid
        assert res.min_clearance_mm >= 0.40
        for p in res.particles:
            assert set(map(int, p.struts.ravel())) <= {0, 1, 2, 3}

    def test_conventional_cell_size_sizing(self):
        from graphite.explicit.interlinked.pams import calibrate_d4tet_edge_length

        a = 10.0
        r = 0.5
        L = calibrate_d4tet_edge_length(a, r, 0.40)
        assert 6.5 <= L <= 8.5
        res = generate_d4tet_diamond_tiling(
            repeats=(1, 1, 1),
            strut_radius=r,
            min_clearance=0.40,
            conventional_cell_size=a,
            build_meshes=False,
        )
        assert abs(float(res.metadata["edge_length"]) - L) < 1e-6
        assert res.metadata["num_particles"] == 8
        assert abs(float(res.metadata["conventional_cell_size"]) - a) < 1e-9
        assert abs(float(res.metadata["bond_length"]) - a * np.sqrt(3.0) / 4.0) < 1e-9
        assert res.metadata["extent_mm"] == (a, a, a)
        assert res.clearance_valid
        assert res.min_clearance_mm >= 0.40
