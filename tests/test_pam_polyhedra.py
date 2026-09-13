"""
Unit tests for higher-order PAM polyhedra (CO, OCT) — Phase 2.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from graphite.explicit.geometry_module import _trimesh_to_manifold
from graphite.explicit.interlinked.pams import (
    align_particle_axis,
    export_pam_multibody_stl,
    generate_c6co_coordination_cell,
    generate_c6co_cubic_tiling,
    generate_cuboctahedral_particle,
    generate_j4oct_interlocked_pair,
    generate_j4oct_square_tiling,
    generate_octahedral_particle,
    generate_pam_lattice,
    particle_pair_clearance,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_C6CO = ROOT / "outputs" / "c6co_interlocked_cell.stl"
OUT_J4OCT = ROOT / "outputs" / "j4oct_interlocked_cell.stl"


class TestPolyhedralGenerators:
    def test_cuboctahedron_topology(self):
        p = generate_cuboctahedral_particle(8.0, center=(1.0, 2.0, 3.0), particle_id=3)
        assert p.geometry_type == "CO"
        assert p.nodes.shape == (12, 3)
        assert p.struts.shape == (24, 2)
        assert p.struts.dtype == np.int64
        assert set(map(int, p.struts.ravel())) <= set(range(12))
        np.testing.assert_allclose(p.center, (1.0, 2.0, 3.0))
        for i, j in p.struts:
            assert abs(float(np.linalg.norm(p.nodes[i] - p.nodes[j])) - 8.0) < 1e-8

    def test_octahedron_topology(self):
        p = generate_octahedral_particle(6.0, particle_id=1)
        assert p.geometry_type == "OCT"
        assert p.nodes.shape == (6, 3)
        assert p.struts.shape == (12, 2)
        edge = 6.0 * np.sqrt(2.0)
        for i, j in p.struts:
            assert abs(float(np.linalg.norm(p.nodes[i] - p.nodes[j])) - edge) < 1e-8

    def test_align_particle_axis(self):
        p = generate_octahedral_particle(5.0)
        q = align_particle_axis(p, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        tip = q.nodes[np.argmax(q.nodes @ np.array([0.0, 0.0, 1.0]))]
        assert tip[2] > 4.5


class TestC6CO:
    @pytest.fixture(scope="class")
    def cell(self):
        return generate_c6co_cubic_tiling(
            repeats=(2, 2, 2),
            size=8.0,
            strut_radius=0.35,
            min_clearance=0.40,
        )

    def test_coordination_shell(self, cell):
        assert cell.tripartite_code == "C-6-CO"
        assert len(cell.particles) == 7  # 1 + 6
        assert cell.metadata["topology"] == "coordination_shell_n6"
        for p in cell.particles:
            assert set(map(int, p.struts.ravel())) <= set(range(12))

    def test_clearance_and_zero_intersection(self, cell):
        assert cell.min_clearance_mm >= 0.40
        assert cell.clearance_valid
        center = cell.meshes[0]
        assert center.is_watertight
        m0 = _trimesh_to_manifold(center)
        for m in cell.meshes[1:]:
            assert m.is_watertight
            mj = _trimesh_to_manifold(m)
            assert float((m0 ^ mj).volume()) <= 1e-4
            assert float(m0.min_gap(mj, 40.0)) >= 0.40

    def test_export_and_api(self, cell):
        path = export_pam_multibody_stl(cell, OUT_C6CO)
        assert path.is_file()
        res = generate_pam_lattice("C-6-CO", 8.0, repeats=(2, 2, 2), strut_radius=0.35)
        assert res.clearance_valid
        assert len(res.particles) == 7
        shell = generate_c6co_coordination_cell(size=8.0, strut_radius=0.35)
        assert shell.clearance_valid


class TestJ4OCT:
    @pytest.fixture(scope="class")
    def cell(self):
        return generate_j4oct_square_tiling(
            repeats=(2, 2),
            size=6.0,
            strut_radius=0.40,
            min_clearance=0.40,
        )

    def test_clearance_and_export(self, cell):
        assert cell.tripartite_code == "J-4-OCT"
        assert len(cell.particles) == 5  # 1 + 4
        assert cell.clearance_valid
        assert cell.min_clearance_mm >= 0.40
        m0 = _trimesh_to_manifold(cell.meshes[0])
        for m in cell.meshes[1:]:
            mj = _trimesh_to_manifold(m)
            assert float((m0 ^ mj).volume()) <= 1e-4
        path = export_pam_multibody_stl(cell, OUT_J4OCT)
        assert path.is_file()
        pair = generate_j4oct_interlocked_pair(size=6.0, strut_radius=0.40)
        assert pair.clearance_valid
        assert particle_pair_clearance(pair.particles[0], pair.particles[1], 0.40) >= 0.40
