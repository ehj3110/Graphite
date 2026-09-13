"""
Unit tests for graphite.explicit.interlinked 3MF instanced export and Phase 5 generator API.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import trimesh

from graphite.explicit.interlinked import (
    ParticleGeometry,
    InterlinkedParticle,
    InterlinkedConfig,
    InterlinkedLatticeResult,
    generate_interlinked_lattice,
    solidify_particle_prototype,
    format_3mf_transform,
    export_interlinked_3mf,
    InterlinkedRegistry,
    C6TTCell,
    D4TetCell,
    European4in1Cell,
    build_perimeter_frame_solid,
    seed_cartesian_lattice,
    instantiate_lattice_on_sites,
)


class TestWriter3MF:
    """Test instanced 3MF writer primitives."""

    def test_format_3mf_transform(self):
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = 10.5
        T[1, 3] = -20.25
        T[2, 3] = 3.0
        # 90 deg rotation about Z
        T[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        s = format_3mf_transform(T)
        parts = [float(x) for x in s.split()]
        assert len(parts) == 12
        # First 9: row-major 3x3 rotation
        np.testing.assert_allclose(parts[:9], [0, -1, 0, 1, 0, 0, 0, 0, 1], atol=1e-6)
        # Last 3: translation
        np.testing.assert_allclose(parts[9:], [10.5, -20.25, 3.0], atol=1e-6)

    def test_solidify_particle_prototype_ring(self):
        geom = ParticleGeometry(
            nodes=np.zeros((16, 3)),
            struts=np.zeros((16, 2), dtype=np.int64),
            geometry_type="ring",
            bounding_radius=5.0,
            metadata={"major_radius": 4.5},
        )
        mesh = solidify_particle_prototype(geom, strut_radius=0.4, circular_segments=16)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_solidify_particle_prototype_truss(self):
        cell = C6TTCell()
        geom = cell.basis_particles[0].geometry
        mesh = solidify_particle_prototype(geom, strut_radius=0.4, circular_segments=16)
        assert isinstance(mesh, trimesh.Trimesh)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_export_interlinked_3mf_basic(self, tmp_path):
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(1, 1, 1), pitch=10.0)
        particles = instantiate_lattice_on_sites(cell, centers, frames, indices, cell_pitch=10.0)

        # Place 4 copies
        all_particles = []
        for i in range(4):
            T_i = np.array([
                [1, 0, 0, i * 12.0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)
            p = particles[0].copy_transformed(new_id=i, new_transform=T_i)
            all_particles.append(p)

        out_3mf = tmp_path / "test_instanced.3mf"
        export_interlinked_3mf(
            particles=all_particles,
            strut_radius=0.4,
            output_path=out_3mf,
            circular_segments=16,
        )

        assert out_3mf.exists()
        assert out_3mf.stat().st_size > 0

        # Load with trimesh
        scene = trimesh.load(str(out_3mf), file_type="3mf")
        assert scene is not None

    def test_3mf_file_size_compression(self, tmp_path):
        """Instanced 3MF should be dramatically smaller than raw monolithic STL for large arrays."""
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(4, 4, 1), pitch=12.0)
        particles = instantiate_lattice_on_sites(cell, centers, frames, indices, cell_pitch=12.0)
        assert len(particles) == 16

        out_3mf = tmp_path / "lattice_16parts.3mf"
        export_interlinked_3mf(
            particles=particles,
            strut_radius=0.4,
            output_path=out_3mf,
            circular_segments=16,
        )

        # Compare with single monolithic STL of all 16 particles
        proto_mesh = solidify_particle_prototype(particles[0].geometry, strut_radius=0.4, circular_segments=16)
        all_meshes = []
        for p in particles:
            m_copy = proto_mesh.copy()
            m_copy.apply_transform(p.transform)
            all_meshes.append(m_copy)
        monolithic_stl = tmp_path / "lattice_16parts.stl"
        trimesh.util.concatenate(all_meshes).export(monolithic_stl)

        size_3mf = out_3mf.stat().st_size
        size_stl = monolithic_stl.stat().st_size

        # 3MF contains 1 mesh definition in resources + 16 transforms in build
        # STL contains 16 copies of the mesh vertices and faces
        compression_ratio = 1.0 - (size_3mf / size_stl)
        assert compression_ratio > 0.80, f"Expected >80% compression, got {compression_ratio * 100:.1f}%"


class TestUnifiedGeneratorPhase5:
    """Test unified generator API supporting modular cells, perimeter framing, and 3MF."""

    def test_generate_with_cell_string(self):
        cfg = InterlinkedConfig(
            cell="c6tt",
            grid_size=(2, 2, 1),
            pitch=12.0,
            wire_radius=0.4,
        )
        res = generate_interlinked_lattice(cfg)
        assert isinstance(res, InterlinkedLatticeResult)
        assert len(res.particles) == 4
        assert res.num_rings == 4
        assert res.mesh.is_watertight
        assert res.clearance_valid

    def test_generate_with_cell_instance(self):
        cell = European4in1Cell()
        cfg = InterlinkedConfig(
            cell=cell,
            grid_size=(2, 2, 1),
            pitch=10.0,
            wire_radius=0.35,
        )
        res = generate_interlinked_lattice(cfg)
        assert len(res.particles) == 4  # 1 basis particle * 4 sites
        assert res.mesh.is_watertight

    def test_generate_with_perimeter_frame(self, tmp_path):
        cfg = InterlinkedConfig(
            cell="c6tt",
            grid_size=(2, 2, 1),
            pitch=12.0,
            wire_radius=0.4,
            add_perimeter_frame=True,
            frame_wall_thickness=1.0,
            frame_margin=0.5,
        )
        res = generate_interlinked_lattice(cfg)
        assert res.frame_mesh is not None
        assert res.frame_mesh.is_watertight

        stl_path = tmp_path / "framed_lattice.stl"
        exported_stl = res.export_stl(stl_path)
        assert exported_stl.exists()

        mf_path = tmp_path / "framed_lattice.3mf"
        exported_3mf = res.export_3mf(mf_path)
        assert exported_3mf.exists()

    def test_legacy_backwards_compatibility(self):
        """Ensure pattern='european_4in1' without cell still produces legacy ring structure."""
        cfg = InterlinkedConfig(
            pattern="european_4in1",
            grid_size=(2, 2, 1),
            pitch=10.0,
            radius_ratio=0.65,
            wire_radius=0.40,
            num_ring_segments=16,
        )
        res = generate_interlinked_lattice(cfg)
        assert res.num_rings == 4
        assert len(res.rings) == 4
        assert res.mesh.is_watertight
