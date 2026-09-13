"""
Unit tests for Spatial Seeding & Boundary Management (Phase 3):
- Translational seeders (Cartesian, Staggered, Hexagonal, Diamond)
- Analytical primitive mappings (Cylindrical wrap with pitch-matching quantization, Spherical shell)
- Universal site placer (instantiate_lattice_on_sites)
- 3-tier boundary management (SDF Inset Culling, Boundary Anchors, Perimeter Frame Welding)
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.explicit.interlinked import (
    C6TTCell,
    European4in1Cell,
    D4TetCell,
    InterlinkedParticle,
    seed_cartesian_lattice,
    seed_staggered_lattice,
    seed_hexagonal_lattice,
    seed_diamond_lattice,
    seed_cylindrical_wrap,
    seed_spherical_shell,
    instantiate_lattice_on_sites,
    cull_particles_by_sdf,
    cull_particles_by_mesh,
    identify_boundary_particles,
    build_perimeter_frame_solid,
    fuse_perimeter_frame,
)


class TestTranslationalSeeders:
    def test_cartesian_seeder(self):
        centers, frames, indices = seed_cartesian_lattice(
            repeats=(3, 4, 2),
            pitch=10.0,
            origin=(1.0, 2.0, 3.0),
        )
        assert len(centers) == 24
        assert centers.shape == (24, 3)
        assert frames.shape == (24, 3, 3)
        assert indices.shape == (24, 3)

        # Min and max coordinates
        np.testing.assert_allclose(centers[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(centers[-1], [1.0 + 2 * 10.0, 2.0 + 3 * 10.0, 3.0 + 1 * 10.0])

        # All frames are identity
        for F in frames:
            np.testing.assert_allclose(F, np.eye(3))

    def test_staggered_seeder(self):
        centers, frames, indices = seed_staggered_lattice(
            repeats=(4, 4),
            pitch=10.0,
            stagger_fraction=0.5,
        )
        assert len(centers) == 16
        # Row 0 (j=0) has no shift: x = 0, 10, 20, 30
        np.testing.assert_allclose(centers[0:4, 0], [0.0, 10.0, 20.0, 30.0])
        # Row 1 (j=1) has +5.0 shift: x = 5, 15, 25, 35
        np.testing.assert_allclose(centers[4:8, 0], [5.0, 15.0, 25.0, 35.0])

    def test_hexagonal_seeder(self):
        centers, frames, indices = seed_hexagonal_lattice(
            repeats=(3, 3),
            pitch=10.0,
        )
        assert len(centers) == 9
        dy = 10.0 * np.sqrt(3.0) / 2.0
        # Row 0 y = 0
        np.testing.assert_allclose(centers[0, 1], 0.0)
        # Row 1 y = dy, x shifted by 5.0
        np.testing.assert_allclose(centers[3, 1], dy)
        np.testing.assert_allclose(centers[3, 0], 5.0)

    def test_diamond_seeder(self):
        centers, frames, indices, sublattices = seed_diamond_lattice(
            repeats=(1, 1, 1),
            conventional_cell_size=16.0,
        )
        # 1 conventional diamond cell has 8 atoms (4 on A, 4 on B)
        assert len(centers) == 8
        assert sublattices.count("A") == 4
        assert sublattices.count("B") == 4


class TestAnalyticalPrimitives:
    def test_cylindrical_wrap_quantization(self):
        radius = 20.0
        height = 30.0
        target_pitch = 8.0

        res = seed_cylindrical_wrap(
            radius=radius,
            height=height,
            target_pitch=target_pitch,
        )

        N_theta = res["num_theta"]
        a_theta = res["quantized_pitch_theta"]
        N_z = res["num_z"]
        a_z = res["quantized_pitch_z"]

        # Exact pitch-matching quantization: N_theta * a_theta == 2 * pi * R
        np.testing.assert_allclose(N_theta * a_theta, 2.0 * np.pi * radius, rtol=1e-10)
        np.testing.assert_allclose(N_z * a_z, height, rtol=1e-10)

        # Frame orthonormality: F = [t1, t2, n]
        frames = res["frames"]
        for F in frames:
            # F @ F.T == I
            np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-10)
            det = float(np.linalg.det(F))
            assert abs(det - 1.0) < 1e-6, "Surface frame must be a right-handed SO(3) matrix"

        # Normals point radially outward
        centers = res["centers"]
        for c, F in zip(centers, frames):
            n = F[:, 2]  # third column is normal
            rad_vec = np.array([c[0], c[1], 0.0])
            rad_vec /= np.linalg.norm(rad_vec)
            np.testing.assert_allclose(n, rad_vec, atol=1e-6)

        # Seam continuity check: site (N_theta - 1, j) to (0, j)
        for j in range(N_z):
            idx_0 = j * N_theta
            idx_last = j * N_theta + (N_theta - 1)
            c0 = centers[idx_0]
            clast = centers[idx_last]
            chord_dist = float(np.linalg.norm(c0 - clast))
            # Chord length for angle delta_th = 2*pi / N_theta: 2*R*sin(delta_th/2)
            expected_chord = 2.0 * radius * np.sin(np.pi / N_theta)
            np.testing.assert_allclose(chord_dist, expected_chord, atol=1e-6)

    def test_spherical_shell_seeder(self):
        radius = 25.0
        res = seed_spherical_shell(radius=radius, target_pitch=10.0)
        centers = res["centers"]
        frames = res["frames"]

        assert len(centers) > 10
        # All centers must lie exactly on radius R
        radii = np.linalg.norm(centers, axis=1)
        np.testing.assert_allclose(radii, radius, atol=1e-6)

        # All frames orthonormal with normal pointing radially outward
        for c, F in zip(centers, frames):
            np.testing.assert_allclose(F @ F.T, np.eye(3), atol=1e-10)
            n = F[:, 2]
            unit_c = c / np.linalg.norm(c)
            np.testing.assert_allclose(n, unit_c, atol=1e-6)


class TestUniversalPlacer:
    def test_instantiate_c6tt_on_cartesian(self):
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(
            repeats=(2, 2, 2),
            pitch=12.7,
        )
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )
        assert len(particles) == 8
        for p in particles:
            assert isinstance(p, InterlinkedParticle)
            assert len(p.global_nodes()) == 12
            assert len(p.struts) == 18

    def test_instantiate_european_on_cylindrical_wrap(self):
        cell = European4in1Cell()
        cyl = seed_cylindrical_wrap(radius=20.0, height=20.0, target_pitch=8.0)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=cyl["centers"],
            frames=cyl["frames"],
            grid_indices=cyl["indices"],
            cell_pitch=cyl["quantized_pitch_theta"],
        )
        assert len(particles) == cyl["num_theta"] * cyl["num_z"]
        # All particles have non-empty nodes and struts
        for p in particles:
            assert len(p.global_nodes()) == 24
            assert len(p.struts) == 24
            # Bounding radius check
            assert p.bounding_radius > 0


class TestBoundaryPolicies:
    def test_sdf_inset_culling(self):
        # Grid of 5x5x1 C6TT particles at 12.7mm pitch -> spanning 0..50.8mm
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(5, 5, 1), pitch=12.7)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )
        assert len(particles) == 25

        # Define an SDF of a box centered at [25.4, 25.4, 0] with half-widths [18.0, 18.0, 20.0]
        # Particles near the perimeter (x=0, x=50.8, y=0, y=50.8) should be culled
        box_center = np.array([25.4, 25.4, 0.0])
        half_extents = np.array([16.0, 16.0, 20.0])

        def box_sdf(pts: np.ndarray) -> np.ndarray:
            d = np.abs(pts - box_center) - half_extents
            outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            inside_dist = np.minimum(np.max(d, axis=1), 0.0)
            return outside_dist + inside_dist

        surviving, culled = cull_particles_by_sdf(
            particles=particles,
            sdf_fn=box_sdf,
            margin=0.5,
            strict_nodes=True,
        )

        assert len(surviving) > 0
        assert len(culled) > 0
        assert len(surviving) + len(culled) == 25
        # The central particle at (2, 2, 0) should definitely survive
        central_p = [p for p in surviving if np.allclose(p.center, box_center, atol=1.0)]
        assert len(central_p) == 1

    def test_identify_boundary_particles_with_catenation_offsets(self):
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(3, 3, 3), pitch=12.7)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )
        # Assign cell_indices to particles
        for p, idx in zip(particles, indices):
            p.cell_index = tuple(idx.tolist())

        boundary_parts = identify_boundary_particles(particles, cell=cell)
        # In a 3x3x3 cube, only the central site (1, 1, 1) has all 6 Simple Cubic neighbors.
        # So 27 - 1 = 26 particles are boundary particles!
        assert len(boundary_parts) == 26
        # The 1 interior particle should have is_boundary == False
        interior = [p for p in particles if not p.metadata.get("is_boundary", False)]
        assert len(interior) == 1
        assert interior[0].cell_index == (1, 1, 1)

    def test_solid_perimeter_frame_welding(self):
        cell = European4in1Cell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(4, 4, 1), pitch=8.0)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=8.0,
        )

        # Build solid box frame
        frame_mesh = build_perimeter_frame_solid(
            particles=particles,
            wall_thickness=2.0,
            margin=0.5,
            z_padding=1.0,
            frame_shape="box",
        )

        assert isinstance(frame_mesh, trimesh.Trimesh)
        assert frame_mesh.is_watertight
        assert frame_mesh.volume > 0.0

        # Verify frame outer bounds strictly exceed particle bounds
        part_nodes = np.vstack([p.global_nodes() for p in particles])
        p_min = np.min(part_nodes, axis=0)
        p_max = np.max(part_nodes, axis=0)
        f_min, f_max = frame_mesh.bounds

        assert f_min[0] < p_min[0]
        assert f_max[0] > p_max[0]
        assert f_min[1] < p_min[1]
        assert f_max[1] > p_max[1]

    def test_cylinder_perimeter_frame(self):
        cyl = seed_cylindrical_wrap(radius=20.0, height=20.0, target_pitch=8.0)
        cell = European4in1Cell()
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=cyl["centers"],
            frames=cyl["frames"],
            grid_indices=cyl["indices"],
            cell_pitch=cyl["quantized_pitch_theta"],
        )

        frame_mesh = build_perimeter_frame_solid(
            particles=particles,
            wall_thickness=3.0,
            margin=1.0,
            frame_shape="cylinder",
            radius=20.0,
        )

        assert isinstance(frame_mesh, trimesh.Trimesh)
        assert frame_mesh.is_watertight
        assert frame_mesh.volume > 0.0
