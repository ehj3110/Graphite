"""
Unit tests for Two-Tier Vectorized Clearance & Inversion Engine (Phase 4):
- circle_circle_distance (continuous 3D circle-to-circle distance)
- particle_pair_centerline_distance & particle_pair_clearance
- compute_pairwise_particle_clearances (O(N log N) KD-tree broad phase + vectorized narrow phase)
- check_particle_clearance (validation and collision detection)
- particle_linking_number (Gauss linking integral for particles)
- resolve_lattice_pitch & _numerical_resolve_pitch (polymorphic inverse pitch solver)
- calibrate_cell_clearance_curve (clearance law slope & linearity characterization)
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    C6TTCell,
    D4TetCell,
    European4in1Cell,
    InterlinkedRegistry,
    InterlinkedParticle,
    seed_cartesian_lattice,
    instantiate_lattice_on_sites,
    circle_circle_distance,
    particle_pair_centerline_distance,
    particle_pair_clearance,
    compute_pairwise_particle_clearances,
    check_particle_clearance,
    particle_linking_number,
    resolve_lattice_pitch,
    calibrate_cell_clearance_curve,
)


class TestCircleCircleDistance:
    def test_coplanar_concentric_circles(self):
        # Two concentric circles in XY plane: R1 = 10, R2 = 6 -> min distance is 4.0 mm
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([0.0, 0.0, 0.0])
        n2 = np.array([0.0, 0.0, 1.0])

        d = circle_circle_distance(c1, n1, 10.0, c2, n2, 6.0)
        np.testing.assert_allclose(d, 4.0, atol=1e-3)

    def test_parallel_offset_circles(self):
        # Two identical circles R=5, separated by delta_z = 4 along Z axis -> min dist = 4.0 mm
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([0.0, 0.0, 4.0])
        n2 = np.array([0.0, 0.0, 1.0])

        d = circle_circle_distance(c1, n1, 5.0, c2, n2, 5.0)
        np.testing.assert_allclose(d, 4.0, atol=1e-3)

    def test_orthogonal_linked_circles(self):
        # Circle 1 in XY plane at origin, R=5
        # Circle 2 in XZ plane centered at [5, 0, 0], R=5
        # Analytically, min distance is 5.0 mm (when circle 1 passes through center of circle 2)
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([5.0, 0.0, 0.0])
        n2 = np.array([0.0, 1.0, 0.0])

        d = circle_circle_distance(c1, n1, 5.0, c2, n2, 5.0)
        np.testing.assert_allclose(d, 5.0, atol=1e-3)

    def test_orthogonal_intersecting_circles(self):
        # Both circles centered at origin, R=5
        # Intersect at [5, 0, 0] and [-5, 0, 0] -> min distance = 0.0 mm
        c1 = np.array([0.0, 0.0, 0.0])
        n1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([0.0, 0.0, 0.0])
        n2 = np.array([0.0, 1.0, 0.0])

        d = circle_circle_distance(c1, n1, 5.0, c2, n2, 5.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-3)


class TestParticlePairClearance:
    def test_c6tt_face_to_face_clearance(self):
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(2, 1, 1), pitch=12.7)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )
        p0, p1 = particles[0], particles[1]

        # Strut radius = 0.625 mm (1.25 mm diameter)
        r = 0.625
        d_cl = particle_pair_centerline_distance(p0, p1)
        clr = particle_pair_clearance(p0, p1, strut_radius=r)

        # Centerline distance must be greater than 2 * r
        assert d_cl > 2.0 * r
        # Delta = d_centerline - 2r ≈ 0.418 mm
        np.testing.assert_allclose(clr, d_cl - 2.0 * r, atol=1e-6)
        assert clr > 0.30


class TestTwoTierClearanceBroadAndNarrow:
    def test_broad_phase_kd_tree_filtering(self):
        # 3x3x3 = 27 C-6-TT cages at 12.7 mm pitch
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(3, 3, 3), pitch=12.7)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )
        assert len(particles) == 27

        # Total possible pairs is 27 * 26 / 2 = 351
        # With KD-tree broad phase, non-neighboring cages (e.g. corner (0,0,0) and (2,2,2)) must be rejected
        min_clr, records = compute_pairwise_particle_clearances(
            particles=particles,
            strut_radius=0.50,
            broadphase_margin=0.5,
        )

        # Only interacting neighbor pairs should be in records (far fewer than 351)
        assert len(records) < 351
        assert len(records) > 0
        assert min_clr > 0.30

    def test_check_particle_clearance_valid_vs_colliding(self):
        cell = C6TTCell()
        centers, frames, indices = seed_cartesian_lattice(repeats=(2, 2, 1), pitch=12.7)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=12.7,
        )

        # 1. Valid clearance: strut_radius = 0.40 mm (0.80 mm diameter)
        is_valid, min_clr, violations = check_particle_clearance(
            particles=particles,
            strut_radius=0.40,
            min_clearance=0.30,
        )
        assert is_valid is True
        assert len(violations) == 0
        assert min_clr >= 0.30

        # 2. Forced collision: strut_radius = 1.20 mm (2.40 mm diameter, exceeds aperture)
        is_valid_bad, min_clr_bad, violations_bad = check_particle_clearance(
            particles=particles,
            strut_radius=1.20,
            min_clearance=0.30,
        )
        assert is_valid_bad is False
        assert len(violations_bad) > 0
        assert min_clr_bad < 0.0  # Penetration / overlap


class TestParticleLinkingNumber:
    def test_linked_rings_gauss_number(self):
        # Two linked rings (Hopf link)
        cell = European4in1Cell(radius_ratio=0.65, tilt_angle_deg=28.0)
        centers, frames, indices = seed_cartesian_lattice(repeats=(2, 1, 1), pitch=8.0)
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=8.0,
        )
        # European 4-in-1 adjacent rings loop through each other
        lk = particle_linking_number(particles[0], particles[1])
        # Linking number magnitude must round to 1
        assert abs(round(lk)) == 1

    def test_unlinked_separated_rings(self):
        cell = European4in1Cell(radius_ratio=0.65, tilt_angle_deg=28.0)
        # Separate rings by 50 mm along X (far apart, completely unlinked)
        centers = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        frames = np.tile(np.eye(3), (2, 1, 1))
        indices = np.array([[0, 0, 0], [10, 0, 0]])
        particles = instantiate_lattice_on_sites(
            cell=cell,
            centers=centers,
            frames=frames,
            grid_indices=indices,
            cell_pitch=8.0,
        )
        lk = particle_linking_number(particles[0], particles[1])
        assert round(lk) == 0


class TestPitchInversion:
    @pytest.mark.parametrize("cell_cls, pitch_param", [
        (C6TTCell, 12.7),
        (D4TetCell, 16.0),
        (European4in1Cell, 10.0),
    ])
    def test_polymorphic_pitch_inversion_roundtrip(self, cell_cls, pitch_param):
        cell = cell_cls()
        strut_d = 0.80  # mm
        target_clr = 0.45  # mm target gap

        # Invert pitch to find required cell spacing
        req_pitch = resolve_lattice_pitch(
            cell=cell,
            target_clearance=target_clr,
            strut_diameter=strut_d,
        )

        assert req_pitch > 0.0
        # Forward check: forward_clearance at req_pitch must yield target_clr
        calc_clr = cell.forward_clearance(unit_cell_pitch=req_pitch, strut_diameter=strut_d)
        np.testing.assert_allclose(calc_clr, target_clr, atol=1e-4)


class TestClearanceCalibration:
    def test_c6tt_clearance_curve_linearity(self):
        cell = C6TTCell()
        curve = calibrate_cell_clearance_curve(
            cell=cell,
            strut_diameter=1.0,
            pitch_range=(10.0, 20.0),
            num_points=10,
        )
        # C-6-TT scales linearly with pitch: Delta = 0.13137 * a_0 - 1.0
        np.testing.assert_allclose(curve["kappa"], 0.13137, rtol=1e-3)
        np.testing.assert_allclose(curve["intercept_mm"], -1.0, atol=1e-6)
        assert curve["r_squared"] > 0.9999
        assert curve["is_linear"] is True

    def test_d4tet_clearance_curve_linearity(self):
        cell = D4TetCell()
        curve = calibrate_cell_clearance_curve(
            cell=cell,
            strut_diameter=0.75,
            pitch_range=(10.0, 25.0),
            num_points=10,
        )
        # D-4-TET: Delta = 0.1018 * a_conv - 0.75
        np.testing.assert_allclose(curve["kappa"], 0.1018, rtol=1e-3)
        np.testing.assert_allclose(curve["intercept_mm"], -0.75, atol=1e-6)
        assert curve["r_squared"] > 0.9999
        assert curve["is_linear"] is True
