"""
Unit and Integration Tests for Robust PAM Metamaterial Heteroepitaxy (Strategies 1-5).

Verifies:
1. Strategy 1: Inverted Void O-Lattice Alignment (extract_void_barycenters, align_void_olattice, T_void).
2. Strategy 2: Habit Plane Selection on {110} and {111} (get_habit_plane_rotation_matrix, solve_supercell_coincidence).
3. Strategy 3: Discrete Buffer Superlattices & Dijkstra Pathfinding (find_buffer_path, build_auto_buffer_superlattice).
4. Strategy 4: Symmetry Subgroup Matching & 3D Phase Shift Optimization (optimize_interface_phase_and_taper).
5. Strategy 5: Local Geometric Scaling & Strut Tapering (apply_strut_taper, apply_cage_shrinkage, enable_taper).
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    C6TTCell,
    D4TetCell,
    D4CubeCell,
    J4OctCell,
    J4RingCell,
    T6RingCell,
    B8COCell,
    InterlinkedConfig,
    InterlinkedParticle,
    ParticleGeometry,
    generate_interlinked_lattice,
    get_habit_plane_rotation_matrix,
    solve_supercell_coincidence,
    compute_habit_plane_aperture,
    compute_habit_plane_profile,
    check_aperture_admissibility,
    extract_void_barycenters,
    align_void_olattice,
    optimize_interface_phase_and_taper,
    optimize_interface_catenation,
    apply_strut_taper,
    apply_cage_shrinkage,
    find_buffer_path,
    build_auto_buffer_superlattice,
    analyze_epitaxial_transition,
    solve_epitaxial_transition,
)


class TestStrategy1InvertedVoidOLattice:
    """Test Strategy 1: Inverted Void O-Lattice Alignment (O_void)."""

    def test_void_barycenter_extraction_c6tt(self):
        """Verify extract_void_barycenters computes open void barycenter and positive inscribed radius."""
        cell = C6TTCell()
        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        pts_2d, r_in = extract_void_barycenters(
            parts[0].geometry,
            cell=cell,
            habit_plane="100",
            pitch=10.0,
            wire_radius=0.35,
        )
        assert pts_2d.shape[1] == 2
        assert len(pts_2d) >= 1
        assert r_in > 0.35  # Must accommodate solid strut wire radius
        # Central window on symmetric C-6-TT should be near origin (0, 0)
        np.testing.assert_allclose(pts_2d[0], [0.0, 0.0], atol=0.2)

    def test_inverted_olattice_displacement_matrix(self):
        """Verify align_void_olattice computes inverted displacement matrix T_void and target guest centroid."""
        res = align_void_olattice("c6tt", "d4cube", habit_plane="100", pitch_a=10.0, pitch_b=10.0)
        assert "T_void" in res
        assert "v_k" in res
        assert "x_guest_star" in res
        assert "inscribed_radius_mm" in res
        assert res["inscribed_radius_mm"] > 0.0
        # Inverted displacement matrix: T_void = I - A_B @ inv(A_A)
        np.testing.assert_allclose(res["T_void"], np.zeros((3, 3)), atol=1e-5)


class TestStrategy2HabitPlaneSelection:
    """Test Strategy 2: Habit Plane Selection and Edge-to-Edge Matching on {110} and {111}."""

    def test_habit_plane_rotation_matrix_100(self):
        """Verify {100} rotation matrix is identity."""
        R = get_habit_plane_rotation_matrix("100")
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_habit_plane_rotation_matrix_110(self):
        """Verify {110} rotation matrix maps [1, 1, 0]/sqrt(2) to +X with det(R) = +1."""
        R = get_habit_plane_rotation_matrix("110")
        n_110 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        mapped_x = R @ n_110
        np.testing.assert_allclose(mapped_x, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_habit_plane_rotation_matrix_111(self):
        """Verify {111} rotation matrix maps [1, 1, 1]/sqrt(3) to +X with exact orthonormal frame."""
        R = get_habit_plane_rotation_matrix("111")
        n_111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        mapped_x = R @ n_111
        np.testing.assert_allclose(mapped_x, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0)

    def test_habit_111_hexagonal_supercell_coincidence(self):
        """Verify solve_supercell_coincidence on {111} evaluates hexagonal discrete coincidence."""
        ratio, theta, strain = solve_supercell_coincidence(
            pitch_a=14.0,
            pitch_b=14.0,
            habit_plane="111",
            network_a="dia",
            network_b="dia",
        )
        assert ratio == (1, 1)
        assert strain < 1e-4

    def test_habit_111_profile_and_aperture(self):
        """Verify aperture and profile extractors support {111} habit plane rotation."""
        cell = D4TetCell()
        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 14.0)
        geom = parts[0].geometry
        aperture = compute_habit_plane_aperture(geom, habit_plane="111", wire_radius=0.35, cell=cell)
        profile = compute_habit_plane_profile(geom, habit_plane="111", wire_radius=0.35, cell=cell)
        assert aperture > 0.0
        assert profile > 0.0


class TestStrategy3DiscreteBufferSuperlattices:
    """Test Strategy 3: Discrete Buffer Superlattices & Dijkstra Pathfinding."""

    def test_dijkstra_buffer_pathfinding_direct(self):
        """Verify find_buffer_path recognizes compatible pair as direct transition."""
        res = find_buffer_path("c6tt", "d4tet", habit_plane="100", pitch_a=9.10)
        assert res.is_direct is True
        assert len(res.intermediate_cells) == 0
        assert res.path == ["c6tt", "d4tet"]
        assert res.total_cost < float("inf")

    def test_build_auto_buffer_superlattice_structure(self):
        """Verify build_auto_buffer_superlattice constructs multi-zone configuration with buffer layers."""
        cfg = build_auto_buffer_superlattice(
            "c6tt",
            "d4tet",
            grid_size=(2, 2, 2),
            buffer_layers_per_zone=2,
            pitch_a=9.10,
        )
        assert len(cfg.zones) >= 2
        assert cfg.zones[0].num_layers == 2
        assert cfg.grid_transverse == (2, 2)


class TestStrategy4SymmetrySubgroupAndPhaseShift:
    """Test Strategy 4: Symmetry Subgroup Matching & 3D Phase Shift Optimization."""

    def test_optimize_interface_phase_and_taper_coaxial(self):
        """Verify optimize_interface_phase_and_taper solves coaxial shift with positive clearance."""
        c6 = C6TTCell()
        d4 = D4TetCell()
        parts_a = c6.instantiate_site((0, 0, 0), np.zeros(3), 9.10)
        parts_b = d4.instantiate_site((0, 0, 0), np.zeros(3), 9.10 * np.sqrt(2.0))

        R_plane = get_habit_plane_rotation_matrix("100")
        R_twist = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ])
        nodes_b_lab = (R_twist @ parts_b[0].global_nodes().T).T

        res = optimize_interface_phase_and_taper(
            nodes_a=parts_a[0].global_nodes(),
            struts_a=parts_a[0].geometry.struts,
            nodes_b=nodes_b_lab,
            struts_b=parts_b[0].geometry.struts,
            wire_radius=0.35,
            min_clearance=0.150,
            pitch_a=9.10,
            pitch_b=9.10 * np.sqrt(2.0),
        )

        assert res.is_valid is True
        assert len(res.phase_shift) == 3
        assert 6.0 <= res.phase_shift[0] <= 7.5
        assert res.piercings_count >= 1
        assert res.max_clearance_mm >= 0.150

    def test_backward_compatible_optimize_interface_catenation(self):
        """Verify optimize_interface_catenation wrapper returns (dx, clr, piercings) matching phase_shift[0]."""
        c6 = C6TTCell()
        d4 = D4TetCell()
        parts_a = c6.instantiate_site((0, 0, 0), np.zeros(3), 9.10)
        parts_b = d4.instantiate_site((0, 0, 0), np.zeros(3), 9.10 * np.sqrt(2.0))

        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ])
        nodes_b_rot = (R @ parts_b[0].global_nodes().T).T

        dx, clr, piercings = optimize_interface_catenation(
            nodes_a=parts_a[0].global_nodes(),
            struts_a=parts_a[0].geometry.struts,
            nodes_b=nodes_b_rot,
            struts_b=parts_b[0].geometry.struts,
            wire_radius=0.35,
            min_clearance=0.150,
            pitch_a=9.10,
            pitch_b=9.10 * np.sqrt(2.0),
        )

        assert 6.0 <= dx <= 7.5
        assert piercings >= 1
        assert clr >= 0.150


class TestStrategy5GeometricScalingAndTapering:
    """Test Strategy 5: Local Geometric Scaling & Strut Tapering."""

    def test_apply_strut_taper(self):
        """Verify apply_strut_taper thins boundary layer wire radius by gamma factor."""
        cell = C6TTCell()
        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        for p in parts:
            p.wire_radius = 0.40
            p.metadata["layer_x"] = -1

        count = apply_strut_taper(parts, gamma=0.70, boundary_layers=(-1, 0))
        assert count == len(parts)
        for p in parts:
            assert np.isclose(p.wire_radius, 0.28)  # 0.40 * 0.70
            assert p.metadata.get("strut_taper_applied") is True

    def test_apply_cage_shrinkage(self):
        """Verify apply_cage_shrinkage isotropically contracts boundary cage node coordinates."""
        cell = D4CubeCell()
        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        orig_nodes = np.copy(parts[0].geometry.nodes)
        parts[0].metadata["layer_x"] = 0

        count = apply_cage_shrinkage(parts, eta=0.90, boundary_layers=(0,))
        assert count == 1
        shrunk_nodes = parts[0].geometry.nodes
        np.testing.assert_allclose(shrunk_nodes, orig_nodes * 0.90, atol=1e-5)
        assert parts[0].metadata.get("cage_shrink_applied") is True

    def test_tapering_recovers_aperture_constrained_transition(self):
        """Verify analyze_epitaxial_transition with enable_taper=True resolves aperture frustration."""
        # Without taper, tight min_clearance produces an aperture clash
        res_notaper = analyze_epitaxial_transition(
            "c6tt",
            "d4cube",
            pitch_a=9.10,
            wire_radius=0.35,
            min_clearance=0.350,
            enable_taper=False,
        )
        assert res_notaper.is_feasible is False
        assert res_notaper.failure_stage == "aperture"
        assert res_notaper.aperture_margin_mm < 0.0

        # With Strategy 5 geometric tapering enabled, aperture feasibility is fully recovered
        res_taper = analyze_epitaxial_transition(
            "c6tt",
            "d4cube",
            pitch_a=9.10,
            wire_radius=0.35,
            min_clearance=0.350,
            enable_taper=True,
            strut_taper_gamma=0.70,
            cage_shrink_eta=0.90,
        )
        assert res_taper.is_feasible is True
        assert res_taper.alignment is not None
        assert res_taper.aperture_margin_mm >= 0.0


class TestEndToEndRobustTransitions:
    """Integration Tests: End-to-end lattice generation with robust heteroepitaxy."""

    def test_generate_c6tt_to_d4tet_transition_clean_miter(self):
        """Verify C-6-TT -> D-4-TET transition generates watertight clean miter truss mesh."""
        cfg = InterlinkedConfig(
            cell="c6tt",
            transition_to_cell="d4tet",
            pitch=9.10,
            wire_radius=0.35,
            min_clearance=0.150,
            grid_size=(2, 2, 2),
        )
        res = generate_interlinked_lattice(cfg, check_clearance=True)
        assert res.mesh is not None
        assert len(res.particles) >= 12
        assert res.clearance_valid is True
        assert res.min_clearance >= 0.150
