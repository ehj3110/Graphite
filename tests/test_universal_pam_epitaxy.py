"""
Unit and Integration Tests for Universal PAM Discrete Heteroepitaxy (Phases 0-3).

Verifies:
1. Phase 0: Crystallographic Body-Centered Cubic (BCC) Seeder and Universal Zone Dispatcher.
2. Phase 1: Universal Law 1 Commensurate Coincidence Solver with parent networks.
3. Phase 2: Universal Law 2 Aperture and Profile Extractors across all cell families.
4. Phase 3: Universal Law 3 Polygon Face Extraction, Gauss Linking, and Dynamic Catenation Optimization.
"""
from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    seed_bcc_lattice,
    seed_zone_lattice,
    seed_diamond_lattice,
    seed_cartesian_lattice,
    solve_supercell_coincidence,
    compute_habit_plane_aperture,
    compute_habit_plane_profile,
    check_aperture_admissibility,
    extract_triangulated_faces,
    curve_curve_linking_number,
    optimize_interface_catenation,
    analyze_epitaxial_transition,
    solve_epitaxial_transition,
    InterlinkedRegistry,
    C6TTCell,
    D4TetCell,
    J4OctCell,
    J4RingCell,
    D4CubeCell,
    D4CTCell,
    D4IcoCell,
    T6RingCell,
    T6HexCell,
    C6COCell,
    B8COCell,
)


class TestPhase0BCCAndZoneSeeder:
    """Test Phase 0: BCC Seeder and Universal Zone Seeder."""

    def test_seed_bcc_lattice_counts_and_sublattices(self):
        """Verify BCC seeder generates correct corner (A) and body-center (B) sites."""
        repeats = (2, 2, 2)
        cell_size = 14.0
        centers, frames, indices, sublattices = seed_bcc_lattice(repeats, cell_size=cell_size)

        # 2x2x2 unit cells * 2 sites per unit cell = 16 sites
        assert len(centers) == 16
        assert len(frames) == 16
        assert len(indices) == 16
        assert len(sublattices) == 16
        assert sublattices.count("A") == 8
        assert sublattices.count("B") == 8

        # Check body center offset: (0.5, 0.5, 0.5) * a
        p_corner = centers[0]  # [0, 0, 0]
        p_body = centers[1]    # [7, 7, 7]
        np.testing.assert_allclose(p_body - p_corner, [7.0, 7.0, 7.0])

    def test_seed_zone_lattice_dispatch(self):
        """Verify seed_zone_lattice dispatches correctly based on parent network."""
        # 1. PCU
        c_pcu, f_pcu, idx_pcu, subs_pcu = seed_zone_lattice("pcu", (2, 2, 2), 10.0)
        assert len(c_pcu) == 8
        assert subs_pcu is None

        # 2. DIA
        c_dia, f_dia, idx_dia, subs_dia = seed_zone_lattice("dia", (1, 1, 1), 14.0)
        assert len(c_dia) == 8
        assert subs_dia is not None
        assert len(subs_dia) == 8

        # 3. BCC
        c_bcc, f_bcc, idx_bcc, subs_bcc = seed_zone_lattice("bcc", (2, 2, 2), 14.0)
        assert len(c_bcc) == 16
        assert subs_bcc is not None
        assert subs_bcc.count("A") == 8
        assert subs_bcc.count("B") == 8


class TestPhase1Law1UniversalCoincidence:
    """Test Phase 1: Universal 2D Habit Plane Coincidence Solver."""

    def test_pcu_pcu_coincidence(self):
        """pcu <-> pcu cells on {100} are identical square grids: theta=0, ratio=(1,1)."""
        ratio, theta, strain = solve_supercell_coincidence(
            pitch_a=10.0,
            pitch_b=10.0,
            network_a="pcu",
            network_b="pcu",
        )
        assert ratio == (1, 1)
        assert abs(theta) < 1e-4
        assert strain < 1e-5

    def test_pcu_dia_coincidence(self):
        """pcu <-> dia cells on {100} match at 45 deg with pitch_b = sqrt(2) * pitch_a."""
        p_a = 10.0
        p_b = np.sqrt(2.0) * p_a
        ratio, theta, strain = solve_supercell_coincidence(
            pitch_a=p_a,
            pitch_b=p_b,
            network_a="pcu",
            network_b="dia",
        )
        assert ratio == (1, 1)
        assert abs(theta - np.pi / 4.0) < 1e-4
        assert strain < 1e-5

    def test_dia_dia_coincidence(self):
        """dia <-> dia cells on {100} share orientation: theta=0, ratio=(1,1)."""
        ratio, theta, strain = solve_supercell_coincidence(
            pitch_a=14.0,
            pitch_b=14.0,
            network_a="dia",
            network_b="dia",
        )
        assert ratio == (1, 1)
        assert abs(theta) < 1e-4
        assert strain < 1e-5

    def test_pcu_bcc_coincidence(self):
        """pcu <-> bcc cells on {100} have matching square lattice vectors: theta=0, ratio=(1,1)."""
        ratio, theta, strain = solve_supercell_coincidence(
            pitch_a=12.0,
            pitch_b=12.0,
            network_a="pcu",
            network_b="bcc",
        )
        assert ratio == (1, 1)
        assert abs(theta) < 1e-4
        assert strain < 1e-5


class TestPhase2Law2UniversalApertureAndProfile:
    """Test Phase 2: Universal Aperture and Profile Extractors."""

    @pytest.mark.parametrize(
        "cell_cls",
        [
            C6TTCell,
            D4TetCell,
            J4OctCell,
            J4RingCell,
            D4CubeCell,
            D4CTCell,
            D4IcoCell,
            T6RingCell,
            T6HexCell,
            C6COCell,
            B8COCell,
        ],
    )
    def test_all_11_cells_aperture_and_profile_positive(self, cell_cls):
        """Verify that every one of the 11 PAM unit cells yields positive aperture and profile."""
        cell = cell_cls()
        pitch = 14.0
        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), pitch)
        assert len(parts) >= 1
        geom = parts[0].geometry

        d_ap = compute_habit_plane_aperture(geom, habit_plane="100", wire_radius=0.35, cell=cell)
        d_prof = compute_habit_plane_profile(geom, habit_plane="100", wire_radius=0.35, cell=cell)

        assert np.isfinite(d_ap) and d_ap > 0.0, f"{cell.name} aperture non-positive: {d_ap}"
        assert np.isfinite(d_prof) and d_prof > 0.0, f"{cell.name} profile non-positive: {d_prof}"

    def test_c6tt_and_d4tet_exact_analytical_match(self):
        """Verify C-6-TT aperture and D-4-TET profile match published analytical benchmarks."""
        c6 = C6TTCell()
        d4 = D4TetCell()

        p_c6 = c6.instantiate_site((0, 0, 0), np.zeros(3), 9.10)[0].geometry
        p_d4 = d4.instantiate_site((0, 0, 0), np.zeros(3), 12.869)[0].geometry

        d_ap = compute_habit_plane_aperture(p_c6, habit_plane="100", wire_radius=0.35, cell=c6)
        d_prof = compute_habit_plane_profile(p_d4, habit_plane="100", wire_radius=0.35, cell=d4)

        assert abs(d_ap - 7.54) < 0.02
        assert abs(d_prof - 5.49) < 0.02


class TestPhase3Law3CatenationAndFaceExtraction:
    """Test Phase 3: Polygon Face Extraction and Dual-Mode Catenation."""

    def test_polyhedral_face_extraction(self):
        """Verify extract_triangulated_faces extracts correct cycles for tetrahedron, octahedron, and cube."""
        c_tet = D4TetCell().instantiate_site((0, 0, 0), np.zeros(3), 14.0)[0].geometry
        faces_tet = extract_triangulated_faces(c_tet.nodes, c_tet.struts)
        assert len(faces_tet) == 4, f"Tetrahedron should have 4 triangular faces, got {len(faces_tet)}"

        c_oct = J4OctCell().instantiate_site((0, 0, 0), np.zeros(3), 14.0)[0].geometry
        faces_oct = extract_triangulated_faces(c_oct.nodes, c_oct.struts)
        assert len(faces_oct) == 8, f"Octahedron should have 8 triangular faces, got {len(faces_oct)}"

        c_cube = D4CubeCell().instantiate_site((0, 0, 0), np.zeros(3), 14.0)[0].geometry
        faces_cube = extract_triangulated_faces(c_cube.nodes, c_cube.struts)
        # 6 square faces * 2 triangles per square = 12 triangles
        assert len(faces_cube) == 12, f"Cube should have 12 triangulated faces, got {len(faces_cube)}"

    def test_ring_curve_curve_gauss_linking(self):
        """Verify curve_curve_linking_number computes |Lk| = 1 for linked rings, 0 for separated."""
        from graphite.explicit.interlinked import generate_ring

        r1 = generate_ring(center=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0], radius=3.0, wire_radius=0.3)
        r2_linked = generate_ring(center=[2.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0], radius=3.0, wire_radius=0.3)
        r3_unlinked = generate_ring(center=[10.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0], radius=3.0, wire_radius=0.3)

        p0 = r1.nodes[r1.struts[:, 0]]
        p1 = r1.nodes[r1.struts[:, 1]]

        # Linked
        q0_lk = r2_linked.nodes[r2_linked.struts[:, 0]]
        q1_lk = r2_linked.nodes[r2_linked.struts[:, 1]]
        lk_val = curve_curve_linking_number(p0, p1, q0_lk, q1_lk)
        assert abs(round(abs(lk_val)) - 1) < 1e-4

        # Unlinked
        q0_un = r3_unlinked.nodes[r3_unlinked.struts[:, 0]]
        q1_un = r3_unlinked.nodes[r3_unlinked.struts[:, 1]]
        un_val = curve_curve_linking_number(p0, p1, q0_un, q1_un)
        assert abs(un_val) < 0.1

    def test_dynamic_search_range_catenation(self):
        """Verify optimize_interface_catenation finds valid catenation with dynamic search range."""
        c6 = C6TTCell()
        d4 = D4TetCell()

        p_a = 9.10
        p_b = np.sqrt(2.0) * p_a
        parts_a = c6.instantiate_site((0, 0, 0), np.zeros(3), p_a)
        parts_b = d4.instantiate_site((0, 0, 0), np.zeros(3), p_b)

        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ])
        nodes_b_rot = (R @ parts_b[0].global_nodes().T).T

        best_dx, max_clr, piercings = optimize_interface_catenation(
            nodes_a=parts_a[0].global_nodes(),
            struts_a=parts_a[0].geometry.struts,
            nodes_b=nodes_b_rot,
            struts_b=parts_b[0].geometry.struts,
            wire_radius=0.35,
            min_clearance=0.150,
            pitch_a=p_a,
            pitch_b=p_b,
        )

        assert 6.5 <= best_dx <= 7.6
        assert piercings >= 1
        assert max_clr >= 0.150


class TestPhase4UniversalMultiZoneLatticeGeneration:
    """Test Phase 4: Universal Multi-Zone Transition and Superlattice Generation."""

    def test_generate_c6tt_to_d4tet_transition(self):
        """Verify standard C-6-TT -> D-4-TET transition generates valid multi-zone mesh."""
        from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

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
        assert res.metadata.get("transition_type") == "epitaxial"

    def test_generate_c6tt_to_d4cube_transition(self):
        """Verify C-6-TT -> D-4-CUBE transition generates valid multi-zone lattice."""
        from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

        cfg = InterlinkedConfig(
            cell="c6tt",
            transition_to_cell="d4cube",
            pitch=9.10,
            wire_radius=0.35,
            min_clearance=0.150,
            grid_size=(2, 2, 2),
        )
        res = generate_interlinked_lattice(cfg, check_clearance=False)
        assert res.mesh is not None
        assert len(res.particles) >= 12
        assert res.metadata.get("cell_a") == "C-6-TT"
        assert res.metadata.get("cell_b") == "D-4-CUBE"

    def test_generate_c6tt_to_j4ring_transition(self):
        """Verify C-6-TT -> J-4-RING transition generates valid multi-zone cage-ring hybrid."""
        from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

        cfg = InterlinkedConfig(
            cell="c6tt",
            transition_to_cell="j4ring",
            pitch=9.10,
            wire_radius=0.35,
            min_clearance=0.150,
            grid_size=(2, 2, 2),
        )
        res = generate_interlinked_lattice(cfg, check_clearance=False)
        assert res.mesh is not None
        assert len(res.particles) >= 12
        assert res.metadata.get("cell_a") == "C-6-TT"
        assert res.metadata.get("cell_b") == "J-4-RING"

    def test_generate_epitaxial_superlattice_multizone(self):
        """Verify generate_epitaxial_superlattice builds multi-zone buffer without hardcoded templates."""
        from graphite.explicit.interlinked import SuperlatticeConfig, LatticeZoneConfig, generate_epitaxial_superlattice

        cfg = SuperlatticeConfig(
            zones=[
                LatticeZoneConfig(cell="c6tt", num_layers=1, pitch=9.10),
                LatticeZoneConfig(cell="d4tet", num_layers=1, pitch=12.869),
            ],
            grid_transverse=(2, 2),
            wire_radius=0.35,
            min_clearance=0.100,
        )
        res = generate_epitaxial_superlattice(cfg, check_clearance=False)
        assert len(res.particles) >= 8
        assert res.mesh is not None
        assert len(res.mesh.vertices) > 0


class TestPhase5ExpandedCompatibilityMatrixAndClearance:
    """Test Phase 5: Expanded Cross-Family Compatibility Matrix and Ring Clearance Verification."""

    def test_solve_bcc_to_dia_epitaxial_transition(self):
        """Verify B-8-CO (BCC) -> D-4-TET (Diamond) heteroepitaxy solves with positive clearance."""
        res = solve_epitaxial_transition(
            "b8co",
            "d4tet",
            pitch_a=18.0,
            wire_radius=0.35,
            min_clearance=0.100,
        )
        assert res.supercell_ratio == (1, 1)
        assert res.metadata.get("piercings", 0) >= 1
        assert res.metadata.get("min_clearance_mm", 0.0) >= 0.100
        assert res.strain < 1e-4

    def test_solve_ring_ring_epitaxial_transition(self):
        """Verify T-6-RING (PCU) -> J-4-RING (Diamond) ring-ring heteroepitaxy solves with positive clearance."""
        res = solve_epitaxial_transition(
            "t6ring",
            "j4ring",
            pitch_a=28.0,
            wire_radius=0.35,
            min_clearance=0.100,
        )
        assert res.supercell_ratio == (1, 1)
        assert res.metadata.get("piercings", 0) >= 1
        assert res.metadata.get("min_clearance_mm", 0.0) >= 0.100
        assert res.interface_offset > 0.0

    def test_incompatible_transition_diagnostics(self):
        """Verify analyze_epitaxial_transition returns exact physical diagnostics on aperture clash."""
        diag = analyze_epitaxial_transition(
            "d4tet",
            "c6tt",
            pitch_a=14.0,
            pitch_b=14.0 / np.sqrt(2.0),
            wire_radius=0.35,
            min_clearance=0.150,
        )
        assert diag.is_feasible is False
        assert diag.failure_stage == "aperture"
        assert diag.aperture_margin_mm < 0.0
        assert "Aperture clash" in diag.diagnostics.get("error", "")

    def test_ring_lattice_clearance_measurement(self):
        """Verify measure_ring_lattice_clearance evaluates positive clearance and non-zero pairs on multi-cell ring lattices."""
        from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

        cfg = InterlinkedConfig(
            cell="j4ring",
            seeding_type="diamond",
            orientation="111",
            pitch=14.0,
            wire_radius=0.35,
            grid_size=(1, 1, 1),
            min_clearance=0.100,
        )
        res = generate_interlinked_lattice(cfg, check_clearance=True)
        assert res.clearance_valid is True
        assert res.min_clearance > 0.400
        assert "ring_clearance" in res.metadata
        rc = res.metadata["ring_clearance"]
        assert rc["global_min_clearance"] > 0.400
        assert rc["num_pairs_checked"] > 0
        assert rc["catenated_pairs"] > 0
        assert len(rc["violations"]) == 0


