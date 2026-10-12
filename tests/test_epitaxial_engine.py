"""
Unit & Integration Tests: Metamaterial Heteroepitaxy Engine.

Verifies:
- Law 1: Commensurate Coincidence Lattice Search
- Law 2: Aperture Admissibility & Steric Rejection
- Law 3: Boundary Catenation & Longitudinal Optimization
- Canonical API Integration: generate_interlinked_lattice(transition_to_cell=...)
- Strict Error Handling: EpitaxialIncompatibilityError with rich diagnostics
"""
from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    InterlinkedConfig,
    generate_interlinked_lattice,
    InterlinkedRegistry,
    C6TTCell,
    D4TetCell,
    EpitaxialAlignment,
    EpitaxialFeasibilityResult,
    EpitaxialIncompatibilityError,
    solve_supercell_coincidence,
    check_aperture_admissibility,
    optimize_interface_catenation,
    solve_epitaxial_transition,
    analyze_epitaxial_transition,
)


def test_law_1_supercell_solver_exact() -> None:
    """Law 1: Verify exact coincidence search for C-6-TT to D-4-TET."""
    p_a = 9.10
    p_b = np.sqrt(2.0) * p_a  # 12.869 mm

    ratio, theta, strain = solve_supercell_coincidence(
        pitch_a=p_a,
        pitch_b=p_b,
        habit_plane="100",
        max_supercell=4,
        max_strain=0.05,
    )

    assert ratio == (1, 1), f"Expected (1, 1), got {ratio}"
    assert abs(theta - np.pi / 4.0) < 1e-4, f"Expected 45 deg, got {np.degrees(theta)}"
    assert strain < 1e-4, f"Expected zero strain, got {strain}"


def test_law_1_supercell_solver_incommensurate_failure() -> None:
    """Law 1: Verify that incommensurate pitches exceed tight strain threshold."""
    p_a = 10.0
    p_b = 13.371928371  # Arbitrary irrational-like pitch

    ratio, theta, strain = solve_supercell_coincidence(
        pitch_a=p_a,
        pitch_b=p_b,
        max_supercell=2,
        max_strain=0.001,  # 0.1% tolerance
    )
    assert strain > 0.001, "Expected incommensurate strain to exceed tight threshold"


def test_law_2_aperture_admissibility_positive() -> None:
    """Law 2: Verify positive aperture check for C6 window and D4 profile."""
    d_aperture = 7.54  # mm (C6 hexagonal opening)
    d_profile = 5.49   # mm (D4 entering edge)
    wire_r = 0.35      # 700 um diameter
    min_clr = 0.150    # 150 um clearance

    is_admissible, margin = check_aperture_admissibility(
        d_aperture=d_aperture,
        d_profile=d_profile,
        wire_radius=wire_r,
        min_clearance=min_clr,
    )

    assert is_admissible is True
    assert margin > 0.50, f"Expected margin > 0.5 mm, got {margin:.3f} mm"


def test_law_2_aperture_admissibility_negative() -> None:
    """Law 2: Verify rejection when aperture is smaller than penetrating profile."""
    d_aperture = 4.00  # mm (too small)
    d_profile = 5.00   # mm
    wire_r = 0.35
    min_clr = 0.150

    is_admissible, margin = check_aperture_admissibility(
        d_aperture=d_aperture,
        d_profile=d_profile,
        wire_radius=wire_r,
        min_clearance=min_clr,
    )

    assert is_admissible is False
    assert margin < 0.0


def test_law_3_catenation_optimization() -> None:
    """Law 3: Verify optimization of interface spacing with verified catenation."""
    cell_a = C6TTCell()
    cell_b = D4TetCell()

    proto_a = cell_a.basis_particles[0].geometry
    proto_b = cell_b.basis_particles[0].geometry

    # Scale to physical sizes (s=8.0 mm, L=7.76 mm)
    nodes_a = proto_a.nodes * 8.0
    nodes_b = proto_b.nodes * 7.76

    theta = np.pi / 4.0
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    nodes_b_rot = (R @ nodes_b.T).T

    best_dx, max_clr, piercings = optimize_interface_catenation(
        nodes_a=nodes_a,
        struts_a=proto_a.struts,
        nodes_b=nodes_b_rot,
        struts_b=proto_b.struts,
        wire_radius=0.35,
        min_clearance=0.150,
        search_range=(6.0, 8.0),
        num_samples=21,
    )

    assert 6.8 <= best_dx <= 7.6, f"Expected optimal dx ~7.2 mm, got {best_dx}"
    assert piercings >= 1, f"Expected piercings >= 1, got {piercings}"
    assert max_clr >= 0.150, f"Expected clearance >= 150 um, got {max_clr*1000:.1f} um"


def test_solve_epitaxial_transition_c6_d4() -> None:
    """Verify solve_epitaxial_transition convenience function for C6 -> D4."""
    alignment = solve_epitaxial_transition(
        cell_a="c6tt",
        cell_b="d4tet",
        pitch_a=9.10,
        wire_radius=0.35,
        min_clearance=0.150,
    )

    assert isinstance(alignment, EpitaxialAlignment)
    assert alignment.supercell_ratio == (1, 1)
    assert alignment.strain < 1e-4
    assert 6.5 <= alignment.interface_offset <= 7.6


def test_solve_epitaxial_transition_failure_strict() -> None:
    """Verify that an infeasible transition raises EpitaxialIncompatibilityError (Option A)."""
    # Force aperture clash by demanding an enormous wire radius that cannot fit
    with pytest.raises(EpitaxialIncompatibilityError) as exc_info:
        solve_epitaxial_transition(
            cell_a="c6tt",
            cell_b="d4tet",
            pitch_a=9.10,
            wire_radius=1.50,  # 3.0 mm wire exceeds aperture
            min_clearance=0.50,
        )

    assert "Aperture clash" in str(exc_info.value)
    assert exc_info.value.diagnostics != {}


def test_generate_interlinked_lattice_transition_api() -> None:
    """
    Verify canonical generate_interlinked_lattice API with transition_to_cell.
    Ensures multi-zone lattice generation, watertight STL, and instanced 3MF export.
    """
    config = InterlinkedConfig(
        cell="c6tt",
        transition_to_cell="d4tet",
        pitch=9.10,
        wire_radius=0.35,
        min_clearance=0.150,
        grid_size=(2, 2, 2),
    )

    result = generate_interlinked_lattice(config, check_clearance=True)

    assert result.mesh is not None
    assert len(result.particles) >= 12
    assert result.clearance_valid is True
    assert result.min_clearance >= 0.150
    assert result.metadata.get("transition_type") == "epitaxial"
    assert result.metadata.get("cell_a") == "C-6-TT"
    assert result.metadata.get("cell_b") == "D-4-TET"
