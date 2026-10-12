"""Tests for analytical PAM grain boundaries, selection rules, and twin boundaries.

Validates:
1. Closed-form out-of-plane gap and inverted void shift formulas.
2. Topological habit admissibility selection rules (rejecting frustrated {100} transitions).
3. Coherent Sigma 3 {111} twin boundary matrix generation and mesh reflection parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    ZoneSpec,
    assemble,
    D4TetCell,
    D4CubeCell,
    check_topological_habit_admissibility,
    compute_analytical_pam_gap,
    compute_inverted_void_shift,
    get_coherent_twin_transform,
    plan_heteroepitaxial_interface,
    evaluate_interface_quality_index,
)
from graphite.explicit.interlinked.validator import evaluate_inter_zone_catenation


def test_topological_habit_admissibility_rules():
    """Verify selection rules correctly discriminate between viable and forbidden interfaces."""
    # Tetrahedral {100} habit plane is topologically frustrated
    is_adm_100, reason_100 = check_topological_habit_admissibility("100", "D4-TET", "D4-TET")
    assert not is_adm_100
    assert "Topologically Forbidden" in reason_100

    # Tetrahedral {111} habit plane provides ample void clearance
    is_adm_111, reason_111 = check_topological_habit_admissibility("111", "D4-TET", "D4-TET")
    assert is_adm_111
    assert "admissible" in reason_111

    # D4-Cube {100} has cubic symmetry and allows square face catenation
    is_adm_cube, _ = check_topological_habit_admissibility("100", "D4-CUBE", "D4-CUBE")
    assert is_adm_cube


def test_analytical_pam_gap_and_inverted_void_shift():
    """Verify closed-form formulas match expected physical bounds."""
    a = 12.0
    r_w = 0.5

    gap = compute_analytical_pam_gap(
        pitch_a=a,
        pitch_b=a,
        habit_plane="111",
        network_a="dia",
        network_b="dia",
        wire_radius=r_w,
    )
    assert gap > 0.0
    assert np.isclose(gap, np.sqrt(3.0) - r_w * 2.0 / np.cos(np.radians(35.264)), atol=1e-3)

    # Inverted void shift along {111}
    s_star = compute_inverted_void_shift("D4-TET", "D4-TET", habit_plane="111", pitch_a=a, pitch_b=a)
    # Shockley partial dislocation in Y
    assert np.isclose(s_star[0], 0.0)
    assert np.isclose(s_star[1], a * np.sqrt(6.0) / 12.0, atol=1e-3)


def test_coherent_twin_transform():
    """Verify Sigma 3 reflection matrix properties for {111}."""
    R_twin, t_twin = get_coherent_twin_transform("111", in_supercell_frame=False)
    assert R_twin.shape == (3, 3)
    assert np.allclose(t_twin, [0.0, 0.0, 0.0])

    # Determinant of reflection must be -1
    det = np.linalg.det(R_twin)
    assert np.isclose(det, -1.0)

    # Reflection squared is Identity
    assert np.allclose(R_twin @ R_twin, np.eye(3))

    # Analytical values: diag = 1/3, off-diag = -2/3
    expected = (np.eye(3) - 2.0 * np.ones((3, 3)) / 3.0)
    assert np.allclose(R_twin, expected)


def test_sigma3_twin_boundary_assembly_and_mesh_parity():
    """Assemble a Sigma 3 {111} twin boundary on D4-TET and verify watertightness and quality."""
    rw = 0.25
    za_twin = ZoneSpec(cell=D4TetCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, name="Parent_Grain")
    zb_twin = ZoneSpec(cell=D4TetCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, is_twin=True, name="Twin_Grain")

    plan_twin = plan_heteroepitaxial_interface(za_twin, zb_twin, habit_plane="111", wire_radius=rw, is_twin=True)
    assert plan_twin.interface.reflection_matrix is not None
    assert np.isclose(np.linalg.det(plan_twin.interface.reflection_matrix), -1.0)

    res_twin = assemble(
        zones=[plan_twin.zone_a, plan_twin.zone_b],
        interfaces=[plan_twin.interface],
        wire_radius=rw,
        min_clearance=0.10,
        mode=plan_twin.mode,
        check_clearance=True,
    )

    col_twin = evaluate_inter_zone_catenation(res_twin.particles, wire_radius=rw, min_clearance=0.10)
    assert len(col_twin.boundary_colliding_pairs) == 0
    assert col_twin.boundary_min_clearance_mm >= 0.10
    assert col_twin.boundary_linked_count > 0

    mesh_twin = res_twin.mesh
    assert mesh_twin.is_watertight
    assert len(mesh_twin.faces) > 0

    q_score = evaluate_interface_quality_index(
        min_clearance_um=col_twin.boundary_min_clearance_mm * 1000.0,
        min_face_punctures=3,
        avg_face_punctures=3.0,
        num_boundary_links=col_twin.boundary_linked_count,
        max_possible_links=col_twin.boundary_linked_count,
        sigma_index=3,
    )
    assert q_score >= 0.70
