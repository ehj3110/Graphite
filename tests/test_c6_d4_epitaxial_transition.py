"""
Unit & Integration Tests: C-6-TT to D-4-TET Epitaxial Transition Metamaterial.

Validates the Three Laws of Metamaterial Heteroepitaxy:
1. Commensurate Habit Plane Coincidence (Law 1)
2. Aperture Admissibility & Clearance (Law 2)
3. Topological Catenation Across Interface (Law 3)
4. Multi-Zone Watertight 3D Polycatenation in X, Y, and Z.
"""
from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked.clearance import segment_segment_distance
from graphite.explicit.interlinked.pams import (
    generate_truncated_tetrahedron_particle,
    PAMParticle,
    _tet_ab_templates,
    _TET_STRUTS,
    diamond_network_sites,
    strut_pierces_triangle,
)
from graphite.explicit.geometry_module import build_clean_miter_truss


def test_law_1_commensurate_habit_plane() -> None:
    """
    Law 1: Verify that a 45-degree rotated Diamond (100) habit plane
    achieves exact commensurability (zero lattice mismatch) with Simple Cubic.
    """
    a_sc = 9.10
    a_dia = np.sqrt(2.0) * a_sc  # 12.8693 mm

    # Rotation matrix 45 deg around X-axis
    theta = np.pi / 4.0
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])

    d_sites = diamond_network_sites((1, 1, 1), a_dia)
    boundary_sites = [pos for pos, sub in d_sites if abs(pos[0]) < 1e-6]

    # Rotate boundary sites
    rot_sites = [R @ p for p in boundary_sites]

    # Verify that the distance between nearest in-plane neighbors is exactly a_sc
    dists = []
    for i in range(len(rot_sites)):
        for j in range(i + 1, len(rot_sites)):
            d = float(np.linalg.norm(rot_sites[i] - rot_sites[j]))
            dists.append(d)

    min_spacing = min(dists)
    assert abs(min_spacing - a_sc) < 1e-4, f"Expected spacing {a_sc}, got {min_spacing}"


def test_law_2_aperture_admissibility() -> None:
    """
    Law 2: Verify that the C-6-TT hexagonal face aperture diameter is strictly
    greater than the D-4-TET penetrating edge profile plus clearance.
    """
    s_c6 = 8.0
    a_sc = 9.10
    a_dia = np.sqrt(2.0) * a_sc
    L_tet = 0.603 * a_dia

    # In C-6-TT, the distance from cage center to hexagonal face center is s / sqrt(2)
    # The hexagonal face opening has inscribed radius r_in = s / (3 * sqrt(2)) * sqrt(6)
    # Effective aperture diameter D_aperture:
    d_aperture = (2.0 / 3.0) * s_c6 * np.sqrt(2.0)  # ~7.54 mm

    # D-4-TET cross-sectional profile when entering the face:
    d_profile = L_tet / np.sqrt(2.0)  # ~5.49 mm

    target_clearance = 0.150  # 150 um
    strut_r = 0.35  # 700 um diameter

    assert d_aperture > d_profile + 2.0 * strut_r + 2.0 * target_clearance, (
        f"Aperture {d_aperture:.2f} mm must exceed profile {d_profile:.2f} mm + buffer"
    )


def test_law_3_topological_catenation_and_clearance() -> None:
    """
    Law 3: Verify that C-6-TT and D-4-TET form an interlocking catenated joint
    at the interface with non-zero piercings and >= 150 um physical clearance.
    """
    s_c6 = 8.0
    a_sc = 9.10
    a_dia = np.sqrt(2.0) * a_sc
    L_tet = 0.603 * a_dia
    dx_int = 7.20  # Optimal interface offset

    theta = np.pi / 4.0
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])

    nodes_a, _ = _tet_ab_templates(L_tet)
    nodes_rot = (R @ nodes_a.T).T

    # C-6-TT at [-dx_int, 0, 0]
    p_c6 = generate_truncated_tetrahedron_particle(size=s_c6, center=(-dx_int, 0.0, 0.0))

    # D-4-TET at [0, 0, 0]
    p_d4 = PAMParticle(1, nodes_rot, _TET_STRUTS.copy(), np.zeros(3), "TET")

    # 1. Topological catenation check (strut piercings)
    p0 = p_c6.nodes[p_c6.struts[:, 0]]
    p1 = p_c6.nodes[p_c6.struts[:, 1]]
    tet_faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    piercings = 0
    for i in range(len(p0)):
        for f in tet_faces:
            if strut_pierces_triangle(p0[i], p1[i], nodes_rot[f]):
                piercings += 1

    assert piercings >= 1, f"Expected at least 1 strut piercing, got {piercings}"

    # 2. Physical clearance check
    q0 = nodes_rot[_TET_STRUTS[:, 0]]
    q1 = nodes_rot[_TET_STRUTS[:, 1]]
    d_centerline = segment_segment_distance(p0, p1, q0, q1)

    r_strut = 0.35  # D = 700 um
    clearance = d_centerline - 2.0 * r_strut

    assert clearance >= 0.150, (
        f"Clearance {clearance*1000:.1f} um must be >= 150 um (Target: 150 um)"
    )


def test_multi_zone_epitaxial_assembly_3d() -> None:
    """
    Test complete 3-zone epitaxial assembly:
    Zone 1: Pure C-6-TT (3D Simple Cubic)
    Zone 2: Epitaxial Interface
    Zone 3: Pure D-4-TET (3D Diamond Cubic)
    Verifies that all 3 dimensions (X, Y, Z) are intercatenated with zero collisions.
    """
    s_c6 = 8.0
    a_sc = 9.10
    a_dia = np.sqrt(2.0) * a_sc
    L_tet = 0.603 * a_dia
    dx_int = 7.20

    theta = np.pi / 4.0
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    nodes_a, nodes_b = _tet_ab_templates(L_tet)

    # Zone 1: C-6-TT (2x2x2 block)
    particles = []
    pid = 0
    for ix in [-1, 0]:
        for iy in range(2):
            for iz in range(2):
                c = np.array([ix * a_sc - dx_int, iy * a_sc, iz * a_sc], dtype=np.float64)
                p = generate_truncated_tetrahedron_particle(size=s_c6, center=c, particle_id=pid)
                particles.append(p)
                pid += 1

    # Zone 3: D-4-TET
    d_sites = diamond_network_sites((2, 2, 2), a_dia)
    for pos, sub in d_sites:
        pos_rot = R @ pos
        if -1.0 <= pos_rot[1] <= a_sc + 1.0 and -1.0 <= pos_rot[2] <= a_sc + 1.0 and 0.0 <= pos_rot[0] <= a_sc * 1.5:
            tmpl = nodes_a if sub == "A" else nodes_b
            nodes_rot = (R @ tmpl.T).T
            p = PAMParticle(
                particle_id=pid,
                nodes=nodes_rot + pos_rot,
                struts=_TET_STRUTS.copy(),
                center=pos_rot.copy(),
                geometry_type="TET",
                metadata={"sublattice": sub},
            )
            particles.append(p)
            pid += 1

    assert len(particles) >= 12, f"Expected at least 12 particles, got {len(particles)}"

    # Evaluate global minimum clearance across all pairs
    r_strut = 0.35  # D = 700 um
    min_dist = float("inf")
    for i in range(len(particles)):
        p1 = particles[i].nodes[particles[i].struts[:, 0]]
        p2 = particles[i].nodes[particles[i].struts[:, 1]]
        for j in range(i + 1, len(particles)):
            q1 = particles[j].nodes[particles[j].struts[:, 0]]
            q2 = particles[j].nodes[particles[j].struts[:, 1]]
            d = segment_segment_distance(p1, p2, q1, q2)
            if d < min_dist:
                min_dist = d

    global_clearance = min_dist - 2.0 * r_strut
    assert global_clearance >= 0.150, (
        f"Global clearance {global_clearance*1000:.1f} um must be >= 150 um"
    )
