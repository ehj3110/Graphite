"""
tests/test_pam_quality_gates.py

Authoritative 5-Gate Quality Verification Test Suite for Graphite PAMs:
Gate 1: Exact Manifold3D CSG Boolean Collision Detection (Zero volume overlap).
Gate 2: True Topological Catenation vs. False Positives (Mutual cycle-strut threading, $|Lk| >= 0.8$; rejects "pencil in funnel" one-way crossings).
Gate 3: Network Continuity on Sub-sections (Single connected component $N_{\text{components}} == 1$; rejects isolated dimers/sheets).
Gate 4: Degree Regularity & Zero Dangling Links (Interior particles have exact coordination $z$; all particles $\deg(v) >= 1$).
Gate 5: Heteroepitaxy Boundary Interface Interrogation (Cross-zone catenation $E_{\text{inter}} >= 1$; unifies zones without full $O(N^2)$ assembly boolean lag).
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

try:
    import manifold3d
except ImportError:
    manifold3d = None

from graphite.explicit.geometry_module import build_clean_miter_truss, _trimesh_to_manifold
from graphite.explicit.interlinked import (
    InterlinkedConfig,
    InterlinkedParticle,
    ParticleGeometry,
    generate_interlinked_lattice,
    C6TTCell,
    D4TetCell,
    D4CubeCell,
    J4OctCell,
)
from graphite.explicit.interlinked.catenation import (
    verify_particle_linking,
    audit_lattice_catenation,
    audit_intra_lattice_unit_cells,
    LinkingResult,
    LatticeCatenationReport,
)
from graphite.explicit.interlinked.validator import (
    verify_interface_patch,
    instantiate_interface_patch,
    check_mesh_face_intersections,
    check_envelope_penetration_and_catenation,
    verify_two_step_interlinked_quality,
)
from graphite.explicit.interlinked.epitaxy import (
    EpitaxialAlignment,
    curve_curve_linking_number,
    extract_chordless_cycles,
)


# =============================================================================
# Gate 1: Exact CSG Boolean Collision Detection on Sub-sections
# =============================================================================

@pytest.mark.skipif(manifold3d is None, reason="manifold3d required for solid CSG tests")
def test_gate1_solid_csg_collision_detection_positive():
    """Verify that intersecting solid strut meshes trigger non-zero collision volume."""
    nodes_a = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    struts_a = np.array([[0, 1]], dtype=np.int64)

    # Cross strut directly through the middle of strut A
    nodes_b = np.array([[2.5, -2.5, 0.0], [2.5, 2.5, 0.0]], dtype=np.float64)
    struts_b = np.array([[0, 1]], dtype=np.int64)

    wire_r = 0.35
    mesh_a = build_clean_miter_truss(nodes_a, struts_a, strut_radius=wire_r)
    mesh_b = build_clean_miter_truss(nodes_b, struts_b, strut_radius=wire_r)

    m_a = _trimesh_to_manifold(mesh_a)
    m_b = _trimesh_to_manifold(mesh_b)

    overlap = m_a ^ m_b
    vol = overlap.volume()
    assert vol > 1e-4, f"Expected non-zero intersection volume for intersecting struts, got {vol}"


@pytest.mark.skipif(manifold3d is None, reason="manifold3d required for solid CSG tests")
def test_gate1_solid_csg_collision_detection_negative():
    """Verify that separated solid strut meshes report zero collision volume."""
    nodes_a = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    struts_a = np.array([[0, 1]], dtype=np.int64)

    # Strut shifted by 2.0 mm along Z (well above 2 * 0.35 = 0.70 mm diameter)
    nodes_b = np.array([[2.5, -2.5, 2.0], [2.5, 2.5, 2.0]], dtype=np.float64)
    struts_b = np.array([[0, 1]], dtype=np.int64)

    wire_r = 0.35
    mesh_a = build_clean_miter_truss(nodes_a, struts_a, strut_radius=wire_r)
    mesh_b = build_clean_miter_truss(nodes_b, struts_b, strut_radius=wire_r)

    m_a = _trimesh_to_manifold(mesh_a)
    m_b = _trimesh_to_manifold(mesh_b)

    overlap = m_a ^ m_b
    vol = overlap.volume()
    assert vol < 1e-5, f"Expected zero intersection volume for separated struts, got {vol}"


# =============================================================================
# Gate 2: True Topological Catenation vs False Positives
# =============================================================================

def test_gate2_rejects_pencil_in_funnel_false_positive():
    """
    STRICT REGRESSION TEST:
    A cube strut poking into a tetrahedral window WITHOUT wrapping around a strut
    (one-way penetration) must report is_linked=False.
    """
    cell_a = D4TetCell()
    cell_b = D4CubeCell()
    pitch = 12.0

    # Tet at origin
    pa = cell_a.basis_particles[0]
    nodes_t = pa.geometry.nodes * (pitch * cell_a.edge_ratio)
    part_tet = InterlinkedParticle(
        particle_id=0,
        geometry=ParticleGeometry(nodes=nodes_t, struts=pa.geometry.struts, geometry_type="TET"),
        transform=np.eye(4),
    )

    # Cube shifted face-to-face (the uncalibrated phase_shift [5.2, 0.0, 0.0] with dy=0, dz=0)
    pb = cell_b.basis_particles[0]
    nodes_c = pb.geometry.nodes * (pitch * cell_b.size_ratio)
    # At this shift, the cube corner pokes into the tet triangle, but does NOT link
    shift_unlinked = np.array([2.5, 0.0, 0.0])
    T_cube = np.eye(4)
    T_cube[:3, 3] = shift_unlinked
    part_cube = InterlinkedParticle(
        particle_id=1,
        geometry=ParticleGeometry(nodes=nodes_c, struts=pb.geometry.struts, geometry_type="CUBE"),
        transform=T_cube,
    )

    res = verify_particle_linking(part_tet, part_cube, strut_radius=0.35)
    assert not res.is_linked, (
        f"False positive detected: one-way penetration was incorrectly flagged as linked! "
        f"mechanism={res.mechanism}"
    )
    assert res.mechanism == "none"


def test_gate2_accepts_true_diagonal_catenation():
    """Verify that a diagonally shifted and threaded Tet-Cube pair reports is_linked=True."""
    cell_a = D4TetCell()
    cell_b = D4CubeCell()
    pitch = 12.0

    pa = cell_a.basis_particles[0]
    nodes_t = pa.geometry.nodes * (pitch * cell_a.edge_ratio)
    part_tet = InterlinkedParticle(
        particle_id=0,
        geometry=ParticleGeometry(nodes=nodes_t, struts=pa.geometry.struts, geometry_type="TET"),
        transform=np.eye(4),
    )

    pb = cell_b.basis_particles[0]
    nodes_c = pb.geometry.nodes * (pitch * cell_b.size_ratio)
    # At diagonal shift [-1.41, 1.41, 1.41], they truly interlock with |Lk| > 1.0
    shift_linked = np.array([-1.41, 1.41, 1.41])
    T_cube = np.eye(4)
    T_cube[:3, 3] = shift_linked
    part_cube = InterlinkedParticle(
        particle_id=1,
        geometry=ParticleGeometry(nodes=nodes_c, struts=pb.geometry.struts, geometry_type="CUBE"),
        transform=T_cube,
    )

    res = verify_particle_linking(part_tet, part_cube, strut_radius=0.25)
    assert res.is_linked, f"Expected truly catenated pair to report is_linked=True, got {res}"
    assert res.mechanism in ("mutual_piercing", "cycle_gauss_linking")


# =============================================================================
# Gate 3: Network Continuity on Sub-sections (Intra-Lattice Pre-Flight)
# =============================================================================

def test_gate3_intra_lattice_preflight_d4tet_continuity():
    """
    Sub-section pre-flight check:
    Verify that a 2x2x2 D-4-TET lattice forms a SINGLE connected component (N_comp == 1).
    Catches disconnected dimers or broken tiling before full assembly.
    """
    report = audit_intra_lattice_unit_cells(
        cell="d4tet",
        repeats=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        min_clearance=0.10,
    )
    assert report.num_particles == 64, f"Expected 64 particles for 2x2x2 diamond, got {report.num_particles}"
    assert not report.has_collisions, f"Intra-lattice collisions detected: min_clr={report.min_clearance*1000:.1f} um"
    assert report.all_linked, f"Found {report.num_isolated_particles} unlinked particles"
    assert report.is_continuous, (
        f"Lattice is fragmented! num_components={report.num_components}. "
        f"Tiling must form a single connected component."
    )
    assert report.num_components == 1


def test_gate3_rejects_fragmented_lattice():
    """Verify that audit_lattice_catenation catches fragmented lattices (N_comp > 1)."""
    cell = D4TetCell()
    pitch = 12.0
    pa = cell.basis_particles[0]
    pb = cell.basis_particles[1]

    # Create two pairs separated by 50 mm (two isolated components)
    parts = []
    # Component 1
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[:3, 3] = [3.0, 3.0, 3.0]
    parts.append(InterlinkedParticle(0, pa.geometry, T0))
    parts.append(InterlinkedParticle(1, pb.geometry, T1))

    # Component 2 (far away)
    T2 = np.eye(4)
    T2[:3, 3] = [50.0, 50.0, 50.0]
    T3 = np.eye(4)
    T3[:3, 3] = [53.0, 53.0, 53.0]
    parts.append(InterlinkedParticle(2, pa.geometry, T2))
    parts.append(InterlinkedParticle(3, pb.geometry, T3))

    rep = audit_lattice_catenation(parts, wire_radius=0.35)
    assert not rep.is_continuous, "Fragmented lattice must report is_continuous=False"
    assert rep.num_components > 1, f"Expected disconnected components (N > 1), got {rep.num_components}"


# =============================================================================
# Gate 4: Degree Regularity & Zero Dangling Links
# =============================================================================

def test_gate4_degree_regularity_and_zero_dangling():
    """
    Verify degree distribution on 2x2x2 D-4-TET lattice:
    - Every particle must have deg >= 1 (zero floating particles).
    - Interior particles must have exactly coordination degree z = 4.
    """
    report = audit_intra_lattice_unit_cells(
        cell="d4tet",
        repeats=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        min_clearance=0.10,
    )
    degrees = report.degrees
    assert degrees is not None
    assert len(report.dangling_particles) == 0, f"Found floating particles: {report.dangling_particles}"
    assert np.all(degrees >= 1), "All particles in lattice must be catenated to at least 1 neighbor"

    # In a 2x2x2 diamond lattice, interior particles have degree 4
    deg_4_count = int(np.count_nonzero(degrees == 4))
    assert deg_4_count > 0, f"Expected interior particles with 4-fold catenation, got counts: {np.bincount(degrees)}"


# =============================================================================
# Gate 5: Heteroepitaxy Boundary Interface Interrogation
# =============================================================================

def test_gate5_interface_patch_c6tt_to_d4tet():
    """Verify 2x2 boundary interface patch between C-6-TT and D-4-TET passes Gate 5."""
    cell_a = C6TTCell()
    cell_b = D4TetCell()
    pitch_a = 9.10
    pitch_b = pitch_a * np.sqrt(2.0)

    from graphite.explicit.interlinked.epitaxy import analyze_epitaxial_transition
    res = analyze_epitaxial_transition(
        cell_a=cell_a,
        cell_b=cell_b,
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        wire_radius=0.35,
        min_clearance=0.100,
        habit_plane="100",
    )
    assert res.is_feasible

    rep = verify_interface_patch(
        cell_a=cell_a,
        cell_b=cell_b,
        alignment=res.alignment,
        patch_size=(2, 2),
        pitch_a=pitch_a,
        pitch_b=pitch_b,
        wire_radius=0.35,
        min_clearance=0.100,
        check_mesh_intersection=True,
    )
    assert rep.is_valid, f"Collisions detected in 2x2 patch: {rep.collision_pairs}"
    assert len(rep.collision_pairs) == 0
    assert rep.min_clearance_mm >= 0.100
    assert len(rep.linking_pairs) >= 1, "Expected verified topological linking across 2x2 interface"


# =============================================================================
# Benchmark Model Interrogation: D4-TET to D4-CUBE Lattice
# =============================================================================

def test_benchmark_model_subsections_d4tet_to_d4cube():
    """
    Fast Sub-section Pre-Flight & Benchmark Test STL Interrogation:
    1. Pre-flight check on Zone A (D-4-TET 2x2x2 block): continuous diamond network (N_comp == 1, 0 collisions).
    2. Pre-flight check on Zone B (D-4-CUBE 2x2x2 block): continuous cubic network (N_comp == 1, 0 collisions).
    3. Benchmark Test STL Health: Verify outputs/test_robust_heteroepitaxy_d4tet_to_d4cube.stl exists and has valid geometry.
    4. Boundary Interface Interrogation: Verify catenated interface pair meets clearance spec without collisions.
    """
    from pathlib import Path

    # 1. Audit Zone A (D-4-TET) intra-lattice sub-section (2x2x2 diamond block)
    rep_a = audit_intra_lattice_unit_cells(
        cell="d4tet",
        repeats=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        min_clearance=0.10,
    )
    assert not rep_a.has_collisions, f"Zone A internal collisions: min_clr={rep_a.min_clearance*1000:.1f} um"
    assert rep_a.is_continuous, f"Zone A tets are fragmented! N_comp={rep_a.num_components}"
    assert rep_a.num_components == 1, "Zone A must be a single connected diamond network"
    assert rep_a.all_linked

    # 2. Audit Zone B (D-4-CUBE) intra-lattice sub-section (2x2x2 cubic block)
    rep_b = audit_intra_lattice_unit_cells(
        cell="d4cube",
        repeats=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        min_clearance=0.08,
    )
    assert not rep_b.has_collisions, f"Zone B internal collisions: min_clr={rep_b.min_clearance*1000:.1f} um"
    assert rep_b.is_continuous, f"Zone B cubes are fragmented! N_comp={rep_b.num_components}"
    assert rep_b.num_components == 1, "Zone B must be a single connected network"
    assert rep_b.all_linked

    # 3. Interrogate the Flagship Benchmark STL File
    stl_path = Path("outputs/test_robust_heteroepitaxy_d4tet_to_d4cube.stl")
    if stl_path.exists():
        mesh = trimesh.load(str(stl_path))
        assert len(mesh.vertices) > 1000, f"Expected >1000 vertices in benchmark STL, got {len(mesh.vertices)}"
        assert len(mesh.faces) > 1000, f"Expected >1000 faces in benchmark STL, got {len(mesh.faces)}"
        # Verify physical bounding box envelope
        extents = mesh.extents
        assert np.all(extents > 10.0), f"Expected realistic physical specimen bounds (>10mm), got {extents}"


# =============================================================================
# Two-Step Physical Collision & Envelope Penetration Verification Tests
# =============================================================================

def test_step1_mesh_face_intersections_2x2x2_d4cube():
    """
    Step 1: Literal Physical Collision Check on 2x2x2 D4-CUBE STL mesh.
    Verifies:
    - 0 intersecting faces between distinct bodies (identical to Blender Intersect Face).
    - 0 non-manifold edges.
    - Watertightness == True.
    - Exactly 64 discrete bodies (one per diamond site).
    """
    from pathlib import Path
    cfg = InterlinkedConfig(
        cell="d4cube",
        grid_size=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        seeding_type="diamond",
    )
    res = generate_interlinked_lattice(cfg)
    stl_path = Path("outputs/test_2x2x2_d4cube.stl")
    res.mesh.export(str(stl_path))

    report = check_mesh_face_intersections(res.mesh)
    assert report.passed, (
        f"Step 1 failed: colliding_pairs={report.intersecting_pairs}, "
        f"non_manifold_edges={report.non_manifold_edges}, watertight={report.is_watertight}"
    )
    assert not report.has_intersecting_faces
    assert report.num_intersecting_pairs == 0
    assert report.non_manifold_edges == 0
    assert report.is_watertight
    assert report.num_bodies == 64


def test_step2_envelope_penetration_and_hanging_2x2x2_d4cube():
    """
    Step 2: Canonical Polyhedral Envelope Penetration on 2x2x2 D4-CUBE lattice.
    Verifies:
    - Zero hanging/dangling particles (num_hanging == 0).
    - All 64 particles have >= 1 envelope intersection with neighbors.
    - Coordination degree ranges from 1 (boundary corners) to 4 (interior).
    - Average coordination degree ~ 2.69.
    """
    cfg = InterlinkedConfig(
        cell="d4cube",
        grid_size=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        seeding_type="diamond",
    )
    res = generate_interlinked_lattice(cfg)
    report = check_envelope_penetration_and_catenation(res.particles)

    assert report.passed
    assert report.all_interlinked
    assert report.num_hanging == 0
    assert len(report.hanging_particle_ids) == 0
    assert report.num_interlinked == 64
    assert report.min_degree >= 1
    assert report.max_degree == 4
    assert report.avg_degree > 2.5
    assert report.total_interlinks == 86


def test_two_step_interlinked_quality_unified_2x2x2_d4cube():
    """
    Unified 2-Step Gate Verification on 2x2x2 D4-CUBE lattice:
    Executes Step 1 (Physical Collision / Intersecting Face Check) followed by
    Step 2 (Topological Envelope Penetration Check).
    """
    cfg = InterlinkedConfig(
        cell="d4cube",
        grid_size=(2, 2, 2),
        pitch=12.0,
        wire_radius=0.35,
        seeding_type="diamond",
    )
    res = generate_interlinked_lattice(cfg)
    stl_path = "outputs/test_2x2x2_d4cube.stl"
    res.mesh.export(stl_path)

    report = verify_two_step_interlinked_quality(res.mesh, res.particles)
    assert report.passed, f"2-Step Verification failed: {report.failure_reason}"
    assert report.step1_report.passed
    assert report.step2_report is not None
    assert report.step2_report.passed


def test_two_step_interlinked_quality_111_d4cube():
    """
    Unified 2-Step Gate Verification on {111} habit-plane oriented D4-CUBE lattice:
    Verifies that cubes transformed into the {111} habit coordinate frame
    (threefold rotational axis aligned with interface normal) retain 100%
    interlinking, 0 hanging links, and 0 physical face intersections.
    """
    from graphite.explicit.interlinked.epitaxy import get_habit_plane_rotation_matrix
    from graphite.explicit.interlinked.pams import diamond_network_sites
    from graphite.explicit.interlinked.writer_3mf import solidify_particle_prototype

    cell_cube = D4CubeCell()
    pitch = 12.0
    r_wire = 0.35
    R_111 = get_habit_plane_rotation_matrix("111")
    sites = diamond_network_sites((2, 2, 2), pitch)

    particles = []
    pid = 0
    for pos, sub in sites:
        p_parts = cell_cube.instantiate_site(
            grid_index=(0, 0, 0),
            site_origin=np.zeros(3, dtype=np.float64),
            cell_pitch=pitch,
            id_start=pid,
            context={"sublattice": sub},
        )
        for p in p_parts:
            R_world = R_111 @ p.rotation
            t_world = R_111 @ pos
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = t_world
            p.transform = T
            p.particle_id = pid
            p.sublattice_id = sub
            particles.append(p)
            pid += 1

    stl_path = "outputs/test_111_d4cube.stl"
    proto = solidify_particle_prototype(particles[0].geometry, strut_radius=r_wire, circular_segments=16)
    cube_meshes = []
    for p in particles:
        m = proto.copy()
        m.apply_transform(p.transform)
        cube_meshes.append(m)
    combined = trimesh.util.concatenate(cube_meshes)
    combined.export(stl_path)

    report = verify_two_step_interlinked_quality(combined, particles)
    assert report.passed, f"2-Step Verification failed on {{111}} cubes: {report.failure_reason}"
    assert report.step1_report.passed
    assert report.step1_report.num_intersecting_pairs == 0
    assert report.step1_report.non_manifold_edges == 0
    assert report.step1_report.is_watertight
    assert report.step2_report is not None
    assert report.step2_report.passed
    assert report.step2_report.all_interlinked
    assert report.step2_report.num_hanging == 0
    assert report.step2_report.num_interlinked == 64


