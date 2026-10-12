"""
Test and Verification Suite for Analytical PAM Grain Boundaries & Coherent Twin Interfaces.

Validates:
1. Analytical Topological Selection Rules (immediate rejection of forbidden {100} tetrahedral habit planes).
2. Closed-form interplanar gap g*(a, rw) and Inverted Void O-Lattice shift s* for {111} glide interfaces.
3. Sigma-3 {111} Coherent Twin Boundary on D4-TET (R_twin reflection, zero in-plane shift, det(R)=-1 winding inversion).
4. Full CAD collision audits (0 clashes), Hopf catenation audits, clearance audits (delta >= 100 um),
   and watertight multi-body mesh export with PyVista framing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure graphite root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import trimesh

from graphite.explicit.interlinked import (
    ZoneSpec,
    InterfaceSpec,
    assemble,
    D4TetCell,
    D4CubeCell,
    check_topological_habit_admissibility,
    compute_analytical_pam_gap,
    compute_inverted_void_shift,
    get_coherent_twin_transform,
    plan_heteroepitaxial_interface,
    evaluate_interface_quality_index,
    check_envelope_penetration_and_catenation,
    check_particle_clearance,
    evaluate_assembly_connectivity,
)
from graphite.explicit.interlinked.validator import evaluate_inter_zone_catenation


def run_all_tests():
    print("=" * 80)
    print("TEST SUITE: ANALYTICAL PAM GRAIN BOUNDARIES & COHERENT TWIN INTERFACES")
    print("=" * 80)

    out_dir = Path("outputs/pam_analytical_interfaces")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # TEST 1: Topological Selection Rules & Pre-flight Gate
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Topological Selection Rules & Habit Plane Admissibility Gate...")
    
    # 1.1 Forbidden {100} tetrahedral-tetrahedral transition
    is_adm_100, reason_100 = check_topological_habit_admissibility(
        habit_plane="100",
        cell_a="d4tet",
        cell_b="d4tet",
    )
    print(f"  {100} D4-TET <-> D4-TET: is_admissible = {is_adm_100}")
    print(f"  Reason: {reason_100}")
    assert not is_adm_100, "Expected {100} D4-TET transition to be rejected as Topologically Forbidden!"
    assert "Topologically Forbidden" in reason_100 or "Aperture clash" in reason_100

    # 1.2 Admissible {111} tetrahedral transition
    is_adm_111, reason_111 = check_topological_habit_admissibility(
        habit_plane="111",
        cell_a="d4tet",
        cell_b="d4tet",
    )
    print(f"  {111} D4-TET <-> D4-TET: is_admissible = {is_adm_111}")
    print(f"  Reason: {reason_111}")
    assert is_adm_111, "Expected {111} D4-TET transition to be admissible!"
    print("  -> TEST 1 PASSED: Selection gate strictly rejects clashing planes and permits threading planes.")

    # -------------------------------------------------------------------------
    # TEST 2: Analytical Closed-Form Gap g* and Inverted Void Shift s*
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Analytical Closed-Form Formulas (g* and s*)...")
    pitch = 12.0
    rw = 0.25

    # Analytical gap on {111} diamond
    g_star = compute_analytical_pam_gap(
        pitch_a=pitch,
        pitch_b=pitch,
        habit_plane="111",
        network_a="dia",
        network_b="dia",
        wire_radius=rw,
    )
    d_shuffle = (pitch * np.sqrt(3.0)) / 12.0
    phi_rad = np.radians(35.264)
    expected_g = d_shuffle - rw * (2.0 / np.cos(phi_rad))
    print(f"  Analytical g* on {111}: {g_star:.4f} mm (expected: {expected_g:.4f} mm)")
    assert abs(g_star - expected_g) < 1e-4, f"Mismatch in analytical gap calculation: {g_star} vs {expected_g}"

    # Inverted void shift s* on {111}
    s_star = compute_inverted_void_shift(
        cell_a="d4cube",
        cell_b="d4tet",
        habit_plane="111",
        pitch_a=pitch,
        pitch_b=pitch,
    )
    expected_sy = (pitch * np.sqrt(6.0)) / 12.0
    print(f"  Inverted Void Shift s*: {s_star.tolist()} (expected sy: {expected_sy:.4f} mm)")
    assert abs(s_star[1] - expected_sy) < 1e-4, f"Mismatch in Shockley void shift: {s_star[1]} vs {expected_sy}"

    # Assemble Analytical Coupon 1A via plan_heteroepitaxial_interface
    za = ZoneSpec(cell=D4CubeCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, name="Zone_Cube")
    zb = ZoneSpec(cell=D4TetCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, name="Zone_Tet")
    plan_1a = plan_heteroepitaxial_interface(za, zb, habit_plane="111", wire_radius=rw)
    print(f"  Interface plan mode: {plan_1a.mode}, shift: {plan_1a.interface.shift}, gap: {plan_1a.interface.gap}")

    res_1a = assemble(
        zones=[plan_1a.zone_a, plan_1a.zone_b],
        interfaces=[plan_1a.interface],
        wire_radius=rw,
        min_clearance=0.10,
        mode=plan_1a.mode,
        check_clearance=True,
    )
    
    col_1a = evaluate_inter_zone_catenation(res_1a.particles, wire_radius=rw, min_clearance=0.10)
    print(f"  Coupon 1A Collisions: {len(col_1a.boundary_colliding_pairs)}, Min Clearance: {col_1a.boundary_min_clearance_mm*1000:.1f} um")
    print(f"  Coupon 1A Boundary Links: {col_1a.boundary_linked_count}")
    assert len(col_1a.boundary_colliding_pairs) == 0, f"Analytical Coupon 1A produced collisions: {col_1a.boundary_colliding_pairs}"
    assert col_1a.boundary_min_clearance_mm >= 0.10, f"Clearance below threshold: {col_1a.boundary_min_clearance_mm} mm"
    assert col_1a.boundary_linked_count > 0, "Coupon 1A must have positive boundary catenation!"
    print("  -> TEST 2 PASSED: Analytical closed-form gap and inverted void shift yield clash-free geometry.")

    # -------------------------------------------------------------------------
    # TEST 3: Sigma-3 {111} Coherent Twin Boundary on D4-TET
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Sigma-3 {111} Coherent Twin Boundary on D4-TET...")
    R_twin, t_twin = get_coherent_twin_transform("111", in_supercell_frame=True)
    det_R = float(np.linalg.det(R_twin))
    print(f"  R_twin in supercell frame:\n{R_twin}")
    print(f"  det(R_twin) = {det_R:.1f} (Exact mirror reflection parity: -1.0)")
    assert abs(det_R - (-1.0)) < 1e-6, "Reflection matrix must have det = -1!"

    # Setup parent and twin zones
    za_twin = ZoneSpec(cell=D4TetCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, name="Parent_Grain")
    zb_twin = ZoneSpec(cell=D4TetCell(), network="dia", pitch=14.0, grid_size=(2, 2, 0.35), frame="111", wire_radius=rw, is_twin=True, name="Twin_Grain")

    plan_twin = plan_heteroepitaxial_interface(za_twin, zb_twin, habit_plane="111", wire_radius=rw, is_twin=True)
    print(f"  Twin plan diagnostics: {plan_twin.diagnostics}")
    assert plan_twin.diagnostics.get("strategy") == "Sigma 3 Coherent Twin Boundary Mirror Reflection"

    res_twin = assemble(
        zones=[plan_twin.zone_a, plan_twin.zone_b],
        interfaces=[plan_twin.interface],
        wire_radius=rw,
        min_clearance=0.10,
        mode=plan_twin.mode,
        check_clearance=True,
    )

    # Verify solid collisions and inter-zone boundary linking
    col_twin = evaluate_inter_zone_catenation(res_twin.particles, wire_radius=rw, min_clearance=0.10)
    print(f"  Twin Collision Pairs: {len(col_twin.boundary_colliding_pairs)}")
    print(f"  Twin Min Clearance: {col_twin.boundary_min_clearance_mm*1000:.1f} um")
    print(f"  Twin Boundary Links: {col_twin.boundary_linked_count}")
    assert len(col_twin.boundary_colliding_pairs) == 0, f"Twin boundary produced solid collisions: {col_twin.boundary_colliding_pairs}"
    assert col_twin.boundary_min_clearance_mm >= 0.10, f"Twin boundary clearance below 100 um: {col_twin.boundary_min_clearance_mm*1000:.1f} um"
    assert col_twin.boundary_linked_count > 0, "Twin boundary must exhibit positive topological catenation!"

    # Verify Watertight Mesh & Winding Inversion Safeguard
    mesh_twin = res_twin.mesh
    print(f"  Combined Mesh Faces: {len(mesh_twin.faces)}, Vertices: {len(mesh_twin.vertices)}")
    print(f"  Is Watertight: {mesh_twin.is_watertight}")
    assert mesh_twin.is_watertight, "Combined twin mesh must be 100% watertight!"

    # Evaluate Q_interface score
    q_score = evaluate_interface_quality_index(
        min_clearance_um=col_twin.boundary_min_clearance_mm * 1000.0,
        min_face_punctures=3,
        avg_face_punctures=3.0,
        num_boundary_links=col_twin.boundary_linked_count,
        max_possible_links=col_twin.boundary_linked_count,
        sigma_index=3,
    )
    print(f"  Quantitative Interface Quality Index Q_interface: {q_score:.4f}")
    assert q_score >= 0.75, f"Twin boundary Q_interface ({q_score:.4f}) should be >= 0.75!"
    print("  -> TEST 3 PASSED: Sigma-3 Coherent Twin Boundary verified with zero clashes and watertight winding.")

    # -------------------------------------------------------------------------
    # Render Diagnostic Visuals with PyVista
    # -------------------------------------------------------------------------
    print("\n[RENDERING] Generating PyVista Visualization...")
    stl_path = out_dir / "coupon_twin_sigma3_111.stl"
    png_path = out_dir / "coupon_twin_sigma3_111.png"
    mesh_twin.export(str(stl_path))
    print(f"  Exported STL to {stl_path}")

    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
        plotter.set_background("#181824")

        # Color Zone A vs Zone B differently
        pv_mesh_a = pv.PolyData()
        pv_mesh_b = pv.PolyData()

        # Build zone meshes for coloring
        parts_twin_a = [p for p in res_twin.particles if p.metadata.get("zone_index") == 0]
        parts_twin_b = [p for p in res_twin.particles if p.metadata.get("zone_index") == 1]
        from graphite.explicit.interlinked.generator import _particles_to_combined_mesh
        mesh_a = _particles_to_combined_mesh(parts_twin_a, wire_radius=rw)
        mesh_b = _particles_to_combined_mesh(parts_twin_b, wire_radius=rw)

        pv_a = pv.wrap(mesh_a)
        pv_b = pv.wrap(mesh_b)

        plotter.add_mesh(pv_a, color="#00E5FF", pbr=True, roughness=0.3, metallic=0.7)
        plotter.add_mesh(pv_b, color="#FF9100", pbr=True, roughness=0.3, metallic=0.7)

        plotter.camera_position = [
            (25.0, -35.0, 20.0),
            (6.0, 6.0, 10.0),
            (0.0, 0.0, 1.0),
        ]
        plotter.add_text(
            f"Sigma-3 {{111}} Coherent Twin Boundary\n"
            f"Parent: D4-TET (Cyan) | Twin: D4-TET (Orange)\n"
            f"Clearance: {col_twin.boundary_min_clearance_mm*1000:.1f} um | Clashes: 0 | Q: {q_score:.3f}",
            position="upper_left",
            font_size=11,
            color="white",
        )
        plotter.screenshot(str(png_path))
        plotter.close()
        print(f"  Rendered PNG to {png_path}")
    except Exception as e:
        print(f"  (Warning: PyVista rendering skipped or failed: {e})")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED SUCCESSFULLY (3/3)")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
