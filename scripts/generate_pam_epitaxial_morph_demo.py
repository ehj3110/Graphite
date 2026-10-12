#!/usr/bin/env python
"""
Generate Demonstration Specimen for Epitaxially-Constrained Continuous Basis Morphing (Idea 3).

Specimen Features:
- Zone 1: Pure Phase A (C-6-TT, Simple Cubic, tau = 1/3, theta = 0 deg)
- Zone 2: Continuous Epitaxial Morphing Band (tau: 0.22 -> 0.11, theta: 15 deg -> 30 deg)
- Zone 3: Pure Phase B (D-4-TET, Diamond Habit Plane, tau = 0.0, theta = 45 deg)
- Transverse Pinning: Pitch locked to a_sc = 10.0 mm across all intermediate slices (0% strain)
- Thickness Grading: Gentle linear grading along X (r = 0.38 mm -> 0.28 mm)
- Deliverables: STL, instanced 3MF, and publication render under outputs/
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np
import pyvista as pv
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.explicit.interlinked import (
    EpitaxialMorphConfig,
    generate_epitaxial_morph_lattice,
    LinearThicknessGradient,
)


def render_morph_specimen(
    mesh: trimesh.Trimesh,
    particles: list,
    out_png: Path,
    title: str = "Epitaxially-Constrained Continuous Basis Morphing PAM",
) -> None:
    """Render 3D isometric view with zone coloration and informative telemetry."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.set_background("#0a0e14", top="#141a24")

    # Colorize particles by zone
    # Zone A (Pure C6): Cyan/Teal (#00d2d3)
    # Zone Morph (Hybrid): Amber/Gold (#feca57)
    # Zone B (Pure D4): Coral/Red (#ff6b6b)
    zone_colors = {
        "Zone_A_Pure": "#00d2d3",
        "Zone_Morph": "#feca57",
        "Zone_B_Pure": "#ff6b6b",
    }

    from graphite.explicit.geometry_module import build_clean_miter_truss

    # Render each particle with its zone color and clean mitered truss
    for p in particles:
        z = p.metadata.get("zone", "Zone_Morph")
        color = zone_colors.get(z, "#48dbfb")
        r_wire = p.effective_wire_radius(fallback=0.32)
        nodes_w = (p.transform[:3, :3] @ p.geometry.nodes.T).T + p.transform[:3, 3]
        p_mesh = build_clean_miter_truss(
            nodes=nodes_w,
            struts=p.geometry.struts,
            strut_radius=r_wire,
            circular_segments=16,
        )
        pv_m = pv.wrap(p_mesh)
        plotter.add_mesh(
            pv_m,
            color=color,
            specular=0.5,
            specular_power=25,
            roughness=0.3,
            smooth_shading=True,
        )

    # Telemetry HUD
    n_a = sum(1 for p in particles if p.metadata.get("zone") == "Zone_A_Pure")
    n_m = sum(1 for p in particles if p.metadata.get("zone") == "Zone_Morph")
    n_b = sum(1 for p in particles if p.metadata.get("zone") == "Zone_B_Pure")

    hud_text = (
        f"{title}\n"
        f"Zone 1 (Pure C-6-TT): {n_a} cages | Zone 2 (Morph Band): {n_m} cages | Zone 3 (Pure D-4-TET): {n_b} cages\n"
        f"Total: {len(particles)} discrete kinematic cages | Transverse Strain: 0.0% (Zero Z-Tearing)\n"
        f"Catenation: 100% Interlinked across habit planes | All joints clean mitered bisector planes"
    )
    plotter.add_text(
        hud_text,
        position="upper_left",
        font_size=12,
        color="#e0e6ed",
        font="arial",
    )

    # Shadow Plane
    bounds = mesh.bounds
    z_floor = bounds[0, 2] - 1.5
    floor = pv.Plane(
        center=((bounds[0, 0] + bounds[1, 0]) / 2, (bounds[0, 1] + bounds[1, 1]) / 2, z_floor),
        direction=(0, 0, 1),
        i_size=bounds[1, 0] - bounds[0, 0] + 30.0,
        j_size=bounds[1, 1] - bounds[0, 1] + 30.0,
    )
    plotter.add_mesh(floor, color="#121820", edge_color="#243040", show_edges=True, line_width=1.0)

    plotter.camera_position = [(-65.0, -85.0, 60.0), (-10.0, 10.0, 10.0), (0, 0, 1)]
    plotter.camera.zoom(1.20)
    plotter.screenshot(str(out_png))
    plotter.close()


def main() -> int:
    t0 = time.perf_counter()
    out_dir = _REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stl = out_dir / "pam_epitaxial_morph_demo.stl"
    out_3mf = out_dir / "pam_epitaxial_morph_demo.3mf"
    out_png = out_dir / "pam_epitaxial_morph_demo.png"

    print("=================================================================")
    print("Generating Epitaxially-Constrained Continuous Basis Morphing PAM Demo...")
    print("=================================================================")

    # 1. Configure Continuous Morph Lattice
    pitch = 10.0
    r_start = 0.380
    r_end = 0.280
    grid_size_a = (2, 3, 3)
    grid_size_b = (2, 3, 3)
    num_morph = 2

    cfg = EpitaxialMorphConfig(
        cell_a="c6tt",
        cell_b="d4tet",
        grid_size_a=grid_size_a,
        grid_size_b=grid_size_b,
        num_morph_layers=num_morph,
        pitch=pitch,
        wire_radius=0.32,
        min_clearance=0.150,
        thickness_gradient=LinearThicknessGradient(axis="x", r_start=r_start, r_end=r_end),
        num_thickness_buckets=12,
    )

    print(f"  Configuration:")
    print(f"    Zone 1: C-6-TT ({grid_size_a[0]} layers, tau=1/3, theta=0 deg)")
    print(f"    Zone 2: Continuous Morph Band ({num_morph} layers, tau: 0.22 -> 0.11, theta: 15 -> 30 deg)")
    print(f"    Zone 3: D-4-TET ({grid_size_b[0]} layers, tau=0.0, theta=45 deg)")
    print(f"    Transverse Coincidence: Locked to a_sc = {pitch:.1f} mm (0% transverse strain)")
    print(f"    Strut thickness grading: r = {r_start*1000:.0f} um -> {r_end*1000:.0f} um along X")

    # 2. Synthesize Multi-Zone Lattice
    print("\n  [1/3] Synthesizing continuous morph interlinked assembly...")
    result = generate_epitaxial_morph_lattice(cfg, check_clearance=True)

    dim = result.mesh.bounds[1] - result.mesh.bounds[0]
    radii = [p.effective_wire_radius(fallback=0.32) for p in result.particles]

    print(f"    Total cages: {len(result.particles)}")
    print(f"    Strut radius range: {min(radii)*1000:.1f} um to {max(radii)*1000:.1f} um")
    print(f"    Specimen dimensions: {dim[0]:.2f} mm (X) x {dim[1]:.2f} mm (Y) x {dim[2]:.2f} mm (Z)")
    print(f"    Minimum clearance: {result.min_clearance*1000:.1f} um (Target: 150 um)")
    print(f"    Clearance verified: {result.clearance_valid}")

    # 3. Export Assets
    print("\n  [2/3] Exporting CAD assets...")
    result.export_stl(out_stl)
    stl_mb = out_stl.stat().st_size / (1024 * 1024)
    print(f"    Exported Watertight Multi-body STL: {out_stl} ({stl_mb:.2f} MB)")

    result.export_3mf(out_3mf)
    tmf_kb = out_3mf.stat().st_size / 1024
    print(f"    Exported Instanced 3MF: {out_3mf} ({tmf_kb:.1f} KB)")

    # 4. Render Figure
    print("\n  [3/3] Rendering publication figure...")
    render_morph_specimen(result.mesh, result.particles, out_png)
    print(f"    Saved rendering: {out_png}")

    dt = time.perf_counter() - t0
    print(f"\n=================================================================")
    print(f"Epitaxial Morph PAM generation complete in {dt:.2f}s!")
    print(f"=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
