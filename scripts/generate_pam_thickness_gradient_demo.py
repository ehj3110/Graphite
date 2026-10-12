#!/usr/bin/env python
"""
Generate Demonstration of Universal PAM Strut Thickness Grading.

Demonstrates:
- Continuous linear spatial strut thickness grading D = 1.10 mm -> 0.50 mm (r = 0.55 -> 0.25 mm).
- Evaluates pair-specific kinematic clearances across the gradient.
- Exports watertight multi-body STL and O(K) bucketed instanced 3MF package.
- Renders high-resolution engineering verification figure colored by local strut thickness.

Outputs:
- outputs/pam_thickness_gradient_demo.stl (Multi-body watertight STL)
- outputs/pam_thickness_gradient_demo.3mf (O(K) bucketed instanced 3MF package)
- outputs/pam_thickness_gradient_demo.png (High-resolution verification figure)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.explicit.geometry_module import build_clean_miter_truss
from graphite.explicit.interlinked import (
    InterlinkedConfig,
    generate_interlinked_lattice,
    LinearThicknessGradient,
)
from graphite.explicit.interlinked.particle import InterlinkedParticle


def render_graded_lattice_figure(
    particles: list[InterlinkedParticle],
    out_png: Path,
) -> None:
    """Render publication-grade figure colored continuously by local strut radius."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.background_color = "#080c14"

    radii = [p.effective_wire_radius(fallback=0.40) for p in particles]
    r_min, r_max = min(radii), max(radii)

    # Colormap: Plasma (Dark blue/purple -> Orange -> Yellow for thickest)
    cmap = plt.get_cmap("plasma")

    for p in particles:
        r_val = p.effective_wire_radius(fallback=0.40)
        norm_val = (r_val - r_min) / max(r_max - r_min, 1e-6)
        rgba = cmap(norm_val)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        m = build_clean_miter_truss(p.global_nodes(), p.struts, r_val, circular_segments=16)
        plotter.add_mesh(pv.wrap(m), color=hex_color, smooth_shading=True, specular=0.5, specular_power=25)

    # Ambient floor
    floor = pv.Plane(
        center=(20.0, 10.0, -10.0),
        direction=(0, 0, 1),
        i_size=100.0,
        j_size=75.0,
    )
    plotter.add_mesh(floor, color="#121820", edge_color="#243040", show_edges=True, line_width=1.0)

    plotter.camera_position = [(-55.0, -65.0, 55.0), (20.0, 10.0, 5.0), (0, 0, 1)]
    plotter.camera.zoom(1.15)
    plotter.screenshot(str(out_png))
    plotter.close()


def main() -> int:
    t0 = time.perf_counter()
    out_dir = _REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_stl = out_dir / "pam_thickness_gradient_demo.stl"
    out_3mf = out_dir / "pam_thickness_gradient_demo.3mf"
    out_png = out_dir / "pam_thickness_gradient_demo.png"

    print("=================================================================")
    print("Generating Universal Strut Thickness Gradient PAM Demo...")
    print("=================================================================")

    # 1. Configure Graded Lattice
    pitch = 10.0
    r_start = 0.550  # D = 1.10 mm (1,100 um)
    r_end = 0.250    # D = 0.50 mm (500 um)
    grid_size = (5, 3, 2)  # 5 layers along X, 3 in Y, 2 in Z = 30 cages

    cfg = InterlinkedConfig(
        cell="c6tt",
        pitch=pitch,
        grid_size=grid_size,
        gradient_axis="x",
        gradient_radius_range=(r_start, r_end),
        min_clearance=0.150,
        num_thickness_buckets=10,
        num_ring_segments=16,
    )

    print(f"  Configuration:")
    print(f"    Cell: C-6-TT (Simple Cubic)")
    print(f"    Grid layout: {grid_size[0]} layers X x {grid_size[1]} cols Y x {grid_size[2]} tiers Z")
    print(f"    Strut grading: D = {2*r_start*1000:.0f} um -> {2*r_end*1000:.0f} um along X")
    print(f"    Target clearance: Delta >= {cfg.min_clearance*1000:.0f} um")
    print(f"    3MF prototype buckets: {cfg.num_thickness_buckets}")

    # 2. Invoke Canonical Interlinked Engine
    print("\n  [1/3] Synthesizing graded interlinked lattice...")
    result = generate_interlinked_lattice(cfg, check_clearance=True)

    dim = result.mesh.bounds[1] - result.mesh.bounds[0]
    radii = [p.effective_wire_radius(fallback=0.40) for p in result.particles]

    print(f"    Total cages: {len(result.particles)}")
    print(f"    Strut radius range: {min(radii)*1000:.1f} um to {max(radii)*1000:.1f} um")
    print(f"    Specimen dimensions: {dim[0]:.2f} mm (X) x {dim[1]:.2f} mm (Y) x {dim[2]:.2f} mm (Z)")
    print(f"    Physical clearance: min_clearance = {result.min_clearance*1000:.1f} um (Target: 150 um)")
    print(f"    Clearance verified: {result.clearance_valid}")

    # 3. Export CAD Models
    print("\n  [2/3] Exporting CAD assets...")
    result.export_stl(out_stl)
    stl_mb = out_stl.stat().st_size / (1024 * 1024)
    print(f"    Exported Watertight Multi-body STL: {out_stl} ({stl_mb:.2f} MB)")

    result.export_3mf(out_3mf)
    tmf_kb = out_3mf.stat().st_size / 1024
    print(f"    Exported Instanced 3MF: {out_3mf} ({tmf_kb:.1f} KB)")

    # 4. Render Engineering Verification Figure
    print("\n  [3/3] Rendering 3D engineering verification figure...")
    render_graded_lattice_figure(result.particles, out_png=out_png)
    print(f"    Rendered Engineering Figure: {out_png}")

    dt = time.perf_counter() - t0
    print("\n=================================================================")
    print(f"SUCCESS: Universal Strut Thickness Gradient Demo completed in {dt:.2f}s!")
    print(f"  Watertight STL: {out_stl}")
    print(f"  Instanced 3MF:  {out_3mf}")
    print(f"  Verification:   {out_png}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
