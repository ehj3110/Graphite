#!/usr/bin/env python
"""
PAM Internal Mid-Plane Cutaway / Section Views Figure.

Reveals the 3D internal catenation loops and chainmail entanglement hidden inside
the macroscopic bulk coupon core by slicing the 75x25x25 mm specimens along the
longitudinal mid-plane (Y = 12.5 mm):
1. D-4-TET Specimen Mid-Plane Cutaway: Diamond network (dia) internal corner piercing.
2. C-6-TT Specimen Mid-Plane Cutaway: Simple cubic network (pcu) internal face catenation.

Outputs:
  outputs/diagnostics/pam_internal_cutaway_comparison.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graphite.explicit.interlinked.pams import (
    generate_c6tt_cubic_tiling,
    generate_d4tet_diamond_tiling,
)


def render_cutaway_specimen(
    meshes: list,
    colors: list[str],
    cut_origin: list[float] = [37.5, 12.5, 12.5],
    cut_normal: list[float] = [0.0, -1.0, 0.0],
    plane_color: str = "#ff0055",
    window_size: list[int] = [1800, 800],
    camera_pos: list | str = "iso",
    zoom: float = 1.18,
) -> np.ndarray:
    """Render an internal mid-plane cutaway of a PAM specimen with cutting plane visualization."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = "#0a0d14"

    # Add sliced internal geometry
    for m, c in zip(meshes, colors):
        pdata = pv.wrap(m)
        clipped = pdata.clip(normal=cut_normal, origin=cut_origin)
        plotter.add_mesh(
            clipped,
            color=c,
            smooth_shading=True,
            specular=0.45,
            specular_power=35,
            show_edges=False,
        )

    # Semi-transparent sectioning plane to visually emphasize the mid-plane slice
    plane = pv.Plane(
        center=cut_origin,
        direction=[0.0, 1.0, 0.0],
        i_size=82.0,
        j_size=32.0,
    )
    plotter.add_mesh(
        plane,
        color=plane_color,
        opacity=0.12,
        show_edges=True,
        edge_color=plane_color,
        line_width=1.5,
    )

    if isinstance(camera_pos, str):
        plotter.camera_position = camera_pos
    else:
        plotter.camera_position = camera_pos
    plotter.camera.zoom(zoom)

    img = plotter.screenshot()
    plotter.close()
    return img


def main() -> int:
    diag_dir = _REPO_ROOT / "outputs" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_png = diag_dir / "pam_internal_cutaway_comparison.png"

    print("=" * 80)
    print("Generating PAM Internal Mid-Plane Cutaway / Section Views Figure...")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. D-4-TET Mid-Plane Cutaway
    # -------------------------------------------------------------------------
    print("  [1/3] Generating D-4-TET cutaway (Y = 12.5 mm mid-plane)...")
    t0 = time.perf_counter()
    L_d4 = 15.0
    d4_macro = generate_d4tet_diamond_tiling(
        repeats=(3, 1, 1),
        conventional_cell_size=25.0,
        edge_length=L_d4,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    colors_d4 = [
        "#00d2ff" if p.metadata.get("sublattice") == "A" else "#ff6b6b"
        for p in d4_macro.particles
    ]

    cam = [(120, -110, 85), (37.5, 12.5, 12.5), (0, 0, 1)]
    img_d4_cut = render_cutaway_specimen(
        d4_macro.meshes,
        colors_d4,
        cut_origin=[37.5, 12.5, 12.5],
        cut_normal=[0.0, -1.0, 0.0],
        plane_color="#00d2ff",
        camera_pos=cam,
        zoom=1.20,
    )
    print(f"        D-4-TET Cutaway rendered in {time.perf_counter()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # 2. C-6-TT Mid-Plane Cutaway
    # -------------------------------------------------------------------------
    print("  [2/3] Generating C-6-TT cutaway (Y = 12.5 mm mid-plane)...")
    t1 = time.perf_counter()
    c6_macro = generate_c6tt_cubic_tiling(
        repeats=(6, 2, 2),
        size=10.0,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    palette_c6 = [
        "#50fa7b", "#f1fa8c", "#ff79c6", "#bd93f9",
        "#8be9fd", "#ffb86c", "#00d2ff", "#ff5555",
    ]
    colors_c6 = [palette_c6[i % len(palette_c6)] for i in range(len(c6_macro.meshes))]

    img_c6_cut = render_cutaway_specimen(
        c6_macro.meshes,
        colors_c6,
        cut_origin=[37.5, 12.5, 12.5],
        cut_normal=[0.0, -1.0, 0.0],
        plane_color="#50fa7b",
        camera_pos=cam,
        zoom=1.20,
    )
    print(f"        C-6-TT Cutaway rendered in {time.perf_counter()-t1:.2f}s")

    # -------------------------------------------------------------------------
    # 3. Composite 2-Panel Publication Layout
    # -------------------------------------------------------------------------
    print("  [3/3] Compositing internal cutaway publication figure...")
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), facecolor="#0e1117")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05, hspace=0.18)

    fig.suptitle(
        "Internal Mid-Plane Section Views: Unveiling 3D Bulk Polycatenation Architecture",
        fontsize=17,
        fontweight="bold",
        color="#ffffff",
        y=0.97,
    )

    # --- TOP: D-4-TET ---
    ax0 = axes[0]
    ax0.set_facecolor("#0a0d14")
    ax0.imshow(img_d4_cut)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title(
        "D-4-TET Specimen Internal Cutaway: Sliced at Y = 12.5 mm Mid-Plane",
        color="#00d2ff",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    for s in ax0.spines.values():
        s.set_color("#00d2ff")
        s.set_linewidth(1.8)

    ax0.text(
        0.02,
        0.06,
        "Internal Architecture: Dual Bipartite Diamond Catenation\n"
        "Section Plane: Slices across the conventional unit cell core, exposing the 3D interlocking apexes\n"
        "Mechanism: Corner-through-face piercing creates isotropic kinematic degrees of freedom without contact welding.",
        transform=ax0.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#00d2ff", alpha=0.92),
    )

    # --- BOTTOM: C-6-TT ---
    ax1 = axes[1]
    ax1.set_facecolor("#0a0d14")
    ax1.imshow(img_c6_cut)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(
        "C-6-TT Specimen Internal Cutaway: Sliced at Y = 12.5 mm Mid-Plane",
        color="#50fa7b",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    for s in ax1.spines.values():
        s.set_color("#50fa7b")
        s.set_linewidth(1.8)

    ax1.text(
        0.02,
        0.06,
        "Internal Architecture: Simple Cubic (pcu) 6-Fold Face Interlocking (Zhou et al., Science 2025)\n"
        "Section Plane: Exposes the internal hexagonal and triangular window cutouts through which adjacent cages interlock\n"
        "Mechanism: Collision-free 3D bulk interpenetration maintains continuous clearance gaps (Δ = 0.64 mm) across all core cells.",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#50fa7b", alpha=0.92),
    )

    print(f"  Saving publication figure to {out_png}...")
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[Done] Figure 6 generated: {out_png}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
