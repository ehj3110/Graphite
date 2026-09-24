#!/usr/bin/env python
"""
Comparative 75mm x 25mm PAM Macro-Specimen Showcase with Micro-Cell Insets.

Compares macroscopic 75x25x25 mm tensile testing coupons with micro-scale
crystallographic unit cells for:
1. D-4-TET: Diamond Network (dia), z=4, Bipartite A/B Coloring.
2. C-6-TT: Simple Cubic Network (pcu), z=6, Cyclic Pastel Multi-Body Coloring.

Outputs:
  outputs/diagnostics/pam_75x25_specimens_showcase.png
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

from graphite.explicit.interlinked.pams import (
    calibrate_d4tet_edge_length,
    generate_c6tt_cubic_tiling,
    generate_d4tet_diamond_tiling,
    generate_d4tet_interlocked_pair,
)


def render_mesh_list(
    meshes: list,
    colors: list[str],
    window_size: list[int] = [1800, 900],
    camera_pos: str | list = "iso",
    zoom: float = 1.15,
) -> np.ndarray:
    """Render meshes off-screen using PyVista with dark publication styling."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = "#0a0d14"

    for m, c in zip(meshes, colors):
        pdata = pv.wrap(m)
        plotter.add_mesh(
            pdata,
            color=c,
            smooth_shading=True,
            specular=0.45,
            specular_power=35,
            show_edges=False,
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
    out_png = diag_dir / "pam_75x25_specimens_showcase.png"

    print("=" * 80)
    print("Generating 75mm x 25mm PAM Specimen Showcase with Micro-Cell Insets...")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. D-4-TET Specimen & Micro-Cell Inset
    # -------------------------------------------------------------------------
    print("  [1/4] Generating D-4-TET 75x25x25 mm coupon (3x1x1 conventional cells)...")
    t0 = time.perf_counter()
    L_d4 = 15.0  # optimal diamond cage edge length for conventional cell a = 25.0 mm
    d4_macro = generate_d4tet_diamond_tiling(
        repeats=(3, 1, 1),
        conventional_cell_size=25.0,
        edge_length=L_d4,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    colors_d4_macro = [
        "#00d2ff" if p.metadata.get("sublattice") == "A" else "#ff6b6b"
        for p in d4_macro.particles
    ]
    img_d4_macro = render_mesh_list(
        d4_macro.meshes,
        colors_d4_macro,
        window_size=[1800, 800],
        camera_pos=[(120, -110, 85), (37.5, 12.5, 12.5), (0, 0, 1)],
        zoom=1.20,
    )
    print(f"        D-4-TET Macro rendered in {time.perf_counter()-t0:.2f}s (Clearance = {d4_macro.min_clearance_mm:.2f} mm)")

    print("  [2/4] Generating D-4-TET Micro-Cell Inset (Catenation Pair)...")
    t1 = time.perf_counter()
    d4_pair = generate_d4tet_interlocked_pair(
        edge_length=L_d4,
        strut_radius=0.50,
        min_clearance=0.30,
    )
    colors_d4_micro = ["#00d2ff", "#ff6b6b"]
    img_d4_micro = render_mesh_list(
        d4_pair.meshes,
        colors_d4_micro,
        window_size=[800, 800],
        camera_pos="iso",
        zoom=1.25,
    )
    print(f"        D-4-TET Micro rendered in {time.perf_counter()-t1:.2f}s")

    # -------------------------------------------------------------------------
    # 2. C-6-TT Specimen & Micro-Cell Inset
    # -------------------------------------------------------------------------
    print("  [3/4] Generating C-6-TT 75x25x25 mm coupon (6x2x2 cells)...")
    t2 = time.perf_counter()
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
    colors_c6_macro = [palette_c6[i % len(palette_c6)] for i in range(len(c6_macro.meshes))]
    img_c6_macro = render_mesh_list(
        c6_macro.meshes,
        colors_c6_macro,
        window_size=[1800, 800],
        camera_pos=[(120, -110, 85), (37.5, 12.5, 12.5), (0, 0, 1)],
        zoom=1.20,
    )
    print(f"        C-6-TT Macro rendered in {time.perf_counter()-t2:.2f}s (Clearance = {c6_macro.min_clearance_mm:.2f} mm)")

    print("  [4/4] Generating C-6-TT Micro-Cell Inset (2x2x2 Unit Cell)...")
    t3 = time.perf_counter()
    c6_micro = generate_c6tt_cubic_tiling(
        repeats=(2, 2, 2),
        size=10.0,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    colors_c6_micro = [palette_c6[i % len(palette_c6)] for i in range(len(c6_micro.meshes))]
    img_c6_micro = render_mesh_list(
        c6_micro.meshes,
        colors_c6_micro,
        window_size=[800, 800],
        camera_pos="iso",
        zoom=1.20,
    )
    print(f"        C-6-TT Micro rendered in {time.perf_counter()-t3:.2f}s")

    # -------------------------------------------------------------------------
    # 3. Composite 2-Row Publication Layout
    # -------------------------------------------------------------------------
    print("  Compositing high-resolution publication showcase figure...")
    fig = plt.figure(figsize=(22, 13), facecolor="#0e1117")

    fig.suptitle(
        "Polycatenated Metamaterial (PAM) Tensile Specimens (75 mm × 25 mm × 25 mm)",
        fontsize=18,
        fontweight="bold",
        color="#ffffff",
        y=0.97,
    )

    gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.2], height_ratios=[1, 1], wspace=0.04, hspace=0.18, left=0.03, right=0.97, top=0.91, bottom=0.05)

    # --- ROW 1: D-4-TET ---
    ax_d4_macro = fig.add_subplot(gs[0, 0])
    ax_d4_macro.set_facecolor("#0a0d14")
    ax_d4_macro.imshow(img_d4_macro)
    ax_d4_macro.set_xticks([])
    ax_d4_macro.set_yticks([])
    ax_d4_macro.set_title(
        "D-4-TET Macro Specimen: 75 × 25 × 25 mm Coupon (3 × 1 × 1 Conventional Cells, 24 Particles)",
        color="#00d2ff",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    for s in ax_d4_macro.spines.values():
        s.set_color("#00d2ff")
        s.set_linewidth(1.8)

    ax_d4_macro.text(
        0.02,
        0.06,
        "Network: Diamond (dia) | Coordination: z = 4 | Bipartite Sublattices A (Cyan) & B (Coral)\n"
        f"Conventional Cell a = 25.0 mm | Wire: r = 0.50 mm (1.0 mm dia) | Min Clearance: Δ = {d4_macro.min_clearance_mm:.2f} mm",
        transform=ax_d4_macro.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#00d2ff", alpha=0.90),
    )

    ax_d4_micro = fig.add_subplot(gs[0, 1])
    ax_d4_micro.set_facecolor("#0a0d14")
    ax_d4_micro.imshow(img_d4_micro)
    ax_d4_micro.set_xticks([])
    ax_d4_micro.set_yticks([])
    ax_d4_micro.set_title(
        "Micro-Cell: Corner-Piercing Pair",
        color="#00d2ff",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    for s in ax_d4_micro.spines.values():
        s.set_color("#2d3748")
        s.set_linewidth(1.5)

    # --- ROW 2: C-6-TT ---
    ax_c6_macro = fig.add_subplot(gs[1, 0])
    ax_c6_macro.set_facecolor("#0a0d14")
    ax_c6_macro.imshow(img_c6_macro)
    ax_c6_macro.set_xticks([])
    ax_c6_macro.set_yticks([])
    ax_c6_macro.set_title(
        "C-6-TT Macro Specimen: 75 × 25 × 25 mm Coupon (6 × 2 × 2 Cells, 24 Particles)",
        color="#50fa7b",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    for s in ax_c6_macro.spines.values():
        s.set_color("#50fa7b")
        s.set_linewidth(1.8)

    ax_c6_macro.text(
        0.02,
        0.06,
        "Network: Simple Cubic (pcu) | Coordination: z = 6 | Truncated Tetrahedra with Cyclic Multi-Body Coloring\n"
        f"Unit Cell Pitch a0 = 12.5 mm | Wire: r = 0.50 mm (1.0 mm dia) | Min Clearance: Δ = {c6_macro.min_clearance_mm:.2f} mm",
        transform=ax_c6_macro.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#50fa7b", alpha=0.90),
    )

    ax_c6_micro = fig.add_subplot(gs[1, 1])
    ax_c6_micro.set_facecolor("#0a0d14")
    ax_c6_micro.imshow(img_c6_micro)
    ax_c6_micro.set_xticks([])
    ax_c6_micro.set_yticks([])
    ax_c6_micro.set_title(
        "Micro-Cell: 2×2×2 Unit Cluster",
        color="#50fa7b",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    for s in ax_c6_micro.spines.values():
        s.set_color("#2d3748")
        s.set_linewidth(1.5)

    print(f"  Saving publication figure to {out_png}...")
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[Done] Figure 3 generated: {out_png}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
