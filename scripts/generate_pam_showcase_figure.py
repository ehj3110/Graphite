#!/usr/bin/env python
"""
Comparative 3-Panel Polycatenated Architected Materials (PAMs) Showcase Figure.

Compares 3 canonical 3D bulk kinematic PAM polyhedral lattices:
1. D-4-TET: Diamond network (dia) bipartite corner-to-corner catenation of tetrahedral cages (z = 4).
2. C-6-TT: Simple cubic network (pcu) 6-fold face catenation of truncated tetrahedra (z = 6, Zhou et al. Science 2025).
3. J-4-OCT: Square planar network (sql) tip catenation with alternating 45° Z-twist (z = 4).

Outputs:
  outputs/diagnostics/pam_showcase_3panel_comparison.png
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
    generate_c6tt_cubic_tiling,
    generate_d4tet_diamond_tiling,
    generate_j4oct_square_tiling,
)


def render_panel(meshes: list, colors: list[str], camera_pos: str | list = "iso", zoom: float = 1.15) -> np.ndarray:
    """Render a set of PAM particle meshes to a high-resolution RGB image."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1400])
    plotter.background_color = "#0a0d14"

    for m, c in zip(meshes, colors):
        pdata = pv.wrap(m)
        plotter.add_mesh(
            pdata,
            color=c,
            smooth_shading=True,
            specular=0.40,
            specular_power=30,
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
    out_png = diag_dir / "pam_showcase_3panel_comparison.png"

    print("=" * 75)
    print("Generating Comparative 3-Panel PAM Showcase Figure...")
    print("=" * 75)

    # 1. D-4-TET (Diamond Network, 1x1x1 Conventional Cell, 8 Cages)
    print("  [1/3] Generating D-4-TET (Diamond Tetrahedra)...")
    t0 = time.perf_counter()
    d4 = generate_d4tet_diamond_tiling(
        repeats=(1, 1, 1),
        edge_length=10.0,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
        clean_miter=True,
    )
    # Distinct 8-color palette so each individual D4 cage is immediately distinguishable
    palette_d4 = [
        "#ff4757",  # 0: Crimson Red
        "#2ed573",  # 1: Lime Green
        "#1e90ff",  # 2: Royal Blue
        "#ffa502",  # 3: Amber Orange
        "#9b59b6",  # 4: Amethyst Purple
        "#00d2d3",  # 5: Turquoise / Cyan
        "#ff6b81",  # 6: Coral Pink
        "#fed330",  # 7: Canary Sun Yellow
    ]
    colors_d4 = [palette_d4[i % len(palette_d4)] for i in range(len(d4.meshes))]
    img_d4 = render_panel(d4.meshes, colors_d4, camera_pos="iso", zoom=1.18)
    print(f"        Generated D-4-TET in {time.perf_counter()-t0:.2f}s (Clearance = {d4.min_clearance_mm:.3f} mm)")

    # 2. C-6-TT (Simple Cubic Network, 2x2x2 Repeats, 8 Cages)
    print("  [2/3] Generating C-6-TT (Simple Cubic Truncated Tetrahedra)...")
    t1 = time.perf_counter()
    c6 = generate_c6tt_cubic_tiling(
        repeats=(2, 2, 2),
        size=10.0,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    # Distinct pastel/spectral palette to show each individual 3D interlocking cage
    palette_c6 = ["#50fa7b", "#f1fa8c", "#ff79c6", "#bd93f9", "#8be9fd", "#ffb86c", "#00d2ff", "#ff5555"]
    colors_c6 = [palette_c6[i % len(palette_c6)] for i in range(len(c6.meshes))]
    img_c6 = render_panel(c6.meshes, colors_c6, camera_pos="iso", zoom=1.18)
    print(f"        Generated C-6-TT in {time.perf_counter()-t1:.2f}s (Clearance = {c6.min_clearance_mm:.3f} mm)")

    # 3. J-4-OCT (Square Planar Network, 2x2x1 Repeats, 5 Cages)
    print("  [3/3] Generating J-4-OCT (Square Planar Octahedra)...")
    t2 = time.perf_counter()
    j4 = generate_j4oct_square_tiling(
        repeats=(2, 2, 1),
        size=10.0,
        strut_radius=0.50,
        min_clearance=0.30,
        build_meshes=True,
    )
    palette_j4 = ["#bd93f9", "#8be9fd", "#ffb86c", "#50fa7b", "#ff79c6"]
    colors_j4 = [palette_j4[i % len(palette_j4)] for i in range(len(j4.meshes))]
    img_j4 = render_panel(j4.meshes, colors_j4, camera_pos="iso", zoom=1.18)
    print(f"        Generated J-4-OCT in {time.perf_counter()-t2:.2f}s (Clearance = {j4.min_clearance_mm:.3f} mm)")

    # -------------------------------------------------------------------------
    # Composite Publication Figure Assembly via Matplotlib
    # -------------------------------------------------------------------------
    print("  Compositing 3-panel publication figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8.2), facecolor="#0e1117")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.06, wspace=0.08)

    fig.suptitle(
        "Polycatenated Architected Materials (PAMs): Non-Welded Kinematic Metamaterials",
        fontsize=17,
        fontweight="bold",
        color="#ffffff",
        y=0.96,
    )

    panels_info = [
        {
            "ax": axes[0],
            "img": img_d4,
            "title": "Sub-Figure 1: D-4-TET\nDiamond Tetrahedral Catenation",
            "net": "Crystal Network: Diamond (dia)",
            "coord": "Coordination Number: z = 4",
            "mode": "Catenation: Corner-to-Corner Piercing",
            "clearance": f"Physical Clearance: Δ = {d4.min_clearance_mm:.2f} mm (Valid: {d4.clearance_valid})",
            "color": "#00d2ff",
            "notes": "Each of 8 diamond cages uniquely colored",
        },
        {
            "ax": axes[1],
            "img": img_c6,
            "title": "Sub-Figure 2: C-6-TT\nTruncated Tetrahedral Catenation",
            "net": "Crystal Network: Simple Cubic (pcu)",
            "coord": "Coordination Number: z = 6",
            "mode": "Catenation: 6-Fold Face Interlocking",
            "clearance": f"Physical Clearance: Δ = {c6.min_clearance_mm:.2f} mm (Valid: {c6.clearance_valid})",
            "color": "#50fa7b",
            "notes": "Zhou et al. (Science 2025), collision-free 3D bulk",
        },
        {
            "ax": axes[2],
            "img": img_j4,
            "title": "Sub-Figure 3: J-4-OCT\nSquare-Planar Octahedral Catenation",
            "net": "Crystal Network: Square Planar (sql)",
            "coord": "Coordination Number: z = 4",
            "mode": "Catenation: Tip Catenation (45° Z-Twist)",
            "clearance": f"Physical Clearance: Δ = {j4.min_clearance_mm:.2f} mm (Valid: {j4.clearance_valid})",
            "color": "#bd93f9",
            "notes": "Alternating azimuthal rotation prevents node welding",
        },
    ]

    for p in panels_info:
        ax = p["ax"]
        ax.set_facecolor("#0a0d14")
        ax.imshow(p["img"])
        ax.set_xticks([])
        ax.set_yticks([])

        # Subplot Title
        ax.set_title(p["title"], color="#ffffff", fontsize=12.5, fontweight="bold", pad=12)

        # Border styling
        for spine in ax.spines.values():
            spine.set_color("#2d3748")
            spine.set_linewidth(1.5)

        # Educational Specs Card at the bottom of each panel
        specs_text = (
            f"{p['net']}\n"
            f"{p['coord']}  |  {p['mode']}\n"
            f"{p['clearance']}\n"
            f"{p['notes']}"
        )
        ax.text(
            0.5,
            -0.03,
            specs_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="#e2e8f0",
            fontsize=9.0,
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#161b22",
                edgecolor=p["color"],
                linewidth=1.2,
                alpha=0.95,
            ),
        )

    print(f"  Saving high-resolution figure to {out_png}...")
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    # Also save to pam_showcase_3panel_test.png
    test_png = diag_dir / "pam_showcase_3panel_test.png"
    fig.savefig(test_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Export standalone high-res D-4-TET isometric multi-color image
    d4_standalone_png = diag_dir / "d4tet_unit_cell_isometric_multicolor.png"
    plt.figure(figsize=(10, 10), facecolor="#0a0d14")
    plt.imshow(img_d4)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(d4_standalone_png, dpi=300, facecolor="#0a0d14", edgecolor="none")
    plt.close()

    print(f"[Done] Generated: {out_png}")
    print(f"[Done] Generated: {test_png}")
    print(f"[Done] Generated: {d4_standalone_png}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

