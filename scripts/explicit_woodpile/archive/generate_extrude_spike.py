#!/usr/bin/env python
"""
Phase 0 — single-layer woodpile extrusion spike.

Generates one X-oriented and one Y-oriented layer in a 1 mm³ footprint for
visual comparison against implicit marching-cubes woodpile.

Run from repo root::

    python scripts/explicit_woodpile/generate_extrude_spike.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.explicit.woodpile_extrude import (
    bar_polygons_single_layer,
    generate_single_layer_box,
    union_bar_polygons,
    write_mesh_report,
)

OUT_DIR = _REPO_ROOT / "outputs" / "explicit_woodpile" / "P0"
PORE_MM = 0.2
ORIGIN = (0.0, 0.0, 0.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_report: dict = {
        "phase": 0,
        "pore_mm": PORE_MM,
        "box_mm": [1.0, 1.0, PORE_MM],
        "layers": {},
    }

    for axis in ("x", "y"):
        mesh, layer_report = generate_single_layer_box(
            strut_axis=axis,
            pore_mm=PORE_MM,
            width_x_mm=1.0,
            depth_y_mm=1.0,
            origin_x_mm=ORIGIN[0],
            origin_y_mm=ORIGIN[1],
            origin_z_mm=ORIGIN[2],
        )
        stl_name = f"P0_single_{axis.upper()}_layer_box1mm.stl"
        stl_path = OUT_DIR / stl_name
        write_mesh_report(mesh, stl_path, layer_report)
        phase_report["layers"][axis] = layer_report
        print(f"Wrote {stl_path}  faces={layer_report['faces']}  wt={layer_report['watertight']}")

    png_path = OUT_DIR / "P0_single_layer_footprints.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, axis in zip(axes, ("x", "y"), strict=True):
        ox, oy, _ = ORIGIN
        polys = bar_polygons_single_layer(
            strut_axis=axis,
            pore_mm=PORE_MM,
            x_lo=ox,
            x_hi=ox + 1.0,
            y_lo=oy,
            y_hi=oy + 1.0,
            origin_x_mm=ox,
            origin_y_mm=oy,
        )
        footprint = union_bar_polygons(polys)
        geoms = list(footprint.geoms) if footprint.geom_type == "MultiPolygon" else [footprint]
        for poly in geoms:
            px, py = poly.exterior.xy
            ax.fill(px, py, color="#3498db", edgecolor="#1a5276", linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"{axis.upper()}-oriented bars")
    fig.suptitle(f"Phase 0 extrude footprints (pore={PORE_MM} mm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")

    phase_report["footprint_png"] = str(png_path.resolve())
    report_path = OUT_DIR / "P0_report.json"
    report_path.write_text(json.dumps(phase_report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
