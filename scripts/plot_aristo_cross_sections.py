#!/usr/bin/env python
"""
Plot normalized nodal von Mises cross-sections from Aristo VTU files.

Uses Voronoi solid fill (nodal stress in a thin slab). Pass one or more VTU paths
with optional labels and reports.

Run from repo root::

    python scripts/plot_aristo_cross_sections.py \\
        --vtu case_a.vtu --label "Case A" \\
        --vtu case_b.vtu --label "Case B" \\
        --plane xz --slice-center-mm 0.5 --half-thickness-mm 0.012 \\
        --output outputs/cross_section_xz.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.cross_section_viz import FILL_MODES, plot_cross_section_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vtu",
        type=Path,
        action="append",
        required=True,
        help="Aristo FEA VTU (repeat for side-by-side panels).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Panel label (repeat; defaults to VTU stem).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        default=[],
        help="Optional Aristo report JSON per VTU (repeat).",
    )
    parser.add_argument(
        "--plane",
        choices=("xz", "xy", "yz"),
        default="xz",
        help="Slice plane: xz=Y slab; xy=Z slab; yz=X slab (vertical Y–Z cut).",
    )
    parser.add_argument(
        "--slice-center-mm",
        type=float,
        default=None,
        help="Center of slab normal to slice (Y for xz, Z for xy, X for yz). Default: mesh mid.",
    )
    parser.add_argument("--half-thickness-mm", type=float, default=0.012)
    parser.add_argument("--void-distance-mm", type=float, default=0.018)
    parser.add_argument("--voronoi-scale", type=float, default=0.0)
    parser.add_argument(
        "--fill-mode",
        choices=FILL_MODES,
        default="element-slice-soft",
        help="element-slice, element-slice-soft, voronoi-sharp, voronoi-soft, voronoi-bounded, voronoi-bounded-soft, blur-heatmap, or voronoi-disk.",
    )
    parser.add_argument(
        "--stl",
        type=Path,
        action="append",
        default=[],
        help="Watertight STL for geometry-bounded fill modes (repeat per VTU).",
    )
    parser.add_argument(
        "--blur-radius-mm",
        type=float,
        default=0.018,
        help="Gaussian blur radius in mm (element-slice-soft default: 0.015; blur-heatmap default: 0.018).",
    )
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument(
        "--raster-pixels",
        type=int,
        default=1200,
        help="Square raster resolution per panel (default: 1200).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG DPI (default: 300).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--clip-unit-cube",
        action="store_true",
        help="Clip in-plane axes to [0, 1] mm (1 mm³ cube domain).",
    )
    parser.add_argument(
        "--domain-clip",
        type=float,
        nargs=4,
        metavar=("A_MIN", "A_MAX", "B_MIN", "B_MAX"),
        default=None,
        help="Clip plot/raster to in-plane bounds (overrides --clip-unit-cube).",
    )
    args = parser.parse_args()

    labels = list(args.label)
    reports = list(args.report)
    stls = list(args.stl)
    cases = []
    for i, vtu in enumerate(args.vtu):
        if not vtu.is_file():
            print(f"VTU not found: {vtu}")
            return 1
        label = labels[i] if i < len(labels) else vtu.stem
        report = reports[i] if i < len(reports) else None
        case: dict = {"label": label, "vtu": vtu, "report": report}
        if i < len(stls):
            stl = stls[i]
            if not stl.is_file():
                print(f"STL not found: {stl}")
                return 1
            case["stl"] = stl
        cases.append(case)

    import pyvista as pv

    pts = pv.read(str(args.vtu[0])).points
    mid_x = 0.5 * (float(pts[:, 0].min()) + float(pts[:, 0].max()))
    mid_y = 0.5 * (float(pts[:, 1].min()) + float(pts[:, 1].max()))
    mid_z = 0.5 * (float(pts[:, 2].min()) + float(pts[:, 2].max()))
    half = float(args.half_thickness_mm)
    if args.slice_center_mm is not None:
        center = float(args.slice_center_mm)
    elif args.plane == "xz":
        center = mid_y
    elif args.plane == "xy":
        center = mid_z
    else:
        center = mid_x
    x_center = center if args.plane == "yz" else mid_x
    y_center = center if args.plane == "xz" else mid_y
    z_center = center if args.plane == "xy" else mid_z
    x_half = half if args.plane == "yz" else 0.0
    y_half = half if args.plane == "xz" else 0.0
    z_half = half if args.plane == "xy" else 0.0

    if args.domain_clip is not None:
        domain_clip_mm = tuple(float(v) for v in args.domain_clip)
    elif args.clip_unit_cube:
        domain_clip_mm = (0.0, 1.0, 0.0, 1.0)
    else:
        domain_clip_mm = None

    plot_cross_section_comparison(
        tuple(cases),
        plane=args.plane,
        x_center=x_center,
        y_center=y_center,
        z_center=z_center,
        x_half_thickness=x_half,
        y_half_thickness=y_half,
        z_half_thickness=z_half,
        output_path=args.output,
        void_distance_mm=float(args.void_distance_mm),
        voronoi_scale=float(args.voronoi_scale),
        fill_mode=args.fill_mode,
        draw_boundary=not args.no_boundary,
        raster_pixels=int(args.raster_pixels),
        dpi=int(args.dpi),
        blur_radius_mm=float(args.blur_radius_mm),
        domain_clip_mm=domain_clip_mm,
        suptitle=args.title,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
