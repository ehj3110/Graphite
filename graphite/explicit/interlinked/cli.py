"""
graphite/explicit/interlinked/cli.py

Turnkey Command-Line Interface for Graphite Interlinked & Polycatenated Metamaterials.
Allows anyone to generate print-ready, non-glare, watertight 3MF/STL assemblies
and scientific preview renders with a single command.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence, Any

from .quick import (
    quick_lattice,
    quick_fabric,
    PRESETS,
    PREFERRED_SEEDING,
    RISK_PROFILES,
    PROFILE_ALIASES,
    CELL_ALIASES,
    _resolve_target,
)
from .cell import InterlinkedRegistry
from .dimensioning import PAMDimensioner, PAMDimensionResult, DEFAULT_WARNING_CLEARANCE


ALL_PAM_CELLS = [
    "d4tet",
    "c6tt",
    "j4oct",
    "j4ring",
    "d4cube",
    "d4ct",
    "d4ico",
    "t6ring",
    "t6hex",
    "c6co",
    "b8co",
    "euro_4in1",
    "nasa_hexagon",
    "kusari",
]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print clean ASCII / Markdown table."""
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for r in rows:
        print(" | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(r)))


def display_presets_catalog() -> None:
    """Display catalog of available high-level presets."""
    headers = ["Preset Name", "Default Cell", "Type", "Default Strut", "Default Clearance", "Application"]
    rows = []
    for k, v in PRESETS.items():
        rows.append([
            k,
            v["cell"],
            "2D Sheet" if v.get("is_2d") else "3D Lattice",
            f"{v['default_strut_d']:.2f} mm",
            f"{v['default_clearance']:.2f} mm",
            v["description"],
        ])
    print("\n=== Graphite Interlinked Metamaterial Presets ===")
    print_table(headers, rows)
    print()


def display_cells_catalog() -> None:
    """Display catalog of all registered PAM cells with intrinsic parameters."""
    headers = ["Cell Tag", "Full Name", "Network", "kappa", "C_geom", "Jamming Limit Vf_max"]
    rows = []
    for tag in ALL_PAM_CELLS:
        try:
            inst = PAMDimensioner.resolve_cell(tag)
            name = getattr(inst, "name", tag)
            net = getattr(inst, "parent_network", "pcu")
            kappa = PAMDimensioner.get_kappa(inst)
            c_geom = PAMDimensioner.compute_c_geom(inst)
            vf_max = c_geom * (kappa ** 2)
            rows.append([
                tag,
                name,
                net.upper(),
                f"{kappa:.4f}",
                f"{c_geom:.3f}",
                f"{vf_max * 100.0:.2f}%",
            ])
        except Exception:
            pass
    print("\n=== Graphite Polycatenated Metamaterial (PAM) Architecture Catalog ===")
    print_table(headers, rows)
    print()


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m graphite.explicit.interlinked",
        description="Turnkey Generator for Print-Ready Interlinked Metamaterials & Chainmail Fabrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Positional Quick Launch (Low Risk default: 1.0mm strut, 200um gap):
  python scripts/generate_pam.py shock_absorber --size 40 40 40 --out outputs/shock.3mf

  # High Risk Profile (0.75mm strut, 150um gap for fine resolution):
  python scripts/generate_pam.py chainmail --profile high_risk --size 60 60 --out outputs/maille.3mf

  # NASA JPL Space Fabric (80x80 mm sheet):
  python scripts/generate_pam.py space_fabric --size 80 80 --out outputs/fabric.3mf

  # DfAM Preflight Dry Run:
  python scripts/generate_pam.py d4tet --box 50 50 50 --profile low_risk --dry-run
        """,
    )

    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional preset name or cell tag (e.g. 'chainmail', 'shock_absorber', 'space_fabric', 'd4tet', 'c6tt')",
    )

    group_select = parser.add_argument_group("Architecture & Presets")
    group_select.add_argument(
        "--preset",
        type=str,
        default=None,
        help="High-level preset name ('shock_absorber', 'impact_cushion', 'cube_cages', 'chainmail', 'space_fabric', 'octahedra')",
    )
    group_select.add_argument(
        "--cell",
        type=str,
        default=None,
        help="Explicit cell architecture tag ('d4tet', 'c6tt', 'd4cube', 'j4oct', 'j4ring', 'b8co', 'euro_4in1', 'nasa_hexagon')",
    )
    group_select.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["low_risk", "high_risk", "low", "high"],
        help="Risk profile: 'low_risk' (1.0 mm strut, 200 um clearance, default) or 'high_risk' (0.75 mm strut, 150 um clearance)",
    )
    group_select.add_argument("--list-presets", action="store_true", help="List all available presets and exit")
    group_select.add_argument("--list-cells", action="store_true", help="List all registered PAM cells and exit")

    group_geom = parser.add_argument_group("Geometry & Dimensions (in mm)")
    group_geom.add_argument(
        "--size", "-s",
        nargs="+",
        type=float,
        default=None,
        metavar="DIM",
        help="Target physical size in mm: scalar, 2D, or 3D (e.g. --size 40, --size 60 60, or --size 40 40 40)",
    )
    group_geom.add_argument(
        "--box", "--bounding-box",
        dest="box",
        nargs=3,
        type=float,
        default=None,
        metavar=("Lx", "Ly", "Lz"),
        help="CAD target bounding envelope in mm (e.g. --box 40 40 40)",
    )
    group_geom.add_argument(
        "--grid",
        nargs=3,
        type=int,
        default=None,
        metavar=("nx", "ny", "nz"),
        help="Explicit integer repetition counts (e.g. --grid 3 3 3)",
    )
    group_geom.add_argument("--width", type=float, default=None, help="Sheet width in X (mm) for 2D fabrics")
    group_geom.add_argument("--length", type=float, default=None, help="Sheet length in Y (mm) for 2D fabrics")
    group_geom.add_argument(
        "--strut", "-d",
        type=float,
        default=None,
        help="Wire/strut outer diameter in mm (default: 0.75 mm or preset default)",
    )
    group_geom.add_argument(
        "--clearance", "-c",
        type=float,
        default=None,
        help="Minimum inter-cage surface clearance gap in mm (default: 0.20 mm or preset default)",
    )
    group_geom.add_argument(
        "--vf", "--solid-fraction",
        dest="vf",
        type=float,
        default=None,
        help="Optional target solid volume fraction (e.g. 0.10 for 10%%)",
    )

    group_frame = parser.add_argument_group("Handling Perimeter Frame (Policy C)")
    group_frame.add_argument("--frame", action="store_true", help="Synthesize solid CAD perimeter border for tensile testing/gripping")
    group_frame.add_argument("--frame-thickness", type=float, default=2.0, help="Perimeter wall thickness in mm (default: 2.0 mm)")
    group_frame.add_argument("--frame-margin", type=float, default=0.60, help="Gap clearance between struts and frame in mm (default: 0.60 mm)")

    group_out = parser.add_argument_group("Outputs & Visualization")
    group_out.add_argument(
        "--out", "-o",
        type=str,
        default=None,
        help="Destination deliverable path (.3mf or .stl). Defaults to outputs/pam_<cell>_<size>.3mf",
    )
    group_out.add_argument(
        "--preview", "-p",
        type=str,
        default=None,
        help="Optional destination path for matte scientific render PNG",
    )
    group_out.add_argument("--cutaway", action="store_true", help="Render midplane longitudinal cutaway in preview image")
    group_out.add_argument("--theme", type=str, default="light", choices=["light", "neutral", "dark"], help="Matte render color theme")
    group_out.add_argument("--dry-run", "--info", dest="dry_run", action="store_true", help="Solve sizing and run DfAM preflight without generating meshes")
    group_out.add_argument("--json", action="store_true", help="Output machine-readable JSON metadata")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute interlinked generator CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        display_presets_catalog()
        return 0

    if args.list_cells:
        display_cells_catalog()
        return 0

    # 1. Resolve target preset / cell
    raw_target = args.preset or args.cell or args.target or "shock_absorber"
    effective_cell, is_2d, p_info = _resolve_target(raw_target)

    # 2. Resolve risk profile
    eff_profile = "low_risk"
    if args.profile is not None:
        p_key = args.profile.lower().strip()
        if p_key in PROFILE_ALIASES:
            eff_profile = PROFILE_ALIASES[p_key]
        else:
            eff_profile = "low_risk"

    prof_data = RISK_PROFILES[eff_profile]
    base_strut = prof_data["strut_diameter"]
    base_clearance = prof_data["clearance"]

    strut_d = args.strut if args.strut is not None else base_strut
    clearance_val = args.clearance if args.clearance is not None else base_clearance

    # 3. Resolve target dimensions
    if args.box is not None:
        box_dims = tuple(float(x) for x in args.box)
    elif args.size is not None:
        s = [float(x) for x in args.size]
        if len(s) == 1:
            box_dims = (s[0], s[0], 1.0) if is_2d else (s[0], s[0], s[0])
        elif len(s) == 2:
            box_dims = (s[0], s[1], 1.0)
        else:
            box_dims = (s[0], s[1], s[2])
    elif args.width is not None and args.length is not None:
        box_dims = (float(args.width), float(args.length), 1.0)
    elif args.grid is not None:
        box_dims = None
    else:
        # Sensible default envelope
        box_dims = (60.0, 60.0, 1.0) if is_2d else (40.0, 40.0, 40.0)

    # 3. Solve analytical sizing / DfAM preflight
    cell_inst = InterlinkedRegistry.get(effective_cell)
    if callable(cell_inst):
        cell_inst = cell_inst()

    if args.vf is not None:
        dim_res = PAMDimensioner.solve_from_clearance_and_vf(
            cell=cell_inst,
            clearance=clearance_val,
            solid_fraction=float(args.vf),
            raise_on_jamming=False,
        )
        resolved_pitch = dim_res.pitch
        resolved_strut = dim_res.wire_diameter
    else:
        resolved_pitch = cell_inst.resolve_pitch(
            target_clearance=clearance_val,
            strut_diameter=strut_d,
        )
        resolved_strut = strut_d
        dim_res = PAMDimensioner.solve(
            cell=cell_inst,
            pitch=resolved_pitch,
            wire_diameter=resolved_strut,
            warning_clearance=DEFAULT_WARNING_CLEARANCE,
            raise_on_jamming=False,
        )

    # Compute repetitions
    if box_dims is not None:
        nx = max(1, int(round(box_dims[0] / resolved_pitch)))
        ny = max(1, int(round(box_dims[1] / resolved_pitch)))
        nz = 1 if is_2d else max(1, int(round(box_dims[2] / resolved_pitch)))
        grid_counts = (nx, ny, nz)
    elif args.grid is not None:
        grid_counts = tuple(int(x) for x in args.grid)
    else:
        grid_counts = (8, 8, 1) if is_2d else (4, 4, 4)

    # Estimate link count
    net = getattr(cell_inst, "parent_network", "pcu").lower()
    particles_per_cell = 8 if net == "dia" else (2 if net == "bcc" else 1)
    estimated_links = grid_counts[0] * grid_counts[1] * grid_counts[2] * particles_per_cell

    if args.dry_run:
        if args.json:
            payload = {
                "cell": effective_cell,
                "preset": raw_target if p_info else None,
                "risk_profile": eff_profile,
                "unit_cell_pitch_mm": round(resolved_pitch, 4),
                "cage_diameter_mm": round(dim_res.cage_diameter, 4),
                "wire_diameter_mm": round(resolved_strut, 4),
                "clearance_mm": round(dim_res.clearance, 4),
                "solid_fraction_percent": round(dim_res.solid_fraction_percent, 2),
                "jamming_limit_percent": round(dim_res.max_solid_fraction_percent, 2),
                "grid_size": list(grid_counts),
                "estimated_links": estimated_links,
                "is_feasible": dim_res.is_feasible,
                "warnings": dim_res.warnings,
            }
            print(json.dumps(payload, indent=2))
            return 0

        print("\n=== PAM DfAM Sizing & Preflight Assessment ===")
        print(f"  Preset / Target:        {raw_target if p_info else 'custom'}")
        print(f"  Risk Profile:           {eff_profile.upper()} ({prof_data['name']}: {prof_data['description']})")
        print(f"  Architecture:           {effective_cell} ({getattr(cell_inst, 'name', effective_cell)})")
        print(f"  Unit Cell Pitch (a):    {resolved_pitch:.4f} mm")
        print(f"  Cage Outer Dia (D):     {dim_res.cage_diameter:.4f} mm")
        print(f"  Strut Wire Dia (d):     {resolved_strut:.4f} mm ({resolved_strut * 1000.0:.0f} um)")
        print(f"  Surface Clearance (Delta): {dim_res.clearance:.4f} mm ({dim_res.clearance * 1000.0:.1f} um)")
        print(f"  Solid Density (Vf):     {dim_res.solid_fraction_percent:.2f}% (Jamming Limit: {dim_res.max_solid_fraction_percent:.2f}%)")
        print(f"  Grid Repetitions:       {grid_counts[0]} x {grid_counts[1]} x {grid_counts[2]}")
        print(f"  Estimated Link Count:   ~{estimated_links} independent moving bodies")
        print(f"  Feasible Non-Colliding: {'YES' if dim_res.is_feasible else 'NO'}")
        if dim_res.warnings:
            print("\n  DfAM Warnings:")
            for w in dim_res.warnings:
                print(f"    * {w}")
        print()
        return 0

    # 4. Generate Lattice
    print(f"\n=== Generating Interlinked Metamaterial: {effective_cell.upper()} ===")
    print(f"  Pitch: {resolved_pitch:.3f} mm | Strut: {resolved_strut*1000:.0f} um | Clearance: {clearance_val*1000:.0f} um")
    print(f"  Grid: {grid_counts} (~{estimated_links} links)")

    # Resolve output path
    target_tag = raw_target if p_info else effective_cell
    default_stem = f"pam_{target_tag}_{grid_counts[0]}x{grid_counts[1]}x{grid_counts[2]}"
    out_file = Path(args.out) if args.out else Path("outputs") / f"{default_stem}.3mf"

    result = quick_lattice(
        cell=effective_cell,
        grid_size=grid_counts,
        strut_diameter=resolved_strut,
        clearance=clearance_val,
        add_frame=args.frame,
        frame_thickness=args.frame_thickness,
        frame_margin=args.frame_margin,
        export_path=out_file,
    )

    print(f"\n{result.summary()}\n")
    print(f"  [OK] Saved deliverable: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

    # 5. Render Preview if requested
    if args.preview:
        prev_p = Path(args.preview)
        result.render(
            prev_p,
            cutaway=args.cutaway,
            theme=args.theme,
            title=f"{effective_cell.upper()} Interlinked Lattice",
            subtitle=f"Grid: {grid_counts} | Strut: {resolved_strut*1000:.0f} um | Clearance: {result.min_clearance*1000:.1f} um",
        )
        print(f"  [OK] Rendered preview:  {prev_p}")

    if args.json:
        payload = {
            "cell": effective_cell,
            "output_file": str(out_file),
            "num_particles": result.num_rings,
            "min_clearance_mm": result.min_clearance,
            "clearance_valid": result.clearance_valid,
            "file_size_kb": round(out_file.stat().st_size / 1024.0, 1),
        }
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
