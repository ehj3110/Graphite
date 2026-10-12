#!/usr/bin/env python3
"""
scripts/dimension_pam.py

Universal Analytical PAM Dimensioning & Sizing CLI.
Solves the coupled system of inter-cage clearance and solid volume fraction:
  Delta = kappa * a - d
  Vf = C_geom * (d / a)^2
  Vf_max = C_geom * kappa^2 (jamming limit)

Usage:
  # Inverse DfAM sizing (Target clearance + target density -> pitch + wire diameter)
  python scripts/dimension_pam.py --cell j4ring --clearance 0.20 --vf 0.10

  # Comparative cross-architecture matrix
  python scripts/dimension_pam.py --compare --clearance 0.20 --vf 0.08

  # List all registered PAM cells with intrinsic coefficients & jamming limits
  python scripts/dimension_pam.py --list-cells

  # Machine-readable JSON output
  python scripts/dimension_pam.py --cell d4tet --clearance 0.30 --vf 0.08 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.interlinked import (
    InterlinkedRegistry,
    PAMDimensioner,
    PAMDimensionResult,
    DEFAULT_WARNING_CLEARANCE,
    JammingLimitExceededError,
)

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


def list_all_cells() -> None:
    """Display catalog of all PAM architectures with their intrinsic constants."""
    headers = ["Cell Tag", "Full Name", "Network", "kappa", "C_geom", "Jamming Limit Vf_max"]
    rows = []
    for tag in ALL_PAM_CELLS:
        inst = PAMDimensioner.resolve_cell(tag)
        name = getattr(inst, "name", tag)
        net = getattr(inst, "parent_network", "pcu")
        kappa = PAMDimensioner.get_kappa(inst)
        c_geom = PAMDimensioner.compute_c_geom(inst)
        vf_max = c_geom * (kappa ** 2)
        rows.append([
            tag,
            name,
            net,
            f"{kappa:.4f}",
            f"{c_geom:.3f}",
            f"{vf_max * 100.0:.2f}%",
        ])
    print("\n=== Graphite Polycatenated Metamaterial (PAM) Architecture Catalog ===")
    print_table(headers, rows)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal Analytical PAM Dimensioning Engine (Zhou et al., Science 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cell", type=str, default="j4ring", help="Registered PAM cell tag (e.g. j4ring, d4tet, c6tt)")
    parser.add_argument("--pitch", "-a", type=float, default=None, help="Unit cell translation pitch a (mm)")
    parser.add_argument("--wire-diameter", "-d", type=float, default=None, help="Strut/wire diameter d (mm)")
    parser.add_argument("--clearance", "-c", type=float, default=None, help="Target inter-cage surface clearance Delta (mm)")
    parser.add_argument("--vf", "-v", type=float, default=None, help="Target solid volume fraction (e.g. 0.10 or 10.0 for 10%)")
    parser.add_argument(
        "--warning-clearance", "-w",
        type=float,
        default=DEFAULT_WARNING_CLEARANCE,
        help=f"DfAM warning clearance threshold in mm (default: {DEFAULT_WARNING_CLEARANCE:.3f} mm / 250 um)",
    )
    parser.add_argument("--compare", action="store_true", help="Compare all registered PAM cells for the given targets")
    parser.add_argument("--list-cells", action="store_true", help="List all registered PAM cells and intrinsic constants")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args()

    if args.list_cells:
        list_all_cells()
        return

    # Comparative evaluation across all PAM architectures
    if args.compare:
        if args.clearance is None or args.vf is None:
            parser.error("--compare requires both --clearance (-c) and --vf (-v) to be specified.")

        results = PAMDimensioner.compare_architectures(
            cells=ALL_PAM_CELLS,
            clearance=args.clearance,
            solid_fraction=args.vf,
            warning_clearance=args.warning_clearance,
        )

        if args.json:
            out = [
                {
                    "cell_name": r.cell_name,
                    "parent_network": r.parent_network,
                    "pitch_mm": round(r.pitch, 4) if not math.isnan(r.pitch) else None,
                    "wire_diameter_mm": round(r.wire_diameter, 4) if not math.isnan(r.wire_diameter) else None,
                    "clearance_mm": r.clearance,
                    "solid_fraction_percent": r.solid_fraction_percent,
                    "aspect_ratio": round(r.aspect_ratio, 4) if not math.isnan(r.aspect_ratio) else None,
                    "jamming_limit_percent": round(r.max_solid_fraction_percent, 2) if not math.isnan(r.max_solid_fraction_percent) else None,
                    "is_feasible": r.is_feasible,
                    "warnings": r.warnings,
                }
                for r in results
            ]
            print(json.dumps(out, indent=2))
            return

        print(f"\n=== Comparative PAM Sizing Matrix (Target Clearance = {args.clearance:.3f} mm, Target Vf = {results[0].solid_fraction_percent:.1f}%) ===")
        print(f"Warning Clearance Threshold: {args.warning_clearance * 1000.0:.0f} um ({args.warning_clearance:.3f} mm)\n")
        headers = ["Cell", "Network", "Pitch a (mm)", "Cage Dia (mm)", "Wire d (mm)", "Aspect d/a", "Jamming Vf_max", "Feasible?", "Status / Notes"]
        rows = []
        for r in results:
            if r.is_feasible:
                notes = "OK"
                if r.warnings:
                    notes = "; ".join(r.warnings)
                rows.append([
                    r.cell_name,
                    r.parent_network,
                    f"{r.pitch:.3f}",
                    f"{r.cage_diameter:.3f}",
                    f"{r.wire_diameter:.3f}",
                    f"{r.aspect_ratio:.4f}",
                    f"{r.max_solid_fraction_percent:.1f}%",
                    "YES",
                    notes,
                ])
            else:
                rows.append([
                    r.cell_name,
                    r.parent_network,
                    "---",
                    "---",
                    "---",
                    "---",
                    f"{r.max_solid_fraction_percent:.1f}%",
                    "NO",
                    "Exceeds Jamming Limit" if r.solid_fraction >= r.max_solid_fraction else "; ".join(r.warnings),
                ])
        print_table(headers, rows)
        print()
        return

    # Single cell evaluation
    inputs = {
        "pitch": args.pitch,
        "wire_diameter": args.wire_diameter,
        "clearance": args.clearance,
        "solid_fraction": args.vf,
    }
    specified = [k for k, v in inputs.items() if v is not None]

    if len(specified) != 2:
        parser.error(
            f"Exact 2-parameter specification required to solve the remaining 2. "
            f"Provided {len(specified)} parameters: {specified}."
        )

    try:
        res = PAMDimensioner.solve(
            cell=args.cell,
            pitch=args.pitch,
            wire_diameter=args.wire_diameter,
            clearance=args.clearance,
            solid_fraction=args.vf,
            warning_clearance=args.warning_clearance,
            raise_on_jamming=False,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        payload = {
            "cell_name": res.cell_name,
            "parent_network": res.parent_network,
            "pitch_mm": round(res.pitch, 4),
            "cage_diameter_mm": round(res.cage_diameter, 4),
            "wire_diameter_mm": round(res.wire_diameter, 4),
            "wire_radius_mm": round(res.wire_radius, 4),
            "clearance_mm": round(res.clearance, 4),
            "solid_fraction": round(res.solid_fraction, 6),
            "solid_fraction_percent": round(res.solid_fraction_percent, 2),
            "aspect_ratio": round(res.aspect_ratio, 6),
            "kappa": round(res.kappa, 6),
            "c_geom": round(res.c_geom, 6),
            "jamming_limit_vf": round(res.max_solid_fraction, 6),
            "jamming_limit_percent": round(res.max_solid_fraction_percent, 2),
            "warning_clearance_mm": res.warning_clearance,
            "is_feasible": res.is_feasible,
            "warnings": res.warnings,
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"\n=== PAM Dimensioning Result: {res.cell_name} ({res.parent_network.upper()}) ===")
    print(f"  Unit Cell Pitch (a):        {res.pitch:.4f} mm")
    print(f"  Cage Outer Diameter (D):    {res.cage_diameter:.4f} mm")
    print(f"  Wire Diameter (d):          {res.wire_diameter:.4f} mm ({res.wire_diameter * 1000.0:.1f} um)")
    print(f"  Wire Radius (r):            {res.wire_radius:.4f} mm")
    print(f"  Surface Clearance (Delta):  {res.clearance:.4f} mm")
    print(f"  Solid Volume Fraction (Vf): {res.solid_fraction_percent:.2f}%")
    print(f"  Aspect Ratio (d / a):       {res.aspect_ratio:.4f}")
    print(f"  Clearance Slope (kappa):    {res.kappa:.4f}")
    print(f"  Volumetric Factor (C_geom): {res.c_geom:.4f}")
    print(f"  Jamming Limit (Vf_max):     {res.max_solid_fraction_percent:.2f}%")
    print(f"  Feasible Non-Colliding:     {'YES' if res.is_feasible else 'NO'}")
    if res.warnings:
        print("\n  Warnings / DfAM Notices:")
        for w in res.warnings:
            print(f"    * {w}")
    print()


if __name__ == "__main__":
    main()
