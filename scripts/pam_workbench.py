"""
scripts/pam_workbench.py

Command-line tool for the Permissive PAM Pair Experimental Workbench.
Philosophy: "Always Generate, Always Diagnose".

Usage examples:
  python scripts/pam_workbench.py measure --cell-a d4cube --cell-b d4cube --gap 2.5
  python scripts/pam_workbench.py sweep --cell-a d4cube --cell-b d4tet --param twist --range 0,90,19 --out outputs/workbench_review/sweep.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphite.explicit.interlinked.workbench import measure_pair, sweep, PairReport


def cmd_measure(args: argparse.Namespace) -> None:
    t_rel = np.array([float(x) for x in args.t_rel.split(",")], dtype=np.float64) if args.t_rel else np.array([args.gap, 0.0, 0.0])
    
    rep = measure_pair(
        cell_a=args.cell_a,
        cell_b=args.cell_b,
        t_rel=t_rel,
        wire_radius=args.wire_radius,
        pitch_a=args.pitch_a,
        pitch_b=args.pitch_b,
        include_meshes=args.export_mesh,
    )

    print(f"\n=== PAM Pair Report: {rep.cell_a_name} <-> {rep.cell_b_name} ===")
    print(f"Status:            {rep.status.upper()}")
    print(f"Clearance (min):   {rep.min_clearance_mm:.4f} mm")
    print(f"Clash Volume:      {rep.clash_volume_mm3:.6f} mm^3")
    print(f"Linking Number:    |Lk| = {rep.linking_number}")
    print(f"Elapsed Time:      {rep.diagnostics.get('elapsed_s', 0.0)*1000:.2f} ms")


def cmd_sweep(args: argparse.Namespace) -> None:
    parts = [float(x) for x in args.range.split(",")]
    if len(parts) == 3:
        vals = np.linspace(parts[0], parts[1], int(parts[2]))
    else:
        vals = np.array(parts, dtype=np.float64)

    normal = np.array([float(x) for x in args.normal.split(",")], dtype=np.float64) if args.normal else np.array([1.0, 0.0, 0.0])

    print(f"\nRunning sweep: {args.cell_a} <-> {args.cell_b} along {args.param} ({len(vals)} points)...")
    results = sweep(
        cell_a=args.cell_a,
        cell_b=args.cell_b,
        param=args.param,
        values=vals,
        habit_normal=normal,
        wire_radius=args.wire_radius,
        pitch_a=args.pitch_a,
        pitch_b=args.pitch_b,
    )

    records = []
    for r in results:
        records.append({
            args.param: r.diagnostics.get("sweep_value"),
            "status": r.status,
            "min_clearance_mm": r.min_clearance_mm,
            "clash_volume_mm3": r.clash_volume_mm3,
            "linking_number": r.linking_number,
        })
    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=200)

        # Plot clearance
        ax1.plot(df[args.param], df["min_clearance_mm"], marker="o", color="#2a9d8f", label="Min Clearance (mm)")
        ax1.axhline(0.0, color="red", linestyle="--", alpha=0.6, label="Clash Boundary (0 mm)")
        ax1.set_ylabel("Clearance (mm)")
        ax1.set_title(f"PAM Workbench Sweep: {args.cell_a.upper()} <-> {args.cell_b.upper()} vs {args.param}")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="best")

        # Plot linking number
        ax2.step(df[args.param], df["linking_number"], where="mid", color="#e76f51", label="Gauss Linking |Lk|")
        ax2.set_xlabel(f"{args.param}")
        ax2.set_ylabel("Linking Number |Lk|")
        ax2.set_ylim(-0.2, max(2.0, df["linking_number"].max() + 0.5))
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best")

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PAM Pair Experimental Workbench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Measure pair
    p_meas = subparsers.add_parser("measure", help="Measure single particle pair")
    p_meas.add_argument("--cell-a", required=True, help="Cell A tag")
    p_meas.add_argument("--cell-b", required=True, help="Cell B tag")
    p_meas.add_argument("--gap", type=float, default=2.5, help="Nominal gap distance (mm)")
    p_meas.add_argument("--t-rel", help="Relative translation dx,dy,dz (mm)")
    p_meas.add_argument("--wire-radius", type=float, default=0.35, help="Wire radius (mm)")
    p_meas.add_argument("--pitch-a", type=float, default=10.0, help="Pitch A (mm)")
    p_meas.add_argument("--pitch-b", type=float, default=None, help="Pitch B (mm)")
    p_meas.add_argument("--export-mesh", action="store_true", help="Generate meshes")
    p_meas.set_defaults(func=cmd_measure)

    # Sweep
    p_swp = subparsers.add_parser("sweep", help="Sweep parameter across a range")
    p_swp.add_argument("--cell-a", required=True, help="Cell A tag")
    p_swp.add_argument("--cell-b", required=True, help="Cell B tag")
    p_swp.add_argument("--param", choices=["gap", "twist", "shift_y", "shift_z", "wire_radius"], default="gap")
    p_swp.add_argument("--range", required=True, help="min,max,steps e.g. 1.0,8.0,15")
    p_swp.add_argument("--normal", help="Habit normal nx,ny,nz")
    p_swp.add_argument("--wire-radius", type=float, default=0.35, help="Wire radius (mm)")
    p_swp.add_argument("--pitch-a", type=float, default=10.0, help="Pitch A (mm)")
    p_swp.add_argument("--pitch-b", type=float, default=None, help="Pitch B (mm)")
    p_swp.add_argument("--csv", help="Output CSV path")
    p_swp.add_argument("--out", help="Output plot PNG path")
    p_swp.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
