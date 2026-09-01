#!/usr/bin/env python
"""
Side-by-side Vocal flow comparison (CFD-style XZ slices).

Quick default: 1000 fixed steps per case (not convergence). Results are cached
under ``outputs/vocal/cache/``.

Run from repo root with ``.venv_torch``::

    .venv_torch\\Scripts\\python.exe scripts\\plot_vocal_flow_comparison.py --cases both --steps 1000
    .venv_torch\\Scripts\\python.exe scripts\\plot_vocal_flow_comparison.py --cases both --plot-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.case_studies.cube_1mm.specs import (
    default_output_dir,
    splitp_linear_graded_stl,
    splitp_piecewise_stl,
    vocal_comparison_png,
    vocal_comparison_summary_json,
)
from graphite.lbm.vocal_cache import cache_path_for, load_cache
from graphite.lbm.vocal_run import VocalRunConfig, plot_flow_comparison, run_vocal_comparison

_DEFAULT_OUT = default_output_dir()
_PIECEWISE_STL = splitp_piecewise_stl(_DEFAULT_OUT, cleaned=True)
_GRADED_STL = splitp_linear_graded_stl(_DEFAULT_OUT, cleaned=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        choices=("piecewise", "graded", "both"),
        default="both",
        help="Which scaffolds to include (default: both).",
    )
    parser.add_argument("--piecewise-stl", type=Path, default=_PIECEWISE_STL)
    parser.add_argument("--graded-stl", type=Path, default=_GRADED_STL)
    parser.add_argument(
        "--output",
        type=Path,
        default=vocal_comparison_png(_DEFAULT_OUT),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=vocal_comparison_summary_json(_DEFAULT_OUT),
    )
    parser.add_argument("--target-n", type=int, default=128)
    parser.add_argument("--re", type=float, default=5.0)
    parser.add_argument("--ma", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1000, help="Fixed LBM steps (quick default).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--warm-start",
        choices=("none", "cache", "analytic", "auto"),
        default="auto",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "vocal" / "cache",
    )

    converge = parser.add_mutually_exclusive_group()
    converge.add_argument("--converge", action="store_true", help="Use convergence instead of fixed steps.")
    converge.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from cached results (no LBM re-run).",
    )
    parser.add_argument("--quiver-stride", type=int, default=4)

    args = parser.parse_args()

    stl_map = {
        "piecewise": [("Piecewise", args.piecewise_stl)],
        "graded": [("Linear grading", args.graded_stl)],
        "both": [
            ("Piecewise", args.piecewise_stl),
            ("Linear grading", args.graded_stl),
        ],
    }
    selected = stl_map[args.cases]
    for _, path in selected:
        if not path.is_file():
            print(f"[ERROR] STL not found: {path}", file=sys.stderr)
            return 1

    metrics_dir = args.output.parent / "vocal_metrics"
    cases = tuple(
        {
            "label": label,
            "stl_path": stl_path,
            "metrics_json": metrics_dir / f"{stl_path.stem}_metrics.json",
        }
        for label, stl_path in selected
    )

    base = VocalRunConfig(
        geometry="stl",
        target_n=args.target_n,
        boundary_style="flow_chamber",
        re=args.re,
        ma=args.ma,
        steps=args.steps,
        converge=args.converge,
        max_steps=args.max_steps,
        check_interval=args.check_interval,
        tolerance=args.tolerance,
        device=args.device,
        plot=False,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        force_rerun=args.force_rerun,
        warm_start=args.warm_start,
        quiver_stride=args.quiver_stride,
    )

    print(f"Output PNG: {args.output}")

    if args.plot_only:
        plot_cases = []
        for case in cases:
            cfg = VocalRunConfig(
                geometry="stl",
                stl_path=case["stl_path"],
                target_n=args.target_n,
                boundary_style="flow_chamber",
                re=args.re,
                ma=args.ma,
                steps=args.steps,
                converge=args.converge,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
                cache_dir=args.cache_dir,
            )
            cached = load_cache(cache_path_for(cfg, args.cache_dir), cfg)
            if cached is None:
                print(f"[ERROR] No cache for {case['stl_path'].name}", file=sys.stderr)
                return 1
            grid, flow, _ = cached
            plot_cases.append({"label": case["label"], "grid": grid, "flow": flow})
        plot_flow_comparison(
            tuple(plot_cases),
            output_path=args.output,
            quiver_stride=args.quiver_stride,
        )
        print(f"Wrote {args.output} (from cache)")
        return 0

    print(f"Mode: {'converge' if args.converge else f'{args.steps} fixed steps'}")
    run_vocal_comparison(
        cases,
        base_config=base,
        output_png=args.output,
        summary_json=args.summary_json if len(cases) > 1 else None,
    )
    print(f"Wrote {args.output}")
    if len(cases) > 1:
        print(f"Wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
