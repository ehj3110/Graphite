#!/usr/bin/env python
"""
Refine linear graded Split-P Vocal run: n=128 → n=256 warm-start.

Upsamples velocity + pressure from the converged n=128 upflow cache onto a
finer voxel grid, then runs until mean-Uz convergence (or max_steps cap).

Run from repo root with ``.venv_torch``::

    .venv_torch\\Scripts\\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py
    .venv_torch\\Scripts\\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.lbm.vocal_cache import cache_path_for
from graphite.lbm.vocal_run import VocalRunConfig, run_vocal

CACHE_ROOT = _REPO_ROOT / "outputs" / "vocal" / "cache"
CASE_ROOT = _REPO_ROOT / "outputs" / "case_studies" / "cube_1mm"
FIGURES_DIR = CASE_ROOT / "figures"
METRICS_DIR = CASE_ROOT / "fluid" / "vocal_metrics"

STL = CASE_ROOT / (
    "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_"
    "phaseOrigin500um500um_JacobianW_cleaned.stl"
)
SEED_CACHE_N128 = CACHE_ROOT / (
    "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_"
    "phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_"
    "re5_ma0.05_conv_ms6000_tol0.001"
)

DEFAULT_TARGET_N = 256
MAX_STEPS = 6000
MIN_STEPS = 1000
TOLERANCE = 0.001
CHECK_INTERVAL = 250


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if not STL.is_file():
        print(f"[ERROR] STL not found: {STL}", file=sys.stderr)
        return 1
    if not SEED_CACHE_N128.is_dir():
        print(f"[ERROR] n=128 seed cache not found: {SEED_CACHE_N128}", file=sys.stderr)
        return 1

    target_n = int(args.target_n)
    metrics_json = METRICS_DIR / f"SplitP_linearGrad_n{target_n}_refine_metrics.json"
    preview_png = FIGURES_DIR / f"SplitP_Cube1mm_linearGrad_n{target_n}_vocal_XZ_midY.png"

    cfg = VocalRunConfig(
        geometry="stl",
        stl_path=STL,
        target_n=target_n,
        boundary_style="flow_chamber",
        re=5.0,
        ma=0.05,
        steps=MAX_STEPS,
        converge=True,
        max_steps=MAX_STEPS,
        min_steps=MIN_STEPS,
        check_interval=CHECK_INTERVAL,
        tolerance=TOLERANCE,
        warm_start="cache",
        warm_start_cache_dir=SEED_CACHE_N128,
        cache_dir=CACHE_ROOT,
        use_cache=True,
        force_rerun=bool(args.force_rerun),
        plot=True,
        output_png=preview_png,
        metrics_json=metrics_json,
        device=str(args.device),
        plot_title=f"Split-P linear grad — Vocal n={target_n} (warm-start n=128)",
    )
    out_cache = cache_path_for(cfg, CACHE_ROOT)

    print("=" * 72)
    print("Cube 1 mm — linear Split-P Vocal refine n=128 -> n=256")
    print(f"  STL:        {STL.name}")
    print(f"  Seed cache: {SEED_CACHE_N128.name}")
    print(f"  Target n:   {target_n}")
    print(f"  Output:     {out_cache}")
    print(f"  Preview:    {preview_png}")
    print(f"  Metrics:    {metrics_json}")
    print("=" * 72)

    if args.dry_run:
        return 0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    _grid, _flow, metrics = run_vocal(cfg)
    status = "converged" if metrics.converged else f"cap @ {metrics.steps_run} steps"
    print(f"\nDone: {status}")
    print(f"  k = {metrics.permeability_mm2:.6e} mm²")
    print(f"  WSS max = {metrics.wss_max_pa:.1f} Pa")
    print(f"  Cache: {out_cache}")

    summary = {
        "phase": "n256_refine_linear_splitp",
        "seed_cache": str(SEED_CACHE_N128),
        "output_cache": str(out_cache),
        "target_n": target_n,
        "converged": metrics.converged,
        "steps_run": metrics.steps_run,
        "permeability_mm2": metrics.permeability_mm2,
        "wss_max_pa": metrics.wss_max_pa,
        "preview_png": str(preview_png),
    }
    handoff_path = CASE_ROOT / "fluid" / "n256_linear_splitp_status.json"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Status: {handoff_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
