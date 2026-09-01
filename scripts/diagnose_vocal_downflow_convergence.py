#!/usr/bin/env python
"""
Quick convergence diagnostic for downflow (flow_chamber_reverse) Vocal caches.

Loads cached ``f.pkl`` and runs a small number of additional steps in-memory,
printing the mean Uz relative change every check interval.

This does NOT write a new cache (diagnostic only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "outputs" / "vocal" / "cache"


def _mean_uz_lu(solver) -> float:
    return float(solver.flow.u()[2].mean().item())


def _peak_speed_mm_s(grid, solver) -> float:
    u = solver.get_velocity_field()
    speed = np.linalg.norm(u, axis=0)
    fluid = ~grid.solid_mask
    return float(speed[fluid].max() * 1000.0)


def diagnose(cache_dir: Path, *, extra_steps: int, check_interval: int, device: str) -> None:
    from graphite.lbm.vocal_cache import load_solver_from_cache_dir, load_metrics_from_cache_dir

    cache_dir = Path(cache_dir)
    grid, solver = load_solver_from_cache_dir(cache_dir, device=device)
    metrics = load_metrics_from_cache_dir(cache_dir)

    print("\n" + "=" * 72)
    print(f"Cache: {cache_dir.name}")
    print(f"Cached steps_run: {metrics.steps_run}   converged: {metrics.converged}")
    print(f"Cached dp: {metrics.pressure_drop_pa:.3f} Pa   WSS max: {metrics.wss_max_pa:.1f} Pa")

    mean0 = _mean_uz_lu(solver)
    peak0 = _peak_speed_mm_s(grid, solver)
    print(f"Loaded state: mean Uz (LU)={mean0:.6e}   peak |u|={peak0:.2f} mm/s")
    print(f"Running +{extra_steps} steps (check every {check_interval})")
    print(f"{'step':>8}  {'meanUz(LU)':>12}  {'rel_change':>12}  {'peak|u| mm/s':>14}")

    prev = mean0
    rel_change = float("inf")
    total = 0
    for _ in range(0, extra_steps, check_interval):
        n = min(check_interval, extra_steps - total)
        solver.step(n)
        total += n
        cur = _mean_uz_lu(solver)
        if abs(prev) > 1e-12:
            rel_change = abs(cur - prev) / abs(cur)
        else:
            rel_change = float("inf")
        peak = _peak_speed_mm_s(grid, solver)
        print(f"{total:8d}  {cur:12.5e}  {rel_change:12.3e}  {peak:14.2f}")
        prev = cur

    print(f"Last-interval residual estimate: {rel_change:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-steps", type=int, default=500)
    parser.add_argument("--check-interval", type=int, default=50)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--piecewise-cache",
        type=Path,
        default=CACHE_ROOT
        / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_reverse_re5_ma0.05_steps2000",
    )
    parser.add_argument(
        "--linear-cache",
        type=Path,
        default=CACHE_ROOT
        / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_reverse_re5_ma0.05_steps2000",
    )
    args = parser.parse_args()

    diagnose(
        args.piecewise_cache,
        extra_steps=int(args.extra_steps),
        check_interval=int(args.check_interval),
        device=str(args.device),
    )
    diagnose(
        args.linear_cache,
        extra_steps=int(args.extra_steps),
        check_interval=int(args.check_interval),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

