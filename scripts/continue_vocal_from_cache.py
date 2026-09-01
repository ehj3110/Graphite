#!/usr/bin/env python
"""Continue a Vocal cache from f.pkl for N steps; print residuals and save."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CACHE_ROOT = _REPO_ROOT / "outputs" / "vocal" / "cache"
OUT_DIR = _REPO_ROOT / "outputs" / "case_studies" / "cube_1mm"
METRICS_DIR = OUT_DIR / "fluid" / "vocal_metrics"


def _fluid_velocity(solver, grid) -> np.ndarray:
    u = solver.get_velocity_field()
    fluid = ~grid.solid_mask
    return u[:, fluid]


def _report_block(
    *,
    total_steps: int,
    mean_uz: float,
    prev_mean_uz: float | None,
    u_prev: np.ndarray | None,
    u_cur: np.ndarray,
) -> dict:
    if prev_mean_uz is not None and abs(mean_uz) > 1e-12:
        mean_uz_rel = abs(mean_uz - prev_mean_uz) / abs(mean_uz)
    else:
        mean_uz_rel = float("inf")

    if u_prev is not None and u_cur.size:
        du = u_cur - u_prev
        u_norm = np.linalg.norm(u_cur)
        du_norm = np.linalg.norm(du)
        vel_l2_rel = du_norm / (u_norm + 1e-15)
        vel_max_abs = float(np.max(np.abs(du)))
    else:
        vel_l2_rel = float("inf")
        vel_max_abs = float("nan")

    print(
        f"{total_steps:6d}  meanUz_rel={mean_uz_rel:10.3e}  "
        f"vel_L2_rel={vel_l2_rel:10.3e}  max|du|_mm_s={vel_max_abs * 1000.0:10.3f}",
        flush=True,
    )
    return {
        "step": total_steps,
        "mean_uz_lu": mean_uz,
        "mean_uz_rel": mean_uz_rel,
        "velocity_l2_rel": vel_l2_rel,
        "max_du_mm_s": vel_max_abs * 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--extra-steps", type=int, default=3000)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--until-converged",
        action="store_true",
        help="Stop early when mean-Uz relative change falls below --tolerance.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--save-steps", type=int, default=None, help="Total steps label for output cache slug.")
    parser.add_argument("--output-cache-suffix", type=str, default=None)
    args = parser.parse_args()

    from graphite.lbm.vocal_cache import (
        cache_path_for,
        load_metrics_from_cache_dir,
        load_solver_from_cache_dir,
        save_cache,
    )
    from graphite.lbm.vocal_run import VocalRunConfig, collect_metrics

    cache_dir = Path(args.cache_dir)
    prior = load_metrics_from_cache_dir(cache_dir)
    grid, solver = load_solver_from_cache_dir(cache_dir, device=str(args.device))

    print("=" * 72)
    print(f"Source cache: {cache_dir.name}")
    print(f"Prior steps_run: {prior.steps_run}   converged: {prior.converged}")
    print(f"Boundary: {prior.boundary_style}")
    print(f"Running +{args.extra_steps} steps (check every {args.check_interval})")
    print(f"{'step':>6}  {'meanUz_rel':>12}  {'vel_L2_rel':>12}  {'max|du| mm/s':>14}")

    history: list[dict] = []
    prev_mean = float(solver.flow.u()[2].mean().item())
    u_prev = _fluid_velocity(solver, grid)
    total = 0
    extra = int(args.extra_steps)
    interval = int(args.check_interval)
    converged = False

    for _ in range(0, extra, interval):
        n = min(interval, extra - total)
        solver.step(n)
        total += n
        mean_uz = float(solver.flow.u()[2].mean().item())
        u_cur = _fluid_velocity(solver, grid)
        row = _report_block(
            total_steps=total,
            mean_uz=mean_uz,
            prev_mean_uz=prev_mean,
            u_prev=u_prev,
            u_cur=u_cur,
        )
        history.append(row)
        prev_mean = mean_uz
        u_prev = u_cur.copy()
        if args.until_converged and row["mean_uz_rel"] < float(args.tolerance):
            converged = True
            print(
                f"Converged after +{total} steps "
                f"(meanUz_rel={row['mean_uz_rel']:.3e} < {args.tolerance:g}).",
                flush=True,
            )
            break

    metrics = collect_metrics(grid, solver, _config_from_cache(cache_dir), converged=converged)
    metrics.steps_run = prior.steps_run + total
    metrics.converged = converged
    metrics.notes = list(prior.notes) + [f"continued +{total} steps from {cache_dir.name}"]

    total_steps_label = args.save_steps or metrics.steps_run
    if args.output_cache_suffix:
        out_dir = CACHE_ROOT / f"{cache_dir.name}_{args.output_cache_suffix}"
    elif converged:
        out_cfg = _config_from_cache(
            cache_dir,
            converge=True,
            max_steps=total_steps_label,
            tolerance=float(args.tolerance),
        )
        out_dir = cache_path_for(out_cfg, CACHE_ROOT)
    else:
        out_cfg = _config_from_cache(cache_dir, steps=total_steps_label, converge=False)
        out_dir = cache_path_for(out_cfg, CACHE_ROOT)

    velocity = solver.get_velocity_field()
    pressure = solver.get_pressure_field()
    if args.output_cache_suffix:
        save_cfg = _config_from_cache(cache_dir, steps=total_steps_label, converge=False)
    elif converged:
        save_cfg = _config_from_cache(
            cache_dir,
            converge=True,
            max_steps=total_steps_label,
            tolerance=float(args.tolerance),
        )
    else:
        save_cfg = _config_from_cache(cache_dir, steps=total_steps_label, converge=False)
    save_cache(out_dir, save_cfg, grid, velocity, metrics, flow=solver.flow, pressure=pressure)
    print(f"\nSaved: {out_dir}")
    print(f"Total steps_run: {metrics.steps_run}")
    print(f"Pressure drop: {metrics.pressure_drop_pa:.4f} Pa")
    print(f"WSS max: {metrics.wss_max_pa:.1f} Pa")

    report_path = METRICS_DIR / f"{cache_dir.stem}_continue_{total}_residuals.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "source_cache": str(cache_dir),
                "prior_steps": prior.steps_run,
                "extra_steps": total,
                "check_interval": interval,
                "history": history,
                "final": history[-1] if history else {},
                "output_cache": str(out_dir),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"Residual report: {report_path}")
    return 0


def _config_from_cache(
    cache_dir: Path,
    *,
    steps: int | None = None,
    converge: bool | None = None,
    max_steps: int | None = None,
    tolerance: float | None = None,
):
    from graphite.lbm.vocal_run import VocalRunConfig

    manifest = json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
    fp = manifest["fingerprint"]
    stl_path = Path(fp["stl_path"])
    return VocalRunConfig(
        geometry=str(fp["geometry"]),
        stl_path=stl_path,
        target_n=int(fp["target_n"]),
        boundary_style=str(fp["boundary_style"]),
        re=float(fp["re"]),
        ma=float(fp["ma"]),
        steps=int(steps if steps is not None else fp.get("steps", 500)),
        converge=bool(fp["converge"] if converge is None else converge),
        max_steps=int(max_steps if max_steps is not None else fp.get("max_steps", 6000)),
        tolerance=float(tolerance if tolerance is not None else fp.get("tolerance", 1e-3)),
        apply_flow_chamber_pad=bool(fp.get("apply_flow_chamber_pad", True)),
        cache_dir=CACHE_ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
