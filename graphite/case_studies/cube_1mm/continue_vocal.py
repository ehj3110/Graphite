"""Continue Vocal LBM from cached state — diagnostic residual tracking."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from graphite.case_studies.cube_1mm.specs import (
    VOCAL_CHECK_INTERVAL,
    VOCAL_CONTINUE_EXTRA_STEPS,
    VOCAL_MA,
    VOCAL_RE,
    VOCAL_STEPS_PREVIEW,
    VOCAL_TARGET_N,
    default_output_dir,
    splitp_linear_graded_stl,
    splitp_piecewise_stl,
    vocal_cache_dir,
)
from graphite.lbm.lettuce_solver import LettuceSolver
from graphite.lbm.vocal_cache import F_DIST_NAME, cache_path_for, load_cache
from graphite.lbm.vocal_run import VocalRunConfig, extract_unit_cube_xz_slice


def _mean_uz_lu(solver: LettuceSolver) -> float:
    return float(solver.flow.u()[2].mean().item())


def _mean_uz_pu(solver: LettuceSolver) -> float:
    return float(solver.units.convert_velocity_to_pu(_mean_uz_lu(solver)))


def _peak_fluid_speed_mm_s(grid, solver: LettuceSolver) -> float:
    u = solver.get_velocity_field()
    speed = np.linalg.norm(u, axis=0)
    fluid = ~grid.solid_mask
    return float(speed[fluid].max() * 1000.0)


def continue_vocal_from_cache(
    *,
    label: str,
    stl_path: Path,
    prior_steps: int = VOCAL_STEPS_PREVIEW,
    extra_steps: int = VOCAL_CONTINUE_EXTRA_STEPS,
    check_interval: int = VOCAL_CHECK_INTERVAL,
    cache_dir: Path | None = None,
) -> None:
    cache_root = cache_dir or vocal_cache_dir()
    cfg = VocalRunConfig(
        geometry="stl",
        stl_path=stl_path,
        target_n=VOCAL_TARGET_N,
        boundary_style="flow_chamber",
        re=VOCAL_RE,
        ma=VOCAL_MA,
        steps=prior_steps,
        converge=False,
        max_steps=8000,
        tolerance=1e-3,
        apply_flow_chamber_pad=True,
        cache_dir=cache_root,
    )
    cached_path = cache_path_for(cfg, cache_root)
    loaded = load_cache(cached_path, cfg)
    if loaded is None:
        raise SystemExit(f"No steps{prior_steps} cache for {stl_path.name}")

    grid, flow_state, metrics = loaded
    f_path = cached_path / F_DIST_NAME
    if not f_path.is_file():
        raise SystemExit(f"Missing f.pkl at {f_path}")

    u_cached = flow_state.get_velocity_field()
    speed_cached = np.linalg.norm(u_cached, axis=0)
    prior_peak = float(speed_cached[~grid.solid_mask].max() * 1000.0)

    print(f"\n{'=' * 60}")
    print(f"{label}  (continuing from cache steps{prior_steps})")
    print(f"  cache: {cached_path.name}")
    print(f"  prior steps_run (cached): {metrics.steps_run}")
    print(f"  prior peak |u| (fluid, cached): {prior_peak:.2f} mm/s")

    solver = LettuceSolver(
        voxel_grid=grid,
        Re=cfg.re,
        Ma=cfg.ma,
        device="cuda",
        boundary_style=cfg.boundary_style,
        warm_start_f_path=f_path,
    )

    u0 = solver.get_velocity_field()
    peak0 = _peak_fluid_speed_mm_s(grid, solver)
    mu0 = _mean_uz_pu(solver)
    print(f"  loaded state: mean Uz = {mu0 * 1000:.3f} mm/s, peak |u| = {peak0:.2f} mm/s")
    print(f"  running +{extra_steps} steps (check every {check_interval}):")
    print(f"  {'step':>6}  {'mean Uz (mm/s)':>14}  {'rel_change':>12}  {'peak |u| (mm/s)':>16}")

    prev_mean = _mean_uz_lu(solver)
    total_logical = prior_steps
    rel_change = float("inf")

    for chunk_start in range(0, extra_steps, check_interval):
        n = min(check_interval, extra_steps - chunk_start)
        solver.step(n)
        total_logical += n

        mean_lu = _mean_uz_lu(solver)
        mean_pu = solver.units.convert_velocity_to_pu(mean_lu)
        if abs(prev_mean) > 1e-12:
            rel_change = abs(mean_lu - prev_mean) / abs(mean_lu)
        else:
            rel_change = float("inf")

        peak = _peak_fluid_speed_mm_s(grid, solver)
        print(
            f"  {total_logical:6d}  {mean_pu * 1000:14.4f}  {rel_change:12.3e}  {peak:16.2f}"
        )
        prev_mean = mean_lu

    u1 = solver.get_velocity_field()
    fluid = ~grid.solid_mask
    du = np.linalg.norm(u1 - u0, axis=0)
    rel_du = float(du[fluid].mean() / (np.linalg.norm(u0, axis=0)[fluid].mean() + 1e-12))
    _, _, _, _, _, _, slice_peak = extract_unit_cube_xz_slice(grid, solver)

    print(f"  final logical step count: {total_logical}")
    print(f"  final mean Uz residual (last interval): {rel_change:.3e}")
    print(f"  mean |delta-u|/|u0| over fluid: {rel_du:.3e}")
    print(f"  final volume peak |u|: {_peak_fluid_speed_mm_s(grid, solver):.2f} mm/s")
    print(f"  final XZ slice peak |u|: {slice_peak * 1000:.2f} mm/s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        choices=("piecewise", "graded", "both"),
        default="both",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prior-steps", type=int, default=VOCAL_STEPS_PREVIEW)
    parser.add_argument("--extra-steps", type=int, default=VOCAL_CONTINUE_EXTRA_STEPS)
    parser.add_argument("--check-interval", type=int, default=VOCAL_CHECK_INTERVAL)
    args = parser.parse_args(argv)

    out = args.out_dir or default_output_dir()
    cases: list[tuple[str, Path]] = []
    if args.cases in ("piecewise", "both"):
        cases.append(("Piecewise", splitp_piecewise_stl(out, cleaned=True)))
    if args.cases in ("graded", "both"):
        cases.append(
            (
                "Linear grading",
                splitp_linear_graded_stl(out, cleaned=True),
            )
        )

    for label, stl in cases:
        continue_vocal_from_cache(
            label=label,
            stl_path=stl,
            prior_steps=int(args.prior_steps),
            extra_steps=int(args.extra_steps),
            check_interval=int(args.check_interval),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
