"""Vocal LBM pipeline for piecewise cross-hatch woodpile 1 mm³ cube."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from graphite.case_studies.cube_1mm.specs import (
    VOCAL_MA,
    VOCAL_RE,
    VOCAL_STEPS_PREVIEW,
    VOCAL_TARGET_N,
    default_output_dir,
    repo_root,
    resolve_woodpile_stem,
    scripts_dir,
    torch_python,
    vocal_cache_dir,
    vocal_metrics_dir,
)


def _run(cmd: list[str], *, desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root(), check=True)


def run_woodpile_vocal(
    *,
    out_dir: Path | None = None,
    match_splitp_pores: bool = False,
    stem: str | None = None,
    steps: int = VOCAL_STEPS_PREVIEW,
    target_n: int = VOCAL_TARGET_N,
    re: float = VOCAL_RE,
    ma: float = VOCAL_MA,
    warm_start: str = "auto",
    force_rerun: bool = False,
    converge: bool = False,
    max_steps: int = 15000,
    check_interval: int = 250,
    tolerance: float = 1e-3,
    plot_only: bool = False,
    quiver_stride: int = 4,
) -> int:
    out = out_dir or default_output_dir()
    metrics_dir = vocal_metrics_dir(out)
    stem_resolved = resolve_woodpile_stem(match_splitp_pores=match_splitp_pores, stem=stem)
    stl = out / f"{stem_resolved}_cleaned.stl"
    png = out / f"{stem_resolved}_vocal_cross_section_XZ_midY.png"
    metrics_json = metrics_dir / f"{stem_resolved}_vocal_{target_n}_steps{steps}_metrics.json"
    if converge:
        metrics_json = metrics_dir / f"{stem_resolved}_vocal_{target_n}_converge_metrics.json"
    cache_dir = vocal_cache_dir()

    if plot_only:
        from graphite.lbm.vocal_cache import cache_path_for, load_cache
        from graphite.lbm.vocal_run import VocalRunConfig, plot_xz_velocity_slice

        cfg = VocalRunConfig(
            geometry="stl",
            stl_path=stl,
            target_n=target_n,
            boundary_style="flow_chamber",
            re=re,
            ma=ma,
            steps=steps,
            converge=converge,
            max_steps=max_steps,
            tolerance=tolerance,
            cache_dir=cache_dir,
        )
        cached = load_cache(cache_path_for(cfg, cache_dir), cfg)
        if cached is None:
            print(f"No cache for {stl.name} @ steps={steps}", file=sys.stderr)
            return 1
        grid, flow, _metrics = cached
        plot_xz_velocity_slice(
            grid,
            flow,
            output_path=png,
            title="Cross-hatch woodpile — Vocal steady-state flow (XZ slice)",
            quiver_stride=quiver_stride,
        )
        print(f"Wrote {png} (from cache)")
        return 0

    if not stl.is_file():
        print(f"Cleaned STL missing: {stl}", file=sys.stderr)
        print(
            "Run: python scripts/run_cube_1mm_woodpile_aristo.py "
            "--match-splitp-pores --skip-fea"
        )
        return 1

    metrics_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(torch_python()),
        str(scripts_dir() / "run_vocal.py"),
        "--geometry",
        "stl",
        "--stl",
        str(stl),
        "--target-n",
        str(target_n),
        "--boundary-style",
        "flow_chamber",
        "--re",
        str(re),
        "--ma",
        str(ma),
        "--warm-start",
        warm_start,
        "--output",
        str(png),
        "--metrics-json",
        str(metrics_json),
        "--title",
        "Cross-hatch woodpile — Vocal steady-state flow (XZ slice)",
        "--quiver-stride",
        str(quiver_stride),
        "--cache-dir",
        str(cache_dir),
    ]
    if converge:
        cmd.extend(
            [
                "--converge",
                "--max-steps",
                str(max_steps),
                "--check-interval",
                str(check_interval),
                "--tolerance",
                str(tolerance),
            ]
        )
    else:
        cmd.extend(["--steps", str(steps)])
    if force_rerun:
        cmd.append("--force-rerun")

    _run(cmd, desc="Vocal LBM on woodpile cube")

    if metrics_json.is_file():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        print("\n--- Vocal metrics ---")
        print(f"  steps_run:     {data.get('steps_run')}")
        print(f"  converged:     {data.get('converged')}")
        print(f"  permeability:  {data.get('permeability_mm2')} mm²")
        print(f"  pressure_drop: {data.get('pressure_drop_pa')} Pa")
        print(f"  WSS max:       {data.get('wss_max_pa')} Pa")
        print(f"  solid_fraction:{data.get('solid_fraction')}")

    print(f"\nWrote {png}")
    print(f"Wrote {metrics_json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=VOCAL_STEPS_PREVIEW)
    parser.add_argument("--target-n", type=int, default=VOCAL_TARGET_N)
    parser.add_argument("--re", type=float, default=VOCAL_RE)
    parser.add_argument("--ma", type=float, default=VOCAL_MA)
    parser.add_argument("--warm-start", choices=("none", "cache", "analytic", "auto"), default="auto")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--converge", action="store_true")
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--match-splitp-pores", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_woodpile_vocal(
        out_dir=args.out_dir or default_output_dir(),
        match_splitp_pores=bool(args.match_splitp_pores),
        stem=args.stem,
        steps=int(args.steps),
        target_n=int(args.target_n),
        re=float(args.re),
        ma=float(args.ma),
        warm_start=args.warm_start,
        force_rerun=bool(args.force_rerun),
        converge=bool(args.converge),
        max_steps=int(args.max_steps),
        check_interval=int(args.check_interval),
        tolerance=float(args.tolerance),
        plot_only=bool(args.plot_only),
        quiver_stride=int(args.quiver_stride),
    )


if __name__ == "__main__":
    raise SystemExit(main())
