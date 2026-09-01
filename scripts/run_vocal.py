#!/usr/bin/env python
"""
Vocal — Lattice Boltzmann fluid simulation CLI.

Run from repo root with the torch venv::

    .venv_torch\\Scripts\\python.exe scripts\\run_vocal.py --health-check
    .venv_torch\\Scripts\\python.exe scripts\\run_vocal.py --geometry stl --stl path.stl --converge

See docs/VOCAL.md for full usage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.lbm.vocal_run import VocalRunConfig, run_vocal

_DEFAULT_STL = (
    _REPO_ROOT
    / "experiments/implicit_to_volume/output"
    / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl"
)


def _run_health_check() -> int:
    python = sys.executable
    print("Vocal health check")
    print(f"  Python: {python}")

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  PyTorch: {torch.__version__}  CUDA available: {cuda}")
        if cuda:
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as exc:
        print(f"  [FAIL] PyTorch not installed: {exc}")
        return 1

    try:
        import lettuce as lt  # noqa: F401

        print(f"  Lettuce: OK")
    except ImportError as exc:
        print(f"  [FAIL] Lettuce not installed: {exc}")
        return 1

    print("\nRunning unit tests (tests/test_lettuce_solver.py)...")
    result = subprocess.run(
        [python, "-m", "pytest", "tests/test_lettuce_solver.py", "-q"],
        cwd=_REPO_ROOT,
    )
    if result.returncode != 0:
        print("[FAIL] Unit tests failed.")
        return result.returncode

    print("[PASS] Vocal health check complete.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-check", action="store_true", help="Run pytest health check and exit.")

    parser.add_argument(
        "--geometry",
        choices=("stl", "tpms", "graded-tpms"),
        default="stl",
        help="Geometry source (default: stl).",
    )
    parser.add_argument("--stl", type=Path, default=_DEFAULT_STL, help="STL path (geometry=stl).")
    parser.add_argument("--lattice-type", default="gyroid", help="TPMS type (tpms / graded-tpms).")
    parser.add_argument("--unit-cell-size-mm", type=float, default=1.0)
    parser.add_argument("--solid-fraction", type=float, default=0.30)
    parser.add_argument(
        "--domain-size-mm",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        default=(2.0, 2.0, 2.0),
    )
    parser.add_argument("--sf-inlet", type=float, default=0.20, help="Graded TPMS inlet solid fraction.")
    parser.add_argument("--sf-outlet", type=float, default=0.60, help="Graded TPMS outlet solid fraction.")
    parser.add_argument("--target-n", type=int, default=128, help="Voxels along longest axis.")
    parser.add_argument(
        "--no-flow-chamber-pad",
        action="store_true",
        help="Skip pad_flow_chamber_mask on STL / graded grids.",
    )

    parser.add_argument(
        "--boundary-style",
        choices=("periodic", "flow_chamber", "flow_chamber_reverse"),
        default=None,
        help="BC style (default: flow_chamber for stl/graded-tpms, periodic for tpms).",
    )
    parser.add_argument("--re", type=float, default=5.0)
    parser.add_argument("--ma", type=float, default=0.05)
    parser.add_argument("--acceleration-z", type=float, default=1e-4, help="Body-force (periodic BC).")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")

    step = parser.add_mutually_exclusive_group()
    step.add_argument("--steps", type=int, default=500, help="Fixed LBM steps (default when not converging).")
    step.add_argument("--converge", action="store_true", help="Run until mean Uz converges.")
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--check-interval", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-4)

    parser.add_argument("--output", type=Path, default=None, help="XZ velocity PNG path.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Write metrics JSON.")
    parser.add_argument("--title", default=None, help="Plot title override.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument(
        "--warm-start",
        choices=("none", "cache", "analytic", "auto"),
        default="auto",
        help="Initial condition: cache (f or velocity), analytic guess, auto, or cold none.",
    )

    parser.add_argument("--no-cache", action="store_true", help="Disable disk cache.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cache and re-simulate.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "vocal" / "cache",
        help="Vocal cache root directory.",
    )

    args = parser.parse_args()

    if args.health_check:
        return _run_health_check()

    boundary_style = args.boundary_style
    if boundary_style is None:
        boundary_style = "periodic" if args.geometry == "tpms" else "flow_chamber"

    if args.geometry == "stl" and not args.stl.exists():
        print(f"[ERROR] STL not found: {args.stl}", file=sys.stderr)
        return 1

    if args.output is None:
        stem = {
            "stl": args.stl.stem if args.geometry == "stl" else "tpms",
            "tpms": f"{args.lattice_type}_{args.target_n}",
            "graded-tpms": f"{args.lattice_type}_graded_{args.target_n}",
        }[args.geometry]
        args.output = _REPO_ROOT / "outputs" / "vocal" / f"{stem}_flow.png"

    config = VocalRunConfig(
        geometry=args.geometry,
        stl_path=args.stl,
        lattice_type=args.lattice_type,
        unit_cell_size_mm=args.unit_cell_size_mm,
        solid_fraction=args.solid_fraction,
        domain_size_mm=tuple(args.domain_size_mm),
        sf_inlet=args.sf_inlet,
        sf_outlet=args.sf_outlet,
        target_n=args.target_n,
        apply_flow_chamber_pad=not args.no_flow_chamber_pad,
        boundary_style=boundary_style,
        re=args.re,
        ma=args.ma,
        acceleration_z=args.acceleration_z,
        device=args.device,
        steps=args.steps,
        converge=args.converge,
        max_steps=args.max_steps,
        check_interval=args.check_interval,
        tolerance=args.tolerance,
        quiver_stride=args.quiver_stride,
        plot_title=args.title,
        output_png=None if args.no_plot else args.output,
        metrics_json=args.metrics_json,
        plot=not args.no_plot,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        force_rerun=args.force_rerun,
        warm_start=args.warm_start,
    )

    run_vocal(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
