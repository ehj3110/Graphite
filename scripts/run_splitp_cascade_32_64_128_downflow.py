#!/usr/bin/env python
"""
Split-P downflow cascade: n=32 → n=64 → n=128 (converge-or-cap).

Warm-starts the coarsest stage from the current fine (n=128) downflow caches
by downsampling velocity, then upsamples stage-to-stage.

Default schedule (both Split-P cases, downflow only)::

    n=32  : up to 5000 steps or converge
    n=64  : up to 6000 steps or converge
    n=128 : up to 6000 steps or converge

Does **not** run upward flow.

Run with ``.venv_torch`` from repo root when ready::

    .venv_torch\\Scripts\\python.exe scripts\\run_splitp_cascade_32_64_128_downflow.py
    .venv_torch\\Scripts\\python.exe scripts\\run_splitp_cascade_32_64_128_downflow.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.lbm.vocal_cache import cache_path_for
from graphite.lbm.vocal_run import VocalRunConfig, run_vocal

CACHE_ROOT = _REPO_ROOT / "outputs" / "vocal" / "cache"
OUT_DIR = _REPO_ROOT / "outputs" / "case_studies" / "cube_1mm"
METRICS_DIR = OUT_DIR / "vocal_metrics"

FLOW_CHAMBER_PAD_XY = 1
FLOW_CHAMBER_PAD_Z = 3

# Current fine downflow caches (coarse→fine n128 results) used to seed n=32.
DEFAULT_SEED_CACHES = {
    "piecewise": CACHE_ROOT
    / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_reverse_re5_ma0.05_steps2500",
    "linear": CACHE_ROOT
    / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_reverse_re5_ma0.05_steps2500",
}

CASES = (
    {
        "key": "piecewise",
        "label": "Split-P piecewise (bottom X +L/4)",
        "stl": OUT_DIR
        / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned.stl",
    },
    {
        "key": "linear",
        "label": "Split-P linear grading",
        "stl": OUT_DIR
        / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl",
    },
)

# Stage schedule: (target_n, max_steps)
DEFAULT_STAGES = (
    (32, 5000),
    (64, 6000),
    (128, 6000),
)


def _run_stage(
    *,
    stl: Path,
    target_n: int,
    max_steps: int,
    warm_start_cache_dir: Path | None,
    metrics_json: Path,
    force_rerun: bool,
    tolerance: float,
    check_interval: int,
    device: str,
) -> Path:
    """Run one cascade stage: converge if possible, else stop at max_steps."""
    cfg = VocalRunConfig(
        geometry="stl",
        stl_path=stl,
        target_n=target_n,
        boundary_style="flow_chamber_reverse",
        re=5.0,
        ma=0.05,
        steps=max_steps,  # unused when converge=True, kept for fingerprint clarity
        converge=True,
        max_steps=max_steps,
        check_interval=check_interval,
        tolerance=tolerance,
        warm_start="cache" if warm_start_cache_dir is not None else "analytic",
        warm_start_cache_dir=warm_start_cache_dir,
        cache_dir=CACHE_ROOT,
        use_cache=True,
        force_rerun=force_rerun,
        plot=False,
        metrics_json=metrics_json,
        device=device,
    )
    _grid, _flow, metrics = run_vocal(cfg)
    cache = cache_path_for(cfg, CACHE_ROOT)
    status = "converged" if metrics.converged else f"max_steps={metrics.steps_run}"
    print(f"  Stage n={target_n} done: {status}  ->  {cache.name}")
    return cache


def _existing_stage_cache(
    *,
    stl: Path,
    target_n: int,
    max_steps: int,
    tolerance: float,
) -> Path | None:
    """Return existing converge-cache path for a stage if present on disk."""
    cfg = VocalRunConfig(
        geometry="stl",
        stl_path=stl,
        target_n=target_n,
        boundary_style="flow_chamber_reverse",
        re=5.0,
        ma=0.05,
        converge=True,
        max_steps=max_steps,
        tolerance=tolerance,
        cache_dir=CACHE_ROOT,
    )
    cache = cache_path_for(cfg, CACHE_ROOT)
    if cache.is_dir() and (cache / "velocity.npy").is_file() and (cache / "metrics.json").is_file():
        return cache
    return None


def _load_slice(cache_dir: Path):
    velocity = np.load(cache_dir / "velocity.npy")
    solid = np.load(cache_dir / "solid_mask.npy")
    metrics = json.loads((cache_dir / "metrics.json").read_text(encoding="utf-8"))
    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    nx, ny, nz = solid.shape
    j_mid = ny // 2
    i0, i1 = FLOW_CHAMBER_PAD_XY, nx - FLOW_CHAMBER_PAD_XY
    k0, k1 = FLOW_CHAMBER_PAD_Z, nz - FLOW_CHAMBER_PAD_Z
    speed_sl = speed[i0:i1, j_mid, k0:k1]
    u_x_sl = velocity[0, i0:i1, j_mid, k0:k1]
    u_z_sl = velocity[2, i0:i1, j_mid, k0:k1]
    mask_sl = solid[i0:i1, j_mid, k0:k1]
    return speed_sl, u_x_sl, u_z_sl, mask_sl, metrics


def _draw_panel(ax, speed_sl, u_x_sl, u_z_sl, mask_sl, *, title: str, quiver_stride: int = 4):
    n_x, n_z = speed_sl.shape
    extent = [0, n_x, 0, n_z]
    fluid = ~mask_sl
    peak = float(np.max(speed_sl[fluid])) if np.any(fluid) else 0.0
    if peak <= 0.0:
        ax.set_title(f"{title}\n(no fluid)")
        ax.axis("off")
        return
    speed_norm = np.ma.masked_where(mask_sl, speed_sl / (peak + 1e-15))
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="white")
    ax.imshow(
        speed_norm.T,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    mask_rgba = np.zeros((n_x, n_z, 4))
    mask_rgba[mask_sl] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(mask_rgba.transpose((1, 0, 2)), origin="lower", extent=extent)
    x = np.arange(n_x)
    z = np.arange(n_z)
    xg, zg = np.meshgrid(x, z, indexing="ij")
    xs = xg[::quiver_stride, ::quiver_stride]
    zs = zg[::quiver_stride, ::quiver_stride]
    uxs = u_x_sl[::quiver_stride, ::quiver_stride]
    uzs = u_z_sl[::quiver_stride, ::quiver_stride]
    fs = ~mask_sl[::quiver_stride, ::quiver_stride]
    ax.quiver(xs[fs] + 0.5, zs[fs] + 0.5, uxs[fs], uzs[fs], color="white", pivot="middle")
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_z)
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")
    ax.set_title(f"{title}\nSlice peak |u| = {peak * 1000:.2f} mm/s", fontsize=10)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0.0, 1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=r"Normalized $|u|$")


def _plot_comparison(fine_caches: list[tuple[str, Path]], output: Path, dpi: int = 200) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), constrained_layout=True)
    for ax, (label, cache) in zip(axes, fine_caches, strict=True):
        speed_sl, u_x_sl, u_z_sl, mask_sl, metrics = _load_slice(cache)
        steps = int(metrics.get("steps_run", 0))
        conv = "converged" if metrics.get("converged") else f"{steps} steps"
        _draw_panel(
            ax,
            speed_sl,
            u_x_sl,
            u_z_sl,
            mask_sl,
            title=f"{label}\nDownward (-Z), 32→64→128 ({conv})",
        )
    fig.suptitle(
        "Split-P downflow — cascade warm-start (n=32 → 64 → 128)",
        fontsize=12,
        y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _resolve_seed_cache(key: str, override: Path | None) -> Path:
    seed = Path(override) if override is not None else DEFAULT_SEED_CACHES[key]
    if not seed.is_dir():
        raise FileNotFoundError(
            f"Seed cache for '{key}' not found: {seed}\n"
            "Pass --seed-piecewise / --seed-linear to point at an existing n=128 downflow cache."
        )
    if not (seed / "velocity.npy").is_file():
        raise FileNotFoundError(f"Seed cache missing velocity.npy: {seed}")
    return seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n32-max-steps", type=int, default=5000)
    parser.add_argument("--n64-max-steps", type=int, default=6000)
    parser.add_argument("--n128-max-steps", type=int, default=6000)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--check-interval", type=int, default=250)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--seed-piecewise",
        type=Path,
        default=None,
        help="n=128 downflow cache used to warm-start piecewise n=32 (downsample).",
    )
    parser.add_argument(
        "--seed-linear",
        type=Path,
        default=None,
        help="n=128 downflow cache used to warm-start linear n=32 (downsample).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "Cube1mm_SplitP_Downflow_Cascade_32_64_128_2panel.png",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only render the final 2-panel figure from existing n=128 cascade caches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned stages and seed caches, then exit without running LBM.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages that already have a matching converge cache on disk.",
    )
    args = parser.parse_args()

    stages = (
        (32, int(args.n32_max_steps)),
        (64, int(args.n64_max_steps)),
        (128, int(args.n128_max_steps)),
    )
    seed_overrides = {
        "piecewise": args.seed_piecewise,
        "linear": args.seed_linear,
    }

    if args.dry_run:
        print("Cascade plan (downflow only, Split-P):")
        for case in CASES:
            key = case["key"]
            seed = _resolve_seed_cache(key, seed_overrides[key])
            print(f"\n  {case['label']}")
            print(f"    seed (-> n32): {seed.name}")
            for n, max_steps in stages:
                print(f"    n={n}: converge or max_steps={max_steps}  tol={args.tolerance:g}")
        print("\nDry run complete - no LBM executed.")
        return 0

    if args.plot_only:
        fine_caches: list[tuple[str, Path]] = []
        for case in CASES:
            # Reconstruct expected converge-cache slug for n=128 stage.
            cfg = VocalRunConfig(
                geometry="stl",
                stl_path=case["stl"],
                target_n=128,
                boundary_style="flow_chamber_reverse",
                re=5.0,
                ma=0.05,
                converge=True,
                max_steps=int(args.n128_max_steps),
                tolerance=float(args.tolerance),
                cache_dir=CACHE_ROOT,
            )
            cache = cache_path_for(cfg, CACHE_ROOT)
            if not cache.is_dir():
                raise FileNotFoundError(f"Missing fine cascade cache: {cache}")
            fine_caches.append((case["label"], cache))
        out = _plot_comparison(fine_caches, args.output, dpi=int(args.dpi))
        print(f"Wrote {out}")
        return 0

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fine_caches = []

    for case in CASES:
        stl = Path(case["stl"])
        if not stl.is_file():
            raise FileNotFoundError(stl)
        key = case["key"]
        label = case["label"]
        seed = _resolve_seed_cache(key, seed_overrides[key])

        print(f"\n{'=' * 72}")
        print(f"{label}")
        print(f"  Seed for n=32 (downsample from fine): {seed.name}")

        prev_cache: Path | None = seed
        final_cache: Path | None = None
        for target_n, max_steps in stages:
            print(
                f"\n=== {label}: n={target_n}, "
                f"converge or max_steps={max_steps} (tol={args.tolerance:g}) ===",
                flush=True,
            )
            metrics_json = (
                METRICS_DIR
                / f"SplitP_{key}_down_cascade_n{target_n}_ms{max_steps}_tol{args.tolerance:g}_metrics.json"
            )
            if args.resume:
                existing = _existing_stage_cache(
                    stl=stl,
                    target_n=target_n,
                    max_steps=max_steps,
                    tolerance=float(args.tolerance),
                )
                if existing is not None:
                    print(f"  Resume: reusing existing stage cache {existing.name}")
                    prev_cache = existing
                    final_cache = existing
                    continue
            stage_cache = _run_stage(
                stl=stl,
                target_n=target_n,
                max_steps=max_steps,
                warm_start_cache_dir=prev_cache,
                metrics_json=metrics_json,
                force_rerun=True,
                tolerance=float(args.tolerance),
                check_interval=int(args.check_interval),
                device=str(args.device),
            )
            prev_cache = stage_cache
            final_cache = stage_cache

        assert final_cache is not None
        fine_caches.append((label, final_cache))

    out = _plot_comparison(fine_caches, args.output, dpi=int(args.dpi))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
