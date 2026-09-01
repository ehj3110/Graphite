#!/usr/bin/env python
"""
Coarse→fine Vocal warm-start for Split-P 1 mm cubes (downflow only).

1) Run / continue at target_n=64 (flow_chamber_reverse)
2) Upsample velocity → warm-start target_n=128
3) Continue at 128 and write a 2-panel comparison figure

Run with ``.venv_torch`` from repo root::

    .venv_torch\\Scripts\\python.exe scripts\\run_splitp_coarse_to_fine_downflow.py
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


def _run_case(
    *,
    stl: Path,
    target_n: int,
    steps: int,
    warm_start: str,
    warm_start_cache_dir: Path | None,
    metrics_json: Path,
    force_rerun: bool,
) -> Path:
    cfg = VocalRunConfig(
        geometry="stl",
        stl_path=stl,
        target_n=target_n,
        boundary_style="flow_chamber_reverse",
        re=5.0,
        ma=0.05,
        steps=steps,
        converge=False,
        warm_start=warm_start,  # type: ignore[arg-type]
        warm_start_cache_dir=warm_start_cache_dir,
        cache_dir=CACHE_ROOT,
        use_cache=True,
        force_rerun=force_rerun,
        plot=False,
        metrics_json=metrics_json,
        device="cuda",
    )
    run_vocal(cfg)
    return cache_path_for(cfg, CACHE_ROOT)


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
            title=f"{label}\nDownward (-Z), coarse→fine ({conv})",
        )
    fig.suptitle(
        "Split-P downflow — coarse (n=64) → fine (n=128) warm-start",
        fontsize=12,
        y=1.02,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-n", type=int, default=64)
    parser.add_argument("--fine-n", type=int, default=128)
    parser.add_argument("--coarse-steps", type=int, default=3000)
    parser.add_argument("--fine-steps", type=int, default=2500)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "Cube1mm_SplitP_Downflow_CoarseToFine_2panel.png",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fine_caches: list[tuple[str, Path]] = []

    for case in CASES:
        stl = Path(case["stl"])
        if not stl.is_file():
            raise FileNotFoundError(stl)
        key = case["key"]
        label = case["label"]

        print(f"\n=== {label}: coarse n={args.coarse_n}, +{args.coarse_steps} steps ===")
        coarse_cache = _run_case(
            stl=stl,
            target_n=int(args.coarse_n),
            steps=int(args.coarse_steps),
            warm_start="auto",
            warm_start_cache_dir=None,
            metrics_json=METRICS_DIR / f"SplitP_{key}_down_coarse_n{args.coarse_n}_steps{args.coarse_steps}_metrics.json",
            force_rerun=True,
        )
        print(f"Coarse cache: {coarse_cache}")

        print(f"\n=== {label}: fine n={args.fine_n}, +{args.fine_steps} steps (upsample warm-start) ===")
        fine_cache = _run_case(
            stl=stl,
            target_n=int(args.fine_n),
            steps=int(args.fine_steps),
            warm_start="cache",
            warm_start_cache_dir=coarse_cache,
            metrics_json=METRICS_DIR / f"SplitP_{key}_down_fine_n{args.fine_n}_from_n{args.coarse_n}_steps{args.fine_steps}_metrics.json",
            force_rerun=True,
        )
        print(f"Fine cache: {fine_cache}")
        fine_caches.append((label, fine_cache))

    out = _plot_comparison(fine_caches, args.output, dpi=int(args.dpi))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
