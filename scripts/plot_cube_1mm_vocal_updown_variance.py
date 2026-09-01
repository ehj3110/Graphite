#!/usr/bin/env python
"""
1×3 heatmaps of up vs down Vocal variance on XZ mid-Y slices (n=128 caches).

For each lattice column, two variance maps are available (``--mode``):

- ``direct``:     (|u|_up − |u|_down)² at the same (X, Z) voxel
- ``z_mirror``:   (|u|_up − |u|_down,z-flip)² — downflow speed mirrored in Z

Solid voxels are masked (white). Split-P columns use converged 32→64→128 cascades.
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

from scripts.plot_cube_1mm_vocal_updown_comparison import (  # noqa: E402
    FLOW_CHAMBER_PAD_XY,
    FLOW_CHAMBER_PAD_Z,
    _cases,
    _load_cache,
)

OUT_DIR = _REPO_ROOT / "outputs" / "case_studies" / "cube_1mm" / "figures"


def _xz_speed_slice(velocity: np.ndarray, solid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)
    nx, ny, nz = solid.shape
    j_mid = ny // 2
    i0, i1 = FLOW_CHAMBER_PAD_XY, nx - FLOW_CHAMBER_PAD_XY
    k0, k1 = FLOW_CHAMBER_PAD_Z, nz - FLOW_CHAMBER_PAD_Z
    speed_sl = speed[i0:i1, j_mid, k0:k1]
    mask_sl = solid[i0:i1, j_mid, k0:k1]
    return speed_sl, mask_sl


def _variance_map(
    speed_up: np.ndarray,
    speed_down: np.ndarray,
    mask_sl: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, dict]:
    if mode == "direct":
        speed_ref = speed_down
        label = r"$(|u|_\uparrow - |u|_\downarrow)^2$"
    elif mode == "z_mirror":
        speed_ref = speed_down[:, ::-1]
        label = r"$(|u|_\uparrow - |u|_{\downarrow,z\text{-flip}})^2$"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fluid = ~mask_sl
    diff_sq = np.zeros_like(speed_up, dtype=np.float64)
    diff_sq[fluid] = (speed_up[fluid] - speed_ref[fluid]) ** 2

    vals = diff_sq[fluid]
    stats = {
        "mode": mode,
        "mean_mm2_s2": float(np.mean(vals)) * 1e6,
        "rms_mm_s": float(np.sqrt(np.mean(vals))) * 1000.0,
        "max_mm2_s2": float(np.max(vals)) * 1e6 if vals.size else 0.0,
        "label": label,
    }
    return diff_sq, stats


def _draw_variance_panel(
    ax,
    diff_sq: np.ndarray,
    mask_sl: np.ndarray,
    *,
    title: str,
    vmax: float,
) -> None:
    n_x, n_z = diff_sq.shape
    extent = [0, n_x, 0, n_z]
    fluid = ~mask_sl
    plot_mm2 = np.ma.masked_where(mask_sl, diff_sq * 1e6)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="white")
    ax.imshow(
        plot_mm2.T,
        cmap=cmap,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    mask_rgba = np.zeros((n_x, n_z, 4))
    mask_rgba[mask_sl] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(mask_rgba.transpose((1, 0, 2)), origin="lower", extent=extent)
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_z)
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")
    ax.set_title(title, fontsize=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("direct", "z_mirror"),
        default="direct",
        help="direct: same-grid |u| diff; z_mirror: compare up to Z-flipped down |u|",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: Cube1mm_Vocal_UpDown_variance_<mode>.png)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--shared-scale", action="store_true", default=True)
    parser.add_argument("--no-shared-scale", action="store_false", dest="shared_scale")
    args = parser.parse_args()

    cases = _cases()
    panels: list[tuple[str, np.ndarray, np.ndarray, dict]] = []

    for case in cases:
        for key in ("up_cache", "down_cache"):
            if not case[key].is_dir():
                raise FileNotFoundError(f"Missing cache: {case[key]}")

        grid_up, flow_up, metrics_up = _load_cache(case["up_cache"])
        grid_down, flow_down, metrics_down = _load_cache(case["down_cache"])
        speed_up, mask_up = _xz_speed_slice(flow_up.get_velocity_field(), grid_up.solid_mask)
        speed_down, mask_down = _xz_speed_slice(
            flow_down.get_velocity_field(), grid_down.solid_mask
        )
        if not np.array_equal(mask_up, mask_down):
            raise ValueError(f"Solid mask mismatch for {case['label']}")

        diff_sq, stats = _variance_map(speed_up, speed_down, mask_up, mode=str(args.mode))
        stats["up_steps"] = int(metrics_up.get("steps_run", 0))
        stats["down_steps"] = int(metrics_down.get("steps_run", 0))
        panels.append((case["label"], diff_sq, mask_up, stats))

    if args.shared_scale:
        vmax = max(
            float(np.max(p[1][~p[2]])) * 1e6 for p in panels if np.any(~p[2])
        )
    else:
        vmax = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    for ax, (label, diff_sq, mask_sl, stats) in zip(axes, panels, strict=True):
        panel_vmax = vmax if args.shared_scale else float(np.max(diff_sq[~mask_sl]) * 1e6)
        subtitle = (
            f"RMS Δ|u| = {stats['rms_mm_s']:.2f} mm/s\n"
            f"mean var = {stats['mean_mm2_s2']:.1f} (mm/s)²"
        )
        _draw_variance_panel(
            ax,
            diff_sq,
            mask_sl,
            title=f"{label}\n{subtitle}",
            vmax=panel_vmax,
        )

    mode_title = {
        "direct": "same (X, Z) voxel",
        "z_mirror": "up vs Z-mirrored down |u|",
    }[str(args.mode)]
    fig.suptitle(
        f"1 mm³ Vocal up/down variance — {mode_title} (XZ @ mid-Y, n=128)",
        fontsize=12,
        y=1.03,
    )
    sm = ScalarMappable(cmap=plt.get_cmap("inferno"), norm=Normalize(0.0, vmax or 1.0))
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label=r"$(\Delta |u|)^2$  [(mm/s)$^2$]",
    )

    output = args.output
    if output is None:
        output = OUT_DIR / f"Cube1mm_Vocal_UpDown_variance_{args.mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")
    for label, _, _, stats in panels:
        print(
            f"  {label}: RMS={stats['rms_mm_s']:.2f} mm/s  "
            f"mean_var={stats['mean_mm2_s2']:.1f} (mm/s)²  "
            f"max_var={stats['max_mm2_s2']:.1f} (mm/s)²  "
            f"(up {stats['up_steps']} / down {stats['down_steps']} steps)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
