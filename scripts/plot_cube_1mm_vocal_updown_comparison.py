#!/usr/bin/env python
"""
2x3 Vocal comparison for 1 mm cube lattices: upflow vs downflow.

Top row:    flow_chamber (+Z / upward)
Bottom row: flow_chamber_reverse (-Z / downward)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / "outputs" / "vocal" / "cache"
OUT_DIR = REPO_ROOT / "outputs" / "case_studies" / "cube_1mm" / "figures"

FLOW_CHAMBER_PAD_XY = 1
FLOW_CHAMBER_PAD_Z = 3


class CachedFlowState:
    def __init__(self, velocity: np.ndarray):
        self._velocity = velocity

    def get_velocity_field(self) -> np.ndarray:
        return self._velocity


class CachedGrid:
    def __init__(self, solid_mask: np.ndarray):
        self.solid_mask = solid_mask


def _load_cache(cache_dir: Path) -> tuple[CachedGrid, CachedFlowState, dict]:
    required = ("manifest.json", "metrics.json", "velocity.npy", "solid_mask.npy")
    for name in required:
        if not (cache_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {cache_dir}")
    metrics = json.loads((cache_dir / "metrics.json").read_text(encoding="utf-8"))
    velocity = np.load(cache_dir / "velocity.npy")
    solid_mask = np.load(cache_dir / "solid_mask.npy")
    return CachedGrid(solid_mask=solid_mask), CachedFlowState(velocity), metrics


def _extract_xz_slice(grid: CachedGrid, flow: CachedFlowState):
    u_np = flow.get_velocity_field()
    speed = np.sqrt(u_np[0] ** 2 + u_np[1] ** 2 + u_np[2] ** 2)
    solid = grid.solid_mask
    nx, ny, nz = solid.shape
    j_mid = ny // 2
    i0, i1 = FLOW_CHAMBER_PAD_XY, nx - FLOW_CHAMBER_PAD_XY
    k0, k1 = FLOW_CHAMBER_PAD_Z, nz - FLOW_CHAMBER_PAD_Z
    speed_sl = speed[i0:i1, j_mid, k0:k1]
    u_x_sl = u_np[0, i0:i1, j_mid, k0:k1]
    u_z_sl = u_np[2, i0:i1, j_mid, k0:k1]
    mask_sl = solid[i0:i1, j_mid, k0:k1]
    fluid = ~mask_sl
    peak = float(np.max(speed_sl[fluid])) if np.any(fluid) else 0.0
    return speed_sl, u_x_sl, u_z_sl, mask_sl, peak


def _draw_panel(ax, speed_slice, u_x_slice, u_z_slice, mask_slice, *, title: str, quiver_stride: int = 4):
    n_x, n_z = speed_slice.shape
    extent = [0, n_x, 0, n_z]
    fluid = ~mask_slice
    peak = float(np.max(speed_slice[fluid])) if np.any(fluid) else 0.0
    if peak <= 0.0:
        ax.set_title(f"{title}\n(no fluid in slice)")
        ax.axis("off")
        return

    speed_norm = np.ma.masked_where(mask_slice, speed_slice / (peak + 1e-15))
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="white")
    im = ax.imshow(
        speed_norm.T,
        cmap=cmap_obj,
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    mask_rgba = np.zeros((n_x, n_z, 4))
    mask_rgba[mask_slice] = [0.15, 0.15, 0.15, 1.0]
    ax.imshow(mask_rgba.transpose((1, 0, 2)), origin="lower", extent=extent)

    x_coords = np.arange(0, n_x, 1)
    z_coords = np.arange(0, n_z, 1)
    x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing="ij")
    x_stride = x_grid[::quiver_stride, ::quiver_stride]
    z_stride = z_grid[::quiver_stride, ::quiver_stride]
    u_x_stride = u_x_slice[::quiver_stride, ::quiver_stride]
    u_z_stride = u_z_slice[::quiver_stride, ::quiver_stride]
    fluid_stride = ~mask_slice[::quiver_stride, ::quiver_stride]
    ax.quiver(
        x_stride[fluid_stride] + 0.5,
        z_stride[fluid_stride] + 0.5,
        u_x_stride[fluid_stride],
        u_z_stride[fluid_stride],
        color="white",
        pivot="middle",
    )
    ax.set_facecolor("white")
    ax.set_xlim(0, n_x)
    ax.set_ylim(0, n_z)
    ax.set_xlabel("X (voxels)")
    ax.set_ylabel("Z (voxels)")
    ax.set_title(f"{title}\nSlice peak |u| = {peak * 1000.0:.2f} mm/s", fontsize=10)
    sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=r"Normalized $|u|$")


def _cases():
    """Upward row: woodpile from 6-panel cache; Split-P from converged cascade."""
    return (
        {
            "label": "Cross-hatch woodpile (extrude)",
            "up_cache": CACHE_ROOT
            / "Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned_n128_flow_chamber_re5_ma0.05_steps2000",
            "down_cache": CACHE_ROOT
            / "Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned_n128_flow_chamber_reverse_re5_ma0.05_steps2000",
            "up_note": "2000 steps (6-panel cache)",
            "down_note": "3000 steps total (1000 + 2000)",
        },
        {
            "label": "Split-P piecewise (bottom X +L/4)",
            "up_cache": CACHE_ROOT
            / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001",
            "down_cache": CACHE_ROOT
            / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001",
            "up_note": "converged, 32->64->128 cascade (2000 steps @ n128)",
            "down_note": "converged, 32->64->128 cascade (4750 steps @ n128)",
        },
        {
            "label": "Split-P linear grading",
            "up_cache": CACHE_ROOT
            / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001",
            "down_cache": CACHE_ROOT
            / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001",
            "up_note": "converged, 32->64->128 cascade (4250 steps @ n128)",
            "down_note": "converged, 32->64->128 cascade (4750 steps @ n128)",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "Cube1mm_Vocal_UpDown_2x3_comparison.png",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--quiver-stride", type=int, default=4)
    args = parser.parse_args()

    cases = _cases()
    for case in cases:
        for key in ("up_cache", "down_cache"):
            if not case[key].is_dir():
                raise FileNotFoundError(f"Missing cache: {case[key]}")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)

    for col, case in enumerate(cases):
        grid_up, flow_up, _ = _load_cache(case["up_cache"])
        speed_up, u_x_up, u_z_up, mask_up, _ = _extract_xz_slice(grid_up, flow_up)
        _draw_panel(
            axes[0, col],
            speed_up,
            u_x_up,
            u_z_up,
            mask_up,
            title=f"{case['label']}\nUpward (+Z) — {case['up_note']}",
            quiver_stride=args.quiver_stride,
        )

        grid_down, flow_down, _ = _load_cache(case["down_cache"])
        speed_down, u_x_down, u_z_down, mask_down, _ = _extract_xz_slice(grid_down, flow_down)
        _draw_panel(
            axes[1, col],
            speed_down,
            u_x_down,
            u_z_down,
            mask_down,
            title=f"{case['label']}\nDownward (-Z) — {case['down_note']}",
            quiver_stride=args.quiver_stride,
        )

    axes[0, 0].annotate(
        "Vocal — Upward (+Z)",
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    axes[1, 0].annotate(
        "Vocal — Downward (-Z)",
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    fig.suptitle(
        "1 mm³ lattices — Vocal flow direction comparison (Split-P: converged 32→64→128 cascades)",
        fontsize=13,
        y=1.02,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
