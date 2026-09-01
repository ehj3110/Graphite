#!/usr/bin/env python
"""
2×3 comparison: Aristo von Mises (top) and Vocal flow (bottom) for three 1 mm³ cubes.

Column order: Cross-hatch woodpile | Split-P piecewise | Split-P linear grading.

By default each panel is peak-normalized independently (0–1). Pass
``--shared-color-scale`` to divide every panel in a row by that row's max
slice peak so colors are comparable across the three lattices.

Run from repo root::

    python scripts/plot_cube_1mm_aristo_vocal_comparison.py
    python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
    python scripts/plot_cube_1mm_aristo_vocal_comparison.py --shared-color-scale --plot-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.cross_section_viz import (
    DEFAULT_RASTER_PIXELS,
    DEFAULT_SOFT_BLUR_RADIUS_MM,
    _raster_axis_limits,
    element_slice_soft_field,
    load_tet_von_mises,
)
from graphite.case_studies.cube_1mm.specs import (
    default_output_dir,
    fea_dir,
    figures_dir,
    legacy_output_dir,
    resolve_woodpile_stem,
    splitp_fea_stem,
    splitp_piecewise_phase_shift_x_fea_stem,
    splitp_piecewise_phase_shift_x_vocal_cache_slug,
    splitp_linear_graded_vocal_cache_slug,
    vocal_cache_dir,
    woodpile_fea_stem,
    woodpile_vocal_cache_slug,
)

MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
VELOCITY_NAME = "velocity.npy"
SOLID_MASK_NAME = "solid_mask.npy"

FLOW_CHAMBER_PAD_XY = 1
FLOW_CHAMBER_PAD_Z = 3
SLICE_HALF_THICKNESS_MM = 0.012
UNIT_CUBE_CLIP = (0.0, 1.0, 0.0, 1.0)


class CachedFlowState:
    def __init__(self, velocity: np.ndarray):
        self._velocity = velocity

    def get_velocity_field(self) -> np.ndarray:
        return self._velocity


@dataclass
class _CachedGrid:
    solid_mask: np.ndarray


@dataclass(frozen=True)
class AristoSliceSpec:
    plane: str
    x_center: float
    y_center: float
    z_center: float
    xlabel: str
    ylabel: str
    slice_note: str


def _grid_from_mask(solid_mask: np.ndarray, meta: dict) -> _CachedGrid:
    return _CachedGrid(solid_mask=solid_mask)


def extract_unit_cube_xz_slice(grid, flow):
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


def _compute_aristo_slice(
    *,
    vtu: Path,
    slice_spec: AristoSliceSpec,
    domain_clip_mm: tuple[float, float, float, float] = UNIT_CUBE_CLIP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (A, B, field_MPa, slice_peak_MPa)."""
    nodes, elements, element_vm, _quality_ok = load_tet_von_mises(vtu)
    A, B, field, _material = element_slice_soft_field(
        nodes,
        elements,
        element_vm,
        plane=slice_spec.plane,
        x_center=slice_spec.x_center,
        y_center=slice_spec.y_center,
        z_center=slice_spec.z_center,
        x_half_thickness=SLICE_HALF_THICKNESS_MM,
        y_half_thickness=SLICE_HALF_THICKNESS_MM,
        z_half_thickness=SLICE_HALF_THICKNESS_MM,
        n_a=DEFAULT_RASTER_PIXELS,
        n_b=DEFAULT_RASTER_PIXELS,
        domain_clip_mm=domain_clip_mm,
        blur_radius_mm=DEFAULT_SOFT_BLUR_RADIUS_MM,
    )
    if not np.any(np.isfinite(field)):
        return A, B, field, 0.0
    return A, B, field, float(np.nanmax(field))


def _draw_xz_flow_panel(
    ax,
    speed_slice,
    mask_slice,
    u_x_slice,
    u_z_slice,
    *,
    title: str,
    quiver_stride: int = 4,
    colorbar: bool = True,
    display_peak: float | None = None,
    colorbar_label: str | None = None,
) -> float:
    n_x, n_z = speed_slice.shape
    extent = [0, n_x, 0, n_z]
    fluid = ~mask_slice
    peak = float(np.max(speed_slice[fluid])) if np.any(fluid) else 0.0
    if peak <= 0.0:
        ax.set_title(f"{title}\n(no fluid in slice)")
        ax.axis("off")
        return 0.0

    scale = float(display_peak) if display_peak is not None else peak
    if scale <= 0.0:
        scale = peak
    speed_norm = np.ma.masked_where(mask_slice, speed_slice / (scale + 1e-15))
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="white")
    ax.imshow(
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
    shared_note = ""
    if display_peak is not None and abs(display_peak - peak) > 1e-15:
        shared_note = f"  (shared max {display_peak * 1000.0:.2f} mm/s)"
    ax.set_title(
        f"{title}\nSlice peak $|u|$ = {peak * 1000.0:.2f} mm/s{shared_note}",
        fontsize=10,
    )
    if colorbar:
        sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        label = colorbar_label or r"Normalized $|u|$"
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label)
    return peak


def _load_vocal_from_cache(cache_dir: Path) -> tuple[object, CachedFlowState, dict]:
    cache_dir = Path(cache_dir)
    for name in (MANIFEST_NAME, METRICS_NAME, VELOCITY_NAME, SOLID_MASK_NAME):
        if not (cache_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name} in {cache_dir}")
    manifest = json.loads((cache_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    solid_mask = np.load(cache_dir / SOLID_MASK_NAME)
    velocity = np.load(cache_dir / VELOCITY_NAME)
    metrics = json.loads((cache_dir / METRICS_NAME).read_text(encoding="utf-8"))
    grid = _grid_from_mask(solid_mask, manifest["grid_meta"])
    return grid, CachedFlowState(velocity), metrics


def _draw_aristo_panel(
    ax,
    *,
    A: np.ndarray,
    B: np.ndarray,
    field: np.ndarray,
    slice_peak: float,
    label: str,
    slice_spec: AristoSliceSpec,
    domain_clip_mm: tuple[float, float, float, float] = UNIT_CUBE_CLIP,
    display_peak: float | None = None,
    colorbar: bool = True,
    colorbar_label: str | None = None,
) -> float:
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="white")

    if not np.any(np.isfinite(field)) or slice_peak <= 0.0:
        ax.set_title(f"{label}\n(no elements in slice)")
        ax.axis("off")
        return 0.0

    scale = float(display_peak) if display_peak is not None else slice_peak
    if scale <= 0.0:
        scale = slice_peak
    masked = np.ma.masked_invalid(field / (scale + 1e-15))
    x_lo, x_hi, y_lo, y_hi = _raster_axis_limits(A, B, domain_clip_mm=domain_clip_mm)
    ax.imshow(
        masked,
        origin="lower",
        extent=(x_lo, x_hi, y_lo, y_hi),
        aspect="equal",
        cmap=cmap_obj,
        norm=Normalize(vmin=0.0, vmax=1.0),
        interpolation="nearest",
    )
    ax.set_facecolor("white")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(slice_spec.xlabel)
    ax.set_ylabel(slice_spec.ylabel)
    shared_note = ""
    if display_peak is not None and abs(display_peak - slice_peak) > 1e-12:
        shared_note = f"  (shared max {display_peak:.2f} MPa)"
    ax.set_title(
        f"{label}\n{slice_spec.slice_note}  "
        f"Slice peak element $\\sigma_{{vm}}$ = {slice_peak:.2f} MPa{shared_note}",
        fontsize=10,
    )
    if colorbar:
        sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        label_cb = colorbar_label or r"Normalized $\sigma_{vm}$"
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label_cb)
    return slice_peak


def _resolve_vtu(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"VTU not found. Tried: {', '.join(str(p) for p in candidates)}")


def _default_cases() -> tuple[dict, dict, dict]:
    out = fea_dir()
    legacy = legacy_output_dir()
    cache = vocal_cache_dir()
    woodpile_stem = resolve_woodpile_stem(match_splitp_pores=True, generator="extrude")
    woodpile_fea = woodpile_fea_stem(woodpile_stem, h_mm=0.015)
    return (
        {
            "label": "Cross-hatch woodpile (extrude)",
            "vtu": _resolve_vtu(
                (
                    out / f"{woodpile_fea}_1N_aristo_fea.vtu",
                    legacy / f"{woodpile_fea}_1N_aristo_fea.vtu",
                )
            ),
            "aristo_slice": AristoSliceSpec(
                plane="yz",
                x_center=0.3,
                y_center=0.5,
                z_center=0.5,
                xlabel="Y (mm)",
                ylabel="Z (mm)",
                slice_note="YZ @ X = 0.300 mm",
            ),
            "vocal_cache": cache / woodpile_vocal_cache_slug(converge=True, max_steps=3000),
        },
        {
            "label": "Split-P piecewise (bottom X +L/4)",
            "vtu": _resolve_vtu(
                (
                    out
                    / f"{splitp_piecewise_phase_shift_x_fea_stem()}_1N_aristo_fea.vtu",
                    legacy
                    / f"{splitp_piecewise_phase_shift_x_fea_stem()}_1N_aristo_fea.vtu",
                )
            ),
            "aristo_slice": AristoSliceSpec(
                plane="xz",
                x_center=0.5,
                y_center=0.5,
                z_center=0.5,
                xlabel="X (mm)",
                ylabel="Z (mm)",
                slice_note="XZ @ Y = 0.500 mm",
            ),
            "vocal_cache": cache / splitp_piecewise_phase_shift_x_vocal_cache_slug(converge=True),
        },
        {
            "label": "Split-P linear grading",
            "vtu": _resolve_vtu(
                (
                    out / f"{splitp_fea_stem('linearGrad')}_1N_aristo_fea.vtu",
                    legacy
                    / f"{splitp_fea_stem('linearGrad')}_1N_aristo_fea.vtu",
                )
            ),
            "aristo_slice": AristoSliceSpec(
                plane="xz",
                x_center=0.5,
                y_center=0.5,
                z_center=0.5,
                xlabel="X (mm)",
                ylabel="Z (mm)",
                slice_note="XZ @ Y = 0.500 mm",
            ),
            "vocal_cache": cache / splitp_linear_graded_vocal_cache_slug(converge=True),
        },
    )


def plot_six_panel(
    cases: tuple[dict, dict, dict],
    *,
    output_path: Path,
    dpi: int = 200,
    quiver_stride: int = 4,
    shared_color_scale: bool = False,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)

    vocal_slices: list[tuple] = []
    vocal_peaks: list[float] = []
    for case in cases:
        grid, flow, metrics = _load_vocal_from_cache(case["vocal_cache"])
        steps = int(metrics["steps_run"])
        conv = "converged" if metrics.get("converged") else f"{steps} steps"
        case["vocal_subtitle"] = (
            f"{case['label']}\nVocal ({conv})  "
            f"k={float(metrics['permeability_mm2']):.2e} mm²"
        )
        speed_sl, u_x_sl, u_z_sl, mask_sl, peak = extract_unit_cube_xz_slice(grid, flow)
        vocal_slices.append((speed_sl, mask_sl, u_x_sl, u_z_sl))
        vocal_peaks.append(peak)

    aristo_slices: list[tuple] = []
    aristo_peaks: list[float] = []
    for case in cases:
        A, B, field, slice_peak = _compute_aristo_slice(
            vtu=Path(case["vtu"]),
            slice_spec=case["aristo_slice"],
        )
        aristo_slices.append((A, B, field, slice_peak))
        aristo_peaks.append(slice_peak)

    shared_aristo = max(aristo_peaks) if shared_color_scale and aristo_peaks else None
    shared_vocal = max(vocal_peaks) if shared_color_scale and vocal_peaks else None
    aristo_cb = (
        r"$\sigma_{vm}$ / shared max"
        if shared_color_scale
        else r"Normalized $\sigma_{vm}$"
    )
    vocal_cb = (
        r"$|u|$ / shared max"
        if shared_color_scale
        else r"Normalized $|u|$"
    )

    for col, case in enumerate(cases):
        A, B, field, slice_peak = aristo_slices[col]
        _draw_aristo_panel(
            axes[0, col],
            A=A,
            B=B,
            field=field,
            slice_peak=slice_peak,
            label=case["label"],
            slice_spec=case["aristo_slice"],
            display_peak=shared_aristo,
            colorbar_label=aristo_cb,
        )

        speed_sl, mask_sl, u_x_sl, u_z_sl = vocal_slices[col]
        _draw_xz_flow_panel(
            axes[1, col],
            speed_sl,
            mask_sl,
            u_x_sl,
            u_z_sl,
            title=case["vocal_subtitle"],
            quiver_stride=quiver_stride,
            colorbar=True,
            display_peak=shared_vocal,
            colorbar_label=vocal_cb,
        )

    axes[0, 0].annotate(
        "Aristo — 1 N compression",
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    axes[1, 0].annotate(
        "Vocal — Re=5, flow chamber",
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
    )
    scale_note = (
        " — shared color scale per row"
        if shared_color_scale
        else " — per-panel peak normalized"
    )
    fig.suptitle(
        "1 mm³ lattice comparison — extrude woodpile vs Split-P (piecewise vs linear)"
        + scale_note,
        fontsize=13,
        y=1.02,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if shared_color_scale:
        print(
            f"Shared Aristo slice peak = {shared_aristo:.4f} MPa; "
            f"shared Vocal slice peak = {shared_vocal * 1000.0:.4f} mm/s"
        )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --shared-color-scale).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--quiver-stride", type=int, default=4)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only render PNG from existing VTU/cache artifacts.",
    )
    parser.add_argument(
        "--shared-color-scale",
        action="store_true",
        help=(
            "Normalize each row by the max slice peak across the three lattices "
            "(default: independent per-panel peaks)."
        ),
    )
    args = parser.parse_args()
    if not args.plot_only:
        print("Using existing Aristo VTUs and Vocal caches (no re-simulation).")
    cases = _default_cases()
    if args.output is None:
        name = (
            "Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png"
            if args.shared_color_scale
            else "Cube1mm_Aristo_Vocal_6panel_comparison.png"
        )
        output = figures_dir() / name
    else:
        output = args.output
    out = plot_six_panel(
        cases,
        output_path=output,
        dpi=args.dpi,
        quiver_stride=args.quiver_stride,
        shared_color_scale=bool(args.shared_color_scale),
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
