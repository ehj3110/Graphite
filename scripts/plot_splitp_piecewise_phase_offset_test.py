#!/usr/bin/env python
"""
4-panel Split-P piecewise phase-offset test @ XZ slice Y = 0.5 mm.

Compares baseline piecewise FEA against bottom-band lateral shifts of L_bottom/4
in X, Y, or both — same pore sizes and solid fraction, staggered smaller-pore phase.

Run from repo root::

    python scripts/plot_splitp_piecewise_phase_offset_test.py
    python scripts/plot_splitp_piecewise_phase_offset_test.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
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
from graphite.aristo.volume_mesh_inspect import remove_floating_islands
from graphite.case_studies.cube_1mm.generate_splitp import build_piecewise_splitp_cube_mesh
from graphite.case_studies.cube_1mm.specs import (
    ARISTO_E_MPA,
    ARISTO_FORCE_N,
    BAND_HEIGHT_MM,
    SPLITP_ARISTO_H_MM,
    SPLITP_L_BOTTOM_MM,
    default_output_dir,
    figures_dir,
    repo_root,
    scripts_dir,
    splitp_fea_stem,
    splitp_piecewise_stem,
)

UNIT_CUBE_CLIP = (0.0, 1.0, 0.0, 1.0)
SLICE_Y_CENTER_MM = 0.5
SLICE_HALF_THICKNESS_MM = 0.012
BOTTOM_QUARTER_CELL_SHIFT_MM = 0.25 * SPLITP_L_BOTTOM_MM


@dataclass(frozen=True)
class PhaseTestVariant:
    key: str
    label: str
    band_phase_origin_x_mm: tuple[float, float]
    band_phase_origin_y_mm: tuple[float, float]
    use_baseline_vtu: bool = False


VARIANTS: tuple[PhaseTestVariant, ...] = (
    PhaseTestVariant(
        "baseline",
        "Baseline (no bottom-band shift)",
        (0.0, 0.0),
        (0.0, 0.0),
        use_baseline_vtu=True,
    ),
    PhaseTestVariant(
        "shift_x",
        f"Bottom band shift X +{BOTTOM_QUARTER_CELL_SHIFT_MM:.3f} mm (L/4)",
        (BOTTOM_QUARTER_CELL_SHIFT_MM, 0.0),
        (0.0, 0.0),
    ),
    PhaseTestVariant(
        "shift_y",
        f"Bottom band shift Y +{BOTTOM_QUARTER_CELL_SHIFT_MM:.3f} mm (L/4)",
        (0.0, 0.0),
        (BOTTOM_QUARTER_CELL_SHIFT_MM, 0.0),
    ),
    PhaseTestVariant(
        "shift_xy",
        f"Bottom band shift X+Y +{BOTTOM_QUARTER_CELL_SHIFT_MM:.3f} mm (L/4)",
        (BOTTOM_QUARTER_CELL_SHIFT_MM, 0.0),
        (BOTTOM_QUARTER_CELL_SHIFT_MM, 0.0),
    ),
)


def _phase_test_stem(variant_key: str) -> str:
    return f"{splitp_piecewise_stem()}_phaseTest_qL4_{variant_key}"


def _phase_test_fea_stem(variant_key: str) -> str:
    return (
        f"{_phase_test_stem(variant_key)}_h{int(round(SPLITP_ARISTO_H_MM * 1000)):03d}_1N_E25p8MPa"
    )


def _run(cmd: list[str], *, desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root(), check=True)


def _ensure_geometry(variant: PhaseTestVariant, out_dir: Path) -> Path:
    stem = _phase_test_stem(variant.key)
    stl_path = out_dir / f"{stem}.stl"
    cleaned_path = out_dir / f"{stem}_cleaned.stl"
    if cleaned_path.is_file():
        return cleaned_path

    print(f"Meshing {variant.label} ...")
    mesh, meta = build_piecewise_splitp_cube_mesh(
        band_phase_origin_x_mm=variant.band_phase_origin_x_mm,
        band_phase_origin_y_mm=variant.band_phase_origin_y_mm,
    )
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)
    cleaned, island_meta = remove_floating_islands(mesh)
    print(
        f"  islands: {island_meta['n_components_before']} -> "
        f"{island_meta['n_components_after']}"
    )
    cleaned.export(cleaned_path)
    report_path = out_dir / f"{stem}_generation_report.json"
    report_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {stl_path}")
    print(f"Wrote {cleaned_path}")
    return cleaned_path


def _ensure_fea(variant: PhaseTestVariant, out_dir: Path, cleaned_stl: Path) -> Path:
    if variant.use_baseline_vtu:
        vtu = out_dir / f"{splitp_fea_stem('piecewise')}_1N_aristo_fea.vtu"
        if not vtu.is_file():
            raise FileNotFoundError(f"Baseline VTU missing: {vtu}")
        return vtu

    fea_stem = _phase_test_fea_stem(variant.key)
    vtu = out_dir / f"{fea_stem}_1N_aristo_fea.vtu"
    if vtu.is_file():
        return vtu

    _run(
        [
            sys.executable,
            str(scripts_dir() / "run_aristo_fea.py"),
            "--stl",
            str(cleaned_stl),
            "--h",
            str(SPLITP_ARISTO_H_MM),
            "--force-n",
            str(ARISTO_FORCE_N),
            "--youngs-modulus",
            str(ARISTO_E_MPA),
            "--mesh-mode",
            "single_surface",
            "--solver",
            "pardiso",
            "--bc-load-mode",
            "flat_top_vertex_plane",
            "--stem",
            fea_stem,
            "--out-dir",
            str(out_dir),
            "--no-viz",
        ],
        desc=f"Aristo FEA — {variant.label}",
    )
    if not vtu.is_file():
        raise FileNotFoundError(f"Expected VTU not written: {vtu}")
    return vtu


def _draw_panel(ax, *, vtu: Path, title: str) -> float:
    nodes, elements, element_vm, _quality_ok = load_tet_von_mises(vtu)
    A, B, field, _material = element_slice_soft_field(
        nodes,
        elements,
        element_vm,
        plane="xz",
        x_center=0.5,
        y_center=SLICE_Y_CENTER_MM,
        z_center=0.5,
        x_half_thickness=SLICE_HALF_THICKNESS_MM,
        y_half_thickness=SLICE_HALF_THICKNESS_MM,
        z_half_thickness=SLICE_HALF_THICKNESS_MM,
        n_a=DEFAULT_RASTER_PIXELS,
        n_b=DEFAULT_RASTER_PIXELS,
        domain_clip_mm=UNIT_CUBE_CLIP,
        blur_radius_mm=DEFAULT_SOFT_BLUR_RADIUS_MM,
    )
    cmap_obj = plt.get_cmap("turbo").copy()
    cmap_obj.set_bad(color="white")
    if not np.any(np.isfinite(field)):
        ax.set_title(f"{title}\n(no elements in slice)")
        ax.axis("off")
        return 0.0

    slice_peak = float(np.nanmax(field))
    masked = np.ma.masked_invalid(field / (slice_peak + 1e-15))
    x_lo, x_hi, y_lo, y_hi = _raster_axis_limits(A, B, domain_clip_mm=UNIT_CUBE_CLIP)
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
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(
        f"{title}\nXZ @ Y = {SLICE_Y_CENTER_MM:.3f} mm  "
        f"peak element $\\sigma_{{vm}}$ = {slice_peak:.2f} MPa",
        fontsize=9,
    )
    sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=r"Normalized $\sigma_{vm}$")
    return slice_peak


def plot_phase_offset_test(
    *,
    out_dir: Path,
    output_path: Path,
    dpi: int = 200,
    skip_fea: bool = False,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0), constrained_layout=True)
    axes_flat = axes.ravel()

    for ax, variant in zip(axes_flat, VARIANTS, strict=True):
        cleaned_stl = _ensure_geometry(variant, out_dir)
        if skip_fea and not variant.use_baseline_vtu:
            vtu = out_dir / f"{_phase_test_fea_stem(variant.key)}_1N_aristo_fea.vtu"
            if not vtu.is_file():
                ax.set_title(f"{variant.label}\n(VTU missing — run without --plot-only)")
                ax.axis("off")
                continue
        else:
            vtu = _ensure_fea(variant, out_dir, cleaned_stl)
        _draw_panel(ax, vtu=vtu, title=variant.label)

    fig.suptitle(
        "Split-P piecewise — bottom-band lateral phase offset test "
        f"(band break Z = {BAND_HEIGHT_MM:.1f} mm, SF = 33 %)",
        fontsize=12,
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=figures_dir() / "SplitP_Cube1mm_piecewise_phaseOffset_test_4panel.png",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip FEA for new variants (requires existing VTUs).",
    )
    args = parser.parse_args()
    out = args.out_dir or default_output_dir()
    png = plot_phase_offset_test(
        out_dir=out,
        output_path=args.output,
        dpi=int(args.dpi),
        skip_fea=bool(args.plot_only),
    )
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
