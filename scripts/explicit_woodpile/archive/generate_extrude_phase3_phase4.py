#!/usr/bin/env python
"""
Phase 3 + 4 — piecewise cylinder clip + true woodpile extrusion.

Phase 3: default Ø2 mm piecewise cylinder (P800 / P400 / P200 bands).
Phase 4: uniform true woodpile cylinder (same Ø2 × 2.7 mm demo).

Run from repo root::

    python scripts/explicit_woodpile/generate_extrude_phase3_phase4.py
    python scripts/explicit_woodpile/generate_extrude_phase3_phase4.py --skip-implicit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.explicit.woodpile_extrude import (
    _layer_mesh_box,
    generate_crosshatch_cylinder,
    generate_piecewise_crosshatch_cylinder,
    strut_axis_for_crosshatch_layer,
    true_woodpile_phase_shift_mm,
    verify_cylinder_clip,
    write_mesh_report,
)
from graphite.implicit.piecewise_woodpile import woodpile_piecewise_cylinder_single_pass
from graphite.math.woodpile import evaluate_woodpile, evaluate_woodpile_piecewise_cylinder
from graphite.math.woodpile_anchor import compute_woodpile_xy_origin

OUT_P3 = _REPO_ROOT / "outputs" / "explicit_woodpile" / "P3"
OUT_P4 = _REPO_ROOT / "outputs" / "explicit_woodpile" / "P4"

RADIUS_MM = 1.0
HEIGHT_MM = 2.7
Z_BREAKS = [0.0, 2.23, 2.31, 2.7]
PORES_PIECEWISE = [0.8, 0.4, 0.2]
PORE_UNIFORM = 0.8


def _cylinder_domain_sdf(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, *, radius_mm: float
) -> np.ndarray:
    radial = np.sqrt(xx * xx + yy * yy) - float(radius_mm)
    z_lo = -zz
    z_hi = zz - float(HEIGHT_MM)
    return np.maximum.reduce([radial, z_lo, z_hi])


def _implicit_piecewise_xz_mask(
    *,
    z_breaks_mm: list[float],
    pore_mm: list[float],
    origins_x: list[float],
    origins_y: list[float],
    swap_xy: list[bool],
    flips: list[bool],
    true_woodpile: bool,
    n: int = 250,
    y_mm: float = 0.0,
) -> np.ndarray:
    r = float(RADIUS_MM)
    xs = np.linspace(-r, r, n)
    zs = np.linspace(0.0, HEIGHT_MM, n)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = np.full_like(xx, float(y_mm))
    field = evaluate_woodpile_piecewise_cylinder(
        xx,
        yy,
        zz,
        z_breaks_mm=z_breaks_mm,
        pore_mm=pore_mm,
        origin_x_mm=origins_x,
        origin_y_mm=origins_y,
        swap_xy=swap_xy,
        flip_layer_parity=flips,
        true_woodpile=true_woodpile,
    )
    sdf = _cylinder_domain_sdf(xx, yy, zz, radius_mm=r)
    return np.maximum(field, sdf) <= 0.0


def _implicit_uniform_xz_mask(
    *,
    pore_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
    true_woodpile: bool,
    n: int = 250,
    y_mm: float = 0.0,
) -> np.ndarray:
    r = float(RADIUS_MM)
    xs = np.linspace(-r, r, n)
    zs = np.linspace(0.0, HEIGHT_MM, n)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = np.full_like(xx, float(y_mm))
    field = evaluate_woodpile(
        xx,
        yy,
        zz,
        pore_mm,
        true_woodpile,
        origin_x=origin_x_mm,
        origin_y=origin_y_mm,
    )
    sdf = _cylinder_domain_sdf(xx, yy, zz, radius_mm=r)
    return np.maximum(field, sdf) <= 0.0


def _save_field_xz_png(solid: np.ndarray, path: Path, *, title: str) -> None:
    r = float(RADIUS_MM)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        solid.T,
        origin="lower",
        extent=(-r, r, 0.0, HEIGHT_MM),
        aspect="equal",
        cmap="gray_r",
        interpolation="nearest",
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _export_clip_corner_detail(out_path: Path) -> dict:
    """Small bbox export around one strut–cylinder-wall junction (band 0, layer 0)."""
    pore = PORES_PIECEWISE[0]
    lox, loy, _ = compute_woodpile_xy_origin(
        radius_mm=RADIUS_MM, pore_mm=pore, mode="center_void"
    )
    slab = _layer_mesh_box(
        layer_idx=0,
        z0_mm=0.0,
        layer_height_mm=pore,
        pore_mm=pore,
        x_lo=-RADIUS_MM,
        x_hi=RADIUS_MM,
        y_lo=-RADIUS_MM,
        y_hi=RADIUS_MM,
        lattice_origin_x_mm=lox,
        lattice_origin_y_mm=loy,
        flip_layer_parity=False,
        swap_xy=False,
        clip_circle=(0.0, 0.0, RADIUS_MM),
        true_woodpile=False,
    )
    verts = np.asarray(slab.vertices)
    mask = (
        (verts[:, 0] >= 0.55)
        & (verts[:, 0] <= 1.05)
        & (verts[:, 1] >= -0.25)
        & (verts[:, 1] <= 0.25)
        & (verts[:, 2] >= -0.01)
        & (verts[:, 2] <= pore + 0.01)
    )
    face_mask = mask[slab.faces].any(axis=1)
    face_idx = np.where(face_mask)[0]
    detail = slab.submesh([face_idx], append=True) if len(face_idx) else slab
    write_mesh_report(
        detail,
        out_path,
        {
            "generator": "woodpile_extrude",
            "artifact": "clip_corner_detail",
            "pore_mm": pore,
            "bbox_mm": [0.55, 1.05, -0.25, 0.25, 0.0, pore],
            "faces": int(len(detail.faces)),
            "watertight": bool(detail.is_watertight),
        },
    )
    return {"faces": int(len(detail.faces)), "watertight": bool(detail.is_watertight)}


def run_phase3(*, skip_implicit: bool) -> dict:
    OUT_P3.mkdir(parents=True, exist_ok=True)
    mesh, rep = generate_piecewise_crosshatch_cylinder(
        radius_mm=RADIUS_MM,
        height_mm=HEIGHT_MM,
        z_breaks_mm=Z_BREAKS,
        pore_mm=PORES_PIECEWISE,
        anchor_mode="center_void",
        alternate_band_orientation=True,
        true_woodpile=False,
    )
    stl_path = OUT_P3 / "P3_crosshatch_cylinder_D2_piecewise_extrude.stl"
    write_mesh_report(mesh, stl_path, rep)
    print(
        f"P3 extrude: {stl_path}  faces={rep['faces']}  "
        f"wt={rep['watertight']}  clip_ok={rep['clip_qc']['ok']}"
    )

    corner_path = OUT_P3 / "P3_clip_corner_detail.stl"
    corner_meta = _export_clip_corner_detail(corner_path)
    print(f"P3 corner detail: {corner_path}  faces={corner_meta['faces']}")

    solid_ext = _implicit_piecewise_xz_mask(
        z_breaks_mm=Z_BREAKS,
        pore_mm=PORES_PIECEWISE,
        origins_x=[s["origin_x_mm"] for s in rep["slabs"]],
        origins_y=[s["origin_y_mm"] for s in rep["slabs"]],
        swap_xy=[s["swap_xy"] for s in rep["slabs"]],
        flips=[s["flip_layer_parity"] for s in rep["slabs"]],
        true_woodpile=False,
    )
    _save_field_xz_png(
        solid_ext,
        OUT_P3 / "P3_piecewise_extrude_field_XZ_midY.png",
        title="Piecewise extrude field proxy (XZ @ Y=0)",
    )

    phase: dict = {
        "phase": 3,
        "extrude": rep,
        "clip_corner_detail": corner_meta,
        "stl_extrude": str(stl_path.resolve()),
        "stl_corner": str(corner_path.resolve()),
    }

    if not skip_implicit:
        mesh_imp, rep_imp = woodpile_piecewise_cylinder_single_pass(
            radius_mm=RADIUS_MM,
            z_breaks_mm=Z_BREAKS,
            pore_mm=PORES_PIECEWISE,
            resolution_mm=0.015,
            true_woodpile=False,
            anchor_mode="center_void",
            alternate_band_orientation=True,
        )
        imp_path = OUT_P3 / "P3_crosshatch_cylinder_D2_implicit_ref.stl"
        write_mesh_report(mesh_imp, imp_path, rep_imp)
        solid_imp = _implicit_piecewise_xz_mask(
            z_breaks_mm=Z_BREAKS,
            pore_mm=PORES_PIECEWISE,
            origins_x=[s["origin_x_mm"] for s in rep_imp["slabs"]],
            origins_y=[s["origin_y_mm"] for s in rep_imp["slabs"]],
            swap_xy=[s["swap_xy"] for s in rep_imp["slabs"]],
            flips=[s["flip_layer_parity"] for s in rep_imp["slabs"]],
            true_woodpile=False,
        )
        _save_field_xz_png(
            solid_imp,
            OUT_P3 / "P3_piecewise_implicit_field_XZ_midY.png",
            title="Piecewise implicit field (XZ @ Y=0)",
        )
        vol_ext = float(rep["volume_mm3"] or 0.0)
        vol_imp = float(rep_imp.get("volume_mm3") or mesh_imp.volume)
        phase["implicit"] = rep_imp
        phase["stl_implicit"] = str(imp_path.resolve())
        phase["volume_delta_pct"] = (
            100.0 * (vol_ext - vol_imp) / vol_imp if vol_imp else None
        )
        print(f"P3 implicit ref: {imp_path}  faces={rep_imp.get('faces', len(mesh_imp.faces))}")

    report_path = OUT_P3 / "P3_report.json"
    report_path.write_text(json.dumps(phase, indent=2), encoding="utf-8")
    return phase


def run_phase4(*, skip_implicit: bool) -> dict:
    OUT_P4.mkdir(parents=True, exist_ok=True)
    lox, loy, anchor = compute_woodpile_xy_origin(
        radius_mm=RADIUS_MM, pore_mm=PORE_UNIFORM, mode="center_void"
    )
    mesh, rep = generate_crosshatch_cylinder(
        pore_mm=PORE_UNIFORM,
        radius_mm=RADIUS_MM,
        height_mm=HEIGHT_MM,
        lattice_origin_x_mm=lox,
        lattice_origin_y_mm=loy,
        true_woodpile=True,
    )
    rep["anchor"] = anchor
    stl_path = OUT_P4 / "P4_true_woodpile_cylinder_D2_extrude.stl"
    write_mesh_report(mesh, stl_path, rep)
    clip_qc = verify_cylinder_clip(
        mesh, center_x_mm=0.0, center_y_mm=0.0, radius_mm=RADIUS_MM
    )
    rep["clip_qc"] = clip_qc
    print(
        f"P4 extrude: {stl_path}  faces={rep['faces']}  "
        f"wt={rep['watertight']}  clip_ok={clip_qc['ok']}"
    )

    solid_ext = _implicit_uniform_xz_mask(
        pore_mm=PORE_UNIFORM,
        origin_x_mm=lox,
        origin_y_mm=loy,
        true_woodpile=True,
    )
    _save_field_xz_png(
        solid_ext,
        OUT_P4 / "P4_true_woodpile_extrude_field_XZ_midY.png",
        title="True woodpile extrude field proxy (XZ @ Y=0)",
    )

    # Layer-shift QC at Z midplanes for layers 2 and 3 (mod-4 shifts).
    shift_planes = []
    for layer_idx in (2, 3):
        z_mid = layer_idx * PORE_UNIFORM + PORE_UNIFORM / 2.0
        shift_planes.append(
            {
                "layer_idx": layer_idx,
                "z_mid_mm": z_mid,
                "strut_axis": strut_axis_for_crosshatch_layer(layer_idx),
                "phase_shift_mm": list(
                    true_woodpile_phase_shift_mm(layer_idx, PORE_UNIFORM)
                ),
            }
        )

    phase: dict = {
        "phase": 4,
        "extrude": rep,
        "true_woodpile_shift_planes": shift_planes,
        "stl_extrude": str(stl_path.resolve()),
    }

    if not skip_implicit:
        mesh_imp, rep_imp = woodpile_piecewise_cylinder_single_pass(
            radius_mm=RADIUS_MM,
            z_breaks_mm=[0.0, HEIGHT_MM],
            pore_mm=[PORE_UNIFORM],
            resolution_mm=0.015,
            true_woodpile=True,
            anchor_mode="center_void",
            alternate_band_orientation=True,
        )
        imp_path = OUT_P4 / "P4_true_woodpile_implicit_ref.stl"
        write_mesh_report(mesh_imp, imp_path, rep_imp)
        solid_imp = _implicit_uniform_xz_mask(
            pore_mm=PORE_UNIFORM,
            origin_x_mm=lox,
            origin_y_mm=loy,
            true_woodpile=True,
        )
        _save_field_xz_png(
            solid_imp,
            OUT_P4 / "P4_true_woodpile_implicit_field_XZ_midY.png",
            title="True woodpile implicit field (XZ @ Y=0)",
        )
        agreement = float(np.mean(solid_ext == solid_imp))
        phase["implicit"] = rep_imp
        phase["stl_implicit"] = str(imp_path.resolve())
        phase["field_slice_agreement"] = agreement
        print(
            f"P4 implicit ref: {imp_path}  field agreement={agreement:.4f}"
        )

    report_path = OUT_P4 / "P4_report.json"
    report_path.write_text(json.dumps(phase, indent=2), encoding="utf-8")
    return phase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-implicit",
        action="store_true",
        help="Skip slow marching-cubes reference meshes.",
    )
    args = parser.parse_args()

    p3 = run_phase3(skip_implicit=args.skip_implicit)
    p4 = run_phase4(skip_implicit=args.skip_implicit)

    summary = {
        "phases": [3, 4],
        "P3": {
            "watertight": p3["extrude"]["watertight"],
            "clip_ok": p3["extrude"]["clip_qc"]["ok"],
            "interface_ok": p3["extrude"]["interface_layer_qc"]["ok"],
        },
        "P4": {
            "watertight": p4["extrude"]["watertight"],
            "clip_ok": p4["extrude"].get("clip_qc", {}).get("ok"),
            "true_woodpile": True,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
