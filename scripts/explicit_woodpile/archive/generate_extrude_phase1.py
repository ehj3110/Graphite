#!/usr/bin/env python
"""
Phase 1 — full uniform-pore cross-hatch extrusion + implicit MC comparison.

Run from repo root::

    python scripts/explicit_woodpile/generate_extrude_phase1.py
    python scripts/explicit_woodpile/generate_extrude_phase1.py --skip-implicit
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
    generate_crosshatch_box,
    generate_crosshatch_cylinder,
    mid_plane_solid_fraction,
    write_mesh_report,
)
from graphite.implicit.piecewise_woodpile import (
    woodpile_piecewise_box_single_pass,
    woodpile_piecewise_cylinder_single_pass,
)
from graphite.math.woodpile import evaluate_woodpile
from graphite.math.woodpile_anchor import (
    compute_woodpile_xy_origin,
    compute_woodpile_xy_origin_box,
)

OUT_DIR = _REPO_ROOT / "outputs" / "explicit_woodpile" / "P1"


def _comparison_entry(mesh, report: dict) -> dict:
    entry = dict(report)
    entry["mid_y_solid_fraction"] = mid_plane_solid_fraction(
        mesh, center_mm=0.5, x_range=(0.0, 1.0), z_range=(0.0, 1.0), n_samples=128
    )
    return entry


def _box_domain_sdf(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
    """Positive outside 1 mm³ box ``[0,1]³``."""
    return np.maximum.reduce(
        [
            xx - 1.0,
            -xx,
            yy - 1.0,
            -yy,
            zz - 1.0,
            -zz,
        ]
    )


def _implicit_box_mid_y_mask(
    *,
    pore_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
    n: int = 250,
    y_mm: float = 0.5,
) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n)
    zs = np.linspace(0.0, 1.0, n)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = np.full_like(xx, float(y_mm))
    field = evaluate_woodpile(
        xx,
        yy,
        zz,
        pore_size=float(pore_mm),
        true_woodpile=False,
        origin_x=float(origin_x_mm),
        origin_y=float(origin_y_mm),
    )
    sdf = _box_domain_sdf(xx, yy, zz)
    return np.maximum(field, sdf) <= 0.0


def _implicit_box_mid_y_solid_fraction(
    *,
    pore_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
    n: int = 128,
) -> float:
    solid = _implicit_box_mid_y_mask(
        pore_mm=pore_mm,
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
        n=n,
    )
    return float(solid.mean())


def _save_xz_slice_png(mesh: trimesh.Trimesh, path: Path, *, title: str) -> None:
    n = 250
    xs = np.linspace(0.0, 1.0, n)
    zs = np.linspace(0.0, 1.0, n)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    y_vals = [0.488, 0.5, 0.512]
    solid = np.zeros(xx.shape, dtype=bool)
    for y in y_vals:
        pts = np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()])
        solid |= mesh.contains(pts).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        solid.T,
        origin="lower",
        extent=(0, 1, 0, 1),
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


def _save_implicit_xz_slice_png(
    path: Path,
    *,
    title: str,
    pore_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
) -> None:
    solid = _implicit_box_mid_y_mask(
        pore_mm=pore_mm,
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        solid.T,
        origin="lower",
        extent=(0, 1, 0, 1),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-implicit",
        action="store_true",
        help="Skip slow marching-cubes reference meshes.",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase_report: dict = {"phase": 1, "cases": {}}

    # --- 1 mm cube, pore 200 µm ---
    pore_box = 0.2
    box_ox, box_oy, box_anchor = compute_woodpile_xy_origin_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        pore_mm=pore_box,
        mode="center_void",
        origin_x_mm=0.0,
        origin_y_mm=0.0,
    )
    mesh_box, rep_box = generate_crosshatch_box(
        pore_mm=pore_box,
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        lattice_origin_x_mm=box_ox,
        lattice_origin_y_mm=box_oy,
    )
    rep_box["anchor"] = box_anchor
    box_path = OUT_DIR / "P1_crosshatch_box1mm_P200_extrude.stl"
    write_mesh_report(mesh_box, box_path, rep_box)
    box_entry = _comparison_entry(mesh_box, rep_box)
    _save_xz_slice_png(
        mesh_box,
        OUT_DIR / "P1_crosshatch_box1mm_P200_extrude_XZ_midY.png",
        title="Extrude cross-hatch 1 mm³ (P200 µm)",
    )
    print(f"Box extrude: {box_path}  faces={rep_box['faces']}  wt={rep_box['watertight']}")

    if not args.skip_implicit:
        mesh_imp, rep_imp = woodpile_piecewise_box_single_pass(
            width_x_mm=1.0,
            depth_y_mm=1.0,
            height_mm=1.0,
            z_breaks_mm=[0.0, 1.0],
            pore_mm=[pore_box],
            resolution_mm=0.015,
            true_woodpile=False,
            anchor_mode="center_void",
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
        )
        imp_path = OUT_DIR / "P1_crosshatch_box1mm_P200_implicit_ref.stl"
        imp_report = {
            "generator": "implicit_mc",
            "faces": int(len(mesh_imp.faces)),
            "watertight": bool(mesh_imp.is_watertight),
            "volume_mm3": float(mesh_imp.volume) if mesh_imp.volume else None,
            "resolution_mm": 0.015,
            **{k: rep_imp.get(k) for k in ("slabs",) if k in rep_imp},
        }
        write_mesh_report(mesh_imp, imp_path, imp_report)
        imp_sf = _implicit_box_mid_y_solid_fraction(
            pore_mm=pore_box,
            origin_x_mm=box_ox,
            origin_y_mm=box_oy,
        )
        _save_implicit_xz_slice_png(
            OUT_DIR / "P1_crosshatch_box1mm_P200_implicit_ref_XZ_midY.png",
            title="Implicit field cross-hatch 1 mm³ (P200 µm)",
            pore_mm=pore_box,
            origin_x_mm=box_ox,
            origin_y_mm=box_oy,
        )
        vol_ext = float(box_entry["volume_mm3"] or 0.0)
        vol_imp = float(imp_report["volume_mm3"] or 0.0)
        box_entry["implicit_ref"] = {
            "stl": str(imp_path.resolve()),
            "volume_mm3": vol_imp,
            "volume_delta_pct": (
                100.0 * (vol_ext - vol_imp) / vol_imp if vol_imp > 0 else None
            ),
            "faces": imp_report["faces"],
            "mid_y_solid_fraction": imp_sf,
            "mid_y_solid_fraction_method": "implicit_field",
        }
        print(f"Box implicit ref: {imp_path}  faces={imp_report['faces']}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, png, label in zip(
            axes,
            [
                OUT_DIR / "P1_crosshatch_box1mm_P200_extrude_XZ_midY.png",
                OUT_DIR / "P1_crosshatch_box1mm_P200_implicit_ref_XZ_midY.png",
            ],
            ["Extrude", "Implicit field"],
        ):
            img = plt.imread(png)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
        fig.suptitle("P1 cross-hatch 1 mm³ — XZ @ mid-Y (P200 µm)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "P1_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    phase_report["cases"]["box1mm_P200"] = box_entry

    # --- Cylinder Ø2 mm × 2.7 mm, pore 800 µm ---
    pore_cyl = 0.8
    radius = 1.0
    height = 2.7
    ox, oy, anchor_meta = compute_woodpile_xy_origin(
        radius_mm=radius, pore_mm=pore_cyl, mode="center_void"
    )
    mesh_cyl, rep_cyl = generate_crosshatch_cylinder(
        pore_mm=pore_cyl,
        radius_mm=radius,
        height_mm=height,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        origin_z_mm=0.0,
        lattice_origin_x_mm=ox,
        lattice_origin_y_mm=oy,
    )
    cyl_path = OUT_DIR / "P1_crosshatch_cylinder_D2_H2p7_P800_extrude.stl"
    rep_cyl["anchor"] = anchor_meta
    write_mesh_report(mesh_cyl, cyl_path, rep_cyl)
    print(f"Cylinder extrude: {cyl_path}  faces={rep_cyl['faces']}  wt={rep_cyl['watertight']}")

    if not args.skip_implicit:
        mesh_cyl_imp, rep_cyl_imp = woodpile_piecewise_cylinder_single_pass(
            radius_mm=radius,
            z_breaks_mm=[0.0, height],
            pore_mm=[pore_cyl],
            resolution_mm=0.015,
            true_woodpile=False,
            anchor_mode="center_void",
        )
        cyl_imp_path = OUT_DIR / "P1_crosshatch_cylinder_D2_H2p7_P800_implicit_ref.stl"
        cyl_imp_vol = float(mesh_cyl_imp.volume) if mesh_cyl_imp.volume else None
        cyl_imp_report = {
            "generator": "implicit_mc",
            "faces": int(len(mesh_cyl_imp.faces)),
            "watertight": bool(mesh_cyl_imp.is_watertight),
            "volume_mm3": cyl_imp_vol,
            "resolution_mm": 0.015,
        }
        write_mesh_report(mesh_cyl_imp, cyl_imp_path, cyl_imp_report)
        vol_ext = float(rep_cyl["volume_mm3"] or 0.0)
        vol_imp = float(cyl_imp_report["volume_mm3"] or 0.0)
        rep_cyl["implicit_ref"] = {
            "stl": str(cyl_imp_path.resolve()),
            "volume_mm3": vol_imp,
            "volume_delta_pct": (
                100.0 * (vol_ext - vol_imp) / vol_imp if vol_imp > 0 else None
            ),
            "faces": cyl_imp_report["faces"],
        }
        print(f"Cylinder implicit ref: {cyl_imp_path}  faces={cyl_imp_report['faces']}")

    phase_report["cases"]["cylinder_D2_H2p7_P800"] = rep_cyl

    report_path = OUT_DIR / "P1_report.json"
    report_path.write_text(json.dumps(phase_report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
