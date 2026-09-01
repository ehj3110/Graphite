#!/usr/bin/env python
"""
Phase 2 — piecewise cross-hatch box (P139 bottom / P277 top) + implicit ref.

Run from repo root::

    python scripts/explicit_woodpile/generate_extrude_phase2.py
    python scripts/explicit_woodpile/generate_extrude_phase2.py --skip-implicit
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
    generate_piecewise_crosshatch_box,
    write_mesh_report,
)
from graphite.implicit.piecewise_woodpile import woodpile_piecewise_box_single_pass
from graphite.math.woodpile import evaluate_woodpile_piecewise_cylinder

OUT_DIR = _REPO_ROOT / "outputs" / "explicit_woodpile" / "P2"

Z_BREAKS = [0.0, 0.5, 1.0]
PORES = [0.1386, 0.2771]


def _box_domain_sdf(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
    return np.maximum.reduce([xx - 1.0, -xx, yy - 1.0, -yy, zz - 1.0, -zz])


def _implicit_piecewise_xz_mask(
    *,
    z_breaks_mm: list[float],
    pore_mm: list[float],
    origins_x: list[float],
    origins_y: list[float],
    swap_xy: list[bool],
    flips: list[bool],
    n: int = 250,
    y_mm: float = 0.5,
) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n)
    zs = np.linspace(0.0, 1.0, n)
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
        true_woodpile=False,
    )
    sdf = _box_domain_sdf(xx, yy, zz)
    return np.maximum(field, sdf) <= 0.0


def _save_xz_slice_png(mesh: trimesh.Trimesh, path: Path, *, title: str) -> None:
    n = 250
    xs = np.linspace(0.0, 1.0, n)
    zs = np.linspace(0.0, 1.0, n)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    solid = np.zeros(xx.shape, dtype=bool)
    for y in (0.488, 0.5, 0.512):
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
    rep_extrude: dict,
) -> None:
    slabs = rep_extrude["slabs"]
    solid = _implicit_piecewise_xz_mask(
        z_breaks_mm=Z_BREAKS,
        pore_mm=PORES,
        origins_x=[s["origin_x_mm"] for s in slabs],
        origins_y=[s["origin_y_mm"] for s in slabs],
        swap_xy=[s["swap_xy"] for s in slabs],
        flips=[s["flip_layer_parity"] for s in slabs],
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


def _save_xy_interface_png(mesh: trimesh.Trimesh, path: Path, *, z_mm: float, title: str) -> None:
    n = 250
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    solid = np.zeros(xx.shape, dtype=bool)
    for z in (z_mm - 0.004, z_mm, z_mm + 0.004):
        pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
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
    ax.set_ylabel("Y (mm)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-implicit", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_ext, rep_ext = generate_piecewise_crosshatch_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=Z_BREAKS,
        pore_mm=PORES,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    ext_path = OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude.stl"
    write_mesh_report(mesh_ext, ext_path, rep_ext)
    rot = rep_ext.get("interface_layer_qc", rep_ext.get("band_rotation_qc", {}))
    print(
        "Interface layer QC:",
        "ok=",
        rot.get("ok"),
        rot.get("checks"),
    )
    _save_xz_slice_png(
        mesh_ext,
        OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XZ_midY.png",
        title="Extrude piecewise P139/P277 (XZ @ mid-Y)",
    )
    _save_xy_interface_png(
        mesh_ext,
        OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XY_below_interface.png",
        z_mm=0.49,
        title="Extrude XY @ Z=0.49 mm (below interface)",
    )
    _save_xy_interface_png(
        mesh_ext,
        OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XY_above_interface.png",
        z_mm=0.51,
        title="Extrude XY @ Z=0.51 mm (above interface)",
    )
    print(f"Extrude: {ext_path}  faces={rep_ext['faces']}  wt={rep_ext['watertight']}")

    phase_report: dict = {"phase": 2, "cases": {"piecewise_P139_P277": rep_ext}}

    if not args.skip_implicit:
        mesh_imp, rep_imp = woodpile_piecewise_box_single_pass(
            width_x_mm=1.0,
            depth_y_mm=1.0,
            height_mm=1.0,
            z_breaks_mm=Z_BREAKS,
            pore_mm=PORES,
            resolution_mm=0.015,
            true_woodpile=False,
            anchor_mode="center_void",
            alternate_band_orientation=True,
        )
        imp_path = OUT_DIR / "P2_piecewise_box1mm_P139_P277_implicit_ref.stl"
        imp_report = {
            "generator": "implicit_mc",
            "faces": int(len(mesh_imp.faces)),
            "watertight": bool(mesh_imp.is_watertight),
            "volume_mm3": float(mesh_imp.volume) if mesh_imp.volume else None,
            "resolution_mm": 0.015,
            "slabs": rep_imp.get("slabs"),
        }
        write_mesh_report(mesh_imp, imp_path, imp_report)
        _save_implicit_xz_slice_png(
            OUT_DIR / "P2_piecewise_box1mm_P139_P277_implicit_ref_XZ_midY.png",
            title="Implicit field piecewise P139/P277 (XZ @ mid-Y)",
            rep_extrude=rep_ext,
        )

        vol_ext = float(rep_ext["volume_mm3"] or 0.0)
        vol_imp = float(imp_report["volume_mm3"] or 0.0)
        rep_ext["implicit_ref"] = {
            "stl": str(imp_path.resolve()),
            "volume_mm3": vol_imp,
            "volume_delta_pct": (
                100.0 * (vol_ext - vol_imp) / vol_imp if vol_imp > 0 else None
            ),
            "faces": imp_report["faces"],
        }
        print(f"Implicit ref: {imp_path}  faces={imp_report['faces']}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, png, label in zip(
            axes,
            [
                OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XZ_midY.png",
                OUT_DIR / "P2_piecewise_box1mm_P139_P277_implicit_ref_XZ_midY.png",
            ],
            ["Extrude", "Implicit field"],
        ):
            ax.imshow(plt.imread(png))
            ax.set_title(label)
            ax.axis("off")
        fig.suptitle("P2 piecewise 1 mm³ — XZ @ mid-Y (P139 / P277 µm)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "P2_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, png, label in zip(
            axes,
            [
                OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XY_below_interface.png",
                OUT_DIR / "P2_piecewise_box1mm_P139_P277_extrude_XY_above_interface.png",
            ],
            ["Below Z=0.5 (band 0, last layer)", "Above Z=0.5 (band 1, first layer)"],
        ):
            ax.imshow(plt.imread(png))
            ax.set_title(label)
            ax.axis("off")
        fig.suptitle("P2 interface Z=0.5 — ⊥ strut layers (printability)")
        fig.tight_layout()
        fig.savefig(
            OUT_DIR / "P2_interface_XY_comparison.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)

    report_path = OUT_DIR / "P2_report.json"
    report_path.write_text(json.dumps(phase_report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
