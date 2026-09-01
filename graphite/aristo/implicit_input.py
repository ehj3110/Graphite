"""
Implicit-field boundary mesh generation for Aristo (default input path).

Evaluates a piecewise Split-P implicit field, extracts an MC skin in memory,
repairs topology, and optionally drops floating islands. STL file I/O is **not**
required — use ``--stl`` on CLI scripts only as a developer override.

True field → volume mesh (no MC middleman) is not implemented yet; this module
is the canonical **implicit-first** handoff to Gmsh ``single_surface`` meshing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trimesh

from graphite.implicit.calibration import CalibrationConfig, calibrate_tau_at_fixed_period
from graphite.implicit.piecewise_bands import (
    splitp_piecewise_box_single_pass,
    splitp_piecewise_cylinder_single_pass,
)
from graphite.aristo.volume_mesh_inspect import remove_floating_islands


@dataclass
class AristoImplicitSpec:
    """Piecewise Split-P implicit lattice spec (two Z bands)."""

    domain: str = "cylinder"  # cylinder | box
    resolution_mm: float = 0.02
    diameter_mm: float = 2.0
    height_mm: float = 2.0
    width_x_mm: float = 1.0
    depth_y_mm: float = 1.0
    band_height_mm: float = 1.0
    L_bottom_mm: float = 0.5
    L_top_mm: float = 1.0
    solid_fraction: float = 0.33
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    origin_z_mm: float = 0.0
    remove_islands: bool = True
    repair_mesh: bool = True
    stem: str | None = None


def repair_implicit_mc_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light trimesh repair after marching cubes."""
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    return m


def default_stem_from_spec(spec: AristoImplicitSpec) -> str:
    sf = int(round(spec.solid_fraction * 100))
    h = spec.height_mm
    if spec.domain == "cylinder":
        return (
            f"SplitP_Cylinder{spec.diameter_mm:g}x{h:g}mm_"
            f"piecewise_L{int(spec.L_bottom_mm * 1000)}umBottom_"
            f"L{int(spec.L_top_mm * 1000)}umTop_SF{sf}"
        )
    return (
        f"SplitP_Box{spec.width_x_mm:g}x{spec.depth_y_mm:g}x{h:g}mm_"
        f"piecewise_SF{sf}"
    )


def build_piecewise_splitp_boundary_mesh(
    spec: AristoImplicitSpec,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """
    Evaluate piecewise Split-P implicit field → watertight boundary trimesh.

    Returns ``(mesh, report)`` with calibration and pipeline metadata.
    """
    height_mm = float(spec.height_mm)
    band_h = float(spec.band_height_mm)
    n_bands = round(height_mm / band_h)
    if abs(n_bands * band_h - height_mm) > 1e-6 or n_bands != 2:
        raise ValueError(
            "Piecewise implicit spec expects exactly 2 bands "
            "(height_mm / band_height_mm == 2)."
        )

    sf = float(spec.solid_fraction)
    z_breaks = [0.0, band_h, height_mm]
    cfg = CalibrationConfig(
        pore_tolerance_mm=0.03,
        solid_fraction_tolerance=0.02,
        sample_resolution_mm=0.03,
        max_iterations=14,
    )
    cal_bottom = calibrate_tau_at_fixed_period(
        "Split-P", float(spec.L_bottom_mm), sf, config=cfg
    )
    cal_top = calibrate_tau_at_fixed_period(
        "Split-P", float(spec.L_top_mm), sf, config=cfg
    )
    L_cal = [cal_bottom.calibrated_L_mm, cal_top.calibrated_L_mm]
    tau_bands = [cal_bottom.calibrated_tau, cal_top.calibrated_tau]

    domain = spec.domain.strip().lower()
    if domain == "cylinder":
        mesh, meta = splitp_piecewise_cylinder_single_pass(
            radius_mm=float(spec.diameter_mm) / 2.0,
            height_mm=height_mm,
            resolution_mm=float(spec.resolution_mm),
            z_breaks_mm=z_breaks,
            L_mm=L_cal,
            tau=tau_bands,
        )
    elif domain == "box":
        mesh, meta = splitp_piecewise_box_single_pass(
            width_x_mm=float(spec.width_x_mm),
            depth_y_mm=float(spec.depth_y_mm),
            height_mm=height_mm,
            resolution_mm=float(spec.resolution_mm),
            z_breaks_mm=z_breaks,
            L_mm=L_cal,
            tau=tau_bands,
            origin_x=float(spec.origin_x_mm),
            origin_y=float(spec.origin_y_mm),
            origin_z=float(spec.origin_z_mm),
        )
    else:
        raise ValueError(f"domain must be cylinder or box, got {spec.domain!r}.")

    if spec.repair_mesh:
        mesh = repair_implicit_mc_mesh(mesh)

    island_meta: dict[str, Any] | None = None
    if spec.remove_islands:
        mesh, island_meta = remove_floating_islands(mesh)

    report: dict[str, Any] = {
        "fea_input_mode": "implicit",
        "implicit_spec": {
            "domain": domain,
            "resolution_mm": float(spec.resolution_mm),
            "height_mm": height_mm,
            "band_height_mm": band_h,
            "solid_fraction_target": sf,
            "L_bottom_mm": float(spec.L_bottom_mm),
            "L_top_mm": float(spec.L_top_mm),
        },
        "calibration": {
            "bottom": cal_bottom.__dict__,
            "top": cal_top.__dict__,
        },
        "implicit_meta": meta,
        "island_cleanup": island_meta,
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume),
        "stem": spec.stem or default_stem_from_spec(spec),
    }
    return mesh, report


def add_piecewise_implicit_args(
    parser: argparse.ArgumentParser,
    *,
    default_h: float | None = None,
) -> None:
    """Register implicit-field CLI flags (default Aristo input)."""
    parser.add_argument(
        "--domain",
        choices=("cylinder", "box"),
        default="cylinder",
        help="Implicit domain shape (default when --stl is not set).",
    )
    parser.add_argument("--resolution-mm", type=float, default=0.02)
    parser.add_argument("--diameter-mm", type=float, default=2.0)
    parser.add_argument("--height-mm", type=float, default=2.0)
    parser.add_argument("--width-x-mm", type=float, default=1.0)
    parser.add_argument("--depth-y-mm", type=float, default=1.0)
    parser.add_argument("--band-height-mm", type=float, default=1.0)
    parser.add_argument("--L-bottom-mm", type=float, default=0.5)
    parser.add_argument("--L-top-mm", type=float, default=1.0)
    parser.add_argument("--solid-fraction", type=float, default=0.33)
    parser.add_argument("--no-remove-islands", action="store_true")
    if default_h is not None:
        parser.add_argument("--h", type=float, default=default_h, help="FEA mesh scale (mm)")


def add_stl_dev_arg(parser: argparse.ArgumentParser) -> None:
    """Register optional STL override (developer / legacy path)."""
    parser.add_argument(
        "--stl",
        type=Path,
        default=None,
        help="Developer override: load boundary from STL instead of implicit field.",
    )


def implicit_spec_from_args(args: argparse.Namespace) -> AristoImplicitSpec:
    return AristoImplicitSpec(
        domain=str(args.domain),
        resolution_mm=float(args.resolution_mm),
        diameter_mm=float(args.diameter_mm),
        height_mm=float(args.height_mm),
        width_x_mm=float(args.width_x_mm),
        depth_y_mm=float(args.depth_y_mm),
        band_height_mm=float(args.band_height_mm),
        L_bottom_mm=float(args.L_bottom_mm),
        L_top_mm=float(args.L_top_mm),
        solid_fraction=float(args.solid_fraction),
        remove_islands=not bool(getattr(args, "no_remove_islands", False)),
        stem=getattr(args, "stem", None),
    )
