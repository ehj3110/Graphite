"""Generate Split-P piecewise + linear graded 1 mm³ cube STLs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh
from skimage.measure import marching_cubes

from graphite.aristo.volume_mesh_inspect import remove_floating_islands
from graphite.case_studies.cube_1mm.specs import (
    BAND_HEIGHT_MM,
    BOOLEAN_TRIM_MARGIN_MM,
    CUBE_SIZE_MM,
    IMPLICIT_RESOLUTION_BOOLEAN_TRIM_MM,
    IMPLICIT_RESOLUTION_MM,
    SPLITP_L_BOTTOM_MM,
    SPLITP_L_TOP_MM,
    SPLITP_PHASE_ORIGIN_MM,
    SPLITP_PIECEWISE_BOTTOM_BAND_PHASE_SHIFT_X_MM,
    SPLITP_SF,
    default_output_dir,
    repo_root,
    splitp_linear_graded_stem,
    splitp_piecewise_phase_shift_x_stem,
    splitp_piecewise_stem,
)
from graphite.explicit.geometry_module import boolean_intersect_with_cad
from graphite.geometry.masking import axis_aligned_box_grid, axis_aligned_box_sdf
from graphite.implicit.calibration import calibrate_tau_at_fixed_period
from graphite.implicit.piecewise_bands import splitp_piecewise_box_single_pass
from graphite.math.tpms import evaluate_tpms_phase

_user_spec = importlib.util.spec_from_file_location(
    "user_spec_cylinders",
    repo_root() / "scripts" / "generate_three_cylinder_lattices_user_spec.py",
)
_user_mod = importlib.util.module_from_spec(_user_spec)
assert _user_spec.loader is not None
_user_spec.loader.exec_module(_user_mod)


def _design_cube_trimesh(
    *,
    size_mm: float = CUBE_SIZE_MM,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
) -> trimesh.Trimesh:
    """Axis-aligned design cube with corner at ``origin`` and edge ``size_mm``."""
    s = float(size_mm)
    cube = trimesh.creation.box(extents=[s, s, s])
    cube.apply_translation(
        [
            float(origin_x) + 0.5 * s,
            float(origin_y) + 0.5 * s,
            float(origin_z) + 0.5 * s,
        ]
    )
    return cube


def trim_mesh_to_design_cube(
    mesh: trimesh.Trimesh,
    *,
    size_mm: float = CUBE_SIZE_MM,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
) -> tuple[trimesh.Trimesh, dict]:
    """Boolean ∩ lattice with the design cube (flat CAD faces)."""
    cad = _design_cube_trimesh(
        size_mm=size_mm,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    trimmed, elapsed_s = boolean_intersect_with_cad(mesh, cad)
    meta = {
        "boolean_trim": "manifold3d_intersect_design_cube",
        "boolean_trim_seconds": float(elapsed_s),
        "design_cube_origin_mm": [float(origin_x), float(origin_y), float(origin_z)],
        "design_cube_size_mm": float(size_mm),
        "volume_before_trim_mm3": float(mesh.volume),
        "volume_after_trim_mm3": float(trimmed.volume),
        "faces_before_trim": int(len(mesh.faces)),
        "faces_after_trim": int(len(trimmed.faces)),
    }
    return trimmed, meta


def _finalize_splitp_surface(
    mesh: trimesh.Trimesh,
    *,
    boolean_trim: bool,
    size_mm: float = CUBE_SIZE_MM,
) -> tuple[trimesh.Trimesh, dict]:
    """Optional CAD Boolean face trim, light repair, floating-island removal."""
    meta: dict = {}
    working = mesh
    if boolean_trim:
        working, trim_meta = trim_mesh_to_design_cube(working, size_mm=size_mm)
        meta.update(trim_meta)
    meta["watertight_before_repair"] = bool(working.is_watertight)
    working = _user_mod._repair_mesh_after_implicit_crop(working)
    meta["watertight_after_repair"] = bool(working.is_watertight)
    working, island_meta = remove_floating_islands(working)
    meta["island_cleanup"] = island_meta
    meta["faces"] = int(len(working.faces))
    meta["vertices"] = int(len(working.vertices))
    meta["volume_mm3"] = float(working.volume)
    return working, meta


def _oversize_generation_box(
    *,
    size_mm: float,
    margin_mm: float,
) -> tuple[float, float, float, float]:
    """Return (gen_size, origin_x, origin_y, origin_z) for oversize MC domain."""
    m = float(margin_mm)
    return float(size_mm) + 2.0 * m, -m, -m, -m


def _splitp_mesh_graded_box_single_pass(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    height_z_mm: float,
    resolution_mm: float,
    z_knots_mm: np.ndarray,
    L_knots_mm: np.ndarray,
    tau_knots: np.ndarray,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
    phase_origin_x: float = 0.0,
    phase_origin_y: float = 0.0,
    grading_mode: str = "jacobian_integrated",
) -> tuple[trimesh.Trimesh, dict]:
    if grading_mode not in ("jacobian_integrated", "chirp"):
        raise ValueError(
            f"grading_mode must be 'jacobian_integrated' or 'chirp', got {grading_mode!r}."
        )

    X, Y, Z, grid_origin, spacing = axis_aligned_box_grid(
        width_x_mm,
        depth_y_mm,
        height_z_mm,
        resolution_mm,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )
    box_sdf = axis_aligned_box_sdf(
        X,
        Y,
        Z,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        height_z_mm=height_z_mm,
    )
    L_voxel = np.interp(Z, z_knots_mm, L_knots_mm)
    tau_voxel = np.interp(Z, z_knots_mm, tau_knots)
    omega = 2.0 * np.pi / np.maximum(L_voxel, 1e-6)
    if grading_mode == "jacobian_integrated":
        W_phase = _user_mod._phase_w_from_L_profile(Z, z_knots_mm, L_knots_mm)
    else:
        W_phase = Z * omega
    U = (X - phase_origin_x) * omega
    V = (Y - phase_origin_y) * omega
    F = evaluate_tpms_phase("split-p", U, V, W_phase)
    tpms_field = np.abs(F) - tau_voxel
    final_field = np.maximum(tpms_field, box_sdf)

    verts, faces, _n, _v = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=spacing,
    )
    verts = verts + grid_origin
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    meta = {
        "pipeline": (
            "analytic box SDF + linear L(z), tau(z) + "
            + (
                "integrated W(z) Jacobian"
                if grading_mode == "jacobian_integrated"
                else "direct chirp W=Z*omega"
            )
        ),
        "boundary_sdf": "analytic_box",
        "grading_mode": grading_mode,
        "domain_mm": (width_x_mm, depth_y_mm, height_z_mm),
        "origin_mm": (origin_x, origin_y, origin_z),
        "phase_origin_mm": (phase_origin_x, phase_origin_y),
        "z_knots_mm": np.asarray(z_knots_mm, dtype=float).tolist(),
        "L_knots_mm": np.asarray(L_knots_mm, dtype=float).tolist(),
        "tau_knots": np.asarray(tau_knots, dtype=float).tolist(),
    }
    return mesh_out, meta


def _preview_mesh_png(mesh: trimesh.Trimesh, path: Path, title: str) -> None:
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(960, 720))
    plotter.add_mesh(
        pv.wrap(mesh),
        color="#5b9bd5",
        smooth_shading=True,
        specular=0.35,
        specular_power=20,
    )
    plotter.add_text(title, position="upper_left", font_size=11, color="black")
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.background_color = "white"
    plotter.screenshot(str(path))
    plotter.close()


def _montage_png(panel_paths: list[tuple[Path, str]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(panel_paths), figsize=(6.5 * len(panel_paths), 6.5))
    if len(panel_paths) == 1:
        axes = [axes]
    for ax, (img_path, title) in zip(axes, panel_paths):
        ax.imshow(plt.imread(img_path))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_piecewise_splitp_cube_mesh(
    *,
    band_phase_origin_x_mm: Sequence[float] | None = None,
    band_phase_origin_y_mm: Sequence[float] | None = None,
    resolution_mm: float = IMPLICIT_RESOLUTION_MM,
    solid_fraction: float = SPLITP_SF,
    boolean_trim: bool = False,
    boolean_trim_margin_mm: float = BOOLEAN_TRIM_MARGIN_MM,
) -> tuple[trimesh.Trimesh, dict]:
    """Mesh piecewise Split-P cube with optional per-band XY phase offsets.

    When ``boolean_trim`` is True, march on an oversized box then Boolean ∩ the
    design cube so exterior faces are true CAD planes (baseball-style cleanup).
    """
    size_mm = CUBE_SIZE_MM
    band_h = BAND_HEIGHT_MM
    sf = float(solid_fraction)
    cal_bottom = calibrate_tau_at_fixed_period(
        lattice_type="Split-P",
        period_mm=SPLITP_L_BOTTOM_MM,
        target_solid_fraction=sf,
    )
    cal_top = calibrate_tau_at_fixed_period(
        lattice_type="Split-P",
        period_mm=SPLITP_L_TOP_MM,
        target_solid_fraction=sf,
    )
    L_cal = [float(cal_bottom.calibrated_L_mm), float(cal_top.calibrated_L_mm)]
    tau_bands = [float(cal_bottom.calibrated_tau), float(cal_top.calibrated_tau)]
    z_breaks = [0.0, band_h, size_mm]

    if boolean_trim:
        gen_size, ox, oy, oz = _oversize_generation_box(
            size_mm=size_mm, margin_mm=boolean_trim_margin_mm
        )
    else:
        gen_size, ox, oy, oz = size_mm, 0.0, 0.0, 0.0

    mesh, meta = splitp_piecewise_box_single_pass(
        width_x_mm=gen_size,
        depth_y_mm=gen_size,
        height_mm=gen_size,
        resolution_mm=resolution_mm,
        z_breaks_mm=z_breaks,
        L_mm=L_cal,
        tau=tau_bands,
        origin_x=ox,
        origin_y=oy,
        origin_z=oz,
        band_phase_origin_x_mm=band_phase_origin_x_mm,
        band_phase_origin_y_mm=band_phase_origin_y_mm,
    )
    mesh, fin_meta = _finalize_splitp_surface(
        mesh, boolean_trim=boolean_trim, size_mm=size_mm
    )
    meta.update(fin_meta)
    meta["calibrated_L_mm"] = L_cal
    meta["calibrated_tau"] = tau_bands
    meta["resolution_mm"] = float(resolution_mm)
    meta["boolean_trim_enabled"] = bool(boolean_trim)
    meta["boolean_trim_margin_mm"] = float(boolean_trim_margin_mm) if boolean_trim else 0.0
    meta["design_size_mm"] = float(size_mm)
    return mesh, meta


def build_linear_graded_splitp_cube_mesh(
    *,
    resolution_mm: float = IMPLICIT_RESOLUTION_MM,
    solid_fraction: float = SPLITP_SF,
    grading_mode: str = "jacobian_integrated",
    phase_origin_x_mm: float = SPLITP_PHASE_ORIGIN_MM[0],
    phase_origin_y_mm: float = SPLITP_PHASE_ORIGIN_MM[1],
    boolean_trim: bool = False,
    boolean_trim_margin_mm: float = BOOLEAN_TRIM_MARGIN_MM,
) -> tuple[trimesh.Trimesh, dict, list]:
    """Mesh linear graded Split-P cube; returns mesh, meta, calibration knot objects."""
    size_mm = CUBE_SIZE_MM
    sf = float(solid_fraction)
    z_pts = np.array([0.0, 0.5 * size_mm, size_mm], dtype=float)
    L_pts = np.array(
        [SPLITP_L_BOTTOM_MM, 0.5 * (SPLITP_L_BOTTOM_MM + SPLITP_L_TOP_MM), SPLITP_L_TOP_MM],
        dtype=float,
    )
    cal_knots = [
        calibrate_tau_at_fixed_period(
            lattice_type="Split-P",
            period_mm=float(L),
            target_solid_fraction=sf,
        )
        for L in L_pts
    ]
    L_cal = [float(c.calibrated_L_mm) for c in cal_knots]
    tau_knots = [float(c.calibrated_tau) for c in cal_knots]

    if boolean_trim:
        gen_size, ox, oy, oz = _oversize_generation_box(
            size_mm=size_mm, margin_mm=boolean_trim_margin_mm
        )
    else:
        gen_size, ox, oy, oz = size_mm, 0.0, 0.0, 0.0

    mesh, meta = _splitp_mesh_graded_box_single_pass(
        width_x_mm=gen_size,
        depth_y_mm=gen_size,
        height_z_mm=gen_size,
        resolution_mm=resolution_mm,
        z_knots_mm=z_pts,
        L_knots_mm=np.asarray(L_cal, dtype=float),
        tau_knots=np.asarray(tau_knots, dtype=float),
        origin_x=ox,
        origin_y=oy,
        origin_z=oz,
        phase_origin_x=float(phase_origin_x_mm),
        phase_origin_y=float(phase_origin_y_mm),
        grading_mode=grading_mode,
    )
    mesh, fin_meta = _finalize_splitp_surface(
        mesh, boolean_trim=boolean_trim, size_mm=size_mm
    )
    meta.update(fin_meta)
    meta["z_knots_mm"] = z_pts.tolist()
    meta["period_knots_mm"] = L_pts.tolist()
    meta["calibrated_L_mm"] = L_cal
    meta["calibrated_tau"] = tau_knots
    meta["resolution_mm"] = float(resolution_mm)
    meta["boolean_trim_enabled"] = bool(boolean_trim)
    meta["boolean_trim_margin_mm"] = float(boolean_trim_margin_mm) if boolean_trim else 0.0
    meta["design_size_mm"] = float(size_mm)
    return mesh, meta, cal_knots


def regenerate_case_study_splitp_stls(
    *,
    out_dir: Path | None = None,
    resolution_mm: float = IMPLICIT_RESOLUTION_BOOLEAN_TRIM_MM,
    boolean_trim_margin_mm: float = BOOLEAN_TRIM_MARGIN_MM,
    solid_fraction: float = SPLITP_SF,
    write_previews: bool = True,
) -> dict:
    """
    Regenerate the two case-study TPMS cleaned STLs (no Aristo / Vocal).

    1. Piecewise bottom-band X +L/4 (``phaseTest_qL4_shift_x``)
    2. Linear graded Jacobian W(z)
    """
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    shift = float(SPLITP_PIECEWISE_BOTTOM_BAND_PHASE_SHIFT_X_MM)
    results: list[dict] = []

    print(
        f"Regenerating case-study Split-P STLs @ resolution={resolution_mm:.4f} mm, "
        f"boolean_trim margin={boolean_trim_margin_mm:.3f} mm ..."
    )

    print("\n=== Piecewise phaseTest_qL4_shift_x ===")
    mesh_pw, meta_pw = build_piecewise_splitp_cube_mesh(
        band_phase_origin_x_mm=(shift, 0.0),
        band_phase_origin_y_mm=(0.0, 0.0),
        resolution_mm=resolution_mm,
        solid_fraction=solid_fraction,
        boolean_trim=True,
        boolean_trim_margin_mm=boolean_trim_margin_mm,
    )
    stem_pw = splitp_piecewise_phase_shift_x_stem(solid_fraction=solid_fraction)
    stl_pw = out / f"{stem_pw}.stl"
    cleaned_pw = out / f"{stem_pw}_cleaned.stl"
    mesh_pw.export(stl_pw)
    mesh_pw.export(cleaned_pw)
    png_pw = out / f"{stem_pw}_preview.png"
    if write_previews:
        _preview_mesh_png(
            mesh_pw,
            png_pw,
            "Piecewise Split-P (X +L/4)\n"
            f"res={resolution_mm:.3f} mm + Boolean cube trim",
        )
    report_pw = out / f"{stem_pw}_generation_report.json"
    report_pw.write_text(json.dumps(meta_pw, indent=2), encoding="utf-8")
    print(f"Wrote {stl_pw}")
    print(f"Wrote {cleaned_pw}")
    print(f"Wrote {report_pw}")
    results.append(
        {
            "variant": "piecewise_phase_shift_x",
            "stem": stem_pw,
            "stl_path": str(stl_pw),
            "cleaned_stl_path": str(cleaned_pw),
            "mesh_meta": meta_pw,
        }
    )

    print("\n=== Linear graded (Jacobian W) ===")
    mesh_gr, meta_gr, cal_knots = build_linear_graded_splitp_cube_mesh(
        resolution_mm=resolution_mm,
        solid_fraction=solid_fraction,
        grading_mode="jacobian_integrated",
        phase_origin_x_mm=SPLITP_PHASE_ORIGIN_MM[0],
        phase_origin_y_mm=SPLITP_PHASE_ORIGIN_MM[1],
        boolean_trim=True,
        boolean_trim_margin_mm=boolean_trim_margin_mm,
    )
    stem_gr = splitp_linear_graded_stem(solid_fraction=solid_fraction)
    stl_gr = out / f"{stem_gr}.stl"
    cleaned_gr = out / f"{stem_gr}_cleaned.stl"
    mesh_gr.export(stl_gr)
    mesh_gr.export(cleaned_gr)
    png_gr = out / f"{stem_gr}_preview.png"
    if write_previews:
        _preview_mesh_png(
            mesh_gr,
            png_gr,
            "Linear graded Split-P\n"
            f"res={resolution_mm:.3f} mm + Boolean cube trim",
        )
    report_gr = out / f"{stem_gr}_generation_report.json"
    meta_gr_out = dict(meta_gr)
    meta_gr_out["calibration_knots"] = [
        {
            "period_mm": c.calibrated_L_mm,
            "tau": c.calibrated_tau,
            "measured_sf": c.measured_solid_fraction,
            "measured_pore_mm": c.measured_pore_mm,
        }
        for c in cal_knots
    ]
    report_gr.write_text(json.dumps(meta_gr_out, indent=2), encoding="utf-8")
    print(f"Wrote {stl_gr}")
    print(f"Wrote {cleaned_gr}")
    print(f"Wrote {report_gr}")
    results.append(
        {
            "variant": "linear_graded",
            "stem": stem_gr,
            "stl_path": str(stl_gr),
            "cleaned_stl_path": str(cleaned_gr),
            "mesh_meta": meta_gr_out,
        }
    )

    summary = {
        "domain_mm": [CUBE_SIZE_MM, CUBE_SIZE_MM, CUBE_SIZE_MM],
        "solid_fraction_target": float(solid_fraction),
        "resolution_mm": float(resolution_mm),
        "boolean_trim": True,
        "boolean_trim_margin_mm": float(boolean_trim_margin_mm),
        "variants": results,
    }
    summary_path = out / "SplitP_Cube1mm_boolean_trim_regen_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")
    return summary


def generate_splitp_cube_variants(
    *,
    out_dir: Path | None = None,
    resolution_mm: float = IMPLICIT_RESOLUTION_MM,
    solid_fraction: float = SPLITP_SF,
    only: str = "all",
    grading_mode: str = "jacobian_integrated",
    phase_origin_x_mm: float = SPLITP_PHASE_ORIGIN_MM[0],
    phase_origin_y_mm: float = SPLITP_PHASE_ORIGIN_MM[1],
    boolean_trim: bool = False,
    boolean_trim_margin_mm: float = BOOLEAN_TRIM_MARGIN_MM,
) -> dict:
    """Build piecewise and/or linear graded Split-P cube STLs. Returns generation report dict."""
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    size_mm = CUBE_SIZE_MM
    band_h = BAND_HEIGHT_MM
    sf = float(solid_fraction)
    results: list[dict] = []
    preview_panels: list[tuple[Path, str]] = []

    if only in ("all", "piecewise"):
        print(f"Calibrating / meshing piecewise Split-P @ SF={sf:.0%} ...")
        mesh_pw, meta_pw = build_piecewise_splitp_cube_mesh(
            resolution_mm=resolution_mm,
            solid_fraction=sf,
            boolean_trim=boolean_trim,
            boolean_trim_margin_mm=boolean_trim_margin_mm,
        )
        stem_pw = splitp_piecewise_stem(solid_fraction=sf)
        stl_pw = out / f"{stem_pw}.stl"
        cleaned_pw = out / f"{stem_pw}_cleaned.stl"
        png_pw = out / f"{stem_pw}_preview.png"
        mesh_pw.export(stl_pw)
        mesh_pw.export(cleaned_pw)
        _preview_mesh_png(
            mesh_pw,
            png_pw,
            "Piecewise Split-P 1 mm³\nbottom L=500 µm | top L=1 mm | SF=33 %",
        )
        preview_panels.append((png_pw, "Piecewise (hard band @ Z = 0.5 mm)"))
        results.append(
            {
                "variant": "piecewise",
                "stem": stem_pw,
                "stl_path": str(stl_pw),
                "cleaned_stl_path": str(cleaned_pw),
                "preview_png": str(png_pw),
                "bands": [
                    {
                        "z_mm": [0.0, band_h],
                        "L_cal_mm": meta_pw["calibrated_L_mm"][0],
                        "tau": meta_pw["calibrated_tau"][0],
                    },
                    {
                        "z_mm": [band_h, size_mm],
                        "L_cal_mm": meta_pw["calibrated_L_mm"][1],
                        "tau": meta_pw["calibrated_tau"][1],
                    },
                ],
                "mesh_meta": meta_pw,
            }
        )
        print(f"Wrote {stl_pw}")
        print(f"Wrote {cleaned_pw}")
        print(f"Wrote {png_pw}")

    if only in ("all", "graded"):
        print(f"Calibrating / meshing linear graded Split-P @ SF={sf:.0%} ...")
        mesh_gr, meta_gr, cal_knots = build_linear_graded_splitp_cube_mesh(
            resolution_mm=resolution_mm,
            solid_fraction=sf,
            grading_mode=grading_mode,
            phase_origin_x_mm=phase_origin_x_mm,
            phase_origin_y_mm=phase_origin_y_mm,
            boolean_trim=boolean_trim,
            boolean_trim_margin_mm=boolean_trim_margin_mm,
        )
        stem_gr = splitp_linear_graded_stem(
            solid_fraction=sf,
            phase_origin_x_mm=phase_origin_x_mm,
            phase_origin_y_mm=phase_origin_y_mm,
            jacobian=grading_mode == "jacobian_integrated",
        )
        stl_gr = out / f"{stem_gr}.stl"
        cleaned_gr = out / f"{stem_gr}_cleaned.stl"
        png_gr = out / f"{stem_gr}_preview.png"
        mesh_gr.export(stl_gr)
        mesh_gr.export(cleaned_gr)
        _preview_mesh_png(
            mesh_gr,
            png_gr,
            "Linear graded Split-P 1 mm³\nL: 500 µm -> 1 mm | SF=33 %\n"
            f"phase origin ({phase_origin_x_mm:.3f}, {phase_origin_y_mm:.3f}) mm\n"
            f"{grading_mode}",
        )
        preview_panels.append((png_gr, "Linear graded (Jacobian W(z))"))
        results.append(
            {
                "variant": "linear_graded",
                "stem": stem_gr,
                "stl_path": str(stl_gr),
                "cleaned_stl_path": str(cleaned_gr),
                "preview_png": str(png_gr),
                "gradient": {
                    "grading_mode": grading_mode,
                    "phase_origin_mm": [float(phase_origin_x_mm), float(phase_origin_y_mm)],
                    "z_knots_mm": meta_gr["z_knots_mm"],
                    "period_knots_mm": meta_gr["period_knots_mm"],
                    "calibrated_L_mm": meta_gr["calibrated_L_mm"],
                    "calibrated_tau": meta_gr["calibrated_tau"],
                    "calibration_knots": [
                        {
                            "period_mm": c.calibrated_L_mm,
                            "tau": c.calibrated_tau,
                            "measured_sf": c.measured_solid_fraction,
                            "measured_pore_mm": c.measured_pore_mm,
                        }
                        for c in cal_knots
                    ],
                },
                "mesh_meta": meta_gr,
            }
        )
        print(f"Wrote {stl_gr}")
        print(f"Wrote {cleaned_gr}")
        print(f"Wrote {png_gr}")

    if len(preview_panels) > 1:
        montage_path = out / "SplitP_Cube1mm_variants_preview.png"
        _montage_png(preview_panels, montage_path)
        print(f"Wrote {montage_path}")

    report = {
        "domain_mm": [size_mm, size_mm, size_mm],
        "solid_fraction_target": sf,
        "resolution_mm": resolution_mm,
        "boolean_trim": bool(boolean_trim),
        "boolean_trim_margin_mm": float(boolean_trim_margin_mm) if boolean_trim else 0.0,
        "variants": results,
    }
    report_path = out / "SplitP_Cube1mm_variants_generation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-mm", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--solid-fraction", type=float, default=SPLITP_SF)
    parser.add_argument(
        "--only",
        choices=("all", "piecewise", "graded", "case-study"),
        default="all",
        help="'case-study' regenerates phase-shift piecewise + linear graded with Boolean trim.",
    )
    parser.add_argument(
        "--grading-mode",
        choices=("jacobian_integrated", "chirp"),
        default="jacobian_integrated",
    )
    parser.add_argument("--phase-origin-x-mm", type=float, default=SPLITP_PHASE_ORIGIN_MM[0])
    parser.add_argument("--phase-origin-y-mm", type=float, default=SPLITP_PHASE_ORIGIN_MM[1])
    parser.add_argument(
        "--boolean-trim",
        action="store_true",
        help="Oversize MC domain then Boolean ∩ design cube for flat faces.",
    )
    parser.add_argument(
        "--boolean-trim-margin-mm",
        type=float,
        default=BOOLEAN_TRIM_MARGIN_MM,
    )
    args = parser.parse_args(argv)
    out_dir = args.out_dir or default_output_dir()

    if args.only == "case-study":
        res = (
            float(args.resolution_mm)
            if args.resolution_mm is not None
            else IMPLICIT_RESOLUTION_BOOLEAN_TRIM_MM
        )
        regenerate_case_study_splitp_stls(
            out_dir=out_dir,
            resolution_mm=res,
            boolean_trim_margin_mm=float(args.boolean_trim_margin_mm),
            solid_fraction=float(args.solid_fraction),
        )
        return 0

    res = float(args.resolution_mm) if args.resolution_mm is not None else IMPLICIT_RESOLUTION_MM
    generate_splitp_cube_variants(
        out_dir=out_dir,
        resolution_mm=res,
        solid_fraction=float(args.solid_fraction),
        only=args.only,
        grading_mode=args.grading_mode,
        phase_origin_x_mm=float(args.phase_origin_x_mm),
        phase_origin_y_mm=float(args.phase_origin_y_mm),
        boolean_trim=bool(args.boolean_trim),
        boolean_trim_margin_mm=float(args.boolean_trim_margin_mm),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
