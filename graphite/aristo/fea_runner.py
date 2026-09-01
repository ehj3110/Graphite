"""
Shared helpers for Aristo compression FEA CLI scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_log import aristo_stage
from graphite.aristo.aristo_solver import run_aristo
from graphite.aristo.aristo_viz import (
    export_aristo_paraview,
    plot_aristo_result,
    plot_aristo_von_mises_isosurfaces,
)
from graphite.aristo.boundary_detection import (
    detect_boundary_masks_flat_top_load,
    detect_boundary_masks_flat_top_vertex_plane,
)
from graphite.aristo.stiffness_assembly import compute_element_stresses

NODAL_VM_KEY = "von_mises_nodal_MPa"
DEFAULT_Z_BAND_FRACTION = 0.01
DEFAULT_TOP_Z_TOLERANCE_MM = 1e-4
DEFAULT_TOP_NORMAL_Z_MIN = 0.99
DEFAULT_TOP_MIN_VERTICES_AT_PLANE = 100


def loaded_face_area(
    fea_nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_face_mask: np.ndarray,
) -> float:
    loaded = surface_faces[load_face_mask]
    if loaded.size == 0:
        return 0.0
    verts = fea_nodes[loaded]
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    return float(np.linalg.norm(np.cross(e1, e2), axis=1).sum() * 0.5)


def detect_flat_top_bcs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    bc_load_mode: str,
    z_band_fraction: float = DEFAULT_Z_BAND_FRACTION,
    top_min_vertices: int = DEFAULT_TOP_MIN_VERTICES_AT_PLANE,
) -> tuple[np.ndarray, np.ndarray, dict]:
    load_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if bc_load_mode == "flat_top_vertex_plane":
        fixed_mask, load_mask, meta = detect_boundary_masks_flat_top_vertex_plane(
            vertices,
            faces,
            load_dir,
            z_band_fraction,
            min_vertices_at_plane=top_min_vertices,
            top_normal_z_min=DEFAULT_TOP_NORMAL_Z_MIN,
        )
        return fixed_mask, load_mask, meta
    fixed_mask, load_mask = detect_boundary_masks_flat_top_load(
        vertices,
        faces,
        load_dir,
        z_band_fraction,
        z_tolerance_mm=DEFAULT_TOP_Z_TOLERANCE_MM,
        top_normal_z_min=DEFAULT_TOP_NORMAL_Z_MIN,
    )
    return fixed_mask, load_mask, {}


def build_compression_config(
    *,
    pressure_n_per_mm2: float,
    h: float,
    mesh_mode: str | None = None,
    quality_mode: str,
    bc_load_mode: str,
    clean_stl: bool,
    solver: str,
    fea_input_mode: str = "implicit",
    top_min_vertices: int = DEFAULT_TOP_MIN_VERTICES_AT_PLANE,
    youngs_modulus: float = 2800.0,
    poisson_ratio: float = 0.38,
) -> AristoConfig:
    h = float(h)
    if mesh_mode is None:
        mesh_mode = "single_surface" if fea_input_mode == "implicit" else "tpms"
    adaptive = mesh_mode == "tpms"
    return AristoConfig(
        youngs_modulus=float(youngs_modulus),
        poisson_ratio=float(poisson_ratio),
        fea_mesh_resolution=h,
        load_direction=(0.0, 0.0, -1.0),
        load_magnitude=float(pressure_n_per_mm2),
        load_mode="full",
        bc_z_band_fraction=DEFAULT_Z_BAND_FRACTION,
        bc_load_mode=bc_load_mode,
        bc_top_z_tolerance_mm=DEFAULT_TOP_Z_TOLERANCE_MM,
        bc_top_normal_z_min=DEFAULT_TOP_NORMAL_Z_MIN,
        bc_top_min_vertices_at_plane=int(top_min_vertices),
        fea_quality_mode=quality_mode,
        fea_input_mode=fea_input_mode,
        fea_gmsh_mesh_mode=mesh_mode,
        fea_gmsh_adaptive_curvature=adaptive,
        fea_gmsh_char_length_min=max(0.03, h * 0.5),
        fea_gmsh_char_length_max=max(0.15, h * 2.5),
        fea_gmsh_min_elements_per_two_pi=15,
        fea_gmsh_classify_only=mesh_mode == "tpms",
        fea_clean_stl_surface=bool(clean_stl),
        fea_linear_solver=solver,
    )


def run_compression_fea_case(
    mesh: trimesh.Trimesh,
    *,
    stl_path: Path | None,
    out_dir: Path,
    stem: str,
    force_n: float,
    h: float,
    mesh_mode: str | None = None,
    quality_mode: str = "quick",
    bc_load_mode: str = "flat_top_vertex_plane",
    clean_stl: bool = False,
    solver: str = "pardiso",
    fea_input_mode: str = "implicit",
    implicit_report: dict[str, Any] | None = None,
    top_min_vertices: int = DEFAULT_TOP_MIN_VERTICES_AT_PLANE,
    youngs_modulus: float = 2800.0,
    write_viz: bool = True,
) -> dict[str, Any]:
    """Run one compression load case; write report, VTU, and optional PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{int(force_n)}N" if force_n == int(force_n) else f"{force_n:g}N"

    fixed_mask, load_mask, bc_preview = detect_flat_top_bcs(
        mesh.vertices,
        mesh.faces,
        bc_load_mode=bc_load_mode,
        top_min_vertices=top_min_vertices,
    )
    load_area = loaded_face_area(mesh.vertices, mesh.faces, load_mask)
    if load_area <= 0.0:
        raise ValueError("No load faces detected on STL surface.")
    pressure = float(force_n) / load_area

    if mesh_mode is None:
        mesh_mode = "single_surface" if fea_input_mode == "implicit" else "tpms"

    config = build_compression_config(
        pressure_n_per_mm2=pressure,
        h=h,
        mesh_mode=mesh_mode,
        quality_mode=quality_mode,
        bc_load_mode=bc_load_mode,
        clean_stl=clean_stl,
        solver=solver,
        fea_input_mode=fea_input_mode,
        top_min_vertices=top_min_vertices,
        youngs_modulus=float(youngs_modulus),
    )

    with aristo_stage(f"run_aristo ({tag})"):
        result = run_aristo(mesh, config)

    vm_elem = compute_element_stresses(
        result.fea_nodes,
        result.fea_elements,
        result.displacement.ravel(),
        config.youngs_modulus,
        config.poisson_ratio,
    )
    _, load_fea_mask, _ = detect_flat_top_bcs(
        result.fea_nodes,
        result.fea_surface_faces,
        bc_load_mode=bc_load_mode,
        top_min_vertices=top_min_vertices,
    )
    fea_load_area = loaded_face_area(
        result.fea_nodes, result.fea_surface_faces, load_fea_mask
    )

    mq = result.mesh_quality_report
    report: dict[str, Any] = {
        "fea_input_mode": fea_input_mode,
        "stl": str(stl_path) if stl_path is not None else None,
        "implicit_generation": implicit_report,
        "load_case_tag": tag,
        "target_force_N": float(force_n),
        "applied_force_N": pressure * fea_load_area,
        "pressure_N_per_mm2": pressure,
        "stl_load_area_mm2": load_area,
        "fea_load_area_mm2": fea_load_area,
        "n_load_faces_stl": int(load_mask.sum()),
        "n_load_faces_fea": int(load_fea_mask.sum()),
        "fea_mesh_resolution_mm": float(h),
        "fea_gmsh_mesh_mode": mesh_mode,
        "fea_clean_stl_surface": bool(clean_stl),
        "fea_linear_solver": solver,
        "youngs_modulus_MPa": float(youngs_modulus),
        "bc_load_mode": bc_load_mode,
        "bc_load_detection": mq.get("bc_load_detection") or bc_preview,
        "linear_solver": mq.get("linear_solver"),
        "linear_solve_sec": mq.get("linear_solve_sec"),
        "max_displacement_mm": result.max_displacement,
        "max_von_mises_nodal_MPa": result.max_von_mises_raw,
        "max_von_mises_element_raw_MPa": result.max_von_mises_element_raw,
        "mesh_quality_report": mq,
    }

    report_path = out_dir / f"{stem}_{tag}_aristo_report.json"
    vtu_path = out_dir / f"{stem}_{tag}_aristo_fea.vtu"
    report["vtu_path"] = str(vtu_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    export_aristo_paraview(
        result,
        str(vtu_path),
        element_vm_mpa=vm_elem,
        load_face_mask=load_fea_mask,
    )

    if write_viz:
        plot_aristo_result(
            result,
            output_path=str(out_dir / f"{stem}_{tag}_aristo_von_mises.png"),
            scalar_key="von_mises",
            off_screen=True,
            load_direction=(0.0, 0.0, -1.0),
        )
        plot_aristo_von_mises_isosurfaces(
            result,
            str(out_dir / f"{stem}_{tag}_aristo_von_mises_isosurfaces.png"),
            off_screen=True,
        )
        plot_aristo_result(
            result,
            output_path=str(out_dir / f"{stem}_{tag}_aristo_displacement.png"),
            scalar_key="displacement_magnitude",
            off_screen=True,
            load_direction=(0.0, 0.0, -1.0),
        )

    report["report_path"] = str(report_path)
    return report
