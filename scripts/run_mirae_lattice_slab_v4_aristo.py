"""
Linear elastic compression of Mirae_LatticeSlab_V4_fixed.stl via Aristo.

TPMS classifySurfaces + adaptive CharacteristicLength curvature sizing.
Flat-top load BC (Z + normal gate); parametric sweep over compressive loads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_log import aristo_log, aristo_stage
from graphite.aristo.aristo_solver import run_aristo
from graphite.aristo.aristo_viz import (
    export_aristo_paraview,
    plot_aristo_result,
    plot_aristo_von_mises_isosurfaces,
)
from graphite.aristo.fea_runner import detect_flat_top_bcs
from graphite.aristo.stiffness_assembly import compute_element_stresses

STL_NAME = "Mirae_LatticeSlab_V4_fixed.stl"
STEM = "Mirae_LatticeSlab_V4_fixed"
LOAD_CASES_N = [1.0, 5.0, 10.0]
FEA_MESH_MM = 0.15
Z_BAND_FRACTION = 0.01
TOP_Z_TOLERANCE_MM = 1e-4
TOP_NORMAL_Z_MIN = 0.99
FEA_QUALITY_MODE = "thorough"
CHAR_LENGTH_MIN = 0.08
CHAR_LENGTH_MAX = 0.8
MIN_ELEMENTS_PER_TWO_PI = 15


def _load_tag(force_n: float) -> str:
    if force_n == int(force_n):
        return f"{int(force_n)}N"
    return f"{force_n:g}N".replace(".", "p")


def _loaded_face_area(
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
    areas = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
    return float(areas.sum())


def _preview_flat_top_bcs(
    mesh: trimesh.Trimesh,
    *,
    bc_load_mode: str,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    fixed_mask, load_mask, meta = detect_flat_top_bcs(
        mesh.vertices,
        mesh.faces,
        bc_load_mode=bc_load_mode,
    )
    z_max = float(mesh.vertices[:, 2].max())
    return fixed_mask, load_mask, z_max, meta


def _print_result_summary(tag: str, result, mq: dict) -> None:
    print(f"  [{tag}] max displacement:        {result.max_displacement:.6f} mm")
    print(f"  [{tag}] stress field:            {result.stress_field_mode}")
    print(f"  [{tag}] max von Mises (nodal):   {result.max_von_mises_raw:.4f} MPa")
    print(f"  [{tag}] max von Mises (element): {result.max_von_mises_element_raw:.4f} MPa")
    nodal_stress = mq.get("stress_nodal", {})
    if nodal_stress:
        print(
            f"  [{tag}] nodal P50/P99/max:       "
            f"{nodal_stress.get('p50', 0):.3f} / "
            f"{nodal_stress.get('p99', 0):.3f} / "
            f"{nodal_stress.get('max', 0):.3f} MPa"
        )
    print(
        f"  [{tag}] mesh quality: {mq.get('n_poor_elements', 0):,} poor / "
        f"{mq.get('n_elements', 0):,} tets "
        f"({100.0 * mq.get('poor_fraction', 0):.2f}%), "
        f"remesh attempts={mq.get('remesh_attempts', 1)}"
    )
    valid_stress = mq.get("stress_valid_elements", {})
    all_stress = mq.get("stress_all_elements", {})
    if valid_stress:
        print(
            f"  [{tag}] stress valid P50/P99/max: "
            f"{valid_stress.get('p50', 0):.3f} / "
            f"{valid_stress.get('p99', 0):.3f} / "
            f"{valid_stress.get('max', 0):.3f} MPa"
        )
    if all_stress:
        print(
            f"  [{tag}] stress all   P50/P99/max: "
            f"{all_stress.get('p50', 0):.3f} / "
            f"{all_stress.get('p99', 0):.3f} / "
            f"{all_stress.get('max', 0):.3f} MPa"
        )
    print(f"  [{tag}] above 2x P99 (valid):    {mq.get('n_above_2x_p99_valid', 0):,} tets")


def _run_single_load_case(
    mesh: trimesh.Trimesh,
    stl_path: Path,
    out_dir: Path,
    *,
    force_n: float,
    fixed_mask: np.ndarray,
    load_mask: np.ndarray,
    z_max: float,
    z_span: float,
    bc_load_mode: str,
    solver: str | None,
) -> None:
    tag = _load_tag(force_n)
    load_area = _loaded_face_area(mesh.vertices, mesh.faces, load_mask)
    pressure = force_n / load_area
    cz = mesh.triangles_center[:, 2]
    z_tol = Z_BAND_FRACTION * z_span

    print(f"\n=== Load case {force_n:g} N ({tag}) ===")
    print(
        f"  Assembling/solving: {bc_load_mode}, bottom {Z_BAND_FRACTION:.0%} Z-band fixed "
        f"(pressure={pressure:.6f} N/mm², STL load area={load_area:.4f} mm²)..."
    )
    aristo_log(
        f"Mirae V4 load case {tag}: {FEA_QUALITY_MODE} FEA, "
        f"force={force_n:g} N, pressure={pressure:.6f} N/mm², "
        f"bc={bc_load_mode}, solver={solver or 'auto'}"
    )

    config = AristoConfig(
        youngs_modulus=2800.0,
        poisson_ratio=0.38,
        fea_mesh_resolution=FEA_MESH_MM,
        load_direction=(0.0, 0.0, -1.0),
        load_magnitude=pressure,
        load_mode="full",
        bc_z_band_fraction=Z_BAND_FRACTION,
        bc_load_mode=bc_load_mode,
        bc_top_z_tolerance_mm=TOP_Z_TOLERANCE_MM,
        bc_top_normal_z_min=TOP_NORMAL_Z_MIN,
        fea_quality_mode=FEA_QUALITY_MODE,
        fea_gmsh_mesh_mode="tpms",
        fea_gmsh_adaptive_curvature=True,
        fea_gmsh_char_length_min=CHAR_LENGTH_MIN,
        fea_gmsh_char_length_max=CHAR_LENGTH_MAX,
        fea_gmsh_min_elements_per_two_pi=MIN_ELEMENTS_PER_TWO_PI,
        fea_gmsh_classify_only=True,
        fea_clean_stl_surface=False,
        fea_linear_solver=solver,
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
    )
    fea_load_area = _loaded_face_area(
        result.fea_nodes, result.fea_surface_faces, load_fea_mask
    )

    report = {
        "stl": str(stl_path),
        "load_case_tag": tag,
        "gmsh_mode": config.fea_gmsh_mesh_mode,
        "adaptive_curvature": config.fea_gmsh_adaptive_curvature,
        "char_length_min_mm": config.fea_gmsh_char_length_min,
        "char_length_max_mm": config.fea_gmsh_char_length_max,
        "min_elements_per_two_pi": config.fea_gmsh_min_elements_per_two_pi,
        "bc_mode": bc_load_mode,
        "bc_load_mode": config.bc_load_mode,
        "fea_linear_solver": solver,
        "bc_top_z_tolerance_mm": TOP_Z_TOLERANCE_MM,
        "bc_top_normal_z_min": TOP_NORMAL_Z_MIN,
        "fixed_band_fraction": Z_BAND_FRACTION,
        "z_span_mm": z_span,
        "z_band_thickness_mm": z_tol,
        "z_max_mm": z_max,
        "target_force_N": force_n,
        "applied_force_N": pressure * fea_load_area,
        "stl_load_area_mm2": load_area,
        "fea_load_area_mm2": fea_load_area,
        "n_load_faces_stl": int(load_mask.sum()),
        "n_fixed_faces_stl": int(fixed_mask.sum()),
        "n_load_faces_fea": int(load_fea_mask.sum()),
        "load_z_min_mm": float(cz[load_mask].min()),
        "load_z_max_mm": float(cz[load_mask].max()),
        "pressure_N_per_mm2": pressure,
        "fea_mesh_resolution_mm": FEA_MESH_MM,
        "volume_mm3": mesh.volume,
        "n_fea_nodes": int(len(result.fea_nodes)),
        "n_fea_tets": int(len(result.fea_elements)),
        "fea_quality_mode": config.fea_quality_mode,
        "max_displacement_mm": result.max_displacement,
        "stress_field_mode": result.stress_field_mode,
        "max_von_mises_nodal_MPa": result.max_von_mises_raw,
        "max_von_mises_element_raw_MPa": result.max_von_mises_element_raw,
        "hotspot_fraction": result.hotspot_fraction,
        "mesh_quality_report": result.mesh_quality_report,
    }

    report_path = out_dir / f"{STEM}_{tag}_aristo_report.json"
    vtu_path = out_dir / f"{STEM}_{tag}_aristo_fea.vtu"
    stress_png = out_dir / f"{STEM}_{tag}_aristo_von_mises.png"
    iso_png = out_dir / f"{STEM}_{tag}_aristo_von_mises_isosurfaces.png"
    disp_png = out_dir / f"{STEM}_{tag}_aristo_displacement.png"
    report["vtu_path"] = str(vtu_path)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with aristo_stage(f"export_paraview_vtu ({tag})"):
        export_aristo_paraview(
            result,
            str(vtu_path),
            element_vm_mpa=vm_elem,
            load_face_mask=load_fea_mask,
        )

    load_dir_tuple = (0.0, 0.0, -1.0)
    with aristo_stage(f"plot_von_mises_png ({tag})"):
        plot_aristo_result(
            result,
            output_path=str(stress_png),
            scalar_key="von_mises",
            off_screen=True,
            load_direction=load_dir_tuple,
        )
    with aristo_stage(f"plot_isosurfaces_png ({tag})"):
        plot_aristo_von_mises_isosurfaces(result, str(iso_png), off_screen=True)
    with aristo_stage(f"plot_displacement_png ({tag})"):
        plot_aristo_result(
            result,
            output_path=str(disp_png),
            scalar_key="displacement_magnitude",
            off_screen=True,
            load_direction=load_dir_tuple,
        )

    mq = result.mesh_quality_report
    print(
        f"  [{tag}] FEA load faces/area:   {load_fea_mask.sum():,} / "
        f"{fea_load_area:.4f} mm²"
    )
    _print_result_summary(tag, result, mq)
    print(f"Wrote {report_path}")
    print(f"Wrote {vtu_path}")
    print(f"Wrote {stress_png}")
    print(f"Wrote {iso_png}")
    print(f"Wrote {disp_png}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-load-mode",
        choices=("flat_top", "flat_top_vertex_plane"),
        default="flat_top",
        help="Mirae regression default: flat_top. Use flat_top_vertex_plane for implicit-style caps.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "scipy", "pardiso"),
        default="auto",
        help="Linear solver (default auto). Use pardiso when pypardiso is installed.",
    )
    args = parser.parse_args()
    solver = None if args.solver == "auto" else args.solver

    stl_path = _REPO_ROOT / "outputs" / "models" / "user_spec_rect_prism_3x1p5x5mm" / STL_NAME
    out_dir = stl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {stl_path}")
    mesh = trimesh.load_mesh(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    z_span = float(mesh.bounds[1][2] - mesh.bounds[0][2])
    fixed_mask, load_mask, z_max, bc_preview = _preview_flat_top_bcs(
        mesh, bc_load_mode=args.bc_load_mode
    )
    cz = mesh.triangles_center[:, 2]

    print(
        f"  faces={len(mesh.faces):,} watertight={mesh.is_watertight} "
        f"volume={mesh.volume:.4f} mm³ z_span={z_span:.4f} mm z_max={z_max:.6f} mm"
    )
    print(
        f"  BC preview (STL): mode={args.bc_load_mode}, fixed={fixed_mask.sum():,}, "
        f"load={load_mask.sum():,}"
    )
    if bc_preview:
        print(
            f"  cap z={bc_preview.get('z_cap_mm')} floor z={bc_preview.get('z_floor_mm')}"
        )
    elif args.bc_load_mode == "flat_top":
        print(
            f"  (flat_top: z>={z_max - TOP_Z_TOLERANCE_MM:.6f}, normal_z>{TOP_NORMAL_Z_MIN})"
        )
    if load_mask.any():
        print(
            f"  load face z range: [{cz[load_mask].min():.6f}, {cz[load_mask].max():.6f}] mm"
        )

    aristo_log(
        f"Mirae V4 sweep: loads={LOAD_CASES_N} N, {FEA_QUALITY_MODE} FEA, "
        f"bc={args.bc_load_mode}, solver={solver or 'auto'}, "
        f"adaptive cl=[{CHAR_LENGTH_MIN}, {CHAR_LENGTH_MAX}]"
    )

    for force_n in LOAD_CASES_N:
        _run_single_load_case(
            mesh,
            stl_path,
            out_dir,
            force_n=force_n,
            fixed_mask=fixed_mask,
            load_mask=load_mask,
            z_max=z_max,
            z_span=z_span,
            bc_load_mode=args.bc_load_mode,
            solver=solver,
        )

    print(f"\nCompleted {len(LOAD_CASES_N)} load cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
