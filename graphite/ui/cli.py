"""
Headless CLI recipe runner for Graphite.

Allows generating lattices directly from command line arguments or JSON/YAML
recipes without launching the graphical user interface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import trimesh

from graphite.geometry.primitives import generate_primitive
from graphite.explicit import generate_conformal_lattice as generate_explicit_conformal_lattice
from graphite.implicit.conformal import generate_conformal_lattice as generate_implicit_conformal_lattice
from graphite.implicit.density_control import tau_from_wall_thickness_mm
from graphite.implicit.field_driven import generate_field_driven_lattice
from graphite.implicit.graded import generate_graded_lattice
from graphite.implicit.uniform_woodpile import generate_uniform_woodpile
from graphite.implicit.surface_textures import SurfaceTextureConfig, apply_surface_texture
from graphite.io.mesh_export import export_mesh, formats_from_request

SUPPORTED_TPMS_EQUATIONS = [
    "Gyroid",
    "Diamond",
    "Schwarz-P",
    "Schwarz-Diamond",
    "Neovius",
    "Lidinoid",
    "Split-P",
    "Woodpile",
    "Cross-Hatch",
]


def run_headless_recipe(
    modality: str = "implicit",
    explicit_type: str = "SC",
    sc_rule: str = "octahedral",
    strut_radius: float = 0.4,
    dual_width: float | None = None,
    dual_thickness: float | None = None,
    conformal_mode: str = "conformal",
    primitive: str = "Cube",
    size: float = 20.0,
    cad_stl: str | Path | None = None,
    lattice_type: str = "Gyroid",
    cell_size: float = 5.0,
    solid_fraction: float = 0.30,
    wall_thickness_mm: float | None = None,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    formats: str | tuple[str, ...] = "stl",
    enable_grading: bool = False,
    grading_axis: str = "Z",
    grading_mode: str = "cell_size",
    min_solid_fraction: float = 0.15,
    max_solid_fraction: float = 0.45,
    doubling_interval_mm: float = 5.0,
    conformal_deformation: bool = False,
    conformal_iterations: int = 150,
    enable_surface_texture: bool = False,
    texture_type: str = "microgrooves",
    texture_amplitude_mm: float = 0.025,
    texture_wavelength_mm: float = 0.050,
    texture_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    texture_profile: str = "sine",
    texture_displacement_mode: str = "centered",
    texture_triplanar: bool = False,
    enable_micropillars: bool = False,
    pillar_diameter_mm: float = 0.050,
    pillar_height_mm: float = 0.200,
    pillar_spacing_mm: float = 0.200,
    pillar_location: str = "all",
    pillar_distribution: str = "poisson_disk",
    pillar_filter_printable: bool = False,
    woodpile_invert: bool = False,
    resolution: float | None = None,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a headless lattice generation recipe."""
    start_time = time.time()

    if modality in ("interlinked", "pam", "pams"):
        default_out = "outputs/phase4_interlinked/cli_lattice.stl"
    elif modality in ("auxetics", "auxetic"):
        default_out = "outputs/auxetics/cli_lattice.stl"
    elif modality == "explicit":
        default_out = "outputs/phase3_explicit/cli_lattice.stl"
    else:
        default_out = "outputs/phase2_implicit_full/cli_lattice.stl"

    out_p = Path(output_path) if output_path else Path(default_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(formats, str):
        export_formats_tuple = formats_from_request(formats)
    else:
        export_formats_tuple = tuple(str(f).lower() for f in formats)

    # Resolve CAD boundary: custom STL or primitive
    tmp_cad_path: str | None = None
    if cad_stl is not None:
        cad_path = Path(cad_stl)
        if not cad_path.is_file():
            raise FileNotFoundError(f"CAD STL file not found: {cad_path}")
        cad_mesh = trimesh.load(str(cad_path), force="mesh")
        if isinstance(cad_mesh, trimesh.Scene):
            cad_mesh = cad_mesh.dump(concatenate=True)
        input_stl_for_engine = str(cad_path.resolve())
    else:
        cad_mesh = generate_primitive(primitive, size)
        if modality == "implicit":
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                cad_mesh.export(tmp.name)
                tmp_cad_path = tmp.name
            input_stl_for_engine = tmp_cad_path
        else:
            input_stl_for_engine = None

    try:
        if modality in ("interlinked", "pam", "pams"):
            from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

            interlinked_cell = kwargs.get("interlinked_cell", "c6tt")
            seeding_type = kwargs.get("interlinked_seeding", "cartesian")
            wire_r = float(kwargs.get("wire_radius", strut_radius))
            min_clr = float(kwargs.get("min_clearance", 0.35))
            auto_resolve_p = bool(kwargs.get("auto_resolve_pitch", False))
            cull_margin = float(kwargs.get("cull_margin", 0.50))
            add_perimeter_frame = bool(kwargs.get("add_perimeter_frame", False))
            frame_wall_thickness = float(kwargs.get("frame_wall_thickness", 2.0))
            frame_margin = float(kwargs.get("frame_margin", 0.50))
            frame_shape = str(kwargs.get("frame_shape", "box"))
            cylinder_r = kwargs.get("cylinder_radius", None)
            cylinder_h = kwargs.get("cylinder_height", None)
            support_rec = kwargs.get("support_recipe", None)

            p = float(cell_size)
            seeding_mode = str(seeding_type).strip().lower()

            if seeding_mode in ("cylindrical", "cylinder", "cylindrical_wrap"):
                cfg = InterlinkedConfig(
                    cell=interlinked_cell,
                    seeding_type="cylindrical",
                    cylinder_radius=float(cylinder_r) if cylinder_r is not None else 15.0,
                    cylinder_height=float(cylinder_h) if cylinder_h is not None else 25.0,
                    pitch=p,
                    wire_radius=wire_r,
                    min_clearance=min_clr,
                    auto_resolve_pitch=auto_resolve_p,
                    add_perimeter_frame=add_perimeter_frame,
                    frame_wall_thickness=frame_wall_thickness,
                    frame_margin=frame_margin,
                    frame_shape="cylinder" if frame_shape == "box" else frame_shape,
                    support_recipe=support_rec,
                )
                res = generate_interlinked_lattice(cfg, check_clearance=True)
            else:
                min_b, max_b = cad_mesh.bounds[0], cad_mesh.bounds[1]
                grid_size = np.maximum(np.ceil((max_b - min_b) / p).astype(int) + 1, 1)
                origin = min_b + (max_b - min_b - (grid_size - 1) * p) / 2.0

                cfg = InterlinkedConfig(
                    cell=interlinked_cell,
                    seeding_type="cartesian",
                    grid_size=tuple(grid_size),
                    pitch=p,
                    wire_radius=wire_r,
                    min_clearance=min_clr,
                    auto_resolve_pitch=auto_resolve_p,
                    cull_margin=cull_margin,
                    origin=tuple(origin),
                    add_perimeter_frame=add_perimeter_frame,
                    frame_wall_thickness=frame_wall_thickness,
                    frame_margin=frame_margin,
                    frame_shape=frame_shape,
                    support_recipe=support_rec,
                )
                res = generate_interlinked_lattice(cfg, boundary_mesh=cad_mesh, check_clearance=True)

            out_files = []
            stem = out_p.stem
            parent = out_p.parent

            if "stl" in export_formats_tuple:
                stl_dest = parent / f"{stem}.stl"
                res.export_stl(stl_dest, include_frame=add_perimeter_frame)
                out_files.append(str(stl_dest))
            if "3mf" in export_formats_tuple:
                threemf_dest = parent / f"{stem}.3mf"
                res.export_3mf(threemf_dest, include_frame=add_perimeter_frame)
                out_files.append(str(threemf_dest))

            elapsed = time.time() - start_time
            watertight = bool(res.mesh.is_watertight)
            face_count = len(res.mesh.faces)
            vert_count = len(res.mesh.vertices)

            report = {
                "modality": "interlinked",
                "cad_stl": str(cad_stl) if cad_stl else None,
                "primitive": primitive if not cad_stl else None,
                "size": size if not cad_stl else None,
                "cell": interlinked_cell,
                "seeding_type": seeding_mode,
                "pitch_mm": res.metadata.get("pitch", p),
                "wire_radius_mm": wire_r,
                "min_clearance_mm": res.min_clearance,
                "clearance_valid": res.clearance_valid,
                "num_particles": len(res.particles),
                "num_nodes": res.num_nodes,
                "num_struts": res.num_struts,
                "formats": list(export_formats_tuple),
                "output_path": str(out_p),
                "output_files": out_files,
                "watertight": watertight,
                "face_count": face_count,
                "vertex_count": vert_count,
                "generation_time_s": round(elapsed, 2),
            }
            report_file = out_p.parent / "run_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            return report

        if modality in ("auxetics", "auxetic"):
            pattern = kwargs.get("pattern", "tetra_chiral")
            surface = kwargs.get("surface", "plate")
            width = float(kwargs.get("width", kwargs.get("plate_width", 50.0)))
            height = float(kwargs.get("height", kwargs.get("plate_height", 50.0)))
            thickness = float(kwargs.get("thickness", 2.0))
            r_in = float(kwargs.get("r_in", 15.0))
            r_out = float(kwargs.get("r_out", 17.5))
            n_circumferential = int(kwargs.get("n_circumferential", 6))
            r_node = float(kwargs.get("r_node", 2.0))
            strut_w = float(kwargs.get("strut_w", strut_radius * 2.0 if strut_radius else 1.0))
            square_side = float(kwargs.get("square_side", 10.0))
            rotation_angle_deg = float(kwargs.get("rotation_angle_deg", kwargs.get("rotation_angle", 30.0)))
            hinge_radius = float(kwargs.get("hinge_radius", 0.45))
            unit_cell_sz = float(kwargs.get("unit_cell_size", cell_size if cell_size else 10.0))
            r_min = float(kwargs.get("r_min", 0.35))
            r_max = float(kwargs.get("r_max", 0.90))

            pat_lower = str(pattern).strip().lower()
            surf_lower = str(surface).strip().lower()

            if pat_lower == "rotating_squares":
                from graphite.generators.rotating_auxetics import generate_rotating_squares_lattice

                nx = max(int(width / square_side), 2)
                ny = max(int(height / square_side), 2)
                mesh = generate_rotating_squares_lattice(
                    (nx, ny),
                    square_side=square_side,
                    plate_thickness=thickness,
                    hinge_radius=hinge_radius,
                    rotation_angle_deg=rotation_angle_deg,
                )
            elif pat_lower in ("tetra_chiral", "tri_chiral", "anti_tetra_chiral", "anti_tri_chiral", "reentrant"):
                from graphite.explicit.surface_lattice import generate_surface_lattice

                if surf_lower in ("cylinder", "tube"):
                    mesh = generate_surface_lattice(
                        surface="cylinder",
                        pattern=pat_lower,
                        r_in=r_in,
                        r_out=r_out,
                        height=width,
                        n_circumferential=n_circumferential,
                        r_node=r_node,
                        strut_w=strut_w,
                        export_stl=False,
                    )
                else:
                    mesh = generate_surface_lattice(
                        surface="plate",
                        pattern=pat_lower,
                        width=width,
                        height=height,
                        thickness=thickness,
                        n_circumferential=n_circumferential,
                        r_node=r_node,
                        strut_w=strut_w,
                        export_stl=False,
                    )
                if isinstance(mesh, (str, Path)):
                    mesh = trimesh.load(str(mesh), force="mesh")
            elif pat_lower == "pentamode":
                from graphite.generators.pentamode import generate_pentamode_lattice

                mesh = generate_pentamode_lattice(
                    bounds=((0.0, 0.0, 0.0), (width, height, thickness)),
                    unit_cell_size=unit_cell_sz,
                    r_min=r_min,
                    r_max=r_max,
                    output_format="mesh",
                )
            else:
                raise ValueError(f"Unknown auxetic pattern '{pattern}'")

            out_files = []
            stem = out_p.stem
            parent = out_p.parent

            export_mesh(mesh, out_p, formats=export_formats_tuple)
            for fmt in export_formats_tuple:
                dest = parent / f"{stem}.{fmt}"
                if dest.is_file():
                    out_files.append(str(dest))

            elapsed = time.time() - start_time
            watertight = bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else False
            face_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
            vert_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0

            report = {
                "modality": "auxetics",
                "pattern": pattern,
                "surface": surface,
                "width": width,
                "height": height,
                "thickness": thickness,
                "r_in": r_in if surf_lower in ("cylinder", "tube") else None,
                "r_out": r_out if surf_lower in ("cylinder", "tube") else None,
                "n_circumferential": n_circumferential,
                "r_node": r_node if "chiral" in pat_lower else None,
                "strut_w": strut_w,
                "square_side": square_side if pat_lower == "rotating_squares" else None,
                "rotation_angle_deg": rotation_angle_deg if pat_lower == "rotating_squares" else None,
                "hinge_radius": hinge_radius if pat_lower == "rotating_squares" else None,
                "unit_cell_size": unit_cell_sz if pat_lower == "pentamode" else None,
                "formats": list(export_formats_tuple),
                "output_path": str(out_p),
                "output_files": out_files,
                "watertight": watertight,
                "face_count": face_count,
                "vertex_count": vert_count,
                "generation_time_s": round(elapsed, 2),
            }
            report_file = out_p.parent / "run_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            return report

        if modality == "explicit":
            res = generate_explicit_conformal_lattice(
                cad_mesh,
                cell_size=cell_size,
                strut_radius=strut_radius,
                lattice_type=explicit_type,
                rule_name=sc_rule,
                export_dir=out_p.parent,
                mode=conformal_mode,
                dual_width=dual_width,
                dual_thickness=dual_thickness,
                skip_sweep=False,
            )

            lattice_mesh = res.get("mesh") if isinstance(res, dict) else res
            if lattice_mesh is None:
                cand = out_p.parent / "mesh_conformal_lattice.stl"
                if cand.is_file():
                    lattice_mesh = trimesh.load(str(cand), force="mesh")

            # Clean up intermediate generator files if present
            for extra in [out_p.parent / "mesh_conformal_lattice.stl", out_p.parent / "mesh_boundary_skin.stl"]:
                if extra.is_file() and extra.resolve() != out_p.resolve():
                    try:
                        extra.unlink()
                    except OSError:
                        pass

            export_res = export_mesh(
                lattice_mesh,
                out_p,
                formats=export_formats_tuple,
            )

            elapsed = time.time() - start_time
            watertight = export_res.watertight
            face_count = export_res.face_count
            vert_count = export_res.vertex_count
            output_files = [str(p) for p in export_res.paths_written]

            report = {
                "modality": modality,
                "cad_stl": str(cad_stl) if cad_stl else None,
                "primitive": primitive if not cad_stl else None,
                "size": size if not cad_stl else None,
                "explicit_type": explicit_type,
                "sc_rule": sc_rule if explicit_type.upper() == "SC" else None,
                "conformal_mode": conformal_mode,
                "cell_size_mm": cell_size,
                "strut_radius_mm": strut_radius,
                "formats": list(export_formats_tuple),
                "output_path": str(out_p),
                "output_files": output_files,
                "watertight": watertight,
                "face_count": face_count,
                "vertex_count": vert_count,
                "generation_time_s": round(elapsed, 2),
            }
            if export_res.health_report:
                report["is_export_ready"] = export_res.health_report.is_export_ready
                report["euler_characteristic"] = export_res.health_report.euler_characteristic
                report["boundary_edges"] = export_res.health_report.boundary_edges
                report["boundary_loops"] = export_res.health_report.boundary_loops
                report["non_manifold_edges"] = export_res.health_report.non_manifold_edges
                report["non_manifold_vertices"] = export_res.health_report.non_manifold_vertices

            report_file = out_p.parent / "run_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            return report

        # Implicit modality:
        # Determine voxel resolution: 250 um or wall_thickness / 2, whichever is lower
        if wall_thickness_mm is not None and float(wall_thickness_mm) > 0:
            w_eff = float(wall_thickness_mm)
        else:
            w_eff = (float(solid_fraction) / 1.15) * float(cell_size) / np.pi
        actual_res = float(resolution) if resolution is not None else round(min(0.250, max(w_eff / 2.0, 0.03)), 3)

        l_type = lattice_type.strip().lower()
        if l_type in ("woodpile", "cross-hatch"):
            mesh = generate_uniform_woodpile(
                stl_path=input_stl_for_engine,
                pore_size=cell_size,
                true_woodpile=(l_type == "woodpile"),
                resolution=actual_res,
                output_path=out_p,
                export_formats=export_formats_tuple,
                invert_solids=woodpile_invert,
            )
        elif enable_grading:
            if grading_mode.lower() in ("solid_fraction", "sf"):
                axis_name = "Radial" if grading_axis.upper() in ("RADIAL", "CYLINDRICAL") else grading_axis.upper()
                mesh = generate_graded_lattice(
                    stl_path=input_stl_for_engine,
                    lattice_type=lattice_type,
                    gradient_type=axis_name,
                    resolution=actual_res,
                    pore_size=cell_size * (1.0 - 1.15 * solid_fraction),
                    min_solid_fraction=min_solid_fraction,
                    max_solid_fraction=max_solid_fraction,
                    output_path=out_p,
                    export_formats=export_formats_tuple,
                )
            else:
                # Field-driven grading along chosen axis
                axis_map = {"X": "Cartesian X", "Y": "Cartesian Y", "Z": "Cartesian Z"}
                if grading_axis.upper() in ("RADIAL", "CYLINDRICAL"):
                    coord_name = "Cylindrical Radius"
                    extent = float(np.linalg.norm(cad_mesh.bounds[1][:2] - cad_mesh.bounds[0][:2])) / 2.0
                else:
                    coord_name = axis_map.get(grading_axis.upper(), "Cartesian Z")
                    ax_idx = {"X": 0, "Y": 1, "Z": 2}.get(grading_axis.upper(), 2)
                    s_min = float(cad_mesh.bounds[0][ax_idx])
                    s_max = float(cad_mesh.bounds[1][ax_idx])
                    extent = max(s_max - s_min, 1.0)

                # Doubling every doubling_interval_mm:
                L0 = cell_size
                L_end = L0 * (2.0 ** (extent / max(doubling_interval_mm, 0.1)))
                control_points = [(0.0, L0), (extent, L_end)]

                mesh = generate_field_driven_lattice(
                    stl_path=input_stl_for_engine,
                    lattice_type=lattice_type,
                    resolution=actual_res,
                    base_unit_cell_size=cell_size,
                    base_solid_fraction=solid_fraction,
                    grade_unit_cell=True,
                    unit_cell_coordinate=coord_name,
                    unit_cell_control_points=control_points,
                    grade_solid_fraction=False,
                    density_mode="wall_thickness" if wall_thickness_mm else "solid_fraction",
                    base_wall_thickness_mm=wall_thickness_mm or 0.5,
                    export_mode=export_mode,
                    shell_thickness=shell_thickness,
                    output_path=out_p,
                    export_formats=export_formats_tuple,
                )
        elif conformal_deformation:
            from graphite.implicit.conformal_laplace import generate_harmonic_conformal_lattice
            mesh = generate_harmonic_conformal_lattice(
                stl_path=input_stl_for_engine,
                lattice_type=lattice_type,
                resolution=actual_res,
                unit_cell_size=cell_size,
                solid_fraction=solid_fraction,
                wall_thickness_mm=wall_thickness_mm,
                export_mode=export_mode,
                shell_thickness=shell_thickness,
                iterations=int(conformal_iterations),
                output_path=out_p,
                export_formats=export_formats_tuple,
            )
        else:
            # Uniform conformal
            tau_val = None
            if wall_thickness_mm and float(wall_thickness_mm) > 0:
                tau_val = float(tau_from_wall_thickness_mm(wall_thickness_mm, cell_size).ravel()[0])

            mesh = generate_implicit_conformal_lattice(
                stl_path=input_stl_for_engine,
                lattice_type=lattice_type,
                resolution=actual_res,
                unit_cell_size=cell_size,
                solid_fraction=solid_fraction,
                wall_thickness_mm=wall_thickness_mm,
                tau=tau_val,
                export_mode=export_mode,
                shell_thickness=shell_thickness,
                output_path=out_p,
                export_formats=export_formats_tuple,
            )

        # Optional surface micro-texturing pass
        if enable_surface_texture:
            tex_dir = texture_direction
            if isinstance(tex_dir, str):
                dir_map = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
                tex_dir = dir_map.get(tex_dir.upper(), (0.0, 0.0, 1.0))
            tex_cfg = SurfaceTextureConfig(
                texture_type=str(texture_type).lower(),
                amplitude_mm=float(texture_amplitude_mm),
                wavelength_mm=float(texture_wavelength_mm),
                direction=tex_dir,
                profile=str(texture_profile).lower(),
                displacement_mode=str(texture_displacement_mode).lower(),
                use_triplanar=bool(texture_triplanar),
            )
            mesh = apply_surface_texture(mesh, tex_cfg)
            export_mesh(mesh, out_p, formats=export_formats_tuple)

        # Optional explicit micropillars pass
        if enable_micropillars:
            from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
            pill_cfg = MicropillarConfig(
                diameter_mm=float(pillar_diameter_mm),
                height_mm=float(pillar_height_mm),
                spacing_mm=float(pillar_spacing_mm),
                location=str(pillar_location).lower(),
                distribution=str(pillar_distribution).lower(),
                filter_printable=bool(pillar_filter_printable),
            )
            mesh = generate_micropillars(mesh, pill_cfg, boundary_mesh=cad_mesh)
            export_mesh(mesh, out_p, formats=export_formats_tuple)

        elapsed = time.time() - start_time
        watertight = bool(mesh.is_watertight) if hasattr(mesh, "is_watertight") else False
        face_count = len(mesh.faces) if hasattr(mesh, "faces") else 0
        vert_count = len(mesh.vertices) if hasattr(mesh, "vertices") else 0

        stem = out_p.stem
        output_files = [str(out_p.parent / f"{stem}.{fmt}") for fmt in export_formats_tuple]

        report = {
            "modality": modality,
            "cad_stl": str(cad_stl) if cad_stl else None,
            "primitive": primitive if not cad_stl else None,
            "size": size if not cad_stl else None,
            "lattice_type": lattice_type,
            "cell_size_mm": cell_size,
            "solid_fraction": solid_fraction,
            "wall_thickness_mm": wall_thickness_mm,
            "export_mode": export_mode,
            "shell_thickness_mm": shell_thickness,
            "formats": list(export_formats_tuple),
            "enable_grading": enable_grading,
            "grading_axis": grading_axis if enable_grading else None,
            "grading_mode": grading_mode if enable_grading else None,
            "conformal_deformation": conformal_deformation,
            "surface_texture": texture_type if enable_surface_texture else None,
            "micropillars": enable_micropillars,
            "resolution_mm": actual_res,
            "output_path": str(out_p),
            "output_files": output_files,
            "watertight": watertight,
            "face_count": face_count,
            "vertex_count": vert_count,
            "generation_time_s": round(elapsed, 2),
        }

        report_file = out_p.parent / "run_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    finally:
        if tmp_cad_path is not None and os.path.exists(tmp_cad_path):
            try:
                os.remove(tmp_cad_path)
            except OSError:
                pass


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Graphite Headless Recipe Runner")
    parser.add_argument("--recipe", type=str, help="Path to JSON or YAML recipe file")
    parser.add_argument(
        "--modality",
        type=str,
        default="implicit",
        choices=["implicit", "explicit", "interlinked", "auxetics"],
        help="Lattice modality: 'implicit' (TPMS), 'explicit' (strut scaffolds), 'interlinked' (PAMs / chainmail), or 'auxetics' (metamaterials)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="tetra_chiral",
        choices=["tetra_chiral", "tri_chiral", "anti_tetra_chiral", "anti_tri_chiral", "reentrant", "rotating_squares", "pentamode"],
        help="Auxetic metamaterial pattern",
    )
    parser.add_argument(
        "--surface",
        type=str,
        default="plate",
        choices=["plate", "cylinder", "tube"],
        help="Auxetic surface type ('plate' or 'cylinder'/'tube')",
    )
    parser.add_argument("--plate-width", type=float, default=50.0, help="Auxetic plate width or tube length (mm)")
    parser.add_argument("--plate-height", type=float, default=50.0, help="Auxetic plate height (mm)")
    parser.add_argument("--thickness", type=float, default=2.0, help="Auxetic plate / rib thickness (mm)")
    parser.add_argument("--r-in", type=float, default=15.0, help="Auxetic tube inner radius (mm)")
    parser.add_argument("--r-out", type=float, default=17.5, help="Auxetic tube outer radius (mm)")
    parser.add_argument("--n-circumferential", type=int, default=6, help="Auxetic circumferential unit cell count")
    parser.add_argument("--r-node", type=float, default=2.0, help="Auxetic circular node radius (mm)")
    parser.add_argument("--strut-w", type=float, default=1.0, help="Auxetic ligament width (mm)")
    parser.add_argument("--square-side", type=float, default=10.0, help="Rotating squares side length (mm)")
    parser.add_argument("--rotation-angle", type=float, default=30.0, help="Rotating squares deployment angle (deg)")
    parser.add_argument("--hinge-radius", type=float, default=0.45, help="Rotating squares living hinge radius (mm)")
    parser.add_argument(
        "--interlinked-cell",
        type=str,
        default="c6tt",
        choices=["c6tt", "d4tet", "j4oct", "euro_4in1", "kusari", "nasa_space_fabric"],
        help="Interlinked / PAM unit cell type",
    )
    parser.add_argument(
        "--interlinked-seeding",
        type=str,
        default="cartesian",
        choices=["cartesian", "cylindrical"],
        help="Interlinked lattice seeding topology: 'cartesian' (Policy A Inset Culling) or 'cylindrical' (Tube Wrap)",
    )
    parser.add_argument("--wire-radius", type=float, default=0.40, help="Wire/strut radius for interlinked lattice (mm)")
    parser.add_argument("--min-clearance", type=float, default=0.35, help="Minimum physical clearance gap for interlinked metamaterials (mm)")
    parser.add_argument("--auto-resolve-pitch", action="store_true", help="Auto-solve unit cell pitch to satisfy target clearance exactly")
    parser.add_argument("--cull-margin", type=float, default=0.50, help="Perimeter inset margin for Policy A whole-cell culling (mm)")
    parser.add_argument("--add-perimeter-frame", action="store_true", help="Synthesize solid handling/tensile testing perimeter frame (Policy C)")
    parser.add_argument("--frame-wall-thickness", type=float, default=2.0, help="Solid frame wall thickness (mm)")
    parser.add_argument("--frame-margin", type=float, default=0.50, help="Solid frame penetration margin (mm)")
    parser.add_argument("--frame-shape", type=str, default="box", choices=["box", "cylinder"], help="Solid frame geometry shape")
    parser.add_argument("--cylinder-radius", type=float, default=15.0, help="Cylinder radius for cylindrical wrap seeding (mm)")
    parser.add_argument("--cylinder-height", type=float, default=25.0, help="Cylinder height for cylindrical wrap seeding (mm)")
    parser.add_argument(
        "--explicit-type",
        type=str,
        default="SC",
        choices=["A15", "SC"],
        help="Explicit scaffold type: 'A15' (tet Kagome) or 'SC' (modular hex)",
    )
    parser.add_argument(
        "--sc-rule",
        type=str,
        default="octahedral",
        choices=["octahedral", "grid", "star", "octet", "cross", "kelvin", "tesseract", "hex_face_dual", "cubic"],
        help="SC hex topology rule",
    )
    parser.add_argument(
        "--strut-radius",
        type=float,
        default=0.4,
        help="Explicit strut radius in mm",
    )
    parser.add_argument(
        "--dual-width",
        type=float,
        default=None,
        help="Surface dual ribbon width in mm (SC explicit, defaults to max(1.6, 3.2*radius))",
    )
    parser.add_argument(
        "--dual-thickness",
        type=float,
        default=None,
        help="Surface dual shell thickness in mm (SC explicit, defaults to max(0.8, 1.6*radius))",
    )
    parser.add_argument(
        "--conformal-mode",
        type=str,
        default="conformal",
        choices=["conformal", "boolean"],
        help="Explicit conformation mode: 'conformal' or 'boolean'",
    )
    parser.add_argument("--cad-stl", type=str, default=None, help="Path to custom CAD STL file (e.g. test_parts/BaseRing_1to1.STL)")
    parser.add_argument("--primitive", type=str, default="Cube", choices=["Cube", "Cylinder", "Sphere", "Toros", "Torus"])
    parser.add_argument("--size", type=float, default=20.0, help="Primitive dimension (mm)")
    parser.add_argument(
        "--lattice",
        type=str,
        default="Gyroid",
        choices=SUPPORTED_TPMS_EQUATIONS,
        help="TPMS lattice equation",
    )
    parser.add_argument("--cell-size", type=float, default=5.0, help="Unit cell size (mm)")
    parser.add_argument("--solid-fraction", type=float, default=0.30, help="Target solid fraction")
    parser.add_argument("--wall-thickness", type=float, default=None, help="Physical wall thickness (mm)")
    parser.add_argument(
        "--export-mode",
        type=str,
        default="core",
        choices=["core", "skin", "combined"],
        help="Export mode: 'core' (lattice only), 'skin' (solid shell only), or 'combined' (lattice with shell)",
    )
    parser.add_argument("--shell-thickness", type=float, default=2.0, help="Outer shell thickness in mm")
    parser.add_argument(
        "--formats",
        type=str,
        default="stl",
        help="Export file formats: 'stl', '3mf', or 'stl,3mf'",
    )
    parser.add_argument("--grading", action="store_true", help="Enable grading")
    parser.add_argument("--grading-axis", type=str, default="Z", choices=["X", "Y", "Z", "Radial"])
    parser.add_argument("--grading-mode", type=str, default="cell_size", choices=["cell_size", "solid_fraction"])
    parser.add_argument("--min-sf", type=float, default=0.15, help="Minimum solid fraction for SF grading")
    parser.add_argument("--max-sf", type=float, default=0.45, help="Maximum solid fraction for SF grading")
    parser.add_argument("--doubling-interval", type=float, default=5.0, help="Doubling interval in mm (frequency grading)")
    parser.add_argument(
        "--surface-texture",
        type=str,
        default=None,
        choices=["microgrooves", "bumps", "knurling", "spinodal"],
        help="Procedural surface micro-texture type",
    )
    parser.add_argument("--texture-amplitude", type=float, default=0.025, help="Texture amplitude in mm (default 0.025 = 25 um)")
    parser.add_argument("--texture-wavelength", type=float, default=0.050, help="Texture wavelength in mm (default 0.050 = 50 um)")
    parser.add_argument("--texture-direction", type=str, default="Z", choices=["X", "Y", "Z"], help="Microgroove direction axis")
    parser.add_argument("--texture-profile", type=str, default="sine", choices=["sine", "triangle", "square"], help="Microgroove profile")
    parser.add_argument("--texture-displacement-mode", type=str, default="centered", choices=["centered", "emboss", "engrave"], help="Displacement mode")
    parser.add_argument("--texture-triplanar", action="store_true", help="Apply texture using triplanar projection")
    parser.add_argument("--conformal-deformation", action="store_true", help="Enable experimental harmonic Laplace conformal deformation")
    parser.add_argument("--conformal-iterations", type=int, default=150, help="Iterations for harmonic conformal deformation")
    parser.add_argument("--micropillars", action="store_true", help="Synthesize explicit micropillar / microfiber forest")
    parser.add_argument("--pillar-diameter", type=float, default=0.050, help="Micropillar diameter in mm (default 0.050 = 50 um)")
    parser.add_argument("--pillar-height", type=float, default=0.200, help="Micropillar height in mm (default 0.200 = 200 um)")
    parser.add_argument("--pillar-spacing", type=float, default=0.200, help="Micropillar spacing in mm (default 0.200 = 200 um)")
    parser.add_argument("--pillar-location", type=str, default="all", choices=["all", "outer_only", "internal_only"], help="Pillar location")
    parser.add_argument("--pillar-distribution", type=str, default="poisson_disk", choices=["poisson_disk", "random"], help="Pillar distribution")
    parser.add_argument("--pillar-filter-printable", action="store_true", help="Filter unsupported overhang angles for 3D printability")
    parser.add_argument("--woodpile-invert", action="store_true", help="Invert solid and void for woodpile lattices")
    parser.add_argument("--resolution", type=float, default=None, help="Voxel resolution (mm) (default: min(250um, wall/2))")
    parser.add_argument("--output", type=str, default=None, help="Output file path (e.g. outputs/phase3_explicit/basering_a15.stl)")

    args = parser.parse_args()

    if args.recipe:
        recipe_path = Path(args.recipe)
        with open(recipe_path, "r") as f:
            if recipe_path.suffix.lower() == ".json":
                params = json.load(f)
            else:
                import yaml
                params = yaml.safe_load(f)
        report = run_headless_recipe(**params)
    else:
        report = run_headless_recipe(
            modality=args.modality,
            explicit_type=args.explicit_type,
            sc_rule=args.sc_rule,
            strut_radius=args.strut_radius,
            dual_width=args.dual_width,
            dual_thickness=args.dual_thickness,
            conformal_mode=args.conformal_mode,
            primitive=args.primitive,
            size=args.size,
            cad_stl=args.cad_stl,
            lattice_type=args.lattice,
            cell_size=args.cell_size,
            solid_fraction=args.solid_fraction,
            wall_thickness_mm=args.wall_thickness,
            export_mode=args.export_mode,
            shell_thickness=args.shell_thickness,
            formats=args.formats,
            enable_grading=args.grading,
            grading_axis=args.grading_axis,
            grading_mode=args.grading_mode,
            min_solid_fraction=args.min_sf,
            max_solid_fraction=args.max_sf,
            doubling_interval_mm=args.doubling_interval,
            conformal_deformation=args.conformal_deformation,
            conformal_iterations=args.conformal_iterations,
            enable_surface_texture=bool(args.surface_texture),
            texture_type=args.surface_texture or "microgrooves",
            texture_amplitude_mm=args.texture_amplitude,
            texture_wavelength_mm=args.texture_wavelength,
            texture_direction={"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}.get(args.texture_direction, (0.0, 0.0, 1.0)),
            texture_profile=args.texture_profile,
            texture_displacement_mode=args.texture_displacement_mode,
            texture_triplanar=args.texture_triplanar,
            enable_micropillars=args.micropillars,
            pillar_diameter_mm=args.pillar_diameter,
            pillar_height_mm=args.pillar_height,
            pillar_spacing_mm=args.pillar_spacing,
            pillar_location=args.pillar_location,
            pillar_distribution=args.pillar_distribution,
            pillar_filter_printable=args.pillar_filter_printable,
            woodpile_invert=args.woodpile_invert,
            resolution=args.resolution,
            output_path=args.output,
            interlinked_cell=args.interlinked_cell,
            interlinked_seeding=args.interlinked_seeding,
            wire_radius=args.wire_radius,
            min_clearance=args.min_clearance,
            auto_resolve_pitch=args.auto_resolve_pitch,
            cull_margin=args.cull_margin,
            add_perimeter_frame=args.add_perimeter_frame,
            frame_wall_thickness=args.frame_wall_thickness,
            frame_margin=args.frame_margin,
            frame_shape=args.frame_shape,
            cylinder_radius=args.cylinder_radius,
            cylinder_height=args.cylinder_height,
            pattern=args.pattern,
            surface=args.surface,
            width=args.plate_width,
            height=args.plate_height,
            thickness=args.thickness,
            r_in=args.r_in,
            r_out=args.r_out,
            n_circumferential=args.n_circumferential,
            r_node=args.r_node,
            strut_w=args.strut_w,
            square_side=args.square_side,
            rotation_angle_deg=args.rotation_angle,
            hinge_radius=args.hinge_radius,
        )

    print("\n--- Generation Complete ---")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
