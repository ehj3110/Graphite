import os
import sys
import json
import argparse
import trimesh
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Imports from graphite
from graphite.implicit.conformal import generate_conformal_lattice
from graphite.implicit.graded import generate_graded_lattice
from graphite.implicit.chirped import generate_chirped_lattice
from graphite.implicit.osteochondral import generate_osteochondral_lattice
from graphite.implicit.field_driven import generate_field_driven_lattice
from graphite.implicit.woodpile_input import (
    build_piecewise_woodpile_mesh,
    woodpile_spec_from_config,
)
from graphite.io.mesh_export import export_mesh, formats_from_request, resolve_export_formats


def run_headless(stl_path, config_path, output_path, export_format=None):
    print(f"--- Graphite Headless Runner ---")
    print(f"Input: {stl_path}")
    print(f"Config: {config_path}")
    
    import yaml
    import json
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            params = json.load(f)
        else:
            params = yaml.safe_load(f)
        
    engine_type = params.get("engine_type", "Implicit (TPMS)")
    print(f"Engine: {engine_type}")

    export_formats = export_format or params.get("export_format", "stl")
    export_formats = resolve_export_formats(output_path, export_formats)
    print(f"Export formats: {export_formats}")

    if engine_type == "Implicit (TPMS)":
        grading_mode = params.get("grading_mode", "Uniform")
        print(f"Grading Mode: {grading_mode}")
        
        common = dict(
            stl_path=stl_path,
            lattice_type=params["lattice_type"],
            resolution=params["resolution"],
            center_origin=params.get("center_origin", True),
            output_path=output_path,
            export_formats=export_formats,
        )

        if grading_mode == "Uniform":
            generate_conformal_lattice(
                pore_size=params.get("pore_size"),
                unit_cell_size=params.get("unit_cell_size"),
                solid_fraction=params["solid_fraction"],
                export_mode=params.get("export_mode", "core"),
                shell_thickness=params.get("shell_thickness", 2.0),
                **common,
            )
        elif grading_mode == "Field Controls":
            generate_field_driven_lattice(
                base_unit_cell_size=params.get("unit_cell_size", params.get("pore_size", 5.0)),
                base_solid_fraction=float(params.get("solid_fraction", 0.33)),
                grade_unit_cell=bool(params.get("grade_unit_cell", False)),
                unit_cell_coordinate=params.get("unit_cell_coordinate", "Cartesian"),
                unit_cell_control_points=params.get("unit_cell_control_points"),
                unit_cell_cartesian_control_points=params.get(
                    "unit_cell_cartesian_control_points"
                ),
                grade_solid_fraction=bool(params.get("grade_solid_fraction", False)),
                solid_fraction_coordinate=params.get(
                    "solid_fraction_coordinate", "Cartesian"
                ),
                solid_fraction_control_points=params.get("solid_fraction_control_points"),
                solid_fraction_cartesian_control_points=params.get(
                    "solid_fraction_cartesian_control_points"
                ),
                calibrate_solid_fraction=bool(params.get("calibrate_solid_fraction", True)),
                density_mode=params.get("density_mode", "solid_fraction"),
                base_wall_thickness_mm=float(params.get("wall_thickness_mm", 0.5)),
                export_mode=params.get("export_mode", "core"),
                shell_thickness=params.get("shell_thickness", 2.0),
                **common,
            )
        elif grading_mode == "Variable Porosity (Thickness)":
            generate_graded_lattice(
                gradient_type=params.get("gradient_type", "Z"),
                modifier_path=params.get("modifier_path"),
                pore_size=params.get("pore_size", 5.0),
                min_solid_fraction=params.get("min_solid_fraction", 0.10),
                max_solid_fraction=params.get("max_solid_fraction", params["solid_fraction"]),
                transition_width=params.get("transition_width", 5.0),
                **common,
            )
        elif grading_mode == "Variable Pore Size (Chirped)":
            generate_chirped_lattice(
                gradient_type=params.get("gradient_type", "Z"),
                modifier_path=params.get("modifier_path"),
                solid_fraction=params["solid_fraction"],
                transition_width=params.get("transition_width", 5.0),
                **common,
            )
        elif grading_mode == "Osteochondral (Layered Z)":
            generate_osteochondral_lattice(
                z_heights=params["osteo_z"],
                pore_sizes=params["osteo_p"],
                solid_fractions=params["osteo_sf"],
                **common,
            )
        else:
            raise ValueError(f"Unsupported grading_mode for headless runner: {grading_mode}")

    elif engine_type == "Implicit (Woodpile)":
        grading_mode = params.get("grading_mode", "Piecewise Z bands")
        if grading_mode != "Piecewise Z bands":
            raise ValueError(
                f"Implicit (Woodpile) supports grading_mode 'Piecewise Z bands', got {grading_mode!r}."
            )
        spec = woodpile_spec_from_config(params)
        mesh, report = build_piecewise_woodpile_mesh(spec)
        export_mesh(mesh, Path(output_path), formats=export_formats)
        watertight = report.get(
            "watertight_after_repair",
            report.get("watertight_union_after_repair", report.get("watertight")),
        )
        print(
            f"Woodpile piecewise ({spec.generator}): watertight={watertight} "
            f"faces={report['faces']:,}"
        )
    
    elif "Explicit" in engine_type:
        from graphite.explicit import generate_conformal_lattice
        from graphite.explicit.mesh_repair import repair_cad_mesh
        
        print("Generating Explicit Lattice...")
        raw_mesh = trimesh.load(stl_path)
        print(f"Repairing input mesh: {stl_path}")
        repaired_mesh = repair_cad_mesh(raw_mesh)
        
        cell_size = params.get("explicit_cell_size", 5.0)
        strut_radius = params.get("explicit_strut_radius", 0.5)
        lattice_type = params.get("explicit_lattice_type", "A15")
        mode = params.get("explicit_mode", "conformal")
        
        out = Path(output_path)
        export_dir = str(out.parent)
        
        result = generate_conformal_lattice(
            cad_filepath=repaired_mesh,
            cell_size=cell_size,
            strut_radius=strut_radius,
            lattice_type=lattice_type,
            export_dir=export_dir,
            mode=mode,
        )

    print(f"--- Done! Result saved under {output_path} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphite Headless Lattice Generator")
    parser.add_argument("stl", help="Input STL file")
    parser.add_argument("config", help="JSON or YAML configuration file")
    parser.add_argument("output", help="Output base path (extension optional)")
    parser.add_argument(
        "--format",
        dest="export_format",
        choices=["stl", "step", "both"],
        default=None,
        help="Export format (overrides config export_format)",
    )
    
    args = parser.parse_args()
    run_headless(args.stl, args.config, args.output, export_format=args.export_format)
