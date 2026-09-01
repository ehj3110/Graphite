"""
Standalone TPMS/lattice GMSH volume mesh (STL or STEP).

Implements the four-point lattice meshing checklist:
  1. STL → classifySurfaces + createGeometry
  2. Explicit surface loop + volume; open-boundary warning
  3. Adaptive CharacteristicLengthFromCurvature (default) or legacy Mesh.MeshSize*
  4. STEP → gmsh.model.occ.importShapes + healShapes

Usage:
  python scripts/mesh_tpms_lattice_gmsh.py path/to/lattice.stl [--h 0.15]
  python scripts/mesh_tpms_lattice_gmsh.py path/to/part.step --step [--h 0.2]
  python scripts/mesh_tpms_lattice_gmsh.py path/to/lattice.stl --export-vtu lattice_debug.vtu
  python scripts/mesh_tpms_lattice_gmsh.py path/to/lattice.stl --clean-surface --h 0.15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_log import aristo_log, aristo_stage
from graphite.aristo.gmsh_lattice_mesh import (
    generate_lattice_fea_mesh_from_step,
    generate_lattice_fea_mesh_from_stl,
)
from graphite.aristo.mesh_quality import (
    build_mesh_quality_report,
    build_quality_mask,
    mesh_has_giant_elements,
)
from graphite.aristo.stl_surface_clean import (
    clean_stl_surface_file_to_temp_stl,
    clean_stl_surface_to_temp_stl,
    maybe_clean_stl_surface,
    surface_clean_backend,
)


def _write_mesh_vtu(
    nodes: np.ndarray,
    elems: np.ndarray,
    aspects: np.ndarray,
    max_edges: np.ndarray,
    quality_ok: np.ndarray,
    output_path: Path,
) -> None:
    """Build a linear-tet UnstructuredGrid and save VTU with quality cell data."""
    n_cells = elems.shape[0]
    cells = np.column_stack([np.full(n_cells, 4, dtype=np.int64), elems]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid.cell_data["aspect_ratio"] = np.asarray(aspects, dtype=np.float64)
    grid.cell_data["max_edge_mm"] = np.asarray(max_edges, dtype=np.float64)
    grid.cell_data["quality_ok"] = quality_ok.astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="TPMS/lattice GMSH tet mesh")
    parser.add_argument("input", type=Path, help="STL or STEP path")
    parser.add_argument("--h", type=float, default=0.15, help="Target edge length (mm)")
    parser.add_argument("--step", action="store_true", help="Input is STEP (OCC path)")
    parser.add_argument(
        "--no-curvature",
        action="store_true",
        help="Disable curvature-based sizing",
    )
    parser.add_argument(
        "--legacy-curvature",
        action="store_true",
        help="Use legacy h-scaled Mesh.MeshSize* instead of CharacteristicLength*",
    )
    parser.add_argument("--char-min", type=float, default=0.08, help="CharacteristicLengthMin (mm)")
    parser.add_argument("--char-max", type=float, default=0.8, help="CharacteristicLengthMax (mm)")
    parser.add_argument(
        "--elements-per-two-pi",
        type=int,
        default=15,
        help="Mesh.MinimumElementsPerTwoPi",
    )
    parser.add_argument(
        "--allow-single-surface-fallback",
        action="store_true",
        help="Retry without classifySurfaces if classified mesh fails",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON report path")
    parser.add_argument(
        "--export-vtu",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write tet mesh with aspect_ratio and max_edge_mm cell data for ParaView",
    )
    parser.add_argument(
        "--clean-surface",
        action="store_true",
        help="Open3D isotropic surface remesh before gmsh (target edge = --h)",
    )
    args = parser.parse_args()

    config = AristoConfig(
        fea_mesh_resolution=args.h,
        fea_gmsh_mesh_mode="tpms",
        fea_gmsh_adaptive_curvature=not args.legacy_curvature,
        fea_gmsh_char_length_min=args.char_min,
        fea_gmsh_char_length_max=args.char_max,
        fea_gmsh_min_elements_per_two_pi=args.elements_per_two_pi,
        fea_gmsh_classify_only=not args.allow_single_surface_fallback,
        fea_clean_stl_surface=args.clean_surface,
    )
    aristo_log(
        f"mesh_tpms_lattice_gmsh: {args.input} h={args.h} mm "
        f"curvature={not args.no_curvature} adaptive={config.fea_gmsh_adaptive_curvature} "
        f"cl=[{config.fea_gmsh_char_length_min}, {config.fea_gmsh_char_length_max}] "
        f"step={args.step} clean_surface={args.clean_surface}"
    )
    if not args.no_curvature and config.fea_gmsh_adaptive_curvature:
        print(
            f"Meshing {args.input} (classifySurfaces + adaptive curvature: "
            f"min={args.char_min} max={args.char_max} mm)"
        )
    else:
        print(f"Meshing {args.input} (h={args.h} mm, curvature={not args.no_curvature})")

    if args.step:
        with aristo_stage("generate_lattice_fea_mesh_from_step"):
            nodes, elems, surf = generate_lattice_fea_mesh_from_step(
                args.input,
                args.h,
                silent=False,
                use_curvature=not args.no_curvature,
            )
    else:
        mesh = trimesh.load_mesh(str(args.input))
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        print(
            f"  STL: {len(mesh.faces):,} faces, watertight={mesh.is_watertight}, "
            f"volume={mesh.volume:.4f} mm³"
        )
        if args.clean_surface:
            with aristo_stage("clean_stl_surface_to_temp"):
                backend = surface_clean_backend(config)
                aristo_log(f"surface clean backend={backend}")
                if backend == "open3d_subprocess":
                    temp_stl = clean_stl_surface_file_to_temp_stl(
                        args.input.resolve(),
                        args.h,
                        config=config,
                    )
                else:
                    temp_stl = clean_stl_surface_to_temp_stl(
                        mesh, args.h, config=config
                    )
                print(f"  cleaned surface STL ({backend}): {temp_stl}")
                mesh = trimesh.load_mesh(str(temp_stl))
                if not isinstance(mesh, trimesh.Trimesh):
                    mesh = mesh.dump(concatenate=True)
                print(
                    f"  after clean: {len(mesh.faces):,} faces, "
                    f"watertight={mesh.is_watertight}, volume={mesh.volume:.4f} mm³"
                )
        else:
            mesh = maybe_clean_stl_surface(mesh, args.h, config=config)
        with aristo_stage("generate_lattice_fea_mesh_from_stl"):
            nodes, elems, surf = generate_lattice_fea_mesh_from_stl(
                mesh,
                args.h,
                config=config,
                silent=False,
                use_curvature=not args.no_curvature,
            )

    mask, _, aspects, max_edges = build_quality_mask(nodes, elems, config)
    giants, max_edge, vol_ratio = mesh_has_giant_elements(nodes, elems, config)
    report = build_mesh_quality_report(
        nodes,
        elems,
        config,
        quality_mask=mask,
        remesh_attempts=1,
    )
    report["max_edge_mm"] = max_edge
    report["has_giant_elements"] = giants
    report["vol_median_ratio"] = vol_ratio

    print(f"  nodes={len(nodes):,} tets={len(elems):,} surface_tris={len(surf):,}")
    print(
        f"  poor_fraction={report.get('poor_fraction', 0):.4f} "
        f"max_edge={max_edge:.4f} mm giants={giants}"
    )

    out = args.output or args.input.with_suffix(".tpms_mesh_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if args.export_vtu is not None:
        _write_mesh_vtu(nodes, elems, aspects, max_edges, mask, args.export_vtu)
        print(f"Wrote {args.export_vtu}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
