"""
Lattice volume mesh (classifySurfaces / TPMS path).

Default: adaptive CharacteristicLengthFromCurvature sizing, Delaunay 3D fill,
Laplace-only optimization, linear Tet4 (P1). Use ``--strict-uniform`` or
``--order 2`` only for R&D experiments.

Usage:
  python scripts/gmsh_lattice_mesh.py path/to/Mirae_LatticeSlab_V4_fixed.stl
  python scripts/gmsh_lattice_mesh.py path/to/part.stl --h 0.15 --export-vtu debug.vtu
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
from graphite.aristo.gmsh_lattice_mesh import generate_lattice_fea_mesh_from_stl
from graphite.aristo.mesh_quality import (
    build_mesh_quality_report,
    build_quality_mask,
    mesh_has_giant_elements,
)


def _corner_connectivity(elems: np.ndarray) -> np.ndarray:
    """Linear corner nodes for quality metrics on Tet10 meshes."""
    return elems[:, :4] if elems.shape[1] > 4 else elems


def _write_mesh_vtu(
    nodes: np.ndarray,
    elems: np.ndarray,
    aspects: np.ndarray,
    max_edges: np.ndarray,
    quality_ok: np.ndarray,
    output_path: Path,
) -> None:
    n_cells = elems.shape[0]
    if elems.shape[1] == 10:
        cells = np.column_stack([np.full(n_cells, 10, dtype=np.int64), elems]).ravel()
        cell_types = np.full(n_cells, pv.CellType.QUADRATIC_TETRA, dtype=np.uint8)
    else:
        cells = np.column_stack([np.full(n_cells, 4, dtype=np.int64), elems]).ravel()
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid.cell_data["aspect_ratio"] = np.asarray(aspects, dtype=np.float64)
    grid.cell_data["max_edge_mm"] = np.asarray(max_edges, dtype=np.float64)
    grid.cell_data["quality_ok"] = quality_ok.astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lattice GMSH mesh: classifySurfaces + adaptive curvature (default P1)",
    )
    parser.add_argument("input", type=Path, help="Watertight lattice STL")
    parser.add_argument(
        "--h",
        type=float,
        default=0.15,
        help="Target resolution scale (mm); adaptive cl_min/max unless --strict-uniform",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        choices=(1, 2),
        help="Volume element order (2 = straight-sided Tet10; default linear Tet4)",
    )
    parser.add_argument(
        "--strict-uniform",
        action="store_true",
        help="Opt-in global uniform MeshSizeMin/Max (R&D; not recommended for FEA)",
    )
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
        help="Write tet mesh VTU for ParaView",
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
    )

    mesh_meta: dict = {}
    aristo_log(
        f"gmsh_lattice_mesh: {args.input} h={args.h} mm order={args.order} "
        f"strict_uniform={args.strict_uniform} adaptive={config.fea_gmsh_adaptive_curvature} "
        f"classify_only={config.fea_gmsh_classify_only}"
    )
    if args.strict_uniform:
        print(
            f"Meshing {args.input} (classifySurfaces + strict uniform h={args.h} mm, "
            f"Tet{4 if args.order == 1 else 10})"
        )
    elif not args.no_curvature and config.fea_gmsh_adaptive_curvature:
        print(
            f"Meshing {args.input} (classifySurfaces + adaptive curvature: "
            f"min={args.char_min} max={args.char_max} mm, Tet{4 if args.order == 1 else 10})"
        )
    else:
        print(
            f"Meshing {args.input} (h={args.h} mm, curvature={not args.no_curvature}, "
            f"Tet{4 if args.order == 1 else 10})"
        )

    mesh = trimesh.load_mesh(str(args.input))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    print(
        f"  STL: {len(mesh.faces):,} faces, watertight={mesh.is_watertight}, "
        f"volume={mesh.volume:.4f} mm³"
    )

    with aristo_stage("generate_lattice_fea_mesh_from_stl"):
        nodes, elems, surf = generate_lattice_fea_mesh_from_stl(
            mesh,
            args.h,
            config=config,
            silent=False,
            use_curvature=not args.no_curvature,
            strict_uniform=args.strict_uniform,
            mesh_order=args.order,
            mesh_meta=mesh_meta,
        )

    elems_linear = _corner_connectivity(elems)
    mask, _, aspects, max_edges = build_quality_mask(nodes, elems_linear, config)
    giants, max_edge, vol_ratio = mesh_has_giant_elements(nodes, elems_linear, config)
    report = build_mesh_quality_report(
        nodes,
        elems_linear,
        config,
        quality_mask=mask,
        remesh_attempts=1,
    )
    effective_order = mesh_meta.get("effective_mesh_order", args.order)
    if args.strict_uniform:
        report["mesh_mode"] = "tpms_classify_strict_uniform"
    elif config.fea_gmsh_adaptive_curvature and not args.no_curvature:
        report["mesh_mode"] = "tpms_classify_adaptive_linear"
    else:
        report["mesh_mode"] = "tpms_classify_legacy_sizing"
    report["requested_mesh_order"] = args.order
    report["effective_mesh_order"] = effective_order
    report["order_elevation_aborted"] = mesh_meta.get("order_elevation_aborted", False)
    report["illegal_linear_tets"] = mesh_meta.get("illegal_linear_tets", 0)
    report["inverted_linear_tets"] = mesh_meta.get("inverted_linear_tets", 0)
    report["algorithm_3d"] = mesh_meta.get("algorithm_3d", 1)
    report["optimize_netgen"] = mesh_meta.get("optimize_netgen", 0)
    if args.strict_uniform:
        report["uniform_h_mm"] = args.h
    report["mesh_order"] = effective_order
    report["second_order_linear"] = effective_order >= 2
    report["n_nodes"] = int(len(nodes))
    report["n_elements"] = int(len(elems))
    report["n_elements_linear_corners"] = int(len(elems_linear))
    report["max_edge_mm"] = max_edge
    report["has_giant_elements"] = giants
    report["vol_median_ratio"] = vol_ratio

    print(f"  nodes={len(nodes):,} tets={len(elems):,} surface_tris={len(surf):,}")
    print(
        f"  poor_fraction={report.get('poor_fraction', 0):.4f} "
        f"max_edge={max_edge:.4f} mm giants={giants}"
    )
    if report.get("order_elevation_aborted"):
        print(
            "  WARNING: order elevation aborted; exported linear P1 "
            f"(illegal={report['illegal_linear_tets']} "
            f"inverted={report['inverted_linear_tets']})"
        )

    if args.strict_uniform:
        default_out = args.input.with_suffix(".uniform_mesh_report.json")
    else:
        default_out = args.input.with_suffix(".adaptive_mesh_report.json")
    out = args.output or default_out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if args.export_vtu is not None:
        _write_mesh_vtu(nodes, elems, aspects, max_edges, mask, args.export_vtu)
        print(f"Wrote {args.export_vtu}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
