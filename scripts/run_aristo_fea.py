#!/usr/bin/env python
"""
Linear-elastic compression FEA via Aristo.

**Default input:** piecewise Split-P **implicit field** (evaluated in memory).
**Developer override:** ``--stl`` loads a watertight boundary mesh file.

Default BC: ``flat_top_vertex_plane``. Default volume mesh: ``single_surface``.
Use ``--mesh-mode tpms`` with ``--stl`` for Mirae-style classifySurfaces.

Run from repo root::

    python scripts/run_aristo_fea.py --force-n 1 --h 0.005
    python scripts/run_aristo_fea.py --domain box --width-x-mm 1 --depth-y-mm 1 --height-mm 1 --h 0.005
    python scripts/run_aristo_fea.py --stl path/to/lattice.stl --mesh-mode tpms   # dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.fea_runner import (
    DEFAULT_TOP_MIN_VERTICES_AT_PLANE,
    detect_flat_top_bcs,
    run_compression_fea_case,
)
from graphite.aristo.implicit_input import (
    add_piecewise_implicit_args,
    add_stl_dev_arg,
    build_piecewise_splitp_boundary_mesh,
    implicit_spec_from_args,
)


def _load_stl_mesh(stl_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stl_dev_arg(parser)
    add_piecewise_implicit_args(parser, default_h=0.12)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force-n", type=float, default=10.0)
    parser.add_argument(
        "--youngs-modulus",
        type=float,
        default=2800.0,
        help="Young's modulus (MPa). Default 2800 (Resin SLA).",
    )
    parser.add_argument(
        "--mesh-mode",
        choices=("single_surface", "tpms"),
        default=None,
        help="Default: single_surface (implicit) or tpms (--stl dev path).",
    )
    parser.add_argument(
        "--clean-stl",
        action="store_true",
        help="Open3D isotropic surface remesh before volume fill (STL dev path).",
    )
    parser.add_argument(
        "--quality-mode",
        choices=("quick", "thorough"),
        default="quick",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "scipy", "pardiso"),
        default="pardiso",
    )
    parser.add_argument(
        "--bc-load-mode",
        choices=("flat_top_vertex_plane", "flat_top"),
        default="flat_top_vertex_plane",
    )
    parser.add_argument(
        "--top-min-vertices",
        type=int,
        default=DEFAULT_TOP_MIN_VERTICES_AT_PLANE,
    )
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--export-boundary-stl",
        type=Path,
        default=None,
        help="Optional: write MC boundary STL when using implicit input.",
    )
    args = parser.parse_args()

    implicit_report = None
    fea_input_mode = "stl" if args.stl is not None else "implicit"
    mesh_mode = args.mesh_mode
    if mesh_mode is None:
        mesh_mode = "tpms" if fea_input_mode == "stl" else "single_surface"

    if fea_input_mode == "stl":
        if not args.stl.is_file():
            print(f"STL not found: {args.stl}")
            return 1
        print(f"[dev STL] Loading {args.stl}")
        mesh = _load_stl_mesh(args.stl)
        stem = args.stem or args.stl.stem
        out_dir = args.out_dir or args.stl.parent
    else:
        spec = implicit_spec_from_args(args)
        if args.stem:
            spec.stem = args.stem
        print(
            f"[implicit] piecewise Split-P {spec.domain} "
            f"res={spec.resolution_mm} mm h_fea={args.h} mm"
        )
        mesh, implicit_report = build_piecewise_splitp_boundary_mesh(spec)
        stem = spec.stem or implicit_report["stem"]
        out_dir = args.out_dir or (_REPO_ROOT / "outputs" / "implicit")
        if args.export_boundary_stl:
            args.export_boundary_stl.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(args.export_boundary_stl))
            print(f"  boundary STL -> {args.export_boundary_stl}")

    fixed_mask, load_mask, bc_preview = detect_flat_top_bcs(
        mesh.vertices,
        mesh.faces,
        bc_load_mode=args.bc_load_mode,
        top_min_vertices=args.top_min_vertices,
    )
    print(
        f"  {len(mesh.faces):,} faces watertight={mesh.is_watertight} "
        f"volume={mesh.volume:.4f} mm³ "
        f"BC fixed={fixed_mask.sum():,} load={load_mask.sum():,}"
    )
    if bc_preview:
        print(
            f"  cap z={bc_preview.get('z_cap_mm')} floor z={bc_preview.get('z_floor_mm')}"
        )

    report = run_compression_fea_case(
        mesh,
        stl_path=args.stl,
        out_dir=out_dir,
        stem=stem,
        force_n=float(args.force_n),
        h=float(args.h),
        mesh_mode=mesh_mode,
        quality_mode=args.quality_mode,
        bc_load_mode=args.bc_load_mode,
        clean_stl=bool(args.clean_stl),
        solver=args.solver,
        fea_input_mode=fea_input_mode,
        implicit_report=implicit_report,
        top_min_vertices=int(args.top_min_vertices),
        youngs_modulus=float(args.youngs_modulus),
        write_viz=not args.no_viz,
    )
    mq = report["mesh_quality_report"]
    print(f"  max displacement: {report['max_displacement_mm']:.6f} mm")
    print(f"  max von Mises (nodal): {report['max_von_mises_nodal_MPa']:.4f} MPa")
    print(
        f"  mesh: {mq.get('n_poor_elements', 0):,} poor / "
        f"{mq.get('n_elements', 0):,} ({100 * mq.get('poor_fraction', 0):.2f}%)"
    )
    if mq.get("linear_solver"):
        print(f"  solver: {mq.get('linear_solver')} ({mq.get('linear_solve_sec', 0):.2f}s)")
    print(f"Wrote {report['report_path']}")
    print(f"Wrote {report['vtu_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
