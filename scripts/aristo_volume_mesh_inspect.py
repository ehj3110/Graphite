#!/usr/bin/env python
"""
Volume-mesh an Aristo case (Gmsh Tet4) and export quality-gate JSON + debug VTU.

**Default input:** piecewise Split-P implicit field.
**Developer override:** ``--stl``.

Run from repo root::

    python scripts/aristo_volume_mesh_inspect.py --h 0.005
    python scripts/aristo_volume_mesh_inspect.py --stl path/to/lattice.stl --mesh-mode tpms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.implicit_input import (
    add_piecewise_implicit_args,
    add_stl_dev_arg,
    build_piecewise_splitp_boundary_mesh,
    implicit_spec_from_args,
)
from graphite.aristo.volume_mesh_inspect import export_volume_mesh_inspection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_stl_dev_arg(parser)
    add_piecewise_implicit_args(parser, default_h=0.12)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--mesh-mode",
        choices=("single_surface", "tpms"),
        default=None,
    )
    parser.add_argument("--clean-stl", action="store_true")
    parser.add_argument(
        "--allow-single-surface-fallback",
        action="store_true",
        help="Retry without classifySurfaces if tpms mesh fails (STL dev path).",
    )
    parser.add_argument("--stem", type=str, default=None)
    args = parser.parse_args()

    fea_input_mode = "stl" if args.stl is not None else "implicit"
    mesh_mode = args.mesh_mode or ("tpms" if fea_input_mode == "stl" else "single_surface")
    implicit_report = None

    if fea_input_mode == "stl":
        if not args.stl.is_file():
            print(f"STL not found: {args.stl}")
            return 1
        print(f"[dev STL] Loading {args.stl}")
        mesh = trimesh.load_mesh(str(args.stl))
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        stem = args.stem or args.stl.stem
        out_dir = args.out_dir or args.stl.parent
    else:
        spec = implicit_spec_from_args(args)
        if args.stem:
            spec.stem = args.stem
        print(f"[implicit] piecewise Split-P {spec.domain} h={args.h} mm")
        mesh, implicit_report = build_piecewise_splitp_boundary_mesh(spec)
        stem = spec.stem or implicit_report["stem"]
        out_dir = args.out_dir or (_REPO_ROOT / "outputs" / "implicit")

    print(
        f"  {len(mesh.faces):,} faces watertight={mesh.is_watertight} "
        f"volume={mesh.volume:.4f} mm³"
    )

    extra = {
        "fea_input_mode": fea_input_mode,
        "input_stl": str(args.stl) if args.stl else None,
        "implicit_generation": implicit_report,
    }
    report = export_volume_mesh_inspection(
        mesh,
        h=float(args.h),
        out_dir=out_dir,
        stem=stem,
        mesh_mode=mesh_mode,
        clean_stl=bool(args.clean_stl),
        allow_single_surface_fallback=bool(args.allow_single_surface_fallback),
        extra_report=extra,
    )
    gate = report.get("quality_gate_breakdown", {})
    print(
        f"  nodes={report['n_nodes']:,} tets={report['n_elements']:,} "
        f"poor={report.get('poor_fraction', 0)*100:.2f}% "
        f"max_edge={report.get('max_edge_mm', 0):.4f} mm"
    )
    pf = gate.get("primary_failure_among_poor", {})
    if pf:
        parts = [f"{k}={pf[k]:,}" for k in ("min_volume", "aspect_ratio", "max_edge") if pf.get(k)]
        if parts:
            print(f"  poor primary: {', '.join(parts)}")
    print(f"Wrote {report['report_json']}")
    print(f"Wrote {report['debug_vtu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
