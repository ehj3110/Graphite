#!/usr/bin/env python
"""
Remove floating STL islands, export cleaned STL, volume-mesh, and write diagnostics.

Run from repo root::

    python scripts/aristo_clean_and_mesh_stl.py --stl path/to/lattice.stl
    python scripts/aristo_clean_and_mesh_stl.py --stl ... --h 0.005
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.volume_mesh_inspect import (
    export_volume_mesh_inspection,
    remove_floating_islands,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--h", type=float, default=0.005)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-volume-mm3", type=float, default=1e-6)
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument(
        "--skip-volume-inspect",
        action="store_true",
        help="Skip gmsh volume-mesh QC (useful for coarse extrude STLs).",
    )
    args = parser.parse_args()

    if not args.stl.is_file():
        print(f"STL not found: {args.stl}")
        return 1

    out_dir = args.out_dir or args.stl.parent
    stem_in = args.stem or args.stl.stem

    mesh = trimesh.load_mesh(str(args.stl))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    print(
        f"Input {args.stl.name}: {len(mesh.faces):,} faces "
        f"watertight={mesh.is_watertight} volume={mesh.volume:.6f} mm³"
    )

    cleaned, island_meta = remove_floating_islands(
        mesh, min_volume_mm3=float(args.min_volume_mm3)
    )
    print(
        f"  islands: {island_meta['n_components_before']} -> "
        f"{island_meta['n_components_after']} "
        f"(removed {island_meta['n_islands_removed']})"
    )

    cleaned_stem = f"{stem_in}_cleaned"
    cleaned_stl = out_dir / f"{cleaned_stem}.stl"
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.export(cleaned_stl)
    print(f"  cleaned STL -> {cleaned_stl.name}")

    if args.skip_volume_inspect:
        print("  skipped volume mesh inspect (--skip-volume-inspect)")
        return 0

    try:
        report = export_volume_mesh_inspection(
            cleaned,
            h=float(args.h),
            out_dir=out_dir,
            stem=cleaned_stem,
            mesh_mode="single_surface",
            extra_report={
                "input_stl": str(args.stl),
                "cleaned_stl": str(cleaned_stl),
                "island_cleanup": island_meta,
            },
        )
    except Exception as exc:
        print(f"  volume mesh inspect failed ({exc}); cleaned STL kept.")
        return 0
    print(
        f"  mesh: tets={report['n_elements']:,} "
        f"poor={report.get('poor_fraction', 0)*100:.2f}%"
    )
    print(f"Wrote {report['report_json']}")
    print(f"Wrote {report['debug_vtu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
