"""
Isotropic Open3D surface remesh for lattice/TPMS STL skins.

Eliminates needle triangles on flat cap faces before gmsh volume meshing.

Usage:
  python scripts/clean_stl_surface.py path/to/lattice.stl --h 0.15
  python scripts/clean_stl_surface.py path/to/lattice.stl -o path/to/cleaned.stl --h 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.stl_surface_clean import clean_stl_surface, clean_stl_surface_to_temp_stl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Isotropic Open3D surface remesh (target edge length h mm)"
    )
    parser.add_argument("input", type=Path, help="Input STL path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output STL (default: <input>_cleaned.stl)",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.15,
        help="Target surface edge length in mm (default: 0.15)",
    )
    parser.add_argument(
        "--temp",
        action="store_true",
        help="Write to a temporary STL and print its path",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Cap normal axis for plane preservation (default: 2 = Z)",
    )
    args = parser.parse_args()

    mesh = trimesh.load_mesh(str(args.input), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    print(
        f"Input  {args.input}: {len(mesh.faces):,} faces, "
        f"watertight={mesh.is_watertight}, "
        f"max_edge={float(mesh.edges_unique_length.max()) if len(mesh.edges_unique_length) else 0:.4f} mm"
    )

    if args.temp:
        out = clean_stl_surface_to_temp_stl(mesh, args.h, axis=args.axis)
    else:
        cleaned = clean_stl_surface(mesh, args.h, axis=args.axis)
        out = args.output or args.input.with_name(f"{args.input.stem}_cleaned.stl")
        out.parent.mkdir(parents=True, exist_ok=True)
        cleaned.export(str(out))

    result = trimesh.load_mesh(str(out), process=False)
    if not isinstance(result, trimesh.Trimesh):
        result = result.dump(concatenate=True)
    print(
        f"Output {out}: {len(result.faces):,} faces, "
        f"watertight={result.is_watertight}, "
        f"max_edge={float(result.edges_unique_length.max()) if len(result.edges_unique_length) else 0:.4f} mm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
