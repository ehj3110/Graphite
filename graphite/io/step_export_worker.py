"""
Subprocess entry point for Gmsh STEP export.

Gmsh registers SIGINT handlers at import time, which fails outside Python's
main thread (e.g. Streamlit's ScriptRunner). The parent process invokes this
module via ``python -m graphite.io.step_export_worker``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graphite.io.mesh_export import StepExportOptions, gmsh_stl_file_to_step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Graphite mesh STL → STEP (Gmsh)")
    parser.add_argument("input_stl", type=Path, help="Input watertight STL")
    parser.add_argument("output_step", type=Path, help="Output STEP path")
    parser.add_argument(
        "--feature-angle-deg",
        type=float,
        default=45.0,
        help="Gmsh classifySurfaces feature angle (degrees)",
    )
    parser.add_argument(
        "--geometry-tolerance",
        type=float,
        default=None,
        help="Optional Gmsh Geometry.Tolerance",
    )
    args = parser.parse_args(argv)

    opts = StepExportOptions(
        feature_angle_deg=float(args.feature_angle_deg),
        geometry_tolerance=args.geometry_tolerance,
        repair_mesh=False,
        silent_gmsh=True,
    )
    notes = gmsh_stl_file_to_step(args.input_stl, args.output_step, opts)
    sys.stdout.write(json.dumps(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
