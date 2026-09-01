#!/usr/bin/env python
"""
Generate piecewise-constant Split-P implicit lattices (hard Z bands).

Default: Ø2 mm × 2 mm cylinder, bottom L=0.5 mm, top L=1.0 mm @ 33% SF.
Uses **single-pass** full-cylinder EDT (recommended for Gmsh volume mesh).

Run from repo root::

    python scripts/generate_piecewise_splitp_implicit.py
    python scripts/generate_piecewise_splitp_implicit.py --resolution-mm 0.015 --out-dir outputs/implicit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.aristo.implicit_input import (
    AristoImplicitSpec,
    add_piecewise_implicit_args,
    build_piecewise_splitp_boundary_mesh,
    default_stem_from_spec,
    implicit_spec_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_piecewise_implicit_args(parser)
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "implicit")
    parser.add_argument(
        "--combine",
        choices=("single-pass",),
        default="single-pass",
        help="Only single-pass is supported in the canonical script.",
    )
    parser.add_argument("--stem", type=str, default=None)
    args = parser.parse_args()

    spec = implicit_spec_from_args(args)
    spec.stem = args.stem
    mesh, report = build_piecewise_splitp_boundary_mesh(spec)
    stem = spec.stem or default_stem_from_spec(spec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stl_path = args.out_dir / f"{stem}.stl"
    mesh.export(stl_path)
    report["stl"] = str(stl_path)
    report["combine"] = args.combine

    json_path = args.out_dir / f"{stem}_generation_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {stl_path} ({len(mesh.faces):,} faces, watertight={mesh.is_watertight})")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
