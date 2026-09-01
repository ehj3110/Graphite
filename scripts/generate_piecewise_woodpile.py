#!/usr/bin/env python
"""
Generate piecewise woodpile / cross-hatch lattices (Z control-surface bands).

Documentation: ``docs/PIECEWISE_PRISM_LATTICE_GENERATION.md`` (§8).
Extrusion backend: ``docs/EXPLICIT_WOODPILE_EXTRUSION.md``.

Default: Ø2 mm cylinder, **cross-hatch**, **extrude** backend, three discrete bands @ ~50% SF:
  z=[0, 2.23] mm pore 800 µm | z=[2.23, 2.31] mm pore 400 µm | z=[2.31, 2.7] mm pore 200 µm

Run from repo root::

    python scripts/generate_piecewise_woodpile.py
    python scripts/generate_piecewise_woodpile.py --generator implicit --resolution-mm 0.015
    python scripts/generate_piecewise_woodpile.py --domain box --width-x-mm 1 --depth-y-mm 1 --height-mm 1 `
        --z-breaks-mm 0,0.5,1.0 --pore-mm 0.1386,0.2771
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.woodpile_input import (
    add_piecewise_woodpile_args,
    build_piecewise_woodpile_mesh,
    default_stem_from_spec,
    woodpile_spec_from_args,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_piecewise_woodpile_args(parser)
    parser.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "outputs" / "models")
    parser.add_argument("--stem", type=str, default=None)
    args = parser.parse_args()

    spec = woodpile_spec_from_args(args)
    spec.stem = args.stem
    mesh, report = build_piecewise_woodpile_mesh(spec)
    stem = spec.stem or default_stem_from_spec(spec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stl_path = args.out_dir / f"{stem}.stl"
    mesh.export(stl_path)
    report["stl"] = str(stl_path)

    json_path = args.out_dir / f"{stem}_generation_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    txt_path = args.out_dir / f"{stem}_report.txt"
    watertight = report.get("watertight_union_after_repair", report.get("watertight"))
    txt_path.write_text(
        "\n".join(
            [
                "piecewise woodpile",
                f"generator={spec.generator}",
                f"domain={spec.domain}",
                f"diameter_mm={spec.diameter_mm}",
                f"height_mm={report.get('height_mm', spec.z_height_mm)}",
                f"combine_mode={report.get('combine_method', spec.combine_mode)}",
                f"true_woodpile={spec.true_woodpile}",
                f"resolution_mm={spec.resolution_mm}",
                f"watertight={watertight}",
                f"faces={report['faces']}",
                f"stl={stl_path}",
            ]
        ),
        encoding="utf-8",
    )

    print(
        f"Wrote {stl_path} ({len(mesh.faces):,} faces, "
        f"generator={spec.generator}, watertight={watertight})"
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
