#!/usr/bin/env python3
"""
scripts/pam_gallery.py

Headless Scientific Visualization & Gallery Generator for Graphite Interlinked / PAMs.
Produces publication-grade, matte-shaded figures and cross-sectional cutaways of polycatenated metamaterials.

Key Features:
- Non-shiny, matte scientific shading (specular=0.12, ambient=0.40, diffuse=0.65).
- High-contrast color palette distinguishing crystallographic sublattices (Sublattice A vs B).
- Longitudinal cutaway clipping revealing internal interlocking chains without severed struts.
- Exports deliverables to outputs/ with automatic metadata reporting.

Usage:
  python scripts/pam_gallery.py --cell d4tet
  python scripts/pam_gallery.py --cell d4tet --cutaway
  python scripts/pam_gallery.py --all
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence
import numpy as np
import pyvista as pv
import trimesh

# Workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.interlinked import (
    InterlinkedRegistry,
    InterlinkedConfig,
    generate_interlinked_lattice,
    InterlinkedParticle,
)
from graphite.explicit.interlinked.generator import solidify_particle_prototype
from graphite.explicit.interlinked.dimensioning import PAMDimensioner
from graphite.explicit.interlinked.viz import (
    MATTE_PALETTES,
    render_matte_gallery_specimen,
)

OUTPUTS_DIR = WORKSPACE_ROOT / "outputs"

GALLERY_CELLS = [
    {"cell": "d4tet",  "name": "D-4-TET",  "seeding": "diamond",   "pitch": 14.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Diamond Tetrahedron (dia, bipartite A/B dual)"},
    {"cell": "c6tt",   "name": "C-6-TT",   "seeding": "cartesian", "pitch": 10.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Truncated Tetrahedron on Simple Cubic (pcu)"},
    {"cell": "d4cube", "name": "D-4-CUBE", "seeding": "diamond",   "pitch": 14.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Diamond Cube Wireframe (dia, bipartite A/B dual)"},
    {"cell": "b8co",   "name": "B-8-CO",   "seeding": "bcc",       "pitch": 16.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Cuboctahedron on BCC (bcc, corner+body dual)"},
    {"cell": "j4oct",  "name": "J-4-OCT",  "seeding": "cartesian", "pitch": 10.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Square Planar Octahedron Cross (45° relative twist)"},
    {"cell": "j4ring", "name": "J-4-RING", "seeding": "diamond",   "pitch": 12.0, "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Planar Ring on Diamond (dia, 4-in-1 maille basis)"},
    {"cell": "t6ring", "name": "T-6-RING", "seeding": "cartesian", "pitch": 8.0,  "wire_r": 0.35, "grid": (2, 2, 2), "desc": "Hexagonal Ring on Simple Cubic (pcu, 6-in-1 weave)"},
]


def run_gallery(
    cells: list[dict],
    cutaway: bool = False,
    theme: str = "light",
    output_dir: Path = OUTPUTS_DIR,
) -> None:
    """Generate renders for specified PAM cells."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Generating PAM Gallery ({'Cutaway' if cutaway else 'Full'}, Theme: {theme}) ===")

    for spec in cells:
        c_tag = spec["cell"]
        c_name = spec["name"]
        print(f"Processing {c_name} ({c_tag})...")

        cfg = InterlinkedConfig(
            cell=c_tag,
            seeding_type=spec.get("seeding", "cartesian"),
            grid_size=spec["grid"],
            pitch=spec["pitch"],
            wire_radius=spec["wire_r"],
        )
        res = generate_interlinked_lattice(cfg, check_clearance=False)

        # Dimensioning diagnostics
        dim = PAMDimensioner.solve(
            cell=c_tag,
            pitch=spec["pitch"],
            wire_diameter=2.0 * spec["wire_r"],
            warning_clearance=0.20,
            raise_on_jamming=False,
        )

        suffix = "_cutaway" if cutaway else ""
        out_png = output_dir / f"pam_gallery_{c_tag}{suffix}.png"

        subtitle = (
            f"Pitch: {dim.pitch:.2f} mm | Cage Dia: {dim.cage_diameter:.2f} mm | Strut: {dim.wire_diameter*1000.0:.0f} um | "
            f"Clearance: {dim.clearance*1000.0:.1f} um | Vf: {dim.solid_fraction_percent:.1f}% | Cages: {len(res.particles)}"
        )

        render_matte_gallery_specimen(
            particles=res.particles,
            out_png=out_png,
            title=f"{c_name}: {spec['desc']}",
            subtitle=subtitle,
            cutaway=cutaway,
            theme=theme,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Headless Matte Scientific Visualization Generator for PAMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cell", type=str, default=None, help="Target cell tag (e.g. d4tet, c6tt, d4cube)")
    parser.add_argument("--all", action="store_true", help="Render all curated gallery cells")
    parser.add_argument("--cutaway", action="store_true", help="Render longitudinal cutaway cross-section")
    parser.add_argument("--theme", type=str, default="light", choices=["light", "neutral", "dark"], help="Visual color theme")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR), help="Output directory for PNG renders")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.all:
        target_cells = GALLERY_CELLS
    elif args.cell:
        matches = [s for s in GALLERY_CELLS if s["cell"].lower() == args.cell.lower()]
        if matches:
            target_cells = matches
        else:
            # Fallback dynamic cell
            target_cells = [{
                "cell": args.cell,
                "name": args.cell.upper(),
                "pitch": 12.0,
                "wire_r": 0.35,
                "grid": (2, 2, 2),
                "desc": f"Registered PAM Cell ({args.cell})",
            }]
    else:
        # Default to d4tet and c6tt
        target_cells = GALLERY_CELLS[:2]

    run_gallery(target_cells, cutaway=args.cutaway, theme=args.theme, output_dir=out_dir)


if __name__ == "__main__":
    main()
