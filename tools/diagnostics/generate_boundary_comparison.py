"""
Generate Test_Clipped.stl and Test_FullPipe.stl — boundary mode comparison.

20mm cube, Kagome, 5mm element size, 10% Vf.
- Clipped: lattice intersected with boundary (flat faces at edges).
- FullPipe: raw cylinder union with Smart Inset (sits inside 20mm footprint).

Run from project root: python tools/diagnostics/generate_boundary_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh

from solver import optimize_lattice_fraction

_OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "tests"


def main() -> None:
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- CLIPPED (flat look) ---")
    result_clipped = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=0.10,
        target_element_size=5.0,
        topology_type="kagome",
        include_surface_cage=True,
        clipped_boundary=True,
    )
    result_clipped.mesh.export(str(_OUT_DIR / "Test_Clipped.stl"))
    print(f"Exported: {_OUT_DIR / 'Test_Clipped.stl'} (Vf: {result_clipped.volume / boundary_mesh.volume:.4%})")

    print("\n--- FULL-PIPE (pipe look, Smart Inset) ---")
    result_pipe = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=0.10,
        target_element_size=5.0,
        topology_type="kagome",
        include_surface_cage=True,
        clipped_boundary=False,
    )
    result_pipe.mesh.export(str(_OUT_DIR / "Test_FullPipe.stl"))
    print(f"Exported: {_OUT_DIR / 'Test_FullPipe.stl'} (Vf: {result_pipe.volume / boundary_mesh.volume:.4%})")

    print("\nDone. Compare: Clipped has flat faces; FullPipe has rounded tubular junctions.")


if __name__ == "__main__":
    main()
