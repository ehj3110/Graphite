"""
Generate Kagome lattice for MariaTubeRack_Full.STL — Production pipeline.

Production settings:
  - Load: process=True (Path B) for watertight Boolean
  - Topology: kagome
  - Element size: 6.0 mm (fine mesh)
  - Adaptive radius: r = L·k
  - Clipped boundary: True (precision fitting)
  - return_manifold=True for solver efficiency (<1s per iteration)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from solver import optimize_lattice_fraction

STL_PATH = Path("test_parts/MariaTubeRack_Full.STL")
if not STL_PATH.exists():
    STL_PATH = Path("MariaTubeRack_Full.STL")

ELEMENT_SIZE = 6.0
TARGET_VF = 0.15
OUTPUT_PATH = Path("MariaTubeRack_Kagome_15pct.stl")


def main() -> None:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"Not found: {STL_PATH}")

    # Production: Path B (process=True) for watertight Boolean
    boundary_mesh = trimesh.load(str(STL_PATH), force="mesh", process=True)
    if isinstance(boundary_mesh, trimesh.Scene):
        boundary_mesh = boundary_mesh.dump(concatenate=True)

    print("=" * 60)
    print("MariaTubeRack_Full — Production Kagome Lattice")
    print("=" * 60)
    print(f"STL: {STL_PATH}")
    print(f"Vertices: {len(boundary_mesh.vertices)}, Faces: {len(boundary_mesh.faces)}")
    print(f"is_watertight: {boundary_mesh.is_watertight}, is_volume: {boundary_mesh.is_volume}")
    print(f"Bounds: {boundary_mesh.bounds.tolist()}")
    print(f"Element size: {ELEMENT_SIZE} mm")
    print(f"Target Vf: {TARGET_VF:.0%}")
    print(f"Topology: kagome, clipped_boundary=True")
    print()

    result = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=TARGET_VF,
        target_element_size=ELEMENT_SIZE,
        topology_type="kagome",
        include_surface_cage=True,
        clipped_boundary=True,
    )

    result.mesh.export(str(OUTPUT_PATH))

    print(f"\nExported: {OUTPUT_PATH}")
    print(f"Final strut radius: {result.radius:.6f} mm")
    print(f"Achieved Vf: {result.volume / boundary_mesh.volume:.4%}")


if __name__ == "__main__":
    main()
