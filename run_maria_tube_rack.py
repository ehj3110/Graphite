"""
Mesh MariaTubeRack_Full.STL with Kagome lattice, Full-Pipe (no clipped boundaries).

Note: MariaTubeRack_Full.STL has overlapping facets. If GMSH fails, repair the mesh
in MeshLab/MeshMixer (Filters → Cleaning and Repairing → Remove Duplicate Faces,
Remove Zero-Area Faces) or export a clean STL from your CAD tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import trimesh

from solver import optimize_lattice_fraction

ELEMENT_SIZE = 5.0
TARGET_VF = 0.10


def main() -> None:
    stl_path = Path("test_parts/MariaTubeRack_Full.STL")
    if not stl_path.exists():
        raise FileNotFoundError(f"Not found: {stl_path}")

    boundary_mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if isinstance(boundary_mesh, trimesh.Scene):
        boundary_mesh = boundary_mesh.dump(concatenate=True)
    if not boundary_mesh.is_watertight:
        print("Attempting mesh repair...")
        boundary_mesh.fill_holes()
        boundary_mesh.remove_degenerate_faces()
        boundary_mesh.merge_vertices()
        boundary_mesh.fix_normals()
    if not boundary_mesh.is_watertight:
        print("Warning: mesh may not be watertight; GMSH may fail.")

    print(f"Processing: {stl_path}")
    print(f"Volume: {boundary_mesh.volume:.2f} mm³")
    print(f"Element size: {ELEMENT_SIZE} mm, Target Vf: {TARGET_VF:.0%}")

    try:
        result = optimize_lattice_fraction(
            mesh=boundary_mesh,
            target_vf=TARGET_VF,
            target_element_size=ELEMENT_SIZE,
            topology_type="kagome",
            include_surface_cage=True,
            clipped_boundary=False,
            algorithm_3d=10,  # HXT; try 4 (Netgen) or 1 (Delaunay) if it fails
        )
    except RuntimeError as e:
        if "scaffold" in str(e).lower():
            print("\n" + "=" * 60)
            print("MESH REPAIR NEEDED: The STL has overlapping/self-intersecting facets.")
            print("Repair in MeshLab: Filters → Cleaning and Repairing →")
            print("  Remove Duplicate Faces, Remove Zero-Area Faces")
            print("Or use MeshMixer / CAD export a clean watertight STL.")
            print("=" * 60)
        raise

    out_path = Path("test_parts/MariaTubeRack_Kagome_FullPipe.stl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(out_path))

    print(f"\nExported: {out_path}")
    print(f"Vf: {result.volume / boundary_mesh.volume:.4%}")


if __name__ == "__main__":
    main()
