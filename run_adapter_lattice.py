"""
Generate Kagome lattice for Part2_Adapter.STL — Full-Pipe, 15% solid fraction.

Uses one-shot solver with Newton refinement and Surface Dual cage.
Dual-path test: Path A (process=False) vs Path B (process=True) for watertight/Boolean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from solver import optimize_lattice_fraction

STL_PATH = Path("test_parts/Part2_Adapter.STL")
if not STL_PATH.exists():
    STL_PATH = Path("Part2_Adapter.STL")

ELEMENT_SIZE = 6.0
TARGET_VF = 0.15
OUTPUT_PATH = Path("Adapter_Kagome_FullPipe_15pct.stl")


def _mesh_diagnostics(mesh: trimesh.Trimesh, label: str) -> None:
    """Print is_watertight, is_volume, bounds for a mesh."""
    try:
        is_vol = mesh.is_volume
    except Exception:
        is_vol = "N/A"
    print(f"  {label}: is_watertight={mesh.is_watertight}, is_volume={is_vol}")
    print(f"         bounds={mesh.bounds.tolist()}")
    print(f"         center={mesh.centroid.tolist()}")


def main() -> None:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"Not found: {STL_PATH}")

    # Path A: process=False (original CAD coordinates)
    mesh_a = trimesh.load(str(STL_PATH), force="mesh", process=False)
    if isinstance(mesh_a, trimesh.Scene):
        mesh_a = mesh_a.dump(concatenate=True)

    # Path B: process=True (may fix watertight, may shift center)
    mesh_b = trimesh.load(str(STL_PATH), force="mesh", process=True)
    if isinstance(mesh_b, trimesh.Scene):
        mesh_b = mesh_b.dump(concatenate=True)

    print("=" * 60)
    print("Dual-Path Load Test")
    print("=" * 60)
    _mesh_diagnostics(mesh_a, "Path A (process=False)")
    _mesh_diagnostics(mesh_b, "Path B (process=True)")

    # Re-align Path B to original CAD coordinates if center shifted
    center_a = np.asarray(mesh_a.centroid)
    center_b = np.asarray(mesh_b.centroid)
    offset = center_a - center_b
    if np.linalg.norm(offset) > 1e-6:
        print(f"\n  Re-aligning Path B: offset={offset.tolist()}")
        mesh_b.apply_translation(offset)
        _mesh_diagnostics(mesh_b, "Path B (after re-align)")
    else:
        print("\n  Path B center matches Path A, no re-align needed.")

    # Use Path B if watertight; otherwise fall back to Path A
    if mesh_b.is_watertight:
        boundary_mesh = mesh_b
        print("\n  Using Path B (process=True, watertight=True)")
    else:
        boundary_mesh = mesh_a
        print("\n  Using Path A (process=False) — Path B not watertight")

    print("\n" + "=" * 60)
    print("Part2_Adapter — Kagome Lattice (Full-Pipe)")
    print("=" * 60)
    print(f"STL: {STL_PATH}")
    print(f"Element size: {ELEMENT_SIZE} mm")
    print(f"Target Vf: {TARGET_VF:.0%}")
    print(f"Topology: kagome, Surface Dual cage, clipped_boundary=True")
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

    print(f"Exported: {OUTPUT_PATH}")
    print(f"Final strut radius: {result.radius:.6f} mm")
    print(f"Achieved Vf: {result.volume / boundary_mesh.volume:.4%}")


if __name__ == "__main__":
    main()
