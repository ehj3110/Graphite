#!/usr/bin/env python3
"""
Regenerate explicit surface-dual lattice on repaired adapter/top part STL.

Goals:
  - ~20mm unit cell / tet size (target_element_size)
  - second-order (quadratic) tets for curved surface behavior
  - surface-dual cage enabled
  - solid fraction target ~15% via lightweight radius search
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from graphite.explicit import generate_conformal_scaffold, generate_geometry, generate_topology


def _find_repaired_stl(root: Path) -> Path:
    candidates = [
        root / "top_part_new_repaired._V2.stl",
        root / "test_parts" / "top_part_new_repaired._V2.stl",
        root / "test_parts" / "top_part_new_repaired._V2.STL",
        root / "test_parts" / "top_part_new_repaired._V2.stl",
    ]
    for p in candidates:
        if p.is_file():
            return p

    # Fallback: brute-force by substring
    for p in root.glob("**/*top_part_new_repaired*V2*.stl"):
        return p
    raise FileNotFoundError(f"Could not find repaired STL. Looked in: {candidates}")


def _estimate_solid_fraction(
    solid_mesh: trimesh.Trimesh | None,
    part_volume: float,
) -> float | None:
    if solid_mesh is None or part_volume <= 0:
        return None
    try:
        v = float(solid_mesh.volume)
    except Exception:
        return None
    return v / part_volume


def generate_on_repaired_top_part(
    target_element_size: float = 20.0,
    element_order: int = 2,
    solid_fraction_target: float = 0.15,
    radius_bounds: tuple[float, float] = (0.20, 0.60),
    radius_steps: int = 4,
) -> None:
    root = Path(__file__).resolve().parent
    stl_path = _find_repaired_stl(root)

    print(f"Loading repaired STL: {stl_path}")
    mesh = trimesh.load(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    if not mesh_process.is_watertight:
        print("Warning: repaired mesh is not watertight according to trimesh. GMSH may still fail.")

    part_volume = None
    try:
        part_volume = float(mesh_process.volume)
    except Exception:
        pass

    print(
        f"Input mesh volume: {part_volume:.3f} mm^3"
        if part_volume is not None
        else "Input mesh volume: (unavailable; trimesh may not compute volume)"
    )

    # 1) Scaffold
    print(f"\nGenerating conformal scaffold: target_element_size={target_element_size}mm, element_order={element_order}...")
    scaffold = generate_conformal_scaffold(
        mesh_process,
        target_element_size=target_element_size,
        element_order=element_order,
    )

    nodes = scaffold.nodes
    elements = scaffold.elements
    surface_faces = scaffold.surface_faces

    print(f"Scaffold: nodes={nodes.shape[0]} elements={elements.shape} surface_faces={surface_faces.shape}")

    # 2) Topology: "surface dual" cage enabled, internal rhombic for normal interior tets
    print("\nExtracting topology: rhombic interior + curved surface-dual cage...")
    nodes_out, struts = generate_topology(
        nodes,
        elements=elements,
        surface_faces=surface_faces,
        type="rhombic",
        include_surface_cage=True,
        target_element_size=target_element_size,
        merge_short_struts=False,
    )
    print(f"Topology: nodes_out={nodes_out.shape[0]} struts={struts.shape[0]}")

    # 3) Radius search to hit ~15% volume fraction
    # Geometry is expensive, so do a small bracket sweep and pick best.
    r_lo, r_hi = radius_bounds
    candidate_rs = np.linspace(r_lo, r_hi, radius_steps)

    best = None  # (abs_err, r, vol, frac)
    for r in candidate_rs:
        print(f"\nSweeping struts into solid @ strut_radius={r:.3f}mm (crop_to_boundary=True)...")
        # return_manifold avoids trimesh conversion overhead.
        geom = generate_geometry(
            nodes_out,
            struts,
            strut_radius=float(r),
            boundary_mesh=mesh_process,
            crop_to_boundary=True,
            return_manifold=True,
        )
        solid_manifold, vol = geom  # type: ignore[misc]
        frac = float(vol) / part_volume if part_volume is not None and part_volume > 0 else None
        print(f"  -> Lattice volume={vol:.3f} mm^3 | solid_fraction={frac:.4f}" if frac is not None else f"  -> Lattice volume={vol:.3f} mm^3 (solid fraction unavailable)")

        if frac is None:
            continue

        err = abs(frac - solid_fraction_target)
        if best is None or err < best[0]:
            best = (err, float(r), float(vol), float(frac))

    if best is None:
        print("\nCould not compute solid fraction (part volume unavailable). Using mid-radius fallback.")
        chosen_r = float(candidate_rs[len(candidate_rs) // 2])
    else:
        _, chosen_r, chosen_vol, chosen_frac = best
        print(f"\nBest radius: r={chosen_r:.3f}mm -> solid_fraction={chosen_frac:.4f} (target={solid_fraction_target:.4f})")

    # 4) Final solid generation with chosen radius + export
    print(f"\nGenerating final solid @ strut_radius={chosen_r:.3f}mm...")
    solid_geom = generate_geometry(
        nodes_out,
        struts,
        strut_radius=float(chosen_r),
        boundary_mesh=mesh_process,
        crop_to_boundary=True,
        return_manifold=False,
    )
    solid_mesh = solid_geom if not isinstance(solid_geom, tuple) else solid_geom[0]

    # Export
    out_name = f"{stl_path.stem}_ExplicitSurfaceDual_r{chosen_r:.3f}_cell{target_element_size:.1f}mm.stl"
    out_path = root / out_name
    solid_mesh.export(str(out_path))
    print(f"\nExported: {out_path.name}")


if __name__ == "__main__":
    generate_on_repaired_top_part()

