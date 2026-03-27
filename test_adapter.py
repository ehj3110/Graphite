#!/usr/bin/env python3
"""Verify Part2_Adapter STL and run full curved-skin solid CSG pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from graphite.explicit import (
    generate_conformal_scaffold,
    generate_geometry,
    generate_topology,
)

_ROOT = Path(__file__).resolve().parent
STL_CANDIDATES = (
    _ROOT / "Part2_Adapter.stl",
    _ROOT / "Part2_Adapter.STL",
    _ROOT / "test_parts" / "Part2_Adapter.stl",
    _ROOT / "test_parts" / "Part2_Adapter.STL",
)


def _find_adapter_stl() -> Path | None:
    for p in STL_CANDIDATES:
        if p.is_file():
            return p
    return None


def _print_mesh_health(name: str, mesh: trimesh.Trimesh) -> bool:
    """Print boundary sanity checks; return True if looks meshable."""
    wt = mesh.is_watertight
    vol = mesh.is_volume
    euler = mesh.euler_number
    winding = mesh.is_winding_consistent
    print(f"\n--- {name} ---")
    print(f"  Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}")
    print(f"  is_watertight:        {wt}")
    print(f"  is_volume:            {vol}")
    print(f"  euler_number:         {euler}  (2 = single sphere-like shell; other values"
          f" often mean multiple bodies, holes, or handles)")
    print(f"  is_winding_consistent: {winding}")
    # Watertight + volume is the main gate for GMSH; euler==2 is ideal for one simple solid.
    ok = bool(wt and vol and winding)
    if euler != 2:
        print(
            "  Note: euler != 2 is common for parts with voids or multiple shells; "
            "still OK if meshing succeeds."
        )
    if ok:
        print("  Boundary checks: OK for volumetric meshing (watertight + volume).")
    else:
        print("  Warning: mesh may still fail in GMSH (repair STL if meshing fails).")
    return ok


def test_adapter() -> None:
    target = 8.0
    path = _find_adapter_stl()
    if path is None:
        print(
            "Error: Could not find Part2_Adapter.stl. Tried:\n  "
            + "\n  ".join(str(p) for p in STL_CANDIDATES)
        )
        return

    print(f"Loading {path} and generating {target}mm scaffold...")

    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    _print_mesh_health("Part2_Adapter (after process=True)", mesh_process)

    try:
        scaffold = generate_conformal_scaffold(
            mesh_process,
            target_element_size=target,
            algorithm_3d=10,
            element_order=2,
        )
    except Exception as e:
        print(
            f"\nGMSH failed to mesh Part2_Adapter. The STL may have self-intersections "
            f"or non-manifold geometry.\nError: {e}"
        )
        return

    print("\nGMSH: mesh generation succeeded (watertight + meshable for this target size).")

    nodes = scaffold.nodes
    tets = scaffold.elements
    surface_faces = scaffold.surface_faces
    print(
        f"Scaffold element_order={scaffold.element_order} | "
        f"tet connectivity shape {tets.shape} (expect (N, 10) for quadratic)"
    )

    # Exterior mask built from linear corners only (surface_faces may include mids for P2).
    tets_corners = tets[:, :4] if tets.shape[1] > 4 else tets
    surface_corners = surface_faces[:, :3] if surface_faces.shape[1] > 3 else surface_faces
    boundary_nodes = np.unique(surface_corners)
    is_exterior = np.any(np.isin(tets_corners, boundary_nodes), axis=1)
    interior_tets = tets_corners[~is_exterior]

    print(f"Total Tets: {len(tets)} | Interior Tets: {len(interior_tets)}")

    if len(interior_tets) == 0:
        print("Error: The mesh is too thin for the target size to have a pure interior.")
        return

    print("Extracting Topology (Linear Core + Curved Skin)...")
    nodes_out, struts = generate_topology(
        nodes,
        tets,
        surface_faces,
        type="rhombic",
        include_surface_cage=True,
        target_element_size=target,
        merge_short_struts=False,
    )

    print(f"Sweeping {len(struts)} struts into solid geometry (this may take a moment)...")
    solid_mesh = generate_geometry(nodes_out, struts, strut_radius=0.3)
    output_filename = "Part2_Adapter_Conformal_Lattice.stl"
    print(f"Exporting solid lattice to {output_filename}...")
    solid_mesh.export(output_filename)

    pv_mesh = pv.wrap(solid_mesh)
    p = pv.Plotter(title="Part2_Adapter - Final Curved Solid Lattice")
    p.add_mesh(pv_mesh, color="gold", smooth_shading=True, specular=0.5)
    p.show()


if __name__ == "__main__":
    test_adapter()
