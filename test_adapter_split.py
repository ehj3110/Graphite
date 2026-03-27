#!/usr/bin/env python3
"""
Split Adapter into two isolated meshes:
1) Curved surface dual / hinged surface cage only
2) Linear interior core only

Exports:
  - Adapter_5mm_SurfaceOnly.stl
  - Adapter_5mm_InteriorOnly.stl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from graphite.explicit import generate_conformal_scaffold, generate_geometry, generate_topology


def _find_adapter_stl(root: Path) -> Path:
    candidates = (
        root / "Part2_Adapter.stl",
        root / "Part2_Adapter.STL",
        root / "test_parts" / "Part2_Adapter.stl",
        root / "test_parts" / "Part2_Adapter.STL",
    )
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Part2_Adapter.stl not found. Tried: {candidates}")


def generate_split_meshes() -> None:
    target = 5.0
    root = Path(__file__).resolve().parent
    stl_path = _find_adapter_stl(root)

    print(f"Loading {stl_path.name} for Split Mesh Generation...")

    mesh = trimesh.load(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    # 1) Generate Second-Order Scaffold
    print(f"Generating {target}mm Second-Order Scaffold...")
    scaffold = generate_conformal_scaffold(
        mesh_process,
        target_element_size=target,
        element_order=2,
    )

    nodes = scaffold.nodes
    tets = scaffold.elements
    surface_faces = scaffold.surface_faces

    # 2) Isolate Interior Tets (using only the 4 corner nodes)
    #    This determines "interior vs exterior" regardless of curved mids.
    boundary_nodes = np.unique(surface_faces)
    tets_corners = tets[:, :4] if tets.shape[1] >= 4 else tets
    is_exterior = np.any(np.isin(tets_corners, boundary_nodes), axis=1)
    interior_tets_corners = tets_corners[~is_exterior]

    # ==========================================
    # MESH 1: SURFACE CAGE ONLY
    # ==========================================
    print("\n--- Generating MESH 1: Surface Dual Only ---")
    nodes_surf, struts_surf = generate_topology(
        nodes,
        elements=np.empty((0, 4), dtype=int),
        surface_faces=surface_faces,
        type="rhombic",
        include_surface_cage=True,
        target_element_size=target,
        merge_short_struts=False,
    )

    print(f"Sweeping {len(struts_surf)} surface struts...")
    solid_surf = generate_geometry(nodes_surf, struts_surf, strut_radius=0.3)
    surf_output = root / "Adapter_5mm_SurfaceOnly.stl"
    solid_surf.export(str(surf_output))
    print(f"Exported: {surf_output.name}")

    # ==========================================
    # MESH 2: INTERIOR CORE ONLY
    # ==========================================
    print("\n--- Generating MESH 2: Interior Core Only ---")
    nodes_int, struts_int = generate_topology(
        nodes,
        elements=interior_tets_corners,
        surface_faces=np.empty((0, 3), dtype=int),
        type="rhombic",
        include_surface_cage=False,
        target_element_size=target,
        merge_short_struts=False,
    )

    print(f"Sweeping {len(struts_int)} interior struts...")
    solid_int = generate_geometry(nodes_int, struts_int, strut_radius=0.3)
    int_output = root / "Adapter_5mm_InteriorOnly.stl"
    solid_int.export(str(int_output))
    print(f"Exported: {int_output.name}")

    print("\nDone! Both STLs are ready for inspection.")


if __name__ == "__main__":
    generate_split_meshes()

