#!/usr/bin/env python3
"""Generate full volumetric lattice (interior + exterior tets) with no surface cage."""

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


def generate_volumetric_no_skin() -> None:
    target = 5.0
    root = Path(__file__).resolve().parent
    stl_path = _find_adapter_stl(root)
    print(f"Loading {stl_path.name}...")

    mesh = trimesh.load(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    # 1. Generate Linear Scaffold
    print(f"Generating {target}mm Linear Scaffold...")
    scaffold = generate_conformal_scaffold(
        mesh_process, target_element_size=target, element_order=1
    )
    nodes = scaffold.nodes
    tets = scaffold.elements  # Use all tets (interior + exterior)

    # 2. Extract Topology (all tets, no surface cage)
    print("Extracting full volumetric topology (No surface dual)...")
    nodes_out, struts = generate_topology(
        nodes,
        elements=tets,
        surface_faces=np.empty((0, 3), dtype=int),
        type="rhombic",
        include_surface_cage=False,
        target_element_size=target,
        merge_short_struts=False,
    )

    # 3. Sweep and Export
    print(f"Sweeping {len(struts)} volumetric struts...")
    solid = generate_geometry(nodes_out, struts, strut_radius=0.3)

    output_name = "Adapter_5mm_Volumetric_NoSkin.stl"
    solid.export(str(root / output_name))
    print(f"Successfully exported {output_name} for slicer inspection!")


if __name__ == "__main__":
    generate_volumetric_no_skin()

