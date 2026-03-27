#!/usr/bin/env python3
"""Tetrahedral scaffold quality: global / interior / exterior tet diagnostics."""

import numpy as np
import trimesh

from graphite.explicit import generate_conformal_scaffold


def print_stats(name, edges, aspect_ratios, count):
    if count == 0:
        print(f"\n--- {name} Tets: 0 (Mesh is entirely surface/too small) ---")
        return

    print(f"\n--- {name} Tets (Count: {count}) ---")
    print(f"Mean Edge:       {np.mean(edges):.3f} mm")
    print(f"Min Edge:        {np.min(edges):.3f} mm")
    print(f"Max Edge:        {np.max(edges):.3f} mm")
    print(f"Std Deviation:   {np.std(edges):.3f} mm")
    print(f"Mean Aspect:     {np.mean(aspect_ratios):.3f}")
    print(f"Max Aspect:      {np.max(aspect_ratios):.3f} (Worst)")
    print(f"Min Aspect:      {np.min(aspect_ratios):.3f} (Best)")


def analyze_tet_mesh():
    target = 2.5
    print(f"Generating 20mm cube with Target Element Size: {target}mm...")

    cube = trimesh.creation.box(extents=[20, 20, 20])
    cube_process = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces, process=True)

    scaffold = generate_conformal_scaffold(cube_process, target_element_size=target)
    nodes = scaffold.nodes
    tets = scaffold.elements
    surface_faces = scaffold.surface_faces

    # 1. Identify Boundary Nodes
    boundary_nodes = np.unique(surface_faces)

    # 2. Mask Tets: If any of a tet's 4 nodes are in the boundary_nodes array, it is Exterior.
    # np.isin returns a boolean array of shape (num_tets, 4). np.any(axis=1) collapses it.
    is_exterior = np.any(np.isin(tets, boundary_nodes), axis=1)
    is_interior = ~is_exterior

    # Extract vertices for all tets
    p0, p1, p2, p3 = nodes[tets[:, 0]], nodes[tets[:, 1]], nodes[tets[:, 2]], nodes[tets[:, 3]]

    # Calculate all 6 edge lengths
    e1, e2, e3 = np.linalg.norm(p1 - p0, axis=1), np.linalg.norm(p2 - p0, axis=1), np.linalg.norm(
        p3 - p0, axis=1
    )
    e4, e5, e6 = np.linalg.norm(p2 - p1, axis=1), np.linalg.norm(p3 - p1, axis=1), np.linalg.norm(
        p3 - p2, axis=1
    )

    all_edges = np.vstack((e1, e2, e3, e4, e5, e6))  # Shape: (6, num_tets)

    max_edge_per_tet = np.max(all_edges, axis=0)
    min_edge_per_tet = np.min(all_edges, axis=0)
    aspect_ratios = max_edge_per_tet / min_edge_per_tet

    print("\n=== TETRAHEDRAL MESH DIAGNOSTIC REPORT ===")
    print(f"Target Size: {target} mm | Total Nodes: {len(nodes)} | Total Tets: {len(tets)}")

    print_stats("Global", all_edges, aspect_ratios, len(tets))
    print_stats("Interior", all_edges[:, is_interior], aspect_ratios[is_interior], np.sum(is_interior))
    print_stats("Exterior", all_edges[:, is_exterior], aspect_ratios[is_exterior], np.sum(is_exterior))

    print("==========================================\n")


if __name__ == "__main__":
    analyze_tet_mesh()
