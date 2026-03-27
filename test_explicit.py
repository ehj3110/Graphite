#!/usr/bin/env python3
"""
Debug explicit lattice pipeline step-by-step (GMSH -> topology wireframe -> PyVista).

Set PYVISTA_OFF_SCREEN=true for non-interactive runs (e.g. CI).
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh

from graphite.explicit import generate_conformal_scaffold, generate_topology

# ---------------------------------------------------------------------------
# 1. Generate Boundary
# ---------------------------------------------------------------------------
print("\n--- STEP 1: BOUNDARY ---")
cube = trimesh.creation.box(extents=[20, 20, 20])
cube_process = trimesh.Trimesh(vertices=cube.vertices, faces=cube.faces, process=True)
print(f"Boundary bounds: {cube_process.bounds}")

# ---------------------------------------------------------------------------
# 2. GMSH Scaffold
# ---------------------------------------------------------------------------
print("\n--- STEP 2: GMSH SCAFFOLD ---")
target_element_size = 2.5
scaffold = generate_conformal_scaffold(cube_process, target_element_size=target_element_size)
nodes = scaffold.nodes
tets = scaffold.elements
surface_faces = scaffold.surface_faces
print(f"Scaffold Nodes: {nodes.shape}")
print(f"Scaffold Tets: {tets.shape}")
print(f"Scaffold Surface Faces: {surface_faces.shape}")

# Optional Visual: Did GMSH work? Uncomment to see point cloud.
# pv.plot(nodes, title="GMSH Nodes", render_points_as_spheres=True, point_size=5)

# ---------------------------------------------------------------------------
# 3. Topology (Graph Extraction)
# ---------------------------------------------------------------------------
print("\n--- STEP 3: TOPOLOGY RULES ---")
rule_type = "kagome"
include_cage = True
nodes_out, struts = generate_topology(
    nodes,
    tets,
    surface_faces,
    type=rule_type,
    include_surface_cage=include_cage,
    target_element_size=target_element_size,
)
print(f"Final Nodes: {nodes_out.shape}")
print(f"Final Struts: {struts.shape}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC: degree per node (endpoints in strut list)
# ---------------------------------------------------------------------------
print("\n--- DIAGNOSTIC: NODE DEGREE ---")
unique, counts = np.unique(struts.ravel(), return_counts=True)
max_connections_idx = int(np.argmax(counts))
hub_node = int(unique[max_connections_idx])
hub_deg = int(counts[max_connections_idx])
# Each strut contributes one hit per endpoint on ravel → count == graph degree at that node.
print(f"Node {hub_node} has the highest degree: {hub_deg} incident struts")
if hub_deg > 50:
    print("WARNING: Starburst anomaly suspected — a single node has very high degree.")

print(f"Degree stats: min={counts.min()}, max={counts.max()}, mean={counts.mean():.2f}")
print(
    "\nNOTE: If max degree >> 50 with merge_short_struts enabled, _merge_short_struts "
    "was likely chaining ~100k 'short' struts into one union-find component (starburst). "
    "generate_topology(..., merge_short_struts=False) avoids this (current default)."
)

# ---------------------------------------------------------------------------
# 4. Raw Wireframe Visualization (Bypass Manifold3D)
# ---------------------------------------------------------------------------
print("\n--- STEP 4: WIREFRAME VISUALIZATION ---")
# VTK line cells: [2, i, j, 2, i, j, ...]
n_lines = len(struts)
cells = np.empty(n_lines * 3, dtype=np.int32)
cells[0::3] = 2
cells[1::3] = struts[:, 0]
cells[2::3] = struts[:, 1]

wireframe = pv.PolyData(nodes_out)
wireframe.lines = cells

p = pv.Plotter(title="Raw Topology Wireframe (Kagome + Surface Cage)")
p.add_mesh(wireframe, color="black", line_width=2, render_lines_as_tubes=False)
p.add_points(
    nodes_out,
    color="red",
    point_size=5,
    render_points_as_spheres=True,
)
p.show()
