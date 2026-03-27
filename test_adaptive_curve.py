#!/usr/bin/env python3
"""
Adaptive Boundary Projection
-----------------------------
Build a fast linear surface strut graph and locally curve struts by projecting
their intermediate points onto the original CAD surface (STL).

Outputs: interactive PyVista wireframe showing:
  - adaptive curved red struts
  - translucent original CAD mesh
"""

from __future__ import annotations

import numpy as np
import trimesh
import pyvista as pv

from pathlib import Path

from graphite.explicit import generate_conformal_scaffold, generate_geometry, generate_topology


def adaptive_surface_projection(
    nodes: np.ndarray,
    struts: np.ndarray,
    original_mesh: trimesh.Trimesh,
    segments: int = 4,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes linear surface struts, checks if they deviate from the CAD surface,
    and snaps them to the curve if they do.
    """
    print(f"Checking {len(struts)} surface struts for CAD deviation...")
    if len(struts) == 0:
        return nodes, struts

    new_nodes: list[np.ndarray] = list(nodes)
    new_struts: list[list[int]] = []

    # Pre-calculate start and end points of all struts
    p_start = nodes[struts[:, 0]]
    p_end = nodes[struts[:, 1]]

    # Calculate the exact midpoint of every straight strut
    midpoints = (p_start + p_end) / 2.0

    # Ask trimesh how far the straight midpoint is from the true CAD surface
    closest_pts, distances, _ = trimesh.proximity.closest_point(
        original_mesh, midpoints
    )

    # Find struts that deviate by more than the threshold (e.g. cutting through a hole)
    needs_curve = distances > threshold
    print(f"Found {int(np.sum(needs_curve))} struts that cut corners. Bending them...")

    # Process each strut
    for i, (n1, n2) in enumerate(struts):
        if needs_curve[i]:
            # This strut cuts a corner! Subdivide it into a smooth curve.
            line_pts = np.linspace(nodes[n1], nodes[n2], segments + 1)

            # Project the internal points onto the true CAD surface
            projected_pts, _, _ = trimesh.proximity.closest_point(
                original_mesh, line_pts[1:-1]
            )

            # Add these new curved points to our node list
            start_idx = len(new_nodes)
            new_nodes.extend(projected_pts)

            # Chain them together
            chain = [int(n1)] + list(range(start_idx, start_idx + len(projected_pts))) + [int(n2)]
            for j in range(len(chain) - 1):
                new_struts.append([chain[j], chain[j + 1]])
        else:
            # It's flat enough, keep the single fast strut
            new_struts.append([int(n1), int(n2)])

    return np.asarray(new_nodes, dtype=np.float64), np.asarray(new_struts, dtype=np.int64)


def test_adaptive_mesh() -> None:
    target = 6.0  # Use a large target so the "corner cutting" is obvious
    file_name = "Part2_Adapter.stl"

    root = Path(__file__).resolve().parent
    candidates = (
        root / file_name,
        root / file_name.upper(),
        root / "test_parts" / file_name,
        root / "test_parts" / file_name.upper(),
    )
    stl_path: Path | None = None
    for p in candidates:
        if p.is_file():
            stl_path = p
            break
    if stl_path is None:
        raise FileNotFoundError(f"Could not find {file_name}. Tried: {candidates}")

    # 1. Load the true CAD geometry
    mesh = trimesh.load(str(stl_path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    # 2. Generate FAST LINEAR Scaffold (element_order=1)
    print("Generating Linear Scaffold...")
    scaffold = generate_conformal_scaffold(
        mesh_process,
        target_element_size=target,
        element_order=1,
    )
    nodes, tets, surface_faces = scaffold.nodes, scaffold.elements, scaffold.surface_faces

    # 3. Extract Surface Struts Only (surface cage)
    nodes_surf, struts_surf = generate_topology(
        nodes,
        np.empty((0, 4), dtype=int),
        surface_faces,
        type="rhombic",
        include_surface_cage=True,
        target_element_size=target,
        merge_short_struts=False,
    )

    # 4. ADAPTIVE PROJECTION (The Magic)
    curved_nodes, curved_struts = adaptive_surface_projection(
        nodes_surf,
        struts_surf,
        mesh_process,
        segments=5,
        threshold=0.1,
    )

    print(f"Sweeping {len(curved_struts)} adaptive struts into solid geometry...")
    solid_adaptive = generate_geometry(curved_nodes, curved_struts, strut_radius=0.3)
    adaptive_output = "Adapter_6mm_AdaptiveSurface.stl"
    solid_adaptive.export(adaptive_output)
    print(f"Successfully exported {adaptive_output} for slicer inspection!")

    # 5. Plot the result!
    print("Plotting the Adaptively Curved Wireframe...")
    lines = np.empty((len(curved_struts), 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = curved_struts[:, 0]
    lines[:, 2] = curved_struts[:, 1]

    wireframe = pv.PolyData(curved_nodes)
    wireframe.lines = lines

    p = pv.Plotter(title="Adaptive Boundary Projection (Multi-Segment Curves)")
    p.add_mesh(wireframe, color="red", line_width=2)
    p.add_mesh(pv.wrap(mesh_process), color="white", opacity=0.3)
    p.show()


if __name__ == "__main__":
    test_adaptive_mesh()

