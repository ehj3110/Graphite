"""
Kagome Topology Visual Diagnostic (No Solver)

This script overlays:
  1) GMSH boundary triangles (semi-transparent gray)
  2) Kagome struts (internal blue, boundary/cage red)

Use this to visually detect any stray "shooting" struts.
"""

from __future__ import annotations

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from graphite.explicit.scaffold_module import generate_conformal_scaffold
from graphite.explicit.topology_module import generate_topology


def _match_boundary_face_centroid_nodes(
    topo_nodes: np.ndarray,
    scaffold_nodes: np.ndarray,
    surface_faces: np.ndarray,
    decimals: int = 5,
) -> np.ndarray:
    """
    Find topology-node indices that lie at boundary-face centroids.

    We compute centroids from the actual scaffold boundary triangles and then
    match them against topology nodes using rounded coordinate keys.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    boundary_face_centroids = np.mean(scaffold_nodes[surface_faces], axis=1)
    boundary_face_centroids = np.round(boundary_face_centroids, decimals=decimals)
    topo_nodes_rounded = np.round(topo_nodes, decimals=decimals)

    # Hashable row keys for fast vectorized set-style matching.
    c_keys = np.ascontiguousarray(boundary_face_centroids).view(
        np.dtype((np.void, boundary_face_centroids.dtype.itemsize * 3))
    )
    n_keys = np.ascontiguousarray(topo_nodes_rounded).view(
        np.dtype((np.void, topo_nodes_rounded.dtype.itemsize * 3))
    )
    boundary_mask = np.isin(n_keys, c_keys)
    return np.flatnonzero(boundary_mask)


def run_kagome_debug_viz() -> None:
    # -------------------------------------------------------------------------
    # 1) Build mock boundary and scaffold
    # -------------------------------------------------------------------------
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    target_element_size = 5.0
    strut_radius = 0.25  # Visualization-only fixed radius (not used by solver)

    # Local alias for the requested naming in this diagnostic.
    generate_scaffold = generate_conformal_scaffold
    scaffold = generate_scaffold(
        mesh=boundary_mesh,
        target_element_size=target_element_size,
    )

    # -------------------------------------------------------------------------
    # 2) Generate Kagome topology (surface cage enabled), no solver
    # -------------------------------------------------------------------------
    topo_nodes, struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        type="kagome",
        include_surface_cage=True,
    )

    # -------------------------------------------------------------------------
    # 3) Classify struts: boundary/cage (red) vs internal (blue)
    # -------------------------------------------------------------------------
    boundary_node_ids = _match_boundary_face_centroid_nodes(
        topo_nodes=topo_nodes,
        scaffold_nodes=scaffold.nodes,
        surface_faces=scaffold.surface_faces,
        decimals=5,
    )
    boundary_node_mask = np.zeros(topo_nodes.shape[0], dtype=bool)
    boundary_node_mask[boundary_node_ids] = True

    # A strut is classified as boundary/cage if both endpoints are boundary nodes.
    red_mask = boundary_node_mask[struts[:, 0]] & boundary_node_mask[struts[:, 1]]
    red_struts = struts[red_mask]
    blue_struts = struts[~red_mask]

    # -------------------------------------------------------------------------
    # 4) Plot overlay
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 1: GMSH boundary triangles (scaffold surface faces)
    tris_xyz = scaffold.nodes[scaffold.surface_faces]  # (K, 3, 3)
    tri_collection = Poly3DCollection(
        tris_xyz,
        facecolor=(0.85, 0.85, 0.85, 0.18),
        edgecolor=(0.65, 0.65, 0.65, 0.30),
        linewidths=0.25,
    )
    ax.add_collection3d(tri_collection)

    # Plot 2a: internal struts (blue)
    #for a, b in blue_struts:
   #     p0, p1 = topo_nodes[a], topo_nodes[b]
   #     ax.plot(
   #         [p0[0], p1[0]],
   #         [p0[1], p1[1]],
   #         [p0[2], p1[2]],
   #         color="blue",
   #         linewidth=0.6,
   #         alpha=0.55,
   #     )

    # Plot 2b: boundary/cage struts (red)
    for a, b in red_struts:
        p0, p1 = topo_nodes[a], topo_nodes[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="red",
            linewidth=1.2,
            alpha=0.9,
        )

    # Axis framing and styling
    mins = np.min(scaffold.nodes, axis=0)
    maxs = np.max(scaffold.nodes, axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        "Kagome Debug Overlay\n"
        f"target_element_size={target_element_size}, fixed strut_radius={strut_radius}"
    )

    print("=" * 60)
    print("Kagome Diagnostic Ready")
    print(f"Topology nodes:      {topo_nodes.shape[0]}")
    print(f"Total struts:        {struts.shape[0]}")
    print(f"Internal (blue):     {blue_struts.shape[0]}")
    print(f"Boundary/Cage (red): {red_struts.shape[0]}")
    print("=" * 60)

    # Interactive window
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_kagome_debug_viz()
