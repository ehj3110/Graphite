"""
Boundary-Only Kagome Diagnostic Visualization

This script isolates only boundary struts from a Kagome topology and overlays
them on the GMSH surface mesh to diagnose coordinate mapping issues.
"""

from __future__ import annotations

import numpy as np
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from graphite.explicit.scaffold_module import generate_conformal_scaffold
from graphite.explicit.topology_module import generate_topology


def _boundary_node_mask(nodes: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """
    Return a boolean mask indicating whether each node lies on the box boundary.

    A node is considered on-boundary if any coordinate is within eps of the min
    or max value for that axis.
    """
    mins = np.min(nodes, axis=0)
    maxs = np.max(nodes, axis=0)

    near_min = np.abs(nodes - mins) <= eps
    near_max = np.abs(nodes - maxs) <= eps
    return np.any(near_min | near_max, axis=1)


def run_boundary_debug() -> None:
    # -------------------------------------------------------------------------
    # 1) Generate scaffold + Kagome topology (no solver)
    # -------------------------------------------------------------------------
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    target_element_size = 5.0

    # Local alias matching requested naming.
    generate_scaffold = generate_conformal_scaffold
    scaffold = generate_scaffold(
        mesh=boundary_mesh,
        target_element_size=target_element_size,
    )

    topo_nodes, struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        type="kagome",
        include_surface_cage=True,
    )

    # -------------------------------------------------------------------------
    # 2) Keep only struts with BOTH endpoints on boundary
    # -------------------------------------------------------------------------
    eps = 0.01
    on_boundary = _boundary_node_mask(topo_nodes, eps=eps)
    keep = on_boundary[struts[:, 0]] & on_boundary[struts[:, 1]]
    boundary_struts = struts[keep]

    # -------------------------------------------------------------------------
    # 3) 3D overlay: GMSH surface + boundary Kagome skeleton
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot 1: GMSH boundary triangles
    tris_xyz = scaffold.nodes[scaffold.surface_faces]
    tri_collection = Poly3DCollection(
        tris_xyz,
        facecolor=(0.55, 0.75, 1.0, 0.20),  # light blue, alpha=0.2
        edgecolor=(0.0, 0.0, 1.0, 1.0),     # blue edges
        linewidths=1.0,
    )
    ax.add_collection3d(tri_collection)

    # Plot 2: Boundary-only Kagome struts (bright red)
    for a, b in boundary_struts:
        p0 = topo_nodes[a]
        p1 = topo_nodes[b]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color="red",
            linewidth=2.0,
            alpha=0.95,
        )

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
        "Boundary-Only Kagome Debug Overlay\n"
        f"target_element_size={target_element_size}, eps={eps}"
    )

    print("=" * 60)
    print("Boundary Kagome Diagnostic")
    print(f"Topology nodes:         {topo_nodes.shape[0]}")
    print(f"Total struts:           {struts.shape[0]}")
    print(f"Boundary-only struts:   {boundary_struts.shape[0]}")
    print("=" * 60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_boundary_debug()
