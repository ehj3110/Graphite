"""
Natural Hex Surface Visualizer — Verify GMSH's quad surface meshing.

Bypasses STL export and directly extracts Element Type 3 (4-node Quadrangles)
from GMSH after natural hex meshing (RecombineAll=1, no Transfinite).
Plots the quadrilateral surface elements with Poly3DCollection.

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.visualize_natural_hex_surface
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_root = Path(__file__).resolve().parents[2]
STL_PATH = _root / "test_parts" / "20mm_cube.stl"   # 20mm box geometry


def extract_quad_surface() -> np.ndarray:
    """
    Mesh the 20mm box STL with natural hex meshing (RecombineAll=1, no Transfinite).
    Extract Element Type 3 (4-node Quadrangles) as the true surface.

    Returns
    -------
    ndarray, shape (N, 4, 3)
        Corner coordinates for each quadrilateral surface element.
    """
    if not STL_PATH.exists():
        raise FileNotFoundError(
            f"STL not found: {STL_PATH}. "
            "Ensure test_parts/20mm_cube.stl exists (20mm box geometry)."
        )

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("natural_hex")

    gmsh.merge(str(STL_PATH))

    gmsh.model.mesh.classifySurfaces(
        40.0 * math.pi / 180.0,
        boundary=True,
        forReparametrization=True,
        curveAngle=math.pi,
    )
    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        gmsh.finalize()
        raise RuntimeError("No surfaces after classifySurfaces — check STL is closed.")

    surface_tags = [s[1] for s in surfaces]
    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(surface_tags)])
    gmsh.model.geo.synchronize()

    # Natural hex meshing — recombination ON, no Transfinite
    gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)   # Frontal-Delaunay for Quads

    gmsh.model.mesh.generate(3)

    # Extract node coordinates
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords_arr = coords_flat.reshape(-1, 3)
    node_coord = {int(t): coords_arr[i] for i, t in enumerate(node_tags)}

    # Extract Element Type 3 (4-node Quadrangle) — 2D surface elements
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    quad_coords_list: list[np.ndarray] = []

    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 3:   # GMSH type 3 = 4-node quadrangle
            n_elems = len(en) // 4
            node_matrix = en.reshape(n_elems, 4)
            for row in node_matrix:
                quad_coords_list.append(np.array([node_coord[int(t)] for t in row]))

    gmsh.finalize()

    if not quad_coords_list:
        raise RuntimeError(
            "GMSH produced 0 type-3 (quadrangle) surface elements. "
            "Try adjusting MeshSizeMax or mesh options."
        )

    return np.array(quad_coords_list, dtype=np.float64)   # (N, 4, 3)


def main() -> None:
    print("=" * 60)
    print("Natural Hex Surface — Quadrilateral Element Visualizer")
    print("=" * 60)
    print(f"STL: {STL_PATH}")
    print("Mesh: RecombineAll=1, Algorithm=8, MeshSizeMax=5.0 (no Transfinite)")

    quad_coords = extract_quad_surface()
    n_quads = quad_coords.shape[0]
    print(f"Verified: Extracted {n_quads} Quadrilateral surface elements.")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(
        quad_coords,
        facecolors="lightcyan",
        edgecolors="black",
        linewidths=0.8,
        alpha=0.8,
    )
    ax.add_collection3d(poly)

    # Equal aspect ratio
    verts = quad_coords.reshape(-1, 3)
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    mid_x = (x_min + x_max) * 0.5
    mid_y = (y_min + y_max) * 0.5
    mid_z = (z_min + z_max) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("GMSH Natural Hex — Quadrilateral Surface Elements")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
