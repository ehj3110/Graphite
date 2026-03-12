"""
Integer Subdivision Transfinite Meshing — Eliminate skewed elements.

Reconstructs CAD topology from STL, applies integer subdivision to curves
based on goal_size, then transfinite surfaces with recombination.
Includes a Mesh Quality Analyzer to categorize quads by internal angle deviation.

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_even_hex_subdivision
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_root = Path(__file__).resolve().parents[2]
STL_PATH = _root / "test_parts" / "20mm_cube.stl"
GOAL_SIZE = 6.0   # mm — 20/6 forces non-integer, tests rounding


def _quad_internal_angles(quad: np.ndarray) -> np.ndarray:
    """
    Compute the 4 internal angles (degrees) of a quad with vertices A, B, C, D.

    quad: (4, 3) — vertices in order A, B, C, D
    Returns: (4,) angles at A, B, C, D
    """
    angles = np.zeros(4)
    for i in range(4):
        a = quad[i]
        b = quad[(i - 1) % 4]
        c = quad[(i + 1) % 4]
        u = b - a
        v = c - a
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            angles[i] = 90.0
            continue
        cos_a = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        angles[i] = np.degrees(np.arccos(cos_a))
    return angles


def _classify_quad(angles: np.ndarray) -> str:
    """
    Categorize quad by worst angle deviation from 90°:
    - Ideal: all angles 85-95 deg
    - Skewed: any angle 70-85 or 95-110 deg
    - Sliver: any angle < 70 or > 110°
    """
    for a in angles:
        if a < 70 or a > 110:
            return "Sliver"
    for a in angles:
        if a < 85 or a > 95:
            return "Skewed"
    return "Ideal"


def mesh_with_integer_subdivision() -> np.ndarray:
    """
    Load STL, reconstruct topology, apply integer subdivision transfinite,
    generate 2D mesh, extract quads.

    Returns
    -------
    ndarray, shape (N, 4, 3)
        Quadrilateral vertex coordinates.
    """
    if not STL_PATH.exists():
        raise FileNotFoundError(
            f"STL not found: {STL_PATH}. "
            "Ensure test_parts/20mm_cube.stl exists."
        )

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("even_hex")

    gmsh.merge(str(STL_PATH))

    # Reconstruct CAD topology — groups triangles by sharp edges (45°)
    gmsh.model.mesh.classifySurfaces(math.pi / 4)
    gmsh.model.mesh.createGeometry()

    # Integer subdivision on curves
    curves = gmsh.model.getEntities(1)
    for dim, tag in curves:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox
        L = math.sqrt(
            (xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2
        )
        num_cells = max(1, round(L / GOAL_SIZE))
        num_nodes = num_cells + 1
        gmsh.model.mesh.setTransfiniteCurve(tag, num_nodes)

    # Transfinite + recombination on surfaces
    surfaces = gmsh.model.getEntities(2)
    for dim, tag in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(dim, tag)

    gmsh.model.mesh.generate(2)

    # Extract node coordinates
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords_arr = coords_flat.reshape(-1, 3)
    node_coord = {int(t): coords_arr[i] for i, t in enumerate(node_tags)}

    # Extract Element Type 3 (4-node Quadrangle)
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    quad_coords_list: list[np.ndarray] = []

    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 3:
            n_elems = len(en) // 4
            node_matrix = en.reshape(n_elems, 4)
            for row in node_matrix:
                quad_coords_list.append(np.array([node_coord[int(t)] for t in row]))

    gmsh.finalize()

    if not quad_coords_list:
        raise RuntimeError("No quadrangle elements produced.")

    return np.array(quad_coords_list, dtype=np.float64)


def main() -> None:
    print("=" * 60)
    print("Integer Subdivision Transfinite — Mesh Quality Analyzer")
    print("=" * 60)
    print(f"STL: {STL_PATH}")
    print(f"Goal size: {GOAL_SIZE} mm  (20/6 = {20/6:.2f} -> forces rounding)")

    quad_coords = mesh_with_integer_subdivision()
    n_total = quad_coords.shape[0]
    print(f"\nExtracted {n_total} Quadrilateral surface elements.")

    # Mesh quality analysis
    ideal, skewed, sliver = 0, 0, 0
    qualities: list[str] = []

    for i in range(n_total):
        angles = _quad_internal_angles(quad_coords[i])
        q = _classify_quad(angles)
        qualities.append(q)
        if q == "Ideal":
            ideal += 1
        elif q == "Skewed":
            skewed += 1
        else:
            sliver += 1

    pct_ideal = 100.0 * ideal / n_total if n_total else 0
    pct_skewed = 100.0 * skewed / n_total if n_total else 0
    pct_sliver = 100.0 * sliver / n_total if n_total else 0

    print("\n--- Mesh Quality Statistics ---")
    print(f"  Total Elements : {n_total}")
    print(f"  Ideal (85-95 deg)  : {ideal:4d}  ({pct_ideal:5.1f}%)")
    print(f"  Skewed (70-110 deg): {skewed:4d}  ({pct_skewed:5.1f}%)")
    print(f"  Sliver (<70 or >110): {sliver:4d}  ({pct_sliver:5.1f}%)")

    # Colored visualization
    colors = {
        "Ideal": "lightgreen",
        "Skewed": "yellow",
        "Sliver": "red",
    }
    face_colors = [colors[q] for q in qualities]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(
        quad_coords,
        facecolors=face_colors,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
    )
    ax.add_collection3d(poly)

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
    ax.set_title(
        f"Integer Subdivision — Ideal=green, Skewed=yellow, Sliver=red  "
        f"({pct_ideal:.0f}% Ideal)"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
