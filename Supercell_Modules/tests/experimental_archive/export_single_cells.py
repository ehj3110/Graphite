"""
Export individual Unit Cell STLs — Nodes + Struts via Manifold3D.

Generates standalone STLs for A15, Kelvin, Diamond, Cube_6Tets, and Cube_6Tets_Dual.
Demonstrates the Cube-to-Tetrahedron dual and Kagome honeycomb relationship.

Run from Graphite root: python -m Supercell_Modules.tests.export_single_cells
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifold3d
import numpy as np

# Add Graphite root for geometry_module
_graphite_root = Path(__file__).resolve().parents[2]
if str(_graphite_root) not in sys.path:
    sys.path.insert(0, str(_graphite_root))

# Add Supercell_Modules for tools
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    generate_geometry,
    manifold_to_trimesh,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "Unit_Cells"
CELL_SIZE = 10.0
NODE_RADIUS = 0.8
STRUT_RADIUS = 0.4


def _scale_nodes(nodes: np.ndarray, scale: float) -> np.ndarray:
    """Scale node coordinates by cell_size."""
    return np.asarray(nodes, dtype=np.float64) * scale


def _build_with_spheres_and_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
) -> "manifold3d.Manifold":
    """Build cylinders (struts) + spheres (nodes), union, return manifold."""
    # Cylinders for struts
    lattice = generate_geometry(
        nodes=nodes,
        struts=struts,
        strut_radius=STRUT_RADIUS,
        boundary_mesh=None,
        add_spheres=False,
        crop_to_boundary=False,
    )
    if isinstance(lattice, tuple):
        lattice = lattice[0]
    lat_manifold = _trimesh_to_manifold(lattice)

    # Spheres at nodes
    spheres = []
    for xyz in nodes:
        s = manifold3d.Manifold.sphere(NODE_RADIUS).translate(
            tuple(float(x) for x in xyz)
        )
        spheres.append(s)
    sphere_union = manifold3d.Manifold.compose(spheres)
    combined = manifold3d.Manifold.compose([lat_manifold, sphere_union])
    return combined


def _get_a15_cell() -> tuple[np.ndarray, np.ndarray]:
    """A15 (Pm-3n): 8 nodes, BCC + 6 face centers."""
    nodes = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [0.25, 0.5, 0],
            [0.75, 0.5, 0],
            [0, 0.25, 0.5],
            [0, 0.75, 0.5],
            [0.5, 0, 0.25],
            [0.5, 0, 0.75],
        ],
        dtype=np.float64,
    )
    struts = np.array(
        [
            [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
            [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
            [2, 3], [4, 5], [6, 7],
        ],
        dtype=np.int64,
    )
    return nodes, struts


def _get_kelvin_cell() -> tuple[np.ndarray, np.ndarray]:
    """Kelvin (Truncated Octahedron): 12 nodes, permutations of [0, 1/2, 1/4]."""
    from itertools import permutations

    nodes_set: set[tuple[float, float, float]] = set()
    for vals in ([0.0, 0.5, 0.25], [0.0, 0.5, 0.75]):
        for p in permutations(vals):
            nodes_set.add(p)
    nodes = np.array(sorted(nodes_set), dtype=np.float64)

    def transposition_adjacent(t1: tuple[float, ...], t2: tuple[float, ...]) -> bool:
        diffs = [(a, b) for a, b in zip(t1, t2) if abs(a - b) > 1e-6]
        return len(diffs) == 2 and abs(diffs[0][0] - diffs[0][1]) == abs(
            diffs[1][0] - diffs[1][1]
        )

    struts_list: list[tuple[int, int]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if transposition_adjacent(tuple(nodes[i]), tuple(nodes[j])):
                struts_list.append((i, j))
    struts = np.array(struts_list, dtype=np.int64)
    return nodes, struts


def _get_diamond_cell() -> tuple[np.ndarray, np.ndarray]:
    """Diamond: 8 nodes, tetrahedral 4-connected network."""
    nodes = np.array(
        [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ],
        dtype=np.float64,
    )
    bonds = [
        (0, 4), (0, 7), (0, 6), (0, 5),
        (1, 5), (1, 6), (1, 4), (1, 7),
        (2, 6), (2, 5), (2, 7), (2, 4),
        (3, 7), (3, 4), (3, 5), (3, 6),
    ]
    struts = np.unique(np.sort(np.array(bonds, dtype=np.int64), axis=1), axis=0)
    return nodes, struts


def _get_cube_6tets() -> tuple[np.ndarray, np.ndarray]:
    """
    Cube (0,0,0)-(1,1,1) split into 6 tetrahedra (Freudenthal).
    Tets share diagonal 0-7.
    """
    verts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    tets = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ]
    edges_set: set[tuple[int, int]] = set()
    for tet in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = tet[i], tet[j]
                edges_set.add((min(a, b), max(a, b)))
    nodes = verts
    struts = np.array(sorted(edges_set), dtype=np.int64)
    return nodes, struts


def _get_cube_6tets_dual() -> tuple[np.ndarray, np.ndarray]:
    """
    Dual of 6-tet cube: centroids of tets, connected if tets share a face.
    Forms a hexagonal loop (Kagome honeycomb building block).
    """
    verts = np.array(
        [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    tets = [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ]
    centroids = np.array([verts[t].mean(axis=0) for t in tets], dtype=np.float64)
    adj = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    struts = np.array([[min(a, b), max(a, b)] for a, b in adj], dtype=np.int64)
    return centroids, struts


def _export_cell(name: str, nodes: np.ndarray, struts: np.ndarray) -> Path:
    """Scale nodes, build lattice with spheres+struts, export STL."""
    nodes_scaled = _scale_nodes(nodes, CELL_SIZE)
    manifold = _build_with_spheres_and_struts(nodes_scaled, struts)
    mesh = manifold_to_trimesh(manifold)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.stl"
    mesh.export(str(path))
    print(f"  Exported: {path}")
    return path


def main() -> None:
    """Generate all unit cell STLs."""
    print("=" * 60)
    print("Unit Cell STL Export — Nodes (r=0.8) + Struts (r=0.4)")
    print("=" * 60)
    print(f"Cell size: {CELL_SIZE} mm")
    print(f"Output: {OUTPUT_DIR}")
    print()

    cells = [
        ("A15", _get_a15_cell, "Pm-3n space group"),
        ("Kelvin", _get_kelvin_cell, "14-sided truncated octahedron"),
        ("Diamond", _get_diamond_cell, "Tetrahedral 4-connected network"),
        ("Cube_6Tets", _get_cube_6tets, "Freudenthal 6-tet cube"),
        ("Cube_6Tets_Dual", _get_cube_6tets_dual, "Kagome honeycomb building block"),
    ]

    for name, getter, desc in cells:
        print(f"[{name}] {desc}")
        nodes, struts = getter()
        _export_cell(name, nodes, struts)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
