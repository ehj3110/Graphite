"""
Test 2x2x2 seed grids for SC, BCC, and A15 dual lattices.

Pipeline per topology:
  1) Seed generation
  2) Delaunay tetrahedralization
  3) Degenerate tet purge
  4) Centroid dual extraction
  5) Render dual struts + nodes with Manifold3D
  6) Export STL

Run from Graphite root:
    python -m Supercell_Modules.tests.test_2x2x2_grids
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifold3d
import numpy as np
import trimesh
from scipy.spatial import Delaunay

# Add Graphite root for geometry_module import
_graphite_root = Path(__file__).resolve().parents[2]
if str(_graphite_root) not in sys.path:
    sys.path.insert(0, str(_graphite_root))

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from graphite.explicit.geometry_module import manifold_to_trimesh
from core.topologies import (
    generate_a15_seeds,
    generate_bcc_seeds,
    generate_centroid_dual,
    generate_simple_cubic_seeds,
)


NX = 2
NY = 2
NZ = 2
CELL_SIZE = 10.0
STRUT_RADIUS = 0.5
NODE_RADIUS = 0.8
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def tetra_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute tet volumes: abs(det([b-a, c-a, d-a])) / 6."""
    t = points[simplices]
    mat = np.stack(
        (
            t[:, 1, :] - t[:, 0, :],
            t[:, 2, :] - t[:, 0, :],
            t[:, 3, :] - t[:, 0, :],
        ),
        axis=1,
    )
    return np.abs(np.linalg.det(mat)) / 6.0


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return orthonormal frame, length, midpoint for p0->p1."""
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise ValueError("Zero-length segment.")

    z_axis = vec / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    midpoint = 0.5 * (p0 + p1)
    return R, length, midpoint


def build_dual_manifold(
    dual_nodes: np.ndarray,
    dual_struts: np.ndarray,
    strut_radius: float,
    node_radius: float,
) -> manifold3d.Manifold:
    """Build manifold lattice from dual struts and dual node spheres."""
    parts: list[manifold3d.Manifold] = []

    for i, j in dual_struts:
        p0 = dual_nodes[int(i)]
        p1 = dual_nodes[int(j)]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 1e-9:
            continue

        R, seg_len, mid = _frame_from_segment(p0, p1)
        cyl = manifold3d.Manifold.cylinder(
            height=seg_len,
            radius_low=strut_radius,
            radius_high=strut_radius,
            circular_segments=16,
            center=True,
        )
        affine = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(mid[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(mid[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(mid[2])],
        ]
        parts.append(cyl.transform(affine))

    for xyz in dual_nodes:
        parts.append(manifold3d.Manifold.sphere(node_radius).translate(tuple(float(v) for v in xyz)))

    if not parts:
        raise RuntimeError("No manifold parts created from dual graph.")
    return manifold3d.Manifold.compose(parts)


def run_topology(
    name: str,
    points: np.ndarray,
    output_filename: str,
) -> None:
    """Run Delaunay->clean->dual->render/export for one topology."""
    delaunay = Delaunay(points)
    simplices = np.asarray(delaunay.simplices, dtype=np.int64)

    volumes = tetra_volumes(points, simplices)
    clean_simplices = simplices[volumes >= 1e-8]
    if clean_simplices.shape[0] == 0:
        raise RuntimeError(f"{name}: all tetrahedra were degenerate.")

    dual_nodes, dual_struts = generate_centroid_dual(points, clean_simplices)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Seeds: {points.shape[0]}")
    print(f"Tets (raw): {simplices.shape[0]}")
    print(f"Tets (clean): {clean_simplices.shape[0]}")
    print(f"Dual nodes: {dual_nodes.shape[0]}")
    print(f"Dual struts: {dual_struts.shape[0]}")

    lattice = build_dual_manifold(dual_nodes, dual_struts, STRUT_RADIUS, NODE_RADIUS)
    mesh = manifold_to_trimesh(lattice)
    out_path = OUTPUT_DIR / output_filename
    mesh.export(str(out_path))
    print(f"Exported: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("2x2x2 GRID TEST — SC, BCC, A15")
    print("=" * 72)
    print(f"nx={NX}, ny={NY}, nz={NZ}, cell_size={CELL_SIZE}")

    run_topology(
        name="Simple Cubic -> Kagome Dual",
        points=generate_simple_cubic_seeds(NX, NY, NZ, CELL_SIZE),
        output_filename="2x2x2_Kagome.stl",
    )
    run_topology(
        name="BCC -> Voronoi Dual",
        points=generate_bcc_seeds(NX, NY, NZ, CELL_SIZE),
        output_filename="2x2x2_Voronoi.stl",
    )
    run_topology(
        name="A15 -> Icosahedral Dual",
        points=generate_a15_seeds(NX, NY, NZ, CELL_SIZE),
        output_filename="2x2x2_Icosahedral.stl",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
