"""
Export the true 6-tet Kagome dual for a single 10 mm cube.

Run from Graphite root:
    python -m Supercell_Modules.tests.export_true_kagome
"""

from __future__ import annotations

import sys
from pathlib import Path

import manifold3d
import numpy as np
import trimesh
from scipy.spatial import Delaunay

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from core.topologies import generate_centroid_dual


CELL_SIZE = 10.0
STRUT_RADIUS = 0.5
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "True_6Tet_Kagome.stl"


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return orthonormal frame, segment length, and midpoint for p0->p1."""
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


def build_strut_manifold(nodes: np.ndarray, struts: np.ndarray, radius: float) -> manifold3d.Manifold:
    """Create manifold cylinder struts and union them."""
    cylinders: list[manifold3d.Manifold] = []
    for i, j in struts:
        p0 = nodes[int(i)]
        p1 = nodes[int(j)]
        if np.linalg.norm(p1 - p0) <= 1e-9:
            continue

        R, seg_len, mid = _frame_from_segment(p0, p1)
        cyl = manifold3d.Manifold.cylinder(
            height=seg_len,
            radius_low=radius,
            radius_high=radius,
            circular_segments=18,
            center=True,
        )
        affine = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(mid[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(mid[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(mid[2])],
        ]
        cylinders.append(cyl.transform(affine))

    if not cylinders:
        raise RuntimeError("No strut cylinders were created.")
    return manifold3d.Manifold.compose(cylinders)


def main() -> None:
    print("=" * 64)
    print("TRUE 6-TET KAGOME EXPORT")
    print("=" * 64)

    # 2x2x2 grid of corner nodes for a single cube [0,10]^3.
    gx, gy, gz = np.mgrid[0:11:10, 0:11:10, 0:11:10]
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float64)

    # Delaunay of cube corners naturally gives a 6-tet triangulation.
    delaunay = Delaunay(points)
    simplices = np.asarray(delaunay.simplices, dtype=np.int64)

    # Purge any degenerate tets for robustness.
    tet_pts = points[simplices]
    mat = np.stack(
        (
            tet_pts[:, 1, :] - tet_pts[:, 0, :],
            tet_pts[:, 2, :] - tet_pts[:, 0, :],
            tet_pts[:, 3, :] - tet_pts[:, 0, :],
        ),
        axis=1,
    )
    volumes = np.abs(np.linalg.det(mat)) / 6.0
    clean_simplices = simplices[volumes >= 1e-8]

    dual_nodes, dual_struts = generate_centroid_dual(points, clean_simplices)
    print(f"Cube corner nodes: {points.shape[0]}")
    print(f"Delaunay tetrahedra (clean): {clean_simplices.shape[0]}")
    print(f"Dual nodes: {dual_nodes.shape[0]}")
    print(f"Dual struts: {dual_struts.shape[0]}")

    lattice = build_strut_manifold(dual_nodes, dual_struts, STRUT_RADIUS)
    mesh = lattice.to_mesh()
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vert_properties, dtype=np.float64),
        faces=np.asarray(mesh.tri_verts, dtype=np.int64),
        process=False,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(OUTPUT_PATH))
    print(f"Exported: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
