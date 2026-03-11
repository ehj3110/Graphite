"""
Plot the 4 high-contrast supercell isolation outputs in one 3D space.

This builds each topology's classified polyhedral faces (tet/octa/large) and
places the four topologies in a shared scene for side-by-side spatial comparison.

Run:
  python -m Supercell_Modules.tests.plot_isolation_3d_space
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from core.topologies import (
    generate_a15_seeds,
    generate_bitruncated_cubic_seeds,
    generate_rhombicuboct_seeds,
    generate_truncated_oct_tet_seeds,
)


CELL_SIZE = 10.0
ALPHA = 0.9
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "output" / "Supercell_Isolation" / "HighContrast_3D_Space.png"
)

TYPE_COLORS = {
    "tetrahedral": "#1f77b4",  # blue
    "octahedral": "#2ca02c",  # green
    "truncated_large": "#d62728",  # red
}


def tetra_volume(vertices: np.ndarray) -> float:
    a, b, c, d = vertices
    mat = np.stack((b - a, c - a, d - a), axis=1)
    return float(abs(np.linalg.det(mat)) / 6.0)


def group_tets_by_center(points: np.ndarray, simplices: np.ndarray) -> dict[int, list[int]]:
    tet_centroids = points[simplices].mean(axis=1)
    d2 = np.sum((tet_centroids[:, None, :] - points[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    groups: dict[int, list[int]] = {}
    for tid, sid in enumerate(nearest):
        groups.setdefault(int(sid), []).append(int(tid))
    return groups


def classify_clusters(
    points: np.ndarray, simplices: np.ndarray, groups: dict[int, list[int]]
) -> dict[int, str]:
    cluster_ids = sorted(groups.keys())
    means = []
    for cid in cluster_ids:
        vols = [tetra_volume(points[simplices[tid]]) for tid in groups[cid]]
        means.append(float(np.mean(vols)) if vols else 0.0)
    if not means:
        return {}

    means_arr = np.asarray(means, dtype=np.float64)
    q1 = float(np.quantile(means_arr, 1.0 / 3.0))
    q2 = float(np.quantile(means_arr, 2.0 / 3.0))

    out: dict[int, str] = {}
    for cid, mv in zip(cluster_ids, means):
        if mv <= q1:
            out[cid] = "tetrahedral"
        elif mv <= q2:
            out[cid] = "octahedral"
        else:
            out[cid] = "truncated_large"
    return out


def cluster_boundary_faces(points: np.ndarray, simplices: np.ndarray, tet_ids: list[int]) -> list[np.ndarray]:
    face_map: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    face_triplets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for tid in tet_ids:
        tet = simplices[tid]
        for i, j, k in face_triplets:
            face_local = (int(tet[i]), int(tet[j]), int(tet[k]))
            face_key = tuple(sorted(face_local))
            face_map.setdefault(face_key, []).append(face_local)

    faces: list[np.ndarray] = []
    for owners in face_map.values():
        if len(owners) == 1:
            tri = owners[0]
            faces.append(points[list(tri)])
    return faces


def build_faces_by_type(seed_fn) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    points = seed_fn(nx=1, ny=1, nz=1, cell_size=CELL_SIZE)
    simplices = np.asarray(Delaunay(points).simplices, dtype=np.int64)
    keep = [i for i, tet in enumerate(simplices) if tetra_volume(points[tet]) >= 1e-8]
    simplices = simplices[np.asarray(keep, dtype=np.int64)]

    groups = group_tets_by_center(points, simplices)
    cluster_types = classify_clusters(points, simplices, groups)

    faces_by_type: dict[str, list[np.ndarray]] = {
        "tetrahedral": [],
        "octahedral": [],
        "truncated_large": [],
    }
    for cid, tet_ids in groups.items():
        ctype = cluster_types.get(cid, "tetrahedral")
        faces_by_type[ctype].extend(cluster_boundary_faces(points, simplices, tet_ids))
    return faces_by_type, points


def main() -> None:
    topologies = [
        ("Bitruncated", generate_bitruncated_cubic_seeds, np.array([0.0, 0.0, 0.0])),
        ("Big-Small Hybrid", generate_truncated_oct_tet_seeds, np.array([35.0, 0.0, 0.0])),
        ("Rhombicuboctahedron", generate_rhombicuboct_seeds, np.array([0.0, 35.0, 0.0])),
        ("A15", generate_a15_seeds, np.array([35.0, 35.0, 0.0])),
    ]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    all_pts: list[np.ndarray] = []
    for name, seed_fn, offset in topologies:
        faces_by_type, points = build_faces_by_type(seed_fn)
        shifted_points = points + offset[None, :]
        all_pts.append(shifted_points)

        for ctype in ("tetrahedral", "octahedral", "truncated_large"):
            faces = faces_by_type[ctype]
            if not faces:
                continue
            shifted_faces = [face + offset[None, :] for face in faces]
            pc = Poly3DCollection(
                shifted_faces,
                facecolors=TYPE_COLORS[ctype],
                edgecolors="#000000",
                linewidths=1.0,
                alpha=ALPHA,
            )
            ax.add_collection3d(pc)

        pmin = shifted_points.min(axis=0)
        pmax = shifted_points.max(axis=0)
        label_pos = np.array([(pmin[0] + pmax[0]) * 0.5, pmin[1] - 2.0, pmax[2] + 1.5])
        ax.text(label_pos[0], label_pos[1], label_pos[2], name, fontsize=9, color="black")

    pts = np.vstack(all_pts)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.55 * np.max(maxs - mins)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_axis_off()
    fig.suptitle("Supercell Isolation - 4 Topologies in Shared 3D Space", fontsize=13)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=240)
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
