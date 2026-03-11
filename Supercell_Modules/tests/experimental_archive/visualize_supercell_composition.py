"""
Interactive composition viewer for supercell seed topologies.

This view groups Delaunay tetrahedra into polyhedral "cells" by assigning each tet
to its nearest seed-center (based on tet centroid). Clusters are then classified as:
  - tetrahedral (blue)
  - octahedral (green)
  - truncated/large (red)

Outputs:
  - interactive matplotlib viewer (plt.show)
  - static map: Supercell_Modules/output/Supercell_Composition_Map.png

Run:
  python -m Supercell_Modules.tests.visualize_supercell_composition
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "Supercell_Composition_Map.png"
CELL_SIZE = 10.0

TYPE_COLORS = {
    "tetrahedral": "#1f77b4",  # blue
    "octahedral": "#2ca02c",  # green
    "truncated_large": "#d62728",  # red
}


def tetra_volume(vertices: np.ndarray) -> float:
    """Volume of a tetrahedron given (4,3) vertices."""
    a, b, c, d = vertices
    mat = np.stack((b - a, c - a, d - a), axis=1)
    return float(abs(np.linalg.det(mat)) / 6.0)


def classify_element(vertices: np.ndarray) -> dict[str, float | str]:
    """
    Classify one tetrahedron by geometric metrics.

    Returns:
      - volume
      - edge_ratio (max edge / min edge)
      - size_class (small/medium/large) based on normalized volume placeholder
    """
    vol = tetra_volume(vertices)
    edges = [
        np.linalg.norm(vertices[i] - vertices[j])
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    ]
    edge_ratio = float(max(edges) / max(min(edges), 1e-12))
    # Cluster type is assigned later after grouping by common center.
    return {"volume": vol, "edge_ratio": edge_ratio, "size_class": "unassigned"}


def group_tets_by_center(points: np.ndarray, simplices: np.ndarray) -> dict[int, list[int]]:
    """
    Group tetrahedra by nearest seed-center to tet centroid.

    This approximates polyhedral-cell ownership and is robust for one repeating unit.
    """
    tet_centroids = points[simplices].mean(axis=1)
    d2 = np.sum((tet_centroids[:, None, :] - points[None, :, :]) ** 2, axis=2)
    nearest_seed_ids = np.argmin(d2, axis=1)

    groups: dict[int, list[int]] = {}
    for tet_id, seed_id in enumerate(nearest_seed_ids):
        groups.setdefault(int(seed_id), []).append(int(tet_id))
    return groups


def classify_clusters(
    points: np.ndarray, simplices: np.ndarray, groups: dict[int, list[int]]
) -> dict[int, str]:
    """
    Classify each cluster by mean tetra volume into 3 element types.
    """
    cluster_ids = sorted(groups.keys())
    if not cluster_ids:
        return {}

    cluster_mean_vols = []
    for cid in cluster_ids:
        vols = []
        for tid in groups[cid]:
            tet = simplices[tid]
            verts = points[tet]
            vols.append(tetra_volume(verts))
        cluster_mean_vols.append(float(np.mean(vols)) if vols else 0.0)

    vols_arr = np.asarray(cluster_mean_vols, dtype=np.float64)
    q1 = float(np.quantile(vols_arr, 1.0 / 3.0))
    q2 = float(np.quantile(vols_arr, 2.0 / 3.0))

    cluster_types: dict[int, str] = {}
    for cid, mean_v in zip(cluster_ids, cluster_mean_vols):
        if mean_v <= q1:
            cluster_types[cid] = "tetrahedral"
        elif mean_v <= q2:
            cluster_types[cid] = "octahedral"
        else:
            cluster_types[cid] = "truncated_large"
    return cluster_types


def tet_segments(vertices: np.ndarray) -> np.ndarray:
    """Return the 6 edge segments of one tetrahedron."""
    return np.array(
        [
            [vertices[0], vertices[1]],
            [vertices[0], vertices[2]],
            [vertices[0], vertices[3]],
            [vertices[1], vertices[2]],
            [vertices[1], vertices[3]],
            [vertices[2], vertices[3]],
        ],
        dtype=np.float64,
    )


def build_cluster_wireframes(
    points: np.ndarray,
    simplices: np.ndarray,
    groups: dict[int, list[int]],
    cluster_types: dict[int, str],
) -> dict[str, list[np.ndarray]]:
    """
    Build per-type edge segment lists.

    Deliberately keeps duplicates so shared edges can be drawn multiple times.
    """
    edges_by_type: dict[str, list[np.ndarray]] = {
        "tetrahedral": [],
        "octahedral": [],
        "truncated_large": [],
    }
    for cid, tet_ids in groups.items():
        ctype = cluster_types.get(cid, "tetrahedral")
        for tid in tet_ids:
            tet = simplices[tid]
            verts = points[tet]
            edges_by_type[ctype].extend(tet_segments(verts).tolist())
    return edges_by_type


def supercell_boundary_segments(points: np.ndarray) -> np.ndarray:
    """Wireframe segments for bounding box around one repeating unit."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x0, y1, z0],
            [x1, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x0, y1, z1],
            [x1, y1, z1],
        ],
        dtype=np.float64,
    )
    edge_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edge_pairs], dtype=np.float64)


def plot_topology(ax, name: str, points: np.ndarray, simplices: np.ndarray) -> None:
    """Plot grouped polyhedral-cell wireframes."""
    # Remove degenerate tetrahedra.
    clean_ids = []
    for tid, tet in enumerate(simplices):
        verts = points[tet]
        metrics = classify_element(verts)
        if float(metrics["volume"]) >= 1e-8:
            clean_ids.append(tid)
    simplices = simplices[np.asarray(clean_ids, dtype=np.int64)]
    if simplices.size == 0:
        ax.set_title(f"{name}\n(no non-degenerate tets)")
        return

    groups = group_tets_by_center(points, simplices)
    cluster_types = classify_clusters(points, simplices, groups)
    edges_by_type = build_cluster_wireframes(points, simplices, groups, cluster_types)

    # Draw cluster wireframes by class.
    for ctype in ("tetrahedral", "octahedral", "truncated_large"):
        segs = edges_by_type[ctype]
        if not segs:
            continue
        lc = Line3DCollection(
            np.asarray(segs, dtype=np.float64),
            colors=TYPE_COLORS[ctype],
            linewidths=0.65,
            alpha=0.85,
        )
        ax.add_collection3d(lc)

    # Supercell boundary in black.
    boundary = supercell_boundary_segments(points)
    ax.add_collection3d(Line3DCollection(boundary, colors="black", linewidths=1.3, alpha=1.0))

    # Show seeds lightly for orientation.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8, c="black", alpha=0.2)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.max(maxs - mins)
    center = 0.5 * (mins + maxs)
    half = 0.55 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    counts = {
        "tetrahedral": sum(1 for t in cluster_types.values() if t == "tetrahedral"),
        "octahedral": sum(1 for t in cluster_types.values() if t == "octahedral"),
        "truncated_large": sum(1 for t in cluster_types.values() if t == "truncated_large"),
    }
    ax.set_title(
        f"{name}\nclusters B/G/R={counts['tetrahedral']}/{counts['octahedral']}/{counts['truncated_large']}",
        fontsize=9,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def main() -> None:
    topologies = [
        ("Bitruncated", generate_bitruncated_cubic_seeds),
        ("Big-Small Hybrid", generate_truncated_oct_tet_seeds),
        ("Rhombicuboctahedron", generate_rhombicuboct_seeds),
        ("A15", generate_a15_seeds),
    ]

    fig = plt.figure(figsize=(14, 11))

    for idx, (name, seed_fn) in enumerate(topologies, start=1):
        points = seed_fn(nx=1, ny=1, nz=1, cell_size=CELL_SIZE)
        simplices = np.asarray(Delaunay(points).simplices, dtype=np.int64)
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        plot_topology(ax, name, points, simplices)

    fig.suptitle("Supercell Composition Map (Clustered Polyhedral Cells)", fontsize=14)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PATH), dpi=220)
    print(f"Saved static composition map: {OUTPUT_PATH}")

    # Interactive viewer for rotation/inspection.
    plt.show()


if __name__ == "__main__":
    main()
