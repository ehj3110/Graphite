"""
Opaque element-isolation diagnostic for supercell seed topologies.

For each topology, this script:
  1) Generates seeds and Delaunay tetrahedra
  2) Groups tetrahedra into parent polyhedral clusters by nearest seed-center
  3) Classifies clusters into tetrahedral / octahedral / large-core by mean tet volume
  4) Renders three opaque subplots using Poly3DCollection

Exports one PNG per topology to:
  Supercell_Modules/output/Supercell_Isolation/

Run:
  python -m Supercell_Modules.tests.visualize_opaque_elements
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


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "Supercell_Isolation"
CELL_SIZE = 10.0
ALPHA = 0.9

TYPE_COLORS = {
    "tetrahedral": "#1f77b4",  # blue
    "octahedral": "#2ca02c",  # green
    "truncated_large": "#d62728",  # red
}
TYPE_TITLES = {
    "tetrahedral": "Tetrahedral Components",
    "octahedral": "Octahedral Components",
    "truncated_large": "Large Polyhedral Cores",
}


def tetra_volume(vertices: np.ndarray) -> float:
    a, b, c, d = vertices
    mat = np.stack((b - a, c - a, d - a), axis=1)
    return float(abs(np.linalg.det(mat)) / 6.0)


def group_tets_by_center(points: np.ndarray, simplices: np.ndarray) -> dict[int, list[int]]:
    """Assign each tet to nearest seed point based on centroid."""
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
    """Classify each cluster by mean tet volume into 3 size families."""
    cluster_ids = sorted(groups.keys())
    if not cluster_ids:
        return {}

    means = []
    for cid in cluster_ids:
        vols = [tetra_volume(points[simplices[tid]]) for tid in groups[cid]]
        means.append(float(np.mean(vols)) if vols else 0.0)
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
    """
    Return triangular boundary faces for a cluster.

    Faces shared by two tetrahedra in the same cluster are removed (internal faces).
    """
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


def set_equal_axes(ax, all_points: np.ndarray) -> None:
    """Set equal XYZ axes bounds to avoid distortion."""
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.55 * np.max(maxs - mins)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def render_topology_isolation(name: str, seed_fn) -> tuple[plt.Figure, Path]:
    """Render 3-panel opaque element isolation for one topology."""
    points = seed_fn(nx=1, ny=1, nz=1, cell_size=CELL_SIZE)
    simplices = np.asarray(Delaunay(points).simplices, dtype=np.int64)

    # Purge degenerate tets.
    keep_ids = []
    for tid, tet in enumerate(simplices):
        if tetra_volume(points[tet]) >= 1e-8:
            keep_ids.append(tid)
    simplices = simplices[np.asarray(keep_ids, dtype=np.int64)]

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

    fig = plt.figure(figsize=(16, 5))
    plot_order = ["tetrahedral", "octahedral", "truncated_large"]
    for i, ctype in enumerate(plot_order, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        faces = faces_by_type[ctype]
        if faces:
            pc = Poly3DCollection(
                faces,
                facecolors=TYPE_COLORS[ctype],
                edgecolors="#000000",
                linewidths=2.0,
                alpha=ALPHA,
            )
            ax.add_collection3d(pc)

        set_equal_axes(ax, points)
        ax.set_title(TYPE_TITLES[ctype], fontsize=10)
        ax.grid(True, alpha=0.45, linewidth=0.6)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.suptitle(f"{name} - Opaque Element Isolation", fontsize=13)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"HighContrast_{name.replace(' ', '_')}.png"
    fig.savefig(str(out_path), dpi=240)
    return fig, out_path


def main() -> None:
    topologies = [
        ("Bitruncated", generate_bitruncated_cubic_seeds),
        ("Big-Small Hybrid", generate_truncated_oct_tet_seeds),
        ("Rhombicuboctahedron", generate_rhombicuboct_seeds),
        ("A15", generate_a15_seeds),
    ]

    print("=" * 72)
    print("OPAQUE ELEMENT ISOLATION")
    print("=" * 72)
    figures: list[plt.Figure] = []
    for name, fn in topologies:
        fig, out = render_topology_isolation(name, fn)
        figures.append(fig)
        print(f"Saved: {out}")
    print("\nOpening interactive windows (use toolbar to pan/rotate).")
    plt.show()
    for fig in figures:
        plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
