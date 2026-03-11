"""
Verbose diagnostics for:
  Step 1) BCC seed generation
  Step 2) Delaunay triangulation

Run from Graphite root:
    python -m Supercell_Modules.tests.test_triangulation
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from core.topologies import generate_centroid_dual


def generate_bcc_seeds(grid_size: int, cell_size: float) -> np.ndarray:
    """
    Generate BCC seed points for a cubic grid.

    For each cube cell (i, j, k):
      - Corner seed at (i, j, k) * cell_size
      - Body-center seed at (i + 0.5, j + 0.5, k + 0.5) * cell_size
    """
    corners = []
    centers = []

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                corners.append((i * cell_size, j * cell_size, k * cell_size))
                centers.append(
                    (
                        (i + 0.5) * cell_size,
                        (j + 0.5) * cell_size,
                        (k + 0.5) * cell_size,
                    )
                )

    return np.asarray(corners + centers, dtype=np.float64)


def tetra_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """
    Compute volume for each tetrahedron:
      V = abs(det([b-a, c-a, d-a])) / 6
    """
    tets = points[simplices]  # (N, 4, 3)
    a = tets[:, 0, :]
    b = tets[:, 1, :]
    c = tets[:, 2, :]
    d = tets[:, 3, :]

    m = np.stack((b - a, c - a, d - a), axis=1)  # (N, 3, 3)
    det = np.linalg.det(m)
    return np.abs(det) / 6.0


def unique_mesh_edges(simplices: np.ndarray) -> np.ndarray:
    """Return sorted unique vertex-index edges from tetrahedra."""
    edge_set: set[tuple[int, int]] = set()
    for tet in simplices:
        for u, v in combinations(tet, 2):
            a, b = (int(u), int(v)) if u < v else (int(v), int(u))
            edge_set.add((a, b))
    return np.asarray(sorted(edge_set), dtype=np.int32)


def edge_lengths(points: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Compute Euclidean length for each edge."""
    p0 = points[edges[:, 0]]
    p1 = points[edges[:, 1]]
    return np.linalg.norm(p1 - p0, axis=1)


def plot_dual_lattice(dual_nodes: np.ndarray, dual_struts: np.ndarray) -> None:
    """Plot centroid-dual lattice with thin line segments."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError:
        print("\n[Kagome Plot] matplotlib not installed; skipping plot.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    if dual_struts.shape[0] > 0:
        segments = np.array([[dual_nodes[i], dual_nodes[j]] for i, j in dual_struts])
        ax.add_collection3d(Line3DCollection(segments, colors="tab:blue", linewidths=0.6))

    ax.scatter(
        dual_nodes[:, 0],
        dual_nodes[:, 1],
        dual_nodes[:, 2],
        s=8,
        c="black",
        alpha=0.7,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Centroid Dual (Kagome Honeycomb)")

    mins = dual_nodes.min(axis=0)
    maxs = dual_nodes.max(axis=0)
    pad = 1.0
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    output_path = Path(__file__).resolve().parent.parent / "output" / "Triangulation_Kagome_Dual.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=180)
    plt.close(fig)
    print(f"\n[Kagome Plot] Saved: {output_path}")


def main() -> None:
    grid_size = 3
    cell_size = 10.0

    print("=" * 72)
    print("STEP 1: BCC SEED GENERATION")
    print("=" * 72)
    print(f"grid_size = {grid_size} (cells per axis)")
    print(f"cell_size = {cell_size}")

    points = generate_bcc_seeds(grid_size=grid_size, cell_size=cell_size)
    print(f"Total number of Seed Points: {points.shape[0]}")

    print("\n" + "=" * 72)
    print("STEP 2: DELAUNAY TRIANGULATION")
    print("=" * 72)
    delaunay = Delaunay(points)
    simplices = delaunay.simplices
    volumes = tetra_volumes(points, simplices)

    degenerate_tol = 1e-8
    degenerate_mask = volumes < degenerate_tol
    degenerate_indices = np.where(degenerate_mask)[0]
    clean_simplices = simplices[~degenerate_mask]
    clean_volumes = volumes[~degenerate_mask]

    print(f"Degenerate (zero-volume) tets found: {degenerate_indices.size}")
    if degenerate_indices.size > 0:
        print("First 3 degenerate tets (index in simplices array):")
        for idx in degenerate_indices[:3]:
            print(f"  Tet[{int(idx)}] vertex indices: {simplices[idx]}")
            print("  Coordinates:")
            print(points[simplices[idx]])

    print(f"Total number of Tetrahedra generated: {clean_simplices.shape[0]}")
    if clean_simplices.shape[0] == 0:
        raise RuntimeError("All tetrahedra were filtered as degenerate.")

    first_tet_ids = clean_simplices[0]
    first_tet_xyz = points[first_tet_ids]
    print("\nFirst tetrahedron vertex indices:")
    print(first_tet_ids)
    print("First tetrahedron vertex coordinates:")
    print(first_tet_xyz)

    print("\n" + "=" * 72)
    print("GEOMETRIC SANITY CHECKS")
    print("=" * 72)
    print("Tetrahedron Volumes (clean tets only)")
    print(f"  Min Volume: {clean_volumes.min():.6f}")
    print(f"  Max Volume: {clean_volumes.max():.6f}")
    print(f"  Avg Volume: {clean_volumes.mean():.6f}")

    edges = unique_mesh_edges(clean_simplices)
    lengths = edge_lengths(points, edges)
    print("\nMesh Edge Lengths (unique Delaunay edges)")
    print(f"  Edge Count: {edges.shape[0]}")
    print(f"  Min Length: {lengths.min():.6f}")
    print(f"  Max Length: {lengths.max():.6f}")
    print(f"  Avg Length: {lengths.mean():.6f}")

    print("\n" + "=" * 72)
    print("STEP 3: CENTROID DUAL (KAGOME HONEYCOMB)")
    print("=" * 72)
    dual_nodes, dual_struts = generate_centroid_dual(points, clean_simplices)
    print(f"Total Kagome Nodes: {dual_nodes.shape[0]}")
    print(f"Total Kagome Struts: {dual_struts.shape[0]}")

    if dual_struts.shape[0] > 0:
        dual_lengths = edge_lengths(dual_nodes, dual_struts)
        print("Kagome Strut Lengths")
        print(f"  Min Length: {dual_lengths.min():.6f}")
        print(f"  Max Length: {dual_lengths.max():.6f}")
        print(f"  Avg Length: {dual_lengths.mean():.6f}")
    else:
        print("Kagome Strut Lengths")
        print("  No struts generated.")

    if dual_nodes.shape[0] > 0:
        plot_dual_lattice(dual_nodes, dual_struts)

    print("\nDone.")


if __name__ == "__main__":
    main()
