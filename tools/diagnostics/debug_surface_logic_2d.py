"""
Debug Surface Dual Logic — 2D visualization of top face (Z=max).

Verbose audit: face indices, shared vertices, directional arrows.

Run from project root: python tools/diagnostics/debug_surface_logic_2d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from scaffold_module import generate_conformal_scaffold
from topology_module import get_surface_face_adjacency, get_surface_face_adjacency_verbose


def main() -> None:
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    scaffold = generate_conformal_scaffold(
        mesh=boundary_mesh,
        target_element_size=5.0,
    )

    nodes = scaffold.nodes
    surface_faces = scaffold.surface_faces

    z_max = float(np.max(nodes[:, 2]))
    tol = 0.01

    v0 = nodes[surface_faces[:, 0]]
    v1 = nodes[surface_faces[:, 1]]
    v2 = nodes[surface_faces[:, 2]]
    on_top = (
        (np.abs(v0[:, 2] - z_max) < tol)
        & (np.abs(v1[:, 2] - z_max) < tol)
        & (np.abs(v2[:, 2] - z_max) < tol)
    )

    top_faces = surface_faces[on_top]
    n_top_faces = top_faces.shape[0]

    print(f"Z-max plane:        z = {z_max:.2f}")
    print(f"Surface faces on top: {n_top_faces}")

    adj_raw, shared_edges_raw = get_surface_face_adjacency_verbose(top_faces)
    adj = np.unique(adj_raw, axis=0)
    n_adj = adj.shape[0]
    print(f"Adjacent pairs:     {n_adj} (raw: {adj_raw.shape[0]})")

    # Verbose: first 10 pairs with algorithm's shared edge vs actual intersection
    print("\nFirst 10 adjacent pairs (algo shared edge vs actual intersection):")
    bad_pairs = []
    for k in range(min(10, adj_raw.shape[0])):
        i, j = int(adj_raw[k, 0]), int(adj_raw[k, 1])
        algo_edge = tuple(shared_edges_raw[k])
        actual_shared = sorted(set(top_faces[i]) & set(top_faces[j]))
        if set(algo_edge) != set(actual_shared):
            bad_pairs.append((i, j, algo_edge, actual_shared))
        print(f"  Pair: Face {i} and Face {j}. Algo edge: {algo_edge}. Actual shared: {actual_shared}")

    # Full audit
    for k in range(adj_raw.shape[0]):
        i, j = int(adj_raw[k, 0]), int(adj_raw[k, 1])
        algo_edge = set(shared_edges_raw[k])
        actual_shared = set(top_faces[i]) & set(top_faces[j])
        if algo_edge != actual_shared or len(actual_shared) != 2:
            bad_pairs.append((i, j, tuple(algo_edge), sorted(actual_shared)))
    if bad_pairs:
        print(f"\nWARNING: {len(bad_pairs)} pairs have mismatch:")
        for i, j, algo, actual in bad_pairs[:2]:
            print(f"  Face {i}-{j}: algo={algo} actual={actual}")
            print(f"    Face {i} verts: {list(top_faces[i])} has edge {algo}? {set(algo) <= set(top_faces[i])}")
            print(f"    Face {j} verts: {list(top_faces[j])} has edge {algo}? {set(algo) <= set(top_faces[j])}")
        # Which faces actually have the first algo edge?
        algo_edge = set(bad_pairs[0][2])
        faces_with_edge = [idx for idx in range(n_top_faces) if algo_edge <= set(top_faces[idx])]
        print(f"    Faces with edge {tuple(algo_edge)}: {faces_with_edge}")
    else:
        print("\nAudit OK: all pairs share exactly 2 vertices.")

    centroids = (
        nodes[top_faces[:, 0]] + nodes[top_faces[:, 1]] + nodes[top_faces[:, 2]]
    ) / 3.0

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # A. Triangles (light gray)
    for tri in top_faces:
        pts = nodes[tri]
        pts_closed = np.vstack((pts, pts[0]))
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], color="lightgray", linewidth=0.8)

    # B. Face centroids (red dots) + index labels (blue text)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="red", s=20, zorder=5)
    for idx in range(n_top_faces):
        ax.text(
            centroids[idx, 0],
            centroids[idx, 1],
            str(idx),
            fontsize=8,
            color="blue",
            ha="center",
            va="center",
            zorder=6,
        )

    # C. Dual struts with arrowheads (direction i -> j)
    for i, j in adj:
        cx_i, cy_i = centroids[i, 0], centroids[i, 1]
        cx_j, cy_j = centroids[j, 0], centroids[j, 1]
        ax.annotate(
            "",
            xy=(cx_j, cy_j),
            xytext=(cx_i, cy_i),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
            zorder=4,
        )

    ax.set_aspect("equal")
    ax.set_title(f"Top Face (z={z_max:.1f}): {n_top_faces} faces, {n_adj} adjacent pairs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    plt.savefig("debug_surface_logic_2d.png", dpi=150)
    plt.close()
    print("\nSaved: debug_surface_logic_2d.png")


if __name__ == "__main__":
    main()
