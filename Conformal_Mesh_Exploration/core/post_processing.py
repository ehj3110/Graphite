"""
Post-processing for conformal lattice meshes.

- relax_hex_grid: Masked Laplacian smoothing to reduce boundary distortion.
- discretize_boundary_struts: splits boundary struts into multiple segments
  and snaps intermediate points to the CAD surface.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import trimesh


# 12 edges of a standard 8-node hex
_HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

# 6 faces of a standard 8-node hex (4 vertices each)
_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (3, 2, 6, 7),
    (0, 3, 7, 4),
    (1, 2, 6, 5),
)


def relax_hex_grid(
    hexes: np.ndarray,
    max_depth: int = 2,
    iterations: int = 10,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Masked Laplacian smoothing on hex grid interior near the boundary.

    Boundary nodes (depth 0) are locked. Nodes at depth 1..max_depth are
    relaxed toward the average of their neighbors over multiple iterations.

    Parameters
    ----------
    hexes : ndarray, shape (N, 8, 3)
    max_depth : int
        How many layers inward from the boundary to smooth.
    iterations : int
        Number of Laplacian smoothing passes.
    alpha : float
        Blend factor (0 = no move, 1 = full move to neighbor average).

    Returns
    -------
    relaxed_hexes : ndarray, shape (N, 8, 3)
    """
    # Step 1: Flatten to unique nodes
    flat = hexes.reshape(-1, 3)
    unique_nodes, inverse_indices = np.unique(flat, axis=0, return_inverse=True)
    connectivity = inverse_indices.reshape(-1, 8)
    n_unique = unique_nodes.shape[0]

    # Step 2: Build adjacency list and find boundary nodes
    adj_list: dict[int, set[int]] = defaultdict(set)
    face_counts: dict[tuple, int] = {}
    face_to_nodes: dict[tuple, list[int]] = {}

    for elem in connectivity:
        for a, b in _HEX_EDGES:
            adj_list[elem[a]].add(elem[b])
            adj_list[elem[b]].add(elem[a])
        for face in _HEX_FACES:
            face_nodes = tuple(sorted(elem[v] for v in face))
            face_counts[face_nodes] = face_counts.get(face_nodes, 0) + 1
            face_to_nodes[face_nodes] = [int(elem[v]) for v in face]

    boundary_node_set: set[int] = set()
    for face_key, count in face_counts.items():
        if count == 1:
            boundary_node_set.update(face_to_nodes[face_key])

    depth_0_nodes = np.array(sorted(boundary_node_set), dtype=np.int64)

    # Step 3: BFS depth calculation
    depths = np.full(n_unique, np.inf)
    depths[depth_0_nodes] = 0

    q: deque[int] = deque(depth_0_nodes.tolist())
    while q:
        node = q.popleft()
        next_depth = depths[node] + 1
        for neighbor in adj_list[node]:
            if next_depth < depths[neighbor]:
                depths[neighbor] = next_depth
                q.append(neighbor)

    # Depth distribution diagnostic
    unique_depths, counts = np.unique(depths, return_counts=True)
    print("  Node Depth Distribution:")
    for d, c in zip(unique_depths, counts):
        if d == float('inf'):
            print(f"    Depth INF: {c} nodes")
        else:
            print(f"    Depth {int(d)}: {c} nodes")

    # Step 4: Laplacian smoothing
    unlocked = np.where((depths > 0) & (depths <= max_depth))[0]

    n_unlocked = len(unlocked)
    print(f"  Laplacian Smoothing: {n_unlocked} unlocked nodes"
          f" (depth 1..{max_depth}), {iterations} iterations, alpha={alpha}")

    for _ in range(iterations):
        new_nodes = unique_nodes.copy()
        for ni in unlocked:
            neighbors = adj_list[ni]
            if not neighbors:
                continue
            mean_pos = unique_nodes[list(neighbors)].mean(axis=0)
            new_nodes[ni] = (1.0 - alpha) * unique_nodes[ni] + alpha * mean_pos
        unique_nodes = new_nodes

    # Step 5: Reconstruct
    relaxed_hexes = unique_nodes[connectivity]
    return relaxed_hexes


def discretize_boundary_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    mesh: trimesh.Trimesh,
    target_size: float,
    segments: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split boundary struts into multiple segments with inner points snapped
    to the CAD surface.

    A strut is classified as "boundary" if its midpoint is within
    target_size * 0.35 of the mesh surface.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    struts : ndarray, shape (S, 2)
    mesh : trimesh.Trimesh
        The boundary surface to snap to.
    target_size : float
        Voxel cell size (used to threshold boundary proximity).
    segments : int
        Number of sub-segments per boundary strut (default 4).

    Returns
    -------
    new_nodes : ndarray, shape (N', 3)
    new_struts : ndarray, shape (S', 2)
    """
    nodes_list = [pt.copy() for pt in np.asarray(nodes, dtype=np.float64)]
    struts_arr = np.asarray(struts, dtype=np.int64)

    nodes_np = np.array(nodes_list, dtype=np.float64)
    midpoints = (nodes_np[struts_arr[:, 0]] + nodes_np[struts_arr[:, 1]]) / 2.0

    _, distances, _ = trimesh.proximity.closest_point(mesh, midpoints)
    boundary_mask = distances < (target_size * 0.35)

    n_boundary = int(np.sum(boundary_mask))
    n_interior = len(struts_arr) - n_boundary

    new_struts: list[list[int]] = []

    for i, (a, b) in enumerate(struts_arr):
        if not boundary_mask[i]:
            new_struts.append([int(a), int(b)])
            continue

        pt_a = np.array(nodes_list[a])
        pt_b = np.array(nodes_list[b])
        chain_pts = np.linspace(pt_a, pt_b, segments + 1)
        inner_pts = chain_pts[1:-1]

        if len(inner_pts) > 0:
            snapped_inner, _, _ = trimesh.proximity.closest_point(mesh, inner_pts)
        else:
            snapped_inner = inner_pts

        chain_indices = [int(a)]
        for snapped_pt in snapped_inner:
            new_idx = len(nodes_list)
            nodes_list.append(snapped_pt.copy())
            chain_indices.append(new_idx)
        chain_indices.append(int(b))

        for j in range(len(chain_indices) - 1):
            new_struts.append([chain_indices[j], chain_indices[j + 1]])

    print(f"  Strut Discretization: {n_interior} interior + {n_boundary} boundary"
          f" -> {len(new_struts)} total struts ({len(nodes_list)} nodes)")

    return np.array(nodes_list, dtype=np.float64), np.array(new_struts, dtype=np.int64)
