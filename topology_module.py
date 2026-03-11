"""
Graphite Topology Module — Lattice Topology Synthesis

This module converts a tetrahedral scaffold into a strut graph with four
recipes: rhombic, voronoi, kagome, and icosahedral.

Universal Surface Dual (Day 2):
- Kagome/Voronoi use centroid-to-centroid for both internal and surface struts.
- Surface cage: pairwise connections between adjacent boundary face centroids,
  producing a consistent hexagonal skin. No Y-Skin or face-to-edge logic.

Connectivity fix:
- Global face mapping is created once from all tetrahedra.
- Each unique face owns exactly one centroid node.
- A face adjacency map (face_to_tets) is used by Voronoi so internal faces
  connect centroid-to-centroid across neighboring tetrahedra.

Safety check:
- Watershed connectivity filter removes disconnected floater components and
  keeps only the largest component before returning.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

# Canonical tetra entities
_EDGE_PAIRS = np.array(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
    dtype=np.int64,
)
_FACE_TRIPLETS = np.array(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    dtype=np.int64,
)
_ICOSA_FACE_EDGES = np.array(
    [[0, 1, 3], [0, 2, 4], [1, 2, 5], [3, 4, 5]],
    dtype=np.int64,
)


def _row_keys(a: np.ndarray) -> np.ndarray:
    """Create hashable row keys for vectorized row mapping/search."""
    b = np.ascontiguousarray(a)
    return b.view(np.dtype((np.void, b.dtype.itemsize * b.shape[1]))).ravel()


def _merge_short_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    target_element_size: float,
    min_length_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strut merger (de-clutter): merge nodes connected by struts shorter than
    min_length_ratio * target_element_size. Dissolves tiny geometry in concave corners.
    """
    if struts.shape[0] == 0 or target_element_size <= 0:
        return nodes, struts

    min_len = min_length_ratio * target_element_size
    a_xyz = nodes[struts[:, 0]]
    b_xyz = nodes[struts[:, 1]]
    lengths = np.linalg.norm(b_xyz - a_xyz, axis=1)
    short_mask = lengths < min_len

    if not np.any(short_mask):
        return nodes, struts

    # Union-find: merge nodes connected by short struts
    n = nodes.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in np.where(short_mask)[0]:
        union(int(struts[i, 0]), int(struts[i, 1]))

    # Canonical node per equivalence class
    canonical = np.array([find(i) for i in range(n)], dtype=np.int64)

    # Centroid of each class (for merged node position)
    class_sum = np.zeros((n, 3), dtype=np.float64)
    class_count = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c = canonical[i]
        class_sum[c] += nodes[i]
        class_count[c] += 1
    class_count[class_count == 0] = 1
    new_node_xyz = class_sum / class_count[:, None]

    # Build compact node list: one per unique canonical
    unique_can = np.unique(canonical)
    old_to_new = np.full(n, -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(unique_can):
        old_to_new[old_idx] = new_idx

    new_nodes = new_node_xyz[unique_can]
    new_struts = np.column_stack(
        (old_to_new[canonical[struts[:, 0]]], old_to_new[canonical[struts[:, 1]]])
    )

    # Remove degenerate and duplicate
    keep = new_struts[:, 0] != new_struts[:, 1]
    new_struts = new_struts[keep]
    new_struts = np.sort(new_struts, axis=1)
    new_struts = np.unique(new_struts, axis=0)

    merged_count = n - len(unique_can)
    if merged_count > 0:
        warnings.warn(
            f"Strut Merger: merged {merged_count} nodes from {int(np.sum(short_mask))} "
            f"short struts (<{min_length_ratio*100:.0f}% of element size).",
            UserWarning,
            stacklevel=2,
        )
    return new_nodes, new_struts


def _watershed_keep_largest(struts: np.ndarray, n_nodes: int) -> np.ndarray:
    """Keep only struts in the largest connected component."""
    if struts.shape[0] == 0 or n_nodes == 0:
        return struts

    rows = np.concatenate((struts[:, 0], struts[:, 1]))
    cols = np.concatenate((struts[:, 1], struts[:, 0]))
    data = np.ones(rows.size, dtype=np.float64)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    n_components, labels = csgraph.connected_components(
        graph, directed=False, return_labels=True
    )
    if n_components <= 1:
        return struts

    comp_ids, comp_sizes = np.unique(labels, return_counts=True)
    main_comp = comp_ids[np.argmax(comp_sizes)]

    keep = (labels[struts[:, 0]] == main_comp) & (labels[struts[:, 1]] == main_comp)
    filtered = struts[keep]

    removed = int(struts.shape[0] - filtered.shape[0])
    if removed > 0:
        warnings.warn(
            f"Watershed Check: Removed {removed} disconnected struts to ensure "
            "a manifold print.",
            UserWarning,
            stacklevel=2,
        )
    return filtered


def count_connected_components(struts: np.ndarray, n_nodes: int) -> int:
    """Count connected components containing at least one strut."""
    if struts.shape[0] == 0 or n_nodes == 0:
        return 0

    rows = np.concatenate((struts[:, 0], struts[:, 1]))
    cols = np.concatenate((struts[:, 1], struts[:, 0]))
    data = np.ones(rows.size, dtype=np.float64)
    graph = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    n_components, labels = csgraph.connected_components(
        graph, directed=False, return_labels=True
    )
    has_strut = np.zeros(n_components, dtype=bool)
    has_strut[labels[struts[:, 0]]] = True
    return int(np.sum(has_strut))


def get_surface_face_adjacency_verbose(
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like get_surface_face_adjacency but also returns the shared edge (v0, v1)
    for each pair. Uses explicit edge->faces dict for correctness.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    # Build edge -> list of face indices (exactly 2 = shared edge)
    edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi in range(surface_faces.shape[0]):
        tri = surface_faces[fi]
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            va, vb = int(tri[a]), int(tri[b])
            edge = (min(va, vb), max(va, vb))
            edge_to_faces[edge].append(fi)

    pairs_list = []
    shared_list = []
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            i, j = min(faces), max(faces)
            pairs_list.append((i, j))
            shared_list.append(edge)

    if not pairs_list:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    return np.array(pairs_list, dtype=np.int64), np.array(shared_list, dtype=np.int64)


def generate_surface_dual_cage(
    surface_faces: np.ndarray,
    face_to_node_id: np.ndarray,
    centroid_coords: np.ndarray | None = None,
    target_element_size: float | None = None,
) -> np.ndarray:
    """
    Universal Surface Dual: centroid-to-centroid struts for adjacent surface faces.

    Uses get_surface_face_adjacency and face_to_node_id to build struts.
    Optional distance filter: skip struts longer than 1.5 * target_element_size.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)
    if face_to_node_id.shape[0] != surface_faces.shape[0]:
        raise ValueError(
            f"face_to_node_id length {face_to_node_id.shape[0]} must match "
            f"surface_faces {surface_faces.shape[0]}."
        )

    adj = get_surface_face_adjacency(surface_faces)
    if adj.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    max_dist = None
    if target_element_size is not None and centroid_coords is not None:
        max_dist = 1.5 * float(target_element_size)

    struts_list = []
    for i, j in adj:
        node_a = int(face_to_node_id[i])
        node_b = int(face_to_node_id[j])
        if max_dist is not None and centroid_coords is not None:
            d = float(np.linalg.norm(centroid_coords[i] - centroid_coords[j]))
            if d > max_dist:
                continue
        struts_list.append((min(node_a, node_b), max(node_a, node_b)))

    if not struts_list:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.array(struts_list, dtype=np.int64), axis=0)


def get_surface_face_adjacency(surface_faces: np.ndarray) -> np.ndarray:
    """
    Map which surface triangles share an edge.

    Edges are identified by EXACTLY TWO matching vertex IDs (canonical pair).
    Returns an (E, 2) array of face index pairs (i, j) with i < j,
    one row per shared edge. Faces that share an edge are neighbors.

    Vertex IDs must be consistent: the same geometric point must use the same
    index in all faces. Duplicate vertices (same coord, different ID) cause
    ghost adjacencies or missed neighbors.
    """
    pairs, _ = get_surface_face_adjacency_verbose(surface_faces)
    if pairs.shape[0] == 0:
        return pairs
    return np.unique(pairs, axis=0)


def _build_surface_face_to_node_map(
    surface_faces: np.ndarray,
    unique_faces: np.ndarray,
    off_faces: int,
    face_cent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map each surface face index to (node_id, centroid_coords).
    Returns (face_to_node_id, centroid_coords) for use with generate_surface_dual_cage.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    sf_canon = np.sort(surface_faces, axis=1)
    face_keys = _row_keys(unique_faces)
    sf_keys = _row_keys(sf_canon)
    face_order = np.argsort(face_keys)
    face_sorted = face_keys[face_order]
    face_pos = np.searchsorted(face_sorted, sf_keys)
    face_valid = face_pos < face_sorted.shape[0]
    face_pos_safe = np.clip(face_pos, 0, face_sorted.shape[0] - 1)
    face_match = face_valid & (face_sorted[face_pos_safe] == sf_keys)
    if not np.all(face_match):
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    unique_face_ids = face_order[face_pos_safe]
    face_to_node_id = off_faces + unique_face_ids
    centroid_coords = face_cent[unique_face_ids]
    return face_to_node_id, centroid_coords


def _surface_cage_vertex_edges(surface_faces: np.ndarray) -> np.ndarray:
    e01 = surface_faces[:, [0, 1]]
    e12 = surface_faces[:, [1, 2]]
    e20 = surface_faces[:, [2, 0]]
    return np.vstack((e01, e12, e20))


def _surface_skin_face_centroid_to_edge_midpoints(
    surface_faces: np.ndarray,
    unique_faces: np.ndarray,
    off_faces: int,
    unique_edges: np.ndarray,
    off_edges: int,
) -> np.ndarray:
    """
    Build Y-shaped surface skin struts:
        for each surface triangle face, connect its face-centroid node to the
        three edge-midpoint nodes of that same face.

    This avoids neighbor-search logic and uses only local face geometry.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    sf_canon = np.sort(surface_faces, axis=1)

    # Map surface face -> global unique face id
    face_keys = _row_keys(unique_faces)
    sf_keys = _row_keys(sf_canon)
    face_order = np.argsort(face_keys)
    face_sorted = face_keys[face_order]
    face_pos = np.searchsorted(face_sorted, sf_keys)
    face_valid = face_pos < face_sorted.shape[0]
    face_pos_safe = np.clip(face_pos, 0, face_sorted.shape[0] - 1)
    face_match = face_valid & (face_sorted[face_pos_safe] == sf_keys)
    if not np.all(face_match):
        return np.empty((0, 2), dtype=np.int64)
    face_ids = face_order[face_pos_safe]
    face_centroid_ids = off_faces + face_ids  # (K,)

    # Build the 3 edges of each surface face and map to unique edge ids
    e01 = np.sort(surface_faces[:, [0, 1]], axis=1)
    e12 = np.sort(surface_faces[:, [1, 2]], axis=1)
    e20 = np.sort(surface_faces[:, [2, 0]], axis=1)
    face_edges = np.stack((e01, e12, e20), axis=1)  # (K, 3, 2)

    edge_keys = _row_keys(unique_edges)
    fe_keys = _row_keys(face_edges.reshape(-1, 2))
    edge_order = np.argsort(edge_keys)
    edge_sorted = edge_keys[edge_order]
    edge_pos = np.searchsorted(edge_sorted, fe_keys)
    edge_valid = edge_pos < edge_sorted.shape[0]
    edge_pos_safe = np.clip(edge_pos, 0, edge_sorted.shape[0] - 1)
    edge_match = edge_valid & (edge_sorted[edge_pos_safe] == fe_keys)
    if not np.all(edge_match):
        return np.empty((0, 2), dtype=np.int64)
    edge_ids = edge_order[edge_pos_safe].reshape(-1, 3)
    edge_midpoint_ids = off_edges + edge_ids  # (K, 3)

    # Y-shapes: centroid -> each midpoint
    c = face_centroid_ids[:, None]
    struts = np.vstack(
        (
            np.column_stack((c[:, 0], edge_midpoint_ids[:, 0])),
            np.column_stack((c[:, 0], edge_midpoint_ids[:, 1])),
            np.column_stack((c[:, 0], edge_midpoint_ids[:, 2])),
        )
    )
    struts = np.sort(struts, axis=1)
    return np.unique(struts, axis=0)


def _surface_dual_face_centroid_struts(
    surface_faces: np.ndarray,
    unique_faces: np.ndarray,
    off_faces: int,
    nodes: np.ndarray,
    coplanar_cos_threshold: float = 0.99,
) -> np.ndarray:
    """
    Build straight surface-dual struts for Kagome skin:
      - connect centroids of adjacent surface faces that share an edge
      - only keep adjacent pairs with near-coplanar normals
      - if a face has no coplanar neighbor, add one closure strut from that
        face centroid to a boundary vertex node.
    """
    if surface_faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    sf_canon = np.sort(surface_faces, axis=1)

    # Map each surface face to its global unique-face centroid node id.
    uf_keys = _row_keys(unique_faces)
    sf_keys = _row_keys(sf_canon)
    order = np.argsort(uf_keys)
    sorted_keys = uf_keys[order]
    pos = np.searchsorted(sorted_keys, sf_keys)
    valid = pos < sorted_keys.shape[0]
    pos_safe = np.clip(pos, 0, sorted_keys.shape[0] - 1)
    matched = valid & (sorted_keys[pos_safe] == sf_keys)
    if not np.all(matched):
        return np.empty((0, 2), dtype=np.int64)
    face_ids = order[pos_safe]
    centroid_node_ids = off_faces + face_ids  # (K,)

    # Build adjacent face pairs through shared surface edges.
    e01 = np.sort(surface_faces[:, [0, 1]], axis=1)
    e12 = np.sort(surface_faces[:, [1, 2]], axis=1)
    e20 = np.sort(surface_faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    owner_faces = np.repeat(np.arange(surface_faces.shape[0], dtype=np.int64), 3)

    _, inv = np.unique(all_edges, axis=0, return_inverse=True)
    ord_e = np.argsort(inv)
    inv_s = inv[ord_e]
    face_s = owner_faces[ord_e]

    _, starts, counts = np.unique(inv_s, return_index=True, return_counts=True)
    two_mask = counts == 2
    if not np.any(two_mask):
        # No adjacent surface faces; close each face centroid to one boundary vertex.
        closure = np.column_stack(
            (centroid_node_ids, surface_faces[:, 0].astype(np.int64))
        )
        closure = np.sort(closure, axis=1)
        return np.unique(closure, axis=0)

    f0 = face_s[starts[two_mask]]
    f1 = face_s[starts[two_mask] + 1]

    # Coplanarity filter using face normals.
    tri = nodes[surface_faces]  # (K, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_unit = n / np.maximum(n_norm, 1e-12)
    dot = np.abs(np.sum(n_unit[f0] * n_unit[f1], axis=1))
    coplanar = dot >= float(coplanar_cos_threshold)

    dual_pairs = np.empty((0, 2), dtype=np.int64)
    if np.any(coplanar):
        c0 = centroid_node_ids[f0[coplanar]]
        c1 = centroid_node_ids[f1[coplanar]]
        dual_pairs = np.column_stack((np.minimum(c0, c1), np.maximum(c0, c1)))

    # Closure for faces without any coplanar dual neighbor.
    degree = np.zeros(surface_faces.shape[0], dtype=np.int64)
    if np.any(coplanar):
        np.add.at(degree, f0[coplanar], 1)
        np.add.at(degree, f1[coplanar], 1)
    isolated = degree == 0
    closure = np.empty((0, 2), dtype=np.int64)
    if np.any(isolated):
        closure = np.column_stack(
            (
                centroid_node_ids[isolated],
                surface_faces[isolated, 0].astype(np.int64),
            )
        )
        closure = np.sort(closure, axis=1)

    out = np.vstack((dual_pairs, closure)) if closure.size else dual_pairs
    return np.unique(out, axis=0) if out.size else out


def _apply_surface_edge_collinearity_straightener(
    *,
    surface_faces: np.ndarray,
    unique_faces: np.ndarray,
    face_centroids: np.ndarray,
    unique_edges: np.ndarray,
    edge_midpoints: np.ndarray,
) -> np.ndarray:
    """
    Collinearity fix for Y-skin:
    - For each surface edge shared by two surface faces A/B,
      set edge midpoint coordinate to 0.5 * (centroid_A + centroid_B).
    - For corner edges (one face), keep standard midpoint.
    """
    if surface_faces.shape[0] == 0:
        return edge_midpoints

    # Map each surface face to global unique face id
    sf_canon = np.sort(surface_faces, axis=1)
    uf_keys = _row_keys(unique_faces)
    sf_keys = _row_keys(sf_canon)
    face_order = np.argsort(uf_keys)
    face_sorted = uf_keys[face_order]
    fpos = np.searchsorted(face_sorted, sf_keys)
    fvalid = fpos < face_sorted.shape[0]
    fpos_safe = np.clip(fpos, 0, face_sorted.shape[0] - 1)
    fmatch = fvalid & (face_sorted[fpos_safe] == sf_keys)
    if not np.all(fmatch):
        return edge_midpoints
    sf_face_ids = face_order[fpos_safe]  # (K,)

    # Surface edges + owner surface-face index
    e01 = np.sort(surface_faces[:, [0, 1]], axis=1)
    e12 = np.sort(surface_faces[:, [1, 2]], axis=1)
    e20 = np.sort(surface_faces[:, [2, 0]], axis=1)
    all_edges = np.vstack((e01, e12, e20))
    owner_surface_face = np.repeat(np.arange(surface_faces.shape[0], dtype=np.int64), 3)

    # Group by edge and keep only edges with exactly 2 incident surface faces
    _, inv = np.unique(all_edges, axis=0, return_inverse=True)
    ord_e = np.argsort(inv)
    inv_s = inv[ord_e]
    face_s = owner_surface_face[ord_e]
    _, starts, counts = np.unique(inv_s, return_index=True, return_counts=True)
    two_mask = counts == 2
    if not np.any(two_mask):
        return edge_midpoints

    f0_local = face_s[starts[two_mask]]
    f1_local = face_s[starts[two_mask] + 1]
    f0_global = sf_face_ids[f0_local]
    f1_global = sf_face_ids[f1_local]

    # Map each shared surface edge to global unique edge id
    shared_edges = all_edges[ord_e][starts[two_mask]]  # canonical (E2, 2)
    ue_keys = _row_keys(unique_edges)
    se_keys = _row_keys(shared_edges)
    edge_order = np.argsort(ue_keys)
    edge_sorted = ue_keys[edge_order]
    epos = np.searchsorted(edge_sorted, se_keys)
    evalid = epos < edge_sorted.shape[0]
    epos_safe = np.clip(epos, 0, edge_sorted.shape[0] - 1)
    ematch = evalid & (edge_sorted[epos_safe] == se_keys)
    if not np.all(ematch):
        return edge_midpoints
    shared_edge_ids = edge_order[epos_safe]

    # Straightened midpoint = midpoint between adjacent face centroids
    straight_mid = 0.5 * (face_centroids[f0_global] + face_centroids[f1_global])
    out_mid = edge_midpoints.copy()
    out_mid[shared_edge_ids] = straight_mid
    return out_mid


def _surface_cage_edge_midpoints(
    surface_faces: np.ndarray,
    unique_edges: np.ndarray,
    off_edges: int,
) -> np.ndarray:
    """Boundary cage using edge-midpoint triangles per boundary face."""
    if surface_faces.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int64)

    be01 = np.sort(surface_faces[:, [0, 1]], axis=1)
    be12 = np.sort(surface_faces[:, [1, 2]], axis=1)
    be20 = np.sort(surface_faces[:, [2, 0]], axis=1)
    face_edges = np.stack((be01, be12, be20), axis=1)  # (K, 3, 2)

    uniq_keys = _row_keys(unique_edges)
    q_keys = _row_keys(face_edges.reshape(-1, 2))

    order = np.argsort(uniq_keys)
    sorted_keys = uniq_keys[order]
    pos = np.searchsorted(sorted_keys, q_keys)
    valid = pos < sorted_keys.shape[0]
    pos_safe = np.clip(pos, 0, sorted_keys.shape[0] - 1)
    matched = valid & (sorted_keys[pos_safe] == q_keys)

    if not np.all(matched):
        return np.empty((0, 2), dtype=np.int64)

    edge_ids = order[pos_safe].reshape(-1, 3)
    mids = off_edges + edge_ids

    a = mids[:, [0, 1]]
    b = mids[:, [1, 2]]
    c = mids[:, [2, 0]]
    struts = np.vstack((a, b, c))
    struts = np.sort(struts, axis=1)
    return np.unique(struts, axis=0)


def generate_topology(
    nodes: np.ndarray,
    elements: np.ndarray,
    surface_faces: np.ndarray,
    type: str = "rhombic",
    topology_type: str | None = None,
    include_surface_cage: bool = True,
    target_element_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate topology nodes and struts for conformal tetra scaffold.

    Recipes:
    - rhombic: c_vol to each of 4 vertices (BCC style)
    - voronoi: face-adjacency map; internal face -> centroid-centroid, boundary
      face -> centroid-to-face-centroid (when include_surface_cage=True)
    - kagome: complete graph on 4 face-centroid nodes of each tet
    - icosahedral: per-face triangles on edge-midpoint nodes (12 struts/tet)
    """
    topo = topology_type if topology_type is not None else type
    if topo in ("vertex_to_centroid", "bcc_vertex_conformal"):
        topo = "rhombic"

    nodes_np = np.asarray(nodes, dtype=np.float64)
    elements_np = np.asarray(elements, dtype=np.int64)
    surface_faces_np = np.asarray(surface_faces, dtype=np.int64)

    if nodes_np.ndim != 2 or nodes_np.shape[1] != 3:
        raise ValueError(f"`nodes` must have shape (N, 3); got {nodes_np.shape}.")
    if elements_np.ndim != 2 or elements_np.shape[1] != 4:
        raise ValueError(f"`elements` must have shape (M, 4); got {elements_np.shape}.")
    if surface_faces_np.ndim != 2 or surface_faces_np.shape[1] != 3:
        raise ValueError(
            f"`surface_faces` must have shape (K, 3); got {surface_faces_np.shape}."
        )
    if topo not in {"rhombic", "voronoi", "kagome", "icosahedral"}:
        raise ValueError(
            f"Unsupported topology_type='{topo}'. Supported: "
            "'rhombic', 'voronoi', 'kagome', 'icosahedral'."
        )

    n_orig = nodes_np.shape[0]
    n_tets = elements_np.shape[0]
    if n_tets == 0:
        cage = _surface_cage_vertex_edges(surface_faces_np)
        cage = np.sort(cage, axis=1)
        cage = np.unique(cage, axis=0)
        return nodes_np, cage

    tet_xyz = nodes_np[elements_np]  # (M, 4, 3)

    # Unique global edges and midpoints
    tet_edges = np.sort(elements_np[:, _EDGE_PAIRS], axis=2)  # (M, 6, 2)
    flat_edges = tet_edges.reshape(-1, 2)
    unique_edges, edge_inverse = np.unique(flat_edges, axis=0, return_inverse=True)
    edge_inverse = edge_inverse.reshape(n_tets, 6)
    edge_mid = 0.5 * (nodes_np[unique_edges[:, 0]] + nodes_np[unique_edges[:, 1]])

    # Unique global faces and centroids
    tet_faces = np.sort(elements_np[:, _FACE_TRIPLETS], axis=2)  # (M, 4, 3)
    flat_faces = tet_faces.reshape(-1, 3)
    unique_faces, face_inverse = np.unique(flat_faces, axis=0, return_inverse=True)
    face_inverse = face_inverse.reshape(n_tets, 4)
    face_cent = (
        nodes_np[unique_faces[:, 0]]
        + nodes_np[unique_faces[:, 1]]
        + nodes_np[unique_faces[:, 2]]
    ) / 3.0

    # Global face adjacency map (face -> one or two tet indices)
    flat_face_ids = face_inverse.reshape(-1)
    owner_tets = np.repeat(np.arange(n_tets, dtype=np.int64), 4)
    ord_f = np.argsort(flat_face_ids)
    f_sorted = flat_face_ids[ord_f]
    t_sorted = owner_tets[ord_f]
    uniq_face_ids, starts, counts = np.unique(
        f_sorted, return_index=True, return_counts=True
    )
    first_tet = t_sorted[starts]
    second_tet = np.full_like(first_tet, -1)
    has_two = counts == 2
    second_tet[has_two] = t_sorted[starts[has_two] + 1]

    face_to_tets = {
        int(fid): ([int(t0), int(t1)] if t1 >= 0 else [int(t0)])
        for fid, t0, t1 in zip(uniq_face_ids, first_tet, second_tet)
    }
    _ = face_to_tets  # available for debugging/introspection

    # Node offsets in unified node table
    n_edges = unique_edges.shape[0]
    n_faces = unique_faces.shape[0]
    off_edges = n_orig
    off_faces = n_orig + n_edges
    off_vol = n_orig + n_edges + n_faces

    tet_edge_idx = off_edges + edge_inverse
    tet_face_idx = off_faces + face_inverse
    tet_vol_idx = off_vol + np.arange(n_tets, dtype=np.int64)

    # Internal struts by recipe
    if topo == "rhombic":
        internal = np.column_stack((elements_np.reshape(-1), np.repeat(tet_vol_idx, 4)))

    elif topo == "voronoi":
        # Face-adjacency driven Voronoi struts:
        # - two-tet face: connect volume centroids of neighboring tets
        # - one-tet face: connect volume centroid to boundary face centroid
        #   (only when include_surface_cage=True)
        boundary_face_ids = uniq_face_ids[~has_two]

        two_a = off_vol + first_tet[has_two]
        two_b = off_vol + second_tet[has_two]
        struts_internal = (
            np.column_stack((two_a, two_b))
            if two_a.size
            else np.empty((0, 2), dtype=np.int64)
        )

        if include_surface_cage:
            # Use face centroids (off_faces + boundary_face_ids) for boundary
            # so internal struts land on same nodes as Surface Dual cage.
            b_tet = off_vol + first_tet[~has_two]
            b_face = off_faces + boundary_face_ids
            struts_boundary = (
                np.column_stack((b_tet, b_face))
                if b_tet.size
                else np.empty((0, 2), dtype=np.int64)
            )
        else:
            struts_boundary = np.empty((0, 2), dtype=np.int64)

        internal = np.vstack((struts_internal, struts_boundary))

    elif topo == "kagome":
        pairs = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            dtype=np.int64,
        )
        a = tet_face_idx[:, pairs[:, 0]].reshape(-1)
        b = tet_face_idx[:, pairs[:, 1]].reshape(-1)
        internal = np.column_stack((a, b))

    else:  # icosahedral
        mids = tet_edge_idx[:, _ICOSA_FACE_EDGES]  # (M, 4, 3)
        p = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
        tri_edges = mids[:, :, p].reshape(-1, 2)
        internal = tri_edges

    # Unified node array for all recipes (simple, stable indexing)
    # Voronoi uses face_cent for boundary faces; no separate boundary_face_nodes.
    all_nodes = np.vstack((nodes_np, edge_mid, face_cent, np.mean(tet_xyz, axis=1)))

    # Type-specific boundary cage
    if include_surface_cage:
        if topo == "rhombic":
            cage = _surface_cage_vertex_edges(surface_faces_np)
        elif topo in ("voronoi", "kagome"):
            # Universal Surface Dual: centroid-to-centroid struts for adjacent faces
            face_to_node_id, centroid_coords = _build_surface_face_to_node_map(
                surface_faces_np, unique_faces, off_faces, face_cent
            )
            cage = generate_surface_dual_cage(
                surface_faces_np,
                face_to_node_id,
                centroid_coords=centroid_coords,
                target_element_size=target_element_size,
            )
        else:
            cage = _surface_cage_edge_midpoints(surface_faces_np, unique_edges, off_edges)
    else:
        cage = np.empty((0, 2), dtype=np.int64)

    # Deduplicate struts
    all_struts = np.vstack((internal, cage)) if cage.size else internal
    all_struts = np.sort(all_struts, axis=1)
    all_struts = np.unique(all_struts, axis=0)

    # Deduplicate nodes and remap struts
    rounded_nodes = np.round(all_nodes, decimals=6)
    unique_nodes, inv_nodes = np.unique(rounded_nodes, axis=0, return_inverse=True)
    remap_struts = inv_nodes[all_struts]
    remap_struts = np.sort(remap_struts, axis=1)
    remap_struts = np.unique(remap_struts, axis=0)

    # Strut merger: dissolve short struts (<20% element size) to fuse U-bend junctions
    if target_element_size is not None and target_element_size > 0:
        unique_nodes, remap_struts = _merge_short_struts(
            unique_nodes, remap_struts, target_element_size, min_length_ratio=0.20
        )

    # Watershed: keep largest connected component
    remap_struts = _watershed_keep_largest(remap_struts, unique_nodes.shape[0])

    return unique_nodes, remap_struts
