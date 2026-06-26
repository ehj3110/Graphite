"""
Graphite Explicit Engine - Hexahedral Topology Synthesis

This module provides the global topology generation functions for assembling 
strut-based lattice graphs (nodes and struts) from conformal hexahedral grids.
It aggregates local topological rules (e.g. octahedral, grid, face dual) across 
a continuous field of block elements.
"""
from __future__ import annotations

import numpy as np

from .hex_rules import (
    _ADJACENT_FACE_PAIRS,
    _HEX_FACES,
    apply_hex_a15_kagome,
    apply_hex_dual,
    apply_hex_face_dual,
    apply_hex_grid,
    apply_hex_kelvin,
    apply_hex_kelvin14,
    apply_hex_octahedral,
    apply_hex_octet_truss,
    apply_hex_star,
    apply_hex_tesseract,
)

_HEX_RULES = {
    "grid": apply_hex_grid,
    "octahedral": apply_hex_octahedral,
    "star": apply_hex_star,
    "octet": apply_hex_octet_truss,
    "kelvin14": apply_hex_kelvin14,
    "kelvin_14": apply_hex_kelvin14,
    "kelvin": apply_hex_kelvin,
    "tesseract": apply_hex_tesseract,
    "nested_cube": apply_hex_tesseract,
    "hypercube": apply_hex_tesseract,
    "hex_dual": apply_hex_dual,
    "hex_face_dual": apply_hex_face_dual,
    "a15_kagome": apply_hex_a15_kagome,
}


def generate_hex_topology(
    hex_elements: np.ndarray,
    rule_name: str = "octahedral",
    round_decimals: int = 6,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a global lattice topology (nodes and struts) from a set of hex elements.
    
    This function applies a local topology rule to every element in the grid and
    merges coincident nodes globally to ensure continuous connectivity.

    Rules:
    - 'grid': Connects hex corners. Perfectly conformal for Route 3 conformed scaffolds.
    - 'hex_dual': Centroid-to-centroid adjacency across shared faces. Includes surface cage.
    - 'hex_face_dual': Octahedral pattern connecting centers of all 6 quad faces.
    - 'octet': Hybrid corner + face-center connectivity.
    - 'kelvin14' / 'kelvin' / 'kelvin_14': 24-node truncated octahedron (36 struts).
    - 'tesseract' / 'nested_cube' / 'hypercube': 16-node nested cube (32 struts).
    - 'star': Centroid to all 8 corners.
    - 'octahedral': 6 face centers with octahedral edge connectivity.

    Parameters
    ----------
    hex_elements : ndarray
        (N, 8, 3) array of element corner coordinates.
    rule_name : str, optional
        String identifier for the local cell rule, by default "octahedral".
    round_decimals : int, optional
        Precision for coordinating merging based on floating point equality, 
        by default 6.
    kwargs : dict
        Additional parameters forwarded to local rule functions (e.g. sdf_sampler).

    Returns
    -------
    nodes : ndarray
        (V, 3) array of unique global node coordinates.
    struts : ndarray
        (E, 2) array of global strut indices connecting the nodes.

    Raises
    ------
    ValueError
        If the shape of hex_elements is invalid or if the rule_name is not supported.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")
    rule_key = str(rule_name).strip().lower()
    if rule_key not in _HEX_RULES:
        raise ValueError(f"Unsupported hex rule '{rule_name}'. Options: {sorted(_HEX_RULES)}")
    rule_fn = _HEX_RULES[rule_key]

    if rule_key == "hex_dual":
        return _generate_hex_dual_topology(elems, round_decimals=round_decimals)

    import inspect
    sig = inspect.signature(rule_fn)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        rule_kwargs = kwargs
    else:
        rule_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()

    for elem in elems:
        local_nodes, local_struts = rule_fn(elem, **rule_kwargs)
        local_to_global: list[int] = []
        for node in local_nodes:
            key = tuple(np.round(node, round_decimals).tolist())
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(node.copy())
            local_to_global.append(idx)
        for a, b in local_struts:
            ga = local_to_global[int(a)]
            gb = local_to_global[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            strut_set.add((ga, gb))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = np.array(sorted(strut_set), dtype=np.int64) if strut_set else np.empty((0, 2), dtype=np.int64)
    return nodes, struts


def _generate_hex_dual_topology(
    hex_elements: np.ndarray, round_decimals: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")

    # 1. Internal centroids (one per hex cell)
    centroids = np.mean(elems, axis=1)
    nodes_list = [np.round(centroids, round_decimals).astype(np.float64)]

    # Build coordinate-aware global corner-node IDs for face deduplication.
    flat = elems.reshape(-1, 3)
    rounded = np.round(flat, round_decimals)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    elem_corner_ids = inverse.reshape(-1, 8)

    face_idx = np.array([
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]
    ], dtype=np.int32)

    face_to_owners: dict[tuple[int, int, int, int], list[int]] = {}
    
    # Track all faces and their owner(s)
    for hi, corners in enumerate(elem_corner_ids):
        for face in face_idx:
            key = tuple(sorted(int(corners[i]) for i in face))
            if key not in face_to_owners:
                face_to_owners[key] = []
            face_to_owners[key].append(hi)

    edge_set: set[tuple[int, int]] = set()
    boundary_nodes: list[np.ndarray] = []
    
    n_internal = len(centroids)
    
    for key, owners in face_to_owners.items():
        if len(owners) == 2:
            # Internal shared face: connect two centroids
            a, b = owners[0], owners[1]
            if a != b:
                edge_set.add((min(a, b), max(a, b)))
        elif len(owners) == 1:
            # Exposed boundary face: connect centroid to face center
            hi = owners[0]
            # Need actual coordinates for face center calculation
            # Use any owner to get the face coordinates
            face_nodes_coords = elems[hi][list(np.where(np.isin(elem_corner_ids[hi], key))[0])]
            # Fallback: find the face using the key in the element's local corner map
            # (Strictly, the key matches elem_corner_ids[hi][face_idx_row])
            f_center = np.mean(face_nodes_coords, axis=0)
            
            b_idx = n_internal + len(boundary_nodes)
            boundary_nodes.append(f_center)
            edge_set.add((hi, b_idx))

    # Combine internal centroids and boundary face centers
    final_nodes = np.vstack([nodes_list[0]] + boundary_nodes) if boundary_nodes else nodes_list[0]
    final_struts = np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.empty((0, 2), dtype=np.int64)
    
    return final_nodes, final_struts


def generate_hex_octahedral_volume_with_boundary_face_map(
    hex_elements: np.ndarray,
    *,
    volume_emit_mask: np.ndarray | None = None,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """
    Octahedral volume lattice with integer ``(hex_index, face_index)`` -> global node map
    for exterior faces only.

    Args:
        hex_elements: (N, 8, 3) brick coordinates.
        volume_emit_mask: if set, only hexes with True emit interior struts; face nodes
            are still registered for boundary faces on all hexes.

    Returns:
        nodes, struts, boundary_face_to_global_node
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")
    n_hex = elems.shape[0]
    if n_hex == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {},
        )

    flat = elems.reshape(-1, 3)
    rounded = np.round(flat, round_decimals)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    elem_corner_ids = inverse.reshape(-1, 8).astype(np.int32)

    face_count: dict[tuple[int, int, int, int], int] = {}
    for corners in elem_corner_ids:
        for face in _HEX_FACES:
            key = tuple(sorted(int(corners[i]) for i in face))
            face_count[key] = face_count.get(key, 0) + 1

    emit = (
        np.ones(n_hex, dtype=bool)
        if volume_emit_mask is None
        else np.asarray(volume_emit_mask, dtype=bool).ravel()
    )
    if emit.shape[0] != n_hex:
        raise ValueError("volume_emit_mask length must match hex count.")

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    boundary_face_to_node: dict[tuple[int, int], int] = {}

    def _global_index(pt: np.ndarray) -> int:
        key = tuple(np.round(pt, round_decimals).tolist())
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes_list)
            node_map[key] = idx
            nodes_list.append(np.asarray(pt, dtype=np.float64))
        return idx

    for hi, coords in enumerate(elems):
        face_globals: list[int] = []
        for fi, face in enumerate(_HEX_FACES):
            fc = coords[list(face)].mean(axis=0)
            gid = _global_index(fc)
            face_globals.append(gid)
            key = tuple(sorted(int(elem_corner_ids[hi, i]) for i in face))
            if face_count.get(key, 0) == 1:
                boundary_face_to_node[(int(hi), int(fi))] = gid

        if not emit[hi]:
            continue
        for a, b in _ADJACENT_FACE_PAIRS:
            ga = face_globals[int(a)]
            gb = face_globals[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            strut_set.add((ga, gb))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts, boundary_face_to_node


def filter_struts_drop_exterior_shell_pairs(
    struts: np.ndarray,
    boundary_face_node_ids: set[int] | frozenset[int],
) -> tuple[np.ndarray, int]:
    """
    Remove volume struts whose endpoints are both exterior boundary face centers.

    In two-branch export the skin branch owns the surface dual; keeping these
    struts duplicates the conformal/integer face-center cage on the envelope.
    """
    if struts.size == 0:
        return struts, 0
    bset = boundary_face_node_ids
    kept: list[tuple[int, int]] = []
    dropped = 0
    for a, b in np.asarray(struts, dtype=np.int64):
        ia, ib = int(a), int(b)
        if ia in bset and ib in bset:
            dropped += 1
            continue
        kept.append((ia, ib))
    out = (
        np.array(kept, dtype=np.int64)
        if kept
        else np.empty((0, 2), dtype=np.int64)
    )
    return out, int(dropped)
