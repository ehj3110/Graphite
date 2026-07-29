"""
Graphite Explicit Engine - Supercell Oct-Tip Processing

Oct-void and sparse-cell protection for tet-oct Kagome culling.

When the boundary grazes octahedral voids or sharp tips, tet-only voting can drop
all tets in a parent cell while oct samples still overlap the part. This module
expands the kept-tet mask and optionally merges protected struts from the full
pre-cull graph for those cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Unit-cube face centers (6): oct-void proxies in each FCC parent cell of side L.
_OCT_VOID_LOCAL = np.array(
    [
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.5],
        [0.5, 1.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ],
    dtype=np.float64,
)


def _cell_lattice(tet_coords: np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Parent cell index (i,j,k) per tet and lattice origin."""
    tet_centroids = np.mean(tet_coords, axis=1)
    origin = np.min(tet_coords.reshape(-1, 3), axis=0)
    cell_idx = np.floor((tet_centroids - origin) / max(L, 1e-9) + 1e-6).astype(np.int32)
    return cell_idx, origin


def _build_cell_to_tets(cell_idx: np.ndarray, n_tets: int) -> dict[tuple[int, int, int], list[int]]:
    cell_to_tets: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n_tets):
        key = tuple(int(v) for v in cell_idx[i].tolist())
        cell_to_tets.setdefault(key, []).append(i)
    return cell_to_tets


def detect_oct_tip_regions(
    tet_coords: np.ndarray,
    votes: np.ndarray,
    is_kept: np.ndarray,
    sample_sdf,
    L: float,
    sparse_max_kept: int = 2,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], dict[str, Any]]:
    """
    Classify parent FCC cells into sparse (tip-like) and oct-overlap-only regions.

    Parameters
    ----------
    tet_coords : ndarray
        (N, 4, 3) coordinates for the tetrahedron elements.
    votes : ndarray
        (N,) vote counts for how many nodes of each tetrahedron are inside the volume.
    is_kept : ndarray
        (N,) boolean mask of currently kept tetrahedrons.
    sample_sdf : callable
        Function to evaluate the SDF at a set of given coordinates.
    L : float
        Tiling period of the lattice.
    sparse_max_kept : int, optional
        Maximum number of kept tets for a cell to be considered sparse, by default 2.

    Returns
    -------
    sparse_cells : set of tuple of int
        Set of cell (i, j, k) indices where kept tet count <= sparse_max_kept.
    oct_only_cells : set of tuple of int
        Set of cell indices where no tet passes the strong vote but at least one 
        oct-void sample is inside.
    report : dict
        Dictionary of counts and optional debug fields.
    """
    n_tets = tet_coords.shape[0]
    cell_idx, origin = _cell_lattice(tet_coords, L)
    cell_to_tets = _build_cell_to_tets(cell_idx, n_tets)

    sparse_cells: set[tuple[int, int, int]] = set()
    oct_only_cells: set[tuple[int, int, int]] = set()

    for key, tet_ids in cell_to_tets.items():
        kept_count = int(np.sum([is_kept[i] for i in tet_ids]))
        if kept_count <= sparse_max_kept:
            sparse_cells.add(key)

        i, j, k = key
        cell_origin = origin + np.array([i, j, k], dtype=np.float64) * L
        oct_pts = cell_origin + L * _OCT_VOID_LOCAL
        oct_sdfs = sample_sdf(oct_pts)
        oct_inside_any = bool(np.any(oct_sdfs <= 0.0))

        strong_kept = any(votes[tid] >= 3 for tid in tet_ids)
        if oct_inside_any and not strong_kept:
            oct_only_cells.add(key)

    report = {
        "sparse_cell_count": len(sparse_cells),
        "oct_only_cell_count": len(oct_only_cells),
        "union_cell_count": len(sparse_cells | oct_only_cells),
    }
    return sparse_cells, oct_only_cells, report


def expand_is_kept_oct_tip(
    is_kept: np.ndarray,
    votes: np.ndarray,
    tet_coords: np.ndarray,
    tet_elements: list,
    kag_struts: np.ndarray,
    sample_sdf,
    L: float,
    sparse_cells: set[tuple[int, int, int]],
    oct_only_cells: set[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Gently expand kept tets in protected cells so Kagome struts can survive at tips.

    Rules:
    - Union protected cells = sparse | oct_only
    - In protected cells: keep any tet with votes >= 1 (at least one corner inside)
    - In oct_only cells where every tet has votes == 0: keep the tet whose centroid
      has the best (most inside) SDF among oct sample points (single-tet salvage)
    """
    is_kept = np.asarray(is_kept, dtype=bool).copy()
    n_tets = len(is_kept)
    cell_idx, origin = _cell_lattice(tet_coords, L)
    cell_to_tets = _build_cell_to_tets(cell_idx, n_tets)

    protected = sparse_cells | oct_only_cells
    recovered_votes_ge1 = 0
    recovered_oct_salvage = 0

    for key in protected:
        tet_ids = cell_to_tets.get(key, [])
        if not tet_ids:
            continue

        for tid in tet_ids:
            if votes[tid] >= 1:
                if not is_kept[tid]:
                    is_kept[tid] = True
                    recovered_votes_ge1 += 1

        # Oct-only salvage: still nothing kept but oct overlapped part
        if key in oct_only_cells:
            still_none = not any(is_kept[tid] for tid in tet_ids)
            if still_none:
                best_tid = None
                best_sdf = np.inf
                for tid in tet_ids:
                    c = np.mean(tet_coords[tid], axis=0)
                    sdf_c = float(sample_sdf(c.reshape(1, 3))[0])
                    if sdf_c < best_sdf:
                        best_sdf = sdf_c
                        best_tid = tid
                if best_tid is not None:
                    is_kept[best_tid] = True
                    recovered_oct_salvage += 1

    # Merge protected struts: edges fully inside node union of tets in protected cells
    node_in_protected: set[int] = set()
    for key in protected:
        for tid in cell_to_tets.get(key, []):
            for nid in tet_elements[tid]:
                node_in_protected.add(int(nid))

    extra_struts: list[tuple[int, int]] = []
    if node_in_protected and len(kag_struts) > 0:
        ks = np.asarray(kag_struts, dtype=np.int64)
        for a, b in ks:
            ia, ib = int(a), int(b)
            if ia in node_in_protected and ib in node_in_protected:
                if ia > ib:
                    ia, ib = ib, ia
                extra_struts.append((ia, ib))

    report = {
        "recovered_votes_ge1": recovered_votes_ge1,
        "recovered_oct_salvage": recovered_oct_salvage,
        "protected_strut_pairs": len(extra_struts),
    }
    return is_kept, {"extra_struts": extra_struts, **report}


def classify_bridge_struts(kag_nodes: np.ndarray, kag_struts: np.ndarray, L: float) -> np.ndarray:
    """
    Label struts that are inter-tet bridges through octahedral voids (~length L/3).

    Intra-tet Kagome edges are typically shorter / different spacing than L/3.
    """
    if len(kag_struts) == 0:
        return np.empty((0,), dtype=bool)
    target = L / 3.0
    tol = 0.12 * L
    a = np.asarray(kag_nodes)[kag_struts[:, 0]]
    b = np.asarray(kag_nodes)[kag_struts[:, 1]]
    lengths = np.linalg.norm(b - a, axis=1)
    return np.abs(lengths - target) < tol


def bridge_endpoint_mask_old(kag_nodes: np.ndarray, kag_struts: np.ndarray, L: float) -> np.ndarray:
    """Per old Kagome node: True if incident to at least one bridge strut."""
    n = len(kag_nodes)
    mask = np.zeros(n, dtype=bool)
    if len(kag_struts) == 0:
        return mask
    is_bridge = classify_bridge_struts(kag_nodes, kag_struts, L)
    for k, (s, e) in enumerate(np.asarray(kag_struts, dtype=np.int64)):
        if is_bridge[k]:
            mask[int(s)] = True
            mask[int(e)] = True
    return mask


def tip_region_nodes_old(
    n_kag_nodes: int,
    tet_elements: list,
    cell_idx: np.ndarray,
    tip_union: set[tuple[int, int, int]],
) -> np.ndarray:
    """True for nodes that appear in any tet whose parent cell is in tip_union."""
    mask = np.zeros(n_kag_nodes, dtype=bool)
    for tid, elem in enumerate(tet_elements):
        key = tuple(int(v) for v in cell_idx[tid].tolist())
        if key not in tip_union:
            continue
        for nid in elem:
            mask[int(nid)] = True
    return mask


def drop_fully_outside_tets_in_tip_cells(
    is_kept: np.ndarray,
    votes: np.ndarray,
    cell_idx: np.ndarray,
    tip_union: set[tuple[int, int, int]],
) -> tuple[np.ndarray, int]:
    """
    In sparse/oct-tip parent cells, remove tets with no part contact (votes == 0).

    Avoids keeping whole tet Kagome subgraphs that only serve to flatten against
    the boundary; oct-bridge struts are handled separately in conform.
    """
    is_kept = np.asarray(is_kept, dtype=bool).copy()
    dropped = 0
    for tid in range(len(is_kept)):
        key = tuple(int(v) for v in cell_idx[tid].tolist())
        if key not in tip_union:
            continue
        if votes[tid] == 0 and is_kept[tid]:
            is_kept[tid] = False
            dropped += 1
    return is_kept, dropped


def map_old_node_mask_to_compressed(
    old_mask: np.ndarray, old_to_new: np.ndarray, n_new: int
) -> np.ndarray:
    """
    Map a boolean mask from the old node indexing to a new compressed indexing.

    Parameters
    ----------
    old_mask : ndarray
        The boolean mask array on the old nodes.
    old_to_new : ndarray
        The integer mapping array where index is old node ID and value is new node ID.
    n_new : int
        The number of nodes in the new compressed array.

    Returns
    -------
    ndarray
        The translated boolean mask for the new node array.
    """
    out = np.zeros(n_new, dtype=bool)
    for i, use in enumerate(old_mask):
        if not use:
            continue
        j = int(old_to_new[i])
        if j >= 0:
            out[j] = True
    return out


def merge_struts_with_protected_edges(
    new_struts: np.ndarray,
    old_to_new: np.ndarray,
    extra_struts_old: list[tuple[int, int]],
) -> np.ndarray:
    """Union compressed struts with remapped protected edges (dedup)."""
    edge_set = set()
    if len(new_struts) > 0:
        for a, b in np.asarray(new_struts, dtype=np.int64):
            ia, ib = int(a), int(b)
            if ia > ib:
                ia, ib = ib, ia
            edge_set.add((ia, ib))
    for a, b in extra_struts_old:
        ma = int(old_to_new[a])
        mb = int(old_to_new[b])
        if ma < 0 or mb < 0:
            continue
        if ma > mb:
            ma, mb = mb, ma
        edge_set.add((ma, mb))
    if not edge_set:
        return np.empty((0, 2), dtype=np.int32)
    arr = np.array(sorted(edge_set), dtype=np.int32)
    return arr
