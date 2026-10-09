# -*- coding: utf-8 -*-
"""
2D Tri_* / Sq_* coaster strut segment recipes.

Ported from the coaster preview generators (antigravity scratch):
triangle / square cell grids with Tetrahedral, Icosahedral, Kelvin,
Tesseract, Rhombic (tri) and Grid, Icosahedral, Kelvin, Tesseract (sq).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

TRI_TOPOLOGIES = ("Tetrahedral", "Icosahedral", "Kelvin", "Tesseract", "Rhombic")
SQ_TOPOLOGIES = ("Grid", "Icosahedral", "Kelvin", "Tesseract")

# Coaster defaults (100 mm coaster catalog)
COASTER_TRI_R = 6.35  # used as: side = R * sqrt(3)
COASTER_SQ_SIDE = 12.7


def unique_segments(
    segments: Iterable[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deduplicate undirected segments via rounded hash keys."""
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    inv = 1.0 / max(tol, 1e-12)
    for p1, p2 in segments:
        a = np.asarray(p1, dtype=np.float64).ravel()[:2]
        b = np.asarray(p2, dtype=np.float64).ravel()[:2]
        ka = (int(round(a[0] * inv)), int(round(a[1] * inv)))
        kb = (int(round(b[0] * inv)), int(round(b[1] * inv)))
        if ka == kb:
            continue
        key = (ka, kb) if ka < kb else (kb, ka)
        if key in seen:
            continue
        seen.add(key)
        out.append((a.copy(), b.copy()))
    return out


def _triangle_grid(side: float, height: float, i_range: range, j_range: range) -> list[np.ndarray]:
    triangles: list[np.ndarray] = []

    def get_p(i: int, j: int) -> np.ndarray:
        cx = i * side + (j % 2) * (side / 2.0)
        cy = j * height
        return np.array([cx, cy], dtype=np.float64)

    for j in j_range:
        for i in i_range:
            p_ij = get_p(i, j)
            p_ip1_j = get_p(i + 1, j)
            p_ijp1 = get_p(i, j + 1)
            p_ip1_jp1 = get_p(i + 1, j + 1)
            if j % 2 == 0:
                triangles.append(np.array([p_ij, p_ip1_j, p_ijp1]))
                triangles.append(np.array([p_ip1_j, p_ip1_jp1, p_ijp1]))
            else:
                triangles.append(np.array([p_ij, p_ip1_j, p_ip1_jp1]))
                triangles.append(np.array([p_ij, p_ip1_jp1, p_ijp1]))
    return triangles


def get_triangle_segments(
    topo: str,
    side: float,
    height: float | None = None,
    *,
    extent_cells: int = 7,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Flat triangular-cell strut patterns (coaster Tri_* family).

    Parameters
    ----------
    side : float
        Horizontal spacing between same-parity triangle vertices.
    height : float | None
        Row pitch. Default ``side * sqrt(3) / 2`` (equilateral packing).
    extent_cells : int
        Tile indices in ``[-extent_cells, extent_cells)``.
    """
    if height is None:
        height = side * np.sqrt(3.0) / 2.0
    i_range = range(-extent_cells, extent_cells)
    j_range = range(-extent_cells, extent_cells)
    triangles = _triangle_grid(side, height, i_range, j_range)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    if topo == "Tetrahedral":
        for tri in triangles:
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
    elif topo == "Icosahedral":
        for tri in triangles:
            m0 = (tri[0] + tri[1]) / 2.0
            m1 = (tri[1] + tri[2]) / 2.0
            m2 = (tri[2] + tri[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m0))
    elif topo == "Kelvin":
        for tri in triangles:
            p01_a = (2 * tri[0] + tri[1]) / 3.0
            p01_b = (tri[0] + 2 * tri[1]) / 3.0
            p12_a = (2 * tri[1] + tri[2]) / 3.0
            p12_b = (tri[1] + 2 * tri[2]) / 3.0
            p20_a = (2 * tri[2] + tri[0]) / 3.0
            p20_b = (tri[2] + 2 * tri[0]) / 3.0
            segments.extend(
                [
                    (p01_a, p20_b),
                    (p20_b, p20_a),
                    (p20_a, p12_b),
                    (p12_b, p12_a),
                    (p12_a, p01_b),
                    (p01_b, p01_a),
                    (p01_a, p01_b),
                    (p12_a, p12_b),
                    (p20_a, p20_b),
                ]
            )
    elif topo == "Tesseract":
        for tri in triangles:
            centroid = np.mean(tri, axis=0)
            inscribed = centroid + 0.5 * (tri - centroid)
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[0]))
            for i in range(3):
                segments.append((tri[i], inscribed[i]))
    elif topo == "Rhombic":
        from collections import defaultdict
        centroids = [np.mean(tri, axis=0) for tri in triangles]
        edge_to_tri: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
        inv = 1e3
        for idx, tri in enumerate(triangles):
            for k in range(3):
                p1, p2 = tri[k], tri[(k + 1) % 3]
                k1 = (int(round(p1[0] * inv)), int(round(p1[1] * inv)))
                k2 = (int(round(p2[0] * inv)), int(round(p2[1] * inv)))
                ek = (k1, k2) if k1 < k2 else (k2, k1)
                edge_to_tri[ek].append(idx)
        for tri_indices in edge_to_tri.values():
            if len(tri_indices) == 2:
                segments.append((centroids[tri_indices[0]], centroids[tri_indices[1]]))
    else:
        raise ValueError(f"Unknown triangle topology: {topo!r}")

    return unique_segments(segments)


def get_square_segments(
    topo: str,
    side: float,
    *,
    extent_cells: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flat square-cell strut patterns (coaster Sq_* family)."""
    squares: list[np.ndarray] = []
    for row in range(-extent_cells, extent_cells):
        for col in range(-extent_cells, extent_cells):
            cx = col * side
            cy = row * side
            v0 = np.array([cx - side / 2.0, cy - side / 2.0], dtype=np.float64)
            v1 = np.array([cx + side / 2.0, cy - side / 2.0], dtype=np.float64)
            v2 = np.array([cx + side / 2.0, cy + side / 2.0], dtype=np.float64)
            v3 = np.array([cx - side / 2.0, cy + side / 2.0], dtype=np.float64)
            squares.append(np.array([v0, v1, v2, v3]))

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    if topo == "Grid":
        for sq in squares:
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
    elif topo == "Icosahedral":
        for sq in squares:
            m0 = (sq[0] + sq[1]) / 2.0
            m1 = (sq[1] + sq[2]) / 2.0
            m2 = (sq[2] + sq[3]) / 2.0
            m3 = (sq[3] + sq[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m3))
            segments.append((m3, m0))
    elif topo == "Kelvin":
        for sq in squares:
            p01_a = (2 * sq[0] + sq[1]) / 3.0
            p01_b = (sq[0] + 2 * sq[1]) / 3.0
            p12_a = (2 * sq[1] + sq[2]) / 3.0
            p12_b = (sq[1] + 2 * sq[2]) / 3.0
            p23_a = (2 * sq[2] + sq[3]) / 3.0
            p23_b = (sq[2] + 2 * sq[3]) / 3.0
            p30_a = (2 * sq[3] + sq[0]) / 3.0
            p30_b = (sq[3] + 2 * sq[0]) / 3.0
            segments.extend(
                [
                    (p01_a, p30_b),
                    (p30_b, p30_a),
                    (p30_a, p23_b),
                    (p23_b, p23_a),
                    (p23_a, p12_b),
                    (p12_b, p12_a),
                    (p12_a, p01_b),
                    (p01_b, p01_a),
                    (p01_a, p01_b),
                    (p12_a, p12_b),
                    (p23_a, p23_b),
                    (p30_a, p30_b),
                ]
            )
    elif topo == "Tesseract":
        for sq in squares:
            centroid = np.mean(sq, axis=0)
            inscribed = centroid + 0.5 * (sq - centroid)
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[3]))
            segments.append((inscribed[3], inscribed[0]))
            for i in range(4):
                segments.append((sq[i], inscribed[i]))
    else:
        raise ValueError(f"Unknown square topology: {topo!r}")

    return unique_segments(segments)


def segments_to_nodes_struts(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert undirected 2D/3D segments into (nodes, struts) arrays."""
    key_to_idx: dict[tuple[int, ...], int] = {}
    nodes: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    inv = 1.0 / max(merge_tol, 1e-12)

    def _idx(p: np.ndarray) -> int:
        p = np.asarray(p, dtype=np.float64).ravel()
        key = tuple(int(round(c * inv)) for c in p)
        i = key_to_idx.get(key)
        if i is None:
            i = len(nodes)
            key_to_idx[key] = i
            nodes.append(p.copy())
        return i

    for p1, p2 in segments:
        i = _idx(p1)
        j = _idx(p2)
        if i == j:
            continue
        strut_set.add((i, j) if i < j else (j, i))

    if not nodes:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    node_arr = np.asarray(nodes, dtype=np.float64)
    if node_arr.ndim == 1:
        node_arr = node_arr.reshape(-1, 1)
    if node_arr.shape[1] == 2:
        node_arr = np.column_stack([node_arr, np.zeros(len(node_arr))])
    struts = np.asarray(sorted(strut_set), dtype=np.int64)
    if struts.size == 0:
        struts = np.zeros((0, 2), dtype=np.int64)
    return node_arr, struts


def pattern_label(family: str, topo: str) -> str:
    prefix = "Tri" if family.lower().startswith("tri") else "Sq"
    return f"{prefix}_{topo}"
