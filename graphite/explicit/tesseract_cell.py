"""
Tesseract (nested hypercube projection) unit cell — 16 nodes, 32 struts in a cube of side L.

Node layout (indices 0–15):
  - 0–7: outer cube corners at (+/- L/2, +/- L/2, +/- L/2)
  - 8–15: inner cube corners at (+/- L/4, +/- L/4, +/- L/4) with matching sign patterns

Tiling: outer faces sit at +/- L/2; stack cells by L along X/Y/Z and weld boundary
nodes with proximity threshold 1e-4 (same rounding as global hex topology merge).
"""
from __future__ import annotations

from itertools import product

import numpy as np

_TESSERACT_TOL = 1e-4
_TESSERACT_ROUND = 6


def generate_tesseract_cell(
    L: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one tileable tesseract (nested cube) unit cell centered at the origin.

    Bounding box: [-L/2, L/2]³.

    Parameters
    ----------
    L : float
        Cubic cell side length.

    Returns
    -------
    nodes : ndarray, shape (16, 3)
        Node coordinates (outer 0–7, inner 8–15).
    edges : ndarray, shape (32, 2)
        Undirected edge pairs (node indices).
    """
    if L <= 0.0:
        raise ValueError("L must be > 0.")

    half = float(L) / 2.0
    quarter = float(L) / 4.0
    outer: list[tuple[float, float, float]] = []
    inner: list[tuple[float, float, float]] = []

    for sx, sy, sz in product((-1.0, 1.0), repeat=3):
        outer.append(
            (
                round(sx * half, _TESSERACT_ROUND),
                round(sy * half, _TESSERACT_ROUND),
                round(sz * half, _TESSERACT_ROUND),
            )
        )
        inner.append(
            (
                round(sx * quarter, _TESSERACT_ROUND),
                round(sy * quarter, _TESSERACT_ROUND),
                round(sz * quarter, _TESSERACT_ROUND),
            )
        )

    if len(outer) != 8 or len(inner) != 8:
        raise RuntimeError("Tesseract node generation expected 8 outer and 8 inner nodes.")

    nodes = np.array(outer + inner, dtype=np.float64)
    edge_list: list[tuple[int, int]] = []

    edge_list.extend(_cube_axis_edges(range(8), nodes, float(L)))
    edge_list.extend(_cube_axis_edges(range(8, 16), nodes, float(L) / 2.0))
    for i in range(8):
        edge_list.append((i, 8 + i))

    edge_list = _dedupe_edges(edge_list)
    if len(edge_list) != 32:
        raise RuntimeError(
            f"Tesseract edge generation expected 32 struts, got {len(edge_list)}."
        )

    edges = np.array(sorted(edge_list), dtype=np.int64)
    return nodes, edges


def tesseract_tiling_merge_tolerance() -> float:
    """Proximity threshold for welding outer nodes at +/- L/2 when stacking cells."""
    return _TESSERACT_TOL


def map_tesseract_cell_to_hex_brick(
    hex_corners: np.ndarray,
    *,
    L: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map a reference tesseract cell in [-L/2, L/2]³ onto a deformed 8-node hex brick.

    Fractional coordinates ``u = x/L + 0.5`` drive trilinear hex8 shape functions.
    """
    from .kelvin_cell import _hex8_trilinear

    ref_nodes, ref_edges = generate_tesseract_cell(float(L))
    mapped = np.array(
        [
            _hex8_trilinear(
                hex_corners,
                float(pt[0]) / float(L) + 0.5,
                float(pt[1]) / float(L) + 0.5,
                float(pt[2]) / float(L) + 0.5,
            )
            for pt in ref_nodes
        ],
        dtype=np.float64,
    )
    return mapped, ref_edges.copy()


def _cube_axis_edges(
    indices: range,
    nodes: np.ndarray,
    span: float,
) -> list[tuple[int, int]]:
    """Connect nodes that differ by ``span`` on one axis and match on the other two."""
    idx_list = list(indices)
    edges: list[tuple[int, int]] = []
    for ii, i in enumerate(idx_list):
        for j in idx_list[ii + 1 :]:
            diff = np.abs(nodes[int(j)] - nodes[int(i)])
            zero_axes = sum(1 for k in range(3) if diff[k] <= _TESSERACT_TOL)
            long_axes = sum(1 for k in range(3) if abs(diff[k] - span) <= _TESSERACT_TOL)
            if zero_axes == 2 and long_axes == 1:
                edges.append((min(i, j), max(i, j)))
    return edges


def _dedupe_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(set(edges))
