"""
Kelvin (truncated octahedron) unit cell — 24 nodes, 36 struts in a cube of side L.

Tiling: translate the cell by ±L along X/Y/Z and weld nodes whose coordinates
match on opposing faces (|coord| = L/2 = 2a) using the same merge tolerance as
global hex topology (typically 1e-4 to 1e-6 after rounding).
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np

_KELVIN_EDGE_TOL = 1e-4
_KELVIN_ROUND = 6


def generate_kelvin_cell(
    L: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one tileable Kelvin unit cell centered at the origin.

    Bounding box: [-L/2, L/2]³.

    Parameters
    ----------
    L : float
        Cubic cell side length.

    Returns
    -------
    nodes : ndarray, shape (24, 3)
        Node coordinates.
    edges : ndarray, shape (36, 2)
        Undirected edge pairs (node indices).
    """
    if L <= 0.0:
        raise ValueError("L must be > 0.")

    a = float(L) / 4.0
    template = (0.0, a, 2.0 * a)
    node_set: set[tuple[float, float, float]] = set()

    for perm in permutations(template):
        for sx, sy, sz in product((-1.0, 1.0), repeat=3):
            x, y, z = perm
            node_set.add(
                (
                    round(float(x) * sx if x != 0.0 else 0.0, _KELVIN_ROUND),
                    round(float(y) * sy if y != 0.0 else 0.0, _KELVIN_ROUND),
                    round(float(z) * sz if z != 0.0 else 0.0, _KELVIN_ROUND),
                )
            )

    if len(node_set) != 24:
        raise RuntimeError(f"Kelvin node generation expected 24 nodes, got {len(node_set)}.")

    nodes = np.array(sorted(node_set), dtype=np.float64)
    target_len = float(np.sqrt(2.0) * a)
    edge_list: list[tuple[int, int]] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = float(np.linalg.norm(nodes[j] - nodes[i]))
            if abs(dist - target_len) <= _KELVIN_EDGE_TOL:
                edge_list.append((i, j))

    if len(edge_list) != 36:
        raise RuntimeError(
            f"Kelvin edge generation expected 36 struts, got {len(edge_list)}."
        )

    edges = np.array(edge_list, dtype=np.int64)
    return nodes, edges


def kelvin_tiling_merge_tolerance() -> float:
    """Proximity threshold for welding nodes at ±L/2 when stacking cells."""
    return _KELVIN_EDGE_TOL


def map_kelvin_cell_to_hex_brick(
    hex_corners: np.ndarray,
    *,
    L: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map a reference Kelvin cell in [-L/2, L/2]³ onto a deformed 8-node hex brick.

    Fractional coordinates ``u = x/L + 0.5`` (etc.) drive trilinear hex8 shape
    functions so the Kelvin graph conforms to snapped conformal hexes.
    """
    ref_nodes, ref_edges = generate_kelvin_cell(float(L))
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


def _hex8_trilinear(
    corners: np.ndarray,
    u: float,
    v: float,
    w: float,
) -> np.ndarray:
    """Trilinear map from unit cube [0,1]³ to hex8 corner ordering used in Route 3."""
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    u, v, w = float(u), float(v), float(w)
    weights = np.array(
        [
            (1.0 - u) * (1.0 - v) * (1.0 - w),
            u * (1.0 - v) * (1.0 - w),
            u * v * (1.0 - w),
            (1.0 - u) * v * (1.0 - w),
            (1.0 - u) * (1.0 - v) * w,
            u * (1.0 - v) * w,
            u * v * w,
            (1.0 - u) * v * w,
        ],
        dtype=np.float64,
    )
    return np.dot(weights, c)
