"""
Micro-rules for hexahedral element lattice transformations.

Each rule takes a single hex element as coordinates (8, 3) in the standard
ordering below and returns (nodes, struts):

    Node ordering (matches a right-hand cube):
        0: (-x, -y, -z)  bottom face, wound CCW from below
        1: (+x, -y, -z)
        2: (+x, +y, -z)
        3: (-x, +y, -z)
        4: (-x, -y, +z)  top face, wound CCW from below
        5: (+x, -y, +z)
        6: (+x, +y, +z)
        7: (-x, +y, +z)

Face definitions (6 faces, 4 nodes each):
    Bottom : [0, 1, 2, 3]
    Top    : [4, 5, 6, 7]
    Front  : [0, 1, 5, 4]
    Back   : [3, 2, 6, 7]
    Left   : [0, 3, 7, 4]
    Right  : [1, 2, 6, 5]

Returns:
    nodes  : (N, 3) float64
    struts : (S, 2) int64 index pairs into nodes
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # Bottom
    (4, 5, 6, 7),  # Top
    (0, 1, 5, 4),  # Front
    (3, 2, 6, 7),  # Back
    (0, 3, 7, 4),  # Left
    (1, 2, 6, 5),  # Right
)

# Pairs of faces that share an edge of the cube (adjacent face-centers).
_ADJACENT_FACE_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5),
    (3, 4), (3, 5),
)

# 12 edges of the hexahedron: (vertex_i, vertex_j)
_HEX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),   # Bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # Top
    (0, 4), (1, 5), (2, 6), (3, 7),   # Vertical
)


def _validate(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (8, 3):
        raise ValueError(f"coords must have shape (8, 3), got {coords.shape}.")
    return coords


def hex_face_centers(coords: np.ndarray) -> np.ndarray:
    """Return (6, 3) array of face centroids in _HEX_FACES order."""
    return np.array([coords[list(f)].mean(axis=0) for f in _HEX_FACES])


def hex_centroid(coords: np.ndarray) -> np.ndarray:
    """Return (3,) centroid of the hex element."""
    return coords.mean(axis=0)


# ---------------------------------------------------------------------------
# Rule functions
# ---------------------------------------------------------------------------

def apply_hex_grid(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : 8 corners.
    Struts : 12 edges of the hexahedron.
    """
    coords = _validate(coords)
    nodes = coords.copy()
    struts = np.array(_HEX_EDGES, dtype=np.int64)
    return nodes, struts


def apply_hex_octahedral(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : 6 face-centers (internal octahedron vertices).
    Struts : 12 edges connecting adjacent face-centers.
    """
    coords = _validate(coords)
    nodes = hex_face_centers(coords)
    struts = np.array(_ADJACENT_FACE_PAIRS, dtype=np.int64)
    return nodes, struts


def apply_hex_star(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : 8 corners + 1 centroid.
    Struts : 8 segments from centroid to each corner.
    """
    coords = _validate(coords)
    centroid = hex_centroid(coords)
    nodes = np.vstack((centroid[None, :], coords))
    struts = np.array([(0, i) for i in range(1, 9)], dtype=np.int64)
    return nodes, struts


def apply_hex_octet_truss(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : 8 corners + 6 face-centers (14 total).
    Struts : 36 total — each face-center to its 4 corners (24) +
             adjacent face-centers to each other (12).
    """
    coords = _validate(coords)
    face_ctrs = hex_face_centers(coords)
    nodes = np.vstack((coords, face_ctrs))   # (14, 3): corners 0-7, face-centers 8-13

    strut_list: list[tuple[int, int]] = []
    for fi, face in enumerate(_HEX_FACES):
        fc_idx = 8 + fi
        for corner in face:
            strut_list.append((corner, fc_idx))
    for a, b in _ADJACENT_FACE_PAIRS:
        strut_list.append((8 + a, 8 + b))

    struts = np.array(strut_list, dtype=np.int64)
    return nodes, struts
