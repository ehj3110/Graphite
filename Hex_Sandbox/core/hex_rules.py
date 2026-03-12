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

# Pairs of faces that share an edge of the cube — these become struts in the
# Kagome rule.  Opposite face pairs (Bottom/Top, Front/Back, Left/Right) do NOT
# share an edge, so there are 6*4/2 - 3 = 12 adjacent pairs.
_KAGOME_STRUTS: tuple[tuple[int, int], ...] = (
    # Bottom (0) adjacent to: Front(2), Back(3), Left(4), Right(5)
    (0, 2), (0, 3), (0, 4), (0, 5),
    # Top (1) adjacent to: Front(2), Back(3), Left(4), Right(5)
    (1, 2), (1, 3), (1, 4), (1, 5),
    # Front (2) adjacent to: Left(4), Right(5)  [Bottom/Back already counted]
    (2, 4), (2, 5),
    # Back (3) adjacent to: Left(4), Right(5)
    (3, 4), (3, 5),
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

def apply_hex_voronoi(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : centroid (index 0) + 6 face-centers (indices 1-6).
    Struts : 6 segments — centroid to each face-center.
    """
    coords = _validate(coords)
    centroid = hex_centroid(coords)
    face_ctrs = hex_face_centers(coords)
    nodes = np.vstack((centroid[None, :], face_ctrs))   # (7, 3)
    struts = np.array([(0, i) for i in range(1, 7)], dtype=np.int64)  # (6, 2)
    return nodes, struts


def apply_hex_rhombic(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : centroid (index 0) + 8 corner vertices (indices 1-8).
    Struts : 8 segments — centroid to each corner.
    """
    coords = _validate(coords)
    centroid = hex_centroid(coords)
    nodes = np.vstack((centroid[None, :], coords))      # (9, 3)
    struts = np.array([(0, i) for i in range(1, 9)], dtype=np.int64)  # (8, 2)
    return nodes, struts


def apply_hex_kagome(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes  : 6 face-centers — these form the 6 vertices of an internal
             octahedron when the hex is a cube.
    Struts : 12 segments connecting every pair of adjacent face-centers
             (i.e. face pairs that share an edge of the hex).
    """
    coords = _validate(coords)
    nodes = hex_face_centers(coords)                    # (6, 3)
    struts = np.array(_KAGOME_STRUTS, dtype=np.int64)   # (12, 2)
    return nodes, struts
