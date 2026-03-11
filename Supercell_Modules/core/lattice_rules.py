"""
Micro-rules for tetrahedron-to-lattice transformations.

Each rule takes a single tetrahedron as coordinates (4, 3) and returns:
    nodes: (N, 3)
    struts: (S, 2) index pairs into nodes
"""

from __future__ import annotations

import numpy as np


def _tet_faces() -> tuple[tuple[int, int, int], ...]:
    return ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def _tet_edges() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def apply_voronoi_rule(tet_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes: tet centroid + 4 face centroids.
    Struts: 4 segments (centroid -> each face centroid).
    """
    coords = np.asarray(tet_coords, dtype=np.float64)
    if coords.shape != (4, 3):
        raise ValueError("tet_coords must have shape (4, 3).")

    tet_centroid = coords.mean(axis=0)
    face_centroids = np.array([coords[list(face)].mean(axis=0) for face in _tet_faces()])
    nodes = np.vstack((tet_centroid[None, :], face_centroids))

    struts = np.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=np.int64)
    return nodes, struts


def apply_kagome_rule(tet_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes: 4 face centroids.
    Struts: complete graph on 4 nodes -> 6 segments.
    """
    coords = np.asarray(tet_coords, dtype=np.float64)
    if coords.shape != (4, 3):
        raise ValueError("tet_coords must have shape (4, 3).")

    nodes = np.array([coords[list(face)].mean(axis=0) for face in _tet_faces()])
    struts = np.array(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        dtype=np.int64,
    )
    return nodes, struts


def apply_icosahedral_rule(tet_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes: 6 edge midpoints.
    Struts: 12 segments (3 segments per original face).
    """
    coords = np.asarray(tet_coords, dtype=np.float64)
    if coords.shape != (4, 3):
        raise ValueError("tet_coords must have shape (4, 3).")

    edges = _tet_edges()
    nodes = np.array([(coords[a] + coords[b]) * 0.5 for a, b in edges], dtype=np.float64)
    edge_to_idx = {tuple(sorted((a, b))): i for i, (a, b) in enumerate(edges)}

    strut_list: list[tuple[int, int]] = []
    for i, j, k in _tet_faces():
        m_ij = edge_to_idx[tuple(sorted((i, j)))]
        m_ik = edge_to_idx[tuple(sorted((i, k)))]
        m_jk = edge_to_idx[tuple(sorted((j, k)))]
        strut_list.extend([(m_ij, m_ik), (m_ij, m_jk), (m_ik, m_jk)])

    struts = np.array(strut_list, dtype=np.int64)
    return nodes, struts


def apply_rhombic_rule(tet_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nodes: tet centroid + 4 vertices.
    Struts: 4 segments (centroid -> each vertex).
    """
    coords = np.asarray(tet_coords, dtype=np.float64)
    if coords.shape != (4, 3):
        raise ValueError("tet_coords must have shape (4, 3).")

    tet_centroid = coords.mean(axis=0)
    nodes = np.vstack((tet_centroid[None, :], coords))
    struts = np.array([(0, 1), (0, 2), (0, 3), (0, 4)], dtype=np.int64)
    return nodes, struts
