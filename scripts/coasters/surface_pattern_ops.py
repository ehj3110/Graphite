# -*- coding: utf-8 -*-
"""
Apply Tri_* / Sq_* coaster strut recipes onto conformal surface faces (3D).

Same fractional constructions as the flat coaster generators, evaluated on
ordered triangle / quad corner coordinates from A15 or hex skins.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from scripts.coasters.tri_sq_patterns import (
    SQ_TOPOLOGIES,
    TRI_TOPOLOGIES,
    segments_to_nodes_struts,
)


def _segments_from_tri_face(topo: str, tri: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``tri`` is (3, 3) corner coordinates."""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    if topo == "Tetrahedral":
        return [(v0, v1), (v1, v2), (v2, v0)]
    if topo == "Icosahedral":
        m0 = 0.5 * (v0 + v1)
        m1 = 0.5 * (v1 + v2)
        m2 = 0.5 * (v2 + v0)
        return [(m0, m1), (m1, m2), (m2, m0)]
    if topo == "Kelvin":
        p01_a = (2.0 * v0 + v1) / 3.0
        p01_b = (v0 + 2.0 * v1) / 3.0
        p12_a = (2.0 * v1 + v2) / 3.0
        p12_b = (v1 + 2.0 * v2) / 3.0
        p20_a = (2.0 * v2 + v0) / 3.0
        p20_b = (v2 + 2.0 * v0) / 3.0
        return [
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
    if topo == "Tesseract":
        centroid = (v0 + v1 + v2) / 3.0
        inscribed = np.stack(
            [
                centroid + 0.5 * (v0 - centroid),
                centroid + 0.5 * (v1 - centroid),
                centroid + 0.5 * (v2 - centroid),
            ]
        )
        return [
            (v0, v1),
            (v1, v2),
            (v2, v0),
            (inscribed[0], inscribed[1]),
            (inscribed[1], inscribed[2]),
            (inscribed[2], inscribed[0]),
            (v0, inscribed[0]),
            (v1, inscribed[1]),
            (v2, inscribed[2]),
        ]
    raise ValueError(f"Use adjacency path for Rhombic; got {topo!r}")


def _segments_from_quad_face(topo: str, sq: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``sq`` is (4, 3) ordered corner coordinates."""
    v0, v1, v2, v3 = sq[0], sq[1], sq[2], sq[3]
    if topo == "Grid":
        return [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]
    if topo == "Icosahedral":
        m0 = 0.5 * (v0 + v1)
        m1 = 0.5 * (v1 + v2)
        m2 = 0.5 * (v2 + v3)
        m3 = 0.5 * (v3 + v0)
        return [(m0, m1), (m1, m2), (m2, m3), (m3, m0)]
    if topo == "Kelvin":
        p01_a = (2.0 * v0 + v1) / 3.0
        p01_b = (v0 + 2.0 * v1) / 3.0
        p12_a = (2.0 * v1 + v2) / 3.0
        p12_b = (v1 + 2.0 * v2) / 3.0
        p23_a = (2.0 * v2 + v3) / 3.0
        p23_b = (v2 + 2.0 * v3) / 3.0
        p30_a = (2.0 * v3 + v0) / 3.0
        p30_b = (v3 + 2.0 * v0) / 3.0
        return [
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
    if topo == "Tesseract":
        centroid = 0.25 * (v0 + v1 + v2 + v3)
        inscribed = np.stack(
            [
                centroid + 0.5 * (v0 - centroid),
                centroid + 0.5 * (v1 - centroid),
                centroid + 0.5 * (v2 - centroid),
                centroid + 0.5 * (v3 - centroid),
            ]
        )
        return [
            (v0, v1),
            (v1, v2),
            (v2, v3),
            (v3, v0),
            (inscribed[0], inscribed[1]),
            (inscribed[1], inscribed[2]),
            (inscribed[2], inscribed[3]),
            (inscribed[3], inscribed[0]),
            (v0, inscribed[0]),
            (v1, inscribed[1]),
            (v2, inscribed[2]),
            (v3, inscribed[3]),
        ]
    raise ValueError(f"Unknown square topology: {topo!r}")


def apply_tri_rhombic(
    face_corners: np.ndarray,
    *,
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Centroid dual of adjacent triangles (shared edge).

    ``face_corners``: (F, 3, 3)
    """
    faces = np.asarray(face_corners, dtype=np.float64)
    if faces.ndim != 3 or faces.shape[1:] != (3, 3):
        raise ValueError(f"face_corners must be (F,3,3); got {faces.shape}")
    centroids = faces.mean(axis=1)
    edge_to_faces: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]], list[int]
    ] = defaultdict(list)
    inv = 1.0 / max(merge_tol, 1e-12)

    def _vk(p: np.ndarray) -> tuple[int, int, int]:
        return (
            int(round(p[0] * inv)),
            int(round(p[1] * inv)),
            int(round(p[2] * inv)),
        )

    for fi, tri in enumerate(faces):
        keys = [_vk(tri[0]), _vk(tri[1]), _vk(tri[2])]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = (keys[a], keys[b]) if keys[a] < keys[b] else (keys[b], keys[a])
            edge_to_faces[e].append(fi)

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for faces_on_edge in edge_to_faces.values():
        if len(faces_on_edge) != 2:
            continue
        i, j = faces_on_edge
        segments.append((centroids[i], centroids[j]))
    return segments_to_nodes_struts(segments, merge_tol=merge_tol)


def apply_tri_pattern(
    topo: str,
    face_corners: np.ndarray,
    *,
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Tri_* coaster recipe to a conformal triangle skin.

    ``face_corners``: (F, 3, 3) ordered corners.
    """
    if topo not in TRI_TOPOLOGIES:
        raise ValueError(f"Unknown tri topology: {topo!r}")
    faces = np.asarray(face_corners, dtype=np.float64)
    if faces.ndim != 3 or faces.shape[1:] != (3, 3):
        raise ValueError(f"face_corners must be (F,3,3); got {faces.shape}")
    if topo == "Rhombic":
        return apply_tri_rhombic(faces, merge_tol=merge_tol)
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in faces:
        segments.extend(_segments_from_tri_face(topo, tri))
    return segments_to_nodes_struts(segments, merge_tol=merge_tol)


def apply_sq_pattern(
    topo: str,
    face_corners: np.ndarray,
    *,
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Sq_* coaster recipe to a conformal quad skin.

    ``face_corners``: (F, 4, 3) ordered corners.
    """
    if topo not in SQ_TOPOLOGIES:
        raise ValueError(f"Unknown sq topology: {topo!r}")
    faces = np.asarray(face_corners, dtype=np.float64)
    if faces.ndim != 3 or faces.shape[1:] != (4, 3):
        raise ValueError(f"face_corners must be (F,4,3); got {faces.shape}")
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for sq in faces:
        segments.extend(_segments_from_quad_face(topo, sq))
    return segments_to_nodes_struts(segments, merge_tol=merge_tol)


def project_nodes_to_sphere(
    nodes: np.ndarray,
    center: Sequence[float],
    radius: float,
) -> np.ndarray:
    pts = np.asarray(nodes, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    v = pts - c[None, :]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return c[None, :] + radius * (v / n)
