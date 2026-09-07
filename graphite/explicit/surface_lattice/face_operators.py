# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - Face Operators & Direct Surface Mapping.

Applies micro-rules and topological dual operators directly onto 3D triangle and quad
faces of surface meshes, avoiding the volumetric meshing and boundary clipping defects
of 3D dual pipelines.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence
import numpy as np
import trimesh


TRI_TOPOLOGIES = ("Tetrahedral", "Icosahedral", "Kelvin", "Tesseract", "Rhombic")
QUAD_TOPOLOGIES = ("Grid", "Icosahedral", "Kelvin", "Tesseract")


def segments_to_nodes_struts(
    segments: list[tuple[np.ndarray, np.ndarray]],
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert undirected 3D line segments into unique node coordinates and edge indices.
    """
    if not segments:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    inv = 1.0 / max(merge_tol, 1e-12)
    node_map: dict[tuple[int, int, int], int] = {}
    nodes: list[np.ndarray] = []
    edges: set[tuple[int, int]] = set()

    for p1, p2 in segments:
        k1 = (int(round(p1[0] * inv)), int(round(p1[1] * inv)), int(round(p1[2] * inv)))
        k2 = (int(round(p2[0] * inv)), int(round(p2[1] * inv)), int(round(p2[2] * inv)))

        if k1 not in node_map:
            node_map[k1] = len(nodes)
            nodes.append(np.asarray(p1, dtype=np.float64))
        if k2 not in node_map:
            node_map[k2] = len(nodes)
            nodes.append(np.asarray(p2, dtype=np.float64))

        i1, i2 = node_map[k1], node_map[k2]
        if i1 != i2:
            edges.add((min(i1, i2), max(i1, i2)))

    return np.array(nodes, dtype=np.float64), np.array(sorted(edges), dtype=np.int64)


def segments_from_tri_face(topo: str, tri: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate segments for a single 3D triangle face corner array (3, 3)."""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    topo_norm = topo.capitalize()

    if topo_norm == "Tetrahedral" or topo_norm == "Grid":
        return [(v0, v1), (v1, v2), (v2, v0)]

    if topo_norm == "Icosahedral":
        m0 = 0.5 * (v0 + v1)
        m1 = 0.5 * (v1 + v2)
        m2 = 0.5 * (v2 + v0)
        return [(m0, m1), (m1, m2), (m2, m0)]

    if topo_norm == "Kelvin":
        p01_a = (2.0 * v0 + v1) / 3.0
        p01_b = (v0 + 2.0 * v1) / 3.0
        p12_a = (2.0 * v1 + v2) / 3.0
        p12_b = (v1 + 2.0 * v2) / 3.0
        p20_a = (2.0 * v2 + v0) / 3.0
        p20_b = (v2 + 2.0 * v0) / 3.0
        return [
            (p01_a, p20_b), (p20_b, p20_a), (p20_a, p12_b),
            (p12_b, p12_a), (p12_a, p01_b), (p01_b, p01_a),
            (p01_a, p01_b), (p12_a, p12_b), (p20_a, p20_b),
        ]

    if topo_norm == "Tesseract":
        centroid = (v0 + v1 + v2) / 3.0
        inscribed = np.stack([
            centroid + 0.5 * (v0 - centroid),
            centroid + 0.5 * (v1 - centroid),
            centroid + 0.5 * (v2 - centroid),
        ])
        return [
            (v0, v1), (v1, v2), (v2, v0),
            (inscribed[0], inscribed[1]),
            (inscribed[1], inscribed[2]),
            (inscribed[2], inscribed[0]),
            (v0, inscribed[0]), (v1, inscribed[1]), (v2, inscribed[2]),
        ]

    raise ValueError(f"Unknown triangle face topology '{topo}'. Supported: {TRI_TOPOLOGIES}")


def segments_from_quad_face(topo: str, sq: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate segments for a single 3D quad face corner array (4, 3)."""
    v0, v1, v2, v3 = sq[0], sq[1], sq[2], sq[3]
    topo_norm = topo.capitalize()

    if topo_norm == "Grid":
        return [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]

    if topo_norm == "Icosahedral":
        m0 = 0.5 * (v0 + v1)
        m1 = 0.5 * (v1 + v2)
        m2 = 0.5 * (v2 + v3)
        m3 = 0.5 * (v3 + v0)
        return [(m0, m1), (m1, m2), (m2, m3), (m3, m0)]

    if topo_norm == "Kelvin":
        p01_a = (2.0 * v0 + v1) / 3.0
        p01_b = (v0 + 2.0 * v1) / 3.0
        p12_a = (2.0 * v1 + v2) / 3.0
        p12_b = (v1 + 2.0 * v2) / 3.0
        p23_a = (2.0 * v2 + v3) / 3.0
        p23_b = (v2 + 2.0 * v3) / 3.0
        p30_a = (2.0 * v3 + v0) / 3.0
        p30_b = (v3 + 2.0 * v0) / 3.0
        return [
            (p01_a, p30_b), (p30_b, p30_a), (p30_a, p23_b),
            (p23_b, p23_a), (p23_a, p12_b), (p12_b, p12_a),
            (p12_a, p01_b), (p01_b, p01_a), (p01_a, p01_b),
            (p12_a, p12_b), (p23_a, p23_b), (p30_a, p30_b),
        ]

    if topo_norm == "Tesseract":
        centroid = 0.25 * (v0 + v1 + v2 + v3)
        inscribed = np.stack([
            centroid + 0.5 * (v0 - centroid),
            centroid + 0.5 * (v1 - centroid),
            centroid + 0.5 * (v2 - centroid),
            centroid + 0.5 * (v3 - centroid),
        ])
        return [
            (v0, v1), (v1, v2), (v2, v3), (v3, v0),
            (inscribed[0], inscribed[1]), (inscribed[1], inscribed[2]),
            (inscribed[2], inscribed[3]), (inscribed[3], inscribed[0]),
            (v0, inscribed[0]), (v1, inscribed[1]),
            (v2, inscribed[2]), (v3, inscribed[3]),
        ]

    raise ValueError(f"Unknown quad face topology '{topo}'. Supported: {QUAD_TOPOLOGIES}")


def apply_surface_pattern_to_mesh(
    mesh: trimesh.Trimesh,
    pattern: str,
    merge_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a 2D micro-rule or surface dual directly to a 3D surface triangle mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input surface mesh (e.g. icosphere, sphere, or CAD boundary).
    pattern : str
        "Rhombic" (surface centroid dual), "Tetrahedral", "Icosahedral", "Kelvin", "Tesseract".
    merge_tol : float
        Node welding distance tolerance.

    Returns
    -------
    nodes : (N, 3) float64 array of vertices on or near the surface.
    struts : (S, 2) int64 array of edge indices.
    """
    pattern_norm = pattern.capitalize()
    vertices = mesh.vertices
    faces = mesh.faces

    # 1. Exact Surface Centroid Dual
    if pattern_norm in ("Rhombic", "Dual"):
        centroids = vertices[faces].mean(axis=1)
        # Map shared edges to incident faces
        edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
        for fi, f in enumerate(faces):
            for i in range(len(f)):
                u, v = int(f[i]), int(f[(i + 1) % len(f)])
                edge = (min(u, v), max(u, v))
                edge_to_faces[edge].append(fi)

        dual_segments: list[tuple[np.ndarray, np.ndarray]] = []
        for incident in edge_to_faces.values():
            if len(incident) == 2:
                dual_segments.append((centroids[incident[0]], centroids[incident[1]]))

        return segments_to_nodes_struts(dual_segments, merge_tol=merge_tol)

    # 2. Per-triangle micro-rules
    face_corners = vertices[faces]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in face_corners:
        segments.extend(segments_from_tri_face(pattern_norm, tri))

    return segments_to_nodes_struts(segments, merge_tol=merge_tol)
