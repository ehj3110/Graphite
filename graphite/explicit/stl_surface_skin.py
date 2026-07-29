"""
STL-native conforming surface skin for hex volume lattices.

Builds a strut graph on the input triangle mesh (face centroids and edge
midpoints, all projected to the STL) with centroid -> edge-mid -> centroid paths
analogous to ``generate_hex_surface_dual_on_surface_paths``, then bridges each
hex boundary face center to the nearest skin node so the interior octahedral
graph stays aligned.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .boundary_policy import closest_points_with_fallback
from .hex_surface_dual import HEX_FACES, BoundaryQuadTopology

_TRI_EDGE_PAIRS = ((0, 1), (1, 2), (2, 0))


def refine_mesh_for_surface_skin(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    *,
    max_faces: int = 25_000,
    max_subdivisions: int = 6,
) -> trimesh.Trimesh:
    """
    Subdivide until the longest unique edge is at most ``target_edge_length``.

    Coarse input STLs (few large triangles) need refinement before the skin graph
    can follow curvature at the lattice cell scale.
    """
    if target_edge_length <= 0:
        raise ValueError("target_edge_length must be > 0.")
    m = mesh.copy()
    for _ in range(int(max_subdivisions)):
        if len(m.faces) >= int(max_faces):
            break
        lengths = m.edges_unique_length
        if len(lengths) == 0:
            break
        if float(np.max(lengths)) <= float(target_edge_length):
            break
        m = m.subdivide()
    return m


def project_points_to_mesh(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> np.ndarray:
    """Snap points to closest locations on ``mesh``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    closest, _ = closest_points_with_fallback(mesh, pts)
    return np.asarray(closest, dtype=np.float64)


def generate_stl_surface_path_skin(
    mesh: trimesh.Trimesh,
    *,
    project_to_surface: bool = True,
    max_centroid_span: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Surface-path skin on the STL triangle mesh.

    One node per triangle (projected centroid). For each interior mesh edge shared
    by two triangles, add an edge-midpoint node and two struts:
    ``tri_i -> edge_mid -> tri_j``.

    Returns:
        nodes, struts, report
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"`mesh` must be trimesh.Trimesh, got {type(mesh)}")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_tri = int(faces.shape[0])
    if n_tri == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        z2 = np.empty((0, 2), dtype=np.int64)
        return empty, z2, {"n_triangles": 0, "surface_skin_mode": "stl_surface_path"}

    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    if project_to_surface:
        centroids = project_points_to_mesh(mesh, centroids)

    edge_to_tris: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in _TRI_EDGE_PAIRS:
            va, vb = int(tri[a]), int(tri[b])
            key = (min(va, vb), max(va, vb))
            edge_to_tris[key].append(int(fi))

    nodes_list = [centroids[i].copy() for i in range(n_tri)]
    edge_mid_index: dict[tuple[int, int], int] = {}
    strut_set: set[tuple[int, int]] = set()
    n_path = 0
    n_boundary_edges = 0

    def _mid_node(va: int, vb: int) -> int:
        key = (min(va, vb), max(va, vb))
        idx = edge_mid_index.get(key)
        if idx is None:
            mid = 0.5 * (verts[va] + verts[vb])
            if project_to_surface:
                mid = project_points_to_mesh(mesh, mid.reshape(1, 3))[0]
            idx = len(nodes_list)
            nodes_list.append(np.asarray(mid, dtype=np.float64))
            edge_mid_index[key] = idx
        return idx

    def _add(a: int, b: int) -> None:
        if a == b:
            return
        if a > b:
            a, b = b, a
        strut_set.add((int(a), int(b)))

    for key, tris in edge_to_tris.items():
        if len(tris) != 2:
            n_boundary_edges += 1
            continue
        i, j = int(tris[0]), int(tris[1])
        if max_centroid_span is not None:
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            if d > float(max_centroid_span):
                continue
        va, vb = key
        mi = _mid_node(va, vb)
        _add(i, mi)
        _add(mi, j)
        n_path += 2

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )

    max_off = 0.0
    if project_to_surface and len(nodes) > 0:
        _, dists = closest_points_with_fallback(mesh, nodes)
        max_off = float(np.max(dists)) if len(dists) else 0.0

    return nodes, struts, {
        "n_triangles": n_tri,
        "n_skin_nodes": int(len(nodes)),
        "n_skin_struts": int(len(struts)),
        "n_path_struts": int(n_path),
        "n_stl_boundary_edges": int(n_boundary_edges),
        "max_node_surface_offset_mm": float(max_off),
        "surface_skin_mode": "stl_surface_path",
    }


def hex_boundary_face_centers(
    hex_elements: np.ndarray,
    quad_topology: BoundaryQuadTopology,
) -> np.ndarray:
    """World-space center of each boundary quad face, shape ``(F, 3)``."""
    elems = np.asarray(hex_elements, dtype=np.float64)
    out = np.empty((quad_topology.n_quads, 3), dtype=np.float64)
    for qi in range(quad_topology.n_quads):
        hi, fi = quad_topology.quad_to_hex_face(qi)
        corners = elems[int(hi)][list(HEX_FACES[int(fi)])]
        out[qi] = np.mean(corners, axis=0)
    return out


def build_volume_stl_skin_bridge_struts(
    vol_nodes: np.ndarray,
    boundary_face_to_vol_node: dict[tuple[int, int], int],
    quad_topology: BoundaryQuadTopology,
    hex_elements: np.ndarray,
    stl_skin_nodes: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    project_bridges_to_surface: bool = True,
    bridge_merge_tolerance: float | None = None,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """
    Bridge struts from octahedral volume face nodes to nearest STL skin nodes.

    Returns ``(B, 2)`` with global indices into ``vstack(vol_nodes, stl_skin_nodes)``.
    """
    vol_nodes = np.asarray(vol_nodes, dtype=np.float64)
    stl_nodes = np.asarray(stl_skin_nodes, dtype=np.float64)
    n_vol = int(len(vol_nodes))
    face_centers = hex_boundary_face_centers(hex_elements, quad_topology)
    if project_bridges_to_surface and len(face_centers):
        face_centers = project_points_to_mesh(mesh, face_centers)

    tol = bridge_merge_tolerance
    if tol is None:
        ext = float(np.max(mesh.extents)) if len(mesh.extents) else 1.0
        tol = max(0.05, ext * 1e-4)

    tree = cKDTree(stl_nodes) if len(stl_nodes) else None
    bridges: list[tuple[int, int]] = []
    lengths: list[float] = []

    for qi in range(quad_topology.n_quads):
        hi, fi = quad_topology.quad_to_hex_face(qi)
        vol_id = boundary_face_to_vol_node.get((int(hi), int(fi)))
        if vol_id is None or tree is None:
            continue
        fc = face_centers[qi]
        dist, nn = tree.query(fc)
        skin_global = n_vol + int(nn)
        dist = float(dist)
        if dist <= float(tol) and int(vol_id) == skin_global:
            continue
        a, b = int(vol_id), skin_global
        if a > b:
            a, b = b, a
        bridges.append((a, b))
        lengths.append(dist)

    bridge_arr = (
        np.array(bridges, dtype=np.int64)
        if bridges
        else np.empty((0, 2), dtype=np.int64)
    )
    return bridge_arr, {
        "n_volume_skin_bridges": int(len(bridges)),
        "mean_bridge_length_mm": float(np.mean(lengths)) if lengths else 0.0,
        "max_bridge_length_mm": float(np.max(lengths)) if lengths else 0.0,
        "bridge_merge_tolerance_mm": float(tol),
    }


def merge_volume_with_stl_surface_skin(
    vol_nodes: np.ndarray,
    vol_struts: np.ndarray,
    boundary_face_to_vol_node: dict[tuple[int, int], int],
    quad_topology: BoundaryQuadTopology,
    hex_elements: np.ndarray,
    stl_skin_nodes: np.ndarray,
    stl_skin_struts: np.ndarray,
    mesh: trimesh.Trimesh,
    *,
    project_bridges_to_surface: bool = True,
    bridge_merge_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """
    Union volume graph + STL skin with explicit bridges from hex face nodes.

    Volume node indices are preserved. STL skin nodes are appended and offset.
    Each boundary ``(hex, face)`` with a volume node is bridged to the nearest STL
    skin node (after projecting the hex face center onto the mesh).
    """
    vol_nodes = np.asarray(vol_nodes, dtype=np.float64)
    stl_nodes = np.asarray(stl_skin_nodes, dtype=np.float64)
    n_vol = int(len(vol_nodes))
    nodes = (
        np.vstack((vol_nodes, stl_nodes))
        if len(stl_nodes)
        else vol_nodes.copy()
    )

    strut_set: set[tuple[int, int]] = set()

    def _add(a: int, b: int) -> None:
        if a == b:
            return
        if a > b:
            a, b = b, a
        strut_set.add((int(a), int(b)))

    for a, b in np.asarray(vol_struts, dtype=np.int64):
        _add(int(a), int(b))
    for a, b in np.asarray(stl_skin_struts, dtype=np.int64):
        _add(n_vol + int(a), n_vol + int(b))

    bridge_arr, bridge_meta = build_volume_stl_skin_bridge_struts(
        vol_nodes,
        boundary_face_to_vol_node,
        quad_topology,
        hex_elements,
        stl_nodes,
        mesh,
        project_bridges_to_surface=project_bridges_to_surface,
        bridge_merge_tolerance=bridge_merge_tolerance,
    )
    for a, b in bridge_arr:
        _add(int(a), int(b))

    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts, {
        "merge_mode": "volume_plus_stl_surface_skin",
        "n_volume_nodes": n_vol,
        "n_stl_skin_nodes": int(len(stl_nodes)),
        **bridge_meta,
    }


def polylines_for_struts_on_surface(
    mesh: trimesh.Trimesh,
    nodes: np.ndarray,
    struts: np.ndarray,
    *,
    segments_per_strut: int = 6,
) -> list[np.ndarray]:
    """
    Replace each strut with a polyline on the STL (chord subdivision + projection).
    """
    n_seg = max(2, int(segments_per_strut))
    pts = np.asarray(nodes, dtype=np.float64)
    lines: list[np.ndarray] = []
    for a, b in np.asarray(struts, dtype=np.int64):
        p0, p1 = pts[int(a)], pts[int(b)]
        ts = np.linspace(0.0, 1.0, n_seg, dtype=np.float64)
        chord = p0 + ts[:, None] * (p1 - p0)
        chord = project_points_to_mesh(mesh, chord)
        lines.append(chord)
    return lines


def export_strut_polylines_stl(
    mesh: trimesh.Trimesh,
    nodes: np.ndarray,
    struts: np.ndarray,
    path: str | Path,
    *,
    strut_radius: float,
    segments_per_strut: int = 6,
) -> None:
    """Export struts as short cylinders along surface-projected polylines."""
    import manifold3d

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    polylines = polylines_for_struts_on_surface(
        mesh,
        nodes,
        struts,
        segments_per_strut=segments_per_strut,
    )
    manifolds = []
    for poly in polylines:
        for k in range(len(poly) - 1):
            p1, p2 = poly[k], poly[k + 1]
            vec = p2 - p1
            length = float(np.linalg.norm(vec))
            if length < 1e-9:
                continue
            cyl = trimesh.creation.cylinder(
                radius=float(strut_radius), height=length, sections=10
            )
            z = np.array([0.0, 0.0, 1.0])
            direction = vec / length
            cross = np.cross(z, direction)
            if np.linalg.norm(cross) < 1e-8:
                cross = np.array([1.0, 0.0, 0.0])
            angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
            if angle > 1e-8:
                mat = trimesh.transformations.rotation_matrix(angle, cross)
                cyl.apply_transform(mat)
            cyl.apply_translation((p1 + p2) / 2.0)
            manifolds.append(
                manifold3d.Manifold(
                    manifold3d.Mesh(
                        vert_properties=np.asarray(cyl.vertices, dtype=np.float32),
                        tri_verts=np.asarray(cyl.faces, dtype=np.uint32),
                    )
                )
            )

    if not manifolds:
        trimesh.Trimesh().export(str(out))
        return
    combined = manifolds[0]
    for m in manifolds[1:]:
        combined = combined + m
    mesh_raw = combined.to_mesh()
    verts = np.asarray(mesh_raw.vert_properties)
    faces = np.asarray(mesh_raw.tri_verts)
    trimesh.Trimesh(vertices=verts, faces=faces).export(str(out))
