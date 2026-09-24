"""
Topological adjacency graph for surface-dual routing (Task 22).

Routing uses 3D Cartesian edge midpoints — not 2D UV proximity — so seams that
tear in the UV net still connect correctly via shared 3D geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphite.explicit.hex_rules import _HEX_FACES, _hex_face_centers
from graphite.explicit.hex_topology_module import resolve_quantized_cell_rule
from graphite.explicit.sc_contextual_surface import (
    ROLE_FULL,
    ROLE_HALF_APEX,
    ROLE_HALF_CUT,
    ROLE_HALF_SIDE,
    classify_exposed_face_role,
    contextual_surface_nodes_for_face,
    midplane_diamond_nodes,
    native_surface_struts_for_face,
    prune_face_candidate_nodes,
    side_cut_edge_midpoint,
)

# High-precision 3D edge-midpoint key (~1e-5 for mm-scale CAD)
_EDGE_MID_DECIMALS = 5


def _corner_key(pt: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


def _face_key(
    corners: np.ndarray,
    face_local: tuple[int, int, int, int],
    decimals: int,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(sorted(_corner_key(corners[i], decimals) for i in face_local))


def _edge_midpoint_key(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    decimals: int = _EDGE_MID_DECIMALS,
) -> tuple[float, float, float]:
    """Exact 3D midpoint of a Cartesian edge, rounded for dict keying."""
    mid = 0.5 * (
        np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64)
    )
    return _corner_key(mid, decimals)


def _outward_face_normal(corners: np.ndarray, face_index: int) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float64)
    face = _HEX_FACES[int(face_index)]
    p0, p1, p2 = c[face[0]], c[face[1]], c[face[2]]
    n = np.cross(p1 - p0, p2 - p0)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-14:
        return np.zeros(3, dtype=np.float64)
    n = n / nrm
    centroid = c.mean(axis=0)
    face_c = c[list(face)].mean(axis=0)
    if np.dot(n, face_c - centroid) < 0.0:
        n = -n
    return n


@dataclass
class TopoFace:
    """One exposed Cartesian face with Task-20 pruned surface nodes."""

    face_id: int
    hex_i: int
    face_i: int
    rule: str
    role: str
    corners_3d: np.ndarray  # (4, 3)
    edges_3d: tuple[tuple[np.ndarray, np.ndarray], ...]  # 4 segments
    normal: np.ndarray
    nodes_3d: np.ndarray  # (N, 3) pruned surface nodes
    native_locals: tuple[tuple[int, int], ...]  # local strut pairs


@dataclass
class TopoRoutedStrut:
    """One manifold stitch between topological neighbor faces."""

    face_a: int
    node_a: int  # local index into TopoFace.nodes_3d
    face_b: int
    node_b: int
    edge_mid_key: tuple[float, float, float]


@dataclass
class TopologicalDualResult:
    faces: list[TopoFace]
    edge_map: dict[tuple[float, float, float], list[int]]
    native_struts: list[tuple[int, int, int, int]]  # (fid, i, fid, j) same face
    routed_struts: list[TopoRoutedStrut]
    report: dict = field(default_factory=dict)


def build_exposed_topo_faces(
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
) -> list[TopoFace]:
    """Index every exposed Cartesian face and attach pruned surface nodes."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    rules: list[str | None] = []
    for hi, (corners, tag) in enumerate(zip(elems, tags)):
        rule = resolve_quantized_cell_rule(tag)
        rules.append(rule)
        if rule is None:
            continue
        for fi, face in enumerate(_HEX_FACES):
            face_owners.setdefault(
                _face_key(corners, face, round_decimals), []
            ).append((hi, fi))

    out: list[TopoFace] = []
    for hi, (corners, rule) in enumerate(zip(elems, rules)):
        if rule is None:
            continue
        c = np.asarray(corners, dtype=np.float64)
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(c, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            role = classify_exposed_face_role(rule, fi)
            ctx = contextual_surface_nodes_for_face(c, fi, rule)
            fc = _hex_face_centers(c)[fi]
            kept, native_locals = prune_face_candidate_nodes(
                role,
                ctx.points,
                face_center=fc,
                cut_edge_mid=side_cut_edge_midpoint(c, fi, rule)
                if role == ROLE_HALF_SIDE
                else None,
                diamond_targets=midplane_diamond_nodes(c, rule)
                if role == ROLE_HALF_CUT
                else None,
            )
            if role == ROLE_HALF_CUT and not native_locals:
                native_locals = native_surface_struts_for_face(ROLE_HALF_CUT)

            corners_3d = np.array([c[i] for i in face], dtype=np.float64)
            edges = tuple(
                (corners_3d[a].copy(), corners_3d[b].copy())
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            out.append(
                TopoFace(
                    face_id=len(out),
                    hex_i=hi,
                    face_i=fi,
                    rule=rule,
                    role=role,
                    corners_3d=corners_3d,
                    edges_3d=edges,
                    normal=_outward_face_normal(c, fi),
                    nodes_3d=np.asarray(kept, dtype=np.float64).reshape(-1, 3),
                    native_locals=tuple(native_locals),
                )
            )
    return out


def build_edge_map(
    faces: list[TopoFace],
    *,
    decimals: int = _EDGE_MID_DECIMALS,
) -> dict[tuple[float, float, float], list[int]]:
    """
    Step 1 — edge_map keyed by exact 3D midpoint of each Cartesian edge.

    Value: list of Face_IDs that share that edge.
    """
    edge_map: dict[tuple[float, float, float], list[int]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=decimals)
            edge_map.setdefault(key, []).append(int(f.face_id))
    for k, ids in list(edge_map.items()):
        edge_map[k] = list(dict.fromkeys(ids))
    return edge_map


def _nodes_near_edge(
    nodes: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    tol_frac: float = 0.35,
) -> list[int]:
    """Local node indices lying on / near the shared Cartesian edge."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    ab = b - a
    L = float(np.linalg.norm(ab))
    tol = max(tol_frac * L, 1e-6)
    out: list[int] = []
    for i, p in enumerate(np.asarray(nodes, dtype=np.float64)):
        if L < 1e-14:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.clip(np.dot(p - a, ab) / (L * L), 0.0, 1.0))
            d = float(np.linalg.norm(p - (a + t * ab)))
        if d <= tol:
            out.append(i)
    return out


def _route_pair_across_edge(
    fa: TopoFace,
    fb: TopoFace,
    p0: np.ndarray,
    p1: np.ndarray,
    edge_key: tuple[float, float, float],
) -> list[TopoRoutedStrut]:
    """Connect pruned surface nodes of Face A to Face B across one shared edge."""
    na = _nodes_near_edge(fa.nodes_3d, p0, p1)
    nb = _nodes_near_edge(fb.nodes_3d, p0, p1)

    if not na and len(fa.nodes_3d) == 1:
        na = [0]
    if not nb and len(fb.nodes_3d) == 1:
        nb = [0]
    if not na and len(fa.nodes_3d) == 4:
        mid = 0.5 * (np.asarray(p0) + np.asarray(p1))
        na = [int(np.argmin(np.linalg.norm(fa.nodes_3d - mid, axis=1)))]
    if not nb and len(fb.nodes_3d) == 4:
        mid = 0.5 * (np.asarray(p0) + np.asarray(p1))
        nb = [int(np.argmin(np.linalg.norm(fb.nodes_3d - mid, axis=1)))]

    if not na or not nb:
        struts: list[TopoRoutedStrut] = []
        used_b: set[int] = set()
        for i, pa in enumerate(fa.nodes_3d):
            best_j = None
            best_d = float("inf")
            for j, pb in enumerate(fb.nodes_3d):
                if j in used_b:
                    continue
                d = float(np.linalg.norm(pa - pb))
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None and best_d > 1e-9:
                struts.append(
                    TopoRoutedStrut(fa.face_id, i, fb.face_id, best_j, edge_key)
                )
                used_b.add(best_j)
        return struts

    if len(na) == 1 and len(nb) == 1:
        ia, ib = na[0], nb[0]
        if float(np.linalg.norm(fa.nodes_3d[ia] - fb.nodes_3d[ib])) < 1e-9:
            return []
        return [TopoRoutedStrut(fa.face_id, ia, fb.face_id, ib, edge_key)]

    struts = []
    for ia in na:
        for ib in nb:
            if float(np.linalg.norm(fa.nodes_3d[ia] - fb.nodes_3d[ib])) < 1e-9:
                continue
            struts.append(TopoRoutedStrut(fa.face_id, ia, fb.face_id, ib, edge_key))
    return struts


def route_topological_surface_dual(
    faces: list[TopoFace],
    edge_map: dict[tuple[float, float, float], list[int]] | None = None,
    *,
    decimals: int = _EDGE_MID_DECIMALS,
) -> TopologicalDualResult:
    """
    Step 2 — manifold_stitch from topological adjacency (3D edge_map).

    For every edge shared by exactly two faces, connect their pruned
    Surface_Nodes. UV plot distance is ignored here.
    """
    if edge_map is None:
        edge_map = build_edge_map(faces, decimals=decimals)
    by_id = {f.face_id: f for f in faces}

    edge_seg: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=decimals)
            edge_seg.setdefault(key, (p0, p1))

    native: list[tuple[int, int, int, int]] = []
    for f in faces:
        for i, j in f.native_locals:
            native.append((f.face_id, int(i), f.face_id, int(j)))

    routed: list[TopoRoutedStrut] = []
    seen: set[tuple[int, int, int, int]] = set()
    n_shared = 0
    for key, fids in edge_map.items():
        if len(fids) != 2:
            continue
        n_shared += 1
        fa, fb = by_id[fids[0]], by_id[fids[1]]
        p0, p1 = edge_seg[key]
        for s in _route_pair_across_edge(fa, fb, p0, p1, key):
            a = (s.face_a, s.node_a, s.face_b, s.node_b)
            b = (s.face_b, s.node_b, s.face_a, s.node_a)
            if a in seen or b in seen:
                continue
            seen.add(a)
            routed.append(s)

    report = {
        "n_faces": len(faces),
        "n_edge_keys": len(edge_map),
        "n_shared_edges": n_shared,
        "n_boundary_edges": sum(1 for v in edge_map.values() if len(v) == 1),
        "n_native_struts": len(native),
        "n_routed_struts": len(routed),
        "n_full": sum(1 for f in faces if f.role == ROLE_FULL),
        "n_half_cut": sum(1 for f in faces if f.role == ROLE_HALF_CUT),
        "n_half_side": sum(1 for f in faces if f.role == ROLE_HALF_SIDE),
        "n_half_apex": sum(1 for f in faces if f.role == ROLE_HALF_APEX),
    }
    return TopologicalDualResult(
        faces=faces,
        edge_map=edge_map,
        native_struts=native,
        routed_struts=routed,
        report=report,
    )


def compute_topological_surface_dual(
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
    edge_decimals: int = _EDGE_MID_DECIMALS,
) -> TopologicalDualResult:
    """Full Task-22 pipeline: index faces -> edge_map -> topological routing."""
    faces = build_exposed_topo_faces(
        hex_elems, cell_tags, round_decimals=round_decimals
    )
    edge_map = build_edge_map(faces, decimals=edge_decimals)
    return route_topological_surface_dual(faces, edge_map, decimals=edge_decimals)
