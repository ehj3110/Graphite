"""
Proportional geometric unfold (net) of exposed SC faces (Task 18).

BFS spanning tree rooted at the closeness-centrality center of the face-hinge
graph. Full / Half-CUT / Half-apex faces map to 1×1 cells; Half-SIDE faces map
to 1×0.5 rectangles. Surface-dual edges longer than the local gate are drawn
as stubs, not global bridges.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from graphite.explicit.hex_rules import _HEX_FACES, _hex_face_centers
from graphite.explicit.hex_topology_module import resolve_quantized_cell_rule
from graphite.explicit.sc_contextual_surface import (
    ROLE_FULL,
    ROLE_HALF_APEX,
    ROLE_HALF_CUT,
    ROLE_HALF_SIDE,
    classify_exposed_face_role,
    face_plot_size,
)
from graphite.explicit.sc_quantized_surface import (
    _corner_key,
    _edge_key,
    _face_key,
    _outward_face_normal,
)

_LOCAL_GATE = 1.5  # max contiguous dual length on the face grid


@dataclass
class NetFace:
    face_id: int
    hex_i: int
    face_i: int
    corners_3d: np.ndarray  # (4, 3) winding order
    center_3d: np.ndarray
    normal: np.ndarray
    primary_gid: int | None
    corner_keys: tuple[tuple[float, float, float], ...]
    edge_keys: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ]
    role: str = "full"
    size_uv: tuple[float, float] = (1.0, 1.0)  # (width, depth)
    gids: tuple[int, ...] = ()  # contextual surface node ids for this face


@dataclass
class FaceUnfold:
    """Proportional-grid placement of one exposed face."""

    face_id: int
    grid_ij: tuple[float, float]  # lower-left of the face rectangle
    corners_uv: np.ndarray  # (4, 2) matching corners_3d winding
    center_uv: np.ndarray
    size_uv: tuple[float, float] = (1.0, 1.0)  # (width along hinge, depth)
    role: str = "full"
    # Map face-plane offsets → UV within the cell
    origin_3d: np.ndarray = field(default_factory=lambda: np.zeros(3))
    basis_3d: np.ndarray = field(default_factory=lambda: np.eye(2, 3))
    origin_uv: np.ndarray = field(default_factory=lambda: np.zeros(2))
    basis_uv: np.ndarray = field(default_factory=lambda: np.eye(2))
    # For Half SIDE: UV endpoints of the mid-plane cut edge (node sits at midpoint)
    cut_edge_uv: tuple[np.ndarray, np.ndarray] | None = None
    hinge_edge_uv: tuple[np.ndarray, np.ndarray] | None = None


@dataclass
class NetUnfoldResult:
    faces: list[NetFace]
    hinge_adj: dict[int, set[int]]
    hinges: set[tuple[int, int]]
    cuts: set[tuple[int, int]]
    root_id: int
    unfolds: dict[int, FaceUnfold]
    node_uv: dict[int, np.ndarray]
    node_face: dict[int, int]  # surface node → owning/nearest unfolded face
    report: dict = field(default_factory=dict)


def _tag_to_rule(tag: str | None) -> str | None:
    return resolve_quantized_cell_rule(tag)


def build_net_faces(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
    skin_faces=None,
) -> list[NetFace]:
    """Exposed SC faces with full 3D corner geometry and Task-18 roles."""
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    key_to_gid = {_corner_key(p, round_decimals): i for i, p in enumerate(pts)}

    skin_lookup: dict[tuple[int, int], object] = {}
    if skin_faces is not None:
        for sf in skin_faces:
            skin_lookup[(int(sf.hex_i), int(sf.face_i))] = sf

    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    rules: list[str | None] = []
    for hi, (corners, tag) in enumerate(zip(elems, tags)):
        rule = _tag_to_rule(tag)
        rules.append(rule)
        if rule is None:
            continue
        for fi, face in enumerate(_HEX_FACES):
            face_owners.setdefault(_face_key(corners, face, round_decimals), []).append(
                (hi, fi)
            )

    out: list[NetFace] = []
    for hi, (corners, rule) in enumerate(zip(elems, rules)):
        if rule is None:
            continue
        c = np.asarray(corners, dtype=np.float64)
        face_ctrs = _hex_face_centers(c)
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(c, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            corners_3d = np.array([c[i] for i in face], dtype=np.float64)
            primary = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
            corner_keys = tuple(_corner_key(corners_3d[i], round_decimals) for i in range(4))
            edge_keys = tuple(
                _edge_key(corners_3d[a], corners_3d[b], round_decimals)
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            role = classify_exposed_face_role(rule, fi)
            size = face_plot_size(role)
            gids: tuple[int, ...] = ()
            sf = skin_lookup.get((hi, fi))
            if sf is not None:
                role = str(sf.role)
                size = face_plot_size(role)
                gids = tuple(int(g) for g in sf.gids)
                if gids:
                    primary = int(gids[0]) if len(gids) == 1 else primary
            out.append(
                NetFace(
                    face_id=len(out),
                    hex_i=hi,
                    face_i=fi,
                    corners_3d=corners_3d,
                    center_3d=corners_3d.mean(axis=0),
                    normal=_outward_face_normal(c, fi),
                    primary_gid=None if primary is None else int(primary),
                    corner_keys=corner_keys,
                    edge_keys=edge_keys,
                    role=role,
                    size_uv=size,
                    gids=gids,
                )
            )
    return out


def build_face_hinge_graph(
    faces: list[NetFace],
) -> tuple[dict[int, set[int]], dict[tuple, list[int]]]:
    """Adjacency of exposed faces that share a Cartesian edge."""
    edge_to_faces: dict[tuple, list[int]] = {}
    for f in faces:
        for e in f.edge_keys:
            edge_to_faces.setdefault(e, []).append(f.face_id)

    adj: dict[int, set[int]] = {f.face_id: set() for f in faces}
    for fids in edge_to_faces.values():
        uniq = list(dict.fromkeys(fids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                adj[a].add(b)
                adj[b].add(a)
    return adj, edge_to_faces


def select_root_face(faces: list[NetFace], adj: dict[int, set[int]] | None = None) -> int:
    """
    Topological center of the exposed-face hinge graph (closeness centrality).

    Falls back to the largest connected component if the graph is disconnected.
    """
    if not faces:
        raise ValueError("no exposed faces")
    if adj is None:
        adj, _ = build_face_hinge_graph(faces)

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx required for closeness-centrality root") from exc

    g = nx.Graph()
    g.add_nodes_from(f.face_id for f in faces)
    for u, nbrs in adj.items():
        for v in nbrs:
            if u < v:
                g.add_edge(u, v)

    if g.number_of_nodes() == 0:
        return int(faces[0].face_id)

    # Largest connected component
    components = list(nx.connected_components(g))
    largest = max(components, key=len)
    sub = g.subgraph(largest).copy()
    if sub.number_of_edges() == 0:
        # Isolated faces — pick the one with highest Z, then XY-central
        centers = {f.face_id: f.center_3d for f in faces if f.face_id in largest}
        z_max = max(float(c[2]) for c in centers.values())
        xy = np.mean([c[:2] for c in centers.values()], axis=0)
        return min(
            centers.keys(),
            key=lambda fid: (
                0 if centers[fid][2] >= z_max - 1e-9 else 1,
                float(np.linalg.norm(centers[fid][:2] - xy)),
            ),
        )

    closeness = nx.closeness_centrality(sub)
    return int(max(closeness, key=closeness.get))


def bfs_spanning_tree(
    adj: dict[int, set[int]],
    root_id: int,
) -> tuple[set[tuple[int, int]], dict[int, int | None]]:
    """BFS spanning tree — hinges are tree edges ``(min,max)``."""
    parent: dict[int, int | None] = {root_id: None}
    hinges: set[tuple[int, int]] = set()
    q: deque[int] = deque([root_id])
    while q:
        u = q.popleft()
        for v in sorted(adj.get(u, ())):
            if v in parent:
                continue
            parent[v] = u
            hinges.add((u, v) if u < v else (v, u))
            q.append(v)
    return hinges, parent


def _shared_edge_keys(fa: NetFace, fb: NetFace) -> list[tuple]:
    return list(set(fa.edge_keys) & set(fb.edge_keys))


def _edge_corner_indices(face: NetFace, edge_key: tuple) -> tuple[int, int]:
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ka, kb = face.corner_keys[a], face.corner_keys[b]
        ek = (ka, kb) if ka <= kb else (kb, ka)
        if ek == edge_key:
            return a, b
    raise KeyError("edge not on face")


def _side_of_rect_edge(
    p0: np.ndarray,
    p1: np.ndarray,
    width: float,
    height: float,
) -> str:
    """Which side of the rectangle [0,w]×[0,h] an edge lies on."""
    mid = 0.5 * (p0 + p1)
    w = float(width)
    h = float(height)
    dists = {
        "s": float(mid[1]),
        "n": float(h - mid[1]),
        "w": float(mid[0]),
        "e": float(w - mid[0]),
    }
    return min(dists, key=dists.get)


def _local_corner_uv_unit_square(face: NetFace) -> np.ndarray:
    """Map face winding to the unit square [0,1]^2 (root local frame)."""
    c = face.corners_3d
    e1 = c[1] - c[0]
    n1 = float(np.linalg.norm(e1))
    if n1 < 1e-14:
        e1 = c[2] - c[0]
        n1 = float(np.linalg.norm(e1))
    e1 = e1 / max(n1, 1e-14)
    n = np.asarray(face.normal, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-14:
        n = np.cross(c[1] - c[0], c[3] - c[0])
        nn = float(np.linalg.norm(n))
    n = n / max(nn, 1e-14)
    e2 = np.cross(n, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-14)

    local = np.zeros((4, 2), dtype=np.float64)
    for i, p in enumerate(c):
        d = p - c[0]
        local[i, 0] = float(np.dot(d, e1))
        local[i, 1] = float(np.dot(d, e2))
    mn = local.min(axis=0)
    mx = local.max(axis=0)
    span = np.maximum(mx - mn, 1e-14)
    return (local - mn) / span


def _rect_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    eps: float = 1e-9,
) -> bool:
    au, av, aw, ah = a
    bu, bv, bw, bh = b
    return not (
        au + aw <= bu + eps
        or bu + bw <= au + eps
        or av + ah <= bv + eps
        or bv + bh <= av + eps
    )


def _place_root_grid(face: NetFace) -> FaceUnfold:
    local = _local_corner_uv_unit_square(face)
    w, h = face.size_uv
    # Scale unit square to proportional rect with LL at origin
    corners_uv = local * np.array([w, h], dtype=np.float64)
    center_uv = np.array([0.5 * w, 0.5 * h], dtype=np.float64)
    e1_3d = face.corners_3d[1] - face.corners_3d[0]
    e1_3d = e1_3d / max(float(np.linalg.norm(e1_3d)), 1e-14)
    n = face.normal / max(float(np.linalg.norm(face.normal)), 1e-14)
    e2_3d = np.cross(n, e1_3d)
    e2_3d = e2_3d / max(float(np.linalg.norm(e2_3d)), 1e-14)
    origin_3d = face.center_3d.copy()
    uf = FaceUnfold(
        face_id=face.face_id,
        grid_ij=(0.0, 0.0),
        corners_uv=corners_uv,
        center_uv=center_uv,
        size_uv=(w, h),
        role=face.role,
        origin_3d=origin_3d,
        basis_3d=np.vstack((e1_3d, e2_3d)),
        origin_uv=center_uv.copy(),
        basis_uv=_fit_basis_uv(face, corners_uv, origin_3d, e1_3d, e2_3d, center_uv),
    )
    if face.role == ROLE_HALF_SIDE:
        uf.cut_edge_uv = _infer_side_cut_edge_uv(face, uf)
    return uf


def _infer_side_cut_edge_uv(
    face: NetFace,
    uf: FaceUnfold,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick the UV rectangle edge whose midpoint is closest to the 3D cut-edge
    midpoint (side face center = mid-plane diamond node).
    """
    target = _map_point_uv(uf, face.center_3d)
    best = None
    best_d = float("inf")
    c = uf.corners_uv
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        mid = 0.5 * (c[a] + c[b])
        d = float(np.linalg.norm(mid - target))
        if d < best_d:
            best_d = d
            best = (c[a].copy(), c[b].copy())
    assert best is not None
    return best


def _fit_basis_uv(
    face: NetFace,
    corners_uv: np.ndarray,
    origin_3d: np.ndarray,
    e1_3d: np.ndarray,
    e2_3d: np.ndarray,
    origin_uv: np.ndarray,
) -> np.ndarray:
    """Least-squares map: local (e1,e2) coords → UV offset from center."""
    A = []
    B = []
    for i in range(4):
        d = face.corners_3d[i] - origin_3d
        loc = np.array([np.dot(d, e1_3d), np.dot(d, e2_3d)], dtype=np.float64)
        duv = corners_uv[i] - origin_uv
        A.append(loc)
        B.append(duv)
    A = np.asarray(A)
    B = np.asarray(B)
    M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return M.T


def _map_point_uv(uf: FaceUnfold, p3: np.ndarray) -> np.ndarray:
    d = np.asarray(p3, dtype=np.float64) - uf.origin_3d
    loc = uf.basis_3d @ d
    return uf.origin_uv + uf.basis_uv @ loc


def _place_child_grid(
    parent_face: NetFace,
    parent_uf: FaceUnfold,
    child_face: NetFace,
    edge_key: tuple,
    occupied: list[tuple[float, float, float, float]],
) -> FaceUnfold:
    """Seat child across the shared hinge with proportional depth."""
    ia, ib = _edge_corner_indices(parent_face, edge_key)
    p0 = parent_uf.corners_uv[ia]
    p1 = parent_uf.corners_uv[ib]
    gu, gv = parent_uf.grid_ij
    pw, ph = parent_uf.size_uv
    local0 = p0 - np.array([gu, gv], dtype=np.float64)
    local1 = p1 - np.array([gu, gv], dtype=np.float64)
    side = _side_of_rect_edge(local0, local1, pw, ph)

    cw, ch_nominal = child_face.size_uv
    # Depth is the half-size dimension for SIDE faces; along-hinge width stays 1
    depth = float(ch_nominal) if child_face.role == ROLE_HALF_SIDE else float(cw)
    # For SIDE, size_uv is (1.0, 0.5): width along hinge=1, depth=0.5
    if child_face.role == ROLE_HALF_SIDE:
        depth = 0.5
        width = 1.0
    else:
        depth = 1.0
        width = 1.0

    delta_unit = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}[side]
    # Child LL: step outward by parent extent on that side, then place rect
    if side == "n":
        child_ll = (gu, gv + ph)
        child_size = (width, depth)
    elif side == "s":
        child_ll = (gu, gv - depth)
        child_size = (width, depth)
    elif side == "e":
        child_ll = (gu + pw, gv)
        child_size = (depth, width)  # depth along U
        # For e/w hinges the "depth" extends in U; rect is depth×width
        child_size = (depth, width)
    else:  # w
        child_ll = (gu - depth, gv)
        child_size = (depth, width)

    rect = (child_ll[0], child_ll[1], child_size[0], child_size[1])
    if any(_rect_overlap(rect, r) for r in occupied):
        for d in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            cand_ll = (gu + d[0] * max(pw, depth), gv + d[1] * max(ph, depth))
            cand = (cand_ll[0], cand_ll[1], child_size[0], child_size[1])
            if not any(_rect_overlap(cand, r) for r in occupied):
                child_ll = cand_ll
                rect = cand
                delta_unit = d
                break

    into = np.array(delta_unit, dtype=np.float64)
    into = into / max(float(np.linalg.norm(into)), 1e-14)

    e0_uv, e1_uv = p0.copy(), p1.copy()
    child_corners_uv = np.zeros((4, 2), dtype=np.float64)
    child_keys = child_face.corner_keys
    ci, cj = _edge_corner_indices(child_face, edge_key)
    if child_keys[ci] == parent_face.corner_keys[ia]:
        child_corners_uv[ci] = e0_uv
        child_corners_uv[cj] = e1_uv
    else:
        child_corners_uv[ci] = e1_uv
        child_corners_uv[cj] = e0_uv

    nxt = (ci + 1) % 4
    prv = (ci - 1) % 4
    into_d = into * depth
    if nxt == cj:
        far_cj = (cj + 1) % 4
        far_ci = (far_cj + 1) % 4
        child_corners_uv[far_cj] = child_corners_uv[cj] + into_d
        child_corners_uv[far_ci] = child_corners_uv[ci] + into_d
    elif prv == cj:
        far_ci = (ci + 1) % 4
        far_cj = (far_ci + 1) % 4
        child_corners_uv[far_ci] = child_corners_uv[ci] + into_d
        child_corners_uv[far_cj] = child_corners_uv[cj] + into_d
    else:
        others = [k for k in range(4) if k not in (ci, cj)]
        child_corners_uv[others[0]] = child_corners_uv[ci] + into_d
        child_corners_uv[others[1]] = child_corners_uv[cj] + into_d

    center_uv = child_corners_uv.mean(axis=0)
    e1_3d = child_face.corners_3d[1] - child_face.corners_3d[0]
    e1_3d = e1_3d / max(float(np.linalg.norm(e1_3d)), 1e-14)
    n = child_face.normal / max(float(np.linalg.norm(child_face.normal)), 1e-14)
    e2_3d = np.cross(n, e1_3d)
    e2_3d = e2_3d / max(float(np.linalg.norm(e2_3d)), 1e-14)
    origin_3d = child_face.center_3d.copy()
    basis_uv = _fit_basis_uv(
        child_face, child_corners_uv, origin_3d, e1_3d, e2_3d, center_uv
    )
    hinge_edge = (child_corners_uv[ci].copy(), child_corners_uv[cj].copy())
    uf = FaceUnfold(
        face_id=child_face.face_id,
        grid_ij=(float(rect[0]), float(rect[1])),
        corners_uv=child_corners_uv,
        center_uv=center_uv,
        size_uv=(float(rect[2]), float(rect[3])),
        role=child_face.role,
        origin_3d=origin_3d,
        basis_3d=np.vstack((e1_3d, e2_3d)),
        origin_uv=center_uv.copy(),
        basis_uv=basis_uv,
        hinge_edge_uv=hinge_edge if child_face.role == ROLE_HALF_SIDE else None,
    )
    if child_face.role == ROLE_HALF_SIDE:
        uf.cut_edge_uv = _infer_side_cut_edge_uv(child_face, uf)
    return uf


def unfold_faces_integer_grid(
    faces: list[NetFace],
    parent: dict[int, int | None],
    root_id: int,
) -> dict[int, FaceUnfold]:
    """BFS walk placing each face on a compact proportional grid."""
    by_id = {f.face_id: f for f in faces}
    unfolds: dict[int, FaceUnfold] = {root_id: _place_root_grid(by_id[root_id])}
    occupied: list[tuple[float, float, float, float]] = [
        (
            unfolds[root_id].grid_ij[0],
            unfolds[root_id].grid_ij[1],
            unfolds[root_id].size_uv[0],
            unfolds[root_id].size_uv[1],
        )
    ]

    children: dict[int, list[int]] = {fid: [] for fid in parent}
    for fid, par in parent.items():
        if par is not None:
            children[par].append(fid)

    q: deque[int] = deque([root_id])
    while q:
        u = q.popleft()
        for v in children.get(u, []):
            shared = _shared_edge_keys(by_id[u], by_id[v])
            if not shared:
                continue
            uf = _place_child_grid(
                by_id[u], unfolds[u], by_id[v], shared[0], occupied
            )
            occupied.append(
                (uf.grid_ij[0], uf.grid_ij[1], uf.size_uv[0], uf.size_uv[1])
            )
            unfolds[v] = uf
            q.append(v)
    return unfolds


def assign_surface_node_uv(
    nodes: np.ndarray,
    surface_mask: np.ndarray,
    faces: list[NetFace],
    unfolds: dict[int, FaceUnfold],
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """
    Map surface nodes to UV inside proportional face rectangles.

    Full / apex: face center.
    Half SIDE: center of the half-rectangle.
    Half CUT: 4 diamond nodes → midpoints of the 4 rectangle sides (◇).
    """
    pts = np.asarray(nodes, dtype=np.float64)
    mask = np.asarray(surface_mask, dtype=bool)
    face_list = [f for f in faces if f.face_id in unfolds]
    node_uv: dict[int, np.ndarray] = {}
    node_face: dict[int, int] = {}

    # First: place explicitly owned contextual gids
    for f in face_list:
        uf = unfolds[f.face_id]
        gu, gv = uf.grid_ij
        w, h = uf.size_uv
        if f.role == ROLE_HALF_CUT and len(f.gids) == 4:
            # Side midpoints of the proportional rectangle → diamond in UV
            mids = [
                np.array([gu + 0.5 * w, gv], dtype=np.float64),
                np.array([gu + w, gv + 0.5 * h], dtype=np.float64),
                np.array([gu + 0.5 * w, gv + h], dtype=np.float64),
                np.array([gu, gv + 0.5 * h], dtype=np.float64),
            ]
            # Match 3D diamond order to UV mids by angular sort in face plane
            gids = list(f.gids)
            c3 = pts[gids].mean(axis=0)
            e1 = uf.basis_3d[0]
            e2 = uf.basis_3d[1]
            order = sorted(
                range(4),
                key=lambda i: float(
                    np.arctan2(
                        np.dot(pts[gids[i]] - c3, e2),
                        np.dot(pts[gids[i]] - c3, e1),
                    )
                ),
            )
            for k, oi in enumerate(order):
                gid = gids[oi]
                node_uv[gid] = mids[k % 4].copy()
                node_face[gid] = f.face_id
            continue

        if len(f.gids) == 1:
            gid = int(f.gids[0])
            node_uv[gid] = uf.center_uv.copy()
            node_face[gid] = f.face_id
        elif f.primary_gid is not None:
            node_uv[int(f.primary_gid)] = uf.center_uv.copy()
            node_face[int(f.primary_gid)] = f.face_id

    # Remaining surface nodes: nearest unfolded face + clamp into rect
    for gid in np.flatnonzero(mask):
        gid = int(gid)
        if gid in node_uv:
            continue
        p = pts[gid]
        best_f = min(face_list, key=lambda f: float(np.linalg.norm(p - f.center_3d)))
        fid = best_f.face_id
        uf = unfolds[fid]
        uv = _map_point_uv(uf, p)
        gu, gv = uf.grid_ij
        w, h = uf.size_uv
        uv = np.clip(uv, [gu, gv], [gu + w, gv + h])
        node_uv[gid] = uv
        node_face[gid] = fid
    return node_uv, node_face


def compute_surface_net_unfold(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    surface_mask: np.ndarray,
    *,
    round_decimals: int = 6,
    skin_faces=None,
) -> NetUnfoldResult:
    """Faces → centrality root → BFS hinges → proportional unfold → node UV."""
    faces = build_net_faces(
        nodes,
        hex_elems,
        cell_tags,
        round_decimals=round_decimals,
        skin_faces=skin_faces,
    )
    if not faces:
        return NetUnfoldResult(
            faces=[],
            hinge_adj={},
            hinges=set(),
            cuts=set(),
            root_id=-1,
            unfolds={},
            node_uv={},
            node_face={},
            report={"n_faces": 0},
        )

    adj, _ = build_face_hinge_graph(faces)
    root_id = select_root_face(faces, adj)
    hinges, parent = bfs_spanning_tree(adj, root_id)

    cuts: set[tuple[int, int]] = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u >= v:
                continue
            e = (u, v)
            if e not in hinges:
                cuts.add(e)

    unfolds = unfold_faces_integer_grid(faces, parent, root_id)
    node_uv, node_face = assign_surface_node_uv(
        nodes, surface_mask, faces, unfolds
    )

    grids = list(unfolds.values())
    if grids:
        u0 = min(uf.grid_ij[0] for uf in grids)
        v0 = min(uf.grid_ij[1] for uf in grids)
        u1 = max(uf.grid_ij[0] + uf.size_uv[0] for uf in grids)
        v1 = max(uf.grid_ij[1] + uf.size_uv[1] for uf in grids)
        span_u = u1 - u0
        span_v = v1 - v0
    else:
        span_u = span_v = 0.0

    n_side = sum(1 for f in faces if f.role == ROLE_HALF_SIDE)
    n_cut = sum(1 for f in faces if f.role == ROLE_HALF_CUT)
    return NetUnfoldResult(
        faces=faces,
        hinge_adj=adj,
        hinges=hinges,
        cuts=cuts,
        root_id=root_id,
        unfolds=unfolds,
        node_uv=node_uv,
        node_face=node_face,
        report={
            "n_faces": len(faces),
            "n_hinges": len(hinges),
            "n_cuts": len(cuts),
            "n_unfolded": len(unfolds),
            "root_id": root_id,
            "n_node_uv": len(node_uv),
            "n_half_side": n_side,
            "n_half_cut": n_cut,
            "grid_span_u": float(span_u),
            "grid_span_v": float(span_v),
        },
    )


def _stub_to_cell_edge(
    p: np.ndarray,
    target: np.ndarray,
    grid_ij: tuple[float, float],
    size_uv: tuple[float, float] = (1.0, 1.0),
    *,
    frac: float = 0.35,
) -> np.ndarray:
    """Short stub from ``p`` toward ``target``, ending at the local face rect edge."""
    gu, gv = float(grid_ij[0]), float(grid_ij[1])
    w, h = float(size_uv[0]), float(size_uv[1])
    direction = np.asarray(target, dtype=np.float64) - np.asarray(p, dtype=np.float64)
    nrm = float(np.linalg.norm(direction))
    if nrm < 1e-14:
        return np.asarray(p, dtype=np.float64).copy()
    u = direction / nrm
    t_cands: list[float] = []
    if abs(u[0]) > 1e-12:
        for x in (gu, gu + w):
            t = (x - p[0]) / u[0]
            if t > 1e-9:
                y = p[1] + t * u[1]
                if gv - 1e-9 <= y <= gv + h + 1e-9:
                    t_cands.append(t)
    if abs(u[1]) > 1e-12:
        for y in (gv, gv + h):
            t = (y - p[1]) / u[1]
            if t > 1e-9:
                x = p[0] + t * u[0]
                if gu - 1e-9 <= x <= gu + w + 1e-9:
                    t_cands.append(t)
    if not t_cands:
        q = p + float(frac) * u
        return np.clip(q, [gu, gv], [gu + w, gv + h])
    t_edge = min(t_cands)
    q = p + float(frac) * t_edge * u
    return np.clip(q, [gu, gv], [gu + w, gv + h])


def plot_surface_dual_net(
    nodes: np.ndarray,
    surface_mask: np.ndarray,
    native_surface_struts: np.ndarray,
    manifold_struts: np.ndarray,
    net: NetUnfoldResult,
    out_path: str | Path,
    *,
    node_size: float = 10.0,
    edge_alpha: float = 0.55,
    local_gate: float = _LOCAL_GATE,
    stub_frac: float = 0.35,
    dpi: int = 160,
    figsize: tuple[float, float] = (14.0, 12.0),
) -> Path:
    """
    Proportional-grid net plot (Task 18).

    Full / CUT faces: 1×1 grey squares. Half SIDE faces: 1×0.5 rectangles.
    Dual edges with plot distance < ``local_gate`` are drawn fully; longer
    edges become cut stubs into the node's local face rectangle.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if not net.unfolds:
        ax.set_title("Surface dual net (empty)")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    for uf in net.unfolds.values():
        gu, gv = uf.grid_ij
        w, h = uf.size_uv
        is_side = uf.role == ROLE_HALF_SIDE
        ax.add_patch(
            Rectangle(
                (gu, gv),
                w,
                h,
                facecolor="#e8f0fe" if is_side else "#f2f2f2",
                edgecolor="#9db4d8" if is_side else "#cccccc",
                linewidth=0.7,
                zorder=0,
            )
        )

    pos = {gid: (float(uv[0]), float(uv[1])) for gid, uv in net.node_uv.items()}

    native = (
        np.asarray(native_surface_struts, dtype=np.int64).reshape(-1, 2)
        if len(native_surface_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    man = (
        np.asarray(manifold_struts, dtype=np.int64).reshape(-1, 2)
        if len(manifold_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    native_set = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in native
        if int(a) != int(b)
    }
    man_set = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in man
        if int(a) != int(b)
        and (min(int(a), int(b)), max(int(a), int(b))) not in native_set
    }

    n_native = n_man = n_stub = 0
    gate = float(local_gate)

    for ekey, color, is_native in (
        *((e, "#1f77b4", True) for e in native_set),
        *((e, "#d62728", False) for e in man_set),
    ):
        a, b = ekey
        if a not in pos or b not in pos:
            continue
        pa = np.asarray(pos[a], dtype=np.float64)
        pb = np.asarray(pos[b], dtype=np.float64)
        dist = float(np.linalg.norm(pa - pb))
        if dist < gate:
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                color=color,
                lw=0.9 if is_native else 1.1,
                alpha=float(edge_alpha),
                zorder=2,
            )
            if is_native:
                n_native += 1
            else:
                n_man += 1
        else:
            for gid, p, other in ((a, pa, pb), (b, pb, pa)):
                fid = net.node_face.get(gid)
                if fid is None or fid not in net.unfolds:
                    continue
                uf = net.unfolds[fid]
                q = _stub_to_cell_edge(
                    p, other, uf.grid_ij, uf.size_uv, frac=float(stub_frac)
                )
                ax.plot(
                    [p[0], q[0]],
                    [p[1], q[1]],
                    color=color,
                    lw=1.0,
                    alpha=0.35,
                    solid_capstyle="round",
                    zorder=2,
                )
            n_stub += 1

    if pos:
        xy = np.array(list(pos.values()), dtype=np.float64)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=float(node_size),
            c="#222222",
            alpha=0.85,
            linewidths=0,
            zorder=3,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Proportional surface net "
        f"(root=closeness, local_gate={gate:g})\n"
        f"faces={net.report.get('n_faces')} "
        f"(side={net.report.get('n_half_side', 0)} cut={net.report.get('n_half_cut', 0)}) "
        f"span={net.report.get('grid_span_u'):.1f}×{net.report.get('grid_span_v'):.1f}  "
        f"dual: local_native={n_native} local_routed={n_man} stubs={n_stub}"
    )
    ax.set_xlabel("U (proportional face grid)")
    ax.set_ylabel("V (proportional face grid)")
    legend = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="native (CUT diamond)"),
        Line2D([0], [0], color="#d62728", lw=2, label="routed (adjacent dual)"),
        Line2D([0], [0], color="#555555", lw=1.5, alpha=0.35, label="cut stub (≥gate)"),
        Line2D([0], [0], color="#cccccc", lw=2, label="1×1 Full/CUT"),
        Line2D([0], [0], color="#9db4d8", lw=2, label="½ SIDE face"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Task 19–21 — local node placement audit + strict pruning
# ---------------------------------------------------------------------------

# Task 21: no visual inset — CUT edge-mids must coincide with SIDE cut-edge
# midpoints on shared boundaries (weld verification).
_CUT_NODE_INSET = 0.0


def _rect_edge_mids_uv(gu: float, gv: float, w: float, h: float) -> list[np.ndarray]:
    return [
        np.array([gu + 0.5 * w, gv], dtype=np.float64),
        np.array([gu + w, gv + 0.5 * h], dtype=np.float64),
        np.array([gu + 0.5 * w, gv + h], dtype=np.float64),
        np.array([gu, gv + 0.5 * h], dtype=np.float64),
    ]


def _aabb_edges_uv(
    gu: float, gv: float, w: float, h: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four edges of the axis-aligned plot rectangle (south, east, north, west)."""
    sw = np.array([gu, gv], dtype=np.float64)
    se = np.array([gu + w, gv], dtype=np.float64)
    ne = np.array([gu + w, gv + h], dtype=np.float64)
    nw = np.array([gu, gv + h], dtype=np.float64)
    return [(sw, se), (se, ne), (ne, nw), (nw, sw)]


def side_cut_edge_and_midpoint_uv(
    uf: FaceUnfold,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Task 21 — Half SIDE cut edge + exact midpoint on the plot rectangle.

    Locates the cut edge among the 4 AABB edges of the 1×0.5 / 0.5×1 rect,
    then returns that segment and its geometric midpoint.

    The surface_nodes list for a SIDE face must contain ONLY this midpoint
    (never the segment endpoints / corners).
    """
    gu, gv = float(uf.grid_ij[0]), float(uf.grid_ij[1])
    w, h = float(uf.size_uv[0]), float(uf.size_uv[1])
    edges = _aabb_edges_uv(gu, gv, w, h)

    # Prefer previously inferred cut edge, snapped to nearest AABB edge.
    if uf.cut_edge_uv is not None:
        ca = np.asarray(uf.cut_edge_uv[0], dtype=np.float64)
        cb = np.asarray(uf.cut_edge_uv[1], dtype=np.float64)
        cmid = 0.5 * (ca + cb)
        best = min(
            edges,
            key=lambda e: float(np.linalg.norm(0.5 * (e[0] + e[1]) - cmid)),
        )
    elif uf.hinge_edge_uv is not None:
        # Cut edge = AABB edge farthest from the hinge (mid-plane side).
        ha = np.asarray(uf.hinge_edge_uv[0], dtype=np.float64)
        hb = np.asarray(uf.hinge_edge_uv[1], dtype=np.float64)
        hmid = 0.5 * (ha + hb)
        best = max(
            edges,
            key=lambda e: float(np.linalg.norm(0.5 * (e[0] + e[1]) - hmid)),
        )
    else:
        # Longer edges are the length-1 sides of a half-rect; pick the one
        # whose midpoint is farther from the rectangle center along the
        # short axis (prefer boundary, not floating center).
        center = np.array([gu + 0.5 * w, gv + 0.5 * h], dtype=np.float64)
        long_edges = [
            e
            for e in edges
            if float(np.linalg.norm(e[1] - e[0])) >= 0.5 * max(w, h) - 1e-9
        ]
        pool = long_edges if long_edges else edges
        best = max(
            pool,
            key=lambda e: float(np.linalg.norm(0.5 * (e[0] + e[1]) - center)),
        )

    a = np.asarray(best[0], dtype=np.float64).copy()
    b = np.asarray(best[1], dtype=np.float64).copy()
    mid = 0.5 * (a + b)
    return (a, b), mid


def _point_on_segment(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    tol: float = 1e-6,
) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-28:
        return float(np.linalg.norm(p - a)) <= tol
    t = float(np.dot(p - a, ab) / L2)
    if t < -1e-9 or t > 1.0 + 1e-9:
        return False
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)) <= tol


def prune_extracted_face_uv(
    uf: FaceUnfold,
    *,
    cut_inset: float = _CUT_NODE_INSET,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Strict geometric prune for one unfolded face (Tasks 20–21).

    Full / apex (grey):
      retain ONLY the geometric center; native_surface_struts = [].
    Half SIDE (blue):
      retain ONLY the geometric midpoint of the cut edge (delete corners);
      native_surface_struts = [].
    Half CUT (yellow):
      retain exactly 4 edge-midpoint diamond nodes (no inset);
      native_surface_struts = the 4 diamond perimeter edges.
    """
    gu, gv = float(uf.grid_ij[0]), float(uf.grid_ij[1])
    w, h = float(uf.size_uv[0]), float(uf.size_uv[1])
    center = np.array([gu + 0.5 * w, gv + 0.5 * h], dtype=np.float64)

    # --- Full / apex ---------------------------------------------------------
    if uf.role in (ROLE_FULL, ROLE_HALF_APEX) or uf.role not in (
        ROLE_HALF_CUT,
        ROLE_HALF_SIDE,
    ):
        return [center.copy()], []

    # --- Half SIDE: exactly ONE cut-edge midpoint ----------------------------
    if uf.role == ROLE_HALF_SIDE:
        (a, b), mid = side_cut_edge_and_midpoint_uv(uf)
        # Persist snapped AABB cut edge for dashed audit overlay
        uf.cut_edge_uv = (a, b)
        # Defense: never emit endpoints
        if float(np.linalg.norm(mid - a)) < 1e-12 or float(np.linalg.norm(mid - b)) < 1e-12:
            mid = 0.5 * (a + b)
        return [mid.copy()], []

    # --- Half CUT: 4 edge midpoints, no inset --------------------------------
    assert uf.role == ROLE_HALF_CUT
    del cut_inset  # Task 21: force zero inset for weld-overlap verification
    nodes = _rect_edge_mids_uv(gu, gv, w, h)
    native = [(nodes[i], nodes[(i + 1) % 4]) for i in range(4)]
    return nodes, native


def local_audit_nodes_uv(uf: FaceUnfold) -> list[np.ndarray]:
    """Canonical UV nodes after Task-20/21 strict prune."""
    nodes, _native = prune_extracted_face_uv(uf)
    return nodes


def local_audit_native_segments_uv(
    uf: FaceUnfold,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Native local struts after prune — CUT diamond only."""
    _nodes, native = prune_extracted_face_uv(uf)
    return native


def assert_pruned_face_uv(uf: FaceUnfold) -> list[str]:
    """Return violation messages if prune invariants fail."""
    nodes, native = prune_extracted_face_uv(uf)
    errs: list[str] = []
    gu, gv = float(uf.grid_ij[0]), float(uf.grid_ij[1])
    w, h = float(uf.size_uv[0]), float(uf.size_uv[1])
    center = np.array([gu + 0.5 * w, gv + 0.5 * h], dtype=np.float64)

    if uf.role in (ROLE_FULL, ROLE_HALF_APEX):
        if len(nodes) != 1:
            errs.append(f"full/apex expected 1 node, got {len(nodes)}")
        elif float(np.linalg.norm(nodes[0] - center)) > 1e-9:
            errs.append("full/apex node is not face center")
        if native:
            errs.append(f"full/apex must have 0 native struts, got {len(native)}")

    elif uf.role == ROLE_HALF_SIDE:
        if len(nodes) != 1:
            errs.append(f"side expected 1 node, got {len(nodes)}")
        if native:
            errs.append(f"side must have 0 native struts, got {len(native)}")
        (a, b), mid = side_cut_edge_and_midpoint_uv(uf)
        if len(nodes) == 1:
            if float(np.linalg.norm(nodes[0] - mid)) > 1e-6:
                errs.append("side node is not cut-edge midpoint")
            if float(np.linalg.norm(nodes[0] - a)) < 1e-6 or float(
                np.linalg.norm(nodes[0] - b)
            ) < 1e-6:
                errs.append("side node landed on a cut-edge CORNER (illegal)")
            if float(np.linalg.norm(nodes[0] - center)) < 1e-6:
                errs.append("side node coincides with rect center")

    elif uf.role == ROLE_HALF_CUT:
        if len(nodes) != 4:
            errs.append(f"cut expected 4 nodes, got {len(nodes)}")
        if len(native) != 4:
            errs.append(f"cut expected 4 native struts, got {len(native)}")
        # Exact edge mids (no inset)
        expected = _rect_edge_mids_uv(gu, gv, w, h)
        for i, (p, e) in enumerate(zip(nodes, expected)):
            if float(np.linalg.norm(p - e)) > 1e-6:
                errs.append(f"cut node {i} is not exact edge midpoint (inset?)")

    return errs


def compute_node_audit_unfold(
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
) -> NetUnfoldResult:
    """
    Build proportional face net for node audit — no lattice stamp, no skin
    routing, no surface mask required.
    """
    # Dummy empty node array — build_net_faces only needs it for primary lookup
    nodes = np.empty((0, 3), dtype=np.float64)
    faces = build_net_faces(
        nodes, hex_elems, cell_tags, round_decimals=round_decimals, skin_faces=None
    )
    if not faces:
        return NetUnfoldResult(
            faces=[],
            hinge_adj={},
            hinges=set(),
            cuts=set(),
            root_id=-1,
            unfolds={},
            node_uv={},
            node_face={},
            report={"n_faces": 0},
        )

    adj, _ = build_face_hinge_graph(faces)
    root_id = select_root_face(faces, adj)
    hinges, parent = bfs_spanning_tree(adj, root_id)
    cuts: set[tuple[int, int]] = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u >= v:
                continue
            if (u, v) not in hinges:
                cuts.add((u, v))

    unfolds = unfold_faces_integer_grid(faces, parent, root_id)

    # Synthetic local audit node UV keyed by (face_id, local_i)
    node_uv: dict[int, np.ndarray] = {}
    node_face: dict[int, int] = {}
    next_id = 0
    n_full = n_cut = n_side = n_apex = 0
    n_full_nodes = n_cut_nodes = n_side_nodes = 0
    for f in faces:
        if f.face_id not in unfolds:
            continue
        uf = unfolds[f.face_id]
        pts = local_audit_nodes_uv(uf)
        if f.role == ROLE_HALF_CUT:
            n_cut += 1
            n_cut_nodes += len(pts)
        elif f.role == ROLE_HALF_SIDE:
            n_side += 1
            n_side_nodes += len(pts)
        elif f.role == ROLE_HALF_APEX:
            n_apex += 1
            n_full_nodes += len(pts)
        else:
            n_full += 1
            n_full_nodes += len(pts)
        for p in pts:
            node_uv[next_id] = p
            node_face[next_id] = f.face_id
            next_id += 1

    grids = list(unfolds.values())
    if grids:
        u0 = min(uf.grid_ij[0] for uf in grids)
        v0 = min(uf.grid_ij[1] for uf in grids)
        u1 = max(uf.grid_ij[0] + uf.size_uv[0] for uf in grids)
        v1 = max(uf.grid_ij[1] + uf.size_uv[1] for uf in grids)
        span_u, span_v = u1 - u0, v1 - v0
    else:
        span_u = span_v = 0.0

    return NetUnfoldResult(
        faces=faces,
        hinge_adj=adj,
        hinges=hinges,
        cuts=cuts,
        root_id=root_id,
        unfolds=unfolds,
        node_uv=node_uv,
        node_face=node_face,
        report={
            "n_faces": len(faces),
            "n_hinges": len(hinges),
            "n_cuts": len(cuts),
            "n_unfolded": len(unfolds),
            "root_id": root_id,
            "n_node_uv": len(node_uv),
            "n_full": n_full,
            "n_half_cut": n_cut,
            "n_half_side": n_side,
            "n_half_apex": n_apex,
            "n_full_nodes": n_full_nodes,
            "n_cut_nodes": n_cut_nodes,
            "n_side_nodes": n_side_nodes,
            "grid_span_u": float(span_u),
            "grid_span_v": float(span_v),
            "audit": True,
        },
    )


def _face_plane_from_normal(normal: np.ndarray) -> str:
    """
    Classify a face by the Cartesian plane it lies in.

    Dominant |normal| component:
      |n_z| max → XY,  |n_y| max → XZ,  |n_x| max → YZ.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-14:
        return "XY"
    n = np.abs(n) / nrm
    return ("YZ", "XZ", "XY")[int(np.argmax(n))]


# Face fill/edge by plane (node/strut colors stay role-based)
_PLANE_FACE_STYLE: dict[str, tuple[str, str]] = {
    "XY": ("#fce4d6", "#e67e22"),  # orange — normal ±Z
    "XZ": ("#d5f5e3", "#27ae60"),  # green — normal ±Y
    "YZ": ("#d6eaf8", "#2980b9"),  # blue — normal ±X
}


def plot_surface_node_audit(
    net: NetUnfoldResult,
    out_path: str | Path,
    *,
    node_size: float = 28.0,
    dpi: int = 160,
    figsize: tuple[float, float] = (14.0, 12.0),
) -> Path:
    """
    Node-audit plot: faces colored by XY/XZ/YZ plane from normal; nodes and
    native CUT struts keep their existing role-based colors.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if not net.unfolds:
        ax.set_title("Surface node audit (empty)")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    face_by_id = {f.face_id: f for f in net.faces}
    plane_counts = {"XY": 0, "XZ": 0, "YZ": 0}

    for uf in net.unfolds.values():
        gu, gv = uf.grid_ij
        w, h = uf.size_uv
        face = face_by_id.get(uf.face_id)
        plane = _face_plane_from_normal(face.normal) if face is not None else "XY"
        plane_counts[plane] = plane_counts.get(plane, 0) + 1
        fc, ec = _PLANE_FACE_STYLE[plane]
        ax.add_patch(
            Rectangle(
                (gu, gv),
                w,
                h,
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.8,
                zorder=0,
            )
        )

    n_native = 0
    for uf in net.unfolds.values():
        for a, b in local_audit_native_segments_uv(uf):
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#1f77b4",
                lw=1.2,
                alpha=0.85,
                zorder=2,
            )
            n_native += 1

    for uf in net.unfolds.values():
        if uf.role != ROLE_HALF_SIDE:
            continue
        (a, b), _mid = side_cut_edge_and_midpoint_uv(uf)
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#6fa8dc",
            lw=1.4,
            ls="--",
            alpha=0.7,
            zorder=1,
        )

    buckets = {
        ROLE_FULL: [],
        ROLE_HALF_APEX: [],
        ROLE_HALF_CUT: [],
        ROLE_HALF_SIDE: [],
    }
    for uf in net.unfolds.values():
        key = uf.role if uf.role in buckets else ROLE_FULL
        buckets[key].extend(local_audit_nodes_uv(uf))

    def _scatter(pts: list, color: str, size: float, z: int) -> None:
        if not pts:
            return
        xy = np.asarray(pts, dtype=np.float64)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=float(size),
            c=color,
            edgecolors="#222222",
            linewidths=0.35,
            alpha=0.95,
            zorder=z,
        )

    _scatter(buckets[ROLE_HALF_CUT], "#e69100", float(node_size) * 0.85, 3)
    _scatter(buckets[ROLE_HALF_SIDE], "#e60000", float(node_size), 4)
    _scatter(buckets[ROLE_FULL] + buckets[ROLE_HALF_APEX], "#cc0000", float(node_size) * 1.15, 5)

    r = net.report
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Surface node audit — faces by plane (XY / XZ / YZ from normal)\n"
        f"faces={r.get('n_faces')} "
        f"(XY={plane_counts['XY']} XZ={plane_counts['XZ']} YZ={plane_counts['YZ']})  "
        f"nodes: full/apex={r.get('n_full_nodes')} "
        f"cut={r.get('n_cut_nodes')} side={r.get('n_side_nodes')}  "
        f"native_segs={n_native}"
    )
    ax.set_xlabel("U (proportional face grid)")
    ax.set_ylabel("V (proportional face grid)")
    legend = [
        Line2D([0], [0], color="#e67e22", lw=6, label="XY face (n∥Z)"),
        Line2D([0], [0], color="#27ae60", lw=6, label="XZ face (n∥Y)"),
        Line2D([0], [0], color="#2980b9", lw=6, label="YZ face (n∥X)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#cc0000",
            markeredgecolor="#222222",
            markersize=9,
            label="Full center",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e60000",
            markeredgecolor="#222222",
            markersize=8,
            label="SIDE cut-edge mid",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e69100",
            markeredgecolor="#222222",
            markersize=7,
            label="CUT diamond nodes",
        ),
        Line2D([0], [0], color="#1f77b4", lw=2, label="native CUT diamond"),
        Line2D([0], [0], color="#6fa8dc", lw=1.5, ls="--", label="SIDE cut edge"),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_topological_dual_with_portals(
    net: NetUnfoldResult,
    topo,  # TopologicalDualResult
    out_path: str | Path,
    *,
    local_gate: float = _LOCAL_GATE,
    stub_frac: float = 0.35,
    node_size: float = 22.0,
    dpi: int = 160,
    figsize: tuple[float, float] = (14.0, 12.0),
    label_fontsize: float = 6.0,
    face_node_uv: dict[int, list[np.ndarray]] | None = None,
    portal_mode: str = "uv_gate",
    title: str | None = None,
) -> Path:
    """
    Task 22 visualizer: topological dual on the UV net with seam portals.

    ``portal_mode``:
      - ``uv_gate`` (default): UV length < ``local_gate`` → full line; else stub.
      - ``hinge``: spanning-tree hinges → full line; non-tree seams → stub+portal.

    Optional ``face_node_uv`` overrides synthetic audit node placement (used by
    the extent-rule surface dual to plot surviving lattice nodes).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if not net.unfolds:
        ax.set_title("Topological dual (empty)")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    topo_by_id = {f.face_id: f for f in topo.faces}
    unfold_by_hf: dict[tuple[int, int], FaceUnfold] = {}
    for nf in net.faces:
        if nf.face_id in net.unfolds:
            unfold_by_hf[(nf.hex_i, nf.face_i)] = net.unfolds[nf.face_id]

    if face_node_uv is None:
        face_node_uv = {}
        for tf in topo.faces:
            uf = unfold_by_hf.get((tf.hex_i, tf.face_i))
            if uf is None:
                continue
            uf.role = tf.role
            face_node_uv[tf.face_id] = local_audit_nodes_uv(uf)

    plane_counts = {"XY": 0, "XZ": 0, "YZ": 0}
    for uf in net.unfolds.values():
        nf = next((f for f in net.faces if f.face_id == uf.face_id), None)
        plane = _face_plane_from_normal(nf.normal) if nf is not None else "XY"
        plane_counts[plane] = plane_counts.get(plane, 0) + 1
        fc, ec = _PLANE_FACE_STYLE[plane]
        ax.add_patch(
            Rectangle(
                (uf.grid_ij[0], uf.grid_ij[1]),
                uf.size_uv[0],
                uf.size_uv[1],
                facecolor=fc,
                edgecolor=ec,
                linewidth=0.7,
                zorder=0,
            )
        )

    n_native = 0
    for tf in topo.faces:
        if tf.role != ROLE_HALF_CUT:
            continue
        uvs = face_node_uv.get(tf.face_id)
        if not uvs or len(uvs) != 4:
            continue
        for i, j in tf.native_locals:
            if int(i) >= len(uvs) or int(j) >= len(uvs):
                continue
            a, b = uvs[int(i)], uvs[int(j)]
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#1f77b4",
                lw=1.1,
                alpha=0.85,
                zorder=2,
            )
            n_native += 1

    gate = float(local_gate)
    use_hinge = str(portal_mode).lower() == "hinge"
    n_local = n_portal = 0
    for s in topo.routed_struts:
        uva = face_node_uv.get(s.face_a)
        uvb = face_node_uv.get(s.face_b)
        if not uva or not uvb:
            continue
        if s.node_a >= len(uva) or s.node_b >= len(uvb):
            continue
        pa = np.asarray(uva[s.node_a], dtype=np.float64)
        pb = np.asarray(uvb[s.node_b], dtype=np.float64)
        pair = (
            (s.face_a, s.face_b)
            if s.face_a < s.face_b
            else (s.face_b, s.face_a)
        )
        if use_hinge:
            is_local = pair in net.hinges
        else:
            is_local = float(np.linalg.norm(pa - pb)) < gate
        if is_local:
            ax.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                color="#d62728",
                lw=1.0,
                alpha=0.7,
                zorder=2,
            )
            n_local += 1
            continue

        n_portal += 1
        for fid, p, other_fid, other_p in (
            (s.face_a, pa, s.face_b, pb),
            (s.face_b, pb, s.face_a, pa),
        ):
            tf = topo_by_id[fid]
            uf = unfold_by_hf.get((tf.hex_i, tf.face_i))
            if uf is None:
                continue
            q = _stub_to_cell_edge(
                p, other_p, uf.grid_ij, uf.size_uv, frac=float(stub_frac)
            )
            ax.plot(
                [p[0], q[0]],
                [p[1], q[1]],
                color="#d62728",
                lw=1.1,
                alpha=0.55,
                solid_capstyle="round",
                zorder=3,
            )
            mid = q.copy()
            off = q - p
            on = float(np.linalg.norm(off))
            if on > 1e-12:
                mid = q + 0.12 * (off / on)
            ax.text(
                float(mid[0]),
                float(mid[1]),
                f"to {other_fid}",
                fontsize=float(label_fontsize),
                color="#7a1f1f",
                ha="center",
                va="center",
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="#d62728",
                    alpha=0.85,
                    linewidth=0.4,
                ),
            )

    buckets = {
        ROLE_FULL: [],
        ROLE_HALF_APEX: [],
        ROLE_HALF_CUT: [],
        ROLE_HALF_SIDE: [],
    }
    for tf in topo.faces:
        uvs = face_node_uv.get(tf.face_id, [])
        key = tf.role if tf.role in buckets else ROLE_FULL
        buckets[key].extend(uvs)

    def _scatter(pts: list, color: str, size: float, z: int) -> None:
        if not pts:
            return
        xy = np.asarray(pts, dtype=np.float64)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=float(size),
            c=color,
            edgecolors="#222222",
            linewidths=0.3,
            alpha=0.95,
            zorder=z,
        )

    _scatter(buckets[ROLE_HALF_CUT], "#e69100", float(node_size) * 0.85, 4)
    _scatter(buckets[ROLE_HALF_SIDE], "#e60000", float(node_size), 5)
    _scatter(
        buckets[ROLE_FULL] + buckets[ROLE_HALF_APEX],
        "#cc0000",
        float(node_size) * 1.1,
        6,
    )

    tr = topo.report
    ax.set_aspect("equal", adjustable="box")
    if title is None:
        title = (
            "Task 22 — Topological dual (3D edge_map) + UV seam portals\n"
            f"faces={tr.get('n_faces')} "
            f"(XY={plane_counts['XY']} XZ={plane_counts['XZ']} YZ={plane_counts['YZ']})  "
            f"shared_edges={tr.get('n_shared_edges')}  "
            f"routed: local={n_local} portals={n_portal}  native={n_native}"
        )
    else:
        title = (
            f"{title}\n"
            f"faces={tr.get('n_faces')} "
            f"(XY={plane_counts['XY']} XZ={plane_counts['XZ']} YZ={plane_counts['YZ']})  "
            f"shared_edges={tr.get('n_shared_edges')}  "
            f"routed: local={n_local} portals={n_portal}  native={n_native}"
        )
    ax.set_title(title)
    ax.set_xlabel("U (proportional face grid)")
    ax.set_ylabel("V (proportional face grid)")
    legend = [
        Line2D([0], [0], color="#e67e22", lw=6, label="XY face (n∥Z)"),
        Line2D([0], [0], color="#27ae60", lw=6, label="XZ face (n∥Y)"),
        Line2D([0], [0], color="#2980b9", lw=6, label="YZ face (n∥X)"),
        Line2D([0], [0], color="#1f77b4", lw=2, label="native CUT diamond"),
        Line2D([0], [0], color="#d62728", lw=2, label="topo routed (hinge/local)"),
        Line2D(
            [0],
            [0],
            color="#d62728",
            lw=1.5,
            alpha=0.55,
            label="seam stub + portal label",
        ),
    ]
    ax.legend(handles=legend, loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
