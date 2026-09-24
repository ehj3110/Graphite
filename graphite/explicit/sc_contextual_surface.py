"""
Context-aware surface skin router (Task 18).

Exposed Cartesian faces of retained SC cells get surface nodes that depend on
the parent cell state:

  Full / Half apex  -> single face-center node
  Half CUT          -> 4 mid-plane diamond nodes (edge midpoints of cut axis)
  Half SIDE         -> single mid-plane cut-edge midpoint (one diamond node)

Skin struts:
  - Half CUT: diamond perimeter among its 4 nodes (native)
  - Adjacent exposed faces: dual links between their contextual nodes
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graphite.explicit.hex_rules import (
    _HALF_NEG_X_FACE_INDICES,
    _HALF_NEG_Y_FACE_INDICES,
    _HALF_NEG_Z_FACE_INDICES,
    _HALF_POS_X_FACE_INDICES,
    _HALF_POS_Y_FACE_INDICES,
    _HALF_POS_Z_FACE_INDICES,
    _HEX_EDGES,
    _HEX_FACES,
    _hex_face_centers,
)
from graphite.explicit.hex_topology_module import resolve_quantized_cell_rule

ROLE_FULL = "full"
ROLE_HALF_APEX = "half_apex"
ROLE_HALF_CUT = "half_cut"
ROLE_HALF_SIDE = "half_side"

_HALF_FACE_TABLE: dict[str, tuple[int, ...]] = {
    "octahedral_half_neg_z": _HALF_NEG_Z_FACE_INDICES,
    "octahedral_half_pos_z": _HALF_POS_Z_FACE_INDICES,
    "octahedral_half_neg_x": _HALF_NEG_X_FACE_INDICES,
    "octahedral_half_pos_x": _HALF_POS_X_FACE_INDICES,
    "octahedral_half_neg_y": _HALF_NEG_Y_FACE_INDICES,
    "octahedral_half_pos_y": _HALF_POS_Y_FACE_INDICES,
    "octahedral_half_z": _HALF_NEG_Z_FACE_INDICES,
    "octahedral_half_x": _HALF_NEG_X_FACE_INDICES,
    "octahedral_half_y": _HALF_NEG_Y_FACE_INDICES,
}


def _corner_key(pt: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


def _face_key(
    corners: np.ndarray,
    face_local: tuple[int, int, int, int],
    decimals: int,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(sorted(_corner_key(corners[i], decimals) for i in face_local))


def _edge_key(
    p0: np.ndarray,
    p1: np.ndarray,
    decimals: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a = _corner_key(p0, decimals)
    b = _corner_key(p1, decimals)
    return (a, b) if a <= b else (b, a)


def half_apex_face_index(rule: str) -> int:
    return int(_HALF_FACE_TABLE[rule][0])


def half_cut_face_index(rule: str) -> int:
    """Empty / cut Cartesian face omitted from the half leaf stamp."""
    kept = set(_HALF_FACE_TABLE[rule])
    missing = [i for i in range(6) if i not in kept]
    if len(missing) != 1:
        raise ValueError(f"expected one omitted cut face for {rule}, got {missing}")
    return int(missing[0])


def half_side_face_indices(rule: str) -> tuple[int, ...]:
    """Equatorial mid-plane diamond faces (locals 1-4)."""
    return tuple(int(i) for i in _HALF_FACE_TABLE[rule][1:5])


def classify_exposed_face_role(rule: str | None, face_i: int) -> str:
    """Map (parent rule, face index) -> contextual surface role."""
    if rule is None:
        raise ValueError("empty cell has no exposed surface role")
    if rule == "octahedral":
        return ROLE_FULL
    if rule not in _HALF_FACE_TABLE:
        return ROLE_FULL
    fi = int(face_i)
    if fi == half_apex_face_index(rule):
        return ROLE_HALF_APEX
    if fi == half_cut_face_index(rule):
        return ROLE_HALF_CUT
    if fi in half_side_face_indices(rule):
        return ROLE_HALF_SIDE
    return ROLE_HALF_SIDE


def _edges_parallel_to_axis(axis: int) -> tuple[tuple[int, int], ...]:
    """Four hex corner-edges parallel to world axis 0=X, 1=Y, 2=Z."""
    out: list[tuple[int, int]] = []
    for a, b in _HEX_EDGES:
        ca, cb = int(a), int(b)
        diff = ca ^ cb
        if axis == 0 and diff == 1:
            out.append((ca, cb))
        elif axis == 1 and diff == 2:
            out.append((ca, cb))
        elif axis == 2 and diff == 4:
            out.append((ca, cb))
    return tuple(out)


def _cut_axis_from_face(face_i: int) -> int:
    """Axis normal to the cut face (0=X, 1=Y, 2=Z)."""
    if face_i in (0, 1):
        return 2
    if face_i in (2, 3):
        return 1
    return 0


def midplane_diamond_nodes(corners: np.ndarray, rule: str) -> np.ndarray:
    """
    Four mid-plane diamond nodes for a Half cell.

    Equatorial face centers of the half leaf — identical to the midpoints of
    the four mid-plane cut edges on the SIDE faces (and to the stamped
    diamond locals 1–4). On a cube these are the midpoints of the mid-plane
    square edges (not the Cartesian cut-face corners).
    """
    c = np.asarray(corners, dtype=np.float64)
    fcs = _hex_face_centers(c)
    return np.asarray([fcs[i] for i in half_side_face_indices(rule)], dtype=np.float64)


def side_cut_edge_midpoint(corners: np.ndarray, face_i: int, rule: str) -> np.ndarray:
    """
    Midpoint of the mid-plane cut edge on a Half SIDE face.

    Coincides with one mid-plane diamond node (face center of ``face_i``).
    """
    del rule
    return np.asarray(_hex_face_centers(corners)[int(face_i)], dtype=np.float64)


@dataclass
class ContextualFaceNodes:
    """True surface nodes for one exposed Cartesian face."""

    role: str
    points: np.ndarray  # (N, 3)
    labels: tuple[str, ...] = ()


def contextual_surface_nodes_for_face(
    corners: np.ndarray,
    face_i: int,
    rule: str | None,
) -> ContextualFaceNodes:
    """
    Contextual node placement + Task-20 strict prune for an exposed face.

    Full / Half apex:
        ONLY the Cartesian face-center node; no native struts.
    Half CUT:
        ONLY the 4 mid-plane diamond nodes; native diamond perimeter.
    Half SIDE:
        ONLY the mid-plane cut-edge midpoint (never a floating face-center
        distinct from the cut edge); no native struts.
    """
    c = np.asarray(corners, dtype=np.float64)
    if c.shape != (8, 3):
        raise ValueError(f"corners must be (8,3); got {c.shape}")
    role = classify_exposed_face_role(rule, face_i)

    if role in (ROLE_FULL, ROLE_HALF_APEX):
        ctr = _hex_face_centers(c)[int(face_i)].reshape(1, 3)
        return ContextualFaceNodes(role=role, points=ctr, labels=("center",))

    if role == ROLE_HALF_CUT:
        assert rule is not None
        pts = midplane_diamond_nodes(c, rule)
        return ContextualFaceNodes(
            role=role,
            points=pts,
            labels=("d0", "d1", "d2", "d3"),
        )

    if role == ROLE_HALF_SIDE:
        assert rule is not None
        # Strict: cut-edge midpoint only (coincides with mid-plane diamond node).
        # Do NOT emit a separate Cartesian face-center if it differs.
        mid = side_cut_edge_midpoint(c, face_i, rule).reshape(1, 3)
        return ContextualFaceNodes(role=role, points=mid, labels=("cut_mid",))

    raise ValueError(f"unknown face role {role!r}")


def native_surface_struts_for_face(role: str) -> tuple[tuple[int, int], ...]:
    """
    Task-20 strut filter by face role.

    Full / SIDE / apex → no native surface struts.
    CUT → 4 diamond perimeter edges on local nodes (0..3).
    """
    if role == ROLE_HALF_CUT:
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    return ()


def prune_face_candidate_nodes(
    role: str,
    candidates: np.ndarray,
    *,
    face_center: np.ndarray | None = None,
    cut_edge_mid: np.ndarray | None = None,
    diamond_targets: np.ndarray | None = None,
    tol: float = 1e-6,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """
    Strict geometric filter on candidate nodes associated with one face.

    Returns ``(kept_points, native_strut_locals)``.
    """
    pts = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)

    if role in (ROLE_FULL, ROLE_HALF_APEX):
        if face_center is None:
            raise ValueError("face_center required to prune Full/apex")
        fc = np.asarray(face_center, dtype=np.float64).reshape(3)
        if len(pts) == 0:
            return fc.reshape(1, 3), ()
        # Retain ONLY the node at the geometric center
        d = np.linalg.norm(pts - fc, axis=1)
        keep = pts[int(np.argmin(d))].reshape(1, 3)
        # If nothing was near center, emit center itself
        if float(np.min(d)) > tol * 10 and float(np.min(d)) > 1e-4:
            keep = fc.reshape(1, 3)
        else:
            keep = fc.reshape(1, 3)  # snap to exact center
        return keep, ()

    if role == ROLE_HALF_SIDE:
        if cut_edge_mid is None:
            raise ValueError("cut_edge_mid required to prune SIDE")
        mid = np.asarray(cut_edge_mid, dtype=np.float64).reshape(3)
        # Delete Cartesian face-center bleed: keep ONLY cut-edge mid
        return mid.reshape(1, 3), ()

    if role == ROLE_HALF_CUT:
        if diamond_targets is None:
            raise ValueError("diamond_targets required to prune CUT")
        targets = np.asarray(diamond_targets, dtype=np.float64).reshape(4, 3)
        # Retain exactly the 4 diamond targets (ignore any extras)
        return targets.copy(), native_surface_struts_for_face(ROLE_HALF_CUT)

    return pts, ()


def face_plot_size(role: str) -> tuple[float, float]:
    """
    Proportional UV cell size for diagnostic plots.

    Full / Half CUT / Half apex -> 1x1
    Half SIDE -> 1x0.5 (depth half; width along hinge stays 1)
    """
    if role == ROLE_HALF_SIDE:
        return (1.0, 0.5)
    return (1.0, 1.0)


@dataclass
class ExposedSkinFace:
    face_id: int
    hex_i: int
    face_i: int
    rule: str
    role: str
    corners_3d: np.ndarray
    center_3d: np.ndarray
    normal: np.ndarray
    corner_keys: tuple[tuple[float, float, float], ...]
    edge_keys: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ]
    local_points: np.ndarray
    gids: list[int] = field(default_factory=list)


@dataclass
class ContextualSkinResult:
    nodes: np.ndarray
    core_struts: np.ndarray
    surface_mask: np.ndarray
    unlocked_mask: np.ndarray  # Task 23: pruned Surface_Nodes only (TNP unlock)
    native_surface_struts: np.ndarray
    routed_surface_struts: np.ndarray
    faces: list[ExposedSkinFace]
    unlock_normals: np.ndarray
    report: dict = field(default_factory=dict)


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


def _order_diamond_cycle(points: np.ndarray) -> np.ndarray:
    """Order 4 coplanar points into a cyclic perimeter."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) != 4:
        return np.arange(len(pts), dtype=np.int64)
    c = pts.mean(axis=0)
    e1 = pts[1] - pts[0]
    if float(np.linalg.norm(e1)) < 1e-14:
        e1 = pts[2] - pts[0]
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-14)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    if float(np.linalg.norm(n)) < 1e-14:
        n = np.cross(e1, pts[3] - pts[0])
    n = n / max(float(np.linalg.norm(n)), 1e-14)
    e2 = np.cross(n, e1)
    ang = []
    for i, p in enumerate(pts):
        d = p - c
        ang.append((float(np.arctan2(np.dot(d, e2), np.dot(d, e1))), i))
    ang.sort()
    return np.asarray([i for _, i in ang], dtype=np.int64)


def build_contextual_surface_skin(
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    core_nodes: np.ndarray | None = None,
    core_struts: np.ndarray | None = None,
    round_decimals: int = 6,
) -> ContextualSkinResult:
    """
    Map exposed faces -> contextual nodes -> adjacent-face skin dual.

    Merges contextual surface nodes into ``core_nodes`` (lattice stamp) by
    rounded-coordinate weld so Half diamond / Full FC IDs stay consistent.
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    if len(tags) != len(elems):
        raise ValueError("cell_tags length must match hex_elems")

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    if core_nodes is not None and len(core_nodes):
        for p in np.asarray(core_nodes, dtype=np.float64):
            key = _corner_key(p, round_decimals)
            if key not in node_map:
                node_map[key] = len(nodes_list)
                nodes_list.append(np.asarray(p, dtype=np.float64).copy())

    def _gid(pt: np.ndarray) -> int:
        key = _corner_key(pt, round_decimals)
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes_list)
            node_map[key] = idx
            nodes_list.append(np.asarray(pt, dtype=np.float64).copy())
        return idx

    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    rules: list[str | None] = []
    for hi, (corners, tag) in enumerate(zip(elems, tags)):
        rule = resolve_quantized_cell_rule(tag)
        rules.append(rule)
        if rule is None:
            continue
        for fi, face in enumerate(_HEX_FACES):
            face_owners.setdefault(_face_key(corners, face, round_decimals), []).append(
                (hi, fi)
            )

    faces: list[ExposedSkinFace] = []
    surface_gids: set[int] = set()
    unlocked_gids: set[int] = set()  # Task 23 TNP unlock set
    normal_acc: dict[int, np.ndarray] = {}
    normal_w: dict[int, float] = {}

    def _accum_normal(gid: int, n: np.ndarray) -> None:
        n = np.asarray(n, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-14:
            return
        n = n / nrm
        if gid not in normal_acc:
            normal_acc[gid] = np.zeros(3, dtype=np.float64)
            normal_w[gid] = 0.0
        normal_acc[gid] += n
        normal_w[gid] += 1.0

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
            # Task-20: re-assert prune on candidates (defense in depth)
            fc = _hex_face_centers(c)[fi]
            kept, _native_locals = prune_face_candidate_nodes(
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
            ctx = ContextualFaceNodes(
                role=role,
                points=kept,
                labels=ctx.labels[: len(kept)],
            )
            corners_3d = np.array([c[i] for i in face], dtype=np.float64)
            corner_keys = tuple(
                _corner_key(corners_3d[i], round_decimals) for i in range(4)
            )
            edge_keys = tuple(
                _edge_key(corners_3d[a], corners_3d[b], round_decimals)
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            gids = [_gid(p) for p in ctx.points]
            for gid in gids:
                surface_gids.add(gid)
                # Task 23: UNLOCKED = pruned Surface_Nodes only
                # Full face centers, Half SIDE cut-mids, Half CUT diamonds.
                # Internal lattice nodes + Half apex stay LOCKED.
                if role in (ROLE_FULL, ROLE_HALF_SIDE, ROLE_HALF_CUT):
                    unlocked_gids.add(gid)
                    if role == ROLE_HALF_CUT:
                        _accum_normal(gid, _outward_face_normal(c, half_cut_face_index(rule)))
                    else:
                        _accum_normal(gid, _outward_face_normal(c, fi))
                elif role == ROLE_HALF_APEX:
                    # Apex may participate in dual topology but is LOCKED for TNP.
                    pass
                else:
                    _accum_normal(gid, _outward_face_normal(c, fi))

            faces.append(
                ExposedSkinFace(
                    face_id=len(faces),
                    hex_i=hi,
                    face_i=fi,
                    rule=rule,
                    role=role,
                    corners_3d=corners_3d,
                    center_3d=corners_3d.mean(axis=0),
                    normal=_outward_face_normal(c, fi),
                    corner_keys=corner_keys,
                    edge_keys=edge_keys,
                    local_points=np.asarray(ctx.points, dtype=np.float64),
                    gids=gids,
                )
            )

    nodes = (
        np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    )
    surface_mask = np.zeros(len(nodes), dtype=bool)
    if surface_gids:
        surface_mask[list(surface_gids)] = True
    unlocked_mask = np.zeros(len(nodes), dtype=bool)
    if unlocked_gids:
        unlocked_mask[list(unlocked_gids)] = True
    # Explicit lock: everything not in unlocked_gids stays False (LOCKED).
    assert unlocked_mask.shape == surface_mask.shape
    assert not np.any(unlocked_mask & ~surface_mask)

    unlock_normals = np.zeros_like(nodes)
    for gid, acc in normal_acc.items():
        w = normal_w.get(gid, 0.0)
        if w > 0:
            n = acc / w
            nrm = float(np.linalg.norm(n))
            if nrm > 1e-14:
                unlock_normals[gid] = n / nrm

    native: set[tuple[int, int]] = set()
    routed: set[tuple[int, int]] = set()

    def _add(edge_set: set[tuple[int, int]], a: int, b: int) -> None:
        if a == b:
            return
        e = (a, b) if a < b else (b, a)
        edge_set.add(e)

    for ef in faces:
        if ef.role != ROLE_HALF_CUT or len(ef.gids) != 4:
            continue
        order = _order_diamond_cycle(nodes[np.asarray(ef.gids, dtype=np.int64)])
        g = [ef.gids[int(i)] for i in order]
        for i in range(4):
            _add(native, g[i], g[(i + 1) % 4])

    edge_to_faces: dict[tuple, list[int]] = {}
    for ef in faces:
        for e in ef.edge_keys:
            edge_to_faces.setdefault(e, []).append(ef.face_id)
    by_id = {ef.face_id: ef for ef in faces}

    for _ek, fids in edge_to_faces.items():
        uniq = list(dict.fromkeys(fids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                fa, fb = by_id[uniq[i]], by_id[uniq[j]]
                _route_adjacent_face_dual(fa, fb, nodes, routed, native)

    core = (
        np.asarray(core_struts, dtype=np.int64).reshape(-1, 2)
        if core_struts is not None and len(core_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    all_skin = native | routed
    merged_edges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in core
        if int(a) != int(b)
    } | all_skin
    core_out = (
        np.array(sorted(merged_edges), dtype=np.int64)
        if merged_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    native_arr = (
        np.array(sorted(native), dtype=np.int64)
        if native
        else np.empty((0, 2), dtype=np.int64)
    )
    routed_only = routed - native
    routed_arr = (
        np.array(sorted(routed_only), dtype=np.int64)
        if routed_only
        else np.empty((0, 2), dtype=np.int64)
    )

    report = {
        "n_exposed_faces": len(faces),
        "n_full": sum(1 for f in faces if f.role == ROLE_FULL),
        "n_half_cut": sum(1 for f in faces if f.role == ROLE_HALF_CUT),
        "n_half_side": sum(1 for f in faces if f.role == ROLE_HALF_SIDE),
        "n_half_apex": sum(1 for f in faces if f.role == ROLE_HALF_APEX),
        "n_surface_nodes": int(np.count_nonzero(surface_mask)),
        "n_unlocked_nodes": int(np.count_nonzero(unlocked_mask)),
        "n_locked_nodes": int(np.count_nonzero(~unlocked_mask)),
        "n_native_struts": len(native_arr),
        "n_routed_struts": len(routed_arr),
    }
    return ContextualSkinResult(
        nodes=nodes,
        core_struts=core_out,
        surface_mask=surface_mask,
        unlocked_mask=unlocked_mask,
        native_surface_struts=native_arr,
        routed_surface_struts=routed_arr,
        faces=faces,
        unlock_normals=unlock_normals,
        report=report,
    )


def _point_edge_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L2 = float(np.dot(ab, ab))
    if L2 < 1e-28:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _shared_edge_endpoints(
    fa: ExposedSkinFace,
    fb: ExposedSkinFace,
) -> tuple[np.ndarray, np.ndarray] | None:
    shared = set(fa.edge_keys) & set(fb.edge_keys)
    if not shared:
        return None
    ek = next(iter(shared))
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        ka, kb = fa.corner_keys[i], fa.corner_keys[j]
        key = (ka, kb) if ka <= kb else (kb, ka)
        if key == ek:
            return fa.corners_3d[i].copy(), fa.corners_3d[j].copy()
    return None


def _route_adjacent_face_dual(
    fa: ExposedSkinFace,
    fb: ExposedSkinFace,
    nodes: np.ndarray,
    routed: set[tuple[int, int]],
    native: set[tuple[int, int]],
) -> None:
    """Connect contextual nodes of two edge-adjacent exposed faces."""

    def _add(a: int, b: int) -> None:
        if a == b:
            return
        e = (a, b) if a < b else (b, a)
        if e in native:
            return
        routed.add(e)

    ga = list(fa.gids)
    gb = list(fb.gids)
    if not ga or not gb:
        return

    if len(ga) == 1 and len(gb) == 1:
        _add(ga[0], gb[0])
        return

    ends = _shared_edge_endpoints(fa, fb)
    if ends is not None:
        a3, b3 = ends
        edge_len = float(np.linalg.norm(b3 - a3))
        tol = max(0.35 * edge_len, 1e-6)

        def near(gids: list[int]) -> list[int]:
            return [
                g
                for g in gids
                if _point_edge_distance(nodes[g], a3, b3) <= tol
            ]

        na, nb = near(ga), near(gb)
        if na and nb:
            for u in na:
                for v in nb:
                    _add(u, v)
            return

    used_b: set[int] = set()
    for u in ga:
        best = None
        best_d = float("inf")
        for v in gb:
            if v == u or v in used_b:
                continue
            d = float(np.linalg.norm(nodes[u] - nodes[v]))
            if d < best_d:
                best_d = d
                best = v
        if best is not None and best_d > 1e-9:
            _add(u, best)
            used_b.add(best)
