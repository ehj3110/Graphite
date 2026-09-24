"""Layered surface dual roles (hex C / E / F on occupancy-skin supports).

Occupancy helpers live in ``sc_node_plane_trim``. Gold-standard octahedral
compare: both-ends-on-support promote + F-F shared-edge stitch; F-E / F-C
remain for blends. See docs/LAYERED_SURFACE_DUAL_ROLES_ALIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

from graphite.explicit.conformal_core import merge_surface_skin
from graphite.explicit.hex_rules import (
    _HEX_FACES,
    _hex_face_centers,
)
from graphite.explicit.hex_topology_module import get_hex_topology_rule
from graphite.explicit.sc_node_plane_trim import (
    _hex_filled_octants,
    _hex_is_active,
    _plane_kept,
    compute_kept_planes_per_hex,
    hex_occupies_cad,
    hex_span,
)
from graphite.explicit.sc_quantized_surface import _face_key
from graphite.explicit.sc_simple_fc_surface_dual import (
    axis_aligned_face_quad,
    point_on_quad,
    route_geometric_face_stitch,
)
from graphite.explicit.sc_topological_dual import _edge_midpoint_key

RoleName = Literal["C", "E", "F", "B", "K"]

# Face index → (axis, is_max): 0=-Z, 1=+Z, 2=-Y, 3=+Y, 4=-X, 5=+X
_FACE_AXIS_MAX: tuple[tuple[int, bool], ...] = (
    (2, False),
    (2, True),
    (1, False),
    (1, True),
    (0, False),
    (0, True),
)
_AXIS_IS_MAX_TO_FACE: dict[tuple[int, bool], int] = {
    val: fi for fi, val in enumerate(_FACE_AXIS_MAX)
}

_F_RULES = frozenset({"octahedral", "hex_face_dual", "octet", "cross", "star"})
_C_RULES = frozenset({"grid", "tesseract"})
_K_RULES = frozenset({"kelvin"})

_ROLE_ATOL = 0.08
_EDGE_DECIMALS = 5


@dataclass
class RoleSupport:
    hex_i: int
    face_i: int
    is_midplane: bool
    quad: np.ndarray
    edges: tuple[tuple[np.ndarray, np.ndarray], ...]
    gids_by_role: dict[str, list[int]] = field(default_factory=dict)
    covering_neighbors: list[int] = field(default_factory=list)
    seam_edges: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None


@dataclass
class RoleSurfaceDualResult:
    dual_nodes: np.ndarray
    dual_struts: np.ndarray
    surface_gids: set[int]
    supports: list[RoleSupport]
    report: dict = field(default_factory=dict)


def _quad_edges(corners_3d: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    c = np.asarray(corners_3d, dtype=np.float64)
    return tuple((c[a].copy(), c[b].copy()) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)))


def filled_world_octants(
    hex_elems: np.ndarray,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
) -> tuple[set[tuple[int, int, int]], np.ndarray]:
    """Integer octant indices (half-cell voxels) occupied by kept-plane trims."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    filled: set[tuple[int, int, int]] = set()
    half_ref = np.ones(3, dtype=np.float64)
    if elems.ndim != 3 or kept_planes_per_hex is None:
        return filled, half_ref
    for hi, corners in enumerate(elems):
        if hi >= len(kept_planes_per_hex):
            continue
        local, half = _hex_filled_octants(corners, kept_planes_per_hex[hi])
        if hi == 0:
            half_ref = half
        filled.update(local)
    return filled, half_ref


def point_on_occupancy_skin(
    pt: np.ndarray,
    filled: set[tuple[int, int, int]],
    half_size: np.ndarray,
    *,
    eps: float | None = None,
) -> bool:
    """True if ``pt`` lies on the free skin of occupancy (one-sided along some axis)."""
    if not filled:
        return False
    p = np.asarray(pt, dtype=np.float64).reshape(3)
    h = np.asarray(half_size, dtype=np.float64).reshape(3)
    if eps is None:
        eps = 1e-4 * float(np.min(h))

    def occupied(q: np.ndarray) -> bool:
        idx = tuple(int(np.floor(float(q[i]) / float(h[i]) + 1e-12)) for i in range(3))
        return idx in filled

    for ax in range(3):
        dp = np.zeros(3, dtype=np.float64)
        dp[ax] = float(eps)
        if occupied(p + dp) != occupied(p - dp):
            return True
    return False


def point_touches_occupancy(
    pt: np.ndarray,
    filled: set[tuple[int, int, int]],
    half_size: np.ndarray,
    *,
    eps: float | None = None,
) -> bool:
    """True if ``pt`` is inside occupancy or on its boundary (max-face safe)."""
    if not filled:
        return False
    p = np.asarray(pt, dtype=np.float64).reshape(3)
    h = np.asarray(half_size, dtype=np.float64).reshape(3)
    if eps is None:
        eps = 1e-4 * float(np.min(h))

    def occupied(q: np.ndarray) -> bool:
        idx = tuple(int(np.floor(float(q[i]) / float(h[i]) + 1e-12)) for i in range(3))
        return idx in filled

    for dx in (-eps, 0.0, eps):
        for dy in (-eps, 0.0, eps):
            for dz in (-eps, 0.0, eps):
                if occupied(p + np.array([dx, dy, dz], dtype=np.float64)):
                    return True
    return False


def hex_uvw(corners: np.ndarray, pt: np.ndarray) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float64)
    mn = c.min(axis=0)
    span = np.maximum(hex_span(c), 1e-15)
    return (np.asarray(pt, dtype=np.float64).reshape(3) - mn) / span


def classify_uvw(uvw: np.ndarray, *, atol: float = _ROLE_ATOL) -> RoleName:
    u = np.asarray(uvw, dtype=np.float64).reshape(3)

    def near(a: float, b: float) -> bool:
        return abs(float(a) - float(b)) <= atol

    def at01(a: float) -> bool:
        return near(a, 0.0) or near(a, 1.0)

    def at05(a: float) -> bool:
        return near(a, 0.5)

    n01 = sum(1 for i in range(3) if at01(u[i]))
    n05 = sum(1 for i in range(3) if at05(u[i]))
    if n01 == 3:
        return "C"
    if n01 == 2 and n05 == 1:
        return "E"
    if n05 == 3:
        return "B"
    if n01 == 1 and n05 == 2:
        return "F"
    return "K"


def dual_policy_for_rule(rule_name: str) -> str:
    rule = str(rule_name).strip().lower()
    if rule in _K_RULES:
        return "K"
    if rule in _C_RULES:
        return "C"
    if rule in _F_RULES:
        return "F"
    return "F"


def _interval_kept(kept: np.ndarray, lo: float, hi: float) -> bool:
    if len(kept) == 0:
        return False
    return (float(np.min(kept)) <= lo + 0.01) and (float(np.max(kept)) >= hi - 0.01)


def _compute_exposed_subquads(
    corners: np.ndarray,
    face_i: int,
    kept_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    covering_neighbors: list[int],
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    """Compute the exposed axis-aligned sub-rectangles for a partially covered face via 2x2 quadrant decomposition."""
    c = np.asarray(corners, dtype=np.float64)
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    span = np.maximum(mx - mn, 1e-15)
    axis, is_max = _FACE_AXIS_MAX[int(face_i)]
    val = float(mx[axis] if is_max else mn[axis])
    a0, a1 = (i for i in range(3) if i != axis)

    exp: set[tuple[int, int]] = set()
    for q0 in (0, 1):
        lo0, hi0 = (0.0, 0.5) if q0 == 0 else (0.5, 1.0)
        if not _interval_kept(kept_xyz[a0], lo0, hi0):
            continue
        for q1 in (0, 1):
            lo1, hi1 = (0.0, 0.5) if q1 == 0 else (0.5, 1.0)
            if not _interval_kept(kept_xyz[a1], lo1, hi1):
                continue
            is_cov = False
            for n_hi in covering_neighbors:
                n_kp = kept_planes_per_hex[n_hi]
                if _interval_kept(n_kp[a0], lo0, hi0) and _interval_kept(n_kp[a1], lo1, hi1):
                    is_cov = True
                    break
            if not is_cov:
                exp.add((q0, q1))

    if not exp:
        return []

    # Merge into maximal rectangles
    rects: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if exp == {(0, 0), (1, 0), (0, 1), (1, 1)}:
        rects.append(((0.0, 1.0), (0.0, 1.0)))
    elif exp == {(0, 0), (1, 0)}:
        rects.append(((0.0, 1.0), (0.0, 0.5)))
    elif exp == {(0, 1), (1, 1)}:
        rects.append(((0.0, 1.0), (0.5, 1.0)))
    elif exp == {(0, 0), (0, 1)}:
        rects.append(((0.0, 0.5), (0.0, 1.0)))
    elif exp == {(1, 0), (1, 1)}:
        rects.append(((0.5, 1.0), (0.0, 1.0)))
    elif len(exp) == 3:
        # L-shape: 1 missing quadrant
        mq = next(iter({(0, 0), (1, 0), (0, 1), (1, 1)} - exp))
        mq0, mq1 = mq
        other_q1 = 1 - mq1
        lo1, hi1 = (0.0, 0.5) if other_q1 == 0 else (0.5, 1.0)
        rects.append(((0.0, 1.0), (lo1, hi1)))
        other_q0 = 1 - mq0
        lo0, hi0 = (0.0, 0.5) if other_q0 == 0 else (0.5, 1.0)
        m_lo1, m_hi1 = (0.0, 0.5) if mq1 == 0 else (0.5, 1.0)
        rects.append(((lo0, hi0), (m_lo1, m_hi1)))
    else:
        for q0, q1 in exp:
            lo0, hi0 = (0.0, 0.5) if q0 == 0 else (0.5, 1.0)
            lo1, hi1 = (0.0, 0.5) if q1 == 0 else (0.5, 1.0)
            rects.append(((lo0, hi0), (lo1, hi1)))

    quads: list[np.ndarray] = []
    for (r0_lo, r0_hi), (r1_lo, r1_hi) in rects:
        lo0 = mn[a0] + r0_lo * span[a0]
        hi0 = mn[a0] + r0_hi * span[a0]
        lo1 = mn[a1] + r1_lo * span[a1]
        hi1 = mn[a1] + r1_hi * span[a1]
        uv = ((lo0, lo1), (hi0, lo1), (hi0, hi1), (lo0, hi1))
        out = np.zeros((4, 3), dtype=np.float64)
        for i, (u, v) in enumerate(uv):
            out[i, axis] = val
            out[i, a0] = u
            out[i, a1] = v
        quads.append(out)
    return quads


def collect_role_supports(
    hex_elems: np.ndarray,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    round_decimals: int = _EDGE_DECIMALS,
) -> list[RoleSupport]:
    """Exposed outer faces, or mid-plane cuts when the outer plane is not kept."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must have shape (N, 8, 3); got {elems.shape}")
    if len(kept_planes_per_hex) != len(elems):
        raise ValueError("kept_planes_per_hex length must match n_hex")

    active = [
        i
        for i, kp in enumerate(kept_planes_per_hex)
        if _hex_is_active(kp) and bool(_hex_filled_octants(elems[i], kp)[0])
    ]
    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for hi in active:
        corners = elems[int(hi)]
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, round_decimals)
            face_owners.setdefault(fkey, []).append((int(hi), int(fi)))

    supports: list[RoleSupport] = []
    for hi in active:
        corners = elems[int(hi)]
        kept_xyz = kept_planes_per_hex[int(hi)]
        for fi, face in enumerate(_HEX_FACES):
            axis, is_max = _FACE_AXIS_MAX[int(fi)]
            kept = np.asarray(kept_xyz[int(axis)], dtype=np.float64)
            outer_f = 1.0 if is_max else 0.0
            if _plane_kept(kept, outer_f):
                is_mid = False
            elif _plane_kept(kept, 0.5):
                is_mid = True
            else:
                continue
            covering_neighbors: list[int] = []
            if not is_mid:
                fkey = _face_key(corners, face, round_decimals)
                owners = face_owners.get(fkey, [])
                if len(owners) > 1:
                    completely_covered = False
                    for (n_hi, n_fi) in owners:
                        if n_hi == int(hi):
                            continue
                        covering_neighbors.append(int(n_hi))
                        n_kp = kept_planes_per_hex[n_hi]
                        other_axes = [a for a in (0, 1, 2) if a != int(axis)]
                        covers_all = True
                        for ax in other_axes:
                            hi_min, hi_max = float(np.min(kept_xyz[ax])), float(np.max(kept_xyz[ax]))
                            n_min, n_max = float(np.min(n_kp[ax])), float(np.max(n_kp[ax]))
                            if n_min > hi_min + 0.01 or n_max < hi_max - 0.01:
                                covers_all = False
                                break
                        if covers_all:
                            completely_covered = True
                            break
                    if completely_covered:
                        continue

            full_quad = axis_aligned_face_quad(corners, int(fi), midplane=is_mid)
            if not is_mid and covering_neighbors:
                sub_quads = _compute_exposed_subquads(
                    corners, int(fi), kept_xyz, covering_neighbors, kept_planes_per_hex
                )
                for quad in sub_quads:
                    supports.append(
                        RoleSupport(
                            hex_i=int(hi),
                            face_i=int(fi),
                            is_midplane=bool(is_mid),
                            quad=np.asarray(quad, dtype=np.float64),
                            edges=_quad_edges(quad),
                            covering_neighbors=covering_neighbors,
                            seam_edges=_quad_edges(full_quad),
                        )
                    )
            else:
                supports.append(
                    RoleSupport(
                        hex_i=int(hi),
                        face_i=int(fi),
                        is_midplane=bool(is_mid),
                        quad=np.asarray(full_quad, dtype=np.float64),
                        edges=_quad_edges(full_quad),
                        covering_neighbors=[],
                        seam_edges=None,
                    )
                )
    return supports


def bind_roles(
    nodes: np.ndarray,
    supports: list[RoleSupport],
    hex_elems: np.ndarray,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    *,
    plane_eps_frac: float = 1e-4,
    role_atol: float = _ROLE_ATOL,
) -> list[RoleSupport]:
    """Classify volume nodes and bind them onto supports via point_on_quad."""
    pts = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    elems = np.asarray(hex_elems, dtype=np.float64)
    roles = np.empty(len(pts), dtype=object)
    roles[:] = "K"
    # Classify from the owning hex of the first support that contains the point,
    # else from nearest hex AABB.
    for i, p in enumerate(pts):
        classified = False
        for s in supports:
            corners = elems[int(s.hex_i)]
            span = hex_span(corners)
            plane_eps = max(float(plane_eps_frac) * float(np.min(span)), 1e-6)
            if point_on_quad(p, s.quad, plane_eps=plane_eps):
                roles[i] = classify_uvw(hex_uvw(corners, p), atol=role_atol)
                classified = True
                break
        if not classified and len(elems) > 0:
            # Interior / unused: still label from nearest hex for counts.
            d0 = np.linalg.norm(p - elems[0].mean(axis=0))
            hi = 0
            for h in range(1, len(elems)):
                d = float(np.linalg.norm(p - elems[h].mean(axis=0)))
                if d < d0:
                    d0, hi = d, h
            roles[i] = classify_uvw(hex_uvw(elems[hi], p), atol=role_atol)

    for s in supports:
        s.gids_by_role = {"C": [], "E": [], "F": [], "B": [], "K": []}
        corners = elems[int(s.hex_i)]
        span = hex_span(corners)
        plane_eps = max(float(plane_eps_frac) * float(np.min(span)), 1e-6)
        for i, p in enumerate(pts):
            if not point_on_quad(p, s.quad, plane_eps=plane_eps):
                continue
            role = str(roles[i])
            s.gids_by_role.setdefault(role, []).append(int(i))
    return supports


def classify_nodes(
    nodes: np.ndarray,
    hex_corners: np.ndarray,
    *,
    atol: float = _ROLE_ATOL,
) -> np.ndarray:
    """Role labels for nodes of a single hex stamp."""
    pts = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    corners = np.asarray(hex_corners, dtype=np.float64)
    return np.array([classify_uvw(hex_uvw(corners, p), atol=atol) for p in pts], dtype=object)


def synthesize_missing_f(
    nodes: np.ndarray,
    supports: list[RoleSupport],
    hex_elems: np.ndarray,
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, list[RoleSupport]]:
    """Insert dual-only F: face centroid on outer supports, edge midpoints on cuts."""
    pts = [np.asarray(p, dtype=np.float64).copy() for p in np.asarray(nodes, dtype=np.float64)]
    key_to_gid = {
        tuple(np.round(p, round_decimals).tolist()): i for i, p in enumerate(pts)
    }
    elems = np.asarray(hex_elems, dtype=np.float64)

    def _add(pt: np.ndarray) -> int:
        key = tuple(np.round(np.asarray(pt, dtype=np.float64), round_decimals).tolist())
        gid = key_to_gid.get(key)
        if gid is None:
            gid = len(pts)
            pts.append(np.asarray(pt, dtype=np.float64).copy())
            key_to_gid[key] = gid
        return int(gid)

    for s in supports:
        f_gids = list(s.gids_by_role.get("F") or [])
        has_host = bool(s.gids_by_role.get("C") or s.gids_by_role.get("B") or f_gids)
        if not has_host:
            continue
        if s.is_midplane:
            if len(f_gids) >= 4:
                continue
            for a, b in s.edges:
                gid = _add(0.5 * (np.asarray(a) + np.asarray(b)))
                if gid not in f_gids:
                    f_gids.append(gid)
        else:
            if f_gids:
                continue
            quad_center = np.asarray(s.quad, dtype=np.float64).mean(axis=0)
            f_gids.append(_add(quad_center))
        s.gids_by_role["F"] = f_gids
    return np.asarray(pts, dtype=np.float64), supports


def _unique_edges(struts) -> np.ndarray:
    out: list[tuple[int, int]] = []
    arr = np.asarray(struts, dtype=np.int64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    for a, b in arr.reshape(-1, 2):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        out.append((ia, ib) if ia < ib else (ib, ia))
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(set(out)), dtype=np.int64)


def _compact(nodes: np.ndarray, struts: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    pts = np.asarray(nodes, dtype=np.float64)
    edges = _unique_edges(struts)
    if len(pts) == 0 or len(edges) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {},
        )
    used = np.unique(edges.reshape(-1))
    remap = {int(g): i for i, g in enumerate(used.tolist())}
    out_nodes = pts[used]
    out_struts = np.asarray(
        [[remap[int(a)], remap[int(b)]] for a, b in edges],
        dtype=np.int64,
    )
    return out_nodes, out_struts, remap


def _max_edge_length(cell_spans: np.ndarray) -> float:
    m = float(np.max(np.asarray(cell_spans, dtype=np.float64)))
    return float(np.sqrt(2.0) * m)


def _edge_ok(pa: np.ndarray, pb: np.ndarray, max_len: float) -> bool:
    return float(np.linalg.norm(pa - pb)) <= float(max_len) + 1e-9


def _undirected_edge_set(struts) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    arr = np.asarray(struts, dtype=np.int64)
    if arr.size == 0:
        return out
    for a, b in arr.reshape(-1, 2):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        out.add((ia, ib) if ia < ib else (ib, ia))
    return out


def _support_c(s: RoleSupport) -> list[int]:
    return [int(g) for g in (s.gids_by_role.get("C") or [])]


def _quad_scale(quad: np.ndarray) -> float:
    q = np.asarray(quad, dtype=np.float64)
    span = q.max(axis=0) - q.min(axis=0)
    in_plane = sorted(float(span[i]) for i in range(3))
    return max(in_plane[-1], in_plane[-2], 1e-15)


def classify_support_local(
    pt: np.ndarray,
    quad: np.ndarray,
    *,
    atol_frac: float = _ROLE_ATOL,
) -> RoleName:
    """C / E / F / K from where ``pt`` sits on this support rectangle."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 3)
    p = np.asarray(pt, dtype=np.float64).reshape(3)
    atol = max(float(atol_frac) * _quad_scale(q), 1e-9)
    if min(float(np.linalg.norm(p - c)) for c in q) <= atol:
        return "C"
    mids = [0.5 * (q[i] + q[(i + 1) % 4]) for i in range(4)]
    if min(float(np.linalg.norm(p - m)) for m in mids) <= atol:
        return "E"
    center = q.mean(axis=0)
    if float(np.linalg.norm(p - center)) <= atol:
        return "F"
    return "K"


def _support_volume_gids(s: RoleSupport) -> list[int]:
    """Gids bound on this support."""
    out: list[int] = []
    seen: set[int] = set()
    for role, gids in s.gids_by_role.items():
        for g in gids:
            ig = int(g)
            if ig not in seen:
                seen.add(ig)
                out.append(ig)
    return out


def _hex_role_buckets(s: RoleSupport) -> dict[str, list[int]]:
    """Hex UVW roles already bound on the support (not support-local C/E/F)."""
    return {
        "C": [int(g) for g in (s.gids_by_role.get("C") or [])],
        "E": [int(g) for g in (s.gids_by_role.get("E") or [])],
        "F": [int(g) for g in (s.gids_by_role.get("F") or [])],
        "K": [int(g) for g in (s.gids_by_role.get("K") or [])],
    }


def present_hex_roles(buckets: dict[str, list[int]]) -> set[str]:
    return {role for role, gids in buckets.items() if gids}


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _support_has_volume_cross(
    s: RoleSupport,
    buckets: dict[str, list[int]],
    vol_edge_set: set[tuple[int, int]],
) -> bool:
    """Outer face already has a volume C–F cross: dual must not duplicate it.

    Midplane cuts and hanging F (no volume spokes) still need dual.
    """
    if s.is_midplane:
        return False
    fs = buckets.get("F") or []
    cs = buckets.get("C") or []
    if not fs or not cs:
        return False
    for f in fs:
        if any(_pair(int(f), int(c)) in vol_edge_set for c in cs):
            return True
    return False


def _support_local_buckets(
    s: RoleSupport,
    pts: np.ndarray,
    filled: set[tuple[int, int, int]] | None = None,
    half_size: np.ndarray | None = None,
) -> dict[str, list[int]]:
    """Support-local C/E/F hint (where on the rectangle). Not used for identity."""
    buckets: dict[str, list[int]] = {"C": [], "E": [], "F": [], "K": []}
    for g in _support_volume_gids(s):
        p = pts[int(g)]
        if filled:
            if half_size is None or not point_touches_occupancy(p, filled, half_size):
                continue
        role = classify_support_local(p, s.quad)
        buckets.setdefault(role, []).append(int(g))
    return buckets


def _winning_layer(buckets: dict[str, list[int]]) -> str:
    """Presence summary; C/E no longer excludes F."""
    present = present_hex_roles(buckets)
    if ("C" in present or "E" in present) and "F" in present:
        return "CEF"
    if "C" in present or "E" in present:
        return "CE"
    if "F" in present:
        return "F"
    return "K"


def _nearest_gid(
    gids: list[int],
    pts: np.ndarray,
    target: np.ndarray,
    *,
    atol: float,
) -> int | None:
    if not gids:
        return None
    t = np.asarray(target, dtype=np.float64).reshape(3)
    best_g = int(gids[0])
    best_d = float(np.linalg.norm(pts[best_g] - t))
    for g in gids[1:]:
        d = float(np.linalg.norm(pts[int(g)] - t))
        if d < best_d:
            best_d = d
            best_g = int(g)
    if best_d > float(atol):
        return None
    return best_g


def _segment_key(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    decimals: int = _EDGE_DECIMALS,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    k0 = tuple(np.round(np.asarray(p0, dtype=np.float64), decimals).tolist())
    k1 = tuple(np.round(np.asarray(p1, dtype=np.float64), decimals).tolist())
    return (k0, k1) if k0 < k1 else (k1, k0)


def _support_candidate_segments(s: RoleSupport | np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if isinstance(s, RoleSupport):
        all_edges = list(s.edges)
        if s.seam_edges is not None:
            all_edges.extend(s.seam_edges)
        return [(a.copy(), b.copy()) for a, b in all_edges]
    c = np.asarray(s, dtype=np.float64).reshape(4, 3)
    return [
        (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]),
    ]
def _on_support_pattern_edges(
    s: RoleSupport,
    pts: np.ndarray,
    buckets: dict[str, list[int]],
    *,
    atol: float,
) -> list[tuple[int, int]]:
    """Table-driven canonical on-support pattern based on present roles."""
    quad = np.asarray(s.quad, dtype=np.float64).reshape(4, 3)
    out: list[tuple[int, int]] = []

    c_gids = buckets.get("C") or []
    e_gids = buckets.get("E") or []
    f_gids = buckets.get("F") or []
    b_gids = s.gids_by_role.get("B") or []

    c_nodes: list[int | None] = [_nearest_gid(c_gids, pts, quad[k], atol=atol) for k in range(4)]
    
    cand_e = e_gids if not s.is_midplane else (e_gids + f_gids)
    mids = [0.5 * (quad[k] + quad[(k + 1) % 4]) for k in range(4)]
    e_nodes: list[int | None] = [_nearest_gid(cand_e, pts, mids[k], atol=atol) for k in range(4)]

    cand_f = f_gids if not s.is_midplane else (f_gids + b_gids)
    center = quad.mean(axis=0)
    f_node = _nearest_gid(cand_f, pts, center, atol=atol)
    f_nodes: list[int] = [f_node] if f_node is not None else []

    has_c = any(g is not None for g in c_nodes)
    has_e = any(g is not None for g in e_nodes)
    has_f = bool(f_nodes)

    def add_pair(a: int | None, b: int | None) -> None:
        if a is not None and b is not None and a != b:
            out.append(_pair(int(a), int(b)))

    # Case 1: C + E + F (Diamond + Spokes to C + Perimeter C-E)
    if has_c and has_e and has_f:
        for k in range(4):
            add_pair(e_nodes[k], e_nodes[(k + 1) % 4])
        for f in f_nodes:
            for k in range(4):
                add_pair(int(f), c_nodes[k])
        for k in range(4):
            add_pair(c_nodes[k], e_nodes[(k - 1) % 4])
            add_pair(c_nodes[k], e_nodes[k])

    # Case 2: C + E, no F (Grid only)
    elif has_c and has_e and not has_f:
        for k in range(4):
            add_pair(c_nodes[k], c_nodes[(k + 1) % 4])

    # Case 3: E + F, no C (Diamond ONLY, no cross '+')
    elif has_e and has_f and not has_c:
        for k in range(4):
            add_pair(e_nodes[k], e_nodes[(k + 1) % 4])

    # Case 4: Only E (Diamond ONLY)
    elif has_e and not has_c and not has_f:
        for k in range(4):
            add_pair(e_nodes[k], e_nodes[(k + 1) % 4])

    # Case 5: C + F, no E (X Spokes only, no perimeter C-C)
    elif has_c and has_f and not has_e:
        for f in f_nodes:
            for k in range(4):
                add_pair(int(f), c_nodes[k])

    # Case 6: Only C (Grid perimeter)
    elif has_c and not has_e and not has_f:
        for k in range(4):
            add_pair(c_nodes[k], c_nodes[(k + 1) % 4])

    # Case 7: Only F -> no on-support edges (connected via shared-edge stitches)

    return out


def _active_gids(buckets: dict[str, list[int]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for role in ("C", "E", "F", "K"):
        for g in buckets.get(role) or []:
            ig = int(g)
            if ig not in seen:
                seen.add(ig)
                out.append(ig)
    return out


def _layer_actors(s: RoleSupport, buckets: dict[str, list[int]]) -> dict[str, list[int]]:
    f_list = list(buckets.get("F") or [])
    if s.is_midplane:
        b_list = [int(g) for g in (s.gids_by_role.get("B") or [])]
        f_list.extend(b_list)
        e_list = list(buckets.get("E") or []) + list(buckets.get("F") or [])
    else:
        e_list = list(buckets.get("E") or [])
    return {
        "C": list(buckets.get("C") or []),
        "E": e_list,
        "F": f_list,
    }


def _active_gids(s: RoleSupport, buckets: dict[str, list[int]]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    actors = _layer_actors(s, buckets)
    for role in ("C", "E", "F"):
        for g in actors.get(role) or []:
            ig = int(g)
            if ig not in seen:
                seen.add(ig)
                out.append(ig)
    return out


def _cut_axis_count(
    hex_i: int,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None,
) -> int:
    if not kept_planes_per_hex or hex_i >= len(kept_planes_per_hex):
        return 1
    kpx, kpy, kpz = kept_planes_per_hex[hex_i]
    cut_axes = 0
    if not (kpx[0] and kpx[1]):
        cut_axes += 1
    if not (kpy[0] and kpy[1]):
        cut_axes += 1
    if not (kpz[0] and kpz[1]):
        cut_axes += 1
    return max(cut_axes, 1)


def _starshot_stitch(
    f_gid: int,
    s: RoleSupport,
    nb: RoleSupport,
    nb_actors: dict[str, list[int]],
    pts: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    dual_edge_set: set[tuple[int, int]],
    cap: float,
    *,
    n_targets: int,
    atol: float,
) -> list[tuple[int, int]]:
    """Connect F across seam (p0, p1) to nb's C/E nodes when no connection exists."""
    nb_nodes = set(nb_actors["C"] + nb_actors["E"] + nb_actors["F"])
    # Check if f_gid is already connected to any node of nb
    for n in nb_nodes:
        if _pair(int(f_gid), int(n)) in dual_edge_set:
            return []

    fp = pts[int(f_gid)]
    cand_c = [int(g) for g in nb_actors["C"] if np.linalg.norm(pts[int(g)] - fp) <= cap]
    cand_e = [int(g) for g in nb_actors["E"] if np.linalg.norm(pts[int(g)] - fp) <= cap]

    if not cand_c and not cand_e:
        return []

    targets: list[int] = []
    if n_targets >= 2:
        # Check for symmetric E pair along seam
        p_diff = p1 - p0
        seg_len = float(np.linalg.norm(p_diff))
        if seg_len > 1e-9 and len(cand_e) >= 2:
            s_dir = p_diff / seg_len
            mid = 0.5 * (p0 + p1)
            best_pair = None
            best_sym_err = float("inf")
            for i in range(len(cand_e)):
                for j in range(i + 1, len(cand_e)):
                    e1, e2 = cand_e[i], cand_e[j]
                    proj1 = float(np.dot(pts[e1] - mid, s_dir))
                    proj2 = float(np.dot(pts[e2] - mid, s_dir))
                    sym_err = abs(proj1 + proj2)
                    if sym_err < atol and abs(proj1) > 1e-4:
                        if sym_err < best_sym_err:
                            best_sym_err = sym_err
                            best_pair = (e1, e2)
            if best_pair is not None:
                targets = [best_pair[0], best_pair[1]]

        if not targets:
            # Pick 2 closest C nodes
            if len(cand_c) >= 2:
                cand_c.sort(key=lambda g: float(np.linalg.norm(pts[g] - fp)))
                targets = [cand_c[0], cand_c[1]]
            elif len(cand_c) == 1:
                targets = [cand_c[0]]
            elif len(cand_e) > 0:
                cand_e.sort(key=lambda g: float(np.linalg.norm(pts[g] - fp)))
                targets = [cand_e[0]]
    else:
        # Single connection (quarter-cut or corner cut)
        all_cand = cand_c + cand_e
        all_cand.sort(key=lambda g: float(np.linalg.norm(pts[g] - fp)))
        targets = [all_cand[0]]

    return [(_pair(int(f_gid), int(t))) for t in targets if int(f_gid) != int(t)]


def build_layered_surface_dual(
    supports: list[RoleSupport],
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    *,
    volume_struts: np.ndarray | None = None,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    max_edge_length: float | None = None,
    edge_decimals: int = _EDGE_DECIMALS,
    rule_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Presence-gated layered dual: promote both-ends-on-support, then F–F / F–E / F–C."""
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    filled, half = filled_world_octants(elems, kept_planes_per_hex)
    if max_edge_length is None:
        spans = [hex_span(elems[s.hex_i]) for s in supports] or [np.ones(3)]
        max_edge_length = _max_edge_length(np.vstack(spans))

    buckets_list = [_hex_role_buckets(s) for s in supports]
    vol_edge_set = _undirected_edge_set(volume_struts) if volume_struts is not None else set()
    skin_complete = [
        _support_has_volume_cross(s, b, vol_edge_set)
        for s, b in zip(supports, buckets_list)
    ]

    if kept_planes_per_hex is not None:
        active_hexes = [
            i
            for i, kp in enumerate(kept_planes_per_hex)
            if _hex_is_active(kp) and bool(_hex_filled_octants(elems[i], kp)[0])
        ]
    else:
        active_hexes = list(range(len(elems)))

    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for hi in active_hexes:
        corners = elems[int(hi)]
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, edge_decimals)
            face_owners.setdefault(fkey, []).append((int(hi), int(fi)))

    edge_map: dict[tuple, list[tuple[int, tuple[np.ndarray, np.ndarray]]]] = {}
    for i, s in enumerate(supports):
        for p0, p1 in _support_candidate_segments(s):
            key = _segment_key(p0, p1, decimals=edge_decimals)
            edge_map.setdefault(key, []).append(
                (i, (np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)))
            )

    support_pairs_sharing_edge: set[tuple[int, int]] = set()
    for key, entries in edge_map.items():
        sup_ids = {sup_i for sup_i, _ in entries}
        for ia in sup_ids:
            for ib in sup_ids:
                if ia < ib:
                    support_pairs_sharing_edge.add((ia, ib))

    gid_to_supports: dict[int, list[int]] = {}
    for i, s in enumerate(supports):
        for g in _active_gids(s, buckets_list[i]):
            gid_to_supports.setdefault(int(g), []).append(i)

    def gids_share_support(ga: int, gb: int) -> bool:
        sups_a = gid_to_supports.get(int(ga), [])
        sups_b = gid_to_supports.get(int(gb), [])
        if not sups_a or not sups_b:
            return False
        # Rule 2: In-plane (both nodes share the same support)
        if any(sa in sups_b for sa in sups_a):
            return True
        # Rule 2.5: Supports of the same cell, or adjacent supports sharing an edge
        for sa in sups_a:
            for sb in sups_b:
                if supports[sa].hex_i == supports[sb].hex_i:
                    # Must be perpendicular (axes must differ; e.g. cannot be parallel top and bottom along the same axis)
                    axis_a = _FACE_AXIS_MAX[supports[sa].face_i][0]
                    axis_b = _FACE_AXIS_MAX[supports[sb].face_i][0]
                    if axis_a == axis_b:
                        continue
                    # Exactly one must be an exposed cut midplane, and the other an exposed lateral exterior face.
                    # This ensures the strut traverses an exterior boundary, not an interior face shared with another cell.
                    is_mid_a = supports[sa].is_midplane
                    is_mid_b = supports[sb].is_midplane
                    if not ((is_mid_a and not is_mid_b) or (is_mid_b and not is_mid_a)):
                        continue
                    # The strut must lie along an exterior boundary plane of the hex on the remaining third axis.
                    # If it does not lie in a constant plane or is interior to the cell on axis_c, it is cutting
                    # through the 3D volume of the cell and cannot be surface skin.
                    axis_c = 3 - axis_a - axis_b
                    corners = elems[supports[sa].hex_i]
                    c_val_a = pts[int(ga), axis_c]
                    c_val_b = pts[int(gb), axis_c]
                    if not np.isclose(c_val_a, c_val_b, atol=1e-3):
                        continue
                    c_mn = float(np.min(corners[:, axis_c]))
                    c_mx = float(np.max(corners[:, axis_c]))
                    is_face_c = None
                    if np.isclose(c_val_a, c_mn, atol=1e-3):
                        is_face_c = False
                    elif np.isclose(c_val_a, c_mx, atol=1e-3):
                        is_face_c = True
                    if is_face_c is None:
                        continue
                    # Verify that this third-axis exterior face is NOT shared with another occupied cell in the lattice.
                    fi_c = _AXIS_IS_MAX_TO_FACE[(axis_c, is_face_c)]
                    fkey = _face_key(corners, _HEX_FACES[fi_c], edge_decimals)
                    if len(face_owners.get(fkey, [])) > 1:
                        continue
                    return True
                pair = (sa, sb) if sa < sb else (sb, sa)
                if pair in support_pairs_sharing_edge:
                    return True
        return False

    def gid_on_incomplete(gid: int) -> bool:
        for s, buckets, complete in zip(supports, buckets_list, skin_complete):
            if complete:
                continue
            if int(gid) in _active_gids(s, buckets):
                return True
        return False

    def touches(a: int, b: int) -> bool:
        mid = 0.5 * (pts[int(a)] + pts[int(b)])
        return point_touches_occupancy(mid, filled, half)

    def try_add(
        edges: set[tuple[int, int]],
        a: int,
        b: int,
        *,
        require_touch: bool = False,
        check_length: bool = True,
    ) -> bool:
        if a == b:
            return False
        if require_touch and filled and not touches(a, b):
            return False
        e = _pair(int(a), int(b))
        if e in edges:
            return False
        edges.add(e)
        return True

    dual: set[tuple[int, int]] = set()
    n_native = 0
    n_same_cell = 0
    n_stitch = 0
    n_cap_skip = 0

    for s, buckets in zip(supports, buckets_list):
        atol = max(_ROLE_ATOL * _quad_scale(s.quad), 1e-9)
        for a, b in _on_support_pattern_edges(s, pts, buckets, atol=atol):
            if _pair(a, b) in vol_edge_set:
                continue
            if try_add(dual, a, b, require_touch=False, check_length=True):
                n_native += 1
            else:
                n_cap_skip += 1

    active: set[int] = set()
    for s, buckets in zip(supports, buckets_list):
        active.update(_active_gids(s, buckets))

    def is_cc_on_complete_support(ga: int, gb: int) -> bool:
        for buckets, complete in zip(buckets_list, skin_complete):
            if not complete:
                continue
            cs = set(buckets.get("C") or [])
            if int(ga) in cs and int(gb) in cs:
                return True
        return False

    if volume_struts is not None:
        cand = [
            (int(a), int(b))
            for a, b in np.asarray(volume_struts, dtype=np.int64).reshape(-1, 2)
        ]
    else:
        cand = []

    for ga, gb in cand:
        if ga not in active or gb not in active:
            continue
        if is_cc_on_complete_support(ga, gb):
            continue
        if not gids_share_support(ga, gb):
            continue
        if try_add(dual, ga, gb, check_length=False):
            n_native += 1
        else:
            n_cap_skip += 1

    by_id = {i: s for i, s in enumerate(supports)}
    for key, entries in edge_map.items():
        by_sup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for sup_i, seg in entries:
            if sup_i not in by_sup:
                by_sup[sup_i] = seg
        sup_ids = list(by_sup.keys())
        if len(sup_ids) < 2:
            continue
        for i_idx in range(len(sup_ids)):
            for j_idx in range(i_idx + 1, len(sup_ids)):
                ia, ib = sup_ids[i_idx], sup_ids[j_idx]
                fa, fb = by_id[ia], by_id[ib]
                p0, p1 = by_sup[ia]
                mid = 0.5 * (p0 + p1)
                aa = _layer_actors(fa, buckets_list[ia])
                ab = _layer_actors(fb, buckets_list[ib])
                atol_a = max(_ROLE_ATOL * _quad_scale(fa.quad), 1e-9)
                atol_b = max(_ROLE_ATOL * _quad_scale(fb.quad), 1e-9)
                same_hex = int(fa.hex_i) == int(fb.hex_i)
                if skin_complete[ia] and skin_complete[ib] and same_hex:
                    continue

                def count_add(
                    a: int,
                    b: int,
                    *,
                    stitch: bool,
                    require_touch: bool = False,
                    check_length: bool = True,
                ) -> None:
                    nonlocal n_native, n_stitch, n_same_cell, n_cap_skip
                    if try_add(
                        dual, a, b, require_touch=require_touch, check_length=check_length
                    ):
                        if stitch:
                            n_stitch += 1
                            if same_hex:
                                n_same_cell += 1
                        else:
                            n_native += 1
                    else:
                        n_cap_skip += 1

                c0a = _nearest_gid(aa["C"], pts, p0, atol=atol)
                c1a = _nearest_gid(aa["C"], pts, p1, atol=atol)
                c0b = _nearest_gid(ab["C"], pts, p0, atol=atol)
                c1b = _nearest_gid(ab["C"], pts, p1, atol=atol)
                if c0a is not None and c1a is not None and c0b is not None and c1b is not None:
                    ca = c0a if c0a == c0b else c0a
                    cb = c1a if c1a == c1b else c1a
                    if ca != cb:
                        count_add(ca, cb, stitch=False, require_touch=False, check_length=True)

                ga_list = _active_gids(fa, buckets_list[ia])
                gb_list = _active_gids(fb, buckets_list[ib])
                if ga_list and gb_list:
                    ga_pts = pts[np.asarray(ga_list, dtype=np.int64)]
                    gb_pts = pts[np.asarray(gb_list, dtype=np.int64)]
                    pairs = route_geometric_face_stitch(
                        ga_pts, gb_pts, p0, p1, allow_far=(fa.is_midplane or fb.is_midplane)
                    )
                    for ia_n, ib_n in pairs:
                        count_add(
                            int(ga_list[int(ia_n)]),
                            int(gb_list[int(ib_n)]),
                            stitch=True,
                            require_touch=False,
                            check_length=True,
                        )

    # Stage 2.5: Starshot / Octet stair-step corner fallback (Legacy heuristic override - preserved but disabled in favor of universal dual)
    # Connect cut center nodes to surviving unit-cell corner nodes (C) on adjacent higher steps
    legacy_star_fallback = False
    if rule_name == "star" and legacy_star_fallback:
        active_list = list(active)
        for s in supports:
            b_gids = s.gids_by_role.get("B") or []
            f_gids = s.gids_by_role.get("F") or []
            all_f = list(set(b_gids + f_gids))
            if not all_f:
                continue
            corners = elems[s.hex_i]
            atol = max(_ROLE_ATOL * _quad_scale(s.quad), 1e-4)
            for f in all_f:
                for cp in corners:
                    cgid = _nearest_gid(active_list, pts, cp, atol=atol)
                    if cgid is not None and int(cgid) != int(f):
                        if try_add(dual, int(f), int(cgid), require_touch=False, check_length=True):
                            n_stitch += 1
        active_list = list(active)
        for s in supports:
            if not s.is_midplane:
                continue
            cand_e = list(set((s.gids_by_role.get("E") or []) + (s.gids_by_role.get("F") or [])))
            if not cand_e:
                continue
            corners = elems[s.hex_i]
            max_d = float(np.linalg.norm(hex_span(elems[s.hex_i]))) * 0.75
            atol = max(_ROLE_ATOL * _quad_scale(s.quad), 1e-4)
            for e in cand_e:
                for cp in corners:
                    if float(np.linalg.norm(pts[e] - cp)) <= max_d:
                        cgid = _nearest_gid(active_list, pts, cp, atol=atol)
                        if cgid is not None and int(cgid) != int(e):
                            if try_add(dual, int(e), int(cgid), require_touch=False, check_length=True):
                                n_stitch += 1


    struts = (
        np.array(sorted(dual), dtype=np.int64)
        if dual
        else np.empty((0, 2), dtype=np.int64)
    )
    info = {
        "n_native_diamond": int(n_native),
        "n_same_cell_outer": int(n_same_cell),
        "n_stitches": int(n_stitch),
        "n_cap_skipped": int(n_cap_skip),
        "n_f_struts": int(len(struts)),
        "n_skin_complete_supports": int(sum(1 for c in skin_complete if c)),
        "max_edge_length": float(max_edge_length),
        "dual_rule": "layered_surface_dual_roles",
    }
    return struts, info


def build_f_diamond(
    supports: list[RoleSupport],
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    *,
    volume_struts: np.ndarray | None = None,
    invent_same_cell: bool = False,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    max_edge_length: float | None = None,
    edge_decimals: int = _EDGE_DECIMALS,
    rule_name: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Layered C/E-then-F dual (name kept for existing callers)."""
    del invent_same_cell
    return build_layered_surface_dual(
        supports,
        nodes,
        hex_elems,
        volume_struts=volume_struts,
        kept_planes_per_hex=kept_planes_per_hex,
        max_edge_length=max_edge_length,
        edge_decimals=edge_decimals,
        rule_name=rule_name,
    )


def build_c_perimeter(
    supports: list[RoleSupport],
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    *,
    max_edge_length: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Four quad edges among C nodes on each support. No face diagonals."""
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    if max_edge_length is None:
        spans = [hex_span(elems[s.hex_i]) for s in supports] or [np.ones(3)]
        max_edge_length = _max_edge_length(np.vstack(spans))

    dual: set[tuple[int, int]] = set()
    n_native = 0
    n_cap = 0
    for s in supports:
        c_gids = _support_c(s)
        if len(c_gids) < 2:
            continue
        quad = np.asarray(s.quad, dtype=np.float64)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            pa, pb = quad[a], quad[b]
            if not c_gids:
                continue
            ia = int(c_gids[int(np.argmin([np.linalg.norm(pts[g] - pa) for g in c_gids]))])
            ib = int(c_gids[int(np.argmin([np.linalg.norm(pts[g] - pb) for g in c_gids]))])
            if ia == ib:
                continue
            if not _edge_ok(pts[ia], pts[ib], max_edge_length):
                n_cap += 1
                continue
            e = (ia, ib) if ia < ib else (ib, ia)
            if e not in dual:
                dual.add(e)
                n_native += 1
    struts = (
        np.array(sorted(dual), dtype=np.int64)
        if dual
        else np.empty((0, 2), dtype=np.int64)
    )
    return struts, {
        "n_c_perimeter": int(n_native),
        "n_cap_skipped": int(n_cap),
        "n_c_struts": int(len(struts)),
        "max_edge_length": float(max_edge_length),
    }


def build_k_bridges(
    hex_elems: np.ndarray,
    volume_nodes: np.ndarray,
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Kelvin exterior side-wall bridges (existing merge_surface_skin path)."""
    if len(hex_elems) == 0 or len(volume_nodes) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {"n_k_struts": 0},
        )
    rule_obj = get_hex_topology_rule("kelvin")
    all_nodes, skin = merge_surface_skin(
        hex_elems, volume_nodes, rule_obj, round_decimals=round_decimals
    )
    edges = _unique_edges(skin)
    if len(edges) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {"n_k_struts": 0},
        )
    used = np.unique(edges.reshape(-1))
    remap = {int(g): i for i, g in enumerate(used.tolist())}
    dual_nodes = np.asarray(all_nodes[used], dtype=np.float64)
    dual_struts = np.asarray(
        [[remap[int(a)], remap[int(b)]] for a, b in edges],
        dtype=np.int64,
    )
    return dual_nodes, dual_struts, {"n_k_struts": int(len(dual_struts))}


def build_role_surface_dual(
    hex_elems: np.ndarray,
    volume_nodes: np.ndarray,
    *,
    rule_name: str,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    cad_mesh: trimesh.Trimesh | None = None,
    volume_struts: np.ndarray | None = None,
    synthesize_f: bool | None = None,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    round_decimals: int = 6,
) -> RoleSurfaceDualResult:
    """Build F, C, or K dual from trimmed volume nodes + hex supports.

    F-policy lattices (octahedral, octet, star, hex_face_dual) use layered
    surface dual roles (presence-gated promote + F–F / F–E / F–C).
    Star does not synthesize face centers.
    """
    rule = str(rule_name).strip().lower()
    elems = np.asarray(hex_elems, dtype=np.float64)
    nodes = np.asarray(volume_nodes, dtype=np.float64).reshape(-1, 3)
    policy = dual_policy_for_rule(rule)
    if kept_planes_per_hex is None:
        if cad_mesh is None:
            raise ValueError("cad_mesh is required when kept_planes_per_hex is omitted")
        kept_planes_per_hex = compute_kept_planes_per_hex(
            cad_mesh,
            elems,
            rule,
            empty_vf_max=empty_vf_max,
            samples_per_axis=samples_per_axis,
            extent_samples_per_axis=extent_samples_per_axis,
        )
    supports = collect_role_supports(elems, kept_planes_per_hex, round_decimals=round_decimals)
    supports = bind_roles(nodes, supports, elems, kept_planes_per_hex=kept_planes_per_hex)
    do_synth = False if synthesize_f is None else bool(synthesize_f)
    if policy == "F" and do_synth:
        nodes, supports = synthesize_missing_f(nodes, supports, elems, round_decimals=round_decimals)

    report: dict = {
        "rule_name": rule,
        "policy": policy,
        "n_supports": int(len(supports)),
        "n_midplane": int(sum(1 for s in supports if s.is_midplane)),
        "n_outer": int(sum(1 for s in supports if not s.is_midplane)),
        "synthesize_f": bool(do_synth),
        "dual_rule": "layered_surface_dual_roles",
    }

    if policy == "K":
        active = [i for i, kp in enumerate(kept_planes_per_hex) if _hex_is_active(kp)]
        hex_k = elems[np.asarray(active, dtype=np.int64)] if active else np.empty((0, 8, 3))
        dual_nodes, dual_struts, kinfo = build_k_bridges(hex_k, nodes, round_decimals=round_decimals)
        gids = set(range(len(dual_nodes))) if len(dual_nodes) else set()
        report.update(kinfo)
        return RoleSurfaceDualResult(
            dual_nodes=dual_nodes,
            dual_struts=dual_struts,
            surface_gids=gids,
            supports=supports,
            report=report,
        )

    if policy == "C":
        struts, info = build_c_perimeter(supports, nodes, elems)
    else:
        struts, info = build_f_diamond(
            supports,
            nodes,
            elems,
            volume_struts=volume_struts,
            invent_same_cell=bool(do_synth),
            kept_planes_per_hex=kept_planes_per_hex,
            rule_name=rule,
        )
    report.update(info)
    dual_nodes, dual_struts, remap = _compact(nodes, struts)
    surface_gids = set(remap.keys())
    report["n_dual_nodes"] = int(len(dual_nodes))
    report["n_dual_struts"] = int(len(dual_struts))
    return RoleSurfaceDualResult(
        dual_nodes=dual_nodes,
        dual_struts=dual_struts,
        surface_gids=surface_gids,
        supports=supports,
        report=report,
    )
