"""Gated stitch: cut volume nodes → surface dual (V2 surface-first).

Hard gates (SURFACE_FIRST_DUAL_TRIM.md §3):
  - horizontal reach ‖(d* − c)_XY‖ ≤ L with L = max(s_x, s_y)
  - angle from vertical θ ≤ 60°
  - fail → orphan cull hanging stub(s)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class StitchReport:
    n_cut: int = 0
    n_stitched: int = 0
    n_rejected_distance: int = 0
    n_rejected_angle: int = 0
    n_rejected_valence: int = 0
    n_orphaned: int = 0
    unit_cell_length_used: float = 0.0
    stitch_gates: str = "strict"
    valence_cap: int | None = None
    stitch_pairs: list[tuple[int, int]] = field(default_factory=list)
    orphaned_cut_ids: list[int] = field(default_factory=list)
    reject_reasons: dict[int, str] = field(default_factory=dict)


def unit_cell_length_horizontal(cell_size: float | tuple[float, float, float] | np.ndarray) -> float:
    """L = max(s_x, s_y) for anisotropic; scalar cell → that value."""
    cs = np.asarray(cell_size, dtype=np.float64).ravel()
    if cs.size == 1:
        return float(cs[0])
    if cs.size < 2:
        raise ValueError(f"cell_size needs ≥1 component; got {cs}")
    return float(max(cs[0], cs[1]))


def horizontal_reach(c: np.ndarray, d: np.ndarray) -> float:
    """Plan-view (XY) distance between cut and dual nodes."""
    diff = np.asarray(d, dtype=np.float64).ravel()[:2] - np.asarray(c, dtype=np.float64).ravel()[:2]
    return float(np.linalg.norm(diff))


def angle_from_vertical_deg(c: np.ndarray, d: np.ndarray, up: np.ndarray | None = None) -> float:
    """θ = arccos(|u·ẑ| / ‖u‖) in degrees; 0° = vertical, 90° = horizontal."""
    u = np.asarray(d, dtype=np.float64).ravel()[:3] - np.asarray(c, dtype=np.float64).ravel()[:3]
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-15:
        return 0.0
    zhat = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64)
    zhat = zhat / max(float(np.linalg.norm(zhat)), 1e-15)
    cos_theta = abs(float(np.dot(u, zhat))) / nrm
    cos_theta = min(1.0, max(0.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def angle_from_inward_normal_deg(
    cut: np.ndarray,
    dual: np.ndarray,
    dual_outward_normal: np.ndarray,
) -> float:
    """
    Angle between (cut − dual) and the CAD inward normal (−n_outward).

    0° = stitch approaches the dual straight from inside along the face normal
    (correct for floor, roof, and side walls). 90° = tangent slide on the skin.
    """
    u = np.asarray(cut, dtype=np.float64).ravel()[:3] - np.asarray(dual, dtype=np.float64).ravel()[:3]
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-15:
        return 0.0
    n = np.asarray(dual_outward_normal, dtype=np.float64).ravel()[:3]
    nn = float(np.linalg.norm(n))
    if nn < 1e-15:
        return 90.0
    inward = -n / nn
    cos_theta = float(np.dot(u / nrm, inward))
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))


def passes_distance_gate(c: np.ndarray, d: np.ndarray, L: float) -> bool:
    return horizontal_reach(c, d) <= float(L) + 1e-9


def passes_angle_gate(
    c: np.ndarray,
    d: np.ndarray,
    max_angle_deg: float = 60.0,
    up: np.ndarray | None = None,
) -> bool:
    return angle_from_vertical_deg(c, d, up=up) <= float(max_angle_deg) + 1e-9


def passes_approach_gate(
    cut: np.ndarray,
    dual: np.ndarray,
    dual_outward_normal: np.ndarray,
    max_angle_deg: float = 60.0,
) -> bool:
    return (
        angle_from_inward_normal_deg(cut, dual, dual_outward_normal)
        <= float(max_angle_deg) + 1e-9
    )

def stitch_cut_to_dual(
    cut_nodes: np.ndarray,
    dual_nodes: np.ndarray,
    *,
    L: float,
    max_angle_deg: float = 60.0,
    search_radius: float | None = None,
    up: np.ndarray | None = None,
    dual_outward_normals: np.ndarray | None = None,
    angle_mode: str = "inward_normal",
    stitch_gates: str = "strict",
    valence_cap: int | None = None,
) -> tuple[np.ndarray, StitchReport]:
    """
    For each cut node, pick a dual partner and optionally apply hard gates.

    ``stitch_gates``:
      - ``"strict"`` (V2 default): distance <= L and approach/vertical angle gate.
      - ``"off"`` (V2.1 default): nearest dual within search radius; optional
        ``valence_cap`` only.

    ``angle_mode`` (strict only):
      - ``"inward_normal"`` (default): angle between (cut-dual) and -n_outward.
      - ``"world_vertical"``: legacy theta from +Z.

    Returns
    -------
    stitch_struts : (S, 2) int
        Indices as (cut_local_id, dual_local_id) — caller remaps into a global node table.
    report : StitchReport
    """
    cuts = np.asarray(cut_nodes, dtype=np.float64).reshape(-1, 3)
    duals = np.asarray(dual_nodes, dtype=np.float64).reshape(-1, 3)
    gates = str(stitch_gates).strip().lower()
    if gates not in ("strict", "off"):
        raise ValueError(f"stitch_gates must be 'strict' or 'off'; got {stitch_gates!r}")
    report = StitchReport(
        n_cut=int(len(cuts)),
        unit_cell_length_used=float(L),
        stitch_gates=gates,
        valence_cap=None if valence_cap is None else int(valence_cap),
    )
    mode = str(angle_mode).strip().lower()
    if mode not in ("inward_normal", "world_vertical"):
        raise ValueError(f"angle_mode must be inward_normal|world_vertical; got {angle_mode!r}")
    normals = None
    if gates == "strict" and mode == "inward_normal":
        if dual_outward_normals is None:
            raise ValueError("dual_outward_normals required for angle_mode='inward_normal'")
        normals = np.asarray(dual_outward_normals, dtype=np.float64).reshape(-1, 3)
        if len(normals) != len(duals):
            raise ValueError("dual_outward_normals must match dual_nodes length")

    if len(cuts) == 0 or len(duals) == 0:
        report.n_orphaned = int(len(cuts))
        report.orphaned_cut_ids = list(range(len(cuts)))
        for i in range(len(cuts)):
            report.reject_reasons[i] = "no_dual"
        return np.empty((0, 2), dtype=np.int64), report

    R = float(L if search_radius is None else search_radius)
    R = max(R, float(L))
    tree = cKDTree(duals)
    bond_count = np.zeros(len(duals), dtype=np.int32)
    cap = None if valence_cap is None else int(valence_cap)

    order = np.lexsort((cuts[:, 1], cuts[:, 0], cuts[:, 2]))

    pairs: list[tuple[int, int]] = []
    for ci in order:
        ci = int(ci)
        c = cuts[ci]
        idxs = tree.query_ball_point(c, r=R)
        if not idxs:
            report.n_orphaned += 1
            report.orphaned_cut_ids.append(ci)
            report.reject_reasons[ci] = "no_candidate"
            continue

        if gates == "off":
            ranked = sorted(
                (float(np.linalg.norm(duals[int(j)] - c)), int(j)) for j in idxs
            )
            chosen: int | None = None
            valence_blocked = False
            for _dist, j in ranked:
                if cap is not None and int(bond_count[j]) >= cap:
                    valence_blocked = True
                    continue
                chosen = j
                break
            if chosen is None:
                report.n_orphaned += 1
                report.orphaned_cut_ids.append(ci)
                if valence_blocked:
                    report.n_rejected_valence += 1
                    report.reject_reasons[ci] = "valence"
                else:
                    report.reject_reasons[ci] = "no_candidate"
                continue
            pairs.append((ci, chosen))
            report.stitch_pairs.append((ci, chosen))
            report.n_stitched += 1
            bond_count[chosen] += 1
            continue

        best_local: int | None = None
        best_angle = float("inf")
        any_in_distance = False
        valence_blocked = False
        for j in idxs:
            j = int(j)
            d = duals[j]
            if not passes_distance_gate(c, d, L):
                continue
            any_in_distance = True
            if cap is not None and int(bond_count[j]) >= cap:
                valence_blocked = True
                continue
            if mode == "inward_normal":
                ang = angle_from_inward_normal_deg(c, d, normals[j])
            else:
                ang = angle_from_vertical_deg(c, d, up=up)
            if ang <= float(max_angle_deg) + 1e-9 and ang < best_angle:
                best_angle = ang
                best_local = j

        if best_local is None:
            if not any_in_distance:
                report.n_rejected_distance += 1
                report.reject_reasons[ci] = "distance"
            elif valence_blocked:
                report.n_rejected_valence += 1
                report.reject_reasons[ci] = "valence"
            else:
                report.n_rejected_angle += 1
                report.reject_reasons[ci] = "angle"
            report.n_orphaned += 1
            report.orphaned_cut_ids.append(ci)
            continue

        pairs.append((ci, best_local))
        report.stitch_pairs.append((ci, best_local))
        report.n_stitched += 1
        bond_count[best_local] += 1

    if not pairs:
        return np.empty((0, 2), dtype=np.int64), report
    return np.asarray(pairs, dtype=np.int64), report



def orphan_cull_volume_graph(
    nodes: np.ndarray,
    struts: np.ndarray,
    orphan_node_ids: np.ndarray | list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Delete orphan cut nodes and any incident hanging stub edges.

    Returns compact (nodes, struts, old_to_new) where old_to_new[i] = new index or -1.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    drop = set(int(i) for i in np.asarray(orphan_node_ids, dtype=np.int64).ravel())
    if not drop:
        mapping = np.arange(len(nodes), dtype=np.int64)
        return nodes, struts, mapping

    keep_edge = []
    for a, b in struts:
        ai, bi = int(a), int(b)
        if ai in drop or bi in drop:
            continue
        keep_edge.append((ai, bi))

    keep_nodes = [i for i in range(len(nodes)) if i not in drop]
    old_to_new = np.full(len(nodes), -1, dtype=np.int64)
    for new_i, old_i in enumerate(keep_nodes):
        old_to_new[old_i] = new_i
    new_nodes = nodes[keep_nodes] if keep_nodes else np.empty((0, 3), dtype=np.float64)
    new_struts = []
    for a, b in keep_edge:
        na, nb = int(old_to_new[a]), int(old_to_new[b])
        if na < 0 or nb < 0 or na == nb:
            continue
        new_struts.append((min(na, nb), max(na, nb)))
    strut_arr = (
        np.asarray(sorted(set(new_struts)), dtype=np.int64)
        if new_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return new_nodes, strut_arr, old_to_new


def stitch_by_cartesian_origin(
    cut_outside_points: np.ndarray,
    dual_origins: np.ndarray,
    *,
    match_tol: float = 1e-3,
) -> tuple[np.ndarray, StitchReport]:
    """
    Identity stitch: each cut's clipped outside endpoint maps to the dual whose
    pre-projection Cartesian origin coincides with that endpoint.

    Returns (cut_local_i, dual_local_j) pairs; orphans listed when no origin matches.
    """
    outs = np.asarray(cut_outside_points, dtype=np.float64).reshape(-1, 3)
    origins = np.asarray(dual_origins, dtype=np.float64).reshape(-1, 3)
    report = StitchReport(
        n_cut=int(len(outs)),
        stitch_gates="origin_identity",
        unit_cell_length_used=float(match_tol),
    )
    if len(outs) == 0 or len(origins) == 0:
        report.n_orphaned = int(len(outs))
        report.orphaned_cut_ids = list(range(len(outs)))
        return np.empty((0, 2), dtype=np.int64), report

    tree = cKDTree(origins)
    pairs: list[tuple[int, int]] = []
    for i, p in enumerate(outs):
        dist, j = tree.query(p)
        if float(dist) > float(match_tol):
            report.n_orphaned += 1
            report.orphaned_cut_ids.append(int(i))
            report.reject_reasons[int(i)] = "origin_mismatch"
            continue
        pairs.append((int(i), int(j)))
        report.stitch_pairs.append((int(i), int(j)))
        report.n_stitched += 1

    return (
        np.asarray(pairs, dtype=np.int64)
        if pairs
        else np.empty((0, 2), dtype=np.int64),
        report,
    )
