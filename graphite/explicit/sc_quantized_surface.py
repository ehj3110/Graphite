"""
Quantized Topological Boundary — mixed Full/Half surface manifold + TNP.

Pipeline (Task 15 — explicit tagging, no coincidence extraction):
  1. Stamp with SurfaceStampTags (Half diamond + Full exposed FC)
  2. Native surface struts = tagged Half diamond perimeters only
  3. Manifold stitch: Full exposed FC → 4 topological diagonal neighbor
     surface nodes on the unwrapped exposed-face graph
  4. TNP only Surface Set nodes; all others stay Cartesian-locked
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from graphite.explicit.compare_v2_surface_first.dual import target_normal_project
from graphite.explicit.hex_rules import (
    _HALF_NEG_X_FACE_INDICES,
    _HALF_NEG_Y_FACE_INDICES,
    _HALF_NEG_Z_FACE_INDICES,
    _HALF_POS_X_FACE_INDICES,
    _HALF_POS_Y_FACE_INDICES,
    _HALF_POS_Z_FACE_INDICES,
    _HEX_FACES,
    _hex_face_centers,
)
from graphite.explicit.hex_topology_module import (
    SurfaceStampTags,
    resolve_quantized_cell_rule,
)

_HALF_FACE_TABLE = {
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

_TARGET_SURFACE_DEGREE = 4


@dataclass
class UnlockReport:
    n_nodes: int = 0
    n_unlocked: int = 0
    n_locked: int = 0
    n_half_diamond: int = 0
    n_half_center: int = 0
    n_full_exposed: int = 0
    n_surface_nodes: int = 0
    n_inherent_surface_struts: int = 0
    n_manifold_stitches: int = 0
    n_exposed_faces: int = 0
    n_projected_ray: int = 0
    n_projected_closest: int = 0
    n_tnp_hit: int = 0
    max_travel: float = 0.0
    mean_travel: float = 0.0
    notes: list[str] = field(default_factory=list)
    skin_report: dict = field(default_factory=dict)
    skin_faces: list = field(default_factory=list)


@dataclass
class _ExposedFace:
    """One exposed SC hex face on the unwrapped surface manifold."""

    face_id: int
    hex_i: int
    face_i: int
    rule: str
    is_full: bool
    primary_gid: int | None
    corners: tuple[tuple[float, float, float], ...]
    edges: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]
    normal: np.ndarray


def _tag_to_rule(tag: str | None) -> str | None:
    return resolve_quantized_cell_rule(tag)


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


def _outward_face_normal(corners: np.ndarray, face_index: int) -> np.ndarray:
    """Unit outward normal of a hex face (away from cell centroid)."""
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


def _cell_xyz(
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    cs = np.asarray(cell_size, dtype=np.float64).ravel()
    if cs.size == 1:
        return np.full(3, float(cs[0]), dtype=np.float64)
    if cs.size != 3:
        raise ValueError(f"cell_size must be scalar or length-3; got {cs}")
    return cs.astype(np.float64, copy=False)


def _build_exposed_faces(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int,
) -> tuple[list[_ExposedFace], dict[tuple, list[tuple[int, int]]], list[str | None]]:
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    key_to_gid = {_corner_key(p, round_decimals): i for i, p in enumerate(pts)}

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

    exposed: list[_ExposedFace] = []
    for hi, (corners, rule) in enumerate(zip(elems, rules)):
        if rule is None:
            continue
        face_ctrs = _hex_face_centers(corners)
        is_full = rule == "octahedral"
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            primary = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
            # Half empty-side face has no stamped face-center node.
            corners_k = tuple(_corner_key(corners[i], round_decimals) for i in face)
            edges = tuple(
                _edge_key(corners[face[a]], corners[face[b]], round_decimals)
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            exposed.append(
                _ExposedFace(
                    face_id=len(exposed),
                    hex_i=hi,
                    face_i=fi,
                    rule=rule,
                    is_full=is_full,
                    primary_gid=None if primary is None else int(primary),
                    corners=corners_k,
                    edges=edges,
                    normal=_outward_face_normal(corners, fi),
                )
            )
    return exposed, face_owners, rules


# ---------------------------------------------------------------------------
# Step 2 — Surface Set + inherent surface struts
# ---------------------------------------------------------------------------


def identify_surface_set(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
    stamp_tags: SurfaceStampTags | None = None,
) -> tuple[np.ndarray, np.ndarray, UnlockReport, list[_ExposedFace]]:
    """
    Mark Surface Nodes and unlock normals.

    Surface Set (explicit tags / equivalent walk):
      - Half mid-plane diamond corners only (locals 1–4) — NOT apex, NOT center
      - Full-cell exposed face-center nodes
    """
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    if len(tags) != len(elems):
        raise ValueError("cell_tags length must match hex_elems")
    report = UnlockReport(n_nodes=len(pts))
    if len(pts) == 0:
        return (
            np.zeros(0, dtype=bool),
            np.zeros((0, 3), dtype=np.float64),
            report,
            [],
        )

    key_to_gid = {_corner_key(p, round_decimals): i for i, p in enumerate(pts)}
    exposed, face_owners, rules = _build_exposed_faces(
        pts, elems, tags, round_decimals=round_decimals
    )
    report.n_exposed_faces = len(exposed)

    surface = np.zeros(len(pts), dtype=bool)
    normal_acc = np.zeros((len(pts), 3), dtype=np.float64)
    normal_w = np.zeros(len(pts), dtype=np.float64)

    def _mark(gid: int, normal: np.ndarray, *, source: str) -> None:
        surface[gid] = True
        if source == "full_exposed":
            report.n_full_exposed += 1
        elif source == "half_diamond":
            report.n_half_diamond += 1
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(n))
        if nrm > 1e-14:
            normal_acc[gid] += n / nrm
            normal_w[gid] += 1.0

    if stamp_tags is not None:
        surface = np.asarray(stamp_tags.surface_mask, dtype=bool).copy()
        if surface.shape[0] != len(pts):
            raise ValueError("stamp_tags.surface_mask length must match nodes")
        # Normals still from geometry walk for TNP.
        for hi, (corners, rule) in enumerate(zip(elems, rules)):
            if rule is None:
                continue
            face_ctrs = _hex_face_centers(corners)
            if rule in _HALF_FACE_TABLE:
                face_ids = _HALF_FACE_TABLE[rule]
                for local_i in range(1, 5):
                    fi = int(face_ids[local_i])
                    gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                    if gid is None or not surface[gid]:
                        continue
                    report.n_half_diamond += 1
                    n = _outward_face_normal(corners, fi)
                    nrm = float(np.linalg.norm(n))
                    if nrm > 1e-14:
                        normal_acc[gid] += n / nrm
                        normal_w[gid] += 1.0
                continue
            if rule == "octahedral":
                for fi in range(6):
                    fkey = _face_key(corners, _HEX_FACES[fi], round_decimals)
                    if len(face_owners.get(fkey, [])) != 1:
                        continue
                    gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                    if gid is None or not surface[gid]:
                        continue
                    report.n_full_exposed += 1
                    n = _outward_face_normal(corners, fi)
                    nrm = float(np.linalg.norm(n))
                    if nrm > 1e-14:
                        normal_acc[gid] += n / nrm
                        normal_w[gid] += 1.0
    else:
        for hi, (corners, rule) in enumerate(zip(elems, rules)):
            if rule is None:
                continue
            face_ctrs = _hex_face_centers(corners)
            if rule in _HALF_FACE_TABLE:
                face_ids = _HALF_FACE_TABLE[rule]
                for local_i in range(1, 5):
                    fi = int(face_ids[local_i])
                    gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                    if gid is None:
                        continue
                    _mark(gid, _outward_face_normal(corners, fi), source="half_diamond")
                continue
            if rule == "octahedral":
                for fi in range(6):
                    fkey = _face_key(corners, _HEX_FACES[fi], round_decimals)
                    if len(face_owners.get(fkey, [])) != 1:
                        continue
                    gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                    if gid is None:
                        continue
                    _mark(gid, _outward_face_normal(corners, fi), source="full_exposed")

    report.n_surface_nodes = int(np.count_nonzero(surface))
    report.n_unlocked = report.n_surface_nodes
    report.n_locked = int(len(pts) - report.n_unlocked)
    report.notes.append(
        f"surface_set half_diamond={report.n_half_diamond} "
        f"half_center={report.n_half_center} full_exposed={report.n_full_exposed} "
        f"(explicit tags; no half center)"
    )

    normals = np.zeros_like(pts)
    for i in np.flatnonzero(surface):
        if normal_w[i] > 0:
            n = normal_acc[i] / normal_w[i]
            nrm = float(np.linalg.norm(n))
            if nrm > 1e-14:
                normals[i] = n / nrm
    return surface, normals, report, exposed


def build_surface_node_face_ownership(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    surface_mask: np.ndarray,
    *,
    round_decimals: int = 6,
) -> dict[int, list[np.ndarray]]:
    """
    Map each unlocked surface node → outward normals of owning exposed faces.

    Full exposed face-centers own their Cartesian face normal. Half diamond
    corners own the corresponding SIDE face normal. A welded node that sits on
    several exposed faces collects every owning normal (needed for max-travel
    face-normal projection).
    """
    pts = np.asarray(nodes, dtype=np.float64)
    elems = np.asarray(hex_elems, dtype=np.float64)
    tags = list(cell_tags)
    mask = np.asarray(surface_mask, dtype=bool)
    if mask.shape[0] != len(pts):
        raise ValueError("surface_mask length must match nodes")

    key_to_gid = {_corner_key(p, round_decimals): i for i, p in enumerate(pts)}
    _exposed, face_owners, rules = _build_exposed_faces(
        pts, elems, tags, round_decimals=round_decimals
    )

    ownership: dict[int, list[np.ndarray]] = {int(i): [] for i in np.flatnonzero(mask)}

    def _add(gid: int, normal: np.ndarray) -> None:
        if gid not in ownership:
            return
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-14:
            return
        n = n / nrm
        # Dedup near-identical normals
        for existing in ownership[gid]:
            if float(np.dot(existing, n)) > 0.999:
                return
        ownership[gid].append(n)

    for _hi, (corners, rule) in enumerate(zip(elems, rules)):
        if rule is None:
            continue
        face_ctrs = _hex_face_centers(corners)
        if rule in _HALF_FACE_TABLE:
            face_ids = _HALF_FACE_TABLE[rule]
            for local_i in range(1, 5):
                fi = int(face_ids[local_i])
                gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                if gid is None:
                    continue
                _add(int(gid), _outward_face_normal(corners, fi))
            continue
        if rule == "octahedral":
            for fi in range(6):
                fkey = _face_key(corners, _HEX_FACES[fi], round_decimals)
                if len(face_owners.get(fkey, [])) != 1:
                    continue
                gid = key_to_gid.get(_corner_key(face_ctrs[fi], round_decimals))
                if gid is None:
                    continue
                _add(int(gid), _outward_face_normal(corners, fi))

    return ownership


def blend_unit_normals(
    normals: list[np.ndarray],
    *,
    cell_size: float | tuple[float, float, float] | np.ndarray | None = None,
) -> np.ndarray:
    """Sum owning normals weighted by cell axis length.

    A 12×12×4 cell with +X,+Y,+Z ownership travels along (3, 3, 1), not a
    45° equal-weight diagonal. Two equal XY axes remain 1:1 in the XY plane.
    """
    if cell_size is None:
        cell_xyz = np.ones(3, dtype=np.float64)
    else:
        cell_xyz = _cell_xyz(cell_size)
    acc = np.zeros(3, dtype=np.float64)
    for n in normals:
        vec = np.asarray(n, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(vec))
        if nrm < 1e-14:
            continue
        unit = vec / nrm
        axis = int(np.argmax(np.abs(unit)))
        acc += unit * float(cell_xyz[axis])
    tot = float(np.linalg.norm(acc))
    if tot < 1e-14:
        return np.zeros(3, dtype=np.float64)
    return acc / tot


def apply_face_normal_surface_project(
    nodes: np.ndarray,
    unlocked_mask: np.ndarray,
    ownership_normals: dict[int, list[np.ndarray]],
    cad_mesh: trimesh.Trimesh,
    *,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    multi_face: str = "max_travel",
) -> tuple[np.ndarray, UnlockReport]:
    """
    Project unlocked nodes along owning exposed-face normals (Gate morph path).

    ``multi_face``:
      - ``"max_travel"``: per owning face, nearest CAD hit along ± that
        face normal; keep the candidate with **largest** travel (stair side
        over nearby floor). Default, matches
        ``conformal_core.project_points_face_normal_aware``.
      - ``"blend"``: one ray along owning normals weighted by cell axis
        length (12×12×4 → 3×3×1 slope, not a 45° equal-weight diagonal).

    No hit: mandatory closest-point fallback.
    """
    mode = str(multi_face).strip().lower()
    if mode not in ("max_travel", "blend"):
        raise ValueError(
            f"multi_face must be 'max_travel' or 'blend'; got {multi_face!r}"
        )

    from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf

    pts = np.asarray(nodes, dtype=np.float64).copy()
    mask = np.asarray(unlocked_mask, dtype=bool)
    if mask.shape[0] != len(pts):
        raise ValueError("unlocked_mask must align with nodes")

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    max_travel = float(np.max(_cell_xyz(cell_size)))
    report = UnlockReport(
        n_nodes=len(pts),
        n_unlocked=int(np.count_nonzero(mask)),
        n_locked=int(np.count_nonzero(~mask)),
    )
    if not np.any(mask):
        report.notes.append("No unlocked nodes; skipped face-normal projection.")
        return pts, report

    ids = np.flatnonzero(mask)
    origins = pts[ids].copy()
    projected = origins.copy()
    used_ray = np.zeros(len(ids), dtype=bool)
    used_closest = np.zeros(len(ids), dtype=bool)
    n_blended = 0

    query = trimesh.proximity.ProximityQuery(cad)
    closest_all, _, _ = query.on_surface(origins)

    for i, gid in enumerate(ids):
        o = origins[i]
        normals = ownership_normals.get(int(gid), [])
        hit = None
        if mode == "blend" and len(normals) >= 2:
            direction = blend_unit_normals(normals, cell_size=cell_size)
            if float(np.linalg.norm(direction)) > 1e-14:
                hit = _ray_hit_along(cad, o, direction, max_travel=max_travel)
                n_blended += 1
        else:
            face_cands: list[tuple[float, np.ndarray]] = []
            for n in normals:
                cand = _ray_hit_along(cad, o, n, max_travel=max_travel)
                if cand is None:
                    continue
                travel = float(np.linalg.norm(cand - o))
                if travel < 1e-9 or travel > max_travel + 1e-9:
                    continue
                face_cands.append((travel, cand))
            if face_cands:
                hit = face_cands[int(np.argmax([t for t, _ in face_cands]))][1]

        if hit is not None:
            travel = float(np.linalg.norm(hit - o))
            if 1e-9 < travel <= max_travel + 1e-9:
                projected[i] = hit
                used_ray[i] = True
                continue
        projected[i] = closest_all[i]
        used_closest[i] = True

    travels = np.linalg.norm(projected - origins, axis=1)
    report.n_projected_ray = int(np.count_nonzero(used_ray))
    report.n_projected_closest = int(np.count_nonzero(used_closest))
    report.max_travel = float(np.max(travels)) if len(travels) else 0.0
    report.mean_travel = float(np.mean(travels)) if len(travels) else 0.0
    report.notes.append(
        f"face_normal_projected={len(ids)} ray={report.n_projected_ray} "
        f"closest_fallback={report.n_projected_closest} "
        f"multi_face={mode} blended={n_blended}"
    )
    report.notes.append(f"max_travel_gate={max_travel:.6g}")

    pts[ids] = projected
    return pts, report


def identify_inherent_surface_struts(
    struts: np.ndarray,
    surface_mask: np.ndarray,
    *,
    native_surface_struts: np.ndarray | None = None,
) -> tuple[np.ndarray, set[tuple[int, int]], UnlockReport]:
    """
    Native surface struts for the dual.

    Prefer ``native_surface_struts`` from explicit Half diamond tags.
    Coincidence (both endpoints in Surface Set) is intentionally NOT used —
    that falsely traps internal octahedral edges that weld to surface nodes.
    """
    mask = np.asarray(surface_mask, dtype=bool)
    report = UnlockReport(n_surface_nodes=int(np.count_nonzero(mask)))
    if native_surface_struts is not None:
        arr = (
            np.asarray(native_surface_struts, dtype=np.int64).reshape(-1, 2)
            if len(native_surface_struts)
            else np.empty((0, 2), dtype=np.int64)
        )
        edge_set: set[tuple[int, int]] = set()
        clean: list[tuple[int, int]] = []
        for a, b in arr:
            ia, ib = int(a), int(b)
            if ia == ib:
                continue
            e = (ia, ib) if ia < ib else (ib, ia)
            if e in edge_set:
                continue
            edge_set.add(e)
            clean.append(e)
        out = (
            np.asarray(clean, dtype=np.int64)
            if clean
            else np.empty((0, 2), dtype=np.int64)
        )
        report.n_inherent_surface_struts = len(clean)
        report.notes.append(
            f"native_surface_struts={len(clean)} (explicit Half diamond tags)"
        )
        return out, edge_set, report

    # No tags: return empty — do not fall back to coincidence extraction.
    report.n_inherent_surface_struts = 0
    report.notes.append(
        "native_surface_struts=0 (no explicit tags; coincidence extraction disabled)"
    )
    return np.empty((0, 2), dtype=np.int64), set(), report


@dataclass
class StrutSegregation:
    """Strict partition of core + manifold edges for diagnostic exports."""

    internal_struts: np.ndarray
    native_surface_struts: np.ndarray
    manifold_struts: np.ndarray
    complete_surface_dual_struts: np.ndarray


def segregate_surface_struts(
    core_struts: np.ndarray,
    native_surface_struts: np.ndarray,
    manifold_struts: np.ndarray | None = None,
    surface_mask: np.ndarray | None = None,
) -> StrutSegregation:
    """
    Strict strut segregation for diagnostic STLs (Task 15).

    - ``native_surface_struts``: explicitly tagged Half diamond perimeters
    - ``internal_struts``: remaining core edges (including welded Full↔Half
      octahedral edges that only *coincide* with surface nodes)
    - ``complete_surface_dual_struts``: native + newly stitched manifold edges

    ``surface_mask`` is accepted for call-site compatibility but ignored.
    """
    del surface_mask  # coincidence membership is no longer used
    core = (
        np.asarray(core_struts, dtype=np.int64).reshape(-1, 2)
        if core_struts is not None and len(core_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    native_in = (
        np.asarray(native_surface_struts, dtype=np.int64).reshape(-1, 2)
        if native_surface_struts is not None and len(native_surface_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    man = (
        np.asarray(manifold_struts, dtype=np.int64).reshape(-1, 2)
        if manifold_struts is not None and len(manifold_struts)
        else np.empty((0, 2), dtype=np.int64)
    )

    seen_nat: set[tuple[int, int]] = set()
    native: list[tuple[int, int]] = []
    for a, b in native_in:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e not in seen_nat:
            seen_nat.add(e)
            native.append(e)

    internal: list[tuple[int, int]] = []
    seen_int: set[tuple[int, int]] = set()
    for a, b in core:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e in seen_nat:
            continue
        if e not in seen_int:
            seen_int.add(e)
            internal.append(e)

    man_clean: list[tuple[int, int]] = []
    seen_man: set[tuple[int, int]] = set()
    for a, b in man:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e in seen_nat or e in seen_man:
            continue
        seen_man.add(e)
        man_clean.append(e)

    def _arr(edges: list[tuple[int, int]]) -> np.ndarray:
        return (
            np.asarray(edges, dtype=np.int64)
            if edges
            else np.empty((0, 2), dtype=np.int64)
        )

    native_arr = _arr(native)
    man_arr = _arr(man_clean)
    if len(native_arr) and len(man_arr):
        complete = np.unique(
            np.sort(np.vstack((native_arr, man_arr)), axis=1), axis=0
        )
    elif len(man_arr):
        complete = man_arr
    else:
        complete = native_arr

    return StrutSegregation(
        internal_struts=_arr(internal),
        native_surface_struts=native_arr,
        manifold_struts=man_arr,
        complete_surface_dual_struts=complete,
    )


def plot_unwrapped_surface_topology(
    nodes: np.ndarray,
    surface_mask: np.ndarray,
    native_surface_struts: np.ndarray,
    manifold_struts: np.ndarray,
    out_path: str | Path,
    *,
    node_size: float = 10.0,
    edge_alpha: float = 0.5,
    seam_alpha: float = 0.35,
    y_offset_scale: float = 1.5,
    dpi: int = 160,
    figsize: tuple[float, float] = (12.0, 14.0),
) -> Path:
    """
    Clamshell (X, Y) plot of the surface dual — top and bottom shells side-by-side.

    Seam: ``z_mid`` = midpoint of Surface Set Z bounds.
      - Top shell (Z >= z_mid): plot at (X, Y)
      - Bottom shell (Z < z_mid): plot at (X, Y - y_offset)

    Edge buckets:
      - Top / Bottom (both endpoints in same shell): blue=native, red=manifold
      - Seam (crosses z_mid): thin dashed green
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    pts = np.asarray(nodes, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"nodes must be (V, 3); got {pts.shape}")
    mask = np.asarray(surface_mask, dtype=bool)
    if mask.shape[0] != len(pts):
        raise ValueError("surface_mask must align with nodes")
    surface_ids = [int(i) for i in np.flatnonzero(mask)]

    native = (
        np.asarray(native_surface_struts, dtype=np.int64).reshape(-1, 2)
        if native_surface_struts is not None and len(native_surface_struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    man = (
        np.asarray(manifold_struts, dtype=np.int64).reshape(-1, 2)
        if manifold_struts is not None and len(manifold_struts)
        else np.empty((0, 2), dtype=np.int64)
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if not surface_ids:
        ax.set_title("Clamshell surface dual (empty)")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out

    z_vals = pts[surface_ids, 2]
    z_mid = 0.5 * (float(np.min(z_vals)) + float(np.max(z_vals)))
    y_vals = pts[surface_ids, 1]
    y_span = float(np.max(y_vals) - np.min(y_vals))
    if y_span < 1e-9:
        y_span = float(np.max(pts[surface_ids, 0]) - np.min(pts[surface_ids, 0]))
    if y_span < 1e-9:
        y_span = 1.0
    y_offset = float(y_offset_scale) * y_span

    is_top = {gid: bool(pts[gid, 2] >= z_mid) for gid in surface_ids}
    # Top: (X, Y); Bottom: (X, Y - y_offset)
    pos: dict[int, tuple[float, float]] = {}
    for gid in surface_ids:
        x, y = float(pts[gid, 0]), float(pts[gid, 1])
        pos[gid] = (x, y) if is_top[gid] else (x, y - y_offset)

    g = nx.Graph()
    g.add_nodes_from((gid, {"shell": "top" if is_top[gid] else "bottom"}) for gid in surface_ids)

    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    native_set: set[tuple[int, int]] = set()
    for a, b in native:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        native_set.add(_edge_key(ia, ib))

    man_set: set[tuple[int, int]] = set()
    for a, b in man:
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        e = _edge_key(ia, ib)
        if e not in native_set:
            man_set.add(e)

    all_edges = native_set | man_set
    for e in all_edges:
        g.add_edge(*e)

    top_native: list[tuple[int, int]] = []
    top_man: list[tuple[int, int]] = []
    bot_native: list[tuple[int, int]] = []
    bot_man: list[tuple[int, int]] = []
    seam_edges: list[tuple[int, int]] = []

    for a, b in all_edges:
        if a not in is_top or b not in is_top:
            continue
        ta, tb = is_top[a], is_top[b]
        if ta != tb:
            seam_edges.append((a, b))
            continue
        ekey = _edge_key(a, b)
        is_native = ekey in native_set
        if ta:  # both top
            (top_native if is_native else top_man).append(ekey)
        else:
            (bot_native if is_native else bot_man).append(ekey)

    # Draw shells: bottom first so top sits visually "above" in Z-split layout
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[gid for gid in surface_ids if not is_top[gid]],
        ax=ax,
        node_size=float(node_size),
        node_color="#555555",
        linewidths=0,
        alpha=0.75,
        label="bottom shell",
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=[gid for gid in surface_ids if is_top[gid]],
        ax=ax,
        node_size=float(node_size),
        node_color="#222222",
        linewidths=0,
        alpha=0.75,
        label="top shell",
    )

    def _draw_edges(edgelist, color, width, alpha, style="solid"):
        if not edgelist:
            return
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edgelist,
            ax=ax,
            edge_color=color,
            width=width,
            alpha=alpha,
            style=style,
        )

    _draw_edges(bot_native, "#1f77b4", 0.8, float(edge_alpha))
    _draw_edges(bot_man, "#d62728", 1.0, float(edge_alpha))
    _draw_edges(top_native, "#1f77b4", 0.8, float(edge_alpha))
    _draw_edges(top_man, "#d62728", 1.0, float(edge_alpha))
    _draw_edges(seam_edges, "#2ca02c", 0.5, float(seam_alpha), style="dashed")

    n_top = sum(1 for gid in surface_ids if is_top[gid])
    n_bot = len(surface_ids) - n_top
    ax.set_title(
        "Clamshell surface dual (top / bottom split at z_mid)\n"
        f"z_mid={z_mid:.3f}  y_offset={y_offset:.3f}  "
        f"top_nodes={n_top} bottom_nodes={n_bot}  "
        f"seam_edges={len(seam_edges)}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel(f"Y  (bottom shell shifted by −{y_offset:.1f})")

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2, label="native surface"),
        Line2D([0], [0], color="#d62728", lw=2, label="manifold stitch"),
        Line2D(
            [0],
            [0],
            color="#2ca02c",
            lw=1.5,
            ls="--",
            label=f"seam ({len(seam_edges)})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#222222",
            markersize=6,
            label=f"top shell (Z≥z_mid, n={n_top})",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555555",
            markersize=6,
            label=f"bottom shell (Z<z_mid, n={n_bot})",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Step 3 — Manifold unwrapped diagonal stitching
# ---------------------------------------------------------------------------


def _manifold_edge_adjacency(
    exposed: list[_ExposedFace],
) -> tuple[dict[int, set[int]], dict[tuple, list[int]], dict[tuple, list[int]]]:
    """Edge-adjacent exposed faces + corner/edge indexes for diagonal lookup."""
    edge_to_faces: dict[tuple, list[int]] = {}
    corner_to_faces: dict[tuple, list[int]] = {}
    for ef in exposed:
        for e in ef.edges:
            edge_to_faces.setdefault(e, []).append(ef.face_id)
        for c in ef.corners:
            corner_to_faces.setdefault(c, []).append(ef.face_id)

    adj: dict[int, set[int]] = {ef.face_id: set() for ef in exposed}
    for faces in edge_to_faces.values():
        uniq = list(dict.fromkeys(faces))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                adj[uniq[i]].add(uniq[j])
                adj[uniq[j]].add(uniq[i])
    return adj, edge_to_faces, corner_to_faces


def topological_diagonal_face_neighbors(
    face: _ExposedFace,
    edge_adj: dict[int, set[int]],
    corner_to_faces: dict[tuple, list[int]],
) -> list[int]:
    """
    Four diagonal neighbors of ``face`` on the unwrapped exposed-face sheet.

    At each corner of the face, collect exposed faces that share the corner
    but are **not** edge-adjacent to ``face`` (kitty-corner on the manifold).
    """
    edge_n = edge_adj.get(face.face_id, set())
    diags: list[int] = []
    seen: set[int] = set()
    for c in face.corners:
        for other in corner_to_faces.get(c, []):
            if other == face.face_id or other in edge_n or other in seen:
                continue
            seen.add(other)
            diags.append(other)
    return diags


def stitch_manifold_surface_dual(
    nodes: np.ndarray,
    struts: np.ndarray,
    surface_mask: np.ndarray,
    exposed: list[_ExposedFace],
    *,
    surface_edge_set: set[tuple[int, int]] | None = None,
    target_degree: int = _TARGET_SURFACE_DEGREE,
) -> tuple[np.ndarray, np.ndarray, UnlockReport]:
    """
    Manifold diagonal stitch (Task 15).

    For every Full exposed face, take its explicitly tagged surface node
    (face center) and connect it to the explicitly tagged surface nodes of
    its 4 topological diagonal neighbor faces on the unwrapped SC grid.

    Half diamond perimeters are already native; this pass supplies Full
    crosses (X). Coincidence-based native extraction is not used.

    Returns ``(all_struts, manifold_only_struts, report)``.
    """
    del target_degree  # always draw all available diagonal surface links
    pts = np.asarray(nodes, dtype=np.float64)
    mask = np.asarray(surface_mask, dtype=bool)
    base = (
        np.asarray(struts, dtype=np.int64).reshape(-1, 2)
        if struts is not None and len(struts)
        else np.empty((0, 2), dtype=np.int64)
    )
    report = UnlockReport(
        n_nodes=len(pts),
        n_surface_nodes=int(np.count_nonzero(mask)),
        n_exposed_faces=len(exposed),
    )
    empty = np.empty((0, 2), dtype=np.int64)

    existing: set[tuple[int, int]] = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in base
        if int(a) != int(b)
    }
    if surface_edge_set is None:
        surface_edges: set[tuple[int, int]] = set()
    else:
        surface_edges = set(surface_edge_set)

    if not exposed:
        report.notes.append("manifold_stitch skipped (no exposed faces)")
        return base, empty, report

    edge_adj, _edge_ix, corner_ix = _manifold_edge_adjacency(exposed)
    by_id = {ef.face_id: ef for ef in exposed}
    new_edges: list[tuple[int, int]] = []

    for ef in exposed:
        if not ef.is_full or ef.primary_gid is None:
            continue
        gid = int(ef.primary_gid)
        if not mask[gid]:
            continue

        for nbr_id in topological_diagonal_face_neighbors(ef, edge_adj, corner_ix):
            nbr = by_id[nbr_id]
            og = nbr.primary_gid
            if og is None:
                continue
            og = int(og)
            if og == gid or not mask[og]:
                continue
            e = (gid, og) if gid < og else (og, gid)
            if e in existing:
                continue
            existing.add(e)
            surface_edges.add(e)
            new_edges.append(e)

    report.n_manifold_stitches = len(new_edges)
    report.notes.append(
        f"manifold_stitch added={len(new_edges)} "
        f"(Full exposed FC → 4 topological diagonal neighbor surface nodes)"
    )
    if not new_edges:
        return base, empty, report

    manifold_only = np.asarray(new_edges, dtype=np.int64)
    merged = np.vstack((base, manifold_only)) if len(base) else manifold_only
    merged = np.unique(np.sort(merged, axis=1), axis=0)
    return merged, manifold_only, report


# Back-compat alias used by older call sites / tests
def identify_unlocked_surface_nodes(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, UnlockReport]:
    surface, normals, report, _exposed = identify_surface_set(
        nodes, hex_elems, cell_tags, round_decimals=round_decimals
    )
    return surface, normals, report


def _ray_hit_along(
    cad_mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    max_travel: float,
) -> np.ndarray | None:
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-14:
        return None
    d = d / nrm
    o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    best: tuple[float, np.ndarray] | None = None
    for sign in (1.0, -1.0):
        try:
            locs, _ray_id, _tri = cad_mesh.ray.intersects_location(
                ray_origins=o,
                ray_directions=(sign * d).reshape(1, 3),
                multiple_hits=True,
            )
        except Exception:
            continue
        if locs is None or len(locs) == 0:
            continue
        for p in locs:
            t = float(np.linalg.norm(p - o[0]))
            if t < 1e-9 or t > float(max_travel):
                continue
            if best is None or t < best[0]:
                best = (t, np.asarray(p, dtype=np.float64))
    return None if best is None else best[1]


def apply_minor_surface_tnp(
    nodes: np.ndarray,
    unlocked_mask: np.ndarray,
    unlock_normals: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    use_blender_tnp: bool = True,
) -> tuple[np.ndarray, UnlockReport]:
    """
    Project UNLOCKED Surface_Nodes onto CAD; leave LOCKED nodes frozen.

    Task 23 fallback: if the normal ray misses (or Blender TNP fails / exceeds
    the travel gate), every unlocked node **must** fall back to closest-point
    projection. No unlocked node is allowed to skip projection.
    """
    from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf

    pts = np.asarray(nodes, dtype=np.float64).copy()
    mask = np.asarray(unlocked_mask, dtype=bool)
    normals = np.asarray(unlock_normals, dtype=np.float64)
    if mask.shape[0] != len(pts) or normals.shape != pts.shape:
        raise ValueError("unlocked_mask / unlock_normals must align with nodes")

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    cell_xyz = _cell_xyz(cell_size)
    max_travel = float(np.max(cell_xyz))
    report = UnlockReport(
        n_nodes=len(pts),
        n_unlocked=int(np.count_nonzero(mask)),
        n_locked=int(np.count_nonzero(~mask)),
    )
    if not np.any(mask):
        report.notes.append("No unlocked nodes; skipped projection.")
        return pts, report

    ids = np.flatnonzero(mask)
    origins = pts[ids].copy()
    dirs = normals[ids]
    projected = origins.copy()
    used_ray = np.zeros(len(ids), dtype=bool)
    used_closest = np.zeros(len(ids), dtype=bool)

    query = trimesh.proximity.ProximityQuery(cad)
    closest_all, _, _ = query.on_surface(origins)

    # Optional Blender-style TNP on all unlocked nodes first
    tnp_pts = None
    tnp_hit = np.zeros(len(ids), dtype=bool)
    if use_blender_tnp:
        tnp_pts, tnp_hit = target_normal_project(origins, cad)
        report.n_tnp_hit = int(np.count_nonzero(tnp_hit))

    for i, (o, n) in enumerate(zip(origins, dirs)):
        cand = None
        # Prefer TNP hit within travel gate
        if tnp_pts is not None and bool(tnp_hit[i]):
            travel = float(np.linalg.norm(tnp_pts[i] - o))
            if travel <= max_travel + 1e-9:
                cand = tnp_pts[i]
                used_ray[i] = True
        # Else normal raycast
        if cand is None:
            hit = _ray_hit_along(cad, o, n, max_travel=max_travel)
            if hit is not None:
                cand = hit
                used_ray[i] = True
        # Mandatory closest-point fallback — never skip
        if cand is None:
            cand = closest_all[i]
            used_closest[i] = True
        else:
            travel = float(np.linalg.norm(cand - o))
            if travel > max_travel + 1e-9:
                cand = closest_all[i]
                used_ray[i] = False
                used_closest[i] = True
        projected[i] = cand

    travels = np.linalg.norm(projected - origins, axis=1)
    report.n_projected_ray = int(np.count_nonzero(used_ray))
    report.n_projected_closest = int(np.count_nonzero(used_closest))
    if report.n_projected_ray + report.n_projected_closest != len(ids):
        # Any unclassified → force closest (should not happen)
        for i in range(len(ids)):
            if not (used_ray[i] or used_closest[i]):
                projected[i] = closest_all[i]
                used_closest[i] = True
        travels = np.linalg.norm(projected - origins, axis=1)
        report.n_projected_ray = int(np.count_nonzero(used_ray))
        report.n_projected_closest = int(np.count_nonzero(used_closest))

    report.notes.append(
        f"tnp_projected={len(ids)} ray_or_tnp={report.n_projected_ray} "
        f"closest_fallback={report.n_projected_closest}"
    )

    pts[ids] = projected
    report.max_travel = float(np.max(travels)) if len(travels) else 0.0
    report.mean_travel = float(np.mean(travels)) if len(travels) else 0.0
    report.notes.append(f"max_travel_gate={max_travel:.6g}")
    return pts, report


def lock_and_project_quantized_surface(
    nodes: np.ndarray,
    hex_elems: np.ndarray,
    cell_tags: list | np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    struts: np.ndarray | None = None,
    stamp_tags: SurfaceStampTags | None = None,
    round_decimals: int = 6,
    use_blender_tnp: bool = True,
    stitch_manifold: bool = True,
    skip_tnp: bool = False,
    use_contextual_skin: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, UnlockReport, np.ndarray]:
    """
    Contextual surface skin (Task 18) → merge with lattice → TNP.

    When ``use_contextual_skin`` is True (default):
      exposed faces get Full/CUT/SIDE nodes; adjacent-face dual replaces the
      old diagonal manifold stitch; TNP unlocks only those surface nodes.

    Returns ``(nodes, all_struts, surface_mask, report, routed_surface_struts)``.
    """
    from graphite.explicit.sc_contextual_surface import build_contextual_surface_skin

    strut_in = (
        np.asarray(struts, dtype=np.int64).reshape(-1, 2)
        if struts is not None and len(struts)
        else np.empty((0, 2), dtype=np.int64)
    )

    if use_contextual_skin:
        skin = build_contextual_surface_skin(
            hex_elems,
            cell_tags,
            core_nodes=nodes,
            core_struts=strut_in,
            round_decimals=round_decimals,
        )
        nodes_m = skin.nodes
        strut_out = skin.core_struts
        surface = skin.surface_mask
        # Task 23: TNP unlocks only pruned Surface_Nodes (Full/SIDE/CUT)
        unlocked = np.asarray(skin.unlocked_mask, dtype=bool)
        normals = skin.unlock_normals
        routed = skin.routed_surface_struts
        id_report = UnlockReport(
            n_nodes=len(nodes_m),
            n_unlocked=int(np.count_nonzero(unlocked)),
            n_locked=int(np.count_nonzero(~unlocked)),
            n_surface_nodes=int(np.count_nonzero(surface)),
            n_inherent_surface_struts=len(skin.native_surface_struts),
            n_manifold_stitches=len(routed),
            n_exposed_faces=int(skin.report.get("n_exposed_faces", 0)),
            n_full_exposed=int(skin.report.get("n_full", 0)),
            n_half_diamond=int(skin.report.get("n_half_cut", 0)) * 4,
            skin_report=dict(skin.report),
            skin_faces=list(skin.faces),
        )
        id_report.notes.append(
            f"contextual_skin faces={skin.report.get('n_exposed_faces')} "
            f"full={skin.report.get('n_full')} cut={skin.report.get('n_half_cut')} "
            f"side={skin.report.get('n_half_side')} apex={skin.report.get('n_half_apex')} "
            f"native={len(skin.native_surface_struts)} routed={len(routed)} "
            f"unlocked={int(np.count_nonzero(unlocked))} "
            f"locked={int(np.count_nonzero(~unlocked))}"
        )
        # Stash native on report via notes for callers that segregate separately;
        # prefer regenerating from skin_faces / skin_report in the export script.
        id_report.notes.append(
            f"native_surface_struts={len(skin.native_surface_struts)}"
        )
        # Attach native array for segregate convenience
        id_report.skin_report["native_surface_struts"] = skin.native_surface_struts
        id_report.skin_report["routed_surface_struts"] = routed
        # Keep unlock counts in report; full mask stays on ContextualSkinResult.
        if not stitch_manifold:
            # Drop routed dual but keep native CUT diamonds
            routed = np.empty((0, 2), dtype=np.int64)
            id_report.n_manifold_stitches = 0
            id_report.notes.append("adjacent dual routing skipped")
            # Rebuild strut_out without routed edges
            native_set = {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in skin.native_surface_struts
            }
            core_only = {
                (min(int(a), int(b)), max(int(a), int(b)))
                for a, b in strut_in
                if int(a) != int(b)
            } | native_set
            strut_out = (
                np.array(sorted(core_only), dtype=np.int64)
                if core_only
                else np.empty((0, 2), dtype=np.int64)
            )

        if skip_tnp:
            id_report.notes.append("TNP skipped")
            return (
                np.asarray(nodes_m, dtype=np.float64).copy(),
                strut_out,
                surface,
                id_report,
                routed,
            )

        new_nodes, proj_report = apply_minor_surface_tnp(
            nodes_m,
            unlocked,
            normals,
            cad_mesh,
            cell_size=cell_size,
            use_blender_tnp=use_blender_tnp,
        )
        id_report.n_projected_ray = proj_report.n_projected_ray
        id_report.n_projected_closest = proj_report.n_projected_closest
        id_report.n_tnp_hit = proj_report.n_tnp_hit
        id_report.max_travel = proj_report.max_travel
        id_report.mean_travel = proj_report.mean_travel
        id_report.notes.extend(proj_report.notes)
        return new_nodes, strut_out, surface, id_report, routed

    # --- Legacy Task-15 path (explicit stamp tags + diagonal stitch) --------
    surface, normals, id_report, exposed = identify_surface_set(
        nodes,
        hex_elems,
        cell_tags,
        round_decimals=round_decimals,
        stamp_tags=stamp_tags,
    )

    native_in = (
        stamp_tags.native_surface_struts if stamp_tags is not None else None
    )
    _inherent, surface_edges, inh_report = identify_inherent_surface_struts(
        strut_in, surface, native_surface_struts=native_in
    )
    id_report.n_inherent_surface_struts = inh_report.n_inherent_surface_struts
    id_report.notes.extend(inh_report.notes)

    manifold_only = np.empty((0, 2), dtype=np.int64)
    if stitch_manifold:
        strut_out, manifold_only, man_report = stitch_manifold_surface_dual(
            nodes,
            strut_in,
            surface,
            exposed,
            surface_edge_set=surface_edges,
        )
        id_report.n_manifold_stitches = man_report.n_manifold_stitches
        id_report.notes.extend(man_report.notes)
    else:
        strut_out = strut_in
        id_report.notes.append("manifold_stitch skipped")

    if skip_tnp:
        id_report.notes.append("TNP skipped")
        return (
            np.asarray(nodes, dtype=np.float64).copy(),
            strut_out,
            surface,
            id_report,
            manifold_only,
        )

    new_nodes, proj_report = apply_minor_surface_tnp(
        nodes,
        surface,
        normals,
        cad_mesh,
        cell_size=cell_size,
        use_blender_tnp=use_blender_tnp,
    )
    id_report.n_projected_ray = proj_report.n_projected_ray
    id_report.n_projected_closest = proj_report.n_projected_closest
    id_report.n_tnp_hit = proj_report.n_tnp_hit
    id_report.max_travel = proj_report.max_travel
    id_report.mean_travel = proj_report.mean_travel
    id_report.notes.extend(proj_report.notes)
    return new_nodes, strut_out, surface, id_report, manifold_only
