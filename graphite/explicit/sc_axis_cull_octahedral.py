"""
Independent-axis octahedral cull via CAD material extent (0.5 unit rule).

Strict octahedron only (6 face centers, 12 adjacent-face edges). No SDF
node-tolerance band, no center hub, no Half_* dictionary.

For each SC hex, sample CAD material inside the cell, form its local AABB in
unit-cell coordinates, and keep/cull each face-center node by how far solid
penetrates that axis direction relative to the 0.5 mid-plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.hex_rules import (
    _ADJACENT_FACE_PAIRS,
    apply_hex_octahedral,
)
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.sc_boundary_states import (
    _trilinear_hex_points,
    _unit_cube_sample_grid,
    build_edt_sdf_field,
    estimate_hex_volume_fractions,
)

# Face-center local indices (apply_hex_octahedral / _HEX_FACES order)
_FACE_NEG_Z = 0
_FACE_POS_Z = 1
_FACE_NEG_Y = 2
_FACE_POS_Y = 3
_FACE_NEG_X = 4
_FACE_POS_X = 5

HALF_EXTENT_MAX = 0.5  # unit-cell mid-plane threshold (legacy / current)

# Tiered per-axis policy: <25% discard direction, 25–75% half, >75% full
TIERED_EMPTY_MAX = 0.25
TIERED_FULL_MIN = 0.75

# Policies for generate_extent_trimmed_octahedral
EXTENT_POLICY_MIDPLANE_05 = "midplane_05"
EXTENT_POLICY_TIERED_25_75 = "tiered_25_75"


def _corner_key(pt: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


@dataclass(frozen=True)
class CellMaterialExtents:
    """
    Material penetration depths in unit-cell coordinates [0, 1]^3.

    Positive-axis extents are measured from the local origin (min corner)
    to the solid AABB max. Negative-axis extents are measured from the
    opposite max face back toward the origin (1 - solid_min).
    """

    pos_x: float
    neg_x: float
    pos_y: float
    neg_y: float
    pos_z: float
    neg_z: float
    has_material: bool

    @staticmethod
    def empty() -> "CellMaterialExtents":
        return CellMaterialExtents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)


@dataclass
class ExtentCullReport:
    n_hex_background: int = 0
    n_hex_candidate: int = 0
    n_hex_kept: int = 0
    n_nodes_stamped: int = 0
    n_nodes_kept: int = 0
    n_nodes_culled: int = 0
    n_struts_stamped: int = 0
    n_struts_kept: int = 0
    half_extent_max: float = HALF_EXTENT_MAX
    empty_vf_max: float = 0.01
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05
    n_loose_external_removed: int = 0
    notes: list[str] = field(default_factory=list)


def measure_material_extents(
    corners: np.ndarray,
    sdf_field,
    *,
    samples_per_axis: int = 10,
    inside_eps: float = 0.0,
) -> CellMaterialExtents:
    """
    Bounding extent of CAD solid inside one hex, in unit-cell coordinates.

    Samples a parametric grid in the cell; points with SDF <= ``inside_eps``
    define the solid set. Extents are the AABB of that set mapped to [0,1]^3.
    """
    c = np.asarray(corners, dtype=np.float64)
    uvw = _unit_cube_sample_grid(int(samples_per_axis))
    pts = _trilinear_hex_points(c, uvw)
    sdf = np.asarray(sdf_field.sample(pts), dtype=np.float64)
    inside = sdf <= float(inside_eps)
    if not np.any(inside):
        return CellMaterialExtents.empty()

    solid_uvw = uvw[inside]
    u0, v0, w0 = solid_uvw.min(axis=0)
    u1, v1, w1 = solid_uvw.max(axis=0)
    return CellMaterialExtents(
        pos_x=float(u1),
        neg_x=float(1.0 - u0),
        pos_y=float(v1),
        neg_y=float(1.0 - v0),
        pos_z=float(w1),
        neg_z=float(1.0 - w0),
        has_material=True,
    )


def keep_outer_node(extent: float, *, half_max: float = HALF_EXTENT_MAX) -> bool:
    """
    0.5 unit-threshold rule for one axis direction.

    - extent == 0     → no material → cull outer node
    - 0 < extent ≤ 0.5 → Half → cull outer (mid-plane nodes kept by other axes)
    - extent > 0.5    → Full → keep outer face-center node
    """
    e = float(extent)
    if e <= 0.0:
        return False
    if e <= float(half_max):
        return False
    return True


def keep_outer_node_tiered(
    extent: float,
    *,
    empty_max: float = TIERED_EMPTY_MAX,
    full_min: float = TIERED_FULL_MIN,
) -> bool:
    """
    Tiered per-axis rule:

    - extent < 0.25  → throw out (cull outer)
    - 0.25 ≤ extent ≤ 0.75 → half (cull outer; mid-plane via other axes)
    - extent > 0.75  → full (keep outer)
    """
    e = float(extent)
    if e < float(empty_max):
        return False
    if e <= float(full_min):
        return False
    return True


def face_keep_mask_from_extents(
    extents: CellMaterialExtents,
    *,
    half_max: float = HALF_EXTENT_MAX,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    tiered_empty_max: float = TIERED_EMPTY_MAX,
    tiered_full_min: float = TIERED_FULL_MIN,
) -> np.ndarray:
    """Bool mask length-6 for octahedral face centers (−Z,+Z,−Y,+Y,−X,+X)."""
    if not extents.has_material:
        return np.zeros(6, dtype=bool)

    vals = (
        extents.neg_z,
        extents.pos_z,
        extents.neg_y,
        extents.pos_y,
        extents.neg_x,
        extents.pos_x,
    )
    keep = np.zeros(6, dtype=bool)
    policy = str(extent_policy)
    if policy == EXTENT_POLICY_TIERED_25_75:
        # Entire cell thrown out if no direction reaches the half band
        if max(float(v) for v in vals) < float(tiered_empty_max):
            return keep
        for i, e in enumerate(vals):
            keep[i] = keep_outer_node_tiered(
                e, empty_max=tiered_empty_max, full_min=tiered_full_min
            )
        return keep

    for i, e in enumerate(vals):
        keep[i] = keep_outer_node(e, half_max=half_max)
    return keep


def cull_octahedral_cell_by_extent(
    corners: np.ndarray,
    sdf_field,
    *,
    samples_per_axis: int = 10,
    half_max: float = HALF_EXTENT_MAX,
    inside_eps: float = 0.0,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    tiered_empty_max: float = TIERED_EMPTY_MAX,
    tiered_full_min: float = TIERED_FULL_MIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CellMaterialExtents]:
    """
    Stamp a strict octahedron and cull face nodes by per-axis material extent.

    Returns kept_nodes, kept_struts (local), keep_mask (6,), extents.
    """
    local_nodes, local_struts = apply_hex_octahedral(corners)
    extents = measure_material_extents(
        corners,
        sdf_field,
        samples_per_axis=samples_per_axis,
        inside_eps=inside_eps,
    )
    keep = face_keep_mask_from_extents(
        extents,
        half_max=half_max,
        extent_policy=extent_policy,
        tiered_empty_max=tiered_empty_max,
        tiered_full_min=tiered_full_min,
    )
    if not np.any(keep):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            keep,
            extents,
        )

    old_to_new = -np.ones(6, dtype=np.int64)
    kept_idx = np.flatnonzero(keep)
    old_to_new[kept_idx] = np.arange(len(kept_idx))
    kept_nodes = np.asarray(local_nodes[kept_idx], dtype=np.float64)

    kept_struts: list[tuple[int, int]] = []
    for a, b in np.asarray(local_struts, dtype=np.int64):
        ia, ib = int(old_to_new[int(a)]), int(old_to_new[int(b)])
        if ia < 0 or ib < 0 or ia == ib:
            continue
        kept_struts.append((ia, ib) if ia < ib else (ib, ia))

    strut_arr = (
        np.array(sorted(set(kept_struts)), dtype=np.int64)
        if kept_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return kept_nodes, strut_arr, keep, extents


def prune_loose_external_nodes(
    nodes: np.ndarray,
    struts: np.ndarray,
    sdf_field,
    *,
    inside_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Remove external nodes that are only connected to other external nodes.

    Keep an external node iff it has at least one strut neighbor inside CAD.
    Inside nodes are always kept. Iterates until stable.
    """
    pts = np.asarray(nodes, dtype=np.float64).reshape(-1, 3)
    st = np.asarray(struts, dtype=np.int64).reshape(-1, 2)
    if len(pts) == 0:
        return pts, st, 0

    inside = np.asarray(sdf_field.sample(pts), dtype=np.float64) <= float(inside_eps)
    alive = np.ones(len(pts), dtype=bool)
    removed = 0

    while True:
        nbrs: list[set[int]] = [set() for _ in range(len(pts))]
        for a, b in st:
            ia, ib = int(a), int(b)
            if not alive[ia] or not alive[ib]:
                continue
            nbrs[ia].add(ib)
            nbrs[ib].add(ia)

        drop: list[int] = []
        for i in range(len(pts)):
            if not alive[i] or bool(inside[i]):
                continue
            # External: keep only if any alive neighbor is inside
            if any(bool(inside[j]) for j in nbrs[i]):
                continue
            drop.append(i)

        if not drop:
            break
        for i in drop:
            alive[i] = False
            removed += 1

    old_to_new = -np.ones(len(pts), dtype=np.int64)
    kept_idx = np.flatnonzero(alive)
    old_to_new[kept_idx] = np.arange(len(kept_idx))
    new_nodes = pts[kept_idx]
    new_struts: list[tuple[int, int]] = []
    for a, b in st:
        ia, ib = int(old_to_new[int(a)]), int(old_to_new[int(b)])
        if ia < 0 or ib < 0 or ia == ib:
            continue
        new_struts.append((ia, ib) if ia < ib else (ib, ia))
    strut_arr = (
        np.array(sorted(set(new_struts)), dtype=np.int64)
        if new_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return new_nodes, strut_arr, int(removed)


def generate_extent_trimmed_octahedral(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    resolution: float | None = None,
    half_max: float = HALF_EXTENT_MAX,
    round_decimals: int = 6,
    sdf_field=None,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    tiered_empty_max: float = TIERED_EMPTY_MAX,
    tiered_full_min: float = TIERED_FULL_MIN,
    prune_loose_external: bool = False,
    inside_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, ExtentCullReport]:
    """
    Extent-rule trimmed strict octahedral core (no Half dictionary, no TNP).

    1. Candidate cells: VF > ``empty_vf_max``.
    2. Per cell: measure solid AABB extents in unit coordinates.
    3. Per axis: apply ``extent_policy`` (midplane 0.5 or tiered 25/75).
    4. Weld surviving nodes/struts across cells.
    5. Optional: prune loose external nodes (only tied to other externals).
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must have shape (N, 8, 3); got {elems.shape}")

    policy = str(extent_policy)
    if policy not in (EXTENT_POLICY_MIDPLANE_05, EXTENT_POLICY_TIERED_25_75):
        raise ValueError(f"unknown extent_policy {extent_policy!r}")

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    report = ExtentCullReport(
        n_hex_background=len(elems),
        empty_vf_max=float(empty_vf_max),
        half_extent_max=float(half_max),
        extent_policy=policy,
    )

    vf = estimate_hex_volume_fractions(
        cad,
        elems,
        samples_per_axis=int(samples_per_axis),
        resolution=resolution,
        sdf_field=sdf_field,
        inside_eps=inside_eps,
    )
    candidate = np.flatnonzero(vf.volume_fractions > float(empty_vf_max))
    report.n_hex_candidate = int(len(candidate))
    if len(candidate) == 0:
        report.notes.append("no hexes above empty_vf_max")
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            report,
        )

    if sdf_field is None:
        res = float(vf.voxel_resolution) if vf.voxel_resolution > 0 else 0.5
        sdf_field = build_edt_sdf_field(cad, res)

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    n_stamped_nodes = n_stamped_struts = 0
    n_kept_nodes = n_culled = 0
    n_hex_kept = 0

    for hi in candidate:
        local_nodes, local_struts, keep, _ext = cull_octahedral_cell_by_extent(
            elems[int(hi)],
            sdf_field,
            samples_per_axis=int(extent_samples_per_axis),
            half_max=half_max,
            inside_eps=inside_eps,
            extent_policy=policy,
            tiered_empty_max=tiered_empty_max,
            tiered_full_min=tiered_full_min,
        )
        n_stamped_nodes += 6
        n_stamped_struts += len(_ADJACENT_FACE_PAIRS)
        n_kept_nodes += int(np.count_nonzero(keep))
        n_culled += int(np.count_nonzero(~keep))
        if len(local_nodes) == 0:
            continue
        n_hex_kept += 1

        local_to_global: list[int] = []
        for p in local_nodes:
            key = _corner_key(p, round_decimals)
            gid = node_map.get(key)
            if gid is None:
                gid = len(nodes_list)
                node_map[key] = gid
                nodes_list.append(np.asarray(p, dtype=np.float64).copy())
            local_to_global.append(gid)

        for a, b in local_struts:
            ga, gb = local_to_global[int(a)], local_to_global[int(b)]
            if ga == gb:
                continue
            strut_set.add((ga, gb) if ga < gb else (gb, ga))

    report.n_hex_kept = n_hex_kept
    report.n_nodes_stamped = n_stamped_nodes
    report.n_nodes_kept = n_kept_nodes
    report.n_nodes_culled = n_culled
    report.n_struts_stamped = n_stamped_struts
    report.n_struts_kept = len(strut_set)
    if policy == EXTENT_POLICY_TIERED_25_75:
        report.notes.append(
            "strict octahedral; tiered per-axis "
            f"(<{tiered_empty_max:g} throw, "
            f"{tiered_empty_max:g}–{tiered_full_min:g} half, "
            f">{tiered_full_min:g} full)"
        )
    else:
        report.notes.append(
            "strict octahedral; per-axis material extent vs mid-plane "
            f"(keep outer iff extent > {half_max:g})"
        )

    nodes = (
        np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    )
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )

    if prune_loose_external and len(nodes) > 0:
        nodes, struts, n_rm = prune_loose_external_nodes(
            nodes, struts, sdf_field, inside_eps=inside_eps
        )
        report.n_loose_external_removed = int(n_rm)
        report.n_struts_kept = int(len(struts))
        report.notes.append(
            f"loose-external prune removed {n_rm} nodes "
            "(external nodes kept only if tied to an inside neighbor)"
        )

    return nodes, struts, report


# Back-compat alias used by older callers/tests during transition
generate_multi_axis_trimmed_octahedral = generate_extent_trimmed_octahedral
