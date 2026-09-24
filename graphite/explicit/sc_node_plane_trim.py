"""Node-plane trim for SC hex topologies (Nodal Conformation).

Culling  = drop whole unit cells.
Trimming = drop sub-cell nodes/struts by per-axis node-plane extents.

Grid is cull-only: keep the full stamp if VF >= ``GRID_CULL_VF_MIN`` (0.50),
else drop the hex. Other rules cull empty hexes (VF <= ``empty_vf_max``)
then trim remaining cells to kept node planes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.hex_topology_module import get_hex_topology_rule
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.sc_axis_cull_octahedral import (
    measure_material_extents,
    prune_loose_external_nodes,
)
from graphite.explicit.sc_boundary_states import (
    _trilinear_hex_points,
    _unit_cube_sample_grid,
    build_edt_sdf_field,
    estimate_hex_volume_fractions,
)
from graphite.explicit.sc_node_planes import (
    _UNIT_HEX_CORNERS,
    frac_on_kept_trim_planes,
    incident_box_indices,
    interval_indices_for_frac,
    kept_trim_planes,
    trim_plane_policy,
)

GRID_CULL_VF_MIN = 0.50
_PLANE_ATOL = 1e-5


def _corner_key(pt: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


def node_plane_box_volume_fractions(
    corners: np.ndarray,
    sdf_field,
    planes_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    samples_per_axis: int = 10,
    inside_eps: float = 0.0,
) -> np.ndarray:
    """Per-box CAD volume fraction between consecutive node planes."""
    nx = max(0, int(len(planes_xyz[0]) - 1))
    ny = max(0, int(len(planes_xyz[1]) - 1))
    nz = max(0, int(len(planes_xyz[2]) - 1))
    if nx == 0 or ny == 0 or nz == 0:
        return np.zeros((nx, ny, nz), dtype=np.float64)
    uvw = _unit_cube_sample_grid(int(samples_per_axis))
    pts = _trilinear_hex_points(np.asarray(corners, dtype=np.float64), uvw)
    inside = np.asarray(sdf_field.sample(pts), dtype=np.float64) <= float(inside_eps)
    px, py, pz = planes_xyz
    n_in = np.zeros((nx, ny, nz), dtype=np.float64)
    n_all = np.zeros((nx, ny, nz), dtype=np.float64)
    for (u, v, w), is_in in zip(uvw, inside):
        ixs = interval_indices_for_frac(float(u), px)
        iys = interval_indices_for_frac(float(v), py)
        izs = interval_indices_for_frac(float(w), pz)
        for i in ixs:
            for j in iys:
                for k in izs:
                    n_all[int(i), int(j), int(k)] += 1.0
                    if bool(is_in):
                        n_in[int(i), int(j), int(k)] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vf = np.divide(n_in, n_all, out=np.zeros_like(n_in), where=n_all > 0)
    return vf


def occupied_node_plane_boxes(
    corners: np.ndarray,
    sdf_field,
    planes_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    samples_per_axis: int = 10,
    inside_eps: float = 0.0,
    box_vf_min: float = 0.0,
) -> np.ndarray:
    """Bool occupancy of the rectangular cells between consecutive node planes.

    ``box_vf_min <= 0`` keeps the old any-sample rule. ``box_vf_min > 0``
    requires that box's sampled VF to meet the threshold (sliver octants drop,
    which turns a diagonal clip into a stair / three-quarter).
    """
    vf = node_plane_box_volume_fractions(
        corners,
        sdf_field,
        planes_xyz,
        samples_per_axis=samples_per_axis,
        inside_eps=inside_eps,
    )
    if float(box_vf_min) <= 0.0:
        return vf > 0.0
    return vf >= float(box_vf_min)


def _node_has_occupied_box(
    uvw: np.ndarray,
    occ: np.ndarray,
    planes_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    if not np.any(occ):
        return False
    for i, j, k in incident_box_indices(uvw, planes_xyz, atol=_PLANE_ATOL):
        if 0 <= i < occ.shape[0] and 0 <= j < occ.shape[1] and 0 <= k < occ.shape[2]:
            if bool(occ[i, j, k]):
                return True
    return False


def _remap_kept_graph(
    world_nodes: np.ndarray,
    world_struts: np.ndarray,
    keep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(keep):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
        )
    old_to_new = -np.ones(len(world_nodes), dtype=np.int64)
    kept_idx = np.flatnonzero(keep)
    old_to_new[kept_idx] = np.arange(len(kept_idx))
    kept_nodes = world_nodes[kept_idx]
    out_struts: list[tuple[int, int]] = []
    for a, b in np.asarray(world_struts, dtype=np.int64).reshape(-1, 2):
        ia, ib = int(old_to_new[int(a)]), int(old_to_new[int(b)])
        if ia < 0 or ib < 0 or ia == ib:
            continue
        out_struts.append((ia, ib) if ia < ib else (ib, ia))
    strut_arr = (
        np.array(sorted(set(out_struts)), dtype=np.int64)
        if out_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return kept_nodes, strut_arr


def interval_from_extents(extents) -> tuple[np.ndarray, np.ndarray]:
    """AABB [lo, hi] in unit-cell UVW from ``CellMaterialExtents``."""
    lo = np.array(
        [1.0 - extents.neg_x, 1.0 - extents.neg_y, 1.0 - extents.neg_z],
        dtype=np.float64,
    )
    hi = np.array(
        [extents.pos_x, extents.pos_y, extents.pos_z],
        dtype=np.float64,
    )
    return lo, hi


def hex_span(corners: np.ndarray) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float64)
    return c.max(axis=0) - c.min(axis=0)


def _plane_kept(kept: np.ndarray, frac: float, *, atol: float = 1e-5) -> bool:
    k = np.asarray(kept, dtype=np.float64).reshape(-1)
    if k.size == 0:
        return False
    return bool(np.any(np.abs(k - float(frac)) <= atol))


def _hex_is_active(kept_xyz: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    return any(np.asarray(p).size > 0 for p in kept_xyz)


def _axis_half_occupancy(kept: np.ndarray) -> tuple[bool, bool]:
    """Whether local [0, 0.5] and [0.5, 1] contain solid along one axis."""
    k = np.asarray(kept, dtype=np.float64).reshape(-1)
    if k.size == 0:
        return False, False
    lo = _plane_kept(k, 0.0)
    mid = _plane_kept(k, 0.5)
    hi = _plane_kept(k, 1.0)
    return (lo and mid) or (lo and hi), (mid and hi) or (lo and hi)


def _hex_filled_octants(
    corners: np.ndarray,
    kept_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[set[tuple[int, int, int]], np.ndarray]:
    corners = np.asarray(corners, dtype=np.float64)
    filled: set[tuple[int, int, int]] = set()
    span = hex_span(corners)
    half = 0.5 * span
    if not _hex_is_active(kept_xyz):
        return filled, half
    origin = corners.min(axis=0)
    base = np.round(origin / np.maximum(half, 1e-15)).astype(np.int64)
    hx = [_axis_half_occupancy(kept_xyz[a]) for a in range(3)]
    for da in (0, 1):
        if not hx[0][da]:
            continue
        for db in (0, 1):
            if not hx[1][db]:
                continue
            for dc in (0, 1):
                if not hx[2][dc]:
                    continue
                filled.add((int(base[0] + da), int(base[1] + db), int(base[2] + dc)))
    return filled, half


def hex_occupies_cad(
    corners: np.ndarray,
    kept_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    sdf_field,
    *,
    inside_eps: float = 0.0,
) -> bool:
    """True if at least one kept octant center is inside the CAD (SDF <= eps).

    SDF bleed can give a ghost hex a few percent VF and kept planes even when
    every occupied octant sits outside the part. Those ghosts share faces with
    real cells and steal outer supports.
    """
    filled, half = _hex_filled_octants(corners, kept_xyz)
    if not filled:
        return False
    h = np.asarray(half, dtype=np.float64).reshape(3)
    pts = np.array(
        [(np.asarray(ijk, dtype=np.float64) + 0.5) * h for ijk in filled],
        dtype=np.float64,
    )
    sdf = np.asarray(sdf_field.sample(pts), dtype=np.float64)
    return bool(np.any(sdf <= float(inside_eps)))


def compute_kept_planes_per_hex(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    rule_name: str,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    grid_cull_vf_min: float = GRID_CULL_VF_MIN,
    sdf_field=None,
    inside_eps: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-hex kept trim planes (empty tuple of arrays if the hex is dropped)."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    rule = str(rule_name).strip().lower()
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    vf = estimate_hex_volume_fractions(
        cad, elems, samples_per_axis=int(samples_per_axis), sdf_field=sdf_field
    )
    if sdf_field is None:
        res = float(vf.voxel_resolution) if vf.voxel_resolution > 0 else 0.5
        sdf_field = build_edt_sdf_field(cad, res)
    fracs = np.asarray(vf.volume_fractions, dtype=np.float64)
    empty = (np.zeros(0), np.zeros(0), np.zeros(0))
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    full01 = np.array([0.0, 1.0], dtype=np.float64)
    for hi in range(len(elems)):
        if rule == "grid":
            if float(fracs[hi]) >= float(grid_cull_vf_min):
                out.append((full01.copy(), full01.copy(), full01.copy()))
            else:
                out.append(empty)
            continue
        if float(fracs[hi]) <= float(empty_vf_max):
            out.append(empty)
            continue
        ext = measure_material_extents(
            elems[hi],
            sdf_field,
            samples_per_axis=int(extent_samples_per_axis),
            inside_eps=inside_eps,
        )
        if not ext.has_material:
            out.append(empty)
            continue
        lo, hi_uvw = interval_from_extents(ext)
        kept = tuple(kept_trim_planes(float(lo[a]), float(hi_uvw[a]), rule) for a in range(3))
        if not hex_occupies_cad(elems[hi], kept, sdf_field, inside_eps=inside_eps):
            out.append(empty)
            continue
        out.append(kept)
    return out


def trim_stamped_cell(
    corners: np.ndarray,
    sdf_field,
    *,
    rule_name: str,
    samples_per_axis: int = 10,
    inside_eps: float = 0.0,
    box_occupancy: bool | None = None,
    box_vf_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Stamp ``rule_name`` into one hex and drop nodes off kept node planes.

    Non-grid rules also require at least one incident node-plane *box* to
    contain CAD (diagonal / far-corner trim). Pass ``box_occupancy`` to
    override; default is on for every rule except grid.
    ``box_vf_min`` raises occupancy above a sliver (0 = any sample).

    Returns (kept_nodes, kept_struts, info). Empty arrays if the cell
    has no material or no remaining graph.
    """
    rule = str(rule_name).strip().lower()
    use_boxes = bool(box_occupancy) if box_occupancy is not None else (rule != "grid")
    rule_obj = get_hex_topology_rule(rule)
    trim_pol = trim_plane_policy(rule)
    c = np.asarray(corners, dtype=np.float64)
    unit_nodes, unit_struts = rule_obj.builder(_UNIT_HEX_CORNERS.copy())
    world_nodes, world_struts = rule_obj.builder(c)
    unit_nodes = np.asarray(unit_nodes, dtype=np.float64).reshape(-1, 3)
    world_nodes = np.asarray(world_nodes, dtype=np.float64).reshape(-1, 3)
    unit_struts = np.asarray(unit_struts, dtype=np.int64).reshape(-1, 2)
    world_struts = np.asarray(world_struts, dtype=np.int64).reshape(-1, 2)
    if len(unit_nodes) != len(world_nodes):
        raise RuntimeError(f"{rule}: unit/world stamp node count mismatch")
    edges = world_struts if len(world_struts) else unit_struts

    extents = measure_material_extents(
        c, sdf_field, samples_per_axis=int(samples_per_axis), inside_eps=inside_eps
    )
    info = {
        "n_stamped": int(len(world_nodes)),
        "n_kept": 0,
        "has_material": bool(extents.has_material),
        "box_occupancy": use_boxes,
        "box_vf_min": float(box_vf_min),
    }
    if not extents.has_material:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            info,
        )

    lo, hi = interval_from_extents(extents)
    kept_axis = [
        kept_trim_planes(float(lo[a]), float(hi[a]), rule) for a in range(3)
    ]
    if not hex_occupies_cad(c, tuple(kept_axis), sdf_field, inside_eps=inside_eps):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            info,
        )
    occ = None
    if use_boxes:
        occ = occupied_node_plane_boxes(
            c,
            sdf_field,
            trim_pol.planes_xyz,
            samples_per_axis=int(samples_per_axis),
            inside_eps=inside_eps,
            box_vf_min=float(box_vf_min),
        )
        info["n_boxes_occupied"] = int(np.count_nonzero(occ))

    keep = np.ones(len(unit_nodes), dtype=bool)
    for i, uvw in enumerate(unit_nodes):
        on_planes = all(
            frac_on_kept_trim_planes(float(uvw[a]), kept_axis[a], rule)
            for a in range(3)
        )
        if not on_planes:
            keep[i] = False
            continue
        if use_boxes and occ is not None:
            keep[i] = _node_has_occupied_box(uvw, occ, trim_pol.planes_xyz)

    info["n_kept"] = int(np.count_nonzero(keep))
    kept_nodes, strut_arr = _remap_kept_graph(world_nodes, edges, keep)
    return kept_nodes, strut_arr, info


@dataclass
class NodePlaneTrimReport:
    rule_name: str
    n_hex_background: int = 0
    n_hex_candidate: int = 0
    n_hex_kept: int = 0
    n_nodes_stamped: int = 0
    n_nodes_kept: int = 0
    n_struts_kept: int = 0
    n_loose_external_removed: int = 0
    box_vf_min: float = 0.0
    empty_vf_max: float = 0.01
    grid_cull_vf_min: float = GRID_CULL_VF_MIN
    mode: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "mode": self.mode,
            "n_hex_background": self.n_hex_background,
            "n_hex_candidate": self.n_hex_candidate,
            "n_hex_kept": self.n_hex_kept,
            "n_nodes_stamped": self.n_nodes_stamped,
            "n_nodes_kept": self.n_nodes_kept,
            "n_struts_kept": self.n_struts_kept,
            "n_loose_external_removed": self.n_loose_external_removed,
            "box_vf_min": self.box_vf_min,
            "empty_vf_max": self.empty_vf_max,
            "grid_cull_vf_min": self.grid_cull_vf_min,
            "notes": list(self.notes),
        }


def generate_node_plane_trimmed_lattice(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    rule_name: str,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    round_decimals: int = 6,
    sdf_field=None,
    inside_eps: float = 0.0,
    grid_cull_vf_min: float = GRID_CULL_VF_MIN,
    prune_loose_external: bool = False,
    box_vf_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, NodePlaneTrimReport]:
    """
    Overbuild hexes → cull empty cells → trim (or keep-full for grid)
    → optional loose-external prune on the welded graph.

    Grid: whole-cell cull at ``grid_cull_vf_min``; no sub-cell trim.
    Other rules: VF > ``empty_vf_max`` then node-plane trim AND box occupancy.

    Loose-external prune (off by default so the octahedral gold engine is
    unchanged): drop outside nodes whose remaining neighbors are all outside.
    Keep an outside node if it has at least one inside neighbor.
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must have shape (N, 8, 3); got {elems.shape}")

    rule = str(rule_name).strip().lower()
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    report = NodePlaneTrimReport(
        rule_name=rule,
        n_hex_background=len(elems),
        empty_vf_max=float(empty_vf_max),
        grid_cull_vf_min=float(grid_cull_vf_min),
        box_vf_min=float(box_vf_min),
    )

    vf = estimate_hex_volume_fractions(
        cad,
        elems,
        samples_per_axis=int(samples_per_axis),
        sdf_field=sdf_field,
        inside_eps=inside_eps,
    )
    fracs = np.asarray(vf.volume_fractions, dtype=np.float64)

    if False: # rule == "grid":
        candidate = np.flatnonzero(fracs >= float(grid_cull_vf_min))
        report.mode = f"cull_vf_ge_{float(grid_cull_vf_min):g}_full_stamp"
        report.notes.append(
            f"grid cull-only: keep full cell if VF >= {grid_cull_vf_min:g}"
        )
    else:
        candidate = np.flatnonzero(fracs > float(empty_vf_max))
        report.mode = "cull_empty_then_node_plane_trim_and_box_occupancy"
        report.notes.append(
            f"empty cull VF <= {empty_vf_max:g}; trim by node-plane mid "
            "thresholds AND incident box occupancy (diagonal far-corner)"
            + (
                f"; box VF >= {box_vf_min:g}"
                if float(box_vf_min) > 0.0
                else ""
            )
        )

    report.n_hex_candidate = int(len(candidate))
    if len(candidate) == 0:
        report.notes.append("no hexes survived cull")
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            report,
        )

    if sdf_field is None:
        res = float(vf.voxel_resolution) if vf.voxel_resolution > 0 else 0.5
        sdf_field = build_edt_sdf_field(cad, res)

    rule_obj = get_hex_topology_rule(rule)
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    n_stamped = 0
    n_hex_kept = 0

    def weld(p: np.ndarray) -> int:
        key = _corner_key(p, round_decimals)
        gid = node_map.get(key)
        if gid is None:
            gid = len(nodes_list)
            node_map[key] = gid
            nodes_list.append(np.asarray(p, dtype=np.float64).copy())
        return gid

    for hi in candidate:
        corners = elems[int(hi)]
        if rule == "grid":
            local_nodes, local_struts = rule_obj.builder(corners)
            local_nodes = np.asarray(local_nodes, dtype=np.float64).reshape(-1, 3)
            local_struts = np.asarray(local_struts, dtype=np.int64).reshape(-1, 2)
            info = {"n_stamped": int(len(local_nodes)), "n_kept": int(len(local_nodes))}
        else:
            local_nodes, local_struts, info = trim_stamped_cell(
                corners,
                sdf_field,
                rule_name=rule,
                samples_per_axis=int(extent_samples_per_axis),
                inside_eps=inside_eps,
                box_vf_min=float(box_vf_min),
            )
        n_stamped += int(info["n_stamped"])
        if len(local_nodes) == 0:
            continue
        n_hex_kept += 1
        local_to_global = [weld(p) for p in local_nodes]
        for a, b in np.asarray(local_struts, dtype=np.int64).reshape(-1, 2):
            ga, gb = local_to_global[int(a)], local_to_global[int(b)]
            if ga == gb:
                continue
            strut_set.add((ga, gb) if ga < gb else (gb, ga))

    report.n_hex_kept = n_hex_kept
    report.n_nodes_stamped = n_stamped
    report.n_nodes_kept = len(nodes_list)
    report.n_struts_kept = len(strut_set)

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
        report.n_nodes_kept = int(len(nodes))
        report.n_struts_kept = int(len(struts))
        report.notes.append(
            f"loose-external prune removed {n_rm} nodes "
            "(outside kept only if tied to an inside neighbor)"
        )

    return nodes, struts, report
