"""Gold-standard octahedral engine: VF/node-plane trim + shared-edge F dual.

Compare new trim/dual work to this path (locked 19 Aug 2026; cylinder fixture
``outputs/sc_cylinder_50x100_octahedral_temp_engine/``). See
``docs/NODAL_CONFORMATION.md``.

This path is intentionally thin so experimental modules do not bleed in:

  - Volume = ``generate_node_plane_trimmed_lattice`` (VF cull, 25/75 planes,
    box occupancy, ghost-hex CAD occupancy).
  - Dual = octahedral face-center graph on unique-owner exposed faces /
    mid-plane cuts, plus inter-cell stitches across a shared Cartesian edge
    (``route_geometric_face_stitch``). Same-hex opposite-face chords stay
    volume; jump-cut is not dual.

Not included (keep as separate experimental modules):

  - Layered C/E/F role dual (``sc_role_surface_dual``)
  - Node minimization (``node_minimization``)
  - Joint wick / stress deconcentration (``geometry_module``)

Later these can be wired into the explicit production engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.hex_rules import _HEX_FACES
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.proven_topologies import generate_background_grid
from graphite.explicit.sc_node_plane_trim import (
    GRID_CULL_VF_MIN,
    _hex_filled_octants,
    _hex_is_active,
    _plane_kept,
    compute_kept_planes_per_hex,
    generate_node_plane_trimmed_lattice,
    hex_span,
)
from graphite.explicit.sc_quantized_surface import _face_key
from graphite.explicit.sc_simple_fc_surface_dual import (
    axis_aligned_face_quad,
    point_on_quad,
    route_geometric_face_stitch,
)
from graphite.explicit.sc_topological_dual import _edge_midpoint_key

_EDGE_DECIMALS = 5
_FACE_AXIS_MAX = (
    (2, False),
    (2, True),
    (1, False),
    (1, True),
    (0, False),
    (0, True),
)
_OCTAHEDRAL_RULES = frozenset({"octahedral", "hex_face_dual"})


@dataclass
class SharedEdgeFace:
    face_id: int
    hex_i: int
    face_i: int
    is_midplane: bool
    corners_3d: np.ndarray
    edges_3d: tuple[tuple[np.ndarray, np.ndarray], ...]
    node_gids: list[int]


@dataclass
class TrimSharedEdgeResult:
    rule_name: str
    origin_offset: np.ndarray
    hex_elems: np.ndarray
    volume_nodes: np.ndarray
    volume_struts: np.ndarray
    dual_nodes: np.ndarray
    dual_struts: np.ndarray
    surface_gids: set[int]
    faces: list[SharedEdgeFace]
    report: dict = field(default_factory=dict)


def _as_cell_dims(cell_size) -> np.ndarray:
    cell_dims = np.asarray(cell_size, dtype=np.float64)
    if cell_dims.ndim == 0:
        cell_dims = np.full(3, float(cell_dims), dtype=np.float64)
    if cell_dims.shape != (3,) or np.any(cell_dims <= 0.0):
        raise ValueError("cell_size must be a positive scalar or three positive dimensions")
    return cell_dims


def _unique_edges(struts: np.ndarray) -> np.ndarray:
    edges = np.asarray(struts, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = {
        (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        for a, b in edges
        if int(a) != int(b)
    }
    return np.array(sorted(pairs), dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)


def _quad_edges(corners_3d: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    c = np.asarray(corners_3d, dtype=np.float64)
    return tuple((c[a].copy(), c[b].copy()) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)))


def collect_exposed_sc_faces(
    hex_elems: np.ndarray,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    round_decimals: int = 6,
) -> list[tuple[int, int, bool, np.ndarray]]:
    """Unique-owner outer faces, or mid-plane cuts when the outer plane is dropped."""
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

    out: list[tuple[int, int, bool, np.ndarray]] = []
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
            if not is_mid:
                fkey = _face_key(corners, face, round_decimals)
                if len(face_owners.get(fkey, [])) != 1:
                    continue
            quad = axis_aligned_face_quad(corners, int(fi), midplane=is_mid)
            out.append((int(hi), int(fi), bool(is_mid), np.asarray(quad, dtype=np.float64)))
    return out


def build_octahedral_shared_edge_dual(
    hex_elems: np.ndarray,
    volume_nodes: np.ndarray,
    volume_struts: np.ndarray,
    kept_planes_per_hex: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    round_decimals: int = 6,
    edge_decimals: int = _EDGE_DECIMALS,
    plane_eps_frac: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, set[int], list[SharedEdgeFace], dict]:
    """Skin-native octahedral edges plus inter-cell shared-edge F–F stitches."""
    pts = np.asarray(volume_nodes, dtype=np.float64).reshape(-1, 3)
    core_edges = {
        (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        for a, b in _unique_edges(volume_struts)
    }
    elems = np.asarray(hex_elems, dtype=np.float64)

    faces: list[SharedEdgeFace] = []
    for hi, fi, is_mid, quad in collect_exposed_sc_faces(
        elems, kept_planes_per_hex, round_decimals=round_decimals
    ):
        span = hex_span(elems[int(hi)])
        plane_eps = max(float(plane_eps_frac) * float(np.min(span)), 1e-6)
        gids = [i for i, p in enumerate(pts) if point_on_quad(p, quad, plane_eps=plane_eps)]
        if not gids:
            continue
        faces.append(
            SharedEdgeFace(
                face_id=len(faces),
                hex_i=int(hi),
                face_i=int(fi),
                is_midplane=bool(is_mid),
                corners_3d=quad,
                edges_3d=_quad_edges(quad),
                node_gids=gids,
            )
        )

    surface_gids: set[int] = set()
    for f in faces:
        surface_gids.update(int(g) for g in f.node_gids)

    dual_strut_set: set[tuple[int, int]] = set()
    n_from_core = 0
    for a, b in core_edges:
        if a in surface_gids and b in surface_gids:
            dual_strut_set.add((a, b) if a < b else (b, a))
            n_from_core += 1

    edge_map: dict[tuple[float, float, float], list[int]] = {}
    edge_seg: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=edge_decimals)
            edge_map.setdefault(key, []).append(int(f.face_id))
            edge_seg.setdefault(key, (p0, p1))
    for k, ids in list(edge_map.items()):
        edge_map[k] = list(dict.fromkeys(ids))

    by_id = {f.face_id: f for f in faces}
    n_shared = 0
    n_stitches_added = 0
    n_skipped_same_cell = 0
    n_skipped_same_node = 0
    n_already_core = 0
    n_skipped_no_edge_node = 0

    for key, fids in edge_map.items():
        if len(fids) != 2:
            continue
        n_shared += 1
        fa, fb = by_id[fids[0]], by_id[fids[1]]
        if fa.hex_i == fb.hex_i:
            n_skipped_same_cell += 1
            continue
        p0, p1 = edge_seg[key]
        nodes_a = (
            pts[np.asarray(fa.node_gids, dtype=np.int64)]
            if fa.node_gids
            else np.empty((0, 3), dtype=np.float64)
        )
        nodes_b = (
            pts[np.asarray(fb.node_gids, dtype=np.int64)]
            if fb.node_gids
            else np.empty((0, 3), dtype=np.float64)
        )
        local_pairs = route_geometric_face_stitch(nodes_a, nodes_b, p0, p1)
        if not local_pairs:
            n_skipped_no_edge_node += 1
            continue
        for ia, ib in local_pairs:
            ga = int(fa.node_gids[int(ia)])
            gb = int(fb.node_gids[int(ib)])
            if ga == gb:
                n_skipped_same_node += 1
                continue
            pair = (ga, gb) if ga < gb else (gb, ga)
            if pair in dual_strut_set:
                n_already_core += 1
                continue
            dual_strut_set.add(pair)
            n_stitches_added += 1

    used: set[int] = set(surface_gids)
    for a, b in dual_strut_set:
        used.add(a)
        used.add(b)
    old_to_new = {old: i for i, old in enumerate(sorted(used))}
    dual_nodes = (
        np.asarray([pts[old] for old in sorted(used)], dtype=np.float64)
        if used
        else np.empty((0, 3), dtype=np.float64)
    )
    dual_struts = (
        np.array(
            sorted(
                (old_to_new[a], old_to_new[b])
                if old_to_new[a] < old_to_new[b]
                else (old_to_new[b], old_to_new[a])
                for a, b in dual_strut_set
            ),
            dtype=np.int64,
        )
        if dual_strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    report = {
        "n_surface_nodes": int(len(surface_gids)),
        "n_surface_faces": len(faces),
        "n_midplane_faces": sum(1 for f in faces if f.is_midplane),
        "n_outer_faces": sum(1 for f in faces if not f.is_midplane),
        "n_shared_edges": n_shared,
        "n_core_surface_struts": int(n_from_core),
        "n_stitches_added": int(n_stitches_added),
        "n_skipped_same_cell": n_skipped_same_cell,
        "n_skipped_same_node": n_skipped_same_node,
        "n_skipped_no_edge_node": n_skipped_no_edge_node,
        "n_already_core": n_already_core,
        "n_dual_struts": int(len(dual_struts)),
        "n_dual_nodes": int(len(dual_nodes)),
        "rule": "octahedral_shared_edge_f",
    }
    return dual_nodes, dual_struts, surface_gids, faces, report


def generate_octahedral_trim_shared_edge(
    cad_mesh: trimesh.Trimesh,
    cell_size,
    *,
    rule_name: str = "octahedral",
    origin_offset: np.ndarray | tuple[float, float, float] | None = None,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    round_decimals: int = 6,
) -> TrimSharedEdgeResult:
    """
    Temporary generate: node-plane trim + octahedral shared-edge dual.

    ``origin_offset`` is applied as given (default zeros). This function does
    not search node-min phase and does not solidify or wick joints.
    """
    rule = str(rule_name).strip().lower()
    if rule not in _OCTAHEDRAL_RULES:
        raise ValueError(
            f"temporary engine is octahedral shared-edge only; got {rule_name!r}. "
            "Use generate_nodal_conformation / role dual for other lattices."
        )

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    if isinstance(cad, trimesh.Scene):
        cad = trimesh.util.concatenate(tuple(cad.geometry.values()))

    cell_dims = _as_cell_dims(cell_size)
    if origin_offset is None:
        offset = np.zeros(3, dtype=np.float64)
    else:
        offset = np.asarray(origin_offset, dtype=np.float64).reshape(3)

    bounds = np.asarray(cad.bounds, dtype=np.float64)
    gn, cells = generate_background_grid(
        "SC", bounds, cell_dims, origin_offset=offset
    )
    hex_elems = np.asarray(gn[np.asarray(cells, dtype=np.int64)], dtype=np.float64)

    vol_nodes, vol_struts, trim_report = generate_node_plane_trimmed_lattice(
        cad,
        hex_elems,
        rule_name=rule,
        empty_vf_max=float(empty_vf_max),
        samples_per_axis=int(samples_per_axis),
        extent_samples_per_axis=int(extent_samples_per_axis),
        grid_cull_vf_min=GRID_CULL_VF_MIN,
        round_decimals=int(round_decimals),
    )
    vol_struts = _unique_edges(vol_struts)
    if len(vol_nodes) == 0:
        raise RuntimeError(f"{rule}: no nodes survived cull/trim")

    kept_planes = compute_kept_planes_per_hex(
        cad,
        hex_elems,
        rule,
        empty_vf_max=float(empty_vf_max),
        samples_per_axis=int(samples_per_axis),
        extent_samples_per_axis=int(extent_samples_per_axis),
        grid_cull_vf_min=GRID_CULL_VF_MIN,
    )
    dual_nodes, dual_struts, surface_gids, faces, dual_report = (
        build_octahedral_shared_edge_dual(
            hex_elems,
            vol_nodes,
            vol_struts,
            kept_planes,
            round_decimals=int(round_decimals),
        )
    )

    report = {
        "method": "octahedral_trim_shared_edge",
        "temporary": True,
        "rule_name": rule,
        "cell_mm": cell_dims.tolist(),
        "origin_offset_mm": offset.tolist(),
        "node_minimization": False,
        "joint_wick": False,
        "role_dual": False,
        "trim": trim_report.as_dict(),
        "dual": dual_report,
        "n_volume_nodes": int(len(vol_nodes)),
        "n_volume_struts": int(len(vol_struts)),
        "n_dual_nodes": int(len(dual_nodes)),
        "n_dual_struts": int(len(dual_struts)),
        "n_surface_gids": int(len(surface_gids)),
    }
    return TrimSharedEdgeResult(
        rule_name=rule,
        origin_offset=offset,
        hex_elems=hex_elems,
        volume_nodes=vol_nodes,
        volume_struts=vol_struts,
        dual_nodes=dual_nodes,
        dual_struts=dual_struts,
        surface_gids=surface_gids,
        faces=faces,
        report=report,
    )
