"""Nodal Conformation — Cartesian SC + surface dual (project dual only).

Formerly the \"wrist-rest experimental\" path.

Terminology:
  Culling  = drop whole unit cells (VF Full/Empty).
  Trimming = drop sub-cell nodes/struts (halves, quarters, …).

  Grid is cull-only: keep the full stamp if VF >= 0.50, else drop the hex.
  All other SC rules cull empty hexes then node-plane trim AND box occupancy.

1. Node minimization (phase search) — **on by default**
2. Cull empty hexes (grid: VF < 0.50); trim remaining cells by node planes
3. Stamp topology into **undeformed** hexes → volume / core graph
4. Build surface dual / skin on the kept complex
5. Closest-point project **dual nodes only** (core stays Cartesian)
6. Caller solidifies (rectangular dual + cylinders) and Boolean ∩ CAD

This is **not** hex-cage conformal morph (iron + relax of the volume scaffold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh

from graphite.explicit.conformal_core import safe_signed_distance
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.node_minimization import (
    axial_increment,
    axis6_candidate_offsets,
)
from graphite.explicit.proven_topologies import generate_background_grid
from graphite.explicit.sc_boundary_states import build_edt_sdf_field, estimate_hex_volume_fractions
from graphite.explicit.sc_node_planes import node_plane_policy
from graphite.explicit.sc_node_plane_trim import (
    GRID_CULL_VF_MIN,
    generate_node_plane_trimmed_lattice,
)
from graphite.explicit.sc_role_surface_dual import build_role_surface_dual

# Full-cell SC rules for the wrist-rest Nodal Conformation matrix.
NODAL_CONFORMATION_MATRIX_RULES: tuple[str, ...] = (
    "grid",
    "octahedral",
    "star",
    "octet",
    "cross",
    "kelvin",
    "tesseract",
    "hex_face_dual",
)


@dataclass
class NodalConformationResult:
    rule_name: str
    origin_offset: np.ndarray
    hex_elems: np.ndarray
    volume_nodes: np.ndarray
    volume_struts: np.ndarray
    dual_nodes: np.ndarray
    dual_struts: np.ndarray
    dual_nodes_projected: np.ndarray
    surface_gids: set[int]
    report: dict = field(default_factory=dict)
    dual_solid: object = field(default=None)


def _as_cell_dims(cell_size) -> np.ndarray:
    cell_dims = np.asarray(cell_size, dtype=np.float64)
    if cell_dims.ndim == 0:
        cell_dims = np.full(3, float(cell_dims), dtype=np.float64)
    if cell_dims.shape != (3,) or np.any(cell_dims <= 0.0):
        raise ValueError("cell_size must be a positive scalar or three positive dimensions")
    return cell_dims


def _vf_cull_hex_elems(
    cad: trimesh.Trimesh,
    cell_size,
    origin_offset: np.ndarray,
    *,
    volume_fraction_threshold: float,
    samples_per_axis: int,
) -> tuple[np.ndarray, dict]:
    bounds = np.asarray(cad.bounds, dtype=np.float64)
    grid_nodes, cells = generate_background_grid(
        "SC", bounds, cell_size, origin_offset=origin_offset
    )
    cells = np.asarray(cells, dtype=np.int64)
    hex_all = np.asarray(grid_nodes[cells], dtype=np.float64)
    vf = estimate_hex_volume_fractions(
        cad, hex_all, samples_per_axis=int(samples_per_axis)
    )
    fracs = np.asarray(vf.volume_fractions, dtype=np.float64)
    keep = fracs >= float(volume_fraction_threshold)
    kept = hex_all[keep]
    info = {
        "n_hex_total": int(len(hex_all)),
        "n_hex_kept": int(len(kept)),
        "n_partial": int(np.count_nonzero((fracs[keep] < 1.0 - 1e-9))),
        "vf_threshold": float(volume_fraction_threshold),
    }
    return kept, info


def compact_subgraph(
    nodes: np.ndarray,
    struts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unused nodes and reindex struts."""
    pts = np.asarray(nodes, dtype=np.float64)
    edges = _unique_edges(struts)
    if len(pts) == 0 or len(edges) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
        )
    used = np.unique(edges.reshape(-1))
    remap = {int(g): i for i, g in enumerate(used.tolist())}
    out_nodes = pts[used]
    out_struts = np.asarray(
        [[remap[int(a)], remap[int(b)]] for a, b in edges],
        dtype=np.int64,
    )
    return out_nodes, out_struts


def surface_dual_from_volume(
    nodes: np.ndarray,
    struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    emit_nodes: np.ndarray | None = None,
    inside_eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Surface dual of a volume graph: leftover outside nodes plus the inside
    endpoints of struts that cross CAD.

    Mark surface on ``nodes`` (use undeformed coordinates). ``emit_nodes`` may
    be the deformed copy so the dual sits on the snapped core.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    edges = _unique_edges(struts)
    empty = (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 2), dtype=np.int64),
        {"n_outside": 0, "n_surface": 0, "n_crossing": 0, "n_dual_struts": 0},
    )
    if len(pts) == 0 or len(edges) == 0:
        return empty
    emit = np.asarray(pts if emit_nodes is None else emit_nodes, dtype=np.float64)
    if emit.shape != pts.shape:
        raise ValueError("emit_nodes must match nodes shape")
    sdf = np.asarray(safe_signed_distance(cad_mesh, pts), dtype=np.float64)
    outside = sdf > float(inside_eps)
    surface = outside.copy()
    n_crossing = 0
    for a, b in edges:
        ia, ib = int(a), int(b)
        if bool(outside[ia]) != bool(outside[ib]):
            surface[ia] = True
            surface[ib] = True
            n_crossing += 1
    dual_edges: list[tuple[int, int]] = []
    for a, b in edges:
        ia, ib = int(a), int(b)
        if bool(surface[ia]) and bool(surface[ib]):
            dual_edges.append((ia, ib) if ia < ib else (ib, ia))
    dual_nodes, dual_struts = compact_subgraph(
        emit,
        np.asarray(dual_edges, dtype=np.int64) if dual_edges else np.empty((0, 2), dtype=np.int64),
    )
    info = {
        "n_outside": int(np.count_nonzero(outside)),
        "n_surface": int(np.count_nonzero(surface)),
        "n_crossing": int(n_crossing),
        "n_dual_struts": int(len(dual_struts)),
    }
    return dual_nodes, dual_struts, info


def project_nodes_closest(nodes: np.ndarray, cad_mesh: trimesh.Trimesh) -> np.ndarray:
    pts = np.asarray(nodes, dtype=np.float64)
    if len(pts) == 0:
        return pts.copy()
    closest, _dist, _ = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(pts)
    return np.asarray(closest, dtype=np.float64)


def deform_outside_nodes(
    nodes: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    inside_eps: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Closest-project nodes with SDF > ``inside_eps``. Interior stays put."""
    pts = np.asarray(nodes, dtype=np.float64)
    if len(pts) == 0:
        return pts.copy(), 0
    sdf = np.asarray(safe_signed_distance(cad_mesh, pts), dtype=np.float64)
    outside = sdf > float(inside_eps)
    n_out = int(np.count_nonzero(outside))
    if n_out == 0:
        return pts.copy(), 0
    out = pts.copy()
    out[outside] = project_nodes_closest(pts[outside], cad_mesh)
    return out, n_out


def _unique_edges(struts: np.ndarray) -> np.ndarray:
    out: list[tuple[int, int]] = []
    for a, b in np.asarray(struts, dtype=np.int64).reshape(-1, 2):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        out.append((ia, ib) if ia < ib else (ib, ia))
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(set(out)), dtype=np.int64)


def _score_graph_outside(
    nodes: np.ndarray,
    sdf_field,
    *,
    inside_eps: float,
) -> tuple[float, int, int]:
    n = int(len(nodes))
    if n == 0:
        return 1.0, 0, 0
    sdf = np.asarray(sdf_field.sample(nodes), dtype=np.float64)
    outside = sdf > float(inside_eps)
    outside_count = int(np.count_nonzero(outside))
    return float(outside_count) / float(n), outside_count, n


def minimize_nodal_conformation_offset(
    cad_mesh: trimesh.Trimesh,
    cell_size,
    *,
    rule_name: str,
    volume_fraction_threshold: float = 0.5,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    inside_eps: float = 0.0,
    prune_loose_external: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Axis-6 phase search minimizing exterior stamped-lattice nodes.

    Octahedral uses tiered extent trim for scoring; other rules use VF cull + stamp.
    """
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    if isinstance(cad, trimesh.Scene):
        cad = trimesh.util.concatenate(tuple(cad.geometry.values()))

    cell_dims = _as_cell_dims(cell_size)
    increment = axial_increment(rule_name, cell_dims)
    offsets = axis6_candidate_offsets(increment)
    rule = str(rule_name).strip().lower()

    probe, _ = _vf_cull_hex_elems(
        cad,
        cell_dims,
        offsets[0],
        volume_fraction_threshold=volume_fraction_threshold,
        samples_per_axis=samples_per_axis,
    )
    if len(probe) == 0 and rule != "octahedral":
        # Still build SDF from a coarse probe of all hexes at origin
        bounds = np.asarray(cad.bounds, dtype=np.float64)
        gn, cells = generate_background_grid("SC", bounds, cell_dims, origin_offset=offsets[0])
        probe = np.asarray(gn[np.asarray(cells, dtype=np.int64)], dtype=np.float64)
    vf0 = estimate_hex_volume_fractions(cad, probe[: max(1, min(64, len(probe)))], samples_per_axis=samples_per_axis)
    res = float(vf0.voxel_resolution) if vf0.voxel_resolution > 0 else 0.5
    sdf_field = build_edt_sdf_field(cad, res)

    candidates: list[dict] = []
    best_offset = offsets[0]
    best_key: tuple | None = None

    for offset in offsets:
        bounds = np.asarray(cad.bounds, dtype=np.float64)
        gn, cells = generate_background_grid(
            "SC", bounds, cell_dims, origin_offset=offset
        )
        hex_elems = np.asarray(gn[np.asarray(cells, dtype=np.int64)], dtype=np.float64)
        nodes, struts, trim = generate_node_plane_trimmed_lattice(
            cad,
            hex_elems,
            rule_name=rule,
            empty_vf_max=float(empty_vf_max),
            samples_per_axis=int(samples_per_axis),
            extent_samples_per_axis=int(extent_samples_per_axis),
            sdf_field=sdf_field,
            inside_eps=float(inside_eps),
            grid_cull_vf_min=GRID_CULL_VF_MIN,
            prune_loose_external=bool(prune_loose_external),
        )
        n_hex = int(trim.n_hex_kept)

        outside_frac, outside_count, n_nodes = _score_graph_outside(
            nodes, sdf_field, inside_eps=inside_eps
        )
        cand = {
            "offset_mm": np.asarray(offset, dtype=np.float64).tolist(),
            "outside_frac": outside_frac,
            "outside_count": outside_count,
            "n_nodes": n_nodes,
            "n_struts": int(len(struts)),
            "n_hex_kept": n_hex,
        }
        candidates.append(cand)
        key = (
            outside_frac,
            outside_count,
            -n_hex,
            float(np.linalg.norm(offset)),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_offset = np.asarray(offset, dtype=np.float64)

    return best_offset, candidates


def generate_nodal_conformation(
    cad_mesh: trimesh.Trimesh,
    cell_size,
    *,
    rule_name: str = "octahedral",
    origin_offset: np.ndarray | tuple[float, float, float] | None = None,
    run_node_minimization: bool = True,
    volume_fraction_threshold: float = 0.5,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    project_dual: bool = True,
    prune_loose_external: bool = True,
    box_vf_min: float = 0.0,
    prune_floor_shortcuts: bool = False,
    surface_dual_mode: Literal["planar_sweep", "topology_only"] = "planar_sweep",
    sweep_config: Any | None = None,
) -> NodalConformationResult:
    """
    Build undeformed volume + dual graphs (optional dual closest-point project).

    After node-plane trim, drop outside nodes that are only tied to other
    outside nodes. Gold octahedral engine does not use this prune.
    """
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    if isinstance(cad, trimesh.Scene):
        cad = trimesh.util.concatenate(tuple(cad.geometry.values()))

    cell_dims = _as_cell_dims(cell_size)
    rule = str(rule_name).strip().lower()
    planes = node_plane_policy(rule)

    phase_report: dict = {"node_minimization": bool(run_node_minimization)}
    if run_node_minimization and origin_offset is None:
        best, cands = minimize_nodal_conformation_offset(
            cad,
            cell_dims,
            rule_name=rule,
            volume_fraction_threshold=volume_fraction_threshold,
            empty_vf_max=empty_vf_max,
            samples_per_axis=samples_per_axis,
            extent_samples_per_axis=extent_samples_per_axis,
            prune_loose_external=bool(prune_loose_external),
        )
        offset = best
        phase_report["candidates"] = cands
        phase_report["phase_increment_mm"] = axial_increment(rule, cell_dims).tolist()
    elif origin_offset is None:
        offset = np.zeros(3, dtype=np.float64)
    else:
        offset = np.asarray(origin_offset, dtype=np.float64).reshape(3)

    phase_report["origin_offset_mm"] = offset.tolist()

    if rule == "grid":
        from graphite.explicit.grid_boolean_engine import generate_grid_boolean_lattice

        all_nodes, vol_struts, surf_struts, surf_nodes = generate_grid_boolean_lattice(
            cad,
            cell_size=cell_dims,
            origin_offset=offset,
            prune_floor_shortcuts=bool(prune_floor_shortcuts),
        )
        vol_struts = _unique_edges(vol_struts)
        surf_struts = _unique_edges(surf_struts)

        remap_surf = {int(g): i for i, g in enumerate(surf_nodes)}
        dual_nodes = (
            all_nodes[surf_nodes] if len(surf_nodes) else np.empty((0, 3), dtype=np.float64)
        )
        dual_struts = (
            np.asarray(
                [
                    [remap_surf[int(a)], remap_surf[int(b)]]
                    for a, b in surf_struts
                    if int(a) in remap_surf and int(b) in remap_surf
                ],
                dtype=np.int64,
            )
            if len(surf_struts)
            else np.empty((0, 2), dtype=np.int64)
        )
        dual_proj = dual_nodes.copy()
        surface_gids = set(int(g) for g in surf_nodes)

        report = {
            "method": "nodal_conformation_grid_boolean",
            "rule_name": rule,
            "cell_mm": cell_dims.tolist(),
            "node_plane_policy": planes.as_dict(),
            "phase": phase_report,
            "trim": {"mode": "axis_raycast_boolean_crop"},
            "n_volume_nodes": int(len(all_nodes)),
            "n_volume_struts": int(len(vol_struts)),
            "n_dual_nodes": int(len(dual_nodes)),
            "n_dual_struts": int(len(dual_struts)),
            "n_surface_gids": int(len(surface_gids)),
            "project_dual": bool(project_dual),
            "prune_loose_external": bool(prune_loose_external),
            "box_vf_min": float(box_vf_min),
        }
        return NodalConformationResult(
            rule_name=rule,
            origin_offset=offset,
            hex_elems=np.empty((0, 8, 3), dtype=np.float64),
            volume_nodes=all_nodes,
            volume_struts=vol_struts,
            dual_nodes=dual_nodes,
            dual_struts=dual_struts,
            dual_nodes_projected=dual_proj,
            surface_gids=surface_gids,
            report=report,
        )

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
        prune_loose_external=bool(prune_loose_external),
        box_vf_min=float(box_vf_min),
    )
    vol_struts = _unique_edges(vol_struts)
    hex_kept = hex_elems
    trim_info = trim_report.as_dict()

    if len(vol_nodes) == 0:
        raise RuntimeError(f"{rule}: no nodes survived cull/trim")
    role = build_role_surface_dual(
        hex_elems,
        vol_nodes,
        rule_name=rule,
        cad_mesh=cad,
        volume_struts=vol_struts,
        empty_vf_max=float(empty_vf_max),
        samples_per_axis=int(samples_per_axis),
        extent_samples_per_axis=int(extent_samples_per_axis),
    )
    dual_nodes = np.asarray(role.dual_nodes, dtype=np.float64)
    dual_struts = _unique_edges(role.dual_struts)
    surface_gids = set(int(g) for g in role.surface_gids)
    trim_info = {**trim_info, "role_dual": role.report}

    if project_dual and len(dual_nodes) > 0:
        closest, _dist, _ = trimesh.proximity.ProximityQuery(cad).on_surface(dual_nodes)
        dual_proj = np.asarray(closest, dtype=np.float64)
    else:
        dual_proj = np.asarray(dual_nodes, dtype=np.float64).copy()

    dual_solid = None
    sweep_report = None
    if surface_dual_mode == "planar_sweep" and len(dual_proj) > 0 and len(dual_struts) > 0:
        try:
            from graphite.explicit.planar_surface_sweep import (
                PlanarSweepConfig,
                build_planar_slicing_surface_dual,
            )
            cfg = sweep_config if isinstance(sweep_config, PlanarSweepConfig) else PlanarSweepConfig()
            dual_solid, sweep_report = build_planar_slicing_surface_dual(
                cad,
                dual_nodes_projected=dual_proj,
                dual_struts=dual_struts,
                volume_nodes=vol_nodes,
                volume_struts=vol_struts,
                dual_nodes_cartesian=dual_nodes,
                config=cfg,
            )
        except Exception as e:
            sweep_report = {"status": "failed", "error": str(e)}

    report = {
        "method": "nodal_conformation",
        "rule_name": rule,
        "cell_mm": cell_dims.tolist(),
        "node_plane_policy": planes.as_dict(),
        "phase": phase_report,
        "trim": trim_info if isinstance(trim_info, dict) else {"raw": str(trim_info)},
        "n_volume_nodes": int(len(vol_nodes)),
        "n_volume_struts": int(len(vol_struts)),
        "n_dual_nodes": int(len(dual_nodes)),
        "n_dual_struts": int(len(dual_struts)),
        "n_surface_gids": int(len(surface_gids)),
        "project_dual": bool(project_dual),
        "prune_loose_external": bool(prune_loose_external),
        "box_vf_min": float(box_vf_min),
        "surface_dual_mode": str(surface_dual_mode),
    }
    if sweep_report is not None:
        report["planar_sweep"] = sweep_report

    return NodalConformationResult(
        rule_name=rule,
        origin_offset=offset,
        hex_elems=np.asarray(hex_kept, dtype=np.float64),
        volume_nodes=vol_nodes,
        volume_struts=vol_struts,
        dual_nodes=dual_nodes,
        dual_struts=dual_struts,
        dual_nodes_projected=dual_proj,
        surface_gids=surface_gids,
        report=report,
        dual_solid=dual_solid,
    )


def weld_combined_lattice(
    nc: NodalConformationResult,
    cad_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weld volume lattice and surface dual into a single unified conforming lattice.

    Matches dual nodes to volume nodes, updates surface nodes to their projected
    boundary positions, and eliminates duplicate struts between volume and dual.

    Returns:
        (cartesian_nodes, deformed_nodes, unique_combined_struts)
    """
    from scipy.spatial import cKDTree

    vol_nodes = np.asarray(nc.volume_nodes, dtype=np.float64)
    vol_struts = np.asarray(nc.volume_struts, dtype=np.int64)
    dual_nodes = np.asarray(nc.dual_nodes, dtype=np.float64)
    dual_struts = np.asarray(nc.dual_struts, dtype=np.int64)
    dual_proj = np.asarray(nc.dual_nodes_projected, dtype=np.float64)

    unified_cart_nodes = vol_nodes.copy()
    if cad_mesh is not None:
        vol_def, _ = deform_outside_nodes(vol_nodes, cad_mesh)
    else:
        vol_def = vol_nodes.copy()
    unified_def_nodes = vol_def.copy()

    if len(dual_nodes) > 0 and len(vol_nodes) > 0:
        tree = cKDTree(vol_nodes)
        dists, dual_to_vol = tree.query(dual_nodes)
        cell_span = float(np.min(np.ptp(nc.hex_elems.reshape(-1, 3), axis=0))) if len(nc.hex_elems) > 0 else 25.4
        weld_atol = max(1e-4 * cell_span, 1e-4)
        matched = dists < weld_atol

        for di, vi in enumerate(dual_to_vol):
            if matched[di]:
                unified_def_nodes[vi] = dual_proj[di]

        remap_dual = np.zeros(len(dual_nodes), dtype=np.int64)
        extra_cart = []
        extra_def = []
        next_idx = len(vol_nodes)
        for di in range(len(dual_nodes)):
            if matched[di]:
                remap_dual[di] = dual_to_vol[di]
            else:
                remap_dual[di] = next_idx
                extra_cart.append(dual_nodes[di])
                extra_def.append(dual_proj[di])
                next_idx += 1

        if extra_cart:
            unified_cart_nodes = np.vstack([unified_cart_nodes, extra_cart])
            unified_def_nodes = np.vstack([unified_def_nodes, extra_def])

        remapped_dual_struts = np.zeros_like(dual_struts)
        for i, (da, db) in enumerate(dual_struts):
            remapped_dual_struts[i] = [remap_dual[da], remap_dual[db]]

        combined_struts = _unique_edges(np.vstack([vol_struts, remapped_dual_struts]))
    else:
        combined_struts = vol_struts.copy()

    return unified_cart_nodes, unified_def_nodes, combined_struts


def generate_sc_conformal_lattice(
    cad_filepath: str | Path | trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    strut_radius: float = 0.5,
    lattice_type: str = "SC",
    export_dir: str | Path | None = None,
    skip_sweep: bool = False,
    dual_width: float | None = None,
    dual_thickness: float | None = None,
    surface_dual_mode: str = "planar_sweep",
    rule_name: str = "octahedral",
    part_name: str = "conformal_lattice",
    origin_offset: np.ndarray | tuple[float, float, float] | None = None,
    run_node_minimization: bool = False,
    box_vf_min: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Production drop-in entrypoint for SC conformal lattice generation.

    Applies the Nodal Conformation Bookend Pipeline:
      1. Generates background Cartesian SC grid with node-plane trimming.
      2. Snaps outside volume nodes to CAD boundary (leaving interior crystal symmetry 100% rigid).
      3. Builds universal layered role surface dual.
      4. Solidifies via Planar Slicing Contour Sweep + double CAD boolean shell.
    """
    if isinstance(cad_filepath, (str, Path)):
        cad_path = Path(cad_filepath)
        cad = trimesh.load(str(cad_path), force="mesh")
        if isinstance(cad, trimesh.Scene):
            cad = cad.dump(concatenate=True)
        default_pname = cad_path.stem
    else:
        cad = cad_filepath
        if isinstance(cad, trimesh.Scene):
            cad = cad.dump(concatenate=True)
        default_pname = "conformal_lattice"

    pname = part_name if part_name != "conformal_lattice" else default_pname
    r = float(strut_radius)
    dw = float(dual_width) if dual_width is not None else max(1.6, 3.2 * r)
    dt = float(dual_thickness) if dual_thickness is not None else max(0.8, 1.6 * r)

    mode = str(kwargs.get("mode", "conformal")).lower().strip()
    is_boolean = (mode == "boolean")
    if is_boolean:
        surface_dual_mode = "none"

    cfg = None
    if surface_dual_mode == "planar_sweep":
        from graphite.explicit.planar_surface_sweep import PlanarSweepConfig

        sweep_cfg = kwargs.get("sweep_config")
        if isinstance(sweep_cfg, PlanarSweepConfig):
            cfg = sweep_cfg
        else:
            cfg = PlanarSweepConfig(
                dual_width=dw,
                dual_thickness=dt,
                inward_depth_factor=1.5,
                outer_margin=0.4,
                n_sweep_pts=8,
                extend_factor=0.4,
                valley_prune_sdf=kwargs.get("valley_prune_sdf", None),
                boundary_promote_eps=float(kwargs.get("boundary_promote_eps", 1.0)),
                enable_junction_caps=bool(kwargs.get("enable_junction_caps", True)),
                junction_cap_factor=float(kwargs.get("junction_cap_factor", 0.6)),
            )

    nc = generate_nodal_conformation(
        cad,
        cell_size=cell_size,
        rule_name=rule_name,
        origin_offset=origin_offset,
        run_node_minimization=run_node_minimization,
        box_vf_min=box_vf_min,
        surface_dual_mode=surface_dual_mode,
        sweep_config=cfg,
    )

    vol_nodes = np.asarray(nc.volume_nodes, dtype=np.float64)
    vol_struts = np.asarray(nc.volume_struts, dtype=np.int64)
    if is_boolean:
        vol_nodes_def = vol_nodes
        dual_nodes = np.empty((0, 3), dtype=np.float64)
        dual_struts = np.empty((0, 2), dtype=np.int64)
    else:
        vol_nodes_def, _ = deform_outside_nodes(vol_nodes, cad)
        dual_nodes = np.asarray(nc.dual_nodes_projected, dtype=np.float64)
        dual_struts = np.asarray(nc.dual_struts, dtype=np.int64)

    if skip_sweep:
        return {
            "nodes": vol_nodes_def,
            "struts": vol_struts,
            "nodes_relaxed": vol_nodes_def,
            "nodes_3d": vol_nodes,
            "red_struts": vol_struts,
            "cyan_struts": dual_struts,
            "dual_nodes": dual_nodes,
            "dual_struts": dual_struts,
            "nodes_count": len(vol_nodes_def),
            "struts_count": len(vol_struts),
            "dual_nodes_count": len(dual_nodes),
            "dual_struts_count": len(dual_struts),
            "red_struts_count": len(vol_struts),
            "cyan_struts_count": len(dual_struts),
            "mesh": None,
            "core_mesh": None,
            "dual_mesh": None,
            "status": "ok",
            "rule_name": rule_name,
            "report": nc.report,
        }

    from graphite.explicit.geometry_module import (
        _cylinder_manifold_for_joint,
        _manifold_to_trimesh,
        _union_manifolds_for_joint,
        trimesh_to_manifold,
    )

    cad_outer = trimesh_to_manifold(cad)

    core_parts = []
    for u, v in vol_struts:
        cyl = _cylinder_manifold_for_joint(vol_nodes_def[u], vol_nodes_def[v], radius=r, segments=14)
        if cyl is not None:
            core_parts.append(cyl)

    if core_parts:
        core_raw, _ = _union_manifolds_for_joint(core_parts)
        core_trimmed = core_raw ^ cad_outer if core_raw is not None else None
    else:
        core_trimmed = None

    if nc.dual_solid is not None:
        cad_inner_mesh = cad.copy()
        cad_inner_mesh.vertices -= dt * cad_inner_mesh.vertex_normals
        cad_inner = trimesh_to_manifold(cad_inner_mesh)
        dual_trimmed = (nc.dual_solid ^ cad_outer) - cad_inner
    else:
        dual_trimmed = None

    if core_trimmed is not None and dual_trimmed is not None:
        combined_manifold = (core_trimmed + dual_trimmed) ^ cad_outer
    elif core_trimmed is not None:
        combined_manifold = core_trimmed
    elif dual_trimmed is not None:
        combined_manifold = dual_trimmed
    else:
        combined_manifold = None

    mesh_combined = _manifold_to_trimesh(combined_manifold) if combined_manifold is not None else None
    mesh_core = _manifold_to_trimesh(core_trimmed) if core_trimmed is not None else None
    mesh_dual = _manifold_to_trimesh(dual_trimmed) if dual_trimmed is not None else None

    if export_dir is not None:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if mesh_combined is not None:
            mesh_combined.export(str(out_dir / f"{pname}_conformal_lattice.stl"))
        if mesh_core is not None:
            mesh_core.export(str(out_dir / f"{pname}_core.stl"))
        if mesh_dual is not None:
            mesh_dual.export(str(out_dir / f"{pname}_surface_dual.stl"))

    return {
        "nodes": vol_nodes_def,
        "struts": vol_struts,
        "nodes_relaxed": vol_nodes_def,
        "nodes_3d": vol_nodes,
        "red_struts": vol_struts,
        "cyan_struts": dual_struts,
        "dual_nodes": dual_nodes,
        "dual_struts": dual_struts,
        "nodes_count": len(vol_nodes_def),
        "struts_count": len(vol_struts),
        "dual_nodes_count": len(dual_nodes),
        "dual_struts_count": len(dual_struts),
        "red_struts_count": len(vol_struts),
        "cyan_struts_count": len(dual_struts),
        "mesh": mesh_combined,
        "core_mesh": mesh_core,
        "dual_mesh": mesh_dual,
        "status": "ok",
        "rule_name": rule_name,
        "report": nc.report,
    }


# ---------------------------------------------------------------------------
# Conformation Diagnostic Stages Helper (Troubleshooting & Visualization)
# ---------------------------------------------------------------------------


def calculate_sc_strut_radius_for_solid_fraction(
    nodes: np.ndarray,
    struts: np.ndarray,
    n_cells: int,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    target_solid_fraction: float = 0.02,
) -> tuple[float, float]:
    """Calculate the strut radius and diameter to achieve a target solid fraction across a lattice.

    Computes:
        V_domain = n_cells * (cell_size_x * cell_size_y * cell_size_z)
        L_total = sum(strut_lengths)
        strut_radius = sqrt((target_solid_fraction * V_domain) / (pi * L_total))
        strut_diameter = 2 * strut_radius

    Returns:
        (strut_radius, strut_diameter) in mm.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    edges = np.asarray(struts, dtype=np.int64).reshape(-1, 2)
    cell_dims = _as_cell_dims(cell_size)
    v_cell = float(np.prod(cell_dims))
    v_domain = float(n_cells) * v_cell

    if len(edges) == 0 or v_domain <= 0:
        return 0.5, 1.0

    lengths = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
    l_total = float(np.sum(lengths))
    if l_total <= 0:
        return 0.5, 1.0

    phi = float(target_solid_fraction)
    r = float(np.sqrt((phi * v_domain) / (np.pi * l_total)))
    return r, 2.0 * r


@dataclass
class SCConformationStagesResult:
    """Diagnostic data and meshes for the 3 conformal lattice preparation stages."""

    rule_name: str
    cell_size: np.ndarray
    target_solid_fraction: float
    calculated_strut_diameter: float
    calculated_strut_radius: float

    # Node and strut arrays (non-default)
    untrimmed_nodes: np.ndarray
    untrimmed_struts: np.ndarray
    trimmed_nodes: np.ndarray
    trimmed_struts: np.ndarray
    conformed_nodes: np.ndarray
    conformed_struts: np.ndarray

    # Meshes and paths (optional/default None)
    untrimmed_mesh: trimesh.Trimesh | None = None
    untrimmed_stl_path: Path | None = None
    trimmed_mesh: trimesh.Trimesh | None = None
    trimmed_stl_path: Path | None = None
    conformed_mesh: trimesh.Trimesh | None = None
    conformed_stl_path: Path | None = None

    # SC Grid Cage (bounding box wireframe of kept cells)
    grid_cage_nodes: np.ndarray | None = None
    grid_cage_struts: np.ndarray | None = None
    grid_cage_mesh: trimesh.Trimesh | None = None
    grid_cage_stl_path: Path | None = None
    grid_cage_diameter: float = 0.0
    grid_cage_radius: float = 0.0

    # Stage 4: Full Conformal Lattice (with Universal Surface Dual)
    full_conformal_mesh: trimesh.Trimesh | None = None
    full_conformal_stl_path: Path | None = None

    report: dict = field(default_factory=dict)



def generate_sc_conformation_stages(
    cad_filepath: str | Path | trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray = 15.0,
    *,
    rule_name: str = "star",
    target_solid_fraction: float = 0.02,
    strut_diameter: float | None = None,
    empty_vf_max: float = 0.01,
    box_vf_min: float = 0.0,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    origin_offset: np.ndarray | tuple[float, float, float] | None = None,
    export_dir: str | Path | None = None,
    part_name: str = "sc_stages",
    build_meshes: bool = True,
    include_grid_cage: bool = True,
    include_full_conformal: bool = True,
    circular_segments: int = 16,
) -> SCConformationStagesResult:
    """Generate the 3 progressive stages of SC lattice preparation for visual troubleshooting:

    1. Culled, Untrimmed Array: Full unit cells stamped into all kept Cartesian hexes (no sub-cell trimming).
       Allows inspecting raw unit-cell placement and cell culling relative to CAD.
    2. Culled & Trimmed Array: Cartesian array trimmed against CAD by sub-cell node planes (25%/75%).
       Allows inspecting internal crystal symmetry and boundary sub-cell element dropping.
    3. Trimmed & Conformed Array: The trimmed array with outside nodes snapped/deformed to the CAD boundary.
       Allows inspecting boundary node projection and lattice conformance before surface dual addition.

    Also generates an optional SC Grid Cage (wireframe box edges of all kept hexes at target solid fraction)
    to visually map out the underlying simple cubic grid.
    """
    from scipy.spatial import cKDTree
    from graphite.explicit.hex_topology_module import get_hex_topology_rule
    from graphite.explicit.geometry_module import build_clean_miter_truss

    # 1. Load CAD
    if isinstance(cad_filepath, (str, Path)):
        cad_path = Path(cad_filepath)
        cad = trimesh.load(str(cad_path), force="mesh")
        if isinstance(cad, trimesh.Scene):
            cad = cad.dump(concatenate=True)
        default_pname = cad_path.stem
    else:
        cad = cad_filepath
        if isinstance(cad, trimesh.Scene):
            cad = cad.dump(concatenate=True)
        default_pname = "conformal_stages"

    pname = part_name if part_name != "sc_stages" else default_pname
    cell_dims = _as_cell_dims(cell_size)
    rule = str(rule_name).strip().lower()

    if origin_offset is None:
        offset = np.zeros(3, dtype=np.float64)
    else:
        offset = np.asarray(origin_offset, dtype=np.float64).reshape(3)

    # 2. Background Grid & Volume Fraction Culling
    bounds = np.asarray(cad.bounds, dtype=np.float64)
    gn, cells = generate_background_grid("SC", bounds, cell_dims, origin_offset=offset)
    hex_all = np.asarray(gn[np.asarray(cells, dtype=np.int64)], dtype=np.float64)

    vf = estimate_hex_volume_fractions(cad, hex_all, samples_per_axis=int(samples_per_axis))
    fracs = np.asarray(vf.volume_fractions, dtype=np.float64)
    kept_mask = fracs >= float(empty_vf_max)
    kept_hexes = hex_all[kept_mask]

    if len(kept_hexes) == 0:
        raise RuntimeError(f"No hexes kept for {rule} with empty_vf_max={empty_vf_max}")

    # 3. Stage 1: Culled, Untrimmed SC Array
    rule_obj = get_hex_topology_rule(rule)
    all_pts = []
    all_struts = []
    for h in kept_hexes:
        pts_h, struts_h = rule_obj.builder(h)
        base_idx = len(all_pts)
        all_pts.extend(pts_h)
        all_struts.extend(struts_h + base_idx)

    all_pts_arr = np.asarray(all_pts, dtype=np.float64)
    all_struts_arr = np.asarray(all_struts, dtype=np.int64)

    tree1 = cKDTree(all_pts_arr)
    groups1 = tree1.query_ball_tree(tree1, r=1e-4)
    remap1 = np.zeros(len(all_pts_arr), dtype=np.int64)
    visited1 = set()
    u_pts1_list = []
    for i, g in enumerate(groups1):
        if i not in visited1:
            rep = len(u_pts1_list)
            u_pts1_list.append(all_pts_arr[i])
            for idx in g:
                remap1[idx] = rep
                visited1.add(idx)

    stage1_nodes = np.asarray(u_pts1_list, dtype=np.float64)
    stage1_struts = _unique_edges(remap1[all_struts_arr])

    # 4. Determine Strut Radius & Diameter
    if strut_diameter is not None and float(strut_diameter) > 0:
        r_strut = float(strut_diameter) / 2.0
        d_strut = float(strut_diameter)
    else:
        r_strut, d_strut = calculate_sc_strut_radius_for_solid_fraction(
            stage1_nodes,
            stage1_struts,
            len(kept_hexes),
            cell_dims,
            target_solid_fraction=target_solid_fraction,
        )

    # 5. Stage 2: Culled and Node-Plane Trimmed Array (Cartesian)
    stage2_nodes, stage2_struts, trim_rep = generate_node_plane_trimmed_lattice(
        cad,
        kept_hexes,
        rule_name=rule,
        empty_vf_max=float(empty_vf_max),
        samples_per_axis=int(samples_per_axis),
        extent_samples_per_axis=int(extent_samples_per_axis),
        grid_cull_vf_min=GRID_CULL_VF_MIN,
        box_vf_min=float(box_vf_min),
    )
    stage2_struts = _unique_edges(stage2_struts)

    # 6. Stage 3: Trimmed & Conformed Array (Boundary Snapped)
    stage3_nodes, n_deformed = deform_outside_nodes(stage2_nodes, cad)
    stage3_struts = stage2_struts.copy()

    # 7. Optional SC Grid Cage (12 Edges per Kept Hex Box)
    cage_nodes = None
    cage_struts = None
    r_cage = 0.0
    d_cage = 0.0
    if include_grid_cage:
        _HEX_EDGES = (
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        all_corners = kept_hexes.reshape(-1, 3)
        tree_c = cKDTree(all_corners)
        groups_c = tree_c.query_ball_tree(tree_c, r=1e-4)
        remap_c = np.zeros(len(all_corners), dtype=np.int64)
        visited_c = set()
        u_corners_list = []
        for i, g in enumerate(groups_c):
            if i not in visited_c:
                rep = len(u_corners_list)
                u_corners_list.append(all_corners[i])
                for idx in g:
                    remap_c[idx] = rep
                    visited_c.add(idx)
        cage_nodes = np.asarray(u_corners_list, dtype=np.float64)
        edge_list = []
        for h_idx in range(len(kept_hexes)):
            base = h_idx * 8
            for e0, e1 in _HEX_EDGES:
                edge_list.append([remap_c[base + e0], remap_c[base + e1]])
        cage_struts = _unique_edges(np.asarray(edge_list, dtype=np.int64))
        r_cage, d_cage = calculate_sc_strut_radius_for_solid_fraction(
            cage_nodes,
            cage_struts,
            len(kept_hexes),
            cell_dims,
            target_solid_fraction=target_solid_fraction,
        )

    # 8. Build Watertight Clean Miter Meshes
    mesh1 = None
    mesh2 = None
    mesh3 = None
    mesh_cage = None

    if build_meshes:
        if len(stage1_struts) > 0:
            mesh1 = build_clean_miter_truss(
                stage1_nodes, stage1_struts, r_strut, circular_segments=circular_segments
            )
        if len(stage2_struts) > 0:
            mesh2 = build_clean_miter_truss(
                stage2_nodes, stage2_struts, r_strut, circular_segments=circular_segments
            )
        if len(stage3_struts) > 0:
            mesh3 = build_clean_miter_truss(
                stage3_nodes, stage3_struts, r_strut, circular_segments=circular_segments
            )
        if include_grid_cage and cage_nodes is not None and cage_struts is not None and len(cage_struts) > 0:
            mesh_cage = build_clean_miter_truss(
                cage_nodes, cage_struts, r_cage, circular_segments=circular_segments
            )
        mesh4 = None
        if include_full_conformal:
            try:
                full_res = generate_sc_conformal_lattice(
                    cad,
                    cell_size=cell_dims,
                    strut_radius=r_strut,
                    rule_name=rule,
                    origin_offset=offset,
                    box_vf_min=box_vf_min,
                )
                mesh4 = full_res.get("mesh")
            except Exception:
                mesh4 = None
    else:
        mesh4 = None

    # 9. Export STLs if Requested
    p1 = None
    p2 = None
    p3 = None
    p_cage = None
    p4 = None

    if export_dir is not None:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if mesh1 is not None:
            p1 = out_dir / f"{pname}_{rule}_stage1_untrimmed.stl"
            mesh1.export(str(p1))
        if mesh2 is not None:
            p2 = out_dir / f"{pname}_{rule}_stage2_trimmed.stl"
            mesh2.export(str(p2))
        if mesh3 is not None:
            p3 = out_dir / f"{pname}_{rule}_stage3_conformed.stl"
            mesh3.export(str(p3))
        if mesh_cage is not None:
            p_cage = out_dir / f"{pname}_grid_cage_2pct.stl"
            mesh_cage.export(str(p_cage))
        if mesh4 is not None:
            p4 = out_dir / f"{pname}_{rule}_stage4_conformal_full.stl"
            mesh4.export(str(p4))

    report = {
        "rule_name": rule,
        "n_hex_total": len(hex_all),
        "n_hex_kept": len(kept_hexes),
        "target_solid_fraction": float(target_solid_fraction),
        "strut_diameter_mm": float(d_strut),
        "strut_radius_mm": float(r_strut),
        "stage1_untrimmed": {
            "n_nodes": int(len(stage1_nodes)),
            "n_struts": int(len(stage1_struts)),
            "n_faces": int(len(mesh1.faces)) if mesh1 is not None else 0,
            "is_watertight": bool(mesh1.is_watertight) if mesh1 is not None else False,
        },
        "stage2_trimmed": {
            "n_nodes": int(len(stage2_nodes)),
            "n_struts": int(len(stage2_struts)),
            "n_faces": int(len(mesh2.faces)) if mesh2 is not None else 0,
            "is_watertight": bool(mesh2.is_watertight) if mesh2 is not None else False,
        },
        "stage3_conformed": {
            "n_nodes": int(len(stage3_nodes)),
            "n_nodes_deformed": int(n_deformed),
            "n_struts": int(len(stage3_struts)),
            "n_faces": int(len(mesh3.faces)) if mesh3 is not None else 0,
            "is_watertight": bool(mesh3.is_watertight) if mesh3 is not None else False,
        },
    }
    if include_grid_cage and cage_nodes is not None and cage_struts is not None:
        report["grid_cage"] = {
            "cage_diameter_mm": float(d_cage),
            "cage_radius_mm": float(r_cage),
            "n_nodes": int(len(cage_nodes)),
            "n_struts": int(len(cage_struts)),
            "n_faces": int(len(mesh_cage.faces)) if mesh_cage is not None else 0,
            "is_watertight": bool(mesh_cage.is_watertight) if mesh_cage is not None else False,
        }

    return SCConformationStagesResult(
        rule_name=rule,
        cell_size=cell_dims,
        target_solid_fraction=float(target_solid_fraction),
        calculated_strut_diameter=float(d_strut),
        calculated_strut_radius=float(r_strut),
        untrimmed_nodes=stage1_nodes,
        untrimmed_struts=stage1_struts,
        untrimmed_mesh=mesh1,
        untrimmed_stl_path=p1,
        trimmed_nodes=stage2_nodes,
        trimmed_struts=stage2_struts,
        trimmed_mesh=mesh2,
        trimmed_stl_path=p2,
        conformed_nodes=stage3_nodes,
        conformed_struts=stage3_struts,
        conformed_mesh=mesh3,
        conformed_stl_path=p3,
        grid_cage_nodes=cage_nodes,
        grid_cage_struts=cage_struts,
        grid_cage_mesh=mesh_cage,
        grid_cage_stl_path=p_cage,
        grid_cage_diameter=float(d_cage),
        grid_cage_radius=float(r_cage),
        full_conformal_mesh=mesh4,
        full_conformal_stl_path=p4,
        report=report,
    )


def render_sc_conformation_stages_figure(
    stages: SCConformationStagesResult,
    cad_mesh: trimesh.Trimesh,
    output_png_path: str | Path,
) -> Path:
    """Render a 4-panel visual comparison of the SC conformation stages with PyVista and Matplotlib:
    1. Stage 1: Culled Untrimmed Array
    2. Stage 2: Culled and Trimmed Array (Cartesian)
    3. Stage 3: Trimmed and Conformed Array (Boundary Snapped)
    4. Stage 4: Full Conformal Lattice with Universal Surface Dual (or SC Grid Cage)
    """
    import matplotlib.pyplot as plt
    import pyvista as pv

    pv.set_plot_theme("document")
    pv_cad = pv.wrap(cad_mesh)

    if stages.full_conformal_mesh is not None and len(stages.full_conformal_mesh.faces) > 0:
        panels = [
            (f"Stage 1: Culled Untrimmed ({stages.rule_name.capitalize()})\nWhole Cells (No Sub-Cell Trimming)", stages.untrimmed_mesh, "#a855f7"),
            (f"Stage 2: Culled & Trimmed ({stages.rule_name.capitalize()})\nCartesian 25%/75% Node-Plane Trim", stages.trimmed_mesh, "#f59e0b"),
            (f"Stage 3: Trimmed & Conformed ({stages.rule_name.capitalize()})\nOutside Nodes Snapped to Boundary", stages.conformed_mesh, "#10b981"),
            (f"Stage 4: Full Conformal + Dual ({stages.rule_name.capitalize()})\nWatertight Universal Surface Dual", stages.full_conformal_mesh, "#06b6d4"),
        ]
    else:
        panels = [
            ("SC Grid Cage (2% Solid Fraction)\nBackground Hex Bounding Boxes", stages.grid_cage_mesh, "#38bdf8"),
            (f"Stage 1: Culled Untrimmed ({stages.rule_name.capitalize()})\nWhole Cells (No Sub-Cell Trimming)", stages.untrimmed_mesh, "#a855f7"),
            (f"Stage 2: Culled & Trimmed ({stages.rule_name.capitalize()})\nCartesian 25%/75% Node-Plane Trim", stages.trimmed_mesh, "#f59e0b"),
            (f"Stage 3: Trimmed & Conformed ({stages.rule_name.capitalize()})\nOutside Nodes Snapped to CAD Surface", stages.conformed_mesh, "#10b981"),
        ]

    images = []
    for title, m, col in panels:
        p = pv.Plotter(off_screen=True, window_size=(700, 800))
        p.background_color = "#0b121e"
        p.add_mesh(pv_cad, color="#475569", opacity=0.15, show_edges=False)
        if m is not None and len(m.faces) > 0:
            p.add_mesh(pv.wrap(m), color=col, smooth_shading=True, specular=0.4, specular_power=20)
        p.camera_position = [(-90.0, -120.0, 110.0), (0.0, 3.0, 45.0), (0, 0, 1)]
        p.camera.zoom(1.18)
        img = p.screenshot()
        p.close()
        images.append((title, img, col))

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), facecolor="#080c10")
    for ax, (title, img, col) in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, color=col, fontsize=12, fontweight="bold", pad=8)

    fig.suptitle(
        f"Graphite SC Conformation Diagnostics: {stages.rule_name.upper()}\n"
        f"Cell Size = {stages.cell_size} mm, Strut Diameter = {stages.calculated_strut_diameter:.3f} mm ({stages.target_solid_fraction*100:.1f}% Solid Fraction)",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    out_path = Path(output_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_path

