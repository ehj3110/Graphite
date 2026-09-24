"""
V3 hybrid SC conformal compare:

  loose (boolean) cull → dual → shrinkwrap to CAD
  strict VF 0.5 cull → volume stamp (Cartesian)
  morph volume iron → nearest dual (lateral OK)
  valence-gate volume→dual bonds at k_rule
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from graphite.explicit.compare_v3_hybrid.valence import (
    apply_valence_gate,
    local_max_node_degree,
)
from graphite.explicit.conformal_core import (
    classify_boundary_from_hex_elems,
    cull_hex_elements,
    generate_sc_volume_topology,
    hex_volumes,
    safe_signed_distance,
)
from graphite.explicit.conformal_generator import (
    apply_depth_gated_relaxation,
    project_to_cad_surface,
)
from graphite.explicit.geometry_module import (
    generate_rectangular_surface_cage,
    union_lattice_with_spherical_joints,
    union_solid_meshes,
)
from graphite.explicit.hex_surface_dual import generate_hex_surface_dual_cage
from graphite.explicit.hex_topology_module import (
    SKIN_MODE_CORNER_EDGE_CAGE,
    SKIN_MODE_FACE_CENTROID_DUAL,
    get_hex_topology_rule,
)
from graphite.explicit.mesh_repair import repair_cad_mesh


def _round_key(pt: np.ndarray, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


def build_dual_from_hex_complex(
    hex_elems: np.ndarray,
    rule_name: str,
    *,
    round_decimals: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Surface dual / corner-edge cage on the exposed faces of ``hex_elems``.

    Uses rule skin mode: octahedral → face-centroid dual; grid → corner-edge cage.
    """
    rule = get_hex_topology_rule(rule_name)
    scaffold, _hex_ids, boundary_quads, exterior = classify_boundary_from_hex_elems(
        hex_elems, round_decimals=round_decimals
    )
    info: dict[str, Any] = {
        "n_boundary_quads": int(len(boundary_quads)),
        "n_exterior_corners": int(len(exterior)),
        "skin_mode": rule.skin_mode,
    }

    if boundary_quads.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            info,
        )

    if rule.skin_mode == SKIN_MODE_FACE_CENTROID_DUAL:
        dual_nodes, dual_struts = generate_hex_surface_dual_cage(
            scaffold,
            boundary_quads,
            coplanar_cos_threshold=0.0,
            include_isolated_closure=True,
            include_corner_closure=True,
        )
    elif rule.skin_mode == SKIN_MODE_CORNER_EDGE_CAGE:
        used = sorted({int(i) for quad in boundary_quads for i in quad})
        old_to_new = {old: i for i, old in enumerate(used)}
        dual_nodes = scaffold[np.asarray(used, dtype=np.int64)].copy()
        strut_set: set[tuple[int, int]] = set()
        for quad in boundary_quads:
            for i in range(4):
                a = old_to_new[int(quad[i])]
                b = old_to_new[int(quad[(i + 1) % 4])]
                if a != b:
                    strut_set.add((min(a, b), max(a, b)))
        dual_struts = (
            np.array(sorted(strut_set), dtype=np.int64)
            if strut_set
            else np.empty((0, 2), dtype=np.int64)
        )
    else:
        # Fallback: face-centroid dual (safe default for hybrid compare).
        dual_nodes, dual_struts = generate_hex_surface_dual_cage(
            scaffold,
            boundary_quads,
            coplanar_cos_threshold=0.0,
            include_isolated_closure=True,
            include_corner_closure=True,
        )

    info["n_dual_nodes"] = int(len(dual_nodes))
    info["n_dual_struts"] = int(len(dual_struts))
    return (
        np.asarray(dual_nodes, dtype=np.float64),
        np.asarray(dual_struts, dtype=np.int64),
        info,
    )


def shrinkwrap_dual_to_cad(
    dual_nodes: np.ndarray,
    cad_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Project dual nodes onto CAD (closest-point). Returns (projected, displacements)."""
    nodes = np.asarray(dual_nodes, dtype=np.float64)
    if len(nodes) == 0:
        return nodes.copy(), np.empty((0,), dtype=np.float64)
    closest, _ = project_to_cad_surface(nodes, cad_mesh)
    disp = np.linalg.norm(closest - nodes, axis=1)
    return np.asarray(closest, dtype=np.float64), disp


def volume_iron_node_ids(
    hex_elems: np.ndarray,
    volume_nodes: np.ndarray,
    rule_name: str,
    *,
    round_decimals: int = 6,
) -> np.ndarray:
    """
    Volume nodes that should couple to the dual (boundary / iron DOFs).

    - Face-centroid dual rules (octahedral): exposed-face centroids.
    - Corner-edge cage (grid): exposed scaffold corners.
    """
    rule = get_hex_topology_rule(rule_name)
    scaffold, _hex_ids, boundary_quads, exterior = classify_boundary_from_hex_elems(
        hex_elems, round_decimals=max(round_decimals, 6)
    )
    iron_pts: list[np.ndarray] = []
    if rule.skin_mode == SKIN_MODE_FACE_CENTROID_DUAL:
        for quad in boundary_quads:
            iron_pts.append(scaffold[np.asarray(quad, dtype=np.int64)].mean(axis=0))
    else:
        for cid in exterior:
            iron_pts.append(scaffold[int(cid)].copy())

    if not iron_pts:
        return np.empty((0,), dtype=np.int64)

    index = {
        _round_key(volume_nodes[i], round_decimals): i for i in range(len(volume_nodes))
    }
    ids: list[int] = []
    seen: set[int] = set()
    for pt in iron_pts:
        key = _round_key(pt, round_decimals)
        idx = index.get(key)
        if idx is None:
            # Nearest volume node (rounding mismatch / rule node not exact).
            d = np.linalg.norm(volume_nodes - pt, axis=1)
            idx = int(np.argmin(d))
        if idx not in seen:
            seen.add(idx)
            ids.append(idx)
    return np.asarray(ids, dtype=np.int64)


def _volume_node_lookup(
    volume_nodes: np.ndarray,
    *,
    round_decimals: int = 6,
) -> dict[tuple[float, float, float], int]:
    return {
        _round_key(volume_nodes[i], round_decimals): i for i in range(len(volume_nodes))
    }


def _lookup_volume_id(
    pt: np.ndarray,
    volume_nodes: np.ndarray,
    index: dict[tuple[float, float, float], int],
    *,
    round_decimals: int = 6,
    max_dist: float = 1e-3,
) -> int | None:
    key = _round_key(pt, round_decimals)
    idx = index.get(key)
    if idx is not None:
        return int(idx)
    d = np.linalg.norm(volume_nodes - np.asarray(pt, dtype=np.float64), axis=1)
    j = int(np.argmin(d))
    if float(d[j]) <= max_dist:
        return j
    return None


def remove_volume_surface_struts(
    hex_elems: np.ndarray,
    volume_nodes: np.ndarray,
    volume_struts: np.ndarray,
    iron_ids: np.ndarray,
    rule_name: str,
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Remove volume struts that duplicate the surface/skin pattern.

    Rules (documented in the compare report):
    - **grid** / corner-edge cage: drop undirected edges of exposed faces of the
      **strict** hex complex (same edges the corner-edge skin cage would use).
    - **octahedral** / face-centroid dual: drop volume struts whose **both**
      endpoints are iron/boundary nodes (iron–iron surface-layer edges that
      duplicate the dual after morph/weld). Interior–interior and iron→interior
      coupling struts are kept.
    """
    rule = get_hex_topology_rule(rule_name)
    struts = np.asarray(volume_struts, dtype=np.int64)
    n_before = int(len(struts))
    if n_before == 0:
        return struts.copy(), {
            "n_volume_struts_before_drop": 0,
            "n_volume_surface_struts_dropped": 0,
            "drop_volume_surface_rule": "none",
        }

    drop_edges: set[tuple[int, int]] = set()
    if rule.skin_mode == SKIN_MODE_CORNER_EDGE_CAGE:
        rule_note = (
            "grid: drop undirected edges of exposed strict-hex faces "
            "(corner-edge cage perimeter)"
        )
        scaffold, _hex_ids, boundary_quads, _ = classify_boundary_from_hex_elems(
            hex_elems, round_decimals=max(round_decimals, 6)
        )
        index = _volume_node_lookup(volume_nodes, round_decimals=round_decimals)
        for quad in boundary_quads:
            for i in range(4):
                pa = scaffold[int(quad[i])]
                pb = scaffold[int(quad[(i + 1) % 4])]
                ia = _lookup_volume_id(
                    pa, volume_nodes, index, round_decimals=round_decimals
                )
                ib = _lookup_volume_id(
                    pb, volume_nodes, index, round_decimals=round_decimals
                )
                if ia is None or ib is None or ia == ib:
                    continue
                drop_edges.add((min(ia, ib), max(ia, ib)))
    else:
        # Face-centroid / octahedral: iron–iron volume edges are the surface layer.
        rule_note = (
            "octahedral: drop volume struts with both endpoints iron/boundary "
            "(iron–iron edges that duplicate dual after morph)"
        )
        iron_set = {int(i) for i in np.asarray(iron_ids, dtype=np.int64)}
        for a, b in struts:
            ia, ib = int(a), int(b)
            if ia in iron_set and ib in iron_set:
                drop_edges.add((min(ia, ib), max(ia, ib)))

    keep: list[tuple[int, int]] = []
    n_dropped = 0
    for a, b in struts:
        ia, ib = int(a), int(b)
        key = (min(ia, ib), max(ia, ib))
        if key in drop_edges:
            n_dropped += 1
            continue
        keep.append((ia, ib))

    out = (
        np.array(keep, dtype=np.int64)
        if keep
        else np.empty((0, 2), dtype=np.int64)
    )
    return out, {
        "n_volume_struts_before_drop": n_before,
        "n_volume_surface_struts_dropped": int(n_dropped),
        "drop_volume_surface_rule": rule_note,
    }


def drop_volume_struts_coincident_with_dual(
    volume_struts: np.ndarray,
    dual_struts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """After merge: drop any volume strut whose undirected edge is also a dual strut."""
    vol = np.asarray(volume_struts, dtype=np.int64)
    dual = np.asarray(dual_struts, dtype=np.int64)
    if len(vol) == 0:
        return vol.copy(), 0
    dual_set = {
        (min(int(a), int(b)), max(int(a), int(b))) for a, b in dual
    }
    keep: list[tuple[int, int]] = []
    n_dropped = 0
    for a, b in vol:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key in dual_set:
            n_dropped += 1
            continue
        keep.append((int(a), int(b)))
    out = (
        np.array(keep, dtype=np.int64)
        if keep
        else np.empty((0, 2), dtype=np.int64)
    )
    return out, int(n_dropped)


def morph_volume_iron_to_dual(
    volume_nodes: np.ndarray,
    volume_struts: np.ndarray,
    iron_ids: np.ndarray,
    dual_nodes: np.ndarray,
    k_rule: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Snap volume iron nodes to nearest dual node with valence gate.

    Lateral motion is allowed (pure Euclidean nearest). Excess volume→dual
    bonds beyond ``k_rule`` per dual node are rejected.
    """
    nodes = np.asarray(volume_nodes, dtype=np.float64).copy()
    dual = np.asarray(dual_nodes, dtype=np.float64)
    iron = np.asarray(iron_ids, dtype=np.int64)
    report: dict[str, Any] = {
        "n_volume_iron": int(len(iron)),
        "n_morph_to_dual": 0,
        "n_bonds_rejected_valence": 0,
        "k_rule": int(k_rule),
        "max_morph_distance": 0.0,
    }
    if len(iron) == 0 or len(dual) == 0:
        return nodes, report

    # Candidates: each iron → its nearest dual (3D Euclidean).
    candidates: list[tuple[int, int, float]] = []
    for iid in iron:
        dvec = dual - nodes[int(iid)]
        dist = np.linalg.norm(dvec, axis=1)
        j = int(np.argmin(dist))
        candidates.append((int(iid), j, float(dist[j])))
    candidates.sort(key=lambda t: t[2])

    accepted, n_rejected = apply_valence_gate(candidates, k_rule)
    report["n_bonds_rejected_valence"] = int(n_rejected)

    max_disp = 0.0
    for iid, did in accepted.items():
        before = nodes[iid].copy()
        nodes[iid] = dual[did]
        max_disp = max(max_disp, float(np.linalg.norm(nodes[iid] - before)))
    report["n_morph_to_dual"] = int(len(accepted))
    report["max_morph_distance"] = float(max_disp)
    report["accepted_iron_to_dual"] = {int(k): int(v) for k, v in accepted.items()}
    return nodes, report


def _relax_volume_with_iron_fixed(
    nodes: np.ndarray,
    struts: np.ndarray,
    iron_ids: np.ndarray,
    *,
    iterations: int,
    alpha: float,
) -> np.ndarray:
    """Laplacian-relax free volume nodes; iron (all boundary candidates) fixed."""
    if iterations <= 0 or len(nodes) == 0:
        return nodes
    depths = np.ones(len(nodes), dtype=np.int32)
    depths[np.asarray(iron_ids, dtype=np.int64)] = 0
    return apply_depth_gated_relaxation(
        nodes,
        struts,
        depths,
        iterations=int(iterations),
        alpha=float(alpha),
        max_depth=None,
    )


def _merge_graphs(
    volume_nodes: np.ndarray,
    volume_struts: np.ndarray,
    dual_nodes: np.ndarray,
    dual_struts: np.ndarray,
    *,
    round_decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate volume + dual, welding coincident nodes.

    Returns combined nodes, volume strut indices (post-merge), dual strut indices.
    """
    node_map: dict[tuple[float, float, float], int] = {}
    out: list[np.ndarray] = []

    def add(pt: np.ndarray) -> int:
        key = _round_key(pt, round_decimals)
        idx = node_map.get(key)
        if idx is None:
            idx = len(out)
            node_map[key] = idx
            out.append(np.asarray(pt, dtype=np.float64).copy())
        return idx

    vol_map = [add(volume_nodes[i]) for i in range(len(volume_nodes))]
    dual_map = [add(dual_nodes[i]) for i in range(len(dual_nodes))]

    def remap(struts: np.ndarray, mapping: list[int]) -> np.ndarray:
        if len(struts) == 0:
            return np.empty((0, 2), dtype=np.int64)
        sset: set[tuple[int, int]] = set()
        for a, b in struts:
            ga, gb = mapping[int(a)], mapping[int(b)]
            if ga == gb:
                continue
            sset.add((min(ga, gb), max(ga, gb)))
        return (
            np.array(sorted(sset), dtype=np.int64)
            if sset
            else np.empty((0, 2), dtype=np.int64)
        )

    vol_s = remap(volume_struts, vol_map)
    dual_s = remap(dual_struts, dual_map)
    return np.asarray(out, dtype=np.float64), vol_s, dual_s


def _count_outside_nodes(nodes: np.ndarray, cad_mesh: trimesh.Trimesh) -> int:
    if len(nodes) == 0:
        return 0
    sd = safe_signed_distance(cad_mesh, nodes)
    return int(np.count_nonzero(sd > 1e-3))


def _collapsed_hex_count(
    hex_elems_before: np.ndarray,
    volume_nodes_morphed: np.ndarray,
    iron_ids: np.ndarray,
    rule_name: str,
    *,
    collapse_warn_ratio: float = 0.10,
) -> int:
    """
    For grid (corner DOFs): rebuild hex corners from morphed volume nodes and
    count collapsed bricks. For face-centroid rules, hex corners stay Cartesian
    so collapse count is 0 unless we cannot map.
    """
    rule = get_hex_topology_rule(rule_name)
    if rule.skin_mode != SKIN_MODE_CORNER_EDGE_CAGE:
        return 0
    # Volume nodes are hex corners for grid — morph deforms bricks.
    scaffold, hex_ids, _, _ = classify_boundary_from_hex_elems(hex_elems_before)
    morphed_scaffold = scaffold.copy()
    if len(volume_nodes_morphed):
        # Nearest scaffold corner for each volume node (grid: identity).
        for vi in range(len(volume_nodes_morphed)):
            d = np.linalg.norm(scaffold - volume_nodes_morphed[vi], axis=1)
            si = int(np.argmin(d))
            if d[si] < 1e-4:
                morphed_scaffold[si] = volume_nodes_morphed[vi]
    deformed = morphed_scaffold[hex_ids]
    vol_b = hex_volumes(hex_elems_before)
    vol_a = hex_volumes(deformed)
    ratio = np.abs(vol_a) / np.maximum(np.abs(vol_b), 1e-12)
    return int(np.count_nonzero(ratio < float(collapse_warn_ratio)))


def generate_compare_v3(
    cad_filepath: str | Path | trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray = (12.0, 12.0, 4.0),
    strut_radius: float = 0.6,
    rule_name: str = "octahedral",
    volume_fraction_threshold: float = 0.5,
    export_dir: str | Path | None = None,
    export_stem: str | None = None,
    skip_sweep: bool = False,
    solidify_untrimmed: bool = True,
    relax_iterations: int = 40,
    relax_alpha: float = 0.5,
    signed_distance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    k_rule: int | None = None,
    drop_volume_surface_struts: bool = False,
) -> dict[str, Any]:
    """
    Hybrid V3: loose dual shrinkwrap + VF volume + morph-to-dual + valence gate.

    Loose cull uses ``mode="boolean"`` (any face-centroid inside). Strict cull
    uses VF ``volume_fraction_threshold`` (default 0.5). Volume iron nodes morph
    to nearest dual nodes — not primary nearest-CAD morph.

    If ``drop_volume_surface_struts`` is True, volume struts that duplicate the
    surface/skin pattern are removed after morph/relax so the dual alone shows
    the skin (interior volume + iron→interior coupling remain).
    """
    t0 = time.time()
    if isinstance(cad_filepath, trimesh.Trimesh):
        cad_mesh = repair_cad_mesh(cad_filepath)
        part_name = "mesh"
    else:
        cad_mesh = repair_cad_mesh(trimesh.load(str(cad_filepath)))
        part_name = Path(str(cad_filepath)).stem

    rule = get_hex_topology_rule(rule_name)
    k = int(k_rule) if k_rule is not None else local_max_node_degree(rule.name)

    # --- 0/1 Loose complex → dual → shrinkwrap ---
    hex_loose, _grid, _surv_l, n_partial_loose = cull_hex_elements(
        cad_mesh,
        cell_size,
        volume_fraction_threshold=volume_fraction_threshold,
        mode="boolean",
        signed_distance_fn=signed_distance_fn,
    )
    dual_nodes_raw, dual_struts, dual_info = build_dual_from_hex_complex(
        hex_loose, rule.name
    )
    dual_nodes, dual_disp = shrinkwrap_dual_to_cad(dual_nodes_raw, cad_mesh)

    # --- 2 Strict complex → volume (Cartesian) ---
    hex_strict, _grid2, _surv_s, n_partial_strict = cull_hex_elements(
        cad_mesh,
        cell_size,
        volume_fraction_threshold=volume_fraction_threshold,
        mode="conformal",
        signed_distance_fn=signed_distance_fn,
    )
    volume_nodes, volume_struts, _rule = generate_sc_volume_topology(
        hex_strict, rule.name
    )
    iron_ids = volume_iron_node_ids(hex_strict, volume_nodes, rule.name)

    # --- 3/4 Morph volume → dual + valence gate ---
    volume_morphed, morph_report = morph_volume_iron_to_dual(
        volume_nodes, volume_struts, iron_ids, dual_nodes, k
    )
    volume_relaxed = _relax_volume_with_iron_fixed(
        volume_morphed,
        volume_struts,
        iron_ids,
        iterations=relax_iterations,
        alpha=relax_alpha,
    )

    n_collapsed = _collapsed_hex_count(
        hex_strict, volume_relaxed, iron_ids, rule.name
    )

    # --- Optional: strip surface-layer volume struts (keep interior + coupling) ---
    volume_struts_export = np.asarray(volume_struts, dtype=np.int64)
    drop_info: dict[str, Any] = {
        "drop_volume_surface_struts": bool(drop_volume_surface_struts),
        "n_volume_struts_before_drop": int(len(volume_struts_export)),
        "n_volume_surface_struts_dropped": 0,
        "n_volume_struts_dropped_coincident_dual": 0,
        "drop_volume_surface_rule": "disabled",
    }
    if drop_volume_surface_struts:
        volume_struts_export, drop_info_pre = remove_volume_surface_struts(
            hex_strict,
            volume_nodes,  # pre-morph IDs for edge identity
            volume_struts_export,
            iron_ids,
            rule.name,
        )
        drop_info.update(drop_info_pre)
        drop_info["drop_volume_surface_struts"] = True

    combined_nodes, vol_struts_m, dual_struts_m = _merge_graphs(
        volume_relaxed, volume_struts_export, dual_nodes, dual_struts
    )
    if drop_volume_surface_struts and len(vol_struts_m) and len(dual_struts_m):
        vol_struts_m, n_coin = drop_volume_struts_coincident_with_dual(
            vol_struts_m, dual_struts_m
        )
        drop_info["n_volume_struts_dropped_coincident_dual"] = int(n_coin)
        drop_info["n_volume_surface_struts_dropped"] = int(
            drop_info["n_volume_surface_struts_dropped"]
        ) + int(n_coin)

    all_struts = (
        np.vstack([vol_struts_m, dual_struts_m])
        if len(vol_struts_m) and len(dual_struts_m)
        else (vol_struts_m if len(vol_struts_m) else dual_struts_m)
    )

    n_outside = _count_outside_nodes(combined_nodes, cad_mesh)

    result: dict[str, Any] = {
        "rule_name": rule.name,
        "cell_size": tuple(float(x) for x in np.atleast_1d(cell_size).tolist())
        if not np.isscalar(cell_size)
        else (float(cell_size),) * 3,
        "strut_radius": float(strut_radius),
        "loose_cull_mode": "boolean",
        "loose_cull_note": "keep if n_inside_face_centroids >= 1",
        "n_hex_loose": int(len(hex_loose)),
        "n_hex_strict": int(len(hex_strict)),
        "n_partial_loose": int(n_partial_loose),
        "n_partial_strict": int(n_partial_strict),
        "n_dual_nodes": int(len(dual_nodes)),
        "n_dual_struts": int(len(dual_struts)),
        "n_volume_nodes": int(len(volume_relaxed)),
        "n_volume_struts": int(len(volume_struts_export)),
        "n_volume_struts_full": int(len(volume_struts)),
        "n_volume_iron": int(morph_report["n_volume_iron"]),
        "n_morph_to_dual": int(morph_report["n_morph_to_dual"]),
        "n_bonds_rejected_valence": int(morph_report["n_bonds_rejected_valence"]),
        "k_rule": int(k),
        "max_morph_distance": float(morph_report["max_morph_distance"]),
        "max_dual_shrinkwrap_distance": float(dual_disp.max()) if len(dual_disp) else 0.0,
        "n_outside_nodes": int(n_outside),
        "n_collapsed_hexes": int(n_collapsed),
        "nodes_count": int(len(combined_nodes)),
        "struts_count": int(len(all_struts)),
        "elapsed_time": float(time.time() - t0),
        "dual_info": dual_info,
        **drop_info,
        # Topology payloads for callers / tests
        "volume_nodes": volume_relaxed,
        "volume_struts": np.asarray(volume_struts_export, dtype=np.int64),
        "volume_struts_full": np.asarray(volume_struts, dtype=np.int64),
        "dual_nodes": dual_nodes,
        "dual_struts": np.asarray(dual_struts, dtype=np.int64),
        "combined_nodes": combined_nodes,
        "combined_volume_struts": vol_struts_m,
        "combined_dual_struts": dual_struts_m,
        "combined_struts": all_struts,
        "iron_ids": iron_ids,
        "hex_loose": hex_loose,
        "hex_strict": hex_strict,
        "cad_mesh": cad_mesh,
        "part_name": part_name,
    }

    if skip_sweep or not solidify_untrimmed:
        result["elapsed_time"] = float(time.time() - t0)
        return result

    # --- 5 Solidify: cylindrical volume + rectangular dual cage, union ---
    cage_width = 2.0 * float(strut_radius)
    cage_thickness = 0.5 * cage_width
    core = None
    if len(vol_struts_m) > 0:
        core, _ = union_lattice_with_spherical_joints(
            combined_nodes,
            vol_struts_m,
            float(strut_radius),
            joint_scale=1.05,
            cylinder_segments=12,
            sphere_segments=12,
        )
    skin_raw = None
    if len(dual_struts_m) > 0:
        skin_raw = generate_rectangular_surface_cage(
            combined_nodes,
            dual_struts_m,
            cad_mesh,
            width=cage_width,
            thickness=cage_thickness,
            normal_oversize=0.25,
            crop_to_boundary=False,
            project_stations=True,
        )

    lattice = None
    if core is not None and skin_raw is not None and len(skin_raw.faces) > 0:
        lattice = union_solid_meshes([core, skin_raw])
    elif core is not None:
        lattice = core
    elif skin_raw is not None:
        lattice = skin_raw
    else:
        lattice = trimesh.Trimesh()

    result["lattice_untrimmed"] = lattice
    result["elapsed_time"] = float(time.time() - t0)

    if export_dir is not None:
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        cs = result["cell_size"]
        stem = export_stem or (
            f"compare_v3_hybrid_{rule.name}_{int(cs[0])}x{int(cs[1])}x{int(cs[2])}"
            f"_untrimmed"
        )
        stl_path = out / f"{stem}.stl"
        lattice.export(stl_path)
        result["stl_path"] = str(stl_path)
        result["export_stem"] = stem

    return result


def write_compare_v3_report(result: dict[str, Any], path: str | Path) -> None:
    """Write ``*_report.txt`` with required compare keys."""
    cs = result.get("cell_size", ())
    drop_flag = bool(result.get("drop_volume_surface_struts", False))
    lines = [
        f"version=v3_hybrid",
        f"rule_name={result.get('rule_name')}",
        f"cell_size_xyz_mm={list(cs)}",
        f"strut_radius_mm={result.get('strut_radius')}",
        f"loose_cull_mode={result.get('loose_cull_mode')}",
        f"loose_cull_note={result.get('loose_cull_note')}",
        f"drop_volume_surface_struts={'true' if drop_flag else 'false'}",
        f"drop_volume_surface_rule={result.get('drop_volume_surface_rule', 'disabled')}",
        f"n_volume_struts_before_drop={result.get('n_volume_struts_before_drop', result.get('n_volume_struts'))}",
        f"n_volume_surface_struts_dropped={result.get('n_volume_surface_struts_dropped', 0)}",
        f"n_volume_struts_dropped_coincident_dual={result.get('n_volume_struts_dropped_coincident_dual', 0)}",
        f"n_hex_loose={result.get('n_hex_loose')}",
        f"n_hex_strict={result.get('n_hex_strict')}",
        f"n_dual_nodes={result.get('n_dual_nodes')}",
        f"n_dual_struts={result.get('n_dual_struts')}",
        f"n_volume_nodes={result.get('n_volume_nodes')}",
        f"n_volume_struts={result.get('n_volume_struts')}",
        f"n_volume_iron={result.get('n_volume_iron')}",
        f"n_morph_to_dual={result.get('n_morph_to_dual')}",
        f"n_bonds_rejected_valence={result.get('n_bonds_rejected_valence')}",
        f"k_rule={result.get('k_rule')}",
        f"nodes={result.get('nodes_count')}",
        f"struts={result.get('struts_count')}",
        f"n_outside_nodes={result.get('n_outside_nodes')}",
        f"n_collapsed_hexes={result.get('n_collapsed_hexes')}",
        f"max_morph_distance={result.get('max_morph_distance')}",
        f"max_dual_shrinkwrap_distance={result.get('max_dual_shrinkwrap_distance')}",
        f"elapsed_time_s={result.get('elapsed_time')}",
        "note=Volume iron morphs to nearest dual node (lateral OK); "
        "not primary nearest-CAD morph. Loose dual from boolean face-centroid cull.",
        "risk=Loose dual may be denser; valence gate can drop bonds → local gaps.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
