"""V2 surface-first compare pipeline: Cartesian volume + dual + gated stitch."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from graphite.explicit.conformal_core import (
    cull_hex_elements,
    generate_sc_volume_topology,
    safe_signed_distance,
)
from graphite.explicit.geometry_module import (
    boolean_intersect_with_cad,
    generate_rectangular_surface_cage,
    keep_largest_solid_component,
    union_lattice_with_spherical_joints,
    union_solid_meshes,
)

from .clip import clip_volume_graph_to_cad
from .dual import build_surface_native_dual
from .stitch import (
    orphan_cull_volume_graph,
    stitch_cut_to_dual,
    unit_cell_length_horizontal,
)


def _as_cell_tuple(
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    cs = np.asarray(cell_size, dtype=np.float64).ravel()
    if cs.size == 1:
        v = float(cs[0])
        return (v, v, v)
    if cs.size != 3:
        raise ValueError(f"cell_size must be scalar or length-3; got {cs}")
    return (float(cs[0]), float(cs[1]), float(cs[2]))


def _merge_node_tables(
    vol_nodes: np.ndarray,
    dual_nodes: np.ndarray,
    *,
    round_decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate volume + dual with optional coincident merge; return maps."""
    vol_nodes = np.asarray(vol_nodes, dtype=np.float64)
    dual_nodes = np.asarray(dual_nodes, dtype=np.float64)
    node_list: list[np.ndarray] = []
    key_to_idx: dict[tuple[float, float, float], int] = {}

    def add(pt: np.ndarray) -> int:
        key = tuple(np.round(pt, round_decimals).tolist())
        if key in key_to_idx:
            return key_to_idx[key]
        idx = len(node_list)
        node_list.append(np.asarray(pt, dtype=np.float64).copy())
        key_to_idx[key] = idx
        return idx

    vol_map = np.array([add(vol_nodes[i]) for i in range(len(vol_nodes))], dtype=np.int64)
    dual_map = np.array(
        [add(dual_nodes[i]) for i in range(len(dual_nodes))], dtype=np.int64
    )
    nodes = (
        np.vstack(node_list) if node_list else np.empty((0, 3), dtype=np.float64)
    )
    return nodes, vol_map, dual_map


def _format_report(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def generate_compare_v2(
    cad_mesh: trimesh.Trimesh | str | Path,
    *,
    cell_size: float | tuple[float, float, float] = (12.0, 12.0, 4.0),
    volume_fraction_threshold: float = 0.5,
    rule_name: str = "grid",
    strut_radius: float = 0.6,
    surface_cage_width: float | None = None,
    surface_cage_thickness: float | None = None,
    surface_cage_normal_oversize: float = 0.25,
    stitch_max_horizontal: float | None = None,
    stitch_max_angle_from_vertical_deg: float = 60.0,
    cut_inset_factor: float = 0.0,
    dual_mode: str = "face_centroid",
    stitch_gates: str = "strict",
    valence_cap: int | None = None,
    boolean_trim_volume: bool = True,
    boolean_trim_union: bool = False,
    export_dir: str | Path | None = None,
    stem: str | None = None,
    export_stages: bool = False,
    skip_solidify: bool = False,
    cylinder_segments: int = 10,
    version_label: str = "v2_surface_first",
) -> dict[str, Any]:
    """
    Surface-first SC conformal compare (V2 / V2.1).

    Volume: VF cull → stamp undeformed topology → graph-clip to CAD → cut nodes.
    Dual: independent exposed-face dual placed surface-natively → rectangular cage.
    Stitch: gated (V2) or nearest-dual (V2.1 ``stitch_gates='off'``); orphan cull.
    Does **not** call ``morph_hex_scaffold`` / closest-point cage morph.
    """
    t0 = time.perf_counter()
    if isinstance(cad_mesh, (str, Path)):
        cad = trimesh.load(str(cad_mesh), force="mesh")
        cad_path = str(cad_mesh)
    else:
        cad = cad_mesh
        cad_path = "<mesh>"

    cell_xyz = _as_cell_tuple(cell_size)
    L = float(
        stitch_max_horizontal
        if stitch_max_horizontal is not None
        else unit_cell_length_horizontal(cell_xyz)
    )
    cage_w = float(2.0 * strut_radius if surface_cage_width is None else surface_cage_width)
    cage_t = float(0.5 * cage_w if surface_cage_thickness is None else surface_cage_thickness)

    # --- A. Volume (no conformal morph) ---
    hex_elems, _grid_nodes, _surviving, n_partial = cull_hex_elements(
        cad,
        cell_size=cell_xyz,
        volume_fraction_threshold=float(volume_fraction_threshold),
        mode="conformal",
    )
    # Stamp on undeformed hexes — identity scaffold (no morph_hex_scaffold).
    vol_nodes_raw, vol_struts_raw, rule = generate_sc_volume_topology(
        hex_elems, rule_name=rule_name
    )

    vol_nodes, vol_struts, cut_ids, clip_report = clip_volume_graph_to_cad(
        vol_nodes_raw,
        vol_struts_raw,
        cad,
        cut_inset_factor=float(cut_inset_factor),
    )

    # --- B. Independent dual ---
    dual_nodes, dual_struts, dual_normals, dual_report = build_surface_native_dual(
        hex_elems, cad, dual_mode=str(dual_mode)
    )

    # --- C. Stitch ---
    cut_pts = vol_nodes[cut_ids] if len(cut_ids) else np.empty((0, 3), dtype=np.float64)
    stitch_local, stitch_report = stitch_cut_to_dual(
        cut_pts,
        dual_nodes,
        L=L,
        max_angle_deg=float(stitch_max_angle_from_vertical_deg),
        search_radius=max(L, float(max(cell_xyz)) * 2.0),
        dual_outward_normals=dual_normals,
        angle_mode="inward_normal",
        stitch_gates=str(stitch_gates),
        valence_cap=valence_cap,
    )

    orphan_vol_ids = (
        cut_ids[np.asarray(stitch_report.orphaned_cut_ids, dtype=np.int64)]
        if stitch_report.orphaned_cut_ids
        else np.empty(0, dtype=np.int64)
    )
    vol_nodes_cull, vol_struts_cull, old_to_new = orphan_cull_volume_graph(
        vol_nodes, vol_struts, orphan_vol_ids
    )

    global_nodes, vol_map, dual_map = _merge_node_tables(vol_nodes_cull, dual_nodes)

    def remap_struts(struts: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        if len(struts) == 0:
            return np.empty((0, 2), dtype=np.int64)
        out = []
        for a, b in struts:
            ga, gb = int(mapping[int(a)]), int(mapping[int(b)])
            if ga == gb:
                continue
            out.append((min(ga, gb), max(ga, gb)))
        return (
            np.asarray(sorted(set(out)), dtype=np.int64)
            if out
            else np.empty((0, 2), dtype=np.int64)
        )

    g_vol_struts = remap_struts(vol_struts_cull, vol_map)
    g_dual_struts = remap_struts(dual_struts, dual_map)

    # Stitch: local (cut_local_i, dual_local_j) where cut_local indexes cut_pts.
    g_stitch: list[tuple[int, int]] = []
    for ci, dj in stitch_local:
        old_cut = int(cut_ids[int(ci)])
        new_vol = int(old_to_new[old_cut])
        if new_vol < 0:
            continue
        ga = int(vol_map[new_vol])
        gb = int(dual_map[int(dj)])
        if ga != gb:
            g_stitch.append((min(ga, gb), max(ga, gb)))
    g_stitch_struts = (
        np.asarray(sorted(set(g_stitch)), dtype=np.int64)
        if g_stitch
        else np.empty((0, 2), dtype=np.int64)
    )

    # Combined cylinder struts = volume (post-cull) + stitches (dual uses cage).
    cyl_struts = (
        np.vstack([g_vol_struts, g_stitch_struts])
        if len(g_stitch_struts)
        else g_vol_struts
    )

    elapsed_topo = time.perf_counter() - t0

    # Outside-node count on volume (post-cull).
    if len(vol_nodes_cull):
        sd_vol = safe_signed_distance(cad, vol_nodes_cull)
        n_outside = int(np.count_nonzero(sd_vol > 1e-3))
    else:
        n_outside = 0

    result: dict[str, Any] = {
        "version": str(version_label),
        "cad_path": cad_path,
        "cell_size": cell_xyz,
        "rule_name": rule.name if hasattr(rule, "name") else rule_name,
        "volume_fraction_threshold": float(volume_fraction_threshold),
        "strut_radius": float(strut_radius),
        "unit_cell_length_used": L,
        "cut_inset_factor": float(cut_inset_factor),
        "dual_mode": str(dual_mode),
        "stitch_gates": str(stitch_gates),
        "valence_cap": valence_cap,
        "hex_elems": hex_elems,
        "n_hex": int(len(hex_elems)),
        "n_partial_hex": int(n_partial),
        "nodes": global_nodes,
        "volume_nodes_raw": vol_nodes_raw,
        "volume_struts_raw": vol_struts_raw,
        "volume_nodes_clipped": vol_nodes,
        "volume_struts_clipped": vol_struts,
        "cut_ids": cut_ids,
        "volume_nodes": vol_nodes_cull,
        "volume_struts": vol_struts_cull,
        "dual_nodes": dual_nodes,
        "dual_struts": dual_struts,
        "stitch_struts": g_stitch_struts,
        "global_volume_struts": g_vol_struts,
        "global_dual_struts": g_dual_struts,
        "n_cut": int(stitch_report.n_cut),
        "n_stitched": int(stitch_report.n_stitched),
        "n_rejected_distance": int(stitch_report.n_rejected_distance),
        "n_rejected_angle": int(stitch_report.n_rejected_angle),
        "n_rejected_valence": int(getattr(stitch_report, "n_rejected_valence", 0)),
        "n_orphaned": int(stitch_report.n_orphaned),
        "n_outside_volume_nodes": n_outside,
        "clip_report": clip_report,
        "dual_report": dual_report,
        "stitch_report": stitch_report,
        "elapsed_topology_s": elapsed_topo,
        "morph_used": False,
        "projection_mode": "none_volume_cartesian",
        "notes": [
            "Volume stamped on undeformed hexes; no morph_hex_scaffold / closest cage morph.",
            f"Cut nodes via chord raycast (SDF bisect fallback) + inset={cut_inset_factor:g} "
            f"(mean cut SDF={clip_report.mean_cut_sdf:.4f}; "
            f"raycast={clip_report.n_crossings_raycast}, "
            f"bisect={clip_report.n_crossings_bisect}).",
            f"Dual mode={dual_mode} ({dual_report.placement}).",
            f"Stitch gates={stitch_gates}; L={L:g}; valence_cap={valence_cap}.",
        ],
    }

    mesh_untrimmed = trimesh.Trimesh()
    mesh_trimmed = trimesh.Trimesh()
    cage_mesh = trimesh.Trimesh()
    solidify_s = 0.0
    trim_s = 0.0

    if not skip_solidify:
        t_sol = time.perf_counter()
        core = trimesh.Trimesh()
        if len(cyl_struts) > 0:
            core, _ = union_lattice_with_spherical_joints(
                global_nodes,
                cyl_struts,
                float(strut_radius),
                joint_scale=1.05,
                cylinder_segments=int(cylinder_segments),
                sphere_segments=int(cylinder_segments),
            )
        if len(dual_struts) > 0:
            cage_raw = generate_rectangular_surface_cage(
                dual_nodes,
                dual_struts,
                cad,
                width=cage_w,
                thickness=cage_t,
                normal_oversize=float(surface_cage_normal_oversize),
                crop_to_boundary=False,
                project_stations=True,
            )
            cage_mesh, _ = boolean_intersect_with_cad(cage_raw, cad)
        else:
            cage_raw = trimesh.Trimesh()

        parts = [m for m in (core, cage_raw) if m is not None and len(getattr(m, "faces", [])) > 0]
        if parts:
            mesh_untrimmed = union_solid_meshes(parts) if len(parts) > 1 else parts[0]
        else:
            mesh_untrimmed = trimesh.Trimesh()
        solidify_s = time.perf_counter() - t_sol

        if boolean_trim_union and len(mesh_untrimmed.faces) > 0:
            t_tr = time.perf_counter()
            mesh_trimmed, _ = boolean_intersect_with_cad(mesh_untrimmed, cad)
            mesh_trimmed = keep_largest_solid_component(mesh_trimmed, label="v2 union")
            trim_s = time.perf_counter() - t_tr
        elif boolean_trim_volume and len(core.faces) > 0:
            # Volume Boolean trim separately; dual already flush-capable via cage ∩ CAD.
            t_tr = time.perf_counter()
            core_trim, _ = boolean_intersect_with_cad(core, cad)
            cage_use = cage_mesh if len(cage_mesh.faces) else cage_raw
            mesh_trimmed = (
                union_solid_meshes([core_trim, cage_use])
                if len(cage_use.faces)
                else core_trim
            )
            mesh_trimmed = keep_largest_solid_component(mesh_trimmed, label="v2 trim")
            trim_s = time.perf_counter() - t_tr
        else:
            mesh_trimmed = mesh_untrimmed

    result["mesh_untrimmed"] = mesh_untrimmed
    result["mesh_trimmed"] = mesh_trimmed
    result["mesh_dual_cage"] = cage_mesh
    result["elapsed_solidify_s"] = solidify_s
    result["elapsed_trim_s"] = trim_s
    result["elapsed_total_s"] = time.perf_counter() - t0

    report_text = _format_report(
        [
            f"compare_version={version_label}",
            f"cad={cad_path}",
            f"cad_extents_mm={np.asarray(cad.extents, dtype=float).tolist()}",
            f"cell_size_xyz_mm={list(cell_xyz)}",
            f"rule_name={result['rule_name']}",
            f"volume_fraction_threshold={volume_fraction_threshold}",
            f"strut_radius_mm={float(strut_radius):.9f}",
            f"strut_diameter_mm={2.0 * float(strut_radius):.9f}",
            f"surface_cage_width_mm={cage_w:.9f}",
            f"surface_cage_thickness_mm={cage_t:.9f}",
            "morph_hex_scaffold=false",
            "projection_mode=none_volume_cartesian",
            f"cut_inset_factor={float(cut_inset_factor):.9f}",
            f"dual_mode={dual_mode}",
            f"stitch_gates={stitch_gates}",
            f"valence_cap={valence_cap}",
            f"unit_cell_length_used={L:.9f}",
            f"stitch_max_angle_from_vertical_deg={float(stitch_max_angle_from_vertical_deg)}",
            f"n_hex={result['n_hex']}",
            f"n_partial_hex={result['n_partial_hex']}",
            f"n_volume_nodes={len(vol_nodes_cull)}",
            f"n_volume_struts={len(vol_struts_cull)}",
            f"n_dual_nodes={len(dual_nodes)}",
            f"n_dual_struts={len(dual_struts)}",
            f"n_stitch_struts={len(g_stitch_struts)}",
            f"n_cut={stitch_report.n_cut}",
            f"n_stitched={stitch_report.n_stitched}",
            f"n_rejected_distance={stitch_report.n_rejected_distance}",
            f"n_rejected_angle={stitch_report.n_rejected_angle}",
            f"n_rejected_valence={getattr(stitch_report, 'n_rejected_valence', 0)}",
            f"n_orphaned={stitch_report.n_orphaned}",
            f"n_outside_volume_nodes={n_outside}",
            f"clip_method={clip_report.method}",
            f"clip_mean_cut_sdf={clip_report.mean_cut_sdf:.9f}",
            f"clip_crossings_raycast={clip_report.n_crossings_raycast}",
            f"clip_crossings_bisect={clip_report.n_crossings_bisect}",
            f"clip_struts_interior={clip_report.n_struts_kept_interior}",
            f"clip_struts_clipped={clip_report.n_struts_clipped}",
            f"clip_struts_dropped={clip_report.n_struts_dropped_outside}",
            f"dual_placement={dual_report.placement}",
            f"dual_n_projected={dual_report.n_projected}",
            f"dual_n_fallback_closest={dual_report.n_projection_fallback_closest}",
            f"elapsed_topology_s={elapsed_topo:.3f}",
            f"elapsed_solidify_s={solidify_s:.3f}",
            f"elapsed_trim_s={trim_s:.3f}",
            f"elapsed_total_s={result['elapsed_total_s']:.3f}",
            f"untrimmed_faces={len(mesh_untrimmed.faces)}",
            f"trimmed_faces={len(mesh_trimmed.faces)}",
            f"watertight_untrimmed={bool(getattr(mesh_untrimmed, 'is_watertight', False))}",
            f"watertight_trimmed={bool(getattr(mesh_trimmed, 'is_watertight', False))}",
            "--- notes ---",
            *result["notes"],
        ]
    )
    result["report_text"] = report_text

    if export_dir is not None:
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        base = stem or str(version_label)
        (out / f"{base}_report.txt").write_text(report_text, encoding="utf-8")
        if len(mesh_untrimmed.faces):
            mesh_untrimmed.export(out / f"{base}_untrimmed.stl")
        if len(mesh_trimmed.faces):
            mesh_trimmed.export(out / f"{base}_trimmed.stl")
        if len(cage_mesh.faces):
            cage_mesh.export(out / f"{base}_dual_cage.stl")
        if export_stages:
            # Lightweight point/edge stage dumps for debugging.
            if len(vol_nodes_raw) and len(vol_struts_raw):
                _export_wire_stl(
                    out / f"{base}_stage_volume_unclipped.stl",
                    vol_nodes_raw,
                    vol_struts_raw,
                    float(strut_radius) * 0.35,
                )
            if len(vol_nodes) and len(vol_struts):
                _export_wire_stl(
                    out / f"{base}_stage_volume_clipped.stl",
                    vol_nodes,
                    vol_struts,
                    float(strut_radius) * 0.35,
                )
            if len(dual_nodes) and len(dual_struts):
                _export_wire_stl(
                    out / f"{base}_stage_dual.stl",
                    dual_nodes,
                    dual_struts,
                    float(strut_radius) * 0.35,
                )
            if len(g_stitch_struts):
                _export_wire_stl(
                    out / f"{base}_stage_stitches.stl",
                    global_nodes,
                    g_stitch_struts,
                    float(strut_radius) * 0.35,
                )
        result["export_dir"] = str(out)
        result["export_stem"] = base

    return result


def _export_wire_stl(
    path: Path,
    nodes: np.ndarray,
    struts: np.ndarray,
    radius: float,
) -> None:
    """Best-effort thin cylinder dump for stage previews."""
    try:
        mesh, _ = union_lattice_with_spherical_joints(
            np.asarray(nodes, dtype=np.float64),
            np.asarray(struts, dtype=np.int64),
            float(radius),
            joint_scale=1.05,
            cylinder_segments=8,
            sphere_segments=8,
        )
        if mesh is not None and len(getattr(mesh, "faces", [])):
            mesh.export(path)
    except Exception:
        pass
