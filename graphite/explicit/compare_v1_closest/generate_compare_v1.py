"""V1 closest-point SC conformal compare baseline.

Thin orchestration of the existing VF-cull → closest morph → stamp path with
frozen compare constants. Does not alter ``morph_hex_scaffold`` defaults.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from graphite.explicit.conformal_core import (
    cull_hex_elements,
    generate_sc_volume_topology,
    morph_hex_scaffold,
    safe_signed_distance,
)
from graphite.explicit.geometry_module import union_lattice_with_spherical_joints
from graphite.explicit.mesh_repair import repair_cad_mesh

# Frozen fair-compare defaults (three-way index + V1 handoff).
DEFAULT_CELL_SIZE: tuple[float, float, float] = (12.0, 12.0, 4.0)
DEFAULT_STRUT_RADIUS: float = 0.6
DEFAULT_VOLUME_FRACTION: float = 0.5
DEFAULT_RULE_NAME: str = "grid"
DEFAULT_MAX_PROJECTION_FACTOR: float = 0.5
DEFAULT_STAIR_STEP_GATE_FACTOR: float = 1.0
PROJECTION_MODE: str = "closest"

def _stair_floor_limitation(
    cell_size: tuple[float, float, float],
    max_projection_factor: float | None,
    stair_step_normal_gate: bool,
) -> str:
    if max_projection_factor is None:
        budget = "unclamped"
    else:
        b = tuple(float(max_projection_factor) * float(c) for c in cell_size)
        budget = f"{b[0]:g}/{b[1]:g}/{b[2]:g} mm"
    if stair_step_normal_gate:
        return (
            "Known limitation: the stair-step normal gate rescues notch corners "
            "whose closest CAD point sits behind a face they do not expose, but "
            "corners whose aligned target simply exceeds the travel clamp still "
            f"stop short (per-axis budgets {budget} on cell {cell_size})."
        )
    return (
        "Known limitation: stair-side corners often snap to the nearest floor/top "
        "patch rather than the vertical wall; max_projection_factor also leaves "
        f"short insides (per-axis budgets {budget} on cell {cell_size})."
    )


def _load_cad(cad_filepath: str | Path | trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(cad_filepath, trimesh.Trimesh):
        return repair_cad_mesh(cad_filepath)
    return repair_cad_mesh(trimesh.load(str(cad_filepath), force="mesh"))


def _render_png(mesh: trimesh.Trimesh, path: Path, title: str) -> None:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 850))
    plotter.background_color = "white"
    plotter.add_mesh(
        pv.wrap(mesh),
        color="#2c3e50",
        show_edges=False,
        smooth_shading=True,
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.add_title(title, font_size=9)
    plotter.screenshot(str(path))
    plotter.close()


def _format_report_text(report: dict[str, Any]) -> str:
    lines = [f"{key}={report[key]}" for key in report]
    return "\n".join(lines) + "\n"


def generate_compare_v1(
    cad_filepath: str | Path | trimesh.Trimesh,
    *,
    cell_size: tuple[float, float, float] | list[float] | np.ndarray = DEFAULT_CELL_SIZE,
    strut_radius: float = DEFAULT_STRUT_RADIUS,
    rule_name: str = DEFAULT_RULE_NAME,
    volume_fraction_threshold: float = DEFAULT_VOLUME_FRACTION,
    max_projection_factor: float | None = DEFAULT_MAX_PROJECTION_FACTOR,
    snap_outside_nodes: bool = True,
    stair_step_normal_gate: bool = True,
    stair_step_gate_factor: float = DEFAULT_STAIR_STEP_GATE_FACTOR,
    cull_collapsed_hexes: bool = False,
    relax_iterations: int = 200,
    relax_alpha: float = 0.5,
    relax_mode: str = "laplacian",
    build_solid: bool = True,
    export_dir: str | Path | None = None,
    stem: str | None = None,
    write_render: bool = True,
    cad_path_label: str | None = None,
) -> dict[str, Any]:
    """Run the V1 closest-point conformal baseline.

    Always uses ``projection_mode="closest"`` (face_normal is out of scope for V1).

    Returns a dict with topology, optional solid mesh, scaffold morph metrics,
    a flat ``report`` suitable for ``*_report.txt``, and optional export paths.
    """
    start = time.time()
    cell = tuple(float(x) for x in np.asarray(cell_size, dtype=np.float64).ravel()[:3])
    if len(cell) != 3:
        raise ValueError(f"cell_size must have 3 components; got {cell_size!r}")

    cad = _load_cad(cad_filepath)
    if cad_path_label is None:
        if isinstance(cad_filepath, (str, Path)):
            cad_path_label = str(cad_filepath)
        else:
            cad_path_label = "mesh"

    hex_elems, _grid_nodes, _surviving, _n_partial = cull_hex_elements(
        cad,
        cell,
        volume_fraction_threshold=float(volume_fraction_threshold),
        mode="conformal",
    )

    hex_elems, scaffold_report = morph_hex_scaffold(
        hex_elems,
        cad,
        cell_size=cell,
        projection_mode=PROJECTION_MODE,
        max_projection_factor=max_projection_factor,
        snap_outside_nodes=bool(snap_outside_nodes),
        stair_step_normal_gate=bool(stair_step_normal_gate),
        stair_step_gate_factor=float(stair_step_gate_factor),
        cull_collapsed_hexes=bool(cull_collapsed_hexes),
        relax_iterations=int(relax_iterations),
        relax_alpha=float(relax_alpha),
        relax_mode=str(relax_mode),
    )

    nodes, volume_struts, rule = generate_sc_volume_topology(hex_elems, rule_name)
    nodes = np.asarray(nodes, dtype=np.float64)
    volume_struts = np.asarray(volume_struts, dtype=np.int64)

    # Outside-node count uses the same positive=outside convention as morph snap.
    sd = safe_signed_distance(cad, nodes)
    n_outside = int(np.sum(sd > 1e-3))
    elapsed = float(time.time() - start)

    lattice: trimesh.Trimesh | None = None
    if build_solid and len(volume_struts):
        lattice, _ = union_lattice_with_spherical_joints(
            nodes,
            volume_struts,
            float(strut_radius),
            joint_scale=1.05,
            cylinder_segments=12,
            sphere_segments=12,
        )
    elif build_solid:
        lattice = trimesh.Trimesh()

    report: dict[str, Any] = {
        "version": "v1_closest",
        "cad": cad_path_label,
        "cell_size_xyz_mm": list(cell),
        "rule_name": rule.name,
        "volume_fraction_threshold": float(volume_fraction_threshold),
        "projection_mode": PROJECTION_MODE,
        "max_projection_factor": max_projection_factor,
        "snap_outside_nodes": bool(snap_outside_nodes),
        "stair_step_normal_gate": bool(stair_step_normal_gate),
        "stair_step_gate_factor": float(stair_step_gate_factor),
        "n_stair_step_gate_redirects": int(
            scaffold_report.get("n_stair_step_gate_redirects", 0)
        ),
        "cull_collapsed_hexes": bool(cull_collapsed_hexes),
        "strut_radius_mm": float(strut_radius),
        "strut_diameter_mm": 2.0 * float(strut_radius),
        "surface_dual": "none",
        "cad_trim": "none",
        "nodes": int(len(nodes)),
        "volume_struts": int(len(volume_struts)),
        "outside_nodes": n_outside,
        "outside_nodes_tol_mm": 1e-3,
        "n_collapsed_hexes": int(scaffold_report.get("n_collapsed_hexes", 0)),
        "n_inverted_hexes": int(scaffold_report.get("n_inverted_hexes", 0)),
        "n_culled_hexes": int(scaffold_report.get("n_culled_hexes", 0)),
        "min_hex_volume_ratio": float(
            scaffold_report.get("min_hex_volume_ratio", 1.0)
        ),
        "max_projection_distance": float(
            scaffold_report.get("max_projection_distance", 0.0)
        ),
        "elapsed_s": elapsed,
        "limitation": _stair_floor_limitation(
            cell, max_projection_factor, bool(stair_step_normal_gate)
        ),
    }
    if lattice is not None:
        report["watertight"] = bool(lattice.is_watertight)
        try:
            report["lattice_volume_mm3"] = abs(float(lattice.volume))
        except Exception:
            report["lattice_volume_mm3"] = float("nan")

    paths: dict[str, str] = {}
    if export_dir is not None:
        out = Path(export_dir)
        out.mkdir(parents=True, exist_ok=True)
        file_stem = stem or "compare_v1_closest"
        report_path = out / f"{file_stem}_report.txt"
        report_path.write_text(_format_report_text(report), encoding="utf-8")
        paths["report"] = str(report_path)

        if lattice is not None:
            stl_path = out / f"{file_stem}.stl"
            lattice.export(stl_path)
            paths["stl"] = str(stl_path)
            if write_render:
                png_path = out / f"{file_stem}_render.png"
                title = (
                    f"V1 closest | {rule.name} | cell={cell[0]:g}x{cell[1]:g}x{cell[2]:g} | "
                    f"strut r={float(strut_radius):g} | untrimmed"
                )
                _render_png(lattice, png_path, title)
                paths["render"] = str(png_path)

    return {
        "nodes": nodes,
        "volume_struts": volume_struts,
        "skin_struts": np.empty((0, 2), dtype=np.int64),
        "rule_name": rule.name,
        "projection_mode": PROJECTION_MODE,
        "lattice": lattice,
        "scaffold_report": scaffold_report,
        "report": report,
        "paths": paths,
        "elapsed_s": elapsed,
    }
