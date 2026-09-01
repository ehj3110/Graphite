"""
Volume mesh inspection utilities (Gmsh Tet4 + quality gates + debug VTU).

Used by ``scripts/aristo_volume_mesh_inspect.py`` and
``scripts/aristo_clean_and_mesh_stl.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_solver import _generate_fea_mesh_single_surface, _sanitize_trimesh_for_gmsh
from graphite.aristo.gmsh_lattice_mesh import generate_lattice_fea_mesh_from_stl
from graphite.aristo.mesh_quality import (
    build_mesh_quality_report,
    build_quality_mask,
    mesh_has_giant_elements,
)
from graphite.aristo.stl_surface_clean import maybe_clean_stl_surface


def remove_floating_islands(
    mesh: trimesh.Trimesh,
    *,
    min_volume_mm3: float = 1e-6,
    min_volume_fraction: float = 1e-5,
    min_face_fraction: float = 1e-4,
) -> tuple[trimesh.Trimesh, dict]:
    """Drop disconnected shells much smaller than the main lattice body."""
    parts = mesh.split(only_watertight=False)
    parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    if len(parts) <= 1:
        return mesh, {
            "n_components_before": len(parts),
            "n_components_after": 1,
            "n_islands_removed": 0,
            "removed_faces": 0,
            "removed_shells": [],
        }

    main_faces = max(len(p.faces) for p in parts)
    main_vol = max(float(p.volume) for p in parts)
    vol_floor = max(float(min_volume_mm3), main_vol * float(min_volume_fraction))
    face_floor = max(64, int(main_faces * float(min_face_fraction)))

    kept: list[trimesh.Trimesh] = []
    removed_shells: list[dict] = []
    for part in parts:
        n_faces = int(len(part.faces))
        vol = float(part.volume)
        if n_faces >= face_floor and vol >= vol_floor:
            kept.append(part)
        else:
            removed_shells.append(
                {
                    "faces": n_faces,
                    "volume_mm3": vol,
                    "watertight": bool(part.is_watertight),
                    "bounds_mm": part.bounds.tolist(),
                }
            )

    if not kept:
        kept = [parts[0]]

    cleaned = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)
    cleaned.remove_unreferenced_vertices()
    return cleaned, {
        "n_components_before": len(parts),
        "n_components_after": len(kept),
        "n_islands_removed": len(removed_shells),
        "removed_faces": int(len(mesh.faces) - len(cleaned.faces)),
        "removed_shells": removed_shells,
        "face_floor": face_floor,
        "volume_floor_mm3": vol_floor,
    }


def write_mesh_quality_vtu(
    nodes: np.ndarray,
    elems: np.ndarray,
    aspects: np.ndarray,
    max_edges: np.ndarray,
    quality_ok: np.ndarray,
    output_path: Path,
) -> None:
    n_cells = elems.shape[0]
    cells = np.column_stack([np.full(n_cells, 4, dtype=np.int64), elems]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    grid.cell_data["aspect_ratio"] = np.asarray(aspects, dtype=np.float64)
    grid.cell_data["max_edge_mm"] = np.asarray(max_edges, dtype=np.float64)
    grid.cell_data["quality_ok"] = quality_ok.astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path))


def inspect_volume_mesh(
    mesh: trimesh.Trimesh,
    *,
    h: float,
    mesh_mode: str = "single_surface",
    clean_stl: bool = False,
    allow_single_surface_fallback: bool = False,
    char_length_min: float | None = None,
    char_length_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, AristoConfig]:
    """
    Volume-mesh ``mesh`` and return nodes, elems, surface tris, quality report, config.
    """
    h = float(h)
    mesh = _sanitize_trimesh_for_gmsh(mesh, allow_face_removal=True)
    config = AristoConfig(
        fea_mesh_resolution=h,
        fea_gmsh_mesh_mode=mesh_mode,
        fea_gmsh_adaptive_curvature=mesh_mode == "tpms" or clean_stl,
        fea_gmsh_char_length_min=float(char_length_min if char_length_min is not None else max(0.03, h * 0.5)),
        fea_gmsh_char_length_max=float(char_length_max if char_length_max is not None else max(0.15, h * 2.5)),
        fea_gmsh_min_elements_per_two_pi=15,
        fea_gmsh_classify_only=(
            mesh_mode == "tpms" and not allow_single_surface_fallback
        ),
        fea_clean_stl_surface=bool(clean_stl),
    )

    if clean_stl:
        mesh = maybe_clean_stl_surface(mesh, h, config=config)

    mesh_meta: dict = {"mesh_mode": mesh_mode}
    if mesh_mode == "tpms":
        nodes, elems, surf = generate_lattice_fea_mesh_from_stl(
            mesh, h, config=config, silent=False, mesh_meta=mesh_meta
        )
    else:
        nodes, elems, surf = _generate_fea_mesh_single_surface(mesh, h, silent=False)
        mesh_meta["algorithm_3d"] = 1
        mesh_meta["optimize_netgen"] = 0

    mask, _, aspects, max_edges = build_quality_mask(nodes, elems, config)
    giants, max_edge, vol_ratio = mesh_has_giant_elements(nodes, elems, config)
    report = build_mesh_quality_report(
        nodes, elems, config, quality_mask=mask, remesh_attempts=1
    )
    report["mesh_meta_gmsh"] = mesh_meta
    report["stl_faces"] = int(len(mesh.faces))
    report["n_nodes"] = int(len(nodes))
    report["n_elements"] = int(len(elems))
    report["n_surface_tris"] = int(len(surf))
    report["has_giant_elements"] = bool(giants)
    report["vol_median_ratio"] = float(vol_ratio)
    report["max_edge_mm"] = float(max_edge)
    return nodes, elems, surf, report, config


def export_volume_mesh_inspection(
    mesh: trimesh.Trimesh,
    *,
    h: float,
    out_dir: Path,
    stem: str,
    mesh_mode: str = "single_surface",
    clean_stl: bool = False,
    allow_single_surface_fallback: bool = False,
    extra_report: dict | None = None,
) -> dict:
    """Run inspection and write ``{stem}_volume_mesh_report.json`` + debug VTU."""
    nodes, elems, _surf, report, config = inspect_volume_mesh(
        mesh,
        h=h,
        mesh_mode=mesh_mode,
        clean_stl=clean_stl,
        allow_single_surface_fallback=allow_single_surface_fallback,
    )
    mask, _, aspects, max_edges = build_quality_mask(nodes, elems, config)
    if extra_report:
        report.update(extra_report)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_volume_mesh_report.json"
    vtu_path = out_dir / f"{stem}_volume_mesh_debug.vtu"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_mesh_quality_vtu(nodes, elems, aspects, max_edges, mask, vtu_path)
    report["report_json"] = str(json_path)
    report["debug_vtu"] = str(vtu_path)
    return report
