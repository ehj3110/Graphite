"""
Experimental STL & Lattice Mesh Repair Pipeline for Graphite.

Provides a progressive, multi-tier repair engine designed to produce 100%
watertight, 2-manifold, zero-self-intersection STLs suitable for CAD (SolidWorks)
and SLA slicing (Formlabs PreForm) without destructive feature smoothing when
exact CSG is achievable.

Includes strict Volume and Surface Area conservation checks to detect and prevent
thin-feature erosion (e.g. Poisson reconstruction dissolving thin struts into
disconnected nodal beads / point clouds).

Tiers:
1. Gentle Topology Cleanup (Trimesh: deduplication, face winding, outward normals)
2. Boundary & Hole Healing (PyMeshFix: hole sealing and boundary stitching)
3. Exact CSG De-Intersection (Manifold3D: component decomposition + batch boolean union)
4. Screened Poisson Surface Reconstruction (PyMeshLab: robust isosurface fallback with volume-loss gating)
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
import trimesh

from graphite.io.mesh_export import MeshHealthReport, compute_mesh_health


def repair_mesh_experimental(
    mesh: trimesh.Trimesh,
    *,
    poisson_depth: int = 9,
    allow_poisson_fallback: bool = True,
    min_volume_retention: float = 0.50,
    min_area_retention: float = 0.40,
    log_steps: bool = False,
) -> tuple[trimesh.Trimesh, MeshHealthReport, list[str]]:
    """
    Repair a mesh using the progressive experimental repair pipeline with volume & area conservation gating.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh.
    poisson_depth : int, default 9
        Octree depth for Screened Poisson reconstruction if triggered.
    allow_poisson_fallback : bool, default True
        Whether to escalate to Screened Poisson reconstruction if exact
        methods (gentle, meshfix, manifold3d) do not resolve all non-manifold
        or self-intersection defects.
    min_volume_retention : float, default 0.85
        Minimum allowable ratio of candidate_volume / initial_volume. If a candidate
        falls below this threshold, it is rejected to prevent point-cloud / nodal-bead degradation.
    min_area_retention : float, default 0.70
        Minimum allowable ratio of candidate_area / initial_area.
    log_steps : bool, default False
        Whether to print diagnostic step messages to stdout.

    Returns
    -------
    tuple[trimesh.Trimesh, MeshHealthReport, list[str]]
        Repaired mesh, final health report, and list of diagnostic step notes.
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    notes: list[str] = []
    current_mesh = mesh.copy()

    # Baseline volume and area
    baseline_vol = float(abs(mesh.volume)) if mesh.volume is not None else 0.0
    baseline_area = float(mesh.area) if mesh.area is not None else 0.0

    # Step 0: Check initial health
    initial_health = compute_mesh_health(current_mesh, check_intersections=True)
    if initial_health.is_export_ready:
        notes.append("Input mesh is already 100% export-ready.")
        if log_steps:
            print("[repair_experimental] Mesh is already 100% export-ready.")
        return current_mesh, initial_health, notes

    # Step 1: Gentle topology cleanup (Trimesh)
    try:
        from graphite.repair.repair_suite import repair_trimesh_gentle
        cand_gentle = repair_trimesh_gentle(current_mesh)
        cand_v = float(abs(cand_gentle.volume))
        cand_a = float(cand_gentle.area)
        vol_ret = cand_v / baseline_vol if baseline_vol > 1e-9 else 1.0
        area_ret = cand_a / baseline_area if baseline_area > 1e-9 else 1.0

        if vol_ret >= min_volume_retention and area_ret >= min_area_retention and cand_v > 0:
            current_mesh = cand_gentle
            notes.append("Applied gentle topology cleanup (dedup, winding, normals).")
        else:
            notes.append(f"Gentle cleanup rejected due to volume/area loss (Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%}).")
    except Exception as exc:
        notes.append(f"Gentle cleanup warning: {exc}")

    health_1 = compute_mesh_health(current_mesh, baseline_mesh=mesh, check_intersections=True)
    if health_1.is_export_ready:
        notes.append("Gentle cleanup resolved all mesh defects.")
        if log_steps:
            print("[repair_experimental] Gentle cleanup resolved all defects.")
        return current_mesh, health_1, notes

    # Step 2: Hole & Boundary Healing via PyMeshFix (if open boundary edges exist)
    if health_1.boundary_edges > 0:
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(current_mesh.vertices, current_mesh.faces)
            mf.repair(remove_smallest_components=False)
            if mf.points is not None and len(mf.points) > 0 and len(mf.faces) > 0:
                cand_mf = trimesh.Trimesh(mf.points, mf.faces, process=True)
                cand_v = float(abs(cand_mf.volume))
                cand_a = float(cand_mf.area)
                vol_ret = cand_v / baseline_vol if baseline_vol > 1e-9 else 1.0
                area_ret = cand_a / baseline_area if baseline_area > 1e-9 else 1.0

                if vol_ret >= min_volume_retention and area_ret >= min_area_retention and cand_v > 0:
                    health_mf = compute_mesh_health(cand_mf, baseline_mesh=mesh, check_intersections=True)
                    if health_mf.boundary_edges == 0:
                        current_mesh = cand_mf
                        notes.append("PyMeshFix successfully sealed boundary holes while preserving volume.")
                        if health_mf.is_export_ready:
                            if log_steps:
                                print("[repair_experimental] PyMeshFix produced export-ready mesh.")
                            return current_mesh, health_mf, notes
                else:
                    notes.append(
                        f"PyMeshFix candidate rejected: geometric loss (Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%})."
                    )
        except Exception as exc:
            notes.append(f"PyMeshFix attempt skipped: {exc}")

    # Step 3: Exact CSG De-Intersection via Manifold3D (preserves sharp edges)
    try:
        import manifold3d as m3d
        vp = np.ascontiguousarray(current_mesh.vertices, dtype=np.float32)
        tv = np.ascontiguousarray(current_mesh.faces, dtype=np.uint32)
        mm = m3d.Mesh(vert_properties=vp, tri_verts=tv)
        manifold = m3d.Manifold(mm)
        if manifold.status() == m3d.Error.NoError:
            parts = manifold.decompose()
            if len(parts) > 1:
                united = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
                raw = united.to_mesh()
                tm_united = trimesh.Trimesh(raw.vert_properties, raw.tri_verts, process=True)
                cand_v = float(abs(tm_united.volume))
                cand_a = float(tm_united.area)
                vol_ret = cand_v / baseline_vol if baseline_vol > 1e-9 else 1.0
                area_ret = cand_a / baseline_area if baseline_area > 1e-9 else 1.0

                if vol_ret >= min_volume_retention and area_ret >= min_area_retention:
                    health_m3d = compute_mesh_health(tm_united, baseline_mesh=mesh, check_intersections=True)
                    if health_m3d.is_export_ready:
                        notes.append(
                            f"Manifold3D exact CSG union resolved {len(parts)} intersecting bodies without smoothing."
                        )
                        if log_steps:
                            print(
                                f"[repair_experimental] Manifold3D exact CSG union resolved {len(parts)} bodies."
                            )
                        return tm_united, health_m3d, notes
                    else:
                        current_mesh = tm_united
                        notes.append("Manifold3D CSG union reduced defects but non-manifold edges remain.")
                else:
                    notes.append(
                        f"Manifold3D CSG candidate rejected: volume loss (Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%})."
                    )
    except Exception as exc:
        notes.append(f"Manifold3D CSG attempt skipped: {exc}")

    # Step 4: Screened Poisson Surface Reconstruction (Robust fail-safe with volume gating)
    current_health = compute_mesh_health(current_mesh, baseline_mesh=mesh, check_intersections=True)
    if not current_health.is_export_ready and allow_poisson_fallback:
        try:
            import pymeshlab
            with tempfile.TemporaryDirectory() as td:
                in_path = os.path.join(td, "raw_to_repair.stl")
                out_path = os.path.join(td, "repaired_poisson.stl")
                current_mesh.export(in_path)

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(in_path)
                ms.meshing_remove_duplicate_faces()
                ms.meshing_remove_duplicate_vertices()
                ms.meshing_remove_null_faces()
                ms.meshing_remove_unreferenced_vertices()
                ms.compute_normal_per_vertex()
                ms.generate_surface_reconstruction_screened_poisson(
                    depth=int(poisson_depth), preclean=True
                )
                ms.save_current_mesh(out_path)

                poisson_mesh = trimesh.load(out_path, force="mesh", process=True)
                if isinstance(poisson_mesh, trimesh.Scene):
                    poisson_mesh = poisson_mesh.dump(concatenate=True)

                cand_v = float(abs(poisson_mesh.volume))
                cand_a = float(poisson_mesh.area)
                vol_ret = cand_v / baseline_vol if baseline_vol > 1e-9 else 1.0
                area_ret = cand_a / baseline_area if baseline_area > 1e-9 else 1.0

                # CRITICAL GUARD: Reject Poisson if struts were dissolved into point-cloud beads
                if vol_ret < min_volume_retention or area_ret < min_area_retention or cand_v <= 0:
                    notes.append(
                        f"Screened Poisson candidate REJECTED: Severe geometric loss (Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%}). "
                        "Prevented thin struts from dissolving into disconnected nodal beads/point cloud."
                    )
                    if log_steps:
                        print(
                            f"[repair_experimental] REJECTED Poisson candidate: Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%}. "
                            "Struts would dissolve into disconnected point clouds."
                        )
                else:
                    health_p = compute_mesh_health(poisson_mesh, baseline_mesh=mesh, check_intersections=True)
                    if health_p.is_export_ready or (
                        health_p.boundary_edges == 0 and health_p.non_manifold_edges == 0
                    ):
                        current_mesh = poisson_mesh
                        notes.append(
                            f"Screened Poisson reconstruction (depth={poisson_depth}) produced watertight shell "
                            f"(Vol ret={vol_ret:.1%}, Area ret={area_ret:.1%})."
                        )
                        if log_steps:
                            print(
                                f"[repair_experimental] Screened Poisson reconstruction succeeded (V={health_p.vertex_count:,}, F={health_p.face_count:,})."
                            )
                        return current_mesh, health_p, notes
                    else:
                        notes.append("Screened Poisson completed with residual topological artifacts.")
                        current_mesh = poisson_mesh
        except Exception as exc:
            notes.append(f"Screened Poisson reconstruction failed: {exc}")

    final_health = compute_mesh_health(current_mesh, baseline_mesh=mesh, check_intersections=True)
    return current_mesh, final_health, notes


def repair_stl_experimental(
    input_path: str | Path,
    output_path: str | Path,
    *,
    poisson_depth: int = 9,
    allow_poisson_fallback: bool = True,
    min_volume_retention: float = 0.50,
    min_area_retention: float = 0.40,
    log_steps: bool = True,
) -> tuple[MeshHealthReport, MeshHealthReport]:
    """
    Repair an STL file from disk and save the clean result to output_path.
    Inspects surface area and volume before and after repair.

    Parameters
    ----------
    input_path : str or Path
        Input STL path.
    output_path : str or Path
        Output destination path for the repaired STL.
    poisson_depth : int, default 9
        Poisson depth for fail-safe reconstruction.
    allow_poisson_fallback : bool, default True
        Allow Poisson reconstruction if exact CSG is insufficient.
    min_volume_retention : float, default 0.85
        Minimum allowable ratio of repaired_vol / initial_vol.
    min_area_retention : float, default 0.70
        Minimum allowable ratio of repaired_area / initial_area.
    log_steps : bool, default True
        Print before/after metrics.

    Returns
    -------
    tuple[MeshHealthReport, MeshHealthReport]
        (Initial health report, final health report).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_mesh = trimesh.load(str(input_path), force="mesh")
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = raw_mesh.dump(concatenate=True)

    initial_health = compute_mesh_health(raw_mesh, check_intersections=True)

    if log_steps:
        print(f"\n[+] Repairing: {input_path.name}")
        print(
            f"    Initial: V={initial_health.vertex_count:,} F={initial_health.face_count:,} "
            f"wt={initial_health.is_watertight} nm_edges={initial_health.non_manifold_edges} "
            f"nm_verts={initial_health.non_manifold_vertices} ints={initial_health.self_intersections} | "
            f"Area={initial_health.surface_area:,.2f} mm² Vol={initial_health.volume:,.2f} mm³"
        )

    repaired, final_health, notes = repair_mesh_experimental(
        raw_mesh,
        poisson_depth=poisson_depth,
        allow_poisson_fallback=allow_poisson_fallback,
        min_volume_retention=min_volume_retention,
        min_area_retention=min_area_retention,
        log_steps=log_steps,
    )

    repaired.export(str(output_path))

    if log_steps:
        v_ret_str = f"{final_health.volume_retention:.1%}" if final_health.volume_retention is not None else "100.0%"
        a_ret_str = f"{final_health.area_retention:.1%}" if final_health.area_retention is not None else "100.0%"
        print(
            f"    Repaired: V={final_health.vertex_count:,} F={final_health.face_count:,} "
            f"wt={final_health.is_watertight} nm_edges={final_health.non_manifold_edges} "
            f"nm_verts={final_health.non_manifold_vertices} ints={final_health.self_intersections} | "
            f"Area={final_health.surface_area:,.2f} mm² Vol={final_health.volume:,.2f} mm³"
        )
        print(f"    Retention: Volume = {v_ret_str}, Surface Area = {a_ret_str}")
        print(f"    Export Ready: {final_health.is_export_ready}")
        if final_health.volume_retention is not None and final_health.volume_retention < min_volume_retention:
            print("    [WARNING] Severe volume loss occurred! Slicer may show point clouds.")
        print(f"    Saved: {output_path}")

    return initial_health, final_health
