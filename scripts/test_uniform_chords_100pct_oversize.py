"""Benchmark 4-chord and 8-chord uniform surface duals with 100% oversize double-sided trim."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyvista as pv
import trimesh
import manifold3d

from graphite.explicit.conformal_generator import generate_conformal_lattice
from graphite.explicit.geometry_module import (
    generate_rectangular_surface_cage,
    keep_largest_solid_component,
    manifold_to_trimesh,
    trimesh_to_manifold,
    union_lattice_with_spherical_joints,
)
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.meshing_backends import extract_isosurface


def main() -> None:
    cad_path = REPO_ROOT / "test_parts" / "Mouse wrist rest v1.stl"
    out_dir = REPO_ROOT / "outputs" / "wrist_rest_uniform_100pct"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("UNIFORM 4-CHORD VS 8-CHORD DUAL WITH 100% OVERSIZE DOUBLE-SIDED TRIM")
    print("=" * 80)
    print(f"CAD File             : {cad_path.name}")
    print("Target Skin Depth    : 1.20 mm")
    print("Target Strut Width   : 1.20 mm")
    print("Blank Oversize       : +100% (+1.20 mm Outward, +1.20 mm Inward -> 3.60 mm total)")
    print("Outer Blank Limit    : +1.20 mm beyond CAD surface")
    print("Inner Blank Limit    : -2.40 mm depth (1.20 mm past inner cut)")
    print("=" * 80)

    cad = trimesh.load(str(cad_path), force="mesh")
    cell_size = (12.0, 12.0, 4.0)
    strut_radius = 0.6
    cage_width = 1.20
    target_skin_thickness = 1.20
    oversize = 1.20  # 100% oversize

    # 1. Conformal Scaffold
    t0 = time.perf_counter()
    print("\n[1/4] Generating conformal scaffold...")
    result = generate_conformal_lattice(
        cad_filepath=str(cad_path),
        cell_size=cell_size,
        strut_radius=strut_radius,
        lattice_type="SC",
        rule_name="octahedral",
        export_dir=str(out_dir),
        mode="conformal",
        skip_sweep=True,
        volume_fraction_threshold=0.5,
        relax_mode="laplacian",
        relax_iterations=200,
        relax_alpha=0.5,
        max_projection_factor=0.5,
        surface_cage_profile="rectangular",
        surface_cage_width=cage_width,
        surface_cage_thickness=target_skin_thickness,
    )
    nodes = np.asarray(result["nodes_relaxed"], dtype=np.float64)
    volume_struts = np.asarray(result["volume_struts"], dtype=np.int64)
    skin_struts = np.asarray(result["skin_struts"], dtype=np.int64)
    print(f"Scaffold ready in {time.perf_counter()-t0:.2f}s: {len(nodes)} nodes, {len(skin_struts)} skin struts")

    # 2. Extract Inset CAD Surface at level = -1.20 mm
    print("\n[2/4] Extracting CAD SDF at -1.20 mm...")
    t_sdf = time.perf_counter()
    sdf_res = 0.25
    _, _, _, cad_sdf, origin, _, _, _, _ = voxelize_mesh_and_edt(
        cad, resolution=sdf_res, pad_width=4
    )
    iso_inset = extract_isosurface(
        cad_sdf,
        spacing=(sdf_res, sdf_res, sdf_res),
        origin=origin,
        level=-target_skin_thickness,
        enforce_watertight=True,
    )
    cad_inset_mesh = iso_inset.mesh
    print(f"Inset CAD extracted in {time.perf_counter()-t_sdf:.2f}s: {len(cad_inset_mesh.faces):,} faces")

    man_cad = trimesh_to_manifold(cad)
    man_inset = trimesh_to_manifold(cad_inset_mesh)

    # Core volume lattice
    print("\nBuilding core volume lattice...")
    core, _ = union_lattice_with_spherical_joints(
        nodes,
        volume_struts,
        strut_radius,
        joint_scale=1.05,
        cylinder_segments=12,
        sphere_segments=12,
    )
    man_core = trimesh_to_manifold(core)
    man_core_clipped = man_core ^ man_cad

    cases = [
        {"name": "4-Chord Uniform (100% Oversize)", "n_segments": 4, "id": "4chord"},
        {"name": "8-Chord Uniform (100% Oversize)", "n_segments": 8, "id": "8chord"},
    ]

    bench_results = []

    for c in cases:
        print(f"\n[3/4] Processing {c['name']}...")
        t_loft = time.perf_counter()
        raw_cage = generate_rectangular_surface_cage(
            nodes,
            skin_struts,
            cad,
            width=cage_width,
            thickness=target_skin_thickness + oversize,  # 2.40 mm
            normal_oversize=oversize,                    # 1.20 mm
            n_segments=c["n_segments"],
            project_stations=True,
            crop_to_boundary=False,
        )
        t_loft_s = time.perf_counter() - t_loft
        print(f"  Raw cage lofted: {len(raw_cage.faces):,} faces in {t_loft_s:.2f}s")

        t_trim = time.perf_counter()
        man_raw = trimesh_to_manifold(raw_cage)
        man_trimmed = (man_raw ^ man_cad) - man_inset
        clean_dual = manifold_to_trimesh(man_trimmed)
        trimesh.repair.fix_normals(clean_dual)
        clean_dual = keep_largest_solid_component(clean_dual, label=c["id"])
        splits = clean_dual.split(only_watertight=False)
        if len(splits) > 1:
            clean_dual = max(splits, key=lambda s: s.volume)
        t_trim_s = time.perf_counter() - t_trim

        dual_stl = out_dir / f"wrist_rest_dual_{c['id']}_100pct.stl"
        clean_dual.export(str(dual_stl))
        print(f"  Trimmed dual: {len(clean_dual.faces):,} faces, vol={clean_dual.volume:.2f} mm^3 in {t_trim_s:.2f}s (watertight={clean_dual.is_watertight})")

        # Combine with core
        t_comb = time.perf_counter()
        man_comb = man_trimmed + man_core_clipped
        comb_mesh = manifold_to_trimesh(man_comb)
        trimesh.repair.fix_normals(comb_mesh)
        comb_mesh = keep_largest_solid_component(comb_mesh, label=f"{c['id']}_assembly")
        splits_c = comb_mesh.split(only_watertight=False)
        if len(splits_c) > 1:
            comb_mesh = max(splits_c, key=lambda s: s.volume)
        t_comb_s = time.perf_counter() - t_comb

        assembly_stl = out_dir / f"wrist_rest_assembly_{c['id']}_100pct.stl"
        comb_mesh.export(str(assembly_stl))
        print(f"  Assembly fused: {len(comb_mesh.faces):,} faces, vol={comb_mesh.volume:.2f} mm^3 in {t_comb_s:.2f}s (watertight={comb_mesh.is_watertight})")

        bench_results.append({
            **c,
            "raw_faces": len(raw_cage.faces),
            "loft_time": t_loft_s,
            "trim_time": t_trim_s,
            "dual_faces": len(clean_dual.faces),
            "dual_volume": clean_dual.volume,
            "dual_watertight": clean_dual.is_watertight,
            "assembly_faces": len(comb_mesh.faces),
            "assembly_volume": comb_mesh.volume,
            "assembly_watertight": comb_mesh.is_watertight,
            "dual_mesh": clean_dual,
            "assembly_mesh": comb_mesh,
        })

    # Visual renders
    print("\n[4/4] Generating comparative visual renders...")
    # Side-by-side Duals
    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 800))
    p.subplot(0, 0)
    p.background_color = "white"
    p.add_mesh(pv.wrap(bench_results[0]["dual_mesh"]), color="#2980b9", smooth_shading=True)
    p.add_text("4-Chord Uniform Dual (100% Oversize)", font_size=10, color="black")
    p.camera_position = [(-70, -35, 20), (-40, -15, 6), (0, 0, 1)]
    p.camera.zoom(1.4)

    p.subplot(0, 1)
    p.background_color = "white"
    p.add_mesh(pv.wrap(bench_results[1]["dual_mesh"]), color="#27ae60", smooth_shading=True)
    p.add_text("8-Chord Uniform Dual (100% Oversize)", font_size=10, color="black")
    p.camera_position = [(-70, -35, 20), (-40, -15, 6), (0, 0, 1)]
    p.camera.zoom(1.4)

    out_dual_png = out_dir / "wrist_rest_4chord_vs_8chord_dual_comparison.png"
    p.screenshot(str(out_dual_png))
    p.close()
    print(f"Saved dual comparison to {out_dual_png}")

    # Side-by-side Full Assemblies
    p2 = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1600, 800))
    p2.subplot(0, 0)
    p2.background_color = "white"
    p2.add_mesh(pv.wrap(bench_results[0]["assembly_mesh"]), color="#2c3e50", smooth_shading=True)
    p2.add_text("4-Chord Assembly (100% Oversize)", font_size=10, color="black")
    p2.view_isometric()

    p2.subplot(0, 1)
    p2.background_color = "white"
    p2.add_mesh(pv.wrap(bench_results[1]["assembly_mesh"]), color="#2c3e50", smooth_shading=True)
    p2.add_text("8-Chord Assembly (100% Oversize)", font_size=10, color="black")
    p2.view_isometric()

    out_comb_png = out_dir / "wrist_rest_4chord_vs_8chord_assembly_comparison.png"
    p2.screenshot(str(out_comb_png))
    p2.close()
    print(f"Saved assembly comparison to {out_comb_png}")

    print("\n" + "=" * 80)
    print("UNIFORM CHORD (100% OVERSIZE) BENCHMARK SUMMARY")
    print("=" * 80)
    for r in bench_results:
        print(f"{r['name']:<35} | Raw Faces: {r['raw_faces']:>7,} | Dual Faces: {r['dual_faces']:>7,} | Dual Vol: {r['dual_volume']:>7.1f} mm³ | Assembly Vol: {r['assembly_volume']:>7.1f} mm³ | Watertight: {r['dual_watertight']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
