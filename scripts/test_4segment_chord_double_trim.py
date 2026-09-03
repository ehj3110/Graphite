"""Test 4-segment chord with +25% oversize and double-sided Boolean trim on mouse wrist rest."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import pyvista as pv
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
    out_dir = REPO_ROOT / "outputs" / "wrist_rest_4seg_trim"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("4-SEGMENT CHORD (+25% OVERSIZE) DOUBLE-SIDED BOOLEAN TRIM TEST")
    print("=" * 80)
    print(f"CAD File             : {cad_path.name}")
    print("Chord Segments       : 4 segments (3 surface-projected stations per bar)")
    print("Target Strut Depth   : 1.20 mm (Constant Inset Shell)")
    print("Target Strut Width   : 1.20 mm")
    print("Blank Oversize       : +25% (+0.30 mm Outward, +0.30 mm Inward -> Total 1.50 mm depth)")
    print("Core Strut Diameter  : 1.20 mm (Radius: 0.60 mm)")
    print("=" * 80)

    cad = trimesh.load(str(cad_path), force="mesh")
    cell_size = (12.0, 12.0, 4.0)
    strut_radius = 0.6
    cage_width = 1.20
    target_skin_thickness = 1.20  # mm

    # 25% larger on both sides:
    # Outward oversize: +25% of 1.20 mm = +0.30 mm
    # Inward depth: 1.20 mm + 0.30 mm = 1.50 mm (so blank extends 0.30 mm past the inner cut)
    normal_oversize = 0.30
    blank_thickness = target_skin_thickness + normal_oversize  # 1.50 mm

    # 1. Scaffold Generation
    print("\n[1/5] Generating conformal scaffold...")
    t0 = time.perf_counter()
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
    print("\n[2/5] Computing CAD SDF and extracting 1.20 mm inset shell boundary...")
    t_sdf = time.perf_counter()
    sdf_res = 0.25  # 250 um resolution for high boundary fidelity
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
    print(f"Inset CAD extracted in {time.perf_counter()-t_sdf:.2f}s: {len(cad_inset_mesh.faces):,} faces, watertight={cad_inset_mesh.is_watertight}")

    # 3. Loft 4-Segment Rectangular Blanks (+25% Oversize)
    print("\n[3/5] Loftying 4-segment surface cage blanks (+25% oversize)...")
    t_loft = time.perf_counter()
    raw_cage = generate_rectangular_surface_cage(
        nodes,
        skin_struts,
        cad,
        width=cage_width,
        thickness=blank_thickness,     # 1.50 mm
        normal_oversize=normal_oversize,  # 0.30 mm outward
        n_segments=4,                  # 4 segments per chord
        project_stations=True,         # Projected stations match the curved skin
        crop_to_boundary=False,
    )
    t_loft_s = time.perf_counter() - t_loft
    print(f"4-segment raw blanks lofted in {t_loft_s:.2f}s: {len(raw_cage.faces):,} faces")

    # 4. Manifold3D Double-Sided Boolean Trim
    print("\n[4/5] Executing Manifold3D Double-Sided Boolean CSG...")
    t_trim = time.perf_counter()
    man_cad = trimesh_to_manifold(cad)
    man_inset = trimesh_to_manifold(cad_inset_mesh)
    man_raw = trimesh_to_manifold(raw_cage)

    # Clean dual = (raw ∩ CAD_outer) \ CAD_inset
    man_trimmed = (man_raw ^ man_cad) - man_inset
    clean_dual = manifold_to_trimesh(man_trimmed)
    trimesh.repair.fix_normals(clean_dual)
    clean_dual = keep_largest_solid_component(clean_dual, label="clean_4seg_dual")
    t_trim_s = time.perf_counter() - t_trim

    dual_stl = out_dir / "wrist_rest_surface_dual_4segment_clean.stl"
    clean_dual.export(str(dual_stl))
    print(f"Double-sided trimmed dual in {t_trim_s:.2f}s: {len(clean_dual.faces):,} faces, volume={clean_dual.volume:.2f} mm^3, watertight={clean_dual.is_watertight}")

    # 5. Build Core Lattice and Fuse
    print("\n[5/5] Building core volume lattice and fusing assembly...")
    t_core = time.perf_counter()
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
    man_combined = man_trimmed + man_core_clipped
    combined_mesh = manifold_to_trimesh(man_combined)
    trimesh.repair.fix_normals(combined_mesh)
    combined_mesh = keep_largest_solid_component(combined_mesh, label="4seg_assembly")
    t_core_s = time.perf_counter() - t_core

    assembly_stl = out_dir / "wrist_rest_octahedral_4segment_assembly.stl"
    combined_mesh.export(str(assembly_stl))
    print(f"Combined assembly fused in {t_core_s:.2f}s: {len(combined_mesh.faces):,} faces, volume={combined_mesh.volume:.2f} mm^3, watertight={combined_mesh.is_watertight}")

    # 6. Visual Comparisons
    print("\nGenerating visual comparisons (cross-sections and 3D renders)...")
    # A) High-resolution cross-section at x=0
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, sharey=True)

    sl_dual = clean_dual.section(plane_origin=[0, 0, 7], plane_normal=[1, 0, 0])
    if sl_dual is not None:
        for entity in sl_dual.discrete:
            axes[0].plot(entity[:, 1], entity[:, 2], color="royalblue", lw=1.2)
    axes[0].set_title("Clean 4-Segment Surface Dual (Exact 1.2 mm Shell, Perfect Curve Tracking)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Z (mm)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    sl_comb = combined_mesh.section(plane_origin=[0, 0, 7], plane_normal=[1, 0, 0])
    if sl_comb is not None:
        for entity in sl_comb.discrete:
            axes[1].plot(entity[:, 1], entity[:, 2], color="forestgreen", lw=1.2)
    axes[1].set_title("Combined Assembly (4-Segment Dual + Core Octahedral Lattice)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Y (mm)")
    axes[1].set_ylabel("Z (mm)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    slice_png = out_dir / "wrist_rest_4seg_cross_section.png"
    plt.savefig(str(slice_png), dpi=200)
    plt.close()
    print(f"Saved cross section to {slice_png}")

    # B) 3D Isometric Render
    p = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1400, 700))
    p.subplot(0, 0)
    p.background_color = "white"
    p.add_mesh(pv.wrap(clean_dual), color="#3498db", smooth_shading=True)
    p.add_text("4-Segment Surface Dual (+25% Trimmed to 1.2mm)", font_size=10, color="black")
    p.view_isometric()

    p.subplot(0, 1)
    p.background_color = "white"
    p.add_mesh(pv.wrap(combined_mesh), color="#2c3e50", smooth_shading=True)
    p.add_text("Welded Assembly (4-Seg Dual + Octahedral Core)", font_size=10, color="black")
    p.view_isometric()

    render_png = out_dir / "wrist_rest_4seg_3d_render.png"
    p.screenshot(str(render_png))
    p.close()
    print(f"Saved 3D render to {render_png}")

    print("\n" + "=" * 80)
    print("4-SEGMENT CHORD BENCHMARK METRICS SUMMARY")
    print("=" * 80)
    print(f"Raw Blank Faces          : {len(raw_cage.faces):,}")
    print(f"Loft Time                : {t_loft_s:.2f}s")
    print(f"Double-Sided CSG Time    : {t_trim_s:.2f}s")
    print(f"Trimmed Dual Faces       : {len(clean_dual.faces):,}")
    print(f"Trimmed Dual Volume      : {clean_dual.volume:.2f} mm^3")
    print(f"Dual Watertight          : {clean_dual.is_watertight}")
    print(f"Assembly Faces           : {len(combined_mesh.faces):,}")
    print(f"Assembly Volume          : {combined_mesh.volume:.2f} mm^3")
    print(f"Assembly Watertight      : {combined_mesh.is_watertight}")
    print(f"Total Pipeline Time      : {time.perf_counter()-t0:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
