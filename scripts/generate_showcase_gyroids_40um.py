"""Generate 3 showcase Gyroid models at 40um resolution for engine capability demonstration.

1. Raw Gyroid (5mm unit cell, 40um resolution, NO oversized boolean fix - shows native flying edges cut)
2. Micro-Grooved Gyroid (5mm unit cell, 40um resolution, 250um deep micro-grooves on CAD-trimmed scaffold)
3. Knurled Gyroid (5mm unit cell, 40um resolution, diamond knurling pattern on CAD-trimmed scaffold)
"""

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

from graphite.geometry.masking import axis_aligned_box_grid, axis_aligned_box_sdf
from graphite.math.tpms import gyroid
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.surface_textures import SurfaceTextureConfig, apply_surface_texture
from graphite.explicit.geometry_module import trimesh_to_manifold, manifold_to_trimesh


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "textures"
    out_dir.mkdir(parents=True, exist_ok=True)

    size_mm = 10.0      # 10mm specimen (2x2x2 unit cells)
    unit_cell_mm = 5.0  # 5mm unit cell requested
    res_mm = 0.040      # 40um resolution requested
    solid_fraction = 0.33
    half = size_mm / 2.0

    print("=" * 80)
    print("GRAPHITE ENGINE SHOWCASE: 3 GYROID SCAFFOLDS AT 40 µm RESOLUTION")
    print("=" * 80)
    print(f"Specimen Size        : {size_mm:.1f} x {size_mm:.1f} x {size_mm:.1f} mm")
    print(f"Unit Cell Period (L) : {unit_cell_mm:.1f} mm (2x2x2 periodic array)")
    print(f"Solid Fraction       : {solid_fraction*100:.0f}%")
    print(f"Voxel Resolution     : {res_mm*1e3:.0f} µm ({res_mm} mm)")
    print(f"Texture Depth        : 250 µm (0.250 mm peak-to-valley)")
    print("=" * 80)

    # 1. Setup Grid and Distance Field
    t0 = time.perf_counter()
    print("\n[1/4] Generating 40 µm voxel grid and analytical bounds...")
    X, Y, Z, origin, spacing = axis_aligned_box_grid(
        size_mm, size_mm, size_mm, res_mm,
        origin_x=-half, origin_y=-half, origin_z=-half,
        pad_voxels=3,
    )
    cad_sdf = axis_aligned_box_sdf(
        X, Y, Z,
        origin_x=-half, origin_y=-half, origin_z=-half,
        width_x_mm=size_mm, depth_y_mm=size_mm, height_z_mm=size_mm,
    )
    F = gyroid(X, Y, Z, unit_cell_size=unit_cell_mm, iso_offset=solid_fraction, is_sheet=True)
    print(f"Grid generated: {X.shape} ({X.size:,} voxels) in {time.perf_counter()-t0:.2f}s")

    # -------------------------------------------------------------------------
    # MODEL 1: RAW GYROID (NO OVERSIZED BOOLEAN FIX)
    # -------------------------------------------------------------------------
    print("\n[2/4] Generating Model 1: Raw Gyroid (Native Flying Edges boundary cut)...")
    t_m1 = time.perf_counter()
    raw_field = np.maximum(F, cad_sdf)
    iso_raw = extract_isosurface(raw_field, spacing=spacing, origin=origin, enforce_watertight=True)
    mesh_raw = iso_raw.mesh
    m1_path = out_dir / "gyroid_5mm_uc_40um_raw_no_boolean_trim.stl"
    mesh_raw.export(str(m1_path))
    t_m1_s = time.perf_counter() - t_m1
    print(f"  Exported: {m1_path.name}")
    print(f"  Faces: {len(mesh_raw.faces):,}, Vol: {mesh_raw.volume:.2f} mm³, Watertight: {mesh_raw.is_watertight} in {t_m1_s:.2f}s")

    # Base Mesh with Exact CAD Boolean Trim for Texturing
    print("\n[3/4] Generating Base Scaffold with Exact CAD B-Rep Boolean Trim...")
    t_trim = time.perf_counter()
    box_cad = trimesh.creation.box(extents=[size_mm, size_mm, size_mm])
    cad_sdf_dil = cad_sdf - (3.0 * res_mm)  # 3-voxel normal level set dilation
    field_dil = np.maximum(F, cad_sdf_dil)
    iso_dil = extract_isosurface(field_dil, spacing=spacing, origin=origin, enforce_watertight=True)
    man_tpms = trimesh_to_manifold(iso_dil.mesh)
    man_cad = trimesh_to_manifold(box_cad)
    base_trimmed = manifold_to_trimesh(man_tpms ^ man_cad)
    trimesh.repair.fix_normals(base_trimmed)
    t_trim_s = time.perf_counter() - t_trim
    print(f"  Base trimmed scaffold ready: {len(base_trimmed.faces):,} faces in {t_trim_s:.2f}s")

    # -------------------------------------------------------------------------
    # MODEL 2: 250 µm DEEP MICRO-GROOVES
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Model 2: Gyroid with 250 µm Micro-Grooves...")
    t_m2 = time.perf_counter()
    cfg_grooves = SurfaceTextureConfig(
        texture_type="microgrooves",
        amplitude_mm=0.125,  # +/- 125 um = 250 um total peak-to-valley depth
        wavelength_mm=0.600, # 600 um pitch
        direction=(0.0, 0.0, 1.0),
        profile="sine",
        displacement_mode="centered",
        project_transverse_normal=True,
        target_edge_length_mm=res_mm,
        max_faces=5_000_000,
    )
    mesh_grooves = apply_surface_texture(base_trimmed, cfg_grooves)
    m2_path = out_dir / "gyroid_5mm_uc_40um_250um_microgrooves.stl"
    mesh_grooves.export(str(m2_path))
    t_m2_s = time.perf_counter() - t_m2
    print(f"  Exported: {m2_path.name}")
    print(f"  Faces: {len(mesh_grooves.faces):,}, Vol: {mesh_grooves.volume:.2f} mm³, Watertight: {mesh_grooves.is_watertight} in {t_m2_s:.2f}s")

    # -------------------------------------------------------------------------
    # MODEL 3: DIAMOND KNURLING
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating Model 3: Gyroid with Diamond Knurling...")
    t_m3 = time.perf_counter()
    cfg_knurl = SurfaceTextureConfig(
        texture_type="knurling",
        amplitude_mm=0.125,  # +/- 125 um = 250 um total depth
        wavelength_mm=0.800, # 800 um diamond facet pitch
        target_edge_length_mm=res_mm,
        max_faces=5_000_000,
    )
    mesh_knurl = apply_surface_texture(base_trimmed, cfg_knurl)
    m3_path = out_dir / "gyroid_5mm_uc_40um_diamond_knurling.stl"
    mesh_knurl.export(str(m3_path))
    t_m3_s = time.perf_counter() - t_m3
    print(f"  Exported: {m3_path.name}")
    print(f"  Faces: {len(mesh_knurl.faces):,}, Vol: {mesh_knurl.volume:.2f} mm³, Watertight: {mesh_knurl.is_watertight} in {t_m3_s:.2f}s")

    # -------------------------------------------------------------------------
    # VISUAL RENDERS FOR PRESENTATION
    # -------------------------------------------------------------------------
    print("\nGenerating high-resolution comparative renders...")
    # Triple-panel overview
    p = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 600))

    p.subplot(0, 0)
    p.background_color = "white"
    p.add_mesh(pv.wrap(mesh_raw), color="#7f8c8d", smooth_shading=True)
    p.add_text("1. Raw Gyroid\n(Native flying edges boundary cut)", font_size=10, color="black")
    p.view_isometric()

    p.subplot(0, 1)
    p.background_color = "white"
    p.add_mesh(pv.wrap(mesh_grooves), color="#2980b9", smooth_shading=True)
    p.add_text("2. 250 µm Micro-Grooves\n(CAD-trimmed boundary + axial channels)", font_size=10, color="black")
    p.view_isometric()

    p.subplot(0, 2)
    p.background_color = "white"
    p.add_mesh(pv.wrap(mesh_knurl), color="#27ae60", smooth_shading=True)
    p.add_text("3. Diamond Knurling\n(CAD-trimmed boundary + cross-hatch ridges)", font_size=10, color="black")
    p.view_isometric()

    overview_png = out_dir / "showcase_3gyroids_40um_comparison.png"
    p.screenshot(str(overview_png))
    p.close()
    print(f"Saved overview render to {overview_png}")

    # Close-up comparison of the boundary edge (Raw vs Exact Trimmed)
    p_edge = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1400, 700))
    p_edge.subplot(0, 0)
    p_edge.background_color = "white"
    p_edge.add_mesh(pv.wrap(mesh_raw), color="#e74c3c", smooth_shading=True)
    p_edge.add_text("Raw Gyroid Boundary (No Boolean Fix)\nNotice jagged/ragged facet cuts along CAD faces", font_size=10, color="black")
    p_edge.camera_position = [(12, 12, 12), (5, 5, 5), (0, 0, 1)]
    p_edge.camera.zoom(1.6)

    p_edge.subplot(0, 1)
    p_edge.background_color = "white"
    p_edge.add_mesh(pv.wrap(base_trimmed), color="#2ecc71", smooth_shading=True)
    p_edge.add_text("Exact CAD Trim (Our Boolean Fix)\nCrisp, planar CAD faces & razor-sharp edges", font_size=10, color="black")
    p_edge.camera_position = [(12, 12, 12), (5, 5, 5), (0, 0, 1)]
    p_edge.camera.zoom(1.6)

    edge_png = out_dir / "showcase_cad_trim_edge_comparison.png"
    p_edge.screenshot(str(edge_png))
    p_edge.close()
    print(f"Saved boundary edge comparison to {edge_png}")

    # Close-up of Textures (Micro-Grooves vs Knurling)
    p_tex = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1400, 700))
    p_tex.subplot(0, 0)
    p_tex.background_color = "white"
    p_tex.add_mesh(pv.wrap(mesh_grooves), color="#2980b9", smooth_shading=True)
    p_tex.add_text("250 µm Micro-Grooves Surface Detail\nCell alignment / Osseointegration micro-channels", font_size=10, color="black")
    p_tex.camera_position = [(6, 6, 6), (0, 0, 0), (0, 0, 1)]
    p_tex.camera.zoom(2.0)

    p_tex.subplot(0, 1)
    p_tex.background_color = "white"
    p_tex.add_mesh(pv.wrap(mesh_knurl), color="#27ae60", smooth_shading=True)
    p_tex.add_text("Diamond Knurling Detail\nMechanical anti-slip / High friction micro-ridges", font_size=10, color="black")
    p_tex.camera_position = [(6, 6, 6), (0, 0, 0), (0, 0, 1)]
    p_tex.camera.zoom(2.0)

    tex_png = out_dir / "showcase_microgroove_vs_knurl_detail.png"
    p_tex.screenshot(str(tex_png))
    p_tex.close()
    print(f"Saved texture detail comparison to {tex_png}")

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print("SHOWCASE GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"1. Raw Gyroid (No Boolean Fix) : {m1_path.name} ({m1_path.stat().st_size/(1024**2):.1f} MB, {len(mesh_raw.faces):,} faces)")
    print(f"2. 250 µm Micro-Grooves         : {m2_path.name} ({m2_path.stat().st_size/(1024**2):.1f} MB, {len(mesh_grooves.faces):,} faces)")
    print(f"3. Diamond Knurling            : {m3_path.name} ({m3_path.stat().st_size/(1024**2):.1f} MB, {len(mesh_knurl.faces):,} faces)")
    print(f"Total Execution Time           : {total_time:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
