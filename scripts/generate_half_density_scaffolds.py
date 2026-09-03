"""
Generate half-density micropillar scaffold benchmarks:
1. Cylindrical Disk (5mm dia x 2mm h, 1mm unit cell Gyroid, 33% SF, 30um res):
   - 100um dia x 400um height hairs at 400um spacing (943 hairs vs 1,868 original, ~50% density)
2. Cube Scaffold (5mm cube, 1mm pore Gyroid, 33% SF, 30um res):
   - 50um dia x 200um height hairs at 280um spacing (~4,200 hairs vs 8,467 original, ~50% density)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.geometry.masking import axis_aligned_box_grid, axis_aligned_box_sdf
from graphite.implicit.density_control import period_mm_from_sizing
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
from graphite.math.tpms import gyroid


def main():
    out_dir = _REPO_ROOT / "outputs" / "textures"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_mm = 0.030

    print("=" * 65)
    print(" GENERATING HALF-DENSITY BENCHMARK SCAFFOLDS ".center(65, "="))
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. Cylindrical Disk (5mm diameter x 2mm height)
    # -------------------------------------------------------------
    print("\n[1/2] Processing Cylindrical Disk Gyroid (Dia=5mm, H=2mm, UC=1mm, SF=33%)...")
    t0 = time.perf_counter()
    r = 2.5
    h = 2.0
    pad = 3 * res_mm
    x = np.arange(-r - pad, r + pad + res_mm / 2.0, res_mm)
    y = np.arange(-r - pad, r + pad + res_mm / 2.0, res_mm)
    z = np.arange(-h / 2.0 - pad, h / 2.0 + pad + res_mm / 2.0, res_mm)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    dr = np.sqrt(X**2 + Y**2) - r
    dz = np.abs(Z) - h / 2.0
    cad_sdf_cyl = np.sqrt(np.maximum(dr, 0.0)**2 + np.maximum(dz, 0.0)**2) + np.minimum(np.maximum(dr, dz), 0.0)
    F_cyl = gyroid(X, Y, Z, unit_cell_size=1.0, iso_offset=0.33, is_sheet=True)
    field_cyl = np.maximum(F_cyl, cad_sdf_cyl)

    origin_cyl = np.array([x[0], y[0], z[0]], dtype=np.float64)
    spacing = np.array([res_mm, res_mm, res_mm], dtype=np.float64)
    base_disk = extract_isosurface(field_cyl, spacing=spacing, origin=origin_cyl, enforce_watertight=True).mesh

    # Half density: spacing 400um (vs 300um original)
    cfg_disk_half = MicropillarConfig(
        diameter_mm=0.100,
        height_mm=0.400,
        spacing_mm=0.400,
        distribution="relaxed",
        boundary_type="cylinder",
        location="all",
    )
    t1 = time.perf_counter()
    mesh_disk_half = generate_micropillars(base_disk, cfg_disk_half)
    disk_path = out_dir / "gyroid_disk_5mm_dia_2mm_h_33sf_1mm_uc_30um_half_density_pillars.stl"
    mesh_disk_half.export(str(disk_path))
    sz_disk = disk_path.stat().st_size / (1024 ** 2)
    print(f"  Exported Disk: {disk_path.name}")
    print(f"  Faces: {len(mesh_disk_half.faces):,}, Watertight: {mesh_disk_half.is_watertight}, Size: {sz_disk:.1f} MB in {time.perf_counter()-t1:.2f}s")

    # -------------------------------------------------------------
    # 2. Cube Scaffold (5mm x 5mm x 5mm)
    # -------------------------------------------------------------
    print("\n[2/2] Processing Cube Gyroid (5x5x5mm, Pore=1mm, SF=33%)...")
    size_mm = 5.0
    half = size_mm / 2.0
    X_c, Y_c, Z_c, origin_c, _ = axis_aligned_box_grid(
        size_mm, size_mm, size_mm, res_mm,
        origin_x=-half, origin_y=-half, origin_z=-half,
        pad_voxels=3,
    )
    cad_sdf_box = axis_aligned_box_sdf(
        X_c, Y_c, Z_c,
        origin_x=-half, origin_y=-half, origin_z=-half,
        width_x_mm=size_mm, depth_y_mm=size_mm, height_z_mm=size_mm,
    )
    L = period_mm_from_sizing(pore_size_mm=1.00, unit_cell_size_mm=None, solid_fraction_for_pore_mapping=0.33)
    F_box = gyroid(X_c, Y_c, Z_c, unit_cell_size=L, iso_offset=0.33, is_sheet=True)
    field_box = np.maximum(F_box, cad_sdf_box)
    base_box = extract_isosurface(field_box, spacing=spacing, origin=origin_c, enforce_watertight=True).mesh

    # Half density: spacing 280um (vs 200um original)
    cfg_box_half = MicropillarConfig(
        diameter_mm=0.050,
        height_mm=0.200,
        spacing_mm=0.280,
        distribution="relaxed",
        boundary_type="box",
        location="all",
    )
    t2 = time.perf_counter()
    mesh_box_half = generate_micropillars(base_box, cfg_box_half)
    box_path = out_dir / "gyroid_5mm_33sf_1mm_pore_30um_half_density_pillars.stl"
    mesh_box_half.export(str(box_path))
    sz_box = box_path.stat().st_size / (1024 ** 2)
    print(f"  Exported Cube: {box_path.name}")
    print(f"  Faces: {len(mesh_box_half.faces):,}, Watertight: {mesh_box_half.is_watertight}, Size: {sz_box:.1f} MB in {time.perf_counter()-t2:.2f}s")

    print("\n" + "=" * 65)
    print(" HALF-DENSITY GENERATION COMPLETE ".center(65, "="))


if __name__ == "__main__":
    main()
