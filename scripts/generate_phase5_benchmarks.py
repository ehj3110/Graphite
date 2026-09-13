"""
Generate Phase 5 benchmark STLs demonstrating:
1. Poisson-disk even hair distribution (no clumping, uniform spacing)
2. 3D-printability overhang angle filtering (<= 30 deg from horizontal)
3. Selective primitive face selection (+Z / top face only)
4. Internal pore channels only (clean outer envelope)
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


def run_phase5_benchmarks():
    out_dir = _REPO_ROOT / "outputs" / "textures"
    out_dir.mkdir(parents=True, exist_ok=True)

    size_mm = 5.0
    res_mm = 0.015 # 15um locked-in
    pore_size_mm = 1.00 # 1mm pore size
    solid_fraction = 0.33
    half = size_mm / 2.0

    print("=" * 65)
    print(" GENERATING PHASE 5 BENCHMARK SCAFFOLDS ".center(65, "="))
    print(f"Base: 5mm Gyroid, 33% SF, 1mm pores, 15um resolution (Flying Edges)")
    print("=" * 65)

    # 1. Generate Base Mesh once
    t0 = time.perf_counter()
    X, Y, Z, origin, spacing = axis_aligned_box_grid(
        size_mm, size_mm, size_mm, res_mm,
        origin_x=-half, origin_y=-half, origin_z=-half,
        pad_voxels=3,
    )
    cad_sdf = axis_aligned_box_sdf(X, Y, Z, origin_x=-half, origin_y=-half, origin_z=-half, width_x_mm=size_mm, depth_y_mm=size_mm, height_z_mm=size_mm)
    L = period_mm_from_sizing(pore_size_mm=pore_size_mm, unit_cell_size_mm=None, solid_fraction_for_pore_mapping=solid_fraction)
    F = gyroid(X, Y, Z, unit_cell_size=L, iso_offset=solid_fraction, is_sheet=True)
    final_field = np.maximum(F, cad_sdf)

    base_res = extract_isosurface(final_field, spacing=spacing, origin=origin, enforce_watertight=True)
    base_mesh = base_res.mesh
    print(f"Base mesh ready: faces={len(base_mesh.faces):,} in {time.perf_counter()-t0:.2f}s")

    # Common pillar sizing
    d_mm = 0.050 # 50 um dia
    h_mm = 0.200 # 200 um length
    s_mm = 0.200 # 200 um spacing

    benchmarks = [
        (
            "1. Poisson-Disk Even Spacing",
            MicropillarConfig(
                diameter_mm=d_mm, height_mm=h_mm, spacing_mm=s_mm,
                distribution="poisson_disk", location="all",
            ),
            "gyroid_5mm_33sf_1mm_pore_15um_poisson_pillars.stl",
        ),
        (
            "2. 3D-Printability Filtered (<= 30 deg from horizontal)",
            MicropillarConfig(
                diameter_mm=d_mm, height_mm=h_mm, spacing_mm=s_mm,
                distribution="poisson_disk", filter_printable=True,
                max_angle_from_horizontal_deg=30.0, location="all",
            ),
            "gyroid_5mm_33sf_1mm_pore_15um_printable_pillars.stl",
        ),
        (
            "3. Top Planar Face Only (+Z)",
            MicropillarConfig(
                diameter_mm=d_mm, height_mm=h_mm, spacing_mm=s_mm,
                distribution="poisson_disk", selected_faces=("+z",),
            ),
            "gyroid_5mm_33sf_1mm_pore_15um_top_face_pillars.stl",
        ),
        (
            "4. Internal Pores Only (Clean Outer Skin)",
            MicropillarConfig(
                diameter_mm=d_mm, height_mm=h_mm, spacing_mm=s_mm,
                distribution="poisson_disk", location="internal_only",
            ),
            "gyroid_5mm_33sf_1mm_pore_15um_internal_only_pillars.stl",
        ),
    ]

    for title, cfg, filename in benchmarks:
        print("-" * 65)
        print(f"Running: {title}...")
        t_b = time.perf_counter()
        m = generate_micropillars(base_mesh, cfg)
        out_path = out_dir / filename
        m.export(str(out_path))
        sz_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"  Exported: {filename}")
        print(f"  Faces: {len(m.faces):,}, Watertight: {m.is_watertight}, Size: {sz_mb:.1f} MB in {time.perf_counter()-t_b:.2f}s")

    print("=" * 65)
    print(" ALL 4 PHASE 5 BENCHMARKS GENERATED SUCCESSFULLY! ".center(65, "="))


if __name__ == "__main__":
    run_phase5_benchmarks()
