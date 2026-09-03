"""
Regenerate Phase 5 scaffolds at 30um resolution:
1. Evenly spaced (normal-consistent + tangential relaxation)
2. Printability filtered (60 to 90 deg from horizontal)
3. Standard Poisson-disk (normal-consistent thin-wall preservation)
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

    size_mm = 5.0
    res_mm = 0.030  # 30um locked-in
    pore_size_mm = 1.00
    solid_fraction = 0.33
    half = size_mm / 2.0

    print("=" * 65)
    print(" GENERATING 30um REGENERATED SCAFFOLDS ".center(65, "="))
    print(f"Parameters: 5mm Gyroid, 33% SF, 1mm pore, 30um resolution (Flying Edges)")
    print("=" * 65)

    # 1. Base Mesh at 30um
    t0 = time.perf_counter()
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
    L = period_mm_from_sizing(
        pore_size_mm=pore_size_mm, unit_cell_size_mm=None, solid_fraction_for_pore_mapping=solid_fraction
    )
    F = gyroid(X, Y, Z, unit_cell_size=L, iso_offset=solid_fraction, is_sheet=True)
    final_field = np.maximum(F, cad_sdf)

    base_res = extract_isosurface(final_field, spacing=spacing, origin=origin, enforce_watertight=True)
    base_mesh = base_res.mesh
    t_base = time.perf_counter() - t0
    print(f"Base mesh generated: {len(base_mesh.faces):,} faces, watertight={base_mesh.is_watertight} in {t_base:.2f}s")

    d_mm = 0.050  # 50um dia
    h_mm = 0.200  # 200um length
    s_mm = 0.200  # 200um spacing

    benchmarks = [
        (
            "1. Evenly Spaced (Normal-Consistent Tangential Relaxation)",
            MicropillarConfig(
                diameter_mm=d_mm,
                height_mm=h_mm,
                spacing_mm=s_mm,
                distribution="relaxed",
                location="all",
            ),
            "gyroid_5mm_33sf_1mm_pore_30um_even_pillars.stl",
        ),
        (
            "2. Printability Filtered (60 to 90 deg from horizontal)",
            MicropillarConfig(
                diameter_mm=d_mm,
                height_mm=h_mm,
                spacing_mm=s_mm,
                distribution="relaxed",
                filter_printable=True,
                min_angle_from_horizontal_deg=60.0,
                max_angle_from_horizontal_deg=90.0,
                location="all",
            ),
            "gyroid_5mm_33sf_1mm_pore_30um_printable_pillars.stl",
        ),
        (
            "3. Poisson-Disk (Normal-Consistent Thin-Wall Preservation)",
            MicropillarConfig(
                diameter_mm=d_mm,
                height_mm=h_mm,
                spacing_mm=s_mm,
                distribution="poisson_disk",
                location="all",
            ),
            "gyroid_5mm_33sf_1mm_pore_30um_poisson_pillars.stl",
        ),
    ]

    for title, cfg, fname in benchmarks:
        print("-" * 65)
        print(f"Synthesizing {title}...")
        t_b = time.perf_counter()
        m = generate_micropillars(base_mesh, cfg)
        out_path = out_dir / fname
        m.export(str(out_path))
        sz_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"  Exported: {fname}")
        print(f"  Faces: {len(m.faces):,}, Watertight: {m.is_watertight}, Size: {sz_mb:.1f} MB in {time.perf_counter()-t_b:.2f}s")

    print("=" * 65)
    print(" ALL 30um BENCHMARKS COMPLETED SUCCESSFULLY ".center(65, "="))


if __name__ == "__main__":
    main()
