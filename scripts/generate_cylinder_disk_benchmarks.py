"""
Generate the three 5mm diameter x 2mm tall Gyroid disk benchmarks requested:
- 1mm unit cell Gyroid, 33% solid fraction, 30um resolution (Flying Edges)
- 100um diameter x 400um tall micropillars at 300um spacing
- Disk 1: Evenly spaced hairs across all surfaces (relaxed honeycomb)
- Disk 2: 3D-Printable hairs only (60 to 90 deg from horizontal)
- Disk 3: Top and bottom planar surfaces only
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

from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
from graphite.math.tpms import gyroid


def main():
    out_dir = _REPO_ROOT / "outputs" / "textures"
    out_dir.mkdir(parents=True, exist_ok=True)

    diameter_mm = 5.0
    radius_mm = diameter_mm / 2.0  # 2.5 mm
    height_mm = 2.0  # 2.0 mm
    res_mm = 0.030   # 30 um resolution
    unit_cell_mm = 1.0  # 1.0 mm unit cell
    solid_fraction = 0.33

    print("=" * 65)
    print(" GENERATING CYLINDRICAL DISK GYROID SCAFFOLDS ".center(65, "="))
    print(f"Geometry: Cylindrical Disk, Dia={diameter_mm}mm, Height={height_mm}mm")
    print(f"Lattice: Gyroid, Unit Cell={unit_cell_mm}mm, SF={solid_fraction:.0%}, Res={res_mm*1e3:.0f}um")
    print(f"Hairs: 100um diameter x 400um length, 300um spacing")
    print("=" * 65)

    # 1. Generate Base Cylindrical Disk Mesh
    t0 = time.perf_counter()
    pad = 3 * res_mm
    x = np.arange(-radius_mm - pad, radius_mm + pad + res_mm / 2.0, res_mm)
    y = np.arange(-radius_mm - pad, radius_mm + pad + res_mm / 2.0, res_mm)
    z = np.arange(-height_mm / 2.0 - pad, height_mm / 2.0 + pad + res_mm / 2.0, res_mm)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Exact analytic cylinder SDF
    dr = np.sqrt(X**2 + Y**2) - radius_mm
    dz = np.abs(Z) - height_mm / 2.0
    exterior = np.sqrt(np.maximum(dr, 0.0)**2 + np.maximum(dz, 0.0)**2)
    interior = np.minimum(np.maximum(dr, dz), 0.0)
    cad_sdf = exterior + interior

    # TPMS Gyroid field with 1.0 mm unit cell and 33% SF
    F = gyroid(X, Y, Z, unit_cell_size=unit_cell_mm, iso_offset=solid_fraction, is_sheet=True)
    final_field = np.maximum(F, cad_sdf)

    origin = np.array([x[0], y[0], z[0]], dtype=np.float64)
    spacing = np.array([res_mm, res_mm, res_mm], dtype=np.float64)

    base_res = extract_isosurface(final_field, spacing=spacing, origin=origin, enforce_watertight=True)
    base_mesh = base_res.mesh
    t_base = time.perf_counter() - t0
    print(f"Base disk generated: {len(base_mesh.faces):,} faces, watertight={base_mesh.is_watertight}, extents={np.round(base_mesh.extents, 3)} in {t_base:.2f}s")

    pillar_dia = 0.100  # 100 um
    pillar_len = 0.400  # 400 um
    pillar_spacing = 0.300  # 300 um

    benchmarks = [
        (
            "1. Evenly Spaced Hairs (All Surfaces)",
            MicropillarConfig(
                diameter_mm=pillar_dia,
                height_mm=pillar_len,
                spacing_mm=pillar_spacing,
                distribution="relaxed",
                boundary_type="cylinder",
                location="all",
            ),
            "gyroid_disk_5mm_dia_2mm_h_33sf_1mm_uc_30um_even_pillars.stl",
        ),
        (
            "2. Printable Hairs Only (60 to 90 deg from horizontal)",
            MicropillarConfig(
                diameter_mm=pillar_dia,
                height_mm=pillar_len,
                spacing_mm=pillar_spacing,
                distribution="relaxed",
                boundary_type="cylinder",
                filter_printable=True,
                min_angle_from_horizontal_deg=60.0,
                max_angle_from_horizontal_deg=90.0,
                location="all",
            ),
            "gyroid_disk_5mm_dia_2mm_h_33sf_1mm_uc_30um_printable_pillars.stl",
        ),
        (
            "3. Top and Bottom Surfaces Only",
            MicropillarConfig(
                diameter_mm=pillar_dia,
                height_mm=pillar_len,
                spacing_mm=pillar_spacing,
                distribution="relaxed",
                boundary_type="cylinder",
                selected_faces=("top", "bottom"),
            ),
            "gyroid_disk_5mm_dia_2mm_h_33sf_1mm_uc_30um_top_bottom_pillars.stl",
        ),
    ]

    for title, cfg, fname in benchmarks:
        print("-" * 65)
        print(f"Synthesizing: {title}...")
        t_b = time.perf_counter()
        m = generate_micropillars(base_mesh, cfg)
        out_path = out_dir / fname
        m.export(str(out_path))
        sz_mb = out_path.stat().st_size / (1024 ** 2)
        print(f"  Exported: {fname}")
        print(f"  Faces: {len(m.faces):,}, Watertight: {m.is_watertight}, Size: {sz_mb:.1f} MB in {time.perf_counter()-t_b:.2f}s")

    print("=" * 65)
    print(" ALL THREE CYLINDER DISK BENCHMARKS COMPLETE ".center(65, "="))


if __name__ == "__main__":
    main()
