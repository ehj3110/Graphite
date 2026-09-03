"""
Generate high-fidelity Skull Cutout Gyroid scaffold with:
1. 5% overscaled CAD boundary + Manifold3D Boolean trim for razor-sharp CAD edges
2. 800um pores, 33% solid fraction, 60um resolution
3. 100um dia x 400um long Z-aligned fibers on TOP surface only at 300um pitch
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

from graphite.explicit.geometry_module import manifold_to_trimesh, trimesh_to_manifold
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.density_control import period_mm_from_sizing
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
from graphite.math.tpms import gyroid


def main():
    out_dir = _REPO_ROOT / "outputs" / "textures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cad_path = _REPO_ROOT / "test_parts" / "SkullCutout_OriginToZero.stl"
    res_mm = 0.060  # 60 um
    pore_size_mm = 0.800
    solid_fraction = 0.33

    print("=" * 70, flush=True)
    print(" SKULL CUTOUT TRIMMED GYROID + TOP Z-PILLARS ENGINE ".center(70, "="), flush=True)
    print(f"CAD File: {cad_path.name}", flush=True)
    print(f"Pore Size: {pore_size_mm*1e3:.0f} um, SF: {solid_fraction:.2f}, Res: {res_mm*1e3:.1f} um", flush=True)
    print("=" * 70, flush=True)

    cad = trimesh.load(str(cad_path), force="mesh")
    com = np.asarray(cad.center_mass, dtype=np.float64)
    print(f"Center of Mass: {np.round(com, 3)}", flush=True)

    # 1. 5% Overscaled CAD Boundary
    print("[1/5] Overscaling CAD by 5% about center of mass...", flush=True)
    scaled_cad = cad.copy()
    scaled_cad.vertices = (scaled_cad.vertices - com) * 1.05 + com

    # 2. Voxelize Overscaled Domain & Evaluate TPMS
    print(f"[2/5] Voxelizing overscaled CAD domain at {res_mm*1e3:.1f} um...", flush=True)
    t0 = time.perf_counter()
    X, Y, Z, cad_sdf, _, _, nx, ny, nz = voxelize_mesh_and_edt(scaled_cad, resolution=res_mm, pad_width=3)
    print(f"      Domain: {nx}x{ny}x{nz} voxels in {time.perf_counter()-t0:.2f}s", flush=True)

    L = period_mm_from_sizing(pore_size_mm=pore_size_mm, unit_cell_size_mm=None, solid_fraction_for_pore_mapping=solid_fraction)
    print(f"[3/5] Evaluating Gyroid field (L={L:.3f} mm)...", flush=True)
    t1 = time.perf_counter()
    F = gyroid(X, Y, Z, unit_cell_size=L, iso_offset=solid_fraction, is_sheet=True)
    field = np.maximum(F, cad_sdf)

    origin = np.array([X.min(), Y.min(), Z.min()], dtype=np.float64)
    spacing = np.array([res_mm, res_mm, res_mm], dtype=np.float64)
    iso = extract_isosurface(field, spacing=spacing, origin=origin, enforce_watertight=True)
    oversized_tpms = iso.mesh
    print(f"      Oversized TPMS: {len(oversized_tpms.faces):,} faces in {time.perf_counter()-t1:.2f}s", flush=True)

    # 3. Manifold3D Boolean Intersection with Exact Original CAD
    print("[4/5] Performing Manifold3D Boolean trim with original CAD boundary...", flush=True)
    t2 = time.perf_counter()
    man_tpms = trimesh_to_manifold(oversized_tpms)
    man_cad = trimesh_to_manifold(cad)
    man_clean = man_tpms ^ man_cad  # Exact CSG intersection
    clean_tpms = manifold_to_trimesh(man_clean)
    trimesh.repair.fix_normals(clean_tpms)
    print(f"      Boolean Trim completed in {time.perf_counter()-t2:.2f}s!", flush=True)
    print(f"      Clean Base Mesh: {len(clean_tpms.faces):,} faces, watertight={clean_tpms.is_watertight}", flush=True)

    # 4. Synthesize 100um x 400um Z-Aligned Fibers on TOP Surface Only
    print("[5/5] Synthesizing Z-aligned fibers on TOP surface only (d=100um, h=400um, s=300um)...", flush=True)
    t3 = time.perf_counter()
    pillar_cfg = MicropillarConfig(
        diameter_mm=0.100,
        height_mm=0.400,
        spacing_mm=0.300,
        distribution="relaxed",
        selected_faces=("top",),
        orientation="z_aligned",
        boundary_mesh=cad,
        max_boundary_dist_mm=0.200,
        circular_segments=16,
    )
    final_scaffold = generate_micropillars(clean_tpms, pillar_cfg, boundary_mesh=cad)
    print(f"      Fiber synthesis & union completed in {time.perf_counter()-t3:.2f}s", flush=True)

    out_file = out_dir / "skull_cutout_gyroid_800um_pore_33sf_60um_trimmed_top_z_pillars.stl"
    print(f"Exporting final STL to {out_file.name}...", flush=True)
    final_scaffold.export(str(out_file))
    sz_mb = out_file.stat().st_size / (1024 ** 2)
    print(f"DONE! Exported {out_file.name} ({sz_mb:.1f} MB, {len(final_scaffold.faces):,} faces, watertight={final_scaffold.is_watertight})", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
