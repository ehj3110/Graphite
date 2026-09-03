import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import trimesh

from graphite.implicit.conformal import generate_conformal_lattice


def main() -> None:
    cad_path = Path("test_parts/SkullCutout_OriginToZero.stl")
    out_dir = Path("outputs/textures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "skull_cutout_gyroid_5mm_uc_33sf_exact_cad_trimmed.stl"

    print("=" * 70)
    print("Generating Skull Implant Gyroid Scaffold with Exact CAD B-Rep Trim")
    print("=" * 70)
    print(f"Target Part     : {cad_path.name}")
    print("Lattice Type    : Gyroid")
    print("Unit Cell Size  : 5.0 mm")
    print("Solid Fraction  : 33%")
    print("Resolution      : 0.120 mm (120 um)")
    print("Exact CAD Trim  : True (via SDF level-set dilation + Manifold3D CSG)")
    print("=" * 70)

    t0 = time.perf_counter()
    mesh_out = generate_conformal_lattice(
        stl_path=cad_path,
        lattice_type="Gyroid",
        resolution=0.120,
        unit_cell_size=5.0,
        solid_fraction=0.33,
        export_mode="core",
        exact_cad_trim=True,
        output_path=out_path,
        export_formats="stl",
    )
    total_time = time.perf_counter() - t0

    cad_mesh = trimesh.load(cad_path, force="mesh")
    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print("LATTICE METRICS SUMMARY")
    print("=" * 70)
    print(f"Output File     : {out_path}")
    print(f"Total Time      : {total_time:.2f} s")
    print(f"Face Count      : {len(mesh_out.faces):,}")
    print(f"Vertex Count    : {len(mesh_out.vertices):,}")
    print(f"Watertight      : {mesh_out.is_watertight}")
    print(f"File Size       : {file_size_mb:.2f} MB")
    print(f"Lattice Extents : {np.round(mesh_out.extents, 2)} mm")
    print(f"CAD Extents     : {np.round(cad_mesh.extents, 2)} mm")
    print(f"Volume          : {mesh_out.volume:.2f} mm^3")
    print("=" * 70)


if __name__ == "__main__":
    main()
