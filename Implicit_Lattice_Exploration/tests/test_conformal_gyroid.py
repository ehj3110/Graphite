from __future__ import annotations

import time
from pathlib import Path

from Implicit_Lattice_Exploration.core.conformal_gyroid import generate_conformal_gyroid


def main() -> None:
    stl_path = Path("test_parts/20mm_cube.stl")
    resolution = 0.25
    pore_size = 5.0
    solid_fraction = 0.33

    out_path = Path("Implicit_Lattice_Exploration/output/Conformal_Cube_Gyroid.stl")

    print("=" * 60)
    print("Conformal Gyroid on 20mm Cube")
    print("=" * 60)
    print(f"STL: {stl_path}")
    print(f"Resolution: {resolution} mm, Pore size: {pore_size} mm, SF: {solid_fraction:.2f}")

    t0 = time.perf_counter()
    mesh = generate_conformal_gyroid(
        stl_path=stl_path,
        resolution=resolution,
        pore_size=pore_size,
        unit_cell_size=None,
        solid_fraction=solid_fraction,
        output_path=out_path,
    )
    dt = time.perf_counter() - t0

    print(f"Total execution time: {dt:.2f} s")
    print(f"Final face count: {len(mesh.faces):,}")


if __name__ == "__main__":
    main()
