from __future__ import annotations

import time
from pathlib import Path

from Implicit_Lattice_Exploration.core.gyroid_generator import generate_gyroid_block


def main() -> None:
    size = 20.0
    resolution = 0.25
    pore_size = 5.0
    solid_fraction = 0.33

    print("=" * 60)
    print("Basic Gyroid Implicit Block - Parameterization Tests")
    print("=" * 60)

    # Test 1: Pore-size based parameterization
    out_pore = Path("Implicit_Lattice_Exploration/output/Basic_Gyroid_PoreSize.stl")
    print("\n[Test 1] Using pore_size=5.0 mm")
    t0 = time.perf_counter()
    mesh_pore = generate_gyroid_block(
        size=size,
        resolution=resolution,
        pore_size=pore_size,
        unit_cell_size=None,
        solid_fraction=solid_fraction,
        output_path=out_pore,
    )
    dt1 = time.perf_counter() - t0
    print(f"  Total execution time: {dt1:.2f} s")
    print(f"  Final face count: {len(mesh_pore.faces):,}")

    # Test 2: Direct unit-cell size (legacy L) with pore_size=None
    out_L = Path("Implicit_Lattice_Exploration/output/Basic_Gyroid_UnitCell.stl")
    print("\n[Test 2] Using unit_cell_size=5.0 mm (pore_size=None)")
    t1 = time.perf_counter()
    mesh_L = generate_gyroid_block(
        size=size,
        resolution=resolution,
        pore_size=None,
        unit_cell_size=5.0,
        solid_fraction=solid_fraction,
        output_path=out_L,
    )
    dt2 = time.perf_counter() - t1
    print(f"  Total execution time: {dt2:.2f} s")
    print(f"  Final face count: {len(mesh_L.faces):,}")


if __name__ == "__main__":
    main()
