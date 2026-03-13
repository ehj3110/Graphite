from __future__ import annotations

import time
from pathlib import Path

from Implicit_Lattice_Exploration.core.gradient_gyroid import generate_gradient_gyroid


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    stl_path = root / "test_parts" / "20mm_cube.stl"
    modifier_path = root / "test_parts" / "Sphere.stl"
    out_x = root / "Implicit_Lattice_Exploration" / "output" / "Gradient_X_Sheet.stl"
    out_modifier = (
        root / "Implicit_Lattice_Exploration" / "output" / "Gradient_Modifier_Sheet.stl"
    )

    print("=" * 60)
    print("Gradient Gyroid Tests on 20mm Cube")
    print("=" * 60)
    print(f"STL: {stl_path}")
    print("Resolution: 0.25 mm, Pore size: 5.0 mm, SF range: 0.10 -> 0.90")

    print("\n[Test 1] Axis gradient (X)")
    t0 = time.perf_counter()
    mesh_x = generate_gradient_gyroid(
        stl_path=stl_path,
        resolution=0.25,
        pore_size=5.0,
        min_solid_fraction=0.10,
        max_solid_fraction=0.90,
        gradient_type="X",
        output_path=out_x,
    )
    dt_x = time.perf_counter() - t0
    print(f"Total execution time: {dt_x:.2f} s")
    print(f"Final face count: {len(mesh_x.faces):,}")

    print("\n[Test 2] Modifier gradient")
    print(f"Modifier STL: {modifier_path}")
    t1 = time.perf_counter()
    mesh_mod = generate_gradient_gyroid(
        stl_path=stl_path,
        resolution=0.25,
        pore_size=5.0,
        min_solid_fraction=0.10,
        max_solid_fraction=0.90,
        gradient_type="modifier",
        modifier_path=modifier_path,
        transition_width=5.0,
        output_path=out_modifier,
    )
    dt_mod = time.perf_counter() - t1
    print(f"Total execution time: {dt_mod:.2f} s")
    print(f"Final face count: {len(mesh_mod.faces):,}")


if __name__ == "__main__":
    main()
