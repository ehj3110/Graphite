from __future__ import annotations

import time
from pathlib import Path

from Implicit_Lattice_Exploration.core.chirped_gyroid import generate_chirped_gyroid


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    stl_path = root / "test_parts" / "SkullCutout_OriginToZero.stl"
    modifier_path = root / "test_parts" / "NU_OriginToSkullCutout.stl"

    resolution = 0.25
    solid_fraction = 0.33
    widths = [1.0, 3.0, 6.0]

    print("=" * 60)
    print("Chirped Gyroid Skull Smoothness Sweep")
    print("=" * 60)
    print(f"Base STL: {stl_path}")
    print(f"Modifier STL: {modifier_path}")
    print(f"Resolution: {resolution} mm, Solid fraction: {solid_fraction:.2f}")

    for width in widths:
        out_path = (
            root
            / "Implicit_Lattice_Exploration"
            / "output"
            / f"Chirped_Skull_Smooth_{width}.stl"
        )

        print(f"\n--- transition_width = {width:.1f} mm ---")
        t0 = time.perf_counter()
        mesh = generate_chirped_gyroid(
            stl_path=stl_path,
            modifier_path=modifier_path,
            resolution=resolution,
            solid_fraction=solid_fraction,
            transition_width=width,
            output_path=out_path,
        )
        dt = time.perf_counter() - t0
        print(f"Total execution time: {dt:.2f} s")
        print(f"Final face count: {len(mesh.faces):,}")


if __name__ == "__main__":
    main()
