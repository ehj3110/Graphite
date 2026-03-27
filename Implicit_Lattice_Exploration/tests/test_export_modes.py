from __future__ import annotations

import time
from pathlib import Path

from Implicit_Lattice_Exploration.core.conformal_gyroid import generate_conformal_gyroid


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    stl_path = root / "test_parts" / "20mm_cube.stl"
    out_dir = root / "Implicit_Lattice_Exploration" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    resolution = 0.25
    pore_size = 5.0
    shell_thickness = 2.0

    export_jobs = [
        ("core", out_dir / "Cube_Mode_Core.stl", None),
        ("skin", out_dir / "Cube_Mode_Skin.stl", None),
        ("combined", out_dir / "Cube_Mode_Combined.stl", [0]),
    ]

    print("=" * 60)
    print("Conformal Gyroid Export Modes Test")
    print("=" * 60)
    print(f"STL: {stl_path}")
    print(
        f"Resolution: {resolution} mm, Pore size: {pore_size} mm, "
        f"Shell thickness: {shell_thickness} mm"
    )

    for mode, out_path, selected_surfaces in export_jobs:
        print(f"\n--- Export mode: {mode} ---")
        if selected_surfaces is not None:
            print(f"Selected surfaces: {selected_surfaces}")
        t0 = time.perf_counter()
        mesh = generate_conformal_gyroid(
            stl_path=stl_path,
            resolution=resolution,
            pore_size=pore_size,
            solid_fraction=0.33,
            export_mode=mode,
            shell_thickness=shell_thickness,
            selected_surfaces=selected_surfaces,
            output_path=out_path,
        )
        dt = time.perf_counter() - t0
        print(f"Total execution time: {dt:.2f} s")
        print(f"Final face count: {len(mesh.faces):,}")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
