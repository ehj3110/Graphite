from __future__ import annotations

from pathlib import Path
import time

from Implicit_Lattice_Exploration.core.conformal_gyroid import generate_conformal_gyroid


def main() -> None:
    test_parts_root = Path(
        r"C:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\test_parts"
    )
    out_root = Path(
        r"C:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\Implicit_Lattice_Exploration\output"
    )

    test_cases = {
        "Sphere.stl": {"res": 0.25, "pore": 5.0},  # Small enough for high-res
        "Weird.stl": {"res": 0.35, "pore": 8.0},  # Medium bounds
        "Torus.stl": {"res": 0.4, "pore": 10.0},  # Large bounds (71mm), slightly coarser res
        "acoustic wall panel - single.stl": {"res": 0.5, "pore": 12.0},  # Very large bounds, coarse res
    }

    for filename, params in test_cases.items():
        in_path = test_parts_root / filename
        if not in_path.exists() and filename.lower().startswith("torus"):
            # Workspace uses legacy filename "Toros.stl"
            alt = test_parts_root / "Toros.stl"
            if alt.exists():
                in_path = alt
        if not in_path.exists():
            print(f"\n--- Processing {filename} ---")
            print(f"WARNING: Missing file: {in_path}")
            continue

        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"Conformal_{Path(filename).stem}_Gyroid.stl"

        print(f"\n--- Processing {filename} ---")
        t0 = time.perf_counter()
        generate_conformal_gyroid(
            in_path,
            resolution=params["res"],
            pore_size=params["pore"],
            solid_fraction=0.33,
            output_path=out_path,
        )
        dt = time.perf_counter() - t0
        print(f"Total time: {dt:.2f} s")


if __name__ == "__main__":
    main()

