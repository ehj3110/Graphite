# Graphite Development Roadmap

## Current State (Accomplished)

- Successfully migrated from an R&D script collection to a professional Python package structure (`graphite/`).
- Built a multi-axis and radial thickness/frequency grading engine.
- Built a mathematically perfect "Osteochondral" 1D piecewise Z-height engine for layered cartilage plugs.
- Built a "Hybrid Offset-Boundary" Dual-EDT engine for conformal, non-linear chirping between organic CAD faces.
- Implemented a Dihedral Angle-based Surface Picker using PyVista to easily select CAD faces.
- Upgraded the Streamlit UI with a persistent parameter table and 10µm resolution support.

## Next Steps (Immediate)

1. **Geometry Testing:** Create specific test parts (e.g., curved/organic custom STLs) to validate the Dual-Boundary grading in real-world implant scenarios.
2. **Skull Implant Validation:** Run a full end-to-end test on the `SkullCutout` geometry.
3. **Performance Optimization:** Profile the implicit calculations. Investigate speeding up `evaluate_tpms` and the `marching_cubes` extraction, and implement memory chunking for ultra-high-resolution (10µm) generation to prevent RAM exhaustion.

## Future Horizons

- **The Explicit Engine:** Integrate the discrete Hex/Tet strut-based node networks into the UI.
- **Topology Optimization (FEA):** Wire stress-field data directly into the TPMS frequency/thickness multipliers.
- **Headless Mode:** Create a CLI (Command Line Interface) script to allow for batch generation of lattices overnight without using the Streamlit UI.
