# Graphite

Robust conformal lattice R&D pipeline for repairing difficult STL shells, meshing them into tetrahedral volumes, and generating printable lattice solids.

## Project Architecture

- `src/repair/repair_suite.py` - production STL repair pipeline (Trimesh + PyMeshLab)
- `src/repair/blender_repair.py` - Blender-based pre-repair helper
- `src/meshing/generate_conformal_lattices.py` - end-to-end scaffold + lattice generation
- `tests/test_pipeline_smoke.py` - fast smoke test for repair + lattice pipeline
- `tests/test_gmsh.py` - heavy GMSH integration test on the production STL
- `outputs/` - generated lattice STL outputs

## Repair Pipeline

The repair flow is designed to rescue severely corrupted meshes while preserving geometry as much as possible:

1. **Trimesh Triage (fast cleanup)**
   - remove degenerate/duplicate faces
   - remove unreferenced vertices
   - fix winding, inversion, and normals
   - fill obvious holes

2. **PyMeshLab Screened Poisson Reconstruction (industrial fail-safe)**
   - reconstruct a clean continuous surface from the repaired shell

3. **Highlander Purge + Sealing Loop**
   - enforce single dominant shell
   - remove non-manifold edges/vertices
   - close holes aggressively
   - merge close vertices to stitch topology
   - re-orient/recompute normals
   - iterate until strict watertight checks pass

The target outcome is:
- watertight shell
- 1 connected component
- 0 boundary edges
- 0 non-manifold edges/vertices
- 0 self-intersections

## Meshing Pipeline (GMSH)

The meshing flow uses a robust discrete-to-CAD conversion:

1. Merge repaired STL into GMSH
2. `classifySurfaces(...)` on the discrete mesh
3. `createGeometry()` to build CAD-like entities
4. Build volume explicitly:
   - collect all surfaces
   - add `SurfaceLoop`
   - add `Volume`
5. Generate 3D tetrahedral mesh (`generate(3)`)
6. Extract nodes + tetra elements

Target scaffold size currently used: **25 mm**.

## Lattice Generation

Two lattice outputs are produced from the same conformal tet scaffold:

1. **Simple Tet-Edge Lattice**
   - extract all unique edges from tetrahedra
   - sweep edges to cylinders
   - boolean union with `manifold3d`
   - output: `outputs/top_part_new_Tet_Lattice.stl`

2. **Conformal Kagome Surface Dual**
   - use conformal topology transform (`kagome` + surface dual cage)
   - sweep resulting struts to cylinders
   - boolean union with `manifold3d`
   - output: `outputs/top_part_new_Conformal_Kagome.stl`

All sweep logic handles zero-length struts safely.

## Usage

Run from repository root:

```bash
python src/repair/repair_suite.py top_part_new_BMeshRepaired.stl
```

```bash
python src/meshing/generate_conformal_lattices.py
```

Generated artifacts are written to `outputs/`.

## Test Suite (Pytest)

Run the standardized smoke + integration suite from repository root:

```bash
pytest
```

Included tests:
- `tests/test_pipeline_smoke.py`: quick end-to-end test on a generated box mesh
- `tests/test_gmsh.py`: heavier GMSH volume generation integration test
