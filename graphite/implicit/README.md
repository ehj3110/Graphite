# `graphite.implicit` — capability card

## Owns

TPMS / SDF implicit lattice generation, calibration, pore metrics, piecewise Split-P and woodpile fields, isosurface extraction, surface textures / micropillars.

## Status

**Production.**

## Public entrypoints

Typical imports (see package modules / `__init__`):

- `calibrate_tpms_point`, `calibrate_tpms_gradient_profile`
- `extract_isosurface`
- `splitp_piecewise_box_single_pass`, `woodpile_piecewise_box_single_pass`, `build_piecewise_woodpile_mesh`
- `compute_max_inscribed_sphere_pore_size`, `generate_micropillars`
- Generators: `generate_conformal_lattice`, `generate_graded_lattice`, `generate_field_driven_lattice` (implicit modules)

## Does not own

TPMS scalar formulas (`math/`), CAD voxel/EDT (`geometry/`), STL/STEP export (`io/`), strut scaffolds (`explicit/`).

## Mix-and-match

- Callers: Streamlit `app.py`, `scripts/generate_lattice.py`, case studies, Aristo/Vocal pipelines that start from implicit STLs.
- Callees: `graphite.math`, `graphite.geometry` masking, `graphite.io` export.

## CAD Boundary Clipping & Mesh Integrity

1. **Exact B-Rep Boolean Trim (`exact_cad_trim=True`)**:
   - Uses `manifold3d` to execute an exact Boolean intersection (`man_tpms ^ man_cad`) between the dilated TPMS lattice and the input CAD boundary mesh.
   - **Manifold Validation**: The input CAD STL must be a valid closed 2-manifold (`man_cad.status() == Error.NoError`). `conformal.py` pre-checks CAD manifold status before dilating the level set. If the CAD mesh contains non-manifold defects (e.g. boundary cracks or T-junctions), dilation is automatically bypassed, cleanly clipping the lattice to the native implicit CAD boundary (`cad_sdf <= 0`) without leaving unclipped dilated strut stumps.

2. **Preserving Flying Edges Sub-Voxel Accuracy**:
   - `extract_isosurface` must run with `enforce_watertight=False` inside `conformal.py`.
   - If `enforce_watertight=True` is used, any non-watertight intermediate mesh triggers Trimesh's `mesh.voxelized().fill().marching_cubes` fallback, which degrades the smooth, sub-voxel floating-point Flying Edges contour into a coarse 1-bit binary voxel grid and increases meshing runtime tenfold.

## Read next

1. [docs/IMPLICIT_GRADING_AND_TEXTURES.md](../../docs/IMPLICIT_GRADING_AND_TEXTURES.md) — grading modes + textures / micropillars
2. [docs/IMPLICIT_ENGINE.md](../../docs/IMPLICIT_ENGINE.md)
3. [docs/TPMS_CALIBRATION_WORKFLOW.md](../../docs/TPMS_CALIBRATION_WORKFLOW.md)
4. Piecewise: [docs/PIECEWISE_PRISM_LATTICE_GENERATION.md](../../docs/PIECEWISE_PRISM_LATTICE_GENERATION.md)

## Do not open first

- Aristo stress→grade scripts (`stress_mapper`, `generate_graded_bracket_lattice`) — product path dropped
- `docs/history/IMPLICIT_GRADIENT_HISTORY.md` (cold)
- Experimental lofted TPMS under `experiments/lofted_hex/` unless tasked
