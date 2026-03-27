# Graphite Lattice Engine Architecture

## 1) Project Overview

Graphite is a hybrid computational geometry and lattice generation tool that combines:

- an interactive Streamlit wizard (`app.py`) for geometry setup and execution,
- implicit TPMS-based lattice engines in `graphite/implicit`,
- shared TPMS equation evaluation in `graphite/math`,
- geometry utilities in `graphite/geometry`,
- and a placeholder namespace for future explicit (node/strut) engines in `graphite/explicit`.

At its current state, the active production path in the app is implicit generation (uniform, graded, and chirped TPMS), with STL input/primitive support and STL export.

## 2) Directory Structure

Current scope requested: `app.py` and `graphite/`.

```text
Graphite/
├── app.py
└── graphite/
    ├── __init__.py
    ├── math/
    │   ├── __init__.py
    │   └── tpms.py
    ├── geometry/
    │   ├── __init__.py
    │   ├── primitives.py
    │   └── surface_picking.py
    ├── implicit/
    │   ├── __init__.py
    │   ├── conformal.py
    │   ├── graded.py
    │   └── chirped.py
    └── explicit/
        └── __init__.py
```

### Module Notes

- `app.py`
  - Streamlit 6-step wizard.
  - Collects geometry, TPMS mode, sizing, grading, shelling, and execution settings.
  - Routes execution to:
    - `generate_conformal_lattice` (uniform),
    - `generate_graded_lattice` (variable porosity/thickness),
    - `generate_chirped_lattice` (variable pore size/chirped).
  - Handles temporary STL files for base geometry and optional modifier.
  - Provides in-browser STL download after generation.

- `graphite/__init__.py`
  - Package marker and high-level package description.

- `graphite/math/tpms.py`
  - `evaluate_tpms(lattice_type, k, X, Y, Z)` centralizes TPMS field equations.
  - Supported `lattice_type` aliases:
    - `Gyroid`
    - `Schwarz-P` / `Schwarz Primitive` / `Schwarz`
    - `Diamond`
    - `Neovius`
    - `Split-P`

- `graphite/geometry/primitives.py`
  - `generate_primitive(shape, size)` for:
    - cube (`box`),
    - sphere (`icosphere`),
    - cylinder.

- `graphite/geometry/surface_picking.py`
  - PyVista-based surface visualization and picking utility.
  - Uses `mesh.facets` and `mesh.facets_area` to identify surfaces.
  - Colors facet groups, labels surface centroids, and reports picked face/surface IDs.

- `graphite/implicit/conformal.py`
  - `generate_conformal_lattice(...)`:
    - voxelizes input mesh,
    - builds CAD SDF via EDT,
    - evaluates TPMS field with `evaluate_tpms`,
    - supports export modes: `core`, `skin`, `combined`,
    - supports localized shelling using selected facet IDs via point-cloud EDT approximation,
    - optional origin centering.
  - Includes backward-compat wrapper `generate_conformal_gyroid(...)`.

- `graphite/implicit/graded.py`
  - `generate_graded_lattice(...)`:
    - multi-TPMS graded thickness / porosity field,
    - gradient modes: axis-based (`X`, `Y`, `Z`) or modifier-based (`MODIFIER`),
    - uses legacy mapping `L = pore_size / 0.65`, `k = 2*pi/L`,
    - builds `W` weight map and spatial `sf_grid`,
    - intersects TPMS sheet field with CAD SDF,
    - optional origin centering.

- `graphite/implicit/chirped.py`
  - `generate_chirped_lattice(...)`:
    - multi-TPMS chirped frequency field driven by modifier STL,
    - computes smoothstep weight map from modifier SDF,
    - interpolates `K_grid` between base and modifier frequencies,
    - evaluates TPMS field with `evaluate_tpms(lattice_type, K_grid, X, Y, Z)`,
    - intersects with CAD SDF,
    - optional origin centering.

- `graphite/explicit/__init__.py`
  - Placeholder namespace for future explicit strut engines.

## 3) The Generation Pipeline

### A) UI Collection (`app.py`, Steps 1-6)

1. **Step 1: Geometry Selection**
   - Choose geometry source:
     - `Custom STL` (uploaded file),
     - `Primitive` (shape + size),
     - `ASTM Standard` (UI only; execution not wired).
   - Uploaded STL stored in `st.session_state.uploaded_file`.

2. **Step 2: Engine & Topology**
   - Choose `Explicit (Struts)` vs `Implicit (TPMS)`.
   - For implicit mode, choose TPMS type (`Gyroid`, `Diamond`, `Schwarz-P`, `Neovius`, `Split-P`).

3. **Step 3: Sizing & Density**
   - Set `solid_fraction`.
   - Select sizing mode and enter `pore_size` or `unit_cell_size`.
   - Current execution paths consume `pore_size` (uniform/graded) and `solid_fraction` (uniform/chirped).

4. **Step 4: Boundary & Field Control**
   - Toggle conformal masking flag in params.
   - Select functional grading mode:
     - `Uniform`,
     - `Variable Porosity (Thickness)`,
     - `Variable Pore Size (Chirped)`.
   - For non-uniform modes:
     - upload modifier STL (`st.session_state.modifier_file`),
     - set `transition_width`.

5. **Step 5: Shelling & Export Modes**
   - Select export mode:
     - `Core Only` -> `core`,
     - `Hollow Skin Only` -> `skin`,
     - `Combined (Core + Skin)` -> `combined`.
   - Configure `shell_thickness` for skin/combined.
   - Includes a placeholder UI button for launching surface picker.

6. **Step 6: Execution**
   - Set `resolution`, output filename, and `center_origin`.
   - Press **Generate Scaffold** to run backend.

### B) Geometry Materialization

When Generate is pressed:

- If `Primitive`: `generate_primitive(shape, size)` creates `trimesh` geometry.
- If `Custom STL`: uploaded STL is loaded into `trimesh`.
- If `ASTM Standard`: app warns and stops (not wired).

The resulting mesh is exported to a temporary STL file (`temp_input_path`).

If a modifier STL is provided, it is also written to temp (`mod_path`).

### C) Backend Routing

Router key: `params["grading_mode"]`

- `Uniform` -> `generate_conformal_lattice(...)`
- `Variable Porosity (Thickness)` -> `generate_graded_lattice(..., gradient_type="MODIFIER", modifier_path=mod_path, ...)`
- `Variable Pore Size (Chirped)` -> `generate_chirped_lattice(..., modifier_path=mod_path, ...)`

### D) Core Numerical Flow (implicit engines)

Across conformal / graded / chirped engines, the shared pattern is:

1. Load input STL into `trimesh`.
2. Voxelize (`mesh.voxelized(...).fill()`), pad mask.
3. Build world-aligned 3D grid (`X, Y, Z`).
4. Compute TPMS scalar field via `evaluate_tpms(...)`.
5. Build CAD SDF from EDT (`outside_dist - inside_dist`) * `resolution`.
6. Combine TPMS field and CAD SDF with implicit boolean operations.
7. Extract surface with `marching_cubes`.
8. Convert vertices back to world coordinates.
9. Optional origin centering (`verts -= mesh_out.centroid`).
10. Export STL to requested output path.

### E) Output and Cleanup

- App reports success and exposes `st.download_button` for STL.
- Temporary files (`temp_input_path` and optional `mod_path`) are removed in `finally`.

## 4) Current Capabilities

- Multi-TPMS field evaluation (`Gyroid`, `Diamond`, `Schwarz-P`, `Neovius`, `Split-P`) via shared `graphite.math.tpms`.
- Uniform conformal TPMS generation constrained by CAD volume using EDT-derived SDF.
- Export modes in conformal engine:
  - `core`, `skin`, `combined`.
- Localized shelling in conformal engine using selected CAD facets and point-cloud EDT approximation (`selected_surfaces` path).
- Functional grading (thickness/porosity style) in `graphite/implicit/graded.py`:
  - axis gradient or modifier-driven gradient.
- Chirped pore-size style generation in `graphite/implicit/chirped.py` with smoothstep modifier blending.
- Primitive base geometry generation (cube/sphere/cylinder) for direct generation without STL upload.
- Optional post-extraction origin centering in all three implicit engines (`center_origin`).
- PyVista-based surface picker utility with high-contrast facet coloring and click reporting.
- Streamlit wizard flow with parameter persistence and runtime routing.

## 5) Explicit vs. Implicit Segmentation

Graphite currently separates architecture into two conceptual domains:

- **Implicit domain (`graphite/implicit`)**
  - Continuous scalar-field modeling (TPMS + CAD SDF).
  - Operations are field/iso-surface based (EDT, implicit booleans, marching cubes).
  - Produces watertight triangle meshes from volumetric functions.

- **Explicit domain (`graphite/explicit`)**
  - Reserved namespace for discrete node/strut lattice logic.
  - Intended for graph/topology-driven beam frameworks rather than scalar-field extraction.
  - At present, this namespace is a placeholder module only.

