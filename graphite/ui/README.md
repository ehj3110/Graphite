# `graphite.ui` — capability card

## Owns

Modern web and headless user interfaces for Graphite:
- Interactive Trame + PyVista 3D Studio application (`graphite/ui/trame_app.py`) with primary Multi-Modality Selector (`Implicit TPMS` and `Explicit Struts`).
- Carbon-style lightweight 3D surface preview with adaptive subdivision and analytic level-set evaluation (`graphite/ui/surface_preview.py`).
- Sub-second explicit strut wireframe line network preview with surface-ironed boundary node glyphs (`generate_explicit_preview`).
- Headless CLI recipe runner (`graphite/ui/cli.py`) supporting both `--modality implicit` and `--modality explicit`.
- Geometry ingestion: Primitives (`Cube`, `Cylinder`, `Sphere`, `Toros`), Workspace Fixtures (`test_parts/*.STL`), and Custom STL uploads.
- Full 7-TPMS architecture catalog: `Gyroid`, `Diamond`, `Schwarz-P`, `Schwarz-Diamond`, `Neovius`, `Lidinoid`, `Split-P`.
- Explicit strut architecture catalog: A15 Conformal Kagome (`A15`), and Modular Simple Cubic Hexagonal (`SC`) rules (`octahedral`, `cubic`, `kelvin`).
- Conformation modes: `conformal` (valency boundary ironing) and `boolean` (spatial CAD intersection).
- Analytical strut sizing solver integration: solves $r \leftrightarrow \phi \leftrightarrow L$.
- Outer solid shelling: `Core` (lattice only), `Skin` (solid boundary shell), and `Combined` (lattice + shell) with interior cutaway preview.
- Dual-format export: Automated generation and download of watertight `.stl` and native `.3mf` files with direct HTTP streaming.
- Nyquist adaptive resolution rule: $\min(250\,\mu\text{m},\ w/2)$ with manual override.

## Status

**Production (Phase 3: Multi-Modality Integration — Implicit TPMS & Explicit Struts).**

## Public entrypoints

- `create_app(server=None)`: Factory creating the configured Trame server and single-page drawer layout.
- `run_headless_recipe(...)`: CLI entrypoint executing lattice generation recipes from scripts or terminal arguments.
- `generate_surface_tpms_preview(...)`: Generates dual PyVista PolyData objects (`pv_cad`, `pv_wall`) for sub-second surface preview.
- `generate_explicit_preview(...)`: Generates triple PyVista PolyData objects (`pv_cad`, `pv_struts`, `pv_boundary`) for fast explicit surface-dual or full wireframes.
- `extract_fast_sc_surface_dual(...)`: Fast surface-dual chord and boundary node extraction for SC hex lattices.
- `extract_fast_a15_surface_dual(...)`: Fast surface-dual chord and boundary node extraction for A15 Kagome lattices.
- `extract_surface_tpms_walls(...)`: Adaptive surface mesh subdivision and scalar contour clipping.
- `load_cad_mesh(...)`: CAD geometry loader handling primitives, local disk paths, and uploaded files.
- `create_floor_grid(...)`: Carbon/Blender-style dark build-plate floor plane.
- `SUPPORTED_TPMS_EQUATIONS`: Canonical list of supported implicit TPMS architectures.

## Does not own

- Implicit field evaluation math (`graphite/math/tpms.py`, `graphite/implicit/`).
- Direct explicit generator algorithms (`graphite/explicit/conformal_generator.py`, `graphite/explicit/a15_conformal.py`).
- Direct FEA solvers (`graphite/aristo/`).

## Mix-and-match

- Consumes `graphite/implicit/conformal.py` and `graphite/implicit/field_driven.py` for full 3D implicit lattice volume meshing.
- Consumes `graphite/explicit/conformal_generator.py` and `graphite/explicit/a15_conformal.py` for full 3D explicit conformal meshing.
- Consumes `graphite/explicit/sizing_solver.py` for analytical density and strut radius calibration.
- Consumes `graphite/io/mesh_export.py` for dual STL and 3MF container export with manifold verification.
- Consumes `graphite/geometry/primitives.py` for standard CAD primitives.

## Read next

1. [docs/TRAME_UI_AND_CLI.md](../../docs/TRAME_UI_AND_CLI.md)
2. [graphite/explicit/README.md](../explicit/README.md)
3. [graphite/implicit/README.md](../implicit/README.md)
4. [graphite/io/README.md](../io/README.md)
