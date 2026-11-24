# `graphite.ui` — capability card

## Owns

Native PySide6 + VTK desktop studios, web apps, and headless user interfaces for Graphite:
- **Graphite Implicit Studio (`graphite/ui/implicit_studio.py`, `implicit_state.py`, `implicit_vtk_scene.py`)**:
  - Primary desktop entrypoint (`python -m graphite.ui`) styled with Midnight Slate 3-Column Workbench (`#0b111e` / `#0f172a` + `#38bdf8` / `#00f5d4` accents).
  - Pure VTK 3D viewport (zero `pyvista` requirement) with trackball drag navigation (`vtkInteractorStyleTrackballCamera`), ultra-sharp `2D Surface` default overlay over opaque gray CAD, lightweight `3D Flying Edges (Coarse)` preview (`~350k` voxel budget), amber micropillar forest glyphs, hardware GPU live X/Y/Z cutaway during drag with sealed solid caps on release, 3D orientation axes triad, viewport HUD, PNG snapshot export (`Ctrl+P`), and optional post-generation 3D mesh inspection (OFF by default).
  - Cached-SDF 2D Cross-Section Slice Inspector (down to 10 um `0.010 mm`) with physical mm scale bar, live surface texture visualization, and side-by-side QPainter `L(s)` / `w(s)` / `SF(s)` grading profile chart.
  - Full 7 TPMS (`Gyroid`, `Diamond`, `Schwarz-P`, `Schwarz-Diamond`, `Neovius`, `Lidinoid`, `Split-P`) + `Woodpile`, `"Generate Skin"` gate hiding/revealing internal-fillet shell options, always-on `exact_cad_trim=True`, flexible dual-driver 6-pair 4-way sizing calculator (`1.50 mm` cell / `33%` SF / `0.50 mm` wall / `800 um` pore defaults, unchosen inputs grayed out and autofilled) + Nyquist auto-resolution, and 1-click LUT/Newton calibration (`Calibrate Exact`).
  - Unified Grading Mode Selector (`Uniform`, `Linear / Radial Ramp`, `Multi-Knot Control Table` supporting Normalized SDF Blend and Jacobian Phase Integration, `Piecewise Discrete Bands`) + optional Surface Textures & Explicit Micropillars.
  - Non-destructive in-memory CAD repair, origin centering, sequential batch recipe queue (`Run Batch Queue...` & CLI `--recipe`/`--batch`), JSON recipe save/load, automatic parameter manifests (`outputs/App outputs/YYYY-MM-DD/`), and full-resolution Flying Edges watertight `.stl` / `.3mf` export. See [docs/IMPLICIT_STUDIO_GUIDE.md](../../docs/IMPLICIT_STUDIO_GUIDE.md).
- Interactive Trame + PyVista 3D Studio application (`graphite/ui/trame_app.py`, accessible via `python -m graphite.ui --legacy-trame`).
- Interactive Trame + PyVista Real-Space 3D Popup (`graphite/ui/neighborhood_trame_popup.py`):
  - Real-space 3D mouse trackball rotation, zoom, and pan via `plotter_ui(mode="server")`.
  - Multi-layer toggles: Solid kept cell blocks, transparent dotted macro wireframe, surface dual struts, volume struts, node glyphs, CAD clip.
  - 6 diagnostic coloring modes (direction, plane, valency traffic light, CAD deviation, length distortion, uniform).
  - In-browser "What-If" candidate strut injection (`u, v`) with live topological re-diagnosis.
- Aristo Topology Optimization Studio (`graphite/ui/aristo_gui.py`):
  - Pure-Python unified domain setup (`AristoSetup`, `Region`) decoupling solver from GUI text hacks.
  - In-place actor management, debounced viewport updates, and CAD geometry caching (`AristoPreview`).
  - Thread-safe background solver dispatch via asyncio event loop offloading and cooperative cancellation (`cancel_aristo_solve`).
  - Run directory isolation under `outputs/fea/gui_runs/<timestamp>_<mode>/` with automatic retention pruning.
  - Pre-flight constraint and volume fraction validation with live reactive UI alert chips.
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

- `launch_implicit_studio(...)` / `ImplicitStudioWindow`: Native PySide6 + VTK Implicit Studio (`python -m graphite.ui`).
- `ImplicitStudioState`: Serializable headless state, 4-way sizing solver, and generation pipeline (`graphite/ui/implicit_state.py`).
- `create_app(server=None)`: Factory creating the legacy Trame server and single-page drawer layout.
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
