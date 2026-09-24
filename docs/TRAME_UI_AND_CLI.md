# Interactive Trame + PyVista Studio & Headless CLI

This document is the canonical engineering and operational reference for Graphite's interactive 3D web studio and headless CLI interface (`graphite.ui`). It details the application's capabilities, architecture, performance optimizations, pitfalls encountered during development, and guidance for future AI agents and developers.

---

## 1. System Architecture & Modality Routing

The UI architecture cleanly decouples rapid, interactive 3D surface design from computationally heavy volumetric mesh generation:

```
                      +-----------------------------+
                      |   Browser Client (Vue 3)    |
                      |  Vuetify 3 Industrial Theme |
                      +--------------+--------------+
                                     | WebSocket / HTTP
                      +--------------v--------------+
                      |      Trame Server Loop      |
                      |  (SinglePageWithDrawerLayout)|
                      +-------+--------------+------+
                              |              |
          Fast Preview Button |              | Generate Production Button
                              v              v
     +--------------------------+  +----------------------------+
     | graphite.ui.surface_preview| |    graphite.ui.cli /       |
     | - Fast SC Surface Dual   |  | graphite.implicit.conformal|
     | - Fast A15 Kagome Dual   |  | graphite.explicit.*        |
     | - Direct TPMS Isosurface |  | Complete 3D Watertight Mesh|
     | Sub-second PyVista render|  | Multi-iteration Relaxation |
     +--------------------------+  +-------------+--------------+
                                                 |
                                                 v
                                   +----------------------------+
                                   |   graphite.io.mesh_export  |
                                   | Dual STL + Native 3MF      |
                                   | Automated Manifold Triage  |
                                   +-------------+--------------+
                                                 |
                                                 v
                                   +----------------------------+
                                   |  Static HTTP File Server   |
                                   | /outputs/... (Direct D/L)  |
                                   +----------------------------+
```

### Module Responsibilities

| Component | Module | Responsibility |
| :--- | :--- | :--- |
| **Interactive Studio** | [`graphite/ui/trame_app.py`](../graphite/ui/trame_app.py) | Trame Vuetify3 drawer controls, reactive state, PyVista viewport framing, and direct HTTP file downloads. |
| **Surface Preview Engine** | [`graphite/ui/surface_preview.py`](../graphite/ui/surface_preview.py) | Sub-second visual checks: adaptive TPMS boundary clipping, decoupled fast surface dual extraction for explicit lattices, and cutaway sectioning. |
| **Headless CLI** | [`graphite/ui/cli.py`](../graphite/ui/cli.py) | Terminal runner executing full volumetric lattice generation from CLI flags or JSON/YAML recipes. |
| **Dual-Format Exporter** | [`graphite/io/mesh_export.py`](../graphite/io/mesh_export.py) | Dual `.stl` and `.3mf` export with automated manifold verification (`[PASS]` / `[WARN]`). |

---

## 2. Current Capabilities

### A. Primary Modality Switching
The application provides a top-level **Modality Selector** that dynamically reconfigures the drawer controls:
1. **Implicit TPMS**: Continuous, mathematically smooth triply periodic minimal surfaces.
2. **Explicit Struts**: Discrete beam/strut scaffolds with conformal boundary ironed nodes.

### B. Geometry Ingestion
Supports three distinct geometry pipelines:
- **Analytical Primitives**: Parametric `Cube`, `Cylinder`, `Sphere`, and `Toros` generated on-the-fly.
- **Workspace Fixtures**: Auto-discovers all 33 production fixtures in `test_parts/` (e.g. `BaseRing_1to1.STL`, `BookendLatticeSection_Sloped.STL`, `Mouse wrist rest v1.stl`, `PigSkullImplant_SideOfSkull.stl`).
- **Custom STL Uploads**: Interactive browser file upload (`v3.VFileInput`) streaming to `outputs/uploads/`, or direct local filesystem path specification.

### C. Implicit TPMS Catalog & Features
- **7 Minimal Surface Architectures**: Gyroid, Diamond (Schwarz-D), Schwarz-P, Schwarz-Diamond, Neovius, Lidinoid, and Split-P.
- **Outer Solid Shelling**:
  - `Core`: Lattice fills entire CAD interior; edges exposed at boundary.
  - `Skin`: Solid boundary shell of thickness $t_{\text{shell}}$ with hollow core.
  - `Combined`: Conformal Boolean merge of the solid exterior shell with interior lattice.
- **Axial Grading**: Chirp frequency cell size variation along X, Y, or Z with user-defined doubling intervals.
- **2-out-of-3 Density Calculator**: Automatically solves between Unit Cell Size ($L$), Target Solid Fraction ($\phi$), and Physical Wall Thickness ($w$):
  $$\tau \approx \frac{\phi}{1.15}, \qquad w \approx \frac{\tau \cdot L}{\pi}$$
- **Adaptive Nyquist Voxel Resolution**: Automatically enforces $\text{Res} = \min(0.250\text{ mm}, w / 2)$ to guarantee that thin walls are spanned by at least two voxels, eliminating non-manifold holes.

### D. Explicit Strut Architecture Catalog & Features
- **A15 Conformal Kagome**: Body-centered cubic supercells with tetrahedral decomposition, surface-projected depth-0 nodes, and interior Jacobi relaxation.
- **Modular Simple Cubic Hexagonal (SC)**: Hex grid scaffolding with plug-and-play topology rules: `octahedral`, `cubic`, and `kelvin`.
- **Conformation Modes**:
  - `conformal`: Surface-first nodal boundary conformation with ironed valency.
  - `boolean`: Spatial intersection against the CAD boundary.
- **Analytical Sizing Solver**: Computes analytical strut radius from target unit cell size and solid fraction.

### E. Dual STL & 3MF Export with In-Browser Streaming
- Generates both binary `.stl` and native `.3mf` files containing metadata and unit scale.
- Files are saved to `outputs/App outputs/YYYY-MM-DD/`.
- Download buttons trigger instantaneous browser file streams via a hidden client iframe pointing to the HTTP static mount (`/outputs/...`), preventing WebSocket disconnections.

### F. Blender Dark Industrial Theme
- **Viewport Background**: Blender dark charcoal `#2E3035`.
- **Build Plate**: Carbon-style dark grid (`#232528` with `#40444C` edges).
- **CAD Model**: Titanium gray solid body (`#D8DCE3`, opacity 0.88).
- **Lattice Wireframe / TPMS Footprint**: Vibrant electric cyan (`#0FA4AF`, line width 2.5).
- **Boundary / Conformal Nodes**: Warm coral spheres (`#E07A5F`, point size 8).
- **Drawer / UI**: Oceanic slate teal (`#002528` / `#002C30`).

---

## 3. Fast Conformal Surface Dual Preview Engine

### The "Preview Bottleneck" Problem
In earlier iterations, clicking **"Update 3D Preview"** for explicit lattices executed the entire production generator (`generate_a15_conformal_lattice` or `generate_conformal_lattice`) with `skip_sweep=True`. While this avoided 3D pipe sweeping, it still generated all internal 3D cells, performed spatial KDTree stamping of tens of thousands of internal struts, and ran 15 iterations of iterative Jacobi spring relaxation. For an A15 Kagome cube, this took **11.95 seconds** simply to inspect the unit cell size.

### Mathematical Insight
In both A15 Kagome and Modular SC Hex lattices:
1. Exterior boundary conformation projects boundary nodes (depth 0) onto the CAD surface.
2. Jacobi relaxation leaves depth 0 boundary nodes fixed.
3. The visual surface dual chords (`cyan_struts`) depend **strictly** on boundary faces and boundary quads.
4. **Conclusion**: 100% of interior cell stamping, interior strut connectivity, and interior spring relaxation is redundant for inspecting surface conformation and cell pitch.

### Implementation Details (`graphite/ui/surface_preview.py`)
- **`extract_fast_sc_surface_dual`**:
  1. Computes signed distances on bounding-box hex corners to isolate surviving and boundary hex cells.
  2. Identifies outer boundary quads (quads shared by only 1 surviving cell) using vectorized face hashing.
  3. Projects boundary quad corner nodes directly onto the CAD surface using `trimesh.proximity.closest_point`.
  4. Invokes `generate_hex_surface_dual_cage` to produce the triangular surface dual chords and boundary node points.
- **`extract_fast_a15_surface_dual`**:
  1. Builds canonical sorted face triplets across all A15 tets: `np.sort(tets[:, FACE_TRIPLETS], axis=-1).reshape(-1, 3)`.
  2. Uses `np.unique(..., return_counts=True)` to filter boundary faces (count == 1), halving distance queries from 140,000 to 71,982.
  3. Projects unique boundary face centroids to the CAD surface.
  4. Identifies shared boundary edges and connects adjacent face centroids across shared edges into dual chords.

### Benchmark Speedups

| Geometry & Architecture | Cell Size | Old Full Scaffold Time | Fast Surface Dual Time | Speedup Factor | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cube 20mm — SC Octahedral** | $L=5.0\text{ mm}$ | $1.20\text{ s}$ | **$0.176\text{ s}$** | **$6.8\times$** | 576 dual chords, 194 boundary nodes |
| **Cube 20mm — A15 Kagome** | $L=5.0\text{ mm}$ | $11.95\text{ s}$ | **$2.633\text{ s}$** | **$4.5\times$** | 1,080 dual chords, 720 boundary nodes |
| **BaseRing — SC Octahedral** | $L=5.0\text{ mm}$ | $24.80\text{ s}$ | **$5.515\text{ s}$** | **$4.5\times$** | 4,362 dual chords, 1,419 boundary nodes |

---

## 4. Pitfalls & Engineering "Gotchas" (Lessons Learned)

When developing or extending this application, keep these critical pitfalls in mind:

### Pitfall 1: Browser File Downloads Disconnecting Trame WebSocket
- **Symptom**: Using standard HTML `<a href="..." download>` or `window.location.href = ...` redirects the main browser tab, instantly killing the persistent WebSocket connection (`ws://...`) and dropping the Trame session.
- **Solution**:
  1. Mount the output directory statically on the Trame server:
     ```python
     server.serve["outputs"] = str(Path("outputs").resolve())
     ```
  2. Render a hidden, persistent `<iframe>` in the template and trigger downloads via client-side JavaScript iframe navigation, or link to the static `/outputs/...` route with `target="_blank"`:
     ```html
     <iframe id="download_frame" style="display:none;"></iframe>
     ```

### Pitfall 2: PyVista Multi-Actor Scene Leaks
- **Symptom**: Changing the CAD shape (e.g., from Cube to Cylinder) or switching between Implicit and Explicit modalities causes previous geometries to remain visible as ghost artifacts, or causes viewport rendering to grind to a halt.
- **Solution**: Always call `plotter.clear_actors()` before populating the scene, and assign explicit, unique actor names to meshes (`name="cad_body"`, `name="explicit_struts"`, `name="floor_plate"`). Never instantiate a new `pv.Plotter()` inside a callback—reuse the persistent server plotter.

### Pitfall 3: Vuetify 3 Reactive Options Binding
- **Symptom**: Populating `v3.VSelect(items=[...])` with hardcoded Python lists inside conditional containers caused selections to drop or fail to update reactively when switching tabs.
- **Solution**: Store all dropdown item lists directly in Trame reactive state (`state.xxx_options = [...]`) and pass state variable tuples to components:
  ```python
  state.shape_options = ["Cube", "Cylinder", "Sphere", "Toros"]
  v3.VSelect(items=("shape_options",), v_model=("prim_shape", "Cube"))
  ```

### Pitfall 4: Thin-Wall Meshing Failure at Fixed Resolutions
- **Symptom**: Marching cubes / flying edges on fine TPMS with thin walls ($w < 0.35\text{ mm}$) produced disconnected shells, non-manifold edges, and mesh holes when evaluated on a coarse 0.5mm or 1.0mm voxel grid.
- **Solution**: Enforce Nyquist spatial sampling: $\text{Res} \le w / 2$. Capping at $250\,\mu\text{m}$ for speed gives:
  ```python
  resolution = min(0.250, wall_thickness / 2.0)
  ```

### Pitfall 5: Positional vs. Keyword Argument Misbinding
- **Symptom**: In `graphite/ui/surface_preview.py`, `generate_explicit_preview` has `cad_mesh` as parameter 3 and `cad_file` as parameter 4. Calling `generate_explicit_preview("Cube", 20.0, None, "A15", ...)` caused `"A15"` to be parsed as the `cad_file` string, throwing `FileNotFoundError: CAD mesh file not found: A15`.
- **Solution**: **Always call generator and preview functions using keyword arguments**:
  ```python
  generate_explicit_preview(
      shape="Cube",
      size=20.0,
      lattice_type="A15",
      rule_name="octahedral",
      cell_size=5.0,
      surface_only=True,
  )
  ```

### Pitfall 6: Quadratic Face Hashing in Mesh Culling
- **Symptom**: Iterating over all tetrahedron faces with Python loops or set lookups takes several minutes on grids with $>10^5$ cells.
- **Solution**: Vectorize using NumPy:
  ```python
  triplets = np.sort(tets[:, FACE_TRIPLETS], axis=-1).reshape(-1, 3)
  unique_faces, inverse, counts = np.unique(triplets, axis=0, return_inverse=True, return_counts=True)
  boundary_faces = unique_faces[counts == 1]
  ```

---

## 5. Guide for Future AI Agents & Developers

If you are a fresh AI agent or engineer continuing work on Graphite UI, follow these directives:

1. **Verify the External Core Spec**:
   - Run `python scripts/export_core_spec.py --check` before and after modifying any lattice generation, cell topology, or API code. If the spec diverges, update [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md) and ensure `--check` passes with return code 0.
2. **Never Write Output Files to `test_parts/`**:
   - `test_parts/` is strictly read-only for input fixtures.
   - All generated meshes, test models, and exports must be written to `outputs/` (or subdirectories like `outputs/App outputs/YYYY-MM-DD/`).
3. **Preserve the Preview / Production Decoupling**:
   - The interactive 3D preview (`Update 3D Preview`) is designed for fast, sub-second visual feedback. Never put full 3D Marching Cubes, full 3D strut sweeps, or multi-iteration iterative relaxation into the preview path.
   - The production path (`Generate Production Lattice`) is where full watertight volumetric meshes are synthesized.
4. **Adding New Architectures**:
   - **Adding a new TPMS**: Add the mathematical formula to `graphite/math/tpms.py`, register the name in `SUPPORTED_TPMS_EQUATIONS` in `graphite/ui/cli.py`, and add the analytical calibration constant to `tau_from_wall_thickness_mm`.
   - **Adding a new Explicit Unit Cell**: Implement the unit cell topology in `graphite/explicit/unit_cells.py` conforming to the Unit Cell protocol, register the rule in `SC_RULES`, and add fast boundary face extraction logic if it diverges from standard cubic hex cages.
5. **How to Launch the Application Locally**:
   ```bash
   python -m graphite.ui
   ```
   Server defaults to `http://localhost:8080/`. To specify an alternate port:
   ```bash
   python -m graphite.ui --port 8085
   ```
