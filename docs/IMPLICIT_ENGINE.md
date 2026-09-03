# 1. Project Overview & File Manifest

## Goal

The **implicit lattice engine** evaluates **triply periodic minimal surface (TPMS)** and related **implicit scalar fields** \(f(\mathbf{x})\) on a **regular 3D voxel grid** aligned to world coordinates (mm). The **solid region** is defined by a **level set** (typically the zero isosurface of a composite field combining TPMS thickness / solid-fraction control with a **CAD signed distance field**). The volume is meshed with **marching cubes** to produce a triangle mesh for export (STL).

This path is **orthogonal** to the explicit strut/GMSH pipeline: no tetrahedral scaffold; the geometry is **field-first**.

## Canonical filepaths (production / `graphite`)

| Role | Path |
|------|------|
| TPMS scalar formulas (vectorized) | `graphite/math/tpms.py` |
| Voxel grid + CAD SDF (EDT) | `graphite/geometry/masking.py` |
| Uniform conformal TPMS inside STL | `graphite/implicit/conformal.py` |
| **Chirped** spatial frequency (pore size / \(k\) field) | `graphite/implicit/chirped.py` |
| Graded **solid fraction** (constant \(k\), varying threshold) | `graphite/implicit/graded.py` |
| Z-layer osteochondral-style \(L(z)\), \(SF(z)\) | `graphite/implicit/osteochondral.py` |
| **Lofted** 1D unit-cell / SF grading (integrated phase, any Cartesian spine) | `graphite/implicit/field_driven.py` — see [LOFTED_GRADING.md](LOFTED_GRADING.md) |
| Boundary-driven grading (surface IDs + dual EDT blend) | `graphite/implicit/boundary_graded.py` |
| Streamlit wiring | `app.py` (imports above generators) |

## Reference / R&D & tests

| Role | Path |
|------|------|
| Early block gyroid + box SDF | `experiments/Implicit_Lattice_Exploration/core/gyroid_generator.py` |
| Radial chirp diagnostic (pure math cylinder) | `experiments/Implicit_Lattice_Exploration/tests/test_radial_chirp.py` |
| Related implicit tests | `experiments/Implicit_Lattice_Exploration/tests/test_*gyroid*.py`, `test_*chirp*.py` |
| Design spec (Manifold `LevelSet` *not* used in current marching-cubes path) | `docs/graphite_implicit_spec.md` |

---

# 2. The Core Mathematical Equations

All TPMS evaluations share a **wavenumber** \(k = 2\pi / L\) where \(L\) is the **spatial period** (unit-cell scale) in mm. The API passes **`k` as a scalar or as a full 3D array `K_grid`** (same shape as `X, Y, Z`) into `evaluate_tpms`.

## 2.1 Implemented surfaces (`graphite/math/tpms.py`)

Let \(X,Y,Z\) be world-coordinate arrays (mm). For scalar \(k\):

**Gyroid**

\[
G(X,Y,Z) = \sin(kX)\cos(kY) + \sin(kY)\cos(kZ) + \sin(kZ)\cos(kX)
\]

**Schwarz P** (labeled `schwarz-p`, `schwarz primitive`, `schwarz`)

\[
P(X,Y,Z) = \cos(kX) + \cos(kY) + \cos(kZ)
\]

**Diamond** (Schwarz diamond / F-type combination in code)

\[
\begin{aligned}
D(X,Y,Z) = {} & \sin(kX)\sin(kY)\sin(kZ) \\
& + \sin(kX)\cos(kY)\cos(kZ) \\
& + \cos(kX)\sin(kY)\cos(kZ) \\
& + \cos(kX)\cos(kY)\sin(kZ)
\end{aligned}
\]

**Neovius**

\[
N(X,Y,Z) = 3\bigl(\cos(kX)+\cos(kY)+\cos(kZ)\bigr) + 4\cos(kX)\cos(kY)\cos(kZ)
\]

**Lidinoid**

\[
\begin{aligned}
L_d(X,Y,Z) = {} & \sin(2kX)\cos(kY)\sin(kZ) \\
& + \sin(2kY)\cos(kZ)\sin(kX) \\
& + \sin(2kZ)\cos(kX)\sin(kY) \\
& - \cos(2kX)\cos(2kY) - \cos(2kY)\cos(2kZ) - \cos(2kZ)\cos(2kX) + 0.3
\end{aligned}
\]

**Split-P** (code uses mixed harmonics)

\[
\begin{aligned}
t_1 = {} & \sin(2kX)\sin(kZ)\cos(kY) + \sin(2kY)\sin(kX)\cos(kZ) + \sin(2kZ)\sin(kY)\cos(kX) \\
t_2 = {} & \cos(2kX)\cos(2kY) + \cos(2kY)\cos(2kZ) + \cos(2kZ)\cos(2kX) \\
t_3 = {} & \cos(2kX) + \cos(2kY) + \cos(2kZ) \\
F = {} & 1.1\,t_1 - 0.2\,t_2 - 0.4\,t_3
\end{aligned}
\]

**Broadcasting:** When `K_grid` is an array, the same formulas apply elementwise: e.g. \(\sin(K_{\text{grid}} \odot X)\) in NumPy terms.

## 2.2 Sheet solid field (level-set thickness)

All discrete TPMS functions in `graphite/math/tpms.py` (e.g., `gyroid()`, `schwarz_p()`) natively compute spatial scaling from a `unit_cell_size` parameter, and support computing network thickness directly using `iso_offset` and `is_sheet` flags:

- If `is_sheet=True`: \(\text{result} = \lvert F(\mathbf{x})\rvert - \text{iso\_offset}\) (thick network sheets).
- If `is_sheet=False`: \(\text{result} = F(\mathbf{x}) - \text{iso\_offset}\) (solid skeletal networks).

**Conformal** path (`conformal.py`) maps **user pore size** to period \(L\) when `pore_size` is given:

\[
L = \frac{\text{pore\_size}}{1 - 1.15\,\phi}, \quad \phi = \text{solid\_fraction}, \quad k = \frac{2\pi}{L}
\]

(If `unit_cell_size` is passed instead, \(L\) is that value directly.)

Symmetric shell around the zero level set of \(F\):

\[
t = \frac{3}{2}\,(2\,\phi - 1) = 1.5\,(2\phi - 1)
\]

\[
\text{solid\_field}(\mathbf{x}) = \lvert F(\mathbf{x})\rvert - \lvert t\rvert
\]

The **solid** region corresponds to \(\text{solid\_field} \le 0\) (interior of the thickened TPMS sheet in this convention).

**Chirped / graded** paths often use:

\[
\text{solid\_field} = \lvert F\rvert - \text{SF\_grid}
\]

where `SF_grid` is either a scalar solid-fraction proxy or a spatially varying grid (see §3).

## 2.3 Coordinate generation & resolution

**Inside CAD** (`voxelize_mesh_and_edt` in `graphite/geometry/masking.py`):

1. `trimesh` voxelization: `mesh.voxelized(pitch=resolution).fill()`.
2. Pad the binary inside mask by `pad_width` voxels (default **4**).
3. Build **1D axes** in mm:

   `x_axis[i] = i * resolution + padded_min_bound[0]` (similarly for `y`, `z`).

4. **3D world grid** via **`np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")`** → `X, Y, Z` with shape `(nx, ny, nz)`.

**Typical `resolution`:** **0.25 mm** voxel pitch is the default in multiple generators; it directly sets memory \(\propto 1/\text{resolution}^3\) and marching-cubes triangle count.

**Legacy block** (`gyroid_generator.py`): centered **`np.linspace`** on `[-(size+pad)/2, +(size+pad)/2]` then `np.meshgrid(..., indexing="ij")`.

**Note:** `np.ogrid` is **not** used in the canonical path; full dense `meshgrid` arrays are built for EDT and TPMS evaluation.

---

# 3. The "Chirp" (Gradient) Architecture

**Terminology in code:** “Chirping” here means **spatially varying wavenumber** \(k(\mathbf{x})\) (hence **varying unit-cell period** \(L(\mathbf{x}) = 2\pi/k\)), implemented as a **per-voxel array `K_grid`** fed into `evaluate_tpms`. This is **not** a phase-only chirp; the **argument** of the sines/cosines is **\(K \odot X\)** (elementwise), i.e. **local spatial frequency** changes.

## 3.1 Primary implementation: `generate_chirped_lattice` (`chirped.py`)

**Base and modified scales** (derived from mesh **X extent** in the implementation):

\[
L_{\text{base}} = \frac{\text{size}_x}{6}, \quad L_{\text{mod}} = \frac{L_{\text{base}}}{2}
\]

\[
k_{\text{base}} = \frac{2\pi}{L_{\text{base}}}, \quad k_{\text{mod}} = \frac{2\pi}{L_{\text{mod}}}
\]

**Weight field** \(W(\mathbf{x}) \in [0,1]\) with **Hermite smoothstep** (C¹):

\[
W_{\text{lin}} = \mathrm{clip}(\cdots,\,0,\,1), \quad W = 3W_{\text{lin}}^2 - 2W_{\text{lin}}^3
\]

**Drivers for \(W\):**

1. **Axis-aligned (X, Y, Z):**  
   \(t\) is `X`, `Y`, or `Z`; normalize to \([0,1]\) using padded bbox min/max:
   \[
   W_{\text{lin}} = \mathrm{clip}\left(\frac{t - t_{\min}}{t_{\max} - t_{\min} + \varepsilon},\,0,\,1\right)
   \]

2. **Radial (cylindrical):**  
   \(t = \sqrt{X^2 + Y^2}\), \(t_{\min}=0\), \(t_{\max}=\max(t)\).

3. **Modifier STL:** Voxelize modifier, EDT SDF `mod_sdf`, interpolate onto `(X,Y,Z)`. Distance `d` to modifier surface:
   \[
   W_{\text{lin}} = \mathrm{clip}\left(1 - \frac{d}{T},\,0,\,1\right), \quad T = \text{transition\_width}
   \]
   then same smoothstep \(W\).

**Chirped wavenumber field:**

\[
K_{\text{grid}} = k_{\text{base}}\,(1 - W) + k_{\text{mod}}\,W
\]

**TPMS:**

\[
F = \text{evaluate\_tpms}(\text{lattice\_type},\, K_{\text{grid}},\, X,\, Y,\, Z)
\]

**Solid field:**

\[
\text{solid\_field} = \lvert F\rvert - \phi, \quad \phi = \text{solid\_fraction}
\]

So: **large pores ↔ smaller \(k\)** where \(W\to 0\); **small pores ↔ larger \(k\)** where \(W\to 1\). The transition is **linear in \(W\)** in \(k\)-space, smoothed by smoothstep in the **driver** coordinate.

## 3.2 Related gradient modes (not the same as chirp)

- **`graded.py`:** **Constant \(k\)**; gradient is on **solid fraction** \(\text{SF\_grid} = \phi_{\min} + W(\phi_{\max}-\phi_{\min})\) with the same \(W\) machinery (axis / radial / modifier).
- **`piecewise_bands.py`:** **Hard Z bands** (or box height bands) with piecewise-constant **L** and **τ** on a **single-pass** full-domain EDT field — preferred for Gmsh volume meshing. API: `splitp_piecewise_cylinder_single_pass`, `splitp_piecewise_box_single_pass`, `cumulative_phase_w_from_l_profile`. CLI: `scripts/generate_piecewise_splitp_implicit.py`.
- **`piecewise_woodpile.py`:** **Hard Z bands** for **cross-hatch woodpile** (`true_woodpile=False`) on a **single-pass** full-cylinder EDT field — implicit MC backend when `generator='implicit'`. Extrude backend: `graphite/explicit/woodpile_extrude.py`. Phase helpers: `graphite.math.woodpile_anchor`. CLI: `scripts/generate_piecewise_woodpile.py` (default `generator=extrude`). See [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md#8-piecewise-cross-hatch-woodpile-jul-2026) and [EXPLICIT_WOODPILE_EXTRUSION.md](EXPLICIT_WOODPILE_EXTRUSION.md).
- **`osteochondral.py`:** **1D interpolation** along **relative Z**: `L_grid = interp(Z_rel, z_heights, pore_sizes)`, `K_grid = 2π/L_grid`, `SF_grid` interpolated similarly — **piecewise linear** in height; **integrated phase** along Z via `calculate_integrated_phase` (Z-only osteochondral loft).
- **`field_driven.py` (lofted):** Same **integrated-phase** spine logic along **Cartesian X / Y / Z** (or cylindrical/spherical radius) with user control points; `grade_unit_cell=True` + `unit_cell_control_points`. Documented as experimental in [LOFTED_GRADING.md](LOFTED_GRADING.md).
- **`boundary_graded.py`:** **Distance from logical surfaces** (facet groups) drives `L_grid` and `SF_grid` with **dual-EDT smoothstep** blend toward end-surface parameters — preferred when **both** ends are arbitrary non-parallel surfaces.
- **`test_radial_chirp.py`:** Standalone **cylindrical** domain; \(W\) from **radial distance** from a **core radius** `r_core` with `transition_width` — documents an alternate radial chirp prototype.

---

# 4. Meshing & Boundary Logic

## 4.1 Meshing: `skimage.measure.marching_cubes`

Canonical extraction:

```python
verts, faces, _, _ = marching_cubes(
    final_field.astype(np.float32),
    level=0.0,
    spacing=(resolution, resolution, resolution),
)
verts_world = verts * resolution + padded_min_bound
```

The **implicit solid** is the set \(\{\mathbf{x} : \text{final\_field}(\mathbf{x}) \le 0\}\) (convention used with `maximum` composition below).

## 4.2 CAD boundary: EDT-based SDF + `np.maximum`

**Inside mask → SDF** (`masking.py`):

- `inside_dist = edt(inside_mask)`, `outside_dist = edt(~inside_mask)`
- **`cad_sdf = (outside_dist - inside_dist) * resolution`**

Sign convention: **negative inside** the solid CAD, **positive outside** (after scaling by voxel pitch).

**Composition with TPMS solid field:**

\[
\text{final\_field} = \max(\text{solid\_field},\,\text{cad\_sdf})
\]

This is an **R-function-style** max-of-fields: the **zero level set** lies on the **CAD boundary** where the TPMS would otherwise extend outside — i.e. **voxel-domain confinement** of the implicit sheet to the filled voxel hull of the mesh (watertight fill recommended).

**Conformal extras** (`conformal.py`): optional **skin** / **core** / **combined** modes using additional distance fields from **surface sampling** + EDT for localized shell thickness; still merged with `marching_cubes` on `final_field`.

**Primitive box trim** (`gyroid_generator.py`): **analytic box SDF**

\[
\text{box\_sdf} = \max(|X|,|Y|,|Z|) - \text{size}/2
\]

then `final_field = max(solid_field, box_sdf)`.

**Modifier-driven chirp:** Uses **voxelized modifier** and **RegularGridInterpolator** of modifier SDF for **weight \(W\)**, not for the final CAD trim (CAD trim remains `cad_sdf` from the **part** mesh).

---

# 5. Known Limitations / Bottlenecks

1. **Memory:** Full `X, Y, Z`, `cad_sdf`, and field arrays are **dense** `float64`/`float32` arrays of size `nx×ny×nz`. Halving `resolution` can **8×** voxel count → RAM pressure and slower EDT / marching cubes.

2. **Marching cubes cost & artifacts:** `skimage.measure.marching_cubes` time scales with grid size and output complexity; **stairstepping** and **resolution-dependent** wall thickness appear on diagonal features and thin sheets.

3. **Voxelization fidelity:** Boundary confinement follows **`mesh.voxelized(pitch=resolution).fill()`** — accuracy is limited by **pitch** and mesh watertightness; small features below **~1 voxel** are lost or merged.

4. **SDF approximation for skins:** Localized shell modes sample **surface points** and EDT a sparse **surface mask** — a **voxelized approximation** to distance-to-surface, not an exact analytic SDF.

5. **Spec vs implementation:** `docs/graphite_implicit_spec.md` describes a **Manifold3D `LevelSet`** path and **no skimage marching cubes**; the **implemented** generators in `graphite/implicit/*.py` currently use **marching cubes** throughout. Future migration would change meshing only if code is switched.

6. **Chirp parameter coupling:** In `chirped.py`, \(L_{\text{base}}\) is tied to **bounding-box X extent** / 6, not an independent user “pore size” in that module — architects should treat that as a **fixed design choice** unless refactored.

---

# 6. Pore Metrics (Effective vs Cross-Sectional)

Graphite now exposes two pore-size concepts for implicit TPMS workflows:

1. **Effective pore size**  
   Measured from the realized lattice void geometry (voxel mask) using EDT-derived inscribed diameters.
2. **Cross-sectional pore size**  
   Measured from a local parent unit cell with the same TPMS equation, local pore-size input, and local solid fraction:
   - **2D**: largest inscribed circle on the parent cell mid-plane.
   - **3D**: largest inscribed sphere in the parent cell void.

Primary API entry points:

- `graphite.implicit.compute_effective_pore_size(void_mask, resolution_mm, z_bins=...)`
- `graphite.implicit.compute_cross_sectional_pore_size(lattice_type, pore_size_mm, solid_fraction, ...)`
- `graphite.implicit.compute_pore_metrics_for_z_graded(lattice_type, z_samples_mm, pore_sizes_mm, solid_fraction, ...)`

Interpretation notes:

- Effective pore size is geometry-realized and includes discretization and boundary effects.
- Cross-sectional pore size is a local parent-cell proxy and is designed for Z-graded interpretation (slice behaves like local uniform lattice).
- Resolution matters: measurements below ~2 to 3 voxels per diameter are flagged as lower confidence.

---

# 7. TPMS Calibration Workflow

Graphite includes calibration helpers that solve TPMS parameters to match both target pore size and target solid-fraction.

Core API:

- `graphite.implicit.calibrate_tpms_point(...)`
- `graphite.implicit.calibrate_tpms_gradient_profile(...)`

Point calibration solves for:

- `L_local`: local period/frequency control.
- `tau_local`: local sheet threshold (effective thickness control).

Residuals:

- `r_pore = measured_mis_pore_mm - target_pore_mm`
- `r_sf = measured_effective_sf - target_solid_fraction`

Metrics in the loop:

- **Pore metric**: maximal-inscribed-sphere (MIS), with boundary guard to reduce open-edge bias.
- **Solid-fraction metric**: effective voxel occupancy in sampled local TPMS volume.

Gradient modes:

- **Hybrid (default)**: calibrate anchors -> interpolate profile -> sparse corrections at checkpoints.
- **Full profile**: calibrate each control point directly.

Seed initialization:

- Optional lookup from `graphite/implicit/calibration_seed_table.json`.
- Heuristic fallback when no seed match exists.

Guidance:

- Use hybrid first for runtime efficiency.
- Use full-profile for high nonlinearity or strict checkpoint accuracy.
- Keep resolution sufficiently fine for sub-mm targets, or convergence may be noisy.

## 8. High-Performance Meshing Backend (PyVista/VTK Flying Edges)

Implicit extraction uses high-performance VTK Flying Edges as the default meshing backend:

- `graphite/implicit/meshing_backends.py`

Supported backends:

- `pyvista_flying_edges` (DEFAULT): Multi-threaded C++/VTK algorithm offering 5x–10x faster extraction and isotropic, equi-angled triangle meshes that eliminate the diagonal facet striping of classic Marching Cubes.
- `marching_cubes`: Single-threaded scikit-image fallback.
- `auto`: Alias for `pyvista_flying_edges` with automatic fallback.

Interface:

- `extract_isosurface(field, spacing, origin, backend="pyvista_flying_edges", level=0.0, ...)`

Key Features & Quality Gates:

- **Automatic Winding & Normal Repair**: VTK isosurfacing produces clockwise triangle winding; the engine automatically tests `if mesh.volume < 0.0: mesh.invert()` to guarantee outward-facing surface normals for watertight solids.
- **Graceful Fallback**: If PyVista or VTK is unavailable/fails at runtime, extraction automatically falls back to `marching_cubes` and records fallback metadata.
- **Watertight Hardening**: Supports optional post-processing hole filling and voxel rewrap passes (`enforce_watertight=True`).
- **Surface Micro-Structures**: Integrates directly with `graphite.implicit.surface_textures` (transverse normal projection) and `graphite.implicit.micropillars` (smoothed vertex normal hair synthesis).

## 9. Export formats (STL / STEP)

Lattice meshes are triangle surfaces from marching cubes. Export is centralized in:

- `graphite/io/mesh_export.py`

### STL (default)

- Written with `trimesh.export` (binary/ASCII per trimesh defaults).
- Fast; suitable for printing and mesh viewers.

### STEP (optional, faceted solid)

- Converted with **Gmsh**: temporary STL → `classifySurfaces` → `createGeometry` → volume → `gmsh.write`.
- Produces a **tessellated** solid for CAD import/FEA, not an analytic NURBS B-rep.
- Requires `gmsh` (`pip install gmsh`).
- From **Streamlit**, conversion runs in a **subprocess** (`python -m graphite.io.step_export_worker`) because Gmsh uses signal handlers that only work on Python's main thread.
- Large lattices (fine `resolution`) can produce very large files and long conversion times; a default face-count guard applies.

API:

```python
from graphite.io.mesh_export import export_mesh, formats_from_request

export_mesh(mesh, "outputs/lattice", formats=formats_from_request("both"))
# writes outputs/lattice.stl and outputs/lattice.step
```

## 10. Surface Micro-Structures Engine (Microgrooves & Micropillars)

Graphite provides a dual-engine procedural micro-texturing suite for medical scaffolds that operates directly on the extracted isosurface without requiring billions of micro-voxels:

- `graphite/math/textures.py` (Procedural displacement mathematics)
- `graphite/implicit/surface_textures.py` (Mesh-level microgroove displacement)
- `graphite/implicit/micropillars.py` (Explicit microfiber / pillar forest synthesis)

### 10.1 Microgrooves via Transverse Normal Projection

Periodic surface relief (50 µm depth × 50 µm pitch) for osteoblast contact guidance is applied via adaptive mesh subdivision and normal displacement:

$$\mathbf{n}_{\perp} = \mathbf{n} - (\mathbf{n} \cdot \mathbf{d})\mathbf{d}$$

Where $\mathbf{d}$ is the groove alignment direction vector (e.g. $[0, 0, 1]$). Displacing along the transverse normal $\mathbf{n}_{\perp}$ rather than the full 3D normal guarantees zero normal fold-overs, wrinkles, or self-intersections on curved TPMS walls.

### 10.2 Micropillars & Microfibers Forest Synthesis

High-aspect-ratio cylindrical posts (50–100 µm diameter × 200–400 µm length) are synthesized via Manifold3D CSG boolean union:

1. **Normal-Consistent Thin-Wall Preservation**:
   On thin TPMS sheets ($\sim 150\ \mu\text{m}$ wall thickness), Euclidean spheres penetrate through the solid wall and cause opposing faces to cancel out. The engine enforces normal compatibility ($\mathbf{n}_i \cdot \mathbf{n}_j \ge -0.3$), ensuring both sides of every internal porous channel are fully populated.
2. **Even Spacing via Tangential Particle Relaxation (`distribution="relaxed"`)**:
   In addition to blue-noise Poisson-disk (`distribution="poisson_disk"`), points slide along the local 2D tangent plane under neighbor repulsion forces, converging into a near-perfect hexagonal honeycomb grid across all 3D curved surfaces.
3. **3D-Printability Overhang Filtering (`filter_printable=True`)**:
   Calculates the normal inclination relative to the horizontal build plate ($\alpha = \arcsin(|n_z|)$) and retains only self-supporting hairs within $60^\circ \le \alpha \le 90^\circ$ from horizontal ($|n_z| \ge \sin(60^\circ) \approx 0.866$).
4. **Selective Primitive Boundary Face Placement**:
   Supports targeting specific planar and curved faces on primitive envelopes:
   - **Box/Cube**: `+x`, `-x`, `+y`, `-y`, `+z`, `-z` (synonyms: `top`, `bottom`, `front`, `back`, `left`, `right`).
   - **Cylinder**: `top`, `bottom`, `sides`.
   - **Sphere**: `outer`.
   - **Regions**: `all`, `internal_only` (pore lumen only), `outer_only` (exterior skin only).
5. **45° Boundary Normal Clamping**:
   Normals on planar CAD cut edges are clamped strictly to the face axis (e.g. $[0, 0, 1]$ for $+Z$), eliminating corner-tilting artifacts.
6. **Anatomic CAD Boundary Segmentation (`segment_cad_boundary`)**:
   Automatically segments organic, freeform patient-specific implant cutouts (such as cranial plates) into `top` (cranial dome), `bottom` (dura side), and `sides` (through-thickness osteotomy cut) using dihedral crease edge detection ($> 35^\circ$). By selecting `selected_faces=("top", "bottom")`, the side rim remains completely smooth and hair-free for flush surgical insertion.
7. **Directional Fiber Alignment (`orientation`)**:
   - `"local_normal"`: Fibers project perpendicular to the local curved TPMS strut curvature.
   - `"z_aligned"`: Fibers act as vertical micro-pins (top surface points strictly $+Z$, bottom surface points strictly $-Z$).
   - `"cad_normal"`: Fibers project perpendicular to the anatomic skull curvature dome.
8. **Exact B-Rep Boundary Trimming (Overscaled Domain + CSG Intersection)**:
   To eliminate voxel stairstepping, rounded corner bevels, and discretization artifacts along sharp CAD cuts, the boundary envelope is scaled up (e.g. by 5% about its center of mass) during voxelization. The TPMS isosurface is extracted on this overscaled domain and subsequently trimmed against the unscaled, razor-sharp CAD mesh using exact Manifold3D boolean intersection ($\mathcal{M}_{\text{clean}} = \mathcal{M}_{\text{oversized}} \cap \mathcal{M}_{\text{CAD}}$). All exterior cuts retain exact planar and cylindrical CAD facet sharpness before micropillar synthesis.

---

*Document generated from repository source as of the workspace snapshot; equations match `graphite/math/tpms.py` and implicit engines under `graphite/implicit/`.*
