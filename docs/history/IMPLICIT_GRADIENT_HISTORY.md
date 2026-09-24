# Historical Gradient Implicit Scaffolds in Graphite

This document summarizes how Graphite has historically produced gradient implicit scaffolds, including:
- conformal TPMS generation inside arbitrary STL geometry,
- variable pore-size and variable-thickness grading,
- osteochondral control-surface style layering,
- dual-EDT boundary/control-surface workflows.

It complements:
- `docs/Implicit_TPMS_Architecture.md` (math/file manifest),
- `Oldinfo/README.md` (historical Split-P experiments),
- `docs/history/CONTROL_SURFACES_PLAN.md` (design intent for boundary grading),
- `tests/test_splitp_comparisons.py` and `tests/test_matlab_debug.py` (recent method-comparison debug tools).

---

## 1) Big Picture: The Historical Production Pattern

Historically, Graphite's implicit gradient stack has followed a consistent production pipeline:

1. **Load boundary geometry** (`trimesh` STL).
2. **Voxelize boundary + build CAD SDF** via EDT using `graphite/geometry/masking.py` (`voxelize_mesh_and_edt`).
3. **Build TPMS scalar field** (Gyroid, Split-P, etc.) using `graphite/math/tpms.py` (`evaluate_tpms`).
4. **Apply gradient logic** to either:
   - local frequency / pore-scale (`L` or `k` field),
   - local sheet threshold / apparent solid fraction (`SF` field),
   - both (in osteochondral and boundary-driven modes).
5. **Conformal intersection** with CAD: `final_field = max(lattice_field, cad_sdf)` (inside-only geometry).
6. **Extract mesh** with `skimage.measure.marching_cubes(level=0)`.

This field-first approach is the core implicit architecture and differs from explicit strut/scaffold meshing.

---

## 2) Core Conformal Method (Uniform Baseline)

### Engine
- `graphite/implicit/conformal.py` (`generate_conformal_lattice`)

### What it does
- Computes a uniform TPMS period (`L`, `k`) from pore size (or unit cell size).
- Evaluates TPMS everywhere on voxel grid.
- Builds sheet field via `abs(F) - abs(t)` where `t = 1.5*(2*solid_fraction - 1)`.
- Uses CAD SDF for conformal clipping.

### Surface-conforming behavior
- Conformality is not done by direct mesh projection; it is done by implicit boolean composition with `cad_sdf`.
- This gives robust inside-volume clipping even for complex parts, at voxel resolution limits.

### Shell modes and selected surfaces
- Supports `core`, `skin`, and `combined`.
- If surface IDs are provided, it samples those facets and uses EDT to derive localized skin distance fields.
- This was one of the first "surface-aware" implicit controls in the conformal stack.

---

## 3) Gradient Modes in Historical Use

Graphite has shipped several implicit gradient modes exposed through `app.py` Step 4/Step 6 routing:

### A. Variable Pore Size (Chirped)
- Engine: `graphite/implicit/chirped.py` (`generate_chirped_lattice`)
- Strategy:
  - Build smooth weight field `W` (axis/radial/modifier-driven),
  - Blend `k_base` and `k_mod` into `K_grid`,
  - Evaluate TPMS with spatially varying `k`.
- Strength:
  - intuitive local pore-size variation.
- Historical issue:
  - direct substitution style can show phase-shear / "history rewrite" artifacts in aggressive gradients.

### B. Variable Porosity (Thickness)
- Engine: `graphite/implicit/graded.py` (`generate_graded_lattice`)
- Strategy:
  - keep TPMS frequency fixed,
  - vary local sheet threshold (`SF_grid`) with same `W` machinery.
- Strength:
  - easier control of apparent density while preserving base periodicity.
- Limitation:
  - does not directly drive pore scale in the same way as chirped frequency modes.

### C. Osteochondral (Layered Z)
- Engine: `graphite/implicit/osteochondral.py` (`generate_osteochondral_lattice`)
- Strategy:
  - user supplies `z_heights`, `pore_sizes`, `solid_fractions`,
  - interpolate `L_grid(Z_rel)` and `SF_grid(Z_rel)` from bottom reference,
  - **integrated phase** along Z via `calculate_integrated_phase` (not raw `k*Z`),
  - evaluate TPMS with `evaluate_tpms_phase`.
- Why this mattered for control surfaces:
  - this was the practical "control-point" version used for osteochondral-like axial gradients.
  - conceptually closer to defining material zones/surfaces than simple axis chirp.

### D. Lofted grading (experimental, 2026)
- **Implicit:** `graphite/implicit/field_driven.py` — 1D control points along **Cartesian X/Y/Z** (or radial coordinates) with integrated phase; see `docs/LOFTED_GRADING.md`.
- **Explicit:** `generate_brute_force_fixed_grid_hex_scaffold(..., taper_along)` — uniform spine stations, transverse extents from mesh plane slices (trophy `taper_along='y'`).
- Motivation: osteochondral-style pore schedules on parts **without parallel ends** by picking one functional spine direction instead of full dual-EDT boundary grading.

---

## 4) Control Surfaces / Boundary-Driven Grading (Dual-EDT)

### Design intent
- Captured in `docs/history/CONTROL_SURFACES_PLAN.md`.
- Goal: allow Start and End surfaces to act as physical control boundaries.

### Implemented engine
- `graphite/implicit/boundary_graded.py` (`generate_boundary_graded_lattice`)
- Routed in `app.py` as **"Boundary-Driven (Dual-EDT)"**.

### How it works
1. **Surface grouping**
   - Uses `compute_face_surface_ids` with feature-angle grouping to define logical surfaces.
2. **Distance to Start Surface**
   - Samples selected start facets -> voxel point cloud -> EDT => `D_A`.
3. **Distance to End Surface**
   - Same process for end facets => `D_B`.
4. **Start-surface profile interpolation**
   - Interpolates `L_base(D_A)` and `SF_base(D_A)` from user-provided distance knots.
5. **Dual-EDT transition**
   - Shifts start distance by last knot and blends to end-surface targets with smoothstep:
   - `W_raw = D_A'/(D_A'+D_B+eps)`, `W = 3W_raw^2 - 2W_raw^3`.
6. **Final fields**
   - `L_grid = L_base*(1-W) + end_pore*W`
   - `SF_grid = SF_base*(1-W) + end_sf*W`
   - then TPMS + CAD SDF + marching cubes.

### Why this is important historically
- This is Graphite's first true **surface-to-surface** implicit grading architecture.
- It supersedes simple global axis gradients for anatomically meaningful "from boundary A to boundary B" control.

---

## 5) Modifier-Driven and Radial Control

In both `chirped.py` and `graded.py`, Graphite added nontrivial control drivers:
- **Axis drivers**: X/Y/Z normalized over padded bounds.
- **Radial driver**: cylindrical radius from Z axis.
- **Modifier STL driver**:
  - voxelize modifier STL,
  - derive modifier SDF via EDT,
  - interpolate onto part grid (`RegularGridInterpolator`),
  - build smooth transition map from distance to modifier.

This gave practical regional control before full boundary/control-surface workflows matured.

---

## 6) How Conforming to a Surface Has Been Done

Historically, Graphite's conformal behavior has been SDF-based:
- build `cad_sdf` from voxelized part (`outside_dist - inside_dist`),
- combine TPMS field and CAD SDF with `max`,
- extract zero-isosurface.

This means:
- the lattice intrinsically conforms to the part's interior volume,
- boundary clipping is robust and local,
- quality depends on voxel resolution and mesh watertightness.

Localized shelling/surface ID support in `conformal.py` and full dual-EDT boundary grading in `boundary_graded.py` are the two major "surface-aware" milestones.

---

## 7) Historical Limitations and Lessons Learned

1. **Direct chirp phase artifacts**
   - spatially varying `k` directly in trig arguments can produce non-physical transitions in aggressive gradients.

2. **Voxel-memory scaling**
   - all implicit modes allocate dense 3D arrays; memory and runtime scale rapidly with finer resolution.

3. **Marching-cubes sensitivity**
   - output smoothness and sheet fidelity are resolution dependent.

4. **Surface-approximation limits**
   - control-surface EDT fields are point-cloud approximations of surface distance.

5. **UI/model complexity growth**
   - moving from axis gradients to dual-surface graded fields increased parameter complexity but substantially improved anatomical relevance.

---

## 8) Current Practical Interpretation

Historically, Graphite has evolved from:
- **uniform conformal TPMS**
-> **axis/radial/modifier gradients**
-> **layered osteochondral controls**
-> **dual-EDT surface-driven controls**.

For osteochondral and boundary-conditioned work, the strongest historical direction is:
- explicit control surfaces + distance-domain interpolation (not pure axis chirp),
- conformal clipping via CAD SDF,
- TPMS evaluated with per-voxel `L`/`SF` fields derived from physically meaningful distance controls.
