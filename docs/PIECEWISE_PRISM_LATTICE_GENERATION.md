# Piecewise prism lattice generation (Split-P & cross-hatch)

This document records the **May 2026** work on **discrete-thirds** and **linear** implicit lattices in **3 × 1.5 × 5 mm** rectangular prisms, plus the **inverted cross-hatch** variant. It complements [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md) and [TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md).

---

## 1. Problem we solved

### Open caps on slab meshes

Early piecewise tests meshed each height band with:

1. A **binary** solid mask `(field ≤ level) ∧ inside_domain`
2. **`max(binary_field, analytic_box_or_cylinder_SDF)`** for caps

That produced **open-looking top/bottom rims** on individual slabs — not a marching-cubes bug, but **different boundary treatment** than the production woodpile path.

### Reference that works

**`CrossHatch_Cylinder_*`** (from `scripts/archive/implicit_research/generate_cell_research_woodpiles.py`) uses:

1. Watertight **CAD mesh** (cylinder)
2. **`voxelize_mesh_and_edt`** → smooth **`cad_sdf`** (negative inside)
3. **`final_field = max(woodpile_field, cad_sdf)`**
4. **`skimage.measure.marching_cubes`** at level **0**

Caps come from the **EDT of the watertight boundary**, not separate analytic Z caps.

### Single-pass piecewise (volume-mesh safe)

For Gmsh **`single_surface`** volume meshing, prefer **one full-domain implicit field** with piecewise `L(z)` and `τ(z)` — no per-band boolean union (avoids seam self-intersections).

**API:** `graphite.implicit.piecewise_bands`

- `cumulative_phase_w_from_l_profile()` — Jacobian-integrated axial phase `W(z)`
- `splitp_piecewise_cylinder_single_pass()` / `splitp_piecewise_box_single_pass()`

**CLI:** `scripts/generate_piecewise_splitp_implicit.py`

Calibration: `calibrate_tau_at_fixed_period()` with **L = unit-cell period** (not `calibrate_tpms_point` pore diameter).

### Legacy per-band union

**Per-third short watertight solids** (cylinder or box) → same **`generate_uniform_woodpile`** / EDT Split-P pipeline → **`manifold3d` boolean union** of the three bands. Still used in `generate_three_cylinder_lattices_user_spec.py` for discrete-thirds cylinders; may self-intersect at band seams — not recommended for Aristo volume mesh.

---

## 2. Scripts and outputs

| Script | What it builds |
|--------|----------------|
| `scripts/generate_three_cylinder_lattices_user_spec.py` | (1) Linear Split-P cylinder 3×5 mm, (2) discrete-thirds Split-P cylinder, (3) discrete-thirds woodpile cylinder. `--only piecewise-splitp` for (2) only. |
| `scripts/generate_piecewise_splitp_implicit.py` | **Canonical** two-band piecewise Split-P (cylinder or box), **single-pass** EDT — preferred for volume meshing. |
| `scripts/generate_piecewise_woodpile.py` | **Canonical** piecewise **cross-hatch** woodpile (`--generator extrude` default); see [§8](PIECEWISE_PRISM_LATTICE_GENERATION.md#8-piecewise-cross-hatch-woodpile-jul-2026). |
| `scripts/generate_piecewise_woodpile_implicit.py` | Deprecated redirect → `generate_piecewise_woodpile.py` |
| `experiments/implicit_to_volume/generate_woodpile_cube_1mm.py` | 1 mm³ piecewise cross-hatch cube (200/400 µm pores) @ 0.015 mm resolution. |
| `scripts/test_piecewise_slab_union_woodpile.py` | `scripts/archive/tests_diagnostics/` — legacy slab union recipe |
| `scripts/generate_splitp_linear_box_3x1p5x5_sf25.py` | Linear Split-P in **3 × 1.5 × 5 mm** box. `--frame bottom-face-center` (default) or `midwidth-y0-z0`. |
| `scripts/generate_discrete_rect_prism_3x1p5x5_user_spec.py` | Discrete-thirds Split-P + woodpile in the **prism** (centered XY). `--woodpile-phase invert` for complement lattice. |

### Default output folders

| Folder | Contents |
|--------|----------|
| `outputs/models/user_spec_cylinders_3x5mm/` | 3 mm diameter × 5 mm cylinder variants (**deleted Jul 2026 Option 2**; rebuild with `scripts/generate_three_cylinder_lattices_user_spec.py`) |
| `outputs/models/user_spec_rect_prism_3x1p5x5mm/` | 3 × 1.5 × 5 mm prism variants |
| `outputs/models/piecewise_slab_union_test/` | Slab-union experiment meshes (**deleted Jul 2026 Option 2**; rebuild via archive diagnostic) |
| `outputs/models/` | Linear box STLs (`SplitP_Box3x1p5x5mm_*.stl`) |

### Prism frame (aligned with linear box)

**Bottom-face-center** (default for prism and `bottom-face-center` box):

- `X ∈ [-1.5, 1.5]`, `Y ∈ [-0.75, 0.75]`, `Z ∈ [0, 5]` mm  
- Phase: `U = X·ω`, `V = Y·ω` (same symmetry idea as a centered cylinder in XY)

**Mid-width on Y=0 edge** (`--frame midwidth-y0-z0` on linear box only):

- Box `X ∈ [0, 3]`, `Y ∈ [0, 1.5]`, `Z ∈ [0, 5]` mm  
- Phase origin `(1.5, 0)` mm: `U = (X − 1.5)·ω`, `V = Y·ω` — “half of a wider part” look on the `Y = 0` face

---

## 3. Implementation map (`graphite` + `scripts`)

### Shared helpers (`scripts/generate_three_cylinder_lattices_user_spec.py`)

| Symbol | Role |
|--------|------|
| `_boolean_union_slabs` | `manifold3d` union; `trimesh.util.concatenate` fallback |
| `_W_dense_discrete_thirds` | Cumulative axial phase `W(z)` for piecewise `L` on `[0, H]` |
| `_splitp_mesh_one_slab_edt` | One Z band: **short cylinder** + `max(\|F\|−τ, cad_sdf)` + MC |
| `_splitp_mesh_one_slab_edt_rect_prism` | Same for **short box** slab |
| `_splitp_mesh_discrete_thirds` | Three cylinder slabs + union |
| `_splitp_mesh_discrete_thirds_rect_prism` | Three box slabs + union |
| `_splitp_mesh_from_L_tau_box` | Full-volume linear Split-P in a box (analytic box SDF crop) |
| `_woodpile_piecewise_cylinder_mesh` | Three cylinder slabs → `generate_uniform_woodpile` → union |
| `_woodpile_piecewise_rect_prism_mesh` | Three box slabs → same |

### Production API change

**`graphite/implicit/uniform_woodpile.py`** — `generate_uniform_woodpile(..., invert_solids=False)`:

- If `invert_solids=True`, **`woodpile_field = -woodpile_field`** before `max(..., cad_sdf)`  
- Swaps solid/void **inside** the meshed CAD (complement of cross-hatch in-plane)

### Woodpile field (`graphite/math/woodpile.py`)

- `field ≤ 0` → solid (beam); `field > 0` → void  
- Cross-hatch: `true_woodpile=False` alternates `wave_X` / `wave_Y` by layer index  
- Piecewise: `evaluate_woodpile_piecewise_cylinder()` — per-band `pore`, `origin`, `swap_xy`, local `z_layer_origin`  
- Anchoring: `graphite/math/woodpile_anchor.py` — `compute_woodpile_xy_origin()`, `compute_band_orientation()`

### Piecewise woodpile API (`graphite/implicit/piecewise_woodpile.py`)

| Symbol | Role |
|--------|------|
| `woodpile_piecewise_cylinder_single_pass` | **Preferred** — one EDT field, one MC mesh |
| `woodpile_piecewise_cylinder_union` | Legacy per-slab union |
| `_band_parameters` | Per-band origins + `swap_xy` chain |

---

## 4. Calibration (unchanged recipes)

- **Linear Split-P:** `calibrate_tpms_gradient_profile` (hybrid), pore 0.2 → 0.5 → 0.8 mm at Z = 0, mid, 5 mm, SF **25%**
- **Discrete thirds Split-P:** `calibrate_tpms_point` per third (0.2 / 0.4 / 0.8 mm), SF **25%**
- **Woodpile:** no iso bisection in EDT path; level fixed at **0** on `max(woodpile, cad_sdf)`

Default voxel pitch: **0.01 mm** (10 µm) — large RAM / runtime for full prisms.

---

## 5. Key STL names (prism)

| File | Description |
|------|-------------|
| `SplitP_Box3x1p5x5mm_discreteThirdsPore200_400_800um_SF25.stl` | Discrete-thirds Split-P prism |
| `Woodpile_CrossHatch_Box3x1p5x5mm_discreteThirdsPore200_400_800um_SF50.stl` | Standard cross-hatch prism |
| `Woodpile_CrossHatch_Box3x1p5x5mm_discreteThirdsPore200_400_800um_SF50_invertedSolids.stl` | **Inverted** cross-hatch (mid-plane pore along Y) |
| `SplitP_Box3x1p5x5mm_linearPore200to800um_SF25.stl` | Linear gradient box (bottom-face-center) |
| `SplitP_Box3x1p5x5mm_linearPore200to800um_SF25_phaseOriginMidWidth_Y0Z0.stl` | Linear box, phase origin at (1.5, 0, 0) |

---

## 6. Commands (quick reference)

```bash
# Discrete-thirds Split-P cylinder only
python scripts/generate_three_cylinder_lattices_user_spec.py --only piecewise-splitp

# Prism: Split-P + woodpile
python scripts/generate_discrete_rect_prism_3x1p5x5_user_spec.py

# Prism: inverted woodpile only
python scripts/generate_discrete_rect_prism_3x1p5x5_user_spec.py --only woodpile --woodpile-phase invert

# Linear box (both frames)
python scripts/generate_splitp_linear_box_3x1p5x5_sf25.py
python scripts/generate_splitp_linear_box_3x1p5x5_sf25.py --frame midwidth-y0-z0
```

---

## 7. Pitfalls

1. **Do not** run aggressive `fill_holes` on each porous slab before union — can empty meshes; repair **after** union if needed.
2. **`trimesh.is_watertight`** is often `False` per slab; union + light repair is the meaningful check.
3. **Linear box** still uses **full-volume** analytic box SDF (not per-slab EDT) — only **discrete-thirds** prism/cylinder use slab EDT + union.
4. **`invert_solids`** complements the **entire** in-plane pattern, not only the Y centerline; the Y=0 mid-channel is the intended visible effect.

---

## 8. Piecewise cross-hatch woodpile (Jul 2026)

Graded **cross-hatch** woodpile (`true_woodpile=False`) in small cylinders (e.g. Ø2 mm) with **discrete Z bands** (custom pore size + thickness per band). Woodpile has ~**50% in-plane solid fraction** intrinsically (`pitch = 2 × pore`); no TPMS-style calibration.

### 8.1 Canonical pipeline (use this)

**Single-pass** full-cylinder EDT + piecewise woodpile field + one marching-cubes extract — same philosophy as piecewise Split-P.

| Piece | Role |
|-------|------|
| `graphite/implicit/piecewise_woodpile.py` | `woodpile_piecewise_cylinder_single_pass()` |
| `graphite/math/woodpile.py` | `evaluate_woodpile()`, `evaluate_woodpile_piecewise_cylinder()` |
| `graphite/math/woodpile_anchor.py` | `center_void` / `edge_solid` XY phase; `compute_band_orientation()` |
| `graphite/implicit/woodpile_input.py` | `WoodpileLatticeSpec`, `build_piecewise_woodpile_mesh`, CLI arg helpers |
| `graphite/explicit/woodpile_extrude.py` | Extrude backend (`generator: extrude`) |
| `graphite/geometry/masking.py` | `voxelize_cylinder_slab_and_edt()` — analytic cylinder mask (no trimesh subdivide) |
| `scripts/generate_piecewise_woodpile.py` | **CLI** (default `generator=extrude`) |
| `example_woodpile_piecewise_2mm.yaml` | Headless config for `scripts/generate_lattice.py` |

**Field composition (per band):**

\[
\text{final\_field} = \max(\text{woodpile\_piecewise}(X,Y,Z),\,\text{cad\_sdf})
\]

Each voxel uses the woodpile parameters of **its Z band only** (no boolean union).

### 8.2 Phase anchoring (small diameters)

Default implicit woodpile anchors strut centers at **X = 0, Y = 0**. On a Ø2 mm part with 800 µm pores, that leaves a **single central strut** and a weak lattice.

| `anchor_mode` | Origin | Use when |
|---------------|--------|----------|
| **`center_void`** (default) | `origin = pore` | Symmetric **±struts**; void at center; good for Ø ≈ 2–3 pitches |
| `edge_solid` | `origin = radius − pore/2` | Seat strut outer face on cylinder wall |
| `default` | `0` | Legacy; avoid on small cylinders |

### 8.3 Band orientation (90° toggle)

Between Z bands, toggle in-plane orientation so finer sections do not leave struts **aligned** with the coarser band above.

| Mechanism | What it does | Sufficient alone? |
|-----------|--------------|-------------------|
| `flip_layer_parity` | Swaps which Z parity uses X vs Y struts | **No** — XY grid stays aligned |
| **`swap_xy`** | Exchanges X↔Y before wave evaluation (true 90° rotation) | **Yes**, with single-pass |
| **Local Z layering** | `layer_idx = floor((Z − z_{\text{band,0}}) / pore)` per band | Restarts parity at each interface |

**Default:** `alternate_band_orientation=True` → `swap_xy` toggles each band (`not prev_swap_xy`). Band 0 off, band 1 on, band 2 off, …

### 8.4 Why not per-slab boolean union?

Legacy path: each band → short cylinder slab → `generate_uniform_woodpile` → **`manifold3d` union**.

**Problem:** Union **preserves** the previous band’s struts through the next band’s Z range. A thin middle band (e.g. 80 µm) looks identical to the coarse band above — **`swap_xy` on the new band cannot remove merged geometry**.

**Fix:** `woodpile_piecewise_cylinder_single_pass` (default since Jul 2026). Legacy union: `--union` on CLI or `combine_mode: union` in YAML.

### 8.5 Example spec (Ø2 mm × 2.7 mm)

| Z band | Thickness | Pore |
|--------|-----------|------|
| 0 → 2.23 mm | 2230 µm | 800 µm |
| 2.23 → 2.31 mm | 80 µm | 400 µm |
| 2.31 → 2.7 mm | 390 µm | 200 µm |

**Output:** `outputs/models/CrossHatch_Cylinder2x2.7mm_piecewise_P800_400_200um_SF50.stl`

### 8.6 Commands

```bash
# Default: extrude backend, center_void, global layer continuity
python scripts/generate_piecewise_woodpile.py

# Implicit MC reference
python scripts/generate_piecewise_woodpile.py --generator implicit \
  --diameter-mm 2.0 \
  --z-breaks-mm 0,2.23,2.31,2.7 \
  --pore-mm 0.8,0.4,0.2 \
  --resolution-mm 0.01

# Headless YAML
python scripts/generate_lattice.py NUL example_woodpile_piecewise_2mm.yaml \
  outputs/models/CrossHatch_Cylinder2mm

# Legacy slab union (orientation may not toggle visibly)
# Deprecated legacy union (implicit only):
python scripts/generate_piecewise_woodpile.py --generator implicit --union
```

### 8.7 YAML keys (`engine_type: Implicit (Woodpile)`)

```yaml
engine_type: Implicit (Woodpile)
grading_mode: Piecewise Z bands
diameter_mm: 2.0
resolution: 0.01
true_woodpile: false
anchor_mode: center_void          # center_void | edge_solid | default
alternate_band_orientation: true
combine_mode: single-pass           # single-pass | union
z_breaks_mm: [0.0, 2.23, 2.31, 2.7]
pore_mm: [0.8, 0.4, 0.2]
```

### 8.8 Pitfalls

1. **Do not** use slab union when bands need different in-plane orientation — use **single-pass**.
2. **`flip_layer_parity` alone** does not rotate the XY grid; use **`swap_xy`**.
3. **Piecewise box cubes (1 mm³):** toggling `swap_xy` per band is **not sufficient** when pore sizes differ — the Z = 0.5 mm interface also needs an **XY phase offset** so hatch lines do not stay aligned through the band boundary. **Split-P piecewise:** implemented via `band_phase_origin_x_mm` / `band_phase_origin_y_mm` in `piecewise_bands.py`; comparison uses bottom **X +L/4** — see [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md). **Woodpile:** band-interface fix still tracked in [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md).
4. **Do not** run aggressive `fill_holes` on a good manifold union mesh — can break watertightness.
5. **80 µm bands** are only ~8 voxels at 10 µm pitch — thin bands are fragile; orientation is still correct in the field, but mesh detail is limited.
6. **Cross-hatch vs true woodpile:** `true_woodpile=False` (default) is cross-hatch; `true_woodpile=True` adds shifted stacking layers (`--true-woodpile` on CLI).
