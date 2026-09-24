# Implicit grading & surface textures

**Package:** [`graphite/implicit/`](../graphite/implicit/README.md)  
**Agents:** [AGENTS.md](../AGENTS.md) → package card → this file (not a full workspace scan).  
**Math deep-dive:** [LATTICE_MATH_ARCHITECTURE.md](LATTICE_MATH_ARCHITECTURE.md) · engine overview: [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md)

---

## Product note: no Aristo / FEA-driven grading

**Dropped (Sep 2026):** using Aristo von Mises stress to drive a new graded lattice (`stress_to_vf_gradient`, `stress_to_strut_radius_map`). Those paths crash this workstation at product scale.

**Keep Aristo for:** analyzing lattices you already generated (Mirae, cube study, reports).  
**Grade with:** the native modes below.

Details: [ARISTO.md](ARISTO.md#product-decision-sep-2026-no-fea-driven-lattice-grading).

---

## 1. Grading catalog (what to use when)

| Mode | Module / entry | What varies | Best for | Status |
|------|----------------|-------------|----------|--------|
| **Uniform conformal** | `conformal.generate_conformal_lattice` | Constant \(L\), SF | Baseline fill of a CAD shell | Production |
| **Solid-fraction grade** | `graded.generate_graded_lattice` | SF only (constant \(k\)) | Thicker→thinner walls along axis / radial / modifier | Production |
| **Chirped frequency** | `chirped.generate_chirped_lattice` | Wavenumber \(k\) (pore size) | Smooth pore-size change along axis / radial / modifier STL | Production |
| **Piecewise Split-P bands** | `piecewise_bands.splitp_piecewise_*_single_pass` | Hard bands of \(L\) / \(\tau\) | Discrete zones (e.g. cube thirds); Gmsh-friendly single-pass EDT | Production |
| **Piecewise woodpile** | `piecewise_woodpile` / `build_piecewise_woodpile_mesh` | Band pitch / orientation | Cross-hatch woodpile; prefer `generator=extrude` | Production |
| **Field-driven / lofted** | `field_driven.generate_field_driven_lattice` | Integrated-phase \(L(s)\), SF along spine | Multi-knot continuous grading along X/Y/Z or radius | Production UI path; lofted hex analogue is experimental — [LOFTED_GRADING.md](LOFTED_GRADING.md) |
| **Boundary dual-EDT** | `boundary_graded.generate_boundary_graded_lattice` | \(L\), SF from distance to start/end surfaces | Non-parallel control surfaces | Production code; UI may be incomplete — [history/CONTROL_SURFACES_PLAN.md](history/CONTROL_SURFACES_PLAN.md) |
| **Osteochondral Z loft** | `osteochondral` | \(L(z)\), SF\((z)\) piecewise linear | Implant-style height stacks | Production module |
| **Multi-zonal** | `multi_zonal.generate_multi_zonal_lattice` | Region masks / zones | Multi-region fills | Lab / advanced |
| **Calibration** | `calibration.calibrate_tpms_*` | Chooses \(L\), \(\tau\) for target pore / wall / SF | Before shipping a graded profile | Production — [TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md) |

### Drivers shared by SF grade & chirp

Both `graded.py` and `chirped.py` build a smoothstep weight \(W\in[0,1]\) from:

- **Axis** — `X` / `Y` / `Z` over the part AABB  
- **Radial** — \(\sqrt{X^2+Y^2}\)  
- **Modifier** — distance to a second STL (transition width \(T\))

Then:

- **Graded SF:** \(\mathrm{SF} = \phi_{\min} + W(\phi_{\max}-\phi_{\min})\), constant \(k = 2\pi/L\)  
- **Chirped \(k\):** \(K = k_{\mathrm{base}}(1-W) + k_{\mathrm{mod}} W\), constant SF  

Equations: [IMPLICIT_ENGINE.md §3](IMPLICIT_ENGINE.md#3-chirped--graded-fields).

### Piecewise vs continuous

| Prefer piecewise when… | Prefer continuous (chirp / field-driven) when… |
|------------------------|--------------------------------------------------|
| Hard material / pore targets per band | Smooth transition without band seams |
| Downstream Gmsh / Aristo volume mesh needs clean bands | Preview / implant loft with many knots |
| Cube 1 mm case study (Split-P / woodpile) | Streamlit “Field Controls” loft |

Cube recipes: [CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md), [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md).

### Field-driven (Streamlit Step 4)

`generate_field_driven_lattice` / `generate_field_preview_image`:

- Coordinate: Cartesian axis or cylindrical / spherical radius  
- Control points: \((s, L)\) and optional SF knots  
- **Integrated phase** along the spine (avoids naive \(k(x)\,x\) shear) — see [LATTICE_MATH_ARCHITECTURE.md](LATTICE_MATH_ARCHITECTURE.md)  
- `grade_unit_cell=True` + `unit_cell_control_points` for period grading  

Current UI behavior: [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md). Dead UI branches (Variable Porosity, etc.) are noted in [STREAMLIT_REVAMP.md](STREAMLIT_REVAMP.md) — do not treat those as live APIs.

### Calibration before grading

For target pore size or wall thickness at a solid fraction:

1. `calibrate_tpms_point` / `calibrate_tpms_gradient_profile`  
2. Seed / LUT helpers in `tpms_parameter_lut.py`  
3. Guard MIS near boundaries ([TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md))

Then plug the resulting \(L\), \(\tau\) into piecewise bands or field-driven knots.

---

## 2. Surface textures (post-isosurface displacement)

**Not** a grading mode: run **after** marching cubes / Flying Edges on the scaffold mesh.

| Piece | Path |
|-------|------|
| Math fields | `graphite/math/textures.py` |
| Mesh apply | `graphite/implicit/surface_textures.py` |
| Config | `SurfaceTextureConfig` |
| API | `apply_surface_texture(mesh, config)` |

### Texture types

| `texture_type` | Field | Typical use |
|----------------|-------|-------------|
| `microgrooves` | `microgroove_field` (+ optional triplanar) | Contact-guidance striations |
| `bumps` / `nodules` | `bump_field` | Discrete nodules |
| `knurling` / `diamond` | `knurl_field` | Diamond knurl pattern |
| `spinodal` | `spinodal_spectral_field` | Spectral / Cahn–Hilliard-like relief |
| `none` | — | No-op |

### Key `SurfaceTextureConfig` knobs

| Param | Default | Meaning |
|-------|---------|---------|
| `amplitude_mm` | `0.025` | Displacement height (25 µm) |
| `wavelength_mm` | `0.050` | Pitch (50 µm) |
| `direction` | `(0,0,1)` | Groove axis (world) |
| `profile` | `sine` | `sine` / `triangle` / `square` (microgrooves) |
| `displacement_mode` | `centered` | `centered` / `emboss` / `engrave` |
| `use_triplanar` | `False` | Normal-weighted triplanar vs world-space |
| `project_transverse_normal` | `True` | For axial microgrooves: displace in \(\mathbf{n}_\perp\) to reduce wrinkles |
| `target_edge_length_mm` | `wavelength/2` | Subdivision target before displace |
| `max_faces` | `2e6` | Safety cap (textures explode face count) |

Pipeline: **subdivide** → evaluate field → displace along normals → optional repair.

```python
from graphite.implicit import SurfaceTextureConfig, apply_surface_texture

cfg = SurfaceTextureConfig(
    texture_type="microgrooves",
    amplitude_mm=0.025,
    wavelength_mm=0.050,
    direction=(0.0, 0.0, 1.0),
    displacement_mode="centered",
)
textured = apply_surface_texture(scaffold_mesh, cfg)
```

**Caution:** fine wavelengths + large parts → millions of faces / RAM. Prefer texturing **after** confirming the base lattice at coarser resolution; keep `max_faces` in mind for print-scale parts.

More geometry notes: [IMPLICIT_ENGINE.md §10](IMPLICIT_ENGINE.md#10-surface-micro-structures-engine-microgrooves--micropillars).

---

## 3. Micropillars / microfiber forest

**Explicit CSG** cylinders boolean-unioned onto the scaffold (Manifold), not voxel displacement.

| Piece | Path |
|-------|------|
| API | `generate_micropillars`, `sample_pillar_anchors`, `segment_cad_boundary` |
| Config | `MicropillarConfig` |

### Defaults (medical-scaffold scale)

| Param | Default |
|-------|---------|
| `diameter_mm` | `0.050` (50 µm) |
| `height_mm` | `0.200` (200 µm) |
| `spacing_mm` | `0.200` |
| `distribution` | `poisson_disk` |
| `location` | `all` |
| `max_pillars` | `50_000` |

### Placement & printability

- **`location`:** `all` / `internal_only` / `outer_only`  
- **`selected_faces`:** e.g. `("+z", "-z")`, `("sides",)`, synonyms `top`/`bottom`/…  
- **`boundary_type`:** `box` / `cylinder` / `sphere` / `auto`  
- **`filter_printable`:** keep pillars within overhang limits vs build plate  
- **`orientation`:** `local_normal` / `z_aligned` / `cad_normal`  
- **`clamp_boundary_normals`:** kill 45° tilt on sharp CAD cuts  
- **`segment_cad_boundary`:** organic CAD → top / bottom / sides for selective hair  

```python
from graphite.implicit import MicropillarConfig, generate_micropillars

cfg = MicropillarConfig(
    diameter_mm=0.05,
    height_mm=0.2,
    spacing_mm=0.2,
    location="outer_only",
    filter_printable=True,
)
haired = generate_micropillars(scaffold_mesh, cfg)
```

**Order of operations (recommended):**

1. Generate / grade implicit lattice → isosurface  
2. Optional exact CAD trim (overscale + Manifold intersect) if using that pipeline  
3. `apply_surface_texture` **or** `generate_micropillars` (or texture then pillars — expect heavy meshes)  
4. Export via `graphite.io.export_mesh`  
5. Aristo / Vocal only for **analysis**, not to invent the next grade  

---

## 4. Quick “read next”

| Goal | Doc / code |
|------|------------|
| Package map | [graphite/implicit/README.md](../graphite/implicit/README.md) |
| Equations & meshing | [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md) |
| Calibration practice | [TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md) |
| Piecewise Split-P / woodpile | [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md) |
| Lofted experimental hex | [LOFTED_GRADING.md](LOFTED_GRADING.md) |
| Why not FEA grade | [ARISTO.md](ARISTO.md) |
| Tier-1 CLIs | [scripts/README.md](../scripts/README.md) |
