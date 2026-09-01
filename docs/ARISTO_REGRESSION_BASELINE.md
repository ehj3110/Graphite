# Aristo regression baseline — Mirae V4 fixed slab

Pinned reference values for detecting mesh, BC, or solver regressions after refactors.  
**Geometry:** `outputs/models/user_spec_rect_prism_3x1p5x5mm/Mirae_LatticeSlab_V4_fixed.stl`  
**Recorded:** March 2026 (post P1/Delaunay refactor + flat-top BC load sweep)

Related: [ARISTO.md](ARISTO.md), [ARISTO_MESHING.md](ARISTO_MESHING.md)

---

## How to re-run

```bash
# Mesh only (quick smoke test, ~20 s)
python scripts/gmsh_lattice_mesh.py \
  outputs/models/user_spec_rect_prism_3x1p5x5mm/Mirae_LatticeSlab_V4_fixed.stl

# Full FEA sweep 1 / 5 / 10 N (~90 s)
python scripts/run_mirae_lattice_slab_v4_aristo.py
```

**Environment:** No `ARISTO_GMSH_ALGORITHM3D` override required (Delaunay is hardcoded default).

---

## Mesh-only baseline

Source: `Mirae_LatticeSlab_V4_fixed.adaptive_mesh_report.json`  
Command: default `gmsh_lattice_mesh.py` (adaptive P1, classify-only)

| Metric | Expected | Tolerance (guidance) |
|--------|----------|---------------------|
| `mesh_mode` | `tpms_classify_adaptive_linear` | exact |
| `algorithm_3d` | `1` (Delaunay) | exact |
| `optimize_netgen` | `0` | exact |
| `mesh_order` | `1` | exact |
| `illegal_linear_tets` | `0` | must be 0 |
| `inverted_linear_tets` | `0` | must be 0 |
| `n_nodes` | **24,604** | ±5% |
| `n_elements` | **73,813** | ±5% |
| `poor_fraction` | **0.0136** (1.36%) | ±0.005 absolute |
| `max_edge_mm` | **0.992** | ±0.05 mm |
| `has_giant_elements` | `false` | must stay false |

Thorough FEA remesh may produce slightly **more** tets (~74,653) due to `h_scale` retries — both counts are valid.

---

## FEA baseline (flat-top BC, thorough)

Source: `Mirae_LatticeSlab_V4_fixed_{1N,5N,10N}_aristo_report.json`  
Script: `scripts/run_mirae_lattice_slab_v4_aristo.py`

### Shared config (all load cases)

| Parameter | Value |
|-----------|-------|
| Material | E = 2800 MPa, ν = 0.38 |
| `fea_mesh_resolution` | 0.15 mm |
| `fea_gmsh_mesh_mode` | `tpms` |
| Adaptive cl | 0.08 – 0.8 mm, elem/2π = 15 |
| `bc_load_mode` | `flat_top` |
| `bc_top_z_tolerance_mm` | 1e-4 |
| `bc_top_normal_z_min` | 0.99 |
| Fixed support | bottom 1% Z-band |
| `fea_quality_mode` | `thorough` (3 remesh attempts) |

### Boundary patch (STL and FEA should match)

| Metric | Expected |
|--------|----------|
| `n_load_faces_stl` / `n_load_faces_fea` | **165** |
| `n_fixed_faces_stl` | **1,170** |
| `fea_load_area_mm2` | **0.8115** mm² |
| `z_max_mm` | **3.878932** |

### Per load case

| Case | Force (N) | Max nodal VM (MPa) | Max displacement (mm) | Notes |
|------|-----------|-------------------|----------------------|-------|
| **1N** | 1.00 | **29.124** | **5.31×10⁻⁴** | Primary regression anchor |
| **5N** | 5.00 | **145.622** | **2.65×10⁻³** | Should be ~5× 1N stress & disp |
| **10N** | 10.00 | **291.244** | **5.31×10⁻³** | Should be ~10× 1N stress & disp |

**Linear scaling check:**  
`VM(5N) / VM(1N) ≈ 5.00`, `VM(10N) / VM(1N) ≈ 10.00` (verified March 2026).

### Mesh quality inside FEA (1N report, thorough)

| Metric | Expected |
|--------|----------|
| `n_fea_nodes` | **24,734** |
| `n_fea_tets` | **74,653** |
| `poor_fraction` | **0.0128** (1.28%) |
| `remesh_attempts` | **3** |
| `max_von_mises_element_raw_MPa` (1N) | ~133.5 (hotspot on poor tets; nodal field preferred) |

---

## Pass / fail criteria (CI or manual review)

**Hard fail (stop and investigate):**

- Gmsh crash or zero volume elements extracted
- `illegal_linear_tets > 0` or `inverted_linear_tets > 0` on mesh-only run
- `algorithm_3d ≠ 1` without explicit env override in test
- `optimize_netgen ≠ 0` on lattice TPMS path
- Zero load or fixed faces detected
- Non-finite displacement or solve failure
- 1N max nodal VM outside **25–35 MPa** (large regression)

**Soft warn (review, may be acceptable):**

- Tet count outside ±10% of baseline (Gmsh version drift)
- `poor_fraction` between 1% and 2.5%
- Thorough remesh stops above `thorough_max_poor_fraction` (0.5%) — known on thin-wall lattices

---

## Artifact paths to diff

```
outputs/models/user_spec_rect_prism_3x1p5x5mm/
  Mirae_LatticeSlab_V4_fixed.stl
  Mirae_LatticeSlab_V4_fixed.adaptive_mesh_report.json
  Mirae_LatticeSlab_V4_fixed_1N_aristo_report.json   ← primary FEA diff
  Mirae_LatticeSlab_V4_fixed_1N_aristo_fea.vtu       ← ParaView spot-check
```

---

## When to update this document

Update pinned numbers when **intentionally** changing:

- Default mesh options (`gmsh_lattice_mesh.py`)
- BC mode or flat-top tolerances
- `fea_mesh_resolution` or adaptive cl bounds in V4 script
- Material properties in the V4 orchestration script

Do **not** update for one-off experiments (`--strict-uniform`, `--order 2`, env overrides).
