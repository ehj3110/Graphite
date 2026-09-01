# 1 mm³ three-lattice comparison — Aristo + Vocal (handoff)

**Last updated:** August 2026  
**Code:** `graphite/case_studies/cube_1mm/`, `graphite/aristo/cross_section_viz.py`, `graphite/lbm/`  
**Outputs:** `outputs/case_studies/cube_1mm/`, `outputs/vocal/cache/`  
**Related:** [CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md), [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md), [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md), [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md), [EXPLICIT_WOODPILE_EXTRUSION.md](EXPLICIT_WOODPILE_EXTRUSION.md), [BALLS_BASEBALL.md](BALLS_BASEBALL.md) (oversize → Boolean trim precedent)

This document captures the **current credible 1 mm³ comparison** between three lattice families:

| Column | Lattice | Generator |
|--------|---------|-----------|
| 1 | Cross-hatch woodpile | **Extrude** (`generator: extrude`) |
| 2 | Split-P piecewise | Implicit TPMS, **bottom-band X phase +L/4** |
| 3 | Split-P linear grading | Implicit TPMS, Jacobian-warped |

Each column has **Aristo** (1 N compression, von Mises) and **Vocal** (Re=5 flow chamber, permeability / WSS) artifacts. Comparison figures are **post-processed from saved VTUs and Vocal caches** — no solver re-run is required to regenerate PNGs unless inputs change.

---

## Summary figures (canonical)

| Figure | Script | Description |
|--------|--------|-------------|
| **`Cube1mm_Aristo_Vocal_6panel_comparison.png`** | `scripts/plot_cube_1mm_aristo_vocal_comparison.py` | 2×3: Aristo σ_vm (top) + Vocal \|u\| (bottom); per-panel peak normalize |
| **`Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png`** | same + `--shared-color-scale` | Same layout; color scale shared across the three columns (per row) |
| **`Cube1mm_Vocal_WSS_3panel_comparison.png`** | `scripts/plot_cube_1mm_vocal_wss_comparison.py` | 1×3: Vocal WSS + streamlines (black void) |
| **`Cube1mm_SplitP_Downflow_Cascade_32_64_128_2panel.png`** | `scripts/run_splitp_cascade_32_64_128_downflow.py` | Split-P downflow cascade (32→64→128) |
| **`Cube1mm_SplitP_Upflow_Cascade_32_64_128_2panel.png`** | `scripts/run_splitp_cascade_32_64_128_upflow.py` | Split-P upflow cascade (seed: flip Uz from converged downflow) |
| **`Cube1mm_Vocal_UpDown_2x3_comparison.png`** | `scripts/plot_cube_1mm_vocal_updown_comparison.py` | 2×3: Vocal up (+Z) vs down (−Z), all three lattices |
| **`Cube1mm_Vocal_UpDown_variance_direct.png`** | `scripts/plot_cube_1mm_vocal_updown_variance.py` | 1×3: \((\|u\|_\uparrow-\|u\|_\downarrow)^2\) on XZ slice |
| **`Cube1mm_Vocal_UpDown_variance_z_mirror.png`** | `scripts/plot_cube_1mm_vocal_updown_variance.py --mode z_mirror` | 1×3: up vs Z-mirrored down variance |
| **`SplitP_Cube1mm_piecewise_phaseOffset_test_4panel.png`** | `scripts/plot_splitp_piecewise_phase_offset_test.py` | Phase-offset A/B test (Aristo only) |

All live under `outputs/case_studies/cube_1mm/`.

---

## Column 1 — Cross-hatch woodpile (extrude)

### Geometry

- **Stem:** `Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude`
- **STL:** `..._extrude_cleaned.stl` (~48 KB structured surface)
- **Pores:** matched to Split-P MIS equivalents (P139 µm bottom / P277 µm top, SF 50%)
- **Do not use** the implicit marching-cubes STL (`..._SF50_cleaned.stl`, ~4.6 MB) for Vocal or the comparison figures — voxel corners look blocky and solid fraction differs.

### Aristo

- **VTU:** `..._extrude_h0p015_1N_E25p8MPa_1N_aristo_fea.vtu` (h = 0.015 mm mesh)
- **Slice:** **YZ @ X = 0.300 mm** (woodpile struts run in Y/Z; X slice shows cross-hatch)
- **Fill mode:** `element-slice-soft` (default for woodpile extrude plots)

### Vocal

- **Cache:** `Woodpile_..._extrude_cleaned_n128_flow_chamber_re5_ma0.05_steps2000`
- **Slice:** XZ @ mid-Y (flow +Z, flow-chamber BCs)
- Run (if cache missing):

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_vocal.py `
  --geometry stl `
  --stl "outputs/case_studies/cube_1mm/Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned.stl" `
  --target-n 128 --boundary-style flow_chamber --re 5 --ma 0.05 `
  --steps 2000 --warm-start auto --cache-dir outputs/vocal/cache
```

⚠️ Woodpile **band-interface** geometry (90° hatch rotation + XY phase at Z = 0.5 mm) may still need a dedicated fix — see [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md). The extrude path is the correct surface representation for the current comparison; interface QC is tracked separately.

---

## Column 2 — Split-P piecewise (bottom X +L/4)

### Problem solved

At the **Z = 0.5 mm** band interface (L = 500 µm bottom / L = 1000 µm top), the baseline piecewise lattice showed a **large stress concentration and visible pore misalignment** where hatch lines stayed registered through the interface.

### Fix

Per-band lateral XY phase shift on the **bottom band only**:

- `graphite/implicit/piecewise_bands.py` — `band_phase_origin_x_mm` / `band_phase_origin_y_mm` per band; phase enters as `U = (X - phase_x) * ω`, `V = (Y - phase_y) * ω`.
- `graphite/case_studies/cube_1mm/generate_splitp.py` — `build_piecewise_splitp_cube_mesh()` forwards phase kwargs.

**Chosen variant:** **shift X only, +L_bottom/4 = +0.125 mm** (`phaseTest_qL4_shift_x`).

| Variant | Slice peak σ_vm @ Y=0.5 | Global nodal peak σ_vm (1 N) |
|---------|---------------------------|------------------------------|
| Baseline | 18.0 MPa | 125 MPa |
| **Shift X +L/4** | 14.5 MPa | **67 MPa** ← production choice |
| Shift Y +L/4 | 7.7 MPa | 96 MPa |
| Shift X+Y +L/4 | 24.5 MPa | 72 MPa |

### Key assets

| Artifact | Path |
|----------|------|
| Cleaned STL | `SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned.stl` |
| Aristo VTU | `SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_h005_1N_E25p8MPa_1N_aristo_fea.vtu` |
| Vocal cache (up) | `SplitP_..._phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001` |
| Vocal cache (down) | `SplitP_..._phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001` |

### Aristo slice

- **XZ @ Y = 0.500 mm**, `element-slice-soft`

### Regenerate phase-offset test (4-panel Aristo)

```powershell
python scripts/plot_splitp_piecewise_phase_offset_test.py
```

Stems/helpers: `splitp_piecewise_phase_shift_x_*` in `graphite/case_studies/cube_1mm/specs.py`.

---

## Column 3 — Split-P linear grading

### Geometry

- **STL:** `SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl`
- **Aristo VTU:** `SplitP_Cube1mm_linearGrad_centerPhase_h005_1N_E25p8MPa_1N_aristo_fea.vtu` (legacy stem alias in plot script)
- **FEA surface (if needed):** `..._JacobianW_cleaned_fea.stl` — PyMeshLab remesh of the Boolean STL (see § Boolean surface cleanup)
- **Slice:** XZ @ Y = 0.500 mm, `element-slice-soft`

### Vocal

- **Cache (up):** `SplitP_Cube1mm_linearGrad_..._n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001`
- **Cache (down):** `SplitP_Cube1mm_linearGrad_..._n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001`

---

## Boolean surface cleanup (TPMS cube STLs)

**Motivation (Aug 2026):** Hard-bounding a TPMS field to a small box and marching cubes leaves jagged / cusped exterior faces. On the piecewise Split-P cube that produced a spurious downward material spike near **(X ≈ 0.75, Z ≈ 0)** and an Aristo slice-peak hotspot (~14.5 MPa shared-scale) under platen BCs. Same class of exterior artifact as the baseball oversize → Boolean trim recipe ([BALLS_BASEBALL.md](BALLS_BASEBALL.md)).

**Prior path (still available):** evaluate TPMS on the design cube with `final_field = max(|F|−τ, box_sdf)` then MC — analytic box SDF in field space, but MC still approximates the outer faces.

**Production path for both Split-P case-study STLs:**

1. **Finer MC** — resolution **0.010 mm** (was 0.015 mm). Constants: `IMPLICIT_RESOLUTION_BOOLEAN_TRIM_MM`, default case-study regen.
2. **Oversize domain** — march on a box expanded by **`BOOLEAN_TRIM_MARGIN_MM = 0.04`** mm on each side (`origin = −margin`, `size = 1 + 2·margin`). Outer closure still uses the *generation* box SDF so MC stays bounded; TPMS continues through the design faces into the margin.
3. **Boolean ∩ design cube** — `trim_mesh_to_design_cube()` → `boolean_intersect_with_cad()` (`manifold3d`) against an exact 1 mm axis-aligned cube at the origin. Replaces jagged MC exterior with **true planar CAD faces** (`bounds = [0,1]³`).
4. **Light repair + island cleanup** — `_repair_mesh_after_implicit_crop`, then `remove_floating_islands`.
5. Write both `{stem}.stl` and `{stem}_cleaned.stl` (same mesh for this pipeline).

**API / CLI** (`graphite/case_studies/cube_1mm/generate_splitp.py`):

| Symbol | Role |
|--------|------|
| `trim_mesh_to_design_cube` | Manifold Boolean ∩ design cube |
| `build_piecewise_splitp_cube_mesh(..., boolean_trim=True)` | Piecewise (+ optional band phase) with oversize + trim |
| `build_linear_graded_splitp_cube_mesh(..., boolean_trim=True)` | Linear graded with oversize + trim |
| `regenerate_case_study_splitp_stls()` | Piecewise `phaseTest_qL4_shift_x` + linear Jacobian |

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/generate_cube_1mm_splitp.py --only case-study
# or: ... --boolean-trim --resolution-mm 0.01 --only graded
```

Report: `outputs/case_studies/cube_1mm/SplitP_Cube1mm_boolean_trim_regen_report.json`.

**Observed results after regen:** both cleaned STLs watertight, volume ≈ 0.33 mm³ (SF ~33%), exact `[0,1]³` bounds. Piecewise bottom-face cusp removed for inspection; shared-scale Aristo slice peak fell **~14.5 → ~7.2 MPa** after re-FEA.

**Aristo / gmsh note:** Piecewise Boolean `_cleaned.stl` volume-meshed with `single_surface` directly. Linear Boolean skin triggered gmsh **overlapping facet** failures on the large coplanar CAD faces — fix was a **PyMeshLab isotropic remesh** (`targetlen = 0.015 mm`) → `..._cleaned_fea.stl` (still watertight, volume ≈ 0.332 mm³), then FEA. Keep `_cleaned.stl` as the geometry-of-record for display / Vocal; use `_cleaned_fea.stl` only when gmsh needs a remeshed skin.

**Aristo regen settings:** h = 0.005 mm, 1 N, E = 25.8 MPa, `single_surface`, Pardiso, `flat_top_vertex_plane` BCs.

```powershell
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --shared-color-scale --plot-only
```

---

## Vocal cascade warm-start & up/down (Split-P)

Full reference: **[VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md)**.

**Convergence:** mean-\(U_z\) relative change (`check_interval=250`); stop when `rel_change < 1e-3`, `steps_run >= min_steps` (default **1000**), or stage cap.

**Downflow** (`flow_chamber_reverse`) and **upflow** (`flow_chamber`) both use 32→64→128 cascades. Upflow seeds n=32 from converged downflow n=128 with **`warm_start_flip_uz`** and matched cache pressure.

**Up/down figures:**

```powershell
python scripts/plot_cube_1mm_vocal_updown_comparison.py
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode direct
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode z_mirror
```

After fixed warm-start (July 2026), same-grid up/down speed RMS difference is **0.17–0.30 mm/s** for Split-P (woodpile **0.05 mm/s**). Large Z-mirror variance is expected — lattices are not Z-symmetric.

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_splitp_cascade_32_64_128_downflow.py
.\.venv_torch\Scripts\python.exe scripts/run_splitp_cascade_32_64_128_upflow.py --min-steps 1000
```

---

## Aristo compression BCs (cube case study)

All 1 mm³ cube Aristo runs use **`bc_load_mode="flat_top_vertex_plane"`** (CLI: `--bc-load-mode flat_top_vertex_plane`). This is **not** a vertical slice and **not** a thick Z-band.

**Selection rule** (`detect_boundary_masks_flat_top_vertex_plane` in `graphite/aristo/boundary_detection.py`):

1. Among surface nodes, find the dominant **floor** Z (lowest Z with ≥ `min_vertices_at_plane`, default 100) and dominant **cap** Z (highest Z with the same threshold).
2. **Fixed (support):** surface triangles whose **all three vertices** lie on the floor Z (exact coordinate match) **and** outward normal satisfies `n_z < -0.99` (downward-facing).
3. **Load (compression):** surface triangles whose **all three vertices** lie on the cap Z **and** `n_z > 0.99` (upward-facing).
4. Total force (here **1 N**) is distributed over the selected load faces; fixed nodes get zero displacement.

So BCs land only on **flat contact patches flush with the cube top/bottom faces** (platen-style). Side walls, pore mouths, and near-horizontal internal surfaces are excluded. Boolean-trimmed cube faces change that contact footprint versus jagged marching-cubes skins — peak stress can move even when interior lattice topology looks similar.

Default force / modulus for this case study: **1 N**, **E = 25.8 MPa**, mesh mode **`single_surface`**, solver **Pardiso**.

---

## Aristo cross-section visualization

All comparison Aristo panels use **`element-slice-soft`**:

- Tet plane-cut footprints from `von_mises_element_MPa`
- Overlap averaging (no vertical seam artifacts)
- 1-pixel gap fill before blur
- Masked Gaussian blur with **radius in mm** (default **0.015 mm**)
- Nearest `imshow` interpolation; void = white

Full mode reference: [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md).

### Color scales (6-panel figure)

| Output | Flag | Scaling |
|--------|------|---------|
| `Cube1mm_Aristo_Vocal_6panel_comparison.png` | default / `--plot-only` | **Per-panel:** each subplot ÷ its own slice peak → [0, 1] |
| `Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png` | `--shared-color-scale` | **Per row:** all three columns ÷ that row’s max slice peak |

Panel titles always print each case’s absolute slice peak (MPa or mm/s).

```powershell
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --shared-color-scale --plot-only
```

---

## Vocal velocity (6-panel bottom row)

- **Colormap:** turbo; default **per-panel normalized** \|u\| (0–1 by slice peak)
- **Shared scale:** `--shared-color-scale` → all three columns use `max(slice peaks)` so colors are comparable; titles still report each panel’s own peak
- **Overlay:** dark gray solid mask, white quiver (comparison script) or streamlines (WSS script)
- **Crop:** unit cube interior — strips flow-chamber padding (`pad_xy=1`, `pad_z=3`)

Woodpile column **must** point at the **`_extrude_`** Vocal cache slug (`woodpile_vocal_cache_slug()` in `specs.py`).

---

## Vocal WSS (3-panel)

**No LBM re-simulation required** if `f.pkl` exists in the Vocal cache.

| Cache file | Role |
|------------|------|
| `velocity.npy` | Streamlines |
| `solid_mask.npy` | Geometry mask |
| `f.pkl` | Full LBM distribution → `LettuceSolver.compute_wss_field()` |
| `metrics.json` | Steps, k, WSS summary stats |

### Pipeline

```
Vocal cache dir
  → load_solver_from_cache_dir()   # graphite/lbm/vocal_cache.py
  → compute_wss_field()            # stress tensor from f, not velocity
  → XZ mid-Y slice + streamplot
  → black background; only WSS interface bands colored (turbo)
```

```powershell
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py
```

Options: `--shared-color-scale`, `--streamline-density 1.5`, `--device cpu`

Plot helpers: `plot_wss_comparison()`, `_draw_xz_wss_streamline_panel()` in `graphite/lbm/vocal_run.py`.

---

## `specs.py` helpers (cube 1 mm)

| Helper | Purpose |
|--------|---------|
| `resolve_woodpile_stem(..., generator="extrude")` | Extrude woodpile stem |
| `woodpile_vocal_cache_slug(steps=2000)` | Extrude Vocal cache directory name |
| `splitp_piecewise_phase_shift_x_stem()` | Phase-fixed piecewise stem |
| `splitp_piecewise_phase_shift_x_fea_stem()` | Aristo VTU stem |
| `splitp_piecewise_phase_shift_x_vocal_cache_slug(steps=1000)` | Piecewise Vocal cache name |
| `SPLITP_PIECEWISE_BOTTOM_BAND_PHASE_SHIFT_X_MM` | 0.125 mm (= L_bottom/4) |

---

## Environment notes

| Task | Environment |
|------|-------------|
| Aristo plots, Split-P geometry | System `python` + `PYTHONPATH=.` |
| Vocal run / WSS plots | **`.venv_torch`** (`scripts/run_vocal.py`) |
| `run_cube_1mm_woodpile_vocal.py` wrapper | May fail in `.venv_torch` if `skimage` missing — call `run_vocal.py` directly with STL path |

**Windows PowerShell:** use `;` not `&&` between commands.

---

## Regenerate everything (quick reference)

```powershell
# 6-panel Aristo + Vocal (plot-only; needs VTUs + caches on disk)
$env:PYTHONPATH = (Get-Location).Path
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only

# 3-panel Vocal WSS + streamlines (from f.pkl; ~20 s on GPU)
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py

# Up/down comparison + variance (from cache; no LBM)
python scripts/plot_cube_1mm_vocal_updown_comparison.py
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode direct

# Piecewise phase-offset 4-panel Aristo test
python scripts/plot_splitp_piecewise_phase_offset_test.py
```

---

## Code touchpoints (this workstream)

| Area | Files |
|------|-------|
| Piecewise band phase | `graphite/implicit/piecewise_bands.py`, `generate_splitp.py` |
| Aristo element-slice-soft | `graphite/aristo/cross_section_viz.py` |
| 6-panel comparison | `scripts/plot_cube_1mm_aristo_vocal_comparison.py` |
| WSS from cache | `graphite/lbm/vocal_cache.py` (`load_solver_from_cache_dir`), `vocal_run.py` |
| WSS 3-panel | `scripts/plot_cube_1mm_vocal_wss_comparison.py` |
| Up/down comparison | `scripts/plot_cube_1mm_vocal_updown_comparison.py` |
| Up/down variance | `scripts/plot_cube_1mm_vocal_updown_variance.py` |
| Phase test | `scripts/plot_splitp_piecewise_phase_offset_test.py` |
| Stems / paths | `graphite/case_studies/cube_1mm/specs.py` |
| Cascade + warm-start | [VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md) |

---

## Not yet done (out of scope for current figures)

- Promote `phaseTest_qL4_shift_x` to the default `generate_cube_1mm_splitp.py` piecewise output (still opt-in via phase-test stems).
- Woodpile band-interface 90° + XY phase fix ([WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md)).
- Persist `wss.npy` in Vocal cache on first compute (today WSS is recomputed from `f.pkl` at plot time).
- Woodpile up/down still on 2000-step caches (not re-run through cascade).
