# 1 mm³ cube case study — Split-P vs woodpile

**Code:** `graphite/case_studies/cube_1mm/`  
**Outputs:** `outputs/case_studies/cube_1mm/`  
**Related:** [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md) (**current comparison figures**), [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md), [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md), [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md)

End-to-end workflows for the 1 mm cube form factor: implicit / extrude geometry → Aristo FEA → Vocal LBM.

---

## Three-lattice comparison (July 2026)

Canonical side-by-side: **woodpile extrude | Split-P piecewise (X +L/4) | Split-P linear grad**.

| Figure | Command |
|--------|---------|
| 6-panel Aristo + Vocal | `python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only` |
| 3-panel Vocal WSS | `.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py` |
| Piecewise phase-offset test | `python scripts/plot_splitp_piecewise_phase_offset_test.py` |

Full asset list, slice planes, cache slugs, and design rationale: **[CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)**.

**In progress (Aug 2026):** Vocal n=256 refinement — see **[CUBE_1MM_VOCAL_N256_HANDOFF.md](CUBE_1MM_VOCAL_N256_HANDOFF.md)**.

---

## Layout

```
graphite/case_studies/cube_1mm/
  specs.py              # stems, paths, default physics, vocal cache slugs
  generate_splitp.py    # piecewise + linear graded Split-P cubes
  generate_woodpile.py  # cross-hatch woodpile cube
  run_woodpile_aristo.py
  run_woodpile_vocal.py
  continue_vocal.py     # +N steps from Vocal cache

scripts/  (Tier 1 — see scripts/README.md)
  generate_cube_1mm_splitp.py
  generate_cube_1mm_woodpile.py
  run_cube_1mm_woodpile_aristo.py
  run_cube_1mm_woodpile_vocal.py
  continue_cube_1mm_vocal.py
  plot_vocal_flow_comparison.py
  plot_cube_1mm_aristo_vocal_comparison.py
  plot_cube_1mm_vocal_wss_comparison.py
  plot_splitp_piecewise_phase_offset_test.py
```

---

## Split-P cube

```powershell
# Geometry (piecewise + linear graded @ SF=33%)
python scripts/generate_cube_1mm_splitp.py

# Case-study TPMS regen: 0.010 mm MC, oversize domain, Boolean ∩ 1 mm cube
$env:PYTHONPATH = (Get-Location).Path
python scripts/generate_cube_1mm_splitp.py --only case-study

# Piecewise with bottom-band X phase +L/4 (comparison variant)
# See generate_splitp.build_piecewise_splitp_cube_mesh() + plot_splitp_piecewise_phase_offset_test.py

# Vocal comparison (legacy piecewise vs graded only)
.\.venv_torch\Scripts\python.exe scripts/plot_vocal_flow_comparison.py --cases both --steps 1000

# Continue from cache (+500 steps diagnostic)
.\.venv_torch\Scripts\python.exe scripts/continue_cube_1mm_vocal.py --cases both
```

**Boolean surface cleanup** (oversize MC → manifold Boolean ∩ design cube → flat CAD faces): see **[CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)** § Boolean surface cleanup. Same idea as baseball oversize+trim ([BALLS_BASEBALL.md](BALLS_BASEBALL.md)).

Aristo on Split-P cubes: use `scripts/run_aristo_fea.py` + `scripts/aristo_clean_and_mesh_stl.py` with STLs from `outputs/case_studies/cube_1mm/` (h=0.005 mm). Linear graded may need `..._cleaned_fea.stl` (PyMeshLab remesh) if gmsh rejects the raw Boolean skin.

**Piecewise band interface:** per-band `band_phase_origin_x_mm` / `band_phase_origin_y_mm` in `graphite/implicit/piecewise_bands.py`. Production comparison uses **bottom X +0.125 mm** (`phaseTest_qL4_shift_x`).

---

## Woodpile cube

Use the **extrude** generator for Aristo/Vocal comparison (not implicit MC):

```powershell
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores   # extrude default
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores --generator implicit  # MC reference only
python scripts/run_cube_1mm_woodpile_aristo.py --match-splitp-pores
python scripts/run_cube_1mm_woodpile_aristo.py --match-splitp-pores --h 0.015 --plot-only --fill-mode element-slice-soft
```

Vocal on extrude STL (prefer direct `run_vocal.py`):

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_vocal.py `
  --geometry stl `
  --stl "outputs/case_studies/cube_1mm/Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned.stl" `
  --target-n 128 --boundary-style flow_chamber --re 5 --ma 0.05 --steps 2000 `
  --cache-dir outputs/vocal/cache
```

⚠️ Woodpile **band-interface** hatch alignment at Z = 0.5 mm may still need a dedicated fix — see [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md). Extrude surface + `element-slice-soft` Aristo plots are the current best practice.

Cross-section stress plots for extrude woodpile use **`element-slice-soft`** by default. See [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md).

---

## Migration from `experiments/implicit_to_volume/`

| Old | New |
|-----|-----|
| `experiments/.../generate_splitp_cube_1mm_variants.py` | `scripts/generate_cube_1mm_splitp.py` |
| `experiments/.../generate_woodpile_cube_1mm.py` | `scripts/generate_cube_1mm_woodpile.py` |
| `experiments/.../run_cube_woodpile_aristo.py` | `scripts/run_cube_1mm_woodpile_aristo.py` |
| `experiments/.../run_cube_woodpile_vocal.py` | `scripts/run_cube_1mm_woodpile_vocal.py` |
| `experiments/.../continue_vocal_500.py` | `scripts/continue_cube_1mm_vocal.py` |
| `experiments/.../output/` | `outputs/case_studies/cube_1mm/` |

Legacy experiment scripts remain as thin re-exports.
