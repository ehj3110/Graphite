# `outputs/case_studies/cube_1mm/`

Canonical generated artifacts for the [1 mm³ cube case study](../../docs/CASE_STUDY_CUBE_1MM.md).

**Start here:** [`MANIFEST.json`](MANIFEST.json) — machine-readable index of current FEM, fluid, and figure paths.

**Comparison handoff:** [CUBE_1MM_THREE_LATTICE_COMPARISON.md](../../docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md)

---

## Folder layout

| Path | Contents |
|------|----------|
| **`figures/`** | **All comparison PNGs — look here first** |
| **`fea/`** | Current Aristo VTU + report JSON (three lattices) |
| Root `*.stl` | Current geometry (woodpile extrude + Split-P ×2) |
| **`fluid/`** | Vocal run diagnostics (`vocal_metrics/`) |
| **`archive/`** | Superseded sweeps, phase-test variants, old debug outputs |

Vocal simulation **caches** (`f.pkl`, velocity, metrics) live in `outputs/vocal/cache/` — paths listed in `MANIFEST.json`.

---

## Primary figures (August 2026)

| File | Description |
|------|-------------|
| [`figures/Cube1mm_Aristo_Vocal_6panel_comparison.png`](figures/Cube1mm_Aristo_Vocal_6panel_comparison.png) | 2×3: Aristo σ_vm + Vocal \|u\| (per-panel color scale) |
| [`figures/Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png`](figures/Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png) | Same; shared color scale per row |
| [`figures/Cube1mm_Vocal_WSS_3panel_comparison.png`](figures/Cube1mm_Vocal_WSS_3panel_comparison.png) | 1×3: WSS + streamlines (converged Vocal caches) |

**WSS display experiments** (upsample / STL stamp — not canonical): [CUBE_1MM_WSS_DISPLAY_EXPERIMENTS.md](../../docs/CUBE_1MM_WSS_DISPLAY_EXPERIMENTS.md)

Regenerate:

```powershell
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --shared-color-scale --plot-only
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py
```

---

## Secondary figures

| File | Description |
|------|-------------|
| `figures/SplitP_Cube1mm_piecewise_phaseOffset_test_4panel.png` | Phase-offset Aristo A/B test |
| `figures/Cube1mm_Vocal_UpDown_2x3_comparison.png` | Vocal up vs down, three lattices |
| `figures/Cube1mm_Vocal_UpDown_variance_*.png` | Up/down variance maps |
| `figures/Cube1mm_SplitP_*_Cascade_*.png` | Split-P 32→64→128 cascade |

---

## Current FEM (Aristo)

| Lattice | VTU |
|---------|-----|
| Woodpile extrude | `fea/Woodpile_..._extrude_h0p015_1N_E25p8MPa_1N_aristo_fea.vtu` |
| Split-P piecewise (+L/4) | `fea/SplitP_..._phaseTest_qL4_shift_x_h005_1N_E25p8MPa_1N_aristo_fea.vtu` |
| Split-P linear grad | `fea/SplitP_Cube1mm_linearGrad_centerPhase_h005_1N_E25p8MPa_1N_aristo_fea.vtu` |

---

## Current geometry (STL)

| Lattice | File |
|---------|------|
| Woodpile extrude | `Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned.stl` |
| Split-P piecewise (+L/4) | `SplitP_Cube1mm_piecewise_..._phaseTest_qL4_shift_x_cleaned.stl` |
| Split-P linear grad | `SplitP_Cube1mm_linearGrad_..._JacobianW_cleaned.stl` |

---

## Regenerate geometry

```powershell
python scripts/generate_cube_1mm_splitp.py --only case-study
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores
```
