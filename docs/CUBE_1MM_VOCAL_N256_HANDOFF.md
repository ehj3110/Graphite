# Cube 1 mm — Vocal n=256 refinement handoff

**Last updated:** August 31, 2026  
**Status:** **In progress** — linear graded Split-P n=256 run started (warm-start from converged n=128 upflow cache).  
**Parent:** [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md), [VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md)

---

## Why n=256?

The 3-panel WSS figure (`figures/Cube1mm_Vocal_WSS_3panel_comparison.png`) still looks **blocky** at n=128. The existing cascade already warm-starts 32→64→128 by upsampling `(u, p)` and rebuilding `f_eq`. **n=128→n=256 is the same pattern** — not insane, already implemented in `graphite/lbm/vocal_warmstart.py` (`cache_velocity_upsample`).

`f.pkl` **cannot** be reused across resolutions; only velocity + pressure upsample onto the finer voxel grid.

GPU safe ceiling: `target_n=256` (`voxelizer._GPU_MAX_N`).

---

## Progress completed (August 31, 2026)

### Vocal convergence — figures now use converged caches

| Lattice | Canonical cache slug (upflow +Z) | `converged` | `steps_run` | Notes |
|---------|----------------------------------|-------------|-------------|-------|
| Woodpile extrude | `…_extrude_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms3000_tol0.001` | yes | 2250 | Continued +250 steps from `steps2000` |
| Split-P piecewise (+L/4) | `…_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001` | yes | 2000 | Final stage of upflow cascade (warm-started) |
| Split-P linear grad | `…_JacobianW_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001` | yes | 2000 | Final stage of upflow cascade (warm-started) |

**Caveat:** Split-P `steps_run=2000` is **only the final n=128 stage** after 32→64→128 cascade warm-start. Cold n=128 downflow needed **4750** steps. Mean-Uz tolerance can pass before the full field is settled — see [VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md).

### Aristo FEA (Boolean-trimmed Split-P, August 2026)

| Lattice | VTU (`fea/`) |
|---------|----------------|
| Woodpile extrude | `Woodpile_…_extrude_h0p015_1N_E25p8MPa_1N_aristo_fea.vtu` |
| Split-P piecewise (+L/4) | `SplitP_…_phaseTest_qL4_shift_x_h005_1N_E25p8MPa_1N_aristo_fea.vtu` |
| Split-P linear grad | `SplitP_Cube1mm_linearGrad_centerPhase_h005_1N_E25p8MPa_1N_aristo_fea.vtu` |

Settings: 1 N, E = 25.8 MPa, `flat_top_vertex_plane`, `single_surface`. Linear graded uses PyMeshLab remesh FEA surface (`…_cleaned_fea.stl`).

### Folder cleanup

`outputs/case_studies/cube_1mm/` reorganized:

| Path | Contents |
|------|----------|
| **`figures/`** | All comparison PNGs — **start here** |
| **`fea/`** | Canonical Aristo VTU + report JSON |
| Root `*.stl` | 3 canonical cleaned STLs (kept at root for Vocal cache STL paths) |
| **`fluid/vocal_metrics/`** | Per-run residual / metrics JSON |
| **`archive/superseded/`** | Obsolete phase tests, mesh sweeps, old PNGs |
| **`MANIFEST.json`** | Machine-readable index |

### Code touched this session

- `graphite/case_studies/cube_1mm/specs.py` — `figures_dir()`, `fea_dir()`, converged cache slug helpers
- `scripts/plot_cube_1mm_aristo_vocal_comparison.py` — converged caches, outputs to `figures/`
- `scripts/plot_cube_1mm_vocal_wss_comparison.py` — converged caches, outputs to `figures/`
- `scripts/continue_vocal_from_cache.py` — `--until-converged`
- `scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py` — **new** (see below)

---

## Current plan: n=256 refinement (one lattice at a time)

### Phase 1 — **linear graded Split-P only** (n=256 **failed — CUDA OOM**)

**Aug 31 result:** Warm-start upsample succeeded (grid 259×259×263), but the first LBM collide step hit **CUDA out of memory** on the current GPU (~2.5 GB VRAM ceiling).

**Status file:** `fluid/n256_linear_splitp_status.json`

**Retry options:**

```powershell
# Intermediate resolution (recommended on current GPU)
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py --target-n 192 --force-rerun

# Full n=256 on CPU (slow)
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py --device cpu --force-rerun
```

**Goal:** Sharper WSS / velocity fields by doubling voxel resolution.

| Item | Value |
|------|-------|
| STL | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl` |
| Seed cache (n=128) | `outputs/vocal/cache/SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001` |
| Target | `target_n=256`, `flow_chamber`, Re=5, Ma=0.05 |
| Warm start | Upsample `(u, p)` from seed → `f_eq` on finer grid (`warm_start_cache_dir`) |
| Convergence | `--converge`, `max_steps=6000`, `tolerance=0.001`, `min_steps=1000`, `check_interval=250` |
| Expected output cache | `…_JacobianW_cleaned_n256_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001` |

**Run:**

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py
```

**Review deliverables (when done):**

- Cache dir under `outputs/vocal/cache/` (slug above)
- `outputs/case_studies/cube_1mm/fluid/vocal_metrics/SplitP_linearGrad_n256_refine_metrics.json`
- `outputs/case_studies/cube_1mm/figures/SplitP_Cube1mm_linearGrad_n256_vocal_XZ_midY.png`

**WSS display-only experiments** (upsample, STL stamp — abandoned for figures): see [CUBE_1MM_WSS_DISPLAY_EXPERIMENTS.md](CUBE_1MM_WSS_DISPLAY_EXPERIMENTS.md).

**What “looks right”:** Finer scaffold walls on the XZ mid-Y slice; WSS bands less stair-stepped than n=128; `metrics.json` shows `converged: true` (or document steps if cap hit).

### Phase 2 — after user review

1. Regenerate **single-panel WSS** or side-by-side n=128 vs n=256 for linear grad only.
2. If good → run piecewise (+L/4), then woodpile extrude at n=256 (same pattern).
3. Update `plot_cube_1mm_vocal_wss_comparison.py` to use n=256 caches when all three exist.
4. Optional: `continue_vocal_from_cache.py` with field L2 diagnostics if mean-Uz converges too early.

### Phase 3 — not started

- Promote n=256 caches in `MANIFEST.json` and comparison figures
- Document k / WSS deltas n=128 vs n=256 in results text

---

## Canonical figure paths (n=128, current)

```
outputs/case_studies/cube_1mm/figures/Cube1mm_Aristo_Vocal_6panel_comparison.png
outputs/case_studies/cube_1mm/figures/Cube1mm_Aristo_Vocal_6panel_comparison_sharedScale.png
outputs/case_studies/cube_1mm/figures/Cube1mm_Vocal_WSS_3panel_comparison.png
```

Regenerate:

```powershell
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --plot-only
python scripts/plot_cube_1mm_aristo_vocal_comparison.py --shared-color-scale --plot-only
.\.venv_torch\Scripts\python.exe scripts/plot_cube_1mm_vocal_wss_comparison.py
```

---

## Do not use

- Vocal caches with `steps1000` / `steps2000` (non-converged) for publication figures
- Invalid upflow piecewise cache @ 500 steps (pre warm-start fix) — see cascade doc
- Implicit marching-cubes woodpile STL (`…_SF50_cleaned.stl` without `_extrude`)
- Files under `archive/superseded/` unless explicitly recovering history

---

## Open items (unchanged)

- Woodpile band-interface 90° + XY phase fix
- Persist `wss.npy` in Vocal cache on first compute
- Single three-column pore-size table in comparison doc
- Stair-step gaps in octet surface dual (universal dual handoff — separate track)
