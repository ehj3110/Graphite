# Vocal — 1 mm cube piecewise vs linear grading (status handoff)

**Last updated:** July 2026  
**Parent doc:** [VOCAL.md](VOCAL.md)  
**Three-lattice comparison (woodpile + both Split-P):** [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)  
**Woodpile band-interface note:** [WOODPILE_CUBE_COMPARISON.md](WOODPILE_CUBE_COMPARISON.md)  
**FEA counterpart:** [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md)

This document captures the current state of the **Split-P 1 mm cube** Vocal (LBM) comparison and how to extend runs toward steady state.

---

## Goal

Compare steady-state flow through two Jacobian-warped Split-P cubes (SF ≈ 0.33, L = 500 µm bottom → 1000 µm top):

| Case | STL |
|------|-----|
| **Piecewise** | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_cleaned.stl` |
| **Linear grading** | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned.stl` |

Outputs: permeability **k**, pressure drop **ΔP**, WSS, and CFD-style XZ mid-Y velocity slices (viridis speed, dark scaffold walls, white quiver).

---

## Where we are

### Infrastructure (done)

| Item | Location |
|------|----------|
| Vocal orchestration | `graphite/lbm/vocal_run.py` |
| Disk cache (`velocity.npy`, `f.pkl`, `metrics.json`) | `outputs/vocal/cache/{slug}/` |
| Warm start (cache `f` → velocity → analytic) | `graphite/lbm/vocal_warmstart.py` |
| Single-case CLI | `scripts/run_vocal.py` |
| Side-by-side comparison CLI | `scripts/plot_vocal_flow_comparison.py` |
| Residual diagnostic (+N steps from cache) | `scripts/continue_cube_1mm_vocal.py` |

### Simulation settings (all runs so far)

| Parameter | Value |
|-----------|-------|
| Grid | `target_n=128` → 131×131×135 with flow-chamber padding |
| BC | `flow_chamber` (inlet/outlet pressure, lateral solid casing) |
| Re / Ma | 5.0 / 0.05 |
| Voxel pitch | ~7.8 µm |

### Cached results @ 1000 fixed steps (`converged: false`)

Cache slugs end in `_steps1000`. Summary JSON:  
`outputs/case_studies/cube_1mm/SplitP_Cube1mm_vocal_comparison_summary.json`

| Metric | Piecewise | Linear grading |
|--------|-----------|----------------|
| LBM steps (this cache) | 1000 | 1000 |
| Solid fraction | 0.381 | 0.385 |
| ΔP | ~4000 Pa | ~4000 Pa |
| Permeability **k** | 1.61×10⁻⁴ mm² | 1.81×10⁻⁴ mm² |
| WSS max | 348 Pa | 506 Pa |
| Volume peak \|u\| | 34.3 mm/s | **58.9 mm/s** |
| Mean \|u\| (fluid) | 4.95 mm/s | 5.67 mm/s |
| XZ slice peak \|u\| | 34.2 mm/s | 58.9 mm/s |

**Figure (CFD style):**  
`outputs/case_studies/cube_1mm/SplitP_Cube1mm_vocal_cross_section_XZ_midY.png`

### Continuation experiment (+500 steps from `f.pkl`, June 2026)

Both cases were warm-started from the steps-1000 `f.pkl` and advanced **500 more lattice steps** (logical total **1500**). Residual = relative change in **domain-mean Uz** between 50-step checkpoints (same criterion as `run_until_convergence`).

| | Piecewise @ 1500 | Linear grading @ 1500 |
|--|------------------|------------------------|
| Final mean Uz | 2.91 mm/s | 3.36 mm/s |
| Final residual (last 50-step interval) | **7.7×10⁻⁴** | **2.6×10⁻³** |
| Peak \|u\| start → end | 34.3 → 35.7 mm/s | **58.9 → 53.9 mm/s** |
| Mean \|Δu\|/\|u₀\| over fluid (500 steps) | 3.4% | 17.2% |

**Interpretation:**

- Neither case is steady-state at 1000 steps; the comparison PNG is a **quick preview**, not publication-grade.
- The linear-grading **58.9 mm/s spike is transient** — it dropped ~9% over the extra 500 steps while bulk flow was still rising.
- Piecewise is closer to the comparison-script tolerance (1×10⁻³); linear grading still evolves ~3× faster by the mean-Uz residual.

> **Note:** The +500 continuation was run with `continue_vocal_500.py` for diagnostics only. It does **not** write a new cache entry. The on-disk caches are still at **steps1000**.

---

## Convergence criterion

`LettuceSolver.run_until_convergence()` stops when:

```
rel_change = |mean_uz_new - mean_uz_prev| / |mean_uz_new|  <  tolerance
```

checked every `check_interval` steps over the fluid domain’s mean Z-velocity.

| Context | Default tolerance | Default check interval |
|---------|-------------------|------------------------|
| `scripts/run_vocal.py` | 1×10⁻⁴ | 500 |
| `scripts/plot_vocal_flow_comparison.py --converge` | 1×10⁻³ | 250 |

---

## How to run more iterations

All commands from repo root with `.venv_torch`.

### Option A — Converge both cases (recommended for final numbers)

Warm-starts from cached `f.pkl`, runs until residual &lt; tolerance or `max_steps`:

```powershell
.\.venv_torch\Scripts\python.exe scripts\plot_vocal_flow_comparison.py `
  --cases both `
  --converge `
  --warm-start cache `
  --max-steps 15000 `
  --check-interval 250 `
  --tolerance 1e-3
```

Expect **~15–30+ min** on a laptop GPU depending on how many steps are needed. Progress lines print `mean Uz` and `rel_change` each check interval.

### Option B — Fixed step count from warm start (cache a new slug)

`--steps N` runs **N new LBM steps** starting from the warm-started `f` distribution (geometry-matched cache, usually the latest `f.pkl`). Use `--force-rerun` so a new slug is written even if an old cache exists.

```powershell
# Example: 1500-step run from warm-started state (saves ..._steps1500/)
.\.venv_torch\Scripts\python.exe scripts\run_vocal.py `
  --geometry stl `
  --stl outputs/case_studies/cube_1mm/SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_cleaned.stl `
  --target-n 128 --boundary-style flow_chamber `
  --steps 1500 --warm-start cache --force-rerun `
  --output outputs/case_studies/cube_1mm/SplitP_Cube1mm_piecewise_vocal_1500steps.png `
  --metrics-json outputs/case_studies/cube_1mm/vocal_metrics/piecewise_1500_metrics.json
```

Repeat for the linear-graded STL, then replot:

```powershell
.\.venv_torch\Scripts\python.exe scripts\plot_vocal_flow_comparison.py `
  --cases both --steps 1500 --plot-only
```

(`--plot-only` requires a valid cache for the requested `--steps` value.)

### Option C — Residual table without updating cache

Edit `--extra-steps` / `--check-interval` in `scripts/continue_cube_1mm_vocal.py`, then:

```powershell
.\.venv_torch\Scripts\python.exe scripts/continue_cube_1mm_vocal.py
```

Loads steps-1000 `f.pkl` for both STLs, prints mean Uz / rel_change / peak \|u\| every 50 steps. Does not save results.

### Option D — Replot only (no LBM)

```powershell
.\.venv_torch\Scripts\python.exe scripts/plot_vocal_flow_comparison.py `
  --cases both --steps 1000 --plot-only
```

---

## Suggested workflow toward a trustworthy comparison

1. **Converge both cases** (Option A) with `--warm-start cache` and `--tolerance 1e-3`.
2. **Replot** from the converged caches (`--plot-only` with matching `--steps` or after convergence caches are written).
3. **Compare metrics** in `SplitP_Cube1mm_vocal_comparison_summary.json` and per-case `vocal_metrics/*_metrics.json`.
4. **Optional:** align STLs with Aristo analytic-cap geometry before claiming FEA/CFD parity ([CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md)).
5. **Optional:** if peaks remain high in narrow throats, try `target_n=192` (VRAM ~0.8 GB) on one case to check voxel-resolution sensitivity.

---

## Cache layout reference

```
outputs/vocal/cache/
  SplitP_Cube1mm_piecewise_..._steps1000/
    manifest.json    # geometry + BC fingerprint
    metrics.json     # k, ΔP, WSS, steps_run, converged
    velocity.npy     # (3, nx, ny, nz) m/s
    f.pkl            # full Lettuce distribution (best warm start)
  SplitP_Cube1mm_linearGrad_..._steps1000/
    ...
```

Cache key includes STL fingerprint, `target_n`, BCs, Re/Ma, and **step count** (or converge params). Geometry-matched caches with different step counts can still warm-start via `find_geometry_cache()`.

---

## Open items

- [ ] Converged paired comparison figure and metrics table
- [ ] Save post-continuation state (e.g. steps1500 cache) if fixed-step workflow is preferred over `--converge`
- [ ] Analytic-cap STL parity with Aristo FEA inputs
- [ ] Document converged k / ΔP / WSS next to Aristo σ_vm in a single comparison note

---

## Related outputs

| Artifact | Path |
|----------|------|
| Comparison PNG | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_vocal_cross_section_XZ_midY.png` |
| Comparison JSON | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_vocal_comparison_summary.json` |
| Piecewise quick single-panel | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_piecewise_vocal_quick_1000steps.png` |
| Aristo stress comparison (FEA) | `outputs/case_studies/cube_1mm/SplitP_Cube1mm_aristo_cross_section_*` |
