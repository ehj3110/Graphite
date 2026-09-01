# Vocal cascade warm-start & up/down comparison (1 mm cube)

**Last updated:** July 2026  
**Code:** `graphite/lbm/vocal_warmstart.py`, `graphite/lbm/vocal_cache.py`, `graphite/lbm/lettuce_solver.py`  
**Scripts:** `scripts/run_splitp_cascade_32_64_128_*.py`, `scripts/plot_cube_1mm_vocal_updown_*.py`, `scripts/continue_vocal_from_cache.py`  
**Outputs:** `outputs/vocal/cache/`, `outputs/case_studies/cube_1mm/`

Multiscale **32 → 64 → 128** Vocal warm-start for Split-P **downflow** and **upflow**, with validated up/down field comparison and variance maps.

**Related:** [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md), [VOCAL.md](VOCAL.md), [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md)

---

## Summary figures

| Figure | Script | Description |
|--------|--------|-------------|
| `Cube1mm_SplitP_Downflow_Cascade_32_64_128_2panel.png` | `run_splitp_cascade_32_64_128_downflow.py` | Split-P downflow cascade (piecewise + linear) |
| `Cube1mm_SplitP_Upflow_Cascade_32_64_128_2panel.png` | `run_splitp_cascade_32_64_128_upflow.py` | Split-P upflow cascade (seed: downflow + flip Uz) |
| `Cube1mm_Vocal_UpDown_2x3_comparison.png` | `plot_cube_1mm_vocal_updown_comparison.py` | 2×3: up (+Z) vs down (−Z) for all three lattices |
| `Cube1mm_Vocal_UpDown_variance_direct.png` | `plot_cube_1mm_vocal_updown_variance.py --mode direct` | 1×3: \((\|u\|_\uparrow - \|u\|_\downarrow)^2\) heatmaps |
| `Cube1mm_Vocal_UpDown_variance_z_mirror.png` | `plot_cube_1mm_vocal_updown_variance.py --mode z_mirror` | 1×3: up vs Z-mirrored down \|u\| variance |

All under `outputs/case_studies/cube_1mm/`.

---

## Convergence criterion

Cascade stages use `--converge` (not a fixed step count). Each stage stops when **either**:

1. **Converged:** relative change in domain-mean lattice \(U_z\) falls below tolerance **and** `steps_run >= min_steps`, **or**
2. **Cap reached:** stage `max_steps` is exhausted.

### Formula (`LettuceSolver.run_until_convergence`)

At each checkpoint (default **250** steps):

```text
rel_change = |mean(Uz)_k - mean(Uz)_{k-1}| / |mean(Uz)_k|
```

**Converged** when `rel_change < tolerance` **and** `steps_run >= min_steps`.

### Default cascade parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `tolerance` | **1e-3** | Relative mean-\(U_z\) residual |
| `min_steps` | **1000** | Upflow re-run (prevents early false convergence) |
| `check_interval` | **250** | |
| `Re` / `Ma` | 5.0 / 0.05 | |
| n=32 `max_steps` | 5000 | |
| n=64 `max_steps` | 6000 | |
| n=128 `max_steps` | 6000 | |

### Limitations

- **Mean \(U_z\) only** — does not guarantee full-field velocity convergence. A run can pass while local speeds are still evolving (this caused the first broken upflow “converged @ 500 steps” result).
- For continuation diagnostics, use `scripts/continue_vocal_from_cache.py` (reports mean-\(U_z\) rel **and** fluid velocity L2 rel per interval).
- WSS / permeability are not used as convergence metrics.

---

## Warm-start implementation (current)

When `f.pkl` cannot be reused (cross-resolution upsample or flow-direction flip), Vocal rebuilds the LBM state as **equilibrium** from physical \((p, u)\):

```text
f = f_eq(ρ(p), u)    # Lettuce Flow.initialize()
```

### Per-stage transitions

| Transition | Velocity | Pressure | Mode string |
|------------|----------|----------|-------------|
| Downflow n=128 → upflow n=32 | Downsample + **flip \(U_z\)** | Upsample + **Z-flip** | `cache_velocity_upsample_flipUz` |
| n=32 → n=64 → n=128 (same BC) | Upsample trilinear | Upsample trilinear | `cache_velocity_upsample` |
| Same grid + same BC | — | — | `cache_f` (loads `f.pkl` directly) |

### Cache files used

| File | Role |
|------|------|
| `velocity.npy` | Physical velocity (3, nx, ny, nz), m/s |
| `pressure.npy` | Physical pressure (nx, ny, nz), Pa — **saved on all new runs** |
| `f.pkl` | Full LBM distribution (strongest IC when grid matches) |
| `metrics.json` | `converged`, `steps_run`, ΔP, k, WSS stats |

Pressure for warm-start is loaded from `pressure.npy` if present, else extracted from `f.pkl`, then upsampled with `scipy.ndimage.zoom` (order=1).

### Direction reversal (upflow seed from downflow)

1. Downsample **velocity** from converged downflow n=128.
2. Negate **\(U_z\)** (`warm_start_flip_uz=True`).
3. Load/upsample **pressure**, reverse Z index (`p[:,:,k] → p[:,:,nz-1-k]`).
4. Target BC: `flow_chamber` (+Z); downflow seed uses `flow_chamber_reverse`.

### Analytic fallback (`build_analytic_initial_pu`)

Used only when no pressure is available from cache:

- Uniform \(\|u_z\| \approx 5\) mm/s in fluid (sign per BC)
- Linear pressure ramp **4000 Pa** across Z (3D broadcast `(nx, ny, nz)`)
- **Not** interchangeable with a converged field — use cache pressure whenever possible

---

## Bugs found and fixed (July 2026)

The first upflow cascade produced a **dead top layer** (piecewise \|u\| @ top ≈ 0.4 mm/s vs ≈ 3.7 mm/s downflow). Root causes:

| Bug | Symptom | Fix |
|-----|---------|-----|
| Analytic ΔP = **50 Pa** vs steady **~4000 Pa** | Mismatched \(f_\mathrm{eq}\) when pairing cache \(u\) with analytic \(p\) | `_FLOW_CHAMBER_DP_PA = 4000`; prefer cache pressure |
| Pressure IC was **1D** `(nz,)` not `(nx,ny,nz)` | Corrupt equilibrium initialization | Broadcast ramp: `p[:] = ramp[None, None, :]` |
| Coarse→fine warm-start ignored cache **pressure** | n=64→n=128 used analytic \(p\) only | `_load_pressure_array()` + upsample; Z-flip on direction reversal |
| Mean-\(U_z\) tol alone | “Converged @ 500 steps” with evolving field | `min_steps=1000`; field diagnostics in `continue_vocal_from_cache.py` |

**Invalid caches (do not use):** first upflow `conv_ms6000_tol0.001` piecewise run @ 500 steps; `steps3500` continuation (patched state, pre-fix pipeline).

---

## Downflow cascade (converged)

**Script:** `scripts/run_splitp_cascade_32_64_128_downflow.py`  
**BC:** `flow_chamber_reverse` (−Z)

| Case | n=32 | n=64 | n=128 |
|------|------|------|-------|
| Piecewise (X +L/4) | 3000 | 3500 | 4750 |
| Linear graded | 3500 | 3750 | 4750 |

**Fine caches:**

```
SplitP_Cube1mm_piecewise_..._n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001
SplitP_Cube1mm_linearGrad_..._n128_flow_chamber_reverse_re5_ma0.05_conv_ms6000_tol0.001
```

---

## Upflow cascade (fixed warm-start, July 2026)

**Script:** `scripts/run_splitp_cascade_32_64_128_upflow.py --min-steps 1000`  
**BC:** `flow_chamber` (+Z)  
**Seed:** converged downflow n=128 (table above)

| Case | n=32 | n=64 | n=128 |
|------|------|------|-------|
| Piecewise (X +L/4) | 5000 (cap, not converged) | 1000 | 2000 |
| Linear graded | 2500 | 1750 | 2000 |

Piecewise n=32 oscillated on mean \(U_z\) and hit the 5000 cap; downstream stages still converged and produced a healthy n=128 field.

**Fine caches (canonical upflow):**

```
SplitP_Cube1mm_piecewise_..._n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001
SplitP_Cube1mm_linearGrad_..._n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001
```

### Piecewise n=128 Z-profile check (mean \|u\| in fluid, mm/s)

| | Bottom | Mid | Top |
|--|--------|-----|-----|
| Upflow (fixed) | 2.98 | 4.84 | **3.63** |
| Downflow | 2.94 | 4.87 | 3.72 |

---

## Up/down comparison

**Script:** `scripts/plot_cube_1mm_vocal_updown_comparison.py`

| Column | Up cache | Down cache |
|--------|----------|------------|
| Woodpile extrude | `..._extrude_..._flow_chamber_..._steps2000` | `..._flow_chamber_reverse_..._steps2000` |
| Split-P piecewise | converged upflow cascade n=128 | converged downflow cascade n=128 |
| Split-P linear | converged upflow cascade n=128 | converged downflow cascade n=128 |

Woodpile still uses the original 2000-step caches (not re-run through cascade).

### Should upflow and downflow look identical?

**Not in general.** These lattices are **Z-asymmetric**:

- Graded pore sizes (500 µm bottom → 1000 µm top)
- Piecewise bottom-band **X +L/4** phase shift

Reversing flow direction ≠ mirroring the domain in Z.

---

## Variance heatmaps

**Script:** `scripts/plot_cube_1mm_vocal_updown_variance.py`

XZ mid-Y slice, flow-chamber padding stripped. Solid = dark gray mask.

### `--mode direct` (same grid point)

Per fluid voxel:

```text
variance(x, z) = (|u|_up − |u|_down)²
```

| Lattice | RMS Δ\|u\| | Interpretation |
|---------|------------|----------------|
| Woodpile extrude | 0.05 mm/s | Nearly identical speed magnitudes |
| Split-P piecewise | 0.17 mm/s | Excellent agreement after fixed cascade |
| Split-P linear | 0.30 mm/s | Slightly more mismatch |

### `--mode z_mirror` (Z-flipped down)

```text
variance(x, z) = (|u|_up(x,z) − |u|_down(x, nz−1−z)|)²
```

RMS **~13–20 mm/s** — large, as expected for asymmetric geometry (not a convergence failure).

```powershell
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode direct
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode z_mirror
```

---

## Run commands

```powershell
$env:PYTHONPATH = (Get-Location).Path

# Downflow cascade
.\.venv_torch\Scripts\python.exe scripts/run_splitp_cascade_32_64_128_downflow.py

# Upflow cascade (fixed warm-start, min 1000 steps/stage)
.\.venv_torch\Scripts\python.exe scripts/run_splitp_cascade_32_64_128_upflow.py --min-steps 1000

# Continue an existing cache + residual report
.\.venv_torch\Scripts\python.exe scripts/continue_vocal_from_cache.py `
  --cache-dir outputs/vocal/cache/<slug> --extra-steps 3000 --check-interval 250

# Figures (no LBM)
python scripts/plot_cube_1mm_vocal_updown_comparison.py
python scripts/plot_cube_1mm_vocal_updown_variance.py --mode direct
```

Dry-run / plot-only flags are supported on cascade scripts (`--dry-run`, `--plot-only`, `--resume`).

---

## Code touchpoints

| Area | Files |
|------|-------|
| Warm-start IC | `graphite/lbm/vocal_warmstart.py` |
| Cache I/O + `pressure.npy` | `graphite/lbm/vocal_cache.py` |
| `min_steps` convergence | `graphite/lbm/lettuce_solver.py`, `vocal_run.py` (`VocalRunConfig.min_steps`) |
| Flip-Uz seed | `VocalRunConfig.warm_start_flip_uz` |
| Downflow cascade | `scripts/run_splitp_cascade_32_64_128_downflow.py` |
| Upflow cascade | `scripts/run_splitp_cascade_32_64_128_upflow.py` |
| Up/down + variance plots | `scripts/plot_cube_1mm_vocal_updown_comparison.py`, `plot_cube_1mm_vocal_updown_variance.py` |
| Continue + residuals | `scripts/continue_vocal_from_cache.py` |

---

## Superseded workflows

| Workflow | Status |
|----------|--------|
| Fixed 1000/2000-step n=128 only | Too slow; poor convergence |
| n=64→n=128 only (`run_splitp_coarse_to_fine_downflow.py`) | Useful probe; full cascade preferred |
| First upflow cascade (broken warm-start) | **Invalid** — see bugs table |
| Analytic-only pressure with cache velocity | **Fixed** — use cache pressure |
