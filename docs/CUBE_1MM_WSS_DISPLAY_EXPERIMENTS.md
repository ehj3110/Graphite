# Cube 1 mm — WSS figure display experiments (August 2026)

**Status:** Exploratory only — **not** adopted for canonical figures.  
**Parent:** [CUBE_1MM_VOCAL_N256_HANDOFF.md](CUBE_1MM_VOCAL_N256_HANDOFF.md), [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)

---

## Problem

The Vocal WSS cross-section figures looked **blocky / stair-stepped** at n=128 and n=192 (e.g. linear Split-P single-panel WSS).

### Root cause (confirmed)

| Layer | Resolution | Role |
|-------|------------|------|
| **STL** (`…_JacobianW_cleaned.stl`) | ~443k faces, median edge ~10 µm | Smooth watertight surface |
| **Vocal `target_n`** | pitch = 1 mm / `target_n` (e.g. **5.2 µm** at n=192) | **Axis-aligned voxel solid_mask** via `trimesh.voxelized(pitch)` |
| **WSS field** | One voxel thick at fluid–solid interface | Plotted on voxel geometry |

**Stair-stepping comes from Vocal voxelization (`target_n`), not from coarse STL triangles.** Refining the STL without increasing `target_n` does not change the LBM geometry.

Display-only tricks (upsampling the colormap) smooth **color** but not the **wall outline** unless geometry is changed separately.

---

## What we tried

### 1. Higher Vocal resolution (physics path — still open)

| Run | Result |
|-----|--------|
| n=192 linear Split-P | Converged on laptop; WSS still visibly voxelized |
| n=256 | **CUDA OOM** on laptop (~2.5 GB VRAM budget); planned on 32 GB desktop |

Script: `scripts/run_cube_1mm_vocal_refine_n256_linear_splitp.py`

This is the **only** way to remove stair-steps in both geometry and WSS without compositing.

---

### 2. Display upsample (`--wss-display-factor`)

**Code:** `graphite/lbm/vocal_run.py` — `upsample_slice_for_display()`  
**Script:** `scripts/plot_cube_1mm_vocal_wss_comparison.py --wss-display-factor {2,4}`

- Bilinear upsample of WSS slice + nearest-neighbor upsample of voxel solid mask
- Voxel axes unchanged; finer pixels for the colormap only
- **Effect:** Smoother color gradients along walls; **wall outline still stair-stepped**

**Example output:**  
`outputs/case_studies/cube_1mm/figures/SplitP_Cube1mm_linearGrad_n192_vocal_WSS_XZ_midY_x4.png`

---

### 3. Gaussian display smooth (`--wss-smooth-sigma`)

**Code:** `smooth_interface_slice_for_display()` (normalized convolution on interface band)

- Optional alternative to upsample; spreads sparse WSS samples in fluid
- **Effect:** Softer bands; can smear peaks slightly
- **Not preferred** after upsample experiment — user asked for interpolation rather than blur

---

### 4. STL cross-section + Vocal WSS “stamp” (`--stl-cross-section`) — **abandoned**

**Motivation:** Use the **smooth STL Y-slice** for black walls; upsample/interpolate n=192 WSS and paint it in a thin fluid band adjacent to the STL boundary (display composite).

**Code (left in tree, default off):**

- `rasterize_stl_xz_section()` — trimesh plane slice → matplotlib path fill
- `stamp_wss_on_stl_section()` — upsample WSS, optional σ smooth, distance-band stamp on STL fluid
- `plot_wss_comparison(..., use_stl_cross_section=True)`

**Example command:**

```powershell
.\.venv_torch\Scripts\python.exe scripts\plot_cube_1mm_vocal_wss_comparison.py `
  --stl-cross-section --wss-display-factor 4 --wss-smooth-sigma 1.0 `
  --linear-n 192 --no-single-linear `
  --output outputs\case_studies\cube_1mm\figures\SplitP_Cube1mm_linearGrad_n192_vocal_WSS_XZ_midY_stlStamp_x4.png
```

**Example output:**  
`outputs/case_studies/cube_1mm/figures/SplitP_Cube1mm_linearGrad_n192_vocal_WSS_XZ_midY_stlStamp_x4.png`

**Result:** Visually smooth outer walls, but **much of the internal wall structure disappeared** on the mid-plane slice — the composite read as “deleted walls” rather than a faithful cross-section. Likely causes:

- STL plane slice vs Vocal mid-Y index / padding alignment
- Solid fraction mismatch (STL slice ~40% solid vs voxel slice ~35% on same plane)
- Interface band too thin or WSS stamp not covering all voxel-resolved wall segments
- Closed-path rasterization missing nested voids / thin struts on this slice

**Decision:** Do **not** use for publication. Keep flag for reference; fix would need careful alignment QA or abandoning the approach.

---

## Canonical WSS figures (unchanged)

Still use **raw voxel geometry** from converged caches:

```powershell
.\.venv_torch\Scripts\python.exe scripts\plot_cube_1mm_vocal_wss_comparison.py
```

Default: no display upsample, no STL composite, no σ smooth.

| Figure | Notes |
|--------|--------|
| `figures/Cube1mm_Vocal_WSS_3panel_comparison.png` | Woodpile + piecewise n=128; linear **n=192** |
| `figures/SplitP_Cube1mm_linearGrad_n192_vocal_WSS_XZ_midY.png` | Single-panel linear n=192 |

---

## Recommended next step

1. **Desktop:** run n=256 (or n=320) linear Split-P Vocal refine — real geometry + WSS resolution gain.
2. **Figures:** regenerate WSS plots from new cache **without** display compositing unless re-tested.
3. Optional mild `--wss-display-factor 2` for slide decks only, with caption noting display upsampling.

---

## CLI reference (all optional, display-only)

| Flag | Default | Purpose |
|------|---------|---------|
| `--wss-display-factor` | `1` | Bilinear upsample colormap pixels |
| `--wss-display-order` | `1` | `1` bilinear, `3` bicubic |
| `--wss-smooth-sigma` | `0` | Gaussian stamp smooth (voxels) |
| `--stl-cross-section` | off | STL walls + Vocal stamp (**experimental, broken**) |
| `--linear-n` | `192` | Linear Split-P cache column |

Metrics (`wss_max_pa`, permeability, convergence) always come from the Vocal cache — never from display post-processing.
