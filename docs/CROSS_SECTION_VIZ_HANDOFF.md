# Cross-section stress visualization (Aristo VTU)

**Code:** `graphite/aristo/cross_section_viz.py`, `scripts/plot_aristo_cross_sections.py`  
**Woodpile wrapper:** `graphite/case_studies/cube_1mm/run_woodpile_aristo.py`  
**Related:** [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md), [ARISTO.md](ARISTO.md), [CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md)

Cross-section figures are **post-processed from saved FEA VTUs** — no solver re-run is needed to change fill mode, blur, or colormap scaling.

---

## Pipeline

### Nodal modes (`voronoi-*`, `blur-heatmap`)

```
Aristo VTU (von_mises_nodal_MPa)
  → thin slab filter (± half-thickness normal to slice plane)
  → project to 2D in-plane coordinates + dedupe (max σ per rounded XY/XZ key)
  → raster fill (Voronoi / soft blur / full blur)
  → hard scaffold wall mask (void = white)
  → optional domain clip to [0, 1]² mm (--clip-unit-cube)
  → normalize to colorbar → turbo PNG
```

**Input field:** `von_mises_nodal_MPa` (volume-weighted nodal recovery).

### Element modes (`element-slice`, `element-slice-soft`) — recommended for extrude woodpile

```
Aristo VTU (von_mises_element_MPa + tet connectivity)
  → tets overlapping the slab → plane-cut polygon per tet
  → rasterize footprints (average stress when tets overlap a pixel)
  → [soft only] close 1-pixel cracks between footprints
  → [soft only] masked Gaussian blur (radius in mm, confined to solid)
  → void pixels stay white (NaN)
  → normalize to colorbar → turbo PNG
```

**Input field:** `von_mises_element_MPa` (piecewise-constant per tet). Void geometry comes from **which pixels intersect tets**, not a nodal Voronoi wall mask — so sparse nodal sampling cannot leave white holes inside struts.

**Default for woodpile extrude Aristo plots:** `element-slice-soft`.

---

## Fill modes (`--fill-mode`)

| Mode | Geometry | Stress field | Blur | Colormap normalization |
|------|----------|--------------|------|------------------------|
| **`element-slice-soft`** (woodpile default) | Tet plane-cut footprints | Element σ_vm (averaged at overlaps) | Masked Gaussian, **0.015 mm** default | **Raster peak** |
| **`element-slice`** | Tet plane-cut footprints | Element σ_vm | None | Slice peak |
| **`voronoi-sharp`** | Nodal Voronoi + wall mask | Nodal σ_vm | None | Slice peak |
| **`voronoi-soft`** | Nodal Voronoi + wall mask | Nodal σ_vm | σ = **0.009 mm** | Raster peak |
| **`blur-heatmap`** | Nodal seeds | Nodal σ_vm | σ = **0.018 mm** | Raster peak |
| **`voronoi-bounded`** / **`voronoi-bounded-soft`** | STL-slice mask + Voronoi | Nodal σ_vm | optional | varies |
| **`voronoi-disk`** | Legacy disk cap | Nodal σ_vm | — | Slice peak (avoid for new work) |

### Smoothing is by **distance (mm)**, not node count

For `element-slice-soft`, blur radius is specified in **millimeters** and converted to pixels from the raster spacing:

```text
sigma_pixels = blur_radius_mm / pixel_spacing_mm
```

At `--raster-pixels 1200` over a 1 mm clip, pixel spacing ≈ 0.83 µm, so the default **0.015 mm** radius ≈ **18 pixels** σ. Override with `--blur-radius-mm` on `plot_aristo_cross_sections.py` (woodpile wrapper forwards the same flag when plotting).

Blur uses a **masked** Gaussian: only material pixels contribute; void stays white.

### Element-slice-soft artifact fixes (June 2026)

Early soft plots showed thin **gray vertical seams** at mesh column lines (e.g. Y ≈ 0.38, 0.9 on YZ slices). Root causes and fixes:

| Issue | Fix |
|-------|-----|
| Gray `material=0.5` contour overlay on jagged raster edges | Contour **disabled** for `element-slice` / `element-slice-soft` |
| Bilinear `imshow` bleeding NaN void into solid | **Nearest** display interpolation (blur already applied in data space) |
| Last-write-wins tet rasterization along vertical mesh seams | **Average** stress when multiple tets claim a pixel |
| 1-pixel cracks between adjacent tet footprints | **Gap fill** (1-pixel dilation + nearest-neighbor stress) before blur |
| `quality_ok` mask punching holes in footprints | **Not applied** in soft mode (blur handles sliver spikes; holes caused seam lines) |

### What is **raster peak**?

After blur, each displayed pixel is a **weighted average** of nearby values, so local maxima are **lower** than the raw slice peak. If we still divided by the nodal slice peak, the hottest pixel might only reach ~0.6–0.8 on the colorbar and never show red.

**Raster peak** = `max(blurred raster field)` in that panel. The colorbar is **σ_vm / raster peak**, so the brightest pixel in the *plotted* field maps to 1.0 (red). The subplot title still reports **slice peak stress** (true max in the slab before display normalization) for physical reference.

For **`voronoi-sharp`** and **`element-slice`**, raster peak ≈ slice peak (no blur).

---

## Slice planes and subplot titles

**Figure title (suptitle):** `Normalized von Mises - {Vertical|Horizontal cross-section}`

Each **panel** title has two lines:

1. **Case label** — `--label` (e.g. `Cross-hatch woodpile`, `Piecewise`)
2. **Slice peak σ_vm** — max σ_vm in the slab (MPa)

**Colorbar:** `Normalized σ_vm` (0 = min in panel, 1 = peak of displayed raster; blur modes use raster peak).

| `--plane` | Figure cross-section type | Axes |
|-----------|---------------------------|------|
| `xz` | Vertical cross-section | X vs Z |
| `yz` | Vertical cross-section | Y vs Z |
| `xy` | Horizontal cross-section | X vs Y |

---

## Domain clipping

MC / raster padding can paint slightly outside the nominal `[0, 1]³` mm cube. Use **`--clip-unit-cube`** to clip raster data and axis limits to **X, Z ∈ [0, 1]** (XZ), **Y, Z ∈ [0, 1]** (YZ), or **X, Y ∈ [0, 1]** (XY).

Custom bounds: `--domain-clip A_MIN A_MAX B_MIN B_MAX`.

---

## Woodpile extrude cross-sections (h = 0.015 mm example)

```powershell
$env:PYTHONPATH = (Get-Location).Path

# Plot only (reuses existing VTU)
python graphite/case_studies/cube_1mm/run_woodpile_aristo.py `
  --match-splitp-pores --h 0.015 --plot-only --fill-mode element-slice-soft

# Slightly more smoothing
python scripts/plot_aristo_cross_sections.py `
  --vtu outputs/case_studies/cube_1mm/..._aristo_fea.vtu `
  --fill-mode element-slice-soft --blur-radius-mm 0.018 `
  --clip-unit-cube --plane yz --slice-center-mm 0.3 ...
```

Outputs use suffix `elementslicesoft` in the filename stem.

---

## 1 mm³ analytic-cap case study (E = 25.8 MPa, 1 N, h = 0.005)

| Variant | FEA VTU stem |
|---------|----------------|
| Piecewise | `SplitP_Cube1mm_piecewise_h005_1N_analyticCap_E25p8MPa_1N_aristo_fea` |
| Linear graded (Jacobian W) | `SplitP_Cube1mm_linearGrad_centerPhase_h005_1N_analyticCap_E25p8MPa_1N_aristo_fea` |

**Geometry:** box domains use **analytic flat caps** (`axis_aligned_box_sdf` in `graphite/geometry/masking.py`), not voxelized box EDT, to avoid stairstepped bottom/top skins and artificial cap stress concentrations.

**Nodal peaks (model):** ~64 MPa piecewise, ~23 MPa graded @ E = 25.8 MPa.

### Example commands

```bash
# Voronoi soft — recommended for Split-P side-by-side comparisons
python scripts/plot_aristo_cross_sections.py \
  --vtu outputs/case_studies/cube_1mm/SplitP_Cube1mm_piecewise_h005_1N_analyticCap_E25p8MPa_1N_aristo_fea.vtu \
  --label "Piecewise" \
  --plane xz --slice-center-mm 0.5 --half-thickness-mm 0.012 \
  --fill-mode voronoi-soft --clip-unit-cube \
  --output outputs/case_studies/cube_1mm/SplitP_Cube1mm_analyticCap_cross_section_XZ_midY_voronoiSoft.png
```

### Default tuning

| Parameter | Typical value | Role |
|-----------|---------------|------|
| `--half-thickness-mm` | 0.012 (XZ/YZ), 0.05 (XY) | Slab half-thickness — must capture strut nodes |
| `--void-distance-mm` | 0.018 | Voronoi wall radius from nearest seed; outside = white |
| `--raster-pixels` | 1200 | Raster resolution per panel |
| `--dpi` | 300 | PNG export DPI |
| `--blur-radius-mm` | 0.015 (`element-slice-soft`), 0.018 (`blur-heatmap`) | Physical blur radius in mm |

---

## Used in three-lattice comparison figures

The 6-panel Aristo row in `Cube1mm_Aristo_Vocal_6panel_comparison.png` uses **`element-slice-soft`** for all three columns:

| Column | Slice plane | Notes |
|--------|-------------|-------|
| Woodpile extrude | YZ @ X = 0.3 mm | Struts primarily in Y/Z |
| Split-P piecewise (X +L/4) | XZ @ Y = 0.5 mm | Phase-offset variant |
| Split-P linear grad | XZ @ Y = 0.5 mm | Jacobian-warped |

See [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md).

---

*Last updated: July 2026 — element-slice-soft, distance-based blur, seam fixes, woodpile extrude defaults, three-lattice comparison slices.*
