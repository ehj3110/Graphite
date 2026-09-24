# Prusa Core One Filament Spool Caps — Contest Catalog & Reproduction Guide

This document records the complete architecture, geometric calibration, file naming conventions, test suite, and step-by-step commands to recreate all **150 3D-printable spool cap STLs** and their **2D preview sheets** for the Prusa Core One Printables Contest.

---

## 1. Project Overview

The spool cap system combines 2D explicit surface lattices (Square, Triangular, Crystal Laves phases A15/C15, Voronoi tessellations) and 3D TPMS minimal surfaces with the official Prusa Core One fixture ([`test_parts/capplain.stl`](../test_parts/capplain.stl)).

The caps provide aesthetic and functional front-flange covers for filament spools used in the Prusa Core One enclosure, featuring:
- Watertight boolean union via [Manifold3D](https://github.com/elalish/manifold).
- Coordinated scaling across three standard size tiers.
- High in-plane solid-fraction vertical centering for TPMS surfaces.
- A distinctive 8-part numbered series (#1 to #8) with $1.0\,\text{mm}$ outline ribbons.

---

## 2. Dimensional & Physical Specifications

| Parameter | Small Tier | Medium Tier | Large Tier |
| :--- | :--- | :--- | :--- |
| **Outer Diameter ($\varnothing$)** | $45.5\,\text{mm}$ (fixture base) | $122.75\,\text{mm}$ (midpoint) | $200.0\,\text{mm}$ (1kg spool flange) |
| **Lattice Thickness ($H$)** | $1.0\,\text{mm}$ | $5.0\,\text{mm}$ | $5.0\,\text{mm}$ |
| **Rim / Frame Width** | $3.0\,\text{mm}$ (Hexagon & TPMS) | $5.0\,\text{mm}$ (All) | $5.0\,\text{mm}$ (All) |
| **Center Offset ($X, Y$)** | $[124.0,\, 102.5]\,\text{mm}$ | $[124.0,\, 102.5]\,\text{mm}$ | $[124.0,\, 102.5]\,\text{mm}$ |
| **Z-Alignment** | Translated so bottom face is at $Z = 0.0$ on print bed | Translated so bottom face is at $Z = 0.0$ on print bed | Translated so bottom face is at $Z = 0.0$ on print bed |

### Unit Cell Scaling Across Sizes
- **Square Cells:** Small = $8.0\,\text{mm}$, Medium = $18.0\,\text{mm}$, Large = $25.0\,\text{mm}$
- **Triangular Cells:** Small = $7.0\,\text{mm}$, Medium = $18.0\,\text{mm}$, Large = $25.0\,\text{mm}$
- **Crystal A15:** Small = $20.0\,\text{mm}$, Medium = $28.0\,\text{mm}$, Large = $40.0\,\text{mm}$
- **Crystal C15:** Small = $35.0\,\text{mm}$, Medium = $45.0\,\text{mm}$, Large = $50.0\,\text{mm}$
- **TPMS Unit Cell:** Small = $12.0\,\text{mm}$, Medium = $20.0\,\text{mm}$, Large = $28.0\,\text{mm}$
- **Strut Stroke Width:** $1.4\,\text{mm}$ ($2.0\,\text{mm}$ for Kelvin cells)

---

## 3. Lattice Topologies & Vertical Calibration

### 3.1 Explicit 2D Lattices
- **Square:** `Sq_Grid`, `Sq_Icosahedral`, `Sq_Kelvin`, `Sq_Tesseract`
- **Triangular:** `Tri_Tetrahedral`, `Tri_Kelvin`, `Tri_Icosahedral`, `Tri_Rhombic`, `Tri_Tesseract`

### 3.2 Crystal Laves Phases
- **A15:** `A15_v1` ($Z_{\text{frac}} = 0.0$, Kagome planar dual slice)
- **C15 Laves Phase:**
  - `C15_v1`: $Z_{\text{frac}} = 0.000$ (Hexagonal ring network)
  - `C15_v2`: $Z_{\text{frac}} = 0.0625$ (Kagome-like intermediate structure)
  - `C15_v3`: $Z_{\text{frac}} = 0.125$ (Dense tetrahedral cross-section)

### 3.3 TPMS Minimal Surfaces (Peak Solid-Fraction Alignment)
To maximize structural integrity and in-plane connectivity, each TPMS unit cell was analyzed across its height to locate the global maximum solid fraction ($Z_{\text{peak}} / L$). The 3D sheet is evaluated symmetrically outward such that the midplane ($Z = 2.5\,\text{mm}$ on Medium/Large) coincides with the peak cross-section:

| TPMS Surface | $Z_{\text{peak}} / L$ | Peak Solid Fraction | Formula / Characteristic |
| :--- | :---: | :---: | :--- |
| **Gyroid** | $0.1250$ | $28.4\%$ | $\sin(x)\cos(y) + \sin(y)\cos(z) + \sin(z)\cos(x) = 0$ |
| **Diamond (Schwarz D)** | $0.0050$ | $49.1\%$ | Hexagonal nodal arrays at symmetry origin |
| **Lidinoid** | $0.0900$ | $36.9\%$ | Local maxima connecting double-channel tubes |
| **Neovius** | $0.0900$ | $35.5\%$ | Iso-offset $= 0.50$ for robust wall thickness |
| **Split-P** | $0.1900$ | $34.8\%$ | Non-intersecting interlocking labyrinth |
| **Schwarz P** | $0.2450$ | $29.3\%$ | Simple cubic strut-and-neck geometry |

### 3.4 Voronoi Tessellation
Voronoi lattices are seeded using strict physical density parameters (points per $\text{mm}^2$) to preserve visual density consistency across physical sizes:
- **Small ($\varnothing 45.5\,\text{mm}$):** Sparse = $0.015$, Medium = $0.030$, Dense = $0.060\,\text{pts/mm}^2$
- **Medium ($\varnothing 122.75\,\text{mm}$):** Sparse = $0.015$, Medium = $0.030$, Dense = $0.045\,\text{pts/mm}^2$
- **Large ($\varnothing 200.0\,\text{mm}$):** Sparse = $0.010$, Medium = $0.020$, Dense = $0.030\,\text{pts/mm}^2$

---

## 4. Numbered Series (#1 through #8)

Eight distinctive designs were selected across the categories and assigned numbers 1 through 8:
1. **#1:** `Sq_Kelvin` (Circle)
2. **#2:** `Sq_Tesseract` (Hexagon)
3. **#3:** `Tri_Tetrahedral` (Circle)
4. **#4:** `Tri_Kelvin` (Hexagon)
5. **#5:** `Voronoi_Medium` (Circle)
6. **#6:** `A15_v1` (Hexagon)
7. **#7:** `TPMS_Gyroid` (Circle)
8. **#8:** `TPMS_Split-P` (Hexagon)

### Number Embossing Rules
- **Height:** $30.0\,\text{mm}$ ($Y$-extent), using Arial / Helvetica Bold typeface.
- **Center:** Aligned to `[124.0, 102.5]`.
- **Outline Ribbon:** $1.0\,\text{mm}$ wall thickness surrounding the glyph.
- **Lattice Subtraction:** The inner volume of the glyph is cut out from the lattice so the number stands out with high contrast.
- **Orientation:** Mirrored appropriately during 2D generation so that numbers read naturally from left to right on the printed front face.

---

## 5. File Naming Convention & Directory Layout

All generated models live in a flat directory at `outputs/spool_caps/`:

```
outputs/spool_caps/
├── {LatticeType}_{SubType}_{Shape}_{Size}.stl
└── {LatticeType}_{SubType}_Numbered_{Number}_{Shape}_{Size}.stl
```

- **`{LatticeType}`:** `Sq`, `Tri`, `TPMS`, `Voronoi`, `A15`, `C15`.
- **`{SubType}`:** e.g. `Kelvin`, `Tesseract`, `Gyroid`, `v1`, `Dense`.
- **`{Shape}`:** `Circle` or `Hexagon`.
- **`{Size}`:** `Small`, `Medium`, or `Large`.
- **`Framed`:** Omitted completely from all filenames.

### Catalog Totals (150 Models)
- **Small:** 46 STLs (6 TPMS Circle, 32 Strut/Crystal/Voronoi Circle+Hex, 8 Numbered Circle)
- **Medium:** 52 STLs (12 TPMS Circle+Hex, 32 Strut/Crystal/Voronoi Circle+Hex, 8 Numbered Circle+Hex)
- **Large:** 52 STLs (12 TPMS Circle+Hex, 32 Strut/Crystal/Voronoi Circle+Hex, 8 Numbered Circle+Hex)

---

## 6. Reproduction Commands

### 6.1 Generate Full STL Catalog
To recreate all 150 STLs:
```powershell
python scripts/contest/generate_spool_caps.py --full-catalog
```

To recreate a specific size tier or pattern:
```powershell
# Single size tier
python scripts/contest/generate_spool_caps.py --size small
python scripts/contest/generate_spool_caps.py --size medium
python scripts/contest/generate_spool_caps.py --size large

# Filter by pattern
python scripts/contest/generate_spool_caps.py --pattern "TPMS*"
python scripts/contest/generate_spool_caps.py --pattern "*Numbered*"
```

### 6.2 Generate 2D Preview Sheets
To recreate the 7 coaster-style navy-blue cross-section preview sheets:
```powershell
python scripts/contest/generate_spool_cap_previews.py
```
Outputs produced:
- `spool_caps_sizes_a15.png` (Physical 1:1 scale comparison of Small, Medium, Large)
- `spool_caps_numbered_previews.png` (Numbered caps #1 to #8)
- `spool_caps_c15_previews.png` (C15 v1, v2, v3)
- `spool_caps_tpms_previews.png` (Single row at peak solid fraction)
- `spool_caps_explicit_square_previews.png` (Square lattices)
- `spool_caps_explicit_tri_previews.png` (Triangular lattices)
- `spool_caps_voronoi_previews.png` (Voronoi densities)

### 6.3 Automated Validation Suite
To execute the automated regression test suite:
```powershell
pytest tests/test_spool_caps.py
python scripts/export_core_spec.py --check
```

---

## 7. Key Python Files & Dependencies

- **Generator Script:** [`scripts/contest/generate_spool_caps.py`](../scripts/contest/generate_spool_caps.py)
- **Preview Script:** [`scripts/contest/generate_spool_cap_previews.py`](../scripts/contest/generate_spool_cap_previews.py)
- **Test Suite:** [`tests/test_spool_caps.py`](../tests/test_spool_caps.py)
- **Fixture STL:** [`test_parts/capplain.stl`](../test_parts/capplain.stl)
- **Dependencies:** `trimesh`, `manifold3d`, `shapely`, `scikit-image`, `scipy`, `matplotlib`, `numpy`, `pytest`.
