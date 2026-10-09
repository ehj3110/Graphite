# Coaster Collection — Combined Deliverable Figures

This document details the unified 4-figure presentation suite for the complete 3D-printable coaster collection, formatted to match the reference high-contrast presentation canvas (1024:765 aspect ratio, rendered at 2048 × 1530 px high resolution).

---

## 1. Design & Spacing Standard

All 4 deliverable figures adhere to strict mathematical placement, cross-section accuracy, and typographic guidelines:
1. **Side Margins**: Exactly 10% side margins ($204.8\text{ px}$ left and right) on 2-row layouts (`C15/A15` and `Voroni`), ensuring maximum coaster size without edge crowding.
2. **Center-to-Center Spacing**: $\text{pitch} = 1.2 \times D$ (center-to-center distance along $X$ and $Y$ equals exactly $1.2 \times \text{diameter}$).
3. **Typographic Hierarchy (25% Size Reduction)**:
   - Main figure titles reduced to 51 pt bold.
   - Individual coaster titles reduced to 44 pt (2-row figures) / 34 pt (3-row figures) and centered directly above each coaster.
4. **Exact Cross-Section Topology**:
   - **C15**: Full Frank-Kasper Laves phase tiling ($1800$ basis atoms, cutoff $0.45 \times a$, $z$-limits $5.0\text{ mm}$, offsets $0.0$, $3.125$, $6.25\text{ mm}$).
   - **A15**: True cubic Kagome slices (basis atoms, cutoff $0.62 \times a$, $z$-limits $2.5\text{ mm}$, offsets $0.0$, $3.125$, $6.25\text{ mm}$).
   - **Voronoi**: Exact seeds 42, 43, 44 with dense ($n=450$) and sparse ($n=90$) cell counts, dropping seed 45 `large_v4`.
   - **Explicit**: Standardized $1.0\text{ mm}$ strut width, full triangular height scaling ($s = R\sqrt{3}, h = s\sqrt{3}/2$), and $25.4\text{ mm}$ tesseract pitches.
   - **TPMS**: Continuous level-set scalar contouring at $Z=0.0\text{ mm}$ and $Z=2.4\text{ mm}$.
   - **Universal Solid Frame**: Outer diameter $100.0\text{ mm}$, inner diameter $93.65\text{ mm}$ ($3.175\text{ mm}$ solid perimeter rim).
5. **High Contrast & Elevation**: Deep solid carbon (`#141416`) on studio neutral background (`#F8F9FA`), complemented by soft Gaussian drop shadows.

---

## 2. Deliverables Summary

| Figure Title | Output File | Coasters Included | Grid Format | Diameter ($D$) | Pitch ($1.2 \times D$) | Side Margin |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C15/A15** | `outputs/Coasters/c15_a15_previews.png` | 6 | 2 Rows × 3 Cols | 482 px | 578.4 px | 10.0% (204.8 px) |
| **Voroni** | `outputs/Coasters/voroni_previews.png` | 6 | 2 Rows × 3 Cols | 482 px | 578.4 px | 10.0% (204.8 px) |
| **Explicit** | `outputs/Coasters/explicit_previews.png` | 9 | 3 Rows × 3 Cols | 390 px | 468.0 px | 18.0% (369.0 px) |
| **TPMS** | `outputs/Coasters/tpms_previews.png` | 8 | 3 Rows [3, 2, 3] | 360 px | 432.0 px | 20.9% (428.0 px) |

---

## 3. Figures & Gallery

### 1. C15/A15
Combines the C15 Frank-Kasper polyhedral network (Row 1: offsets 0.0, 3.125 mm, 6.25 mm) and A15 cubic Kagome slices (Row 2: offsets 0.0, 3.125 mm, 6.25 mm):

![C15/A15 Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews.png)

---

### 2. Voroni
Combines high-density organic Voronoi cells (Row 1: `small_v1`, `small_v2`, `small_v3`) and low-density organic cellular patterns (Row 2: `large_v1`, `large_v2`, `large_v3`). The least dense sparse pattern (`large_v4`) was excluded to maintain a balanced 2×3 grid:

![Voroni Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews.png)

---

### 3. Explicit
Combines all 9 2D explicit lattice extrusions into a 3×3 grid. Features 5 triangular tessellations (`Tri_Tetrahedral`, `Tri_Icosahedral`, `Tri_Kelvin`, `Tri_Tesseract`, `Tri_Rhombic`) and 4 square tessellations (`Sq_Grid`, `Sq_Icosahedral`, `Sq_Kelvin`, `Sq_Tesseract`):

![Explicit Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews.png)

---

### 4. TPMS
Continuous minimal surfaces across 8 designs. Row 1 features zero-level sets ($Z=0$), Row 2 features centered Neovius transitions ($Z=0$ and $Z=2.4\text{mm}$), and Row 3 features mid-height slices ($Z=2.4\text{mm}$):

![TPMS Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews.png)
