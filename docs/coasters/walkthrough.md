# Coaster Collection — Combined Deliverable Figures

This document details the unified 4-figure presentation suite for the complete 3D-printable coaster collection, formatted to match the reference high-contrast presentation canvas (1024:765 aspect ratio, rendered at 2048 × 1530 px high resolution) in both **Black (`#141416`)** and **Dark Blue (`#1e3a8a`)** editions.

---

## 1. Design & Spacing Standard

All deliverable figures adhere to strict mathematical placement, cross-section accuracy, and typographic guidelines:
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
5. **Color Editions**:
   - **Version 1 (Black)**: Deep solid carbon (`#141416`) on studio neutral background (`#F8F9FA`).
   - **Version 2 (Blue)**: Deep royal navy (`#1e3a8a`) on studio neutral background (`#F8F9FA`).

---

## 2. Deliverables Summary

| Figure Title | Black Output File | Blue Output File (`#1e3a8a`) | Coasters | Grid Format | Diameter ($D$) | Pitch ($1.2 \times D$) | Side Margin |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C15/A15** | `outputs/Coasters/c15_a15_previews.png` | `outputs/Coasters/c15_a15_previews_blue.png` | 6 | 2 Rows × 3 Cols | 482 px | 578.4 px | 10.0% (204.8 px) |
| **Voroni** | `outputs/Coasters/voroni_previews.png` | `outputs/Coasters/voroni_previews_blue.png` | 6 | 2 Rows × 3 Cols | 482 px | 578.4 px | 10.0% (204.8 px) |
| **Explicit** | `outputs/Coasters/explicit_previews.png` | `outputs/Coasters/explicit_previews_blue.png` | 9 | 3 Rows × 3 Cols | 390 px | 468.0 px | 18.0% (369.0 px) |
| **TPMS** | `outputs/Coasters/tpms_previews.png` | `outputs/Coasters/tpms_previews_blue.png` | 8 | 3 Rows [3, 2, 3] | 360 px | 432.0 px | 20.9% (428.0 px) |

---

## 3. Figures & Gallery

### 1. C15/A15
Combines the C15 Frank-Kasper polyhedral network (Row 1: offsets 0.0, 3.125 mm, 6.25 mm) and A15 cubic Kagome slices (Row 2: offsets 0.0, 3.125 mm, 6.25 mm):

![C15/A15 Preview (Black)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews.png)
![C15/A15 Preview (Blue)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews_blue.png)

---

### 2. Voroni
Combines high-density organic Voronoi cells (Row 1: `small_v1`, `small_v2`, `small_v3`) and low-density organic cellular patterns (Row 2: `large_v1`, `large_v2`, `large_v3`):

![Voroni Preview (Black)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews.png)
![Voroni Preview (Blue)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews_blue.png)

---

### 3. Explicit
Combines all 9 2D explicit lattice extrusions into a 3×3 grid:

![Explicit Preview (Black)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews.png)
![Explicit Preview (Blue)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews_blue.png)

---

### 4. TPMS
Continuous minimal surfaces across 8 designs:

![TPMS Preview (Black)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews.png)
![TPMS Preview (Blue)](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews_blue.png)
