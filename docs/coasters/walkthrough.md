# Coaster Collection — Combined Deliverable Figures

This document details the unified 4-figure presentation suite for the complete 3D-printable coaster collection, formatted to match the reference high-contrast presentation canvas (1024:765 aspect ratio, rendered at 2048 × 1530 px high resolution).

---

## 1. Design & Spacing Standard

All 4 deliverable figures adhere to strict mathematical placement and typographic guidelines:
1. **Center-to-Center Spacing**: $dx = dy = 1.25 \times D$ (pitch equals exactly 1.25 times the coaster diameter).
2. **Coaster Titles**: Scaled ~3× as large (~46 pt to 58 pt bold) and placed directly above each coaster.
3. **High Contrast**: Deep solid carbon (`#141416`) on a clean studio neutral background (`#F8F9FA`).
4. **Product Depth**: Soft Gaussian drop shadows underneath each coaster for elevated presentation.
5. **Combined Groupings**:
   - **`C15/A15`**: 6 coasters (3 C15 on row 1, 3 A15 on row 2).
   - **`Voroni`**: 6 coasters (3 Dense `small_v1`–`v3` on row 1, 3 Sparse `large_v1`–`v3` on row 2; dropped `large_v4` as the least dense).
   - **`Explicit`**: 9 coasters in a 3×3 grid (5 Triangle + 4 Square explicit struts).
   - **`TPMS`**: 8 coasters in a balanced 3-row layout (Gyroid, Diamond, Lidinoid at $Z=0$ and $Z=2.4\text{mm}$, with Neovius $Z=0$ and $Z=2.4\text{mm}$ centered on row 2; Split-P dropped).

---

## 2. Deliverables Summary

| Figure Title | Output File | Coasters Included | Grid Format | Diameter ($D$) | Pitch ($1.25 D$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **C15/A15** | `outputs/Coasters/c15_a15_previews.png` | 6 | 2 Rows × 3 Cols | 540 px | 675 px |
| **Voroni** | `outputs/Coasters/voroni_previews.png` | 6 | 2 Rows × 3 Cols | 540 px | 675 px |
| **Explicit** | `outputs/Coasters/explicit_previews.png` | 9 | 3 Rows × 3 Cols | 360 px | 450 px |
| **TPMS** | `outputs/Coasters/tpms_previews.png` | 8 | 3 Rows [3, 2, 3] | 360 px | 450 px |

---

## 3. Figures & Gallery

### 1. C15/A15
Combines the C15 Frank-Kasper polyhedral network (Row 1: offsets 0.0, 0.0625, 0.125) and A15 cubic Kagome slices (Row 2: offsets 0.0, 0.125, 0.25):

![C15/A15 Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/c15_a15_previews.png)

---

### 2. Voroni
Combines high-density organic Voronoi cells (Row 1: `small_v1`, `small_v2`, `small_v3`) and low-density organic cellular patterns (Row 2: `large_v1`, `large_v2`, `large_v3`). The least dense sparse pattern (`large_v4`, solid area 1867 mm²) was excluded to maintain an exact, balanced 2×3 grid:

![Voroni Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/voroni_previews.png)

---

### 3. Explicit
Combines all 9 2D explicit lattice extrusions into a 3×3 grid. Features 5 triangular tessellations (`Tri_Tetrahedral`, `Tri_Icosahedral`, `Tri_Kelvin`, `Tri_Tesseract`, `Tri_Rhombic`) and 4 square tessellations (`Sq_Grid`, `Sq_Icosahedral`, `Sq_Kelvin`, `Sq_Tesseract`):

![Explicit Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/explicit_previews.png)

---

### 4. TPMS
Continuous minimal surfaces across 8 designs. Row 1 features zero-level sets ($Z=0$), Row 2 features centered Neovius transitions ($Z=0$ and $Z=2.4\text{mm}$), and Row 3 features mid-height slices ($Z=2.4\text{mm}$):

![TPMS Preview](file:///C:/Users/ehunt/OneDrive/Documents/Python%20Scripts/Graphite/outputs/Coasters/tpms_previews.png)
