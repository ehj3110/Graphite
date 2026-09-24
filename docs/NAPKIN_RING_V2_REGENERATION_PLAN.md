# Napkin Ring V2: Regeneration Architecture & Cleanup Plan

## 1. New Base CAD Models & Geometric Specifications

The user has updated and placed the new base CAD files in `test_parts/`. All models feature **3.0 mm wall thickness** (updated from 4.0 mm) and **chamfered bottom rims** for flawless 3D printability.

### 1.1 Verified File Metrics

| File Name | Total Height | Lattice Height ($H$) | Collar Height (Each) | Inner Dia ($D_{\text{in}}$) | Outer Dia ($D_{\text{out}}$) | Wall Thickness | Ratio ($H : D_{\text{in}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BaseRing_1to2.STL`** | 32.35 mm | **19.05 mm** (0.75") | 6.65 mm | 38.10 mm (1.50") | 50.80 mm (2.00") | **3.00 mm** | **1 : 2** (Shortest) |
| **`BaseRing_2to3.STL`** | 38.70 mm | **25.40 mm** (1.00") | 6.65 mm | 38.10 mm (1.50") | 50.80 mm (2.00") | **3.00 mm** | **2 : 3** (Mid / Classic) |
| **`BaseRing_1to1.STL`** | 51.40 mm | **38.10 mm** (1.50") | 6.65 mm | 38.10 mm (1.50") | 50.80 mm (2.00") | **3.00 mm** | **1 : 1** (Maximum Height) |

### 1.2 Master Lattice Section & Scaling Rules
- **Master File**: `test_parts/LatticeSection_1to2.STL`
  - Dimensions: $H = 19.05\text{ mm}$, $R_{\text{in}} = 19.05\text{ mm}$, $R_{\text{out}} = 22.05\text{ mm}$, $\text{Wall} = 3.00\text{ mm}$.
- **Axial Scaling for other aspect ratios**:
  - For **1:2 Ratio** (`1to2`): Base scale $S_y = 1.000$ ($H = 19.05\text{ mm}$).
  - For **2:3 Ratio** (`2to3`): Axial scale $S_y = 25.40 / 19.05 = \mathbf{4 / 3 \approx 1.3333}$ ($H = 25.40\text{ mm}$).
  - For **1:1 Ratio** (`1to1`): Axial scale $S_y = 38.10 / 19.05 = \mathbf{2.0000}$ ($H = 38.10\text{ mm}$).
- **Radial Dimensions**: Strictly invariant ($R_{\text{in}} = 19.05\text{ mm}, R_{\text{out}} = 22.05\text{ mm}$).

---

## 2. Printability Improvements in V2 CAD

1. **Chamfered Base Ring Interface:**
   - The bottom collar rim transitions at a clean 45° self-supporting chamfer.
   - Sits on the build plate with a flat $3.35\text{ mm}$ wide annular contact ring ($r \in [22.05, 25.40]$ mm).
   - Eliminates overhang sagging, elephant's foot distortion, and need for support material.
2. **Updated Wall Thickness (3.0 mm):**
   - Provides optimal strength-to-weight ratio while enabling lighter, more open minimal surface pores and delicate explicit struts.

---

## 3. Cleanup Status

- **Workspace Root**: Completely cleaned up. 28 temporary test STLs, PLY/OBJ meshes, and test PNG renders have been archived to `outputs/archive/root_tests/`.
- **Legacy Deliverables**: The V1 deliverables have been archived to `outputs/archive/deliverables_v1/`.
- **Legacy Split-P Reviews**: Archived to `outputs/archive/split_p_review_v1/`.
- **Readiness**: The workspace is pristine and ready for generation.
