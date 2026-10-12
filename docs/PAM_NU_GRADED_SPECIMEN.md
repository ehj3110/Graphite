# J-4-RING Diamond PAM: Graded NU Wire Diameter Specimens ($750\ \mu\text{m}$ vs $250\ \mu\text{m}$ Extreme Limit)

**Specimen Dimensions:** $50.0 \times 50.0 \times 25.0\text{ mm}$  
**Crystallographic Orientation:** $\{111\}$ aligned to laboratory $+Z$ (100% self-supporting)  
**Parent Network:** Diamond (`dia`) polycatenation (6-ring octahedral clusters, Zhou et al., *Science* 2025)  
**Kinematic Parameters:** Unit cell pitch $a = 14.0\text{ mm}$ | Ring major radius $R = 2.52\text{ mm}$ | Cluster offset $d = 3.36\text{ mm}$  

---

## 1. Visual Comparison: Standard Graded vs Extreme Limit

Both specimens share the full-face $50.0 \times 50.0\text{ mm}$ "NU" mask oriented upright in the $X-Y$ plane with full through-thickness penetration ($40\text{ mm}$ $Z$-extrusion). 

- **$250\ \mu\text{m}$ Extreme Limit Render:** [`outputs/j4_nu_graded_250um_50x50x25.png`](../outputs/j4_nu_graded_250um_50x50x25.png)
- **$750\ \mu\text{m}$ Standard Graded Render:** [`outputs/j4_nu_graded_50x50x25.png`](../outputs/j4_nu_graded_50x50x25.png)

---

## 2. Quantitative Comparison Table

| Metric | $750\ \mu\text{m}$ Graded Specimen | $250\ \mu\text{m}$ Extreme Limit Specimen | Notes / Physical Implication |
|:---|:---:|:---:|:---|
| **Matrix Wire Diameter** | **$1000\ \mu\text{m}$** ($1.0\text{ mm}$, $r=0.500$) | **$1000\ \mu\text{m}$** ($1.0\text{ mm}$, $r=0.500$) | Baseline structural perimeter |
| **NU Ring Wire Diameter** | **$750\ \mu\text{m}$** ($0.75\text{ mm}$, $r=0.375$) | **$250\ \mu\text{m}$** ($0.25\text{ mm}$, $r=0.125$) | **$4\times$ thickness ratio** ($1.0\text{ mm}$ vs $0.25\text{ mm}$) |
| **Ring Aspect Ratio ($R/r$)** | $6.72$ | **$20.16$** | Ultra-slender wire regime |
| **Rings in NU Mask** | $474$ rings ($62.9\%$) | $472$ rings ($62.7\%$) | Forms bold, high-contrast letterforms |
| **Rings in Matrix** | $279$ rings ($37.1\%$) | $281$ rings ($37.3\%$) | Surrounding lattice envelope |
| **Total Specimen Rings** | **$753$ rings** | **$753$ rings** | 100% catenated, $0$ missing rings |
| **Solid Material Volume** | $6,785.2\text{ mm}^3$ | **$3,861.3\text{ mm}^3$** | **$43.1\%$ mass reduction** |
| **Solid Volume Fraction ($\phi$)** | $10.86\%$ | **$6.18\%$** | Deep ultralightweight PAM regime |
| **Global Min Clearance** | $+0.1879\text{ mm}$ ($188\ \mu\text{m}$) | **$+0.1879\text{ mm}$** ($188\ \mu\text{m}$) | Governed by matrix-matrix contacts |
| **Local NU Clearance** | $+0.3129\text{ mm}$ ($313\ \mu\text{m}$) | **$+0.5629\text{ mm}$** ($563\ \mu\text{m}$) | Greatly expanded free kinematic play |
| **Clearance Violations** | **$0$** | **$0$** | 100% collision-free |
| **Mesh Watertightness** | **True** (2-manifold) | **True** (2-manifold) | Validated via `trimesh.is_watertight` |
| **Mesh Geometry** | $578,304$ v / $1,156,608$ f | $578,304$ v / $1,156,608$ f | Constant topological fidelity |

---

## 3. Deliverables in `outputs/`

### A. $250\ \mu\text{m}$ Extreme Limit Deliverables
- **Combined Watertight STL:** [`outputs/j4_nu_graded_250um_50x50x25.stl`](../outputs/j4_nu_graded_250um_50x50x25.stl) ($57.8\text{ MB}$)
- **NU Rings Only ($250\ \mu\text{m}$):** [`outputs/j4_nu_graded_250um_50x50x25_nu_only.stl`](../outputs/j4_nu_graded_250um_50x50x25_nu_only.stl) ($36.2\text{ MB}$, $472$ ultra-slender rings)
- **Matrix Rings Only ($1.0\text{ mm}$):** [`outputs/j4_nu_graded_250um_50x50x25_matrix_only.stl`](../outputs/j4_nu_graded_250um_50x50x25_matrix_only.stl) ($21.6\text{ MB}$, $281$ structural rings)
- **Instanced 3MF Package:** [`outputs/j4_nu_graded_250um_50x50x25.3mf`](../outputs/j4_nu_graded_250um_50x50x25.3mf) ($174.8\text{ KB}$)
- **Studio 3-View Render:** [`outputs/j4_nu_graded_250um_50x50x25.png`](../outputs/j4_nu_graded_250um_50x50x25.png)
- **Diagnostic JSON Report:** [`outputs/j4_nu_graded_250um_50x50x25_report.json`](../outputs/j4_nu_graded_250um_50x50x25_report.json)

### B. $750\ \mu\text{m}$ Standard Graded Deliverables
- **Combined Watertight STL:** [`outputs/j4_nu_graded_50x50x25.stl`](../outputs/j4_nu_graded_50x50x25.stl) ($57.8\text{ MB}$)
- **NU Rings Only ($750\ \mu\text{m}$):** [`outputs/j4_nu_graded_50x50x25_nu_only.stl`](../outputs/j4_nu_graded_50x50x25_nu_only.stl) ($36.4\text{ MB}$)
- **Matrix Rings Only ($1.0\text{ mm}$):** [`outputs/j4_nu_graded_50x50x25_matrix_only.stl`](../outputs/j4_nu_graded_50x50x25_matrix_only.stl) ($21.4\text{ MB}$)
- **Instanced 3MF Package:** [`outputs/j4_nu_graded_50x50x25.3mf`](../outputs/j4_nu_graded_50x50x25.3mf) ($174.3\text{ KB}$)
- **Studio 3-View Render:** [`outputs/j4_nu_graded_50x50x25.png`](../outputs/j4_nu_graded_50x50x25.png)
- **Diagnostic JSON Report:** [`outputs/j4_nu_graded_50x50x25_report.json`](../outputs/j4_nu_graded_50x50x25_report.json)

---

## 4. Inspection in Blender

To inspect the extreme limit specimen in Blender:
1. Open Blender.
2. Select **File > Import > Stl (.stl)** and select **both** sub-meshes simultaneously:
   - [`outputs/j4_nu_graded_250um_50x50x25_nu_only.stl`](../outputs/j4_nu_graded_250um_50x50x25_nu_only.stl)
   - [`outputs/j4_nu_graded_250um_50x50x25_matrix_only.stl`](../outputs/j4_nu_graded_250um_50x50x25_matrix_only.stl)
3. Press **Numpad 7** to switch to the orthographic **Top View (XY plane)**.
4. The $250\ \mu\text{m}$ wire rings form the letters "N" and "U" across the entire $50 \times 50\text{ mm}$ face with high visual contrast against the $1.0\text{ mm}$ matrix rings.
