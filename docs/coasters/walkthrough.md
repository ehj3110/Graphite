# Coaster Collection & Deliverable Figures — Complete Documentation

This document records the complete design, geometric parameters, 2D cross-section extraction pipeline, and deliverable figure layout specifications for the **Graphite Mathematical & Crystal Lattice Coaster Collection**.

Related coaster documentation in this folder:
- [`2d_lattice_extrusion_documentation.md`](2d_lattice_extrusion_documentation.md) — Technical reference on the 2D Z-slab projection, Shapely buffering, and `trimesh.path.Path2D` extrusion workflow.
- [`coaster_collection_description.md`](coaster_collection_description.md) — Public collection overview and 3D printing recommendations.

---

## 1. Coaster Physical & Geometric Specifications

All coasters are designed for support-free FDM/FFF and resin 3D printing with mathematically flat top and bottom faces:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Outer Dimension ($D_{\text{outer}}$)** | $100.0\,\text{mm}$ ($R_{\text{outer}} = 50.0\,\text{mm}$) | Circle diameter, Hexagon circumdiameter ($R = 50\,\text{mm}$), or Square side length |
| **Inner Lattice Boundary ($R_{\text{inner}}$)** | $46.825\,\text{mm}$ | Leaves a $3.175\,\text{mm}$ ($1/8\,\text{in}$) solid outer rim on framed coasters |
| **Coaster Thickness ($H$)** | $5.0\,\text{mm}$ | Extruded symmetrically across $Z \in [-2.5, +2.5]\,\text{mm}$ |
| **Strut Width ($w$)** | $1.0\,\text{mm}$ | Buffered in 2D using `LineString.buffer(0.5, cap_style=2)` |
| **Variants per Design** | 6 (`3 shapes × Framed/Unframed`) | Shapes: `Circle`, `Hexagon`, `Rectangle` |

---

## 2. Lattice Families & Cross-Section Parameters

### 2.1 Crystal Structures — Frank-Kasper & Laves Phases (`C15/A15`)
3D crystal wireframe edges are generated from fractional atomic coordinates, shifted vertically by a slice offset $Z_{\text{offset}}$, filtered within a proportional slab window $z \in [-z_{\text{limit}}, +z_{\text{limit}}]$ ($z_{\text{limit}} = 0.10 \times L$), projected onto the $XY$ plane, buffered to $1.0\,\text{mm}$ strut width, and unioned with the circular frame:

- **C15 Laves Phase** ($\text{MgCu}_2$-type, 24 atoms/unit cell, cutoff $= 0.45\,L$, unit cell size $L = 50.0\,\text{mm}$, $z_{\text{limit}} = 5.0\,\text{mm}$):
  - `C15_v1`: $Z_{\text{offset}} = 0.0\,\text{mm}$ ($0$)
  - `C15_v2`: $Z_{\text{offset}} = 3.125\,\text{mm}$ ($L/16$)
  - `C15_v3`: $Z_{\text{offset}} = 6.25\,\text{mm}$ ($L/8$)
- **A15 Frank-Kasper Phase** ($\text{Cr}_3\text{Si}$-type, 8 atoms/unit cell, cutoff $= 0.62\,L$, unit cell size $L = 25.0\,\text{mm}$, $z_{\text{limit}} = 2.5\,\text{mm}$):
  - `A15_v1`: $Z_{\text{offset}} = 0.0\,\text{mm}$ ($0$)
  - `A15_v2`: $Z_{\text{offset}} = 3.125\,\text{mm}$ ($L/8$)
  - `A15_v3`: $Z_{\text{offset}} = 6.25\,\text{mm}$ ($L/4$)

### 2.2 Voronoi Tessellations (`Voroni`)
2D Voronoi ridges (`scipy.spatial.Voronoi`) are generated from uniform random seed points inside $[-55, 55]^2\,\text{mm}$ with deterministic seeds (`np.random.seed(s)`) and buffered to $1.0\,\text{mm}$ strut thickness:
- **Dense (`100` seed points):** `small_v1` (seed `42`), `small_v2` (seed `43`), `small_v3` (seed `44`).
- **Sparse (`50` seed points):** `large_v1` (seed `42`), `large_v2` (seed `43`), `large_v3` (seed `44`).
  *(Note: The ultra-sparse 25-seed variant `large_v4` is excluded from the combined deliverable figure.)*

### 2.3 Explicit 2D Strut Lattices (`Explicit`)
Nine planar projections of polyhedral and polytope unit cells tiled across the circular domain:
- **Triangular-Symmetry Family (`5` designs):**
  - `Tri_Tetrahedral` ($\text{pitch} = 15.0\,\text{mm}$, triangular grid with centroid star struts)
  - `Tri_Icosahedral` ($\text{pitch} = 16.0\,\text{mm}$, Kagome-like inset triangle network)
  - `Tri_Kelvin` ($\text{pitch} = 8.0\,\text{mm}$, honeycomb dual with $0.8\,\text{mm}$ strut width)
  - `Tri_Tesseract` ($\text{pitch} = 22.0\,\text{mm}$, triangular grid + centroid + inner triangle + medial ring)
  - `Tri_Rhombic` ($\text{pitch} = 10.0\,\text{mm}$, dual honeycomb + rhombic edge network)
- **Square-Symmetry Family (`4` designs):**
  - `Sq_Grid` ($\text{cell} = 15.0\,\text{mm}$, orthogonal Cartesian grid)
  - `Sq_Icosahedral` ($\text{cell} = 15.0\,\text{mm}$, $45^\circ$ diagonal diamond lattice)
  - `Sq_Kelvin` ($\text{cell} = 18.0\,\text{mm}$, truncated-square & octagonal tiling)
  - `Sq_Tesseract` ($\text{cell} = 25.0\,\text{mm}$, 2D hypercube projection with inner square $[0.25, 0.75]^2$ and corner diagonals)

### 2.4 Triply Periodic Minimal Surfaces (`TPMS`)
Evaluated on a $600 \times 600$ 2D grid over $[-50, 50]^2\,\text{mm}$ at two horizontal slice planes—midplane $Z = 0.0\,\text{mm}$ and near-top $Z = 2.4\,\text{mm}$—using unit cell size $L = 25.0\,\text{mm}$ ($\omega = 2\pi / 25.0$) and sheet half-thickness iso-threshold $t$:
- **`Gyroid`**: $|\sin X \cos Y + \sin Y \cos Z + \sin Z \cos X| \le 0.4$
- **`Diamond`**: $|\sin X \sin Y \sin Z + \sin X \cos Y \cos Z + \cos X \sin Y \cos Z + \cos X \cos Y \sin Z| \le 0.4$
- **`Lidinoid`**: $|0.5(\sin 2X \cos Y \sin Z + \sin 2Y \cos Z \sin X + \sin 2Z \cos X \sin Y) - 0.5(\cos 2X \cos 2Y + \cos 2Y \cos 2Z + \cos 2Z \cos 2X) + 0.15| \le 0.3$
- **`Neovius`**: $|3(\cos X + \cos Y + \cos Z) + 4 \cos X \cos Y \cos Z| \le 0.5$

---

## 3. Deliverable Figure Specifications

All deliverable preview figures are rendered at **$1800 \times 1350\,\text{px}$** ($4:3$ aspect ratio) on a **pure white background (`#FFFFFF`)** with **zero drop shadows**.

### 3.1 Layout & Geometry Table

| Figure Title | Coasters | Grid Layout | Coaster Diameter ($D$) | Center-to-Center Pitch ($1.2 \times D$) | Side Margins | Label Font Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C15/A15`** | 6 | $2 \times 3$ (`[3, 3]`) | $482\,\text{px}$ | $578.4\,\text{px}$ | $10.0\%$ ($180.2\,\text{px}$) | $33\,\text{pt}$ |
| **`Voroni`** | 6 | $2 \times 3$ (`[3, 3]`) | $482\,\text{px}$ | $578.4\,\text{px}$ | $10.0\%$ ($180.2\,\text{px}$) | $33\,\text{pt}$ |
| **`Explicit`** | 9 | $3 \times 3$ (`[3, 3, 3]`) | $390\,\text{px}$ | $468.0\,\text{px}$ | $18.2\%$ ($327.0\,\text{px}$) | $25\,\text{pt}$ |
| **`TPMS`** | 8 | $3 - 2 - 3$ (`[3, 2, 3]`) | $360\,\text{px}$ | $432.0\,\text{px}$ | $21.0\%$ ($378.0\,\text{px}$) | $25\,\text{pt}$ |

- **Main Figure Title**: Centered at $Y = 26\,\text{px}$ in `44 pt` Bold (`Segoe UI Bold` / `Arial Bold`), color `#0f172a`.
- **Coaster Sub-Titles**: Centered directly above each coaster (`10–12 px` vertical clearance above top rim), color `#1e293b`.

### 3.2 Three Visual Style Editions

Each of the 4 combined figures was produced in three visual styles:
1. **Solid Black (`*_previews.png`)**:
   - Solid polygon fill in high-contrast charcoal black `#141416` (`RGB(20, 20, 22)`), hollow white cells (`#FFFFFF`).
2. **Solid Lighter Blue (`*_previews_blue.png`)**:
   - Solid polygon fill in medium-light slate blue `#3b66b0` (`RGB(59, 102, 176)`), hollow white cells (`#FFFFFF`).
3. **Original Blue Outline (`*_previews_outline.png`)**:
   - Unfilled white strut interiors (`facecolor="none"`) with crisp dark navy-blue outlines in `#1e3a8a` (`RGB(30, 58, 138)`).
   - **Linewidth Calibration**:
     - Strut lattices (`C15/A15`, `Voroni`, `Explicit`): `linewidth = 1.8`
     - Implicit surfaces (`TPMS`): `linewidth = 2.5`
