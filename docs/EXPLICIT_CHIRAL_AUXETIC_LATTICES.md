# Explicit Chiral & Auxetic Metamaterial Lattices

## 1. Overview & Mechanical Principles

This document serves as the architectural specification and user guide for the **chiral and auxetic mechanical metamaterial** pipelines in Graphite Explicit (`graphite.explicit`).

Mechanical metamaterials with a **negative Poisson's ratio** ($\nu < 0$), known as **auxetics**, expand laterally when stretched and contract laterally when compressed. Unlike conventional positive-Poisson materials that thin under tension, auxetic structures offer:
- **Synclastic Curvature:** Naturally forming dome-like double curvature without saddle distortion when wrapped onto curved 3D geometries.
- **Enhanced Energy Absorption & Fracture Toughness:** Dissipating impact energy through collective cell rotation or hinging.
- **Superior Shear Modulus:** Resisting indentation and shear deformation.

Graphite Explicit provides two major families of 2D/3D auxetic structures:
1. **Chiral Metamaterials:** Central circular nodes with tangential straight ligaments that deform through coupled ligament bending and node rotation (Prall & Lakes 1997; Spadoni et al. 2006; Alderson et al. 2010).
2. **Re-Entrant Honeycomb Metamaterials:** Bow-tie / hourglass unit cells featuring inward-angled ribs and optional directional reinforcement ribs (Chen et al. 2020).

Both families are fully integrated into cylindrical mapping pipelines to produce 100% watertight, print-ready 3D mechanical rings (such as napkin rings, medical stents, or compliant sleeves) with CAD collar rim fusion.

---

## 2. Metamaterial Topologies Implemented

### 2.1 Tetra-Chiral ("Square" Basis)
- **Geometry:** 4 tangential straight ligaments connected to each circular node at 90° intervals on a square lattice.
- **Lattice Pitch:** $D = \sqrt{L^2 + 4r^2}$ where $L$ is ligament length and $r$ is circular node radius.
- **Rotation Angle:** $\alpha = \arctan\left(\frac{2r}{L}\right)$.
- **Kinematics:** Symmetrical in-plane contraction ($\nu \approx -1$). When subjected to tensile strain, the central nodes rotate, unwinding the flexed ligaments and causing uniform lateral expansion.

### 2.2 Tri-Chiral ("Triangular / Hexagonal" Basis)
- **Geometry:** 3 tangential straight ligaments connected to each circular node at 120° intervals on a triangular lattice.
- **Basis Vectors:** $\mathbf{a}_1 = [D, 0]$, $\mathbf{a}_2 = \left[\frac{D}{2}, \frac{D\sqrt{3}}{2}\right]$.
- **Offset Angle:** $\phi = \arcsin\left(\frac{2r}{D}\right)$.
- **Kinematics:** Hexagonal symmetry yielding isotropic in-plane Poisson's ratio and enhanced shear compliance.

### 2.3 Anti-Chiral Variants
- **Anti-Tetra-Chiral:** Ligaments connect adjacent nodes on the same side of the center line, creating alternating node rotations.
- **Anti-Tri-Chiral:** Hexagonal arrangement where ligament tangents preserve mirror symmetry rather than rotational symmetry.

### 2.4 Chen et al. (2020) Re-Entrant Auxetic Honeycombs
- **Base Cell:** Bow-tie / hourglass geometry with inward-slanted ribs connected at the waist by horizontal bridging struts.
- **Type-A Variant:** Base cell reinforced with a horizontal transverse rib across the waist for enhanced axial stiffness.
- **Type-B Variant:** Base cell reinforced with a vertical central longitudinal rib for superior buckling resistance.

---

## 3. Cylindrical Conformal Mapping & Seam-Free Periodicity

A common failure mode when rolling planar chiral lattices onto cylindrical surfaces is **helical seam shearing**:
In an unrotated tetra-chiral lattice, connecting circles across opposite tangents produces a $+2r$ axial shift per column. Over $N$ columns, this creates an open helix rather than a closed cylinder.

### Rotated Basis Solution
Graphite Explicit solves this analytically:
1. The cylinder mid-surface circumference is $C_{\text{mid}} = 2\pi R_{\text{mid}}$, where $R_{\text{mid}} = \frac{R_{\text{in}} + R_{\text{out}}}{2}$.
2. The circumferential pitch is constrained to $D = \frac{C_{\text{mid}}}{N}$, where $N \in \mathbb{Z}^+$ is the number of circumferential unit cells.
3. For a chosen node radius $r < D/2$, the ligament length is computed as:
   $$L = \sqrt{D^2 - 4r^2}$$
4. The lattice basis is rotated by $-\alpha$, where:
   $$\alpha = \arctan\left(\frac{2r}{L}\right)$$
5. This rotation forces the column translation vector $\mathbf{a}_1 = [D, 0]^T$ to lie strictly along the circumferential coordinate $u \in [0, C_{\text{mid}}]$.
6. **Result:** Nodes at $u = 0$ coincide with nodes at $u = C_{\text{mid}}$ with machine precision, producing **zero helical distortion and 100% periodic seam closure**.

---

## 4. Core Python API (`graphite.explicit`)

The chiral metamaterial generators are exposed in `graphite.explicit`:

```python
from graphite.explicit import generate_tetrachiral_cell, generate_trichiral_cell

# 1. Tetra-Chiral (Square) Unit Cell
nodes, struts, meta = generate_tetrachiral_cell(
    L=10.0,                  # Ligament length in mm
    r=2.0,                   # Central circle radius in mm
    t=1.0,                   # Nominal strut thickness in mm
    n_circle_segments=16,    # Polygon chord discretization for circular node
    chiral=True,             # True = chiral, False = anti-chiral
)

print(f"Tetra-Chiral Pitch D: {meta['D_pitch']:.2f} mm")
print(f"Rotation angle alpha: {meta['alpha_deg']:.2f} deg")
print(f"Nodes shape: {nodes.shape}, Struts shape: {struts.shape}")

# 2. Tri-Chiral (Triangular) Unit Cell
nodes_tri, struts_tri, meta_tri = generate_trichiral_cell(
    L=10.0,
    r=2.0,
    t=1.0,
    n_circle_segments=16,
    chiral=True,
)

print(f"Tri-Chiral Pitch D: {meta_tri['D_pitch']:.2f} mm")
print(f"Tangent angle phi: {meta_tri['phi_deg']:.2f} deg")
```

### Return Values
- `nodes`: `(N, 2)` NumPy array of 2D node coordinates.
- `struts`: `(S, 2)` NumPy array of integer index pairs representing strut connectivity.
- `metadata`: Dictionary containing computed pitch $D$, angles ($\alpha$ or $\phi$), and configuration details.

---

## 5. 3D Piecewise Prism Extrusion Pipeline

To transform 2D line segments $(u_1, y_1) \to (u_2, y_2)$ into solid, print-ready 3D cylindrical struts:
1. **Discretization:** Long struts and circular chords are partitioned into chord steps no larger than $\Delta u_{\text{max}} = 1.0\text{ mm}$ (`MAX_CHORD_STEP`).
2. **Radial Prism Extrusion (`build_prisms_from_2d_segments`):**
   Each 2D segment is mapped to cylindrical coordinates:
   $$\theta = \frac{u}{R_{\text{mid}}}, \quad r \in [R_{\text{in}}, R_{\text{out}}], \quad y = y$$
   The strut width $w$ defines tangential offsets, creating an 8-vertex 3D hexahedral wedge (prism).
3. **Boundary Rings & Solid Trim:**
   - Upper and lower boundary stabilization rings are boolean-added to prevent free floating cantilever ligaments at the cylinder ends.
   - A smooth cylindrical sleeve trim $(R_{\text{in}} \le r \le R_{\text{out}}, y_{\text{base}} \le y \le y_{\text{base}} + H)$ is intersected (`^`) to ensure perfectly cylindrical inner/outer faces.
4. **CAD Collar Rim Fusion:**
   - Existing base ring CAD STLs (`test_parts/NapkingRing_BaseRing_V1.STL` or `test_parts/BaseRing_1to2.STL`) have their middle lattice section carved out using `m3d.Manifold` subtraction.
   - The remaining solid top and bottom collar rims are unioned (`+`) with the generated chiral lattice sleeve, forming a single watertight part.

---

## 6. Command-Line Usage (Headless Execution)

### 6.1 Chiral Pipeline (`generate_cylinder_chiral.py`)
```bash
# Run all phases (2D diagnostics + standalone sleeves + full assembled napkin rings)
python scripts/cylinders/generate_cylinder_chiral.py --phase 0

# Run specific phases:
python scripts/cylinders/generate_cylinder_chiral.py --phase 1  # 2D unrolled comparison plot
python scripts/cylinders/generate_cylinder_chiral.py --phase 2  # 3D standalone sleeves
python scripts/cylinders/generate_cylinder_chiral.py --phase 3  # Full napkin rings with CAD rims
```

### 6.2 Re-Entrant Auxetic Pipeline (`generate_cylinder_auxetic.py`)
```bash
# Run all phases for Chen et al. bow-tie lattices
python scripts/cylinders/generate_cylinder_auxetic.py --phase 0
```

---

## 7. Deliverable File Catalog

### Core Python Modules & Tests
| File | Role |
|---|---|
| `graphite/explicit/chiral_cell.py` | 2D unit cell generators for tetra-chiral, tri-chiral, and anti-chiral metamaterials. |
| `graphite/explicit/__init__.py` | Package-level exports for `generate_tetrachiral_cell` and `generate_trichiral_cell`. |
| `tests/test_chiral_cell.py` | Unit tests verifying node/strut counts, ligament lengths, and pitch math (`pytest` clean). |
| `scripts/cylinders/generate_cylinder_chiral.py` | Complete 2D/3D generation pipeline for chiral cylinder lattices and napkin rings. |
| `scripts/cylinders/generate_cylinder_auxetic.py` | Complete generation pipeline for Chen et al. (2020) re-entrant auxetic honeycombs. |

### Generated 3D Models (`outputs/cylinders/`)
| STL File | Topology | Genus | Manifold Status | Bounding Box ($X \times Y \times Z$ mm) |
|---|---|---|---|---|
| `LatticeSection_1p5inch_TetraChiral.stl` | Tetra-Chiral Sleeve | 61 | **Error.NoError** | $46.1 \times 38.1 \times 46.1$ |
| `LatticeSection_1p5inch_TriChiral.stl` | Tri-Chiral Sleeve | 111 | **Error.NoError** | $46.1 \times 38.1 \times 46.1$ |
| `NapkinRing_1p5inch_TetraChiral.stl` | V1 Classic Napkin Ring | 59 | **Error.NoError** | $50.8 \times 51.4 \times 50.8$ |
| `NapkinRing_1p5inch_TriChiral.stl` | V1 Classic Napkin Ring | 109 | **Error.NoError** | $50.8 \times 51.4 \times 50.8$ |
| `NapkinRing_1to2_TetraChiral.stl` | V2 1:2 Chamfered Ring | 41 | **Error.NoError** | $50.8 \times 32.3 \times 50.8$ |
| `NapkinRing_1to2_TriChiral.stl` | V2 1:2 Chamfered Ring | 71 | **Error.NoError** | $50.8 \times 32.3 \times 50.8$ |

> [!NOTE]
> All generated STLs are verified watertight manifolds with zero non-manifold edges, zero self-intersections, and positive non-zero volumes.
