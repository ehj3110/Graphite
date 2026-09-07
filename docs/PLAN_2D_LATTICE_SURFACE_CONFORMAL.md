# Architectural Plan: Unified "2D Lattice & Surface Conformal" Mode in Graphite Explicit

## 1. Executive Summary & Context

Graphite has developed two highly successful but previously isolated 2D-to-3D workflows:
1. **Coaster Engine:** 2D crystal/chiral/planar lattices extruded along $+Z$ into flat plates with sharp corners and flat caps (`docs/coasters/2d_lattice_extrusion_documentation.md`).
2. **Cylindrical Napkin Ring Engine:** 2D periodic lattices wrapped onto cylindrical annular shells via 3D radial prism extrusion (`build_prisms_from_2d_segments`) and fused with solid CAD collar rims.

Both workflows share a critical engineering advantage: **they completely avoid the mathematical pitfalls of 3D volumetric tetrahedralization and freeform conformal meshing**. By keeping the lattice math in 2D and projecting onto isometric surfaces, they preserve **exact unit cell geometry** (true circular nodes, exact tangent departure, zero angle distortion) and produce lightweight, 100% watertight STLs in seconds.

This plan details how to promote these capabilities into a first-class **"2D Lattice" engine** in `graphite.explicit`, resolving the "surface selection" dilemma and making it intuitive and production-ready.

---

## 2. The "Surface Dilemma": How to Make it Usable Without Rabbit Holes

### 2.1 The Problem with Arbitrary 3D Surfaces
Attempting to map a 2D metamaterial onto an arbitrary organic or double-curved 3D surface ($K \ne 0$) triggers severe mathematical issues:
- **Metric Distortion:** Non-zero Gaussian curvature forces local area and angle distortions. Circular nodes become stretched ellipses, and straight tangent ligaments lose their tangency.
- **Singularities & Seams:** Unwrapping closed organic shapes requires cut seams (branch cuts) where ligaments will shear or fail to align.
- **Lost Metamaterial Physics:** Auxetic Poisson's ratio ($\nu < 0$) relies on symmetric rotating nodes; once distorted, the negative-Poisson behavior degrades into shear collapse.

### 2.2 The Solution: The "Developable & Extruded Manifold" Scope
Instead of pursuing generic freeform mapping, we restrict the surface engine to **surfaces with zero Gaussian curvature ($K = 0$) and extruded profiles**:

```
                       ┌───────────────────────────────────────────────┐
                       │           2D Metamaterial Engine              │
                       │ (Tetra-chiral, Tri-chiral, Kagome, Voronoi)   │
                       └───────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────▼───────────────────────┐
                       │      Surface Conformal Projection             │
                       └───────┬───────────────────────────────┬───────┘
                               │                               │
                ┌──────────────▼──────────────┐ ┌──────────────▼──────────────┐
                │        MODE A: PLANAR       │ │      MODE B: CYLINDRICAL    │
                │ (Extruded Plate / Coasters) │ │ (Annular Sleeve / Rings)    │
                └──────────────┬──────────────┘ └──────────────┬──────────────┘
                               │                               │
                • Orthogonal Z extrusion        • Radial prism wedge extrusion
                • Flat caps, crisp profiles     • Flush R_in and R_out walls
                • Zero metric distortion        • Exact periodic seam closure
```

1. **Planar Surfaces ($X, Y \to Z$):** Flat plates, baffles, shear panels, energy-absorbing pads, coasters.
2. **Cylindrical Surfaces ($\theta, Y \to R$):** Rings, sleeves, bushings, medical stents, pipe sleeves, compliant collars.
3. **Conical Frustums ($\theta, s \to R$):** Tapered nozzles, adapters, funnels.

**Why this is NOT niche:** 
Over 90% of real-world mechanical metamaterial applications (stents, acoustic meta-plates, shock-absorbing sleeves, crash tubes, wearable braces) are either flat plates or cylindrical/conical tubes. By mastering these two surfaces completely, the engine delivers rock-solid, production-grade parts without edge cases.

---

## 3. Package Architecture in `graphite.explicit`

We propose housing this under a new submodule: `graphite/explicit/surface_2d/`:

```
graphite/explicit/surface_2d/
├── __init__.py               # Public API exports
├── unit_cells.py             # Pure unit-cell generators (Tetra-chiral, Tri-chiral, Kagome, etc.)
├── tessellation.py           # 2D domain tiling, pitch calculation, periodic closure
├── mappers/
│   ├── base.py               # Abstract BaseSurfaceMapper
│   ├── planar.py             # Shapely 2D buffer + Path2D extrusion (Coaster path)
│   └── cylindrical.py        # Radial prism wedge extrusion (Napkin Ring path)
└── cad_rims.py               # Automated collar rim extraction and boolean fusion
```

### 3.1 Step 1: Unit Cell Generation (`unit_cells.py`)
Generates single-cell topology graphs parameterized by:
- $r$: circular node radius
- $L$: connection ligament length
- $t$: strut thickness
- `topology`: `"tetra"` (square), `"tri"` (triangular/hexagonal), `"anti_tetra"`, `"anti_tri"`.

### 3.2 Step 2: Domain Tessellation (`tessellation.py`)
Tiles the unit cell over a 2D bounding area $[0, U] \times [0, V]$:
- For **Cylinders:** $U = C_{\text{mid}} = 2\pi R_{\text{mid}}$, $V = H$. Computes pitch $D = C_{\text{mid}} / N$ and rotates the square basis by $-\alpha = -\arctan(2r/L)$ to ensure **zero-defect seam closure**.
- For **Plates:** $U = \text{Width}$, $V = \text{Height}$. Clips or pads boundaries with optional perimeter stabilization frames.

### 3.3 Step 3: Surface Solidification Mappers (`mappers/`)
Each mapper takes the deduplicated 2D line segments `(p1, p2)` and solidifies them into a `manifold3d.Manifold`:

* **`PlanarMapper`:**
  - Buffers segments into 2D polygonal ribbons with square caps.
  - Performs 2D `unary_union`.
  - Extrudes orthogonally along $+Z$ by thickness $T$.
  - Generates perfectly flat top and bottom faces.
* **`CylindricalMapper`:**
  - Subdivides segments into arc steps ($\Delta u \le 1\text{ mm}$).
  - Extrudes each chord radially into an 8-vertex hexahedral wedge spanning $r \in [R_{\text{in}}, R_{\text{out}}]$.
  - Intersects with a smooth bounding sleeve cylinder and unions with boundary stabilization rings.
  - Produces perfectly concentric inner and outer cylindrical faces.

---

## 4. Making it Usable: How Does the User Specify the "Surface"?

To make this workflow effortless for users and coworkers, we define three simple ways to specify the surface:

### Method A: Direct Parametric Primitive (Zero CAD input required)
The user provides physical dimensions directly:
```python
from graphite.explicit.surface_2d import generate_surface_lattice

# 1. Cylinder
mesh = generate_surface_lattice(
    surface="cylinder",
    r_in=19.05,
    r_out=22.05,
    height=38.1,
    cell_type="tetra_chiral",
    n_circumferential=10,
    r_node=2.2,
    strut_w=2.0,
)

# 2. Flat Plate
plate_mesh = generate_surface_lattice(
    surface="plate",
    width=100.0,
    height=100.0,
    thickness=4.0,
    cell_type="tri_chiral",
    pitch=12.0,
    r_node=2.0,
    strut_w=1.5,
)
```

### Method B: "Auto-Fit to STL" (Extract Surface from CAD Fixture)
If the user uploads an STL part (e.g. `BaseRing_1to2.STL` or a custom hollow tube):
1. **Bounding Cylinder Extraction:** The engine analyzes the mesh bounds and axial symmetry (typically $Y$ or $Z$ axis).
2. **Automatic Parameter Deduction:**
   - Detects inner bore $R_{\text{in}}$ and outer diameter $R_{\text{out}}$.
   - Detects height $H$.
3. **Window Carve & Collar Rim Fusion (`cad_rims.py`):**
   - Carves a middle cutout window for the lattice.
   - Retains the solid top and bottom collar rims.
   - Unions the generated chiral lattice with the rims in a single automated step.

### Method C: UI Integration (Streamlit)
Add a dedicated **"2D Metamaterial Surface Mode"** selector:
1. **Surface Type:** `[Cylinder / Sleeve]` or `[Flat Plate / Coaster]`.
2. **Input Source:**
   - `Manual Dimensions` (Sliders for $R_{\text{in}}, R_{\text{out}}, H$ or Width, Height, Depth).
   - `CAD Base Part` (Upload STL $\to$ auto-detects diameters and height).
3. **Cell Selection:**
   - `Tetra-Chiral (Square)` or `Tri-Chiral (Hexagonal)`.
4. **Parametric Sliders:**
   - Number of cells $N$ (or Pitch $D$).
   - Central node circle radius $r$ (with dynamic validation preventing $2r \ge D$).
   - Strut thickness $w$.
5. **Real-time 2D Diagnostic Preview + 1-click 3D STL Export.**

---

## 5. Implementation Roadmap

### Phase 1: Engine Consolidation
- Create `graphite/explicit/surface_2d/`.
- Move `build_prisms_from_2d_segments` and `dedupe_segments` from `scripts/cylinders/generate_cylinder_lattice.py` into `graphite/explicit/surface_2d/mappers/cylindrical.py`.
- Move 2D planar polygon extrusion logic from `scripts/archive/` into `graphite/explicit/surface_2d/mappers/planar.py`.
- Move chiral unit cell definitions from `graphite/explicit/chiral_cell.py` into `graphite/explicit/surface_2d/unit_cells.py`.

### Phase 2: Unified Interface & Test Suite
- Expose top-level helper functions in `graphite/explicit/__init__.py`.
- Write unit tests in `tests/test_surface_2d.py`:
  - Test planar extrusion (watertightness, cap flatness, face counts).
  - Test cylindrical prism extrusion (watertightness, seam closure, manifold status).
  - Test auto-parameter calculation (pitch $D$, ligament $L$, rotation $\alpha$).

### Phase 3: Automated CAD Collar Rim Detection
- Build `cad_rims.py` to inspect input cylindrical STLs, calculate bounding cylinders, carve the middle lattice zone, and fuse collar rims automatically.

### Phase 4: Streamlit UI Integration
- Add the 2D Surface Mode tab in `app.py`.
