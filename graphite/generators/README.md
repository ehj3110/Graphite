# `graphite.generators` — High-Compliance & Kinematic Metamaterials

## Owns

Specialized continuous and discrete metamaterial generators:
1. **Transversely Isotropic Hexagonal Pentamode**: Layered hexagonal prism frameworks with $Z = 4$ mid-plane hubs, basal flexural rings, and biconical out-of-plane pillars ($G_{xy}\to 0$, $E_z \gg G_{xy}$).
2. **Continuous Pentamode (Meta-Fluid) Networks**: 4-coordinated diamond cubic lattices ($Z = 4$) with biconical strut envelopes and hierarchical sub-trusses (tetrahedral and C15-inspired).
3. **Interlocking Contact Assemblies**: Discrete, non-bonded kinematic unit cell chains (3D re-entrant auxetic bowties, alternating hook arrays, and chainmail textiles) that drape with near-zero in-plane shear resistance and lock under normal displacement.

---

## Status

**Production** generator subsystem.

---

## Module Map

| Module | Core Functions / Classes | Description |
| :--- | :--- | :--- |
| `hexagonal_pentamode.py` | `generate_hexagonal_pentamode_lattice`, `hexagonal_pentamode_cell` | Transversely isotropic hex pentamode: basal rings + biconical verticals, $Z=4$ hubs |
| `pentamode.py` | `generate_pentamode_lattice`, `LatticeGraph`, `ImplicitField` | Milton-Cherkaev diamond cubic meta-fluid networks, bicones, hierarchical sub-trusses, and SDF |
| `interlocking.py` | `generate_interlocking_auxetic_sheet`, `verify_interlocking_clearance`, `combine_interlocking_meshes` | Discrete non-welded kinematic unit cell assemblies (re-entrant auxetics, space-fabric, chainmail) |

---

## Kinematic & Physical Foundations

### 0. Hexagonal Pentamode (`hexagonal_pentamode.py`)
- **Macro Topology**: ABAB-stacked planar hexagonal honeycombs (beam length $a$, pitch $c$) with mid-plane hubs.
- **AB stagger (translation, not rotation):** Layers A and B share identical strut orientations. Layer B is shifted by $\Delta\mathbf{r}=(a\sqrt{3}/2,\, a/2,\, c)$ into the void pockets of Layer A — **no** in-plane $30^\circ$ phase spin.
- **Coordination:** Hubs at $z=c/2$ connect to 2 A + 2 B nodes ($Z=4$). Honeycomb vertices have 3 in-plane struts + 1 inclined bicone ($Z=4$). No crossing verticals.
- **API:** `generate_transverse_hexagonal_pentamode(nx, ny, nz, a, c)` and `generate_hexagonal_pentamode_lattice(..., nx=, ny=, nz=)`.
- **Outputs**: `"graph"` / `"implicit_sdf"` / `"mesh"` via shared `LatticeGraph` / `ImplicitField`.

```python
from graphite.generators import generate_hexagonal_pentamode_lattice

mesh = generate_hexagonal_pentamode_lattice(
    bounds=((0, 0, 0), (30, 30, 20)),
    a=5.0,
    c=7.5,
    r_min=0.28,
    r_max=0.85,
    r_basal=0.22,
    output_format="mesh",
)
```

### 1. Pentamode Meta-Fluids (`pentamode.py`)
- **Macro Symmetry**: Diamond cubic crystal structure ($Fd\bar{3}m$, 8 basis nodes per conventional cell).
- **Coordination**: Every internal node has tetrahedral coordination degree $Z = 4$ and bond angle $\arccos(-1/3) \approx 109.47^\circ$.
- **Biconical Strut Profile**:
  $$r(t) = r_{\min} + 4(r_{\max} - r_{\min}) \cdot t(1 - t), \quad t \in [0, 1]$$
  - Neck radius $r_{\min}$ controls compliant hinge stiffness ($G \propto (r_{\min}/L)^4$).
  - Belly radius $r_{\max}$ controls axial bulk modulus ($K \propto (r_{\max}/L)^2$) and suppresses Euler buckling.
- **Hierarchical Sub-Truss Mode (`hierarchical=True`)**:
  - Infills the biconical envelope with a localized micro-space-frame (tetrahedral or C15-inspired) converging to apex nodes at the macro vertices.
  - Maximizes the effective second moment of area ($I_{\text{eff}}$) and radius of gyration ($r_g = \sqrt{I/A}$) for high energy absorption and crashworthiness while minimizing additive manufacturing thermal/cure mass.
- **Output Formats**:
  - `"graph"`: Returns `LatticeGraph(nodes, struts, radii)`.
  - `"implicit_sdf"`: Returns `ImplicitField(field, origin, spacing)` with exact analytical signed distance.
  - `"mesh"`: Returns a watertight `trimesh.Trimesh` composed via `manifold3d`.

### 2. Interlocking Contact Assemblies (`interlocking.py`)
- **Discrete Kinematic Assemblies**:
  - Generates unbonded, non-welded unit cell chains with positive clearance gap $\delta$ across all adjacent surfaces.
  - Rigid-body articulation with $G \approx 0$ under zero in-plane load (draping).
- **3D Re-entrant Auxetic Bowtie Links (`cell_topology="reentrant_bowtie"`)**:
  - Features an hourglass waist with re-entrant angle $\theta < 90^\circ$ and out-of-plane chiral interlocking hooks.
  - Under normal displacement or indentation, the re-entrant arms draw inward and wedge against adjacent waists, triggering a geometric jamming transition into a rigid plate.
- **Output Format**:
  - Returns `list[trimesh.Trimesh]` where each element is an independent, watertight mesh ready for multi-body 3MF/STL print preparation.

---

## Quickstart API Usage

### Pentamode Meta-Fluid
```python
from graphite.generators.pentamode import generate_pentamode_lattice

# 1. Generate solid biconical mesh
mesh = generate_pentamode_lattice(
    bounds=((0, 0, 0), (20, 20, 20)),
    unit_cell_size=10.0,
    r_min=0.30,
    r_max=1.00,
    output_format="mesh",
)
mesh.export("pentamode_mesh.stl")

# 2. Generate hierarchical tetrahedral sub-truss graph
graph = generate_pentamode_lattice(
    bounds=((0, 0, 0), (20, 20, 20)),
    unit_cell_size=10.0,
    r_min=0.30,
    r_max=1.00,
    hierarchical=True,
    sub_element_type="tetrahedral",
    output_format="graph",
)
graph.export_inp("pentamode_beam_elements.inp")
```

### Interlocking Auxetic Sheet
```python
from graphite.generators.interlocking import (
    generate_interlocking_auxetic_sheet,
    combine_interlocking_meshes,
    verify_interlocking_clearance,
)

# Generate 4x4 array of discrete 3D re-entrant auxetic bowtie links
link_meshes = generate_interlocking_auxetic_sheet(
    dimensions=(4, 4),
    cell_pitch=10.0,
    clearance_gap=0.40,
    cell_topology="reentrant_bowtie",
)

# Verify clearance (zero boolean collisions)
report = verify_interlocking_clearance(link_meshes)
print(f"Clearance valid: {report['clearance_valid']}, min gap: {report['min_clearance_mm']:.3f} mm")

# Combine into multi-body mesh for rendering
combined = combine_interlocking_meshes(link_meshes)
combined.export("auxetic_sheet.stl")
```

---

## Solid Fraction Calibration & Boundary Clipping

`generate_pentamode_lattice` supports exact boundary clipping via `crop_to_bounds=True`, ensuring all struts and spherical fillets are cleanly cut at domain boundary planes ($x_{\min}, x_{\max}, y_{\min}, y_{\max}, z_{\min}, z_{\max}$). This guarantees exact volume matching and seamless periodic face-matching when tiling unit cells.

### Analytical & Empirical Calibration Formulas
For a diamond unit cell of side length $a$ in mm, bounding volume $V_{\text{cell}} = a^3$:
- The biconical radii follow the standard stiffness ratio $r_{\max} / r_{\min} \approx 2.5714$.
- Calibration parameters for standard target solid fractions:
  - **10% Solid Fraction ($a = 10\text{ mm}$, $V_{\text{target}} = 100.0\text{ mm}^3$):**
    - $r_{\min} = 0.3344\text{ mm}$, $r_{\max} = 0.8599\text{ mm} \implies V = 99.982\text{ mm}^3$ ($10.00\%$ SF).
  - **8% Solid Fraction ($a = 5\text{ mm}$, $V_{\text{target}} = 10.0\text{ mm}^3$ per cell):**
    - $r_{\min} = 0.1484\text{ mm}$, $r_{\max} = 0.3817\text{ mm} \implies V = 1000.06\text{ mm}^3$ for a $50 \times 25 \times 10\text{ mm}$ block ($8.00\%$ SF).

### Solid Skin Integration
Top and bottom solid skins (e.g. $3.175\text{ mm}$ / $1/8$-inch plates) can be integrated with the lattice core using Manifold3D boolean union:
```python
import manifold3d as m3d
from graphite.generators.pentamode import generate_pentamode_lattice
from graphite.explicit.geometry_module import _manifold_to_trimesh, _trimesh_to_manifold

# 1. Generate lattice core (50 x 25 x 10 mm, 8% SF)
core_mesh = generate_pentamode_lattice(
    bounds=((0.0, 0.0, 0.0), (50.0, 25.0, 10.0)),
    unit_cell_size=5.0,
    r_min=0.1484,
    r_max=0.3817,
    output_format="mesh",
    crop_to_bounds=True,
)
m_core = _trimesh_to_manifold(core_mesh)

# 2. Construct 3.175 mm top and bottom plates
t_skin = 3.175
m_bottom = m3d.Manifold.cube([50.0, 25.0, t_skin]).translate([0.0, 0.0, -t_skin])
m_top = m3d.Manifold.cube([50.0, 25.0, t_skin]).translate([0.0, 0.0, 10.0])

# 3. Boolean union into a single watertight solid (total height 16.35 mm)
m_skinned = (m_core + m_bottom + m_top).translate([0.0, 0.0, t_skin])
skinned_mesh = _manifold_to_trimesh(m_skinned)
skinned_mesh.export("pentamode_sample_skinned_16p35mm.stl")
```

---

## Interlocking Auxetic Chainmail Architecture

In `graphite.generators.interlocking`, the `reentrant_bowtie` topology implements true **topological loop-in-loop chainmail connectivity**:
- **Re-entrant Wishbone Geometry:** Each tile features an 8-point central re-entrant auxetic frame with 4 concave waists and 4 diagonal lobes. Dual-strut wishbone arms extend from adjacent lobes in the 4 cardinal directions ($+X, -X, +Y, -Y$) to support circular eyelet rings.
- **Alternating Orientation:** 
  - Even cells ($(i+j) \pmod 2 == 0$) place connecting rings horizontally in the $XY$ plane.
  - Odd cells ($(i+j) \pmod 2 == 1$) place connecting rings vertically in the $XZ$ / $YZ$ planes.
- **Linking Guarantee:**
  - The vertical ring of each odd cell loops directly through the aperture of the neighboring horizontal ring, producing a linking number $L = 1$ with guaranteed physical clearance ($\delta > 0.35\text{ mm}$) and 0 boolean collisions.
  - When tiled in a grid (e.g. $6 \times 6$ or $8 \times 8$), all tiles are physically interlinked, forming a contiguous, draping auxetic metamaterial sheet.

### 3D Volumetric Auxetic Metamaterial Assemblies
When `dimensions` is specified as a 3-tuple `(nx, ny, nz)` with `nz > 1` (e.g. `(2, 2, 2)`):
- **3D Cardinal Linking:** Each unit cell is equipped with 6 cardinal eyelets along $\pm X, \pm Y, \pm Z$, supported by wishbone struts radiating from an 8-point 3D re-entrant corner frame.
- **3D Checkerboard Parity:**
  - $\text{parity} = (i + j + k) \pmod 2 == 0$: $X$-rings in $XY$, $Y$-rings in $YZ$, $Z$-rings in $XZ$.
  - $\text{parity} = (i + j + k) \pmod 2 == 1$: $X$-rings in $XZ$, $Y$-rings in $XY$, $Z$-rings in $YZ$.
- Along every Cartesian direction, opposing rings from adjacent cells cross the cell boundary and loop through each other with linking number $L = 1$, achieving isotropic 3D chainmail topological interlock with zero boolean intersections and positive print clearance ($\delta > 0.40\text{ mm}$).

---

## Rotating Rigid Squares Auxetics

In `graphite.generators.rotating_auxetics`, `generate_rotating_squares_lattice` implements the canonical **Grima & Evans (2000)** rotating rigid unit auxetic mechanism:
- **Kinematic Principle:** Rigid square plates of side length $s$ and thickness $t$ connected at corner vertices via living hinges of radius $r_h$.
- **Alternating Rotation:** Checkerboard parity $(u + v) \pmod 2$ governs deployment angle $\pm\phi = \pm\theta/2$. As $\theta$ increases from $0^\circ$ (closed) to $60^\circ$ (deployed), internal rhombus/diamond voids expand uniformly across both Cartesian axes.
- **Poisson's Ratio:** The ideal kinematic Poisson's ratio is $\nu = -1$ everywhere in the deployment range.
- **3D Stacking:** Multi-layer assemblies (`dimensions=(nx, ny, nz)`) include vertical cylindrical hinge pins connecting layer corner vertices.
- **Printability:** 100% monolithic, watertight Manifold3D meshes ready for additive manufacturing with zero boolean self-collisions.

```python
from graphite.generators import generate_rotating_squares_lattice

# 4x4 monolithic 2D auxetic sheet deployed at 25 degrees
mesh_2d = generate_rotating_squares_lattice(
    dimensions=(4, 4),
    square_side=10.0,
    plate_thickness=2.0,
    hinge_radius=0.45,
    rotation_angle_deg=25.0,
)

# 3x3x2 3D volumetric rotating cubes auxetic block
mesh_3d = generate_rotating_squares_lattice(
    dimensions=(3, 3, 2),
    square_side=10.0,
    plate_thickness=2.0,
    hinge_radius=0.45,
    rotation_angle_deg=20.0,
    layer_spacing=5.0,
)
```

---

## Hashin-Shtrikman Optimal Plate-Lattices

In `graphite.generators.plate_lattice`, `generate_plate_lattice` implements closed-cell and open-cell plate-lattices (Berger et al. 2017, Tancogne-Dejean et al. 2018) that reach the theoretical **Hashin-Shtrikman upper bound** for stiffness:

| Topology | Crystallographic Planes | Key Mechanical Advantage |
| :--- | :--- | :--- |
| **`sc`** | 3 mutually orthogonal $\{100\}$ midplanes ($XY, XZ, YZ$) | **3.3× stiffer** than octet truss at 10% SF; maximum uniaxial membrane modulus |
| **`bcc`** | 6 diagonal $\{110\}$ planes (rhombic dodecahedral symmetry) | High shear modulus along principal axes; triangular sub-cells |
| **`fcc`** | 4 body-diagonal $\{111\}$ planes (octahedral symmetry) | Isotropic multi-axial energy absorption and crush resistance |
| **`sc_bcc`** | 9 combined $\{100\} + \{110\}$ planes | Optimal multi-axial stiffness and shear resistance |

### Solid Fraction Calibration
Provides `calibrate_plate_thickness(unit_cell_size, target_solid_fraction, topology)` using Brent's numerical root-finding on exact Manifold3D cell volumes, ensuring achieved relative density matches the target within $0.01\%$.

```python
from graphite.generators import generate_plate_lattice, calibrate_plate_thickness

# Simple Cubic plate lattice calibrated to exactly 10.0% solid fraction
mesh_sc = generate_plate_lattice(
    bounds=((0.0, 0.0, 0.0), (50.0, 50.0, 25.0)),
    unit_cell_size=10.0,
    target_solid_fraction=0.10,
    topology="sc",
    crop_to_bounds=True,
)
```


