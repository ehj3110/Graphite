# Polycatenated Architected Materials (PAMs) — Architecture & Implementation Guide

## 1. Overview & Theoretical Framework

Polycatenated Architected Materials (PAMs) are a class of **kinematic 3D metamaterials** constructed from discrete, unbonded polyhedral wireframe cages that are topologically catenated (interlinked) across crystallographic parent networks ([Zhou et al., *Science* 2025](https://doi.org/10.1126/science.adp5092)).

Unlike standard structural lattices (which weld intersecting struts at common node junctions), PAMs maintain a strictly positive surface clearance:
$$\\Delta = d_{\\text{centerline}} - (r_1 + r_2) \\ge \\Delta_{\\min} > 0$$
across all struts. When printed in additive manufacturing (via SLS, SLA, or multi-material jetting), the independent cages move freely without mechanical bonding, giving rise to:
- Non-linear topological kinematics (finite free-play before kinematic locking).
- Giant energy dissipation through inter-cage friction and contact.
- Auxetic negative Poisson's ratios under tensile and compressive loading.

---

## 2. Tripartite Naming Convention (`X-n-abc`)

In accordance with Reticular Chemistry Structure Resource (RCSR) and Zhou et al. (2025), PAM architectures follow the tripartite descriptor **`X-n-abc`**:

| Component | Code | Meaning | Examples |
| :--- | :--- | :--- | :--- |
| **`X`** | Parent Network | The underlying crystallographic graph connecting particle centers | `C` (primitive cubic `pcu`), `D` (diamond `dia`), `S` (shifted cubic `pcu-b`), `J` (planar square cluster) |
| **`n`** | Coordination Number | Number of catenated cage interlocks per particle | $n = 4$ (tetrahedral/square), $n = 6$ (octahedral/cubic), $n = 6/2$ (hybrid primary/secondary) |
| **`abc`** | Particle Geometry | Wireframe polyhedral cage geometry | `TET` (tetrahedron), `TT` (truncated tetrahedron), `OCT` (octahedron), `CO` (cuboctahedron) |

---

## 3. Core 3D Bulk Periodic Topologies

### 3.1 `C-6-TT` (Cubic Truncated Tetrahedral)
- **Crystallographic Network:** Primitive Simple Cubic (`pcu`), coordination $n = 6$.
- **Particle Geometry:** Truncated Tetrahedron (12 vertices, 18 struts).
  - Derived from truncating a regular tetrahedron at $1/3$ edge length.
  - Possesses $T_d$ tetrahedral point-group symmetry.
  - Forms 4 hexagonal faces (normals along $\\langle 1, 1, 1 \\rangle$) and 4 triangular face cutouts (normals along opposite $\\langle 1, 1, 1 \\rangle$).
- **Resolution of 3D Cuboctahedron Frustration:**
  - *Theoretical Paradox:* A standard cuboctahedron (`CO`) with diamond-oriented square faces cannot achieve a relative $45^\\circ$ twist along all three orthogonal Cartesian axes simultaneously using any rigid $SO(3)$ rotation matrix, because orthogonal 3D rotations do not commute. Setting an unrotated or 2-fold rotated cuboctahedron on a simple cubic grid causes strut collisions ($\\Delta = 0.0\\text{ mm}$).
  - *Truncated Tetrahedron Solution:* The Truncated Tetrahedron wireframe presents open hexagonal and triangular window cutouts directly aligned with the neighbor approach vectors. Adjacent unrotated particles interpenetrate through these open windows with **zero solid collisions**.
- **Validated Lattice Parameters:**
  - Unit cell spacing: $a_0 = 1.25 \\times \\text{size}$.
  - Edge length: $L_{\\text{edge}} = \\frac{2}{3} \\times \\text{size}$.
  - For $\\text{size} = 10.0\\text{ mm}$, $a_0 = 12.50\\text{ mm}$, $r = 0.50\\text{ mm}$ (1.0 mm wire):
    - Minimum surface clearance: **$\\Delta = 0.6421\\text{ mm} \\ge 0.30\\text{ mm}$** (exceeds SLA/SLS DFAM requirements).
    - Boolean collision volume: **$0.0\\text{ mm}^3$** across all interacting pairs.

### 3.2 `D-4-TET` (Diamond Tetrahedral)
- **Crystallographic Network:** Diamond (`dia`), coordination $n = 4$.
- **Particle Geometry:** Regular Tetrahedron (4 vertices, 6 struts).
- **Sublattice Parity & Relative Twist:**
  - Diamond sites are partitioned into bipartite sublattices: Sublattice A ($P=0$) and Sublattice B ($P=1$), displaced along $\\langle 1, 1, 1 \\rangle / 4$.
  - Particle A has vertex 0 aligned with $[+1, +1, +1]$.
  - Particle B has vertex 0 aligned with $[-1, -1, -1]$ with a relative $60^\\circ$ twist around the bond axis.
  - Catenation occurs corner-through-face along all 4 tetrahedral coordination vectors.
- **Validated Lattice Parameters:**
  - Bond length: $d \\approx 1.10 \\times L_{\\text{edge}}$ (or calibrated via `calibrate_d4tet_edge_length`).
  - Conventional cell size: $a = \\frac{4d}{\\sqrt{3}}$.
  - Tested: $3 \\times 3 \\times 2$ conventional block (144 particles), $\\Delta \\ge 0.10\\text{ mm} - 0.40\\text{ mm}$.

### 3.3 `J-4-OCT` (Octahedral Planar Cross)
- **Crystallographic Network:** 2D Square-planar grid ($n = 4$).
- **Particle Geometry:** Regular Octahedron (6 vertices, 12 struts).
- **Catenation Mechanism:** Central octahedron connects tip-to-tip with 4 orthogonal neighbors along $\\pm X$ and $\\pm Y$ with a $45^\\circ$ relative bond twist.

### 3.4 `S-6/2-OCT` (Shifted Octahedral Hybrid)
- **Crystallographic Network:** Shifted cubic network (`pcu-b`), hybrid coordination $n = 6$ (primary corner catenation along Cartesian axes) + $n = 2$ (secondary edge catenation along body diagonals).

---

## 4. Clearance Verification & DfAM Rules

### Global Broad-Phase Clearance Check
The internal function `_min_clearance_among_particles` in `graphite/explicit/interlinked/pams.py`:
1. Constructs a `scipy.spatial.cKDTree` over all particle centroids.
2. Queries all pairs within the global interaction radius $R_{\\text{search}} = 2.2 \\times R_{\\text{outer}} + 2r$.
3. Evaluates analytical segment-segment minimum 3D distances between all strut pairs:
   $$d_{\\min} = \\min_{s_A \\in P_A, s_B \\in P_B} \\text{dist}(s_A, s_B)$$
4. Computes surface clearance: $\\Delta = d_{\\min} - (r_A + r_B)$.
5. Flags invalid if $\\Delta < \\Delta_{\\min}$ (default $0.30\\text{ mm}$ for SLA/SLS).

### Manifold3D Solid Intersection Check
Solid geometry is verified via `manifold3d`:
$$\\text{Vol}(M_A \\cap M_B) = 0.0\\text{ mm}^3$$
guaranteeing that touching or penetrating volumes are strictly absent.

---

## 5. Python API Reference

```python
from graphite.explicit.interlinked import (
    generate_truncated_tetrahedron_particle,
    generate_c6tt_cubic_tiling,
    generate_d4tet_diamond_tiling,
    generate_pam_lattice,
    recalibrate_pam_lattice,
    export_pam_multibody_stl,
)

# 1. Generate a single discrete Truncated Tetrahedron particle
# Supports both object access and tuple unpacking:
particle = generate_truncated_tetrahedron_particle(size=10.0, center=(0, 0, 0))
nodes, struts = particle  # Unpacks (12, 3) float64 and (18, 2) int64

# 2. Tessellate a 3D periodic bulk C-6-TT lattice
result = generate_c6tt_cubic_tiling(
    repeats=(2, 2, 2),        # 8-particle bulk block
    size=10.0,                # Particle size in mm
    strut_radius=0.50,        # 1.0 mm wire diameter
    min_clearance=0.30,       # DFAM threshold in mm
    build_meshes=True,        # Construct Manifold3D meshes
)

print(f"Code: {result.tripartite_code}")
print(f"Clearance Valid: {result.clearance_valid}")
print(f"Min Surface Clearance: {result.min_clearance_mm:.4f} mm")

# 3. Export to multi-body STL (discrete non-welded islands)
export_pam_multibody_stl(result, "outputs/c6tt_3d_bulk.stl")

# 4. Automated recalibration for target bond/size
optimal_spacing = recalibrate_pam_lattice("C-6-TT", target_d=10.0, r=0.50)
print(f"Optimal a0 spacing: {optimal_spacing:.4f} mm")

# 5. Top-level unified router
pam_lattice = generate_pam_lattice(
    tripartite_code="C-6-TT",
    unit_cell_size=10.0,
    repeats=(2, 2, 2),
    strut_radius=0.50,
)
```

---

## 6. Verification & Test Suite

The PAM subsystem is covered by comprehensive unit and integration tests:
- `tests/test_pam_c6tt.py`: Topology, tuple unpacking, clearance verification, Boolean intersection, multibody STL export, routing, and parameter recalibration.
- `tests/test_pam_polyhedra.py`: Cuboctahedra, Octahedra, coordination shell star, and planar crosses.
- `scripts/generate_c6tt_tiling_review.py`: Generates deliverables `outputs/c6tt_3d_bulk.stl` and `outputs/c6tt_3d_bulk.png`.

---

## 7. Unit Cell Sizing & Strut Clearance Scaling Laws

### 7.1 Unit Cell Definition by Network Geometry

| Topology | Crystallographic Network | Unit Cell Definition | Formula / Relationship |
| :--- | :--- | :--- | :--- |
| **`C-6-TT`** | Simple Cubic (`pcu`) | **Lattice Repeat Pitch ($a_0$)** | Center-to-center translational pitch between adjacent cubic sites ($L = N \cdot a_0$). Particle size relates by $a_0 = 1.25 \times s$. |
| **`D-4-TET`** | Diamond (`dia`) | **Option A: Bond Pitch ($d_{\text{bond}}$)**<br>**Option B: Conventional Cube ($a_{\text{conv}}$)** | Nearest-neighbor center-to-center distance along tetrahedral $\langle 1, 1, 1 \rangle$ bonds ($d_{\text{bond}} \approx 1.10 \times L_{\text{edge}}$).<br>Conventional 8-particle cube: $a_{\text{conv}} = \frac{4 d_{\text{bond}}}{\sqrt{3}} \approx 2.3094 \times d_{\text{bond}}$. |

### 7.2 Analytical Clearance Scaling Laws

The centerline gap $d_{\text{centerline}}$ between interpenetrating struts scales linearly with the unit cell dimension:
$$\Delta = d_{\text{centerline}} - D_{\text{strut}} = \kappa \cdot a - D_{\text{strut}}$$

- **For `C-6-TT`:**
  $$\kappa_{\text{C6TT}} \approx 0.1314 \implies d_{\text{centerline}} = 0.1314 \times a_0$$
  - At $a_0 = 5.0\text{ mm}$, $d_{\text{centerline}} = 0.657\text{ mm}$.
  - A $1.0\text{ mm}$ strut diameter causes a solid collision of $\Delta = 0.657\text{ mm} - 1.000\text{ mm} = -0.343\text{ mm}$.
  - To support a $1.0\text{ mm}$ strut with $\Delta \ge 0.30\text{ mm}$, the cell pitch must be $a_0 \ge \frac{1.0 + 0.30}{0.1314} \approx 9.89\text{ mm} \approx 10\text{ mm}$.
  - At $a_0 = 5.0\text{ mm}$, the maximum allowable strut diameter is $D_{\text{strut}} = 0.35\text{ mm}$ ($\Delta = 0.3069\text{ mm}$).

- **For `D-4-TET`:**
  $$\kappa_{\text{D4TET}} \approx 0.2354 \implies d_{\text{centerline}} = 0.2354 \times d_{\text{bond}}$$
  - At $d_{\text{bond}} = 5.0\text{ mm}$, $d_{\text{centerline}} = 1.177\text{ mm}$.
  - For $D_{\text{strut}} = 1.0\text{ mm}$, surface clearance is positive: $\Delta = 1.177\text{ mm} - 1.000\text{ mm} = 0.1770\text{ mm} > 0$.
  - Thus, **`D-4-TET` can cleanly accommodate a $1.0\text{ mm}$ strut diameter at $d_{\text{bond}} = 5.0\text{ mm}$**.

### 7.3 Strip Deliverables ($75 \times 25 \times 20\text{ mm}$) — Clean Mitered Trusses

Both strip deliverables have been regenerated with **clean mitered truss connections** (`add_spheres=False`), eliminating spherical nodes in favor of crisp, directly intersecting cylinder junctions:

- **`C-6-TT` Clean Mitered Strip:** [`outputs/c6tt_strip_75x25x20.stl`](../outputs/c6tt_strip_75x25x20.stl) | Render: [`outputs/c6tt_strip_75x25x20.png`](../outputs/c6tt_strip_75x25x20.png)
  - Dimensions: $75 \times 25 \times 20\text{ mm}$ ($15 \times 5 \times 4$ cells, 300 discrete particles).
  - Cell pitch: $a_0 = 5.0\text{ mm}$, Strut diameter: $D = 500\ \mu\text{m}$ ($0.50\text{ mm}$, $r = 0.25\text{ mm}$).
  - Min surface clearance: $\Delta = 156.9\ \mu\text{m}$ ($0.1569\text{ mm}$).
- **`D-4-TET` Clean Mitered Strip:** [`outputs/d4tet_strip_75x25x20.stl`](../outputs/d4tet_strip_75x25x20.stl) | Render: [`outputs/d4tet_strip_75x25x20.png`](../outputs/d4tet_strip_75x25x20.png)
  - Dimensions: $80.8 \times 23.1 \times 23.1\text{ mm}$ ($7 \times 2 \times 2$ conventional cells, 112 discrete particles).
  - Bond pitch: $d_{\text{bond}} = 5.0\text{ mm}$ ($a_{\text{conv}} = 11.55\text{ mm}$), Strut diameter: $D = 750\ \mu\text{m}$ ($0.75\text{ mm}$, $r = 0.375\text{ mm}$).
  - Min surface clearance: $\Delta = 427.0\ \mu\text{m}$ ($0.4270\text{ mm}$).

### 7.4 Scaled Deliverable: `C-6-TT` $3 \times 3 \times 3$ Cube ($a_0 = 12.7\text{ mm}$, $D = 1.25\text{ mm}$)

- **STL File:** [`outputs/c6tt_3x3x3_12.7mm_d1.25.stl`](../outputs/c6tt_3x3x3_12.7mm_d1.25.stl) | Render: [`outputs/c6tt_3x3x3_12.7mm_d1.25.png`](../outputs/c6tt_3x3x3_12.7mm_d1.25.png)
  - Dimensions: $38.1 \times 38.1 \times 38.1\text{ mm}$ ($1.50\text{ in}$ cube, $3 \times 3 \times 3 = 27$ discrete particles).
  - Cell pitch: $a_0 = 12.7\text{ mm}$ ($0.50\text{ in}$), Particle size: $s = 10.16\text{ mm}$.
  - Strut diameter: $D = 1.25\text{ mm}$ ($r = 0.625\text{ mm}$).
  - Min surface clearance: $\Delta = 418.4\ \mu\text{m}$ ($0.4184\text{ mm} \ge 0.30\text{ mm}$).
  - Joint style: Clean mitered trusses (`add_spheres=False`, `circular_segments=24`).
  - File size: $2.10\text{ MB}$ (watertight multi-body solid).
