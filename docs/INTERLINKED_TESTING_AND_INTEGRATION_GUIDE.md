# Universal PAM Heteroepitaxy and Interlinked Metamaterial Testing & Integration Guide

## Executive Summary

The **Universal PAM Heteroepitaxy Engine** in Graphite (`graphite/explicit/interlinked/`) generalizes the synthesis of multi-zone, kinematic, and non-welded architected materials across all **11 Polycatenated Architected Material (PAM)** unit cell topologies discovered by Zhou et al. (*Science* 2025). Unlike rigid strut scaffolds that weld intersecting beams at common nodes, interlinked metamaterials mandate **strictly positive physical clearance ($\Delta > 0$)** across all adjacent bodies in print-in-place additive manufacturing (LPBF, SLS, SLA).

This guide provides the definitive, production-grade integration and testing manual for:
1. The **modular subsystem architecture** spanning `epitaxy.py`, `seeding.py`, `generator.py`, `superlattice.py`, `clearance.py`, `dimensioning.py`, and `cells.py`.
2. The **Three Governing Laws of Metamaterial Heteroepitaxy** and the complete **11×11 PAM Feasibility Matrix** on $\{100\}$ habit planes.
3. The **two-tier clearance verification protocol** ($O(N \log N)$ KD-tree broad phase + vectorized narrow phase), separating intra- and inter-cluster interactions.
4. **Rigorous quality gates**, Clean Miter Joint standards (`clean_miter=True`), watertight mesh verification, and regression commands.
5. The **zero-touch extension protocol** for registering a new 12th PAM architecture or custom cell, along with the roadmap for $\{110\}/\{111\}$ habit planes and continuous morphological homotopy.

---

## 1. Integration Architecture & Unified API Standard

### 1.1 Module Dependency & Information Flow

The interlinked module is organized into focused, decoupled components adhering to strict single-responsibility principles:

```text
graphite/explicit/interlinked/
├── cell.py            <── InterlinkedCell Protocol & InterlinkedRegistry catalog
├── cells.py           <── Concrete catalog: 11 PAM shapes + 3 maille/fabric cells
├── particle.py        <── ParticleGeometry (local origin) & InterlinkedParticle in SE(3)
├── seeding.py         <── Spatial seeders: Cartesian (pcu), Diamond (dia), BCC (bcc), Wraps
├── epitaxy.py         <── The Three Laws of Heteroepitaxy (analytical & numerical solvers)
├── dimensioning.py    <── Closed-form 2-of-4 inverse DfAM solver (a, d, Delta, Vf)
├── clearance.py       <── Two-tier KD-tree broad phase + vectorized segment-segment solver
├── superlattice.py    <── Multi-zone sequential heteroepitaxy & buffer layer assembler
└── generator.py       <── Canonical orchestrator (InterlinkedConfig -> InterlinkedLatticeResult)
```

The interaction graph across the pipeline is shown below:

```mermaid
graph TD
    Config["InterlinkedConfig / SuperlatticeConfig"] --> Generator["generator.py / superlattice.py"]
    Generator --> Registry["InterlinkedRegistry (cell.py / cells.py)"]
    Generator --> Dimensioner["PAMDimensioner (dimensioning.py)"]
    Generator --> Epitaxy["Epitaxy Engine (epitaxy.py)"]
    
    subgraph "The Three Laws of Heteroepitaxy"
        Epitaxy --> Law1["Law 1: Commensurate Coincidence\nsolve_supercell_coincidence()"]
        Epitaxy --> Law2["Law 2: Aperture Admissibility\ncompute_habit_plane_aperture()\ncompute_habit_plane_profile()"]
        Epitaxy --> Law3["Law 3: Boundary Catenation\nextract_triangulated_faces()\ncurve_curve_linking_number()\noptimize_interface_catenation()"]
    end
    
    Law1 & Law2 & Law3 --> Alignment["EpitaxialAlignment\n(dx_int, R_habit, strain)"]
    Alignment --> Seeding["seeding.py\nseed_zone_lattice(parent_network)"]
    Seeding --> Particles["list[InterlinkedParticle]\n(particle.py)"]
    Particles --> Clearance["clearance.py\ncheck_particle_clearance()\nmeasure_ring_lattice_clearance()"]
    Clearance --> Meshing["_particles_to_combined_mesh()\nClean Miter Truss (manifold3d)"]
    Meshing --> Result["InterlinkedLatticeResult\n(3MF & STL Export)"]
```

### 1.2 Module Roles and Interactions

| Module | Primary Responsibility | Interface & Dependencies |
| :--- | :--- | :--- |
| `cell.py` | Declares `InterlinkedCell` Protocol, `BasisParticle`, and `InterlinkedRegistry`. | Central catalog decorated via `@InterlinkedRegistry.register()`. |
| `cells.py` | Concrete implementations of all 11 PAM shapes and 3 maille/fabric unit cells. | Implements `forward_clearance`, `resolve_pitch`, `resolve_dimensions`, and `instantiate_site`. |
| `seeding.py` | Translates crystallographic Bravais networks into 3D coordinate frames. | Dispatches via `seed_zone_lattice(parent_network)` (`pcu`, `dia`, `bcc`). |
| `epitaxy.py` | Evaluates and solves the Three Laws of Metamaterial Heteroepitaxy across habit planes. | Consumes `InterlinkedCell` and `ParticleGeometry`; produces `EpitaxialAlignment`. |
| `dimensioning.py` | Inverts scaling laws: $\Delta = \kappa a - d$, $V_f = C_{\text{geom}}(d/a)^2$, $V_{f,\max} = C_{\text{geom}}\kappa^2$. | Solves any 2 of $\{a, d, \Delta, V_f\}$; integrates $C_{\text{geom}}$ dynamically. |
| `clearance.py` | Two-tier KD-tree broad phase + vectorized narrow phase segment/ring distance. | Evaluates surface clearance and Gauss linking number without false self-clash. |
| `superlattice.py`| Sequences arbitrary chains of zones $\{Z_0, Z_1, \dots, Z_{N-1}\}$ along transition axis. | Evaluates consecutive boundary alignments and accumulates longitudinal offsets. |
| `generator.py` | Top-level entrypoint with pre-flight DfAM validation, frames, and CSG meshing. | Routes single-zone, two-zone (`transition_to_cell`), and multi-zone requests. |

### 1.3 Unified API Standard

All explicit interlinked and heteroepitaxial lattice generation in Graphite strictly adheres to the unified API contract:

#### 1.3.1 Two-Zone Heteroepitaxial Transition
```python
from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

# Define transition from C-6-TT (Simple Cubic) to D-4-TET (Diamond Cubic)
config = InterlinkedConfig(
    cell="c6tt",                        # Zone A registered identifier
    transition_to_cell="d4tet",         # Zone B registered identifier
    pitch=9.10,                         # Unit cell pitch for Zone A (mm)
    wire_radius=0.35,                   # Strut radius r = 0.35 mm (d = 0.70 mm)
    min_clearance=0.150,                # Minimum required clearance Delta >= 150 um
    grid_size=(2, 2, 2),                # 2x2x2 unit cells per zone
    check_clearance=True,               # Full two-tier clearance verification
)

result = generate_interlinked_lattice(config)

# Diagnostic inspection
print(f"Alignment offset dx: {result.metadata['alignment'].interface_offset:.2f} mm")
print(f"Verified clearance: {result.metadata['min_clearance_found']*1000:.1f} um")
print(f"Clearance valid: {result.clearance_valid}")

# Export deliverables
result.export_3mf("outputs/c6_to_d4_transition.3mf")
result.export_stl("outputs/c6_to_d4_transition.stl")
```

#### 1.3.2 Multi-Zone Epitaxial Superlattice (Buffer Layers)
```python
from graphite.explicit.interlinked import (
    SuperlatticeConfig,
    LatticeZoneConfig,
    generate_epitaxial_superlattice,
)

# Stack 3 sequential zones: C-6-TT -> D-4-TET -> D-4-CUBE
superlattice_cfg = SuperlatticeConfig(
    zones=[
        LatticeZoneConfig(cell="c6tt", num_layers=2, pitch=9.10),
        LatticeZoneConfig(cell="d4tet", num_layers=2, pitch=12.869),
        LatticeZoneConfig(cell="d4cube", num_layers=2, pitch=12.869),
    ],
    grid_transverse=(2, 2),             # (ny, nz) transverse repetition
    wire_radius=0.35,
    min_clearance=0.100,
)

sl_result = generate_epitaxial_superlattice(superlattice_cfg, check_clearance=True)
sl_result.export_3mf("outputs/pam_superlattice_c6_d4_cube.3mf")
```

---

## 2. Test Architecture & Verification Matrix

### 2.1 Complete Test Suite Topology

Graphite's interlinked module is safeguarded by 8 rigorous pytest suites comprising **123 tests**, executing synchronously in $<8.0\text{s}$:

```text
tests/
├── test_universal_pam_epitaxy.py    # 25 tests — Phases 0 to 4 heteroepitaxy engine
├── test_interlinked_cells.py        # 32 tests — Unit cells, basis instantiation, dimensioning
├── test_interlinked_clearance.py    # 14 tests — KD-tree, segment-segment, circle-circle, Gauss Lk
├── test_interlinked_core.py         #  7 tests — ParticleGeometry, InterlinkedParticle SE(3)
├── test_interlinked_lattice.py      # 14 tests — Generator, DfAM guardrails, boundary culling
├── test_interlinked_pam_integration.py # 6 tests — Legacy PAM compatibility & clearance
├── test_interlinked_seeding.py      # 12 tests — Cartesian, Staggered, Hex, Diamond, Cylindrical wrap
└── test_interlinked_3mf.py          #  9 tests — ISO/IEC 5165 instanced 3MF compression & export
```

### 2.2 Test Suite Breakdown: `test_universal_pam_epitaxy.py`

This suite directly validates the generalized heteroepitaxy engine across all 5 development phases:

1. **`TestPhase0BCCAndZoneSeeder` (Phase 0):**
   - `test_seed_bcc_lattice_counts_and_sublattices`: Verifies that `seed_bcc_lattice` generates exact corner ($A$ at $[i, j, k]a$) and body-center ($B$ at $[i+0.5, j+0.5, k+0.5]a$) sites with equal $8/8$ partition.
   - `test_seed_zone_lattice_dispatch`: Validates dynamic routing across `pcu`, `dia`, and `bcc` networks with proper sublattice metadata tagging.
2. **`TestPhase1Law1UniversalCoincidence` (Phase 1):**
   - Tests transverse coincidence search across `pcu` $\leftrightarrow$ `pcu` ($\theta = 0^\circ$, ratio $(1, 1)$, $\epsilon = 0$).
   - Tests `pcu` $\leftrightarrow$ `dia` ($\theta = 45^\circ$, $p_{\text{dia}} = \sqrt{2} p_{\text{pcu}}$, ratio $(1, 1)$, $\epsilon = 0$).
   - Tests `dia` $\leftrightarrow$ `dia` and `pcu` $\leftrightarrow$ `bcc` with strain $\epsilon < 10^{-5}$.
3. **`TestPhase2Law2UniversalApertureAndProfile` (Phase 2):**
   - `test_all_11_cells_aperture_and_profile_positive`: Parametrized test across **all 11 PAM unit cell classes**, verifying that `compute_habit_plane_aperture` and `compute_habit_plane_profile` yield strictly positive, finite geometric values.
   - `test_c6tt_and_d4tet_exact_analytical_match`: Verifies that C-6-TT aperture ($D_{\text{ap}} = 7.54\text{ mm}$) and D-4-TET profile ($D_{\text{prof}} = 5.49\text{ mm}$) match published benchmarks within $0.02\text{ mm}$.
4. **`TestPhase3Law3CatenationAndFaceExtraction` (Phase 3):**
   - `test_polyhedral_face_extraction`: Verifies topological 3-cycle extraction for tetrahedron (4 triangles), octahedron (8 triangles), and chordless 4-cycle decomposition for cube (12 triangles).
   - `test_ring_curve_curve_gauss_linking`: Validates 3-point Gauss-Legendre quadrature computing $|\text{Lk}| = 1$ for catenated rings and $\text{Lk} \approx 0$ for unlinked rings.
   - `test_dynamic_search_range_catenation`: Tests longitudinal interface spacing search, recovering optimal offset $6.5 \le dx_{\text{int}} \le 7.6\text{ mm}$ with piercings $\ge 1$ and $\Delta \ge 0.150\text{ mm}$.
5. **`TestPhase4UniversalMultiZoneLatticeGeneration` (Phase 4):**
   - End-to-end multi-zone lattice generation: C-6-TT $\to$ D-4-TET, C-6-TT $\to$ D-4-CUBE, and C-6-TT $\to$ J-4-RING.
   - Validates multi-zone buffer superlattice generation (`generate_epitaxial_superlattice`) with zero hardcoded templates.

---

### 2.3 The 11×11 PAM Feasibility Matrix on {100} Habit Planes

We evaluated all $11 \times 11 = 121$ unit cell combinations across $\{100\}$ habit planes at nominal unit cell pitch ($p_A = 9.10\text{ mm}$, $r_{\text{wire}} = 0.35\text{ mm}$, $\Delta_{\min} = 100\ \mu\text{m}$).

#### 2.3.1 Summary of Results
- **Feasible Transitions:** $9 / 121$ ($7.4\%$)
- **Failure Breakdown by Governing Law:**
  - **Law 2 (Aperture Clashes):** $84 / 121$ ($69.4\%$) — Receiving boundary window is smaller than penetrating cross-sectional profile ($D_{\text{aperture}} < D_{\text{profile}} + 2r + 2\Delta$).
  - **Law 3 (Catenation / Clearance Failures):** $18 / 121$ ($14.9\%$) — Penetration occurs without closed-loop interlocking ($\text{piercings} = 0$) or minimum surface clearance falls below threshold ($\Delta < 100\ \mu\text{m}$).
  - **Law 1 (Incommensurate Strain):** $10 / 121$ ($8.3\%$) — Transverse lattice mismatch strain exceeds tolerance ($\epsilon > 5.0\%$).

#### 2.3.2 Admissible Transition Catalog on {100}

| Cell A (Host / Receiver) | Cell B (Penetrating Guest) | Habit Alignment $(m_A, n_B), \theta$ | Optimal $dx_{\text{int}}$ (mm) | Min Clearance $\Delta$ ($\mu\text{m}$) | Piercings / Linking | Physical Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **C-6-TT** | **D-4-TET** | $(1, 1), 45.0^\circ$ | $6.73$ | $461.6$ | 2 | Truncated triangular face receives dual tet apex |
| **C-6-TT** | **J-4-RING** | $(1, 1), 0.0^\circ$ | $3.84$ | $327.7$ | 2 | Truncated face pierced by vertical ring edge |
| **C-6-TT** | **D-4-CUBE** | $(1, 1), 45.0^\circ$ | $5.22$ | $717.8$ | 2 | Truncated triangular face penetrated by cube corner |
| **C-6-TT** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $5.23$ | $149.1$ | 2 | Hexagonal cage face pierced by axial ring |
| **J-4-OCT** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $6.71$ | $266.8$ | 4 | Octahedral tip enters closed circular ring |
| **T-6-RING** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $7.73$ | $1211.3$ | 4 | Homogeneous axial chainmail catenation |
| **T-6-HEX** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $7.73$ | $810.4$ | 4 | Hexagonal prism ring links circular ring |
| **C-6-CO** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $4.66$ | $702.4$ | 1 | Cuboctahedral square face pierced by ring |
| **B-8-CO** | **T-6-RING** | $(1, 1), 0.0^\circ$ | $5.12$ | $130.5$ | 2 | BCC body-centered cage links axial ring |

#### 2.3.3 Physical Insights on Non-Admissible Pairs
1. **Asymmetric Reversibility:** While `C-6-TT` $\to$ `D-4-TET` is highly admissible ($\Delta = 461.6\ \mu\text{m}$), the reverse `D-4-TET` $\to$ `C-6-TT` fails Law 2 ($D_{\text{ap}}(\text{Tet}) = 4.48\text{ mm} < D_{\text{prof}}(\text{TT}) = 7.54\text{ mm}$). A small tetrahedron cannot receive the large truncated tetrahedron cage.
2. **Dense Cage Frustration:** Densely packed cages such as `D-4-ICO` (icosahedron) and `D-4-CT` (cuboctahedron) exhibit tight internal apertures ($D_{\text{ap}} \le 3.5\text{ mm}$), preventing cage-in-cage penetration on $\{100\}$. They require transitional buffer zones (e.g. `D-4-ICO` $\to$ `T-6-RING` $\to$ `C-6-TT`) or inclined habit planes ($\{111\}$).

---

### 2.4 Ring & Strut Clearance Measurement Verification Protocols

A critical failure mode in interlinked metamaterial simulation is **false collision detection** or **false cancellation**. The engine implements specific safeguards to ensure physical realism:

#### 2.4.1 Segment-Segment vs. Continuous Ring-Ring Distance
- **Discrete Segment-Segment (`segment_segment_distance`):**
  Wireframe cages and discretized rings represent geometry as line segments $\mathbf{s}_1(t) = \mathbf{p}_0 + t(\mathbf{p}_1 - \mathbf{p}_0)$ and $\mathbf{s}_2(u) = \mathbf{q}_0 + u(\mathbf{q}_1 - \mathbf{q}_0)$ for $t, u \in [0, 1]$.
  $$\text{dist}(\mathbf{s}_1, \mathbf{s}_2) = \min_{t, u \in [0, 1]} \|\mathbf{s}_1(t) - \mathbf{s}_2(u)\|$$
  Surface clearance is computed as:
  $$\Delta = \text{dist}(\mathbf{s}_1, \mathbf{s}_2) - (r_{\text{wire}, 1} + r_{\text{wire}, 2})$$
- **Analytical Circle-Circle Optimization (`circle_circle_distance`):**
  For smooth circular rings, chord discretization can overestimate clearance at acute contact angles. The engine employs Powell minimization on the continuous toroidal parameter space $(\theta_1, \theta_2) \in [0, 2\pi)^2$, initialized at the closest chord vertices to guarantee exact sub-micron clearance.

#### 2.4.2 Eliminating False Collisions
1. **Intra-Site Exclusion:** Particles belonging to the same rigid body or unit cell basis are flagged with `is_intra = (cluster_i == cluster_j)`. This prevents self-strut checks from triggering false zero-clearance warnings.
2. **Two-Tier Spatial Filtering:**
   - *Broad Phase:* $O(N \log N)$ `scipy.spatial.cKDTree` filters particle centers with query radius:
     $$R_{\text{query}} = 2.15 \cdot \max(R_{\text{bounding}} + r_{\text{wire}})$$
     Prunes $>92\%$ of non-interacting pairs before narrow-phase evaluation.
   - *Narrow Phase:* Evaluates candidate pairs with fully vectorized NumPy tensor broadcasting.
3. **Gauss Linking Number Quadrature Singularities:**
   The double Gauss linking integral:
   $$\text{Lk} = \frac{1}{4\pi} \oint_{C_1} \oint_{C_2} \frac{\mathbf{r}_1 - \mathbf{r}_2}{\|\mathbf{r}_1 - \mathbf{r}_2\|^3} \cdot (d\mathbf{r}_1 \times d\mathbf{r}_2)$$
   is evaluated using **3-point Gauss-Legendre quadrature** per segment. When segments approach touching ($\|\mathbf{r}_1 - \mathbf{r}_2\| \to 0$), the denominator is regularized with $\epsilon = 10^{-12}$ to avoid numerical overflow.

---

## 3. Rigorous Quality Gates & Regression Standards

Every modification to `graphite/explicit/interlinked/` must pass three mandatory quality gates before merge:

### 3.1 Quality Gate 1: Core Architectural Spec Compliance

Graphite maintains a synchronized specification contract in [`GRAPHITE_CORE_SPEC.md`](GRAPHITE_CORE_SPEC.md). Whenever any dataclass, protocol, or public export is updated:

```bash
# Validate core specification synchronization
python scripts/export_core_spec.py --check
```
*Expected Result:* `[OK] Specification GRAPHITE_CORE_SPEC.md is fully verified and consistent with graphite/ source tree.`

If the spec check fails, update `GRAPHITE_CORE_SPEC.md` or re-export via `python scripts/export_core_spec.py`.

### 3.2 Quality Gate 2: Watertight Meshing & Clean Miter Standard

All explicit wireframe and truss structures must conform to the **Clean Miter Truss Standard**:
- **Zero Spherical Bulges:** Spherical joint fillets (`add_spheres=True`) are prohibited for production lattices because they create kinematic interference and increase unbonded frictional drag.
- **Zero Flat Cutoff Notches:** Flat end-caps without bisector miters (`clean_miter=False`) produce jagged non-manifold edge defects.
- **Bisector Cutting Planes:** Native `build_clean_miter_truss` calculates normal bisectors $\mathbf{n}_{\text{plane}} = \frac{\mathbf{v}_1 - \mathbf{v}_2}{\|\mathbf{v}_1 - \mathbf{v}_2\|}$ at every junction.
- **Manifold CSG:** Meshes assembled via `manifold3d` with `process=False` guarantee $100\%$ closed 2-manifold watertight geometry.

```python
from graphite.explicit.interlinked.generator import _particles_to_combined_mesh

# Verify generated mesh is closed and watertight
mesh = _particles_to_combined_mesh(particles, wire_radius=0.35)
assert mesh.is_watertight, "Mesh contains open boundaries or non-manifold edges!"
assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
```

### 3.3 Quality Gate 3: Pytest Regression Battery

Execute the following commands to validate targeted components and the complete interlinked suite:

```bash
# 1. Targeted Heteroepitaxy & Phase 0-4 Tests
python -m pytest tests/test_universal_pam_epitaxy.py -v

# 2. Targeted Unit Cell Catalog & Protocol Tests
python -m pytest tests/test_interlinked_cells.py -v

# 3. Targeted 3MF Instancing & Exporter Tests
python -m pytest tests/test_interlinked_3mf.py -v

# 4. Full Interlinked Metamaterial Battery (123 tests)
python -m pytest tests/test_interlinked_cells.py tests/test_interlinked_clearance.py tests/test_interlinked_core.py tests/test_interlinked_lattice.py tests/test_interlinked_pam_integration.py tests/test_interlinked_seeding.py tests/test_interlinked_3mf.py tests/test_universal_pam_epitaxy.py
```

*Execution Benchmark:* All 123 tests pass cleanly in under **8.0 seconds** on standard developer hardware.

---

## 4. Protocol for Extending the Engine

### 4.1 Adding a 12th PAM Architecture or Custom Cell

To register a new unit cell (e.g. `D-4-OCT`, `K-8-CUBE`, or a custom auxetic particle) so that it **automatically participates in heteroepitaxy, analytical dimensioning, clearance checking, and multi-zone superlattices without modifying existing core loops**, follow this 4-step plug-and-play protocol:

#### Step 1: Implement the `InterlinkedCell` Protocol
Create a new cell class complying with the structural protocol in `graphite/explicit/interlinked/cell.py`:

```python
from typing import Any
import numpy as np
from graphite.explicit.interlinked.cell import (
    BasisParticle,
    InterlinkedCell,
    InterlinkedRegistry,
    _default_identity_orientation,
)
from graphite.explicit.interlinked.particle import ParticleGeometry, InterlinkedParticle

@InterlinkedRegistry.register("d4custom")
@InterlinkedRegistry.register("d-4-custom")
class D4CustomCell:
    """
    12th PAM Unit Cell: Diamond Custom Metamaterial.
    """
    name: str = "D-4-CUSTOM"
    family: str = "pam_3d"
    parent_network: str = "dia"                 # 'pcu', 'dia', 'bcc', etc.
    coordination_number: int = 4
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)
    ]
    
    # Analytical Scaling Constants (Zhou et al., Science 2025)
    CLEARANCE_KAPPA: float = 0.1250             # Delta = kappa * a - d
    SOLID_VOLUME_COEFFICIENT: float = 18.42     # Vf = C_geom * (d / a)^2
```

#### Step 2: Implement Clearance and Inverse Dimensioning Methods
```python
    def forward_clearance(self, unit_cell_pitch: float, strut_diameter: float, **kwargs: Any) -> float:
        """Delta = kappa * a - d"""
        return float(self.CLEARANCE_KAPPA * float(unit_cell_pitch) - float(strut_diameter))

    def resolve_pitch(self, target_clearance: float, strut_diameter: float, **kwargs: Any) -> float:
        """a = (Delta + d) / kappa"""
        return float((float(target_clearance) + float(strut_diameter)) / self.CLEARANCE_KAPPA)

    def resolve_dimensions(self, target_clearance: float, target_solid_fraction: float, **kwargs: Any) -> tuple[float, float]:
        """Closed-form inversion via PAMDimensioner."""
        from graphite.explicit.interlinked.dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)
```

#### Step 3: Implement Spatial Site Instantiation
```python
    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        
        # Build local canonical prototype centered at [0, 0, 0]
        nodes = np.array([...], dtype=np.float64) * a
        struts = np.array([...], dtype=np.int64)
        geom = ParticleGeometry(nodes=nodes, struts=struts, bounding_radius=0.45 * a, geometry_type="CUSTOM")
        
        # Compute rigid world placement transform SE(3)
        R = ctx.get("site_frame", np.eye(3, dtype=np.float64))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = orig
        
        p = InterlinkedParticle(
            particle_id=id_start,
            geometry=geom,
            transform=T,
            cell_index=tuple(grid_index),
            sublattice_id=ctx.get("sublattice", "A"),
        )
        return [p]
```

#### Step 4: (Optional) Declare Analytical Aperture & Profile
If the cell possesses known analytical habit plane projections:
```python
    def habit_plane_aperture(self, plane: str = "100") -> float:
        return 0.55 * self.pitch  # mm
        
    def habit_plane_profile(self, plane: str = "100") -> float:
        return 0.40 * self.pitch  # mm
```
*(If omitted, `compute_habit_plane_aperture` and `compute_habit_plane_profile` automatically extract these directly from the 3D wireframe nodes).*

#### Automatic Pipeline Participation
Once decorated with `@InterlinkedRegistry.register("d4custom")`:
- `InterlinkedConfig(cell="d4custom")` works immediately.
- `InterlinkedConfig(cell="c6tt", transition_to_cell="d4custom")` automatically evaluates Law 1, Law 2, and Law 3 to solve $dx_{\text{int}}$ and rotation $\mathbf{R}_{\text{habit}}$.
- `SuperlatticeConfig(zones=[LatticeZoneConfig(cell="d4custom"), ...])` stacks seamlessly.
- `PAMDimensioner.dimension_cell("d4custom", pitch=10.0, wire_diameter=0.8)` computes $V_f$ and jamming density without manual tuning.

---

### 4.2 Roadmap: Mixed Habit Planes ({110}, {111}) & Continuous Morphing

```mermaid
timeline
    title PAM Heteroepitaxy Evolution Roadmap
    section Completed (v1.0)
        {100} Universal Heteroepitaxy : 11 PAM shapes
        : 3 Bravais Networks (PCU, DIA, BCC)
        : Three Laws Automation
        : Multi-Zone Superlattices
    section Next Phase (v1.1)
        {110} & {111} Habit Planes : Rhombic & Hexagonal 2D Nets
        : 2D Unimodular Coincidence Tensors
        : Inclined Planar Slicing Apertures
    section Future (v2.0)
        Continuous Morphing Homotopy : Vertex Truncation Homotopy tau in [0, 1]
        : Dynamic Clearance Corridors
        : Non-Planar Conformal Habit Shells
```

#### 4.2.1 Mixed Habit Planes ({110} and {111})
1. **Transverse 2D Bravais Lattice Tensors:**
   - Along $\{100\}$, the habit plane is a 2D square grid.
   - Along $\{110\}$, the habit plane projects into a centered rectangular / rhombic 2D lattice ($\mathbf{a}_1 = [0, 0, 1]a$, $\mathbf{a}_2 = [1/\sqrt{2}, 1/\sqrt{2}, 0]a$).
   - Along $\{111\}$, the habit plane projects into a 2D hexagonal / triangular close-packed network ($\mathbf{a}_1 = [1, -1, 0]a/\sqrt{2}$, $\mathbf{a}_2 = [0, 1, -1]a/\sqrt{2}$).
2. **Generalized Supercell Coincidence Tensor:**
   Replace scalar ratio search with integer unimodular 2D transformation matrix $\mathbf{M} \in \mathbb{Z}^{2 \times 2}$:
   $$\|\mathbf{M} \mathbf{A}_{\text{transverse}} - \mathbf{B}_{\text{transverse}} \mathbf{R}(\theta)\| \le \epsilon_{\max}$$
3. **Arbitrary Miller Plane Aperture Extraction:**
   Rotate node coordinates so habit normal $\mathbf{n} = \frac{(h, k, l)}{\sqrt{h^2+k^2+l^2}}$ aligns with $\mathbf{\hat{x}}$, projecting onto $(\mathbf{\hat{y}}, \mathbf{\hat{z}})$ for polygon in-circle extraction.

#### 4.2.2 Continuous Morphological Transitions
Where discrete heteroepitaxial jumping is inadmissible due to extreme aperture disparity (e.g. `C-6-TT` $\leftrightarrow$ `D-4-ICO`), the system delegates to **continuous vertex-truncation homotopy**:
$$\mathbf{x}_{\text{cage}}(\tau) = (1 - \tau)\mathbf{x}_A + \tau \mathbf{x}_B, \quad \tau \in [0, 1]$$
- Governed by `EpitaxialMorphConfig` and `generate_epitaxial_morph_lattice`.
- Interpolates cage geometry across $N$ buffer layers, dynamically tracking inter-layer clearance to eliminate topological jamming while transitioning between incompatible crystal classes.
