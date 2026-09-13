# Architecture Handoff: Unified Extensible System for `graphite.explicit.interlinked`

**Document Purpose:** Architectural design brief and specifications for a frontier cloud model to plan and guide the modular, unit-cell-driven framework for `graphite.explicit.interlinked`.  
**Status:** Canonical handoff specification.  
**Target Capability:** Achieve the same plug-and-play versatility for kinematic/interlinked/catenated metamaterials that `graphite.explicit` has for welded truss lattices (declarative unit cell registration, spatial tessellation across volumes and analytical primitives, two-tier vectorized clearance verification, and watertight instanced 3MF solid meshing).  
**Primary References in Codebase:**
- Subsystem root: [`graphite/explicit/interlinked/`](../graphite/explicit/interlinked/)
- Canonical particle abstraction: [`graphite/explicit/interlinked/particle.py`](../graphite/explicit/interlinked/particle.py)
- Declarative cell protocol & registry: [`graphite/explicit/interlinked/cell.py`](../graphite/explicit/interlinked/cell.py)
- PAM implementation: [`graphite/explicit/interlinked/pams.py`](../graphite/explicit/interlinked/pams.py)
- Chainmail & rings: [`graphite/explicit/interlinked/patterns.py`](../graphite/explicit/interlinked/patterns.py)
- Clearance & linking: [`graphite/explicit/interlinked/clearance.py`](../graphite/explicit/interlinked/clearance.py)
- Conformal seeding & culling: [`graphite/explicit/interlinked/conformal.py`](../graphite/explicit/interlinked/conformal.py)
- Solid meshing module: [`graphite/explicit/geometry_module.py`](../graphite/explicit/geometry_module.py)
- Core architectural spec: [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md)

---

## 1. Executive Summary & Product Vision

### 1.1 The Goal
`graphite` currently possesses two distinct explicit geometric paradigms:
1. **`graphite.explicit` (Welded Trusses):** Highly modular and versatile. Developers and users can register a unit cell (e.g., Simple Cubic, Octet, Kelvin, Kagome, A15), specify repeating grid dimensions $(N_x, N_y, N_z)$ or a conformal CAD domain, and the engine automatically builds the lattice, deduplicates shared boundary nodes, and solids the result.
2. **`graphite.explicit.interlinked` (Kinematic / Non-Welded Metamaterials):** Highly capable with validated implementations of:
   - 2D/2.5D chainmail fabrics (European 4-in-1, Japanese Kusari).
   - 3D volumetric interlocking ring cages (`cubic_8ring`).
   - Complex space fabrics (NASA JPL hexagonal fabric with spiral hook arms).
   - 3D Polycatenated Architected Materials (PAMs from Zhou et al., *Science* 2025: `C-6-TT` truncated tetrahedra, `D-4-TET` diamond tetrahedra, `J-4-OCT` planar crosses, `C-6-CO` coordination stars).

However, historical interlinked families were developed with **bespoke, standalone generator pipelines**. Adding a new chainmail pattern, polyhedral catenation cage, or interlocking auxetic required writing a separate generator script rather than declaring a new **`InterlinkedCell`** within a unified registry.

### 1.2 The Directive for the Unified Engine
The interlinked subsystem is refactored into a **declarative, extensible, unit-cell-driven system** organized across 6 decoupled layers:
- Defining a new interlinked metamaterial requires only writing a concise, self-contained **`InterlinkedCell` class** registering its asymmetric basis, orientation field, and polymorphic clearance laws.
- The core engine handles spatial tessellation (Cartesian 3D, Diamond 3D, Hexagonal close-pack 2D, Cylindrical wraps, Spherical shells), coordinate transforms, multi-tier boundary management, two-tier vectorized clearance verification, and Manifold3D instanced 3MF multi-body solidification automatically.

---

## 2. Hard Codebase Rules & Engineering Guardrails

Any extension to `graphite.explicit.interlinked` **must strictly adhere** to existing Graphite conventions:

| Guardrail | Mandate | Rationale / Enforcing Code |
| :--- | :--- | :--- |
| **Strict Non-Welding** | Particles must **never** share nodes or weld at boundaries. Minimum surface clearance $\Delta \ge \Delta_{\min} > 0$. | Printing unbonded multi-body assemblies (SLS, SLA, SLM) requires mechanical free-play. Fused joints destroy kinematic mobility. |
| **Zero Cut Rings** | **No naive boolean boundary slicing.** Slicing a closed ring or cage breaks the loop, destroying catenation. | Must use **Modular Boundary Policies** (SDF Inset Culling, Dedicated Boundary Anchors, or Exterior Solid Frame Welding). |
| **Naming Conventions** | Topology arrays must be named **`nodes`** (`shape=(V, 3)`, float64) and **`struts`** (`shape=(E, 2)`, int64). Do **not** use `edges`. | Enforced across `graphite.explicit` and [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md). |
| **Local Index Spaces** | Each discrete particle in an interlinked lattice must maintain its own **local node/strut index space** $(0 \dots V-1)$. | Prevents accidental node-merging or KD-tree welding. |
| **Output Directories** | All generated STL, 3MF, and PNG deliverables **must** be written under `outputs/`. | Never write generated files to `test_parts/` (read-only CAD fixtures) or project root. |
| **Core Spec Integrity** | Any change to public dataclasses, protocols, or generators must pass `python scripts/export_core_spec.py --check`. | Enforces architectural consistency across human and AI developers. |
| **Solid Engine** | Solid CSG must use **`manifold3d`** (via `graphite.explicit.geometry_module`). | Guarantees 100% watertight, 2-manifold multi-body meshes. |

---

## 3. Critical Engineering Analysis & Resolved Blind Spots

A rigorous review of the physics and mechanics of non-welded assemblies identified five critical architectural fallacies that this unified engine resolves:

### 3.1 Resolution 1: Polymorphic Clearance vs. Single-Scalar Fallacy
* **The Fallacy:** Assuming that a single linear equation $\Delta = \kappa \cdot a_0 - D_{\text{strut}}$ governs all interlinked metamaterials.
* **The Reality:** Linear scaling holds strictly for isotropic, self-similar polyhedral cages (`C-6-TT`, `D-4-TET`). In contrast:
  - For 2D/2.5D chainmail (European 4-in-1), clearance is non-linearly coupled to row tilt angle $\theta$, ring aspect ratio, and gravitational settling.
  - For complex kinematic mechanisms (NASA space fabric), clearance depends on spiral arm sweep curvature, engagement depth, and z-stacking gaps.
* **The Architecture:** Clearance calculation is formalized as a **polymorphic protocol method**:
  - `forward_clearance(unit_cell_pitch, strut_diameter, **kwargs) -> float`
  - `resolve_pitch(target_clearance, strut_diameter, **kwargs) -> float`
  Backed by closed-form laws for isotropic cages and fast numerical stencils for complex kinematics.

### 3.2 Resolution 2: Multi-Tier Boundary Policies vs. Naive Inset Culling
* **The Fallacy:** Solely relying on binary SDF Inset Culling (discarding particles that cross the boundary envelope).
* **The Reality:** Pure inset culling leaves perimeter links with missing neighbors. In LPBF/SLS, loose boundary links shift during recoating and snag the blade, causing catastrophic build crashes; in SLA/DLP, unconstrained links wash off during solvent agitation.
* **The Architecture:** A 3-tier boundary management engine:
  - **Policy A (SDF Inset Culling):** Retained for bulk material specimen testing.
  - **Policy B (Dedicated Boundary Anchors):** Substitutes perimeter sites with closed termination loops, eyelets, or half-cages.
  - **Policy C (Solid Perimeter Frame Welding):** Retains inset particles and generates an exterior solid CAD handling frame whose inner face fuses with outer perimeter struts, locking the assembly into a stable border for tensile testing or physical handling.

### 3.3 Resolution 3: Canonical Particle Separation vs. Abstraction Proliferation
* **The Fallacy:** Introducing competing classes (`ParticleTemplate`, `BasisParticle`, `PAMParticle`) with overlapping roles.
* **The Reality:** Storing stateful particles inside static cell declarations creates semantic confusion, while mutating node arrays breaks local canonical coordinates.
* **The Architecture:** Clear separation of concerns:
  1. [`ParticleGeometry`](../graphite/explicit/interlinked/particle.py): Frozen, lightweight canonical prototype at local origin $[0, 0, 0]$ (`nodes`, `struts`, `bounding_radius`, `geometry_type`).
  2. [`InterlinkedParticle`](../graphite/explicit/interlinked/particle.py): Placed instance with an $SE(3)$ homogeneous transform matrix (`transform`), evaluating `global_nodes()` lazily on the fly.
  3. `PAMParticle` remains 100% intact with bidirectional bridge methods (`to_pam()` and `from_pam()`), guaranteeing zero regressions across 1400+ lines in `pams.py`.

### 3.4 Resolution 4: Two-Tier Vectorized Distance Checks vs. CSG Boolean Bottlenecks
* **The Fallacy:** Relying on Manifold3D exact boolean evaluation ($\text{Vol}(A \cap B) == 0$) for collision detection.
* **The Reality:** Evaluating solid boolean intersections across hundreds or thousands of unbonded bodies causes generation times to explode from milliseconds to tens of minutes.
* **The Architecture:** Collision detection and clearance verification execute entirely in the **discrete wireframe and analytical domain**:
  - **Broad-Phase:** `scipy.spatial.cKDTree` over particle centroids with search radius $R_1 + R_2 + \text{margin}$ ($O(N \log N)$ filtering).
  - **Narrow-Phase:** Vectorized segment-segment 3D distance for wireframes, analytical circle-circle distance for torus rings, and local convex hull distance for custom meshes.
  - Solid CSG meshing via Manifold3D is invoked **only once** the configuration is mathematically proven collision-free.

### 3.5 Resolution 5: Analytical Primitive Mappings vs. Freeform Conformal Scope Creep
* **The Fallacy:** Over-engineering generalized non-linear conformal metric relaxation for arbitrary freeform surfaces.
* **The Reality:** Freeform surface warping distorts open window cutouts and causes solid collisions in kinematic assemblies. Graphite targets well-defined analytical primitives (boxes, cylinders, spheres).
* **The Architecture:**
  - **Cylindrical Wraps:** Euclidean $(x, y) \to (R, \theta, z)$ with **exact pitch-matching quantization**: $2\pi R = N \cdot a_0$. This guarantees zero seam shearing or link collision across the $0 \to 2\pi$ seam line.
  - **Spherical Shells:** Analytical geodesic projection (subdivided icosahedra) or Fibonacci spherical lattices with analytical radial frames.
  - **Strict Particle Rigidity:** Particles mapped to primitives remain strictly rigid; only their global positions and orientation frames conform to the surface.

### 3.6 Resolution 6: Kinematic Frustration, Stagger Contracts & Pre-Solidification Clearance Gates
* **The Root Cause:** In non-welded kinematics, particles whose bounding diameters exceed grid pitch ($2R > a_0$) cannot simply be repeated with identical orientation along rows without coplanar collision.
  - For example, European 4-in-1 on an unstaggered square grid requires a **diagonal weave axis** $\mathbf{k} = [1, 1, 0]/\sqrt{2}$ with **checkerboard parity** $(i + j) \pmod 2$. If oriented along $X$ with row parity $j \pmod 2$, horizontal neighbors are coplanar and smash into each other ($\Delta = -0.80\text{ mm}$).
  - Alternatively, if oriented along $X$, rows must possess a **half-pitch translational stagger** ($x \to x + a_0/2$).
* **The Systemic Danger (Silent CSG Welding):**
  When intersecting solids are passed to `Manifold.compose()` or a standard boolean union, the CSG engine silently fuses the intersecting bodies into a single monolithic component with triangulated intersection seams.
* **The Architectural Safeguards:**
  1. **Mandatory Pre-Solidification Clearance Gate:** The top-level generation pipeline must evaluate wireframe/analytical clearance *before* invoking solid meshing. If $\min(\Delta) \le 0$ or violates target clearance, it must raise a `ClearanceViolationError` with the exact colliding particle IDs and coordinates, preventing silent fusion.
  2. **Cell Stagger Contracts:** When registering an `InterlinkedCell`, the class must declare whether its parent network is simple Cartesian, brick-staggered, hexagonal, or diamond.
  3. **Disjoint Multi-Body Export:** Multi-body unbonded assemblies must never use boolean union or `Manifold.compose()`. Instead, each particle must be converted to an independent mesh island and assembled via `trimesh.util.concatenate()` or instanced 3MF, ensuring components remain separate in slicing and CAD viewers.

---

## 4. The 6-Layer Engine Architecture

```
┌────────────────────────────────────────────────────────────┐
│ 1. Crystallographic Parent Network (Spatial Sites)         │
│    • Simple Cubic (pcu), Diamond (dia), BCC, FCC, Hex 2D   │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 2. Asymmetric Basis & Cell Declarations                    │
│    • BasisParticle prototypes + SO(3) orientation fields   │
│    • Polymorphic forward_clearance & resolve_pitch         │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Analytical Primitive Seeding Engine                     │
│    • Cartesian volumes, quantized cylindrical wraps,       │
│      spherical shells with strict particle rigidity        │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 4. Two-Tier Vectorized Clearance & Inversion Engine        │
│    • Broad-phase KDTree on particle centroids              │
│    • Narrow-phase vectorized segment & circle distances    │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 5. Perimeter Boundary Management                           │
│    • Policy A: SDF Inset Culling                           │
│    • Policy B: Dedicated Boundary Anchor loops/half-cages  │
│    • Policy C: Exterior Solid Frame Welding                │
└─────────────────────────────┬──────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────┐
│ 6. Multi-Body Instanced 3MF Solidification Engine          │
│    • Canonical mesh resource defined once                  │
│    • Assembly defined by lightweight SE(3) instance list   │
│    • Manifold3D clean_miter, embedded_spheres, tori       │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Core Code Abstractions (Implemented in Phase 1)

### 5.1 Canonical Particle Representation (`particle.py`)

```python
@dataclass(frozen=True)
class ParticleGeometry:
    nodes: np.ndarray             # (V, 3) float64 centered at local [0, 0, 0]
    struts: np.ndarray            # (E, 2) int64 local indices (0 ... V-1)
    bounding_radius: float        # Outer sphere enclosing all local nodes
    geometry_type: str            # 'TT', 'TET', 'CO', 'OCT', 'ring', etc.
    metadata: dict[str, Any]      # Immutable parameters

@dataclass
class InterlinkedParticle:
    particle_id: int
    geometry: ParticleGeometry
    transform: np.ndarray         # (4, 4) float64 homogeneous matrix in SE(3)
    sublattice_id: str = ""
    cell_index: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def global_nodes(self) -> np.ndarray:
        R = self.transform[:3, :3]
        t = self.transform[:3, 3]
        return (R @ self.geometry.nodes.T).T + t

    def to_pam(self) -> PAMParticle:
        ...

    @classmethod
    def from_pam(cls, pam: PAMParticle) -> InterlinkedParticle:
        ...
```

### 5.2 Declarative Protocol & Global Registry (`cell.py`)

```python
@dataclass(frozen=True)
class BasisParticle:
    geometry: ParticleGeometry
    fractional_offset: np.ndarray
    sublattice_id: str = ""
    orientation_fn: Callable[[tuple[int, int, int], np.ndarray, dict], np.ndarray] = ...
    metadata: dict[str, Any] = field(default_factory=dict)

class InterlinkedCell(Protocol):
    name: str
    family: str
    parent_network: str
    basis_particles: list[BasisParticle]
    coordination_number: int
    neighbor_catenation_offsets: list[tuple[int, int, int]]

    def forward_clearance(self, unit_cell_pitch: float, strut_diameter: float, **kwargs) -> float: ...
    def resolve_pitch(self, target_clearance: float, strut_diameter: float, **kwargs) -> float: ...
    def instantiate_site(self, grid_index, site_origin, cell_pitch, id_start, context=None) -> list[InterlinkedParticle]: ...

class InterlinkedRegistry:
    @classmethod
    def register(cls, name: str): ...
    @classmethod
    def get(cls, name: str) -> type[InterlinkedCell]: ...
    @classmethod
    def list_cells(cls) -> list[dict[str, Any]]: ...
```

### 5.3 Pilot Reference Cell: `C6TTCell`
Fully implemented and verified in Phase 1:
- Registered as `"c6tt"` / `"c-6-tt"`.
- Truncated Tetrahedron wireframe (12 vertices, 18 struts) on Simple Cubic (`pcu`).
- Coordination number 6 with 6 Cartesian catenation offsets.
- Validated clearance law: $\Delta = 0.131370 \cdot a_0 - D_{\text{strut}}$.

---

## 6. Phased Execution Roadmap

The implementation proceeds in five modular phases:

| Phase | Milestone | Deliverables / Status |
| :--- | :--- | :--- |
| **Phase 1 (DONE)** | **Core Foundation & Particle Abstraction** | [`particle.py`](../graphite/explicit/interlinked/particle.py), [`cell.py`](../graphite/explicit/interlinked/cell.py), `C6TTCell` pilot, bridge methods, and [`tests/test_interlinked_core.py`](../tests/test_interlinked_core.py) (all 7 passed, all regression tests green). |
| **Phase 2 (DONE)** | **Modular Cell Declarations** | Registered `D4TetCell` (diamond bipartite dual), `J4OctCell` (twisted octahedron), `European4in1Cell` (alternating tilt), `JapaneseKusariCell` (orthogonal flat/arch), and `NasaSpaceFabricCell` into the registry. [`tests/test_interlinked_cells.py`](../tests/test_interlinked_cells.py) (all 16 passed). |
| **Phase 3 (DONE)** | **Analytical Primitive Seeding & Boundaries** | Implemented Cartesian, Staggered, Hexagonal, and Diamond cubic coordinates, Cylindrical wrap with exact $2\pi R = N_\theta \cdot a_\theta$ pitch quantization, Spherical Fibonacci shells, and the 3-tier boundary management engine (Policy A/B/C). [`tests/test_interlinked_seeding.py`](../tests/test_interlinked_seeding.py) (all 12 passed). |
| **Phase 4 (DONE)** | **Two-Tier Vectorized Clearance & Inversion** | Two-tier KD-tree broad phase + vectorized segment-segment and circle-circle narrow phase, Gauss linking integral, and automated closed-loop `resolve_pitch()` inverse solver. [`tests/test_interlinked_clearance.py`](../tests/test_interlinked_clearance.py) (all 14 passed). |
| **Phase 5 (DONE)** | **Instanced 3MF Solidification & Unified API** | Production-grade instanced 3MF package exporter ([`writer_3mf.py`](../graphite/explicit/interlinked/writer_3mf.py)) achieving 98.6% file compression. Unified top-level [`generate_interlinked_lattice()`](../graphite/explicit/interlinked/generator.py) API integrating all 5 phases. [`tests/test_interlinked_3mf.py`](../tests/test_interlinked_3mf.py) (all 9 passed). |

---

## 7. Verification Checklist for Subsequent Phases

Any implementation phase executed in this workspace must ensure:
- [x] `python scripts/export_core_spec.py --check` passes cleanly.
- [x] All unit tests in `tests/test_interlinked_core.py`, `tests/test_pam_c6tt.py`, `tests/test_pam_polyhedra.py`, and `tests/test_interlinked_lattice.py` pass without regression.
- [x] Multi-body files write exclusively to `outputs/` (never `test_parts/`).
- [x] Adding a new interlinked cell type requires $< 100$ lines of declarative code registering an `InterlinkedCell` with no bespoke generator scripts.
