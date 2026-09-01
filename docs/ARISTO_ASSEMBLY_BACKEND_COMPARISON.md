# Aristo FEA assembly backend — planning brief

**Audience:** Planning / architecture specialist helping choose between Graphite’s current **custom `scipy.sparse` + PARDISO** FEA path and **scikit-FEM**.

**Decision scope:** Linear static elasticity assembly and solve for **P1 Tet4** volume meshes on **lattice / TPMS** parts. This document does **not** compare mesh generators (Gmsh), implicit field evaluation, or topology optimization stacks.

**Project intent (documented):** Keep the current backend for now; **migrate to scikit-FEM eventually** if benefits outweigh integration cost. See [ARISTO.md](ARISTO.md) § “Future direction: scikit-FEM”.

**Date context:** June 2026 — Aristo default input is now **implicit field → MC skin (in memory) → Gmsh `single_surface` → FEA**, with STL as a developer override.

---

## 1. What Aristo actually needs

| Requirement | Detail |
|-------------|--------|
| Physics | 3D linear static isotropic elasticity |
| Elements | **P1 Tet4 only** in production (Tet10 export exists for debug; assembly is P1-only today) |
| Units | mm, MPa (= N/mm²), N |
| Mesh source | Gmsh volume mesh → raw `nodes (N×3)`, `elements (M×4)`, surface triangles |
| BCs | Custom heuristics on porous lattices: bottom cap fixed, top cap loaded (`flat_top`, `flat_top_vertex_plane`) |
| Loads | Uniform pressure on detected load-face subset (not a simple box BC) |
| Postprocessing | Element + nodal von Mises, mesh quality gates (`quality_ok`), VTU/PNG export |
| Scale | ~75k tets (Mirae V4 regression) to ~1.2M tets (fine implicit dev cubes @ h=0.005) |
| Solver | Sparse **direct** solve; **PARDISO** (via `pypardiso`) when available, else SciPy `spsolve` |

**Critical separation:** Most engineering effort went into **meshing and BC detection on lattice STLs**, not into hand-rolling FEM math. Either backend still needs Gmsh + custom BC/QC glue.

---

## 2. Option A — Current stack (custom assembly + SciPy / PARDISO)

### Architecture

```
Gmsh Tet4 mesh (numpy arrays)
  → custom vectorized K assembly (stiffness_assembly.py)
  → custom pressure load vector (aristo_solver.py)
  → apply BC masks → sparse K, F
  → PARDISO or scipy.sparse.linalg.spsolve
  → custom element stress + nodal recovery + quality gates
```

**Code:** `graphite/aristo/stiffness_assembly.py`, `linear_solver.py`, `aristo_solver.py`  
**Dependencies:** `numpy`, `scipy`, optional `pypardiso` ([requirements-aristo.txt](../requirements-aristo.txt))

### Pros

| Category | Benefit |
|----------|---------|
| **Dependencies** | No scikit-FEM in the core Aristo install; stack is already required elsewhere (`numpy`, `scipy`, `gmsh`) |
| **Scope match** | Implements exactly one element (P1 tet) and one physics (linear elasticity) — no unused FEM framework surface area |
| **Performance** | Fully vectorized NumPy/`einsum` assembly over all elements; assembly is not the bottleneck at current scales |
| **Solver choice** | PARDISO integration is first-class (`fea_linear_solver`, ~20–30 s solves on ~1M DOF in recent implicit-cube runs) |
| **Control** | Full visibility into `K` assembly, element B-matrices, von Mises, and which tets enter stress statistics |
| **Lattice-specific QC** | Poor-tet gates, “giant element” rejection, and `quality_ok` VTU fields are tightly coupled to raw element arrays |
| **Regression stability** | Mirae V4 baseline is pinned to this path ([ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)) |
| **Debugging** | Easy to inspect intermediate arrays; no framework abstraction between mesh and DOF numbering |

### Cons

| Category | Drawback |
|----------|----------|
| **Maintenance** | Project owns correctness of FEM formulation (B matrix, Voigt ordering, symmetry, load consistency) |
| **Feature growth** | P2 elements, hex elements, thermal, contact, or multi-physics require significant new code |
| **BC abstraction** | No standard `condense`/forms API — BCs and loads are custom Python on face masks |
| **Ecosystem** | Does not plug into scikit-topt / standard FEM tutorials without an adapter layer |
| **Testing** | No external reference implementation to diff against (beyond manual Mirae numbers and spot checks) |
| **Team onboarding** | New contributors must read custom assembly code rather than skfem docs/examples |
| **Duplication** | `graphite/topt` already uses scikit-FEM — two FEM styles in one repo |

---

## 3. Option B — scikit-FEM

### Architecture (typical)

```
Gmsh Tet4 mesh
  → skfem.MeshTet(nodes.T, elements.T)   # note transpose convention
  → Basis(mesh, ElementVector(ElementTetP1()))
  → linear_elasticity + asm → K
  → condense/solve for BCs
  → postprocess via skfem helpers + custom viz
```

**Existing usage in repo:** `graphite/topt/` (topology optimization), `scripts/testbed_scikit_fem_compression.py` (`.inp`/`.msh` validation)  
**Dependencies:** `scikit-fem>=11,<12` ([requirements-topt.txt](../requirements-topt.txt)) — **not** in `requirements-aristo.txt` today

### Pros

| Category | Benefit |
|----------|--------|
| **Correctness confidence** | Well-tested P1 (and P2, hex, etc.) forms; standard weak-form API |
| **Maintainability** | Less custom FEM math to own; upgrades follow library releases |
| **Extensibility** | Natural path to Tet10, mixed meshes, additional physics if product needs them |
| **BC/load API** | `condense`, `solve`, boundary DOF selection — documented patterns |
| **Unification** | Same FEM library as `graphite/topt`; one conceptual model for FEA + optimization |
| **Community** | Examples, papers, and support for skfem-based workflows |
| **Validation** | Can cross-check against testbed script and external `.inp` meshes via meshio |
| **Future adjoints** | If sensitivity analysis or optimization-on-lattice is needed, skfem integration is already proven in topt |

### Cons

| Category | Drawback |
|----------|----------|
| **Dependency** | Adds `scikit-fem` to Aristo production requirements; version pinning and CI matrix |
| **Still need custom glue** | Gmsh tag remapping, connected-component cleanup, lattice BC heuristics, pressure on partial caps, quality gates — **skfem does not replace these** |
| **Mesh conversion** | Must maintain reliable `numpy ↔ MeshTet` adapter (DOF ordering, 0/1-based indices, transpose layout) |
| **Performance uncertainty** | Assembly may be fine but needs benchmarking vs vectorized custom code at 75k–1.2M tets |
| **Solver** | skfem typically still uses SciPy sparse solvers under the hood; PARDISO would remain a separate integration |
| **Postprocessing** | Nodal stress recovery, `quality_ok` masking, and VTU export likely stay custom |
| **Migration cost** | Regression re-baseline (Mirae V4 + implicit cases); dual-backend period or flag |
| **Abstraction leak** | Debugging lattice issues may require understanding both skfem Basis DOF maps and Aristo mesh QC |

---

## 4. Side-by-side comparison

| Criterion | Custom scipy + PARDISO (A) | scikit-FEM (B) |
|-----------|----------------------------|----------------|
| **Solves current product need (P1 tet, linear static)** | ✅ Production-proven | ✅ Supported |
| **Extra pip dependency** | None (beyond optional pypardiso) | `scikit-fem` |
| **Lines of FEM code we maintain** | ~400 (assembly + stress) | ~50–150 adapter + config |
| **Gmsh / lattice meshing** | Same | Same |
| **Lattice BC heuristics** | Same custom code | Same custom code |
| **PARDISO support** | ✅ Native | ⚠️ Still custom after assembly |
| **P2 / hex / multi-physics** | ❌ Large effort | ✅ Incremental |
| **Alignment with topt** | ❌ Separate path | ✅ Shared |
| **Mirae regression risk** | Baseline today | Requires re-pin |
| **Implicit-first pipeline (June 2026)** | ✅ Works | ✅ Works (same mesh arrays) |
| **Paper / external audit trail** | “In-house P1 tet code” | “scikit-FEM P1 tet” |

---

## 5. What neither option fixes

These dominated past meshing pain and remain **orthogonal** to the assembly choice:

- Gmsh `classifySurfaces` failures on marching-cubes / union STLs → mitigated by **implicit-first + `single_surface`**
- Through-wall resolution → requires **fine enough `h`**, not a FEM library change
- MC floating islands → **island cleanup** before volume mesh
- Cap/load face detection on porous tops → **boundary_detection.py**
- Sliver tet stress speckling → **mesh quality gates** + optional surface remesh

A planner should not expect scikit-FEM to reduce Gmsh or lattice-topology work.

---

## 6. Hybrid / phased options

| Strategy | Description | When it makes sense |
|----------|-------------|---------------------|
| **A only (status quo)** | Keep custom assembly indefinitely | Team is small; P1-only forever; minimize deps |
| **B replace A** | Single skfem backend | Want one FEM story; plan P2 or topt convergence |
| **Dual backend (flag)** | `fea_assembly_backend=scipy\|skfem` | Safe migration with regression parity tests |
| **B for new features only** | Keep A for Mirae; skfem for new physics | Risk-averse; regression frozen on A |
| **Shared mesh adapter only** | Extract `numpy → skfem.MeshTet` module first | De-risk before swapping assembly |

**Documented project preference:** dual-backend migration with flag, then default to skfem if parity holds.

---

## 7. Suggested evaluation plan (for the specialist)

1. **Parity tests** — Same Gmsh mesh, same BCs/loads: compare max displacement and nodal von Mises P99 between A and B (Mirae V4 @ 1 N; one implicit cube @ h=0.005).
2. **Timing** — Assembly time, solve time, peak RAM at 75k and ~1.2M tets.
3. **DOF mapping audit** — Verify load patch area and reaction force match (conservation check).
4. **Dependency review** — Python version matrix (Aristo uses 3.13 in some envs; Open3D limited to ≤3.12; skfem 11.x compatibility).
5. **Roadmap fit** — Ask whether product needs P2, hex, thermal, or optimization sensitivities within 12 months.
6. **Ownership** — Who maintains custom assembly vs skfem upgrade churn?

**Pass criteria (example):** nodal displacement within 0.1%; nodal von Mises P99 within 2%; no regression in load patch area; solve time within 2×.

---

## 8. Recommendation framing (neutral)

| If the priority is… | Lean toward… |
|---------------------|--------------|
| Minimal dependencies, frozen P1 scope, pinned Mirae regression | **A (keep current)** |
| Long-term FEM extensibility, alignment with topt, less owned math | **B (scikit-FEM)** |
| Risk management during transition | **Dual backend + parity tests** |

**Not recommended:** Full rewrite of Gmsh meshing or BC layers as part of a skfem migration — low ROI.

---

## 9. Key repository references

| Topic | Location |
|-------|----------|
| Custom assembly | `graphite/aristo/stiffness_assembly.py` |
| PARDISO / SciPy solve | `graphite/aristo/linear_solver.py` |
| FEA orchestration | `graphite/aristo/aristo_solver.py` |
| skfem testbed | `scripts/testbed_scikit_fem_compression.py` |
| skfem in optimization | `graphite/topt/topt_solver.py`, [TOPOLOGY_OPTIMIZATION.md](TOPOLOGY_OPTIMIZATION.md) |
| Meshing (Gmsh) | [ARISTO_MESHING.md](ARISTO_MESHING.md) |
| Accuracy / implicit path | [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md) |
| Implicit-first input | `graphite/aristo/implicit_input.py` |
| Aristo requirements | `requirements-aristo.txt` |
| Topt / skfem requirements | `requirements-topt.txt` |

---

## 10. Questions for the planning specialist

1. Is **P1 Tet4 linear static** the long-term ceiling, or do we need **P2 / hex / multi-physics** within a defined horizon?
2. Should Aristo and **topology optimization (`topt`)** share one FEM dependency tree in production?
3. What **regression tolerance** is acceptable when swapping assembly backends?
4. Is **PARDISO** a hard requirement for large implicit meshes (~1M+ tets), regardless of assembly library?
5. Do we want **external auditability** (skfem) vs **minimal dependency surface** (custom)?
6. Who owns ongoing validation when **scikit-fem major versions** ship?

---

*This is a planning document only — no backend change is implied until parity tests and a migration schedule are approved.*
