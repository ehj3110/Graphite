# Handoff: 3D Auxetics, Interlinked Lattices & Proposed PAMs

**Audience:** Planning AI / literature-informed architect drafting the next PAM (Polycatenated Architected Materials) phase for Graphite.  
**Status:** Capability inventory + architecture constraints. **No PAM/`D-4-TET` code exists yet.**  
**Date context:** 2026-09 (repo state at handoff authorship).

Agents implementing code: start at [`AGENTS.md`](../AGENTS.md) → [`graphite/generators/README.md`](../graphite/generators/README.md) and/or [`graphite/explicit/interlinked/README.md`](../graphite/explicit/interlinked/README.md). Do **not** invent `graphite/core/`, `graphite/tiling/`, or `graphite/validation/` packages without revisiting the conflicts below.

---

## 1. Product framing (what Graphite already is)

Graphite is a Python/NumPy conformal lattice R&D stack. For kinematic / auxetic / chainmail work, three layers matter:

| Layer | Package | Role |
|-------|---------|------|
| **Welded explicit struts** | `graphite.explicit` | Continuous graphs: `nodes (N,3)`, `struts (M,2)`, radii → Manifold solids via `generate_geometry` |
| **Interlinked (non-welded)** | `graphite.explicit.interlinked` | Discrete closed rings / multi-body cells with **Δ > 0** clearance; print-in-place chainmail & space fabric |
| **High-compliance generators** | `graphite.generators` | Pentamodes, plate-lattices, interlocking auxetic sheets, rotating squares |

**Canonical naming:** connectivity arrays are called **`struts`**, not `edges` (see [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md)). `LatticeGraph` lives in `graphite.generators.pentamode` (`nodes`, `struts`, `radii`, `metadata`) — there is **no** `graphite/core/types.py`.

**Outputs rule:** generated STLs/reports go under `outputs/` only. `test_parts/` is read-only CAD fixtures (e.g. NASA hexagon STL).

---

## 2. Capabilities already implemented

### 2.1 Interlinked engine — `graphite.explicit.interlinked`

Production subsystem for **kinematic, non-welded multi-body** scaffolds.

| Module | Capability |
|--------|------------|
| `patterns.py` | `Ring` dataclass; European 4-in-1; Japanese Kusari; cubic 8-ring cell; volumetric Kusari |
| `clearance.py` | Vectorized segment–segment distance; surface clearance Δ = d − (r₁+r₂); pairwise checks; **Gauss linking number**; Lipschitz clamps on graded pitch/radius |
| `conformal.py` | Planar seeding; surface-conformal frames; **SDF inset culling** (drop whole rings so none are cut at CAD boundaries) |
| `generator.py` | `InterlinkedConfig` → `InterlinkedLatticeResult`; stacks rings → `generate_geometry` Manifold mesh |
| `importer.py` | Multi-body STL decompile; hexagonal sheet/rosette seeds; tessellation with decoupled component scaling |
| `nasa_hexagon.py` | Parametric NASA JPL 6-fold space fabric + graded sheets |

**Default DfAM knobs (interlinked):** `min_clearance ≈ 0.30 mm` (config default); NASA fabric verified ~**0.185–0.192 mm** at pitch **12.75 mm**. Auxetic interlocking sheets typically target **≥ 0.40 mm**.

**Patterns (topology vocabulary already in code):**

- **European 4-in-1** — square grid, alternating tilt (~28°), adjacent rings link with |Lk| = 1.
- **Kusari** — flat XY rings + vertical XZ/YZ arch links.
- **Cubic 8-ring** — 8 interlocking rings per cube cell (omnidirectional kinematic cage).
- **Volumetric Kusari** — 3D periodic extension along X/Y/Z.
- **NASA hexagon** — plate + top torus + six hook arms; hexagonal close-pack tiling.

### 2.2 Interlocking auxetics — `graphite.generators.interlocking`

Discrete **list[trimesh.Trimesh]** assemblies (one body per tile), not a single welded graph.

| Topology string | What it builds |
|-----------------|----------------|
| `reentrant_bowtie` | 2D sheet: 8-point re-entrant frame + wishbone arms + cardinal eyelets; checkerboard ring orientation for loop-in-loop. 3D `(nx,ny,nz)`: six cardinal eyelets, parity `(i+j+k) mod 2`, isotropic chainmail linking |
| `hook_array` / `nasa_hexagon` | Tessellated NASA hexagon cells |
| `european_ring` | One mesh per European ring (revolved tori) |
| `kusari_ring` | One mesh per Kusari ring |

APIs: `generate_interlocking_auxetic_sheet`, `combine_interlocking_meshes`, `verify_interlocking_clearance` (Manifold boolean intersection volume + `min_gap`).

**Validated behaviors (tests):** watertight bodies; zero adjacent boolean collision; min clearance > ~0.30 mm on 3×3 bowtie sheets; 2D and 3D volumetric grids.

### 2.3 Rotating rigid squares — `graphite.generators.rotating_auxetics`

Grima & Evans (2000) mechanism: rigid squares + living hinges; ideal ν = −1; **monolithic** Manifold mesh (not polycatenated). Supports 2D sheets and 3D stacked layers with vertical hinge pins.

### 2.4 Welded / continuous “high compliance” relatives (not PAMs)

Useful literature neighbors already in-repo:

| Generator | Mechanics idea | Output |
|-----------|----------------|--------|
| `pentamode.py` | Milton–Cherkaev diamond cubic (Z=4), biconical struts, optional hierarchical sub-truss | `LatticeGraph` / SDF / mesh |
| `hexagonal_pentamode.py` | Transversely isotropic hex pentamode (G_xy → 0) | graph / SDF / mesh |
| `plate_lattice.py` | Hashin–Shtrikman-oriented plate lattices (`sc`, `bcc`, `fcc`, `sc_bcc`) | mesh + thickness calibration |
| `graphite.explicit.chiral_cell` + surface lattice | Tetra-/tri-chiral, anti-chiral, re-entrant honeycomb → plates/cylinders | welded strut solids |

Docs: [`docs/EXPLICIT_CHIRAL_AUXETIC_LATTICES.md`](EXPLICIT_CHIRAL_AUXETIC_LATTICES.md), [`docs/CUSTOM_CELLS_AND_INTERLINKED.md`](CUSTOM_CELLS_AND_INTERLINKED.md), [`graphite/generators/README.md`](../graphite/generators/README.md).

### 2.5 Experimental sandbox (not production)

[`experiments/explicit_auxetic/`](../experiments/explicit_auxetic/README.md) — welded re-entrant hex strut cells (θ-parameterized elbows). Generates geometry only; **Poisson ratio not FEA-validated**. Do not treat as PAM or as the interlocking path.

---

## 3. What we have already tried / shipped as reviews

| Artifact / script | Intent |
|-------------------|--------|
| `scripts/generate_interlinked_review.py` | European 4-in-1, Kusari, 2×2×2 cube STLs/PNGs → `outputs/interlinked_review/` |
| `scripts/generate_custom_cells_review.py` | Imported + procedural NASA sheets, grading demos |
| `scripts/inspect_cell_stl.py` | Body count, genus, C₄/C₆ symmetry, pitch estimate, code emission |
| `scripts/generate_tiled_interlocking_auxetic.py` | Large tiled bowtie chainmail sheets |
| `scripts/generate_auxetic_cube_2x2x2.py` | 2×2×2 volumetric interlocking auxetic cube (L=1 cardinal linking) |
| `scripts/generate_rotating_auxetics_review.py` | 4×4 θ=25° sheet + 3×3×2 rotating cubes → `outputs/rotating_auxetics_review/` |
| Tests | `tests/test_interlinked_lattice.py`, `tests/test_interlocking_auxetics.py`, `tests/test_rotating_auxetics.py`, `tests/test_custom_cells.py` |

**Engineering lessons already baked in:**

1. **Never cut closed loops at CAD boundaries** — cull whole rings via SDF inset (`SDF(center) ≤ −(R_outer + margin)`).
2. **Clearance ≠ centerline gap** — use surface Δ = centerline min distance − (r₁+r₂); optionally Manifold `min_gap` / boolean intersection for solid meshes.
3. **Linking number** — Gauss integral used to assert true interlocking (|Lk|=1), not merely proximity.
4. **Checkerboard / parity orientations** — required so adjacent eyelets cross rather than collide (2D and 3D auxetic chainmail).
5. **Manifold3D is the production solidizer** — cylinders, spheres, revolve tori, compose/boolean; not OpenCASCADE, not a separate “pure polygonal dilation” package.
6. **Graded interlocks need Lipschitz limits** on pitch and wire radius so larger rings do not bind inside smaller neighbors.

---

## 4. Explicitly **not** implemented (PAM gap) — Phase 1 status

**Phase 1 landed (2026-09):** `graphite/explicit/interlinked/pams.py` with `PAMParticle`, `PAMLatticeResult`, tetrahedral (+ cuboctahedral stub) generators, and `generate_d4tet_interlocked_pair` / `generate_pam_lattice("D-4-TET", repeats=(1,1,1))`. Review STL: `outputs/d4tet_interlocked_pair.stl`. Tests: `tests/test_pam_d4tet.py`.

**Phase 2 (polyhedra) landed:** `generate_cuboctahedral_particle`, `generate_octahedral_particle`, `align_particle_axis`, `C-6-CO` coordination shell (1+6), `J-4-OCT` planar cross (1+4). Review: `outputs/c6co_interlocked_cell.stl`, `outputs/j4oct_interlocked_cell.stl`. Tests: `tests/test_pam_polyhedra.py`.

**D-4-TET volume sizing:** `generate_d4tet_diamond_tiling(..., conventional_cell_size=a)` sets diamond pitch `d=a√3/4` and calibrates tet edge via `calibrate_d4tet_edge_length`. Review block: `outputs/d4tet_20x100x100/` (a=10 mm, repeats 2×10×10, r=0.5). **C-6-CO full SC volume remains paused** (orientation-field obstruction).

Still missing / later phases:

- Full SC / diamond **periodic tilings** for CO/OCT (orientation field beyond local shells)
- `ICO`, `HEX`, `S-6/2-CO`, ring PAMs (`J-4-ring`, `T-6-ring`)
- Packages `graphite/core/`, `graphite/tiling/`, `graphite/validation/` — still **must not** be created


**Closest analogs to PAM literature concepts:**

| Literature PAM idea | Closest Graphite artifact |
|---------------------|---------------------------|
| Ring-based polycatenanes / chainmail | `european_4in1`, `kusari`, cubic 8-ring, volumetric Kusari |
| Jammed / unjammed kinematic DOFs | Interlocking auxetic bowtie (drape → jam narrative in docs; **not** discrete-element sim) |
| Particle independence | `list[Ring]` or `list[Trimesh]` — no KD-tree node merge across bodies |
| Wireframe polyhedral PAM (TET/CO) | **Missing** — would be new particle generators + network placer |
| Continuous dual of diamond network | `generate_pentamode_lattice` (welded meta-fluid, **not** catenated cages) |

---

## 5. Architecture conflicts for planners (do not propose these)

A prior implementation brief proposed greenfield modules that **fight the repo**:

| Proposed | Why it conflicts |
|----------|------------------|
| `graphite/core/types.py` extending `LatticeGraph` | No `core` package; `LatticeGraph` is in `pentamode` with `struts` |
| `graphite/validation/clearance.py` | Clearance already in `explicit/interlinked/clearance.py` (+ interlocking mesh checks) |
| `graphite/meshing/dilation.py` without CAD Booleans | `graphite/meshing` is **legacy GMSH**; production path is `explicit.geometry_module.generate_geometry` + **manifold3d** |
| New top-level `graphite/tiling/` | Seeding/culling already in `interlinked/conformal.py` |
| Ban Manifold/Boolean solids for PAM meshing | Would orphan every existing interlinked/auxetic pipeline |

**Preferred home for true PAMs:** extend `graphite.explicit.interlinked` (particle types + catenation placer + reuse clearance + Manifold export), optionally thin-wrap from `graphite.generators`. Keep continuous lattices’ KD-tree node merge **out** of PAM particle graphs.

**Process rules:** one phase then stop with a reviewable file under `outputs/`; sync [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md) via `python scripts/export_core_spec.py --check` after API/schema changes.

---

## 6. Python stack & packages (planner-relevant)

Core geometry / lattice stack used by interlinked & auxetic paths:

| Package | Use in this domain |
|---------|-------------------|
| **numpy** | Nodes, struts, vectorized distances |
| **scipy** | `cKDTree` neighbor queries (clearance pairing, conformal helpers) |
| **trimesh** | Mesh I/O, concatenate multi-body STLs, watertight checks |
| **manifold3d** | CSG cylinders/spheres/revolve tori, compose, boolean intersection, `min_gap` |
| **pyvista** | Review PNGs / framing via `graphite.viz` (scripts) |

Related optional stacks elsewhere in Graphite (not required to *generate* interlocks, useful for *analysis*):

| Package area | Role |
|--------------|------|
| Aristo FEA (`requirements-aristo.txt`) | Linear-elastic analysis of STLs — **not** used for grading lattices |
| Vocal LBM | Permeability / WSS on lattices |
| Topology opt (`requirements-topt.txt`) | MMA / scikit-topt experiments |

There is **no** OpenCASCADE / CadQuery dependency in the interlinked path. Prefer staying on NumPy + Manifold3D + trimesh for PAM solids.

---

## 7. Key APIs (copy-paste map)

```python
# Interlinked rings (skeleton + clearance + mesh)
from graphite.explicit.interlinked import (
    InterlinkedConfig,
    generate_interlinked_lattice,
    generate_european_4in1_rings,
    generate_kusari_rings,
    generate_cubic_8ring,
    check_ring_clearance,
    gauss_linking_number,
)

# Discrete interlocking auxetic / chainmail bodies
from graphite.generators import (
    generate_interlocking_auxetic_sheet,
    verify_interlocking_clearance,
    combine_interlocking_meshes,
    generate_rotating_squares_lattice,
)

# Welded strut solidization (reuse for polyhedral cage graphs)
from graphite.explicit.geometry_module import generate_geometry
```

Schema source of truth: [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md) § Ring, InterlinkedConfig, InterlinkedLatticeResult, LatticeGraph, interlocking APIs.

---

## 8. Literature mapping hints (for planning with papers)

When mapping academic PAM / auxetic / chainmail designs onto Graphite, use this lens:

1. **RCSR / reticular nets as particle centers** — Graphite already tiles diamond-like graphs for **welded** pentamodes; PAM work should place **independent cages** at those sites without merging nodes (opposite of conformal strut welding).
2. **Catenation coordination n** in `X-n-abc` — analogous to how many neighbors each ring/eyelet links to (European 4-in-1 ≈ planar n=4; cubic 8-ring / 3D bowtie ≈ higher isotropic coordination). New TET/CO particles need explicit symmetry-axis interlocking rules.
3. **Unjammed vs jammed kinematics** — code today produces **geometry + clearance**; it does **not** run DEM/FEM contact dynamics. Planning should separate (a) DfAM-valid interlocking geometry from (b) later kinematic/jamming simulation (Aristo or external DEM).
4. **DfAM clearance** — literature often cites ~0.2–0.5 mm for powder/resin processes; repo practice: **0.30 mm** interlinked default, **0.40 mm** auxetic interlocking target, NASA ~**0.19 mm** at known pitch (process-dependent — document process when tightening).
5. **Monolithic auxetics vs polycatenanes** — rotating squares and chiral cylinders are **connected solids**; PAMs and interlocking sheets are **multi-body**. Do not conflate in API design or FEA boundary conditions.
6. **Wireframe dilation** — prefer existing Manifold strut tubing (`generate_geometry`) over proposing a parallel meshing package; multi-body export can concatenate islands or keep `list[Trimesh]` like interlocking.

Canonical external touchstones already reflected in docs/code comments: Grima & Evans rotating squares (2000); Prall & Lakes / Spadoni chiral; Chen et al. re-entrant honeycombs; Milton–Cherkaev pentamodes; Berger / Tancogne-Dejean plate-lattices; NASA JPL space fabric.

---

## 9. Suggested planning constraints (acceptance-oriented)

If drafting the next PAM implementation plan, constrain it to:

1. **One phase** with a reviewable STL/PNG/report under `outputs/` (e.g. two interlocking tetrahedral cages, then stop).
2. **Reuse** `interlinked.clearance` + `generate_geometry` (or Manifold tori/cylinders); do not add `validation/` or `meshing/dilation` packages.
3. **New types** (`Particle`, `PAMGraph` or extend `Ring`-like records) live under `graphite/explicit/interlinked/` (or a submodule), using **`struts`** naming and independent per-particle index spaces.
4. **Tests** mirror existing style: watertightness, zero boolean collision, Δ ≥ target, and (for graphs) zero shared global node indices across particles.
5. **Update** `GRAPHITE_CORE_SPEC.md` + generators/interlinked README when APIs land.
6. Treat **tripartite naming** and RCSR mapping as parameterization on top of the interlinked architecture, not a parallel “Graphite Interlinked rewrite.”

---

## 10. File index (quick open list)

```
graphite/explicit/interlinked/     # primary PAM-adjacent engine
graphite/generators/interlocking.py
graphite/generators/rotating_auxetics.py
graphite/generators/pentamode.py   # LatticeGraph definition
graphite/explicit/geometry_module.py
docs/CUSTOM_CELLS_AND_INTERLINKED.md
docs/EXPLICIT_CHIRAL_AUXETIC_LATTICES.md
graphite/generators/README.md
graphite/explicit/interlinked/README.md
GRAPHITE_CORE_SPEC.md
experiments/explicit_auxetic/      # welded re-entrant sandbox only
test_parts/nasa_fabric_hexagon.stl # fixture, read-only
```

---

## 11. One-sentence summary for the planning AI

Graphite already ships a production **interlinked multi-body / chainmail / NASA fabric** stack and **discrete interlocking 3D auxetic** generators on **NumPy + SciPy + trimesh + Manifold3D**, with clearance and linking-number tooling; **true polycatenated polyhedral PAMs (TET/CO on RCSR nets) are the greenfield gap**, and they should extend `graphite.explicit.interlinked` rather than new `core`/`tiling`/`validation`/`meshing` packages or a Boolean-free dilation fork.
