# Explicit Lattice Capabilities — Technical Documentation

This document records the **current state** of explicit (discrete strut / node) lattice generation in the Graphite repository: which components exist, where they live, what dependencies they use, and how boundary conformance is achieved. It complements the high-level roadmap in `ROADMAP.md` and the original design spec in `lattice_generator_spec.md`.

---

## 1. Executive summary

| Question | Answer |
|----------|--------|
| Is there an Explicit engine inside `graphite/`? | **No.** `graphite/explicit/` contains only a placeholder `__init__.py`. The Streamlit app and packaged `graphite` modules today focus on **implicit** TPMS workflows. |
| Does explicit lattice code exist in `graphite/`? | **Yes (core pipeline).** The canonical scaffold/topology/geometry modules live under **`graphite/explicit/`** (`scaffold_module.py`, `topology_module.py`, `geometry_module.py`). Additional experiments and routers remain in sibling packages (`Universal_Lattice_Engine`, `Supercell_Modules`, etc.). |
| Primary geometry kernel for struts | **manifold3d** (cylinders, batch union, boolean intersection). |
| Primary mesh / IO / containment | **trimesh** |
| Volumetric scaffold | **GMSH** (conformal tetrahedral meshes) |

**Implication:** Integrating “Explicit” into the `graphite` package and UI is largely a **wiring and API-surface** task on top of existing engines, not a greenfield algorithm effort—though packaging, defaults, tests, and UX still need to be built.

---

## 2. Terminology

- **Explicit lattice** — A structure defined by **nodes** (3D points) and **struts** (undirected or directed pairs of node indices), later thickened into cylinders (and optionally spheres) to form a printable solid.
- **Scaffold** — A background volumetric discretization (classically a **tetrahedral mesh**) used to place nodes and infer connectivity.
- **Micro-rule / topology rule** — A function that, given one cell (e.g. one tetrahedron or one hex), emits local nodes and struts; results are merged and deduplicated globally.
- **Trimming / masking** — Removing or clipping geometry that lies outside the part. This repo uses several distinct mechanisms (graph-level trim vs solid boolean), described in §6.

---

## 3. Status of `graphite/explicit/`

The package path **`graphite/explicit/`** holds the **primary explicit lattice pipeline** and a small public API:

- **`graphite/explicit/__init__.py`** — re-exports `generate_conformal_scaffold`, `generate_topology`, and `generate_geometry`.
- **Submodules** — `scaffold_module.py`, `topology_module.py`, `geometry_module.py` (import as `graphite.explicit.scaffold_module`, etc., for types and helpers such as `ScaffoldResult`, `manifold_to_trimesh`).

Hex/tet **routers** and sandboxes may remain under `Universal_Lattice_Engine/`, `Hex_Sandbox/`, etc., and call into `graphite.explicit.geometry_module` where needed.

---

## 4. Core pipeline modules (`graphite/explicit/`)

These Python modules form the main **conformal tet → topology → solid** story described in `docs/lattice_generator_spec.md`.

### 4.1 `graphite/explicit/scaffold_module.py` — Conformal tetrahedral meshing (GMSH)

**Purpose:** Turn a watertight **boundary** triangle mesh into a **volume-filling tetrahedral mesh** suitable for topology synthesis.

**Responsibilities:**

- Interface with the **GMSH** Python API.
- Export/merge boundary, define volume regions, set mesh sizing.
- Return structured NumPy data: global **node coordinates**, **tet connectivity** (4 nodes per element), and **surface face** connectivity for the outer boundary.

**Libraries:** `gmsh`, `numpy`, `trimesh` (boundary handling / export paths as implemented).

**Boundary behavior:** The scaffold is **conformal to the meshed domain** defined from the input STL—not a simple axis-aligned bounding box fill. Coordinate scaling / “smart inset” behaviors are documented in the module docstring (e.g. alignment with original CAD bounds after scaling).

---

### 4.2 `graphite/explicit/topology_module.py` — Tet scaffold → global node/strut graph

**Purpose:** Convert the tetrahedral scaffold into a single **global graph** of nodes and struts according to a chosen **recipe**.

**Documented recipes / themes:**

- Multiple lattice “families” (e.g. rhombic, Voronoi, Kagome, icosahedral—see module docstring and code).
- **Surface handling:** Includes logic for **surface dual / cage** style connections (e.g. adjacent boundary face centroids) so the outer skin is better behaved than raw internal-only struts.

**Libraries:**

- **`numpy`** — vectorized geometry (centroids, midpoints, keys for face deduplication).
- **`scipy.sparse` / `scipy.sparse.csgraph`** — graph connectivity and component filtering (e.g. removing small disconnected “floaters”).

**Note:** A workspace-wide search did not show **NetworkX** imports in `.py` files; graph work here uses **SciPy** sparse graph routines.

---

### 4.3 `graphite/explicit/geometry_module.py` — Strut solid generation and boundary trim

**Purpose:** Turn `(nodes, struts)` into a **watertight-style** triangle mesh suitable for STL export, using **manifold3d** CSG.

**Capabilities (see module docstring):**

- Per-strut **cylinders** (uniform or per-strut radius arrays).
- Optional **spheres** at nodes for stress relief / smoother joints.
- **Batch union** of primitives (not sequential per-strut booleans in the intended path).
- Optional **`crop_to_boundary`**: **boolean intersection** of the merged lattice solid against a **`trimesh.Trimesh`** boundary.
- Optional fast path returning **`manifold3d.Manifold`** (and volume) without immediate conversion to `trimesh` for optimization loops.

**Libraries:** **`manifold3d`**, **`numpy`**, **`trimesh`**.

**Boundary behavior:** When enabled, this is **true conformal clipping** of the **solid** lattice against the boundary mesh (manifold pipeline), distinct from only dropping struts whose endpoints fail a point-in-poly test (see §6).

---

### 4.4 `solver.py` — Target solid fraction (iterative)

**Purpose:** Adjust strut thickness (or related parameters) so that the **lattice solid** inside the boundary approaches a **target volume fraction**, using the geometry module’s volume evaluation paths.

**Libraries:** Depends on **`trimesh`**, **`numpy`**, and **`graphite.explicit.geometry_module`** / topology outputs as wired in the solver functions.

---

### 4.5 `io_module.py` — Input boundary preparation

**Purpose:** Load STL (or similar), repair/consistency checks, report **bounding box** and volume—**feeds** the conformal pipeline.

**Libraries:** **`trimesh`** (per project constraints in `lattice_generator_spec.md`).

**Boundary behavior:** Prepares the **actual part surface** for GMSH and downstream steps; not lattice trimming by itself.

---

## 5. Satellite packages and experiment trees

These directories contain **additional explicit or hybrid workflows**, tests, and prototypes. They are important for understanding “what we already know how to do” even if they are not yet imported from `graphite`.

### 5.1 `Universal_Lattice_Engine/`

**Role:** “Universal router” style logic: apply named **micro-rules** per **tet** (or hex in tests), merge into a global **`nodes` / `struts`** array, then optionally build **manifold3d** solids in tests.

**Notable locations:**

- `Universal_Lattice_Engine/core/tet_rules.py` — per-tet rule implementations.
- `Universal_Lattice_Engine/core/topology_module.py` — orchestration / merging patterns used by the engine.
- `Universal_Lattice_Engine/tests/` — end-to-end checks (e.g. tet compatibility, hex STL, pipeline rounding).

**Libraries:** **`numpy`**, **`manifold3d`**, **`trimesh`**, **GMSH** (in hex/tet diagnostic tests).

---

### 5.2 `Hex_Sandbox/`

**Role:** **Hexahedral** cell–local rules: from a structured hex grid, emit **nodes and struts** for specific topologies.

**Notable file:**

- `Hex_Sandbox/core/hex_rules.py` — defines local connectivity for cube/hex-based lattices.

**Libraries:** **`numpy`** (and test harnesses may use **`manifold3d`** for visualization/export).

---

### 5.3 `Supercell_Modules/`

**Role:** Supercell construction, Cartesian grids, **topologies**, and **graph-level trimming** against an STL.

**Notable files:**

- `Supercell_Modules/core/grid_generator.py` — **`generate_cartesian_nodes`** (bbox-based grid) and **`trim_to_stl`**, which filters the **node/strut graph** using **`trimesh.contains`** (watertight requirement noted in docstring).
- `Supercell_Modules/core/lattice_rules.py`, `factory.py`, `topologies.py` — rule application and topology variants.
- `Supercell_Modules/tests/test_supercell.py` — pipelines that distinguish **boundary-intersecting** vs **internal** struts, export core vs clipped final STL via **`graphite.explicit.geometry_module.generate_geometry`**.

**Libraries:** **`numpy`**, **`trimesh`**, **`manifold3d`** (via geometry module), **`pyvista`** / **matplotlib** in some visualization tests.

---

### 5.4 `Conformal_Mesh_Exploration/`

**Role:** **Conformal hex** placement driven by a **voxelizer**-style pipeline (project naming: “conformal hexes”), then application of truss-like rules in tests.

**Notable file:**

- `Conformal_Mesh_Exploration/core/voxelizer.py` — **`generate_conformal_hexes`** and related logic.

**Tests** (e.g. `test_conformal_stl_export.py`, `test_hard_curves.py`) combine this with **`trimesh`**, **proximity** queries, and export paths.

**Boundary behavior:** **Conformal placement** of cells inside curved boundaries; distinct from the GMSH+tet mainline but still “explicit” in the sense of discrete cells and struts.

---

### 5.5 `experiments/` and `scripts/`

**Role:** One-off and archival drivers: e.g. debug Kagome on a box, production batch scripts, solver demos.

**Typical pattern:** Load or create a **`trimesh`** boundary → **`generate_conformal_scaffold`** → **`generate_topology`** (or equivalent) → **`generate_geometry`** / export.

---

### 5.6 `PreviousTPMSGeneration/` and legacy generators

**Role:** Historical R&D scripts; **`General_Lattice_Generator.py`** mixes **implicit** lattice concepts with **boolean** mesh operations. It is **not** the canonical GMSH + strut explicit pipeline, but it documents earlier approaches to clipping and voxel-related trimming in that codebase.

---

## 6. Bounding-box masking vs conformal boundary handling

This section distinguishes mechanisms that are often conflated.

### 6.1 Axis-aligned bounding box (graph seeding)

**Where:** e.g. `Supercell_Modules/core/grid_generator.py` (`generate_cartesian_nodes`).

**What:** Nodes are created on a **regular grid inside a bbox**. Struts follow Cartesian or supercell rules.

**Limitation:** The **scaffold** is box-shaped; conformity to an **organic** STL comes only after **trimming** or **solid booleans**.

---

### 6.2 Graph-level trim (`trim_to_stl`)

**Where:** `Supercell_Modules/core/grid_generator.py` — **`trim_to_stl(nodes, struts, stl_mesh)`**.

**What:** Uses **`trimesh.contains`** (ray-casting / point-in-mesh) to **drop or remap** nodes and struts that fall outside the watertight boundary.

**Nature:** **Topological / discrete** filtering—not thickening struts into solids first. Fast for graph culling; may not match the exact **surface** of the part at sub-strut resolution until solids are built and clipped.

---

### 6.3 Conformal tet scaffold (GMSH)

**Where:** `graphite/explicit/scaffold_module.py`.

**What:** Volume mesh **respects the boundary** you give GMSH; topology is derived from **elements inside the part**.

**Nature:** **Conformal** at the meshing resolution; quality depends on GMSH settings and watertight input.

---

### 6.4 Solid boolean intersection (`crop_to_boundary`)

**Where:** `graphite/explicit/geometry_module.py` — **`generate_geometry(..., boundary_mesh=..., crop_to_boundary=True)`**.

**What:** Union strut cylinders (and optional spheres), then **intersect** with the boundary using **manifold3d**.

**Nature:** **Conformal clipping of the final solid**—the strongest guarantee among the listed methods for “no struts sticking out” **if** the boundary and manifold pipeline are well-conditioned.

---

### 6.5 Conformal hex / voxel-style exploration

**Where:** `Conformal_Mesh_Exploration`.

**What:** Places **hex-like** cells conformally inside geometry; subsequent strut rules operate on those cells.

**Nature:** Hybrid **discrete cell** + **conformal placement**; different performance and fidelity tradeoffs vs GMSH tets.

---

## 7. Visualization and debugging

- **`pyvista`** — Used in several Supercell and experimental tests for **3D visualization** of struts and boundaries (wireframe/surface styles).
- **`graphite/geometry/surface_picking.py`** — **PyVista**-based picker for **implicit / CAD face ID** workflows; not the explicit strut engine, but part of the broader Graphite UX.

---

## 8. Documentation and specs (reference)

| Document | Relevance |
|----------|-----------|
| `docs/lattice_generator_spec.md` | Target architecture for conformal explicit lattices (GMSH, manifold3d, anti-patterns). |
| `LATTICE_ARCHITECTURE.md` | Topology / micro-rule narrative (Delaunay, rules, manifold export). |
| `docs/experimental_tet_lattices.md` | Ideas for future tet-rule variants (jitter, chiral, hybrid rules). |
| `ROADMAP.md` | Product direction; Explicit engine UI integration listed under **Future Horizons**. |

---

## 9. Recommended next steps (for `graphite` integration)

When implementing **`graphite.explicit`** for real:

1. **Choose a canonical backend** for v1 (e.g. GMSH scaffold + `graphite.explicit.topology_module` + `graphite.explicit.geometry_module`) vs supercell bbox + `trim_to_stl`—document tradeoffs for users (organic parts vs speed).
2. **Extend facades** in `graphite/explicit/__init__.py` (or submodules) with stable signatures for the UI (paths in, `trimesh` / STL out, typed options) as needed.
3. **Centralize dependencies** in `pyproject.toml` / requirements where missing (GMSH, manifold3d).
4. **Add integration tests** that mirror `experiments/` scripts but run under `pytest` from CI.
5. **Extend Streamlit** (or a CLI per roadmap) to select explicit mode without duplicating parameter state.

---

## 10. Revision history

| Date | Note |
|------|------|
| (this file) | Initial verbose documentation from workspace scan. |
| — | Core explicit modules migrated into `graphite/explicit/`; public API exports the three primary entrypoints. |

---

*End of document.*
