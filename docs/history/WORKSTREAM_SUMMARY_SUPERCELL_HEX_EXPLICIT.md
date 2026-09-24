# Workstream Summary: Supercell, Hex Sandbox, and Explicit Lattices

**Date:** 2026-03-23  
**Scope:** `Supercell_Modules/`, `Hex_Sandbox/`, explicit lattice work under `graphite/explicit/`, and cross-checks against `docs/` and top-level project docs.

---

## Executive Summary

This repository now has three active but partially overlapping explicit-lattice tracks:

1. **`graphite/explicit/` (primary modern path)**  
   Conformal tet scaffold (GMSH) -> topology graph -> manifold solid sweep.  
   This is the strongest candidate for production and Streamlit integration.

2. **`Supercell_Modules/` (rule-factory + graph trimming + exploratory pipelines)**  
   Useful for topology experimentation and Cartesian/supercell studies; still valuable, but includes compatibility shims and test/archive overlap.

3. **`Hex_Sandbox/` (single-cell hex rule validation)**  
   Clean, focused validation sandbox for hex micro-rules and STL export.

The biggest project-level issue is now **documentation drift**: several docs still describe `graphite/explicit/` as a placeholder or “future” even though it is active and heavily exercised.

---

## A) `graphite/explicit/` Current State (Explicit - Fast)

### What is implemented

- **`graphite/explicit/scaffold_module.py`**
  - GMSH conformal tet meshing from boundary STL.
  - Robust STL import handling (`createGeometry` fallback to `createTopology`).
  - Strict size-lock options and anti-refinement controls.
  - Supports `element_order=1` and `element_order=2`.
  - Quadratic extraction support:
    - Tet type 11 (10-node tets)
    - Surface type 9 (6-node triangles)
  - Post-mesh optimization paths and stability-oriented high-order settings.

- **`graphite/explicit/topology_module.py`**
  - Recipes: rhombic, voronoi, kagome, icosahedral.
  - Connectivity cleanup/watershed to remove disconnected floaters.
  - Now handles second-order boundaries (`surface_faces` shape `(K, 6)`) for curved/hinged cage struts.
  - Interior safety slicing for higher-order elements (`elements[:, :4]` for interior logic).

- **`graphite/explicit/geometry_module.py`**
  - Cylinder/sphere sweep via manifold3d.
  - Batch CSG compose and optional boundary clipping.
  - Conversion helpers between `trimesh` and manifold.

- **`graphite/explicit/__init__.py`**
  - Public API is live:
    - `generate_conformal_scaffold`
    - `generate_topology`
    - `generate_geometry`
  - Docstring has been updated to the **Explicit - Fast (Delaunay)** identity.

### What was validated in this session wave

- Adapter workflows (`Part2_Adapter.stl`) with:
  - linear and quadratic scaffold modes,
  - interior-only / surface-only / full-solid exports,
  - adaptive projection experiments,
  - large-target sizing behavior.
- Multiple purpose scripts were created and run successfully for:
  - split outputs,
  - adaptive curved surface projection,
  - volumetric no-skin generation,
  - super-cell test generation,
  - grid clipping experiments.

### Key caveats observed

- Some STLs are still non-manifold/self-intersecting and can fail GMSH parametrization.
- Certain manifold boolean combinations (especially scripted Kagome clip experiments) can yield **zero-volume intersections** despite non-zero input volumes. This indicates API/operator semantic mismatch and/or manifold validity issues for those specific synthetic constructions.

---

## B) `Supercell_Modules/` Current State

### What it does well

- **Factory-driven topology + rule composition** in `core/factory.py`:
  - topology selection
  - Delaunay tetrahedralization
  - degenerate tet filtering
  - rule application
  - global merge/dedupe
- **Grid and trim tools** in `core/grid_generator.py`:
  - Cartesian node generation
  - graph-level STL trimming (`trim_to_stl`) using `trimesh.contains`
- **Rich topology catalog** in `core/topologies.py`:
  - simple cubic, BCC, bitruncated, truncated oct-tet, rhombicuboct, A15, centroid-dual, etc.
- **Integration testing harness** in `tests/test_supercell.py`:
  - boundary-intersecting vs internal strut classification
  - core-only and clipped final exports
  - practical use of `graphite.explicit.geometry_module`.

### Architectural note

- `core/lattice_rules.py` is currently a **shim** re-exporting from `Universal_Lattice_Engine/core/tet_rules.py`.  
  This works, but increases coupling and makes source-of-truth less obvious.

### Risks / debt

- Heavy `output/` volume with many historical artifacts; hard to identify canonical references.
- `tests/experimental_archive/` still contains useful experiments but mixes “retired” and “active learning” material.

---

## C) `Hex_Sandbox/` Current State

### What it does well

- Clear, compact sandbox for hex local rule logic:
  - `core/hex_rules.py` defines Voronoi/Rhombic/Kagome rules for a single hex.
- Good single-element validation workflow:
  - `tests/test_20mm_hex.py` generates STL exports for each rule and reports segment lengths.
- Produces deterministic outputs with minimal dependencies beyond manifold/trimesh stack.

### Limits

- Not integrated into Streamlit.
- Scope is validation/demo, not yet a full production pipeline with robust clipping, conformity, and parameter handling.

---

## D) Explicit Lattice Experiments Completed (High-Level)

This cycle included extensive explicit R&D scripts, including:

- **Adapter profiling and exports**
  - full curved-skin output
  - interior-only and surface-only splits
  - volumetric no-skin variants
- **Adaptive boundary projection**
  - detect strut deviation from CAD and locally subdivide/project points to surface
  - CSG-swept adaptive export generated
- **Kagome super-cell studies**
  - octet-derived midpoint and face-centroid variants
  - single-cell STL generation confirmed
- **Grid clipping studies**
  - tiled and global FCC-grid approaches
  - manifold conversion resilience improvements
  - diagnostics added for volume and array-shape handling

The net result: the **Explicit - Fast** path is strong for rapid stochastic/conformal generation, while fully deterministic symmetric global Kagome clipping still needs boolean robustness work.

---

## E) Documentation Audit: What Needs Update / Merge / Delete

## Priority 1 (must update now)

- **`ARCHITECTURE.md`**
  - Currently says `graphite/explicit` is placeholder/future.
  - This is no longer true.
  - **Action:** update to reflect live explicit modules and app wiring status.

- **`docs/EXPLICIT_LATTICE_CAPABILITIES.md`**
  - Contains contradictory statements (e.g., “No explicit engine in `graphite/`” vs later sections describing explicit modules).
  - **Action:** normalize to one consistent reality:
    - explicit modules are active
    - clearly separate production path vs experiments.

## Priority 2 (merge/refactor)

- **`LATTICE_ARCHITECTURE.md`** + **`EXPLICIT_MIGRATION_PLAN.md`**
  - Overlap heavily with explicit architecture narrative.
  - **Action:** merge into one canonical “Explicit Architecture & Migration Status” doc, keep the other as a short pointer.

- **`test_checkpoint.md`** + a future engineering changelog
  - Useful session checkpoint, but likely to become stale.
  - **Action:** either:
    - fold into `docs/development_log.md`, or
    - create dated `docs/checkpoints/` and move it there.

## Priority 3 (mark archival)

- **`docs/experimental_tet_lattices.md`**
  - Still useful idea bank, but should be clearly marked “R&D concepts / not implemented by default”.
  - **Action:** keep, add implementation-status table.

- **`PreviousTPMSGeneration/*` docs**
  - Historical but easy to confuse with current architecture.
  - **Action:** mark as legacy at top of each doc.

## Candidate deletions (only after confirmation)

- None should be hard-deleted immediately without owner approval.
- Safer pattern:
  1. Mark legacy/archived,
  2. Add pointer to canonical docs,
  3. Remove only after one release cycle.

---

## F) Suggested Canonical Doc Structure Going Forward

1. **`docs/architecture_current.md`**  
   Single source of truth for current production architecture (implicit + explicit).

2. **`docs/explicit_engine.md`**  
   Explicit-only canonical technical reference:
   - scaffold
   - topology
   - geometry
   - known failure modes / mitigations

3. **`docs/experiments_archive.md`**  
   Index of supercell/hex/experimental scripts and what they proved.

4. **`docs/changelog.md`**  
   Dated technical checkpoints replacing ad-hoc root markdown snapshots.

---

## G) Bottom-Line Status

- **Explicit - Fast** is not a prototype anymore; it is an operational engine with broad test coverage in scripts.
- `Supercell_Modules` and `Hex_Sandbox` remain highly valuable R&D and validation tracks.
- The biggest immediate blocker is **documentation consistency**, not missing core algorithms.

