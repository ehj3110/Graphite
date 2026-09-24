# Supercell Recovery Audit Report

This report implements the recovery audit plan deliverables D1-D5 for the tet-oct/supercell path and records the minimal stabilization edits applied.

## D1: Source-of-Truth Map

- Seed generation (periodic node tiling): `graphite/explicit/supercell_module.py`
  - `generate_cartesian_nodes(...)`
  - `get_effective_tiling_period(...)`
- Tet-oct Kagome graph synthesis (bbox-driven canonical path): `graphite/explicit/supercell_module.py`
  - `apply_face_centric_kagome(..., target_bbox=...)`
- Conformal clipping/snapping (graph-level): `graphite/explicit/supercell_module.py`
  - `apply_topological_snapping(...)`
  - `apply_strict_clipping(...)`
- Explicit tessellation pipeline (general non-supercell path): `graphite/explicit/`
  - `scaffold_module.py` -> `generate_conformal_scaffold(...)`
  - `topology_module.py` -> `generate_topology(...)`
  - `geometry_module.py` -> `generate_geometry(...)`
- App integration point (supercell runtime route): `app.py`
  - `"Explicit - Supercell (Beta)"` branch in generation flow.

Canonical owner decision: supercell/tet-oct generation stays in `graphite/explicit/supercell_module.py`, while scaffold/topology/geometry remain canonical explicit primitives for the Delaunay path.

## D2: Drift Matrix (Active vs Backups)

### Function-level drift

- `supercell_module.py` adds modern conformal stages not present in old backup:
  - `cull_kagome_lattice(...)`
  - `generate_boundary_skin(...)`
  - `apply_surface_snapping(...)`
  - `apply_supercell_stretching(...)`
  - `generate_supercell_surface_skin(...)` (missing from oldest backup)
- `supercell_module.phase1.bak.py` has `apply_element_culling(...)` which is replaced by `cull_kagome_lattice(...)` in active.

### API drift

- `apply_face_centric_kagome(...)`
  - active: `(..., target_bbox=None)` -> returns `(nodes, struts, tet_elements, active_tet_coords)`
  - phase1 backup: no `target_bbox` -> returns `(nodes, struts, final_elements, face_to_node)`
  - older backup: no `target_bbox` -> returns `(nodes, struts)` only
- This introduces caller unpacking drift in older scripts/tests expecting 2-tuple or legacy element payload.

### Safety/behavioral drift

- Duplicate `apply_topological_snapping` existed in active and backup variants.
- Active had a non-returning path in `apply_topological_snapping` after node compression, causing undefined caller behavior.
- Bounding-box deterministic tiling existed but was not enforced at the app callsite for Kagome generation.

## D3: Integration Contract

### Explicit pipeline contract (canonical)

- `generate_conformal_scaffold(mesh, target_element_size, ...) -> ScaffoldResult`
  - `nodes`: `(N, 3)` float
  - `elements`: `(M, 4)` or `(M, 10)` int
  - `surface_faces`: `(K, 3)` or `(K, 6)` int
- `generate_topology(nodes, elements, surface_faces, ...) -> (topology_nodes, struts)`
  - `topology_nodes`: `(P, 3)` float
  - `struts`: `(Q, 2)` int
- `generate_geometry(nodes, struts, strut_radius, boundary_mesh, ...) -> trimesh or tuple`
  - expects valid indexable `nodes/struts`, non-empty `struts`.

### Supercell integration contract (stabilized)

- `generate_cartesian_nodes(bounds, cell_size, cell_type, padding_blocks=1) -> (N, 3)`
- `apply_face_centric_kagome(nodes, cell_size, cell_type, target_bbox=bounds) -> (nodes, struts, tet_elements, tet_coords)`
- `apply_topological_snapping(nodes, struts, mesh, mode=STRICT|SNAP) -> (nodes, struts)`
- `generate_geometry(nodes, struts, radius, ...)` for final solid.

## D4: Breakage List (Deleted/Migrated Path and API Mismatch Risks)

- Tests/scripts with supercell API unpacking mismatches against active return signatures:
  - `tests/test_vf_mapping.py` (expects 2 return values)
  - `tests/test_true_kagome_connectivity.py` (expects `(tets, octs)` tuple shape not produced by active)
  - `tests/test_torture_suite.py` (expects 2 return values in Kagome branch)
  - `tests/test_feature_sandbox.py` (incorrect call signature using struts as arg2)
- Historical/deprecated references to migrated trees exist in docs/scripts and experiment paths (`Supercell_Modules`, `Universal_Lattice_Engine`, archived historical references). These are not runtime blockers for the stabilized app path but represent cleanup debt.

## D5: Stabilization Patch Plan and Execution

### Planned minimal sequence

1. Fix conformal clipping function duplication and return contract.
2. Enforce bbox-based deterministic Kagome tiling at app callsite.
3. Validate deterministic tet-oct conformal export via smoke run.
4. Defer broad test migration to dedicated API-normalization pass.

### Applied edits

- `graphite/explicit/supercell_module.py`
  - removed duplicate wrapper `apply_topological_snapping(...)` definition
  - fixed empty return shapes for clipping
  - restored final return after compression:
    - `compressed_nodes`
    - `compressed_struts`
- `app.py`
  - updated Kagome call to pass `target_bbox=bounds` for deterministic bbox tiling.

### Validation results

- Deterministic smoke run passed:
  - bbox node generation -> face-centric Kagome -> strict clipping -> geometry export
  - output file: `outputs/smoke_tetoct_bbox_conformal.stl`

## Current Recovery State

Recovered target achieved for a minimal runnable path:

- single callable supercell tet-oct route,
- conformal clipping integrated,
- deterministic bbox-anchored tiling,
- STL export validated.

Remaining work is primarily API normalization for legacy tests/scripts expecting old `apply_face_centric_kagome` return shapes.
