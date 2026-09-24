# Conformal Lattice Engine V2 — Full State Document
**Last Updated:** 2026-07-18  
**Project:** `c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite`

> **Status (Jul 2026):** The “promote `experiments/conformal_v2` → `a15_conformal.py`” plan in this document is **done**. Production entry points are `generate_a15_conformal_lattice` / `generate_conformal_lattice` (see [history/walkthrough.md](history/walkthrough.md) Phases 1–7). Keep this file as the experiment/seam history; do not treat the integration checklist as unfinished work.

---

## Session History

| Date | Agent | Work Done |
|---|---|---|
| 2026-07-10 | Claude (me) | 3-stage pipeline built, 4 bugs fixed, sphere + Part2_Adapter validated, global vertex weld added, docs written |
| 2026-07-10–15 | Second AI agent | Extended `conformal_utils.py` with 12 new functions, diagnosed tiling seam issue, built boolean trim pipeline, ran Toros + Trophy tests |

---

## Current State of `conformal_utils.py`

**Location:** `experiments/conformal_v2/conformal_utils.py`  
**Lines:** 1,403 (was ~810 on 2026-07-10)

### Full Function Index (with line numbers)

| Line | Function | Status |
|---|---|---|
| 19 | `_pt_key(p, decimals)` | Refactored — now accepts `decimals` param |
| 23 | `_coord_face_key(nodes, tet, fv)` | **NEW** — coordinate-based face key (replaces index-based) |
| 27 | `_unique_tet_edges(tets)` | **NEW** — extracts unique edges from tets for debug export |
| 39 | `_tet_layer_distance_from_skin(tets, boundary_faces)` | Original |
| 119 | `safe_signed_distance(cad_mesh, points, chunk_size)` | Original |
| 135 | `calibrate_strut_radius(cell_size, target_fraction)` | Original |
| 242 | `project_to_cad_surface(nodes, cad_mesh)` | Original |
| 271 | `apply_sdf_ironing(nodes_3d, struts, boundary_node_ids, cad_mesh)` | Original |
| 339 | `apply_depth_gated_relaxation(nodes, struts, node_depths, iterations, alpha)` | Original |
| 401 | `_rotation_matrix_from_z(vec)` | Original |
| 419 | `sweep_to_manifold(nodes, struts, radius)` | Original (binary-tree union) |
| 470 | `_manifold_from_trimesh(mesh)` | **NEW** — helper for manifold3d conversion |
| 482 | `_manifold_to_trimesh(manifold)` | **NEW** — helper for manifold3d conversion |
| 489 | `_cylinder_manifold(p0, p1, radius, segments)` | **NEW** — builds one manifold cylinder |
| 522 | `_union_manifolds(parts, chunk_size)` | **NEW** — binary-tree union with timing |
| 559 | `union_lattice_with_spherical_joints(nodes, struts, strut_radius, ...)` | **NEW** — cylinders + spheres at joints, per-strut radius support |
| 669 | `boolean_intersect_with_cad(lattice_mesh, cad_mesh)` | **NEW** — clips lattice to CAD using manifold3d boolean |
| 696 | `sweep_struts_concat(nodes, struts, radius)` | **NEW** — fast concat (no boolean) for debug visualization |
| 724 | `project_points_to_surface(nodes, cad_mesh)` | **NEW** — vectorized surface projection |
| 749 | `conform_strut_polyline(p0, p1, cad_mesh, n_steps)` | **NEW** — bends a strut to hug the CAD surface |
| 777 | `_square_prism_between(p0, p1, side)` | **NEW** — square cross-section strut primitive |
| 790 | `_union_trimesh_list(meshes)` | **NEW** — iterative trimesh boolean union |
| 828 | `_square_prism_surface_oriented(p0, p1, normal, side)` | **NEW** — surface-normal-oriented square prism |
| 887 | `sweep_square_surface_struts(nodes, struts, cad_mesh, side)` | **NEW** — sweeps square struts along CAD surface |
| 933 | `sweep_square_straight_struts(nodes, struts, side)` | **NEW** — sweeps square struts (straight, interior) |
| 1018 | `trim_mesh_with_inset_sphere(mesh, center, radius, inset)` | **NEW** — trims a mesh against an inset sphere |
| 1042 | `generate_conformal_part(cad_filepath, cell_size, strut_radius, export_dir, ...)` | Upgraded — 6 new kwargs |

### New `generate_conformal_part` Kwargs
```python
generate_conformal_part(
    cad_filepath: str,
    cell_size: float,
    strut_radius: float,
    export_dir: str,
    *,
    export_debug_stls: bool = True,   # NEW: exports 4 debug STLs per run
    skin_only: bool = False,           # NEW: export only the surface dual
    skin_output_name: str | None = None, # NEW: custom filename for skin-only mode
    skip_sweep: bool = False,          # NEW: returns topology dict without meshing
    signed_distance_fn=None,           # NEW: injectable SDF (for testing)
) -> dict
```

The return dict now includes (when `skip_sweep=True`):
```python
{
    "nodes_relaxed": np.ndarray,
    "cyan_struts": np.ndarray,
    "red_struts": np.ndarray,
    "cad_mesh": trimesh.Trimesh,
    "tet_nodes": np.ndarray,
    "boundary_tris": np.ndarray,
    "primal_surface_struts": np.ndarray,
    # + nodes_count, struts_count, boundary_faces_count, ...
}
```

---

## What the Second AI Agent Did (2026-07-10–15)

### 1. Seam Investigation (`diagnose_tiling_seams.py`)
- Wrote a standalone harness to trace valency-1 boundary faces that appear on internal cell planes.
- Found that the global vertex weld I added (decimals=5) was **insufficient** — boundary faces at cell interfaces were still classified as singular because the coordinate-based face key `_coord_face_key` treats faces by rounded coordinate, not by node index. This is the correct fix.
- Added `_coord_face_key()` as a module-level function.
- Exported `debug_internal_seams.stl` in `experiments/conformal_v2/`.

### 2. Debug STL Pipeline
Added 5 numbered debug export stages inside `generate_conformal_part`:
- `debug_01_raw_tet_grid.stl` — background grid before culling
- `debug_02_culled_tet_grid.stl` — surviving tet edges
- `debug_02_5_conformed_tet_grid.stl` — ironed boundary nodes
- `debug_03_conformed_core.stl` — core struts only
- `debug_04_final_lattice.stl` — combined final output

These are controlled by `export_debug_stls=True` (default).

### 3. Square-Prism Strut System
Added a full surface-oriented square strut pipeline for skin struts:
- `_square_prism_between`: axis-aligned square cross-section
- `_square_prism_surface_oriented`: CAD-normal-aligned square strut
- `sweep_square_surface_struts`: sweeps skin struts with square cross-section
- `sweep_square_straight_struts`: sweeps core struts with square cross-section

This enables a cube-strut skin that better matches CAD surface curvature.

### 4. Boolean Trim Workflow
Added `boolean_intersect_with_cad(lattice_mesh, cad_mesh)` for final post-processing trim:
- Converts both meshes to manifold3d
- Performs `m_lattice & m_cad` (intersection)
- Returns trimesh result

Also added `trim_mesh_with_inset_sphere(mesh, center, radius, inset)` for spherical part trimming.

### 5. Validated New Parts
| Part | cell_size | strut_radius | Nodes | Struts | Time |
|---|---|---|---|---|---|
| `Toros.stl` | dynamic | calibrated | — | — | ~120s |
| `Trophy_base_Rescaled.STL` | dynamic | calibrated | — | — | ~90s |
| Sphere 70mm (various cells) | 12.7–70mm | 2–6.35mm | — | — | varies |
| Debug 05: jointed lattice union | — | — | — | — | test |
| Debug 06: boolean trimmed lattice | — | — | — | — | test |

---

## Key Architecture Decision: Coordinate-Based Face Keys

The most important change made by the second agent was switching **all** boundary face lookups from index-based to **coordinate-based** using `frozenset` of rounded coordinate tuples:

```python
# OLD (index-based) — breaks across cell boundaries where same physical point has 2 indices
fkey = tuple(sorted(list(tet[list(fv)])))

# NEW (coordinate-based) — robust to any index aliasing
fkey = frozenset(_pt_key(tet_nodes[i]) for i in tet[list(fv)])
```

This is the real fix for the tiling seam gaps — the global vertex weld is a necessary pre-condition, but the coordinate-based face key is what makes the boundary detection robust.

---

## New Files Created (2026-07-10–15)

| File | Location | Purpose |
|---|---|---|
| `diagnose_tiling_seams.py` | `experiments/conformal_v2/` | Seam investigation harness |
| `debug_internal_seams.stl` | `experiments/conformal_v2/` | Valency-1 internal faces output |
| `universal_shared_edge_skin.py` | `graphite/` | 2D cross-section skin generator (for flat cuts) |
| `export_torture_stls.py` | `tests/` | Batch torture-test STL exporter |
| `plot_gating_comparisons.py` | `tests/` | Side-by-side gating comparison plotter |
| `BALLS_BASEBALL.md` | `docs/` | Baseball project doc (separate workflow) |

---

## Outstanding Issues

### The Seam/Gap Problem — Status: Partially Fixed
- **Root cause confirmed:** Boundary faces at cell interfaces were being double-counted due to floating-point mismatches in index-based face keys.
- **Fixes applied:** Global vertex weld (decimals=5) + coordinate-based face keys.
- **Current status:** The other agent ran `Part2_Adapter` after these fixes — the debug STLs show the tet grid is clean, but the user reported gaps still visible in the final mesh. This likely means the seam issue persists at the Kagome node/strut level (not the tet level), specifically in how cyan (skin) struts are wired.
- **Next investigation:** The topological wiring in Step 9 still uses `edge_to_faces` with index-based `boundary_faces` tuples. These need to be switched to coordinate-keyed lookups as well.

### `skin_only` mode not in `docs/CONFORMAL_ENGINE_V2.md`
The doc we wrote on 2026-07-10 predates the second agent's changes. It does not cover: debug STL pipeline, `skip_sweep`, `skin_only`, square-prism struts, `boolean_intersect_with_cad`, or `union_lattice_with_spherical_joints`.

---

## Integration Plan: Wiring into Main Graphite Engine

The experiment lives in `experiments/conformal_v2/conformal_utils.py`. The target home in the production engine is `graphite/explicit/`.

### Proposed Module Layout
```
graphite/explicit/
  a15_kagome.py                   ← exists: trilinear_warp lives here
  a15_conformal.py                ← NEW: home for the V2 engine
  conformal_surface_struts.py     ← NEW: square-prism skin strut system
```

### Functions to Promote (from experiment → production)

**Core pipeline** → `graphite/explicit/a15_conformal.py`:
- `calibrate_strut_radius()`
- `safe_signed_distance()` (or import from `graphite/geometry/`)
- `project_to_cad_surface()`
- `apply_sdf_ironing()`
- `apply_depth_gated_relaxation()`
- `generate_conformal_part()` (as `generate_a15_conformal_lattice()`)

**Mesh primitives** → existing `graphite/explicit/`:
- `union_lattice_with_spherical_joints()` → `geometry_module.py`
- `boolean_intersect_with_cad()` → `geometry_module.py`
- `sweep_to_manifold()` → already exists in Phase 3 form, merge
- `_union_manifolds()`, `_cylinder_manifold()`, etc. → `geometry_module.py`

**Square struts** → `graphite/explicit/conformal_surface_struts.py`:
- `sweep_square_surface_struts()`
- `sweep_square_straight_struts()`
- `_square_prism_between()`
- `_square_prism_surface_oriented()`

**Universal skin** → `graphite/explicit/universal_shared_edge_skin.py`:
- Already exists at `graphite/universal_shared_edge_skin.py` — needs moving one level deeper

### Things to NOT promote yet
- `trim_mesh_with_inset_sphere()` — too specialized, keep in experiment
- `conform_strut_polyline()` — experimental, not validated at scale
- All `debug_*` export logic — move to a `graphite/debug/` or strip from production API

---

## How to Run the Current Production Wrapper

```python
from experiments.conformal_v2.conformal_utils import (
    generate_conformal_part,
    calibrate_strut_radius,
)
import trimesh, numpy as np

mesh = trimesh.load("test_parts/MyPart.STL")
cell_size = float(np.min(mesh.extents)) / 2.0      # half smallest dim
strut_radius = calibrate_strut_radius(cell_size, target_fraction=0.10)

summary = generate_conformal_part(
    cad_filepath="test_parts/MyPart.STL",
    cell_size=cell_size,
    strut_radius=strut_radius,
    export_dir="outputs/conformal_outputs",
    export_debug_stls=False,   # set True to get debug_01..04 STLs
)
```

**Outputs** always go to `outputs/conformal_outputs/` — never `test_parts/`.  
**Scratch scripts** live in `C:\Users\ehunt\.gemini\antigravity\scratch\`.

---

## Files Map (Complete)

```
experiments/conformal_v2/
  conformal_utils.py              ← Master experiment file (1,403 lines)
  diagnose_tiling_seams.py        ← Seam investigation tool
  debug_internal_seams.stl        ← Seam debug output
  test_phase1_voxel_bfs.py        ← Phase 1 prototype
  test_phase2_ironing.py          ← Phase 2 prototype  
  test_phase3_relaxation.py       ← Phase 3 prototype (ground truth)
  relaxed_tet_edges_Case2.stl     ← Phase 3 sphere reference output
  relaxed_boundary_faces_Case2.stl
  ironed_only_tet_edges_Case2.stl

graphite/
  universal_shared_edge_skin.py   ← 2D skin generator (needs relocation)
  explicit/
    a15_kagome.py                  ← trilinear_warp lives here

outputs/conformal_outputs/
  test_sphere_ph3_conformal_lattice.stl
  test_sphere_ph3_boundary_skin.stl
  Hook_Part2_Template_conformal_lattice.stl
  Hook_Part2_Template_boundary_skin.stl
  Part2_Adapter_conformal_lattice.stl   ← cell_size=7.9375mm
  Part2_Adapter_boundary_skin.stl
  Part2_Adapter_surface_dual_half_cell.stl
  Toros_conformal_lattice.stl
  Toros_boundary_skin.stl
  Trophy_base_Rescaled_conformal_lattice.stl
  Trophy_base_Rescaled_boundary_skin.stl
  sphere_70mm*.stl (multiple resolutions)
  debug_01..06_*.stl (pipeline debug exports)

docs/
  CONFORMAL_ENGINE_V2.md          ← This document (updated)

tests/
  test_conformal_batch.py
  export_torture_stls.py
  plot_gating_comparisons.py

scratch/ (C:\Users\ehunt\.gemini\antigravity\scratch\)
  test_part2_adapter.py           ← Dynamic-scaling test for Part2_Adapter
  test_sphere_production.py       ← Phase 3 sphere regression test
  analyze_gaps.py                 ← Connectivity analysis script
```
