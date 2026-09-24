# Hex Explicit Engine: Current State and Integration Guide

## Purpose

This document is the central reference for **explicit hexahedral lattice generation** in Graphite:

- current implementation status,
- active pipeline behavior,
- boundary-conformance capabilities,
- overlap with tet and supercell techniques,
- recommended next integration steps.

It complements:

- `docs/EXPLICIT_ENGINE.md` (broader explicit architecture),
- `docs/history/WORKSTREAM_SUMMARY_SUPERCELL_HEX_EXPLICIT.md` (historical audit),
- code-level modules under `graphite/explicit/`.

---

## Executive Snapshot

- **Default hex mesh:** **SC Nodal Conformation + Planar Slicing Surface Dual** (`generate_sc_conformal_lattice` / `generate_nodal_conformation`). This combines Cartesian background grid volume, outside-node boundary snapping, universal role surface dual wiring, and planar slicing contour sweep arc lofts for deliverable 2-manifold watertight meshes. Full spec: **[A15_KAGOME_AND_SC_SURFACE_DUAL.md](A15_KAGOME_AND_SC_SURFACE_DUAL.md)** and **[PLANAR_SLICING_SURFACE_DUAL_RETROSPECTIVE.md](PLANAR_SLICING_SURFACE_DUAL_RETROSPECTIVE.md)**.
- Historical hex-cage morphs in `conformal_generator.py` and GMSH scaffolds are archived legacy.
- Shared boundary policy (`boundary_policy.py`) is production-ready.

---

## Default production hex pipeline — Conformal Dual

**This is the mesh to wire into Graphite.**

| Step | Module | Function |
|------|--------|----------|
| 1. Scaffold | `hex_scaffold_module.py` | `generate_conformed_hex_scaffold(..., conformal_dual_mode=True, cull_mostly_external_hexes=True, ...)` |
| 2. Synthesis | `hex_scaffold_module.py` | `synthesize_conformal_dual_lattice(hex_elements)` |
| 3. Geometry | `geometry_module.py` | `generate_geometry(..., crop_to_boundary=True)` |

**Public API:** `graphite.explicit.generate_conformed_hex_scaffold`, `graphite.explicit.synthesize_conformal_dual_lattice`.

**Reference exports:**

- Toros: `scripts/export_route3_conformal_dual_toros.py` → `outputs/Route3_ConformalDual_Fixed_Toros.stl`
- Trophy: `scripts/export_trophy_base_thin_conformal_dual.py` → `outputs/Trophy_base_thin_ConformalDual.stl`

**Mandatory scaffold flags** (see CONFORMAL_DUAL_HEX.md): `conformal_dual_mode=True`, `cull_mostly_external_hexes=True`, `neighbor_stretch=False`, `boundary_stretch_out=False`.

---

## Other hex routes (non-default)

## 1) Legacy UI entry points

Hex engines in advanced `app.py` (not Conformal Dual):

- `Explicit - Conformal Hexahedral (Gmsh)` — Route 1
- `Explicit - Cropped Hexahedral` — Route 2

These use `generate_hex_topology` with a user-selected local rule (`grid`, `octahedral`, etc.) and do **not** run the Conformal Dual synthesis path.

## 2) Hex scaffold generation (alternatives)

### Conformed hex scaffold (Route 3 base, without Dual synthesis)

Module: `graphite/explicit/hex_scaffold_module.py`  
Function: `generate_conformed_hex_scaffold(...)`

Route 3 field-snapping scaffold only. **Conformal Dual** adds `conformal_dual_mode`, mandatory VF cull, and `synthesize_conformal_dual_lattice` — do not substitute `generate_hex_topology(..., rule_name="grid")` for the default mesh.

---

### Conformal Hexahedral (Gmsh - Route 1)

Module: `graphite/explicit/hex_scaffold_module.py`  
Function: `generate_conformal_hex_scaffold(...)`

*Note: Highly sensitive to STL quality. Often fails to produce hexes on non-sweepable organic geometry. Use Route 3 for production robustness.*

### Cropped Hexahedral (Route 2)

Module: `graphite/explicit/hex_scaffold_module.py`  
Function: `generate_cropped_hex_scaffold(...)`

*Note: Basic Cartesian crop. No boundary snapping. Good for fast diagnostics but not structural fidelity.*

### Lofted brute-force grid (experimental)

Module: `graphite/explicit/hex_scaffold_module.py`  
Function: `generate_brute_force_fixed_grid_hex_scaffold(..., taper_along='z'|'y')`

Fixed `nx×ny×nz` topology with **uniform station spacing along the spine** and **transverse extents from mesh plane slices** at each station — explicit analogue of single-axis lofted grading (see [LOFTED_GRADING.md](LOFTED_GRADING.md)).

Used for trophy-base experiments (`8×4×4`, `taper_along='y'`) with Kelvin, Tesseract, or two-branch topology. Scripts: `scripts/archive/exports/export_trophy_base_thin_brute_force_*.py`.

---

## 3) Hex topology generation

**Conformal Dual (default)** does not use `generate_hex_topology` for the production mesh. It calls `synthesize_conformal_dual_lattice`, which builds octahedral volume + integer surface dual internally.

For **non-default** routes (cropped hex, local rule experiments, Kelvin/Tesseract on brute grids):

Modules:
- `graphite/explicit/hex_rules.py` (local rules)
- `graphite/explicit/hex_topology_module.py` (global merge/dedupe)

Local rules include `octahedral`, `octet`, `grid`, `star`, `kelvin` / `kelvin14`, `tesseract`, `hex_dual`, `hex_face_dual`.

## 4) Geometry and clipping

Module: `graphite/explicit/geometry_module.py`  
Function: `generate_geometry(...)`

Shared behavior with tet/supercell:

- Struts swept to manifold cylinders,
- optional boolean crop to boundary (`crop_to_boundary=True`).

---

## Tet and Supercell Pipelines Worth Reusing

## Tet conformal pipeline (mature)

Modules:

- `graphite/explicit/scaffold_module.py`
- `graphite/explicit/topology_module.py`

Key reusable strengths:

- robust Gmsh STL -> CAD fallback handling (`createGeometry` -> `createTopology`),
- optional P2 tetra support (`element_order=2`) with curved boundary extraction,
- global adjacency-driven topology generation,
- dual/cage boundary graph construction with consistency checks,
- optional short-strut merge and watershed connectivity cleanup.

## Supercell boundary-state pipeline (mature experimental logic)

Modules:

- `graphite/explicit/supercell_module.py`
- `graphite/explicit/supercell_oct_tip.py`

Key reusable strengths:

- state-aware culling and conform decisions near tips/sparse regions,
- detection of oct-void bridge-like struts by geometric scale,
- selective conforming instead of blanket boundary flattening,
- tiered outside/inside boundary policy in hybrid conform mode.

---

## Boundary-Conformance Overlap Matrix

| Capability | Tet Pipeline | Supercell Pipeline | Hex Pipeline (current) | Reuse Opportunity |
|---|---|---|---|---|
| Conformal volumetric scaffold | Strong (Gmsh tets, P1/P2) | Not primary | Partial (Gmsh recombined hex) | Reuse tet Gmsh tolerance/fallback settings |
| Grid + crop route | Limited | Strong | Strong (cropped hex) | Share crop policy and diagnostics |
| Boundary-node snapping | Limited in tet topology stage | Strong (`apply_topological_snapping`, hybrid rules) | Basic (crop + final boolean clip) | Port tiered snap/stretch policy to hex |
| State-based boundary handling | Limited | Strong | None | Add hex boundary states by cell-face occupancy |
| Global adjacency graph logic | Strong | Strong | Minimal (local-rule merge only) | Add hex-cell neighbor graph and face-adjacency logic |
| Connectivity cleanup | Strong (watershed) | Moderate | None | Add optional watershed pass to hex graph |

---

## Hex Topology Integration Assessment

## Current math model

Current hex rules are local per-cell decorations and then global dedupe by rounded coordinate:

- very fast,
- deterministic,
- but no explicit cell-neighbor-aware strut routing.

## Why neighbor-aware logic matters

In tet Voronoi-like behavior, meaningful struts often follow **neighbor-centroid relationships**, not just within one element.  
For hex cells, the analog is:

- compute one node per cell center (or selected face center set),
- connect to **neighbor cell centers across shared faces** (or shared-face-derived anchors),
- optionally enforce straight-line continuity constraints across boundary perturbations.

This avoids over-localized “face-center only” behavior and produces stronger global lattice coherence (closer to what you described for tet centroid-to-centroid intent).

## Recommended integration path

1. Keep current local hex rules as baseline (`grid/octahedral/star/octet`).
2. Add a new neighbor-aware hex topology family in production, e.g.:
   - `hex_dual` (cell centroid to adjacent cell centroid),
   - `hex_face_dual` (shared-face anchor to shared-face anchor).
3. Build adjacency from structured hex index space (cropped route) and from recovered face-sharing map (conformal route).
4. Add optional boundary policy:
   - hard outside pull,
   - soft near-surface blend,
   - inside-near-boundary stretch (supercell hybrid analog).

---

## Immediate Code-Sharing Targets

## 1) Boundary-policy helper layer

Candidate shared utility module (new):

- `graphite/explicit/boundary_policy.py`

Unify:

- SDF sampling,
- hard/soft/stretch tier masks,
- closest-point projection fallback behavior.

## 2) Graph post-processing

Port from tet topology to hex topology:

- optional short-strut merge,
- optional largest-component watershed.

## 3) Gmsh robustness settings

Port from tet scaffold to conformal hex scaffold:

- geometry tolerance scaling,
- stricter size lock re-assertion after synchronize,
- optional post-mesh optimize passes,
- better diagnostics before declaring no type-5 output.

---

## Known Gaps and Risks

- No canonical central hex capability doc existed before this file.
- `docs/EXPLICIT_ENGINE.md` currently mixes old and new statements and has drift.
- Conformal Gmsh hex route can fail on complex organic STLs (zero type-5 extraction).
- Hex topology currently lacks neighbor-aware dual-style modes.
- Hex-specific boundary-state logic is not yet implemented.

---

## Suggested Next Steps

1. Add a fallback path in `generate_conformal_hex_scaffold(...)`:
   - if no type-5 hexes, optionally fallback to cropped hex with warning.
2. Introduce `hex_dual` topology mode with explicit cell-neighbor connectivity.
3. Add shared boundary-policy utility and apply to:
   - supercell hybrid conform,
   - new hex conform modes.
4. Add hex pipeline integration tests:
   - conformal hex smoke,
   - cropped hex smoke,
   - topology count sanity for each hex rule,
   - boundary clipping regression checks.

---

## Source Modules Referenced

- `graphite/explicit/scaffold_module.py`
- `graphite/explicit/topology_module.py`
- `graphite/explicit/geometry_module.py`
- `graphite/explicit/supercell_module.py`
- `graphite/explicit/supercell_oct_tip.py`
- `graphite/explicit/hex_scaffold_module.py`
- `graphite/explicit/hex_topology_module.py`
- `graphite/explicit/hex_rules.py`
- `experiments/Conformal_Mesh_Exploration/core/voxelizer.py`
- (Removed Jul 2026) former `experiments/Supercell_Modules/core/grid_generator.py` — use `graphite/explicit/proven_topologies.py` instead.
- `experiments/Universal_Lattice_Engine/core/hex_rules.py`

