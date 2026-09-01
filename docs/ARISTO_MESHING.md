# Aristo meshing — TPMS / discrete STL volume fill

This document covers **Gmsh volume meshing for Aristo FEA** on watertight lattice STLs (especially Mirae TPMS slab exports). It records the **production defaults**, **geometry selection guidance**, and **issues encountered** during the March 2026 Mirae workstream.

**Solver / BC overview:** [ARISTO.md](ARISTO.md)  
**Implementation:** `graphite/aristo/gmsh_lattice_mesh.py`, `graphite/aristo/aristo_solver.py`  
**CLI:** `scripts/gmsh_lattice_mesh.py`, `scripts/mesh_tpms_lattice_gmsh.py`

---

## Pipeline overview

```
Watertight STL (trimesh)
    → inject discrete surface into Gmsh
    → classifySurfaces (90°) + createTopology   [NOT createGeometry on dense soup]
    → explicit SurfaceLoop + Volume
    → adaptive CharacteristicLength sizing
    → cap-only duplicate-node merge (1e-4 mm)
    → generate(3) with Delaunay + Laplace optimize
    → [optional] setOrder(2) with safety gate
    → extract linear Tet4 (+ surface tris)
    → Aristo P1 stiffness assembly
```

**Mode flag:** `AristoConfig.fea_gmsh_mesh_mode="tpms"` or env `ARISTO_GMSH_FEA_MESH_MODE=tpms`.

---

## Production defaults (March 2026)

These are the **intended defaults** after the P1 / Delaunay refactor. Do not revert to uniform Tet10 + Netgen without reading [Failed strategies](#failed-strategies-and-root-causes) below.

| Gmsh / config option | Production value | Notes |
|----------------------|------------------|-------|
| `Mesh.Algorithm` | `6` | Frontal-Delaunay 2D |
| `Mesh.Algorithm3D` | **`1` (Delaunay)** | HXT (`10`) crashes on V4 classify path |
| `Mesh.CharacteristicLengthFromCurvature` | `1` | Adaptive sizing on |
| `Mesh.CharacteristicLengthMin` | **0.08** mm | `fea_gmsh_char_length_min` |
| `Mesh.CharacteristicLengthMax` | **0.8** mm | `fea_gmsh_char_length_max` |
| `Mesh.MinimumElementsPerTwoPi` | **15** | `fea_gmsh_min_elements_per_two_pi` |
| `Mesh.Optimize` | `1` | Laplace smoothing |
| `Mesh.OptimizeNetgen` | **`0`** | Required for discrete STL lattices |
| `Mesh.Smoothing` | `10` | |
| Element order | **1 (Tet4)** | P1 FEA assembly |
| `Mesh.SecondOrderLinear` | `1` | Only if `--order 2` and linear mesh passes gate |
| Cap merge tolerance | **1e-4** mm | Cap patches only — see [Cap merge](#cap-merge-and-slivers) |
| `fea_gmsh_classify_only` | `True` | No single-surface fallback unless opted in |

**Override 3D algorithm:** `ARISTO_GMSH_ALGORITHM3D=10` (HXT) for experiments only.

**Legacy uniform sizing:** CLI `--strict-uniform --h 0.20` or `--legacy-curvature` — R&D only.

---

## Geometry selection

Test matrix from the Mirae 3×1.5×5 mm slab folder:

| STL | ~Faces | Mesh / FEA | Recommendation |
|-----|--------|------------|----------------|
| **V4_fixed** | ~48k | ✅ ~74k tets, ~1.3% poor | **Use for production FEA** |
| V2 test | ~223k | ✅ ~360k tets, more speckling | Legacy reference |
| V5_main | ~903k | ⚠️ ~1.56M tets @ h=0.35, ~28 min solve | Decimate to ~50–100k faces first |
| V6 / V7 | 524k–883k | ❌ classify / PLC failures | Not ready without cap repair workflow |
| V3 lineage | ~46k | Superseded | Archive / delete |

**Watertight check:** `trimesh.Trimesh.is_watertight` before meshing.  
**Self-intersections:** PyMeshLab check on V4_fixed → 0 intersections.

---

## Issues encountered

### 1. HXT (Algorithm3D = 10) crash on classified V4

**Symptom:** Process crash or failure during Steiner insertion on the classify + topology path.  
**Cause:** Default HXT mesher unstable on this discrete TPMS skin + classified patches.  
**Fix:** Hardcode **Delaunay (1)** in `_set_meshing_algorithm_options()`. Env override retained for testing.

---

### 2. Netgen optimization on discrete STL skins

**Symptom:** ~326 **illegal linear tets** remain after meshing; ParaView shows broken volume; stiffness matrix unreliable if ingested uncleaned.  
**Cause:** Netgen (`Mesh.OptimizeNetgen=1`) cannot repair through-thickness sliver tets constrained by fixed STL facets.  
**Fix:** **`Mesh.OptimizeNetgen=0`** for all lattice STL paths. Keep Laplace only (`Mesh.Optimize=1`).

**Note:** Gmsh may still log ~500 “ill-shaped” tets after Laplace (low SICN quality). These are **not** the same as Netgen illegal PLC failures. Aristo drops ~2 near-degenerate tets at extract via orientation repair.

---

### 3. Strict uniform h + quadratic (Tet10) experiment

**Symptom:** Mesh completes but illegal tets persist; curvilinear midside nodes snap to STL facets without `SecondOrderLinear`; even with straight-sided Tet10, linear base mesh still broken.  
**Cause:** Uniform `MeshSizeMin/Max` + Netgen on thin-wall discrete boundary.  
**Fix:** Default back to **adaptive CharacteristicLength** + **P1**; Tet10 opt-in via `--order 2` with pre-elevation health gate.

**Aristo implication:** Stiffness assembly is **P1-only**. Tet10 meshes are export/debug until assembly upgraded.

---

### 4. createGeometry vs createTopology on lattice soup

**Symptom:** `createGeometry()` re-meshes 2D patches and can collapse the skin to a handful of nodes → PLC failure.  
**Fix:** Default **`createTopology(True, True)`** after `classifySurfaces`. Opt into re-parametrization with `ARISTO_GMSH_CLASSIFY_FOR_REPARAM=1` only when intentional.

---

### 5. Dense STL skin dominates tet count (V5)

**Symptom:** ~1.56M tets at h=0.35 whether h is 0.15 or 0.35 — reducing `h` does not reduce DOFs.  
**Cause:** Single-surface path injects full ~900k-face skin (~451k surface nodes); interior sizing cannot overcome boundary node budget.  
**Mitigation:** Blender / mesh decimation to ~50–100k faces, then TPMS classify + adaptive path. Single-surface + uniform h is not a DOF control knob on dense exports.

---

### 6. Cap slivers and overlapping facets (V5, V6, V7)

**Symptom:** `classifySurfaces` unusable; gmsh overlapping-facet / PLC errors; knife-edge nodes at z ≈ 3.3–3.6 mm on slab caps.  
**Cause:** Microscopic cap slivers in exported STL.  
**What worked:** Targeted collapse of **3 faces** on V4 via `scripts/fix_lattice_stl_slivers.py`; V5 needed merge of vertex 1214→1213 before meshing.  
**What failed:** **Bulk** sliver collapse on dense meshes — creates new overlaps.

---

### 7. Global Geometry.Tolerance too small on single-surface path

**Symptom:** HXT / PLC failure on V5 when tiny tolerance auto-applied.  
**Cause:** Collapses valid cap nodes on dense discrete shell.  
**Fix:** Single-surface path only applies tolerance if `ARISTO_GMSH_GEOMETRY_TOLERANCE` is set. Classify path uses bbox-scaled tolerance for topology, then **1e-4 mm cap-only merge** immediately before `generate(3)`.

---

### 8. Thin-wall sliver tets (expected, not a bug)

**Symptom:** ~1–2% of tets fail Aristo `quality_ok` gates (aspect ratio up to ~12, high vol/median ratio).  
**Cause:** Discrete STL → volume fill naturally creates through-thickness needle tets in thin TPMS walls.  
**Handling:** Excluded from stress percentile stats; still in global `K`. Typical V4 poor fraction **~1.3%**.

---

### 9. Top Z-band load BC picked side walls

**Symptom:** 214 “load” faces on V4 with 1% top Z-band — includes vertical faces on upper struts.  
**Fix:** `bc_load_mode="flat_top"` — `z ≥ z_max − 1e-4` and `normal_z > 0.99` → **165 faces**. Bottom fixed remains 1% Z-band (friction platen model).

---

## Failed strategies and root causes

| Strategy | Outcome | Root cause |
|----------|---------|------------|
| HXT 3D on V4 classify | Crash | Algorithm / discrete skin incompatibility |
| Uniform h + Tet10 + Netgen | Illegal tets, broken export | Netgen + fixed STL skin |
| Single-surface uniform h on V5 | ~1.56M tets always | Skin node budget |
| Bulk sliver fix on dense mesh | New overlaps | Over-merging |
| Tiny global tolerance (V5 single-surface) | PLC failure | Cap node collapse |
| V6/V7 without cap repair | Mesh fail | Unmeshable cap topology |

---

## Cap merge and slivers

Knife-edge nodes live on **planar cap patches**, not the large lattice skin.

1. `_cap_surface_dim_tags()` — largest classified surface = skin; smaller patches = caps.
2. `removeDuplicateNodes` / `removeDuplicateElements` on **cap entities only**.
3. `Geometry.Tolerance = 1e-4` and `Mesh.ToleranceInitialDelaunay = 1e-4` scoped to pre-`generate(3)` cleanup window.

Do **not** run global duplicate merge on the full skin — overlapping facets on dense meshes.

---

## Order elevation gate (`--order 2`)

When quadratic export is requested:

1. After `generate(3)`, count illegal Gmsh tets (`minSICN ≤ 0`) and inverted tets (negative signed volume).
2. If any found → **abort `setOrder(2)`**, log critical warning, export **linear P1**.
3. Else → set `Mesh.SecondOrderLinear = 1`, then `setOrder(2)` (straight-sided Tet10, no STL facet snap).

Metadata returned via optional `mesh_meta` dict on `generate_lattice_fea_mesh_from_stl()`.

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `ARISTO_GMSH_FEA_MESH_MODE` | `tpms` / `lattice` / `single_surface` |
| `ARISTO_GMSH_ALGORITHM3D` | Override 3D mesher (default Delaunay `1`) |
| `ARISTO_GMSH_CLASSIFY_ONLY` | `1` = no single-surface fallback |
| `ARISTO_GMSH_CLASSIFY_FOR_REPARAM` | `1` = use `createGeometry` instead of topology |
| `ARISTO_GMSH_CLASSIFY_ANGLE_DEG` | classifySurfaces angle (default 90°) |
| `ARISTO_GMSH_CHAR_LENGTH_MIN/MAX` | Override adaptive cl bounds (mm) |
| `ARISTO_GMSH_MIN_ELEMENTS_PER_TWO_PI` | Curvature density |
| `ARISTO_GMSH_LEGACY_CURVATURE_SIZING` | `1` = legacy `Mesh.MeshSize*` path |
| `ARISTO_GMSH_GEOMETRY_TOLERANCE` | Single-surface tolerance override only |
| `ARISTO_VERBOSE` | Stage logging |

---

## CLI tools

### Mesh only (no FEA)

```bash
# Default: adaptive P1, Delaunay, classify-only
python scripts/gmsh_lattice_mesh.py path/to/Mirae_LatticeSlab_V4_fixed.stl \
  --export-vtu outputs/.../mesh_debug.vtu

# R&D: strict uniform (not recommended for FEA)
python scripts/gmsh_lattice_mesh.py path/to/part.stl --strict-uniform --h 0.20

# R&D: quadratic export attempt
python scripts/gmsh_lattice_mesh.py path/to/part.stl --order 2
```

### Full FEA (V4 reference)

```bash
python scripts/run_mirae_lattice_slab_v4_aristo.py
```

---

## VTU and quality fields

**Mesh-only VTU** (`--export-vtu`):

- `aspect_ratio`, `max_edge_mm`, **`quality_ok`** (int8, 1 = pass)

**FEA VTU** (`export_aristo_paraview`):

- Nodal: `von_mises_nodal_MPa`, `displacement_mm`
- Cell: `quality_ok`, `tet_volume_mm3`, `aspect_ratio`, `max_edge_mm`, optional `touches_load_face`

**Quality gate definition** (`mesh_quality.build_quality_mask`):

- `aspect_ratio ≤ max_tet_aspect_ratio` (default **20**)
- Volume within min/max bounds (median-based)
- `max_edge_mm ≤ resolve_max_tet_edge(config)`

**Gate breakdown** (`quality_gate_breakdown` in JSON reports): per-gate failure counts plus `primary_failure_among_poor` (exclusive reason among poor tets). See [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md).

---

## Implicit marching-cubes STLs (`single_surface`)

Piecewise implicit cylinders (`experiments/implicit_to_volume/`) export **dense MC skins** (~200k–500k tris). Unlike Mirae V4:

| Topic | Mirae V4 | Implicit MC STL |
|-------|----------|-----------------|
| Gmsh entry | `classifySurfaces` + adaptive CL | **`single_surface`** only |
| Typical `poor_fraction` @ h≈0.07–0.12 | ~1.3% | ~**11%** |
| Primary poor gates | Thin-wall slivers | **`min_volume` + `aspect_ratio`** |
| `max_edge` gate failures | Rare | **None** (in tested cases) |

**Do not** use per-band manifold union before meshing — seam overlaps break classify and single-surface PLC.

**Repair + fine `h` (validated):** MC STLs mesh reliably after island cleanup and wall-thickness-appropriate `h` — see [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md) § "Practical recipe". Coarse `h` was the main driver of ~11% poor fractions, not Gmsh inability to tet-fill lattices.

**Wall sizing:** for ~**80 µm** strut walls (0.5 mm unit cell, SF 33%), target **h ≤ 0.02 mm** (~4 tets/wall); fine sweep **h = 0.01 mm** (~8 tets/wall, `MeshSizeMin` = 5 µm). See `mesh_wall_thickness_sweep.py`.

---

## Troubleshooting matrix

| Observation | Likely cause | Action |
|-------------|--------------|--------|
| Gmsh crash during 3D | HXT on classify path | Confirm Delaunay default; avoid `ARISTO_GMSH_ALGORITHM3D=10` |
| Hundreds of “illegal tets” after optimize | Netgen on | Confirm `OptimizeNetgen=0` |
| ~1.5M+ tets, h change ignored | Dense STL skin | Decimate STL; use classify path |
| classify fails / overlapping facets | Cap slivers | Targeted sliver fix; check V6/V7 diagnostics JSON |
| 0 load faces with flat_top | No upward cap at z_max | Check STL orientation; relax `bc_top_normal_z_min` slightly |
| Stress speckles at caps | Needle boundary tris | Optional Open3D surface clean (`fea_clean_stl_surface`) — not default on V4 |
| Poor fraction ~1–2% on V4 | Thin-wall slivers | Expected; review `quality_ok` in ParaView |

---

## Regression baseline

Full pinned metrics, tolerances, and pass/fail rules: **[ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)**.

Quick re-run:

```bash
python scripts/gmsh_lattice_mesh.py \
  outputs/models/user_spec_rect_prism_3x1p5x5mm/Mirae_LatticeSlab_V4_fixed.stl

python scripts/run_mirae_lattice_slab_v4_aristo.py
```

**Expect (approximate):**

- Mesh-only: ~73.8k Tet4, `algorithm_3d: 1`, `optimize_netgen: 0`, 0 illegal/inverted linear tets  
- FEA 1N: max nodal von Mises **29.124 MPa**, 165 load faces, 0.8115 mm² load area  
- Linear scaling: 5 N → **145.622 MPa**, 10 N → **291.244 MPa**  

---

## Related documentation

| Path | Purpose |
|------|---------|
| [docs/ARISTO.md](../../../docs/ARISTO.md) | **Canonical** module overview and progress |
| [docs/ARISTO_MESHING.md](../../../docs/ARISTO_MESHING.md) | **Canonical** meshing issues and defaults |
| `MIRAE_LATTICE_FEA_WORKLOG.md` | Run-specific notes (this folder) |
| `CLEANUP_GUIDE.md` | Artifact retention (this folder) |
