# Aristo — FEA module for discrete lattice STLs

**Aristo** is Graphite’s linear-elastic FEA path for lattice structures. The **default input** is a **piecewise Split-P implicit field** (evaluated in memory → MC skin → Gmsh `single_surface` volume mesh). **STL files** are a developer/legacy override (Mirae regression, exported meshes).

**Code:** `graphite/aristo/`  
**Entry point:** `run_aristo(mesh, config)` — boundary `trimesh` may come from implicit eval or STL load  
**Implicit input API:** `graphite/aristo/implicit_input.py`  
**Meshing deep-dive:** [ARISTO_MESHING.md](ARISTO_MESHING.md)  
**Regression baseline:** [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)

---

## Recent progress (Mirae lattice slab workstream)

Work focused on the **3 × 1.5 × 5 mm** Mirae TPMS slab in  
`outputs/models/user_spec_rect_prism_3x1p5x5mm/`.

### What is production-ready today

| Area | Status |
|------|--------|
| **Canonical geometry** | `Mirae_LatticeSlab_V4_fixed.stl` (~48k faces, watertight, 0 PyMeshLab self-intersections) |
| **Volume mesh** | TPMS `classifySurfaces` + adaptive `CharacteristicLength*` + **Delaunay 3D** + **linear Tet4** |
| **Optimization** | Laplace only (`Mesh.Optimize=1`); **Netgen disabled** on discrete STL lattices |
| **FEA assembly** | P1 Tet4 only; ~74k tets / ~25k nodes on V4 |
| **Boundary conditions** | Bottom **1% Z-band** fixed; top **flat-cap** load (`normal_z > 0.99`, `z ≥ z_max − 1e-4`) |
| **Load sweep** | 1 N / 5 N / 10 N compressive cases with unique JSON + VTU + PNG outputs |
| **ParaView** | FEA VTU with `quality_ok`, stress, displacement; mesh VTU with quality cell data |

### Milestones completed

1. **Mesh default refactor** — Reverted broken “strict uniform h + Tet10 + Netgen” experiment; restored adaptive P1 as CLI and library default (`scripts/gmsh_lattice_mesh.py`, `graphite/aristo/gmsh_lattice_mesh.py`).
2. **Delaunay hardening** — `Mesh.Algorithm3D = 1` by default (HXT / algo 10 crashes on V4 classify path).
3. **Netgen removal** — `Mesh.OptimizeNetgen = 0` for lattice STLs; avoids hundreds of irrecoverable illegal linear tets.
4. **Order-elevation safety gate** — If `--order 2` is requested, abort `setOrder(2)` when illegal/inverted linear tets remain; fall back to exportable P1 mesh.
5. **Flat-top BC mode** — `bc_load_mode="flat_top"` routes through `detect_boundary_masks_flat_top_load()` in solver and orchestration scripts (165 load faces vs 214 with naive top Z-band on V4).
6. **Parametric load sweep** — `scripts/run_mirae_lattice_slab_v4_aristo.py` runs 1 / 5 / 10 N with linear stress scaling verified.

### What did not work (see meshing doc for detail)

- HXT 3D on classified V4 skin  
- Uniform h + Tet10 + Netgen on discrete STL boundaries  
- Dense V5 (~900k faces) without decimation — tet count dominated by skin injection  
- V6 / V7 without targeted cap sliver repair — classify / overlapping-facet failures  
- Bulk sliver collapse on dense meshes — creates new overlaps  

Field notes and artifact cleanup guidance live in the output folder:

- `outputs/models/user_spec_rect_prism_3x1p5x5mm/MIRAE_LATTICE_FEA_WORKLOG.md`
- `outputs/models/user_spec_rect_prism_3x1p5x5mm/CLEANUP_GUIDE.md`

---

## Module layout

| File | Role |
|------|------|
| `aristo_config.py` | `AristoConfig` — material, mesh mode, BC mode, quality gates |
| `aristo_solver.py` | `run_aristo`, gmsh FEA mesh strategies, BC application, solve |
| `gmsh_lattice_mesh.py` | TPMS / lattice STL volume mesh (`classifySurfaces`, adaptive sizing) |
| `boundary_detection.py` | Fixed/load face masks (`z_band`, `flat_top`, `flat_top_vertex_plane`) |
| `stiffness_assembly.py` | P1 tet `K` assembly, element stresses |
| `mesh_quality.py` | Aspect ratio / volume gates, `quality_ok` mask |
| `aristo_viz.py` | PNG plots, ParaView VTU export |
| `stress_postprocess.py` | Nodal stress recovery, hotspot metrics |

---

## Quick start

**Implicit field (default):**

```bash
python scripts/run_aristo_fea.py --force-n 1 --h 0.005
python scripts/aristo_volume_mesh_inspect.py --h 0.005
```

**Developer STL override (Mirae / exported meshes):**

```bash
python scripts/run_aristo_fea.py --stl outputs/models/.../Mirae_LatticeSlab_V4_fixed.stl --mesh-mode tpms
```

```python
from graphite.aristo import AristoConfig, run_aristo
from graphite.aristo.implicit_input import AristoImplicitSpec, build_piecewise_splitp_boundary_mesh

mesh, meta = build_piecewise_splitp_boundary_mesh(AristoImplicitSpec(domain="cylinder"))
config = AristoConfig(
    fea_input_mode="implicit",
    fea_gmsh_mesh_mode="single_surface",
    fea_mesh_resolution=0.005,
    ...
)
result = run_aristo(mesh, config)
```

**Mirae STL regression (developer):**

```python
import trimesh
from graphite.aristo import AristoConfig, run_aristo

mesh = trimesh.load("outputs/models/.../Mirae_LatticeSlab_V4_fixed.stl")
config = AristoConfig(
    fea_input_mode="stl",
    youngs_modulus=2800.0,
    poisson_ratio=0.38,
    fea_mesh_resolution=0.15,
    load_direction=(0.0, 0.0, -1.0),
    load_magnitude=pressure_N_per_mm2,
    load_mode="full",
    bc_z_band_fraction=0.01,
    bc_load_mode="flat_top",
    fea_quality_mode="thorough",
    fea_gmsh_mesh_mode="tpms",
    fea_gmsh_adaptive_curvature=True,
    fea_gmsh_char_length_min=0.08,
    fea_gmsh_char_length_max=0.8,
    fea_gmsh_min_elements_per_two_pi=15,
    fea_gmsh_classify_only=True,
)
result = run_aristo(mesh, config)
```

Headless reference run (load sweep):

```bash
python scripts/run_mirae_lattice_slab_v4_aristo.py
```

Mesh-only check (no solve):

```bash
python scripts/gmsh_lattice_mesh.py path/to/Mirae_LatticeSlab_V4_fixed.stl --export-vtu debug.vtu
```

---

## Configuration highlights

### FEA mesh mode

| `fea_gmsh_mesh_mode` | Use when |
|----------------------|----------|
| `tpms` / `lattice` | Discrete TPMS / porous STL — **classifySurfaces** path (default for Mirae slabs) |
| `None` / `single_surface` | Simple closed shells; uniform `MeshSize*` sizing |

Set via config or `ARISTO_GMSH_FEA_MESH_MODE=tpms`.

### FEA quality mode

| Mode | Behavior |
|------|----------|
| `quick` | Single mesh attempt; poor tets excluded from stress stats only |
| `thorough` | Up to 3 remesh attempts with tighter `h` scale (0.85^n); rejects meshes with “giant” edges |

`thorough_use_netgen_optimize` defaults to **False** for lattices.

### Boundary conditions

| Setting | Description |
|---------|-------------|
| `bc_load_mode="z_band"` | Symmetric top/bottom bands (`bc_z_band_fraction`, e.g. 0.01) |
| `bc_load_mode="flat_top"` | Bottom Z-band fixed; load only upward-facing top cap |
| `bc_load_mode="flat_top_vertex_plane"` | **Cube case-study default.** Load/fixed = triangles with all vertices on the dominant top/bottom Z plane **and** \|n_z\| ≈ 1 (not a vertical slice; not a thick Z-band). See [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md) § Aristo compression BCs. |
| `bc_top_z_tolerance_mm` | Default `1e-4` — max distance below `z_max` for load centroids |
| `bc_top_normal_z_min` | Default `0.99` — minimum +Z normal component on load faces |

Explicit face IDs (`fixed_face_ids`, `load_face_ids`) override heuristics when set.

### Stress quality gates

Poor tets are flagged when **any** of: aspect ratio > `max_tet_aspect_ratio` (default **20**), volume out of band, max edge too long. Flag exported as **`quality_ok`** (1 = pass) in VTU cell data. Poor elements remain in the stiffness matrix but are excluded from stress percentile stats.

### Linear solver

| Setting | Description |
|---------|-------------|
| `fea_linear_solver="auto"` | **Default** — MKL PARDISO via `pypardiso` when installed, else SciPy `spsolve` |
| `fea_linear_solver="pardiso"` | Intel MKL PARDISO (multi-threaded direct solve) |
| `fea_linear_solver="scipy"` | SciPy SuperLU / UMFPACK (single-threaded) |
| Env | `ARISTO_LINEAR_SOLVER=auto\|scipy\|pardiso` |

Install: `pip install pypardiso` (see `requirements-aristo.txt`). Timing is logged as `linear_solve_sec` in `mesh_quality_report`.

---

## Script status (March 2026)

| Script | Status | Input / role |
|--------|--------|--------------|
| `scripts/run_mirae_lattice_slab_v4_aristo.py` | **Canonical FEA** | `Mirae_LatticeSlab_V4_fixed.stl` — flat-top BC, 1/5/10 N sweep |
| `scripts/gmsh_lattice_mesh.py` | **Canonical mesh-only** | Any watertight lattice STL — adaptive P1 JSON + optional VTU |
| `scripts/mesh_tpms_lattice_gmsh.py` | **Supported** | STL or STEP — extended TPMS CLI (surface clean, legacy flags) |
| `scripts/fix_lattice_stl_slivers.py` | **Utility** | Targeted cap sliver repair — not bulk collapse |
| `scripts/archive/aristo_legacy/run_mirae_lattice_slab_v3_aristo.py` | **Legacy / obsolete** | Superseded by V4 |
| `scripts/archive/aristo_legacy/run_mirae_lattice_slab_aristo.py` | **Legacy reference** | V2 |
| `scripts/archive/aristo_legacy/run_mirae_lattice_slab_v5_aristo.py` | **Experimental** | Dense V5 |
| `scripts/archive/aristo_legacy/export_mirae_aristo_paraview.py` | **Legacy helper** | V2 ParaView export |
| `experiments/implicit_to_volume/` | **Deprecated** | Use `graphite/case_studies/cube_1mm/` — see [CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md) |

Deleting STLs without updating this table will break **Legacy** and **Experimental** scripts only; **Canonical** paths must keep `Mirae_LatticeSlab_V4_fixed.stl`.

---

## Reference scripts (detail)

| Script | Input STL | Notes |
|--------|-----------|-------|
| `scripts/run_mirae_lattice_slab_v4_aristo.py` | `Mirae_LatticeSlab_V4_fixed.stl` | **Primary** — flat-top BC (default), 1/5/10 N sweep; `--bc-load-mode`, `--solver` |
| `scripts/run_aristo_fea.py` | any watertight lattice STL | General FEA — vertex-plane BC default, `single_surface` mesh |
| `scripts/aristo_volume_mesh_inspect.py` | any lattice STL | Mesh-only JSON + debug VTU |
| `scripts/aristo_clean_and_mesh_stl.py` | MC implicit STL | Island cleanup + volume mesh |
| `scripts/plot_aristo_cross_sections.py` | Aristo VTU | Voronoi cross-section stress PNGs |
| `scripts/generate_piecewise_splitp_implicit.py` | — | Piecewise Split-P implicit STL (single-pass) |
| `scripts/gmsh_lattice_mesh.py` | any watertight lattice STL | Mesh-only JSON + optional VTU |
| `scripts/mesh_tpms_lattice_gmsh.py` | STL or STEP | Extended TPMS mesh CLI |
| `scripts/archive/aristo_legacy/run_mirae_lattice_slab_v5_aristo.py` | `Mirae_LatticeSlab_V5_main.stl` | Dense V5 experiment |
| `scripts/archive/aristo_legacy/run_mirae_lattice_slab_aristo.py` | `Mirae_LatticeSlabTest_V2.stl` | V2 reference |

**Script index:** all Tier 1 entry points are listed in [`scripts/README.md`](../scripts/README.md). Older scripts live under `scripts/archive/`.

---

## Typical V4 results (flat-top, thorough)

See pinned values in **[ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)**.

| Metric | Approximate value |
|--------|-------------------|
| Nodes / tets | ~24.7k / ~74.7k |
| Poor-tet fraction | ~1.3% |
| Load patch area | ~0.81 mm² (165 FEA faces) |
| Max nodal von Mises @ 1 N | **29.124 MPa** (scales linearly: 5 N → 145.6, 10 N → 291.2) |

Outputs per load case:  
`Mirae_LatticeSlab_V4_fixed_{1N,5N,10N}_aristo_report.json`, `_aristo_fea.vtu`, PNGs.

---

## Known limitations & roadmap

- **P2 elements:** Tet10 export supported for debug; stiffness assembly remains P1-only.
- **V5 scale:** Requires STL decimation before classify path is practical for routine FEA.
- **Thorough poor-fraction target:** 0.5% target not always reached on thin-wall lattices; ~1.3% still usable.
- **Load sweep remeshes each case:** Mesh is load-independent; future optimization could reuse `K`.
- **Multi-axis loads:** Phase 2 (config supports uniaxial only today).

### Future direction: scikit-FEM (intentional, not scheduled)

**Current:** Aristo assembles P1 Tet4 stiffness in custom vectorized code (`stiffness_assembly.py`) and solves with `scipy.sparse` / PARDISO. This is **kept by design for now** — narrow scope, no extra dependency, tight coupling to Gmsh array extraction and mesh-quality gates.

**Intent (project owner):** **Migrate to scikit-FEM eventually** for assembly, BC condensation, and element/postprocessing APIs. The custom `scipy.sparse` path remains the production implementation until that migration is explicitly scheduled; scikit-FEM is already used in `graphite/topt` (topology optimization) and the `.inp` validation testbed (`scripts/testbed_scikit_fem_compression.py`).

**Likely migration shape:** keep all Gmsh meshing, BC detection, and quality-gate logic; replace `assemble_global_K` / hand-rolled stress recovery with scikit-FEM `Basis` + `linear_elasticity`. Does **not** remove the need for good volume meshes — see [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md).

### Meshing: implicit field vs STL middleman

Most Aristo mesh pain comes from **discretizing the boundary as an STL** (marching cubes, booleans, classify), not from the linear solve. Starting from the **implicit field directly** (adaptive isosurface or implicit-to-tet meshing) would avoid many STL-specific failures — MC needle triangles, union seam overlaps, `classifySurfaces` breakage — but volume fill and through-wall resolution rules still apply.

**Empirical finding (implicit → volume path, June 2026):** once the MC STL is **repaired first** (island cleanup via `aristo_clean_and_mesh_stl.py` / `remove_floating_islands`) and **`h` is small enough** for strut wall thickness (≥ ~4 tets through the wall; e.g. **h ≤ 0.005 mm** on 1 mm³ dev cubes, **h ≤ 0.01 mm** on 80 µm walls), Gmsh **`single_surface`** Delaunay meshing is **routine** — poor fractions drop to ~1–4% and FEA proceeds without classify or Netgen. The bottleneck is **resolution + clean topology**, not an fundamental Gmsh failure mode. Details: [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md), [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md).

---

## Related documentation

- **[ARISTO_ASSEMBLY_BACKEND_COMPARISON.md](ARISTO_ASSEMBLY_BACKEND_COMPARISON.md)** — Planning brief: custom scipy/PARDISO vs scikit-FEM (for architecture handoff)
- **[ARISTO_MESHING.md](ARISTO_MESHING.md)** — Gmsh pipeline, defaults, failure modes, env vars, troubleshooting
- **[ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)** — pinned V4 mesh/FEA numbers and pass/fail criteria
- **[ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md)** — voxel vs isotropic remesh vs implicit mesh (paper-grade interface stress)
- **[OUTPUTS_MODELS_CLEANUP_HANDOFF.md](OUTPUTS_MODELS_CLEANUP_HANDOFF.md)** — repo-wide `outputs/models/` cleanup and phased git commits
- **[IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md)** — Upstream implicit TPMS generation (orthogonal to FEA mesh path)
- **[PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md)** — Parent prism / Split-P geometry context
