# Implicit → volume meshing (piecewise Split-P cylinders)

R&D path: **implicit TPMS field → marching-cubes STL → Gmsh Tet4 volume mesh → (optional) Aristo FEA**. Complements Mirae production STLs documented in [ARISTO_MESHING.md](ARISTO_MESHING.md).

**Experiment folder:** `experiments/implicit_to_volume/`  
**Upstream implicit math:** [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md), [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md)

---

## Test geometries

### Production-scale (2 × 2 mm)

| Parameter | Value |
|-----------|-------|
| Domain | Ø **2 mm**, height **2 mm** |
| Bottom `Z ∈ [0, 1]` | Split-P, **L = 0.5 mm**, SF **33%** |
| Top `Z ∈ [1, 2]` | Split-P, **L = 1.0 mm**, SF **33%** |
| Transition | Hard band at **Z = 1 mm** (piecewise L/τ, shared `W(z)`) |

Canonical STL: `experiments/implicit_to_volume/output/SplitP_Cylinder2x2mm_piecewise_L500umBottom_L1mmTop_SF33.stl`

**FEA reference (10 N):** `..._10N_aristo_fea.vtu` — flat-top BC, `single_surface` mesh @ **h = 0.12 mm**.

### Dev-scale (2 × 1 mm, 0.5 mm bands)

Same L/SF targets; **0.5 mm** per band; total height **1 mm**. Used for faster mesh iteration.

Canonical STL (15 µm implicit): `SplitP_Cylinder2x1mm_piecewise_L500umBottom_L1000umTop_SF33.stl`

**Wall thickness (fine band, L ≈ 0.5 mm, SF 33%):** strut walls ≈ **80 µm** (0.08 mm) — use this when choosing Gmsh `h`.

---

## Implicit generation pipeline

1. Calibrate Split-P **τ** per band (`calibrate_tau_at_fixed_period`, SF = 33%).
2. **Single-pass** implicit (default): full-domain EDT + piecewise `L(z)`, `τ(z)` → `max(|F|−τ, cad_sdf)` → marching cubes.
3. Trimesh repair (`nondegenerate_faces`, winding, fill holes).

**Avoid** per-band manifold union for volume meshing — union seams produce overlapping facets; Gmsh `classifySurfaces` fails.

```bash
python scripts/generate_piecewise_splitp_implicit.py --out-dir outputs/implicit
python scripts/generate_piecewise_splitp_implicit.py \
  --domain cylinder --height-mm 2 --band-height-mm 1 --resolution-mm 0.015
```

Legacy dev wrapper (same CLI): `experiments/implicit_to_volume/generate_piecewise_splitp_cylinder_2x2mm.py`

| Implicit pitch | Effect |
|----------------|--------|
| 0.02 mm | Default; ~240k faces (2×2 mm) |
| 0.015 mm | **Recommended** for 2×1 mm dev part (~227k faces) |
| 0.01 mm | Finer skin but *more* STL noise; often **worse** `poor_fraction` |

---

## Volume meshing (no FEA)

### Mode: `single_surface` (implicit MC STLs)

Marching-cubes lattice STLs are **discrete soups**. Use **`fea_gmsh_mesh_mode="single_surface"`** — one boundary shell, no `classifySurfaces`.

Gmsh sizing (single-surface path in `aristo_solver._generate_fea_mesh_single_surface`):

| Option | Value |
|--------|--------|
| `Mesh.Algorithm3D` | **1** (Delaunay) |
| `Mesh.OptimizeNetgen` | **0** |
| `Mesh.MeshSizeMin` | **0.5 × h** |
| `Mesh.MeshSizeMax` | **2.0 × h** |

`h` = `AristoConfig.fea_mesh_resolution` (mm), anchor characteristic length.

**TPMS/classify mode** remains for Mirae V4_fixed; it **fails** on implicit union/MC STLs (overlapping facets).

### Practical recipe (repair + fine `h`)

June 2026 implicit-to-volume work established a repeatable pattern — meshing failures were **not** fundamental once prerequisites were satisfied:

| Step | Action | Why |
|------|--------|-----|
| 1 | **Clean topology** — `scripts/aristo_clean_and_mesh_stl.py` (floating MC island removal) | Extra shells and seam debris make `K` singular or pollute BC detection |
| 2 | **Size to wall thickness** — `h` such that ≥ ~4 tets span the strut wall (80 µm walls → `h ≤ 0.02 mm`; 1 mm³ dev cubes @ `h = 0.005` → ~1–2% poor) | Coarse `h` leaves through-thickness slivers and drives `poor_fraction` to ~6–11% |
| 3 | **`single_surface` mesh mode** | Avoids classify on MC skins; Delaunay 3D + Laplace only |

After steps 1–3, Gmsh volume meshing **proceeds routinely**; remaining poor tets are thin-wall slivers at expected rates (~1–4%), not classify/PLC blow-ups. Finer implicit MC pitch alone does **not** substitute for fine **`h`** — STL noise can worsen `poor_fraction` at 10 µm MC vs 15 µm.

**Long-term:** mesh directly from the implicit field (skip STL) to avoid MC skin artifacts entirely; through-wall sizing rules still apply. See [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md) §3.

### Wall-thickness sizing (80 µm struts)

Target **≥ 4 linear tets through the strut wall** for continuum fill:

\[
h \lesssim t_\text{wall} / n_\text{through} \quad\Rightarrow\quad h \lesssim 0.08\,\text{mm} / 4 = 0.02\,\text{mm}
\]

| Case | `h` (mm) | CL min–max (mm) | Tets | Poor % | Max edge (mm) | AR median | Notes |
|------|----------|-----------------|------|--------|---------------|-----------|-------|
| **h040** | 0.04 | 0.02–0.08 | 396k | **9.2%** | 0.151 | 3.24 | Max edge ~2× wall — under-resolved |
| **h020** | 0.02 | 0.01–0.04 | 477k | **6.6%** | 0.079 | 2.53 | ~1 tet across 80 µm wall |
| **h010** | 0.01 | 0.005–0.02 | 859k | **3.6%** | 0.041 | 1.69 | **Recommended** for through-wall fill |

Outputs: `experiments/implicit_to_volume/output/mesh_sweep_2x1mm/*_wallSweep_h{040,020,010}_volume_mesh_debug.vtu`  
Summary: `..._wallSweep_summary.json`

```bash
python scripts/aristo_volume_mesh_inspect.py --stl ... --h 0.02
python experiments/implicit_to_volume/mesh_wall_thickness_sweep.py
```

---

## Mesh quality: `poor_fraction` and gate breakdown

After meshing, Aristo scores each tet (`quality_ok` in VTU). **Poor** if any gate fails:

| Gate | Default |
|------|---------|
| Aspect ratio (max edge / min edge) | ≤ **20** |
| Volume | [1% × median, 50× median] |
| Max edge | ≤ **12 × h** |

**`poor_fraction`** = fraction failing any gate. Poor tets stay in the mesh; stress stats exclude them.

Reports include **`quality_gate_breakdown`**:

- `fail_any_gate` — per-gate counts (overlapping)
- `primary_failure_among_poor` — exclusive main reason

Typical implicit-cylinder failures: **`min_volume`** + **`aspect_ratio`** on thin struts (not `max_edge`).

| Case | Poor fraction | Notes |
|------|---------------|-------|
| Mirae V4 @ h=0.15 | ~**1.3%** | Production baseline |
| 2×2 implicit @ h=0.12 | ~**11%** | MC STL, no surface clean |
| 2×1 implicit @ h=0.07 | ~**11%** | Same gates |

---

## Scripts (canonical)

| Script | Role |
|--------|------|
| `scripts/generate_piecewise_splitp_implicit.py` | **Piecewise** Split-P cylinder/box STL (single-pass EDT) |
| `scripts/aristo_volume_mesh_inspect.py` | Volume mesh + quality gate JSON + debug VTU |
| `scripts/aristo_clean_and_mesh_stl.py` | Island cleanup + volume mesh |
| `scripts/run_aristo_fea.py` | Compression FEA (vertex-plane BC, PARDISO default) |
| `scripts/plot_aristo_cross_sections.py` | Voronoi cross-section von Mises PNGs |

**Library:** `graphite.implicit.piecewise_bands`, `graphite.aristo.fea_runner`, `graphite.aristo.volume_mesh_inspect`, `graphite.aristo.cross_section_viz`

**R&D sandbox** (wrappers + local output): `experiments/implicit_to_volume/`

| Experiment-only | Role |
|-----------------|------|
| `mesh_wall_thickness_sweep.py` | h sweep on 2×1 mm dev cylinder |
| `generate_splitp_cube_1mm_variants.py` | 1 mm³ dev form-factor STLs (not canonical) |

---

## 1 mm³ Split-P cube case study

Dev-scale **axis-aligned cube** `[0, 1]³` mm (Z up) for comparing **piecewise** vs **Jacobian-integrated linear grading** at SF **33%**, then volume meshing, Aristo FEA, and 2D cross-section stress maps.

### Geometries

| Variant | L(z) profile | Phase / notes | Canonical STL |
|---------|--------------|---------------|---------------|
| **Piecewise** | 0.5 mm bottom half, 1.0 mm top half | Hard band at **Z = 0.5 mm** | `SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33.stl` |
| **Linear graded** | 0.5 → 1.0 mm over height | **Jacobian-integrated** `W(z) = ∫ω(z)dz` (not chirp `z·ω`); in-plane phase origin **(0.5, 0.5) mm** | `SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW.stl` |

**Calibration:** `calibrate_tau_at_fixed_period()` holds **L fixed** per band and solves τ only for SF = 33%. Do **not** use `calibrate_tpms_point(target_pore_mm=L)` here — that API treats the argument as **inscribed pore diameter**, not unit-cell period, and over-opens the lattice.

Implicit pitch default **15 µm** (`--resolution-mm 0.015`).

### End-to-end commands

```bash
# Canonical pipeline (any form factor)
python scripts/generate_piecewise_splitp_implicit.py --out-dir outputs/implicit
python scripts/aristo_clean_and_mesh_stl.py --stl outputs/implicit/....stl --h 0.005
python scripts/run_aristo_fea.py --stl ..._cleaned.stl --h 0.005 --force-n 1
python scripts/plot_aristo_cross_sections.py --vtu ... --output ...
```

### 1 mm³ dev cube case study (optional)

```bash
# Dev-only generators under experiments/implicit_to_volume/
python experiments/implicit_to_volume/generate_splitp_cube_1mm_variants.py
python scripts/aristo_clean_and_mesh_stl.py --stl experiments/implicit_to_volume/output/....stl --h 0.005
python scripts/run_aristo_fea.py --stl ..._cleaned.stl --h 0.005 --force-n 1 --stem SplitP_Cube1mm_piecewise_h005_1N
python scripts/plot_aristo_cross_sections.py --vtu ... --output ...
```

Legacy experiment wrappers delegate to the `scripts/` entries above.

### Volume mesh @ h = 0.005 mm

After island cleanup, `single_surface` mesh on the 1 mm cube:

| Variant | ~Tets | Poor % (quick gates) |
|---------|-------|----------------------|
| Piecewise | ~1.19 M | ~1.6% |
| Linear graded | ~1.29 M | ~1.2–1.3% |

`h = 0.005` was chosen as the FEA resolution after a poor-fraction sweep on cleaned STLs (`h = 0.02` / `0.01` were ~6–7% poor).

### FEA reference (1 N)

| Case | Nodal peak σ_vm | Element peak σ_vm (raw P1) | PARDISO solve |
|------|-----------------|----------------------------|---------------|
| Piecewise `..._h005_1N` | ~125 MPa | ~1765 MPa | ~22 s |
| Graded `..._centerPhase_h005_1N` | ~31 MPa | ~2228 MPa | ~29 s |

BC mode **`flat_top_vertex_plane`**: load/fix only faces whose **all three vertices** lie on the top/bottom cap plane (≥100 vertices at plane, `|n_z| > 0.99`). See `graphite/aristo/boundary_detection.py`.

Stress field in VTU: **`von_mises_nodal_MPa`** (volume-weighted nodal recovery) and **`von_mises_element_MPa`** (raw per-tet constants). Cross-section figures use **nodal** stress for display.

---

## Cross-section stress figures (Voronoi solid fill)

**Script:** `plot_cube_aristo_cross_sections.py`  
**Outputs:**

| Figure | Plane | Purpose |
|--------|-------|---------|
| `SplitP_Cube1mm_aristo_cross_section_XZ_midY.png` | XZ @ **Y = 0.5 mm** (±12 µm slab) | Mid-height cut through both grading regions |
| `SplitP_Cube1mm_aristo_cross_section_XY_midZ_slab.png` | XY @ **Z = 0.5 mm** (±50 µm slab) | Band interface / piecewise transition |

Default inputs: piecewise and center-phase graded `*_h005_1N_aristo_fea.vtu` + matching `*_aristo_report.json`.

### Why cells look overlapped (not true Voronoi)

The current fill is **nearest-neighbor assignment + radial disk clip** (`void_distance_mm`), not a planar Voronoi tessellation. Disks larger than ~½ the in-plane node spacing paint past natural cell boundaries, so struts look **thick and overlapping** even though each pixel has only one owner. See [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md) for alternatives (true Voronoi, IDW heatmap).

### Visualization method

Goal: **solid-looking** strut cross-sections with **white voids**, comparable hotspot contrast to early nodal scatter plots, without mesh-sliver artifacts from raw element coloring.

```
VTU nodal σ_vm
    → thin slab node extract (2D projection)
    → in-plane dedupe (XY slab: max σ per rounded (x,y))
    → 2D Voronoi nearest-seed raster (480×480)
    → cap radius void_distance_mm (default 18 µm) → NaN / white outside
    → normalize ÷ slice peak per panel → turbo colormap
    → optional material-footprint contour (gray)
```

**Step by step**

1. **Load** `von_mises_nodal_MPa` from `*_aristo_fea.vtu` (PyVista).
2. **Slab filter** — keep nodes within half-thickness of the cut plane (captures thin struts an exact zero-thickness plane misses).
3. **Project** to 2D: XZ uses `(x, z)`; XY uses `(x, y)`. For the XY slab, collapse duplicate in-plane keys (0.1 µm rounding) keeping **max** nodal stress.
4. **Voronoi solid fill** — for each raster pixel, assign the **nearest slab node**’s stress (`scipy.spatial.cKDTree`). Paint only if distance ≤ **`void_distance_mm`** (default **0.018 mm**); otherwise leave **white**. This is a clipped Voronoi diagram: cells tile among seeds but do not bridge large pores.
5. **Normalize** each panel independently: `σ / slice_peak` so piecewise vs graded are comparable in hue. Titles also report model-wide nodal peak from the JSON report.
6. **Outline** — `matplotlib.contour` at 0.5 on the binary material mask (disable with `--no-boundary`).

### Approaches we tried (and why not)

| Approach | Issue |
|----------|-------|
| Nodal scatter + `griddata` / bilinear raster | Heatmap bled into voids; looked like low-stress material |
| Nodal bin mean/max raster + distance mask | Worked but pointillist, not solid |
| **PyVista plane cut + element face polygons** | Missing thin struts; **raw P1 element** outliers (1000+ MPa on slivers) dominated normalization → almost all dark blue |
| **Voronoi nodal fill + void cap** (current) | Solid struts, white voids, nodal hotspots preserved |

Use **nodal** stress for publication-style cross-sections; reserve **element** stress for mesh-quality debugging or iso-surface views (`plot_aristo_von_mises_isosurfaces`).

### Tuning CLI

```bash
python experiments/implicit_to_volume/plot_cube_aristo_cross_sections.py \
  --y-half-thickness-mm 0.012 \
  --z-half-thickness-mm 0.05 \
  --void-distance-mm 0.018 \
  --voronoi-scale 0        # optional extra cap at scale × NN spacing
```

| Flag | Default | Effect |
|------|---------|--------|
| `--void-distance-mm` | `0.018` | Max paint radius from each seed; larger → thicker solid struts |
| `--y-half-thickness-mm` | `0.012` | XZ slab half-thickness (~2× `h`) |
| `--z-half-thickness-mm` | `0.05` | XY slab half-thickness (band overlap) |
| `--voronoi-scale` | `0` | If >0, also cap cells at `scale ×` nearest-neighbor spacing |
| `--no-boundary` | off | Skip gray footprint contour |

---

## Outputs (ParaView)

| Suffix | Content |
|--------|---------|
| `*_volume_mesh_debug.vtu` | Tet4 + `quality_ok`, `aspect_ratio`, `max_edge_mm` |
| `*_volume_mesh_report.json` | Counts, thresholds, `quality_gate_breakdown` |
| `*_10N_aristo_fea.vtu` | Displacement + von Mises (FEA only) |

Slice at **Z = band height** to inspect hard L transition.

---

## Related

- [ARISTO_MESHING.md](ARISTO_MESHING.md) — Mirae TPMS classify path
- [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md) — implicit-to-mesh vs STL remesh
- [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md) — V4 pinned numbers
