# Baseball Balls (Ø74 mm) — Project Catalog & Workflows

Graphite’s baseball R&D track: radially graded **TPMS** solids plus **surface strut cages** (conformal Tri/Sq, Voronoi, A15 surface dual). Planned later on the same diameter conventions: tennis (67 mm), ping-pong (40 mm).

**Status (Jul 2026):** FINAL catalog is complete — every lattice has a plain (`*_noSeam`) and a solid baseball-seam (`*_seam`) sibling. Mesh binaries were uploaded externally; regenerate locally with the scripts in §7 (STLs are not kept in this workspace).

---

## 0. Directory rules

**`outputs/Balls/baseball/FINAL/` is approved deliverables only.**  
Do **not** write new experiments there unless explicitly promoted.

```
outputs/Balls/baseball/
├── FINAL/
│   ├── TPMS/                         # graded implicit lattices
│   ├── scaffolds/                    # conformal Tri_* / Sq_* cages
│   ├── voronoi/                      # Spherical Voronoi cages
│   ├── tpms_baseball_unitcell_ref.json
│   └── README.md
├── a15_review/                       # A15 dual experiments (square + jointed cyl)
└── (scratch / obsolete review dirs may appear — do not treat as finals)
```

Canonical write-up: this file. Short pointers: `outputs/Balls/README.txt`, `FINAL/README.md`.

### Local mesh policy
After upload, delete `*.stl` under `outputs/Balls/baseball/` to reclaim disk (~2.6 GB for a full regen). Keep this doc, `FINAL/README.md`, calibration JSON, and generators. Re-run §7 to rebuild.

---

## 0b. File naming

Folders stay as-is (`FINAL/TPMS`, `FINAL/scaffolds`, `FINAL/voronoi`).  
**Filenames are only:**

```
{latticeId}_{noSeam|seam}.stl
```

| Part | Rule |
|------|------|
| `latticeId` | Short camelCase identity (`gyroid`, `triKelvin`, `sqGrid`, `voronoi_fib200`, …). Optional `_V2` (etc.) only when two otherwise identical lattices must be distinguished. |
| `noSeam` / `seam` | Plain cage/solid vs same geometry Boolean-unioned with a solid baseball-seam ribbon |

**Examples:** `gyroid_noSeam.stl`, `triKelvin_seam.stl`, `voronoi_rand200_V2_noSeam.stl`

No diameter, cell, width, thickness, or calibration strings in the filename — those live in this doc / generator tables.

---

## 0c. FINAL catalog (approved)

| Folder | Plains (`*_noSeam`) | Seams (`*_seam`) | Count |
|--------|---------------------|------------------|-------|
| `TPMS/` | `gyroid`, `diamond`, `schwarz`, `schwarzDiamond`, `neovius`, `neovius_V2`, `splitP` | same IDs | 14 |
| `scaffolds/` | `triTetrahedral`, `triIcosahedral`, `triKelvin`, `triTesseract`, `triRhombic`, `sqGrid`, `sqIcosahedral`, `sqKelvin`, `sqTesseract` | same IDs | 18 |
| `voronoi/` | `voronoi_fib200`, `voronoi_fib300`, `voronoi_rand{200,300}_V{1,2,3}` | same IDs | 16 |

**Total FINAL meshes:** 48 STLs (24 plain + 24 seam).

---

## 1. Shared geometry

| Parameter | Value |
|-----------|--------|
| Baseball diameter | **74 mm** |
| Design radius `R` | **37 mm** |
| Coordinate frame | Sphere centered at `(R, R, R)` |
| TPMS generate | Often oversize (~Ø75 mm) then Boolean truncate to Ø74 |
| Square strut cages (after flush trim) | Envelope ~**Ø73.5 mm** (`R − 0.25` trim sphere) |

---

## 2. Square surface-strut workflow (required)

Surface duals with **square** struts must use this joint-cleanup recipe. It replaced pure prism unions, which left messy strut junctions.

### Recipe
1. Build the surface **wire graph** (nodes on / near the sphere, undirected edges).
2. Sweep each edge as a straight **surface-normal-oriented** rectangular prism:
   - In-plane **width** = tangential strut width `W` (typically 1.0 or 1.6 mm).
   - Along the outward surface normal: **thickness** `T` (catalog default **5.0 mm**; if targeting a thin nominal, use **`T_nominal + 2.0 mm`** — see print rule below), plus **+0.25 mm oversize** biased outward for joint cleanup.
3. CSG-union the prisms (manifold boolean).
4. **Boolean ∩** a sphere of radius **`R − 0.25 mm`** (same center). This flushes the outer joints.

API: `sweep_square_straight_struts(..., side=W, thickness=T, normal_oversize=0.25, trim_sphere_radius=R-0.25)`.

### Print thickness rule (surface dual) — required for future parts

**Surface duals must be rectangular**, not cylinders (unless explicitly reviewing a cylindrical path).

| Dimension | Rule |
|-----------|------|
| In-plane **width** | Nominal strut size `W` |
| Surface-normal **thickness** | Catalog default **5.0 mm**. When starting from a thin target, author **`T_nominal + 2.0 mm`** |

**Why (print lesson, Jul 2026):** Thin ~1×1 mm skins printed weak. Pad thin nominals by **+2 mm**. Sphere **Boolean ∩** cleanup can also clip roughly half the normal-direction stock if the strut straddles the skin. Default cage authoring remains **`T = 5.0 mm`**.

### Why (joint cleanup)
Growing slightly in the normal direction, then cutting with a slightly smaller ball, removes overlapping prism corners at nodes without ad-hoc filleting. Empirically far cleaner than union-only square bars.

### APIs
| Symbol / helper | Location |
|-----------------|----------|
| `sweep_square_straight_struts(..., normal_oversize=0.25, trim_sphere_radius=R-0.25)` | `experiments/conformal_v2/conformal_utils.py` |
| `_square_prism_surface_oriented(..., normal_oversize=…)` | same |
| `trim_mesh_with_inset_sphere(mesh, design_radius=R, inset_mm=0.25)` | same — post-process existing STLs |
| Batch re-trim FINAL scaffolds + voronoi | `tests/trim_balls_square_struts_inset.py` |

```powershell
# Re-apply inset trim to existing square-strut STLs in FINAL/
python tests/trim_balls_square_struts_inset.py
```

Generators that already bake this in:
- `tests/generate_balls_conformal_patterns.py`
- `tests/generate_balls_voronoi_surface.py`
- `tests/generate_balls_a15_surface_dual.py` (square path)

---

## 2b. Solid baseball seams (required for `*_seam`)

Every FINAL lattice ships a **seam** sibling: the approved plain mesh Boolean-unioned with a solid baseball-seam ribbon, then trimmed with the same **`R − 0.25`** sphere used for square cages.

### Recipe (fast path — preferred)
1. Build / load the approved `{latticeId}_noSeam.stl`.
2. Sample the baseball seam curve (`baseball_seam_points`, `A≈0.44`, **8192** samples) on the design sphere.
3. Sweep a rectangular ribbon along the curve: width **5 mm**, thickness **5 mm**, `normal_oversize=0.25` via `_square_prism_surface_oriented`.
4. CSG-union ribbon segments → one ribbon mesh; union ribbon with the plain mesh.
5. Boolean ∩ sphere of radius **`R − 0.25 mm`**.
6. Write `{latticeId}_seam.stl`.

Do **not** rebuild conformal / Voronoi graphs just to add a seam — reuse the uploaded plains.

### Scripts
| Family | Script |
|--------|--------|
| Scaffolds | `tests/generate_balls_scaffold_seams_from_plain.py` |
| Voronoi / Fib | `tests/generate_balls_voronoi_seams_from_plain.py` |
| TPMS | Same ribbon + union recipe over `FINAL/TPMS/*_noSeam.stl` (underscores in IDs such as `neovius_V2` are allowed) |

Curve helper: `tests/generate_balls_a15_seam_graded.py` → `baseball_seam_points`.

---

## 3. TPMS baseballs (implicit) — `FINAL/TPMS/`

### Intent
Radially grade period `L(r)` so **pores** go from coarse at the center to fine at the skin at **constant solid fraction**.

### Final recipe
| Target | Value |
|--------|--------|
| Solid fraction | **25%** (`τ` from `|F|` quantile; solve `L` for MIS pore) |
| Center pore | **25.4 mm** |
| Skin pore | **6.35 mm** |
| Radial grade | `L(r)` vs spherical radius; outer **plateau** from ~**65% of R** |
| Mesh resolution | ~**0.4 mm**; Neovius also at **0.3 mm** (`neovius_V2`) |

### Calibration notes
Lattice-agnostic period / wall heuristics failed (pores not matched across types; `τ ≈ πw/L` badly wrong vs EDT walls). Finals use **point-wise unit-cell** `calibrate_period_for_pore_at_sf`, then interpolate `L(r)` only.

Reference: `graphite/implicit/tpms_baseball_unitcell_ref.json` (copied under `FINAL/`).

### Lattice set (short names)

| `latticeId` | Notes |
|-------------|--------|
| `gyroid` | |
| `diamond` | |
| `schwarz` | |
| `schwarzDiamond` | |
| `neovius` | ~0.4 mm res |
| `neovius_V2` | finer ~0.3 mm res |
| `splitP` | |

Each writes `{id}_noSeam.stl`; seam siblings via §2b.

```powershell
python tests/generate_balls_baseball_tpms.py
# writes outputs/Balls/baseball/FINAL/TPMS/*_noSeam.stl
# then apply §2b seam recipe for *_seam.stl
```

Related: `graphite/implicit/tpms_parameter_lut.py`, `calibrate_period_for_pore_at_sf` in `calibration.py`.

---

## 4. Conformal Tri_* / Sq_* scaffolds — `FINAL/scaffolds/`

### Intent
Map the **coaster** Tri/Sq strut recipes onto the ball **evenly**. Stereographic / UV wraps were tried and **rejected** (pole densification). Finals use conformal unit-cell tiling + per-face operators.

### Method
1. Goal **cell size** → tile interior.
2. **Tri:** A15 conformal tet skin → ordered boundary triangles.
3. **Sq:** Hex conformal dual scaffold → ordered boundary quads.
4. Apply coaster micro-rule on each face (Rhombic = shared-edge centroid dual).
5. Radial-project nodes onto the design sphere.
6. Square strut sweep with **§2 oversize + inset trim**.

Code: `scripts/coasters/surface_pattern_ops.py`, `scripts/coasters/tri_sq_patterns.py`.

### Tuned catalog

In-plane **width** preserved from prior sides; surface-normal **thickness = 5.0 mm** for all (+0.25 oversize then `R−0.25` trim).

| Pattern | `latticeId` | Cell (mm) | Width (mm) | Thickness (mm) |
|---------|-------------|-----------|------------|----------------|
| Tri_Tetrahedral | `triTetrahedral` | 12.7 | 1.6 | 5.0 |
| Tri_Icosahedral | `triIcosahedral` | 12.7 | **1.0** | 5.0 |
| Tri_Kelvin | `triKelvin` | **25.4** | 1.6 | 5.0 |
| Tri_Tesseract | `triTesseract` | **25.4** | **1.0** | 5.0 |
| Tri_Rhombic | `triRhombic` | 12.7 | 1.6 | 5.0 |
| Sq_Grid | `sqGrid` | **6.35** | 1.6 | 5.0 |
| Sq_Icosahedral | `sqIcosahedral` | 12.7 | 1.6 | 5.0 |
| Sq_Kelvin | `sqKelvin` | 12.7 | 1.6 | 5.0 |
| Sq_Tesseract | `sqTesseract` | 12.7 | **1.0** | 5.0 |

| Micro-rule | Meaning on a face |
|------------|-------------------|
| Tetrahedral / Grid | Face edges (primal) |
| Icosahedral | Edge midpoints |
| Kelvin | Edge ⅓-points (truncated rings) |
| Tesseract | Outer + inscribed + spokes |
| Rhombic (Tri) | Adjacent-face centroid dual (= A15 cyan dual topology) |

At equal cell size, A15 triangle skins are denser than hex quad skins — density overrides above compensate.

Tables live in `tests/generate_balls_conformal_patterns.py` (`CELL_BY_LABEL`, `STRUT_BY_LABEL`).

```powershell
python tests/generate_balls_conformal_patterns.py
python tests/generate_balls_scaffold_seams_from_plain.py
# writes FINAL/scaffolds/ (*_noSeam + *_seam)
```

---

## 5. Spherical Voronoi — `FINAL/voronoi/`

### Intent
Organic surface cages from a spherical Voronoi diagram of seed points.

### Seed placement
| Mode | Description |
|------|-------------|
| **Fibonacci** | Golden-angle spiral — deterministic, approximately even coverage |
| **Random** | i.i.d. Gaussian 3-vectors → normalize (uniform on the sphere); fixed RNG family for reproducibility |

Fibonacci ≠ random. Prefer Fibonacci for even density; random for variety / coaster-like organic look.

### Catalog

| Mode | Seeds | Variants | `latticeId` examples | Cross-section |
|------|-------|----------|----------------------|---------------|
| Fibonacci | 200, 300 | 1 each | `voronoi_fib200` | width **1.6** × thickness **5.0** mm + §2 trim |
| Random | 200, 300 | **3** each (`_V1`…`_V3`) | `voronoi_rand200_V2` | width **1.6** × thickness **5.0** mm + §2 trim |

```powershell
python tests/generate_balls_voronoi_surface.py
python tests/generate_balls_voronoi_seams_from_plain.py
# writes FINAL/voronoi/ (*_noSeam + *_seam)
```

---

## 6. A15 surface dual (review) — `a15_review/`

### Intent
Conformal **A15 Kagome / cyan** surface dual (same graph family as Tri_Rhombic), without C15. Coaster C15 Z-slices are **skipped** on balls — only A15 conformal skin.

### Cell / sizing
- Cell **12.7 mm**
- Rectangular: width **1.6 mm**, thickness **5.0 mm** + §2 oversize/trim
- Cylinder: radius **0.8 mm** (review path; optional)

### Cylinder / joint cleanup (Part2 method)
For cylindrical A15 duals:
1. Cylinder per strut.
2. Sphere at every used node with `r_joint = strut_r × 1.05`.
3. Robust manifold **CSG union** (`union_lattice_with_spherical_joints`).
4. **Boolean ∩** design sphere CAD (`boolean_intersect_with_cad`) to flush the exterior.

```powershell
python tests/generate_balls_a15_surface_dual.py
```

Typical outputs under `a15_review/`:
- `..._sq1.6mm.stl` — square + inset trim  
- `..._cyl0.8mm_joints_union.stl` — cylinders + joints, union only  
- `..._cyl0.8mm_joints_trimmed.stl` — union then ∩ sphere (**preferred cylindrical look**)

Promote into `FINAL/` only when explicitly approved.

---

## 7. Script index

| Script | Role |
|--------|------|
| `tests/generate_balls_baseball_tpms.py` | Graded TPMS plains → `FINAL/TPMS/*_noSeam.stl` |
| `tests/generate_balls_conformal_patterns.py` | 9 Tri/Sq conformal plains → `FINAL/scaffolds/` |
| `tests/generate_balls_scaffold_seams_from_plain.py` | Union scaffold plains with solid baseball seam → `*_seam.stl` |
| `tests/generate_balls_voronoi_surface.py` | Fibonacci + random Voronoi plains → `FINAL/voronoi/` |
| `tests/generate_balls_voronoi_seams_from_plain.py` | Union Voronoi plains with solid baseball seam → `*_seam.stl` |
| `tests/generate_balls_a15_surface_dual.py` | A15 dual square + jointed cylinder → `a15_review/` |
| `tests/trim_balls_square_struts_inset.py` | Batch `R−0.25` Boolean trim of FINAL square cages |

Full FINAL rebuild (plains + seams):

```powershell
python tests/generate_balls_baseball_tpms.py
# TPMS seams: reuse §2b ribbon union on FINAL/TPMS/*_noSeam.stl
python tests/generate_balls_conformal_patterns.py
python tests/generate_balls_scaffold_seams_from_plain.py
python tests/generate_balls_voronoi_surface.py
python tests/generate_balls_voronoi_seams_from_plain.py
```

Obsolete / superseded generators (do not use for finals): stereographic `generate_balls_coaster_patterns.py`, early dual review scripts, thickness-graded seam experiments (`generate_balls_a15_seam_graded.py` is kept only for the seam **curve** helper).

---

## 8. Historical cleanup

Removed when `FINAL/` was established:

- Interim TPMS (radialPlateau, SF=20%, wall-calibrated knots)
- Early Kagome / hex dual size sweeps
- Stereographic `coaster_patterns/`
- Scratch spheres and `_work` trees at the baseball root
- Long descriptive filenames (diameter / cell / calibration strings) → short `{latticeId}_{noSeam|seam}.stl`

Local `*.stl` under `outputs/Balls/baseball/` may be deleted after external upload; regenerate with §7.

---

## 9. Related docs

- Coasters: [`docs/coasters/walkthrough.md`](coasters/walkthrough.md)
- TPMS engine / calibration: [`IMPLICIT_ENGINE.md`](IMPLICIT_ENGINE.md), [`TPMS_CALIBRATION_WORKFLOW.md`](TPMS_CALIBRATION_WORKFLOW.md)
- Hex conformal dual: [`CONFORMAL_DUAL_HEX.md`](CONFORMAL_DUAL_HEX.md)
- Spherical-joint + CAD intersect (Part2): `tests/run_part2_boolean_trim.py`, `union_lattice_with_spherical_joints` in `conformal_utils.py`
