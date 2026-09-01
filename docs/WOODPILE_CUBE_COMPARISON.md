# Cross-hatch woodpile 1 mm cube — Phase 0–2 handoff

**Last updated:** 2026-06-10  
**Status:** ⚠️ **All woodpile Aristo/Vocal results below are invalid** until band-interface geometry is fixed (see [§ Band interface — redo required](#band-interface--redo-required)).  
**Parent docs:** [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md) §8, [VOCAL.md](VOCAL.md), [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md)  
**Split-P counterpart:** [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md)

Third comparison lattice in the 1 mm cube form factor: **piecewise cross-hatch woodpile** vs Split-P piecewise and Split-P linear graded.

---

## Band interface — redo required

**All woodpile cube runs to date must be regenerated** before Aristo/Vocal numbers are used in the three-lattice comparison.

### Requirement

At the **Z = 0.5 mm band interface**, the top section’s in-plane hatch must be **rotated 90°** relative to the bottom section, with a **phase offset** so strut lines do not stay continuous through the interface.

| What we need | Why |
|--------------|-----|
| **90° rotation** (`swap_xy` toggled per band) | Classic cross-hatch: fine bottom hatch ⊥ coarse top hatch |
| **XY phase offset at interface** | Independent per-band `center_void` origins (scaled to each pore) are **not enough** when pore sizes differ — grids can still align through Z = 0.5 mm |
| **Single-pass field** | Per-slab boolean union would freeze prior-band struts (see PIECEWISE §8.4) |

### Current behaviour (incorrect for comparison)

`alternate_band_orientation=True` toggles `swap_xy` (band 0 off, band 1 on) and recomputes `compute_woodpile_xy_origin_box()` **per band** from each pore size. QC on P139/P277 run showed **both bands reporting the same dominant strut axis at mid-band** — the interface does not present a clean 90° offset.

### Code touchpoints (fix TBD)

| Module | Role |
|--------|------|
| `graphite/math/woodpile_anchor.py` | `compute_band_orientation()`, per-band origins |
| `graphite/implicit/piecewise_woodpile.py` | `_band_parameters()`, `woodpile_piecewise_box_single_pass()` |
| `tests/test_woodpile_piecewise_box.py` | Add interface-orientation assertion (XY @ Z=0.5⁻ vs Z=0.5⁺) |

### After fix — rerun pipeline

```powershell
python scripts/generate_cube_1mm_woodpile.py --match-splitp-pores
python scripts/run_cube_1mm_woodpile_aristo.py --match-splitp-pores
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_woodpile_vocal.py --match-splitp-pores
```

Invalidate / archive prior `P200/P400` and `P139/P277` Aristo VTUs and Vocal caches when replacing.

---

## Target geometry

| Property | Value |
|----------|-------|
| Lattice | **Cross-hatch** (`true_woodpile=False`) — through-thickness Z channels |
| Domain | `[0, 1]³` mm, Z up |
| Solid fraction | ~50% intrinsic (woodpile pitch = 2 × pore); voxel SF higher for fine pores |
| Implicit resolution | **0.015 mm** (matches Split-P cube) |
| Pipeline | `woodpile_piecewise_box_single_pass` (analytic box SDF, single-pass) |

### Pore sizing variants

| Variant | Bottom pore | Top pore | Notes |
|---------|-------------|----------|-------|
| **Original** | 200 µm | 400 µm | First comparison attempt |
| **Split-P MIS matched** | **138.6 µm** | **277.1 µm** | Measured MIS from Split-P @ SF=33%, L=500/1000 µm — **not** SF-matched |

Split-P specifies **unit-cell period L**, not pore diameter. MIS pores from `calibrate_tau_at_fixed_period()`:

| Split-P band | L | MIS pore |
|--------------|---|----------|
| Bottom | 500 µm | ~139 µm |
| Top | 1000 µm | ~277 µm |

---

## Phase 0 — Geometry (CLI)

**Code:** `graphite/case_studies/cube_1mm/`  
**Outputs:** `outputs/case_studies/cube_1mm/`  
**CLI:** `scripts/generate_cube_1mm_*.py`, `scripts/run_cube_1mm_woodpile_*.py`

### Canonical CLI

```powershell
python scripts/generate_piecewise_woodpile.py --domain box `
  --width-x-mm 1 --depth-y-mm 1 --height-mm 1 `
  --z-breaks-mm 0,0.5,1.0 --pore-mm 0.1386,0.2771 `
  --resolution-mm 0.015 `
  --out-dir outputs/case_studies/cube_1mm `
  --stem Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50
```

### Code

| Item | Location |
|------|----------|
| Box single-pass API | `woodpile_piecewise_box_single_pass()` |
| Box anchoring | `compute_woodpile_xy_origin_box()` |
| Pore-match constants | `SPLITP_CUBE_MIS_PORE_*_MM` in `woodpile_input.py` |
| Stem helper | `woodpile_cube_1mm_stem()` |
| 1 mm wrapper | `generate_woodpile_cube_1mm.py` |
| Aristo / Vocal wrappers | `--match-splitp-pores`, `--stem` on `run_cube_woodpile_*.py` |

---

## Phase 1 — Aristo FEA

```powershell
python scripts/run_cube_1mm_woodpile_aristo.py --match-splitp-pores
```

| Parameter | Value |
|-----------|-------|
| Mesh | `single_surface`, **h = 0.008 mm** (h=0.005 OOMs on woodpile) |
| Load / E | 1 N, 25.8 MPa |
| BC | `flat_top_vertex_plane` |

### Preliminary results (invalid — pending interface fix)

| Variant | Tets | σ_vm peak | Disp max | Notes |
|---------|------|-----------|----------|-------|
| P200/P400 | ~545k | 19.3 MPa | 0.103 mm | Coarser pores |
| P139/P277 | ~615k | 7.7 MPa | 0.057 mm | Pore-matched; interface not verified |

---

## Phase 2 — Vocal LBM

```powershell
.\.venv_torch\Scripts\python.exe scripts/run_cube_1mm_woodpile_vocal.py --match-splitp-pores
```

Settings: `target_n=128`, `flow_chamber`, Re=5, Ma=0.05, 1000-step preview.

### Preliminary Vocal @ 1000 steps (invalid — pending interface fix)

| Variant | SF (voxel) | k (mm²) | WSS max | vs Split-P piecewise k |
|---------|------------|---------|---------|-------------------------|
| P200/P400 | 0.50 | 6.78×10⁻⁴ | 694 Pa | ~4.2× |
| P139/P277 | 0.57 | 2.00×10⁻⁴ | 559 Pa | ~1.2× |

Pore-matching brought permeability closer to Split-P; SF mismatch (~50% intrinsic vs Split-P 33%) remains.

---

## Three-lattice comparison (Split-P valid; woodpile pending redo)

| Lattice | Grading | Design pores | SF | Aristo | Vocal |
|---------|---------|--------------|-----|--------|-------|
| Split-P piecewise | L 500/1000 µm | ~139 / ~277 µm MIS | 33% | ✅ h=0.005 | ✅ @ 1000 steps |
| Split-P linear | Jacobian W(z) | same | 33% | ✅ | ✅ @ 1000 steps |
| Cross-hatch woodpile | pore 139/277 µm | explicit pore | ~50% | ⚠️ redo | ⚠️ redo |

---

## Open items

- [ ] **Fix band-interface 90° rotation + XY phase offset** (`woodpile_anchor` / `piecewise_woodpile`)
- [ ] Regenerate STL → Aristo → Vocal for pore-matched variant
- [ ] Interface QC: field slice @ Z=0.5 mm + dominant strut axis check per band
- [ ] Converged Vocal run after geometry fix
- [ ] Optional: three-way comparison figure with Split-P
