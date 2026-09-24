# Trophy base hex lattices — test-part workstream

**Default hex mesh (Graphite-wide):** **Conformal Dual** — see [`CONFORMAL_DUAL_HEX.md`](CONFORMAL_DUAL_HEX.md). Validated on Toros; reproduced on trophy (`Trophy_base_thin_ConformalDual.stl`). All new hex wiring in Graphite should use this recipe unless a task explicitly requests an experimental variant.

**This note** records trophy test-part exports: **Method A = default**; Methods B–D are comparisons and R&D (two-branch, STL skin, lofted brute-force, Kelvin/Tesseract).

Checkpoint: [`optimization/checkpoints/conformal-dual-hex-2026-05/`](../optimization/checkpoints/conformal-dual-hex-2026-05/CHECKPOINT.md).

---

## Test parts

| STL | Notes | Prior scripts |
|-----|-------|---------------|
| `test_parts/Trophy_base_thin.STL` | Near-cuboid, **coarse** (~12 triangles); flat regions | `export_trophy_base_thin_stl_skin.py`, shrink-only |
| `test_parts/Trophy_base_Rescaled.STL` | Rescaled variant | implicit / gradient exports in `outputs/` |

**Default cell for thin base (existing convention):** `12.7 mm`  
**Default strut diameter (existing):** `1.4 mm` → radius `0.7 mm`

Toros used `strut_radius = (0.06/1.5) × cell_size`; for trophy, keep **explicit mm** strut sizing unless calibrating to a target volume fraction.

---

## Method A — Conformal Dual (**default**)

Same pipeline as Toros Phase 17 (Graphite default hex mesh):

1. `generate_conformed_hex_scaffold` with **`conformal_dual_mode=True`** and **`cull_mostly_external_hexes=True`**.
2. `synthesize_conformal_dual_lattice(hexes)`.
3. `generate_geometry(..., crop_to_boundary=True)`.

**Prepared script:** `scripts/export_trophy_base_thin_conformal_dual.py`

```bash
python scripts/export_trophy_base_thin_conformal_dual.py
```

**Expected outputs:**

- `outputs/Trophy_base_thin_ConformalDual.stl` — merged core + skin
- Optional diagnostics: `*_SkinOnly.stl`, `*_CoreOnly.stl`

**Hold point checklist (Trophy_base_thin — Jun 2026):**

| Run | Cell | Skin | Hexes | Inversions | Output |
|-----|------|------|-------|------------|--------|
| Integer Conformal Dual | 12.7 mm | Hex quad dual | 84 | 0 | `Trophy_base_thin_ConformalDual.stl` |
| **Two-branch (Phase 18)** | 12.7 mm | Surface-path dual + curved ribbons | vol 84 / skin 104 | vol 24 / skin 68 | `Trophy_base_thin_TwoBranch.stl` |
| Decoupled STL skin | **6.35 mm** | STL path + refine | **684** | 4 | `Trophy_base_thin_cell6.35mm_STLskin_refined.stl` |

Decoupled STL skin (6.35 mm experiment — denser, not default): `scripts/archive/exports/export_trophy_base_thin_decoupled_stl_skin.py --refine-skin`

---

## Method C — Two-branch (Phase 18, preferred organic skin)

Volume and skin scaffolds are split: VF-kept hexes carry octahedral core only; **all** boundary-band hexes (including VF-dropped) deform for the skin envelope. Skin topology uses `generate_hex_surface_dual_on_surface_paths` (centroid → edge mid → centroid on deformed quads), exported with CAD-projected polylines (`generate_decoupled_core_and_curved_skin`).

**Script:** `scripts/archive/exports/export_trophy_base_thin_two_branch.py`

```bash
python scripts/archive/exports/export_trophy_base_thin_two_branch.py
```

**Outputs:** `outputs/Trophy_base_thin_TwoBranch.stl`, `outputs/Trophy_base_thin_TwoBranch_iso.png`

**First run (Jun 2026):** watertight STL; synthesis must use **surface-path skin only** — suppress integer face-center dual and drop volume struts whose both endpoints are exterior face centers (`filter_struts_drop_exterior_shell_pairs`, 92 struts on trophy). Without that filter, core octahedral spokes stack on the path dual and look like overlapping surface cages. Default `include_sliver_cages=False` avoids direct centroid chords duplicating path edges on VF-dropped hexes.

---

## Method B — STL-native skin (comparison / flat regions)

Existing path for coarse cuboids — **does not** use integer hex dual:

- `synthesize_hex_volume_with_stl_surface_skin`
- Skin: triangle centroid → edge mid → centroid on STL
- Bridges: cKDTree from hex face centers to nearest skin node

**Script:** `scripts/archive/exports/export_trophy_base_thin_stl_skin.py`

**Gap vs Toros methods:** That script still uses **shrink-only** scaffold with **`cull_mostly_external_hexes=False`**. Before relying on it for production, align scaffold flags with Method A (VF cull + conformal dual **or** at minimum VF cull + pull-only snap).

---

## Method D — Lofted brute-force grid (experimental)

Fixed **8×4×4** hex topology with **lofted** scaffold: uniform **Y** stations, **X/Z** cross-section from mesh slices (`taper_along='y'`). No VF cull; all 128 bricks deform. See [LOFTED_GRADING.md](LOFTED_GRADING.md).

| Variant | Interior rule | Script | Output |
|---------|---------------|--------|--------|
| Two-branch octahedral + surface skin | Octahedral volume + path dual | `scripts/archive/exports/export_trophy_base_thin_brute_force_8x4x4.py` | `Trophy_base_thin_BruteForce8x4x4_TwoBranch.*` |
| Kelvin (24-node) | Volume only | `scripts/archive/exports/export_trophy_base_thin_brute_force_kelvin14.py` | `Trophy_base_thin_BruteForce8x4x4_Kelvin14.*` |
| Tesseract (16-node) | Volume only | `scripts/archive/exports/export_trophy_base_thin_brute_force_tesseract.py` | `Trophy_base_thin_BruteForce8x4x4_Tesseract.*` |

---

## Suggested comparison matrix

| Run | Scaffold | Skin | Script |
|-----|----------|------|--------|
| T1 | Conformal dual + VF cull | Integer dual | `scripts/export_trophy_base_thin_conformal_dual.py` |
| T2 | Two-branch + VF cull | Surface-path + polyline skin | `scripts/archive/exports/export_trophy_base_thin_two_branch.py` |
| T3 | Conformal dual + VF cull | STL-native | TBD (fork stl_skin script with Phase 17 flags) |
| T4 | Legacy shrink-only | STL-native | `scripts/archive/exports/export_trophy_base_thin_stl_skin.py` (baseline) |
| T5 | **Lofted** brute-force 8×4×4 | Kelvin / Tesseract / two-branch | `scripts/archive/exports/export_trophy_base_thin_brute_force_*.py` |

---

## Implementation notes for thin base

1. **Coarse STL + integer dual:** Boundary quads follow **hex** skin after snap; on a 12-triangle STL this may still look stair-stepped on true CAD curvature — acceptable on flat trophy faces.
2. **VF cull:** Even on cuboids, cull removes bbox padding cells; expect fewer hexes than `cull=False` legacy runs.
3. **Cell size:** `12.7 mm` may yield a very coarse interior; consider `extents/4` rule (Toros-style) as an alternate experiment.
4. **Rescaled part:** Duplicate export script with path override or CLI `--stl` argument when adding `Trophy_base_Rescaled.STL`.

---

## Files to touch when executing

| Action | File |
|--------|------|
| Run Conformal Dual export | `scripts/export_trophy_base_thin_conformal_dual.py` |
| Run Two-branch export | `scripts/archive/exports/export_trophy_base_thin_two_branch.py` |
| Update index | `docs/README.md` |
| Archive results | `outputs/diagnostics/hex_test_parts/trophy_base_thin/` or `outputs/` root |

No Streamlit wiring until trophy hold point passes.
