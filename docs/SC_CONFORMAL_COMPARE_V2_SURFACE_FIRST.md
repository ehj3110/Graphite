# V2 — Surface-First Dual + Volumetric Trim + Gated Stitch

**Agent role:** Implement the decoupled surface-first architecture as comparison version 2.  
**Parent index:** [SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md)  
**Full architecture:** [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md) (authoritative for gates and rationale)

---

## Thesis

The **surface dual** is the conformal authority on the CAD. The **volume lattice** stays essentially Cartesian (VF 50% cull, **no** closest-point cage morph), is Boolean-trimmed (or graph-clipped) to the solid, and reconnects to the dual only through **gated stitches**. Failed stitches → **orphan cull** hanging stubs.

---

## Algorithm (normative)

### A. Volume (no conformal morph)

1. `cull_hex_elements(..., volume_fraction_threshold=0.5)`.
2. Stamp topology on **undeformed** hexes (`morph_hex_scaffold` skipped, or identity).
3. Solidify volume struts; **Boolean intersect** with CAD (or clip struts to interior). Record **cut / hanging** nodes near the boundary.

### B. Independent dual

1. From the same VF complex (or its exposed faces), build skin/dual graph.
2. Place dual nodes with a **surface-native** rule (face-centroid → CAD along normal / exposed-face normal — **not** volume nearest-point ironing).
3. Loft rectangular cage; Boolean flush to CAD (reuse `generate_rectangular_surface_cage` + intersect).

### C. Stitch (hard gates)

For each cut node \(c\), nearest dual node \(d^\star\):

- **Distance:** horizontal reach \(\|(d^\star-c)_{XY}\| \le L\), with \(L = \max(s_x,s_y)\) for anisotropic cells (one unit cell length in plan).
- **Angle:** angle from vertical \(\theta \le 60^\circ\).
- **Fail:** delete \(c\) and hanging stub(s) (orphan cull).
- **Pass:** add stitch strut \(\{c,d^\star\}\).

### D. Union and export

Union volume (post-cull) + dual + stitches; write compare artifacts.

---

## Code ownership

| Path | Purpose |
|---|---|
| `graphite/explicit/compare_v2_surface_first/` | `generate_compare_v2(...)`, stitch helpers, reports |
| `scripts/compare_sc_v2_wrist_rest.py` | Wrist-rest export |
| `tests/compare_v2/` | Gate unit tests (distance/angle reject + orphan) + smoke export |

**Do not** edit V1/V3 packages. Prefer new modules over rewriting `morph_hex_scaffold`.

---

## Deliverables

- `test_parts/mouse_wrist_rest_grid/compare_v2_surface_first_grid_12x12x4_untrimmed.stl` (and/or trimmed)
- `_render.png`, `_report.txt` with `n_cut`, `n_stitched`, `n_rejected_distance`, `n_rejected_angle`, `n_orphaned`
- Import: `from graphite.explicit.compare_v2_surface_first import generate_compare_v2`

---

## Acceptance

- Volume path does **not** use `projection_mode="closest"` cage morph as the skin authority.
- Dual is Boolean-flush capable.
- Stitch gates match [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md) §3.
- Report exposes cull/stitch stats.
