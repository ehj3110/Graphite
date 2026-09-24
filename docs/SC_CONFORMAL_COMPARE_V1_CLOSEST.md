# V1 — Closest-Point Morph (Comparison Baseline)

**Agent role:** Package and freeze the **current** closest-point SC conformal path as comparison version 1.  
**Parent index:** [SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md)

---

## Thesis

Retain a VF-gated SC hex cage, project every **exposed** scaffold corner to the **nearest point on the CAD**, Laplacian-relax interior corners, stamp topology, optionally build surface dual from the **already-morphed** skin. This is the known baseline: visually reasonable on the wrist rest, but stair side-wall nodes often snap to floor/top.

---

## Algorithm (normative)

1. Load CAD (`Mouse wrist rest v1.stl` for compare).
2. `cull_hex_elements(..., volume_fraction_threshold=0.5, mode="conformal")`.
3. `morph_hex_scaffold(..., projection_mode="closest", max_projection_factor=0.5, snap_outside_nodes=True, stair_step_normal_gate=True, cull_collapsed_hexes=False)`  
   - Use `safe_signed_distance` semantics inside any custom snap (positive = outside).  
   - Do **not** use `face_normal` for V1.
   - The stair-step normal gate re-aims *interior* corners whose closest CAD point lies behind a face the corner does not expose. This is not `face_normal` projection: it fires only on corners closest-point would drag through their own brick, and it takes the **nearest** allowed normal hit, so it cannot overshoot laterally.
4. Stamp `rule_name` (`grid` primary; also produce `octahedral` if cheap).
5. Optional: rectangular surface cage from skin struts (existing generator path) — label clearly if included.
6. Solidify cylinders + joints; export STL + render + report.

---

## Known failure modes (document in report, do not “fix” in V1)

- Stair corners: nearest patch is often floor/top, not side wall. **Addressed** by `stair_step_normal_gate` for the re-entrant notch case — the corner that keeps hexes above *and* below it, exposes only lateral faces, and was previously dragged 2 mm down into the floor and stranded mid-brick (stretching the strut above it and crushing the one below). Corners whose *aligned* target simply exceeds the travel clamp still stop short.
- `max_projection_factor=0.5` → per-axis budgets `6/6/2` mm on `12×12×4`; short insides remain; residual snap only finishes **outside** nodes. Gated corners get `stair_step_gate_factor` (default `1.0`, one full cell) instead, since an axis-aligned normal ray cannot cross into a neighbouring node's territory.
- Gate cost on coarse anisotropic cells: on `12×12×4` it clears all 12 stranded corners but takes collapsed hexes 1 → 3 (`min_vol` 0.093 → 0.036). On `6×6×4` it is free (3 → 1 stranded, quality unchanged). Neither size inverts a hex.
- Dual (if used) inherits morphed chords.

---

## Code ownership

| Path | Purpose |
|---|---|
| `graphite/explicit/compare_v1_closest/` | Thin wrapper API `generate_compare_v1(...)` calling existing generator/core |
| `scripts/compare_sc_v1_wrist_rest.py` | Wrist-rest export |
| `tests/compare_v1/` | Smoke: runs without error; report keys present |

**Do not** change V2/V3 packages. **Do not** change default `projection_mode` globally unless already `"closest"`.

---

## Deliverables

- `test_parts/mouse_wrist_rest_grid/compare_v1_closest_grid_12x12x4_untrimmed.stl`
- Matching `_render.png` and `_report.txt`
- Wrapper importable: `from graphite.explicit.compare_v1_closest import generate_compare_v1`

---

## Acceptance

- Pipeline uses `projection_mode="closest"` end-to-end.
- Reproducible script with fixed cell/strut constants matching the three-way index.
- Report notes stair/floor limitation in one sentence (honesty for compare).
