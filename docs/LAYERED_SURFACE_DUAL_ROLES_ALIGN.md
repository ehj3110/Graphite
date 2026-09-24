# Layered surface dual roles — align with gold (phased)

**Status:** implemented (Aug 2026) — octahedral Cartesian dual matches gold; **visually signed off** 19 Aug 2026 (cylinder, blend F–C, wrist rest).  
**Product name:** universal dual. **Code name:** layered surface dual roles (`dual_rule` unchanged).  
**Gold oracle:** `generate_octahedral_trim_shared_edge` + cylinder `outputs/sc_cylinder_50x100_octahedral_temp_engine/`  
**Spec:** [SURFACE_DUAL_ROLES.md](SURFACE_DUAL_ROLES.md)  
**Code:** `graphite/explicit/sc_role_surface_dual.py`  
**Do not touch:** `sc_trim_shared_edge_engine.py` (oracle). Node-min and wick stay out.

Goal: octahedral dual **graphs match gold**. F–E and F–C stay in the table for blends. Rules change; routes are not deleted.

Leave `generate_nodal_conformation` on the old connector until **Phase 6** (cylinder) passes. Each phase is one PR-sized change plus tests; stop and review before the next.

---

## Locked rules (all later phases assume these)

1. **Hex identity.** A node’s role is `classify_uvw` (C / E / F / B / K). Support-local C/E/F is only *where* it sits on the rectangle (corner / edge mid / face center). Hex F on a cut stays **F** even if it sits at a support-E.
2. **Presence, not exclusion.** Whatever hex roles are on a support stay on it. C does not turn F off. Presence decides **where a neighbor attaches**, not which nodes we drop.
3. **Promote.** A volume bar is dual iff **both ends lie on some support**. No occupancy-skin midpoint test for promote.
4. **Add (keep all rows).** On a shared Cartesian edge of two supports:

   | A has (hex) | B has (hex) | Dual |
   |-------------|-------------|------|
   | F | F | `route_geometric_face_stitch` (1-node FC may stand for the face; multi-node cut does not pair far-side nodes) |
   | F | E | F → E at shared-edge midpoint |
   | F | C | F → two corners of the shared edge |
   | C | C | C–C at the two corners (usually already welded) |
   | E | E | E–E at the mid if both have it |

   Several rows may fire if several roles are present. Octahedral–octahedral only has F, so only F–F plus promote.
5. **Occupancy.** Occupancy still decides **which hex emits a support** (ghost hex / unique-owner / mid-plane). Occupancy midpoint gate applies to **invented** F–E / F–C only, not to promote or F–F.
6. **Invent on-support C–C / adjacent E–E** only if that hex role is present **and** volume did not already draw the bar. Octahedral F-only: no invention.

---

## Current tests that encode the *old* rules

These must **flip** when the matching phase lands (not be preserved as regressions):

| Test today | Encodes | After |
|------------|---------|--------|
| `test_isolated_full_octahedral_volume_struts_not_on_skin` | isolated full cube dual empty (`require_skin`) | **12** dual bars, 0 stitches |
| `test_half_neg_z_octahedral_diamond_only` | 4 bars, apex **not** dual | diamond **+** apex spokes; apex **is** a dual node |
| `test_isolated_half_z_native_diamond_apex_not_dual` | same | apex in dual |
| `test_octet_full_is_c_cage_f_off` | C excludes F | C **and** F may both be destinations (Phase 8) |

Keep ghost-hex / unique-owner support tests; they are trim, not connector.

---

## Phase 0 — Oracle helper (no dual-rule change)

**Why.** Every later phase needs a gold graph without regenerating the 50×100 solid.

**Do.**

- Add `tests/helpers/gold_octahedral_dual.py` (or a test-only function in `test_sc_role_surface_dual.py`) that calls `generate_node_plane_trimmed_lattice` + `build_octahedral_shared_edge_dual` (same as the temp engine’s dual, not the full generate wrapper).
- Helper: `assert_dual_graphs_equal(a, b)` — weld by rounded XYZ, compare undirected edges.

**Test.** Isolated 10 mm cube: gold dual has 12 struts. Two welded 10 mm cubes: stitches > 0. Do not call `build_role_surface_dual` yet.

**Done when.** Helper is the only compare used in later octahedral tests.

---

## Phase 1 — Hex identity vs support-local

**Why.** If a cut diamond is winning-layer E, F–E fires on octahedral and gold is lost. F–E must wait for a real hex E.

**Do.**

- Keep `classify_uvw` as the identity stored on the support (`gids_by_role` keyed by **hex** role).
- Keep `classify_support_local` only as a geometric hint (nearest corner / edge mid / face center) for F–E / F–C targeting.
- `_support_local_buckets` / `_winning_layer`: bucket by **hex** role, not support-local E for octahedral FCs.

**Test.**

- Isolated half-Z octahedral: four equatorial nodes classify as hex **F**, not E. Winning presence on the cut support is `{F}`.
- Isolated full: six outer supports, each one hex F.
- Grid stamp unchanged: hex C on outer faces.
- Existing `test_classify_uvw_edge_midpoint_is_e` still true for a **true** edge-midpoint UVW.

**Done when.** No octahedral cut support reports hex E unless the stamp actually has E (octet/star). F–E code paths exist but do not run on these fixtures.

**Review.** If coupon overlays look wrong, stop; do not “fix” by relabeling F→E.

---

## Phase 2 — Presence, not exclusion

**Why.** `_winning_layer` returning `"CE"` and ignoring F is exclusion. Destination set = all hex roles present on that support.

**Do.**

- Replace winner-take-all `_winning_layer` with `present_roles(support) -> set`.
- Shared-edge loop: for each pair of supports, fire every routing row whose both sides have that hex role. Do not skip F because C or E is also present.
- Do **not** yet change promote or occupancy (still old). Octahedral fixtures only have F, so dual counts may still be wrong until Phase 3.

**Test.**

- Octahedral half/full: present roles `{F}` only (sanity).
- Synthetic support with C **and** F bound (can inject gids): `present_roles == {C, F}`; routing would consider both F–F and F–C if a neighbor F exists. Unit-test the table, not a full octet CAD yet.

**Done when.** No code path sets `layer == "CE"` then `_layer_actors(..., F=[])`.

---

## Phase 3 — Promote = both ends on supports

**Why.** Gold dual includes every volume bar with both ends on the skin graph (full-cell 12-bar octahedron; half-cell apex spokes). `require_skin` is the mismatch.

**Do.**

- Promote: both endpoints in the union of support gids. Drop `require_skin` for this verb.
- Same-hex opposite-face F–F: still only if both F are on supports (isolated full: yes, all 12). Interior weld FCs on a shared two-owner face: not on a support → those bars stay volume (gold).

**Test (must match gold helper).**

- Isolated full octahedral: **12** dual struts, **0** stitches. Flip `test_isolated_full_octahedral_volume_struts_not_on_skin`.
- Isolated half-Z: dual includes diamond **and** spokes to the kept outer apex. Flip apex-not-dual tests.
- Isolated quarter: still only bars gold would promote (no opposite-face jump). Compare to gold helper on the same `kept_planes`.
- Two full cells: shared-face FCs not dual; outer F–F + stitches match gold helper.

**Done when.** Those four graphs equal the oracle helper. F–E / F–C still unused.

---

## Phase 4 — Occupancy and length gates

**Why.** Role dual skips F–F when the stitch midpoint fails `point_touches_occupancy` on both hexes; gold does not. `max_edge_length` can drop a legal F–F.

**Do.**

- Promote + F–F: no occupancy-midpoint skip; no `max_edge_length` kill (or cap ≥ hex face diagonal).
- Invented F–E / F–C: **keep** occupancy-touch and a length cap so blend bars cannot jump an empty half.

**Test.**

- Re-run Phase 3 gold compares (must still pass).
- `test_quarter_plus_neighbor_does_not_bridge_empty_half`: still no **jump-cut** (no shared edge). If it only passed via occupancy-on-F–F, rewrite so it asserts “no shared Cartesian edge → no add”.
- Optional: two hexes that share an edge whose midpoint is numerically awkward — F–F must still exist if gold has it.

**Done when.** Gold helper still matches; jump-cut still forbidden.

---

## Phase 5 — Bind like gold; invent only if needed

**Why.** Gold binds `point_on_quad` on unique-owner / mid-plane quads. Role dual drops neighbors that fail occupancy-at-node. On-support C–C / E–E invention must not run on F-only supports.

**Do.**

- Bind: volume nodes on the quad (gold). Hex occupancy still used only in `collect_role_supports` / `hex_occupies_cad`.
- Invention: C–C perimeter and adjacent E–E iff hex C/E present and bar not already in volume.

**Test.**

- Octahedral gold helper still matches (bind change must not add junk nodes on isolated cubes).
- Grid isolated hex: 12 C–C edges (already `test_isolated_grid_hex_twelve_edges`).
- Two grid cells: interior face not dual (`test_two_grid_cells_no_interior_face_dual_no_diagonals`).

**Done when.** Octahedral graphs still equal gold; grid C-perimeter still works.

---

## Phase 6 — Cylinder oracle

**Why.** Visual gold. Small tests cannot catch wall-stair skip counts.

**Do.**

- Script or pytest (slow/opt-in): same CAD as `export_sc_cylinder_temp_engine.py` (Ø50 × 100, cell 10, offset 0). Compare **Cartesian** dual node count, strut count, `n_stitches_added` / equivalent to `report.json`: **408 / 880 / 104**.
- Prefer exact undirected graph match after coordinate weld (decimals=6). If counts match but graphs differ, dump a small mismatch report (extra vs missing edges) — do not “fix” with F–E.

**Test.** `pytest tests/test_layered_dual_cylinder_oracle.py` marked `slow`, or a script `scripts/compare_cylinder_layered_vs_gold.py` that exits 0 on match.

**Done when.** Exact or documented 1-edge tolerance. Human glance at `dual_graph_cartesian` overlay optional.

**Then** point `generate_nodal_conformation` octahedral/hex_face_dual at layered dual (still no node-min/wick by default). Temp engine remains the oracle.

---

## Phase 7 — Blend: F–C / F–E still exist

**Why.** Prove we did not delete versatility. Not a gold compare.

**Do.** One packed coupon (or two hexes): octahedral full cell | grid (or octet) neighbor sharing a wall. Shared face is interior (not a support). Shared **crease** between an octahedral outer F support and a grid C support.

**Test.**

- At least one F–C bar (F → two corners of the shared edge), and/or F–E if the C-lattice puts E on that edge.
- Octahedral-only gold tests from Phase 3 **unchanged**.

**Done when.** Blend test green without regressing cylinder/oracle.

---

## Phase 8 — Octet / star / Kelvin (do not block gold)

**Why.** Presence-not-exclusion changes octet “F off”. That is expected, not gold.

**Do.** Update `test_octet_full_is_c_cage_f_off` to: C cage **plus** F still eligible as destinations. Star B-on-cut still deferred (explicit test). Kelvin still K helper until a K row is defined.

**Test.** Existing grid/tesseract/kelvin tests adjusted only if they depended on exclusion. No octahedral gold change.

**Done when.** Docs in SURFACE_DUAL_ROLES.md match: C-turns-F-off removed; blend table as above.

---

## Suggested order and stop points

```text
0 oracle helper
1 hex identity          → review: cut is F
2 presence not exclude  → review: no CE-kills-F
3 promote               → review: 12-bar cube + apex half  ★ gold graphs
4 occupancy/length      → review: jump-cut still dead
5 bind / invent         → review: grid still 12
6 cylinder              → review: 408/880/104            ★ visual gold
7 blend F-C             → review: versatility
8 octet/star docs+tests
```

Stars are the two “does this still look like the cylinder” gates. If Phase 3 or 6 fails, do not proceed; do not reintroduce F→E relabel or `require_skin` to paper over it.

---

## Out of scope (later)

- Node minimization, joint wick, closest-point (compare Cartesian graphs first).
- Synthesize missing F; exposed B as a fourth graph.
- Changing the temp engine.
- Streamlit / Conformal Dual production default.
