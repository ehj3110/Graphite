# V2.1 — Evolve Surface-First (Plan)

**Status:** implementation in progress (WP0–WP5).  
**Code:** `graphite/explicit/compare_v2_1_surface_first/` · `scripts/compare_sc_v2_1_wrist_rest.py` · `tests/compare_v2_1/`  
**Parent:** [SC_CONFORMAL_COMPARE_V2_SURFACE_FIRST.md](SC_CONFORMAL_COMPARE_V2_SURFACE_FIRST.md), [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md)

## 1. Goal

Keep V2’s architecture (Cartesian volume, independent dual, no cage morph) and
make the **cut memory + dual + stitch** path reliable enough that stair-step
height changes no longer look like collapsed or double surface struts.

V2.1 is **not** V3 (no morph volume→dual). It is trim + connect.

## 2. What V2 already has

| Stage | Today |
|---|---|
| Volume | VF 50% cull → stamp undeformed topology → **no** `morph_hex_scaffold` |
| Clip | Chord bisect + surface snap + **inset** along hanging stub → cut nodes |
| Dual | Exposed-face **centroids** raycast along exposed normal (`dual.py`) |
| Stitch | Distance ≤ L + approach-angle (inward normal) + orphan cull |
| Known win | Inset + angle-aware dual pick fixed many bottom orphans |
| Known gap | Dual is face-centroid, not “project exterior face corners/edges”; stitch still gate-heavy; clip/stitch diagnostics are report-text only; no stage previews |

## 3. Design principles for V2.1

1. **Volume never morphs to CAD.** Cartesian until clip.
2. **Cuts are first-class.** Every clipped strut leaves a remembered hanging stub + cut node id; nothing is “forgotten” between clip and stitch.
3. **Dual is the skin authority.** Built from the unmorphed complex’s exterior, then placed on CAD by a surface-native rule (test variants).
4. **Stitch starts simple, gates return only if needed.** Default mental model: nearest dual node. Re-enable distance/angle as opt-in or after junk appears.
5. **Stage artifacts.** Export volume-only, dual-only, cut markers, stitch pass/fail so we debug without Blender archaeology.
6. **Own package / stem.** New code under `compare_v2_1_*` or clearly versioned flags inside `compare_v2_surface_first` with `version=v2_1` in reports — do not silently change frozen V2 compare baselines.

## 4. Proposed pipeline (normative)

```text
CAD
 │
 ├─ A. Background SC grid + VF 0.5 cull
 │     stamp rule on undeformed hexes
 │     preview: volume_unclipped.*
 │
 ├─ B. Surface dual from exterior of that same complex
 │     B0. Choose dual recipe (A/B test — see §5)
 │     B1. Place nodes on CAD (surface-native)
 │     B2. Optional rectangular cage loft + Boolean flush
 │     preview: dual_only.*
 │
 ├─ C. Graph-clip volume to CAD (hardened)
 │     keep interior struts
 │     crossing → cut node + hanging stub (stable ids, inset policy)
 │     drop fully outside
 │     preview: volume_clipped + cut_nodes markers
 │
 ├─ D. Stitch cut → dual
 │     v2.1 default: nearest dual (3D), optional valence cap
 │     optional gates: distance ≤ L, approach angle ≤ 60°
 │     orphan cull on hard fail
 │     preview: stitches_pass / stitches_reject colored
 │
 └─ E. Union volume + dual + stitches → export + QA report
```

## 5. Work packages (ordered)

### WP0 — Baseline freeze & naming (½ day)

- Keep current V2 artifacts as regression reference (`compare_v2_surface_first_*`).
- Add `generate_compare_v2_1(...)` (new module or `version="v2.1"` path) and
  `scripts/compare_sc_v2_1_wrist_rest.py`.
- Report always includes: clip counts, cut count, stitch accept/reject reasons,
  dual placement mode, inset factor.

### WP1 — Harden cut memory (highest priority) (1–2 days)

This is the failure you named: trim while still remembering legitimate stubs.

- Audit `clip_volume_graph_to_cad`: every clipped strut must yield exactly one
  cut node linked to its interior parent edge; no silent drop of near-tangent
  crossings.
- Keep inset (current ~0.45 along to-interior) but make it a **named parameter**
  (`cut_inset_factor`) with a short sweep (0.2 / 0.45 / 0.6) on the wrist rest.
- Prefer **not** snapping the cut flush then relying on angle gates; keep cut
  **inside** the solid so “connect to dual” has a clear approach vector.
- Emit `cut_parent_strut`, `cut_interior_node`, `cut_direction` in the report
  (or a sidecar NPZ) for debugging.
- Unit tests: synthetic box — N crossing edges → N cut nodes; all cuts have
  SD &lt; 0 (inside); all cuts retain a hanging strut to an interior node.

**Done when:** on wrist rest 6×6×4 and 12×12×4, `n_cut_nodes` is stable across
reruns and bottom-face cuts exist in non-zero count.

### WP2 — Dual from exterior faces (A/B) (2–3 days)

Today’s dual = face **centroids** along exposed normals. Your preferred idea =
project the **exterior** of the unmorphed grid more faithfully.

Test three dual recipes on the same volume clip (no stitch yet):

| ID | Recipe | Intent |
|---|---|---|
| D0 | Current: face-centroid + normal ray (V2) | Baseline |
| D1 | **Exposed-face corner nodes** projected along face normal (or outward), then dual edges = exposed quad edges | Closest to “project exterior faces” |
| D2 | Exposed-face edge midpoints + corners (denser cage) | If D1 undersamples curves |

For each: dual-only STL, mean/max distance of dual nodes to CAD, visual check
on stair risers (2-story ↔ 3-story columns).

**Pick winner** by: flush to CAD on walls *and* top, no self-crossing cage,
reasonable node count (not 3× denser than volume skin without benefit).

**Done when:** one recipe is default in V2.1; others remain flags
(`dual_mode="face_centroid"|"exposed_corners"|...`).

### WP3 — Simple stitch, then optional gates (1–2 days)

Per discussion: start with **closest dual node** per cut.

1. Nearest dual (KD-tree), always propose a stitch.
2. Optional **valence cap** (reuse V3’s idea lightly): max bonds per dual node
   ≈ local rule degree (grid corner 3 / octahedral 4) to avoid pile-ups.
3. Soft report metrics: length, angle from inward normal — **do not reject** in
   v2.1-default unless `stitch_gates="strict"` (legacy V2 behavior).
4. Orphan cull only for: no dual within `search_radius` (e.g. 2L), or valence
   overflow with no alternate dual.

**Done when:** stitch rate ≫ legacy V2 on bottom/top; wrist-rest 6×6×4 shows
volume connected to dual at height steps without requiring closest-point morph.

### WP4 — Stage diagnostics (parallel, 1 day)

- Export: `*_volume_unclipped.stl`, `*_volume_clipped.stl`, `*_dual.stl`,
  `*_stitches_only.stl`, optional colored GLB (cut=red, stitch accept=green,
  reject=orange).
- Report histograms: cut depth (SDF), stitch length, approach angle.
- One script flag `--stages` on the wrist-rest exporter.

### WP5 — Fair compare refresh (½ day)

Regenerate wrist rest at **6×6×4** and **12×12×4**:

- V2 (frozen) vs V2.1  
- Side-by-side: outside nodes, cut/stitch/orphan counts, collapsed hexes
  (should stay ~0 — volume is unmorphed), visual height-step junctions.

### Out of scope for V2.1

- Closest-point / stair-gate / surface-relax volume morph (V1/V4).
- Morph volume iron → dual (that is V3).
- CAD quad remesh dual (future B1) — only if D0–D2 all fail.
- Streamlit wiring (follow later from SURFACE_FIRST_DUAL_TRIM §4).
- Optimal bipartite matching stitch (future B4) — only if nearest+valence fails.

## 6. Suggested implementation order

```text
WP0 naming → WP1 cut memory → WP2 dual A/B → WP3 simple stitch → WP4 diagnostics → WP5 compare
```

Do **not** change stitch policy before cuts are trustworthy; bad cuts made us
blame the angle gate last time.

## 7. Success criteria

- Volume hexes remain undeformed (no morph collapse at 2↔3 story transitions).
- Every clipped boundary strut has a cut node that survives until stitch.
- Dual lies on CAD without inheriting volume stair chords as the final skin.
- Height-step regions show **one** dual skin + clear stitch legs, not double
  morphed surface struts.
- Report is honest: `version=v2_1`, dual mode, inset, stitch mode, reject counts.
- Existing V2 tests still pass; new tests cover clip cut-count and dual modes.

## 8. Risks

| Risk | Mitigation |
|---|---|
| D1 dual too sparse on curves | Fall back to D0 or D2 |
| Ungated nearest stitch creates resin-trap tangents | Keep strict gates as flag; measure angles in report |
| Inset too deep shortens volume | Sweep inset; floor at min stub length |
| Dual and volume exterior edges overlap visually | Optional drop of volume skin edges that duplicate dual (V3 already has a helper) |

## 9. First concrete commit (when implementing)

1. Scaffold `compare_v2_1` wrapper calling shared clip/dual/stitch with new defaults.
2. Parameterize cut inset; add clip unit test on a box.
3. Add `dual_mode=exposed_corners` experimental path behind a flag.
4. Default stitch = nearest dual + valence soft-cap; `stitch_gates="off"|"strict"`.
