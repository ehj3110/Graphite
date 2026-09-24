# Handoff: Nodal Conformation + universal dual (19 Aug 2026)

> **Doc role:** Active handoff. Agents: [AGENTS.md](../AGENTS.md) → [graphite/explicit/README.md](../graphite/explicit/README.md), then this file for dual/wrist-rest work only.

**For a new agent.** Read this first, then [NODAL_CONFORMATION.md](NODAL_CONFORMATION.md) and [SURFACE_DUAL_ROLES.md](SURFACE_DUAL_ROLES.md). Do **one phase**, then stop with a reviewable STL/PNG under `outputs/`.

**Repo:** Graphite (`graphite/`, `scripts/`, `docs/`, `outputs/`).  
**CAD (read only):** `test_parts/Mouse wrist rest v1.stl` — never write generated meshes into `test_parts/`.  
**Generated review files:** `outputs/` only.

---

## Where we stopped (do this next)

The user **signed off τ = 0.50** octant occupancy for octet trim, then asked to **extract** the volume C–F star onto dual (not leave dual empty on full faces). That extraction is **in**. Last review pack:

`outputs/wrist_rest_octet_box_vf_050_deformed/`

| File | Role |
|------|------|
| `volume_deformed.stl` | Octet volume, τ=0.50, loose-external prune, **232** outside nodes snapped to CAD |
| `dual_cartesian.stl` | Universal dual, undeformed (overlay on Cartesian volume) |
| `dual_deformed.stl` | **Final surface dual** after closest-project of dual nodes onto CAD |
| `report.json` | Counts |

**Last dual counts (after extract, no extra F–E/C–C on complete faces):** **298 nodes / 590 struts**.

**Stair-Step Gap Resolution, 2x2 Quadrant Partition & Rule 2.5 (Landed 26 Aug 2026):** 
1. **Neighbor Coverage & 2x2 Discrete Quadrant Partitioning:** `collect_role_supports` in `sc_role_surface_dual.py` decomposes partially covered shared faces into discrete $2 \times 2$ half-cell quadrants. Buried quadrants are excluded while exposed quadrants are emitted as maximal non-overlapping rectangles (`_compute_exposed_subquads`), eliminating non-convex L-shape over-extension.
2. **Seam Edge Preservation:** `RoleSupport.seam_edges` preserves the unclipped full-face perimeter edges so that inter-cell cross-support stitching (`_support_candidate_segments`) pairs shared seams across cell boundaries without dropped connections.
3. **Volume Promotion (Rule 2 & Rule 2.5):**
   - **Rule 2 (Coplanar):** Struts lying on the same 2D support face are promoted.
   - **Rule 2.5 (Perpendicular & Exterior Traversal):** Promotes volume struts connecting perpendicular supports of the same hex *only* when one is an exposed cut midplane and the other is an exposed lateral exterior face, and the remaining third-axis boundary plane is not shared with an occupied neighbor cell. Struts spanning parallel opposite faces or traversing faces shared with other lattice cells are strictly rejected.
   - **Octahedral Guardrail:** Internal 3D apex chords ($F_{\text{apex}} \to F_{\text{diamond}}$) across the cell height are excluded from dual promotion.
4. **CSG Boolean Meshing Fix:** Replaced fast disjoint `compose` with true exact `Manifold.batch_boolean(parts, OpType.Add)` in `geometry_module.py` (`_union_manifolds_for_joint`), eliminating coincident overlapping triangle facets and planar meshing artifacts.
5. **Locked Reference 8-Model Bookend Review Suite (`outputs/bookend_review/`):**

| Part Variant | Lattice Rule | Volume (Nodes / Struts) | Dual (Nodes / Struts) | Combined Struts | Health & Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sloped** | **Octet** | 276 / 1141 | 214 / **491** | **1632** | Clean C–F star extraction; 0 buried stair-step struts |
| **Angled** | **Octet** | 354 / 1575 | 230 / **560** | **2135** | Complete, unbroken surface skin network |
| **Sloped** | **Star** | 142 / 310 | 107 / **257** | **567** | Exterior step anchors intact; 0 internal crossing chords |
| **Angled** | **Star** | 185 / 450 | 115 / **295** | **745** | Boundary edge anchors fully preserved |
| **Sloped** | **Octahedral** | 204 / 575 | 162 / **357** | **932** | Exact gold standard; clean diamond cycle skin |
| **Angled** | **Octahedral** | 264 / 805 | 180 / **401** | **1206** | Complete watertight boundary skin |
| **Sloped** | **Cross** | 266 / 670 | 204 / **463** | **1133** | Rogue quadrant strut eliminated; clean stair-step |
| **Angled** | **Cross** | 354 / 965 | 230 / **525** | **1490** | Clean watertight boundary skin |

*Review pack:* `outputs/bookend_review/`. All 42 unit tests pass.

---

## Locked philosophy

1. **Cartesian volume lattice.** Conformity is cull/trim + **surface dual** (Nodal Conformation), not hex-cage morph.
2. **Gold octahedral dual** is the oracle. Do **not** rewrite `sc_trim_shared_edge_engine.py` to match experimental dual.
3. **Universal dual** is the product name for layered surface dual roles (one connector for all SC stamps). Code `dual_rule` is still `"layered_surface_dual_roles"`.
4. **Streamlit default** is still Conformal Dual (hex morph path). Do not change the production UI default unless asked.
5. **Out of scope unless asked:** node-min, joint wick / stress deconcentration, print Boolean, rewriting gold.

Process: `.cursor/rules/one-phase-review.mdc` — one phase, named `outputs/` path, what “looks right” means, then stop.

---

## Gold standard (octahedral only)

Locked 19 Aug 2026 after Ø50 × 100 mm cylinder looked right.

| | |
|--|--|
| Engine | `generate_octahedral_trim_shared_edge` in `graphite/explicit/sc_trim_shared_edge_engine.py` |
| Volume | VF cull, 25/75 node-plane trim, **any-sample** box occupancy, ghost-hex CAD occupancy. **No** loose-external prune. **No** octant VF τ. |
| Dual | Shared-edge F: promote skin volume bars + inter-cell F–F stitches. Jump-cut is not dual. |
| Fixture | `outputs/sc_cylinder_50x100_octahedral_temp_engine/` — 10 mm cells, offset `[0,0,0]`, Cartesian dual **408 / 880 / 104 stitches** |
| Regen | `python scripts/export_sc_cylinder_temp_engine.py` |

**User signed off:** universal dual **graph-matches** gold on that cylinder and on the octahedral wrist rest (12×12×4, offset `[0,0,0]`): **219 / 449**, 0 extra / 0 missing. Blend F–C (octahedral \| grid) **looks fantastic**.

Tests: `tests/test_gold_octahedral_dual.py`, `tests/test_layered_dual_cylinder_oracle.py` (`slow`), `scripts/compare_cylinder_layered_vs_gold.py`.

If a change would alter gold graphs, keep gold flags off (`box_vf_min=0`, `prune_loose_external=False` on `generate_node_plane_trimmed_lattice` as called from the gold engine).

---

## Pipeline (nodal conformation)

```text
CAD
  → SC background hexes (offset; node-min optional)
  → Cull empty hexes
  → Stamp topology (undeformed)
  → Node-plane trim (25/75) + box occupancy
  → Optional octant VF threshold (box_vf_min)
  → Loose-external prune (nodal default ON; gold OFF)
  → Universal dual on that volume graph
  → Optional: snap leftover outside volume nodes; project dual nodes onto CAD
```

API: `generate_nodal_conformation` in `graphite/explicit/nodal_conformation.py`.

| Flag | Gold engine | Current octet review |
|------|-------------|----------------------|
| `origin_offset` | `[0,0,0]` on gold cylinder; frozen octahedral wrist-rest table is `[0,0,-1]` but gold-vs-universal wrist rest used **`[0,0,0]`** | `[0,0,0]` |
| `run_node_minimization` | n/a | **False** |
| `prune_loose_external` | False | **True** (almost no effect after τ=0.50: 0–1 nodes) |
| `box_vf_min` | **0** (any CAD sample in an octant counts) | **0.50** (user OK) |
| `project_dual` | dual on CAD for print fixtures | **True** on deformed pack |

Cell for wrist rest: **12 × 12 × 4 mm**. Rule for current review: **`octet`**.

---

## Trim (why octet was busy)

**Culling** = drop whole hexes. **Trimming** = drop sub-cell nodes.

25/75 is **per-axis AABB**. A diagonal clip can still span ~0–1 on X and Y → cell looks full → busy octet stamp.

**Box occupancy (any-sample):** keep a node if *any* incident ½-cell octant contains CAD. A sliver nick keeps hanging face-centers. This is still gold.

**Octant VF τ:** same eight octants, but occupied only if **sampled VF ≥ τ**. τ=0.25 then **0.50**. User is **okay with 0.50**. A ~50% diagonal stairs to a **half** (only ≥50% octants keep nodes), not a true 45° clip.

**Loose-external prune:** drop outside nodes whose remaining neighbors are all outside. Keep an outside node with an **inside** neighbor. On this wrist rest after occupancy it barely fires. Helper: `prune_loose_external_nodes` in `sc_axis_cull_octahedral.py`; wired post-weld in `generate_node_plane_trimmed_lattice`.

Do **not** clip struts at the CAD surface (that is a different engine).

---

## Universal dual (current octet rules)

Code: `graphite/explicit/sc_role_surface_dual.py` — `build_layered_surface_dual` / `build_role_surface_dual`.

**Supports:** outer unique-owner faces + cut midplanes. Identity is **hex UVW** (C/E/F), not support-local. Presence, not exclusion (C does not turn F off).

**Skin-complete support:** outer (not midplane) with C **and** F and at least one **volume** F–C spoke (`_support_has_volume_cross`).

| On a skin-complete octet face | Do |
|------------------------------|----|
| Volume C–F star | **Extract / promote** onto dual (same-support volume bars). This **is** the skin. |
| Invented C–C ring | **Skip** |
| Invented F–E (or extra F–C) from that F | **Skip** |
| Volume F–F that only joins two complete outer faces | **Skip** (not the face star) |

Midplane / hanging F (no volume spokes): still **add** F–E / F–C / stitches. Occupancy-touch + length cap only on **invented** F–E / F–C, not on promote or F–F.

Octahedral faces are F-only → not skin-complete → gold promote + F–F stitch unchanged.

Isolated full octet cube test: **24** dual struts (6×4 spokes), **14** nodes. `tests/test_sc_role_surface_dual.py::test_octet_full_extracts_fc_star_no_invented_ring`.

`dual_deformed.stl` **is** the surface dual after projection. Volume still contains interior bars; dual should be the complete **skin** (extracted stars + stair/hanging adds).

---

## Review fixture trail (octet wrist rest)

All 12×12×4, offset `[0,0,0]`, CAD `Mouse wrist rest v1.stl`. Overlay on `outputs/wrist_rest_octet_untrimmed/hex_cages.stl` to see unit cells.

| Folder | What |
|--------|------|
| `outputs/wrist_rest_octet_untrimmed/` | Full AABB stamps, **no cull/trim/dual**. 1080 hexes. Cages + volume. |
| `outputs/wrist_rest_octet_universal_dual/` | Trim only (any-sample occupancy), Cartesian dual. Volume 584/2701, dual 339/1186 |
| `outputs/wrist_rest_octet_universal_dual_loose_prune/` | + prune. Volume 583/2698, dual 338/1180 |
| `outputs/wrist_rest_octet_box_vf_025/` | τ=0.25. Volume 565/2619, dual 320/1070 |
| `outputs/wrist_rest_octet_box_vf_050/` | τ=0.50 Cartesian. Volume **543/2486**, dual at the time was pre-sparse-extract |
| `outputs/wrist_rest_octet_box_vf_050_deformed/` | **Current.** Same volume graph; dual after extract rule **298/590** |

Other signed-off packs:

| Folder | What |
|--------|------|
| `outputs/layered_dual_review/` | Cylinder gold vs universal + coupons + blend |
| `outputs/wrist_rest_gold_vs_universal_dual/` | Octahedral wrist rest gold vs universal, match |

Regen current octet deformed pack:

```text
python scripts/export_wrist_rest_octet_box_vf_deformed.py
```

Other useful scripts: `export_layered_dual_review.py`, `export_wrist_rest_gold_vs_universal_dual.py`, `export_wrist_rest_octet_untrimmed.py`, `export_wrist_rest_octet_box_vf.py` (currently set to τ=0.50 output folder).

Preview struts are **0.15 mm** cylinders (review, not print).

---

## Tests to run after dual/trim edits

```text
python -m pytest tests/test_sc_role_surface_dual.py tests/test_gold_octahedral_dual.py tests/test_sc_node_plane_trim.py -q
```

Slow cylinder oracle (optional): `python -m pytest tests/test_layered_dual_cylinder_oracle.py -m slow`.

If gold graphs move, you broke the oracle.

---

## Code map

| Piece | Path |
|-------|------|
| Nodal generate | `graphite/explicit/nodal_conformation.py` |
| Trim + octant VF + prune | `graphite/explicit/sc_node_plane_trim.py` (`box_vf_min`, `prune_loose_external`) |
| Gold octahedral | `graphite/explicit/sc_trim_shared_edge_engine.py` |
| Universal dual | `graphite/explicit/sc_role_surface_dual.py` |
| Octet stamp | `apply_hex_octet_truss` in `hex_rules.py` (C–F per face + adjacent F–F) |

---

## Do not

- Write into `test_parts/` (except reading CAD).
- Implement a multi-phase plan in one shot.
- Change gold trim/dual to “match” octet experiments.
- Turn Streamlit off Conformal Dual unless asked.
- Start stair-step dual fixes without a visual/description from the user.

---

## User language (keep it)

- **Universal dual** = layered surface dual roles.
- **Extract** = promote volume skin bars onto dual.
- **Add** = invent bars that were never volume (C–C ring, F–E cross on a face that already has an F–C star).
- They were **wrong** when they said full faces need no dual; they meant **do not add F–E** on top of an existing F–C star. Extract the star.
- Blend coupon looked **fantastic**. Octahedral gold vs universal on wrist rest looked **perfect**.

---

## Locked Reference 8-Model Bookend Review Suite

Hardened Universal Engine (active `face_owners` + 3rd-axis exterior boundary invariant, zero rule-name branching):

| Part Variant | Lattice Rule | Volume (Nodes / Struts) | Dual (Nodes / Struts) | Combined Welded Struts |
| :--- | :--- | :--- | :--- | :--- |
| **Sloped** | **Octet** | 276 / 1141 | 214 / 509 | **1210** |
| **Angled** | **Octet** | 354 / 1575 | 230 / 560 | **1656** |
| **Sloped** | **Octahedral** | 204 / 575 | 162 / 357 | **666** |
| **Angled** | **Octahedral** | 264 / 805 | 180 / 401 | **897** |
| **Sloped** | **Cross** | 266 / 670 | 204 / 463 | **898** |
| **Angled** | **Cross** | 354 / 965 | 230 / 525 | **1241** |
| **Sloped** | **Star** | 142 / 310 | 107 / 203 | **513** |
| **Angled** | **Star** | 185 / 450 | 115 / 237 | **687** |

*Verified: `pytest tests/test_sc_role_surface_dual.py tests/test_gold_octahedral_dual.py tests/test_sc_node_plane_trim.py` (42 passed).*

---

## Planar Slicing Contour Sweep (Production Default as of September 2026)

The Planar Slicing Contour Sweep (`graphite.explicit.planar_surface_sweep`) is the canonical surface dual cleanup and solidification method for all SC hexahedral meshes. It directly replaces box-chord extrusions and cylindrical duals.

### Architecture & Math Summary
1. **Slicing Plane $\Pi$:** Normal $\mathbf{w} = \text{normalize}(\mathbf{t}_{\text{chord}} \times \mathbf{n}_{\text{mid}})$ stays constant along the entire chord arc, ensuring parallel flat sidewalls with zero twisting/torsion artifacts.
2. **CAD Surface Arc Projection:** $N=8$ sample points along chord are projected to closest CAD surface points $\mathbf{q}_k$, then projected into plane $\Pi$:
   $$\mathbf{c}_k = \mathbf{q}_k - ((\mathbf{q}_k - p_0) \cdot \mathbf{w})\mathbf{w}$$
3. **Continuous Quad-Lofted Rings:** 4 profile vertices per station oriented with in-plane normal $\mathbf{d}_k = \mathbf{t}_k \times \mathbf{w}$ pointing outward. Endpoints extended along tangent by $0.4 \times W$ for seamless joint overlap.
4. **Golden Depth Parameters:**
   - Width $W = 1.6\text{ mm}$
   - Finished thickness $T_{\text{dual}} = 0.8\text{ mm}$
   - Inward depth $T_{\text{in}} = 1.5 \times T_{\text{dual}} = 1.2\text{ mm}$ (golden ratio: bridges chord sagitta without inner corner shark fins)
   - Outward margin $M_{\text{out}} = 0.4\text{ mm}$ (guarantees zero-gap CAD boolean)
5. **Topological Refinements:**
   - **Boundary volume strut promotion:** Promotes volume struts where both nodes lie on CAD boundary ($d < 0.1\text{ mm}$) to seal rim and corner gaps (68 struts on foam squeezer).
   - **Empty valley pruning:** Prunes non-volume dual struts where midpoint $\text{SDF} > 0.5\text{ mm}$ to eliminate floating chords across concave valleys (24 struts pruned on foam squeezer).
6. **Double Boolean Shell:**
   $$\mathcal{S}_{\text{final}} = (\mathcal{S}_{\text{raw}} \cap \Omega_{\text{CAD}}) \setminus \Omega_{\text{CAD-inner}}$$
   where $\Omega_{\text{CAD-inner}} = V_{\text{CAD}} - T_{\text{dual}}\mathbf{n}_{\text{vertex}}$.

See [PLANAR_SLICING_SURFACE_DUAL_RETROSPECTIVE.md](PLANAR_SLICING_SURFACE_DUAL_RETROSPECTIVE.md) for full engineering retrospective and benchmarks.

