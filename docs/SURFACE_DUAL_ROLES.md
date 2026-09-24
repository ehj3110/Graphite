# Universal dual (layered surface dual roles)

**Product name:** universal dual  
**Code identifier (unchanged):** `layered_surface_dual_roles` in `graphite/explicit/sc_role_surface_dual.py`  
**Status:** octahedral Cartesian dual **matches gold** (reviewed 19 Aug 2026). Other lattices (octet, …) use the same engine without a gold oracle.  
**Gold standard (octahedral only):** [NODAL_CONFORMATION.md](NODAL_CONFORMATION.md) § Gold standard — do not rewrite `sc_trim_shared_edge_engine.py` to “match” this dual.  
**Align plan:** [LAYERED_SURFACE_DUAL_ROLES_ALIGN.md](LAYERED_SURFACE_DUAL_ROLES_ALIGN.md)

The dual engine does **not** branch on lattice name. **Hex identity** (C / E / F / B / K from UVW) is what sits on a support. Support-local C/E/F is only *where* on the rectangle. Presence decides **where a neighbor attaches**; C does **not** turn F off.

---

## Review lock (19 Aug 2026)

What landed, and what you signed off visually:

| Fixture | Regenerator | Output | Result |
|---------|-------------|---------|--------|
| Ø50 × 100 mm octahedral cylinder, 10 mm cells, offset `[0,0,0]` | `scripts/export_layered_dual_review.py` | `outputs/layered_dual_review/cylinder/` | Gold vs universal graphs **match**: 408 nodes / 880 struts / 104 stitches |
| Trim coupons (halves / quarters / eights / mix) | same script | `outputs/layered_dual_review/coupons/` | Dual on volume + CAD |
| Octahedral \| grid crease (F–C) | same script | `outputs/layered_dual_review/blend/` | F–C from left +Z face-center to two grid corners on the shared top crease at x = 4 — **looks right** |
| Wrist rest octahedral 12×12×4 mm, offset `[0,0,0]` | `scripts/export_wrist_rest_gold_vs_universal_dual.py` | `outputs/wrist_rest_gold_vs_universal_dual/` | Gold vs universal graphs **match**: 219 / 449, 0 extra / 0 missing |
| Wrist rest octet 12×12×4 mm | `scripts/export_wrist_rest_octet_box_vf_deformed.py` | `outputs/wrist_rest_octet_box_vf_050_deformed/` | τ=0.50 + extract C–F star; dual **298/590**. **Next:** stair-step dual gaps (user to describe). |

Older coupon STLs under `outputs/trim_transition_coupons/` are pre-align; use `layered_dual_review/coupons/` for the current connector.

**Not in these reviews:** node-min, joint wick / stress deconcentration, hex-cage morph, Streamlit Conformal Dual default.

**How the compare is fair:** one Cartesian volume from the gold trim engine (octahedral) or from `generate_nodal_conformation` (other lattices). Universal dual is `build_role_surface_dual` on that same volume. Overlay `dual_gold.stl` vs `dual_universal.stl` when gold exists.

---

## Supports

A **support** is one rectangle on the skin of occupancy:

- An **outer hex face** with no occupied neighbor (unique owner).
- An **outer hex face shared with a cut neighbor** (stair-step boundary): When `len(owners) > 1`, the engine checks the neighbor's `kept_planes_per_hex` across transverse axes. If the neighbor is trimmed/cut such that it does not completely cover the shared interface, the face is recognized as partially exposed and retained as a support.
  - **Sub-Quad Clipping:** For partially covered supports, the support quad `s.quad` is clipped directly to the exposed sub-rectangle (`_compute_exposed_subquad`).
  - **Seam Preservation:** The original unclipped full-face perimeter edges are stored in `s.seam_edges` so the inter-cell seam stitcher (`_support_candidate_segments`) can continue matching adjacent faces across stair-step boundaries without dropping connections.
  - **Natural Node Binding:** Nodes on the buried half are naturally excluded during `bind_roles` (since they fall outside `s.quad`), eliminating internal buried struts on octet stair-steps without requiring heuristic node/midpoint filters.
  - **Synthesized F Centering:** `synthesize_missing_f` places dual-only F-nodes at `s.quad.mean(axis=0)` (the center of the exposed region).
- A **cut midplane** when that half of the hex is empty (the free face of a half or quarter).

Interior faces where two fully-covering occupied cells meet are not supports. Dual never jumps a gap that does not share a Cartesian hex **edge**.

---

## Promotion vs Addition

1. **Promote (Rule 2: Coplanar)** — copy a **volume** strut onto dual when **both endpoints share the same 2D support face** (`gids_share_support`). This prevents internal 3D diagonal volume struts from being mistakenly promoted into the 2D surface dual in thin regions (such as 1-element thick slabs).
2. **Promote (Rule 2.5: Perpendicular / Shared-Edge & Exterior Traversal)** — copy a **volume** strut onto dual across distinct supports under these exact constraints:
   - **Inter-Cell Seam:** Endpoints sit on adjacent supports of *different* hexes that share an exterior Cartesian edge segment in `edge_map`.
   - **Same-Hex Perpendicular Boundary:** Endpoints sit on *different* supports $S_a$ and $S_b$ of the *same* hex cell:
     - **Perpendicularity Check:** $S_a$ and $S_b$ must be perpendicular (`axis_a != axis_b`). Struts connecting parallel opposite faces (e.g. $+Z$ and $-Z$) across the model height are strictly rejected.
     - **Exterior Face Traversal Check:** Exactly one support must be an exposed cut midplane (e.g. $Z=0.5$), and the other must be an **exposed exterior lateral support** of that hex. If the lateral face is shared with an occupied neighbor in the lattice (an interior interface), it has no support in `supports`, and the strut traversing that interior wall is rejected.
     - **Octahedral Guardrail:** Octahedral same-hex 3D apex chords ($F_{\text{apex}} \to F_{\text{diamond}}$) remain excluded from surface dual promotion.
3. **Add / Stitch** — route dual bars across a **shared hex edge** between two adjacent supports using the role routing table. Occupancy-touch and length caps apply to invented F–E / F–C stitches.

---

## Roles

### Hex UVW (how a node sits in the unit cell)

| Role | UVW | Meaning |
|------|-----|---------|
| **C** | three axes at 0 or 1 | Corner |
| **E** | two axes at 0 or 1, one at 0.5 | Hex **edge midpoint** |
| **F** | one axis at 0 or 1, two at 0.5 | Face center |
| **B** | three axes at 0.5 | Body center |
| **K** | anything else on a face | Kelvin-type leftover |

**B** is not dual while it is inside the cell. If a cut exposes it, it is **not** a fourth graph type: it plays the support role below. Star/body cuts are deferred.

### Support-local (how a node sits on *this* rectangle)

| Role | Where on the rectangle |
|------|------------------------|
| **C** | A corner |
| **E** | Midpoint of an edge |
| **F** | Center of the rectangle |

Hex labels and support labels are not always the same point. Examples:

- Octahedral **cut** diamond: hex **F** (side-face centers) sit at the **E** of the cut rectangle.
- Octahedral **outer** face: hex **F** is the **F** of that rectangle.
- Exposed **B** on a **half** cut: center of the cut → support **F**.
- Exposed **B** on a **quarter**: midpoint of the two-cut crease → support **E**.
- Exposed **B** on an **eighth**: remaining octant corner → support **C**.

---

## Presence (per support)

Whatever hex roles are bound on the rectangle stay there. Routing uses those nodes as destinations; it does not drop F because C or E is also present.

Invent C–C around the rectangle / adjacent E–E only if those hex roles exist and volume did not already draw the bar.

Octet full face has C **and** F → both eligible.  
Octahedral outer face has only F → F.  
Octahedral cut diamond is hex **F** sitting at support-E locations (not hex E).

---

## Two verbs

1. **Promote** — copy a **volume** strut onto dual when **both ends lie on some support**. Occupancy-skin midpoint is not required (full-cell octahedral 12-bar; half-cell apex spokes).
2. **Add** — invent a bar only along a **shared hex edge** between two supports, using the routing below. Never because “both nodes are on the surface.” Occupancy-touch and length caps apply to **invented F–E / F–C** only, not to promote or F–F.

---

## Routing on a shared edge

Applies to two supports that share a Cartesian hex edge (**same hex or neighbors**). Connect through the **shared edge**, not face-center to face-center of a C face.

| Pair | Dual bars |
|------|-----------|
| **C next to C** | C–C at the two corners of the shared edge (usually already welded). |
| **E next to E** | E–E at the midpoint of the shared edge if both have an E there; otherwise adjacent-edge E already handled on-support. |
| **F next to E** | F draws to the **midpoint of the shared edge** (the E). One bar, no dangling end. |
| **F next to C** | F draws to the **two corners** of the shared edge (the C nodes), **not** to the C-face center. |
| **F next to F** | **Different hexes only:** old 1-node F–F stitch (face center stands for the face). **Same hex:** do not add (that chord is through the cell). |

Do not jump a cut that does not share an edge.

---

## What each lattice is, in this language

| Volume lattice | Nodes on a typical outer face | Dual |
|----------------|------------------------------|------|
| grid, tesseract outer | C | C–C perimeter |
| octahedral, hex_face_dual | F | Promote F–F + shared-edge F–F stitch; cuts stay hex F |
| octet | C and F | Extract volume C–F star onto dual. Do **not** invent C–C or F–E on a face that already has that star. Stairs / hanging F still add. |
| star | C (and B inside) | C; exposed B deferred |
| kelvin | K | K last |

---

## Coupons

Current octahedral fill: `outputs/layered_dual_review/coupons/` (`scripts/export_layered_dual_review.py`). Overlay `*_octahedral_dual.stl` on `*_octahedral_volume.stl` and `*_cad.stl`. Older `outputs/trim_transition_coupons/` is pre-align.
