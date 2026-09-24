# Quantized Overbuild + Extent Trim + Geometric Surface Dual

**Status:** see **[NODAL_CONFORMATION.md](NODAL_CONFORMATION.md)** (current cull/trim/box-occupancy/specials). This file is the octahedral lab trail.  
**Stress part:** `test_parts/Mouse wrist rest v1.stl`  
**Cell:** \((12, 12, 4)\,\mathrm{mm}\) SC (octahedral was the first proven rule; multi-rule matrix uses the same path)  
**Origin offset:** phase-searched (node minimization)  
**API:** `graphite/explicit/nodal_conformation.py` · matrix: `scripts/export_wrist_rest_nodal_conformation_matrix.py` → `test_parts/mouse_wrist_rest_nodal_conformation/`  
**Related:** [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md), [SC_CONFORMAL_CROP_THEN_DUAL_RESTART.md](SC_CONFORMAL_CROP_THEN_DUAL_RESTART.md)

This document records the **Nodal Conformation** workstream (quantized overbuild + extent/VF trim + geometric or rule skin dual): do **not** morph the volume hex cage into the CAD. That cage-morph path is separate (“conformal morph”) and was **not** the intended multi-lattice wrist-rest compare.

---

## 1. What we are doing now

```text
CAD Ω
  │
  ├─ SC background hex grid, phase-shifted by node minimization
  ├─ VF > 0.01 candidate cells
  ├─ Per-axis material AABB in unit-cell coords
  │     < 0.25  throw (cull that outer face-center)
  │     0.25–0.75  half (cull outer; remaining nodes live on the mid-plane)
  │     > 0.75  full (keep outer face-center)
  ├─ Stamp strict octahedron (6 face-centers + 12 edges). Weld by coordinate.
  │     → CORE lattice (undeformed)
  │
  └─ Geometric surface dual
        Surface node = any core node that lies on an exposed SC face quad
          (outer Cartesian rectangle if that direction was kept,
           or the parallel mid-plane cut if the outer was culled)
        Dual struts = core edges with both ends on the skin
                     + inter-cell stitches across shared Cartesian edges
        Morph dual nodes only → Euclidean closest point on ∂Ω
```

**Core is not morphed.** Overbuild + trim already puts remaining nodes near the boundary; the dual is a subset of that graph, so closest-point is a short, clean snap rather than a cage-crushing morph.

Wrist-rest snapshot (offset `[0,0,-1]`, tiered 25/75):

| Graph | Nodes | Struts |
|--------|------:|-------:|
| Core octahedral | 450 | 1368 |
| Geometric dual (undeformed) | 283 | 596 (498 extracted + 98 stitches) |
| Dual closest-point | 283 | 596 (same connectivity) |

Mean closest travel **2.26 mm**, max **6.98 mm**. Exposed faces: 86 outer + 154 mid-plane.

---

## 2. Why this path exists

Earlier SC conformal work (V1 closest volume morph, V2/V2.1 clip+dual, TNP / axis-constrained dual projection) asked a **rigid hex cage** to become the CAD skin. On the wrist rest that produced stair crush, floor-snaps, far-side ray jumps, and inverted bricks. Those experiments are documented in [SC_CONFORMAL_CROP_THEN_DUAL_RESTART.md](SC_CONFORMAL_CROP_THEN_DUAL_RESTART.md) and [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md).

The quantized path inverts the authority:

1. **Overbuild** a Cartesian octahedral lattice that is allowed to stick out of the CAD.
2. **Trim topologically** (keep/cull face-centers from material extent), not by Boolean spaghetti or SDF node-tolerance fuzz.
3. **Extract the skin graph** from whatever nodes remain on exposed faces.
4. **Morph only that skin**, with the simplest projector that looks good once the graph is already close.

Volume stays printable and load-bearing; dual owns conformity.

---

## 3. Chronology — what we tried

### 3.1 Quantized Full / Half dictionary (abandoned as the dual authority)

Inspired by Marching Cubes / Dual Contouring: tag each hex Empty / Half / Full from VF, stamp `octahedral_half_{pos,neg}_{x,y,z}` (six polarities) plus full octahedra.

**Fixes that mattered**

- Half cells originally used only three rules, always negative apex → **inverted halves** (empty half kept, solid half deleted). Expanded to six polarities.
- `octahedral_transition` tried to bridge half cells to full-cell **corners**. Octahedral lattices have **no corner nodes**. Deleted the transition pass; coincident face-centers weld.

**What still failed**

- UV / CUT / SIDE dual *roles* were invented on the unwrapped surface net and then routed independently of the trimmed octahedral graph. The dual looked plausible in 2D UV and disconnected in 3D.
- SDF node-tolerance bands reintroduced continuous fuzz (nodes slightly inside/outside) and fought the discrete Full/Half idea.

**Keep:** six-polarity half octahedra as a *volume* trim primitive; drop role-based dual routing.

### 3.2 Independent-axis extent cull (kept)

Replace “is this node inside by SDF?” with **how far does solid penetrate this cell along each axis?**

For each VF-candidate hex, sample CAD inside the brick, form a local AABB in unit-cell coordinates \([0,1]^3\), and decide each of the six face-centers independently.

First rule: **mid-plane 0.5** — keep the outer face-center iff extent \(> 0.5\). That is a binary half/full with no “throw the sliver away.”

Then **tiered 25/75** (current):

| Per-axis extent | Cell treatment on that axis | Outer face-center |
|-----------------|-----------------------------|-------------------|
| \(< 0.25\) | throw | cull |
| \(0.25\)–\(0.75\) | half | cull (skin lives on the mid-plane) |
| \(> 0.75\) | full | keep |

A cell with two axes in the half band is a **quarter**; three axes → **eighth**. Strict octahedron only: 6 face-centers, 12 adjacent-face edges. No center hub, no SDF node-tolerance.

Optional `prune_loose_external` drops external nodes whose neighbors are all external. Not required for the wrist-rest exports.

**This is the trim that worked.** It overbuilds, then cuts with the same topology the dual will read.

### 3.3 Node minimization (kept)

Octahedral face-centers sit at \(0 / 0.5 / 1\) along each axis, so the natural phase increment is **0.25 × cell** = \((3, 3, 1)\,\mathrm{mm}\).

Search origin + six axial shifts. Score = fraction of *trimmed* lattice nodes outside the CAD (lower is better). Tie-breaks: fewer outside nodes, more kept hexes, smaller offset.

Wrist rest: baseline outside fraction **0.517** → best offset **`[0, 0, -1]`** at **0.376**.

`generate_background_grid` now accepts `origin_offset`. All later duals use this phase.

### 3.4 Surface dual definitions

#### A. Role-based UV dual (discarded)

CUT / SIDE / FULL roles on the UV net, then 3D stitches. Disconnected from the extent-trimmed graph. Abandoned.

#### B. Simple face-center dual (too sparse)

Surface node = kept **outer** face-center of an exposed Cartesian face. Dual = core edges between those nodes + inter-cell stitches.

Wrist rest: ~86 nodes / ~114 struts. Missed **half-cell mid-plane diamonds** (the four equatorial face-centers that remain after an outer is culled). Visually a broken skin.

#### C. Geometric dual (current topology)

Lattice-agnostic occupancy, not “octahedral FC only”:

- Build the exposed SC face quad: outer rectangle if that direction was kept, else the **parallel mid-plane cut** through the cell center.
- Any trimmed lattice node on that rectangle (axis-aligned on-plane + in-rectangle test, `plane_eps_frac=1e-4`) is a surface node.
- **Extracted struts:** any core octahedral edge whose both ends are surface nodes (includes same-cell diamonds).
- **Added stitches:** faces that share a 3D Cartesian edge but belong to different cells.

Core STL has **no** added stitches. Dual STL = extracted + stitches.

This recovered half/quarter diamonds and is the dual we deform.

### 3.5 Inter-cell stitches — quarter-cell far-side bug (fixed)

**Symptom:** on elements cut in two axes (e.g. +X and +Y culled), dual struts jumped from a remaining “central” node to the **far-side face-center of the adjacent quarter**.

**Cause:** stitches used `_route_pair_across_edge`. If no node was within \(0.35 \times\) edge length of the shared edge, a 4-node face fell back to “nearest to edge mid,” otherwise **greedy nearest-neighbor pairing of all nodes on both faces**.

On a 12×12×4 quarter, the shared edge with a Y-neighbor is the short **Z edge (L = 4 mm)**, tolerance ≈ 1.4 mm. Remaining nodes (−Y, ±Z face-centers) sit ~6 mm off that edge. Near-edge fails → greedy draws long far-side stitches. The node that *should* have sat on that edge is the **culled** +Y face-center.

**Fix** (`route_geometric_face_stitch`):

1. Prefer nodes actually on/near the shared edge (tighter `near_tol_frac=0.15`).
2. A face with **exactly one** node (typical outer face-center) may represent the whole face even if it is not on the edge.
3. Multi-node faces **never** pair far-side nodes. If nobody is on the shared edge, **skip**.

Wrist rest stitches: **273 → 98**. Shared edges skipped for no on-edge node: **138**. Dual **771 → 596** struts. Extracted core surface edges unchanged (498).

### 3.6 Dual morph experiments (only the dual; core frozen)

| Method | Intent | What we saw | Verdict |
|--------|--------|-------------|---------|
| Face-normal, **max travel** among owning faces | Stair side-wall beats nearby floor | Quarter nodes own two mid-planes; the projector fired **each axis separately** and kept the longer hit → motion in **one dimension** | Rejected for this dual |
| **Equal-weight blend** of owning normals | 45° for two-axis cuts | XY quarters looked diagonal; ignored cell aspect on Z | Better, not final |
| **Aspect-weighted blend** | 12×12×4 → direction \((3,3,1)\) | Two equal XY axes stay 1:1 in plan; XZ is 3:1; nodes that also own ±Z travel \((3,3,\pm 1)\) | Principled, still a constrained ray |
| **Closest point** on CAD | No ray, no ownership | Clean wrap given overbuild+trim; mean 2.26 mm, max 6.98 mm. Stair nodes may prefer the nearer top/bottom patch | **Chosen for now** |

Blender-style TNP and axis-constrained / Z-only policies from the older compare-V2 dual were **not** re-run on this geometric graph. Those fights (floor-snap vs backward stair pull) are why we stopped inventing ray rules until the graph itself was honest. Closest-point is the honest default once nodes are already near \(\partial\Omega\).

---

## 4. Current technique (why we chose it)

**Overbuild + discrete trim** puts remaining octahedral nodes on outer faces or mid-planes that already approximate the CAD. The geometric dual is a subset of that graph, not a second cage invented in UV.

Closest-point morph then:

- does not require classifying quarter vs half vs full for a travel direction;
- does not pick a single axis or a 45°/3×3×1 ray that can miss the local surface;
- cannot jump to the far kidney wall along a long axis-aligned ray (travel is just the Euclidean gap, here typically 1.5–2.3 mm);
- leaves **core Cartesian**, so interior load paths stay regular.

The known closest-point failure mode — stair corners snapping to the nearer floor instead of the side wall — is much milder here than on a full hex-cage morph, because the dual nodes that represent side cuts already sit on mid-planes a few millimetres from the wall, not on a Manhattan stair a cell away. Review of the closest STL looked cleaner than the blended-normal dual, so it is the **main morph** until a specific sidewall defect reappears.

Core remains undeformed on purpose. Volume trim is topological; dual owns skin contact.

---

## 5. Code map

| Piece | Module / script |
|-------|-----------------|
| Extent trim, 25/75, strict octahedron | `graphite/explicit/sc_axis_cull_octahedral.py` |
| Phase search (7 candidates) | `graphite/explicit/node_minimization.py` |
| Geometric dual, stitch router, ownership normals | `graphite/explicit/sc_simple_fc_surface_dual.py` |
| Face-normal / blend project (kept, not default) | `graphite/explicit/sc_quantized_surface.py` (`apply_face_normal_surface_project`, `blend_unit_normals`) |
| Tests | `tests/test_sc_simple_fc_surface_dual.py`, `tests/test_node_minimization.py`, `tests/test_face_normal_projection.py` |
| **Nodal Conformation** (generate + phase search) | `graphite/explicit/nodal_conformation.py` |
| Multi-lattice wrist-rest matrix | `scripts/export_wrist_rest_nodal_conformation_matrix.py` → `test_parts/mouse_wrist_rest_nodal_conformation/` |
| Wrist-rest combined export (octahedral reference) | `scripts/export_wrist_rest_core_and_rect_dual.py` |
| Rectangular cage / turning radius / crease relax | `graphite/explicit/geometry_module.py` (`generate_rectangular_surface_cage`) |

Older duals still in tree (not this path): `sc_extent_surface_dual.py`, `sc_topological_dual.py`, `sc_contextual_surface.py`, `build_simple_fc_surface_dual`.

---

## 6. Wrist-rest artifacts

Directory: `test_parts/mouse_wrist_rest_grid/`

| File | Contents |
|------|----------|
| `trimmed_core_lattice_geometric.stl` | Undeformed core (450 / 1368) |
| `surface_dual_geometric_extracted.stl` | Undeformed geometric dual (283 / 596) |
| `surface_dual_geometric_extracted_deformed.stl` | Aspect-weighted blended-normal morph (reference, not default) |
| `surface_dual_geometric_extracted_closest.stl` | **Current default morph** (cylindrical review) |
| `surface_dual_closest_rect.stl` | **Current dual solid:** rectangular bars + wick joints (dual sphere Ø = 1.1× width, taper over 2× Dj), 3 mm turning radius, crease relax, Boolean ∩ CAD |
| `core_and_dual_closest_rect.stl` | **Current combined:** Cartesian core + wick joints (core sphere Ø = 1.5× strut Ø) + rectangular dual, Boolean ∩ CAD. Mixed core ends sit 0.25 × strut radius inward of the skin. |
| `surface_dual_closest_cyl.stl` | Trial: cylinders + spherical joints, r = 1.5 × core strut radius. Abandoned — CAD clip left struts too thin |
| `core_and_dual_closest_cyl.stl` | Trial combined with cylindrical dual (abandoned) |
| `culled_sc_grid_geometric.stl` | Kept hex overlay (162 hexes) |
| `surface_dual_geometric_report.json` | Counts + closest / blend travel stats |

CAD: `test_parts/Mouse wrist rest v1.stl`. Strut diameter 0.6 mm for review solids.

---

## 7. Issues log (short)

| Issue | Fix | Status |
|-------|-----|--------|
| Half cells spawn backward | Six polarities, not three always-negative | Fixed in volume rules |
| Half→full “transition” to corners | Delete; octahedral FCs weld | Fixed |
| UV/CUT/SIDE dual disconnected | Drop roles; dual from trimmed graph | Abandoned roles |
| Simple-FC dual misses half diamonds | Geometric occupancy on outer **or** mid-plane quads | Current dual |
| Quarter dual struts to far-side FC | Skip stitch unless a node is on the shared edge | Fixed |
| Quarter morph only one axis | Blend owning normals | Tried |
| Blend ignores 12×12×4 aspect | Weight by cell axis length → 3×3×1 | Tried |
| Ray rules still fussier than the graph | Closest-point on dual only | **Default** |
| Volume morph crushes cells | Do not morph core | Policy |
| Dual bars chord through space | Raycast loft stations along interpolated endpoint normals; pin ends to one canonical CAD point per node | Current dual solid |
| Messy joints (overshoot + Bishop frame) | Shared node position + shared face normal; T projected into that tangent plane | Current dual solid |
| Slot-style shortened bars at joints | Bars still meet at the node; add a normal-axis cylinder hub (r = 1.1 × strut r, h = thickness) | Current dual solid |
| Rectangular dual fights sharp CAD edges | Min turning radius on loft paths; crease-node relax (wrist rest only); nlerp section normals | Wrist rest: 3 mm turn + 3 mm crease relax. Adapter: mild turn = strut Ø, crease relax off |
| Aggressive turn radius / crease relax on Adapter | `0.25×cell` (~1.32 mm) fairing + crease relax over-pulled bars → Boolean slivers / tet debris | Fixed: turn R = strut diameter (~0.41 mm); crease_relax = 0 |
| SDF / face-normal flip on dual nodes | Re-orient CAD triangle normals via SDF | Reverted — raw CAD face normals; flip was blamed for post-Boolean slivers |
| Cylindrical dual CAD clip too thin | Revert to rectangular bars | Abandoned |

---

## 8. What is explicitly out of scope (for now)

- Morphing the **core** lattice.
- Volume ↔ dual **stitch / spoke** reconnect after trim (the architectural Stage C in SURFACE_FIRST_DUAL_TRIM). The geometric dual already includes inter-cell skin stitches; core-to-dual spokes are not exported yet.
- Boolean volumetric clip of core struts (graph trim is topological, not a CAD intersection).
- Reintroducing TNP / axis-constrained / Z-only projectors as the default.

---

## 9. Suggested next steps

1. Review closest dual vs CAD; if a sidewall floor-snap shows up, try **closest among hits along the aspect-weighted blend ray**, not a return to max-travel.
2. Optional core↔dual spokes for cut/hanging volume nodes (distance + angle gates from SURFACE_FIRST_DUAL_TRIM).
3. ~~Pack the pipeline behind one generate function~~ → **`generate_nodal_conformation`** (done). Multi-lattice wrist-rest matrix uses it with node min default.

---

## 10. Rectangular dual solid (what actually prints)

Graph connectivity is §3–4. This section is the **solid** that gets Boolean-clipped to CAD. Implementation: `generate_rectangular_surface_cage` in `graphite/explicit/geometry_module.py`, exported by `scripts/export_wrist_rest_core_and_rect_dual.py`.

### 10.1 Dimensions

| Piece | Size |
|-------|------|
| Core cylinder diameter | \(0.6\,\mathrm{mm}\) (review; not a VF target on the wrist rest) |
| Dual bar width | same as core diameter (\(0.6\,\mathrm{mm}\)), in the tangent plane |
| Dual bar thickness | \(1.5\times\) diameter (\(0.9\,\mathrm{mm}\)), **inward** from CAD |
| **Node stress + wick joints (default)** | Core joint sphere Ø \(= 1.5\times\) strut Ø; dual joint sphere Ø \(= 1.1\times\) bar width. Strut/bar section eases from joint-sphere size at each end down to nominal over \(1.25\times\) joint-sphere diameter (smoothstep; short struts clamp). Replaces dual cylinder hubs. |
| Legacy node hub (option off) | cylinder, axis = CAD normal, radius \(= 1.1\times\) strut radius, height \(=\) bar thickness |
| Mixed core ends | inset \(0.25\times\) strut radius (\(0.075\,\mathrm{mm}\)) along the inward normal so cylinders do not stand proud |

### 10.2 Centerline and joints

1. Snap each dual node to CAD **once**. Every incident bar is pinned to that exact point and that node's face normal \(N\) (**raw CAD face normals** — no SDF re-orientation).
2. Sample the 3D chord, raycast interiors onto CAD along interpolated endpoint normals (follows the skin; does not jump to a nearer floor).
3. **Minimum turning radius (optional):** fair interiors whose Menger radius is tighter than the threshold, pulling them *off* the CAD through a crease. Gentle patches stay on the surface. Ends stay shared.
   - Wrist rest: \(3\,\mathrm{mm}\) (fixed).
   - Part2_Adapter: **strut diameter** (\(\approx 0.41\,\mathrm{mm}\)). Earlier \(0.25\times\) cell (\(\approx 1.32\,\mathrm{mm}\)) was too aggressive and produced Boolean slivers; \(R=0\) left hard crease copies but clean Booleans — mild diameter-scale fairing is the compromise.
4. **Crease-node relax:** optional inward pull at shared joints. On for wrist rest (\(3\,\mathrm{mm}\)); **off** on Part2_Adapter (was sinking dual into the solid on ordinary convex curvature).
5. Section frame: thickness along a **blend** of the two endpoint normals (not per-triangle CAD facets). Path tangent \(T\) is projected into that tangent plane so \(T \perp N\). Width \(U = N \times T\).
6. **Joint spheres + wick (default):** spheres at used nodes; strut/bar section eases from sphere size at each end to nominal over \(1.25\times\) sphere diameter (`joint_wick`, smoothstep). Core sphere Ø = \(1.5\times\) strut Ø; dual sphere Ø = \(1.1\times\) bar width. Dual spheres sit on the CAD-snapped node (Boolean \(\cap\) CAD keeps the inward portion). When off: dual cylinder hubs + core `joint_scale=1.05`.
7. Combined solid = core cylinders ∪ dual bars ∪ joint spheres, then Boolean \(\cap\) CAD.

Cylindrical dual (r = \(1.5\times\) core radius, Boolean clip) was tried and **abandoned**: after the clip the remaining lens was too thin.

### 10.3 Why this survived review

- Strict on-surface loft copied every CAD crease into the dual (stair-step).
- Bishop-transported frames and overshot ends made joints messy.
- Slot-style shortened bars were the wrong joint.
- Cylinders whose centerlines lie on CAD get sliced to a sliver.
- Over-aggressive fairing (large turn radius or unconstrained crease relax) pulls bars deep inside the CAD so the Boolean \(\cap\) leaves thin/sliver fragments.

Bars still meet at one point; they may leave the tangent plane at a crease with a **mild** turning radius instead of a knife edge.

---

## 11. Part2_Adapter (cubic octahedral, ~5% VF)

**CAD:** `test_parts/Part2_Adapter.STL`  
**Script:** `scripts/export_part2_adapter_octahedral.py`  
**Outputs:** `test_parts/part2_adapter_octahedral/`

This run uses the **modular SC conformal** path (hex cage morph → stamp octahedral → surface dual), then the rectangular dual solid recipe from §10 with Adapter-specific fairing defaults.

| Spec | Value |
|------|-------|
| Cell | cubic **5.292 mm** (smallest extent \(15.875 / 3\)) |
| Topology | octahedral, hex cage morphed (542 exposed corners, max disp 3.54 mm) |
| Graph | 1104 volume nodes / 3324 volume struts; 3276 dual struts (542 skin-only nodes appended) |
| Strut diameter | **0.410 mm** (\(r = 0.205\,\mathrm{mm}\)) from \(\approx 5\%\) VF with overlap \(0.72\) |
| Dual bars | width \(=\) strut Ø, thickness \(= 1.5\times\) Ø (\(\approx 0.615\,\mathrm{mm}\)) |
| Turning radius | **strut diameter** (\(\approx 0.410\,\mathrm{mm}\)); earlier \(0.25\times\) cell \(\approx 1.32\,\mathrm{mm}\) was too aggressive |
| Crease-node relax | **off** (\(0\)) |
| Joint mode | **wick** (default) |
| Core joint sphere | Ø \(= 1.5\times\) strut Ø (\(\approx 0.615\,\mathrm{mm}\)); earlier trials used \(1.25\times\) |
| Dual joint sphere | Ø \(= 1.1\times\) width (\(\approx 0.451\,\mathrm{mm}\)) |
| Wick taper | \(1.25\times D_j\) smoothstep from sphere size → nominal (short struts clamp) |
| Normals | raw CAD face normals (SDF re-orient reverted) |
| Mixed core inset | \(0.051\,\mathrm{mm}\) (\(0.25 r\)) |
| Combined | `core_and_dual.stl` / `surface_dual.stl` (Boolean ∩ CAD, largest component kept) |

### 11.1 Fairing and joint lessons (Adapter)

| Trial | Result |
|-------|--------|
| Turn R = \(0.25\times\) cell (~1.32 mm) + crease relax | Dual sat deep; Boolean left sliver/tet debris |
| SDF outward normal flip on dual nodes | Suspected contributor to post-Boolean slivers; **reverted** |
| Turn R = 0, crease relax = 0 | Clean Boolean / no slivers; hard crease copies remain |
| Turn R = strut Ø, crease relax = 0 | Mild rounding without over-pull (**kept**) |
| Cylinder hubs only | Baseline joints; sharp bar–bar creases |
| Spheres only (core \(1.25\times\) Ø) | Softens crotches; abrupt step from sphere to strut |
| Wick + core \(1.25\times\) Ø, taper \(2 D_j\) | Compare artifact `compare/core_and_dual_wick.stl` |
| **Wick + core \(1.5\times\) Ø, taper \(1.25 D_j\)** | **Current default** (re-export to refresh primary STLs) |

Trial / compare STLs live under `test_parts/part2_adapter_octahedral/compare/` (hubs, no-turn-radius, earlier wick).

Re-run: `python scripts/export_part2_adapter_octahedral.py`.

---

## 12. Settled Adapter recipe (Aug 2026)

What we kept after the fairing / joint experiments:

| Layer | Choice |
|-------|--------|
| Grid | Modular SC conformal hex cage morph (exposed corners → CAD), then stamp topology |
| Dual connectivity | Rule `skin_mode` (octahedral: face-centroid dual) |
| Dual solid | Rectangular bars, width = strut Ø, thickness = \(1.5\times\) Ø |
| Path fairing | `min_turn_radius` = strut Ø; `crease_relax` = 0; raw CAD face normals |
| Joints | **Wick:** spheres + end tapers; core Ø \(= 1.5\times\) strut Ø; dual Ø \(= 1.1\times\) width; taper \(= 1.25\times D_j\) |
| Core | Cylinders on non-skin volume edges; surface ends inset \(0.25 r\); Boolean \(\cap\) CAD |

Code: `joint_wick` / `joint_wick_length_scale` in `geometry_module.py`; Adapter script defaults above. Primary STLs may lag param tweaks until re-export.

Wrist-rest path (§1–10) differs: Cartesian overbuild + extent trim + geometric dual (core not morphed). Same dual solid + wick joint recipe applies once the dual graph exists.

**Node-plane generalization (in progress):** see [SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md](SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md). Octahedral 25/75 is the midplane policy for planes \(\{0,0.5,1\}\). **Wrist-rest multi-topology compare uses Nodal Conformation** (not cage morph): `python scripts/export_wrist_rest_nodal_conformation_matrix.py`. The older conformal-morph matrix (`export_wrist_rest_sc_matrix.py`) is the wrong path for that compare.
