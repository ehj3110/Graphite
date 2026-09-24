# Surface-First Dual + Volumetric Trim (SC Conformal Pivot)

**Status:** architectural target (Aug 2026) — not yet the production `morph_hex_scaffold` path  
**Stress part:** `test_parts/Mouse wrist rest v1.stl`  
**Related:** [MODULAR_SC_CONFORMAL.md](MODULAR_SC_CONFORMAL.md), [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md), [SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md](SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md) (wrist-rest overbuild + extent trim + geometric dual; closest-point morph)

This document records the pivot away from **projecting a rigid SC hex cage onto CAD** and toward a **decoupled surface dual + trimmed volume + gated stitch**, in the spirit of enterprise surface-first lattices (e.g. Carbon Design Engine–style skins).

---

## 1. Executive Summary & The Morphing Problem

### 1.1 What we were doing

The modular SC conformal pipeline retained a Cartesian hex background grid with a face-centroid volume-fraction (VF) gate, then **morphed every exposed hex corner onto the CAD** (`closest` or `face_normal`), Laplacian-relaxed the interior scaffold, stamped a topology rule into the deformed bricks, and optionally lofted a surface dual from the **already-morphed** skin edges.

That design assumes a local projection \(p \mapsto \pi_{\partial\Omega}(p)\) can turn a **rigid, axis-aligned complex** into a mesh whose boundary is homeomorphic (or at least manufacturable) on \(\partial\Omega\).

### 1.2 Why local projection fails on non-isomorphic boundaries

Let \(\mathcal{H}\) be the VF-retained SC hex complex and \(\partial\mathcal{H}\) its combinatorial boundary (faces owned by exactly one hex — including stair risers). The CAD solid \(\Omega\) has boundary \(\partial\Omega\) that is **not** a small normal graph over \(\partial\mathcal{H}\):

- **Non-isomorphism / stair mismatch.** After a 50% face-centroid cull, \(\partial\mathcal{H}\) is a Manhattan hull. A single stair corner often owns both a **floor** face and a **side** face. Nearest-point projection
  \[
  \pi(p)=\arg\min_{q\in\partial\Omega}\|q-p\|
  \]
  sends that corner to the **geometrically nearer** patch. On a thin wrist rest, the floor/top is typically nearer than the side wall, so “side-wall” combinatorial nodes snap vertically. Face-normal rays with a **farthest-hit** tie-break fix the wall preference but allow long axis-aligned rays to hit the **far** side of a kidney outline (\(\sim 20\,\mathrm{mm}\) pure \(XY\) jumps), inverting column order.

- **Lateral folding / negative Jacobians.** Large tangential displacements between neighboring corners destroy hex maps. If corner images cross in the parametric sense, the hex Jacobian (or scalar triple product of edge frames) changes sign → crushed / inverted bricks, sub-diameter struts, fused solids that read as holes.

- **Clamp vs contact.** Bounding travel per axis by \(\alpha\cdot(s_x,s_y,s_z)\) (e.g. \(\alpha=0.5\) on \(12\times12\times4\,\mathrm{mm}\) → budgets \(6/6/2\,\mathrm{mm}\)) prevents some folds but leaves nodes short of \(\partial\Omega\). Residual “snap still-outside” with nearest point reintroduces bunching; it does **not** finish short **inside** nodes whose nearest hit was floor/top.

**Honest limit:** no single local rule on hex corners has simultaneously delivered (a) stair side-wall preference, (b) non-folding element quality, and (c) full surface contact on this class of CAD. Forcing the volume graph to *be* the conformal skin is the wrong authority.

### 1.3 Architectural direction

**Decouple surface topology from internal volume.**

- The **surface dual** is the conformal authority: an independent graph (and solid cage) registered to \(\partial\Omega\).
- The **volume lattice** stays (mostly) Cartesian / lightly processed, then is **Boolean-trimmed** to \(\Omega\), accepting cut or hanging stubs at the boundary.
- A **heuristic stitch** reconnects eligible cut volume nodes to the dual under manufacturability gates; failures are culled, not forced.

The dual may (and should) conform **more perfectly** than the volume. Volume provides interior connectivity and stiffness; dual + trim provide the printable skin.

---

## 2. The Decoupled Architecture (The Solution)

### 2.1 Pipeline overview

```text
CAD Ω
  │
  ├─► (A) SC background hex grid + VF cull     → internal hex complex H
  │         stamp topology (grid / …)            → volume graph G_vol
  │         [optional light inward-only morph]
  │         Boolean / clip against Ω             → cut volume G_vol∩Ω
  │                                                   cut nodes / hanging stubs
  │
  └─► (B) Independent surface dual on ∂Ω       → surface graph G_dual
            loft rectangular (or rule) cage
            Boolean flush to CAD                 → solid skin

  (C) Heuristic stitch: eligible cut nodes ──struts──► dual nodes
        distance gate · angle gate · orphan cull

  (D) Union solids (volume + dual + stitch) → export
```

### 2.2 Stage A — Raw internal SC hex grid

- Generate axis-aligned SC hex bricks over CAD bounds; keep cells whose face-centroid inside fraction meets `volume_fraction_threshold` (existing `cull_hex_elements`).
- Stamp the chosen volume rule into **undeformed or lightly treated** bricks.
- **Do not** use bidirectional nearest-point / farthest face-normal ironing as the primary conformal step.
- Optional: inward-only cleanup of clearly outside corners, or a mild clamp, solely to reduce gross exterior junk before trim — not to achieve watertight skin conformity.
- **Volumetric trim:** intersect the volume solid (or clip the strut graph) with \(\Omega\). Edges that cross \(\partial\Omega\) become **cut**: a retained interior endpoint plus a hanging stub toward the boundary (or a new cut-node on \(\partial\Omega\) if the clipper inserts one). Those cut nodes are stitch candidates.

Geometric honesty: trim does **not** invent a perfect stair-to-wall hex. It leaves an intentional gap between Cartesian interior and CAD skin for the dual + stitch to own.

### 2.3 Stage B — Independent surface dual

- Build the dual from **combinatorial exposed faces of \(H\)** (same face-adjacency definition as today) **or** from a CAD-native sampling of \(\partial\Omega\) guided by those faces — but **place** dual vertices with a **surface-native** rule, e.g.:
  - face-centroid → project along **local CAD normal** / exposed-face normal onto \(\partial\Omega\);
  - dual edges as chords lofted with station projection onto CAD (existing rectangular cage machinery).
- Solidify with width / thickness / outward oversize, then Boolean-intersect with CAD so the exterior is flush.
- This stage does **not** inherit failed volume-corner projections. Skin conformity is the dual’s job.

Existing hooks to reuse: `find_exposed_faces`, skin modes in `hex_topology_module`, `generate_rectangular_surface_cage`, Boolean trim helpers — rewired so dual placement is **not** downstream of a full cage morph.

### 2.4 Why this resembles enterprise surface-first designs

Enterprise engines commonly separate:

| Layer | Responsibility |
|---|---|
| Skin / dual | Exact boundary conformity, washability of outer junctions |
| Core lattice | Interior connectivity; may be clipped rather than projected |
| Transition | Short, gated bonds — not arbitrary long projections |

We accept the same trade: some cut stubs are deleted rather than stretched into unprintable geometry.

---

## 3. Heuristic Stitching Algorithm (Core Logic)

Connect **cut / hanging volume nodes** from the trimmed interior to **nodes of the independent surface dual**. Goal: manufacturable bridges across the stair gap without recreating long shallow struts or resin traps.

### 3.1 Inputs

- \(V_{\mathrm{cut}}\): cut or hanging nodes on the volume side after trim (and optionally their incident hanging stub directions).
- \(V_{\mathrm{dual}}\): dual nodes already on or flush to \(\partial\Omega\).
- Unit cell length \(L\): primary SC pitch used for the volume grid. For anisotropic cells \((s_x,s_y,s_z)\), define
  \[
  L=\min(s_x,s_y,s_z)
  \]
  unless the UI exposes an explicit stitch length (default: one **unit cell length** as specified below — for isotropic grids \(L=s\); for wrist-rest \(12\times12\times4\), product default should be documented in UI as \(L=\min(s_i)\) or “longest horizontal pitch”; **implementation must pick one and stick to it** — recommended: \(L=\max(s_x,s_y)\) for horizontal reach on flat parts, or the planner-mandated “exactly 1 unit cell length” as the **configured cell size scalar** when isotropic, and \(\max(s_x,s_y)\) when anisotropic so side-wall stitches can span one full plan-cell).

> **Parameter lock for v1:** Distance gate max reach \(= 1\times\) **configured unit cell length**. In the anisotropic UI, bind “unit cell length” for this gate to \(\max(s_x,s_y)\) (horizontal cell size). Do not silently use \(s_z\).

### 3.2 Per cut-node procedure

For each node \(c\in V_{\mathrm{cut}}\):

1. **Candidate dual nodes**  
   Query dual nodes within a ball of radius \(R_{\mathrm{search}}\ge L\) (spatial hash / KD-tree). If none, fail → orphan path.

2. **Primary candidate**  
   Prefer the nearest dual node \(d^\star\) in Euclidean distance. Optional secondary ranking: prefer dual nodes whose owning dual edge/face normal aligns with the hanging stub or local CAD outward normal (tie-break only; gates below are hard).

3. **Distance gate (hard)**  
   Let \(r=\|(d^\star-c)_{XY}\|\) be the **horizontal** (plan-view) reach — coordinates in the build / grid \(XY\) plane unless a part-aligned frame is defined.
   - **Accept only if** \(r\le L\) (exactly **one unit cell length** max horizontal reach).
   - If \(r>L\), **reject** the connection (do not fall back to a farther dual node outside the gate).

4. **Angle gate (hard)**  
   Let \(\mathbf{u}=d^\star-c\) and \(\hat{z}\) the build vertical (or part “up” axis used for wash orientation — default CAD \(+Z\) for the wrist rest).
   - Define the angle from vertical:
     \[
     \theta=\arccos\left(\frac{|\mathbf{u}\cdot\hat{z}|}{\|\mathbf{u}\|}\right)
     \]
     so \(\theta=0^\circ\) is vertical, \(\theta=90^\circ\) is horizontal.
   - **Accept only if** \(\theta\le 60^\circ\).
   - Shallower (more horizontal) stitches are **rejected**: they form acute wash-trapping junctions with the skin and act as extreme stress concentrators under compression/bending.

5. **Commit or cull**  
   - If both gates pass: insert stitch strut \(\{c,d^\star\}\) (and solidify with the volume strut diameter unless a dedicated stitch radius is set).
   - If either gate fails: **orphan cull** — delete \(c\) and its attached hanging stub edge(s) from the volume graph. Do not leave a free-floating stub into empty space inside the wash volume.

### 3.3 Batch / ordering notes

- Process cut nodes in a stable order (e.g. increasing \(Z\), then Hilbert on \(XY\)) so results are reproducible.
- One dual node may accept multiple stitches only if local valence / minimum angle between stitches stays printable; v1 may allow multiple without extra gates, then tighten if wash tests fail.
- After orphan cull, drop any volume component that is disconnected from the main interior graph and smaller than a size threshold (optional cleanup).

### 3.4 Manufacturing rationale (no fluff)

- **Distance \(\le 1\) cell:** longer horizontal jumps recreate the folding / cross-column problem and leave long unsupported spans in resin.
- **Angle \(\le 60^\circ\) from vertical:** shallow skin junctions trap uncured resin and are hard to flush in wash/cure; they also concentrate stress at the dual–volume interface.
- **Orphan cull over forced stitch:** a missing local bond is preferable to an unprintable or wash-trapping strut. The dual skin still closes the exterior.

### 3.5 What this algorithm deliberately does *not* do

- It does not morph the entire hex cage onto \(\partial\Omega\).
- It does not use farthest face-normal travel to choose dual targets.
- It does not keep hanging stubs that fail gates “just to fill the stair.”

---

## 4. Sequential Pipeline Integration (Streamlit)

Graphite’s UI is a **sequential wizard** (`app.py`, see [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md)). Surface-first SC conformal should collect parameters in order so each stage can preview before the next commits.

### 4.1 Recommended step mapping

Extend or branch the explicit / conformal path so parameters are collected **sequentially**:

1. **Internal SC hex grid (density / cell size)**  
   - CAD / part selection (existing geometry step).  
   - Isotropic or anisotropic cell size \((s_x,s_y,s_z)\).  
   - VF threshold, topology rule (`grid`, `octahedral`, …).  
   - Strut radius for volume.  
   - Preview: culled unsnapped (or lightly trimmed) volume only.

2. **Surface dual generation**  
   - Skin mode / dual recipe (corner-edge cage, face-centroid dual, …).  
   - Cage width, thickness, normal oversize.  
   - Dual placement mode (normal / face-centroid project — **not** volume nearest-point).  
   - Preview: dual solid alone, and dual overlaid on CAD.

3. **Heuristic stitching thresholds**  
   - Unit cell length binding for distance gate (display computed \(L\)).  
   - Max horizontal reach (default \(=L\); advanced override only in Advanced UI).  
   - Max angle from vertical (default \(60^\circ\)).  
   - Optional stitch strut radius.  
   - Preview: stitch candidates colored pass/fail; orphan count; final union.

4. **Execute / export**  
   - Run trim → dual → stitch → union.  
   - Export STL/STEP + parameter manifest (gate values, cull counts, cell size).

### 4.2 UI honesty

- Show **orphan cull counts** and **rejected stitch reasons** (distance vs angle). Users must see that empty stair pockets are intentional when gates fail.
- Do not imply the volume lattice is “fully conformal” in this mode; label the mode **Surface-first dual + trim**.
- Keep legacy `morph_hex_scaffold` projection modes available under Advanced / experimental for comparison, not as the default for this path.

### 4.3 Headless / API sketch

```text
generate_surface_first_sc_lattice(
    cad,
    cell_size=...,
    volume_fraction_threshold=0.5,
    rule_name="grid",
    strut_radius=...,
    dual_*=...,
    stitch_max_horizontal=None,  # default → 1 unit cell length
    stitch_max_angle_from_vertical_deg=60.0,
    ...
) → nodes, volume_struts, dual_struts, stitch_struts, report
```

`report` must include: `n_cut_nodes`, `n_stitched`, `n_rejected_distance`, `n_rejected_angle`, `n_orphaned`, `unit_cell_length_used`.

---

## 5. Relation to Prior Attempts (Context)

| Approach | Outcome |
|---|---|
| Closest-point ironing of exposed hex corners | Good global look; stairs snap to floor/top; clamp leaves short insides |
| Face-normal + farthest multi-face hit | Stairs can hit walls; lateral overshoot folds columns |
| Travel clamp + residual outside closest | Limits folds; stranded insides remain; residual closest bunches |
| Dual loft after volume morph | Dual inherits bad chords or diverges when stations re-project |
| **Surface-first dual + trim + gated stitch** | Target architecture in this document |

SDF note for any residual volume cleanup: use `safe_signed_distance` (positive = outside), not raw trimesh signs, on the wrist-rest class of meshes.

---

## 6. Success criteria

- Dual exterior is flush to CAD (Boolean) across curved and stair regions.
- Volume interior remains non-folded (no systematic negative Jacobians from projection).
- Stitches obey **horizontal reach \(\le 1\) cell** and **\(\theta\le 60^\circ\) from vertical**; failures orphan-culled.
- Streamlit collects grid → dual → stitch parameters in sequence with actionable previews.
- Wash/cure on printed wrist-rest-class parts does not show resin-trap acute skin junctions attributable to shallow stitches.

---

## 7. Out of scope for v1

- Soft spring coupling of all hex corners to dual vertices (possible later).
- Replacing VF cull with a true volumetric fraction oracle.
- Guaranteeing every stair riser receives a stitch (gates may leave gaps; dual still skins the part).
