# V3 — Hybrid: Intersect-All Dual Shrinkwrap + VF Volume + Morph-to-Dual

**Agent role:** Implement the hybrid comparison version 3.  
**Parent index:** [SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md)

---

## Thesis

Run **two culls** of the same background SC grid:

1. **Loose cull (dual scaffold):** keep every hex that intersects the CAD **at all** (not the 50% VF rule). Shrinkwrap that complex onto the CAD to form the **surface dual**.
2. **Strict cull (volume):** keep hexes with the normal **50% face-centroid VF gate**. Do **not** morph volume corners to nearest CAD point. Instead morph volume boundary DOFs to the **nearest surface-dual node**, allowing lateral motion when the dual sits on a side wall.

Cap how many **internal (volume) struts** may land on one dual node by the **per-element topological valency** of that node class in the volume rule (e.g. octahedral face-center node valency **4**). Excess candidates are rejected (no pile-up that exceeds a single cell’s local degree).

---

## Algorithm (normative)

### 0. Background grid

Generate the same SC background hex grid over CAD bounds (shared `cell_size`).

### 1. Loose complex → dual

1. Cull with **any-intersection** retention:
   - Practical v1 implementation: treat as `cull_hex_elements(..., mode="boolean")` **or** keep a cell if **any** of the 6 face centroids is inside, **or** if any corner is inside / any hex edge intersects CAD.  
   - **Preferred clarity:** `keep if n_inside_face_centroids >= 1` (boolean face-centroid mode). Document which you chose in the report.
2. Identify exposed faces of this **loose** complex.
3. Build dual graph from those faces (rule skin mode: for `grid` use corner-edge cage / for `octahedral` use face-centroid dual as appropriate).
4. **Shrinkwrap dual onto CAD:** project dual nodes (and/or loft stations) onto \(\partial\Omega\) with a surface-native method (closest on dual nodes is OK here because dual nodes already lie near the boundary of a looser hull; prefer normal/face-centroid project if both are easy). Goal: dual is the conformal skin.
5. Solidify dual cage (rectangular loft + Boolean flush recommended).

### 2. Strict complex → volume

1. `cull_hex_elements(..., volume_fraction_threshold=0.5, mode="conformal")` on the **same** background grid.
2. Stamp volume topology on these hexes (Cartesian / undeformed scaffold positions).
3. Identify volume **boundary / iron** nodes (exposed faces of the **strict** complex, and/or volume nodes that should couple to skin).

### 3. Morph volume → dual (not CAD)

1. For each volume iron node \(v\), find nearest dual node \(d^\star\) (3D Euclidean; **lateral moves allowed**).
2. Move \(v\) toward / onto \(d^\star\) (v1: snap to \(d^\star\); optional later: clamp travel).
3. Laplacian-relax remaining free volume scaffold nodes with iron fixed (optional but recommended).
4. Re-stamp or update volume strut geometry from morphed nodes.

### 4. Valence gate (internal → dual)

When creating **connections from volume/internal struts to dual nodes** (either by morph coincidence weld or explicit stitch edges):

- Let \(k_{\mathrm{rule}}\) = maximum nodal degree **inside one parent element** for that topology rule’s local graph.  
  - **Octahedral:** each local node has valency **4** → \(k_{\mathrm{rule}}=4\).  
  - **Grid:** hex-corner cage edges: a corner in one hex has degree **3** along the element’s edges; use the rule’s documented local degree (for SC grid skin, use **3** for pure corner cage, or match `HexTopologyRule.valency_cutoff` only if that matches “within its own element” — **prefer counting local strut degree in `apply_hex_*` for one element**).
- For each dual node \(d\), count accepted volume→dual incident bonds \(b(d)\).
- **Reject** any additional bond that would make \(b(d) > k_{\mathrm{rule}}\).
- Counting rule: only **internal/volume→dual** bonds count toward the cap; pure dual–dual cage edges do **not** consume the budget.

### 5. Export

Union volume + dual (+ any explicit coupling struts); write compare artifacts and valence rejection stats.

---

## Why this might beat V1 on stairs

The dual is built from a **fatter** boundary (any-intersect hexes), so shrinkwrap can bridge stair gaps on the skin. Volume nodes aim at **dual nodes already on the wall**, not at nearest CAD (floor). Lateral morph is intentional.

## Risks (document honestly)

- Loose dual may be denser / bulkier than VF dual.
- Many volume nodes mapping to one dual node → valence gate drops bonds → local gaps.
- Morph-to-dual can still fold hexes if dual sampling is coarse; report collapsed-hex counts.

---

## Code ownership

| Path | Purpose |
|---|---|
| `graphite/explicit/compare_v3_hybrid/` | Loose cull helper, shrinkwrap dual, morph-to-dual, valence gate, `generate_compare_v3(...)` |
| `scripts/compare_sc_v3_wrist_rest.py` | Wrist-rest export |
| `tests/compare_v3/` | Valence cap unit test; loose vs strict cull count smoke; export smoke |

**Do not** edit V1/V2 packages.

---

## Deliverables

- `test_parts/mouse_wrist_rest_grid/compare_v3_hybrid_grid_12x12x4_untrimmed.stl`  
  (also `..._octahedral_...` if validating valence=4)
- `_render.png`, `_report.txt` with:
  - `n_hex_loose`, `n_hex_strict`
  - `n_dual_nodes`, `n_volume_iron`
  - `n_morph_to_dual`
  - `n_bonds_rejected_valence`
  - `k_rule`
- Import: `from graphite.explicit.compare_v3_hybrid import generate_compare_v3`

---

## Acceptance

- Dual comes from **any-intersection** (or boolean face-centroid) complex, not from VF 50% alone.
- Volume uses VF 50% and morphs to **dual nodes**, not `project_to_cad_surface` as primary.
- Valence cap enforced at \(k_{\mathrm{rule}}\) (octahedral demo with 4 required in tests).
- Lateral motion to dual is allowed (no requirement that morph be purely normal).
