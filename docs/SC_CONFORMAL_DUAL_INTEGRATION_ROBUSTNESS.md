# SC conformal dual — Graphite explicit integration & cubic robustness

**Status:** current method is **[NODAL_CONFORMATION.md](NODAL_CONFORMATION.md)** (Aug 2026). This page keeps the node-plane idea and cubic-type table.

---

## Terminology (settled)

| Term | Meaning |
|------|---------|
| **Culling** | Drop **whole unit cells** (hexes). VF Full/Empty is culling. |
| **Trimming** | Drop **sub-cell** nodes/struts (halves, quarters, …) while the hex still contributes a partial lattice. |

**Current Nodal Conformation:** every SC rule **culls** empty hexes, then **trims** remaining cells to kept node planes (table below). **Grid** is the exception: cull-only at VF ≥ 0.50 (full stamp; no sub-cell trim).

| Lattice | Node planes | Trim cuts | Notes |
|---------|-------------|-----------|--------|
| grid | 0, 1 | — | Cull whole cell if VF < 0.50 |
| octahedral / star / octet / hex_face_dual | 0, 0.5, 1 | 0.25, 0.75 | Half at mid-plane **plus** box occupancy |
| kelvin | 0, 0.5, 1 (trim; nodes also at 0.25/0.75) | 0.25, 0.75 | Same cuts as octahedral; 0.25/0.75 nodes only in a kept half |
| tesseract | 0, 0.25, 1 as stops | 0.125, then jump | Inner shell at 0.25 only; 0.25–0.75 still 0.25; >0.75 keeps the full cell |

API: `graphite/explicit/sc_node_plane_trim.py`. Wrist-rest cores: `scripts/export_wrist_rest_trimmed_undeformed.py`.

---

## Node-plane extent policy (preferred over per-rule half cuts)

Rather than maintaining `octahedral_half_*` (and someday kelvin/tesseract halves), drive boundary trim from **where nodes sit in the unit cell**.

### Idea

1. Stamp each hex rule into the unit cube \([0,1]^3\).
2. Collect unique fractional coordinates per axis → **node planes** \(f_0 < f_1 < \cdots < f_n\).
3. Place decision thresholds at midpoints \(t_i = (f_i + f_{i+1}) / 2\).
4. Compare material extent on that axis to the \(t_i\)’s → keep the outermost still-covered node plane(s); cull nodes beyond; drop struts that touch culled nodes.
5. Skin / dual = graph on the **outer kept planes** of exposed faces.

Octahedral \(\{0, 0.5, 1\}\) recovers the historical **25/75** tiers automatically (\(t \in \{0.25, 0.75\}\)). Grid \(\{0, 1\}\) → single mid threshold \(0.5\). Tesseract \(\{0, 0.25, 0.75, 1\}\). Kelvin is denser — **derive**, don’t hardcode.

### Caveats

- **Star:** body center at \(0.5\) is not a face midplane — later distinguish planar shells vs interior-only fractions.
- Culling nodes must leave a valid strut subgraph (or regenerate from kept nodes).
- Quarters / eighths still compose from independent X/Y/Z decisions.

### Early code (test workflow)

| Piece | Path |
|-------|------|
| Node-plane trim | `graphite/explicit/sc_node_plane_trim.py` |
| Plane derivation | `graphite/explicit/sc_node_planes.py` |
| Unit tests | `tests/test_sc_node_planes.py`, `tests/test_sc_node_plane_trim.py` |
| **Nodal Conformation** API | `graphite/explicit/nodal_conformation.py` |
| Wrist-rest multi-lattice matrix | `scripts/export_wrist_rest_nodal_conformation_matrix.py` → `test_parts/mouse_wrist_rest_nodal_conformation/` |
| Legacy conformal-morph matrix (wrong compare path) | `scripts/export_wrist_rest_sc_matrix.py` → `test_parts/mouse_wrist_rest_sc_matrix/` |

**Nodal Conformation** matrix: Cartesian + dual-only closest-point, **node minimization on**, rectangular dual, **no stress deconcentration**. Volume: node-plane trim (grid: VF ≥ 0.50 cull-only).

Rules: `grid`, `octahedral`, `star`, `octet`, `kelvin`, `tesseract`, `hex_face_dual`.

---

## Settled solid recipe to package (when stress decon is on)

1. Hex cage morph (exposed corners → CAD) via modular SC conformal path  
2. Stamp registered hex topology into deformed cells  
3. Build skin / dual graph from rule `skin_mode`  
4. Solids: volume cylinders + rectangular dual bars + optional **wick joints**  
   - Core sphere Ø = **1.5×** strut Ø  
   - Dual sphere Ø = **1.1×** bar width  
   - Taper length = **1.25×** joint-sphere diameter  
5. Mixed core ends inset \(0.25 r\); Boolean ∩ CAD  

Wrist-rest **matrix** run intentionally leaves wick / stress spheres **off** for a clean topology compare.

---

## Cubic types in scope

From `hex_topology_module._default_rules`:

| Rule family | Examples | Skin mode | Main risk |
|-------------|----------|-----------|-----------|
| Face-centroid dual | `octahedral`, `hex_dual`, `hex_face_dual`, half-octahedra | `face_centroid_dual` | Lowest — matches Adapter |
| Corner-edge cage | `grid`, `tesseract` | `corner_edge_cage` | Stair-step silhouette; corner valency |
| Face-local rule | `star`, `octet` | `face_local_rule` | Shared-face double coverage / gaps |
| Kelvin bridge | `kelvin`, `kelvin14` | `kelvin_face_bridge` | Long chords; fairing |
| Deferred | `a15_kagome` | `none` | **Out of scope** until skin exists |

---

## Robustness plan (updated)

### 1. Node-plane trim (primary)

Implement extent cull using `node_plane_policy(rule)` mid thresholds — one policy for all cubic rules. Half-octahedron stamps become optional/legacy.

### 2. Skin graph contract (fail loud)

Still validate connectivity / coverage per `skin_mode` after trim.

### 3. Matrix regression

Wrist-rest matrix script is the first visual harness. Next: sphere fixture + watertight asserts in `tests/`.

### 4. Rule-agnostic solids

Wick / rectangular cage / hubs take abstract `(nodes, struts)` only.

### 5. Production API (later)

`generate_sc_conformal_lattice_solid(...)` after the matrix looks good.

---

## Non-goals (for this kickoff)

- Full graphite-explicit replacement of wrist-rest geometric extent trim  
- Stress deconcentration on the matrix exports  
- `a15_kagome` until `skin_mode != none`  
