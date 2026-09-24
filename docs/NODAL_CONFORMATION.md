# Nodal Conformation (Aug 2026)

**Status:** active wrist-rest / Adapter workstream  
**Not:** hex-cage conformal morph (that path was the wrong compare)  
**CAD:** `test_parts/Mouse wrist rest v1.stl`  
**Cell:** 12 x 12 x 4 mm  
**API:** `graphite/explicit/sc_trim_shared_edge_engine.py` (gold-standard octahedral), `sc_node_plane_trim.py`, `sc_node_planes.py`  
**Lab trail:** [SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md](SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md)

Nodal Conformation keeps the **volume lattice Cartesian**. Conformity is a **surface dual** (and/or a closest snap of leftover outside nodes). The hex cage is never morphed into the CAD.

---

## Terminology

| Term | Meaning |
|------|---------|
| **Culling** | Drop **whole unit cells**. |
| **Trimming** | Drop **sub-cell** nodes/struts (halves, quarters, …) while the hex still contributes. |
| **Box occupancy** | Keep a node only if at least one **incident node-plane box** contains CAD. Catches 45-degree cuts whose AABB still looks full. |
| **Deform (cores)** | Same connectivity; nodes with SDF > 0 snap to **closest point** on CAD. Interior stays Cartesian. |

---

## Pipeline

```text
CAD
  → SC background hexes (node-min phase)
  → Cull empty hexes
  → Stamp topology (undeformed)
  → Trim by node-plane policy (+ box occupancy, except grid)
  → Loose-external prune (nodal conformation only): drop outside nodes whose remaining neighbors are all outside; keep an outside node if it has an inside neighbor
  → Optional: closest-project leftover outside nodes
  → Surface dual from the kept graph
  → Closest-point project dual nodes only
  → Solidify (cylinders / rectangular dual); Boolean intersect CAD when printing
```

**Grid** is cull-only: keep the full stamp if VF >= 0.50, else drop the hex. No sub-cell trim (one box = the whole cell).

---

## Trim cuts (current)

Decision planes are mid-thresholds between node planes, except the specials below. Box occupancy is ANDed with plane keep for every type except grid.

| Lattice | Trim planes | What happens |
|---------|-------------|--------------|
| grid | — | Cull if VF < 0.50; full cell otherwise |
| octahedral, star, octet, hex_face_dual | 0, 0.5, 1 | Cuts at 25/75; box occupancy |
| kelvin | 0, 0.5, 1 | Same cuts as octahedral. Nodes at 0.25/0.75 still exist, but only inside a **kept half** (not their own trim stops) |
| tesseract | 0, 0.25, then jump | Trim to 0.25 as usual. **0.25–0.75 still only 0.25** (no 0.75 stop — that made hanging struts). **> 0.75 keeps the full cell** |

Independent-axis AABB cannot see a diagonal. Box occupancy uses the eight octants between planes 0 / 0.5 / 1.

- **Gold / default (`box_vf_min=0`):** an octant is occupied if **any** CAD sample is inside.
- **Octet experiment (`box_vf_min=0.50`, user OK):** occupied only if that octant’s sampled VF ≥ 0.50. Sliver diagonals stair to half/three-quarter cells. See [UNIVERSAL_DUAL_HANDOFF.md](UNIVERSAL_DUAL_HANDOFF.md).

---

## Deform vs dual morph

Two different "deformed" artifacts:

1. **Trimmed core, deformed** — leftover outside **volume** nodes closest-projected. Used for cull/trim review. Folders: `trimmed_box_occupancy/`, `kelvin_tesseract_trim/`.
2. **Surface dual, deformed** — dual graph of the trimmed lattice after leftover outside volume nodes are closest-projected, then dual nodes closest-projected onto CAD. Core is not in these files. Folder: `surface_dual_deformed/` (`scripts/export_wrist_rest_surface_dual_deformed.py`). Dual = outside nodes plus inside endpoints of CAD-crossing struts (octahedral/skin fallback if that graph is empty).

Do not confuse with **conformal morph** (hex corners projected, volume cage crushed).

---

## Gold standard (compare against this)

**Locked 19 Aug 2026** after the Ø50 × 100 mm octahedral cylinder looked right.

Future trim / dual / print work should be judged against this stack, not against universal dual, node-min, or wick, until those are re-integrated on purpose. The gold engine stays the **oracle** — do not rewrite it to match universal dual.

| Piece | What it is |
|-------|------------|
| Engine | `generate_octahedral_trim_shared_edge` in `graphite/explicit/sc_trim_shared_edge_engine.py` |
| Volume | VF cull, 25/75 node-plane trim, box occupancy, ghost-hex CAD occupancy |
| Dual | Shared-edge octahedral F: skin-native volume bars + inter-cell stitches on a shared Cartesian edge. Same-hex opposite-face chords stay volume; jump-cut is not dual. Midplane faces with several nodes do not stitch far-side nodes. |
| Not in this baseline | Universal dual (`sc_role_surface_dual`), node minimization, joint wick / stress deconcentration, loose-external prune |
| Dual on CAD | Closest-point project dual nodes only (core stays Cartesian) |
| Review fixture | `outputs/sc_cylinder_50x100_octahedral_temp_engine/` — Ø50 × 100 mm, 10 mm cells, origin `[0,0,0]`, 10% VF on the volume graph, dual bars 1.2× thickness, no wick. Regenerator: `scripts/export_sc_cylinder_temp_engine.py` |

## Universal dual (layered surface dual roles)

**Product name:** universal dual. **Code:** `build_role_surface_dual` / `build_layered_surface_dual` (`dual_rule`: `layered_surface_dual_roles`). Spec: [SURFACE_DUAL_ROLES.md](SURFACE_DUAL_ROLES.md).

Intent: one connector for every SC stamp (octahedral, octet, grid blends, …) without deleting F–E / F–C. On **octahedral** it is required to **graph-match gold**. Align phases: [LAYERED_SURFACE_DUAL_ROLES_ALIGN.md](LAYERED_SURFACE_DUAL_ROLES_ALIGN.md).

**Reviewed 19 Aug 2026 (looks right):**

- Cylinder gold vs universal: `outputs/layered_dual_review/` — 408 / 880 / 104 exact.
- Blend F–C (octahedral \| grid): same folder `blend/`.
- Wrist rest octahedral 12×12×4: `outputs/wrist_rest_gold_vs_universal_dual/` — 219 / 449 exact.
- Wrist rest octet 12×12×4 (no gold oracle): `outputs/wrist_rest_octet_universal_dual/`.

`generate_nodal_conformation` already calls this dual (pass `volume_struts` for promote). Grid/tesseract still use the C-perimeter helper; Kelvin still uses K-bridge. Star does not synthesize face centers (C on the skin; exposed B on cuts is deferred).

Do not use `surface_dual_from_volume` (crossing-layer) for new exports — that kept volume chords through the CAD.

---

## Node minimization

Axis-6 phase search (origin ± dx, dy, dz). Increment = 0.5 × (min node-plane gap) × cell. Frozen wrist-rest offsets:

| Rule | Offset (mm) |
|------|-------------|
| grid, star, octet, tesseract | [0, 0, 0] |
| octahedral | [0, 0, -1] |
| kelvin | [1.5, 0, 0] |
| hex_face_dual | [0, -3, 0] |

---

## Export map (wrist rest)

Current gold vs universal overlays live under **`outputs/`** (never `test_parts/` root):

| Folder | Contents |
|--------|----------|
| `outputs/layered_dual_review/` | Cylinder + coupons + blend |
| `outputs/wrist_rest_gold_vs_universal_dual/` | Octahedral wrist rest gold + universal |
| `outputs/wrist_rest_octet_universal_dual/` | Octet wrist rest, trim only (pre–loose-external) |
| `outputs/wrist_rest_octet_box_vf_025/` | Octet, octant VF ≥ 0.25 |
| `outputs/wrist_rest_octet_box_vf_050/` | Octet, octant VF ≥ 0.50 |
| `outputs/wrist_rest_octet_box_vf_050_deformed/` | Same graph; volume outside snap + dual on CAD |

Older matrix STLs under `test_parts/mouse_wrist_rest_nodal_conformation/`:

| Folder | Contents |
|--------|----------|
| `cull_compare/` | Unculled vs culled Cartesian cores (pre–box occupancy) |
| `trimmed_undeformed/` | First node-plane trim (AABB only) |
| `trimmed_box_occupancy/` | Plane trim + box occupancy; undeformed and closest-deformed cores |
| `kelvin_tesseract_trim/` | Kelvin 0/0.5/1 cuts; tesseract no-0.75-stop |
| `surface_dual_deformed/` | Dual only, after deformed cores (`export_wrist_rest_surface_dual_deformed.py`) |
| root `*.stl` | Older combined core+dual matrix (pre-trim generalization) |

Scripts: `scripts/export_wrist_rest_*.py`. Review solids: 0.6 mm cylinders, **no** stress decon / wick, **no** Boolean unless noted. Dual solids: rectangular bars, 3 mm turn radius, crease relax off, hubs on.

---

## Code

| Piece | Path |
|-------|------|
| Generate (volume + dual) | `graphite/explicit/nodal_conformation.py` |
| Universal dual (layered roles) | `graphite/explicit/sc_role_surface_dual.py`. **Handoff:** [UNIVERSAL_DUAL_HANDOFF.md](UNIVERSAL_DUAL_HANDOFF.md). Review: `scripts/export_layered_dual_review.py`, `export_wrist_rest_gold_vs_universal_dual.py`, `export_wrist_rest_octet_box_vf_deformed.py` |
| Cull / trim / box occupancy | `graphite/explicit/sc_node_plane_trim.py` |
| Planes, kelvin/tesseract specials | `graphite/explicit/sc_node_planes.py` (`trim_plane_policy`, `kept_trim_planes`) |
| Octahedral geometric dual (legacy working dual) | `graphite/explicit/sc_simple_fc_surface_dual.py` |
| Tests | `tests/test_sc_node_planes.py`, `tests/test_sc_node_plane_trim.py`, `tests/test_sc_role_surface_dual.py` |
