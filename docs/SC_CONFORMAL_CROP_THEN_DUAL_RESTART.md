# SC Conformal — Crop Lattice → Connect Surface Dual (Restart)

**Status:** Restart. Prior projection / stair / dual-snap experiments abandoned as a mess.  
**Date:** 2026-08-10  
**Goal:** Crop (clip) an interior Cartesian SC lattice to the CAD, then connect it to a **surface dual** shell. Do not morph the volume to the surface via closest-point or unconstrained projection of volume nodes.

Related code (still in tree; treat as reference, not gospel):

- `graphite/explicit/compare_v2_surface_first/` — clip, dual, stitch, outer-ring dual
- `graphite/explicit/compare_v2_1_surface_first/` — pipeline wrapper
- `scripts/generate_sphere_grid_baselines.py`
- `scripts/generate_wrist_rest_connected.py`

Prior docs (historical): `SC_CONFORMAL_COMPARE_V*.md`, `SC_CONFORMAL_THREE_WAY_COMPARE.md`, `SC_CONFORMAL_FUTURE_WORK.md`.  
Implemented follow-on (wrist rest, Aug 2026): [SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md](SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md).

---

## Restart recipe (intended)

1. **Stamp** a Cartesian SC volume lattice over the CAD bbox (VF cull optional for empty cells).
2. **Crop / clip** that lattice to the CAD solid so only interior (or interior + short surface stubs) remains.
3. **Build a surface dual** independently (exposed faces / outer ring / interface faces — *not* “delete intersecting struts from a full SC pad”).
4. **Connect** cropped volume ↔ dual with spokes/stitches (and only those). Dual owns the skin edges; volume owns interior edges. Do not stack full volume skin + dual + stitches without stripping dual-owned edges from the volume graph.
5. **Project the dual only** onto the CAD (keep policy simple until topology is solid). Avoid axis/diagonal rabbit holes until crop→connect is trustworthy.

---

## What we tried (chronological)

### 1. Closest-point volume morph (V1)

- VF cull → morph volume nodes toward CAD by closest point, with travel clamps / stair gates / residual outside snap / Laplacian relax.
- **Result:** Stair crush, wrong snaps, non-robust on wrist rest. **Abandoned** as the primary path.

### 2. Surface-first pipelines (V2 / V2.1)

- VF cull → Cartesian volume stamp → graph-clip volume to CAD → dual from exposed faces → stitch cut stubs to dual.
- Dual modes: `exposed_corners`, `face_centroid`.
- Projection modes on dual: `closest`, `exposed_normal`, then Blender-style **Target Normal Project (TNP)**.
- Clip used `cut_inset_factor≈0.45` initially → stubs too short / off-chord snaps. Fixed toward **chord raycast** (SDF bisect fallback) and **inset ≈ 0**.

**Useful takeaway:** Separate volume crop from surface dual; connect with stitches. Don’t ask the volume lattice to become the skin.

### 3. Outer-ring dual

- Pad bbox, delete hexes intersecting CAD, keep the next layer of cells, keep only **interface faces**.
- Produced a continuous exterior shell. Shrinkwrap with **Target Normal Project** looked good in Blender.
- User preference: dual from outer-ring / exposed-face logic, **not** “delete intersecting struts from a full SC.”

### 4. Connected topology (“looks like three STLs”)

- Early “connected” exports stacked: full clipped volume (including skin) + dual + stitches → triple skin / confusing mesh.
- **Fix that still matters:** dual owns skin edges; volume keeps interior only; interior↔skin become spokes; weld coincident nodes.

### 5. Sphere baselines

- Scripted stages: culled → trimmed → dual → connected (interior + dual + spokes).
- User asked to stop dumping ~6 stage STLs at once; prefer **one STL at a time** when debugging.

### 6. Dual projection experiments (wrist rest) — the mess

| Mode | Intent | Outcome |
|------|--------|---------|
| `target_normal` (TNP) | Project along CAD face normal | Good conformity; some sidewall nodes can floor-snap if unconstrained |
| Axis-constrained (±X/Y/Z from exposed normals) | Stop floor snap on walls | Sidewalls better; tops/bottoms broken |
| Cell-weighted diagonal for multi-axis “stairs” | Allow stair diagonals | Top stairs pulled **backward** (riser outward normals point into the notch) |
| Z-only when Z present | Stop backward top pull | Helped tops; user then said wall stair steps in the **overlay** (not dual) were fine and asked to revert |

Also explored (partially): stair/hex classification (`simple` / `stair` / `corner`), skip volume snap on stairs, let dual own the skin. Got complicated without a clean win.

**Lesson:** Unconstrained TNP on the dual can work for conformity; axis/diagonal constraints fight stair geometry. Stabilize **crop → dual → connect** before inventing projection rules.

### 7. Other historical branches (earlier compare docs)

- V3 hybrid (loose-cull shrinkwrap dual + strict volume + morph-to-dual).
- V4 surface relax.
- Face-normal / quality-gate / cage-chord isolations on wrist rest.
- Kelvin / octahedral / tesseract / radial / Laplacian sphere variants.

These informed vocabulary and failure modes; none replace the restart recipe above.

---

## Hard lessons (keep)

1. **Don’t stack** volume skin + dual + stitches without stripping dual-owned edges from the volume graph.
2. **Inset clip** systematically shortens stubs; surface cuts want inset ≈ 0 and on-chord hits.
3. **Unconstrained TNP** on dual conforms well but can floor-snap side nodes on flat-bottom parts.
4. **Per-axis rays / notch diagonals** fight stair riser normals; easy to make tops worse.
5. Debug **one mesh at a time** (dual-only vs connected separately).
6. Prefer **outer-ring / exposed-face dual** construction over “delete intersecting struts from full SC.”

---

## Cleanup (2026-08-10)

Deleted generated test STLs under:

- `test_parts/sphere_sc_20mm/`
- `test_parts/mouse_wrist_rest_grid/`

Kept source CAD elsewhere (e.g. `test_parts/Mouse wrist rest v1.stl`). Reports/PNG leftovers may remain; regenerate as needed.

---

## Suggested next implementation focus

Minimal pipeline, no projection gymnastics:

1. Crop Cartesian SC to CAD (clip graph).
2. Build surface dual (outer-ring or exposed-face interface).
3. Connect with spokes only; weld; export **dual** and **connected** as separate single STLs when testing.
4. Optional: project dual with plain TNP once topology looks right; revisit sidewall floor-snap only if it still appears.
