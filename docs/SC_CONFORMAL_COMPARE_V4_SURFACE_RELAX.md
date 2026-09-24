# V4 — Closest Morph + Tangent Surface Relaxation (Experimental)

**Agent role:** Package V1 closest-point morph with **tangent-constrained surface
node relaxation** and sharp-feature pinning as comparison version 4.  
**Parent index:** [SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md)  
**Design notes:** [SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md](SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md)

---

## Thesis

Keep V1’s VF cull → closest-point ironing → stair-step normal gate, then **even
out iron-node spacing on the CAD** before the interior Laplacian runs. Iron
nodes near sharp CAD dihedral edges stay pinned so relaxation cannot undo the
stair-step gate by sliding a wall node onto the floor.

---

## Algorithm (normative)

1. Same as V1 through projection, stair-step gate, residual outside snap.
2. `relax_surface_nodes_tangent` on iron nodes:
   - Skin adjacency from exposed faces.
   - Pin nodes within `0.15 * min(cell)` of a CAD sharp edge (dihedral ≥ 60°).
   - Free nodes: **edge-length springs** toward the median skin edge length
     (not position Laplacian — that shrinks open skins); keep only the
     tangential component (CAD normal); re-snap with closest-point; reject
     steps that flip the CAD normal (patch jump).
   - Cap cumulative travel at `0.35 * min(cell)`.
3. Interior Laplacian relax (same as V1).
4. Stamp + solidify + export.

Defaults: `surface_relax_iterations=25`, `surface_relax_alpha=0.5`.

---

## Known limitations

- Crease *sliding* (Idea 2 role=1) is not implemented; feature-proximal nodes
  are fully pinned. Spacing near true CAD creases will not redistribute.
- Does not replace V2/V3 dual architectures; it only cleans V1 skin spacing.
- Travel clamp / short insides from V1 remain.

---

## Code ownership

| Path | Purpose |
|---|---|
| `graphite/explicit/compare_v4_surface_relax/` | `generate_compare_v4(...)` |
| `scripts/compare_sc_v4_wrist_rest.py` | Wrist-rest export |
| `tests/compare_v4/` | Unit + smoke |
| Shared (opt-in): `conformal_core.relax_surface_nodes_tangent` etc. | Default off in morph |

**Do not** change V2/V3 packages. Do not turn surface relax on by default in V1.

---

## Deliverables

- `test_parts/mouse_wrist_rest_grid/compare_v4_surface_relax_grid_6x6x4_untrimmed.*`
- (Optional) matching 12×12×4 for fair compare with V1–V3 defaults
- Import: `from graphite.explicit.compare_v4_surface_relax import generate_compare_v4`

---

## Acceptance

- Report includes `skin_edge_cv_before` / `skin_edge_cv_after`, free/pinned counts.
- `projection_mode="closest"` end-to-end; stair gate on.
- Smoke test passes; wrist-rest 6×6×4 exports without inverted hexes.
