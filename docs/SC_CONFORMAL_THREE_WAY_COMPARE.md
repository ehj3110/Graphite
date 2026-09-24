# SC Conformal — Three-Way Comparison (Agent Index)

> **Doc role:** Active compare map. Agents: [AGENTS.md](../AGENTS.md) → [graphite/explicit/README.md](../graphite/explicit/README.md), then open **one** version handoff below — not all.

**Status:** comparison program (Aug 2026)  
**Stress CAD:** `test_parts/Mouse wrist rest v1.stl`  
**Shared cell / strut (default compare):** `cell_size=(12,12,4)`, `rule_name="grid"` and/or `"octahedral"`, strut Ø `1.2` mm, VF `0.5` where applicable.

We compare **three independent algorithms** for conformal SC lattices. Each has its own handoff doc and **isolated code ownership** so agents do not edit the same files.

| Version | Name | Handoff doc | Code ownership (agents MUST stay here) |
|---|---|---|---|
| **V1** | Closest-point morph (baseline) | [SC_CONFORMAL_COMPARE_V1_CLOSEST.md](SC_CONFORMAL_COMPARE_V1_CLOSEST.md) | `graphite/explicit/compare_v1_closest/` · `scripts/compare_sc_v1_*.py` · `tests/compare_v1/` |
| **V2** | Surface-first dual + trim + stitch | [SC_CONFORMAL_COMPARE_V2_SURFACE_FIRST.md](SC_CONFORMAL_COMPARE_V2_SURFACE_FIRST.md) · detail [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md) | `graphite/explicit/compare_v2_surface_first/` · `scripts/compare_sc_v2_*.py` · `tests/compare_v2/` |
| **V2.1** | V2 evolved: hardened cuts, exterior-corner dual, nearest stitch | [SC_CONFORMAL_COMPARE_V2_1_PLAN.md](SC_CONFORMAL_COMPARE_V2_1_PLAN.md) | `graphite/explicit/compare_v2_1_surface_first/` · `scripts/compare_sc_v2_1_*.py` · `tests/compare_v2_1/` |
| **V3** | Hybrid: intersect-all dual shrinkwrap + VF volume + morph-to-dual | [SC_CONFORMAL_COMPARE_V3_HYBRID.md](SC_CONFORMAL_COMPARE_V3_HYBRID.md) | `graphite/explicit/compare_v3_hybrid/` · `scripts/compare_sc_v3_*.py` · `tests/compare_v3/` |
| **V4** | V1 + tangent surface relax (experimental) | [SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md](SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md) | `graphite/explicit/compare_v4_surface_relax/` · `scripts/compare_sc_v4_*.py` · `tests/compare_v4/` |

**Shared read-only dependencies (do not rewrite for one version):**  
`conformal_core.cull_hex_elements`, `find_exposed_faces`, `generate_hex_topology`, `generate_rectangular_surface_cage`, geometry Boolean helpers, existing tests outside `tests/compare_v*`.

**Shared compare outputs:** `test_parts/mouse_wrist_rest_grid/compare_v{1,2,3}_*/`

### Implementation status (Aug 2026)

| Version | Status | Primary artifacts |
|---|---|---|
| V1 | Done | `compare_v1_closest_grid_12x12x4_untrimmed.*` (+ `6x6x4`) |
| V2 | Done | `compare_v2_surface_first_grid_12x12x4_{untrimmed,trimmed,dual_cage}.*` |
| V2.1 | In progress | `compare_v2_1_surface_first_grid_*` |
| V3 | Done | `compare_v3_hybrid_{grid,octahedral}_12x12x4_untrimmed.*` |
| V4 | Experimental | `compare_v4_surface_relax_grid_{6x6x4,12x12x4}_untrimmed.*` |

Scripts: `scripts/compare_sc_v{1,2,3,4}_wrist_rest.py`

---

## One-line theses

- **V1:** Keep today’s morph: VF 50% cage → project exposed corners to **nearest CAD point** → relax → stamp → optional dual from morphed skin.
- **V2:** Dual is the skin authority on CAD; volume stays Cartesian and is **trimmed**; gated **stitch** reconnects cut nodes (distance ≤ 1 cell horizontal, ≤ 60° from vertical, else orphan cull).
- **V3:** Build a **loose** dual from “any intersection” hexes and shrinkwrap it to CAD; build **strict** VF 50% volume; morph volume boundary to **nearest dual node** (lateral OK); cap dual-node valence from internal struts at **per-element rule valency** (e.g. octahedral node = 4).
- **V4:** V1 + stair-step gate + **tangent surface relaxation** (sharp CAD features pinned) to even skin node spacing after projection.

---

## Fair comparison checklist (every version)

Export for the wrist rest:

1. Untrimmed lattice solid (volume ± dual ± stitches as applicable).
2. Optional CAD-Boolean trimmed solid.
3. PNG isometric render.
4. `*_report.txt` with: cell size, rule, node/strut counts, outside-node count (`safe_signed_distance`), collapsed-hex warnings, timing.
5. Same `STRUT_RADIUS=0.6` unless the version’s doc says otherwise.

---

## How to point a new agent

1. Open **only** that version’s handoff doc.  
2. Respect **code ownership** paths above.  
3. Prefer wrapping existing `graphite.explicit` APIs over editing `morph_hex_scaffold` in place.  
4. Deliver a runnable `scripts/compare_sc_vN_wrist_rest.py` that writes under `test_parts/mouse_wrist_rest_grid/`.
