# PAM Blindspots — Implementation Plan (v2, revised after re-review)

> [!IMPORTANT]
> **Handoff document for an implementing model.** Execute **one phase at a time**, then stop with reviewable outputs under `outputs/` (AGENTS.md rule). Do not start the next phase without user approval.

---

## 0. Context & Honest Status (read first)

### What exists
- Multi-zone assembly: `graphite/explicit/interlinked/assemble.py` (`ZoneSpec`, `InterfaceSpec`, `assemble`, modes `discrete` / `continuous`).
- Validators: `graphite/explicit/interlinked/validator.py`
  - `check_envelope_penetration_and_catenation` (linked / entangled / hanging)
  - `evaluate_inter_zone_catenation` (cross-zone links + clearance)
  - `evaluate_insufficient_linking` (Phase 5: orphaned boundary particles, `inter>=1 and intra==0`)
  - `verify_two_step_interlinked_quality` (mesh collision + envelope)
- Phase 5 culling: `ZoneSpec.cull_dangling` / `assemble(..., cull_dangling=)` — single pass, removes only degree-0 particles per zone.
- Deterministic sweep: `graphite/explicit/interlinked/search.py` (`search_interface`). Powell retired to `legacy/solver.py`.
- Coupon script: `scripts/generate_phase3_transitions.py` → `outputs/phase3_review/`.

### Known defects (not previously reported correctly)
1. **Coupon 1 ({111} D4-CUBE → D4-TET, continuous mode) has 3 orphaned boundary particles** (IDs 15, 16, 18, all in Zone B, `intra=0`). All 3 of its "boundary links" go to D4-TETs detached from their own lattice. Its "3 links, 200 µm" headline is misleading.
2. **8 failing interlinked tests** (243 passed) in the last full interlinked run:
   - `tests/test_universal_pam_epitaxy.py`:
     `test_polyhedral_face_extraction`, `test_dynamic_search_range_catenation`, `test_generate_c6tt_to_d4tet_transition`, `test_generate_c6tt_to_d4cube_transition`, `test_generate_c6tt_to_j4ring_transition`, `test_solve_ring_ring_epitaxial_transition`
   - `tests/test_void_olattice_and_robust_epitaxy.py`:
     `test_optimize_interface_phase_and_taper_coaxial`, `test_backward_compatible_optimize_interface_catenation` (phase_shift = 2.61, expected 6.0–7.5)
   - Suspected cause: Phase 4 replacement of Powell polish with grid polish in `epitaxy.py`. **Not yet confirmed.**
3. **Root cause of orphans is in seeding, not assembly.** `seed_orthogonal_lattice` (`seeding.py` ~L353) includes sites on *both* faces of the box (`-eps <= x <= max+eps`). For `dia` 1×1×1 that yields 18 sites, not 8; diamond corner sites whose 4 bonds all point outward get zero intra-zone links. Phase 5 culling patches the symptom.
4. **Phase 5 culling is one pass and degree-0 only.** Removing particles can strand others; small disconnected clusters (degree ≥ 1, but not attached to the main lattice) survive.

---

## Global Rules for the Implementer

- Generated files go **only** under `outputs/`. Never write to `test_parts/`.
- Clean miter joints (`clean_miter=True` / `build_clean_miter_truss`). No spheres.
- No numerical optimizers (no Powell / scipy.optimize). Deterministic, bounded sweeps only.
- "Always Generate, Always Diagnose": never raise on geometric mismatch unless `strict=True`.
- **Reuse existing modules before writing new ones** (see file map, §6). Don't add a new module if an existing one covers ≥ 50% of the need.
- If a cell pair cannot interlink, **report an honest failure**. Don't force it (user guidance).
- After any API change: `python scripts/export_core_spec.py --check` must pass. Update `GRAPHITE_CORE_SPEC.md` if not.
- Report **all** test results, including failures, in the phase review. Never report only the passing subset.

### Mandatory linkage gate (every new or regenerated coupon, every phase)
Every generated coupon must report all of the following in its review markdown:
1. `clearance_valid`, global min clearance (µm)
2. `boundary_linked_count` (multi-zone)
3. `evaluate_insufficient_linking(...).is_valid` + orphan count
4. **Connected components of the full catenation graph** (Phase 0 adds this): count + size of largest component
5. `res.mesh.is_watertight`
6. `verify_two_step_interlinked_quality(stl_path, particles)` pass/fail + reason

A coupon is "PASS" only if 1–6 are all clean (single connected component, zero orphans, zero collisions).

---

## Phase 0 — Pay Down Debt (do this first)

**Goal:** Every existing coupon and test is honestly green or honestly diagnosed before new features are added.

### 0.1 Triage the 8 failing tests
- Run: `pytest tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py -x -q`
- For each failure decide, and record in the review:
  - **(a) Regression from Phase 4** (grid polish vs. Powell): fix the grid polish in `epitaxy.py` so it reaches equivalent optima (e.g. widen range / refine step around coarse best). Do not reintroduce Powell.
  - **(b) Test encoded Powell-specific numbers**: if the new result is physically valid (clearance ≥ target, links ≥ 1, verified by the linkage gate), update the assertion to a physical criterion (e.g. `clr >= min_clearance and piercings >= 1`) rather than a magic number. Justify in a comment.
- Use `git stash`/`git diff` on `epitaxy.py` to compare against pre-Phase-4 behavior if needed.

### 0.2 Bond-aware seeding (fix the root cause)
File: `graphite/explicit/interlinked/seeding.py`
- Add an optional parameter to `seed_orthogonal_lattice(..., prune_unbonded: bool = False)`.
- When `True`, after collecting sites, compute for each site the number of *network* neighbors within the seeded set:
  - `dia`: neighbor distance = `a * sqrt(3)/4`
  - `pcu`: `a`
  - `bcc`: `a * sqrt(3)/2`
  - tolerance: `1e-3 * a`
  - use `scipy.spatial.cKDTree.query_pairs`
- Iteratively remove sites with 0 network neighbors until stable.
- Keep only the **largest connected component** of the site-bond graph.
- Default stays `False` (back-compat). `assemble()` passes `prune_unbonded=True` when `cull_dangling` is set.

### 0.3 Make particle culling iterative + component-based
File: `graphite/explicit/interlinked/assemble.py`
- Replace the single-pass degree-0 cull (both continuous and discrete branches) with a helper:
  ```python
  def _cull_to_largest_component(parts, wire_radius) -> list[InterlinkedParticle]:
      # build graph from check_envelope_penetration_and_catenation(parts).linked_pairs
      # keep largest connected component (scipy.sparse.csgraph.connected_components)
  ```
- Applies per zone, before interface transforms (current position of the cull). Note: culling may change `z_cur_min`, which is fine because gap placement uses the surviving set.

### 0.4 Whole-assembly connectivity check
File: `graphite/explicit/interlinked/validator.py`
- Add `evaluate_assembly_connectivity(particles, wire_radius) -> ConnectivityReport` with fields: `num_components`, `component_sizes` (sorted desc), `largest_fraction`, `is_single_component`, `component_of` (particle → component id).
- Reuse `check_envelope_penetration_and_catenation(...).linked_pairs`. **Don't recompute linking** (it's the slow step). Better: let `evaluate_insufficient_linking` and the new function accept a precomputed `linked_pairs` to avoid 3× recomputation inside `assemble()`.
- In `assemble()`: store `metadata["connectivity_report"]`, `metadata["num_components"]`. Under `strict=True` with `cull_dangling`, raise if `num_components > 1`.
- Export from `__init__.py` (`ConnectivityReport`, `evaluate_assembly_connectivity`).

### 0.5 Honest re-run of all three Phase 3 coupons
- Update `scripts/generate_phase3_transitions.py`:
  - Coupon 1: run once without and once with `cull_dangling=True`. Report both. If culling removes all boundary links, **Coupon 1 is reported as FAIL (not interlinked)** — don't tune it to pass.
  - Coupon 2: unchanged (diagnosed unlinked).
  - Coupon 3: `cull_dangling=True` (already set). Confirm single component.
- Report markdown must include the full linkage gate table for every coupon.

### 0.6 Tests
- `tests/test_assemble.py`:
  - Fix `test_continuous_mode_111_interlinks` to also assert on `is_sufficiently_linked`. If Coupon 1 fails honestly, change the test to assert the diagnosed state, and add `xfail` with a reason only if the user approves.
  - Add `test_connectivity_single_component_c6tt_d4tet`.
- `tests/test_interlinked_seeding.py`: add `test_prune_unbonded_dia_1x1x1` (no site has zero bonded neighbors; one component).

### Phase 0 exit criteria
- Full interlinked battery green (the 19 files in §7), or each remaining failure documented with cause and user sign-off.
- `export_core_spec.py --check` passes.
- Review artifact `phase0_debt_review.md` + regenerated `outputs/phase3_review/` with honest gate tables. **Stop.**

---

## Phase 1 — Connectivity Diagnostics in Existing Outputs (lightweight Blindspot 4)

**Goal:** Make problems visible without building a new HUD.

- Statuses in `assemble()`: add `Zone_k_detached` for particles not in the largest component (after `orphaned`, before `hanging` in precedence).
- `writer_3mf.py` / `DEFAULT_STATUS_COLORS`: add colors for `*_orphaned` (yellow `#FACC15`) and `*_detached` (orange `#FB923C`). Check `resolve_particle_color` handles the `Zone_k_` prefix pattern.
- `scripts/generate_phase3_transitions.py` `render_assembly_pyvista`: optionally draw the catenation graph as thin lines between particle centers (inter-zone links in green, intra in grey). Reuse `graphite/viz/` framing helpers if applicable.
- **Don't** build a Trame / interactive HUD here. If wanted later, it belongs in `graphite/ui/` (per AGENTS.md module map), as a separate approved phase.
- Tests: 3MF color group contains orphan/detached colors when present.
- Exit: re-rendered coupons with visible orphan/detached coloring + review. **Stop.**

---

## Phase 2 — Curved Boundary: Cylindrical Multi-Zone (Blindspot 3, scoped)

**Goal:** Core/sleeve cylinder (e.g. C-6-TT core, D-4-TET sleeve) with honest per-sector diagnosis.

### Key physics the old plan missed
A cylindrical interface through a cubic lattice has **no single habit plane**. The local interface normal rotates through all {hk0} orientations around the circumference, so a gap and twist that interlink at θ = 0° won't generally work at θ = 22.5° or 45°. Expect sector-dependent success.

### Implementation (reuse first)
- Build both zones as 3D lattices clipped with `ZoneSpec.region` predicates (`r < R_core` / `R_core + gap <= r < R_outer`). Use `cull_particles_by_sdf` from `seeding.py` if it fits better. **Don't** use `seed_cylindrical_wrap` here (it's a 2D surface seeder).
- Extend `assemble()` discrete placement: when `InterfaceSpec.normal == "radial"` (new option), skip the planar `delta_n` translation; zones share the global frame and are separated by their region predicates.
- New diagnostic in `validator.py`: `evaluate_sector_linking(particles, axis, n_sectors=8)` → per-sector counts of boundary links, orphans, and min clearance.
- Script: `scripts/generate_cylindrical_transition.py` → `outputs/phase2_cylinder/` (STL, 3MF, render, report with a per-sector table).

### Out of scope (document as future work, don't implement)
- Spherical multi-zone shells: a sphere can't be tiled without topological defects (≥ 12 five-fold sites). Fibonacci seeding gives irregular neighbor spacing, so consistent interlinking isn't a seeding problem.
- Arbitrary CAD SDF interfaces: same rotating-normal issue, but worse.

### Exit
Cylinder coupon with linkage gate table + sector table. Sectors that don't link are reported as such. **Stop.**

---

## Phase 3 — Buffer Layers (Blindspot 2, audit-first)

**Goal:** Determine what the existing morph/buffer code can already do before writing anything.

### Step 3.1 — Audit (write-up only, no new code)
Read and document capabilities/limits of:
- `graphite/explicit/interlinked/epitaxial_morph.py` (or wherever `EpitaxialBasisMorpher`, `EpitaxialMorphConfig`, `generate_epitaxial_morph_lattice`, `build_morphing_tetrahedron_geometry` live — locate via `__init__.py` imports)
- `superlattice.py`: `find_buffer_path`, `build_auto_buffer_superlattice`, `generate_epitaxial_superlattice`
- Existing tests: `tests/test_epitaxial_morph.py`, `tests/test_void_olattice_and_robust_epitaxy.py::TestStrategy3DiscreteBufferSuperlattices`

### Constraints the old plan missed
- Node interpolation `x(λ) = (1-λ)x_A + λx_B` needs **identical topology** (same node/strut count and connectivity). D4-CUBE (8 nodes / 12 struts) vs D4-TET (4 / 6) aren't compatible. Bipartite matching doesn't produce valid closed cages.
- Every buffer layer adds **another interface** that must pass the linkage gate on both sides.
- **D4-CUBE ↔ D4-TET on {100} is an accepted failure** (user guidance). Don't target it.

### Step 3.2 — Only if the audit shows a gap
- Topology-compatible pairs only (same cell family, different size/taper, e.g. D4-TET at pitch a₁ → D4-TET at pitch a₂, or the taper/shrink already in `apply_strut_taper` / `apply_cage_shrinkage`).
- Wire into `assemble()` via existing modules; no new `morphing.py` unless the audit justifies it.

### Exit
`phase3_buffer_audit.md`, plus a coupon only if 3.2 was approved. **Stop.**

---

## Phase 4 — Incommensurate Pitches / CSL (Blindspot 1, DEFERRED)

**Status:** Deferred until a real design needs pitch ratios that don't match. Current coupons use commensurate pitches (e.g. `a_dia = √2·a_sc`).

### Corrections to the old plan (for when this is picked up)
- **Σ is not `n₁ₓ·n₁ᵧ`.** For one lattice, Σ = (CSL cell area) / (primitive cell area). For two *different* lattices it's a **near-coincidence** problem: find integer supercells (m₁, m₂) with `|m₁a₁ − m₂a₂| / (m₁a₁)` small.
- Σ5/13/17 twist boundaries are same-lattice rotations. They don't validate hetero-lattice matching.
- **Rigid particles don't strain.** The right acceptance metric is cumulative misregistration across the coupon width `W`: `δ_max = W · |m₁a₁ − m₂a₂| / (m₁a₁)` must be ≤ a fraction of the clearance slack (e.g. ≤ 50% of `min_clearance_measured − min_clearance_target`), checked by the linkage gate at the coupon edges.
- Reuse `solve_supercell_coincidence` in `epitaxy.py` (Phase 1 of the earlier universal epitaxy work) before writing `csl.py`.

---

## 5. Additional Blindspots to Track (no phase assigned yet)

| Item | Risk | Suggested check |
|---|---|---|
| Linking-number thresholds | `face_cycle_linking` rounds at 0.70 / 0.85; noisy near tangency → false positives/negatives | Unit test with known linked/unlinked loop pairs at small clearance |
| Envelope pre-filter | Convex-hull envelope overlap gate could skip real links on non-convex cells (rings, custom) | Compare against an unfiltered linking pass on a small coupon |
| Linking recomputation cost | `assemble()` calls envelope/link evaluation ~3× | Pass precomputed `linked_pairs` (done in Phase 0.4) |
| Outer-face dangling | Orphans/detached particles at outer zone faces, not just at the interface | Covered by connectivity report (Phase 0.4) |
| Printability | Trapped powder / support access in dense interlinks | Out of scope; flag for user |

---

## 6. Existing Modules to Reuse (don't duplicate)

| Need | Existing symbol | Location (via `interlinked/__init__.py`) |
|---|---|---|
| Lattice seeding (habit planes) | `seed_orthogonal_lattice`, `orthogonal_supercell` | `seeding.py` |
| Cylinder / sphere surface seeding | `seed_cylindrical_wrap`, `seed_spherical_shell` | `seeding.py` |
| SDF / mesh culling | `cull_particles_by_sdf`, `cull_particles_by_mesh` | `seeding.py` (exported) |
| Perimeter frame | `build_perimeter_frame_solid`, `fuse_perimeter_frame` | exported in `__init__` |
| Morphing | `EpitaxialBasisMorpher`, `generate_epitaxial_morph_lattice`, `build_morphing_tetrahedron_geometry` | exported in `__init__` |
| Buffers | `find_buffer_path`, `build_auto_buffer_superlattice`, `generate_epitaxial_superlattice` | `superlattice.py` |
| Taper / shrink | `apply_strut_taper`, `apply_cage_shrinkage` | `epitaxy.py` |
| Supercell coincidence | `solve_supercell_coincidence` | `epitaxy.py` |
| Status colors / 3MF | `DEFAULT_STATUS_COLORS`, `resolve_particle_color`, `export_interlinked_3mf` | `writer_3mf.py` |
| Interactive UI (future) | Trame Studio | `graphite/ui/` |

---

## 7. Test Battery (run in full every phase; report all results)

```bash
pytest tests/test_assemble.py tests/test_c6_d4_epitaxial_transition.py tests/test_interlinked_search.py tests/test_interlinked_3mf.py tests/test_interlinked_cells.py tests/test_interlinked_clearance.py tests/test_interlinked_core.py tests/test_interlinked_lattice.py tests/test_interlinked_pam_integration.py tests/test_interlinked_seeding.py tests/test_pam_c6tt.py tests/test_pam_catenation.py tests/test_pam_d4tet.py tests/test_pam_dimensioning.py tests/test_pam_polyhedra.py tests/test_pam_quality_gates.py tests/test_pam_workbench.py tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py
python scripts/export_core_spec.py --check
```

> [!NOTE]
> The full battery takes ~13 minutes. Run it in the background. `pytest tests/` (whole tree) fails at collection on unrelated legacy modules (`src`, `Supercell_Modules`, `torch`, `test_torture_suite`). That's pre-existing and out of scope. Use the explicit file list above.

---

## 8. Phase Summary

| Phase | Scope | New files | Deliverable |
|---|---|---|---|
| **0** | Debt: 8 tests, bond-aware seeding, component culling, connectivity check, honest coupon re-run | none (edits only) | `phase0_debt_review.md`, regenerated `outputs/phase3_review/` |
| **1** | Orphan/detached statuses, colors, graph overlay in renders | none | Re-rendered coupons + review |
| **2** | Cylindrical core/sleeve with per-sector diagnosis | `scripts/generate_cylindrical_transition.py` | `outputs/phase2_cylinder/` |
| **3** | Audit existing morph/buffer code; build only if gap is proven | audit doc | `phase3_buffer_audit.md` |
| **4** | CSL / incommensurate pitches | deferred | — |
