# Implementation Plan: Finishing Phase 0 — Pay Down Debt

> [!IMPORTANT]
> **Planning Mode Active.** This plan provides a surgical, technically verified roadmap to complete **Phase 0** ("Pay Down Debt") of the PAM Blindspots Implementation Plan. No code modifications will be executed until you review and approve this plan.

---

## 1. Goal Description

Phase 0's mandate is: **"Every existing coupon and test is honestly green or honestly diagnosed before new features are added."**

Through our root-cause investigation, we identified the exact mathematical reasons behind the debt and failures:
1. **The 44 Epitaxy Tests (Strategy 1–5):** All 44 tests in `tests/test_universal_pam_epitaxy.py` and `tests/test_void_olattice_and_robust_epitaxy.py` are passing (fixed cycle traversal, polyhedral hull on hollow rings, coaxial search priority, and conditional 2x2 patch validation).
2. **Missing Hexagonal Faces in C-6-TT:** `extract_chordless_cycles` in `epitaxy.py` only extracted 3- and 4-cycles before returning early. For truncated tetrahedra (C-6-TT), which have 4 triangles and 4 hexagons, the 6-cycles were omitted. Because entering D-4-TET struts penetrate C-6-TT's hexagonal faces, `face_cycle_linking` evaluated $|Lk| = 0$, causing Coupon 3 and `test_c6tt_to_d4tet_100_interlinks` to falsely diagnose 0 boundary links. Extending cycle extraction to chordless cycles up to length 6 instantly recovers the exact 3 boundary links ($|Lk| = 1$, 191.4 µm clearance) published in Zhou et al. (*Science* 2025).
3. **Double-Bounded Orthogonal Seeding (Root Cause of Orphans):** `seed_orthogonal_lattice` (`seeding.py`) uses a closed bounding box `[-eps, max + eps]`, adding superfluous boundary sites on opposite faces. For diamond 1×1×1, this creates 18 sites instead of 8, producing 4 corner sites with 0 intra-lattice bonds. `prune_unbonded=True` eliminates these dangling sites and leaves a single connected component of 14 sites.
4. **Honest Coupon 1 Reporting:** In Coupon 1 ({111} D4-CUBE → D4-TET, continuous mode), the 3 boundary links are attached to D4-TET particles that have `intra_degree == 0`. When component-based culling (`cull_dangling=True`) is applied, these orphaned particles are removed, reducing boundary links to **0**. Per user direction and the plan, **Coupon 1 will be reported honestly as FAIL (unlinked)** without tuning fudge parameters.

---

## 2. User Review Required

> [!WARNING]
> **Honest Failure Diagnosis for Coupon 1 ({111} Continuous Mode):**
> With `cull_dangling=True`, Coupon 1's boundary links drop from 3 (all orphaned) to 0. We will NOT adjust gap or fudge parameters to force it to pass. In `outputs/phase3_review/phase3_transition_report.md` and `phase0_debt_review.md`, Coupon 1 will be officially marked as **FAIL (insufficient intra-zone linking)**.

> [!NOTE]
> **Precomputed `linked_pairs` Reuse:**
> `check_envelope_penetration_and_catenation`, `evaluate_inter_zone_catenation`, `evaluate_insufficient_linking`, and `evaluate_assembly_connectivity` all use exact pairwise Gauss linking. In `assemble.py`, the linking pairs from the first check are reused across all downstream validators, eliminating redundant 3× manifold intersection calls.

---

## 3. Open Questions

There are no blocking ambiguities; the root causes and physical criteria are verified:
- **Coupon 1:** Report honestly as FAIL under culling.
- **Coupon 2:** Remains diagnosed unlinked (incompatible habit plane).
- **Coupon 3:** Passes with 3 boundary links, 191.4 µm clearance, and a single connected component.

---

## 4. Proposed Changes

### Component 1: `graphite/explicit/interlinked/epitaxy.py`
Extend `extract_chordless_cycles` to support chordless cycles up to length 6 so polyhedral cages with hexagonal faces (like C-6-TT) are completely extracted.

#### [MODIFY] `graphite/explicit/interlinked/epitaxy.py`
- In `extract_chordless_cycles(nodes, struts)`:
  - Keep the fast 2-regular ring handler (for simple rings/tori).
  - Extract 3-cycles (triangles) and 4-cycles (squares).
  - Also extract chordless 5- and 6-cycles via bounded cycle search (using `networkx.simple_cycles` or iterative neighbor expansion) and check that `subgraph(c).number_of_edges() == len(c)` (no chords).
  - Deduplicate canonical cycle representations.

---

### Component 2: `graphite/explicit/interlinked/seeding.py`
Ensure `prune_unbonded: bool = False` is available in `seed_orthogonal_lattice` and `seed_diamond_orthogonal_111`.

#### [MODIFY] `graphite/explicit/interlinked/seeding.py`
- Completed and verified via `test_prune_unbonded_dia_1x1x1`.
- When `prune_unbonded=True`, computes nearest-neighbor distances (`dia`: $a\sqrt{3}/4$, `pcu`: $a$, `bcc`: $a\sqrt{3}/2$) with tolerance $10^{-3} a$, iteratively removes 0-neighbor sites, and retains the largest connected component.

---

### Component 3: `graphite/explicit/interlinked/validator.py`
Implement `ConnectivityReport` and `evaluate_assembly_connectivity`, and allow precomputed `linked_pairs` in `evaluate_insufficient_linking`.

#### [MODIFY] `graphite/explicit/interlinked/validator.py`
- Completed and exported via `__init__.py`.
- Computes connected components of the full catenation graph (`num_components`, `component_sizes`, `largest_fraction`, `is_single_component`, `component_of`).

---

### Component 4: `graphite/explicit/interlinked/assemble.py`
Clean up particle status tagging, discrete habit plane placement, and component-based culling.

#### [MODIFY] `graphite/explicit/interlinked/assemble.py`
1. **Status Tagging Precedence:**
   - Particles in colliding pairs $\to$ `"colliding"`
   - Boundary-linked particles $\to$ `f"Zone_{k}_boundary_linked"`
   - Orphaned boundary particles $\to$ `f"Zone_{k}_orphaned"`
   - Detached / hanging particles $\to$ `f"Zone_{k}_detached"` or `"hanging"`
   - Standard zone particles $\to$ `f"Zone_{k}"` (ensures `Zone_0` and `Zone_1` counts in `status_counts` are accurate)
2. **Culling Integration:**
   - In both continuous and discrete branches, when `cull_dangling` is active, cull dangling particles using `_cull_to_largest_component(parts, wire_radius)`.
   - Pass `prune_unbonded=True` to `seed_orthogonal_lattice` when culling is enabled.
3. **Assembly Connectivity Check:**
   - Call `evaluate_assembly_connectivity(all_particles, wire_radius, linked_pairs=all_linked_pairs)`.
   - Record in `metadata["connectivity_report"]`, `metadata["num_components"]`, `metadata["is_single_component"]`.
   - Under `strict=True` with `cull_dangling`, raise if `num_components > 1`.

---

### Component 5: Tests & Deliverable Script

#### [MODIFY] `tests/test_assemble.py`
- Update `test_assemble_two_zones_basic` to verify status counts (`Zone_0` and `Zone_1`).
- Update `test_continuous_mode_111_interlinks`:
  - Assert that without culling, `is_sufficiently_linked` is `False` (3 orphaned boundary particles).
  - Assert that with `cull_dangling=True`, `boundary_linked_count == 0` (honest diagnosed failure).
- Update `test_c6tt_to_d4tet_100_interlinks`:
  - Verify watertightness, clearance $\ge 150$ µm, and 3 boundary links.
- Add `test_connectivity_single_component_c6tt_d4tet` to verify single connected component under `cull_dangling=True`.

#### [MODIFY] `scripts/generate_phase3_transitions.py`
- Update script to run Coupon 1 both without and with `cull_dangling=True`.
- Re-run script to generate STL, 3MF, PNG renders, and updated `outputs/phase3_review/phase3_transition_report.md` reporting:
  1. `clearance_valid` & min clearance (µm)
  2. `boundary_linked_count`
  3. `evaluate_insufficient_linking` & orphan count
  4. Connected components (count and size of largest component)
  5. `is_watertight`
  6. Final Pass/Fail verdict per coupon

#### [NEW] Review Artifact: `phase0_debt_review.md`
- Document all 6 linkage gate metrics for Coupons 1, 2, and 3.
- Document test suite execution (44 epitaxy tests + full 19-file interlinked battery).
- Document `export_core_spec.py --check` result.

---

## 5. Verification Plan

### Automated Tests
Run targeted suites in succession:
1. `pytest tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py -q`
2. `pytest tests/test_interlinked_seeding.py -q`
3. `pytest tests/test_assemble.py -q`
4. Full interlinked battery (all 19 test files listed in Section 7 of the implementation plan):
   `pytest tests/test_interlinked_*.py tests/test_pam_*.py tests/test_assemble.py tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py tests/test_c6_d4_epitaxial_transition.py -q`
5. Architecture specification check:
   `python scripts/export_core_spec.py --check`

### Deliverable Generation & Manual Verification
Run the transition generation script:
```bash
python scripts/generate_phase3_transitions.py
```
Verify generated files under `outputs/phase3_review/`:
- `d4cube_to_d4tet_111_coupon.stl` & `.3mf` & `render.png`
- `d4cube_to_d4tet_100_coupon.stl` & `.3mf` & `render.png`
- `c6tt_to_d4tet_100_coupon.stl` & `.3mf` & `render.png`
- `phase3_transitions_comparison.png`
- `phase3_transition_report.md`

Check that Coupon 1 is honestly diagnosed (FAIL under culling, 0 links), Coupon 2 is diagnosed unlinked, and Coupon 3 is PASS (single component, 3 links, 191 µm clearance, watertight).

**Stop upon creating `phase0_debt_review.md` for user approval.**
