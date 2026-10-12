# PAM Heteroepitaxy & Multi-Zone Transitions — Fresh Agent Handoff

**Target Audience:** Fresh AI Agent / New Pair Programmer  
**Location:** `graphite.explicit.interlinked`  
**Date:** October 2026  
**Status:** Phase 0 Debt Cleared (100% Green); Ready for Phase 1 Diagnostics & Phase 2 Cylindrical Transitions  

---

## 1. Executive Summary & Where We Are

You are picking up Graphite's **Polycatenated Architected Materials (PAMs)** engine. 
Graphite generates 3D printing-ready, interlinked metamaterials (chainmail, polycatenated polyhedral frames) and stitches different unit cell types across multi-zone boundaries without chemical bonds (heteroepitaxy).

### Current Baseline:
- **Phase 0 ("Pay Down Debt") is 100% COMPLETE.**
  - All 44 universal epitaxy tests are passing (`tests/test_universal_pam_epitaxy.py`, `tests/test_void_olattice_and_robust_epitaxy.py`).
  - SVD planarity checks and chordless 6-cycle extraction are in `graphite/explicit/interlinked/epitaxy.py`.
  - Bond-aware seeding (`prune_unbonded=True`) in `seeding.py` eliminates artificial corner sites.
  - Whole-assembly connectivity reporting (`evaluate_assembly_connectivity`, `ConnectivityReport`) is live in `validator.py`.
  - Component-based culling (`_cull_to_largest_component`) is live in `assemble.py`.
  - Core specification passes: `python scripts/export_core_spec.py --check`.

---

## 2. The Strict Rules (Do Not Break)

1. **Clean Miter Standard:** All truss lattices must use `clean_miter=True` / `build_clean_miter_truss`. Never use spherical bulges (`add_spheres=True`).
2. **Permissive Assembly ("Always Generate, Always Diagnose"):** Never raise exceptions on unlinked or colliding interfaces unless `strict=True`. Generate the geometry, then return authoritative diagnostics in `metadata`.
3. **No Continuous Optimizers:** Legacy SciPy Powell loops are retired to `interlinked/legacy/solver.py`. All interface tuning uses bounded grid sweeps via `graphite.explicit.interlinked.search.search_interface`.
4. **Outputs Rule:** Generated meshes and reports go **only** under `outputs/`. Never write to `test_parts/`.
5. **One Phase at a Time:** Deliver one reviewable phase with working outputs, then stop and await user review.
6. **Core Spec Check:** After any API or dataclass modification, always verify `python scripts/export_core_spec.py --check`.

---

## 3. What Works vs. What Fails (The Honest Ground Truth)

Do not trust old notes claiming all cell combinations link. Here is the verified truth from Phase 0:

| Transition Pair | Habit Plane | Reality | What the Tests / Gates Confirm |
|---|:---:|:---:|---|
| **C-6-TT $\leftrightarrow$ D-4-TET** | `{100}` | **PASS (Benchmark)** | **Working Gold Standard.** 3 boundary links, $191.4\,\mu\text{m}$ clearance, single connected component (`num_components == 1`). Watertight solid mesh. Follows Zhou et al. (*Science* 2025). |
| **D4-CUBE $\leftrightarrow$ D4-TET** | `{100}` | **FAIL (Incompatible)** | Collides ($< 0$ clearance). Tetrahedral bond angles ($54.7^\circ$) cannot penetrate square cube faces without interference. Diagnosed honestly as `FAIL`. |
| **D4-CUBE $\leftrightarrow$ D4-TET** | `{111}` | **FAIL (Unlinked under culling)** | Without culling, it shows 3 boundary links, but **all 3 are attached to orphaned boundary tets** that have zero bonds into the rest of their own lattice (`intra_degree == 0`). Under `cull_dangling=True`, boundary links drop to **0**. Diagnosed honestly as `FAIL`. |

---

## 4. Critical Crystallographic Pitfalls (Read Before Coding!)

Two major pitfalls were uncovered during recent testing. **Do not repeat them:**

### Pitfall A: Orthogonal Supercell Stacking Height Mismatch
- On diamond along $\{111\}$, the orthogonal supercell has vertical period $H_Z = \sqrt{3} \cdot a \approx 17.32\text{ mm}$ (for $a = 10\text{ mm}$).
- Because diamond stacks in an $ABCABC$ sequence, **one single orthogonal supercell repeat contains 3 cubic unit cells (6 diamond layers)**.
- If a user asks for "3 cells tall", passing `grid_size=(3, 3, 3)` creates 3 supercells tall = **9 cubic unit cells tall (18–19 layers, $\approx 104\text{ mm}$)**!
- **Rule:** If the user asks for $N$ unit cells along $\{111\}$, 1 orthogonal supercell repeat already provides 3 unit cells.

### Pitfall B: The Global Twist Disaster (Breaking Bulk Lattice Symmetry)
- Diamond along $[1, 1, 1]$ has **3-fold ($C_3$) symmetry, NOT 6-fold ($C_6$) symmetry**.
- Each site has 1 vertical bond along $+Z$ and 3 transverse bonds pointing into the $XY$ plane at $120^\circ$ angles.
- If you apply `twist_deg = 60.0` as a global rotation to all particles in Zone B without transforming their site coordinates:
  - The vertical bonds stay connected.
  - But the 3 transverse bonds point directly into empty space between sites.
  - **All transverse bonds across the entire $XY$ plane are severed!** Zone B fractures into 105 disconnected vertical pairs (dimers), leaving huge open gaps across the lattice.
- **Rule:** `normal` is a **global 3D tilt** (rotates both sites and particles together, keeping all neighbor bonds intact). `twist_deg` is an **interface adapter parameter** at the seam—never spin bulk lattice particles locally!

---

## 5. Active Implementation Roadmap

The master plan is stored in [`docs/plans/PAM_BLINDSPOTS_IMPLEMENTATION_PLAN.md`](plans/PAM_BLINDSPOTS_IMPLEMENTATION_PLAN.md).

### Next Phase: Phase 1 — Connectivity Diagnostics in Existing Outputs
1. **Status Tagging:** Ensure `assemble()` tags particles not in the largest component as `f"Zone_{k}_detached"`.
2. **Color Palette:** In `graphite/explicit/interlinked/writer_3mf.py`, support:
   - `*_orphaned` $\to$ Yellow (`#FACC15`)
   - `*_detached` $\to$ Orange (`#FB923C`)
3. **Graph Overlay in Renders:** In `scripts/generate_phase3_transitions.py` / PyVista visualizer, render catenation graph edges between particle centers (green for inter-zone links, grey for intra-zone links).
4. **Deliverable:** Re-run `scripts/generate_phase3_transitions.py` and inspect `outputs/phase3_review/`.

### Follow-up Phase: Phase 2 — Cylindrical Multi-Zone Transitions
1. Core/sleeve cylindrical transition (e.g. C-6-TT core, D-4-TET sleeve).
2. Clip lattices via `ZoneSpec.region` predicates (`r < R_core` vs `r >= R_core + gap`).
3. Note: Cylindrical normal rotates through all $\{hk0\}$ orientations; expect sector-dependent linking.
4. Output: `scripts/generate_cylindrical_transition.py` $\to$ `outputs/phase2_cylinder/`.

---

## 6. Key Code References & Quick Verification

### Core Modules:
- [`graphite/explicit/interlinked/assemble.py`](../graphite/explicit/interlinked/assemble.py): Multi-zone placement, continuous & discrete modes, component culling.
- [`graphite/explicit/interlinked/seeding.py`](../graphite/explicit/interlinked/seeding.py): Habit plane supercells (`{100}`, `{110}`, `{111}`), `prune_unbonded`.
- [`graphite/explicit/interlinked/validator.py`](../graphite/explicit/interlinked/validator.py): `evaluate_assembly_connectivity`, `evaluate_insufficient_linking`.
- [`graphite/explicit/interlinked/epitaxy.py`](../graphite/explicit/interlinked/epitaxy.py): Chordless cycle extraction (3, 4, 6 cycles), aperture geometry, Gauss linking number.
- [`graphite/explicit/interlinked/search.py`](../graphite/explicit/interlinked/search.py): Deterministic bounded interface parameter sweep.

### Fast Verification Commands:
```powershell
# 1. Check architectural specification compliance
python scripts/export_core_spec.py --check

# 2. Run universal epitaxy suites (44 tests, ~17s)
pytest tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py -q

# 3. Run assembly and seeding suites (24 tests, ~6s)
pytest tests/test_assemble.py tests/test_interlinked_seeding.py -q

# 4. Run search and C6-TT benchmark transition (9 tests, ~8s)
pytest tests/test_interlinked_search.py tests/test_c6_d4_epitaxial_transition.py -q
```
