# PAM Metamaterial Heteroepitaxy: Grid Search Audit & Grain Boundary Combinatorics

## 1. Audit of the Initial Grid Search (Phase 3)

In Phase 3, we executed a **Representative Heteroepitaxy Pilot Grid** across 6 selected benchmark systems to validate the autonomous engine (`plan_heteroepitaxial_interface`) and the scoring index ($\mathcal{Q}_{\text{interface}}$).

### What Was Evaluated:
1. **D4-CUBE $\to$ D4-TET on {111}:** Isotypic diamond cubic, matching habit planes ($h_A = \{111\}, h_B = \{111\}$). **$\mathcal{Q} = 0.951$ (PASS)**.
2. **C-6-TT $\to$ B-8-CO on {100}:** Cross-network PCU to BCC, matching habit planes ($h_A = \{100\}, h_B = \{100\}$). **$\mathcal{Q} = 0.868$ (PASS)**.
3. **C-6-TT $\to$ D-4-TET on {100}:** Cross-network PCU to DIA with $45^\circ$ CSL rotation, matching habit planes ($h_A = \{100\}, h_B = \{100\}$). **$\mathcal{Q} = 0.807$ (PASS)**.
4. **D4-CUBE $\to$ D4-TET on {110}:** Zig-zag diamond cleavage plane, matching habit planes ($h_A = \{110\}, h_B = \{110\}$). **$\mathcal{Q} = 0.805$ (PASS)**.
5. **D4-TET $\to$ D4-TET on {111}:** Homotypic diamond without dislocation offset ($h_A = \{111\}, h_B = \{111\}$). **$\mathcal{Q} = 0.000$ (FAIL - Clashes)**.
6. **C-6-TT $\to$ C-6-TT on {100}:** Homotypic simple cubic baseline ($h_A = \{100\}, h_B = \{100\}$). **$\mathcal{Q} = 0.000$ (FAIL - Clashes)**.

### What Was NOT Evaluated:
**We did not evaluate hetero-orientational (cross-plane / grain boundary) combinations.** In all 6 runs above, both Zone A and Zone B were cut along the **same** crystallographic habit plane family ($h_A = h_B$).

---

## 2. Hetero-Orientational (Grain Boundary) Combinatorics

When the two sublattices belong to the same material (e.g., **D4-TET $\to$ D4-TET**) but are cut along different crystallographic planes, the interface is formally a **grain boundary**.

Given the 4 planes of interest:
$$\mathcal{P} = \{ (100), \; (101), \; (110), \; (111) \}$$

The complete pairwise cross-plane matrix consists of exactly $\binom{4}{2} = 6$ combinations:

| Boundary # | Habit Plane Transition | Interplanar Angle | 2D Footprint Coupling | Boundary Type |
|:---:|:---:|:---:|:---:|:---|
| **1** | **(100) $\longleftrightarrow$ (101)** | $45.00^\circ$ | Square ($a \times a$) $\longleftrightarrow$ Centered Rectangular ($a \times a/\sqrt{2}$) | $45^\circ$ Tilt Grain Boundary |
| **2** | **(100) $\longleftrightarrow$ (110)** | $45.00^\circ$ | Square ($a \times a$) $\longleftrightarrow$ Centered Rectangular ($a \times a/\sqrt{2}$) | $45^\circ$ Tilt Grain Boundary (Orthogonal Azimuth) |
| **3** | **(100) $\longleftrightarrow$ (111)** | $54.74^\circ$ | Square ($a \times a$) $\longleftrightarrow$ Hexagonal / Triangular ($a/\sqrt{2} \times a\sqrt{6}/4$) | $54.74^\circ$ Asymmetric Tilt Boundary |
| **4** | **(110) $\longleftrightarrow$ (101)** | $60.00^\circ$ | Centered Rectangular $\longleftrightarrow$ Centered Rectangular | $60^\circ$ Twist / Tilt Boundary (within $\{110\}$ family) |
| **5** | **(110) $\longleftrightarrow$ (111)** | $35.26^\circ$ | Centered Rectangular $\longleftrightarrow$ Hexagonal / Triangular | $35.26^\circ$ Mixed Habit Boundary |
| **6** | **(101) $\longleftrightarrow$ (111)** | $35.26^\circ$ | Centered Rectangular $\longleftrightarrow$ Hexagonal / Triangular | $35.26^\circ$ Mixed Habit Boundary (Conjugated Azimuth) |

---

## 3. Physical & Mathematical Challenges of Cross-Plane Transitions

Why do cross-plane transitions require automated 2D CSL matching?
1. **Dissimilar Surface Footprints:** A $\{100\}$ surface is square ($p_x = p_y = a$). A $\{110\}$ surface is rectangular ($p_x = a, p_y = a/\sqrt{2}$). A $\{111\}$ surface is hexagonal ($p_x = a/\sqrt{2}, p_y = a\sqrt{6}/4$). Simply abutting them creates periodic edge mismatches ("hanging edges") unless supercell multipliers ($M_1, M_2$) are solved to match the total boundary aperture.
2. **Out-of-Plane Void Mismatch:** The interplanar spacing along $\{100\}$ is $a/4$, along $\{110\}$ is $a\sqrt{2}/4$, and along $\{111\}$ is $a\sqrt{3}/12 + a\sqrt{3}/4$. The gap $g^*$ must be optimized to prevent strut clashes while allowing multi-face interpenetration ($N_A \ge 2, N_B \ge 2$).

---

## 4. Documentation Inventory & Sync Status

All files documenting this research, implementation, and catalog are maintained under version control:

| Document / Asset | File Path | Scope & Status |
|:---|:---|:---|
| **Authoritative Interface Catalog** | [`docs/PAM_INTERFACE_CATALOG.md`](PAM_INTERFACE_CATALOG.md) | Official Stage 2 leaderboard, metrics, and models. |
| **Crystallography & Biomimetic Spec** | [`docs/PAM_CRYSTALLOGRAPHIC_AND_BIOMIMETIC_SPEC.md`](PAM_CRYSTALLOGRAPHIC_AND_BIOMIMETIC_SPEC.md) | Analytical glide-set equations, O-lattice algorithms, Q index. |
| **Research AI Handoff Document** | [`docs/PAM_CRYSTALLOGRAPHY_AND_CERAMIC_BIOMIMICRY_RESEARCH_HANDOFF.md`](PAM_CRYSTALLOGRAPHY_AND_CERAMIC_BIOMIMICRY_RESEARCH_HANDOFF.md) | Background handoff detailing the Blender discovery. |
| **Grain Boundary Analysis** | [`docs/PAM_GRAIN_BOUNDARY_AND_GRID_SEARCH_ANALYSIS.md`](PAM_GRAIN_BOUNDARY_AND_GRID_SEARCH_ANALYSIS.md) | This document: combinatorics and cross-plane roadmap. |
| **Phase 1 Completion Report** | `PHASE1_ANALYTICAL_COUPON1A_COMPLETION.md` | Artifact: Coupon 1A glide-set verification. |
| **Phase 2 Completion Report** | `PHASE2_CORE_AUTOMATION_ENGINE_COMPLETION.md` | Artifact: Core automation engine verification. |
| **Phase 3 Completion Report** | `PHASE3_GRID_SEARCH_CATALOG_COMPLETION.md` | Artifact: Stage 2 grid search leaderboard. |
| **Core Architecture Specification** | [`GRAPHITE_CORE_SPEC.md`](../GRAPHITE_CORE_SPEC.md) | External API specification (verified green). |
| **Physical 3D Assets** | `outputs/pam_stage1_coupons/`, `outputs/pam_stage2_catalog/` | Exported .3mf, .stl, and PyVista renders. |
