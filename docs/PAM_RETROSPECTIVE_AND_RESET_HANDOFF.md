# PAM Development Retrospective, Ground-Truth Reset & Fresh Agent Handoff

**Scope:** `graphite.explicit.interlinked`  
**Date:** October 2026  
**Status:** Architecture Reset & Clean Ground-Truth Handoff  

---

## 1. Full Development History: What Worked, Why We Moved On, and What Failed

### Phase 1: The Two-Cell Experimental Workbench
- **What it was:** A fast diagnostic tool (`graphite.explicit.interlinked.workbench`) that placed **only two single particles** (e.g. one D4-CUBE and one D4-TET) in empty space and evaluated their clearance, clash volume, and Gauss linking number $|Lk|$ in under 100 ms across gap and twist parameters.
- **What worked:**
  - Proved that deterministic geometric sweeps could completely replace hanging SciPy numerical solvers (Powell/SLSQP).
  - Empirically diagnosed the **window aspect ratio mismatch** between D4-CUBE (4-fold square windows, 90° corners) and D4-TET (3-fold triangular windows, 60° corners).
  - Generated the diagnostic heatmap (`outputs/workbench_review/d4cube_to_d4tet_diagnosis.png`) showing a viable linking pocket for a single pair at $60^\circ$ twist and $4.33\text{ mm}$ gap along $\langle 111 \rangle$.
- **Why we moved on:** Phase 1 was strictly a 2-particle workbench. Once we knew two isolated particles could interlock in empty space, we needed to build actual 3D materials.
- **The Hidden Trap:** An isolated particle in empty space has complete rotational freedom. In a 3D lattice, a particle is tethered to multiple neighbors. What works for two free-floating links does not automatically work when those links are embedded in a continuous crystal lattice.

### Phase 2: Single-Zone Homogeneous Engine
- **What it was:** A multi-plane generator testing homogeneous, single-cell lattices across 8 cell architectures and 3 habit planes (`{100}`, `{110}`, `{111}`).
- **What worked:**
  - 100% of coupons produced watertight 2-manifold solid meshes (`outputs/phase2_review/phase2_matrix_report.md`).
  - Standard cells (C-6-TT, D-4-TET, D-4-CUBE, D-4-CT, D-4-ICO, B-8-CO) achieved perfect catenation with zero hanging links and $>200\,\mu\text{m}$ clearance in their native orientations.
  - Proved that unrotated, homogeneous PAM generation is rock-solid.
- **Why we moved on:** Homogeneous lattices were solved. The project mandate was multi-zone transitions (heteroepitaxy) between dissimilar PAM types.

### Phase 3: Multi-Zone Heteroepitaxial Transitions (Where It Stalled)
- **What we attempted:** Stacking Zone A (e.g., D4-CUBE) and Zone B (e.g., D4-TET) across a shared interface plane.
- **What broke and why:**
  1. **The Symmetry Severing Bug:** To mate with D4-CUBE along $\{111\}$, the code applied a global $60^\circ$ rotation to all particles in Zone B. But diamond along $[1, 1, 1]$ has **3-fold ($C_3$) trigonal symmetry**, NOT 6-fold ($C_6$) symmetry. Rotating each tetrahedron by $60^\circ$ locally preserved only the single vertical bond along $Z$, but pointed the 3 transverse bonds directly into empty space between diamond sites. Zone B shattered into 105 disconnected vertical pairs (dimers), leaving huge open gaps across the array.
  2. **Scale / Supercell Explosion (19 cells tall instead of 3):** Slicing along $\{111\}$ uses an orthogonal supercell whose period is $c = \sqrt{3} a \approx 17.32\text{ mm}$, which contains 3 cubic unit cells (6 diamond layers) per repeat. Requesting a $3 \times 3 \times 3$ grid produced 9 cubic unit cells per zone (18–19 physical layers tall, $\approx 104\text{ mm}$) instead of 3 unit cells.
  3. **The Culling Distraction:** Culling routines were repeatedly added to strip away degree-0 particles. Instead of fixing why the geometry didn't link, culling masked the underlying issue or stripped out valid peripheral parts, leaving disconnected floating pieces at the edges.

### Phase 0: Paying Down Debt (The Clean Baseline)
- To eliminate misleading passes, we paused and cleared validator debt:
  - Extended cycle extraction to chordless 6-cycles in `epitaxy.py`, restoring the gold standard benchmark: **C-6-TT $\leftrightarrow$ D-4-TET on $\{100\}$ passes cleanly with 3 boundary links and $191.4\,\mu\text{m}$ clearance (Zhou et al., Science 2025)**.
  - Added full-assembly connectivity analysis (`evaluate_assembly_connectivity`) to report the exact number of disconnected components.
  - Fixed all 44 universal epitaxy tests (`pytest tests/test_universal_pam_epitaxy.py tests/test_void_olattice_and_robust_epitaxy.py`).

---

## 2. Core Architectural Resets & New Definitions

To move forward without confusion or bloated structures, the new agent must adopt these strict definitions:

### 1. New Scale & Terminology Standard
* **"Unit Cell" = A SINGLE LINK / PARTICLE.**
  - A unit cell is one individual physical link (e.g., one D4-TET wireframe frame, one D4-CUBE frame, one C-6-TT cage).
  - Stop referring to an entire periodic repeating lattice block as a "unit cell."
* **"Super-Cell" = AN ARRAY OF LINKED PARTICLES.**
  - Any 3D periodic collection of links (e.g. $2 \times 2 \times 2$ links, $3 \times 3 \times 3$ links) is a **super-cell**.
  - When the user asks for "3 cells tall", they mean **3 links tall** ($\approx 30\text{–}35\text{ mm}$ depending on pitch), NOT 3 orthogonal habit supercells that expand into 19 layers.

### 2. Throw Out the Culling Test
* **Stop aggressive culling:** Do not use iterative culling to hide unlinked particles or artificially force a pass.
* **Keep unlinked pieces visible:** If a particle fails to link at an interface, leave it in place. Seeing exactly where and why it fails in CAD/Blender is vastly more valuable than stripping it away and guessing.

### 3. Clean Miter Standard (Unchanged)
* All truss and wireframe lattices must use Clean Mitered joints (`clean_miter=True` / `build_clean_miter_truss`). No spherical bulges (`add_spheres=True` is prohibited).

---

## 3. The Immediate Mandate for the Fresh AI Agent

The incoming agent should follow this 3-step reboot:

### Step 1: Baseline Verification of Homogeneous, Unrotated PAM Types
Generate a clean, homogeneous super-cell ($2 \times 2 \times 2$ or $3 \times 3 \times 3$ links) for each core PAM type in its **native, unrotated orientation**:
1. `c6tt` (Truncated Tetrahedron on Simple Cubic)
2. `d4tet` (Tetrahedron on Diamond)
3. `d4cube` (Cube on Diamond)
4. `b8co` (Cuboctahedron on BCC)
5. `j4oct` (Octahedron on Diamond)

Verify that each homogeneous super-cell forms **a single, 100% interconnected topological component** with positive clearance and zero disconnected pieces.

### Step 2: Investigate the Fundamental Rotation Mystery
Why did D4-TET disconnect when rotated?
- Diamond consists of two interpenetrating FCC sublattices (Sublattice A and Sublattice B) displaced by $(\frac{1}{4}, \frac{1}{4}, \frac{1}{4})$.
- On Sublattice A, tetrahedra point in orientation $\mathbf{R}_A$. On Sublattice B, tetrahedra point in the inverted orientation $\mathbf{R}_B$.
- When an interface demands a rotation (e.g. $60^\circ$ about $[1, 1, 1]$), how must the sublattice sites, bond vectors, and particle orientations transform *together* so that internal catenation is not destroyed?
- Formulate a clear mathematical and geometric explanation before writing code.

### Step 3: Reboot Phase 1 with Ground Truth
Once homogeneous baselines are verified and the rotation mechanism is understood, formulate a clean, step-by-step implementation plan starting back at Phase 1 to tackle multi-zone transitions systematically.

---

## 4. Key References & Verification Commands

- **Benchmark Working Heteroepitaxy:** `c6tt_to_d4tet_100_coupon` in `outputs/phase3_review/` (C-6-TT to D-4-TET on $\{100\}$).
- **Two-Cell Workbench:** `graphite/explicit/interlinked/workbench.py` & `scripts/pam_workbench.py`.
- **Lattice Assembly:** `graphite/explicit/interlinked/assemble.py`.
- **Orthogonal Habit Seeding:** `graphite/explicit/interlinked/seeding.py`.
- **Topological Catenation & Linking:** `graphite/explicit/interlinked/epitaxy.py`.

### Verification Suite:
```powershell
# Verification Suite:
python scripts/export_core_spec.py --check
pytest tests/test_universal_pam_epitaxy.py tests/test_analytical_pam_grain_boundaries.py -q
```

---

## 5. Resolution & Validated Implementation (October 2026 Ground Truth)

All 3 reboot steps have been fully executed, proven mathematically, and validated in code:
1. **The Rotation Mystery Solved:** Proved that Diamond cubic ($Fd\bar{3}m$) along $[1, 1, 1]$ has trigonal $C_3$ symmetry, not $C_6$ hexagonal symmetry. A $60^\circ$ local frame spin severed transverse bulk diamond bonds, explaining the isolated dimers. Established the **Shared-Network 0° Twist Law**: shared-network transitions (e.g. Diamond CUBE $\to$ TET) require zero twist.
2. **SE(3) Gauss Linking Invariance Proved:** Proved that simultaneous rigid transformations on site coordinates and particle frames leave the double Gauss linking integral strictly invariant: $Lk(\gamma_A', \gamma_B') = Lk(\gamma_A, \gamma_B)$.
3. **Phase-Aware Glide Cleavage (Coupon 1A):** Snapping to 3-bond glide planes and applying the Shockley partial dislocation shift $\mathbf{s}^* = [0, a\sqrt{6}/12, 0]$ jumped clearance to $+381.4\,\mu\text{m}$ with simultaneous 3-face punctures on both polyhedra ($N_A=3, N_B=3$).
4. **Analytical Grain Boundaries & Coherent Twins:** Implemented closed-form gap equation $g^*(a, r_w)$, inverted void shift $\mathbf{s}^*$, and $\Sigma 3$ coherent twin reflection $\mathbf{R}_{\text{twin}}$ with automatic triangle winding inversion (`faces[:, [0, 2, 1]]`).
5. **Parametric Scaling Laws:** Formulated $a = \frac{D + \Delta}{\kappa}$ across all cell architectures.
6. **Authoritative Master Reference:** See [docs/POLYCATED_ARCHITECTED_MATERIALS.md](POLYCATED_ARCHITECTED_MATERIALS.md) (Sections 12–16) and [graphite/explicit/interlinked/README.md](../graphite/explicit/interlinked/README.md).
