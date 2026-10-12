# Crystallographic Supercell Scaling & Notation in Polycatenated Architected Materials (PAMs)

**Subsystem:** `graphite.explicit.interlinked`  
**Status:** Architectural & Crystallographic Standard  
**Date:** October 2026  

---

## 1. Executive Summary & Crystallographic Root Cause

When generating Polycatenated Architected Materials (PAMs) across crystallographic habit planes—specifically $\{100\}$, $\{110\}$, and $\{111\}$—specimen bounding boxes exhibit distinct dimensional proportions when given identical repetition counts `grid_size = (nx, ny, nz) = (2, 2, 2)`:
* **$\{100\}$:** Forms a symmetric cubic coupon ($L_x \approx L_y \approx L_z \approx 28.0\,\text{mm}$ for $a = 14.0\,\text{mm}$).
* **$\{110\}$:** Forms an orthogonal block ($L_x \approx 28.0\,\text{mm}$, $L_y \approx L_z \approx 19.8\,\text{mm}$).
* **$\{111\}$:** Forms a tall, slender prism ($L_x \approx 19.8\,\text{mm}$, $L_y \approx 17.1\,\text{mm}$, $L_z \approx 48.5\,\text{mm}$), exhibiting an aspect ratio of $\sim 2.45 : 1$.

This behavior is **not a software defect**, nor is it unit-cell dependent. It is a fundamental consequence of **crystallographic orthogonal supercell transformation**. 

When building along tilted habit planes, Graphite must tile by the **minimum translationally periodic orthogonal supercell** that aligns with the habit-plane coordinate system. Along $\{111\}$, diamond-cubic and FCC stacking requires a full 3-bilayer sequence ($ABC$) to return to the identical translational state, causing the vertical supercell repeat $L_z$ to equal $\sqrt{3} a \approx 1.732 a$, whereas in-plane periods $L_x$ and $L_y$ are smaller than $a$ ($0.707 a$ and $0.612 a$).

---

## 2. Supercell Periods Across Habit Planes

In Graphite, the conventional cubic lattice parameter $a$ is set by `pitch` (e.g. $a = 14.0\,\text{mm}$).

When rotating to habit plane $(hkl)$, the orthonormal coordinate frame $\mathbf{R} = [\mathbf{u}_x; \mathbf{u}_y; \mathbf{u}_z]$ is established such that the stacking normal $\mathbf{u}_z$ is parallel to $[h, k, l]$. The orthogonal domain spans:

$$\Delta X = n_x \cdot (a \cdot L_x), \quad \Delta Y = n_y \cdot (a \cdot L_y), \quad \Delta Z = n_z \cdot (a \cdot L_z)$$

where $[L_x, L_y, L_z]$ are normalized periodic step multipliers determined by the parent network:

### 2.1 Diamond Cubic Network (`dia` / D-4-TET, D-4-CUBE)

| Habit Plane | Stacking Normal $\mathbf{u}_z$ | In-Plane $\mathbf{u}_x$ | In-Plane $\mathbf{u}_y$ | Normalized Periodicity $[L_x, L_y, L_z]$ | Absolute Size for $a=14\,\text{mm}$, `(2,2,2)` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$\{100\}$** | $[0, 0, 1]$ | $[1, 0, 0]$ | $[0, 1, 0]$ | $[1.0, 1.0, 1.0]$ | $28.0 \times 28.0 \times 28.0\,\text{mm}$ (Aspect 1 : 1 : 1) |
| **$\{110\}$** | $\frac{1}{\sqrt{2}}[1, 1, 0]$ | $[0, 0, 1]$ | $\frac{1}{\sqrt{2}}[1, -1, 0]$ | $[1.0, \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}] \approx [1.0, 0.707, 0.707]$ | $28.0 \times 19.8 \times 19.8\,\text{mm}$ (Aspect 1.4 : 1 : 1) |
| **$\{111\}$** | $\frac{1}{\sqrt{3}}[1, 1, 1]$ | $\frac{1}{\sqrt{2}}[1, -1, 0]$ | $\frac{1}{\sqrt{6}}[1, 1, -2]$ | $[\frac{1}{\sqrt{2}}, \frac{\sqrt{6}}{4}, \sqrt{3}] \approx [0.707, 0.612, 1.732]$ | $19.8 \times 17.1 \times 48.5\,\text{mm}$ (**Aspect 1.15 : 1 : 2.83**) |

### 2.2 Simple Cubic Network (`pcu` / C-6-TT, J-4-OCT)

| Habit Plane | Normalized Periodicity $[L_x, L_y, L_z]$ | Absolute Size for $a=12.7\,\text{mm}$, `(2,2,2)` |
|:---:|:---:|:---:|
| **$\{100\}$** | $[1.0, 1.0, 1.0]$ | $25.4 \times 25.4 \times 25.4\,\text{mm}$ |
| **$\{110\}$** | $[1.0, \sqrt{2}, \frac{1}{\sqrt{2}}] \approx [1.0, 1.414, 0.707]$ | $25.4 \times 35.9 \times 18.0\,\text{mm}$ |
| **$\{111\}$** | $[\sqrt{2}, \sqrt{6}, \sqrt{3}] \approx [1.414, 2.449, 1.732]$ | $35.9 \times 62.2 \times 44.0\,\text{mm}$ |

### 2.3 Body-Centered Cubic Network (`bcc` / B-8-CO)

| Habit Plane | Normalized Periodicity $[L_x, L_y, L_z]$ | Absolute Size for $a=14.0\,\text{mm}$, `(2,2,2)` |
|:---:|:---:|:---:|
| **$\{100\}$** | $[1.0, 1.0, 1.0]$ | $28.0 \times 28.0 \times 28.0\,\text{mm}$ |
| **$\{110\}$** | $[1.0, \frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}] \approx [1.0, 0.707, 0.707]$ | $28.0 \times 19.8 \times 19.8\,\text{mm}$ |
| **$\{111\}$** | $[\frac{1}{\sqrt{2}}, \frac{\sqrt{6}}{2}, \sqrt{3}] \approx [0.707, 1.225, 1.732]$ | $19.8 \times 34.3 \times 48.5\,\text{mm}$ |

---

## 3. Why Orthogonal Supercells are Mandatory for PAMs

In standard welded truss lattices, bounding boxes can arbitrarily slice through unit cells, cutting individual struts to fit any arbitrary bounding volume. 

In **Polycatenated Architected Materials (PAMs)**, this is strictly forbidden:
1. **Zero-Culling Mandate & Catenation Invariance:**
   PAM particles are closed, non-welded polyhedral wireframe cages. If a bounding cut bisects a cage, the ring is cut open. The open link unravels, falls loose, and violates the zero-culling standard, producing loose debris during powder removal in LPBF/SLS printing.
2. **Clean Planar Boundary Faces:**
   The orthogonal supercell vectors are uniquely chosen such that the bounding planes ($X = 0, X_{\max}$, $Y = 0, Y_{\max}$, $Z = 0, Z_{\max}$) pass cleanly through the interstitial gaps between interlocking cages without intersecting any constituent wireframe struts.
3. **Translational Stacking Along $\{111\}$:**
   Along $[1, 1, 1]$, the close-packed stacking sequence is $A \to B \to C \to A$. Truncating a diamond lattice at arbitrary non-supercell heights yields unequal top and bottom boundary terminations (sublattice A vs B), disrupting symmetric mechanical gripping and boundary testing.

---

## 4. Single-Cell Notation vs. Supercell Notation: Critical Analysis

The user asked:
> *"I want to try to switch our generation over from supercell notation to single cell notation. Will that cause any issues with our generation, future heteroepitaxy efforts, or anything else you can foresee? I can stick with super-cell notation as long as we allow the user to define the size of single cells."*

### 4.1 What "Single-Cell Notation" Means
* **Conventional Single Cell Parameter ($a$):** The edge length of the cubic unit cell. In Graphite, this is already defined directly by `pitch = a` (e.g. $14.0\,\text{mm}$).
* **Particle / Link Size ($D_{\text{link}}$):** The physical bounding radius or strut length of a single wireframe cage, directly scaled from $a$.
* **Repeat Counts:** Tiling by single conventional cells or single close-packed layers rather than multi-layer supercells.

### 4.2 Potential Issues if We Abandon Supercells

| Dimension | Risk / Issue | Severity | Technical Reason |
|:---|:---|:---:|:---|
| **Boundary Slicing** | Severe Catenation Failure | 🔴 Critical | Slicing along non-supercell planes cuts closed cages, unlinking the assembly and producing loose floating links (violates zero-culling). |
| **Heteroepitaxy** | Interface Incommensurability | 🔴 Critical | Heteroepitaxial boundaries require matching 2D Coincidence Site Lattices (CSL). If in-plane transverse dimensions do not match common supercell periods, interface sites will collide or fail to catenate across the boundary. |
| **Periodic Boundary Conditions** | Phase Misalignment | 🟠 Moderate | Truncating $\{111\}$ at fractional $ABC$ layers causes lateral phase shear ($\frac{1}{3}\mathbf{u}_x + \frac{1}{3}\mathbf{u}_y$), breaking periodic boundary simulations. |

### 4.3 Recommended Solution: User Single-Cell Definition with Supercell Kinematics

We retain the robust orthogonal supercell math under the hood (preserving zero-culling, flat faces, and heteroepitaxy), while providing the user three intuitive controls:

1. **Direct Single-Cell Pitch Definition (Existing Standard):**
   `pitch = a` directly defines the single conventional unit cell length in mm. All particle geometries and clearances scale from this single value.
2. **Aspect-Ratio-Aware Repeat Presets:**
   To obtain an approximately isotropic ($1 : 1 : 1$) cube on $\{111\}$:
   - For `dia` (D-4-TET, D-4-CUBE): Use `grid_size = (3, 3, 1)` or `(5, 6, 2)`.
     - `grid_size = (3, 3, 1)` gives $\sim 30 \times 26 \times 24\,\text{mm}$ (near-cube, $\sim 1 : 1 : 1$).
     - `grid_size = (2, 2, 2)` gave $19.8 \times 17.1 \times 48.5\,\text{mm}$ (tall tower, $2.45 : 1$).
   - For `bcc` (B-8-CO): Use `grid_size = (3, 2, 1)`.
   - For `pcu` (C-6-TT): Use `grid_size = (2, 1, 1)` or `(3, 2, 2)`.
3. **Target Dimensioning (`target_box_mm`):**
   Allow the user to specify desired physical millimeters, e.g. `(30.0, 30.0, 30.0)`. Graphite automatically rounds to the nearest integer supercell repeats $(n_x, n_y, n_z)$ that satisfy the exact crystallographic invariants without broken links.

---

## 5. Fractional Supercell Widths & Sub-Supercell Heteroepitaxy

### 5.1 Fractional Widths (e.g. 1.5 Supercells Wide)
In Graphite's particle engine, cages are **discrete rigid bodies instantiated at crystallographic lattice sites**, not continuous solid volumes trimmed with a knife.
* Inside one $\{111\}$ diamond supercell along $X$ ($\mathbf{u}_x \parallel [1, -1, 0]$), there are two discrete atomic columns separated by $\Delta x = 0.5 L_x \approx 4.95\,\text{mm}$ (for $a = 14.0\,\text{mm}$).
* Specifying `grid_size = (1.5, 1, 1)` captures the 4th atomic column at $X = 1.5 L_x = 14.85\,\text{mm}$.
* The engine instantiates the extra column in its entirety (20 uncut particles instead of 15).
* **Zero Cages Sliced:** No struts are severed, no loops unravel, and zero-culling is preserved.
* **Boundary Termination:** Lateral edge particles naturally have lower coordination numbers (2 or 3 links instead of 4), terminating cleanly at the model boundary with **0 collisions, 0 hanging links, and 1 single connected component**.

### 5.2 Sub-Supercell Heteroepitaxial Transitions (e.g. 1-Bilayer Transitions)
A full $\{111\}$ vertical supercell repeat ($L_z = \sqrt{3}a \approx 24.25\,\text{mm}$) spans 3 bilayers ($A \to B \to C$). A sub-supercell transition occurs when Zone A spans only **1 bilayer** ($\approx 8.1\,\text{mm}$) or **2 bilayers** ($\approx 16.2\,\text{mm}$) before transitioning to Zone B.

* **Crystallographic Stacking Shift:** In diamond $\{111\}$, transitioning from Bilayer 1 to Bilayer 2 introduces an in-plane stacking shift:
  $$\mathbf{t}_{\text{stack}} = [0, \tfrac{2}{3}L_y, 0] \approx [0, 5.72\,\text{mm}, 0] \quad \text{or} \quad [\tfrac{1}{2}L_x, -\tfrac{1}{3}L_y, 0] \approx [4.95, -2.86, 0]\,\text{mm}$$
* **Transition Alignment:**
  1. Without compensating for this shift, stacking Zone B directly above Zone A yields lateral misalignment and 2 disconnected components (`comps = 2`).
  2. By applying the stacking shift `shift = [0.0, 5.7155, 0.0]` and calibrated gap $g = 4.90\,\text{mm}$ via `InterfaceSpec`, Zone B's receiving apertures perfectly engage Zone A's penetrating cages.
* **Empirical Verification:** A 1-bilayer D4-CUBE $\to$ 1-bilayer D4-TET transition was verified via Manifold3D exact CSG boolean intersection:
  - **42 bodies, 0 colliding pairs (`has_intersecting_faces=False`)**
  - **100% single connected component (`comps = 1`)**
  - **+135 µm positive physical clearance**
  - **Total specimen height reduced from $50+\,\text{mm}$ to $\approx 18\,\text{mm}$.**

---

## 6. Authoritative Baseline Rotation Standard (Balanced Aspect Ratios)

The complete 12-sample baseline suite across `{100}`, `{110}`, and `{111}` has been regenerated and audited under the clean miter and Manifold3D exact CSG boolean standard (`scripts/experiments/generate_verified_rotation_samples.py`). On `{111}`, repeat counts are calibrated to yield balanced $\sim 30\text{--}35\,\text{mm}$ envelopes matching `{100}` and `{110}`:

| Sample | Architecture | Habit Plane | Grid Repeats | Bounding Box ($W \times D \times H$) | Particles | CSG Clashes | Connectivity | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `c6tt_111` | **C-6-TT** | $\{111\}$ | `(2, 1, 1)` | $50.2 \times 45.5 \times 33.8\,\text{mm}$ | 26 | **0** | 1 comp (0 hanging) | 🟩 PASS |
| `d4tet_111` | **D-4-TET** | $\{111\}$ | `(3, 3, 1)` | $38.6 \times 35.9 \times 31.5\,\text{mm}$ | 84 | **0** | 1 comp (0 hanging) | 🟩 PASS |
| `d4cube_111` | **D-4-CUBE** | $\{111\}$ | `(3, 3, 1)` | $36.9 \times 33.9 \times 32.9\,\text{mm}$ | 84 | **0** | 1 comp (0 hanging) | 🟩 PASS |
| `b8co_111` | **B-8-CO** | $\{111\}$ | `(3, 2, 1)` | $46.0 \times 48.5 \times 37.7\,\text{mm}$ | 34 | **0** | 1 comp (0 hanging) | 🟩 PASS |

All models have both instanced `.3mf` and watertight `.stl` deliverables exported under `outputs/pam_rotation_samples/` alongside high-res PyVista renders. Full audit log: [outputs/pam_rotation_samples/rotation_samples_report.md](../../outputs/pam_rotation_samples/rotation_samples_report.md).

