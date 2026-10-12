# Authoritative Engineering Specification: Crystallographic Heteroepitaxy, CSL Matching & Biomimetic PAM Architecture

> **Document Status:** Authoritative Architectural & Mathematical Specification  
> **Source:** Deep Research Synthesis & Graphite PAM Empirical Foundations  
> **Applies to:** `graphite/explicit/interlinked/` (`assemble.py`, `epitaxy.py`, `seeding.py`, `validator.py`, `cells.py`, `generator.py`)

---

## 1. Executive Overview

This specification establishes the mathematical formulations, closed-form equations, and algorithmic blueprints for five foundational pillars of Polycatenated Metamaterial (PAM) architecture:
1. **Analytical Formula for the Diamond $\{111\}$ Glide-Set Interface** (optimal gap $g^*$ and Shockley shift vector $\mathbf{s}^*$).
2. **Automated 2D Coincident Site Lattice (CSL) Commensurate Matcher** (O-lattice tensor and integer supercell finder).
3. **Phase-Aware Boundary Slicing Engine** (bilayer phase detection and glide-plane snapping).
4. **Biomimetic Strut Geometry for Non-Linear Strain Hardening** (dovetailed nodal flaring and sinusoidal undulations).
5. **Quantitative Interface Quality Index ($\mathcal{Q}_{\text{interface}}$)** (Heaviside-gated multi-objective fitness function).

---

## 2. Specification Pillar 1: Analytical Formula for the Diamond $\{111\}$ Glide-Set Interface

### 2.1 Problem Definition
In transitioning between dissimilar Diamond-family PAMs (e.g. D4-CUBE $\to$ D4-TET along $\{111\}$), naive planar cutting places the boundary on the **shuffle-set (1-bond) cleavage plane**. This creates a flat triangular floor that obstructs penetration and leads to shallow, unlinked 1-face pierces ($N \le 1$).

To maximize steric clearance $\delta_{\min}$ while guaranteeing simultaneous 3-face punctures on both polyhedra ($N_A = 3, N_B = 3$), the interface must sit on the **3-bond glide plane**, positioning the downward-pointing 3-strut tripod of Sublattice B into the upward-pointing 3-strut corner tripod of the cube.

### 2.2 Optimal Out-of-Plane Gap ($g^*$)
The interplanar spacing of the shuffle-plane void in a diamond cubic lattice is:
$$d_{\text{shuffle}} = \frac{a\sqrt{3}}{12}$$

To center the structural nodes within the opposing void space while accounting for strut dilation by wire radius $r_w$ at the characteristic strut inclination angle ($35.26^\circ$ relative to the $\{111\}$ habit plane):

$$\boxed{g^*(a, r_w) = \frac{a\sqrt{3}}{12} - r_w \left(1 + \frac{1}{\cos(35.26^\circ)}\right) = \frac{a\sqrt{3}}{12} - r_w \left(1 + \sqrt{\frac{3}{2}}\right)}$$

For standard parameters ($a = 14.0\,$mm, $r_w = 0.25\,$mm):
$$g^*(14.0, 0.25) \approx 2.0207 - 0.25 \times (1 + 1.2247) = 2.0207 - 0.5562 = \mathbf{1.4645\,\text{mm}}$$

### 2.3 Analytical In-Plane Translation Vector ($\mathbf{s}^*$)
In the orthonormal supercell frame defined by basis vectors:
$$\mathbf{e}_1 = \frac{1}{\sqrt{2}}[1, -1, 0], \quad \mathbf{e}_2 = \frac{1}{\sqrt{6}}[1, 1, -2], \quad \mathbf{e}_3 = \frac{1}{\sqrt{3}}[1, 1, 1]$$

The crystallographic **Shockley partial dislocation vector** $\mathbf{b} = \frac{a}{6}\langle 11\bar{2}\rangle$ aligns entirely along the $\mathbf{e}_2$ axis. The required in-plane shift vector is:

$$\boxed{\mathbf{s}^* = \left[ 0, \; \frac{a\sqrt{6}}{6} \right] = \left[ 0, \; \frac{a}{\sqrt{6}} \right]}$$

For $a = 14.0\,$mm:
$$\mathbf{s}^* = [0, \; 5.7155]\,\text{mm}$$

---

## 3. Specification Pillar 2: Automated 2D CSL Commensurate Matcher

### 3.1 Problem Definition
When joining two zones with different unit cell pitches ($a_1 \ne a_2$) or different 2D Bravais surface lattices (e.g. Simple Cubic square $\{100\}$ to Diamond hexagonal $\{111\}$), non-commensurate boundaries generate unlinked "dangling" perimeter links that shed loose fragments upon loading.

### 3.2 Formulation: Inverting Bollmann's O-Lattice Equation
Let $\mathbf{A}_1, \mathbf{A}_2 \in \mathbb{R}^{2 \times 2}$ be the 2D primitive surface basis matrices of Zone 1 and Zone 2 in their respective habit plane coordinates.

#### Algorithm Steps:
1. **Discretize Candidate Rotations:**  
   Evaluate rotation angles $\theta_{\text{CSL}} \in [0, \pi/2]$ in discrete angular steps (e.g., $\Delta \theta = 0.5^\circ$).
2. **Compute Transformation Tensor:**  
   $$\mathbf{T}(\theta) = \mathbf{A}_2^{-1} \mathbf{R}(\theta_{\text{CSL}}) \mathbf{A}_1$$
3. **Diophantine Integer Optimization:**  
   Search for pairs of integer matrices $\mathbf{M}_1, \mathbf{M}_2 \in \mathbb{Z}^{2 \times 2}$ satisfying the domain constraint:
   $$\det(\mathbf{M}_1) \le \frac{L_{\max}^2}{\det(\mathbf{A}_1)}$$
   minimizing the Frobenius norm residual:
   $$\min_{\mathbf{M}_1, \mathbf{M}_2} \left\| \mathbf{M}_2 - \mathbf{M}_1 \mathbf{T}(\theta) \right\|_F$$
4. **Compute Residual Misfit Strain Tensor:**  
   $$\mathbf{E} = \mathbf{M}_2^{-1} \mathbf{M}_1 \mathbf{T}(\theta) - \mathbf{I}$$
5. **Acceptance Threshold:**  
   Accept the configuration if the areal strain satisfies:
   $$\Delta A_{\text{strain}} = \det(\mathbf{I} + \mathbf{E}) - 1 \le \epsilon \quad (\text{typically } \epsilon \le 0.02 = 2\%)$$
   The solver returns:
   - $\mathbf{M}_1$: Integer supercell multiplier for Zone A ($N_1 \times M_1$)
   - $\mathbf{M}_2$: Integer supercell multiplier for Zone B ($N_2 \times M_2$)
   - $\theta_{\text{CSL}}$: Coincident Site Lattice rotation angle
   - $\mathbf{E}$: Residual misfit strain tensor

---

## 4. Specification Pillar 3: Phase-Aware Boundary Slicing Engine

### 4.1 Theoretical Background
A diamond or zincblende cubic lattice possesses periodic bilayer spacing along $[111]$:
$$\begin{aligned}
d_{\text{shuffle}} &= \frac{a\sqrt{3}}{12} \quad (\text{1-bond plane; unstable; low coordination}) \\
d_{\text{glide}} &= \frac{a\sqrt{3}}{4} \quad (\text{3-bond plane; stable; high coordination}) \\
d_{\text{bilayer}} &= d_{\text{shuffle}} + d_{\text{glide}} = \frac{a\sqrt{3}}{3}
\end{aligned}$$

### 4.2 Algorithmic Implementation Logic
```python
import numpy as np

def slice_pam_boundary(mesh, normal_vec, d_target, a_pitch):
    """
    Slices a PAM boundary ensuring the cut interface terminates exclusively
    on the high-coordination 3-bond glide plane.
    """
    # 1. Calculate bilayer spacings
    d_shuffle = (a_pitch * np.sqrt(3.0)) / 12.0
    d_glide = (a_pitch * np.sqrt(3.0)) / 4.0
    d_bilayer = d_shuffle + d_glide
    
    # 2. Local phase within bilayer period
    local_phase = d_target % d_bilayer
    
    # 3. Detect shuffle plane vs glide plane
    if local_phase < d_shuffle:
        # Snap cut forward into the 3-bond glide plane
        d_snapped = d_target + (d_shuffle - local_phase)
        requires_shockley_shift = True
    else:
        # Already within glide plane
        d_snapped = d_target
        requires_shockley_shift = False
        
    # 4. Strict manifold boundary slice
    sliced_mesh = mesh.boolean_difference(Plane(normal_vec, d_snapped))
    
    # 5. Apply Shockley partial dislocation vector if phase was snapped
    if requires_shockley_shift:
        e2_vector = calculate_112_direction(normal_vec)
        b_shift = (a_pitch * np.sqrt(6.0) / 6.0) * e2_vector
        sliced_mesh.translate(b_shift)
        
    return sliced_mesh
```

---

## 5. Specification Pillar 4: Biomimetic Strut Geometry for Non-Linear Strain Hardening

Standard cylindrical struts ($r(t) = \text{const}$) exhibit linear compliance followed by abrupt rigid contact. Drawing inspiration from nacreous tablet pullout and glass sponge spicule bundling, struts are parameterized as non-uniform solids of revolution along normalized curve coordinate $t \in [0, 1]$.

### 5.1 Dovetailed Nodal Flaring (Progressive Wedge Locking)
By flaring the strut radius non-linearly near the nodes while maintaining slender mid-spans, initial deformation is kinematic and compliant. Under tension or shear, the flared vertices wedge against the opposing mesh apertures, generating progressive geometric strain hardening:

$$\boxed{r(t) = r_{\min} + \Delta r \left( |2t - 1|^k \right)}$$

- $r_{\min}$: Slender mid-span wire radius (ensures $\delta_{\min} \ge 100\,\mu\text{m}$ at rest).
- $\Delta r$: Nodal flaring amplitude ($\Delta r \le 0.5 \, r_{\min}$).
- $k \ge 3$: Flaring concentration exponent ($k=3$ or $k=4$ ensures the flare is strictly localized at the nodes, preventing rest-state mid-span collisions).

### 5.2 Wavy Sinusoidal Undulations (Ratchet Yielding & Friction)
Inspired by the microscopic surface waviness of aragonite tablets in nacre, a sinusoidal modulation is applied to the central axis $\mathbf{p}(t)$ normal to the primary slip direction:

$$\boxed{\mathbf{p}_{\text{new}}(t) = \mathbf{p}_{\text{old}}(t) + A \sin(2\pi n t) \, \mathbf{n}_\perp}$$

- $n \in \{1, 2\}$: Spatial frequency along the strut length.
- $A$: Wave amplitude constrained by the rest clearance:
  $$A \le \delta_{\min} - 100\,\mu\text{m}$$
This forces sliding struts to elastically flex as they pass through opposing apertures, yielding high frictional energy dissipation and progressive pullout resistance.

---

## 6. Specification Pillar 5: Quantitative Interface Quality Index ($\mathcal{Q}_{\text{interface}}$)

### 6.1 Objective Function Formulation
To rank thousands of candidate interface parameter sets $(g, \mathbf{s}, \theta)$ in automated catalog searches, the fitness function is formulated as a **Heaviside-gated multi-objective metric**:

$$\boxed{\mathcal{Q} = \mathcal{H}(\delta_{\min} - 100) \cdot \mathcal{H}(\min(N_A, N_B) - 2) \cdot \left[ w_1 \left( \frac{\mathcal{C}}{\mathcal{C}_{\max}} \right) + w_2 \Sigma^{-1} + w_3 \left( 1 - e^{-k(\delta_{\min} - 100)} \right) + w_4 \left( \frac{\bar{N}}{3} \right) \right]}$$

### 6.2 Component Definitions
| Symbol | Component | Mathematical Definition / Value | Purpose |
|:---:|:---|:---|:---|
| $\mathcal{H}(x)$ | Heaviside Step Gate | $1 \text{ if } x \ge 0 \text{ else } 0$ | Strict zero-tolerance for Gate 1 (clashes) and Gate 4 ($N \ge 2$). |
| $\delta_{\min}$ | Minimum Physical Clearance | $\min_{i, j} \text{dist}(\text{Link}_i, \text{Link}_j)$ in $\mu\text{m}$ | Ensures positive gap ($\ge 100\,\mu\text{m}$). |
| $N_A, N_B$ | Multi-Face Penetrations | Face punctures on parent polyhedra | Eliminates 1-face pierces ($N \le 1$). |
| $\mathcal{C} / \mathcal{C}_{\max}$ | Catenation Density | Realized inter-zone links / Max theoretical links | Rewards high connectivity across the boundary. |
| $\Sigma^{-1}$ | Coincidence Density | Reciprocal CSL density (e.g. $1/5$ for $\Sigma 5$) | Rewards small commensurate supercell domains. |
| $1 - e^{-k(\Delta - 100)}$ | Clearance Robustness | Saturation metric ($k = 0.01\,\mu\text{m}^{-1}$) | Rewards safety margin beyond the $100\,\mu\text{m}$ threshold. |
| $\bar{N} / 3$ | Penetration Depth | Average face punctures / 3 | Normalizes depth against the ideal 3-face $\{111\}$ puncture. |
| $w_1, w_2, w_3, w_4$ | Normalized Weights | $w_1=0.40, \; w_2=0.20, \; w_3=0.20, \; w_4=0.20$ | Balances connectivity, pitch coincidence, and clearance. |

---

## 7. Immediate Code Integration Mapping

| Specification Pillar | Target Graphite Module | Target Classes / Functions |
|:---|:---|:---|
| **Pillar 1:** $\{111\}$ Glide-Set Formula | `graphite/explicit/interlinked/assemble.py` | `InterfaceSpec`, `assemble()`, `_resolve_normal()` |
| **Pillar 2:** Automated 2D CSL Matcher | `graphite/explicit/interlinked/epitaxy.py` | `find_commensurate_supercell()`, `ZoneSpec` |
| **Pillar 3:** Phase-Aware Boundary Slicing | `graphite/explicit/interlinked/seeding.py` | `seed_orthogonal_lattice()`, `slice_pam_boundary()` |
| **Pillar 4:** Biomimetic Strut Geometry | `graphite/explicit/interlinked/writer_3mf.py` | `solidify_particle_prototype()`, `build_clean_miter_truss()` |
| **Pillar 5:** Interface Quality Index ($\mathcal{Q}$) | `graphite/explicit/interlinked/validator.py` | `evaluate_interface_quality()`, `InterZoneBoundaryReport` |
