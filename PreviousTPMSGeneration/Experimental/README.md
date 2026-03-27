# Experimental Gradient TPMS Scaffold Generation

## Overview

This folder contains all scripts and output STL files from an exploratory series
of experiments in generating functionally graded Triply Periodic Minimal Surface
(TPMS) scaffolds for bone implant research. The goal was to produce 3D-printable
STL files with controllable pore size gradients, targeting 200–800 µm pore sizes
at a 33% solid fraction.

---

## Background: What is a Sheet-TPMS Scaffold?

A TPMS is a minimal surface that is periodic in all three spatial directions. In
scaffold design, the **sheet** variant is preferred over the **solid** (nodal)
variant because it keeps both fluid channels of the bi-continuous structure open,
which supports cell migration and nutrient transport through the implant.

The surface used throughout this work is the **Split-P** (Schwarz Primitive variant):

```
F(x,y,z) = T1 + T2 + T3

T1 =  1.1 * ( sin(2kx)·sin(kz)·cos(ky) + sin(2ky)·sin(kx)·cos(kz) + sin(2kz)·sin(ky)·cos(kx) )
T2 = -0.2 * ( cos(2kx)·cos(2ky) + cos(2ky)·cos(2kz) + cos(2kz)·cos(2kx) )
T3 = -0.4 * ( cos(2kx) + cos(2ky) + cos(2kz) )
```

The sheet is the region where |F| < τ (tau). Tau is calibrated so that the
volume of the solid sheet equals the target solid fraction (33%).

The spatial frequency k relates to pore size P:
```
k = 2π / (P × 2.5)
```

---

## Approaches Explored

### 1. Voronoi / IDW Blending (`tpms_implant_generator.py`, `Implant_generator`)

**Concept:** Generate a stochastic point cloud (seed cloud) whose local density
encodes the desired pore size at each height. For each voxel, find the 4 nearest
seeds and blend their local Split-P fields using Inverse Distance Weighting (IDW).

```
F_total = Σ w_i · F_i(k_i)     where w_i ∝ 1/dist_i^4
```

**Geometry:** 2.2 mm diameter cylinder × 1.8 mm height (initial), then adapted
to a 2 × 2 × 2 mm cube, then to a 2 × 2 × 5 mm prism.

**Gradient:** 200 µm → 400 µm → 400 µm plateau → 800 µm, defined by piecewise
linear interpolation nodes along Z.

**Issues encountered:**
- Single-voxel-thick walls before wall thickening was investigated.
- Root cause identified: the simple `cos(kx) + cos(ky) + cos(kz)` formula was
  being used instead of the correct hybrid Split-P (T1+T2+T3).
- Once corrected, wall thickness is properly determined by tau and target VF.
- Blocky appearance at lower resolutions; resolution raised to 220 voxels.
- IDW graded approach was functional but exhibited messy transition zones.

**Output STLs:** `implant_uniform_33vf.stl`, `implant_graded_33vf.stl`,
`hybrid_implant.stl`, `hybrid_sheet_33.stl`, `graded_sheet_final.stl`,
`graded_sheet_v2.stl`, `graded_sheet_v3.stl`, `graded_splitp_implant.stl`,
`manifold_hybrid_cube.stl`

---

### 2. Sigmoid Field Morphing (`sigmoidal_morph_generator.py`)

**Concept:** Evaluate two complete Split-P fields independently — one for each
pore size — across the entire grid, then blend them using a sigmoid weight W(z):

```
F_total = (1 - W(z)) · F_A  +  W(z) · F_B
```

**First attempt:** Blend the signed fields, then take |F_total|.
- **Problem:** In the transition zone (W ≈ 0.5), F_A and F_B are out of phase
  and partially cancel, driving |F_total| artificially low. This creates phantom
  extra walls and elevated apparent solid fraction in the center.
- **Fix:** Blend the absolute (sheet) fields instead:
  `F_abs = (1 - W)·|F_A| + W·|F_B|`
  This eliminates inter-field cancellation. Tau jumped from ~0.39 to ~0.49,
  confirming the previous version was falsely elevated.

**Also implemented:** Chirped TPMS (Method A, in the same file) for direct
comparison — a single field with spatially-varying k, no blending artifacts.
The chirped approach showed "blooming" distortion where cells stretch/compress
rapidly.

**Output STLs:** `gradient_chirped.stl`, `gradient_sigmoid_morph.stl`,
`sigmoidal_morph_33.stl`

---

### 3. Chirped TPMS — Angled Cylinder (`angled_cylinder_generator.py`)

**Concept:** A 2 mm diameter cylinder with a top face cut at 30° from horizontal.
Uses a single chirped Split-P field with k varying continuously along the
normalized column height:

```
t(x,y,z) = z / z_top(x)    where z_top(x) = H + (x - R)·tan(30°)

k(t) = exp( (1-W)·ln(k_bottom) + W·ln(k_top) )    [log-space interpolation]
W(t) = smoothstep(t) = 3t² - 2t³
```

**Key design decision:** `t = z / z_top(x)` (column-local normalized height)
means the gradient rate adapts per column — shorter columns on the short side
compress the gradient faster — ensuring the angled top face always shows uniform
200 µm pore size regardless of angular position.

**Transition width discussion:**
- Original sigmoid (steepness=10) concentrated 90% of pore change into 59% of height.
- Switched to **smoothstep** (W = 3t²-2t³): zero slope at both faces, linear
  mid-section, transition spread across 100% of height.
- Centered gradient at the scaffold's center of mass in Z.

**Output STL:** `angled_cylinder_chirped.stl`

---

### 4. Final Design — Two-Zone Gradient Cylinder (`gradient_cylinder.py`)

**This is the primary output script.** Three generator functions:

#### `generate_flat()`
- 2 mm diameter × 5 mm tall, flat top and bottom
- Gradient: 800 µm (z=0) → 200 µm (z=5mm), purely Z-based
- `t = z / H` — radially symmetric, every horizontal cross-section is uniform
- **Output:** `gradient_cylinder_full.stl`, `gradient_cylinder_half.stl`

#### `generate_angled()`
- 2 mm diameter, 30° angled top, 3.85–5.00 mm tall
- Column-local gradient: `t = z / z_top(x)`
- 800 µm at flat bottom, 200 µm at angled top face
- **Output:** `gradient_angled_full.stl`, `gradient_angled_half.stl`

#### `generate_angled_twostage()`
- Same geometry as `generate_angled()`
- **Two-zone gradient:**
  - **Zone 1** (top 1 mm, perpendicular to angled face): 200 µm → 400 µm
    - Parameter: perpendicular distance from angled face
    - `t1 = (z_top - z)·cos(θ) / 1mm`
    - The 400 µm boundary plane is geometrically parallel to the 200 µm top face
  - **Zone 2** (bottom, from 400 µm plane to flat base): 400 µm → 800 µm
    - Parameter: column-local height within zone
    - `t2 = z / z_mid(x)`  where `z_mid(x) = z_top(x) - 1/cos(θ)`
- **Output:** `gradient_angled_twostage_full.stl`, `gradient_angled_twostage_half.stl`

Each function also outputs a **half-cylinder** (cut at y=0) for cross-section inspection.

---

## Key Technical Decisions

### Pore size interpolation: geometric, not linear
```
k(t) = k_bottom^(1-W) × k_top^W     [geometric / log-space]
P(t) = P_bottom^(1-W) × P_top^W
```
For a 4× pore size range (200–800 µm), geometric interpolation gives equal
*multiplicative* steps per unit height — perceptually uniform. The midpoint
geometric mean is √(200×800) ≈ 400 µm, not the arithmetic mean of 500 µm.

### Solid fraction and wall thickness
Tau is found by binary search over F_abs to satisfy:
```
fraction of interior voxels with F_abs < tau == TARGET_VF (0.33)
```
Wall thickness is a direct consequence of tau and the target solid fraction.
It is NOT set independently — higher VF → higher tau → thicker walls.

### Boundary stair-stepping
The XY stair-stepping on the cylinder wall (visible when looking down Z) and
the angled-cut stair-stepping are caused by hard voxel masking (`F[outside] = 10`).
Marching cubes snaps to voxel faces rather than interpolating sub-voxel positions.
The correct fix is a **soft signed-distance mask** that ramps F smoothly from
the lattice value to void over 2–3 voxels near each boundary. This was identified
but not yet implemented.

---

## Resolution Guide

| `RES_XY` | `RES_Z` | XY voxel | Z voxel | Voxels/200µm pore | Run time (approx) |
|----------|---------|----------|---------|-------------------|-------------------|
| 140 | 140 | 14 µm | 14 µm | 14 | ~5–10 s |
| 180 | 300 | 11 µm | 17 µm | 18 | ~30 s |
| 260 | 430 | 8 µm | 12 µm | 26 | ~2–3 min |
| 400 | 660 | 5 µm | 8 µm | 40 | ~10–15 min |

Minimum recommended for manufacturing: `RES_XY=260`, `RES_Z=430` (~26 voxels per smallest pore).

---

## Files in This Folder

### Scripts

| File | Description |
|------|-------------|
| `gradient_cylinder.py` | **Primary script.** Three generators: flat-top, angled, two-zone angled. |
| `sigmoidal_morph_generator.py` | Chirped vs. sigmoid morph comparison. Two-field and single-field blending. |
| `angled_cylinder_generator.py` | Earlier standalone angled cylinder with smoothstep gradient. |
| `tpms_implant_generator.py` | Voronoi/IDW graded scaffold. Supports uniform and graded modes. |
| `Implant_generator` | Reference script from Gemini with detailed comments. Source of working Split-P formula. |

### STL Outputs

| File | Description |
|------|-------------|
| `gradient_angled_twostage_full.stl` | **Primary output.** Two-zone angled cylinder, full. |
| `gradient_angled_twostage_half.stl` | **Primary output.** Two-zone angled cylinder, cross-section. |
| `gradient_cylinder_full.stl` | Flat-top gradient cylinder, full. |
| `gradient_cylinder_half.stl` | Flat-top gradient cylinder, cross-section. |
| `gradient_angled_full.stl` | Angled top, single gradient, full. |
| `gradient_angled_half.stl` | Angled top, single gradient, cross-section. |
| `gradient_chirped.stl` | Chirped TPMS 5mm prism (comparison). |
| `gradient_sigmoid_morph.stl` | Sigmoid morph 5mm prism (comparison). |
| `angled_cylinder_chirped.stl` | Earlier angled cylinder iteration. |
| `implant_uniform_33vf.stl` | Uniform 400 µm pore 2×2×2 mm cube. |
| `implant_graded_33vf.stl` | Voronoi IDW graded 2×2×5 mm prism. |
| `sigmoidal_morph_33.stl` | Early sigmoid morph (before fix). |
| `hybrid_*.stl`, `graded_*.stl`, `manifold_*.stl` | Earlier experimental iterations. |

---

## Dependencies

```
numpy
scipy
scikit-image (skimage)
numpy-stl (stl)
```

Install with:
```
pip install numpy scipy scikit-image numpy-stl
```
