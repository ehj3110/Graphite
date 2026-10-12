# PAM Transition Lookup Table & Crystallographic Reference

**Subsystem:** `graphite.explicit.interlinked`  
**Status:** Validated Reference Standard  
**Catalog Version:** 1.0 (Core 5 Metamaterials across {100}, {110}, {111})  

---

## 1. Overview & Classification Schema

This document provides the authoritative, precomputed lookup table for all kinematically admissible
polycatenated metamaterial transitions across habit planes `{100}`, `{110}`, and `{111}`.
Transition parameters are validated against Graphite's 5 Quality Gates (zero collisions $\Delta > 0$,
mutual closed-loop catenation $|Lk| \ge 1$, single connected component $N_{\text{components}} = 1$,
and zero dangling links).

### Feasibility Classes:
* **DIRECT (Class 1):** Directly commensurate, non-colliding, and topologically catenated across the boundary.
  Zero intermediate buffer required.
* **BUFFERED (Class 2):** Geometrically frustrated as a direct cut, but admissible via a multi-step bridge route
  (e.g., intermediate cell or alternate habit plane path). All recommended routes in this catalog are
  constructed from verified DIRECT transitions.
* **FRUSTRATED (Class 3):** Sterically impossible due to aperture restrictions ($D_{\text{aperture}} < D_{\text{profile}}$)
  or fundamental space-group symmetry breaking without intermediate buffer layers.

---

## 2. Same-Cell Plane-to-Plane & Rotational Transitions

Transitions within the *same* unit cell type between distinct crystallographic habit planes
(e.g., $\{100\} \to \{111\}$ reorientations, elbow turns, and twin boundaries):

| Cell | Boundary Type | Habit Planes | Gap ($g/a$) | Twist (deg) | Clearance (µm) | Status | Notes / Recommended Routes |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `C6TT` | Homogeneous | {100} $\to$ {100} | 0.650 | 0.0° | 2715.0 | 🟩 DIRECT | Homogeneous translational stacking |
| `C6TT` | Homogeneous | {110} $\to$ {110} | 0.650 | 0.0° | 2715.0 | 🟩 DIRECT | Homogeneous translational stacking |
| `C6TT` | Homogeneous | {111} $\to$ {111} | 0.650 | 0.0° | 2715.0 | 🟩 DIRECT | Homogeneous translational stacking |
| `C6TT` | Reorientation | {100} $\to$ {110} | 0.650 | 45.0° | 2715.0 | 🟩 DIRECT | Plane reorientation from {100} to {110} |
| `C6TT` | Reorientation | {100} $\to$ {111} | 0.383 | 45.0° | 1313.0 | 🟩 DIRECT | Plane reorientation from {100} to {111} |
| `C6TT` | Reorientation | {110} $\to$ {111} | 0.650 | 0.0° | 1334.2 | 🟩 DIRECT | Plane reorientation from {110} to {111} |
| `D4TET` | Elbow / Turn | {100} $\to$ {100} | 0.250 | 90.0° | 1357.5 | 🟩 DIRECT | Kink / tilt boundary (twist=90.0°) |
| `D4TET` | Elbow / Turn | {110} $\to$ {110} | 0.250 | 90.0° | 1357.5 | 🟩 DIRECT | Kink / tilt boundary (twist=90.0°) |
| `D4TET` | Elbow / Turn | {111} $\to$ {111} | 0.250 | 90.0° | 1357.5 | 🟩 DIRECT | Kink / tilt boundary (twist=90.0°) |
| `D4TET` | Reorientation | {100} $\to$ {110} | 0.250 | 45.0° | 810.6 | 🟩 DIRECT | Plane reorientation from {100} to {110} |
| `D4TET` | Reorientation | {100} $\to$ {111} | 0.250 | 45.0° | 712.5 | 🟩 DIRECT | Plane reorientation from {100} to {111} |
| `D4TET` | Reorientation | {110} $\to$ {111} | 0.250 | 60.0° | 595.3 | 🟩 DIRECT | Plane reorientation from {110} to {111} |
| `D4CUBE` | Homogeneous | {100} $\to$ {100} | 0.250 | 0.0° | 658.2 | 🟩 DIRECT | Homogeneous translational stacking |
| `D4CUBE` | Homogeneous | {110} $\to$ {110} | 0.250 | 0.0° | 658.2 | 🟩 DIRECT | Homogeneous translational stacking |
| `D4CUBE` | Homogeneous | {111} $\to$ {111} | 0.250 | 0.0° | 658.2 | 🟩 DIRECT | Homogeneous translational stacking |
| `D4CUBE` | Reorientation | {100} $\to$ {110} | 0.250 | 30.0° | 658.2 | 🟩 DIRECT | Plane reorientation from {100} to {110} |
| `D4CUBE` | Reorientation | {100} $\to$ {111} | 0.250 | 45.0° | 760.4 | 🟩 DIRECT | Plane reorientation from {100} to {111} |
| `D4CUBE` | Reorientation | {110} $\to$ {111} | 0.250 | 0.0° | 280.6 | 🟩 DIRECT | Plane reorientation from {110} to {111} |
| `B8CO` | Homogeneous | {100} $\to$ {100} | 0.650 | 45.0° | 366.1 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `B8CO` | Homogeneous | {110} $\to$ {110} | 0.650 | 45.0° | 366.1 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `B8CO` | Homogeneous | {111} $\to$ {111} | 0.650 | 45.0° | 366.1 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `B8CO` | Reorientation | {100} $\to$ {110} | 0.650 | 0.0° | 366.1 | 🟩 DIRECT | Plane reorientation from {100} to {110} |
| `B8CO` | Reorientation | {100} $\to$ {111} | — | — | — | 🟨 BUFFERED | • `b8co {100} -> intermediate buffer -> b8co {111}` |
| `B8CO` | Reorientation | {110} $\to$ {111} | — | — | — | 🟨 BUFFERED | • `b8co {110} -> intermediate buffer -> b8co {111}` |
| `J4OCT` | Homogeneous | {100} $\to$ {100} | 0.650 | 45.0° | 2064.5 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `J4OCT` | Homogeneous | {110} $\to$ {110} | 0.650 | 45.0° | 2064.5 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `J4OCT` | Homogeneous | {111} $\to$ {111} | 0.650 | 45.0° | 2064.5 | 🟩 DIRECT | Kink / tilt boundary (twist=45.0°) |
| `J4OCT` | Reorientation | {100} $\to$ {110} | 0.650 | 0.0° | 2064.5 | 🟩 DIRECT | Plane reorientation from {100} to {110} |
| `J4OCT` | Reorientation | {100} $\to$ {111} | 0.650 | 0.0° | 437.8 | 🟩 DIRECT | Plane reorientation from {100} to {111} |
| `J4OCT` | Reorientation | {110} $\to$ {111} | 0.650 | 45.0° | 793.5 | 🟩 DIRECT | Plane reorientation from {110} to {111} |

---

## 3. Inter-Cell Heteroepitaxial Transitions

Transitions between *different* unit cell architectures across habit planes `{100}`, `{110}`, and `{111}`:

| Transition | Habit Plane | Supercell Ratio | Gap ($g/a$) | Twist (deg) | Boundary Clearance (µm) | Status | Recommended Routes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `C6TT` $\to$ `D4TET` | {100} | 1.414 | 0.544 | 45.0° | 191.4 | 🟩 DIRECT | Direct |
| `C6TT` $\to$ `D4TET` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {110}`<br/>• `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110}` |
| `C6TT` $\to$ `D4TET` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {111}`<br/>• `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4tet {111}` |
| `C6TT` $\to$ `D4CUBE` | {100} | 1.414 | — | — | — | 🟨 BUFFERED | • `c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}` |
| `C6TT` $\to$ `D4CUBE` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110}`<br/>• `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110} -> d4cube {110}` |
| `C6TT` $\to$ `D4CUBE` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111}`<br/>• `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4tet {111} -> d4cube {111}` |
| `C6TT` $\to$ `B8CO` | {100} | 1.000 | 0.472 | 0.0° | 183.7 | 🟩 DIRECT | Direct |
| `C6TT` $\to$ `B8CO` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `c6tt {110} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `c6tt {110} -> c6tt {111} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `C6TT` $\to$ `B8CO` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `c6tt {111} -> intermediate buffer -> b8co {111}` |
| `C6TT` $\to$ `J4OCT` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `c6tt {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `C6TT` $\to$ `J4OCT` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `c6tt {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `C6TT` $\to$ `J4OCT` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `c6tt {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `D4TET` $\to$ `C6TT` | {100} | 0.707 | 0.544 | 45.0° | 191.4 | 🟩 DIRECT | Direct |
| `D4TET` $\to$ `C6TT` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {110}`<br/>• `d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {111} -> c6tt {110}` |
| `D4TET` $\to$ `C6TT` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {111}`<br/>• `d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {110} -> c6tt {111}` |
| `D4TET` $\to$ `D4CUBE` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}`<br/>• `d4tet -> d4ct -> d4cube (continuous morph buffer)` |
| `D4TET` $\to$ `D4CUBE` | {110} | 1.000 | 0.286 | 0.0° | 104.7 | 🟩 DIRECT | Direct |
| `D4TET` $\to$ `D4CUBE` | {111} | 1.000 | 0.350 | 0.0° | 135.0 | 🟩 DIRECT | Direct |
| `D4TET` $\to$ `B8CO` | {100} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {100} -> c6tt {100} -> b8co {100}` |
| `D4TET` $\to$ `B8CO` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `d4tet {110} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `D4TET` $\to$ `B8CO` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {111} -> intermediate buffer -> b8co {111}` |
| `D4TET` $\to$ `J4OCT` | {100} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `D4TET` $\to$ `J4OCT` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `D4TET` $\to$ `J4OCT` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4tet {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `D4CUBE` $\to$ `C6TT` | {100} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100}` |
| `D4CUBE` $\to$ `C6TT` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {110}`<br/>• `d4cube {110} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {110}` |
| `D4CUBE` $\to$ `C6TT` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {111}`<br/>• `d4cube {111} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {111}` |
| `D4CUBE` $\to$ `D4TET` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100}`<br/>• `d4cube -> d4ct -> d4tet (continuous morph buffer)` |
| `D4CUBE` $\to$ `D4TET` | {110} | 1.000 | 0.286 | 0.0° | 104.7 | 🟩 DIRECT | Direct |
| `D4CUBE` $\to$ `D4TET` | {111} | 1.000 | 0.350 | 0.0° | 135.0 | 🟩 DIRECT | Direct |
| `D4CUBE` $\to$ `B8CO` | {100} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100}` |
| `D4CUBE` $\to$ `B8CO` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `d4cube {110} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `D4CUBE` $\to$ `B8CO` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {111} -> intermediate buffer -> b8co {111}` |
| `D4CUBE` $\to$ `J4OCT` | {100} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `D4CUBE` $\to$ `J4OCT` | {110} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `D4CUBE` $\to$ `J4OCT` | {111} | 0.707 | — | — | — | 🟨 BUFFERED | • `d4cube {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `B8CO` $\to$ `C6TT` | {100} | 1.000 | 0.472 | 0.0° | 183.7 | 🟩 DIRECT | Direct |
| `B8CO` $\to$ `C6TT` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `b8co {110} -> b8co {100} -> c6tt {100} -> c6tt {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> c6tt {111} -> c6tt {110}` |
| `B8CO` $\to$ `C6TT` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `b8co {111} -> intermediate buffer -> c6tt {111}` |
| `B8CO` $\to$ `D4TET` | {100} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {100} -> c6tt {100} -> d4tet {100}` |
| `B8CO` $\to$ `D4TET` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110}` |
| `B8CO` $\to$ `D4TET` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {111} -> intermediate buffer -> d4tet {111}` |
| `B8CO` $\to$ `D4CUBE` | {100} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}` |
| `B8CO` $\to$ `D4CUBE` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110} -> d4cube {110}` |
| `B8CO` $\to$ `D4CUBE` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `b8co {111} -> intermediate buffer -> d4cube {111}` |
| `B8CO` $\to$ `J4OCT` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `b8co {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `B8CO` $\to$ `J4OCT` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `b8co {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `B8CO` $\to$ `J4OCT` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `b8co {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `J4OCT` $\to$ `C6TT` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {100} -> intermediate Kagome buffer -> c6tt {100}` |
| `J4OCT` $\to$ `C6TT` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {110} -> intermediate Kagome buffer -> c6tt {110}` |
| `J4OCT` $\to$ `C6TT` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {111} -> intermediate Kagome buffer -> c6tt {111}` |
| `J4OCT` $\to$ `D4TET` | {100} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {100} -> intermediate Kagome buffer -> d4tet {100}` |
| `J4OCT` $\to$ `D4TET` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {110} -> intermediate Kagome buffer -> d4tet {110}` |
| `J4OCT` $\to$ `D4TET` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {111} -> intermediate Kagome buffer -> d4tet {111}` |
| `J4OCT` $\to$ `D4CUBE` | {100} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {100} -> intermediate Kagome buffer -> d4cube {100}` |
| `J4OCT` $\to$ `D4CUBE` | {110} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {110} -> intermediate Kagome buffer -> d4cube {110}` |
| `J4OCT` $\to$ `D4CUBE` | {111} | 1.414 | — | — | — | 🟨 BUFFERED | • `j4oct {111} -> intermediate Kagome buffer -> d4cube {111}` |
| `J4OCT` $\to$ `B8CO` | {100} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {100} -> intermediate Kagome buffer -> b8co {100}` |
| `J4OCT` $\to$ `B8CO` | {110} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {110} -> intermediate Kagome buffer -> b8co {110}` |
| `J4OCT` $\to$ `B8CO` | {111} | 1.000 | — | — | — | 🟨 BUFFERED | • `j4oct {111} -> intermediate Kagome buffer -> b8co {111}` |

---

## 4. The Hall of Frustration: Geometrically Inadmissible Direct Interfaces

To prevent repeating invalid design iterations, the following transitions are proven
to be frustrated as direct planar cuts, and their verified non-frustrated bridge paths are documented:

| Frustrated Transition | Habit Plane | Physical Root Cause | Recommended Non-Frustrated Route |
|:---|:---:|:---|:---|
| `B8CO` $\to$ `B8CO` | {100->111} | Geometric boundary frustration between {100} and {111} facet profiles | • `b8co {100} -> intermediate buffer -> b8co {111}` |
| `B8CO` $\to$ `B8CO` | {110->111} | Geometric boundary frustration between {110} and {111} facet profiles | • `b8co {110} -> intermediate buffer -> b8co {111}` |
| `C6TT` $\to$ `D4TET` | {110} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {110}`<br/>• `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110}` |
| `C6TT` $\to$ `D4TET` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {111}`<br/>• `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4tet {111}` |
| `C6TT` $\to$ `D4CUBE` | {100} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}` |
| `C6TT` $\to$ `D4CUBE` | {110} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110}`<br/>• `c6tt {110} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110} -> d4cube {110}` |
| `C6TT` $\to$ `D4CUBE` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111}`<br/>• `c6tt {111} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4tet {111} -> d4cube {111}` |
| `C6TT` $\to$ `B8CO` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (7.54 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -3613.0 um. | • `c6tt {110} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `c6tt {110} -> c6tt {111} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `C6TT` $\to$ `B8CO` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (7.54 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -3613.0 um. | • `c6tt {111} -> intermediate buffer -> b8co {111}` |
| `C6TT` $\to$ `J4OCT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (7.54 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -6967.5 um. | • `c6tt {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `C6TT` $\to$ `J4OCT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (7.54 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -6967.5 um. | • `c6tt {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `C6TT` $\to$ `J4OCT` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (7.54 mm) is smaller than penetrating profile + clearance (12.01 mm). Margin: -4462.7 um. | • `c6tt {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `D4TET` $\to$ `C6TT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -7693.3 um. | • `d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {110}`<br/>• `d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {111} -> c6tt {110}` |
| `D4TET` $\to$ `C6TT` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {111}`<br/>• `d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {110} -> c6tt {111}` |
| `D4TET` $\to$ `D4CUBE` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (7.05 mm). Margin: -2567.6 um. | • `d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}`<br/>• `d4tet -> d4ct -> d4cube (continuous morph buffer)` |
| `D4TET` $\to$ `B8CO` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -6675.1 um. | • `d4tet {100} -> c6tt {100} -> b8co {100}` |
| `D4TET` $\to$ `B8CO` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -6675.1 um. | • `d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `d4tet {110} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `D4TET` $\to$ `B8CO` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4tet {111} -> intermediate buffer -> b8co {111}` |
| `D4TET` $\to$ `J4OCT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -10029.6 um. | • `d4tet {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `D4TET` $\to$ `J4OCT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.48 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -10029.6 um. | • `d4tet {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `D4TET` $\to$ `J4OCT` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4tet {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `D4CUBE` $\to$ `C6TT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -7798.1 um. | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100}` |
| `D4CUBE` $\to$ `C6TT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -7798.1 um. | • `d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {110}`<br/>• `d4cube {110} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {110}` |
| `D4CUBE` $\to$ `C6TT` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> c6tt {111}`<br/>• `d4cube {111} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> c6tt {111}` |
| `D4CUBE` $\to$ `D4TET` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (6.35 mm). Margin: -1971.7 um. | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100}`<br/>• `d4cube -> d4ct -> d4tet (continuous morph buffer)` |
| `D4CUBE` $\to$ `B8CO` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -6779.9 um. | • `d4cube {100} -> d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100}`<br/>• `d4cube {100} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100}` |
| `D4CUBE` $\to$ `B8CO` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -6779.9 um. | • `d4cube {110} -> d4tet {110} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}`<br/>• `d4cube {110} -> d4cube {111} -> d4tet {111} -> d4tet {100} -> c6tt {100} -> b8co {100} -> b8co {110}` |
| `D4CUBE` $\to$ `B8CO` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4cube {111} -> intermediate buffer -> b8co {111}` |
| `D4CUBE` $\to$ `J4OCT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -10134.4 um. | • `d4cube {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `D4CUBE` $\to$ `J4OCT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (4.38 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -10134.4 um. | • `d4cube {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `D4CUBE` $\to$ `J4OCT` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `d4cube {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `B8CO` $\to$ `C6TT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (5.15 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -7026.0 um. | • `b8co {110} -> b8co {100} -> c6tt {100} -> c6tt {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> c6tt {111} -> c6tt {110}` |
| `B8CO` $\to$ `C6TT` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (5.94 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -6229.6 um. | • `b8co {111} -> intermediate buffer -> c6tt {111}` |
| `B8CO` $\to$ `D4TET` | {100} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `b8co {100} -> c6tt {100} -> d4tet {100}` |
| `B8CO` $\to$ `D4TET` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (5.15 mm) is smaller than penetrating profile + clearance (6.35 mm). Margin: -1199.6 um. | • `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110}` |
| `B8CO` $\to$ `D4TET` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `b8co {111} -> intermediate buffer -> d4tet {111}` |
| `B8CO` $\to$ `D4CUBE` | {100} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110} -> d4cube {100}`<br/>• `b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4cube {111} -> d4cube {100}` |
| `B8CO` $\to$ `D4CUBE` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (5.15 mm) is smaller than penetrating profile + clearance (7.05 mm). Margin: -1900.3 um. | • `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {110} -> d4cube {110}`<br/>• `b8co {110} -> b8co {100} -> c6tt {100} -> d4tet {100} -> d4tet {111} -> d4tet {110} -> d4cube {110}` |
| `B8CO` $\to$ `D4CUBE` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `b8co {111} -> intermediate buffer -> d4cube {111}` |
| `B8CO` $\to$ `J4OCT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (7.28 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -7230.0 um. | • `b8co {100} -> intermediate Kagome buffer -> j4oct {100}` |
| `B8CO` $\to$ `J4OCT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (5.15 mm) is smaller than penetrating profile + clearance (14.51 mm). Margin: -9362.3 um. | • `b8co {110} -> intermediate Kagome buffer -> j4oct {110}` |
| `B8CO` $\to$ `J4OCT` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (5.94 mm) is smaller than penetrating profile + clearance (12.01 mm). Margin: -6061.1 um. | • `b8co {111} -> intermediate Kagome buffer -> j4oct {111}` |
| `J4OCT` $\to$ `C6TT` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -5348.7 um. | • `j4oct {100} -> intermediate Kagome buffer -> c6tt {100}` |
| `J4OCT` $\to$ `C6TT` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -5348.7 um. | • `j4oct {110} -> intermediate Kagome buffer -> c6tt {110}` |
| `J4OCT` $\to$ `C6TT` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (11.15 mm) is smaller than penetrating profile + clearance (12.17 mm). Margin: -1028.5 um. | • `j4oct {111} -> intermediate Kagome buffer -> c6tt {111}` |
| `J4OCT` $\to$ `D4TET` | {100} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `j4oct {100} -> intermediate Kagome buffer -> d4tet {100}` |
| `J4OCT` $\to$ `D4TET` | {110} | [CATENATION_OR_CLEARANCE] Steric clash or unlinked boundary | • `j4oct {110} -> intermediate Kagome buffer -> d4tet {110}` |
| `J4OCT` $\to$ `D4TET` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `j4oct {111} -> intermediate Kagome buffer -> d4tet {111}` |
| `J4OCT` $\to$ `D4CUBE` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (7.05 mm). Margin: -223.0 um. | • `j4oct {100} -> intermediate Kagome buffer -> d4cube {100}` |
| `J4OCT` $\to$ `D4CUBE` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (7.05 mm). Margin: -223.0 um. | • `j4oct {110} -> intermediate Kagome buffer -> d4cube {110}` |
| `J4OCT` $\to$ `D4CUBE` | {111} | [COMMENSURABILITY] Lattices are incommensurate on habit plane (111). Minimum strain 5.72% exceeds tolerance 5.0%. | • `j4oct {111} -> intermediate Kagome buffer -> d4cube {111}` |
| `J4OCT` $\to$ `B8CO` | {100} | [APERTURE] Aperture clash on habit plane (100). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -4330.5 um. | • `j4oct {100} -> intermediate Kagome buffer -> b8co {100}` |
| `J4OCT` $\to$ `B8CO` | {110} | [APERTURE] Aperture clash on habit plane (110). Cell A aperture (6.82 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -4330.5 um. | • `j4oct {110} -> intermediate Kagome buffer -> b8co {110}` |
| `J4OCT` $\to$ `B8CO` | {111} | [APERTURE] Aperture clash on habit plane (111). Cell A aperture (11.15 mm) is smaller than penetrating profile + clearance (11.16 mm). Margin: -10.3 um. | • `j4oct {111} -> intermediate Kagome buffer -> b8co {111}` |

---

## 5. Programmatic API Integration

Precomputed entries are queried in $O(1)$ time via `graphite.explicit.interlinked.catalog`:

```python
from graphite.explicit.interlinked import (
    get_transition_entry,
    list_feasible_transitions,
    recommend_transition_path,
    assemble,
)

# 1. Instant lookup in < 1 ms without running numerical solvers
entry = get_transition_entry('d4cube', 'd4tet', habit_plane='111')
if entry.is_feasible:
    print(f'Optimal gap: {entry.gap_mm} mm, twist: {entry.twist_deg} deg, clearance: {entry.min_clearance_um} um')
else:
    print(f'Transition frustrated: {entry.failure_reason}')
    print(f'Recommended routes: {entry.recommended_routes}')

# 2. Find optimal multi-step transition sequence
path = recommend_transition_path('d4cube', 'b8co', plane_a='100', plane_b='100')
print('Recommended Path:', path)
```
