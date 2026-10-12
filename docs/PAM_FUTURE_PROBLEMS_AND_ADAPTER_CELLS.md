# Polycatenated Metamaterials (PAMs) — Future Research: Boundary Adapter Cells

This document records deferred research concepts, architectural considerations, and a future development roadmap for **Boundary Adapter Cells** and **Supermesh Homotopy** in Graphite's polycatenated metamaterials (`graphite.explicit.interlinked`).

---

## 1. Background & Problem Statement

In discrete polycatenated metamaterials (PAMs), joining two distinct lattice zones across a boundary typically relies on:
1. **Continuous Epitaxy:** Identical parent networks sharing an uninterrupted space group (e.g. D4-CUBE $\to$ D4-TET on diamond $\{111\}$ via Shockley partial dislocation).
2. **Discrete Coincidence Supercell (CSL) Matching:** Inverting Bollmann's O-lattice equation ($\mathbf{M}_1 \mathbf{A}_1 = \mathbf{R}(\theta) \mathbf{M}_2 \mathbf{A}_2$) to discover commensurate 2D supercells across a habit plane.

However, when interfacing **highly incompatible symmetries**—such as a 4-fold square $\{100\}$ plane (PCU) meeting a 3-fold/6-fold hexagonal $\{111\}$ plane (DIA)—pure discrete CSL matching may require extremely large supercell dimensions ($M_1, M_2 > 8$), resulting in prohibitive simulation domains and localized aperture pinching.

To bridge these incompatible geometries without discrete supercell scaling, the research agent proposed **Supermesh Homotopy Boundary Adapter Cells**.

---

## 2. The Supermesh Homotopy Concept

### Theoretical Formulation
The concept proposes bridging incompatible unit cells by constructing a single transition layer of **adapter unit cells** via spherical projection:
1. **Spherical Graph Embedding:** The wireframe graphs $G_A = (V_A, E_A)$ and $G_B = (V_B, E_B)$ of adjacent unit cells are projected radially onto a common unit sphere $\mathbb{S}^2$.
2. **Topological Supermesh:** A shared planar graph (supermesh) $G^* = (V^*, E^*)$ is formed on $\mathbb{S}^2$ containing the union of vertices and edges from both parent graphs.
3. **Homotopy Interpolation:** For a transition layer at fractional height $t \in (0, 1)$, vertex positions and strut radii are interpolated:
   $$\mathbf{x}_i(t) = (1 - t)\,\mathbf{x}_i^A + t\,\mathbf{x}_i^B$$
4. **Adapter Solidification:** The interpolated graph forms a bespoke intermediate cell bridging Zone A and Zone B.

---

## 3. Current Architectural Challenges & Why It Is Deferred

While theoretically intriguing, integrating dynamically morphed adapter cells presents substantial challenges within Graphite's production explicit engine:

| Challenge | Impact on Production Engine | Detail |
|-----------|-----------------------------|--------|
| **Strut Cross-Section Uniformity** | High | Unconstrained spherical homotopy stretches and warps struts non-uniformly. This compromises the calibrated wire radius $r_w$ and risks unphysical necking or local wall thinning. |
| **Clean Miter Truss Frustration** | Critical | Graphite enforces bisector-plane cut strut ends (`build_clean_miter_truss`). Morphing arbitrary angles produces non-crystallographic dihedral vertices, resulting in knife-edge degeneracies and non-manifold boolean mesh errors. |
| **3MF Instancing Invalidation** | High | Graphite achieves up to 98.6% file compression by instancing a single prototype mesh per cell type (`export_interlinked_3mf`). If adapter cells vary across the interface, every cell must be exported as an independent mesh resource, dramatically inflating file sizes. |
| **Kinematic Clearance Guarantee** | Critical | PAMs require strictly positive physical clearance ($\delta_{\min} > 0$) across all kinematic degrees of freedom. Smooth geometric homotopy does not guarantee topological catenation or non-interference under relative cage articulation. |

---

## 4. Future Research & Development Roadmap

When this problem is revisited in future phases, the following milestones should be pursued:

### Milestone A: Discrete Symmetry-Adapted Polyhedra
Instead of continuous spherical morphing, curate a catalog of **crystallographically well-defined transition polyhedra** (e.g. truncated cuboctahedra, rhombic dodecahedra, or Johnson solids) whose face symmetries natively exhibit square $\{100\}$ facets on one side and triangular $\{111\}$ facets on the opposite side.

### Milestone B: Constrained Homotopy with Miter Angle Regularization
Develop an energy-minimization solver that optimizes vertex positions under strict geometric constraints:
- Minimum dihedral angle $\alpha_{\min} \ge 35^\circ$ to prevent miter knife edges.
- Constant strut radius maintenance along all edges.
- Minimum cage aperture diameter $D_{\text{ap}} \ge D_{\text{strut}} + 2 \delta_{\text{target}}$.

### Milestone C: Topological Linking Verification for Non-Periodic Cells
Extend the Hopf linking and face-cycle auditor (`validator.py`) to handle non-periodic, asymmetric adapter rings, verifying:
- $N_A \ge 2$ and $N_B \ge 2$ simultaneous face punctures across both adapter boundaries.
- Broad-phase KD-tree clearance verification for the transition layer.

---

*Status: Archived for future research. Production grain boundaries currently use discrete CSL matching, Shockley glide translations, and coherent twin reflections.*
