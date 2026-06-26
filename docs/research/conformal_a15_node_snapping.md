# Conformal A15 Kagome Lattice Generation via Direct Node Snapping

This document describes the design and mathematical formulation of the conformal A15 Kagome lattice generation pipeline in Graphite. This architecture resolves the dual challenges of boundary culling and boundary-snapping distortions without shearing the highly-ordered internal struts.

---

## 1. The Core Problem

High-performance aerospace and biomedical components require structured lattices (such as A15 or C15 Kagome topologies) that conform to arbitrary 3D boundary surfaces while maintaining perfect symmetry and strut order. 

Standard conformal meshing techniques fail for these lattices in two distinct ways:
1. **Delaunay-Based Meshing (Asymmetry):** Delaunay tetrahedralization of boundary-conformed points results in highly anisotropic, randomly oriented tetrahedra. Mapping structured unit cells (like A15) onto a Delaunay mesh destroys the crystalline symmetry, leading to unpredictable mechanical behavior and inconsistent pore sizing.
2. **Macro-Hex Trilinear Warping (Shear Distortion):** In this approach, a regular hexahedral mesh is conformed to the boundary using boundary snapping and Laplacian smoothing. The unit cell geometry is then trilinearly warped from the unit cube $[0, 1]^3$ to each physical warped hex. While this preserves topology, snapped boundary hexes are often sheared, flattened, or stretched. Warping the A15 rule coordinates through these deformed hexes introduces severe bending and shear on the internal struts (creating "spaghetti" struts), which dramatically reduces the buckling resistance of the lattice.

---

## 2. The Base Scaffold: Loose Bounding Shell

To ensure that no boundary-straddling unit cells are prematurely culled, we generate a single Cartesian hexahedral grid representing the bounding box of the part. Instead of filtering this grid using a strict $SDF \le 0.0$ threshold at the cell centroids, we apply a loose bounding shell:

$$\text{Keep Hex} \iff \text{SDF}(\mathbf{c}_{\text{hex}}) \le \text{CELL\_SIZE} \times \frac{\sqrt{3}}{2} \approx \text{CELL\_SIZE} \times 0.866$$

Where $\mathbf{c}_{\text{hex}}$ is the centroid of the hex cell. This loose shell ensures that any hex cell that intersects or grazes the boundary surface is fully preserved. This is a critical prerequisite for sub-cell gating: culling must be performed at the sub-cell level, not the macro-hex level.

---

## 3. Strict Sub-Cell Gating (75% Volume Inclusion)

Within each preserved macro-hex cell, we tile the canonical A15 atomic basis and perform a deterministic 4-clique reconstruction to extract the individual tetrahedra (sub-cells).

To prevent under-filled or jagged elements at the boundary, we enforce a strict **75% Volume Inclusion Gating** policy on each tetrahedron $T$ with vertices $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\}$ and centroid $\mathbf{c}_T$:

$$\text{Keep } T \iff \left(\bigwedge_{i=1}^4 \text{SDF}(\mathbf{v}_i) \le 0.0\right) \lor \left(\text{SDF}(\mathbf{c}_T) \le -0.15 \times \text{CELL\_SIZE}\right)$$

* **All-Vertices Inside:** If all four vertices are inside or on the boundary, the entire tetrahedron is guaranteed to be within the part.
* **Deep Centroid:** If the centroid is sufficiently deep inside the part, the tetrahedron is kept even if some corners project slightly outside.
This stricter policy culls tetrahedra that only marginally intersect the part boundary, preventing thin, weak, or isolated struts at the surface.

---

## 4. Direct Node Snapping

Instead of conforming the macro-hexes and warping the struts, we generate the topology using the rigid, undeformed Cartesian grid (Baseline 2). We then identify the boundary nodes of the resulting lattice and project them directly onto the zero-isosurface of the SDF.

### A. Boundary Node Identification
We identify boundary nodes using a combination of two mathematical criteria:

1. **Topological Criterion (Severed Struts):**
   Let $V_{\text{raw}}$ and $E_{\text{raw}}$ be the set of nodes and struts generated without sub-cell gating (Baseline 1). Let $V_{\text{gated}}$ and $E_{\text{gated}}$ be the gated subset (Baseline 2). We define a mapping $f: V_{\text{gated}} \to V_{\text{raw}}$ using coordinate matching. A strut $(u, v) \in E_{\text{raw}}$ is defined as **severed** if:
   $$(u \in f(V_{\text{gated}}) \land v \notin f(V_{\text{gated}})) \lor (v \in f(V_{\text{gated}}) \land u \notin f(V_{\text{gated}}))$$
   The surviving endpoint of any severed strut is classified as a topological boundary node.
2. **Geometric Criterion (SDF Proximity Band):**
   Any node $\mathbf{x} \in V_{\text{gated}}$ that lies within a narrow band of the boundary is classified as a geometric boundary node:
   $$|\text{SDF}(\mathbf{x})| \le 0.5 \times L_{\text{strut}}$$
   where $L_{\text{strut}}$ is the average strut length of the lattice.

The set of boundary nodes $V_{\text{boundary}}$ is the union of these two sets.

### B. Projection and Distance Guard
For each node $\mathbf{x} \in V_{\text{boundary}}$, we compute the closest point on the target surface $\mathbf{x}_{\text{closest}}$. To prevent severe stretching of struts connected to outlier boundary nodes, we enforce a distance guard:

$$\mathbf{x}_{\text{snapped}} = \begin{cases} \mathbf{x}_{\text{closest}} & \text{if } \|\mathbf{x}_{\text{closest}} - \mathbf{x}\| \le 1.2 \times L_{\text{strut}} \\ \mathbf{x} & \text{otherwise} \end{cases}$$

This snaps the outer skin of the A15 Kagome lattice flush to the boundary while leaving the internal struts perfectly ordered and unstrained.
