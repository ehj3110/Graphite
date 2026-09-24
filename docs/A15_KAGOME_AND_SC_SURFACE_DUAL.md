# Unified Engineering Specification: A15 Conformal Kagome & Simple Cubic (SC) Surface Dual

**Status:** Canonical Reference Specification  
**Packages:** `graphite/explicit/` (`a15_conformal.py`, `a15_kagome.py`, `conformal_generator.py`, `hex_topology_module.py`, `surface_preview.py`)

---

## 1. Executive Summary & Graphite Defaults

Graphite establishes two primary explicit lattice architectures for conformal structural engineering:

| Grid Modality | Architecture | Topology Infill | Boundary Treatment | Canonical File |
| :--- | :--- | :--- | :--- | :--- |
| **Tetrahedral Default** | **A15 Conformal Kagome** | 3D Pyrochlore Kagome Honeycomb ($K_4$ Octahedra) | Conformal Face-Centroid Ironing + Shared-Edge Surface Dual (`Tri_Rhombic`) | [`graphite/explicit/a15_conformal.py`](../graphite/explicit/a15_conformal.py) |
| **Hexahedral Default** | **SC Nodal Conformation + Planar Slicing Surface Dual** | Modular Rules (`octahedral`, `cubic`, `kelvin`) | Outside-Node Snapping + Universal Role Dual + Planar Slicing Contour Sweep | [`graphite/explicit/nodal_conformation.py`](../graphite/explicit/nodal_conformation.py) & [`planar_surface_sweep.py`](../graphite/explicit/planar_surface_sweep.py) |

Both architectures decouple into:
1. **Volumetric Internal Lattice**: High-strength, uniform-coordination structural beam network filling the component volume.
2. **Conformal Surface Dual**: A triangulated, boundary-conforming surface wireframe cage lying flush on the CAD boundary, providing continuous skin support, load redistribution, and attachment surfaces without jagged cantilever strut ends.
3. **Fast Interactive Preview**: A lightweight extraction path computing strictly the surface dual and boundary nodes in sub-second time without generating the internal cell volume or running multi-iteration spring relaxation.

---

## 2. A15 Conformal Kagome Architecture

```
                  +-----------------------------------+
                  |      Input CAD Mesh (Repaired)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Integer-Space A15 Tet Grid     |
                  |     (4-Clique Crystallography)    |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |  Exact Face-Centroid Cull (SDF)   |
                  |  conformal: all 4 faces inside    |
                  |  boolean: >= 1 face inside        |
                  +--------+-----------------+--------+
                           |                 |
          Fast Preview Path|                 | Full Generation Path
                           v                 v
          +-----------------------+  +-------------------------------+
          | Boundary Face Extract |  | Topological BFS Depth Tagging |
          |   (counts == 1)       |  |  (depth 0: boundary tets)     |
          +-----------+-----------+  +---------------+---------------+
                      |                              |
                      v                              v
          +-----------------------+  +-------------------------------+
          | Surface Dual Wiring   |  | Valency-Gated Surface Ironing |
          |  np.tile face edges   |  | (snap outside or degree <= 3) |
          +-----------+-----------+  +---------------+---------------+
                      |                              |
                      v                              v
          +-----------------------+  +-------------------------------+
          | CAD Surface Projection|  | Depth-Gated Jacobi Relaxation |
          |  (closest_point query)|  | (15 iters, boundary pinned)   |
          +-----------+-----------+  +---------------+---------------+
                      |                              |
                      v                              v
          +-----------------------+  +-------------------------------+
          | Sub-Second 3D Preview |  | Manifold3D Sweep & Joint Union|
          | (Dual chords + nodes) |  | (Cylinders + spherical joints)|
          +-----------------------+  +---------------+---------------+
                                                     |
                                                     v
                                     +-------------------------------+
                                     | CAD Boolean Intersection Trim |
                                     | (Flush boundary joint cutoff) |
                                     +-------------------------------+
```

### A. Crystallographic Foundation (Delaunay Elimination)
- **The Problem**: Standard Delaunay tetrahedralization produces non-deterministic, anisotropic sliver tets when applied to symmetric crystal lattices, destroying Kagome symmetry and introducing random boundary gaps.
- **The Breakthrough** ([`docs/research/a15_kagome_supercell_breakthrough.md`](research/a15_kagome_supercell_breakthrough.md)):
  The Frank-Kasper A15 phase ($Pm\bar{3}n$, space group 223) is parameterized with 8 crystallographic basis sites:
  $$\mathbf{b}_1 = (0, 0, 0), \quad \mathbf{b}_2 = \left(\tfrac{1}{2}, \tfrac{1}{2}, \tfrac{1}{2}\right)$$
  $$\mathbf{a} \in \left\{\left(\tfrac{1}{4}, 0, \tfrac{1}{2}\right), \left(\tfrac{3}{4}, 0, \tfrac{1}{2}\right), \left(\tfrac{1}{2}, \tfrac{1}{4}, 0ight), \left(\tfrac{1}{2}, \tfrac{3}{4}, 0\right), \left(0, \tfrac{1}{2}, \tfrac{1}{4}\right), \left(0, \tfrac{1}{2}, \tfrac{3}{4}\right)\right\}$$
- **4-Clique Reconstruction**:
  Primary bonds are formed via distance threshold $d_{ij} \le 0.62 \times L_{\text{cell}}$.
  Graph-theoretic search for mutually connected 4-cliques extracts the exact irregular space-filling tetrahedra (28 tets in $1\times 1\times 1$, 296 tets in $2\times 2\times 2$).

### B. Face-Centroid Kagome Mapping
- **Nodes**: Each Kagome node is placed at the centroid of a triangular tetrahedral face:
  $$\mathbf{c}_{\triangle} = \frac{1}{3}(\mathbf{v}_1 + \mathbf{v}_2 + \mathbf{v}_3)$$
- **Core Struts ($K_4$ Inverted Octahedra)**:
  Within each tetrahedron, its 4 face-centroid nodes form an inscribed inverted tetrahedron (6 struts per tet):
  $$\text{Struts}_{\text{tet}} = \{(c_i, c_j) \mid 0 \le i < j \le 3\}$$
  Shared faces across adjacent tets share the centroid node identically, forming the isotropic 3D pyrochlore Kagome lattice.

### C. Boundary Face Extraction & Surface Dual Wiring
- **Exposed Boundary Faces**: Faces belonging to exactly one surviving tetrahedron ($count = 1$).
- **Topological Surface Dual Wiring**:
  Two boundary faces sharing a boundary edge connect their face centroids:
  ```python
  edges_01 = b_faces[:, [0, 1]]
  edges_12 = b_faces[:, [1, 2]]
  edges_02 = b_faces[:, [0, 2]]
  all_b_edges = np.vstack([edges_01, edges_12, edges_02])
  
  # CRITICAL: Use np.tile, NOT np.repeat
  face_owner = np.tile(np.arange(len(b_faces)), 3)
  ```
  This creates the **Tri_Rhombic / cyan Kagome surface dual**, providing 100% boundary triangulation over the tetrahedral boundary.

### D. Conformal Ironing, Valency Snapping & Jacobi Relaxation
1. **Valency-Gated Ironing**:
   - Outside boundary nodes: Always snapped to CAD surface via `project_to_cad_surface`.
   - Inside boundary nodes: Snapped only if degree $\le 3$ (low-valency nodes that would otherwise hang unsupported).
2. **Depth-Gated Jacobi Relaxation**:
   - Tet depth 0 = boundary tets. Internal tets receive topological BFS depths $1, 2, \dots$.
   - Depth 0 nodes are **pinned** (fixed on the CAD surface).
   - Internal nodes (depth $\ge 1$) relax under linear spring forces for 15 iterations (step size $\alpha = 0.5$):
     $$\mathbf{x}_i^{(k+1)} = (1 - \alpha)\mathbf{x}_i^{(k)} + \alpha \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j^{(k)}$$
   - Internal layers radially absorb boundary displacement, preventing strut buckling and preserving interior cell uniformity.

---

## 3. Modular Simple Cubic (SC) Conformal & Surface Dual

### A. Hex Scaffold & 50% Volume Fraction Gating
- Cartesian bounding grid of hexahedral cells ($L \times L \times L$).
- **50% VF Threshold**: Boundary hexes are evaluated across their 6 face centroids. Hexes with $< 50\%$ internal volume are dropped. This prevents "pancaking" on convex CAD boundaries.

### B. Hex Cage Morph & Boundary Projection
- Boundary quads (quads with $count = 1$) are extracted.
- Unique corner nodes of all exposed quads are projected to the CAD boundary (algebraic projection on spheres, closest-point on organic meshes).
- Sub-surface transition layers are smoothed using Laplacian relaxation with depth 0 corner nodes fixed.

### C. Modular Volume Topology Rules
Hex cells are populated via trilinear brick interpolation using [`graphite/explicit/hex_topology_module.py`](../graphite/explicit/hex_topology_module.py):
- `octahedral`: 12 face-diagonal struts per cube forming high-shear-modulus octahedral trusses.
- `cubic`: 12 perimeter cube edges.
- `kelvin`: Truncated octahedron (Kelvin tetrakaidecahedron) infill.

### D. Hex Surface Dual Cage (`generate_hex_surface_dual_cage`)
- Each exposed boundary quad $[v_0, v_1, v_2, v_3]$ receives a face centroid hub $\mathbf{c} = \frac{1}{4}\sum v_i$.
- Corner-to-hub struts $[v_i, \mathbf{c}]$ and perimeter chords triangulate the boundary quad, forming a watertight conformal dual cage.

---

## 4. Solid Geometry & Boolean Trim (Ball & Adapter Workflows)

### A. Part2 Adapter (`Part2_Adapter.STL`)
Documented in [`docs/CONFORMAL_ENGINE_V2.md`](CONFORMAL_ENGINE_V2.md) and [`tests/run_part2_boolean_trim.py`](../tests/run_part2_boolean_trim.py):
1. Conformal topology generation: Core Kagome struts (`red_struts`) + Surface dual chords (`cyan_struts`).
2. Solid Sweep: Cylinders along struts + spherical joints at all nodes ($r_{\text{joint}} = 1.05 \times r_{\text{strut}}$).
3. CSG Union: Manifold3D binary-tree union (`union_lattice_with_spherical_joints`).
4. **CAD Boolean Intersection Trim (`boolean_intersect_with_cad`)**:
   $$\mathcal{M}_{\text{final}} = \mathcal{M}_{\text{lattice}} \cap \mathcal{M}_{\text{CAD}}$$
   Flushes all outward-protruding spherical joints and strut ends cleanly against the CAD solid.

### B. Baseball Sphere (Ø74 mm)
Documented in [`docs/BALLS_BASEBALL.md`](BALLS_BASEBALL.md):
- **Cylindrical Path**: Cylinders + spherical joints followed by `boolean_intersect_with_cad` against the sphere.
- **Square Strut Path**: Surface-normal rectangular prisms ($W=1.6\text{ mm}, T=5.0\text{ mm}$ via `sweep_square_straight_struts`) followed by Boolean intersection with an inset sphere of radius $R - 0.25\text{ mm}$ to flush joints.
- **Topological Identity**: The approved `Tri_Rhombic` pattern in the catalog is the exact topological equivalent of the A15 Kagome cyan surface dual.

### C. Special Cube (`A15_SpecialCube`)
Documented in [`graphite/explicit/a15_kagome.py`](../graphite/explicit/a15_kagome.py):
- Quarter-cell alignment ($0.25 \times L_{\text{cell}}$).
- Valency-gated planar snaps on box faces ($X, Y, Z = 0, L$).
- Shared-edge dual extraction without proximity distortion.

---

## 5. Decoupled Fast Surface Dual 3D Preview

To eliminate the 12–25 second latency of running full volume stamping and 15 Jacobi iterations during interactive design, the **"Update 3D Preview"** button executes decoupled surface-only extraction:

| Metric | SC Octahedral (20mm Cube) | A15 Kagome (20mm Cube) | BaseRing (SC Octahedral) |
| :--- | :--- | :--- | :--- |
| **Prior Scaffold Time** | $1.20\text{ s}$ | $11.95\text{ s}$ | $24.80\text{ s}$ |
| **Fast Surface Dual Time** | **$0.176\text{ s}$** | **$2.633\text{ s}$** | **$5.515\text{ s}$** |
| **Speedup Factor** | **$6.8\times$** | **$4.5\times$** | **$4.5\times$** |
| **Preview Output** | 576 dual chords, 194 boundary nodes | 1,080 dual chords, 720 boundary nodes | 4,362 dual chords, 1,419 boundary nodes |

### Critical Engineering Pitfall: `np.repeat` vs `np.tile`
When vectorizing boundary face edge extraction:
- `all_b_edges = np.vstack([edges_01, edges_12, edges_02])` stacks three arrays of length $N$.
- **`np.repeat(np.arange(N), 3)`** produces $[0, 0, 0, 1, 1, 1, \dots]$, incorrectly assigning edge $k$ to face $\lfloor k/3 \rfloor$. This maps edges from face 0 to face $N/3$ on the opposite side of the CAD part, generating spurious diagonal chords $>25\text{ mm}$ cutting across the body.
- **`np.tile(np.arange(N), 3)`** produces $[0, 1, \dots, N-1, 0, 1, \dots, N-1, \dots]$, correctly mapping every edge to its parent face and ensuring all dual struts remain strictly local ($1.50 - 1.67\text{ mm}$).

---

## 6. API Quick-Reference & Code Index

| Operation | Function / API | File Path |
| :--- | :--- | :--- |
| **A15 Production Lattice** | `generate_a15_conformal_lattice` | [`graphite/explicit/a15_conformal.py`](../graphite/explicit/a15_conformal.py) |
| **SC Production Lattice** | `generate_sc_conformal_lattice` / `generate_conformal_lattice` | [`graphite/explicit/nodal_conformation.py`](../graphite/explicit/nodal_conformation.py) |
| **Fast A15 Surface Dual** | `extract_fast_a15_surface_dual` | [`graphite/ui/surface_preview.py`](../graphite/ui/surface_preview.py) |
| **Fast SC Surface Dual** | `extract_fast_sc_surface_dual` | [`graphite/ui/surface_preview.py`](../graphite/ui/surface_preview.py) |
| **Unified 3D Preview** | `generate_explicit_preview` | [`graphite/ui/surface_preview.py`](../graphite/ui/surface_preview.py) |
| **Hex Dual Cage Generator**| `generate_hex_surface_dual_cage`| [`graphite/explicit/hex_topology_module.py`](../graphite/explicit/hex_topology_module.py) |
| **CSG Spherical Joints** | `union_lattice_with_spherical_joints` | [`graphite/explicit/geometry_module.py`](../graphite/explicit/geometry_module.py) |
| **CAD Boolean Trim** | `boolean_intersect_with_cad` | [`graphite/explicit/geometry_module.py`](../graphite/explicit/geometry_module.py) |
| **Analytical Sizing** | `solve_sizing` / `calculate_strut_radius` | [`graphite/explicit/sizing_solver.py`](../graphite/explicit/sizing_solver.py) |
