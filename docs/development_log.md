# Graphite Engine Status: Day 1 Checkpoint

## 1. Project Overview
The **Graphite Engine** is a custom pipeline designed to generate research-grade, conformal tetrahedral lattices for fracture mechanics. The goal is to move beyond simple repeating blocks to "Dual-Graph" lattices (Voronoi/Kagome) that follow the geometry of complex parts.

## 2. Core Architecture
The engine is split into a modular pipeline to ensure that geometry, topology, and physics can be updated independently:
* **Module 1 (Scaffold):** GMSH Integration. Generates the tetrahedral "skeleton."
* **Module 2 (Topology):** The "Recipe" layer. Converts tetrahedrons into 4 types: **Rhombic, Icosahedral, Voronoi, and Kagome.**
* **Module 3 (Solver):** A bisection-based volume optimizer that hits the target Solid Fraction (e.g., 12.5% or 10.0%).
* **Module 4 (Geometry):** Manifold3D Integration. Handles the heavy Boolean unions to create 100% watertight STLs.

## 3. Key Breakthroughs (Day 1)
### 3D Topology & Connectivity
* **Global Adjacency Mapping:** Moved from local element math to a global face-to-tetrahedron map. This fixed the "broken handshake" problem where neighboring struts didn't meet at shared faces.
* **Kagome/Voronoi Straightening:** Implemented direct Centroid-to-Centroid logic. Internal struts are now mathematically straight vectors, eliminating the "zig-zag" look caused by unstructured tetrahedrons.
* **The Watershed Filter:** Integrated `scipy.sparse.csgraph` to identify and prune "floater" struts, ensuring 100% connectivity for 3D printing.

### Performance & Speed
* **Smart Seeding:** Implemented a geometric radius guess ($r \approx \sqrt{V / \pi L}$), cutting solver iterations from 10+ down to ~4.
* **Audit Box (RVE) Solver:** Created a "Representative Volume Element" path that solves for density using only the center 40% of the lattice, achieving a **~3x speedup** on complex variants.

## 4. Current Status of Lattices
| Variant | Logic | Status |
| :--- | :--- | :--- |
| **Rhombic** | Vertex-to-Centroid | Stable & Manifold |
| **Icosahedral** | Edge-to-Edge | Stable & Manifold |
| **Voronoi** | Centroid-to-Centroid | Straight internal struts; Skin pending |
| **Kagome** | Face-Centroid Pairwise | Straight internal struts; Skin pending |

## 5. Ongoing Challenges
* **Internal Singularities:** Unstructured GMSH meshing causes "nests" or high-valence nodes in the center of the volume.
* **Surface Skin:** Previous "Y-Skin" (Face-to-Edge) logic caused kinks at the boundary. The current goal is to transition to a **Tapered Surface Dual** (Direct Centroid-to-Centroid on surface faces) or a **Chamfered Edge** look.

---

## Day 2 Objectives
1.  **Uniformity Fields:** Refine GMSH "Box Fields" and the **HXT algorithm** to ensure consistent internal density across the 20mm cube.
2.  **The "Chamfered" Surface Dual:** Implement the new surface skin logic that shortcuts corner edges for an engineered, tapered look.
3.  **The Supercell Pivot:** Experiment with **Structured Tetrahedralization** (6-Tet or 24-Tet cubes) to replicate "Carbon-style" equilateral patterns.