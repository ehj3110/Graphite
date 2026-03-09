# Experimental Tetrahedral Lattices (Future Upgrades)

These topologies offer unique, non-standard mechanical responses and can be integrated into `topology_module.py` for advanced structural testing.

## 1. Stochastic Trabecular Tet (Biomimetic)
* **Concept:** Inspired by the random microstructures of trabecular bone.
* **Implementation:** Instead of placing the central node exactly at the mathematical centroid of the tetrahedron, a random Gaussian offset (jitter) is applied to its coordinates before connecting the struts.
* **Mechanical Advantage:** Forces cracks to take a highly tortuous path, increasing energy absorption and preventing catastrophic shear plane failure.

## 2. Chiral Tetrahedral Lattice
* **Concept:** Instead of struts meeting directly at a single central point, the central node is replaced by a small, rigid ring or a smaller central tetrahedron. Struts connect tangentially to this central geometry.
* **Mechanical Advantage:** Compression along an axis forces the internal nodes to twist. This rotation converts linear compression into rotational strain, making it an elite topology for impact and shock absorption.

## 3. Interpenetrating Compound (Hybrid) Lattices
* **Concept:** Combining multiple distinct topologies within the exact same tetrahedral scaffold. For example, mapping both the Diamond (centroid-to-adjacent-centroid) and the Vertex-to-Centroid networks in the same space.
* **Mechanical Advantage:** Provides a staged failure mechanism. The stiffer network takes the initial load and buckles first. The softer network then catches the load, creating a sequential energy absorption profile.