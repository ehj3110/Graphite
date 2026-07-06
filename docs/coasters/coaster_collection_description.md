# Mathematical & Crystal Lattice Coaster Collection

Welcome to this unique collection of mathematical and crystal lattice coasters! This set provides a variety of highly detailed, structurally fascinating designs perfect for 3D printing, ranging from smooth implicit surfaces to complex atomic crystal arrangements. 

## What Are These Models?

This collection features over 150 unique, 3D-printable coasters generated using advanced procedural geometry. The designs are broken down into three main categories:
1. **Triply Periodic Minimal Surfaces (TPMS):** Beautiful, continuous, non-intersecting surfaces like the Gyroid, Schwarz P, and Schwarz D.
2. **Explicit Strut Lattices:** Strong, engineered geometric patterns including square grids, triangles, hexagons, and Voronoi tessellations.
3. **Crystal Structures (Frank-Kasper Phases):** Highly complex patterns derived from the atomic arrangements of the A15 and C15 crystal lattices.

## How Were They Made?

Rather than using traditional 3D modeling software, these coasters were algorithmically generated using Python (`trimesh` and `manifold3d`). 

To ensure the best possible 3D printing experience, we developed a **2D Projection and Extrusion** methodology. Often, generating complex 3D lattices (like marching cubes over implicit functions) results in bumpy, voxelized top and bottom surfaces that print poorly and don't function well as flat coasters. 

Instead, our script calculates the exact mathematical cross-section of the lattice at specific Z-heights. We extract these 2D paths, filter out any disconnected "hanging struts" that would fail during printing, and extrude the 2D path perfectly vertically into a robust, watertight 3D solid. This guarantees perfectly flat, smooth top and bottom layers for your glass to sit on, while preserving the stunning internal geometry of the lattice.

## Understanding the Crystal Structures (A15 & C15)

The highlight of this collection is the inclusion of the **A15** and **C15 Laves phase** structures. These are known as Frank-Kasper phases—complex atomic arrangements found in certain metal alloys.

*   **A15 Structure:** This lattice is known for its high symmetry and is historically significant in superconductivity. The unit cell consists of atoms arranged in a body-centered cubic-like structure with orthogonal chains of atoms running across the faces. For these coasters, we used a unit cell size of $25\text{mm}$.
*   **C15 Structure:** A face-centered cubic Laves phase structure, featuring a highly complex, interconnected network of tetrahedra. It's notoriously difficult to model accurately. For these coasters, we scaled the unit cell to a massive $50\text{mm}$ to truly capture its intricate, diamond-like sub-network across the coaster's surface.

For both structures, we sliced the unit cells at multiple different heights (offsets), resulting in drastically different geometric patterns depending on where the crystal was intersected!

## File and Folder Structure

The download is organized logically so you can easily find the design you want to print:

*   `tpms/` - Contains the Gyroid, Schwarz P, and Schwarz D coasters.
*   `explicit/` - Contains the standard geometric lattices (Square, Triangle, Hexagon, Octagon).
*   `voronoi/` - Contains the Voronoi patterns, categorized by density (large, medium, small).
*   `a15/` - Contains 3 unique coasters (`v1`, `v2`, `v3`) based on different cross-sectional heights of the A15 crystal structure.
*   `c15/` - Contains 3 unique coasters (`v1`, `v2`, `v3`) based on different cross-sectional heights of the C15 crystal structure.

## Printing Recommendations

*   **Material:** PLA, PETG, or ABS are all excellent choices.
*   **Layer Height:** 0.2mm works perfectly.
*   **Infill:** 100% (These are solid struts, so solid infill ensures structural integrity).
*   **Supports:** None required! The 2D extrusion method ensures all features are perfectly printable straight on the build plate.

Happy printing!
