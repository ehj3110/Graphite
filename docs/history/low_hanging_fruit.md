# Low Hanging Fruit for "Graphite"

These features leverage the existing tech stack (trimesh, gmsh, manifold3d) to add high-value functionality for fracture mechanics and resin printing research with minimal computational overhead.

## 1. Attractor-Based Density Gradients
* **Concept:** Allow the lattice to naturally become denser around high-stress areas or specific coordinates.
* **Implementation:** Expose `gmsh`'s built-in field manipulation. The user inputs an [X, Y, Z] coordinate and a radius. Use `gmsh.model.mesh.field.add("Distance")` and `gmsh.model.mesh.field.add("Threshold")` to force the tetrahedral scaffold to refine (generate smaller tets) near that point. The downstream topology mapping will naturally follow this density gradient.

## 2. Solid Skinning (Hybrid Lattices)
* **Concept:** Generate a solid outer shell (or solid grip-points) integrated seamlessly with the internal lattice so testing grips (e.g., Instron) do not crush the exposed struts.
* **Implementation:** 1. Take the input STL.
    2. Use `trimesh` or `manifold3d` to create an inward offset (hollow the mesh), leaving a solid shell of thickness T.
    3. Feed the hollowed inner cavity to GMSH to generate the lattice.
    4. In the final `geometry_module`, perform `manifold3d.batch_union()` on both the lattice AND the solid shell.

## 3. Dedicated "Preview Mode"
* **Concept:** A fast visualization tool to verify unit cell size and topology without waiting for the heavy Boolean operations.
* **Implementation:** Add a `--preview` flag. If triggered, the script halts before calling `manifold3d`. It extracts the 1D line segments (the skeleton) from the `topology_module` and plots them using `pyvista`. The user gets an instant 3D wireframe popup.

## 4. The "Coupon Maker" (Auto-Boundaries)
* **Concept:** Automatically generate standard ASTM test coupons (cubes, cylinders, dogbones) in memory if the user does not provide an external STL.
* **Implementation:** Add a `generate_boundary` helper function. If no filepath is provided, use `trimesh.creation.cylinder()` or `trimesh.creation.box()` to instantly generate the boundary STL and pass it straight into the pipeline.

## 5. Lattice Statistics Export
* **Concept:** Automatically output geometric data required for resin curing profiles and theoretical mass calculations.
* **Implementation:** At the end of the script, save a `graphite_stats.txt` or `.json` file to the output directory containing:
    * Total Nodes: `len(unique_nodes)`
    * Total Struts: `len(struts)`
    * Target Solid Fraction
    * Achieved Solid Fraction
    * Final Strut Radius
    * **Average Pore Size:** Calculated by taking the bounding volume of a standard unit cell, subtracting the solid strut volume within it, and approximating the diameter of the inscribed sphere of the remaining void space.