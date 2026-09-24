# **Industry Analysis: Carbon3D Meshing Strategies**

**Relevance to Graphite Engine: Conformal Boundaries & Surface Duals**

Based on an analysis of Carbon3D's Design Engine documentation, we can draw several critical insights into how they handle complex shapes, prevent element distortion, and generate their signature "clean" surface cages. Their approach validates many of the features we have built into the Graphite engine, while also highlighting alternative "surface-first" paradigms.

## **1\. Surface Tolerance (The Volume Fraction Gate)**

**Reference:** [Carbon3D Surface Tolerance](https://learn.carbon3d.com/software/surface-tolerance)

Carbon utilizes a "Surface Tolerance" parameter that dictates how strictly the lattice conforms to the true CAD boundary.

**How it works:** In explicit voxel/hex meshing, boundary elements inevitably straddle the part boundary. Surface Tolerance acts as a Volume Fraction (VF) gate. A "loose" tolerance allows elements that are mostly in empty space to be kept and stretched inward, while a "strict" tolerance culls those elements, leaving only deeply embedded cells.

**Takeaway for Graphite:**

Carbon's reliance on this confirms that there is no "magic bullet" for Cartesian boundary straddling. Our implementation of **Sub-Cell (Tetrahedral) Gating** and **VF Culling Thresholds** (vf\_cull\_threshold) is mathematically identical to this feature. By setting a strict 75% inclusion threshold, we force the engine to drop outer elements rather than stretch them, exactly mirroring Carbon's best practices for preserving internal lattice symmetry.

## **2\. Surface Isotropy (The "Mesh-Inwards" Paradigm)**

**Reference:** [Carbon3D Surface Isotropy](https://learn.carbon3d.com/software/surface-isotropy)

Carbon offers a highly computationally expensive feature called "Surface Isotropy." When enabled, the surface dual (the cage) perfectly wraps complex, organic curvatures with equidistant, highly uniform struts, avoiding the "stair-stepping" or stretching normally associated with Cartesian grids.

**How it works:**

The computational time warning implies Carbon is using a **"Mesh-Inwards" (or Surface-First) algorithm** paired with adaptive relaxation.

Instead of taking a 3D block of hexes and projecting them *out* to the surface, the engine likely:

1. Generates a pristine, highly isotropic 2D mesh (triangles or quads) directly on the CAD surface using geodesic mapping.  
2. Uses this 2D surface mesh as the absolute constraint (the cage).  
3. Propagates the 3D hex or tet mesh *inwards* from the surface, using adaptive relaxation/smoothing to stitch the beautiful boundary cage to the rigid internal crystal lattice.

**Takeaway for Graphite:**

This explains why Carbon's cages look so flawless on complex shapes—they are literally meshing the surface first. Our **Direct Node Snapping** strategy is an excellent, lightweight approximation of this. To take Graphite to Carbon's level of isotropy in the future, we would need to implement an adaptive transitional layer: generating a geodesic surface mesh first, and mathematically blending it into the rigid A15 macro-grid beneath it.

## **3\. Hex Mesh Deformation (The "Squish" Factor)**

**Reference:** [Carbon3D Hex Mesh](https://learn.carbon3d.com/software/hex-mesh)

Carbon's documentation explicitly acknowledges the limitations of Hexahedral (Hex) meshes when conforming to boundaries. They warn users that pulling hex elements too far to meet a surface will "squish" or distort them, which can compromise the mechanical properties of the struts inside.

**How it works:**

Interestingly, Carbon relies primarily on the *user's awareness* to fix this. They provide the tools (like Surface Tolerance), but expect the user to tune the mesh resolution and orientation so that the hexes don't have to stretch too far. They do not magically prevent trilinear shear—they just warn you it exists.

**Takeaway for Graphite:**

This is a massive validation of the Graphite engine. Extreme trilinear shear ("spaghetti struts") is a fundamental mathematical limitation of mapping cubic spaces onto spherical geometries.

However, Graphite is arguably *safer* in this regard. While Carbon relies on user intuition to prevent inverted elements, Graphite has built-in **Jacobian-Gated Line Searches** (min\_jacobian\_proxy). Our engine programmatically refuses to snap a node if it will crush the volume of the parent hex cell below a safe threshold, automating the exact manual tuning Carbon expects of its users.

## **Conclusion & Strategic Roadmap**

Carbon3D achieves its results not by defying computational geometry, but by combining strict culling, surface-first meshing, and careful grid alignment.

For the Graphite explicit engine, our current architecture is highly competitive:

* Our **Stricter Sub-Cell Gating** perfectly mimics their Surface Tolerance.  
* Our **Direct Node Snapping** provides a clean, flush skin without the extreme computational overhead of true Surface Isotropy.  
* Our **Jacobian Gates** natively protect against the hex "squish" that Carbon warns about.

**Next Steps for Surface Duals:** To replicate Carbon's explicit surface cage using our current tools, we should use the undeformed macro-staircase. By extracting the flat, exposed faces of our rigid bounding hexes *before* snapping, we can instantly generate a predictable 2D cage topology. We can then pass the nodes of that cage through our Direct Node Snapping pipeline, achieving a conformal, closed shell that envelopes the A15 core.