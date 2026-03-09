# Graphite-Implicit: Continuous Surface Generation Engine

## 1. Project Objective
Build a dedicated extension for "Graphite" that generates Implicit Surfaces—specifically Triply Periodic Minimal Surfaces (TPMS) and Spinodal structures. 
Unlike the core Graphite engine (which uses discrete nodes and struts), Graphite-Implicit evaluates volumetric mathematical fields. The tool will evaluate a 3D equation, extract a watertight mesh using a zero-level set, and Boolean-intersect that mesh with an arbitrary boundary STL.

## 2. The Tech Stack & Paradigm Shift
This extension strictly abandons gmsh and discrete scaffolding. 
* manifold3d (Level Sets): The core engine. Use manifold3d.LevelSet() to generate the mesh directly from a 3D scalar field. This bypasses fragile marching cubes algorithms and guarantees a manifold, watertight output.
* numpy: Used strictly for generating the 3D grid of scalar values (the mathematical field) passed into the Manifold LevelSet engine.
* trimesh: Retained for STL import, bounding box calculation, and final volume validation.

## 3. Mathematical Library (The Equations)
The extension must support evaluating these implicit equations across a parameterized (X, Y, Z) coordinate grid. The variable 't' (isovalue) is adjusted by the solver to hit the target Solid Fraction.

* Gyroid (TPMS): f(x,y,z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x) - t
* Schwarz Diamond (TPMS): f(x,y,z) = sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z) - t
* Spinodal (Gaussian Random Field): Summation of N random standing waves (cos(k*r + phi)) - t

## 4. Modular Architecture for Implicit Surfaces

### Module 1: equation_library.py
Logic: Contains pure NumPy vectorized functions for the TPMS and Spinodal equations. Intakes a 3D grid of coordinates and an isovalue 't', returns a 3D array of scalar values.

### Module 2: implicit_engine.py
Logic: Reads the user's STL bounding box, generates a 3D coordinate grid (np.meshgrid), passes the grid to equation_library.py, and feeds the resulting scalar field into manifold3d.LevelSet().
Constraint: Must expose a resolution parameter. Grid density must scale reasonably to prevent RAM crashes.

### Module 3: boolean_trim.py
Logic: Takes the raw cubic TPMS/Spinodal block and uses manifold3d.boolean(op='intersect') to clip it precisely to the user's original STL boundary.

### Module 4: implicit_solver.py
Logic: Runs a Newton-Raphson loop. Instead of adjusting strut radius, it adjusts the isovalue 't' (thickening or thinning the sheet) until the trimmed mesh volume matches the Target Volume.

## 5. Strict Anti-Patterns (DO NOT DO THESE)
* NO gmsh: Do not use the gmsh library for this extension. There are no tetrahedrons here.
* NO custom marching cubes: Do not import skimage.measure.marching_cubes. Rely entirely on manifold3d.LevelSet() to generate the mesh.