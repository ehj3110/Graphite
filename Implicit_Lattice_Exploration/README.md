# Implicit Lattice Exploration

This module contains a production-grade Conformal Implicit TPMS Engine built around Signed Distance Fields (SDFs), Euclidean Distance Transforms (EDT), and Marching Cubes to generate advanced sheet lattices.

The core idea is to evaluate TPMS level-set equations directly on a dense voxel grid, combine them with CAD-aware distance fields using implicit boolean operations, and extract watertight triangle meshes without relying on explicit strut-by-strut geometry construction.

## The Four Engines

1. **Basic Gyroid (`core/gyroid_generator.py`)**
   Generates a simple, parameterized gyroid block with padded watertight boundary capping. This is the foundational SDF prototype for testing TPMS equations, pore-size parameterization, and marching-cubes extraction on a clean cube domain.

2. **Conformal Gyroid (`core/conformal_gyroid.py`)**
   Uses fast Trimesh rasterization and EDT to crop the implicit lattice to arbitrary CAD STLs without expensive pointwise ray-casting. The CAD volume is voxelized once, converted into a smooth signed-distance-like field, and intersected implicitly with the TPMS sheet field.

3. **Gradient Gyroid (`core/gradient_gyroid.py`)**
   Implements variable porosity grading by modifying the sheet thickness (effective volume fraction) across the `X`, `Y`, or `Z` axes, or from a secondary modifier STL. It uses a weight map and Smoothstep-style interpolation to create controlled transitions in density.

4. **Chirped Gyroid (`core/chirped_gyroid.py`)**
   Implements variable pore-size grading by smoothly varying the spatial frequency (unit cell size) across the domain. Instead of changing sheet thickness, it changes the local wave number so lattice density adapts continuously without intentionally severing the overall TPMS topology. This can be driven radially in pure-math diagnostics or by a modifier STL in conformal workflows.

## TPMS Equation Reference

For all equations below:

$$
k = \frac{2 \pi}{L}
$$

where `L` is the unit cell size.

To create a **sheet lattice**, evaluate:

$$
|F(x,y,z)| < t
$$

where `t` is the half-thickness / sheet-thickness proxy.

To create a **network lattice**, evaluate:

$$
F(x,y,z) < t
$$

### 1. Gyroid (G)

$$
F(x,y,z) = \sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx)
$$

### 2. Schwarz Primitive (P)

$$
F(x,y,z) = \cos(kx) + \cos(ky) + \cos(kz)
$$

### 3. Diamond (D)

$$
F(x,y,z) = \sin(kx)\sin(ky)\sin(kz) + \sin(kx)\cos(ky)\cos(kz) + \cos(kx)\sin(ky)\cos(kz) + \cos(kx)\cos(ky)\sin(kz)
$$

### 4. Neovius

$$
F(x,y,z) = 3(\cos(kx) + \cos(ky) + \cos(kz)) + 4(\cos(kx)\cos(ky)\cos(kz))
$$

### 5. Split-P (Modified Schwarz-P)

$$
F(x,y,z) =
1.1 \big[
\sin(2kx)\sin(kz)\cos(ky)
+ \sin(2ky)\sin(kx)\cos(kz)
+ \sin(2kz)\sin(ky)\cos(kx)
\big]
- 0.2 \big[
\cos(2kx)\cos(2ky)
+ \cos(2ky)\cos(2kz)
+ \cos(2kz)\cos(2kx)
\big]
- 0.4 \big[
\cos(2kx) + \cos(2ky) + \cos(2kz)
\big]
$$

