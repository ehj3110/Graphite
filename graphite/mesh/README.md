# `graphite.mesh` — capability card

## Owns

High-performance isosurface extraction (multi-threaded Flying Edges via PyVista/VTK with Marching Cubes fallback) and volume-preserving surface smoothing (Taubin alternating curvature relaxation).

## Status

**Production.**

## Public entrypoints

- `extract_isosurface_flying_edges(field, origin, spacing, level=0.0)` — extracts smooth isosurface meshes in world coordinates using `vtkFlyingEdges3D`.
- `smooth_mesh_taubin(mesh, iterations=15, lamb=0.5, nu=-0.53, inplace=False)` — alternating shrinkage/deflation filter removing voxel terracing while strictly preserving macro volume ($|\Delta V/V_0| < 0.5\%$).
- `compute_mean_curvature(mesh)` — vertex-level mean curvature evaluation.

## Does not own

TPMS or GRF scalar field evaluation (`math/`), voxelization / EDT (`geometry/`), STL/STEP export (`io/`).

## Mix-and-match

- Callers: `graphite.implicit.spinodal`, implicit TPMS pipelines, surface post-processing.
- Callees: `pyvista`, `trimesh`, `scipy.sparse`.
