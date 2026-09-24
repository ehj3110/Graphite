# `graphite.geometry` — capability card

## Owns

Shared geometry helpers: CAD voxel / EDT masks, primitives, surface picking, Vedo BC preprocessor.

## Status

**Production** helpers (package `__init__` may be thin — import modules directly).

## Public entrypoints

- `voxelize_mesh_and_edt`, `voxelize_cylinder_slab_and_edt`
- `axis_aligned_box_sdf`, `generate_primitive`
- `compute_face_surface_ids`, `visualize_surfaces`
- `BCPreprocessor` (`vedo_preprocessor`)

## Does not own

TPMS evaluation (`math/`), Manifold strut primitives (`explicit.geometry_module`), Gmsh volume meshing (`aristo/` / `legacy_gmsh/`).

## Mix-and-match

- Heavy use from `implicit/` for domain masks; TO sandbox may use Vedo preprocessor.

## Read next

1. [docs/IMPLICIT_ENGINE.md](../../docs/IMPLICIT_ENGINE.md) (masking / domain)
2. [graphite/math/README.md](../math/README.md) for scalar fields

## Do not open first

- `experiments/` copies of older voxelizers unless tasked
