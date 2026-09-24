# `graphite.math` — capability card

## Owns

Shared TPMS / phase / texture / woodpile **scalar-field math** (no meshing, no FEA).

## Status

**Production.**

## Public entrypoints

- `evaluate_tpms`, `evaluate_tpms_phase`, `calculate_integrated_phase`
- Named fields: `gyroid`, `schwarz_p`, `split_p`, `bump_field`, `knurl_field`, `spinodal_spectral_field`, `triplanar_map`
- Woodpile helpers: `evaluate_woodpile`, `compute_woodpile_xy_origin`

## Does not own

Voxel grids & CAD SDF (`geometry/`), lattice engines (`implicit/` / `explicit/`), calibration loops (`implicit` calibration modules).

## Mix-and-match

- Callers: `implicit/` generators, grading / lofted experiments that need phase math.

## Read next

1. [docs/LATTICE_MATH_ARCHITECTURE.md](../../docs/LATTICE_MATH_ARCHITECTURE.md)
2. [docs/IMPLICIT_GRADING_AND_TEXTURES.md](../../docs/IMPLICIT_GRADING_AND_TEXTURES.md) (textures use `graphite/math/textures.py`)
3. [graphite/implicit/README.md](../implicit/README.md)

## Do not open first

- Full calibration workflow docs unless changing MIS / seed tables
