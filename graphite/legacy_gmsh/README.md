# `graphite.legacy_gmsh` — capability card

## Owns

Archived GMSH tet/hex scaffolders and older Conformal Dual hex synthesis APIs kept for reference.

## Status

**Legacy.** New work should use GMSH-free `graphite/explicit/` (A15 / modular SC).

## Public entrypoints

Import modules directly (no package `__init__`):

- `scaffold_module.generate_conformal_scaffold`
- Hex: `generate_conformed_hex_scaffold`, `synthesize_conformal_dual_lattice`, `generate_cropped_hex_scaffold`
- `build_hex_surface_skin`, `validate_hex_inversion`

## Does not own

Current A15/SC path (`explicit/`), FEA Gmsh meshing (`aristo/`).

## Mix-and-match

- Some Tier-1 export scripts and docs still describe Conformal Dual using these names — check whether the live call site imports `legacy_gmsh` or a re-export before editing.

## Read next

1. [docs/CONFORMAL_DUAL_HEX.md](../../docs/CONFORMAL_DUAL_HEX.md)
2. [graphite/explicit/README.md](../explicit/README.md) (current defaults)

## Do not open first

- As the default place to implement new hex features
