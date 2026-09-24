# `graphite.meshing` — capability card

## Owns

Older GMSH-based conformal / Kagome demo scripts (scaffold → topology → manifold sweep).

## Status

**Legacy / lab.** Not the current GMSH-free explicit default (see `graphite/explicit/`).

## Public entrypoints

Script-style APIs (import modules directly): `generate_base_scaffold`, `generate_kagome_dual_struts`, `sweep_to_manifold`, `apply_kagome_rule`, `build_global_kagome_graph`, `generate_fcc_tetrahedra_array`.

## Does not own

Production A15 / SC (`explicit/`), FEA Gmsh meshing (`aristo/`), archived scaffold modules (`legacy_gmsh/`).

## Mix-and-match

Prefer `explicit/` for new work. Open this package only when maintaining or comparing legacy GMSH Kagome demos.

## Read next

1. [graphite/explicit/README.md](../explicit/README.md) (current path)
2. [graphite/legacy_gmsh/README.md](../legacy_gmsh/README.md)

## Do not open first

- Treat as source of truth for “how Graphite works today”
