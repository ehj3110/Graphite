# `graphite.repair` — capability card

## Owns

STL health diagnostics and repair: PyMeshLab gentle triage + Poisson fail-safe; optional Blender path.

## Status

**Production** support utility (no rich package `__init__` — import modules directly).

## Public entrypoints

- `repair_stl`, `run_gentle_triage`, `run_poisson_reconstruction`
- `get_pymeshlab_health`, `print_health_report`
- `repair_stl_with_bmesh` (Blender)

## Does not own

Explicit CAD sanitize (`explicit.mesh_repair`), woodpile-specific repair in implicit, Aristo mesh-quality gates.

## Mix-and-match

- Callers: UI / pre-processing before implicit or explicit generation.
- Prefer this package for **input CAD** triage; use Aristo quality tools for **FEA meshes**.

## Read next

1. [docs/MASTER_ARCHITECTURE.md](../../docs/MASTER_ARCHITECTURE.md) (repair mentioned in layout)
2. Sibling: [graphite/explicit/README.md](../explicit/README.md) if strut CAD sanitize is needed

## Do not open first

- Blender-only paths unless the user asks for Blender repair
