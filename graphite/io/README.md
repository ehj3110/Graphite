# `graphite.io` — capability card

## Owns

Shared mesh / FEA export and implicit run manifests (STL, native 3MF, STEP options, Abaqus INP, parametric STEP).

## Status

**Production.**

## Public entrypoints

- `export_mesh`, `ExportResult`, `StepExportOptions`
- `formats_from_request`, `resolve_export_formats`
- `write_implicit_parameters_manifest`
- `export_voxel_to_abaqus_inp`, `export_parametric_tpms_step`

## Does not own

Geometry generation, FEA solve (`aristo/`), Vocal caches (`lbm/`).

## Mix-and-match

- Callers: implicit/explicit pipelines and Streamlit export step.

## Read next

1. [docs/IMPLICIT_ENGINE.md](../../docs/IMPLICIT_ENGINE.md) (export notes)
2. [graphite/implicit/README.md](../implicit/README.md)

## Do not open first

- Worker internals (`step_export_worker.py`) unless debugging STEP export
