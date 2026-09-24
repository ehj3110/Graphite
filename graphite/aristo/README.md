# `graphite.aristo` — capability card

## Owns

Linear-elastic FEA on watertight STLs / implicit-built meshes (P1 tet, Gmsh meshing, stress postprocess, cross-section viz).

## Status

**Production** for analysis. **Not** a lattice-grading driver (Sep 2026): FEA → VF/radius remap is dropped (crashes workstation). See [docs/ARISTO.md](../../docs/ARISTO.md).

## Public entrypoints

- `AristoConfig`, `AristoResult`, `run_aristo`, `MATERIAL_PRESETS`
- Modules: `implicit_input`, `gmsh_lattice_mesh`, `cross_section_viz`, `mesh_quality`
- Legacy only: `stress_to_vf_gradient`, `stress_to_strut_radius_map` in `stress_mapper.py` — do not extend

## Does not own

Fluid LBM (`lbm/`), topology optimization (`topt/`), lattice generation / grading (`implicit/` / `explicit/`).

## Mix-and-match

- Callers: `scripts/run_aristo_fea.py`, Mirae / cube case-study runners.
- Env note: Open3D paths may use `.venv312`; see `requirements-aristo.txt`.

## Read next

1. [docs/ARISTO.md](../../docs/ARISTO.md)
2. [docs/ARISTO_MESHING.md](../../docs/ARISTO_MESHING.md)
3. Regression: [docs/ARISTO_REGRESSION_BASELINE.md](../../docs/ARISTO_REGRESSION_BASELINE.md)
4. Grading instead: [docs/IMPLICIT_GRADING_AND_TEXTURES.md](../../docs/IMPLICIT_GRADING_AND_TEXTURES.md)

## Do not open first

- `stress_mapper.py` / `scripts/generate_graded_bracket_lattice.py` for new product work
- `docs/ARISTO_ACCURACY_PATHS.md` (research) unless paper-grade stress is the task
