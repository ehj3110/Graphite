# `graphite.case_studies` — capability card

## Owns

End-to-end **1 mm³ cube** workflows (Split-P / woodpile → Aristo / Vocal): canonical stems, paths, cache slugs.

## Status

**Production** canonical case study (orchestration only).

## Public entrypoints

Under `cube_1mm/`:

- Path helpers: `default_output_dir`, `repo_root`, stem helpers (`splitp_piecewise_stem`, `woodpile_stem_default`, …)
- Scripts/modules: `generate_splitp`, `generate_woodpile`, `run_woodpile_aristo`, `run_woodpile_vocal`, `continue_vocal`

Thin wrappers also live in `scripts/generate_cube_1mm_*` and `scripts/run_cube_1mm_*`.

## Does not own

Core engines (`implicit/`, `explicit/`, `aristo/`, `lbm/`) — only orchestrates them for this form factor.

## Mix-and-match

- Outputs: `outputs/case_studies/cube_1mm/`
- Prefer these wrappers over ad-hoc experiments when regenerating the cube study.

## Read next

1. [docs/CASE_STUDY_CUBE_1MM.md](../../docs/CASE_STUDY_CUBE_1MM.md)
2. [docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md](../../docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md)
3. [scripts/README.md](../../scripts/README.md) (Tier 1 cube table)

## Do not open first

- `experiments/implicit_to_volume/` copies of older cube runners unless migrating them
