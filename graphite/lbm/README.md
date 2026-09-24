# `graphite.lbm` (Vocal) — capability card

## Owns

Lattice Boltzmann permeability / WSS pipeline: voxelize → Lettuce D3Q19 → metrics and comparison plots.

## Status

**Production.** Run under **`.venv_torch`** (Python 3.12).

## Public entrypoints

- `run_vocal`, `run_vocal_comparison`, `VocalRunConfig`, `VocalMetrics`
- `LettuceSolver`, `build_voxel_grid`, `voxelize_stl_to_grid`, `plot_flow_comparison`
- Legacy Taichi helpers may remain in-tree; prefer Lettuce path.

## Does not own

Structural FEA (`aristo/`), lattice generation, shared PNG framing (`viz/` — optional helpers only).

## Mix-and-match

- Callers: `scripts/run_vocal.py`, cube_1mm Vocal runners / plots.
- Inputs: watertight lattice STLs from implicit/explicit.

## Read next

1. [docs/VOCAL.md](../../docs/VOCAL.md)
2. Cube compare: [docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md](../../docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md)
3. Cascades: [docs/VOCAL_CASCADE_WARMSTART.md](../../docs/VOCAL_CASCADE_WARMSTART.md)

## Do not open first

- Regenerable Vocal caches under `outputs/` as design docs
- Non-Tier-1 scripts in `scripts/archive/`
