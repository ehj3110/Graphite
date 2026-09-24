# `graphite.topt` — capability card

## Owns

Unstructured-mesh topology optimization (scikit-topt / MMA): compliance and stress P-norm objectives, filtering, isosurface extract / viz.

## Status

**Lab** — isolated from Streamlit Standard mode. Sandbox: `experiments/scikit_topt_sandbox/`. Checkpoints: `optimization/checkpoints/topology-optimization-2026-05/`.

## Public entrypoints

- `ToptConfig`, `run_topt_optimization`, `create_sktopt_task`, `ToptTask`
- `Compliance_Heaviside_Optimizer`, `Stress_P_Norm_Optimizer`
- `extract_isosurface`, `render_topt_isosurface` (postprocess / viz modules)

## Does not own

Lattice FEA (`aristo/`), Streamlit BC pickers in the sandbox app, production lattice engines.

## Mix-and-match

- Callers: native TO runners / sandbox scripts; see [optimization/README.md](../../optimization/README.md).
- Deps: `requirements-topt.txt` (Python ≤3.13 typical).

## Read next

1. [docs/TOPOLOGY_OPTIMIZATION.md](../../docs/TOPOLOGY_OPTIMIZATION.md)
2. [experiments/scikit_topt_sandbox/README.md](../../experiments/scikit_topt_sandbox/README.md)
3. Checkpoint: [optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md](../../optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md)

## Do not open first

- Entire `optimization/checkpoints/.../notes/` dump unless debugging a named handoff
