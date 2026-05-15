# Graphite

Graphite is a conformal lattice R&D pipeline for repairing difficult STL shells, tetrahedralizing them, and generating printable lattice solids.

## Quickstart

Run the UI:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

## Rapid Iteration & Headless Mode

To bypass the UI for production runs or batch processing:

1. **Config Management**: In Step 1 of the UI, use the "Configuration Management" expander to Download/Upload your parameters as **YAML** (human-readable) or JSON.
2. **Headless Generation**: Use the CLI script to generate lattices without the GUI:
   ```bash
   python scripts/generate_lattice.py <input.stl> <config.yaml> <output.stl>
   ```

## Core Package Layout

- `graphite/explicit/`: conformal scaffold, topology synthesis, and strut geometry generation
- `graphite/explicit/rules/`: tetrahedral lattice rule modules and registry
- `graphite/implicit/`: TPMS-based implicit generation
- `graphite/topt/`: topology optimization engine (MMA, OC solvers)
- `graphite/repair/`: robust mesh repair pipeline
- `graphite/meshing/`: meshing utilities
- `graphite/math/`: TPMS equations and math helpers
- `graphite/geometry/`: primitive and geometry utilities

## Topology Optimization (topt)

Graphite integrates a Pythonic topology optimization engine built on `scikit-fem` and `nlopt`.

- **Solvers**: Supports Optimality Criteria (OC) and Method of Moving Asymptotes (MMA).
- **Objectives**: Global Compliance minimization and P-Norm Aggregated Stress minimization.
- **Constraints**: Volume fraction, Compliance ceilings, and Stress aggregation.
- **Filtering**: Helmholtz filtering and Heaviside projection for crisp, binary results.

See [docs/TOPOLOGY_OPTIMIZATION.md](docs/TOPOLOGY_OPTIMIZATION.md) for technical details and case studies.

- **Sandbox & scripts:** [experiments/scikit_topt_sandbox/README.md](experiments/scikit_topt_sandbox/README.md) (Streamlit apps, BC manifests, `scripts/` validation and compliance demos).
- **Native drivers:** `native_topt_run.py`, `native_topt_half_run.py` at repository root.
- **Handoff notes (moved off root):** [optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md](optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md).

## Explicit Pipeline (Production)

1. Generate conformal tetra scaffold in `graphite/explicit/scaffold_module.py`
2. Convert tetrahedra to lattice graph in `graphite/explicit/topology_module.py`
3. Sweep/boolean strut geometry in `graphite/explicit/geometry_module.py`

Supported topology types:

- `rhombic`
- `voronoi`
- `kagome`
- `icosahedral`

Rule implementations:

- Local per-element rules: `graphite/explicit/rules/local_tet_rules.py`
- Vectorized production builders + registry: `graphite/explicit/rules/tet_topology_rules.py`

## Hexahedral Pipeline (Advanced)

1. Generate conformed hex scaffold in `graphite/explicit/hex_scaffold_module.py` (Route 3 - Field Snapping).
2. Relax internal nodes via Laplacian Smoothing in `graphite/explicit/boundary_policy.py`.
3. Synthesize topology (e.g., `grid`, `dual`) in `graphite/explicit/hex_topology_module.py`.

Supported hex rules:
- `grid` (surface-conforming)
- `hex_dual` (centroid-based with surface cage)
- `hex_face_dual` (octahedral face-centered)
- `octet`

See [docs/HEX_EXPLICIT_ENGINE.md](docs/HEX_EXPLICIT_ENGINE.md) for detailed integration guides.

## Repository layout

| Path | Role |
|------|------|
| `graphite/` | Production library (implicit, explicit, repair, meshing, `topt/`) |
| `app.py` | Streamlit UI |
| `scripts/` | Headless lattice and batch runners |
| `tests/` | `pytest` suite for `graphite/` |
| `test_parts/` | Input STL fixtures |
| `outputs/` | Generated lattice STLs, metrics, diagnostics |
| `optimization/` | Topology-optimization checkpoints, archives, and `runs/` |
| `experiments/` | Isolated R&D sandboxes (e.g. `scikit_topt_sandbox/`) |
| `docs/` | Architecture and engine documentation |
| `tools/` | Ad-hoc debug utilities |

**Local-only (gitignored, not deleted):** `brain/` (Cursor agent scratch), `scratch/` (throwaway repro scripts), `result/` and `results/` (solver/pytest dumps). See [docs/MASTER_ARCHITECTURE.md](docs/MASTER_ARCHITECTURE.md).

## Outputs

All generated artifacts are organized under `outputs/`:

- `outputs/models/`
- `outputs/metrics/`
- `outputs/diagnostics/`

## Mesh Repair Pipeline

The repair flow is designed to rescue severely corrupted meshes while preserving geometry as much as possible:

1. **Trimesh Triage (fast cleanup)**
   - remove degenerate/duplicate faces
   - remove unreferenced vertices
   - fix winding, inversion, and normals
   - fill obvious holes

2. **PyMeshLab Screened Poisson Reconstruction (industrial fail-safe)**
   - reconstruct a clean continuous surface from the repaired shell

3. **Highlander Purge + Sealing Loop**
   - enforce single dominant shell
   - remove non-manifold edges/vertices
   - close holes aggressively
   - merge close vertices to stitch topology
   - re-orient/recompute normals
   - iterate until strict watertight checks pass

The target outcome is:
- watertight shell
- 1 connected component
- 0 boundary edges
- 0 non-manifold edges/vertices
- 0 self-intersections
