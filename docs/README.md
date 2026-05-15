# Graphite Documentation Index

This directory contains the central architectural design patterns, specification histories, and operational notes for the Graphite project.

## UI Modes
- **Standard mode (default):** Keeps the Streamlit workflow coworker-safe by hiding experimental controls.
- **Advanced mode:** Reveals experimental controls (including supercell beta and advanced grading options) for full power-user workflows.
- **Where to switch:** Use the **UI Mode** toggle in the app sidebar.

## Repository layout

- **Production code:** `graphite/`, UI in `app.py`, headless entry points in `scripts/`, CI in `tests/`.
- **Outputs:** `outputs/` (models, metrics, diagnostics); topology-optimization handoffs under `optimization/`.
- **R&D:** `experiments/` (named sandboxes); ad-hoc helpers in `tools/`.
- **Local-only (`.gitignore`, keep on disk):** `brain/`, `scratch/`, `result/`, `results/` — see [MASTER_ARCHITECTURE.md](MASTER_ARCHITECTURE.md#2-directory-structure).

## Core Architecture
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: The foundational document describing the split Implicit vs Explicit engines, Streamlit UI, and directory structure.
- **[LATTICE_ARCHITECTURE.md](LATTICE_ARCHITECTURE.md)**: Details the node/strut mathematical behaviors, Delaunay meshing, and Micro-Rules (Voronoi, Kagome) handling inside the explicit sub-engine.
- **[EXPLICIT_LATTICE_CAPABILITIES.md](EXPLICIT_LATTICE_CAPABILITIES.md)**: Original technical readout of the capabilities of volumetric meshing over hexahedral trimming.
- **[HEX_EXPLICIT_ENGINE.md](HEX_EXPLICIT_ENGINE.md)**: Current-state reference for production and experimental hex explicit pipelines, boundary handling, and tet/supercell integration opportunities.
- **[graphite_implicit_spec.md](graphite_implicit_spec.md)** - Spec for the implicit generation engine.
- **[IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md)**: Implemented implicit TPMS architecture, equations, meshing notes, pore metrics, and calibration overview.
- **[PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md)**: Discrete-thirds / linear Split-P and cross-hatch in 3×1.5×5 mm prisms (EDT slab + union, inverted woodpile).
- **[TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md)**: Practical guide for point/gradient calibration, MIS boundary guarding, and seed-table usage.

## Internal Mechanics
- **[experimental_tet_lattices.md](experimental_tet_lattices.md)**: Research notes regarding isotropic tetrahedral cell formations.
- **[lattice_generator_spec.md](lattice_generator_spec.md)**: Base generator implementation details.

## Rule Modules
- **[local_tet_rules.py](../graphite/explicit/rules/local_tet_rules.py)**: Pure per-element tetrahedral micro-rules.
- **[tet_topology_rules.py](../graphite/explicit/rules/tet_topology_rules.py)**: Vectorized production rule builders and rule registry used by topology generation.

## Optimization checkpoints
- **[explicit-hex-routes-2026-02](../optimization/checkpoints/explicit-hex-routes-2026-02/CHECKPOINT.md)**: Explicit hex Route 2/3, shared boundary policy, `hex_dual`, VF shrink/grow; bundled `Route*.stl` copies under `optimization/checkpoints/explicit-hex-routes-2026-02/artifacts/`.
- **[topology-optimization-2026-05](../optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md)**: `graphite.topt`, scikit-topt sandbox, native runners; filter-disconnect handoff markdown under `optimization/checkpoints/topology-optimization-2026-05/notes/`.

## History & Archives
The `history/` directory holds milestone reports, past checkpoint data, and legacy upgrade plans. Check here if you need context on previously accomplished work streams:
- [Roadmaps & Log](history/ROADMAP.md)
- [Legacy Explicit Migration Plan](history/EXPLICIT_MIGRATION_PLAN.md)
- [Past Day 1/2 Progress Reports](history/PROGRESS.md)
