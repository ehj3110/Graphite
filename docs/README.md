# Graphite Documentation Index

This directory contains the central architectural design patterns, specification histories, and operational notes for the Graphite project.

## UI Modes
- **Standard mode (default):** Keeps the Streamlit workflow coworker-safe by hiding experimental controls.
- **Advanced mode:** Reveals experimental controls (including supercell beta and advanced grading options) for full power-user workflows.
- **Where to switch:** Use the **UI Mode** toggle in the app sidebar.

## Repository layout

- **Production code:** `graphite/`, UI in `app.py`, headless entry points in [`scripts/README.md`](../scripts/README.md) (Tier 1 only), CI in `tests/`.
- **Outputs:** `outputs/` (models, metrics, diagnostics); **1 mm cube case study:** `outputs/case_studies/cube_1mm/`; topology-optimization handoffs under `optimization/`.
- **R&D:** `experiments/` (named sandboxes; **1 mm cube moved to** `graphite/case_studies/cube_1mm/`); ad-hoc helpers in `tools/`.
- **Local-only (`.gitignore`, keep on disk):** `brain/`, `scratch/`, `result/`, `results/` — see [MASTER_ARCHITECTURE.md](MASTER_ARCHITECTURE.md#2-directory-structure).

## Default explicit hex mesh

**Conformal Dual** is Graphite’s default hexahedral lattice pipeline (Route 3). Use it for new hex exports and UI wiring unless a task explicitly calls for an experimental variant.

- **Doc:** [CONFORMAL_DUAL_HEX.md](CONFORMAL_DUAL_HEX.md)
- **API:** `generate_conformed_hex_scaffold(..., conformal_dual_mode=True)` → `synthesize_conformal_dual_lattice` → `generate_geometry`
- **Reference STL:** `outputs/Route3_ConformalDual_Fixed_Toros.stl`
- **Export template:** `scripts/export_route3_conformal_dual_toros.py`, `scripts/export_trophy_base_thin_conformal_dual.py`

## Default explicit tet mesh (A15)

**A15 conformal Kagome** is Graphite’s default tetrahedral / supercell path (GMSH-free integer-grid recipe).

- **API:** `generate_a15_conformal_lattice` / `generate_conformal_lattice(..., lattice_type='A15')` in `graphite/explicit/`
- **Modules:** `a15_conformal.py`, `a15_kagome.py`, `proven_topologies.py`
- **Refactor record:** [history/walkthrough.md](history/walkthrough.md), [history/task.md](history/task.md)
- **Research recipes:** [research/a15_kagome_supercell_breakthrough.md](research/a15_kagome_supercell_breakthrough.md), [research/conformal_a15_node_snapping.md](research/conformal_a15_node_snapping.md)
- **Legacy / beta:** tet-oct `supercell_module.py` (Advanced UI “supercell beta”); GMSH scaffold under `graphite/legacy_gmsh/`

## Core Architecture
- **[MASTER_ARCHITECTURE.md](MASTER_ARCHITECTURE.md)**: Current architecture overview (implicit vs explicit, directory layout). Older `ARCHITECTURE.md` / `LATTICE_ARCHITECTURE.md` copies live under [history/archive/](history/archive/).
- **[EXPLICIT_ENGINE.md](EXPLICIT_ENGINE.md)**: Explicit engine notes — prefer A15 conformal + Conformal Dual docs above for current defaults; see Jul 2026 status banner in-file.
- **[EXPLICIT_CHIRAL_AUXETIC_LATTICES.md](EXPLICIT_CHIRAL_AUXETIC_LATTICES.md)**: **Chiral & Auxetic Metamaterial Lattices** — Tetra-Chiral (square), Tri-Chiral (hexagonal), anti-chiral variants, and Chen et al. re-entrant honeycombs mapped onto cylindrical parts with seam-free periodicity.
- **[HEX_EXPLICIT_ENGINE.md](HEX_EXPLICIT_ENGINE.md)**: Hex explicit engine guide — **Conformal Dual is the default mesh**; also documents Route 1/2, experiments, and tet/supercell integration.
- **[LATTICE_MATH_ARCHITECTURE.md](LATTICE_MATH_ARCHITECTURE.md)**: Implicit Jacobian / TPMS phase grading math (not the explicit mesh architecture).
- **[CONFORMAL_DUAL_HEX.md](CONFORMAL_DUAL_HEX.md)**: **Default production hex mesh** — VF-cropped Route 3 scaffold, integer surface dual, Toros + trophy validation, export scripts.
- **[MODULAR_SC_CONFORMAL.md](MODULAR_SC_CONFORMAL.md)**: Modular SC hex conformal engine (VF cull → morph → stamp → skin). Current projection-based path.
- **[SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md)**: **Architectural pivot** — surface-first dual + volumetric trim + gated stitch (distance ≤ 1 cell, angle ≤ 60° from vertical, orphan cull); Streamlit sequential params.
- **[UNIVERSAL_DUAL_HANDOFF.md](UNIVERSAL_DUAL_HANDOFF.md)**: **Agent handoff (19 Aug 2026)** — gold + universal dual + octet τ=0.50 trim; where we stopped (stair-step dual gaps); scripts, counts, do-not list.
- **[SURFACE_DUAL_ROLES.md](SURFACE_DUAL_ROLES.md)**: **Universal dual** (code: layered surface dual roles) — hex C/E/F on supports. Octahedral Cartesian dual matches gold (cylinder + wrist rest, 19 Aug 2026). **Phased align plan:** [LAYERED_SURFACE_DUAL_ROLES_ALIGN.md](LAYERED_SURFACE_DUAL_ROLES_ALIGN.md).
- **[SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md](SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md)**: Lab trail for the octahedral overbuild/trim/dual experiments that became Nodal Conformation (§10–12 solids).
- **[SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md](SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md)**: Node-plane policy notes; points at Nodal Conformation.
- **[SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md)**: **Three-way compare** — V1 closest morph · V2 surface-first · V3 hybrid dual-shrinkwrap + morph-to-dual · **V4 surface-relax** (agent handoffs + code ownership).
- **[SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md](SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md)**: V4 handoff — V1 + tangent surface relaxation with sharp-feature pinning.
- **[SC_CONFORMAL_COMPARE_V2_1_PLAN.md](SC_CONFORMAL_COMPARE_V2_1_PLAN.md)**: **V2.1** — hardened cut memory, `exposed_corners` dual, ungated nearest stitch + valence; `generate_compare_v2_1`, stage exports.
- **[SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md](SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md)**: **Future work (not implemented)** — evening out surface node spacing after projection: tangent-constrained Laplacian, patch/crease locking, edge-length springs, coupled relax, quad-quality smoothing.
- **[SC_CONFORMAL_FUTURE_WORK.md](SC_CONFORMAL_FUTURE_WORK.md)**: **Future work catalog (not implemented)** — cleaner conformance (ARAP, height-field mapping, adaptive refinement), better duals (quad remesh, optimal stitching), and workflow/diagnostics upgrades, with priorities.
- **[STREAMLIT_REVAMP.md](STREAMLIT_REVAMP.md)**: **UI revamp plan (not implemented)** — gap analysis vs Graphite today, Streamlit-vs-alternatives recommendation, multipage target IA, and phased delivery.
- **[TROPHY_BASE_HEX_NEXT.md](TROPHY_BASE_HEX_NEXT.md)**: Trophy test-part runs — **Method A (Conformal Dual) is the default**; Methods B–D are experimental comparisons.
- **[graphite_implicit_spec.md](graphite_implicit_spec.md)** - Spec for the implicit generation engine.
- **[IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md)**: Implemented implicit TPMS architecture, equations, meshing notes, pore metrics, and calibration overview.
- **[STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md)**: Current 6-step app flow, finalized Step 4 field-controls behavior, preview/origin handling, and output-path conventions.
- **[PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md)**: Discrete-thirds / linear Split-P and **piecewise cross-hatch woodpile** (single-pass, phase anchoring, band orientation).
- **[EXPLICIT_WOODPILE_EXTRUSION.md](EXPLICIT_WOODPILE_EXTRUSION.md)**: 2D bar extrusion woodpile backend (`generator: extrude`) vs implicit MC; integration and legacy cleanup plan.
- **[TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md)**: Practical guide for point/gradient calibration, MIS boundary guarding, and seed-table usage.
- **[LOFTED_GRADING.md](LOFTED_GRADING.md)**: **Experimental lofted engine** — single-spine unit-cell grading (explicit slice-lofted hex scaffold + implicit integrated-phase TPMS). Sandbox: `experiments/lofted_hex/`.

## Vocal LBM (fluid permeability & WSS)
- **[VOCAL.md](VOCAL.md)**: Lattice Boltzmann fluid simulation — voxelization, Lettuce D3Q19 solver, flow-chamber BCs, permeability/WSS, CLI (`scripts/run_vocal.py`), `.venv_torch` setup.
- **[VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md)**: 1 mm cube piecewise vs linear grading — current metrics, cache state, how to run more iterations / converge.
- **[CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md)**: **Three-lattice comparison** (woodpile extrude \| piecewise X+L/4 \| linear grad) — 6-panel Aristo/Vocal, 3-panel WSS, up/down + variance, phase-offset fix, cache slugs, regenerate commands.
- **[VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md)**: **Vocal 32→64→128 cascades** — warm-start fixes, convergence, up/down comparison, variance heatmaps, invalid caches.
- **[CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md)**: **Canonical 1 mm cube workflows** — `graphite/case_studies/cube_1mm/`, CLI scripts, output paths.

## Aristo FEA (discrete lattice STLs)
- **[ARISTO.md](ARISTO.md)**: Linear-elastic FEA on watertight STLs — module overview, Mirae V4 progress, BC modes, scripts, and known limits.
- **[ARISTO_ASSEMBLY_BACKEND_COMPARISON.md](ARISTO_ASSEMBLY_BACKEND_COMPARISON.md)**: Planning brief — custom `scipy.sparse` + PARDISO vs scikit-FEM (pros/cons, migration options, evaluation plan).
- **[CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md)**: Cross-section stress PNG handoff — Aristo session state, Voronoi overlap analysis, fill-mode alternatives.
- **[ARISTO_MESHING.md](ARISTO_MESHING.md)**: Gmsh TPMS meshing defaults (P1 + Delaunay + adaptive sizing), geometry guidance, failure modes, env vars, and troubleshooting.
- **[ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md)**: Pinned V4 mesh/FEA numbers, re-run commands, pass/fail criteria.
- **[ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md)**: Voxel vs isotropic remesh vs implicit-to-mesh — research notes for paper-grade interface stress.
- **[IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md)**: Piecewise Split-P → Gmsh `single_surface` meshing, quality gates, Aristo FEA; canonical scripts in `scripts/`.
- **[OUTPUTS_MODELS_CLEANUP_HANDOFF.md](OUTPUTS_MODELS_CLEANUP_HANDOFF.md)**: Repo-wide `outputs/models/` cleanup guardrails (Option 2; preserve Mirae prism folder).
- **[BALLS_BASEBALL.md](BALLS_BASEBALL.md)**: Ø74 mm baseball project — graded TPMS, conformal Tri/Sq scaffolds, Voronoi (each with `*_noSeam` + `*_seam`); solid baseball-seam ribbon recipe; square-strut oversize + R−0.25 trim; FINAL catalog + regen scripts (STLs optional locally after upload).

## Internal Mechanics
- **[research/experimental_tet_lattices.md](research/experimental_tet_lattices.md)**: Research notes regarding isotropic tetrahedral cell formations.
- **[research/explicit_supercell_methodology.md](research/explicit_supercell_methodology.md)**: Deterministic A15 bonds / clique Kagome methodology (supersedes Delaunay-periodicity notes).

## Rule Modules
- **[local_tet_rules.py](../graphite/explicit/rules/local_tet_rules.py)**: Pure per-element tetrahedral micro-rules.
- **[tet_topology_rules.py](../graphite/explicit/rules/tet_topology_rules.py)**: Vectorized production rule builders and rule registry used by topology generation.

## Optimization checkpoints
- **[conformal-dual-hex-2026-05](../optimization/checkpoints/conformal-dual-hex-2026-05/CHECKPOINT.md)**: Conformal Dual synthesis, Phase 17 VF-cull fix (Toros 704 hex / 136 inversions), deformation diagnostics, trophy-base next steps.
- **[explicit-hex-routes-2026-02](../optimization/checkpoints/explicit-hex-routes-2026-02/CHECKPOINT.md)**: Explicit hex Route 2/3, shared boundary policy, `hex_dual`, VF shrink/grow; bundled `Route*.stl` copies under `optimization/checkpoints/explicit-hex-routes-2026-02/artifacts/`.
- **[topology-optimization-2026-05](../optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md)**: `graphite.topt`, scikit-topt sandbox, native runners; filter-disconnect handoff markdown under `optimization/checkpoints/topology-optimization-2026-05/notes/`.

## History & Archives
The `history/` directory holds milestone reports, past checkpoint data, and legacy upgrade plans. Check here if you need context on previously accomplished work streams:
- [Roadmaps & Log](history/ROADMAP.md)
- [Legacy Explicit Migration Plan](history/EXPLICIT_MIGRATION_PLAN.md)
- [Past Day 1/2 Progress Reports](history/PROGRESS.md)
- [A15 conformal refactor walkthrough](history/walkthrough.md) / [task checklist](history/task.md)
- [A15 sprint root cleanup inventory](history/A15_SPRINT_ROOT_CLEANUP.md) (2026-07-24 deleted root STLs/PNGs)
- [Implicit gradient history](history/IMPLICIT_GRADIENT_HISTORY.md)
