# Graphite Documentation Index

Human hub for deep guides. **Agents:** start at [`../AGENTS.md`](../AGENTS.md), then one package `README.md` — do not load this whole file first.

## Start here

| Role | Link |
|------|------|
| Agent router | [AGENTS.md](../AGENTS.md) |
| Explicit struts (A15 / SC / duals) | [graphite/explicit/README.md](../graphite/explicit/README.md) |
| Implicit TPMS | [graphite/implicit/README.md](../graphite/implicit/README.md) |
| FEA | [graphite/aristo/README.md](../graphite/aristo/README.md) |
| Vocal LBM | [graphite/lbm/README.md](../graphite/lbm/README.md) |
| Headless CLIs (Tier 1) | [scripts/README.md](../scripts/README.md) |

**Defaults:** A15 conformal (tet); modular SC / Conformal Dual (hex). Generated meshes → `outputs/` only (never `test_parts/`).

**UI modes:** Standard (coworker-safe) vs Advanced (experimental) via sidebar **UI Mode** toggle. Current flow: [STREAMLIT_WORKFLOW.md](STREAMLIT_WORKFLOW.md). Revamp plan (not implemented): [STREAMLIT_REVAMP.md](STREAMLIT_REVAMP.md).

## By module

| Module | Capability card | Canonical doc(s) |
|--------|-----------------|------------------|
| Explicit | [graphite/explicit/README.md](../graphite/explicit/README.md) | [EXPLICIT_ENGINE.md](EXPLICIT_ENGINE.md), [HEX_EXPLICIT_ENGINE.md](HEX_EXPLICIT_ENGINE.md), [MODULAR_SC_CONFORMAL.md](MODULAR_SC_CONFORMAL.md), [CONFORMAL_DUAL_HEX.md](CONFORMAL_DUAL_HEX.md) |
| Implicit | [graphite/implicit/README.md](../graphite/implicit/README.md) | [IMPLICIT_GRADING_AND_TEXTURES.md](IMPLICIT_GRADING_AND_TEXTURES.md), [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md), [TPMS_CALIBRATION_WORKFLOW.md](TPMS_CALIBRATION_WORKFLOW.md), [PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md) |
| Math | [graphite/math/README.md](../graphite/math/README.md) | [LATTICE_MATH_ARCHITECTURE.md](LATTICE_MATH_ARCHITECTURE.md) |
| Aristo FEA | [graphite/aristo/README.md](../graphite/aristo/README.md) | [ARISTO.md](ARISTO.md), [ARISTO_MESHING.md](ARISTO_MESHING.md), [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md) |
| Vocal LBM | [graphite/lbm/README.md](../graphite/lbm/README.md) | [VOCAL.md](VOCAL.md), [CUBE_1MM_THREE_LATTICE_COMPARISON.md](CUBE_1MM_THREE_LATTICE_COMPARISON.md) |
| Topology opt | [graphite/topt/README.md](../graphite/topt/README.md) | [TOPOLOGY_OPTIMIZATION.md](TOPOLOGY_OPTIMIZATION.md) |
| Case study 1 mm | [graphite/case_studies/README.md](../graphite/case_studies/README.md) | [CASE_STUDY_CUBE_1MM.md](CASE_STUDY_CUBE_1MM.md) |
| Repair / geometry / io / viz | [repair](../graphite/repair/README.md) · [geometry](../graphite/geometry/README.md) · [io](../graphite/io/README.md) · [viz](../graphite/viz/README.md) | [MASTER_ARCHITECTURE.md](MASTER_ARCHITECTURE.md) |
| Legacy GMSH | [graphite/legacy_gmsh/README.md](../graphite/legacy_gmsh/README.md) | [CONFORMAL_DUAL_HEX.md](CONFORMAL_DUAL_HEX.md) (historical APIs) |

### Repository layout

- **Production:** `graphite/`, `app.py`, Tier-1 [`scripts/`](../scripts/README.md), `tests/`
- **Outputs:** `outputs/` (models, metrics, diagnostics); cube study: `outputs/case_studies/cube_1mm/`
- **R&D:** `experiments/` (named sandboxes); TO handoffs: `optimization/`
- **Local-only (gitignored):** `brain/`, `scratch/`, `result/`, `results/`

## Active handoffs

Open these only when the task matches the sprint. Prefer the package card first.

| Doc | Topic |
|-----|--------|
| [UNIVERSAL_DUAL_HANDOFF.md](UNIVERSAL_DUAL_HANDOFF.md) | Nodal Conformation / universal dual / wrist-rest (read before dual coding) |
| [NODAL_CONFORMATION.md](NODAL_CONFORMATION.md) | Nodal Conformation design |
| [SURFACE_DUAL_ROLES.md](SURFACE_DUAL_ROLES.md) | Universal / layered surface dual roles |
| [SC_CONFORMAL_THREE_WAY_COMPARE.md](SC_CONFORMAL_THREE_WAY_COMPARE.md) | V1–V4 compare ownership map |
| [PLAN_2D_LATTICE_SURFACE_CONFORMAL.md](PLAN_2D_LATTICE_SURFACE_CONFORMAL.md) | 2D Lattice & Surface Conformal engine design (plates, cylinders, surface duals) |
| [EXPLICIT_CHIRAL_AUXETIC_LATTICES.md](EXPLICIT_CHIRAL_AUXETIC_LATTICES.md) | Chiral & auxetic metamaterials architecture & API guide |
| [STREAMLIT_REVAMP.md](STREAMLIT_REVAMP.md) | UI revamp plan (**not implemented**) |
| [OUTPUTS_MODELS_CLEANUP_HANDOFF.md](OUTPUTS_MODELS_CLEANUP_HANDOFF.md) | `outputs/models/` cleanup guardrails |

### Explicit dual / SC lab (open one version, not all)

- [SURFACE_FIRST_DUAL_TRIM.md](SURFACE_FIRST_DUAL_TRIM.md), [LAYERED_SURFACE_DUAL_ROLES_ALIGN.md](LAYERED_SURFACE_DUAL_ROLES_ALIGN.md)
- [SC_CONFORMAL_COMPARE_V2_1_PLAN.md](SC_CONFORMAL_COMPARE_V2_1_PLAN.md), [SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md](SC_CONFORMAL_COMPARE_V4_SURFACE_RELAX.md)
- [SC_CONFORMAL_FUTURE_WORK.md](SC_CONFORMAL_FUTURE_WORK.md), [SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md](SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md) — **not implemented**
- [SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md](SC_QUANTIZED_OVERBUILD_TRIM_DUAL.md), [SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md](SC_CONFORMAL_DUAL_INTEGRATION_ROBUSTNESS.md)
- Related compare notes: V1 / V2 / V3 / crop-then-dual docs in this folder (agent handoffs; demote to history when sprint closes)

### Other topic guides

- Implicit grading + surface textures / micropillars: [IMPLICIT_GRADING_AND_TEXTURES.md](IMPLICIT_GRADING_AND_TEXTURES.md)
- Chiral / auxetic: [EXPLICIT_CHIRAL_AUXETIC_LATTICES.md](EXPLICIT_CHIRAL_AUXETIC_LATTICES.md)
- Woodpile extrude: [EXPLICIT_WOODPILE_EXTRUSION.md](EXPLICIT_WOODPILE_EXTRUSION.md)
- Lofted grading (experimental): [LOFTED_GRADING.md](LOFTED_GRADING.md) · sandbox `experiments/lofted_hex/`
- Trophy hex methods: [TROPHY_BASE_HEX_NEXT.md](TROPHY_BASE_HEX_NEXT.md)
- Aristo (analysis only; **no FEA-driven grading**): [ARISTO.md](ARISTO.md), [ARISTO_ASSEMBLY_BACKEND_COMPARISON.md](ARISTO_ASSEMBLY_BACKEND_COMPARISON.md), [ARISTO_ACCURACY_PATHS.md](ARISTO_ACCURACY_PATHS.md), [CROSS_SECTION_VIZ_HANDOFF.md](CROSS_SECTION_VIZ_HANDOFF.md), [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md)
- Vocal cube / cascades: [VOCAL_CUBE_COMPARISON.md](VOCAL_CUBE_COMPARISON.md), [VOCAL_CASCADE_WARMSTART.md](VOCAL_CASCADE_WARMSTART.md), [CUBE_1MM_VOCAL_N256_HANDOFF.md](CUBE_1MM_VOCAL_N256_HANDOFF.md)
- Baseball / balls: [BALLS_BASEBALL.md](BALLS_BASEBALL.md)
- Implicit overview: [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md) (prefer over any archived implicit specs under `history/archive/`)

## Archive

Cold start — **do not** open for routine feature work.

### History & roadmaps

- [history/](history/) — [ROADMAP.md](history/ROADMAP.md), [PROGRESS.md](history/PROGRESS.md), [EXPLICIT_MIGRATION_PLAN.md](history/EXPLICIT_MIGRATION_PLAN.md)
- A15 refactor: [history/walkthrough.md](history/walkthrough.md), [history/task.md](history/task.md)
- [history/A15_SPRINT_ROOT_CLEANUP.md](history/A15_SPRINT_ROOT_CLEANUP.md), [history/IMPLICIT_GRADIENT_HISTORY.md](history/IMPLICIT_GRADIENT_HISTORY.md)
- Older architecture copies: [history/archive/](history/archive/)

### Research

- [research/a15_kagome_supercell_breakthrough.md](research/a15_kagome_supercell_breakthrough.md)
- [research/conformal_a15_node_snapping.md](research/conformal_a15_node_snapping.md)
- [research/explicit_supercell_methodology.md](research/explicit_supercell_methodology.md)
- [research/experimental_tet_lattices.md](research/experimental_tet_lattices.md)

### Optimization checkpoints

- [conformal-dual-hex-2026-05](../optimization/checkpoints/conformal-dual-hex-2026-05/CHECKPOINT.md)
- [explicit-hex-routes-2026-02](../optimization/checkpoints/explicit-hex-routes-2026-02/CHECKPOINT.md)
- [topology-optimization-2026-05](../optimization/checkpoints/topology-optimization-2026-05/CHECKPOINT.md)

### Rule modules (code)

- [local_tet_rules.py](../graphite/explicit/rules/local_tet_rules.py)
- [tet_topology_rules.py](../graphite/explicit/rules/tet_topology_rules.py)
