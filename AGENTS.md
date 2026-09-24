# Graphite — agent entrypoint

Read this first. Then open **one** matching package `README.md`. Do not glob `docs/` or the whole tree unless the task names a file.

## Product

Conformal lattice R&D: repair shells, generate printable lattices (implicit TPMS + explicit strut scaffolds), FEA (Aristo), LBM (Vocal), optional topology optimization.

## Defaults (use unless the task says otherwise)

| Kind | Default | Package |
|------|---------|---------|
| Explicit tet | A15 conformal Kagome | `graphite/explicit/` |
| Explicit hex | SC Nodal Conformation + Planar Slicing Surface Dual (`generate_nodal_conformation` / `generate_sc_conformal_lattice`) | `graphite/explicit/` |
| Interlinked / PAM | Modular Cell Pipeline (`generate_interlinked_lattice(InterlinkedConfig(cell=...))`) | `graphite/explicit/interlinked/` |
| Explicit Strut Joints | Clean Mitered Truss (`clean_miter=True` / `build_clean_miter_truss`) — no spherical bulges, no notched flat caps | `graphite/explicit/` |
| Implicit | TPMS / Split-P / woodpile in `graphite/implicit/` | `graphite/implicit/` |
| Implicit grading | Native field / piecewise / chirp / SF — **not** Aristo FEA remap | [docs/IMPLICIT_GRADING_AND_TEXTURES.md](docs/IMPLICIT_GRADING_AND_TEXTURES.md) |
| Generated meshes | Write under `outputs/` only | never `test_parts/` |
| CAD fixtures | Read-only under `test_parts/` | — |

## Module map

| When the task is about… | Open next |
|-------------------------|-----------|
| Explicit struts, A15, SC hex, duals, woodpile extrude, chiral cells | [graphite/explicit/README.md](graphite/explicit/README.md) |
| Interlinked, chainmail, polycatenated metamaterials (PAMs) | [graphite/explicit/interlinked/README.md](graphite/explicit/interlinked/README.md) |
| Hex / diamond pentamodes, interlocking auxetics, plate-lattices | [graphite/generators/README.md](graphite/generators/README.md) |
| TPMS fields, calibration, piecewise Split-P / woodpile, grading, textures | [graphite/implicit/README.md](graphite/implicit/README.md) → [docs/IMPLICIT_GRADING_AND_TEXTURES.md](docs/IMPLICIT_GRADING_AND_TEXTURES.md) |
| Linear-elastic FEA on STLs (**analysis only** — not for grading lattices) | [graphite/aristo/README.md](graphite/aristo/README.md) |
| Fluid permeability / WSS (LBM) | [graphite/lbm/README.md](graphite/lbm/README.md) |
| Topology optimization (MMA / scikit-topt) | [graphite/topt/README.md](graphite/topt/README.md) |
| STL health / PyMeshLab repair | [graphite/repair/README.md](graphite/repair/README.md) |
| CAD voxel / EDT / primitives | [graphite/geometry/README.md](graphite/geometry/README.md) |
| TPMS scalar math only | [graphite/math/README.md](graphite/math/README.md) |
| Mesh / STEP / INP export | [graphite/io/README.md](graphite/io/README.md) |
| Interactive Trame + PyVista Studio & UI CLI | [graphite/ui/README.md](graphite/ui/README.md) |
| Shared PyVista PNG framing | [graphite/viz/README.md](graphite/viz/README.md) |
| 1 mm cube case study orchestration | [graphite/case_studies/README.md](graphite/case_studies/README.md) |
| Archived GMSH scaffolders | [graphite/legacy_gmsh/README.md](graphite/legacy_gmsh/README.md) |
| Headless CLIs (Tier 1 only) | [scripts/README.md](scripts/README.md) |
| Human doc hub | [docs/README.md](docs/README.md) |

## Hard rules (also in `.cursor/rules/`)

- **One phase, then stop** with a reviewable file under `outputs/` (or a named runnable), unless the user explicitly says to continue or finish all todos.
- **Never write generated STLs/reports into `test_parts/`.**
- **Explicit Hex Standard:** All explicit hexahedral lattice requests MUST use the Nodal Conformation Bookend Pipeline (`generate_nodal_conformation` / `generate_conformal_lattice(..., lattice_type='SC')`) with Cartesian volume, outside-node boundary snapping, and Planar Slicing Contour Sweep surface dual. The old hex-cage morph in `conformal_generator.py` is archived legacy and must NEVER be used for production lattice generation.
- **Interlinked / PAM Standard:** All interlinked, chainmail, and polycatenated metamaterial requests MUST use the unified modular engine `generate_interlinked_lattice(InterlinkedConfig(cell=...))` in `graphite/explicit/interlinked/`. Procedural generators in `pams.py` are legacy research adapters; do not write ad-hoc procedural lattice loops for new requests.
- **Truss Joint Standard:** All explicit wireframe and truss lattice requests MUST use clean mitered joints (`clean_miter=True` / `build_clean_miter_truss`) with bisector-plane cut strut ends. Spherical fillet bulges (`add_spheres=True`) and raw flat cutoff notches (`clean_miter=False`) must not be used for production lattices unless explicitly requested.
- Wrist rest / octahedral / surface dual / Nodal Conformation: read [docs/UNIVERSAL_DUAL_HANDOFF.md](docs/UNIVERSAL_DUAL_HANDOFF.md) before coding; do **not** rewrite `graphite/explicit/sc_trim_shared_edge_engine.py` (gold oracle).
- **Architecture / Lattice changes:** Whenever an architectural, API, or lattice generation change is completed, run `python scripts/export_core_spec.py --check` and ensure [GRAPHITE_CORE_SPEC.md](GRAPHITE_CORE_SPEC.md) is updated.

## Do not open first

Unless the user names them:

- `docs/history/`, `docs/research/` (cold archive)
- `experiments/` sandboxes (except their README if the task is that sandbox)
- `scripts/archive/`, whole `scripts/` inventory
- `optimization/checkpoints/*/notes/` dump
- Bulk `docs/SC_CONFORMAL_COMPARE_*.md` — use the explicit package card + one compare doc if needed

## Read ladder

```text
AGENTS.md  →  graphite/<pkg>/README.md  →  at most one docs/*.md  →  code
```
