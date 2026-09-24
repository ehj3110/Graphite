# `graphite.explicit` — capability card

## Owns

GMSH-free explicit strut lattices: conformal tet/hex scaffolds, topology rules, strut geometry (Manifold), SC surface dual / Nodal Conformation, woodpile extrusion, chiral unit cells.

## Status

**Production** core (A15 + modular SC). Lab: `compare_*` packages and many `sc_*` dual experiments.

## Defaults

| Path | Entry |
|------|--------|
| Tet | `generate_a15_conformal_lattice` / `generate_conformal_lattice(..., lattice_type='A15')` |
| Hex (Canonical Default) | `generate_sc_conformal_lattice` / `generate_conformal_lattice(..., lattice_type='SC')` / `generate_nodal_conformation(..., surface_dual_mode='planar_sweep')` |
| Hex (Archived legacy morph) | `generate_legacy_conformal_lattice` / `generate_conformal_lattice(..., lattice_type='SC', legacy=True)` |
| Interlinked / Maille | `generate_interlinked_lattice` / `InterlinkedConfig` in `graphite/explicit/interlinked/` |
| Gold octahedral trim+dual | `sc_trim_shared_edge_engine.generate_octahedral_trim_shared_edge` — **do not rewrite** |

## Public entrypoints

From package `__init__.py` (lazy wrappers):

- `generate_a15_conformal_lattice`, `generate_conformal_scaffold`, `generate_conformal_lattice`
- SC Nodal Conformation & Universal Dual: `generate_nodal_conformation`, `deform_outside_nodes`, `weld_combined_lattice`, `build_planar_slicing_surface_dual`, `PlanarSweepConfig`
- `generate_interlinked_lattice`, `InterlinkedConfig`
- `generate_topology`, `generate_hex_topology`, `generate_geometry`, `solve_sizing`, `repair_cad_mesh`
- `generate_tetrachiral_cell`, `generate_trichiral_cell`
- Health: `check_explicit_health`

Important modules (import directly when needed):

- Interlinked: `interlinked/` (`particle`, `cell`, `cells`, `seeding`, `boundary`, `clearance`, `writer_3mf`, `generator`, `patterns`, `pams`, `importer`, `nasa_hexagon`)
- Custom cells / rules: `custom_rules` (register dynamic hex trusses from canonical skeletons into explicit core)
- Tet: `a15_conformal`, `a15_kagome`, `proven_topologies`, `rules/`
- Hex SC: `conformal_generator`, `conformal_core`, `hex_rules`, `hex_topology_module`, `nodal_conformation`, `planar_surface_sweep`
- Dual / trim: `planar_surface_sweep` (canonical planar slicing sweep default), `sc_trim_shared_edge_engine` (gold), `sc_role_surface_dual`, `sc_node_plane_trim`, `hex_surface_dual`
- Lofted: `lofted_scaffold` (GMSH-free spine-lofted hex scaffold & multi-rule synthesis)
- Damage-programmable: `damage/` (`dp_cells`, microfiber-reinforced BCC hierarchy T0–T3, crack-guidance spatial partitioning, clean mitered joints)
- Multilayer & Hybrid Level-Sets: `hybrid/` (`levelset`, regularized Heaviside blending, 3D mean curvature diagnostics, minimal surface zero-bending interfaces)
- Woodpile: `woodpile_extrude`
- Compares: `compare_v1_closest`, `compare_v2_surface_first`, `compare_v2_1_surface_first`, `compare_v3_hybrid`, `compare_v4_surface_relax`

## Does not own

TPMS scalar fields (`implicit/`, `math/`), FEA (`aristo/`), LBM (`lbm/`), archived GMSH scaffolders (`legacy_gmsh/`).

## Mix-and-match

- Callers: `app.py`, Tier-1 scripts under `scripts/`, case studies.
- Callees: `geometry_module` (Manifold), often CAD from `test_parts/` (read-only).

## Read next

1. [docs/A15_KAGOME_AND_SC_SURFACE_DUAL.md](../../docs/A15_KAGOME_AND_SC_SURFACE_DUAL.md) — canonical A15 Kagome & SC surface dual specification
2. [docs/EXPLICIT_ENGINE.md](../../docs/EXPLICIT_ENGINE.md) — overview + status banner
3. [docs/HEX_EXPLICIT_ENGINE.md](../../docs/HEX_EXPLICIT_ENGINE.md) / [docs/MODULAR_SC_CONFORMAL.md](../../docs/MODULAR_SC_CONFORMAL.md) — hex
4. Wrist-rest / dual sprint: [docs/UNIVERSAL_DUAL_HANDOFF.md](../../docs/UNIVERSAL_DUAL_HANDOFF.md)
5. Multi-lattice blending: [docs/MULTI_LATTICE_BLENDING.md](../../docs/MULTI_LATTICE_BLENDING.md)

## Do not open first

- `docs/history/`, bulk completed compare plans unless the task names a version
- `scripts/archive/`, regenerable STLs under `outputs/` as “source of truth”
