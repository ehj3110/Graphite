# Graphite Lattice Engine Architecture

## 1) Project Overview

Graphite is a hybrid computational geometry and lattice generation tool that combines:
- an interactive Streamlit wizard (`app.py`) for geometry setup and execution
- continuous scalar-field implicit engines in `graphite/implicit/`
- volumetric node/strut explicit engines in `graphite/explicit/`
- pre-processing repair pipelines in `graphite/repair/`
- geometry and meshing helpers in `graphite/geometry/` and `graphite/meshing/`

## 2) Directory Structure

```text
Graphite/
├── app.py
├── graphite/
│   ├── explicit/
│   │   ├── scaffold_module.py
│   │   ├── topology_module.py
│   │   ├── geometry_module.py
│   │   └── rules/
│   │       ├── local_tet_rules.py
│   │       └── tet_topology_rules.py
│   ├── implicit/
│   ├── geometry/
│   ├── math/
│   ├── meshing/
│   └── repair/
├── outputs/
│   ├── models/
│   ├── metrics/
│   └── diagnostics/
├── tests/
└── test_parts/
```

## 3) Two Methodologies: Implicit vs Explicit

Graphite divides generation into two domains.

**Implicit domain (`graphite/implicit`)**:
- continuous TPMS fields plus CAD distance fields
- implicit boolean composition in voxel space
- mesh extraction with marching cubes

**Explicit domain (`graphite/explicit`)**:
- tetrahedral scaffold generation with GMSH
- topology synthesis from tetra entities into strut graphs
- strut sweep and booleans through manifold3d

## 4) Explicit Pipeline

1. `generate_conformal_scaffold(...)` in `graphite/explicit/scaffold_module.py`
2. `generate_topology(...)` in `graphite/explicit/topology_module.py`
3. `generate_geometry(...)` in `graphite/explicit/geometry_module.py`

Topology synthesis is now split into explicit rule modules:
- `graphite/explicit/rules/local_tet_rules.py`: pure single-element rules
- `graphite/explicit/rules/tet_topology_rules.py`: vectorized production builders and rule registry

Plugin-style extension points:
- `register_topology_rule(...)`
- `unregister_topology_rule(...)`
- `get_topology_rule(...)`

These live in `graphite/explicit/rules/tet_topology_rules.py` and are re-exported by `graphite/explicit/rules/__init__.py`.

Supported topology names are `rhombic`, `voronoi`, `kagome`, and `icosahedral`.

Golden fixtures and parity coverage:
- `tests/test_explicit_rule_fixtures.py`
- `tests/fixtures/explicit_rules/golden_single_tet.json`

## 5) Core Supporting Modules

`graphite/repair`:
- robust STL cleanup and watertight repair

`graphite/meshing` and `graphite/geometry`:
- meshing helpers, primitive generation, and surface selection tools

`graphite/math`:
- shared TPMS equations and math utilities
