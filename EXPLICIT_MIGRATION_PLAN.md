# Conformal Explicit Lattice Architecture & Migration Plan

## 1. The Carbon3D Methodology (Reverse-Engineered)

Based on Carbon3D's Design Engine documentation and lattice catalog, their ability to create perfectly conformal, non-trimmed explicit lattices relies on **Volumetric Tetrahedral Meshing** rather than Boolean-trimmed Hexahedral grids.

### The "Boundary Morphing" Secret

* **Hex Meshes:** To make a grid of cubes fit an organic boundary without trimming, the software must warp and morph *every* cell in the part, which heavily distorts the mechanical properties.
* **Tet Meshes:** A tetrahedral mesher packs the volume with nearly uniform, isotropic pyramids. It **only** morphs the tetrahedrons that physically touch the boundary to make them snap to the design space. The internal structure remains mathematically highly regular.

### The "Two Patterns" Secret: Core vs. Skin

Carbon offers two versions of most tet-based lattices (e.g., *Voronoi* vs. *Voronoi Surface*). This is achieved by extracting two distinct mathematical sets from the volumetric mesh:

1. **The Volume (Internal Tets):** Extracting struts from the internal pyramids creates the bulk material behavior. However, this leaves floating, spiky nodes at the boundary.
2. **The Boundary (Surface Triangles):** Extracting the 2D shell of triangles that hugs the CAD surface allows the generation of surface struts or surface duals.

By combining the Internal Struts with the Surface Struts, the lattice gains a perfectly smooth, printable "cage" or skin that conforms flawlessly to the CAD geometry with zero floating shards.

---

## 2. The Graphite Reality

According to our repository analysis (`docs/EXPLICIT_LATTICE_CAPABILITIES.md`), this Carbon-style pipeline **already exists** in the codebase. The canonical implementation now lives under **`graphite/explicit/`**:

| Module | Role |
|--------|------|
| **`graphite/explicit/scaffold_module.py`** | Uses **GMSH** to generate the conformal tetrahedral volume and boundary surface. |
| **`graphite/explicit/topology_module.py`** | Converts the tet mesh into **nodes/struts**, including **surface dual / cage** connections. |
| **`graphite/explicit/geometry_module.py`** | Uses **manifold3d** to sweep graph edges into watertight cylinders (and optional node spheres), with optional boundary trim. |

Downstream scripts and tests import these via `graphite.explicit` or `graphite.explicit.<module>`.

---

## 3. Execution Plan: The Great Migration

Goal: install these engines as the **official backend** for `graphite/explicit/` and expose a stable public API for the app (including Streamlit).

### Step 1: Move the Core Modules

Move the following files from the repository root (or legacy folders) into the **`graphite/explicit/`** directory:

- `scaffold_module.py`
- `topology_module.py`
- `geometry_module.py`

**Status:** Completed — all three modules reside in `graphite/explicit/`. Root copies were removed after updating import paths across the repository.

### Step 2: Establish the Public API

Rewrite `graphite/explicit/__init__.py` to expose these tools cleanly:

```python
from .scaffold_module import generate_conformal_scaffold
from .topology_module import generate_topology
from .geometry_module import generate_geometry

__all__ = ['generate_conformal_scaffold', 'generate_topology', 'generate_geometry']
```

**Note:** Primary entrypoint names were verified against the modules (`generate_conformal_scaffold`, `generate_topology`, `generate_geometry`).

**Status:** Completed — see `graphite/explicit/__init__.py`.

### Step 3: Fix Internal Imports

Scan `scaffold_module.py`, `topology_module.py`, and `geometry_module.py` under `graphite/explicit/`. If they import each other, use package-relative imports (e.g. `from . import geometry_module`).

**Status:** No cross-imports between these three modules were present; only third-party and stdlib imports (`gmsh`, `numpy`, `trimesh`, `manifold3d`, `scipy`, …). No changes required.

### Step 4: Verification

- Run a **linter** on `graphite/explicit/`.
- Run **`python -m py_compile`** on each `.py` file in `graphite/explicit/` to confirm syntax and import graph.

**Status:** Completed in-repo — see §4.

---

## 4. Verification checklist (Step 4)

Execute from the repository root (with the `graphite` package importable):

```bash
python -m py_compile graphite/explicit/__init__.py graphite/explicit/scaffold_module.py graphite/explicit/topology_module.py graphite/explicit/geometry_module.py
python -c "from graphite.explicit import generate_conformal_scaffold, generate_topology, generate_geometry"
```

Optional: IDE / Ruff / Pylint on `graphite/explicit/`.

**Last run:** `python -m py_compile` on all four modules, import smoke test `from graphite.explicit import …`, and IDE diagnostics — all **passed** (no linter issues on `graphite/explicit/*.py`).

---

## 5. Related documentation

- `docs/EXPLICIT_LATTICE_CAPABILITIES.md` — file-level capabilities and libraries.
- `docs/lattice_generator_spec.md` — original modular architecture spec (paths updated for `graphite/explicit/`).
