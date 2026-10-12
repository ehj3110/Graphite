# `graphite.explicit.surface_lattice` — 2D Planar & Conformal Lattice Engine

## Overview

The `graphite.explicit.surface_lattice` subsystem generates **2D planar lattices**, **cylindrical-conformal wraps**, and **surface-normal skin scaffolds** in Graphite.

It unifies:
- **Metamaterial & Standard Topologies:** Re-entrant auxetics, chiral lattices (tetra-chiral, tri-chiral, anti-chiral), regular grids, triangles, Kagome, Kelvin, and Tesseract patterns.
- **Continuous Spatial Grading:** Spatial in-plane strut width $w(x, y)$, out-of-plane thickness $t_z(x, y)$, and multi-scale cell sizing $L(x, y)$ under constant solid-fraction constraints.
- **Clinical Pore Sizing (MIC):** Direct Maximal Inscribed Circle pore diameter metrology and dimensional solving for tissue ingrowth (hernia meshes, osteochondral grafts).
- **Developable Thermoforming Compensation:** Inverse pre-closing of planar auxetic sheets undergoing cylindrical bending ($K = 0$) over 1D sine waves to achieve uniform pore area after forming.
- **Three Solidification Sweeper Backends:**
  1. `plate`: 2D planar buffering + orthogonal Z-extrusion with continuous trapezoidal tapering.
  2. `cylinder`: Circumferential cylindrical conformal unrolling with automatic collar rim fusion.
  3. `mesh`: Direct surface-normal bar sweeps for arbitrary 3D triangular/quad meshes.

---

## Subsystem Architecture

```text
graphite/explicit/surface_lattice/
├── __init__.py         # Public exports (generate_surface_lattice, grading helpers, pore analyzers)
├── pipeline.py         # Unified generator orchestrator (generate_surface_lattice), input validation, export
├── unit_cells.py       # Whole-cell integer tessellators (re-entrant, chiral, graded cell domain)
├── grading.py          # Spatial grading, Gibson-Ashby micromechanics, MIC pore analysis, sine curvature
├── sweepers.py         # Solidification sweepers (sweep_planar_2d, sweep_cylindrical_prisms, sweep_surface_skin)
├── cad_fixtures.py     # Collar rim fusion, CAD fixture inspection, and Boolean carving
└── face_operators.py   # Surface-normal operators for direct mesh face mapping (spheres, shells)
```

---

## Defaults & Standards

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| **Tessellation Boundary** | Integer whole unit cells | Eliminates dangling strut remnants and half-cut chiral node rings. |
| **Strut Tapering** | Continuous trapezoidal (`tapered=True`) | Smooth width transitions with circular joint discs, preventing knife-edge gaps. |
| **Re-entrant Cap / Waist** | $w_{\text{top}} = 0.65 a$, $w_{\text{waist}} = 0.30 a$ | Proven kinematic stability; clamped $w_{\text{waist}} \in [0.10 a, 0.85 a]$. |
| **Minimum Printability** | $w_{\min} \ge 0.40\text{ mm}$ | Prevents un-bonded disconnected node islands. |
| **Export Formats** | `.stl`, `.3mf`, `.ply` | Auto-detected from file suffix; produces watertight 2-manifold closed solids. |

---

## Module Reference

### 1. `pipeline.py` — High-Level Generator & Validation
- **`generate_surface_lattice(surface, pattern, ...)`**: Primary public entrypoint.
  - Supported `surface`: `'plate'`, `'cylinder'`, `'mesh'`.
  - Supported `pattern`: `'reentrant'`, `'tetra_chiral'`, `'tri_chiral'`, `'anti_tetra_chiral'`, `'anti_tri_chiral'`, `'grid'`, `'tetrahedral'`, `'icosahedral'`, `'kelvin'`, `'tesseract'`, `'rhombic'`.
  - Upfront validation: raises informative `ValueError` with supported lists for unrecognized surfaces/patterns or invalid dimensions ($w, h, t \le 0$).
  - Auto-export: exports directly to file when `output_path` is passed.

### 2. `unit_cells.py` — Whole-Cell Tessellators
- **`tessellate_reentrant_domain`**: Generates 2D line segments for Chen et al. (2020) re-entrant honeycombs. Supports per-column angle modulation $\theta_i(x)$ with safety clamping.
- **`tessellate_chiral_domain`**: Generates square-basis (`tetra`) and hexagonal-basis (`tri`) chiral lattices with centered circular nodes.
- **`tessellate_graded_cell_domain`**: Generates continuous multi-scale grids where cell pitch $L(x, y)$ varies across space.

### 3. `grading.py` — Mechanics, Curvature & Pore Sizing
- **`analyze_pore_maximal_inscribed_circles`**: Uses the pole of inaccessibility (`polylabel`) to calculate exact clear void diameters for all interior pores.
- **`solve_reentrant_dimensions_from_pore`**: Closed-form analytical solver calculating cell pitch $a$ and height $b$ required to satisfy a target MIC pore size and wall thickness.
- **`SineCurvatureMapper`**: Computes exact curvature $\kappa(s)$ and arc length $s(x)$ for developable sine waves $z(x) = A \sin(2\pi x / \lambda)$, providing evaluated pre-closed angles $\theta_0(s)$ and strut stiffening $t_{\text{strut}}(s)$.
- **`ProfileCurvatureMapper`**: Computes developable arc length $s(x)$ and curvature $\kappa(s)$ for arbitrary multi-harmonic or semi-random 1D profiles $z(x)$, driving continuous spatial variation in unit cell size $L(s)$ and column boundary discretization.
- **`width_for_constant_solid_fraction`**: Solves strut width $w(L)$ as a function of local cell size $L$ to maintain constant solid fraction $\phi^*$.

### 4. `sweepers.py` — Solidification Backends
- **`sweep_planar_2d`**: Buffers 2D segments into polygons with trapezoidal width tapering, and extrudes along $Z$. Automatically verifies positive manifold volume.
- **`sweep_cylindrical_prisms`**: Radial arc subdivision projecting 2D unrolled patterns into concentric cylindrical prisms with optional solid rim collars.
- **`sweep_surface_skin`**: Sweeps rectangular cross-section bars oriented along surface normals on arbitrary 3D triangular meshes.

---

## Quickstart Examples

### Example 1: Basic Planar Re-Entrant Auxetic Plate
```python
from graphite.explicit import generate_surface_lattice

mesh = generate_surface_lattice(
    surface="plate",
    pattern="reentrant",
    width=60.0,
    height=40.0,
    thickness=2.0,
    n_circumferential=12,
    strut_w=0.8,
    output_path="outputs/reentrant_plate.stl",
)
```

### Example 2: Graded Strut Width Plate
```python
# Strut width grades linearly from 0.8 mm (left) to 2.0 mm (right)
w_gradient = lambda x, y: 0.8 + (x / 100.0) * (2.0 - 0.8)

mesh = generate_surface_lattice(
    surface="plate",
    pattern="tetra_chiral",
    width=100.0,
    height=50.0,
    thickness=3.0,
    n_circumferential=15,
    strut_w=w_gradient,
    tapered=True,
    output_path="outputs/graded_chiral_plate.stl",
)
```

### Example 3: Cylindrical Conformal Ring
```python
mesh = generate_surface_lattice(
    surface="cylinder",
    pattern="tri_chiral",
    r_in=18.0,
    r_out=21.0,
    height=25.0,
    n_circumferential=12,
    r_node=1.5,
    output_path="outputs/cylindrical_sleeve.stl",
)
```

### Example 4: Biological Pore Sizing (Hernia Mesh)
```python
from graphite.explicit.surface_lattice import (
    solve_reentrant_dimensions_from_pore,
    analyze_pore_maximal_inscribed_circles,
    generate_surface_lattice,
)

# 1. Solve cell pitch for 2.0 mm clear pore with 0.4 mm strut walls
params = solve_reentrant_dimensions_from_pore(
    target_mic_pore_mm=2.0,
    wall_thickness_mm=0.4,
    w_waist_ratio=0.30,
)
# a = (2.0 + 0.4) / 0.30 = 8.0 mm cell pitch

# 2. Generate mesh
mesh = generate_surface_lattice(
    surface="plate",
    pattern="reentrant",
    width=80.0,
    height=40.0,
    thickness=1.0,
    n_circumferential=10,  # 80 mm / 8 mm = 10 cells
    strut_w=0.4,
    output_path="outputs/mic_hernia_mesh.stl",
)

# 3. Audit pore diameters
pore_audit = analyze_pore_maximal_inscribed_circles(
    domain_width=80.0,
    domain_height=40.0,
    n_circumferential=10,
    strut_w=0.4,
)
print(f"Mean Inscribed Pore Diameter: {pore_audit['mean_d_mic']:.2f} mm")
```

### Example 5: Developable Sine Forming Compensation
```python
mesh = generate_surface_lattice(
    surface="plate",
    pattern="reentrant",
    width=132.07,  # Developable arc length S for lambda=100, A=20
    height=50.0,
    thickness=2.0,
    n_circumferential=24,
    sine_compensation={
        "amplitude": 20.0,
        "wavelength": 100.0,
        "sheet_thickness": 2.0,
        "theta_nominal_deg": 20.0,
        "t_base": 0.8,
        "gamma": 0.15,
    },
    output_path="outputs/sine_compensated_sheet.stl",
)
```

### Example 6: Semi-Random Surface Profile Curvature Compensation (Variable Cell Size)
```python
mesh = generate_surface_lattice(
    surface="plate",
    pattern="reentrant",
    height=40.0,
    thickness=1.5,
    profile_compensation={
        "length": 140.0,
        "seed": 42,
        "cell_size_min": 3.5,
        "cell_size_max": 14.0,
        "warp_power": 0.5,
        "modulate_cell_size": True,
        "modulate_theta": True,
        "gamma": 0.15,
    },
    strut_w=0.8,
    output_path="outputs/semirandom_curvature_compensated_sheet.stl",
)
```

---

## Related Documentation

- Technical Retrospective: [`docs/2D_LATTICE_ENGINE_RETROSPECTIVE.md`](../../docs/2D_LATTICE_ENGINE_RETROSPECTIVE.md)
- Original Design Specification: [`docs/PLAN_2D_LATTICE_SURFACE_CONFORMAL.md`](../../docs/PLAN_2D_LATTICE_SURFACE_CONFORMAL.md)
- Explicit Core Capability Card: [`graphite/explicit/README.md`](../README.md)
- Core Spec: [`GRAPHITE_CORE_SPEC.md`](../../GRAPHITE_CORE_SPEC.md)
