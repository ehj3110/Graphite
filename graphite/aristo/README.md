# `graphite.aristo` — capability card

## Owns

1. **Direct Linear-Elastic FEA**: Analysis of watertight STLs / implicit-built meshes (P1 tet, Gmsh meshing, stress postprocess, cross-section viz).
2. **Aristo Adapt (Two-Scale Homogenized Lattice Optimization)**: Closed-loop Fully Stressed Design (FSD) driven by asymptotic unit-cell homogenization, tensor surrogate models, macro continuum bridging, and clean mitered truss realization.

## Status

**Production** for:
- Direct analysis on discrete meshes (`run_aristo`).
- Two-scale stress-adaptive lattice optimization (`run_aristo_adaptive`).

> [!NOTE]
> The legacy 1-pass micro-remap in `stress_mapper.py` (which attempted full-resolution Gmsh meshing of micro-struts and crashed workstations) is officially superseded by **Aristo Adapt**. Aristo Adapt operates on a coarse macro continuum mesh, evaluates the local effective constitutive tensor $\mathbf{C}^H(\phi)$ from offline homogenization surrogates, and converges in seconds without memory explosion.

## Public entrypoints

- **Direct FEA Engine**: `AristoConfig`, `AristoResult`, `run_aristo`, `MATERIAL_PRESETS`
- **Aristo Adapt (Optimization)**: `AristoAdaptConfig`, `AristoAdaptResult`, `run_aristo_adaptive`, `StressAdaptationConfig`, `TwoScaleOptimizationResult`
- **Micromechanics & Homogenization**: `homogenize_voxel_rve`, `homogenize_octet_cell`, `homogenize_tpms_cell`, `homogenize_strut_cell`, `MaterialTensorSurrogate`, `build_octet_homogenization_surrogate`, `build_tpms_homogenization_surrogate`
- **Physical Realization**: `realize_optimized_strut_lattice` (clean mitered joints), `realize_optimized_tpms_lattice`
- **Continuum Bridge**: `MacroMesh`, `create_box_continuum_mesh`, `generate_macro_continuum_mesh`, `run_two_scale_macro_fea`
- **Legacy Lab-Only**: `stress_mapper.py` (`stress_to_vf_gradient`, `stress_to_strut_radius_map`) — do not extend.

## Does not own

Fluid LBM (`lbm/`), pure SIMP topology optimization (`topt/`), unconstrained geometric lattice generators (`implicit/` / `explicit/`).

## Quick start: Aristo Adapt

```python
from graphite.aristo import AristoAdaptConfig, run_aristo_adaptive

config = AristoAdaptConfig(
    target_stress=30.0,            # MPa
    target_volume_fraction=0.15,   # Global volume conservation constraint
    move_limit=0.10,
    max_iterations=15,
)

result = run_aristo_adaptive(
    part=cad_mesh,                 # trimesh.Trimesh or bounding box
    config=config,
    lattice_type="octet",          # or 'gyroid', 'diamond', etc.
    cell_size_mm=2.0,
    fixed_face="-x",
    load_face="+x",
    total_force_N=1200.0,
    realize_lattice=True,          # Synthesize physical 3D mesh
    clean_miter=True,              # Bisector-cut clean mitered joints
    output_stl="outputs/fea/adaptive_lattice.stl",
    output_vtu="outputs/fea/macro_opt.vtu",
)
```

## Read next

1. [docs/ARISTO.md](../../docs/ARISTO.md)
2. [docs/ARISTO_MESHING.md](../../docs/ARISTO_MESHING.md)
3. Regression: [docs/ARISTO_REGRESSION_BASELINE.md](../../docs/ARISTO_REGRESSION_BASELINE.md)
4. Grading comparison: [docs/IMPLICIT_GRADING_AND_TEXTURES.md](../../docs/IMPLICIT_GRADING_AND_TEXTURES.md)

