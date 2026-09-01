# Graphite scripts

Headless CLIs beside `app.py`. **Library code lives in `graphite/`** — scripts should stay thin.

**Case study workflows:** prefer `graphite/case_studies/` + the `generate_cube_1mm_*` / `run_cube_1mm_*` wrappers below.

---

## Tier 1 — Canonical (run these)

### Aristo FEA & meshing

| Script | Role |
|--------|------|
| [`run_aristo_fea.py`](run_aristo_fea.py) | General FEA (implicit default or `--stl`) |
| [`run_mirae_lattice_slab_v4_aristo.py`](run_mirae_lattice_slab_v4_aristo.py) | Mirae V4 slab — 1/5/10 N sweep, regression baseline |
| [`aristo_clean_and_mesh_stl.py`](aristo_clean_and_mesh_stl.py) | MC island cleanup + volume mesh inspect |
| [`aristo_volume_mesh_inspect.py`](aristo_volume_mesh_inspect.py) | Mesh-only quality JSON + debug VTU |
| [`plot_aristo_cross_sections.py`](plot_aristo_cross_sections.py) | Voronoi stress slices from VTU |
| [`gmsh_lattice_mesh.py`](gmsh_lattice_mesh.py) | Adaptive P1 mesh JSON (+ optional VTU) |
| [`mesh_tpms_lattice_gmsh.py`](mesh_tpms_lattice_gmsh.py) | Extended TPMS mesh CLI |
| [`fix_lattice_stl_slivers.py`](fix_lattice_stl_slivers.py) | Targeted cap sliver repair |
| [`clean_stl_surface.py`](clean_stl_surface.py) | Optional Open3D surface prep |

### Vocal LBM

| Script | Role |
|--------|------|
| [`run_vocal.py`](run_vocal.py) | Single-case LBM (`.venv_torch`) |
| [`plot_vocal_flow_comparison.py`](plot_vocal_flow_comparison.py) | Split-P cube side-by-side flow PNG |
| [`plot_cube_1mm_vocal_wss_comparison.py`](plot_cube_1mm_vocal_wss_comparison.py) | 1 mm cube WSS + streamlines from Vocal `f.pkl` cache (`.venv_torch`) |
| [`continue_cube_1mm_vocal.py`](continue_cube_1mm_vocal.py) | +N steps from Vocal cache (diagnostic) |

### Implicit lattice generation

| Script | Role |
|--------|------|
| [`generate_lattice.py`](generate_lattice.py) | YAML/JSON headless lattice driver |
| [`generate_piecewise_splitp_implicit.py`](generate_piecewise_splitp_implicit.py) | Piecewise Split-P cylinder/box (single-pass) |
| [`generate_piecewise_woodpile.py`](generate_piecewise_woodpile.py) | Piecewise cross-hatch woodpile (`--generator extrude` default) |
| [`generate_piecewise_woodpile_implicit.py`](generate_piecewise_woodpile_implicit.py) | Deprecated → `generate_piecewise_woodpile.py` |
| [`generate_three_cylinder_lattices_user_spec.py`](generate_three_cylinder_lattices_user_spec.py) | 3×5 mm cylinder comparison bundle |
| [`generate_discrete_rect_prism_3x1p5x5_user_spec.py`](generate_discrete_rect_prism_3x1p5x5_user_spec.py) | Discrete-thirds prism STLs |
| [`generate_splitp_linear_box_3x1p5x5_sf25.py`](generate_splitp_linear_box_3x1p5x5_sf25.py) | Linear graded Split-P box |
| [`generate_splitp_gradient_cylinder_3x4p8_200to800_sf25_calibrated.py`](generate_splitp_gradient_cylinder_3x4p8_200to800_sf25_calibrated.py) | Calibrated gradient cylinder |

### 1 mm cube case study

| Script | Delegates to |
|--------|----------------|
| [`generate_cube_1mm_splitp.py`](generate_cube_1mm_splitp.py) | `graphite.case_studies.cube_1mm.generate_splitp` |
| [`generate_cube_1mm_woodpile.py`](generate_cube_1mm_woodpile.py) | `graphite.case_studies.cube_1mm.generate_woodpile` |
| [`run_cube_1mm_woodpile_aristo.py`](run_cube_1mm_woodpile_aristo.py) | `graphite.case_studies.cube_1mm.run_woodpile_aristo` |
| [`run_cube_1mm_woodpile_vocal.py`](run_cube_1mm_woodpile_vocal.py) | `graphite.case_studies.cube_1mm.run_woodpile_vocal` |
| [`plot_cube_1mm_aristo_vocal_comparison.py`](plot_cube_1mm_aristo_vocal_comparison.py) | 2×3 Aristo + Vocal comparison figure |
| [`plot_cube_1mm_vocal_wss_comparison.py`](plot_cube_1mm_vocal_wss_comparison.py) | 1×3 Vocal WSS + streamlines (from cache) |
| [`run_splitp_cascade_32_64_128_downflow.py`](run_splitp_cascade_32_64_128_downflow.py) | Split-P downflow 32→64→128 Vocal cascade (`.venv_torch`) |
| [`run_splitp_cascade_32_64_128_upflow.py`](run_splitp_cascade_32_64_128_upflow.py) | Split-P upflow 32→64→128 (seed: flip Uz from converged downflow) |
| [`plot_cube_1mm_vocal_updown_comparison.py`](plot_cube_1mm_vocal_updown_comparison.py) | 2×3 Vocal up vs down comparison |
| [`plot_cube_1mm_vocal_updown_variance.py`](plot_cube_1mm_vocal_updown_variance.py) | 1×3 up/down \|u\| variance heatmaps |
| [`continue_vocal_from_cache.py`](continue_vocal_from_cache.py) | Continue from `f.pkl` + residual diagnostics |
| [`plot_splitp_piecewise_phase_offset_test.py`](plot_splitp_piecewise_phase_offset_test.py) | Piecewise band phase-offset Aristo 4-panel test |

Outputs: `outputs/case_studies/cube_1mm/` — see [docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md](../docs/CUBE_1MM_THREE_LATTICE_COMPARISON.md).

### Explicit hex (default production path)

| Script | Role |
|--------|------|
| [`export_route3_conformal_dual_toros.py`](export_route3_conformal_dual_toros.py) | Route 3 Conformal Dual — Toros reference |
| [`export_trophy_base_thin_conformal_dual.py`](export_trophy_base_thin_conformal_dual.py) | Trophy base — Conformal Dual (Method A) |
| [`run_adapter_lattice.py`](run_adapter_lattice.py) | Adapter / Delaunay Kagome pipeline |

---

## Tier 2 — Archive

Older Mirae versions, one-off exports, A15/Kagome experiments, diagnostics, and superseded generators live under **[`archive/`](archive/)**.

See [`archive/README.md`](archive/README.md) for the full file index and when to use archived scripts.

**Rule of thumb:** if it is not in the Tier 1 table above, look in `archive/` first before adding a new script.

---

## Adding new scripts

1. Implement logic in `graphite/` (or `graphite/case_studies/` for bundled workflows).
2. Add a thin CLI here only if it needs a stable entry point.
3. Update this README (Tier 1) or `archive/README.md`.
4. Do not add duplicate wrappers — extend an existing canonical script with flags when possible.
