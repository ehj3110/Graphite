# Aristo accuracy paths — beyond Delaunay Tet4 on STL

Research notes on meshing paradigms for **TPMS lattice FEA** when thin-wall sliver tets or STL discretization error pollute stress fields (especially **pore-size interfaces** for publication).

**Current production path:** discrete STL → `classifySurfaces` → adaptive Delaunay Tet4 → P1 Aristo solve. See [ARISTO_MESHING.md](ARISTO_MESHING.md).

**Context:** V4_fixed achieves ~1.3% “poor” tets (Aristo `quality_ok` gate). These are mostly **through-thickness slivers** at smooth walls — expected for STL-bound Delaunay fill, not necessarily solver-breaking. Interface stress for a paper may still need a cleaner geometric representation than the bulk mesh average.

---

## Summary recommendation (March 2026)

| Approach | Fit for interface stress paper | Graphite status |
|----------|-------------------------------|-----------------|
| **Voxel / structured Hex8** | **Poor** — stair-step singularities mask real interface peaks | Implicit voxel export exists; **not** wired to Aristo FEA |
| **Isotropic STL surface remesh** | **Good next experiment** — low integration cost | **Implemented** — off by default on V4 |
| **Implicit → volume mesh (no STL)** | **Best long-term** — removes Boolean/STL artifacts | Implicit field gen exists; **no** Aristo handoff yet |
| **Finer adaptive tets + quality gates** | **Current baseline** — adequate for global response, debatable at interface | **Production** on V4 |

---

## 1. Voxel meshing (gridded Hex8)

### Idea

Overlay a 3D grid; mark cells solid where the TPMS implicit field is inside the wall; emit perfect cubes (aspect ratio 1.0).

### Pros (Gemini assessment — agree)

- No Delaunay slivers; robust assembly.
- Bypasses STL, `classifySurfaces`, and cap Boolean issues.
- Natural fit for Graphite’s **implicit** TPMS evaluation on a regular grid ([IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md)).

### Cons for pore-interface stress studies (agree — **fatal for this paper goal**)

- **Stair-step surface:** curved TPMS walls become piecewise-flat; every internal 90° kink is a **geometric stress singularity** in linear FEA.
- Artificial peaks on jagged walls can **dominate or mask** the physical concentration at a smooth pore-size transition.
- **Shear locking:** thin walls need ~3–4 Hex8 elements through thickness for bending; DOFs explode unless you use selective reduced integration or Tet/Hex mixed schemes (not in Aristo today).

### Verdict

Use voxels for **mass properties, permeability, or visualization** — not as the primary mesh for **interface von Mises** claims unless you add surface smoothing / level-set projection and validate against a smooth-surface reference.

---

## 2. Isotropic surface remeshing (pre-Gmsh)

### Idea

Long, skinny STL triangles from Boolean cuts force Gmsh to create matching through-thickness sliver tets. **Remesh the skin** to near-equilateral triangles before volume fill.

### Pros

- Preserves **smooth** TPMS geometry (no voxel stairs).
- Fits existing pipeline: STL → remesh → Gmsh → Aristo.
- Industry-standard mitigation for discrete-boundary tet meshes.

### Graphite / Aristo implementation (already exists)

| Piece | Location |
|-------|----------|
| Config flag | `AristoConfig.fea_clean_stl_surface=True` or `ARISTO_CLEAN_STL_SURFACE=1` |
| Open3D isotropic remesh | `graphite/aristo/stl_surface_clean.py` |
| Python 3.12 worker subprocess | `scripts/clean_stl_surface.py` (when Open3D unavailable on 3.13) |
| Trimesh fallback | `subdivide_to_size` in same module |
| Hook in solve path | `maybe_clean_stl_surface()` in `aristo_solver._generate_fea_mesh()` |
| CLI | `mesh_tpms_lattice_gmsh.py --clean-surface` |

**V4 script today:** `fea_clean_stl_surface=False` in `run_mirae_lattice_slab_v4_aristo.py` — deliberate; V4_fixed already has ~48k reasonably uniform faces after manual sliver fix.

### Suggested A/B experiment

1. Run mesh-only with and without clean on V4_fixed; compare `poor_fraction`, max aspect ratio p99, tet count.
2. Run 1N flat-top FEA; compare nodal VM near pore interface (ParaView clip plane at third boundaries).
3. Target edge length ≈ `fea_mesh_resolution` (0.15 mm) when enabling clean.

**Risk:** Aggressive remesh can **move** the zero-level surface slightly (volume drift) — log `mesh.volume` before/after and cap intersection features.

### Verdict

**Highest ROI next step** for paper accuracy without abandoning Tet4 Aristo. Gemini’s PyMeshLab suggestion is equivalent; Graphite already wraps Open3D/Trimesh.

---

## 3. Direct implicit-to-volume mesh (bypass STL)

### Idea

Mesh from the TPMS implicit field (CGAL, Gmsh implicit, Netgen, etc.) with adaptive surface refinement — no STL middleman.

### Would starting from the implicit field avoid our meshing issues?

**Partially yes — it removes the STL-specific failure modes, not all meshing work.**

| Problem class | STL path (today) | Implicit-first path |
|---------------|------------------|---------------------|
| MC needle / zigzag skin triangles | Common | **Avoided** — no marching-cubes discretization step |
| Per-band boolean union seams | Breaks `classifySurfaces` | **Avoided** — single continuous field (see piecewise single-pass) |
| Floating MC island shells | Needs `remove_floating_islands` | **Reduced** — one isosurface extraction |
| Through-wall tet count | Needs **small enough `h`** | **Still required** — Delaunay fill must resolve ~80 µm walls |
| Gmsh Delaunay on thin struts | Works once skin is clean + `h` fine | Same physics; cleaner input surface helps |

So: **implicit-first avoids the STL repair/classify circus**, but you still need enough volumetric resolution and a watertight boundary for tet fill. The June 2026 implicit-cylinder work showed that even on the **STL middleman path**, meshing is **straightforward once prerequisites are met**:

1. **Repair first** — drop floating MC islands (`scripts/aristo_clean_and_mesh_stl.py`).
2. **Size `h` to wall thickness** — target ≥ 4 linear tets through the strut wall (`h ≲ t_wall / 4`; see wall-thickness sweep in [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md)).
3. **Use `single_surface`** — no `classifySurfaces` on MC exports.

With those three, `poor_fraction` fell from ~11% (@ coarse `h`) to **~1–4%** and volume mesh + FEA completed routinely. The issue was **under-resolution and dirty topology**, not an unsolvable mesher.

A true implicit-to-mesh pipeline (field → adaptive surface or field → tets) would skip step 1 for most cases and give smoother caps; it is still the **correct long-term architecture** for publication-grade interface stress.

### Pros

- Eliminates STL faceting, cap slivers, and classify failures at the source.
- Mathematically aligned with how Graphite **generates** Split-P / TPMS structures ([PIECEWISE_PRISM_LATTICE_GENERATION.md](PIECEWISE_PRISM_LATTICE_GENERATION.md)).

### Cons / effort

- New pipeline branch: field → surface mesh → volume mesh → Aristo (or field → tet mesh directly).
- Piecewise thirds / Mirae-style slabs need **multi-region implicit CSG** preserved in the mesher.
- CGAL / Netgen dependencies and Windows toolchain friction.
- Regression story must be rebuilt (new baseline, not V4 STL).

### Graphite today

- **Implicit generation:** `graphite/implicit/*`, marching cubes for export STL.
- **Aristo:** consumes **trimesh STL** today — no direct implicit-field handoff yet.
- **Practical STL path (validated):** implicit field → MC STL → **island cleanup** → **`single_surface` Gmsh @ fine `h`** → Aristo. Works reliably when sized correctly (see [IMPLICIT_TO_VOLUME_MESHING.md](IMPLICIT_TO_VOLUME_MESHING.md)).
- **Conformal / voxel R&D:** separate from Aristo FEA path.

### Verdict

**Correct long-term architecture** for publication-grade interface stress; **large project**. Pursue after isotropic-remesh A/B and interface-focused mesh refinement show STL path is insufficient.

---

## 4. Stay on Delaunay Tet4 but refine locally

Before changing paradigm, consider cheaper improvements on the **current** path:

| Tactic | Notes |
|--------|-------|
| **`quality_ok` stress gating** | Already excludes ~1.3% poor tets from stress stats — use in paper methods section |
| **Interface band refinement** | Background mesh field in Gmsh keyed to Z-thirds or implicit pore boundary — not implemented |
| **Finer `cl_min`** | e.g. 0.05 mm at cost of DOFs; thorough remesh already scales h |
| **Flat-top BC** | Done — removes spurious load on sidewalls |
| **Netgen off** | Done — avoids illegal tets worse than slivers |

Slivers in **included** elements still affect `K` — gating helps post-processing more than stiffness. For interface-dominated claims, geometry fidelity matters more than excluding 1% of elements from percentiles.

---

## V4 STL provenance and remesh insertion point

`Mirae_LatticeSlab_V4_fixed.stl` is a **curated export** (~48k faces, cap sliver fix), not a single one-shot script output in-repo. Related generators:

- `scripts/generate_discrete_rect_prism_3x1p5x5_user_spec.py` — discrete-thirds Split-P / woodpile prism STLs
- `scripts/generate_splitp_linear_box_3x1p5x5_sf25.py` — linear gradient box

**Insertion point for automated isotropic remesh (any script-based STL):**

```
implicit / CAD export → STL
    → [optional] fix_lattice_stl_slivers.py (cap only)
    → [NEW DEFAULT FOR FEA] fea_clean_stl_surface / maybe_clean_stl_surface
    → generate_lattice_fea_mesh_from_stl / run_aristo
```

Wire into orchestration:

```python
config = AristoConfig(
    ...
    fea_clean_stl_surface=True,
    fea_mesh_resolution=0.15,
)
```

Or precompute once:

```bash
python scripts/clean_stl_surface.py input.stl output_clean.stl --target-edge 0.15
```

---

## Decision tree (paper-focused)

```
Need interface von Mises trustworthy?
├─ Yes, STL path good enough after remesh A/B?
│   ├─ Yes → document isotropic pre-mesh + quality_ok gates in methods
│   └─ No  → implicit-to-mesh R&D OR interface-local h refinement
└─ Voxel hex for sanity check only (compare trends, not absolute peaks)
```

---

## Related docs

- [ARISTO_MESHING.md](ARISTO_MESHING.md) — current Gmsh defaults and failures
- [ARISTO_REGRESSION_BASELINE.md](ARISTO_REGRESSION_BASELINE.md) — V4 pinned metrics for A/B comparison
- [IMPLICIT_ENGINE.md](IMPLICIT_ENGINE.md) — upstream TPMS fields
