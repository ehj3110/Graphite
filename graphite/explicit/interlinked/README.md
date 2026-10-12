# graphite.explicit.interlinked — Modular Interlinked Metamaterial Engine

## Overview

The `graphite.explicit.interlinked` subsystem generates **kinematic metamaterials**, **chainmail textiles**, and **non-welded print-in-place multi-body assemblies** in Graphite.

Unlike standard explicit strut lattices (which weld intersecting struts at common nodal junctions to form rigid structural trusses), interlinked lattices maintain **strictly positive physical clearance ($\Delta > 0$)** between adjacent closed rings, catenated polyhedral cages, or interlocking hooks. When fabricated via Selective Laser Melting (SLM), Laser Powder-Bed Fusion (LPBF), SLS, or multi-material jetting, the unbonded bodies move freely relative to one another, producing flexible space fabrics, energy-absorbing polycatenated lattices (PAMs), and kinematic compliant mechanisms.

## Quickstart (Beginner-Friendly One-Liners)

For users who want print-ready chainmail or 3D shock absorbers without dealing with low-level crystallography, use the one-liner presets with pre-tuned **Risk Profiles**:

- **Low Risk** *(Default)*: $1.0\,\text{mm}$ strut diameter, $200\,\mu\text{m}$ clearance (robust, high-reliability printing).
- **High Risk**: $0.75\,\text{mm}$ strut diameter, $150\,\mu\text{m}$ clearance (fine features, lightweight).

```python
from graphite.explicit.interlinked import chainmail, space_fabric, shock_absorber, impact_cushion

# 1. Flexible 60x60 mm chainmail sheet
chainmail(width=60, length=60, export_path="outputs/chainmail.3mf")

# 2. NASA JPL 80x80 mm interlocking space fabric
space_fabric(width=80, length=80, export_path="outputs/space_fabric.3mf")

# 3. 3D compliant shock absorber cube (40x40x40 mm)
shock_absorber(size=40, export_path="outputs/shock_absorber.3mf")

# 4. High-risk fine-resolution cushion block
impact_cushion(size=(50, 50, 25), profile="high_risk", export_path="outputs/cushion.3mf")
```

Full beginner tutorial, slicing advice, and CLI guide: **[docs/INTERLINKED_USER_GUIDE.md](../../../docs/INTERLINKED_USER_GUIDE.md)**.

---

## Subsystem Architecture (Phases 1–5)

```text
graphite/explicit/interlinked/
├── __init__.py         # Public exports (generate_interlinked_lattice, assemble, search_interface, etc.)
├── particle.py         # Canonical ParticleGeometry (local origin) & InterlinkedParticle (SE(3) instance)
├── cell.py             # Declarative BasisParticle, InterlinkedCell protocol, InterlinkedRegistry catalog
├── cells.py            # Registered cell library (C-6-TT, D-4-TET, J-4-OCT, Euro 4-in-1, Kusari, NASA Space Fabric)
├── seeding.py          # Spatial seeders (Cartesian, Staggered, Hex, Diamond, Cylindrical Wrap, Spherical Shell)
├── assemble.py         # Deterministic Multi-Zone Assembly Engine (Continuous & Discrete Stacking Modes)
├── search.py           # Bounded Deterministic Coarse-to-Fine Interface Parameter Sweep Engine
├── validator.py        # Two-step quality gate (literal physical collisions & polyhedral catenation)
├── clearance.py        # Two-tier KD-tree broad phase + vectorized narrow phase, Gauss linking, pitch solver
├── catenation.py       # Exact topological catenation audit, cycle extraction, and Hopf linking
├── boundary.py         # 3-Tier boundary engine (Policy A: Inset culling, Policy B: Boundary tag, Policy C: Frame)
├── writer_3mf.py       # ISO/IEC 5165 instanced 3MF package exporter (O(1) prototype storage, 98.6% compression)
├── generator.py        # Top-level unified orchestrator (InterlinkedConfig, InterlinkedLatticeResult)
├── patterns.py         # Analytical closed rings (European 4-in-1, Japanese Kusari, 3D cubic 8-ring)
├── conformal.py        # Planar grid seeding, surface conformal frames, and SDF Inset Culling
├── pams.py             # Polycatenated Architected Materials (Zhou et al., Science 2025)
├── legacy/             # Archived non-deterministic numerical solvers (Powell optimizer)
└── nasa_hexagon.py     # Code-driven parametric NASA JPL Hexagonal Space Fabric generator
```

---

## Module Reference

### 1. `particle.py` — Particle Abstraction in $SE(3)$
- **`ParticleGeometry`**: Frozen, lightweight canonical prototype centered at the local origin $[0, 0, 0]$ (`nodes`, `struts`, `bounding_radius`, `geometry_type`).
- **`InterlinkedParticle`**: Rigid instance placed in world space via an $SE(3)$ homogeneous transform matrix:
  $$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}$$
  - Lazy coordinate evaluation: `p.global_nodes() = (R @ nodes.T).T + t`.
  - Tuple unpacking support: `nodes, struts = p`.
  - Zero cross-body node sharing: maintains independent vertex indices to prevent manifold merging.
  - Bidirectional bridges: `p.to_pam()` and `InterlinkedParticle.from_pam(pam)`.

### 2. `cell.py` — Declarative Cell Protocol & Registry
- **`BasisParticle`**: Constituent particle specification within a unit cell asymmetric repeat unit (`geometry`, `fractional_offset`, `sublattice_id`, `orientation_fn`).
- **`InterlinkedCell` Protocol**: Universal structural contract declaring:
  - `name`, `family`, `parent_network`, `coordination_number`, `neighbor_catenation_offsets`.
  - `forward_clearance(unit_cell_pitch, strut_diameter)` $\to \Delta$ in mm.
  - `resolve_pitch(target_clearance, strut_diameter)` $\to a_0$ in mm.
  - `instantiate_site(grid_index, site_origin, cell_pitch, id_start, context)` $\to$ `list[InterlinkedParticle]`.
- **`InterlinkedRegistry`**: Central catalog decorator `@InterlinkedRegistry.register("name")` enabling dynamic lookup via `InterlinkedRegistry.get("cell_name")`.

### 3. `cells.py` — Registered Modular Cell Library
Comprehensive catalog of modular kinematic unit cells:
- **`C6TTCell` (`"c6tt"`, `"c-6-tt"`):** Truncated Tetrahedron on Simple Cubic (`pcu`) with 6-fold face catenation (Zhou et al., *Science* 2025). Resolves 3D cuboctahedral collision frustration.
- **`D4TetCell` (`"d4tet"`, `"d-4-tet"`):** Diamond cubic (`dia`) bipartite A/B dual cages with point-inversion corner catenation. Uses calibrated crystal clearance constant $\kappa = 0.1018$ and single-particle sublattice placement.
- **`J4OctCell` (`"j4oct"`, `"j-4-oct"`):** Square planar octahedra with alternating $45^\circ$ bond-axis twist ($\pm X$ or $\pm Y$) for tip catenation.
- **`European4in1Cell` (`"european_4in1"`, `"euro_4in1"`):** Classic diagonal checkerboard maille weave ($\pm 28^\circ$ alternating tilt).
- **`JapaneseKusariCell` (`"japanese_kusari"`, `"kusari"`):** Tri-particle basis (1 horizontal flat ring + 2 vertical linking arch links).
- **`NasaSpaceFabricCell` (`"nasa_space_fabric"`, `"nasa_hexagon"`):** Hexagonal close-pack space fabric with 6-fold spiral interlocking legs.

### 4. `seeding.py` — Spatial Lattice Seeding Topologies
- **Translational Seeders:**
  - `seed_cartesian_lattice(repeats, pitch, origin)`: Regular simple cubic / square grid.
  - `seed_staggered_lattice(repeats, pitch, stagger_fraction, origin)`: 2D/3D brick-staggered lattice.
  - `seed_hexagonal_lattice(repeats, pitch, origin)`: Hexagonal close-pack ($d_y = \frac{\sqrt{3}}{2} d$).
  - `seed_diamond_lattice(repeats, conventional_cell_size, origin)`: Face-centered diamond cubic network returning `(centers, frames, indices, sublattice_ids)`.
- **Analytical Surface Primitives:**
  - `seed_cylindrical_wrap(radius, height, target_pitch, origin)`: Mandates **exact pitch-matching quantization** ($2\pi R = N_\theta \cdot a_\theta$), eliminating seams and link shearing across $0 \to 2\pi$.
  - `seed_spherical_shell(radius, target_pitch, origin)`: Uniform Fibonacci golden spiral distribution with outward radial surface normals.
- **Universal Placer:**
  - `instantiate_lattice_on_sites(cell, centers, frames, grid_indices, cell_pitch, sublattice_ids)`: Places arbitrary `InterlinkedCell` instances onto arbitrary 3D coordinate frames with sublattice context injection.

### 5. `boundary.py` — 3-Tier Perimeter Boundary Engine
- **Policy A (Inset Culling):**
  - `cull_particles_by_sdf(particles, sdf_fn, margin)` & `cull_particles_by_mesh(particles, mesh, margin)`.
  - Drops particles whose bounding spheres intersect the domain boundary ($SDF \le -(R + \text{margin})$). Guarantees **zero broken or cut rings/cages**.
- **Policy B (Perimeter Identification):**
  - `identify_boundary_particles(particles, cell)`: Flags boundary particles via catenation graph topology or KD-tree coordination deficits.
- **Policy C (Solid Perimeter Frame Welding):**
  - `build_perimeter_frame_solid(particles, wall_thickness, margin, frame_shape)`: Synthesizes a rigid exterior CAD border (rectangular box or cylindrical annular collar) that penetrates outermost struts by `margin`, locking loose perimeter elements and providing tensile gripping borders for mechanical testing.

### 6. `clearance.py` — Two-Tier Vectorized Clearance & Pitch Inversion
- **Two-Tier Clearance Verification:**
  - **Tier 1 (Broad Phase):** $O(N \log N)$ `scipy.spatial.cKDTree` interaction sphere filtering. Prunes $>90\%$ of candidate pairs.
  - **Tier 2 (Narrow Phase):** Vectorized segment-segment 3D distance and continuous analytical circle-circle distance (`circle_circle_distance`) with Powell optimization.
- **Topological Linking Integral (`particle_linking_number`):**
  - Evaluates the double Gauss linking path integral via 3-point Gauss-Legendre quadrature.
- **Polymorphic Pitch Inversion (`resolve_lattice_pitch`):**
  - Inverts clearance laws to solve unit cell pitch $a_0$ for any specified strut diameter $D$ and target clearance $\Delta_{\text{target}}$:
    $$\Delta = \kappa \cdot a_0 - D \implies a_0 = \frac{\Delta_{\text{target}} + D}{\kappa}$$
  - Automatically falls back to Brent's root-finding method for non-linear kinematic profiles.

### 7. Universal Clean Miter Solidification & 3MF Export
- **Clean Miter Truss Standard (`build_clean_miter_truss`):**
  - All explicit truss and wireframe particles use bisector cutting planes at each node. Flat end-caps, notch gaps, and spherical bulges are eliminated.
  - Native `manifold3d` CSG with `process=False` ensures 100% closed 2-manifold watertight components without trimesh vertex-welding artifacts.
- **Instanced 3MF Exporter (`writer_3mf.py`):**
  - Implements the **3MF Core Specification (ISO/IEC 5165)**:
    - Prototype mesh geometry is written **exactly once** in `<resources>`.
    - Each particle instance is an `<item objectid="..." transform="..."/>` reference in `<build>`.
    - Achieves **$98.6\%$ file size reduction** compared to monolithic STL files ($17\text{ KB}$ vs $1.25\text{ MB}$ for a 16-cage assembly).

### 8. `generator.py` — Canonical Unified Engine & DfAM Guardrails
Unified entrypoint for all interlinked lattice production:
- **DfAM Pre-Flight Guardrail:** Checks `cell.forward_clearance(pitch, 2 * wire_radius)` before allocating sites. Rejects physical collision configurations with an informative `ValueError` stating the minimum collision-free pitch required.
- **Canonical Usage Example:**
```python
from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

# Configure an inverse-calibrated C-6-TT lattice with a solid perimeter frame
config = InterlinkedConfig(
    cell="c6tt",                # "c6tt", "d4tet", "j4oct", "euro_4in1", "kusari", "nasa_hexagon"
    grid_size=(4, 4, 1),
    wire_radius=0.45,
    min_clearance=0.50,
    auto_resolve_pitch=True,    # Solves pitch so clearance is exactly 0.50 mm
    add_perimeter_frame=True,   # Policy C solid handling border
    frame_wall_thickness=2.0,
    frame_margin=0.6,
)

result = generate_interlinked_lattice(config)

# Export deliverables
result.export_3mf("lattice.3mf")  # Ultra-compact instanced 3MF package
result.export_stl("lattice.stl")  # Watertight monolithic STL
```

---

## Canonical Standard vs. Procedural Legacy Bridge

> [!IMPORTANT]
> **Production Standard:** All future interlinked lattice generation MUST use `generate_interlinked_lattice(InterlinkedConfig(cell=...))` with registered cells from `InterlinkedRegistry`.  
> The procedural functions in `pams.py` (`generate_pam_lattice`, `generate_d4tet_diamond_tiling`, etc.) are retained strictly as backward-compatible bridges and unit-test harnesses. Do not write new ad-hoc procedural lattice loops.

---

## Functionally Graded PAMs & Universal Heteroepitaxial Transitions

The interlinked engine supports continuous spatial functional grading, morphing transitions, and universal discrete heteroepitaxy across all 11 PAM unit cells while strictly maintaining physical non-bonding clearance ($\Delta > 0$):
- **Universal Discrete Heteroepitaxy:** Connect any two disparate PAM cell architectures (polyhedra, rings, polygons) across $\{100\}$ habit planes. The engine automatically solves:
  1. **Law 1 (Supercell Coincidence):** Evaluates transverse lattice vectors, integer supercell multipliers $(m_A, n_B)$, and rotation $\theta$ (e.g. $\theta = 45^\circ$ for `pcu` $\leftrightarrow$ `dia` with $p_B = \sqrt{2} p_A$).
  2. **Law 2 (Aperture Admissibility):** Dynamically computes receiving window apertures and penetrating profile envelopes from 3D geometry without hardcoded constants.
  3. **Law 3 (Topological Catenation):** Automatically extracts polyhedral face cycles (triangles/quads) or evaluates Gauss linking numbers ($\text{Lk} \ne 0$) for closed rings, optimizing longitudinal interface offset $dx_{\text{int}}$ to maximize clearance.
- **Universal Multi-Zone Seeding:** Zone-by-zone generation routes through `cell.instantiate_site()`, completely eliminating hardcoded templates across both two-zone transitions (`generate_interlinked_lattice`) and multi-zone buffer superlattices (`generate_epitaxial_superlattice`).
- **Functionally Graded PAMs:** Continuously grade strut radius $r(\mathbf{x})$ along spatial gradients (e.g. $r = 0.22 \to 0.58\text{ mm}$ on C-6-TT) while guaranteeing $\Delta(x) \ge \Delta_{\min}$ across all interacting particles ([`outputs/graded_pam_c6tt_thickness_gradient.png`](../../../outputs/graded_pam_c6tt_thickness_gradient.png)).
- **C-6-TT $\leftrightarrow$ D-4-TET Morphological Transitions:** Continuous parametric vertex truncation $\tau \in [0, 1/3]$ applied across multi-layer spatial lattices ($6 \times 2 \times 2 = 24$ cages) with verified positive clearance $\Delta = +0.614\text{ mm}$ ([`outputs/transition_pam_c6_to_d4.png`](../../../outputs/transition_pam_c6_to_d4.png)).
- Full mathematical theory, scaling laws, and case studies: see [docs/POLYCATED_ARCHITECTED_MATERIALS.md](../../../docs/POLYCATED_ARCHITECTED_MATERIALS.md).

### First-Principles Heteroepitaxy Standard (October 2026 Reset)
For multi-zone assembly and heteroepitaxial transitions across planar habit boundaries:
1. **Zero Iterative Culling:** All generated assemblies must form 100% interconnected topological components by physical construction without relying on post-hoc degree-0 culling to hide unlinked floating fragments.
2. **Shared-Network Symmetry Preservation:** Unit cells sharing the same parent crystallographic network (e.g. D4-CUBE and D4-TET on Diamond `dia`) must be assembled with **zero azimuthal twist (`twist_deg = 0.0`)** along the supercell stacking normal $[0, 0, 1]$. An artificial $60^\circ$ twist severs the parent network's $C_3$ trigonal symmetry, breaking all transverse bonds and shattering the lattice into isolated vertical dimers.
3. **Rigid $\mathrm{SE}(3)$ Invariance:** When rotating or transforming crystal zones, transformations must act simultaneously on particle centroids $\mathbf{c}_p$ and local orientation frames $\mathbf{R}_p$. Under proper $\mathrm{SE}(3)$ rigid-body transformations, the Gauss Linking Integral $|Lk(\gamma_A, \gamma_B)|$ is strictly invariant.
4. **Commensurate Centering:** Cross-network transitions on $\{100\}$ (such as PCU $\to$ DIA) that rotate convention cells by $45^\circ$ require a transverse centering shift $\mathbf{t}_{\text{shift}} = [a_{\text{sc}}, 0, 0]$ to align zone centroids and prevent lateral half-cell shear.
5. **Phase-Aware Glide-Set Slicing (Diamond $\{111\}$):** Slicing diamond cubic lattices along $[1, 1, 1]$ must snap to the high-coordination 3-bond **glide plane** ($d_{\text{glide}} = a\sqrt{3}/4$) rather than the low-coordination 1-bond shuffle plane ($d_{\text{shuffle}} = a\sqrt{3}/12$). Incorporating the crystallographic Shockley partial dislocation shift $\mathbf{s}^* = [0, a\sqrt{6}/12, 0]$ nests the downward-pointing 3-strut tripod of the entering cage directly into the upward-pointing 3-strut tripod of the receiving cage, producing simultaneous 3-face punctures on both polyhedra ($N_A = 3, N_B = 3$) with $+381.4\,\mu\text{m}$ clearance (Coupon 1A).
6. **Analytical Grain Boundaries & $\Sigma 3$ Coherent Twin Interfaces:**
   - **Closed-Form Gap Equation:** $g^*(a, r_w) = \frac{1}{2}(\Delta z_A + \Delta z_B) - r_w\left(\frac{1}{\cos\phi_A} + \frac{1}{\cos\phi_B}\right)$ computes the optimal out-of-plane interface gap instantaneously ($<1\,\mu\text{s}$) without numerical solvers.
   - **Inverted Void O-Lattice Shift:** $\mathbf{s}^* = \mathbf{c}_{F_B} - \mathbf{x}_A^V$ maps solid vertex protrusions directly into receiving polygonal void centers.
   - **$\Sigma 3$ $\{111\}$ Coherent Twins:** Exact reflection across the $\{111\}$ habit plane ($\mathbf{R}_{\text{twin}} = \mathbf{I} - 2\mathbf{n}\mathbf{n}^T$) with zero in-plane misfit strain ($\Sigma = 3$). Because $\det(\mathbf{R}_{\text{twin}}) = -1$, the engine automatically inverts triangle winding (`faces[:, [0, 2, 1]]`) to guarantee outward-facing watertight 2-manifold STLs.
7. **Comprehensive Retrospective (What Worked vs. What Didn't Work):**
   - *Worked:* First-principles symmetry preservation ($0^\circ$ twist on shared networks), glide-set snapping with Shockley shift, closed-form interface gap/shift equations, clean mitered joints with bisector cutting planes, ISO/IEC 5165 instanced 3MF packaging ($98.6\%$ compression).
   - *Didn't Work / Do Not Repeat:* Naive local frame spinning ($60^\circ$ in place broke $C_3$ symmetry into isolated dimers), post-hoc degree-0 culling (masked bugs and cut perimeter rings), shuffle-plane cleavage on $\{111\}$ (flat triangular floor blocked insertion), coherent twin local rotations without spatial center inversion (caused intra-zone collisions in Zone B), supermesh homotopy / metamorphic adapter cells (distorted strut profiles and bloated 3MF files), tetrahedral $\{100\}$ cleavage ($>69.4\%$ clash rate due to pinched facets).
8. **Parametric Scaling Laws & Inversion Guidelines:** Unit cell pitch scales linearly as $a = \frac{D + \Delta}{\kappa}$. Calibrated packing constants: $\kappa_{\text{D4-TET}} = 0.1018$, $\kappa_{\text{D4-CUBE}} = 0.0900$, $\kappa_{\text{C6-TT}} = 0.1314$, $\kappa_{\text{J4-RING}} = 0.0848$, $\kappa_{\text{T6-RING}} = 0.1855$. For a standard $19.5\,\text{mm}$ outer cage diameter, the required strut diameters are $2.49\,\text{mm}$ for D4-TET, $2.78\,\text{mm}$ for D4-CUBE, and $3.82\,\text{mm}$ for T6-RING.
9. **Authoritative Master Documentation:**
   - Full mathematical derivations, proofs, retrospective post-mortems, scaling tables, and dimensioning theory: [docs/POLYCATED_ARCHITECTED_MATERIALS.md](../../../docs/POLYCATED_ARCHITECTED_MATERIALS.md) (Sections 12–19).
   - First-principles heteroepitaxy proofs: [docs/PAM_FIRST_PRINCIPLES_HETEROEPITAXY.md](../../../docs/PAM_FIRST_PRINCIPLES_HETEROEPITAXY.md).
   - Precomputed $O(1)$ transition lookup catalog: [docs/PAM_TRANSITION_LOOKUP_TABLE.md](../../../docs/PAM_TRANSITION_LOOKUP_TABLE.md).
   - Crystallographic supercell scaling across $\{100\}$, $\{110\}$, $\{111\}$: [docs/PAM_SUPERCELL_SCALING_AND_NOTATION.md](../../../docs/PAM_SUPERCELL_SCALING_AND_NOTATION.md).

10. **Headless CLI Utilities & Visualization:**
   - **Turnkey Generator CLI:** `python scripts/generate_pam.py --preset shock_absorber --box 40 40 40 --out outputs/shock.3mf` (or `python -m graphite.explicit.interlinked`) generates print-ready instanced 3MF/STL lattices, DfAM preflight checks (`--dry-run`), solid CAD handling borders (`--frame`), and scientific preview PNGs (`--preview`).
   - **Dimensioning CLI:** `python scripts/dimension_pam.py --cell d4tet --wire-diameter 0.75 --clearance 0.20` solves pitch, solid fraction, jamming limits, and physical cage outer diameters ($D_{\text{cage}}$).
   - **Matte Gallery Visualizer:** `python scripts/pam_gallery.py --cell d4tet --cutaway` renders publication-grade matte studio cards with longitudinal cross-sections.
   - **Two-Particle Workbench:** `python scripts/pam_workbench.py` provides instant (<100 ms) clearance and Gauss linking parameter sweeps.

---

## Unit Testing & Verification

Run the complete interlinked, polycatenated metamaterial, and integration test battery:
```bash
python -m pytest tests/test_universal_pam_epitaxy.py tests/test_epitaxial_*.py tests/test_c6_d4_*.py tests/test_interlinked_*.py tests/test_pam_*.py
```
All **150+ tests** pass cleanly in $<10\text{s}$.