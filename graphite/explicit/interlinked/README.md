# graphite.explicit.interlinked — Modular Interlinked Metamaterial Engine

## Overview

The `graphite.explicit.interlinked` subsystem generates **kinematic metamaterials**, **chainmail textiles**, and **non-welded print-in-place multi-body assemblies** in Graphite.

Unlike standard explicit strut lattices (which weld intersecting struts at common nodal junctions to form rigid structural trusses), interlinked lattices maintain **strictly positive physical clearance ($\Delta > 0$)** between adjacent closed rings, catenated polyhedral cages, or interlocking hooks. When fabricated via Selective Laser Melting (SLM), Laser Powder-Bed Fusion (LPBF), SLS, or multi-material jetting, the unbonded bodies move freely relative to one another, producing flexible space fabrics, energy-absorbing polycatenated lattices (PAMs), and kinematic compliant mechanisms.

---

## Subsystem Architecture (Phases 1–5)

```text
graphite/explicit/interlinked/
├── __init__.py         # Public exports (generate_interlinked_lattice, InterlinkedConfig, InterlinkedRegistry, etc.)
├── particle.py         # Canonical ParticleGeometry (local origin) & InterlinkedParticle (SE(3) instance)
├── cell.py             # Declarative BasisParticle, InterlinkedCell protocol, InterlinkedRegistry catalog
├── cells.py            # Registered cell library (C-6-TT, D-4-TET, J-4-OCT, Euro 4-in-1, Kusari, NASA Space Fabric)
├── seeding.py          # Spatial seeders (Cartesian, Staggered, Hex, Diamond, Cylindrical Wrap, Spherical Shell)
├── boundary.py         # 3-Tier boundary engine (Policy A: Inset culling, Policy B: Boundary tag, Policy C: Frame)
├── clearance.py        # Two-tier KD-tree broad phase + vectorized narrow phase, Gauss linking, pitch solver
├── writer_3mf.py       # ISO/IEC 5165 instanced 3MF package exporter (O(1) prototype storage, 98.6% compression)
├── generator.py        # Top-level unified orchestrator (InterlinkedConfig, InterlinkedLatticeResult)
├── patterns.py         # Analytical closed rings (European 4-in-1, Japanese Kusari, 3D cubic 8-ring)
├── conformal.py        # Planar grid seeding, surface conformal frames, and SDF Inset Culling
├── pams.py             # Polycatenated Architected Materials (Zhou et al., Science 2025)
├── importer.py         # Multi-body STL decompiler, decoupled component scaling, and sheet/rosette tiling
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
- **`D4TetCell` (`"d4tet"`, `"d-4-tet"`):** Diamond cubic (`dia`) bipartite A/B dual cages with point-inversion corner catenation.
- **`J4OctCell` (`"j4oct"`, `"j-4-oct"`):** Square planar octahedra with alternating $45^\circ$ Z-twist for tip catenation.
- **`European4in1Cell` (`"european_4in1"`, `"euro_4in1"`):** Classic diagonal checkerboard maille weave ($\pm 28^\circ$ alternating tilt).
- **`JapaneseKusariCell` (`"japanese_kusari"`, `"kusari"`):** Tri-particle basis (1 horizontal flat ring + 2 vertical linking arch links).
- **`NasaSpaceFabricCell` (`"nasa_space_fabric"`, `"nasa_hexagon"`):** Hexagonal close-pack space fabric with 6-fold spiral interlocking legs.

### 4. `seeding.py` — Spatial Lattice Seeding Topologies
- **Translational Seeders:**
  - `seed_cartesian_lattice(repeats, pitch, origin)`: Regular simple cubic / square grid.
  - `seed_staggered_lattice(repeats, pitch, stagger_fraction, origin)`: 2D/3D brick-staggered lattice.
  - `seed_hexagonal_lattice(repeats, pitch, origin)`: Hexagonal close-pack ($d_y = \frac{\sqrt{3}}{2} d$).
  - `seed_diamond_lattice(repeats, pitch, origin)`: Face-centered diamond cubic network.
- **Analytical Surface Primitives:**
  - `seed_cylindrical_wrap(radius, height, target_pitch, origin)`: Mandates **exact pitch-matching quantization** ($2\pi R = N_\theta \cdot a_\theta$), eliminating seams and link shearing across $0 \to 2\pi$.
  - `seed_spherical_shell(radius, target_pitch, origin)`: Uniform Fibonacci golden spiral distribution with outward radial surface normals.
- **Universal Placer:**
  - `instantiate_lattice_on_sites(cell, centers, frames, grid_indices, cell_pitch)`: Places arbitrary `InterlinkedCell` instances onto arbitrary 3D coordinate frames.

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

### 7. `writer_3mf.py` — Production Instanced 3MF Exporter
- Implements the **3MF Core Specification (ISO/IEC 5165)**:
  - Prototype mesh geometry is written **exactly once** in `<resources>`.
  - Each particle instance is an `<item objectid="..." transform="..."/>` reference in `<build>`.
  - Achieves **$98.6\%$ file size reduction** compared to monolithic STL files ($17\text{ KB}$ vs $1.25\text{ MB}$ for a 16-cage assembly).
  - Packaged as a standard OPC ZIP archive compatible with PrusaSlicer, Bambu Studio, Cura, and Magics.

### 8. `generator.py` — Top-Level Unified API
Unified orchestrator supporting both modular cells and legacy patterns:
```python
from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice

# Configure an inverse-calibrated C-6-TT lattice with a solid perimeter frame
config = InterlinkedConfig(
    cell="c6tt",
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

## Unit Testing & Verification

Run the complete interlinked and polycatenated metamaterial test battery:
```bash
pytest tests/test_interlinked_*.py tests/test_pam_*.py
```
All **94/94 tests** pass cleanly.