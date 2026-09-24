# Retrospective: Centroid-Agglomerated Quad Dual (CAQD)

**Author / Context:** Experimental evaluation of Decoupled 2D Surface Dual with Cell Agglomeration and Inward Centroid Bridging for Simple Cubic (SC) hexahedral lattices (Sept 2026).  
**Code location:** [`experiments/hex_surface_dual_v2/`](../experiments/hex_surface_dual_v2/)  
**Review outputs:** [`outputs/hex_surface_dual_v2/`](../outputs/hex_surface_dual_v2/) vs [`outputs/current_standard_octahedral/`](../outputs/current_standard_octahedral/)  
**Status:** **Archived / Sunset (Negative Result)**. Preserved for reference; production remains **Nodal Conformation with Layered Role Dual** ([`docs/NODAL_CONFORMATION.md`](NODAL_CONFORMATION.md)).

---

## 1. Executive Summary & Naming

This experiment evaluated an alternative boundary architecture proposed by a frontier AI model to address simple cubic / octahedral boundary conformity, given the moniker:

> **Centroid-Agglomerated Quad Dual (CAQD)**  
> *(Alternative descriptor: Decoupled 2D Quad-Centroid Conformal Dual)*

### The Core Premise:
1. **Decouple the 2D surface dual** completely from the internal lattice rule, generating a pure quad/diamond mesh directly on the exterior boundary quad faces of the bounding hex grid.
2. **Agglomerate cut sliver cells** below a volume threshold ($V_{\text{cut}} / V_0 < 0.35$) into adjacent neighbor cells to prevent truncated micro-elements.
3. **Synthesize internal load paths via inward bridging**, connecting underlying cell centroids ($\mathbf{C}_p$) directly to surface quad face centers ($\mathbf{P}_k$).
4. **Conform to curved boundaries** using boundary node pull-in snapping and normal insetting ($\mathbf{P}_k - r_{\text{strut}}\mathbf{n}$) to eliminate half-cylinder overhangs.

### Final Conclusion:
While CAQD achieved all targeted mathematical and geometric goals on canonical fixtures (unbroken quad dual valence, zero micro-struts, watertight Manifold3D compound solids), visual and structural evaluation across complex test parts (**Mouse Wrist Rest**, **Part2 Adapter**, and **Trophy Base**) showed that **CAQD is not worth further exploration**. The maintenance cost, cell distortion, geometric artifacts, and loss of crystallographic symmetry far outweigh any minor inconveniences of Graphite's current standard (**Nodal Conformation**).

---

## 2. Technical Architecture of CAQD

```text
CAD Volume (SDF / EDT)
  │
  ├──► Background Cartesian Hex Grid
  │
  ├──► Sub-voxel Monte Carlo / EDT Volume Fraction Integration
  │      └── Slivers (V_cut / V_0 < 0.35) ──► Agglomerate / Merge into Neighbors
  │
  ├──► Pull-in Node Snapping (Outside nodes with φ > 0 snapped to φ = 0)
  │
  ├──► 2D Quad Dual Extraction (Vectorized edge pairing on exterior quads)
  │      └── Normal Insetting: P_k_inset = P_k - r_strut * n(P_k)
  │
  ├──► Inward Bridging Synthesis: e_bridge = (C_p, P_k_inset)
  │
  └──► Octahedral Volume Lattice Stamp + Direct Joint Union (add_spheres=False)
```

### Modules Implemented in `experiments/hex_surface_dual_v2/`:
- [`agglomeration.py`](../experiments/hex_surface_dual_v2/agglomeration.py):
  - 3D sub-voxel evaluation of inside volume fractions.
  - Cell classification: interior ($V \ge 0.99$), cut ($0.35 \le V < 0.99$), sliver ($V < 0.35$), and exterior ($V \approx 0$).
  - Boundary node pull-in projection onto CAD zero-level set.
  - Surface vertex normal insetting.
- [`dual_topology.py`](../experiments/hex_surface_dual_v2/dual_topology.py):
  - Fully vectorized exterior quad face identification and area centroid calculation.
  - Vectorized edge pairing (`np.tile` + lexicographic sorting) to extract the dual graph without Python loops.
  - Inward bridging generator ($C_p \to P_k$).
  - Unified lattice assembler and sub-graph separator (`surface_dual_struts`, `volume_and_bridge_struts`, `combined_struts`).

---

## 3. Benchmark Comparison: CAQD vs. Current Standard

Both pipelines were executed on identical CAD fixtures at identical cell sizes ($L = \frac{1}{3} \min \Delta$), matching strut radii ($V_f \approx 10\%$, $r / L \approx 0.0722$), direct joint connections (`add_spheres=False`), and shared coordinate origins.

### Quantitative Summary:

| Part & Geometry | Metric | Current Standard (`nodal_conformation`) | CAQD Experiment (`hex_surface_dual_v2`) |
| :--- | :--- | :--- | :--- |
| **Mouse Wrist Rest**<br>$96.8 \times 126.8 \times 13.6\text{ mm}$<br>$L = 4.55\text{ mm}$, $r = 0.329\text{ mm}$ | **Volume Struts**<br>**Dual Struts**<br>**Combined Struts**<br>**File Size (Dual / Vol / Comb)** | 6,901<br>1,825<br>7,887<br>6.37 MB / 18.65 MB / 20.88 MB | 6,812 (incl. bridges)<br>1,236<br>8,048<br>5.85 MB / 23.05 MB / 26.91 MB |
| **Part2 Adapter**<br>$15.9 \times 107.0 \times 77.0\text{ mm}$<br>$L = 5.29\text{ mm}$, $r = 0.382\text{ mm}$ | **Volume Struts**<br>**Dual Struts**<br>**Combined Struts**<br>**File Size (Dual / Vol / Comb)** | 4,715<br>1,576<br>5,267<br>4.42 MB / 11.91 MB / 12.87 MB | 4,500 (incl. bridges)<br>964<br>5,464<br>4.27 MB / 18.39 MB / 20.96 MB |
| **Trophy Base**<br>$100.0 \times 100.0 \times 37.5\text{ mm}$<br>$L = 12.50\text{ mm}$, $r = 0.902\text{ mm}$ | **Volume Struts**<br>**Dual Struts**<br>**Combined Struts**<br>**File Size (Dual / Vol / Comb)** | 1,540<br>487<br>1,701<br>1.51 MB / 4.02 MB / 4.34 MB | 1,416 (incl. bridges)<br>284<br>1,700<br>1.23 MB / 5.46 MB / 6.33 MB |

---

## 4. Why CAQD Was Sunset: Critical Engineering Downsides

### 1. Inward Centroid Bridging Looks Unnatural and Clutters the Core
- **The Ideal:** On an uncropped planar cube, connecting cell centroids $\mathbf{C}_p$ to exterior face centers $\mathbf{P}_k$ forms clean, symmetrical 4-strut pyramid caps.
- **The Reality on Curved CAD:** On complex freeform boundaries, boundary hexes are cut at arbitrary angles. The cell centroid $\mathbf{C}_p$ is often displaced or eccentric. Bridging struts shoot inward at bizarre, non-crystallographic angles, cutting across octahedral diamond cells and cluttering what should be open void space.
- **Current Standard Advantage:** Nodal Conformation preserves native octahedral face chords that naturally obey the $45^\circ$ / $90^\circ$ lattice planes.

### 2. Cell Agglomeration Distorts Element Topology
- Merging sliver cells ($< 35\%$ volume) into adjacent neighbors creates non-hexahedral, irregular polyhedra with 9 to 14 vertices.
- Stamping standard simple cubic topologies into agglomerated irregular polyhedra creates skewed struts, variable strut lengths, and unpredictable local stiffness.
- In contrast, Nodal Conformation's node-plane trimming simply drops sub-cell nodes based on incident box occupancy without altering the underlying grid coordinates.

### 3. Step-Stairing on Inclined Boundaries
- Pull-in snapping snaps outside nodes ($\phi > 0$) onto the CAD zero-level set, but inside nodes ($\phi \le 0$) remain pinned to their Cartesian grid positions.
- Across shallow sloped faces (such as the wrist rest bevel or the trophy base chamfers), this creates artificial stepped terraces in the surface dual.
- In contrast, Nodal Conformation's direct closest-point projection smoothly distributes dual nodes along the continuous CAD surface.

### 4. Iron-Stamping vs. Flexibility across Multiple Cell Rules
- CAQD was designed around the octahedral face-center dual. Adapting it to other rules in Graphite (such as `octet`, `star`, `cross`, `grid`, and `kelvin`) requires re-architecting how centroid bridging interacts with edge midpoints, face diagonals, and body vertices.
- Graphite's **Layered Surface Dual Roles** (`sc_role_surface_dual.py`) already unifies all SC rules under a single, proven, role-based paradigm ($C$, $E$, $F$, $B$, $K$).

---

## 5. Decision & Recommendation

1. **Do NOT adopt or integrate CAQD into `graphite/explicit/`.**
2. **Retain existing production standard:** Continue using **Nodal Conformation with Layered Surface Dual Roles** (`generate_nodal_conformation`).
3. **Archive experiment:** Leave [`experiments/hex_surface_dual_v2/`](../experiments/hex_surface_dual_v2/) and its review outputs in place as a documented negative result so future researchers do not repeat the investigation.
