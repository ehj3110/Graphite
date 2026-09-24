# SC Conformal — Future Work Catalog

> **Doc role:** Ideas only — cold for coding. Agents: [AGENTS.md](../AGENTS.md) → [graphite/explicit/README.md](../graphite/explicit/README.md). Do not implement from this file unless the user picks an item.

Status: **ideas only, nothing here is implemented.** Collected Aug 2026 after
the V1/V2/V3 three-way compare and the stair-step normal gate fix.

Related docs:

- `SC_CONFORMAL_SURFACE_RELAXATION_IDEAS.md` — evening out surface node
  spacing (detailed; not repeated here).
- `SC_CONFORMAL_THREE_WAY_COMPARE.md`, `SURFACE_FIRST_DUAL_TRIM.md` — current
  architecture and the V1/V2/V3 split.

Current state, for context: V1 = VF-cull → closest-point morph with travel
clamp, stair-step normal gate, residual outside snap, interior Laplacian
relax. V2 = independent surface dual + volumetric trim + gated stitch. V3 =
loose-cull shrinkwrap dual + strict volume + morph-to-dual with valence gate.

---

## A. Cleaner lattice conformance (morphing / projection)

### A1. Interleaved project–relax outer loop
Today projection happens once, then the interior relaxes once. Instead,
alternate small projection steps with relax passes (2–5 outer iterations).
Each projection step only travels a fraction of the remaining distance, so the
volume follows the skin gradually and distortion spreads over many cells
instead of concentrating at the boundary. Cheap to try: wrap the existing
pieces in a loop with a per-pass travel cap.

### A2. Elasticity- or ARAP-based morph instead of Laplacian
Treat the scaffold as an elastic solid: prescribe surface displacements
(projection targets) as boundary conditions and solve linear elasticity — or
as-rigid-as-possible (ARAP) energy — for the interior. Both distribute
distortion far better than Jacobi Laplacian and resist element inversion by
construction. ARAP is the sweet spot: rotation-aware, no material constants to
tune, standard local–global solver, sparse Cholesky on ~10³–10⁴ nodes is fast.

### A3. Height-field (2.5D) conformal mapping for extrusion-like parts
The wrist rest is a kidney-shaped extrusion with a curved top — exactly the
class where stair-steps are self-inflicted. For parts with a flat bottom and a
height-field top `h(x, y)`: keep the XY grid, and scale each column of cells
vertically so N layers span `[0, h(x, y)]` exactly (`z → z · h(x,y)/H`).
No stair-steps exist at all; every cell deforms smoothly; top and bottom skins
are perfect. Generalizes to two height fields (top + bottom). Limitation:
no overhangs/undercuts — detect applicability by raycasting the CAD from above.
This would likely beat V1/V2/V3 outright on this part family and is worth
prototyping early.

### A4. Adaptive/octree refinement near the boundary
Projection artifacts scale with travel-distance ÷ cell-size. Refine boundary
cells one or two octree levels (with hanging-node bridges like the existing
Kelvin face bridges) so travel is always a small fraction of the local cell.
Interior stays coarse for printability and speed. Big lift, big payoff; this
is what commercial tools (nTop, Rhino conformal) do.

### A5. SDF-gradient marching instead of closest-point queries
March nodes along the signed-distance gradient in small steps rather than
jumping to the closest point. Near the medial axis (where two surface patches
are equidistant) closest-point is ambiguous and can flip targets between
neighbouring nodes; gradient marching is continuous and neighbours stay
coherent. Also gives a natural place to enforce the exposed-normal gate at
every step, not just at the start.

### A6. Feature-edge snapping
Detect CAD feature edges (dihedral angle threshold) and, before the general
projection, snap the nearest chain of scaffold edges onto each crease. Lattice
edges then align with part edges instead of straddling them — visually much
cleaner borders and fewer near-tangent struts. Needs Idea 2 from the surface
relaxation doc (patch classification) as infrastructure.

### A7. Quality gates on scaled Jacobian, not volume ratio
`min_hex_volume_ratio` misses shear: a brick can keep its volume while
becoming a parallelepiped that stamps terrible struts. Gate on per-corner
scaled Jacobian (already computed in `_hex_corner_jacobians`) and report a
histogram in every run. Better collapse detection for free.

---

## B. Better surface duals

### B1. Quad remesh of the CAD directly (fully decoupled dual)
Build the dual from the CAD surface itself, not from the lattice skin:
cross-field-guided quad-dominant remeshing (Instant-Meshes-style) at the cell
pitch, then use its vertices/edges as dual nodes/struts. Uniform spacing,
follows curvature, zero inheritance of morph artifacts. This is the logical
endpoint of the V2 philosophy — the dual becomes a first-class surface mesh.
Cost: new dependency or a from-scratch remesher; stitching logic unchanged.

### B2. Geodesic Poisson-disk sampling + surface Delaunay
Lighter-weight alternative to B1: farthest-point/Poisson-disk sample the
surface at cell pitch, connect via geodesic Delaunay (or Euclidean k-NN with a
normal-consistency filter) to form the dual graph. Less structured than a quad
remesh but much simpler, and spacing uniformity is guaranteed by construction.

### B3. Progressive shrinkwrap for the V3 dual
Replace the single-shot projection with an inflate/deflate loop: start the
loose dual slightly offset outside the CAD, pull it in over several steps with
a per-step travel cap and a self-intersection check between steps. Kills the
crossed-strut artifacts single-shot shrinkwrap produces on concave regions.

### B4. Globally optimal stitching (V2/V3)
Current stitch is greedy nearest-with-gates per cut node. Formulate as
bipartite matching instead: cut nodes × dual nodes, edge cost = length +
angle-from-inward-normal penalty, with valence caps on both sides. Solve with
Hungarian or min-cost-flow (scipy). Removes order dependence, prevents the
"one dual node grabs five stitches while its neighbour starves" pattern, and
subsumes the valence gate naturally.

### B5. Manufacturability gates as first-class stitch constraints
Fold print-physics into the stitch acceptance test alongside distance/angle:
minimum strut length (avoid stubs that ball up), maximum unsupported
horizontal span, and a resin-drainage check (no closed pockets formed by
stitch + dual + skin). These exist informally in the V2 design notes; promote
them to coded gates with per-printer presets.

---

## C. Workflow, diagnostics, and testing

### C1. Color-coded diagnostic exports
Export GLB/PLY with vertex colors: nodes colored by signed distance to CAD,
struts by length deviation from nominal. Inspecting the render in Blender
would have found the (96, 132) stranded node in seconds instead of an
arrow-annotated screenshot round-trip. One small utility, huge debugging
leverage — probably the highest value-per-effort item in this doc.

### C2. Torture-part library + metric regression tests
Small synthetic parts, each encoding one failure mode: stair notch (the bug
just fixed), thin wall (< 1 cell), dome (curvature), overhang, chamfered edge,
through-hole. For each, assert metrics — zero stranded iron nodes, zero
inversions, strut length std within bounds — rather than golden meshes.
Any future projection change gets instant, interpretable coverage.
(`tests/test_torture_suite.py` exists but imports a long-gone API; rebuild on
the current `morph_hex_scaffold`.)

### C3. Standard QA report per run
Every generation writes one HTML/PNG dashboard: strut length histogram,
node-to-surface distance histogram, scaled Jacobian histogram, stranded/
clamped/redirected counts, and 3–4 fixed camera renders. The three compare
pipelines already emit `report.txt`; unify the keys and add the plots so V1 vs
V2 vs V3 comparisons stop being manual eyeball work.

### C4. Parameter sweep harness
One command that runs a part across a grid of (cell size, VF threshold,
projection factor, gate on/off) and tabulates the QA metrics. We effectively
did this by hand for the stair gate (gate × factor × two cell sizes in
`scripts/diagnose_stair_step_gate.py`); generalize it.

### C5. Unify the compare pipelines behind one interface
`generate_compare_v1/v2/v3` already share read-only dependencies; give them a
common signature + shared metrics module so the Streamlit flow (and C3/C4) can
treat conformance strategy as a dropdown instead of three code paths.

### C6. Performance: batch and cache geometry queries
Ray casts and nearest-point queries are issued in many small batches across
cull, morph, gate, snap, and diagnostics. Consolidate per-stage into single
batched calls, ensure the embree-backed `RayMeshIntersector` is used when
available, and cache one `ProximityQuery` per mesh per run. Matters once cell
sizes shrink (6×6×4 already runs ~10 s; 3×3×2 will hurt).

---

## Suggested priority

| Priority | Item | Why |
|---|---|---|
| 1 | C1 color-coded exports | Trivial effort, immediately speeds up every future debug loop |
| 2 | Surface relaxation Ideas 1+2 (companion doc) | Fixes the visible deformation that motivated all of this |
| 3 | C2 torture library | Locks in the stair-gate fix and everything after it |
| 4 | A3 height-field mapping | Likely eliminates stair-steps entirely for this part class |
| 5 | A1 interleaved project–relax | Cheap experiment, may soften remaining clamp artifacts |
| 6 | B4 optimal stitching | Biggest known V2/V3 quality lever |
| 7 | A2 ARAP morph | Structural upgrade once quick wins are exhausted |
