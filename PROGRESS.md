# Graphite Engine — Day 2 Checkpoint

## Today's Strides

### 1. Universal Surface Dual
Standardized the **Kagome-style hexagonal skin** for all centroid-based lattices (Voronoi, Kagome). Surface struts now use direct centroid-to-centroid logic on boundary faces, producing a consistent hexagonal pattern at the skin. Eliminates the previous "Y-Skin" kinks and zig-zag artifacts.

### 2. Coordinate Integrity
Fixed the **Warping/Aspect Ratio** bug:
- **Path B (process=True) loading**: Trimesh `process=True` ensures watertight meshes for manifold3d Boolean operations. Non-watertight meshes caused empty intersection results.
- **Dual-path test**: Load with both `process=False` (Path A) and `process=True` (Path B); use Path B when watertight; re-align Path B to original CAD coordinates if center shifts.
- **Inverse-scaling for insets**: Smart Inset scales the boundary for scaffold generation, then inverse-scales nodes back to original coordinates for lattice placement.

### 3. Performance Optimization
Achieved a **~200x speedup** in volume calculations:
- Stay in the **Manifold domain** during solver iterations; use `manifold.volume()` directly instead of converting to trimesh.
- **`return_manifold=True`**: First pass returns `(Manifold, volume)` and skips `to_trimesh` conversion (~40s saved per iteration).
- Convert to trimesh **only at the very end** for STL export.
- Boolean intersection (Manifold.intersect) holds at **~0.001s** even on complex geometry.

### 4. Adaptive Strut Thickness
Implemented **r = L · k** (per-strut radius proportional to length):
- Prevents "solid blobs" in dense mesh regions where short struts would otherwise be over-thick.
- Analytical one-shot: `k = sqrt(target_volume / (π · sum(L³)))`, capped at `K_MAX = 0.4`.
- Per-strut radii: `r_i = L_i · k` for uniform visual density.

### 5. Profiling Layer
Added verbose **Performance Audit** timers in `graphite/explicit/geometry_module`:
- Cylinder creation
- Manifold.compose (union)
- Manifold.intersect (clip)
- Manifold.to_trimesh (or skipped)
- Manifold.volume()

---

## Current Issues / Bottlenecks

### Complexity Wall
Large parts like **MariaTubeRack_Full** still take significant time (~10 min). The bottleneck is GMSH 3D tetrahedral meshing, not the Boolean intersection. Consider:
- Coarser element sizes for initial scaffold
- GMSH algorithm tuning (HXT vs Netgen)
- Pre-meshed or simplified input geometry

### Intersection Failures
Non-watertight meshes cause manifold3d's Boolean engine to return **empty results** (0 verts, 0 faces). Root cause: `process=False` loading leaves trimesh with `is_watertight=False` even when external CAD tools confirm watertightness. **Mitigation**: Use Path B (`process=True`) for production; dual-path test for verification.

### Memory Usage
Monitor RAM when unioning **>10k unique cylinders**. `Manifold.compose` and `to_mesh` scale with strut count. No hard limit observed yet; MariaTubeRack completes successfully.

---

## Workspace Layout (Post-Cleanup)

```
Graphite/
├── results/
│   └── tests/          # 20mm cube suite outputs, boundary comparisons
├── tools/
│   └── diagnostics/    # sweep_box_mesh, sweep_rack_mesh, diagnose_adapter_mesh, etc.
├── test_parts/         # Input STLs (Part2_Adapter, MariaTubeRack_Full, 20mm_cube)
├── graphite/
│   └── explicit/       # scaffold_module, topology_module, geometry_module
├── solver.py
├── run_adapter_lattice.py
├── run_mariatube_rack.py
└── generate_suite_20mm.py
```

---

## Next Steps (Prioritization)

1. **Batch Unioning**: Explore manifold3d batch/parallel union strategies for >10k cylinders.
2. **GMSH Simplification**: Reduce scaffold complexity (coarser mesh, algorithm tuning) for large parts.
3. **Production hardening**: Consolidate dual-path logic into a single `load_mesh_for_boolean()` helper.
