# Handoff: Universal Surface Dual Hardening & Grid Lattice Next Steps

**Date:** 2026-09-09  
**Status:** 
- Universal Surface Dual hardened & verified across 8 benchmark models (Octet, Star, Octahedral, Cross).
- All rule-name hardcodes eliminated.
- Next active objective: **Grid Lattice Conformation** (overbuild → chord-clip at interface → connect surface dual).

---

## 1. The Breakthrough: Fixing the Geometric Invariant in Rule 2.5

### The Root Cause Flaw ("Phantom Fallthrough")
In `graphite/explicit/sc_role_surface_dual.py`, Rule 2.5 promotes volume struts connecting perpendicular faces of the same hex (e.g. cut midplane on $Z$, exterior lateral face on $X$) to capture exterior boundary steps.

Previously, the code attempted to verify that the face along the remaining 3rd axis ($Y$, or `axis_c`) was not shared with an occupied neighbor cell:

```python
# FLAWED ORIGINAL LOGIC:
axis_c = 3 - axis_a - axis_b
if np.isclose(c_val_a, c_val_b):
    if np.isclose(c_val_a, c_mn) or np.isclose(c_val_a, c_mx):
        if len(face_owners[fkey]) > 1:
            continue
return True  # <-- BUG: Any strut in the middle of the cell fell through here!
```

**Why this broke multiple lattices:**
If a strut did not lie on the boundary plane of the 3rd axis (for example, if it traversed the **interior/center** of the cell at $Y = 12.7\text{ mm}$ rather than $Y = 0$ or $25.4\text{ mm}$), it skipped the check and executed `return True`. It promoted 3D volume chords cutting through the center of the cell into the 2D surface dual skin!

* In **Star**: Struts connecting the body center (node at $[12.7, 12.7, 12.7]$) to corners were falsely promoted, injecting **54 interior volume crossing chords** in Sloped and **58 in Angled**.
* In **Octahedral**: 3D pyramid apex spokes cutting through the cell interior leaked into the dual. Previously, a hacky patch (`if rule_name == "octahedral": continue`) was added to hide this symptom.
* In **`face_owners`**: `build_layered_surface_dual` constructed its face dictionary over *all* grid hexes (including empty, culled cells outside the CAD model), causing empty space to be treated as an "interior shared wall".

### The Fix Implemented

In `graphite/explicit/sc_role_surface_dual.py`:

1. **Strict 3rd-Axis Exterior Boundary Invariant:**
   A strut connecting perpendicular faces of the same hex is **only** surface skin if it lies along an exterior boundary plane of the remaining 3rd axis (`axis_c`). If it sits in the interior of the cell, it is rejected immediately:
   ```python
   # 1. Must lie in a constant plane along axis_c
   if not np.isclose(c_val_a, c_val_b, atol=1e-3):
       continue
   c_mn = float(np.min(corners[:, axis_c]))
   c_mx = float(np.max(corners[:, axis_c]))
   is_face_c = None
   if np.isclose(c_val_a, c_mn, atol=1e-3):
       is_face_c = False
   elif np.isclose(c_val_a, c_mx, atol=1e-3):
       is_face_c = True
   # IF NOT ON AN EXTERIOR BOUNDARY FACE, IT'S AN INTERIOR VOLUME STRUT:
   if is_face_c is None:
       continue
   # Verify the exterior boundary face is NOT shared with an active neighbor:
   fi_c = _AXIS_IS_MAX_TO_FACE[(axis_c, is_face_c)]
   fkey = _face_key(corners, _HEX_FACES[fi_c], edge_decimals)
   if len(face_owners.get(fkey, [])) > 1:
       continue
   return True
   ```
2. **Removed `if rule_name == "octahedral"`**:
   The engine is now **100% universal and rule-agnostic**. Octahedral apex spokes and Star body-center chords are cleanly rejected purely by geometry.
3. **Unified `face_owners`**:
   Standardized to `_EDGE_DECIMALS = 5` and filtered strictly to active, filled cells (`_hex_is_active(kp) and bool(_hex_filled_octants(elems[i], kp)[0])`).
4. **Adaptive Node Welding (`nodal_conformation.py`)**:
   `weld_combined_lattice` KDTree matching tolerance is now scale-adaptive: `max(1e-4 * cell_span, 1e-4)` instead of hardcoded `1e-3`. Deduplicates shared struts and updates surface nodes to projected coordinates.

---

## 2. Locked Reference Benchmarks (Do Not Break!)

Run verification via:
```bash
python -m pytest tests/test_sc_role_surface_dual.py tests/test_gold_octahedral_dual.py tests/test_sc_node_plane_trim.py -q
```
*Current test suite: **42 passed**.*

### Locked 8-Model Bookend Suite (`outputs/bookend_review/`)

| Part Variant | Lattice Rule | Volume Struts | Dual Struts | Welded Combined Struts | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sloped** | **Octet** | 1141 | 509 | **1210** | 100% exact match |
| **Angled** | **Octet** | 1575 | 560 | **1656** | 100% exact match |
| **Sloped** | **Octahedral** | 575 | 357 | **666** | 100% exact match (zero rule-name hardcodes) |
| **Angled** | **Octahedral** | 805 | 401 | **897** | 100% exact match (zero rule-name hardcodes) |
| **Sloped** | **Cross** | 670 | 463 | **898** | 100% exact match |
| **Angled** | **Cross** | 965 | 525 | **1241** | 100% exact match |
| **Sloped** | **Star** | 310 | **203** | **513** | 54 interior body-center struts eliminated |
| **Angled** | **Star** | 450 | **237** | **687** | 58 interior body-center struts eliminated |
| **Sloped** | **Grid** | **150** | **188** | **338** | Boundary raycast crop + perimeter dual (0 snapped outside) |
| **Angled** | **Grid** | **218** | **192** | **410** | Boundary raycast crop + perimeter dual (0 snapped outside) |

---

## 3. Grid Lattice Conformation (COMPLETED)

### Overview
Grid lattice conformity is fully resolved using the **Axis-Raycast Crop & Cyclical Perimeter Surface Dual** engine in `graphite/explicit/grid_boolean_engine.py`, wired into `nodal_conformation.py` under `rule == "grid"`.

### Key Architectural Solutions
1. **Axis-Raycast Truncation:**
   * Fires rays along $X, Y, Z$ Cartesian grid lines to find exact entry/exit pierce nodes on $\partial\Omega$.
   * Retains Cartesian interior chains between pierce nodes.
   * Eliminates cell-crushing and preserves 100% rigid, orthogonal interior cells.
2. **Cyclical Perimeter Surface Dual (No Crossing "X" Diagonals):**
   * Quad face boundaries are parameterized cyclically ($s \in [0, 4)$ counter-clockwise: Bottom $\to$ Right $\to$ Top $\to$ Left).
   * Connects adjacent pierce nodes along the perimeter walk, eliminating crossing interior diagonals.
3. **Surface Boundary Plane Containment Fix (`_is_inside_or_on_surface`):**
   * Solved false negatives from `trimesh.contains()` on coplanar boundary faces by checking signed distance ($|SDF| \le 10^{-4}\text{ mm}$).
   * Restores all boundary perimeter struts on $Z = 0$, $Z = 127$, and outer walls.
4. **Floor Shortcut Pruning & Stair-Step Preservation (`prune_floor_shortcuts`):**
   * Identified and resolved rogue diagonal struts on organic CAD geometries (e.g. mouse wrist rest) where 2D cell-face perimeter walks connected interior vertical column floor contacts ($Z \approx 0.15\text{ mm}$) to exterior wall pierces ($Z \ge 4.0\text{ mm}$) across empty interior bays.
   * `prune_floor_shortcuts=True` cleanly eliminates all 15 volume-cutting diagonals while preserving all 103 legitimate stair-step struts climbing the CAD slope tiers ($Z = 4 \to 8 \to 12$).
   * Backward-compatible default `prune_floor_shortcuts=False` preserves all locked bookend benchmark counts exactly.
5. **Mouse Wrist Rest Verification:**
   * Tested on `test_parts/Mouse wrist rest v1.stl` across standard ($12 \times 12 \times 4\text{ mm}$, 631 struts clean) and fine ($6 \times 6 \times 4\text{ mm}$) grids.
   * Watertight manifold solid: `outputs/grid_exploration_review/wrist_rest_grid_with_stair_steps.stl`.
   * Verified by automated unit test `test_grid_wrist_rest_prune_floor_shortcuts` in `tests/test_grid_conformation.py`.
   * Total test suite: **48 passed**.

