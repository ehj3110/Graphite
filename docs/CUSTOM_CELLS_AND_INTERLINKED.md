# Custom Unit Cells and Interlinked Lattices in Graphite

This document serves as the comprehensive engineering guide for creating, importing, grading, and tessellating custom unit cells in **Graphite Explicit** (welded structural trusses) and **Graphite Interlinked** (kinematic, non-welded multi-body metamaterials).

---

## 1. Explicit vs. Interlinked: Core Architectural Distinction

Graphite provides two fundamentally different explicit lattice paradigms:

| Characteristic | Standard Explicit (`graphite.explicit`) | Interlinked (`graphite.explicit.interlinked`) |
| :--- | :--- | :--- |
| **Physics / Mechanism** | **Rigid structural scaffold**: load-bearing, stiff trusses and frames | **Kinematic metamaterial**: print-in-place chainmail, fabrics, flexible joints |
| **Topology** | Open 1D graph edges $(u, v)$ sharing node vertices | Disjoint closed loops (tori) or multi-body hooks with clearance |
| **Joint Treatment** | **Welded**: intersecting struts are boolean-unioned at nodes | **Non-welded**: strict positive clearance ($\Delta > 0$) between bodies |
| **Boundary Conformation** | Trimming, surface snapping, surface duals, and planar cuts | **SDF Inset Culling**: culls whole rings to guarantee zero broken loops |
| **Aspect Ratio** | Variable (elongated cells, sheared hexahedra, conformal tets) | **1:1:1 aspect ratio** (circular loops or isotropic regular polygons) |
| **Grading Mode** | Variable node positions and cylinder radii $r(x)$ | Decoupled component scaling, pitch $d(x)$, and wire radius $r(x)$ |

---

## 2. Workflow A: Custom Unit Cells in Standard Graphite Explicit

To introduce a new structural unit cell into standard Graphite Explicit, define its canonical nodes and strut connectivity in normalized coordinate space $[0, 1]^3$, then register it via `register_custom_hex_truss`.

### Python API Example

```python
import numpy as np
from graphite.explicit.custom_rules import register_custom_hex_truss
from graphite.explicit.geometry_module import generate_geometry

# 1. Define canonical vertices in [0, 1]^3
diamond_nodes = np.array([
    [0.5, 0.5, 0.0],  # Bottom center
    [0.5, 0.5, 1.0],  # Top center
    [0.5, 0.0, 0.5],  # Front center
    [0.5, 1.0, 0.5],  # Back center
    [0.0, 0.5, 0.5],  # Left center
    [1.0, 0.5, 0.5],  # Right center
], dtype=np.float64)

# 2. Define strut connectivity (0-indexed vertex pairs)
diamond_struts = np.array([
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
], dtype=np.int64)

# 3. Register into the global HexTopologyRule registry
rule = register_custom_hex_truss("my_octahedron", diamond_nodes, diamond_struts)

# 4. Use in conformal hex meshing or evaluate builder on deformed hex corners
# corners shape: (8, 3) representing hex8 vertices
warped_nodes, warped_struts = rule.builder(hex_corners)
```

### Automatic Trilinear Warping & Shared-Face Welding
- **Trilinear Warping**: `custom_rules.trilinear_warp` automatically maps canonical $[0, 1]^3$ coordinates into any arbitrarily deformed, curved, or tapered conformal hexahedron.
- **Shared Boundary Welding**: When evaluated across multiple adjoining hexahedra, nodes lying on shared boundaries are deduplicated using high-precision coordinate hashing, and redundant coincident struts are purged.

---

## 3. Workflow B: Custom Cell STL Import & Decompilation

When you have an external CAD model or mesh of an intricate kinematic cell (e.g. `test_parts/nasa_fabric_hexagon.stl`), Graphite provides automated inspection and decompilation tools.

### 1. Automated Inspection CLI (`scripts/inspect_cell_stl.py`)

Run the inspector on any unit cell STL:
```bash
python scripts/inspect_cell_stl.py test_parts/nasa_fabric_hexagon.stl
```

**Inspector Outputs**:
1. **Body Count & Separation**: Splits disjoint bodies using mesh face connectivity.
2. **Topological Classification**:
   - Genus 0 ($\chi = 2$): Classifies base plates and curved hook legs.
   - Genus 1 ($\chi = 0$): Identifies through-hole rings and tori.
3. **Symmetry Order**: Evaluates azimuthal centroid distribution to automatically detect $C_4$ (Cartesian) or $C_6$ (hexagonal) symmetry.
4. **Estimated Pitch**: Recommends close-packed center-to-center tile spacing $d$.
5. **Code Emission**: Emits a self-contained Python draft class for the cell.

### 2. Multi-Body STL Tesselation (`ImportedInterlinkedCell`)

```python
from graphite.explicit.interlinked.importer import (
    ImportedInterlinkedCell,
    generate_hexagonal_sheet_seeds,
    tessellate_imported_cell,
)

# 1. Load multi-body cell
cell = ImportedInterlinkedCell("test_parts/nasa_fabric_hexagon.stl")

# 2. Generate rectangular seed array with alternating row offsets
seeds = generate_hexagonal_sheet_seeds(num_rows=4, num_cols=4, pitch=12.75)

# 3. Tessellate with optional component grading
mesh = tessellate_imported_cell(
    cell=cell,
    seed_points=seeds,
    scale_xy_field=1.0,     # overall cell size
    scale_arms_field=1.0,   # decoupled arm thickness
)
mesh.export("outputs/interlinked_review/my_hex_sheet.stl")
```

---

## 4. Workflow C: Parametric NASA JPL Space Fabric

Graphite includes a pure code-driven implementation of the NASA JPL 6-fold symmetric space fabric in `graphite.explicit.interlinked.nasa_hexagon`.

### Geometric Specification

1. **Base Hexagonal Plate**:
   - Width across flats $W = 2 R \cos(30^\circ) = R \sqrt{3}$.
   - For circumscribed radius $R = 7.0\text{ mm}$, $W \approx 12.124\text{ mm}$.
   - **Crucial Orientation**: Rotated by $30^\circ$ so that flat vertical edges lie at $x = \pm 6.062\text{ mm}$, aligning outward edge normals directly with lattice neighbor vectors ($0^\circ, 60^\circ, 120^\circ, 180^\circ, 240^\circ, 300^\circ$).
2. **Top Central Torus Ring**:
   - Revolved circular cross-section at height $Z = 5.898\text{ mm}$ with major radius $R_{\text{major}} = 3.75\text{ mm}$ and wire radius $r_{\text{wire}} = 0.60\text{ mm}$.
3. **Six Interlocking Hook Arms**:
   - Positioned at $60^\circ$ rotational increments.
   - Each arm loops outward across a flat edge to link with the opposing arm of the neighboring tile.

### Hexagonal Sheet Tiling Mathematics

```
          Row 1: y = dy, x offset = +d/2
             /\        /\
            /  \      /  \
           |    |----|    |
            \  /      \  /
             \/   dy   \/
             /\   |    /\
            /  \  v   /  \
           |    |----|    |
            \  /  <-dx-> /
             \/        \/
          Row 0: y = 0, x offset = 0
```

- **Nominal Pitch**: $d = 12.75\text{ mm}$ ($12.124\text{ mm} + 0.626\text{ mm}$ clearance gap).
- **Row Step in Y**: $\Delta y = \frac{\sqrt{3}}{2} d \approx 11.0418\text{ mm}$.
- **Alternating Row Offset in X**: $\Delta x = (j \bmod 2) \cdot \frac{d}{2} \approx 6.375\text{ mm}$.

### Continuous Spatial Grading API

```python
from graphite.explicit.interlinked.nasa_hexagon import generate_nasa_hexagon_lattice

# Define spatially varying wire radius field (thin/flexible center -> thick/rigid edge)
def wire_r_field(pt):
    dist_from_center = np.linalg.norm(pt[:2])
    return 0.22 + 0.16 * min(1.0, dist_from_center / 25.0)

mesh = generate_nasa_hexagon_lattice(
    num_rows=4,
    num_cols=4,
    pitch=12.75,
    wire_radius=0.30,
    wire_radius_field=wire_r_field,
    plate_radius=7.0,
    plate_thickness=0.45,
)
```

---

## 5. Verification & Testing

To run the complete test suite verifying both custom cell explicit registration and interlinked lattice generation:

```powershell
pytest tests/test_custom_cells.py tests/test_interlinked_lattice.py -v
python scripts/export_core_spec.py --check
```

**Test Coverage Highlights**:
- Geometric classification and C6 symmetry detection.
- Rectangular sheet row-offset coordinate verification.
- Positive print clearance ($\Delta \ge 0.18\text{ mm}$) and zero boolean intersection.
- Watertight multi-body mesh assembly via Manifold3D.
- Core specification contract consistency with `GRAPHITE_CORE_SPEC.md`.
