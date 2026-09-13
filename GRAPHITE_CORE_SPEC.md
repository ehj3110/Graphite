# Graphite Core Architecture Specification

> **Source of Truth for External Knowledge Base Integration**  
> *Target System:* Graphite Conformal Lattice R&D Engine  
> *Scope:* Dataclass schemas, array contracts, abstract protocols, functional signatures, coordinate systems, and mathematical conventions.  
> *Exclusions:* Internal logging configs, test fixtures, and non-interface implementation details.

---

## 1. Fundamental Array Shapes & Lattice Representations

Graphite maintains a strict mathematical boundary between explicit strut scaffolds, conformal hexahedral/tetrahedral elements, and implicit scalar fields.

### 1.1 Explicit Strut & Conformal Scaffolds

| Representation | Symbol / Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Node Coordinates** | `nodes` | `(N, 3)` | `float64` | Cartesian nodal coordinates \((x, y, z)\) in millimeters. |
| **Strut Topology** | `struts` / `edges` | `(M, 2)` or `(E, 2)` | `int64` | 0-indexed undirected node vertex pairs \((u, v)\) with \(u < v\). |
| **Hexahedral Elements** | `hex_elements` | `(E_hex, 8, 3)` | `float64` | 8 corner coordinates per hexahedron in standard hex8 ordering. |
| **Tetrahedral Elements** | `tet_elements` | `(E_tet, 4, 3)` | `float64` | 4 corner coordinates per tetrahedron in standard tet4 ordering. |
| **Strut Radii** | `strut_radius` | `(M,)` or `float` | `float64` | Cross-sectional strut cylinder radii in millimeters. |
| **Surface Mask** | `surface_mask` | `(N,)` | `bool` | Boolean indicator tagging nodes lying on the conformal skin / boundary. |
| **Native Surface Struts** | `native_surface_struts` | `(E_surf, 2)` | `int64` | Strut edges that lie strictly on exposed boundary faces (e.g., diamond perimeters). |
| **Node Degrees** | `valencies` | `(N,)` | `int64` | Incident strut count per node, used for valency-gated surface snapping. |

### 1.2 Implicit TPMS & Voxel Scalar Fields

| Representation | Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- | :--- |
| **3D Scalar Field** | `field` | `(N_x, N_y, N_z)` | `float32` or `float64` | Voxel grid containing signed distance or TPMS level-set evaluations. |
| **Grid Spacing** | `spacing` | `(3,)` tuple | `float64` | Physical voxel dimension \((dx, dy, dz)\) in millimeters. |
| **Grid Origin** | `origin` | `(3,)` tuple | `float64` | Minimum bounding box coordinate \((x_{\min}, y_{\min}, z_{\min})\) in mm. |
| **Isovalue Level** | `level` | scalar | `float64` | Isosurface zero-crossing contour threshold (default `0.0`). |

---

## 2. Dataclass Schemas

All dataclasses are strictly typed and located within the `graphite/` package tree.

### 2.1 Explicit Subsystem Schemas

#### `HexTopologyRule`
*Module:* [`graphite.explicit.hex_topology_module`](graphite/explicit/hex_topology_module.py)
```python
@dataclass(frozen=True)
class HexTopologyRule:
    """Definition of a specific hexahedral lattice topology rule."""
    name: str
    builder: HexBuilder  # Callable[[ndarray (8, 3)], tuple[ndarray (V, 3), ndarray (E, 2)]]
    cage_mode: str = "surface_dual"
    overlap_factor: float | Callable[[float], float] = 0.85
    conform_dofs: frozenset[str] = frozenset({"face_centroids"})
    skin_mode: str = "face_centroid_dual"
    valency_cutoff: int = 4
```

#### `SurfaceStampTags`
*Module:* [`graphite.explicit.hex_topology_module`](graphite/explicit/hex_topology_module.py)
```python
@dataclass
class SurfaceStampTags:
    """Global surface-dual tags after welding a multi-rule stamp."""
    surface_mask: np.ndarray  # shape: (V,), dtype: bool - explicitly tagged surface nodes
    native_surface_struts: np.ndarray  # shape: (E, 2), dtype: int64 - half diamond perimeters
    full_exposed_primaries: list[tuple[int, int, int]]  # (hex_i, face_i, gid)
    half_diamond_primaries: list[tuple[int, int, int]]  # (hex_i, face_i, gid)
```

#### `LeafSurfaceSpec`
*Module:* [`graphite.explicit.hex_rules`](graphite/explicit/hex_rules.py)
```python
@dataclass(frozen=True)
class LeafSurfaceSpec:
    """Explicit surface membership for one octahedral leaf rule."""
    surface_node_locals: tuple[int, ...]
    native_surface_strut_locals: tuple[tuple[int, int], ...]
    exposed_candidate_locals: tuple[int, ...] = ()
    local_to_face_index: tuple[int, ...] = ()
```

#### `TopologyRule` (Tetrahedral)
*Module:* [`graphite.explicit.rules.tet_topology_rules`](graphite/explicit/rules/tet_topology_rules.py)
```python
@dataclass(frozen=True)
class TopologyRule:
    """Definition of a specific lattice topology rule for conformal tetrahedral meshes."""
    name: str
    internal_builder: Callable[..., np.ndarray]  # returns struts (S, 2) int64
    cage_mode: str  # "surface_cage" | "surface_dual" | "edge_midpoints"
    overlap_factor: float | Callable[[float], float] = 0.85
```

#### `SharedEdgeFace` & `TrimSharedEdgeResult`
*Module:* [`graphite.explicit.sc_trim_shared_edge_engine`](graphite/explicit/sc_trim_shared_edge_engine.py)
```python
@dataclass
class SharedEdgeFace:
    face_id: int
    hex_i: int
    face_i: int
    is_midplane: bool
    corners_3d: np.ndarray  # shape: (4, 3), dtype: float64
    edges_3d: tuple[tuple[np.ndarray, np.ndarray], ...]
    node_gids: list[int]

@dataclass
class TrimSharedEdgeResult:
    rule_name: str
    origin_offset: np.ndarray  # shape: (3,), dtype: float64
    hex_elems: np.ndarray  # shape: (E_hex, 8, 3), dtype: float64
    volume_nodes: np.ndarray  # shape: (V_vol, 3), dtype: float64
    volume_struts: np.ndarray  # shape: (S_vol, 2), dtype: int64
    dual_nodes: np.ndarray  # shape: (V_dual, 3), dtype: float64
    dual_struts: np.ndarray  # shape: (S_dual, 2), dtype: int64
    surface_gids: set[int]
    faces: list[SharedEdgeFace]
    report: dict = field(default_factory=dict)
```

#### `NodalConformationResult`
*Module:* [`graphite.explicit.nodal_conformation`](graphite/explicit/nodal_conformation.py)
```python
@dataclass
class NodalConformationResult:
    rule_name: str
    origin_offset: np.ndarray  # shape: (3,), dtype: float64
    hex_elems: np.ndarray  # shape: (E_hex, 8, 3), dtype: float64
    volume_nodes: np.ndarray  # shape: (V_vol, 3), dtype: float64
    volume_struts: np.ndarray  # shape: (S_vol, 2), dtype: int64
    dual_nodes: np.ndarray  # shape: (V_dual, 3), dtype: float64
    dual_struts: np.ndarray  # shape: (S_dual, 2), dtype: int64
    dual_nodes_projected: np.ndarray  # shape: (V_dual, 3), dtype: float64
    surface_gids: set[int]
    report: dict = field(default_factory=dict)
```

#### `DependencyStatus`
*Module:* [`graphite.explicit.health`](graphite/explicit/health.py)
```python
@dataclass(frozen=True)
class DependencyStatus:
    module: str
    present: bool
    version: str | None
```

#### `Ring`
*Module:* [`graphite.explicit.interlinked.patterns`](graphite/explicit/interlinked/patterns.py)
```python
@dataclass
class Ring:
    """Parametric closed ring representation for print-in-place chainmail and interlinked lattices."""
    center: np.ndarray  # shape: (3,), dtype: float64 - Cartesian coordinates of ring center in mm
    normal: np.ndarray  # shape: (3,), dtype: float64 - unit normal vector of the ring plane
    radius: float  # major radius (center to wire centerline) in mm
    wire_radius: float  # minor radius (strut wire cross-section radius) in mm
    nodes: np.ndarray  # shape: (V, 3), dtype: float64 - discretized polygonal nodes
    struts: np.ndarray  # shape: (S, 2), dtype: int64 - closed-loop edge connectivity
    tag: str = ""  # semantic tag ('flat', 'arch_x', 'arch_y', 'tilt_pos', 'tilt_neg')
    cell_index: tuple[int, ...] = field(default_factory=tuple)
```

#### `InterlinkedConfig`
*Module:* [`graphite.explicit.interlinked.generator`](graphite/explicit/interlinked/generator.py)
```python
@dataclass
class InterlinkedConfig:
    """Configuration specification for explicit interlinked lattice generation."""
    pattern: str = "european_4in1"  # 'european_4in1' | 'kusari' | 'cubic_8ring' | 'volumetric_kusari'
    pitch: float = 10.0  # grid cell pitch L in mm
    radius_ratio: float = 0.65  # ratio of major radius to pitch (R / L)
    wire_radius: float = 0.40  # strut wire radius r in mm
    tilt_angle_deg: float = 28.0  # weave tilt angle in degrees for European 4-in-1
    grid_size: tuple[int, int, int] = (5, 5, 1)  # (nx, ny, nz) grid dimensions
    num_ring_segments: int = 24  # discretization segments per ring
    min_clearance: float = 0.30  # required minimum surface-to-surface gap in mm
    cull_margin: float = 0.50  # SDF inset culling buffer margin in mm
    add_spheres: bool = True  # fillet spheres at ring polygon nodes
    flat_radius: float | None = None
    arch_radius: float | None = None
    arch_z_radius: float | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
```

#### `InterlinkedLatticeResult`
*Module:* [`graphite.explicit.interlinked.generator`](graphite/explicit/interlinked/generator.py)
```python
@dataclass
class InterlinkedLatticeResult:
    """Output payload of the interlinked lattice generation pipeline."""
    mesh: trimesh.Trimesh
    rings: list[Ring]
    num_rings: int
    num_nodes: int
    num_struts: int
    min_clearance: float
    clearance_valid: bool
    volume: float
    bounds: np.ndarray  # shape: (2, 3), dtype: float64
    metadata: dict = field(default_factory=dict)
```

#### `ImportedInterlinkedCell`
*Module:* [`graphite.explicit.interlinked.importer`](graphite/explicit/interlinked/importer.py)
```python
class ImportedInterlinkedCell:
    """Encapsulates an imported multi-body kinematic unit cell mesh (e.g. NASA space fabric)."""
    full_mesh: trimesh.Trimesh
    bounds: np.ndarray  # shape: (2, 3), dtype: float64
    extents: np.ndarray  # shape: (3,), dtype: float64
    body_manifolds: list[m3d.Manifold]
    plate_idx: int | None
    ring_idx: int | None
    arm_indices: list[int]
    symmetry_order: int  # 6 for hexagonal, 4 for cartesian
    lattice_type: str  # 'hexagonal' | 'cartesian'
    estimated_pitch: float

    def build_instance(
        self,
        scale_xy: float = 1.0,
        scale_z: float = 1.0,
        scale_arms: float = 1.0,
        scale_plate_t: float = 1.0,
    ) -> m3d.Manifold: ...
```

#### `NasaHexagonCell`
*Module:* [`graphite.explicit.interlinked.nasa_hexagon`](graphite/explicit/interlinked/nasa_hexagon.py)
```python
class NasaHexagonCell:
    """Parametric generator for a single 6-fold symmetric NASA Space Fabric tile."""
    pitch: float = 12.75  # center-to-center tile spacing d in mm
    wire_radius: float = 0.30  # nominal strut wire radius in mm
    plate_radius: float = 7.0  # circumscribed base plate radius in mm (flats at +-6.062 mm)
    plate_thickness: float = 0.45  # base plate thickness in mm
    ring_radius: float = 3.75  # major radius of top torus in mm
    ring_height: float = 5.898  # torus height above base in mm

    def build_manifold(self) -> m3d.Manifold: ...
```

#### `PAMParticle`
*Module:* [`graphite.explicit.interlinked.pams`](graphite/explicit/interlinked/pams.py)
```python
@dataclass
class PAMParticle:
    """Discrete polycatenated wireframe particle with a local node/strut index space."""
    particle_id: int
    nodes: np.ndarray  # shape: (V, 3), dtype: float64
    struts: np.ndarray  # shape: (S, 2), dtype: int64 — local indices only (no cross-particle merge)
    center: np.ndarray  # shape: (3,), dtype: float64
    geometry_type: str  # 'TET' | 'CO' | 'OCT' | 'ring' | ...
    metadata: dict = field(default_factory=dict)
```

#### `PAMLatticeResult`
*Module:* [`graphite.explicit.interlinked.pams`](graphite/explicit/interlinked/pams.py)
```python
@dataclass
class PAMLatticeResult:
    """Multi-body PAM assembly with DfAM clearance diagnostics."""
    particles: list[PAMParticle]
    tripartite_code: str  # e.g. 'D-4-TET', 'C-6-CO'
    clearance_valid: bool
    min_clearance_mm: float
    strut_radius: float = 0.8
    meshes: list[trimesh.Trimesh] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def combined_mesh(self) -> trimesh.Trimesh: ...
```

#### `LatticeGraph`
*Module:* [`graphite.generators.pentamode`](graphite/generators/pentamode.py)
```python
@dataclass
class LatticeGraph:
    """Represents an explicit nodal graph with per-strut cross sections."""
    nodes: np.ndarray  # shape: (V, 3), dtype: float64
    struts: np.ndarray  # shape: (E, 2), dtype: int64
    radii: np.ndarray | float  # shape: (E,) or scalar float64
    metadata: dict = field(default_factory=dict)

    def to_trimesh(self, add_spheres: bool = True) -> trimesh.Trimesh: ...
    def export_inp(self, filename: str | Path, element_type: str = "B31") -> Path: ...
```

#### `ImplicitField`
*Module:* [`graphite.generators.pentamode`](graphite/generators/pentamode.py)
```python
@dataclass
class ImplicitField:
    """Represents a 3D scalar distance field grid for isosurface extraction."""
    field: np.ndarray  # shape: (Nx, Ny, Nz), dtype: float32 or float64
    origin: tuple[float, float, float]  # (xmin, ymin, zmin)
    spacing: tuple[float, float, float]  # (dx, dy, dz)
    metadata: dict = field(default_factory=dict)

    def to_trimesh(self, level: float = 0.0) -> trimesh.Trimesh: ...
```

---

### 2.2 Implicit & Calibration Schemas

#### `IsosurfaceExtractionResult`
*Module:* [`graphite.implicit.meshing_backends`](graphite/implicit/meshing_backends.py)
```python
@dataclass(frozen=True)
class IsosurfaceExtractionResult:
    """Dataclass wrapping the result of an isosurface extraction."""
    mesh: trimesh.Trimesh
    backend_requested: str
    backend_used: str
    fallback_used: bool
    fallback_reason: str | None
    runtime_seconds: float
    notes: list[str]
```

#### `WoodpileLatticeSpec`
*Module:* [`graphite.implicit.woodpile_input`](graphite/implicit/woodpile_input.py)
```python
@dataclass
class WoodpileLatticeSpec:
    """Piecewise woodpile / cross-hatch lattice in a cylinder or axis-aligned box."""
    domain: Literal["cylinder", "box"] = "cylinder"
    diameter_mm: float = 2.0
    width_x_mm: float = 1.0
    depth_y_mm: float = 1.0
    height_mm: float = 1.0
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    origin_z_mm: float = 0.0
    resolution_mm: float = 0.01
    true_woodpile: bool = False
    invert_solids: bool = False
    repair_mesh: bool = True
    z_breaks_mm: list[float] = field(default_factory=lambda: [0.0, 2.23, 2.31, 2.7])
    pore_mm: list[float] = field(default_factory=lambda: [0.8, 0.4, 0.2])
    anchor_mode: WoodpileAnchorMode = "center_void"  # "center_void" | "center_solid"
    alternate_band_orientation: bool = True
    combine_mode: str = "single-pass"
    generator: Literal["implicit", "extrude"] = "extrude"
    stem: str | None = None
```

#### `CalibrationConfig` & `CalibrationPointResult`
*Module:* [`graphite.implicit.calibration`](graphite/implicit/calibration.py)
```python
@dataclass(frozen=True)
class CalibrationConfig:
    pore_tolerance_mm: float = 0.03
    solid_fraction_tolerance: float = 0.02
    max_iterations: int = 20
    damping: float = 0.7
    fd_relative_step_L: float = 0.03
    fd_absolute_step_tau: float = 0.02
    min_L_mm: float = 0.05
    max_L_mm: float = 50.0
    min_tau: float = 0.01
    max_tau: float = 2.0
    sample_resolution_mm: float = 0.02
    sample_cells: float = 2.0
    boundary_guard_cells: float = 0.5
    min_voxels_per_period: int = 28

@dataclass(frozen=True)
class CalibrationPointResult:
    lattice_type: str
    target_pore_mm: float
    target_solid_fraction: float
    calibrated_L_mm: float
    calibrated_tau: float
    converged: bool
    iterations: int
    residual_pore_mm: float
    residual_solid_fraction: float
    measured_pore_mm: float
    measured_solid_fraction: float
    hit_bounds: bool
    history: list[CalibrationIteration] = field(default_factory=list)
    seed_source: str = "heuristic"
    notes: list[str] = field(default_factory=list)
```

#### `TpmsParameterLookup`
*Module:* [`graphite.implicit.tpms_parameter_lut`](graphite/implicit/tpms_parameter_lut.py)
```python
@dataclass(frozen=True)
class TpmsParameterLookup:
    lattice_type: str
    target_pore_mm: float
    target_solid_fraction: float
    L_mm: float
    tau: float
    measured_pore_mm: float
    measured_solid_fraction: float
    converged: bool
    residual_pore_mm: float
    residual_solid_fraction: float
    notes: list[str]
```

#### `MeshHealthReport`, `MeshRepairConfig` & `ExportResult`
*Module:* [`graphite.io.mesh_export`](graphite/io/mesh_export.py)
```python
@dataclass
class MeshHealthReport:
    """Detailed topological and geometric health report for a mesh."""
    vertex_count: int = 0
    face_count: int = 0
    euler_characteristic: int = 0
    is_watertight: bool = False
    boundary_edges: int = 0
    boundary_loops: int = 0
    non_manifold_edges: int = 0
    non_manifold_vertices: int = 0
    self_intersections: int | None = None
    is_export_ready: bool = False
    notes: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class MeshRepairConfig:
    """Options for automated mesh repair and validation prior to export."""
    auto_repair: bool = True
    repair_mode: Literal["gentle", "full", "explicit", "implicit", "none"] = "gentle"
    escalate_to_full: bool = False
    poisson_depth: int = 10
    max_non_manifold_edges: int = 0
    max_boundary_loops: int = 0
    check_intersections: bool = False
    log_health: bool = True

@dataclass
class ExportResult:
    """Paths and metadata from export_mesh."""
    paths_written: list[Path] = field(default_factory=list)
    face_count: int = 0
    vertex_count: int = 0
    watertight: bool = False
    step_notes: list[str] = field(default_factory=list)
    health_report: MeshHealthReport | None = None
```

---

## 3. Abstract Base Protocols & Functional Signatures

### 3.1 Hex Unit Cell Builders

```python
HexBuilder = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""
Contract for single-cell hexahedral topology generators.

Parameters:
    coords: ndarray of shape (8, 3), dtype float64.
            Corner coordinates of the hex element in standard hex8 ordering.

Returns:
    nodes: ndarray of shape (V, 3), dtype float64.
           Generated node coordinates in physical space.
    struts: ndarray of shape (E, 2), dtype int64.
            Undirected strut connectivity edges indexing into `nodes`.
"""
```

Canonical Hex Builder Functions in [`graphite.explicit.hex_rules`](graphite/explicit/hex_rules.py):
- `apply_hex_grid(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 8 corner nodes, 12 cube edge struts.
- `apply_hex_octahedral(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 6 quad face center nodes, 12 adjacent face-pair struts (regular octahedron).
- `apply_hex_octahedral_half_neg_z(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (and `pos_z`, `neg_x`, `pos_x`, `neg_y`, `pos_y`)  
  *Returns:* 5 nodes (1 apex + 4 equatorial face centers), 8 struts (4 apex spokes + 4 diamond perimeter edges).
- `apply_hex_star(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 9 nodes (1 centroid + 8 corners), 8 centroid-to-corner struts.
- `apply_hex_octet_truss(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 14 nodes (8 corners + 6 face centers), 36 struts.
- `apply_hex_cross(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 14 nodes (8 corners + 6 face centers), 36 struts (24 face diagonal spokes + 12 perimeter struts).
- `apply_hex_dual(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 1 node (centroid), 0 local struts (struts formed globally between adjacent centroids).
- `apply_hex_face_dual(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 6 face centers, 12 adjacent face struts.
- `apply_hex_kelvin14(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 24 nodes, 36 struts mapped into hex brick via trilinear interpolation.
- `apply_hex_tesseract(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  
  *Returns:* 16 nodes, 32 struts mapped into hex brick via trilinear interpolation.

### 3.2 Tetrahedral Topology Builders

```python
InternalBuilder = Callable[..., np.ndarray]
"""
Vectorized builder for internal struts of conformal tetrahedral elements.

Parameters:
    corners: ndarray of shape (N_tet, 4, 3)
    face_centers: ndarray of shape (N_faces, 3)
    edge_midpoints: ndarray of shape (N_edges, 3)
    centroids: ndarray of shape (N_tet, 3)

Returns:
    struts: ndarray of shape (S, 2), dtype int64.
"""
```

Canonical Tet Topology Rules in [`graphite.explicit.rules.tet_topology_rules`](graphite/explicit/rules/tet_topology_rules.py):
- `vertex_to_centroid` / `rhombic`
- `edge_midpoints`
- `centroid_to_faces`
- `face_to_edges`
- `centroid_to_edges`
- `kagome`

### 3.3 Chiral & Auxetic Metamaterial Generators

Canonical 2D / Cylindrical Chiral Generators in [`graphite.explicit.chiral_cell`](graphite/explicit/chiral_cell.py):
- `generate_tetrachiral_cell(L: float, r: float, t: float, n_circle_segments: int = 16, chiral: bool = True) -> tuple[np.ndarray, np.ndarray, dict]`  
  *Returns:* 2D nodes `(N, 2)`, struts `(S, 2)`, metadata `dict` with lattice pitch \(D = \sqrt{L^2 + 4r^2}\).
- `generate_trichiral_cell(L: float, r: float, t: float, n_circle_segments: int = 16, chiral: bool = True) -> tuple[np.ndarray, np.ndarray, dict]`  
  *Returns:* 2D nodes `(N, 2)`, struts `(S, 2)`, metadata `dict` with lattice pitch \(D = \sqrt{3}(L + \sqrt{3}r)\).

### 3.4 Implicit TPMS Math Evaluators

```python
TPMSEvaluator = Callable[[np.ndarray, np.ndarray, np.ndarray, float, float, bool], np.ndarray]
"""
Standard mathematical signature for TPMS level-set scalar fields.

Parameters:
    x, y, z: ndarray of float64 (broadcastable 3D coordinates).
    unit_cell_size: float (physical period length L in mm).
    iso_offset: float (level-set threshold tau, controlling wall thickness / solid fraction).
    is_sheet: bool (if True, evaluate as sheet network |f(x,y,z)| - tau; if False, solid f(x,y,z) - tau).

Returns:
    scalar_field: ndarray of float64 with shape matching broadcast(x, y, z).
"""
```

Canonical TPMS Functions in [`graphite.math.tpms`](graphite/math/tpms.py):
- `gyroid(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{gyroid}} = \sin(k x)\cos(k y) + \sin(k y)\cos(k z) + \sin(k z)\cos(k x)$$
- `schwarz_p(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{schwarz\_p}} = \cos(k x) + \cos(k y) + \cos(k z)$$
- `schwarz_d(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{schwarz\_d}} = \sin(k x)\sin(k y)\sin(k z) + \sin(k x)\cos(k y)\cos(k z) + \cos(k x)\sin(k y)\cos(k z) + \cos(k x)\cos(k y)\sin(k z)$$
- `lidinoid(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{lidinoid}} = \sin(2kx)\cos(ky)\sin(kz) + \sin(2ky)\cos(kz)\sin(kx) + \sin(2kz)\cos(kx)\sin(ky) - \cos(2kx)\cos(2ky) - \cos(2ky)\cos(2kz) - \cos(2kz)\cos(2kx) + 0.3$$
- `split_p(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{split\_p}} = 1.1\,t_1 - 0.2\,t_2 - 0.4\,t_3$$
  where \(t_1 = \sum_{\text{cyc}} \sin(2kx)\sin(kz)\cos(ky)\), \(t_2 = \sum_{\text{cyc}} \cos(2kx)\cos(2ky)\), \(t_3 = \sum_{\text{cyc}} \cos(2kx)\).
- `neovius(x, y, z, unit_cell_size, iso_offset=0.0, is_sheet=True)`:
  $$f_{\text{neovius}} = 3(\cos(kx) + \cos(ky) + \cos(kz)) + 4\cos(kx)\cos(ky)\cos(kz)$$

where \(k = \frac{2\pi}{L}\).

### 3.5 Meshing & Geometry Synthesis Routines

#### `extract_isosurface`
*Module:* [`graphite.implicit.meshing_backends`](graphite/implicit/meshing_backends.py)
```python
def extract_isosurface(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    backend: str = "pyvista_flying_edges",  # "pyvista_flying_edges" | "auto" | "marching_cubes"
    level: float = 0.0,
    postprocess: bool = True,
    fill_holes: bool = True,
    enforce_watertight: bool = False,
) -> IsosurfaceExtractionResult:
    """Extract an isosurface mesh from a 3D scalar field."""
```

#### `generate_geometry`
*Module:* [`graphite.explicit.geometry_module`](graphite/explicit/geometry_module.py)
```python
def generate_geometry(
    nodes: np.ndarray,  # (N, 3) float64
    struts: np.ndarray,  # (M, 2) int64
    strut_radius: float | np.ndarray,  # scalar or (M,) float64
    boundary_mesh: trimesh.Trimesh | None = None,
    add_spheres: bool = False,
    joint_sphere_scale: float = 1.15,
    trim_strut_ends: bool | None = None,
    crop_to_boundary: bool = True,
    return_manifold: bool = False,
    circular_segments: int = 16,
) -> trimesh.Trimesh | tuple[trimesh.Trimesh, float] | tuple[manifold3d.Manifold, float]:
    """
    Generate explicit lattice solid geometry from topology nodes + struts
    using native manifold3d CSG cylinders, node spheres, and boolean trimming.
    ``circular_segments`` controls cylinder and joint-sphere tessellation (>=16 recommended for round caps).
    """
```

#### `generate_conformal_lattice`
*Module:* [`graphite.explicit.conformal_generator`](graphite/explicit/conformal_generator.py)
```python
def generate_conformal_lattice(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float],
    target_solid_fraction: float = 0.15,
    lattice_type: str = "SC",
    rule_name: str = "octahedral",
    **kwargs,
) -> dict[str, Any]:
    """Unified entrypoint for conformal lattice generation."""
```

#### Lofted Scaffold & Multi-Lattice APIs
*Module:* [`graphite.explicit.lofted_scaffold`](graphite/explicit/lofted_scaffold.py)
```python
def compute_equal_phase_stations(
    s_min: float,
    s_max: float,
    control_points: list[tuple[float, float]] | np.ndarray,
    *,
    target_n_stations: int | None = None,
) -> tuple[np.ndarray, int]:
    """Analytical equal-phase station placement from 1D target cell height control points."""

def generate_lofted_hex_scaffold(
    mesh: trimesh.Trimesh,
    *,
    nx: int = 8,
    ny: int = 4,
    nz: int = 4,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    station_coords: np.ndarray | list[float] | None = None,
    station_control_points: list[tuple[float, float]] | np.ndarray | None = None,
    snap_surface_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate structured hex scaffold lofted along a spine axis with optional equal-phase grading."""

def synthesize_lofted_lattice(
    hex_elements: np.ndarray,
    *,
    rule_name: str | list[str] | dict[int, str] | Callable[[int], str] = "octahedral",
    rule_schedule: list[str] | dict[int, str] | Callable[[int], str] | None = None,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    grid_shape: tuple[int, int, int] | None = None,
    insert_interface_pyramids: bool = True,
    topology_round_decimals: int = 6,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Synthesize uniform or multi-lattice topologies onto deformed lofted hex bricks."""

def synthesize_lofted_multilattice(
    hex_elements: np.ndarray,
    rule_schedule: list[str] | dict[int, str] | Callable[[int], str],
    *,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    grid_shape: tuple[int, int, int] | None = None,
    insert_interface_pyramids: bool = True,
    topology_round_decimals: int = 6,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Synthesize multi-lattice topology across lofted hex layers with automatic transition pyramids."""
```

### 3.7 Pentamode & Interlocking Metamaterial Generators

#### Transversely Isotropic Hexagonal Pentamode
*Module:* [`graphite.generators.hexagonal_pentamode`](graphite/generators/hexagonal_pentamode.py)
```python
def ab_layer_shift(a: float, c: float = 0.0) -> np.ndarray:
    """AB translation (a*sqrt(3)/2, a/2, c); no honeycomb rotation."""

def generate_transverse_hexagonal_pentamode(
    nx: int = 3,
    ny: int = 3,
    nz: int = 1,
    a: float = 1.0,
    c: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ABAB stacked honeycombs; returns nodes (N,3), edges (M,2), roles (M,)."""

def hexagonal_pentamode_cell(
    a: float = 1.0,
    c: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Single AB unit (nx=ny=nz=1)."""

def generate_hexagonal_pentamode_graph(
    bounds=None,
    a: float = 1.0,
    c: float = 1.5,
    *,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def generate_hexagonal_pentamode_lattice(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    a: float = 5.0,
    c: float = 7.5,
    r_min: float = 0.25,
    r_max: float = 0.80,
    r_basal: float = 0.20,
    output_format: Literal["graph", "implicit_sdf", "mesh"] = "graph",
    grid_resolution: int = 64,
    crop_to_bounds: bool = False,
) -> LatticeGraph | ImplicitField | trimesh.Trimesh:
    """Hex pentamode: basal cylinders (r_basal) + vertical bicones r(t)=r_min+4(r_max-r_min)t(1-t).
    LatticeGraph.metadata['strut_roles'] stores per-strut role codes.
    """
```

#### Continuous Pentamode (Meta-Fluid) Networks
*Module:* [`graphite.generators.pentamode`](graphite/generators/pentamode.py)
```python
def generate_pentamode_lattice(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    unit_cell_size: float = 10.0,
    r_min: float = 0.35,
    r_max: float = 0.90,
    hierarchical: bool = False,
    sub_element_type: Literal["tetrahedral", "c15"] = "tetrahedral",
    output_format: Literal["graph", "implicit_sdf", "mesh"] = "graph",
    grid_resolution: int = 64,
) -> LatticeGraph | ImplicitField | trimesh.Trimesh:
    """Generate diamond-cubic coordination (Z=4, bond angle ~109.47 deg) with biconical strut envelopes.
    
    Parameters:
        bounds: ((xmin, ymin, zmin), (xmax, ymax, zmax)) bounding volume box.
        unit_cell_size: Diamond cubic cubic cell dimension L in mm.
        r_min: Strut waist radius at mid-span in mm.
        r_max: Strut terminal radius at nodal connections in mm.
        hierarchical: When True, substitutes solid struts with secondary space-frame trusses.
        sub_element_type: 'tetrahedral' or 'c15' hierarchical infill topology.
        output_format: 'graph' (LatticeGraph), 'implicit_sdf' (ImplicitField), or 'mesh' (watertight trimesh.Trimesh).
        grid_resolution: Voxel resolution per axis for 'implicit_sdf' output.
    """
```

#### Interlocking Kinematic Contact Assemblies
*Module:* [`graphite.generators.interlocking`](graphite/generators/interlocking.py)
```python
def generate_interlocking_auxetic_sheet(
    dimensions: tuple[int, int] = (3, 3),
    cell_pitch: float = 10.0,
    clearance_gap: float = 0.40,
    cell_topology: Literal["reentrant_bowtie", "hook_array", "european_ring", "kusari_ring"] = "reentrant_bowtie",
) -> list[trimesh.Trimesh]:
    """Generate discrete, non-welded kinematic unit cell chains with guaranteed clearance gap > 0."""

def combine_interlocking_meshes(
    meshes: list[trimesh.Trimesh],
) -> trimesh.Trimesh:
    """Concatenate discrete unbonded meshes into a single multi-body Trimesh for visualization and STL export."""

def verify_interlocking_clearance(
    meshes: list[trimesh.Trimesh],
    max_neighbor_distance: float = 20.0,
) -> dict[str, Any]:
    """Verify physical clearance delta > 0 and zero boolean volume collision across all adjacent cell pairs."""
```

#### Polycatenated Architected Materials (PAMs)
*Module:* [`graphite.explicit.interlinked.pams`](graphite/explicit/interlinked/pams.py)
```python
def generate_tetrahedral_particle(
    edge_length: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
    tip_direction: np.ndarray | Sequence[float] | None = None,
    twist_rad: float = 0.0,
) -> PAMParticle:
    """4-vertex / 6-strut TET wireframe; optional 3-fold tip axis for diamond corner catenation."""

def generate_cuboctahedral_particle(
    size: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """12-vertex / 24-strut cuboctahedron; vertices size*(±1,±1,0)/√2 permutations."""

def generate_truncated_tetrahedron_particle(
    size: float = 1.0,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """12-vertex / 18-strut truncated tetrahedron wireframe; 4 hexagonal faces + 4 triangular face cutouts."""

def generate_octahedral_particle(
    size: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """6-vertex / 12-strut octahedron; vertices size*(±e_i)."""

def align_particle_axis(
    particle: PAMParticle,
    source_axis: np.ndarray | Sequence[float],
    target_axis: np.ndarray | Sequence[float],
) -> PAMParticle:
    """Rodrigues-align an internal symmetry axis to a network direction."""

def generate_d4tet_interlocked_pair(
    edge_length: float = 12.0,
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    twist_rad: float | None = None,
) -> PAMLatticeResult:
    """Two dual TET cages on one diamond bond (cylinder joints via generate_geometry)."""

def generate_d4tet_diamond_tiling(
    repeats: tuple[int, int, int],
    edge_length: float | None = None,
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    conventional_cell_size: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """Tile D-4-TET on diamond supercell; size by edge_length or conventional_cell_size a (d=a√3/4)."""

def calibrate_d4tet_edge_length(
    conventional_cell_size: float,
    strut_radius: float,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
) -> float:
    """Find tet edge L for fixed diamond conventional cell a (default bond d=a√3/4)."""

def generate_c6co_cubic_tiling(
    repeats: tuple[int, int, int] = (2, 2, 2),
    size: float = 8.0,
    strut_radius: float = 0.35,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """C-6-CO coordination shell (1+6); Phase-2 cubic cell motif."""

def generate_c6tt_cubic_tiling(
    repeats: tuple[int, int, int] = (2, 2, 2),
    size: float = 10.0,
    strut_radius: float = 0.50,
    min_clearance: float = 0.30,
    *,
    unit_cell_size: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """3D bulk periodic polycatenated material with Truncated Tetrahedra (C-6-TT on pcu network)."""

def generate_j4oct_square_tiling(
    repeats: tuple[int, int, int] | tuple[int, int] = (2, 2),
    size: float = 6.0,
    strut_radius: float = 0.40,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """J-4-OCT planar cross (1+4 tip-to-tip neighbors)."""

def recalibrate_pam_lattice(
    lattice_type: str,
    target_d: float,
    r: float,
    repeats: tuple[int, int, int] = (2, 2, 2),
) -> float:
    """Automated calibration tool sweeping geometric parameters to maximize global clearance."""

def generate_pam_lattice(
    tripartite_code: str,
    unit_cell_size: float,
    repeats: tuple[int, int, int] = (1, 1, 1),
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
) -> PAMLatticeResult:
    """Tripartite X-n-abc entrypoint. Supported: D-4-TET, C-6-TT, C-6-CO, J-4-OCT."""
```

---

## 4. Coordinate Systems, Periodicity & Tiling Conventions

### 4.1 Canonical Hex8 Unit Cell & Indexing

Conformal hexahedral elements adhere to the standard finite element hex8 ordering in the unit cube \([0, 1]^3\):

```
       7 ------------- 6
      /|              /|
     / |             / |
    4 ------------- 5  |
    |  |            |  |
    |  3 -----------|-- 2
    | /             | /
    |/              |/
    0 ------------- 1
```

#### Corner Coordinates in Reference Space \([0, 1]^3\)
- `0`: \((0, 0, 0)\)
- `1`: \((1, 0, 0)\)
- `2`: \((1, 1, 0)\)
- `3`: \((0, 1, 0)\)
- `4`: \((0, 0, 1)\)
- `5`: \((1, 0, 1)\)
- `6`: \((1, 1, 1)\)
- `7`: \((0, 1, 1)\)

#### Quad Faces Definition & Outward Normals
Hex elements define 6 boundary faces indexed 0 through 5:

| Face Index | Corner Tuple | Plane Axis & Value | Outward Normal | Face Center Coordinate \((u, v, w)\) |
| :--- | :--- | :--- | :--- | :--- |
| **Face 0** | `(0, 1, 2, 3)` | \(w = 0.0\) | \([0, 0, -1]^T\) (\(-Z\)) | \((0.5, 0.5, 0.0)\) |
| **Face 1** | `(4, 5, 6, 7)` | \(w = 1.0\) | \([0, 0, +1]^T\) (\(+Z\)) | \((0.5, 0.5, 1.0)\) |
| **Face 2** | `(0, 1, 5, 4)` | \(v = 0.0\) | \([0, -1, 0]^T\) (\(-Y\)) | \((0.5, 0.0, 0.5)\) |
| **Face 3** | `(3, 2, 6, 7)` | \(v = 1.0\) | \([0, +1, 0]^T\) (\(+Y\)) | \((0.5, 1.0, 0.5)\) |
| **Face 4** | `(0, 3, 7, 4)` | \(u = 0.0\) | \([-1, 0, 0]^T\) (\(-X\)) | \((0.0, 0.5, 0.5)\) |
| **Face 5** | `(1, 2, 6, 5)` | \(u = 1.0\) | \([+1, 0, 0]^T\) (\(+X\)) | \((1.0, 0.5, 0.5)\) |

#### Adjacent Face Pairs (Octahedral Connectivity)
The 12 octahedral struts in an SC unit cell connect centroids of adjacent faces that share an edge:
```python
_ADJACENT_FACE_PAIRS = (
    (0, 2), (0, 3), (0, 4), (0, 5),  # -Z connected to -Y, +Y, -X, +X
    (1, 2), (1, 3), (1, 4), (1, 5),  # +Z connected to -Y, +Y, -X, +X
    (2, 4), (2, 5), (3, 4), (3, 5),  # equatorial face connections
)
```
Opposite parallel face pairs `(0, 1)`, `(2, 3)`, and `(4, 5)` are strictly **unconnected**.

---

### 4.2 Bounding Box Conventions: Origin-Centered vs. Positive Octant

| Topology / Unit Cell | Bounding Box Convention | Coordinate Range | Reference File |
| :--- | :--- | :--- | :--- |
| **Canonical Hex Conformal Brick** | Positive Octant | \([0, 1]^3\) | [`hex_rules.py`](graphite/explicit/hex_rules.py) |
| **Kelvin Cell (Truncated Octahedron)** | Origin-Centered | \([-L/2, L/2]^3\) | [`kelvin_cell.py`](graphite/explicit/kelvin_cell.py) |
| **Tesseract (Nested Cube)** | Origin-Centered | \([-L/2, L/2]^3\) | [`tesseract_cell.py`](graphite/explicit/tesseract_cell.py) |
| **A15 Frank-Kasper Basis** | Positive Octant | \([0, L]^3\) (Quarter slices \(\Delta = 0.25 L\)) | [`a15_kagome.py`](graphite/explicit/a15_kagome.py) |
| **Proven Topologies (SC, BCC, FCC)** | Positive Octant | \([0, L]^3\) tiled from fractional \([0, 1]^3\) | [`proven_topologies.py`](graphite/explicit/proven_topologies.py) |
| **TPMS Scalar Evaluations** | Arbitrary Domain | Evaluated on dense grid \([x_0, x_1] \times [y_0, y_1] \times [z_0, z_1]\) | [`math/tpms.py`](graphite/math/tpms.py) |
| **Woodpile / Cross-Hatch** | Positive Z Octant | Transverse \([x_0, x_1] \times [y_0, y_1]\), layers stacked along \(+Z\) | [`woodpile_extrude.py`](graphite/explicit/woodpile_extrude.py) |

#### A15 Frank-Kasper Crystallographic Basis in \([0, 1]^3\)
```python
A15_BASIS = np.array([
    [0.0,  0.0,  0.0 ],   # Corner
    [0.5,  0.5,  0.5 ],   # Center
    [0.25, 0.0,  0.5 ],   # X-face pair
    [0.75, 0.0,  0.5 ],
    [0.5,  0.25, 0.0 ],   # Y-face pair
    [0.5,  0.75, 0.0 ],
    [0.0,  0.5,  0.25],   # Z-face pair
    [0.0,  0.5,  0.75],
], dtype=np.float64)
```
Tiling increments strictly enforce quarter-cell bounds: \(k \times 0.25 \times \text{cell\_size}\).

---

### 4.3 Periodicity Basis Vectors & Gradient Integration

#### 1. Orthogonal Cartesian Periodicity
For uniform periodic lattices, the real-space basis vectors are:
$$\mathbf{a}_1 = \begin{bmatrix} L_x \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{a}_2 = \begin{bmatrix} 0 \\ L_y \\ 0 \end{bmatrix}, \quad \mathbf{a}_3 = \begin{bmatrix} 0 \\ 0 \\ L_z \end{bmatrix}$$
Angular spatial wavenumber:
$$k_x = \frac{2\pi}{L_x}, \quad k_y = \frac{2\pi}{L_y}, \quad k_z = \frac{2\pi}{L_z}$$

#### 2. Jacobian-Integrated Phase for Functional Grading (Spine along \(Z\))
To eliminate phase-shear distortion, frequency doubling, and local history rewriting across steep gradients, phase along the gradient axis \(Z\) is accumulated via path integration:
$$\omega(z) = \frac{2\pi}{L(z)}$$
$$W(z) = \int_0^z \omega(t)\,dt \quad \approx \quad \text{cumulative\_trapezoid}(\omega, z, \text{initial}=0)$$
Orthogonal coordinates are scaled by instantaneous local frequency:
$$U(x, z) = \omega(z) \cdot x, \quad V(y, z) = \omega(z) \cdot y$$
TPMS equations evaluate directly on phase coordinates \((U, V, W)\) with period \(2\pi\).

---

### 4.4 Transformation Matrix Layouts

#### 1. Manifold3D Affine Strut Placement
Strut cylinders are generated as canonical Z-aligned primitives centered at the origin:
`manifold3d.Manifold.cylinder(length, radius_start, radius_end, circular_segments, center=True)`
and placed via a \(3 \times 4\) row-major affine transformation matrix \([R \mid \mathbf{t}]\):
$$\mathbf{p}' = R \mathbf{p} + \mathbf{t}$$

- **Direction unit vector:** \(\mathbf{t}_{\text{dir}} = \frac{\mathbf{p}_1 - \mathbf{p}_0}{\|\mathbf{p}_1 - \mathbf{p}_0\|}\)
- **Midpoint translation:** \(\mathbf{t} = \frac{\mathbf{p}_0 + \mathbf{p}_1}{2}\)
- **Right-handed rotation matrix \(R = [\mathbf{u} \mid \mathbf{v} \mid \mathbf{t}_{\text{dir}}]\):**
  $$\mathbf{u} = \frac{\mathbf{up} \times \mathbf{t}_{\text{dir}}}{\|\mathbf{up} \times \mathbf{t}_{\text{dir}}\|}, \quad \mathbf{v} = \mathbf{t}_{\text{dir}} \times \mathbf{u}$$
  where \(\mathbf{up} = [0, 0, 1]^T\) (or \([1, 0, 0]^T\) if aligned within 0.95 with Z).
- **Row-major layout:**
  $$\begin{bmatrix}
  R_{00} & R_{01} & R_{02} & t_0 \\
  R_{10} & R_{11} & R_{12} & t_1 \\
  R_{20} & R_{21} & R_{22} & t_2
  \end{bmatrix}$$

#### 2. Trilinear Hexahedral Conformal Mapping
Any reference point \((u, v, w) \in [0, 1]^3\) is mapped to physical coordinate \(\mathbf{x}\) via 8-node trilinear shape functions:
$$\mathbf{x}(u, v, w) = \sum_{i=0}^7 N_i(u, v, w) \,\mathbf{c}_i$$
$$N_0 = (1-u)(1-v)(1-w), \quad N_1 = u(1-v)(1-w), \quad N_2 = uv(1-w), \quad N_3 = (1-u)v(1-w)$$
$$N_4 = (1-u)(1-v)w, \quad N_5 = u(1-v)w, \quad N_6 = uvw, \quad N_7 = (1-u)vw$$
where \(\mathbf{c}_i\) are the physical coordinates of the 8 hex corners.
