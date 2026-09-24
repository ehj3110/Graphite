"""
Graphite Explicit Interlinked — Modular Unit Cell Definitions (Phase 2)

Implements and registers standard InterlinkedCell classes:
- C6TTCell: Simple Cubic Truncated Tetrahedron (C-6-TT, n=6).
- D4TetCell: Diamond Tetrahedron (D-4-TET, n=4, bipartite A/B crystallographic dual basis).
- J4OctCell: Square-Planar Octahedron Cross (J-4-OCT, n=4, 45-deg relative twist).
- European4in1Cell: 2D European 4-in-1 chainmail (n=4, alternating row tilt +/-28 deg).
- JapaneseKusariCell: 2D Japanese Kusari chainmail (n=4, orthogonal flat + arch links).
- NasaSpaceFabricCell: 2D NASA JPL 6-fold symmetric space fabric (n=6, spiral hook arms).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np

from .particle import ParticleGeometry, InterlinkedParticle
from .cell import (
    BasisParticle,
    InterlinkedCell,
    InterlinkedRegistry,
    OrientationFn,
    _default_identity_orientation,
    C6TTCell,
)


# =============================================================================
# Helper: Analytical Wireframe Rings
# =============================================================================

def _build_ring_geometry(
    major_radius: float,
    num_segments: int = 24,
    geometry_type: str = "ring",
) -> ParticleGeometry:
    """Generate circular wireframe polygon nodes and closed-loop struts in XY plane."""
    R = float(major_radius)
    n = int(num_segments)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    nodes = np.column_stack([
        R * np.cos(angles),
        R * np.sin(angles),
        np.zeros(n, dtype=np.float64),
    ])
    struts = np.column_stack([
        np.arange(n, dtype=np.int64),
        np.roll(np.arange(n, dtype=np.int64), -1),
    ])
    return ParticleGeometry(
        nodes=nodes,
        struts=struts,
        bounding_radius=R,
        geometry_type=geometry_type,
        metadata={"major_radius": R, "num_segments": n},
    )


# =============================================================================
# 1. D-4-TET: Diamond Tetrahedron Cell
# =============================================================================

_TET_TEMPLATE = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)

_TET_STRUTS = np.array(
    [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3],
    ],
    dtype=np.int64,
)


def _build_tet_geometry(edge_length: float, dual: bool = False) -> ParticleGeometry:
    """Regular tetrahedron geometry centered at origin. If dual=True, applies point inversion."""
    L = float(edge_length)
    raw = _TET_TEMPLATE.copy()
    raw_edge = float(np.linalg.norm(raw[0] - raw[1]))
    nodes = raw * (L / raw_edge)
    nodes -= nodes.mean(axis=0)
    if dual:
        nodes = -nodes
    bounding_radius = float(np.max(np.linalg.norm(nodes, axis=1)))
    return ParticleGeometry(
        nodes=nodes,
        struts=_TET_STRUTS.copy(),
        bounding_radius=bounding_radius,
        geometry_type="TET",
        metadata={"edge_length": L, "dual": dual},
    )


@InterlinkedRegistry.register("d4tet")
@InterlinkedRegistry.register("d-4-tet")
class D4TetCell:
    """
    Diamond Tetrahedron (D-4-TET) PAM Unit Cell.
    Zhou et al., Science 2025. 4-fold corner catenation on Diamond (dia) network.
    Uses bipartite A/B basis with point-inversion crystallographic dual stagger.
    """
    name: str = "D-4-TET"
    family: str = "pam_polyhedra"
    parent_network: str = "dia"
    coordination_number: int = 4

    # 4 nearest neighbors in diamond FCC conventional cell:
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]

    # Validated empirical clearance factor when unit_cell_pitch is conventional cell size a_conv:
    # Delta = kappa * a_conv - D_strut, with kappa ≈ 0.1018 (full diamond crystal FCC+basis)
    CLEARANCE_KAPPA: float = 0.1018
    # Ratio of tet edge length to diamond bond length d: L / d ≈ 1.393
    # With d = a_conv * sqrt(3)/4 ≈ 0.4330 * a_conv, L ≈ 0.603 * a_conv
    DEFAULT_EDGE_RATIO: float = 0.603

    def __init__(self, edge_ratio: float = 0.603):
        self.edge_ratio = float(edge_ratio)
        proto_a = _build_tet_geometry(edge_length=1.0, dual=False)
        proto_b = _build_tet_geometry(edge_length=1.0, dual=True)

        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto_a,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "A"},
            ),
            BasisParticle(
                geometry=proto_b,
                fractional_offset=np.array([0.25, 0.25, 0.25], dtype=np.float64),
                sublattice_id="B",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "B"},
            ),
        ]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """Delta = kappa * a_conv - D_strut."""
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """a_conv = (target_clearance + strut_diameter) / kappa."""
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        """
        Instantiate diamond cell site.
        If context specifies 'sublattice' ('A' or 'B'), instantiate only that single
        crystallographic particle at site_origin. Otherwise (for local 2-particle cell),
        instantiate both basis particles (A and B).
        """
        ctx = context or {}
        a_conv = float(cell_pitch)
        L = a_conv * self.edge_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)

        if "sublattice" in ctx:
            sub = str(ctx["sublattice"]).upper()
            is_dual = (sub == "B")
            geom = _build_tet_geometry(edge_length=L, dual=is_dual)
            R = _default_identity_orientation(grid_index, orig, ctx)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = orig

            p = InterlinkedParticle(
                particle_id=int(id_start),
                geometry=geom,
                transform=T,
                sublattice_id=sub,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "edge_length": L,
                    "conventional_cell_size": a_conv,
                    "sublattice": sub,
                },
            )
            return [p]

        particles = []
        for i, bp in enumerate(self.basis_particles):
            is_dual = (bp.sublattice_id == "B")
            geom = _build_tet_geometry(edge_length=L, dual=is_dual)
            center = orig + bp.fractional_offset * a_conv

            R = bp.orientation_fn(grid_index, center, ctx)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = center

            p = InterlinkedParticle(
                particle_id=int(id_start + i),
                geometry=geom,
                transform=T,
                sublattice_id=bp.sublattice_id,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "edge_length": L,
                    "conventional_cell_size": a_conv,
                    **bp.metadata,
                },
            )
            particles.append(p)
        return particles


# =============================================================================
# 2. J-4-OCT: Square-Planar Octahedron Cross Cell
# =============================================================================

_OCT_NODES = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

_OCT_STRUTS = np.array(
    [
        [0, 2], [0, 3], [0, 4], [0, 5],
        [1, 2], [1, 3], [1, 4], [1, 5],
        [2, 4], [2, 5], [3, 4], [3, 5],
    ],
    dtype=np.int64,
)


def _build_oct_geometry(size: float = 1.0) -> ParticleGeometry:
    s = float(size)
    nodes = _OCT_NODES * s
    return ParticleGeometry(
        nodes=nodes,
        struts=_OCT_STRUTS.copy(),
        bounding_radius=s,
        geometry_type="OCT",
        metadata={"size": s, "edge_length": s * np.sqrt(2.0)},
    )


@InterlinkedRegistry.register("j4oct")
@InterlinkedRegistry.register("j-4-oct")
class J4OctCell:
    """
    Square-Planar Octahedron Cross (J-4-OCT) Unit Cell.
    Zhou et al., Science 2025. 4-fold tip catenation with 45-deg relative twist.
    """
    name: str = "J-4-OCT"
    family: str = "pam_polyhedra"
    parent_network: str = "square_2d"
    coordination_number: int = 4
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
    ]

    CLEARANCE_KAPPA: float = 0.1700
    DEFAULT_SIZE_RATIO: float = 0.75  # size / pitch

    def __init__(self, size_ratio: float = 0.75):
        self.size_ratio = float(size_ratio)
        proto = _build_oct_geometry(size=1.0)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.zeros(3, dtype=np.float64),
                sublattice_id="center",
                orientation_fn=self._checkerboard_orientation,
            )
        ]

    @staticmethod
    def _checkerboard_orientation(
        grid_index: tuple[int, int, int],
        site_center: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Apply 45-deg relative twist about the catenation bond axis.
        For square-planar grid (i, j):
        - Even parity ((i + j) % 2 == 0): unrotated identity orientation.
        - Odd parity along X (i % 2 != 0, j % 2 == 0): 45-deg twist about X.
        - Odd parity along Y (i % 2 == 0, j % 2 != 0): 45-deg twist about Y.
        - Diagonal odd parity: 45-deg twist about Z.
        """
        i, j, _ = grid_index
        parity = (i + j) % 2
        if parity == 0:
            return np.eye(3, dtype=np.float64)

        angle = 0.25 * np.pi
        c, s = np.cos(angle), np.sin(angle)
        if i % 2 != 0 and j % 2 == 0:
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s,  c],
            ], dtype=np.float64)
        elif i % 2 == 0 and j % 2 != 0:
            return np.array([
                [c,  0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ], dtype=np.float64)
        else:
            return np.array([
                [c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a0 = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a0 - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a0 = float(cell_pitch)
        s = a0 * self.size_ratio
        geom = _build_oct_geometry(size=s)
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)

        bp = self.basis_particles[0]
        R = bp.orientation_fn(grid_index, orig, ctx)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = orig

        p = InterlinkedParticle(
            particle_id=int(id_start),
            geometry=geom,
            transform=T,
            sublattice_id=bp.sublattice_id,
            cell_index=tuple(grid_index),
            metadata={"cell_name": self.name, "size": s, "unit_cell_pitch": a0},
        )
        return [p]


# =============================================================================
# 3. European 4-in-1: 2D/2.5D Chainmail Cell
# =============================================================================

@InterlinkedRegistry.register("european_4in1")
@InterlinkedRegistry.register("european-4in1")
@InterlinkedRegistry.register("euro_4in1")
class European4in1Cell:
    """
    Classic European 4-in-1 Chainmail Weave Unit Cell.
    Square translational grid with alternating row tilts (+/- theta) about the weave axis.
    Each ring loops through 4 coordinating neighbors.
    """
    name: str = "European 4-in-1"
    family: str = "chainmail_2d"
    parent_network: str = "square_2d"
    coordination_number: int = 4
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
    ]

    DEFAULT_RADIUS_RATIO: float = 0.65
    DEFAULT_TILT_DEG: float = 28.0

    def __init__(
        self,
        radius_ratio: float = 0.65,
        tilt_angle_deg: float = 28.0,
        num_segments: int = 24,
    ):
        self.radius_ratio = float(radius_ratio)
        self.tilt_angle_deg = float(tilt_angle_deg)
        self.num_segments = int(num_segments)

        proto = _build_ring_geometry(major_radius=1.0, num_segments=self.num_segments)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.zeros(3, dtype=np.float64),
                sublattice_id="ring",
                orientation_fn=self._alternating_tilt_orientation,
            )
        ]

    CLEARANCE_KAPPA: float = 0.1142

    def _alternating_tilt_orientation(
        self,
        grid_index: tuple[int, int, int],
        site_center: np.ndarray,
        context: dict[str, Any],
    ) -> np.ndarray:
        """
        Diagonal checkerboard weave: rotates +/- tilt_angle around diagonal axis [1, 1, 0]/sqrt(2)
        with checkerboard parity (i + j) mod 2. This ensures adjacent neighbors in all 4 directions
        loop through each other cleanly without coplanar edge intersections.
        """
        i, j, _ = grid_index
        sign = 1.0 if ((i + j) % 2 == 0) else -1.0
        theta_rad = np.radians(self.tilt_angle_deg * sign)

        k = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
        K = np.array([
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ], dtype=np.float64)

        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        R = np.eye(3, dtype=np.float64) * cos_t + K * sin_t + np.outer(k, k) * (1.0 - cos_t)
        return R

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """
        Calculates minimum surface clearance for diagonal checkerboard European 4-in-1:
        Delta = CLEARANCE_KAPPA * pitch - strut_diameter
        """
        a0 = float(unit_cell_pitch)
        d_wire = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a0 - d_wire)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """Invert clearance relationship to solve required grid pitch a0."""
        tc = float(target_clearance)
        d_wire = float(strut_diameter)
        return float((tc + d_wire) / self.CLEARANCE_KAPPA)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a0 = float(cell_pitch)
        R = a0 * self.radius_ratio
        geom = _build_ring_geometry(major_radius=R, num_segments=self.num_segments)
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)

        bp = self.basis_particles[0]
        R_mat = bp.orientation_fn(grid_index, orig, ctx)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = orig

        p = InterlinkedParticle(
            particle_id=int(id_start),
            geometry=geom,
            transform=T,
            sublattice_id=f"row_{grid_index[1] % 2}",
            cell_index=tuple(grid_index),
            metadata={
                "cell_name": self.name,
                "major_radius": R,
                "unit_cell_pitch": a0,
                "tilt_deg": self.tilt_angle_deg,
            },
        )
        return [p]


# =============================================================================
# 4. Japanese Kusari: Orthogonal Flat + Arch Rings Cell
# =============================================================================

@InterlinkedRegistry.register("japanese_kusari")
@InterlinkedRegistry.register("kusari")
class JapaneseKusariCell:
    """
    Japanese Kusari Chainmail Unit Cell.
    2D square network with multi-particle basis:
    - 1 Flat ring in XY plane at [0, 0, 0]
    - 1 Arch ring in XZ plane at [0.5, 0, 0]
    - 1 Arch ring in YZ plane at [0, 0.5, 0]
    """
    name: str = "Japanese Kusari"
    family: str = "chainmail_2d"
    parent_network: str = "square_2d"
    coordination_number: int = 4
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
    ]

    DEFAULT_FLAT_RATIO: float = 0.36
    DEFAULT_ARCH_RATIO: float = 0.41

    def __init__(
        self,
        flat_ratio: float = 0.36,
        arch_ratio: float = 0.41,
        num_segments: int = 24,
    ):
        self.flat_ratio = float(flat_ratio)
        self.arch_ratio = float(arch_ratio)
        self.num_segments = int(num_segments)

        proto_flat = _build_ring_geometry(major_radius=1.0, num_segments=self.num_segments, geometry_type="ring_flat")
        proto_arch = _build_ring_geometry(major_radius=1.0, num_segments=self.num_segments, geometry_type="ring_arch")

        # Rotation for XZ plane (rotate 90 deg about X)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        # Rotation for YZ plane (rotate 90 deg about Y)
        Ry = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto_flat,
                fractional_offset=np.array([0.0, 0.0, 0.0]),
                sublattice_id="flat",
                orientation_fn=_default_identity_orientation,
            ),
            BasisParticle(
                geometry=proto_arch,
                fractional_offset=np.array([0.5, 0.0, 0.0]),
                sublattice_id="arch_x",
                orientation_fn=lambda idx, c, ctx: Rx,
            ),
            BasisParticle(
                geometry=proto_arch,
                fractional_offset=np.array([0.0, 0.5, 0.0]),
                sublattice_id="arch_y",
                orientation_fn=lambda idx, c, ctx: Ry,
            ),
        ]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """Physical gap between flat ring and arch link."""
        a0 = float(unit_cell_pitch)
        d_wire = float(strut_diameter)
        rf = self.flat_ratio * a0
        ra = self.arch_ratio * a0
        # Centerline separation between flat and arch is 0.5 * a0
        # Overlap engagement is (rf + ra) - 0.5 * a0
        engagement = (rf + ra) - 0.5 * a0
        return float(engagement - d_wire)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d_wire = float(strut_diameter)
        denom = (self.flat_ratio + self.arch_ratio) - 0.5
        return float((tc + d_wire) / max(1e-6, denom))

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a0 = float(cell_pitch)
        rf = a0 * self.flat_ratio
        ra = a0 * self.arch_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)

        particles = []
        for i, bp in enumerate(self.basis_particles):
            R_major = rf if bp.sublattice_id == "flat" else ra
            geom = _build_ring_geometry(major_radius=R_major, num_segments=self.num_segments, geometry_type=f"ring_{bp.sublattice_id}")
            center = orig + bp.fractional_offset * a0

            R_mat = bp.orientation_fn(grid_index, center, ctx)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_mat
            T[:3, 3] = center

            p = InterlinkedParticle(
                particle_id=int(id_start + i),
                geometry=geom,
                transform=T,
                sublattice_id=bp.sublattice_id,
                cell_index=tuple(grid_index),
                metadata={"cell_name": self.name, "major_radius": R_major, "unit_cell_pitch": a0},
            )
            particles.append(p)
        return particles


# =============================================================================
# 5. NASA Space Fabric Cell: Hexagonal Plate + Spiral Hook Arms
# =============================================================================

@InterlinkedRegistry.register("nasa_hexagon")
@InterlinkedRegistry.register("nasa_space_fabric")
class NasaSpaceFabricCell:
    """
    NASA JPL 3D-Printed Space Fabric Unit Cell.
    Hexagonal lattice with 6-fold spiral interlocking hook legs.
    """
    name: str = "NASA Space Fabric"
    family: str = "space_fabric"
    parent_network: str = "hex_2d"
    coordination_number: int = 6

    # 6 hexagonal nearest neighbor directions:
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (1, -1, 0), (-1, 1, 0),
    ]

    def __init__(
        self,
        pitch: float = 12.75,
        wire_radius: float = 0.30,
        plate_radius: float = 7.0,
        plate_thickness: float = 0.45,
    ):
        self.pitch = float(pitch)
        self.wire_radius = float(wire_radius)
        self.plate_radius = float(plate_radius)
        self.plate_thickness = float(plate_thickness)

        # Approximate hexagonal tile proxy wireframe
        n_arm = 6
        angles = np.linspace(0.0, 2.0 * np.pi, n_arm, endpoint=False)
        outer_nodes = np.column_stack([
            self.plate_radius * np.cos(angles),
            self.plate_radius * np.sin(angles),
            np.zeros(n_arm, dtype=np.float64),
        ])
        center_node = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        nodes = np.vstack([center_node, outer_nodes])
        struts = np.array([[0, i] for i in range(1, n_arm + 1)], dtype=np.int64)

        proto = ParticleGeometry(
            nodes=nodes,
            struts=struts,
            bounding_radius=self.plate_radius,
            geometry_type="nasa_hexagon_proxy",
            metadata={"pitch": self.pitch, "wire_radius": self.wire_radius},
        )

        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.zeros(3, dtype=np.float64),
                sublattice_id="tile",
                orientation_fn=_default_identity_orientation,
            )
        ]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """Estimated radial clearance between adjacent hook arms."""
        p = float(unit_cell_pitch)
        d_wire = float(strut_diameter)
        arm_reach = 0.55 * p
        gap = (2.0 * arm_reach) - p
        return float(gap - d_wire)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d_wire = float(strut_diameter)
        # gap = 0.10 * p - d_wire
        return float((tc + d_wire) / 0.10)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        p = float(cell_pitch)
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)

        bp = self.basis_particles[0]
        R_mat = bp.orientation_fn(grid_index, orig, ctx)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = orig

        p_obj = InterlinkedParticle(
            particle_id=int(id_start),
            geometry=bp.geometry,
            transform=T,
            sublattice_id=bp.sublattice_id,
            cell_index=tuple(grid_index),
            metadata={"cell_name": self.name, "pitch": p},
        )
        return [p_obj]
