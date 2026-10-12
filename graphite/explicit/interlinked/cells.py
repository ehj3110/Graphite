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
    InterlinkedCellBase,
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
class D4TetCell(InterlinkedCellBase):
    """
    Diamond Tetrahedron (D-4-TET) PAM Unit Cell.
    Zhou et al., Science 2025. 4-fold corner catenation on Diamond (dia) network.
    Uses bipartite A/B basis with point-inversion crystallographic dual stagger.
    """
    name: str = "D-4-TET"
    family: str = "pam_polyhedra"
    parent_network: str = "dia"
    coordination_number: int = 4
    preferred_seeding_type: str = "diamond"  # auto-selected by generator when seeding_type not explicitly set

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

    SOLID_VOLUME_COEFFICIENT: float = 22.732564441375736

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def habit_plane_aperture(self, plane: str = "100", edge_length: float = 7.76, cell_pitch: float | None = None) -> float:
        """Internal clearance aperture diameter (mm) of boundary window."""
        a = float(cell_pitch) if cell_pitch is not None else float(edge_length) / self.edge_ratio
        if plane == "111":
            # On {111}, threefold axes expose wide face-on triangular windows and large open hexagonal voids
            # (Zhou et al., Science 2025: D_aperture ≈ 0.625 * a ≈ 7.5 mm at a = 12.0 mm)
            return float(0.625 * a)
        L = a * self.edge_ratio
        return float(L / np.sqrt(3.0))

    def habit_plane_profile(self, plane: str = "100", edge_length: float = 7.76, cell_pitch: float | None = None) -> float:
        """External cross-sectional profile diameter (mm) entering habit plane."""
        a = float(cell_pitch) if cell_pitch is not None else float(edge_length) / self.edge_ratio
        L = a * self.edge_ratio
        return float(L / np.sqrt(2.0))

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
class J4OctCell(InterlinkedCellBase):
    """
    Square-Planar Octahedron Cross (J-4-OCT) Unit Cell.
    Zhou et al., Science 2025. 4-fold tip catenation with 45-deg relative twist.
    """
    name: str = "J-4-OCT"
    family: str = "pam_polyhedra"
    parent_network: str = "square_2d"
    coordination_number: int = 4
    preferred_seeding_type: str = "cartesian"
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

    SOLID_VOLUME_COEFFICIENT: float = 9.996486610856323

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

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
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        bp = self.basis_particles[0]
        R_local = bp.orientation_fn(grid_index, orig, ctx)
        R_world = site_frame @ R_local
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_world
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
class European4in1Cell(InterlinkedCellBase):
    """
    Classic European 4-in-1 Chainmail Weave Unit Cell.
    Square translational grid with alternating row tilts (+/- theta) about the weave axis.
    Each ring loops through 4 coordinating neighbors.
    """
    name: str = "European 4-in-1"
    family: str = "chainmail_2d"
    parent_network: str = "square_2d"
    coordination_number: int = 4
    preferred_seeding_type: str = "cartesian"  # auto-selected by generator when seeding_type not explicitly set
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

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

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
class JapaneseKusariCell(InterlinkedCellBase):
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
    preferred_seeding_type: str = "cartesian"  # auto-selected by generator when seeding_type not explicitly set
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

    @property
    def CLEARANCE_KAPPA(self) -> float:
        return (self.flat_ratio + self.arch_ratio) - 0.5

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

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

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
class NasaSpaceFabricCell(InterlinkedCellBase):
    """
    NASA JPL 3D-Printed Space Fabric Unit Cell.
    Hexagonal lattice with 6-fold spiral interlocking hook legs.
    """
    name: str = "NASA Space Fabric"
    family: str = "space_fabric"
    parent_network: str = "hex_2d"
    coordination_number: int = 6
    preferred_seeding_type: str = "hexagonal"  # auto-selected by generator when seeding_type not explicitly set

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

    CLEARANCE_KAPPA: float = 0.10

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d_wire = float(strut_diameter)
        return float((tc + d_wire) / 0.10)

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

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


# =============================================================================
# 6. J-4-RING: Diamond Octahedral Six-Ring Cluster PAM Cell
# =============================================================================

@InterlinkedRegistry.register("j4ring")
@InterlinkedRegistry.register("j-4-ring")
@InterlinkedRegistry.register("j4")
class J4RingCell(InterlinkedCellBase):
    """
    Diamond Octahedral Six-Ring Cluster (J-4-RING) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    4-coordinated polycatenation on the Diamond (dia) network.
    Each lattice site features an octahedral 6-ring cluster (Oh symmetry) with
    rings radially offset along the 6 Cartesian semi-axes:
        c_±x = (±d_off, 0, 0), c_±y = (0, ±d_off, 0), c_±z = (0, 0, ±d_off)
        n_±x = (±1, 0, 0), n_±y = (0, ±1, 0), n_±z = (0, 0, ±1)

    All clusters share the same global orientation (no bipartite twist needed
    due to inversion centrosymmetry i in Oh). Along each <111> bond, the 3 rings
    pointing into that octant interlink with the 3 opposing rings of the neighbor
    cluster, producing 6 catenated ring-ring pairs (|Lk| = 1).
    """
    name: str = "J-4-RING"
    family: str = "pam_rings"
    parent_network: str = "dia"
    coordination_number: int = 4
    preferred_seeding_type: str = "diamond"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]

    DEFAULT_RADIUS_RATIO: float = 0.18000  # R / a (e.g. 2.52 / 14.0)
    DEFAULT_OFFSET_RATIO: float = 0.24000  # d_off / a (e.g. 3.36 / 14.0)
    CLEARANCE_KAPPA: float = 0.08485       # Delta = kappa * a - D_wire

    def __init__(
        self,
        radius_ratio: float = 0.18000,
        offset_ratio: float = 0.24000,
        num_segments: int = 32,
    ):
        self.radius_ratio = float(radius_ratio)
        self.offset_ratio = float(offset_ratio)
        self.num_segments = int(num_segments)

        proto = _build_ring_geometry(major_radius=1.0, num_segments=self.num_segments, geometry_type="j4_ring")

        axes = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        ]
        sublattices = ["pos_x", "neg_x", "pos_y", "neg_y", "pos_z", "neg_z"]

        self.basis_particles: list[BasisParticle] = []
        for ax, sub in zip(axes, sublattices):
            R_rot = self._get_ring_rotation(ax)
            self.basis_particles.append(
                BasisParticle(
                    geometry=proto,
                    fractional_offset=self.offset_ratio * ax,
                    sublattice_id=sub,
                    orientation_fn=lambda idx, c, ctx, R_m=R_rot: R_m,
                    metadata={"axis": sub, "normal": ax.tolist()},
                )
            )

    @staticmethod
    def _get_ring_rotation(normal: np.ndarray) -> np.ndarray:
        n = np.asarray(normal, dtype=np.float64)
        n = n / float(np.linalg.norm(n))
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.allclose(n, z):
            return np.eye(3, dtype=np.float64)
        if np.allclose(n, -z):
            return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
        axis = np.cross(z, n)
        axis = axis / float(np.linalg.norm(axis))
        angle = np.arccos(np.clip(float(np.dot(z, n)), -1.0, 1.0))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a = float(unit_cell_pitch)
        d_wire = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d_wire)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d_wire = float(strut_diameter)
        return float((tc + d_wire) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 42.63669101270603

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        R_major = a * self.radius_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        particles = []
        for i, bp in enumerate(self.basis_particles):
            geom = _build_ring_geometry(major_radius=R_major, num_segments=self.num_segments, geometry_type="j4_ring")
            d_vec = bp.fractional_offset * a
            center = orig + site_frame @ d_vec

            R_local = bp.orientation_fn(grid_index, center, ctx)
            R_world = site_frame @ R_local

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = center

            p = InterlinkedParticle(
                particle_id=int(id_start + i),
                geometry=geom,
                transform=T,
                sublattice_id=bp.sublattice_id,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "major_radius": R_major,
                    "unit_cell_pitch": a,
                    "cluster_site": orig.tolist(),
                    "cluster_id": id_start // 6,
                    **bp.metadata,
                },
            )
            particles.append(p)
        return particles


# =============================================================================
# 7. D-4-CUBE: Diamond Cube Wireframe Cell
# =============================================================================

def _build_cube_geometry(half_side: float) -> ParticleGeometry:
    """Generate 8-vertex / 12-strut regular cube wireframe centered at origin."""
    s = float(half_side)
    nodes = np.array([
        [-s, -s, -s],
        [-s, -s,  s],
        [-s,  s, -s],
        [-s,  s,  s],
        [ s, -s, -s],
        [ s, -s,  s],
        [ s,  s, -s],
        [ s,  s,  s],
    ], dtype=np.float64)
    struts = []
    for i in range(8):
        for j in range(i + 1, 8):
            if np.isclose(np.linalg.norm(nodes[i] - nodes[j]), 2.0 * s, atol=1e-5):
                struts.append([i, j])
    strut_arr = np.array(struts, dtype=np.int64)
    return ParticleGeometry(
        nodes=nodes,
        struts=strut_arr,
        bounding_radius=float(np.sqrt(3.0) * s),
        geometry_type="CUBE",
        metadata={"half_side": s, "edge_length": 2.0 * s},
    )


@InterlinkedRegistry.register("d4cube")
@InterlinkedRegistry.register("d-4-cube")
class D4CubeCell(InterlinkedCellBase):
    """
    Diamond Cube Wireframe (D-4-CUBE) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    4-fold corner catenation on the Diamond (dia) network.
    8 vertices (+-s, +-s, +-s), 12 struts connecting edges of length 2s.
    Centrosymmetric (Oh); body diagonals align with <111> diamond bonds.
    """
    name: str = "D-4-CUBE"
    family: str = "pam_polyhedra"
    parent_network: str = "dia"
    coordination_number: int = 4
    preferred_seeding_type: str = "diamond"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]

    DEFAULT_SIZE_RATIO: float = 0.1700
    CLEARANCE_KAPPA: float = 0.0900

    def __init__(self, size_ratio: float = 0.1700):
        self.size_ratio = float(size_ratio)
        proto = _build_cube_geometry(half_side=1.0)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "A"},
            ),
            BasisParticle(
                geometry=proto,
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
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 25.635396053292713

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def habit_plane_aperture(self, plane: str = "100", cell_pitch: float | None = None) -> float:
        """Internal clearance aperture diameter (mm) of square window."""
        a = float(cell_pitch) if cell_pitch is not None else 12.0
        s = a * self.size_ratio
        return float(2.0 * s)

    def habit_plane_profile(self, plane: str = "100", cell_pitch: float | None = None) -> float:
        """External cross-sectional profile diameter (mm) entering habit plane."""
        a = float(cell_pitch) if cell_pitch is not None else 12.0
        s = a * self.size_ratio
        return float(2.0 * np.sqrt(2.0) * s)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a_conv = float(cell_pitch)
        s = a_conv * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        proto = _build_cube_geometry(half_side=s)
        sub_req = ctx.get("sublattice", None)

        particles = []
        if sub_req in ("A", "B"):
            bp = self.basis_particles[0] if sub_req == "A" else self.basis_particles[1]
            R_local = bp.orientation_fn(grid_index, orig, ctx)
            R_world = site_frame @ R_local
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = orig
            p = InterlinkedParticle(
                particle_id=id_start,
                geometry=proto,
                transform=T,
                sublattice_id=sub_req,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "half_side": s,
                    "unit_cell_pitch": a_conv,
                    "sublattice": sub_req,
                },
            )
            particles.append(p)
        else:
            for i, bp in enumerate(self.basis_particles):
                proto = _build_cube_geometry(half_side=s)
                shift = site_frame @ (bp.fractional_offset * a_conv)
                pos = orig + shift
                R_local = bp.orientation_fn(grid_index, pos, ctx)
                R_world = site_frame @ R_local
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_world
                T[:3, 3] = pos
                p = InterlinkedParticle(
                    particle_id=id_start + i,
                    geometry=proto,
                    transform=T,
                    sublattice_id=bp.sublattice_id,
                    cell_index=tuple(grid_index),
                    metadata={
                        "cell_name": self.name,
                        "half_side": s,
                        "unit_cell_pitch": a_conv,
                        "sublattice": bp.sublattice_id,
                    },
                )
                particles.append(p)
        return particles


# =============================================================================
# 8. D-4-CT: Diamond Triakis / Catalan Tetrahedron Cell
# =============================================================================

def _build_ct_geometry(size_out: float, dual: bool = False) -> ParticleGeometry:
    """
    Generate 8-vertex / 18-strut Triakis / Catalan Tetrahedron wireframe.
    4 outer apexes + 4 inner base vertices (s_in = 0.6 * s_out).
    """
    s_out = float(size_out)
    s_in = 0.6 * s_out
    outer_v = s_out * np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ], dtype=np.float64)
    inner_v = s_in * np.array([
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
    ], dtype=np.float64)
    nodes = np.vstack([outer_v, inner_v])
    if dual:
        nodes = -nodes
    struts = np.array([
        # 6 base struts connecting inner vertices
        [4, 5], [4, 6], [4, 7],
        [5, 6], [5, 7], [6, 7],
        # 12 pyramid struts connecting apexes to base faces
        [0, 5], [0, 6], [0, 7],
        [1, 4], [1, 6], [1, 7],
        [2, 4], [2, 5], [2, 7],
        [3, 4], [3, 5], [3, 6],
    ], dtype=np.int64)
    return ParticleGeometry(
        nodes=nodes,
        struts=struts,
        bounding_radius=float(np.sqrt(3.0) * s_out),
        geometry_type="CT",
        metadata={"size_out": s_out, "size_in": s_in, "dual": dual},
    )


@InterlinkedRegistry.register("d4ct")
@InterlinkedRegistry.register("d-4-ct")
class D4CTCell(InterlinkedCellBase):
    """
    Diamond Triakis / Catalan Tetrahedron (D-4-CT) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    8 vertices, 18 struts. Non-centrosymmetric (Td symmetry).
    Sublattice A: R_A = I; Sublattice B: R_B = -I (point inversion).
    """
    name: str = "D-4-CT"
    family: str = "pam_polyhedra"
    parent_network: str = "dia"
    coordination_number: int = 4
    preferred_seeding_type: str = "diamond"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]

    DEFAULT_SIZE_RATIO: float = 0.1800
    CLEARANCE_KAPPA: float = 0.0635

    def __init__(self, size_ratio: float = 0.1800):
        self.size_ratio = float(size_ratio)
        proto_a = _build_ct_geometry(size_out=1.0, dual=False)
        proto_b = _build_ct_geometry(size_out=1.0, dual=True)
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
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 34.54785772711946

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a_conv = float(cell_pitch)
        s_out = a_conv * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        sub_req = ctx.get("sublattice", None)
        particles = []

        if sub_req in ("A", "B"):
            dual = (sub_req == "B")
            proto = _build_ct_geometry(size_out=s_out, dual=dual)
            bp = self.basis_particles[0] if sub_req == "A" else self.basis_particles[1]
            R_local = bp.orientation_fn(grid_index, orig, ctx)
            R_world = site_frame @ R_local
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = orig
            p = InterlinkedParticle(
                particle_id=id_start,
                geometry=proto,
                transform=T,
                sublattice_id=sub_req,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "size_out": s_out,
                    "unit_cell_pitch": a_conv,
                    "sublattice": sub_req,
                },
            )
            particles.append(p)
        else:
            for i, bp in enumerate(self.basis_particles):
                dual = (bp.sublattice_id == "B")
                proto = _build_ct_geometry(size_out=s_out, dual=dual)
                shift = site_frame @ (bp.fractional_offset * a_conv)
                pos = orig + shift
                R_local = bp.orientation_fn(grid_index, pos, ctx)
                R_world = site_frame @ R_local
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_world
                T[:3, 3] = pos
                p = InterlinkedParticle(
                    particle_id=id_start + i,
                    geometry=proto,
                    transform=T,
                    sublattice_id=bp.sublattice_id,
                    cell_index=tuple(grid_index),
                    metadata={
                        "cell_name": self.name,
                        "size_out": s_out,
                        "unit_cell_pitch": a_conv,
                        "sublattice": bp.sublattice_id,
                    },
                )
                particles.append(p)
        return particles


# =============================================================================
# 9. D-4-ICO: Diamond Regular Icosahedron Cell
# =============================================================================

def _build_ico_geometry(scale: float) -> ParticleGeometry:
    """Generate 12-vertex / 30-strut regular icosahedron wireframe centered at origin."""
    s = float(scale)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    raw_verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            raw_verts.append([0.0, s1, s2 * phi])
            raw_verts.append([s1, s2 * phi, 0.0])
            raw_verts.append([s2 * phi, 0.0, s1])
    nodes = np.unique(np.round(np.array(raw_verts, dtype=np.float64) * s, 8), axis=0)
    edge_len = 2.0 * s
    struts = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(np.linalg.norm(nodes[i] - nodes[j]), edge_len, atol=1e-4 * s):
                struts.append([i, j])
    strut_arr = np.array(struts, dtype=np.int64)
    return ParticleGeometry(
        nodes=nodes,
        struts=strut_arr,
        bounding_radius=float(np.sqrt(1.0 + phi ** 2) * s),
        geometry_type="ICO",
        metadata={"scale": s, "edge_length": edge_len},
    )


@InterlinkedRegistry.register("d4ico")
@InterlinkedRegistry.register("d-4-ico")
class D4IcoCell(InterlinkedCellBase):
    """
    Diamond Regular Icosahedron (D-4-ICO) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    12 vertices from golden ratio permutations s * (0, +-1, +-phi), 30 struts.
    Triangular faces along <111> bonds catenate face-to-face. R_A = I, R_B = I.
    """
    name: str = "D-4-ICO"
    family: str = "pam_polyhedra"
    parent_network: str = "dia"
    coordination_number: int = 4
    preferred_seeding_type: str = "diamond"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]

    DEFAULT_SIZE_RATIO: float = 0.1650
    CLEARANCE_KAPPA: float = 0.0634

    def __init__(self, size_ratio: float = 0.1650):
        self.size_ratio = float(size_ratio)
        proto = _build_ico_geometry(scale=1.0)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "A"},
            ),
            BasisParticle(
                geometry=proto,
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
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 62.20353485403631

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a_conv = float(cell_pitch)
        s = a_conv * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        proto = _build_ico_geometry(scale=s)
        sub_req = ctx.get("sublattice", None)
        particles = []

        if sub_req in ("A", "B"):
            bp = self.basis_particles[0] if sub_req == "A" else self.basis_particles[1]
            R_local = bp.orientation_fn(grid_index, orig, ctx)
            R_world = site_frame @ R_local
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = orig
            p = InterlinkedParticle(
                particle_id=id_start,
                geometry=proto,
                transform=T,
                sublattice_id=sub_req,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "scale": s,
                    "unit_cell_pitch": a_conv,
                    "sublattice": sub_req,
                },
            )
            particles.append(p)
        else:
            for i, bp in enumerate(self.basis_particles):
                proto = _build_ico_geometry(scale=s)
                shift = site_frame @ (bp.fractional_offset * a_conv)
                pos = orig + shift
                R_local = bp.orientation_fn(grid_index, pos, ctx)
                R_world = site_frame @ R_local
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_world
                T[:3, 3] = pos
                p = InterlinkedParticle(
                    particle_id=id_start + i,
                    geometry=proto,
                    transform=T,
                    sublattice_id=bp.sublattice_id,
                    cell_index=tuple(grid_index),
                    metadata={
                        "cell_name": self.name,
                        "scale": s,
                        "unit_cell_pitch": a_conv,
                        "sublattice": bp.sublattice_id,
                    },
                )
                particles.append(p)
        return particles


# =============================================================================
# 10. T-6-RING & T-6-HEX: Tetrahedral 4-Loop Clusters
# =============================================================================

def _build_hex_loop_geometry(radius: float) -> ParticleGeometry:
    """Generate 6-vertex / 6-strut planar regular hexagon in XY plane centered at origin."""
    R = float(radius)
    theta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    nodes = np.column_stack([
        R * np.cos(theta),
        R * np.sin(theta),
        np.zeros(6, dtype=np.float64),
    ])
    struts = np.column_stack([
        np.arange(6, dtype=np.int64),
        np.roll(np.arange(6, dtype=np.int64), -1),
    ])
    return ParticleGeometry(
        nodes=nodes,
        struts=struts,
        bounding_radius=R,
        geometry_type="HEX_LOOP",
        metadata={"radius": R},
    )


_TET_NORMALS = [
    np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([1.0, -1.0, -1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([-1.0, 1.0, -1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([-1.0, -1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
]


@InterlinkedRegistry.register("t6ring")
@InterlinkedRegistry.register("t-6-ring")
class T6RingCell(InterlinkedCellBase):
    """
    Tetrahedral 4-Ring Cluster (T-6-RING) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    6-coordinated network catenating through the 6 tetrahedral edges.
    Each cluster features 4 circular tori centered at c_k = d_off * n_k
    with plane normals n_k along regular tetrahedral directions.
    """
    name: str = "T-6-RING"
    family: str = "pam_rings"
    parent_network: str = "pcu"
    coordination_number: int = 6
    preferred_seeding_type: str = "cartesian"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    DEFAULT_RADIUS_RATIO: float = 0.4500
    DEFAULT_OFFSET_RATIO: float = 0.4500
    CLEARANCE_KAPPA: float = 0.1855

    def __init__(
        self,
        radius_ratio: float = 0.4500,
        offset_ratio: float = 0.4500,
        num_segments: int = 32,
    ):
        self.radius_ratio = float(radius_ratio)
        self.offset_ratio = float(offset_ratio)
        self.num_segments = int(num_segments)

        proto = _build_ring_geometry(major_radius=1.0, num_segments=self.num_segments, geometry_type="t6_ring")
        self.basis_particles: list[BasisParticle] = []
        for k, n in enumerate(_TET_NORMALS):
            R_rot = J4RingCell._get_ring_rotation(n)
            self.basis_particles.append(
                BasisParticle(
                    geometry=proto,
                    fractional_offset=self.offset_ratio * n,
                    sublattice_id=f"loop_{k}",
                    orientation_fn=lambda idx, c, ctx, R_m=R_rot: R_m,
                    metadata={"loop_index": k, "normal": n.tolist()},
                )
            )

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 8.882643960980422

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        R_major = a * self.radius_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        particles = []
        for i, bp in enumerate(self.basis_particles):
            geom = _build_ring_geometry(major_radius=R_major, num_segments=self.num_segments, geometry_type="t6_ring")
            d_vec = bp.fractional_offset * a
            center = orig + site_frame @ d_vec

            R_local = bp.orientation_fn(grid_index, center, ctx)
            R_world = site_frame @ R_local

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = center

            p = InterlinkedParticle(
                particle_id=int(id_start + i),
                geometry=geom,
                transform=T,
                sublattice_id=bp.sublattice_id,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "major_radius": R_major,
                    "unit_cell_pitch": a,
                    "cluster_site": orig.tolist(),
                    "cluster_id": id_start // 4,
                    **bp.metadata,
                },
            )
            particles.append(p)
        return particles


@InterlinkedRegistry.register("t6hex")
@InterlinkedRegistry.register("t-6-hex")
class T6HexCell(InterlinkedCellBase):
    """
    Tetrahedral 4-Hexagon Cluster (T-6-HEX) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    6-coordinated network catenating through the 6 tetrahedral edges.
    Each cluster features 4 regular planar hexagonal wireframes centered at
    c_k = d_off * n_k with plane normals n_k along regular tetrahedral directions.
    """
    name: str = "T-6-HEX"
    family: str = "pam_polygons"
    parent_network: str = "pcu"
    coordination_number: int = 6
    preferred_seeding_type: str = "cartesian"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    DEFAULT_RADIUS_RATIO: float = 0.4500
    DEFAULT_OFFSET_RATIO: float = 0.4500
    CLEARANCE_KAPPA: float = 0.1168

    def __init__(
        self,
        radius_ratio: float = 0.4500,
        offset_ratio: float = 0.4500,
    ):
        self.radius_ratio = float(radius_ratio)
        self.offset_ratio = float(offset_ratio)

        proto = _build_hex_loop_geometry(radius=1.0)
        self.basis_particles: list[BasisParticle] = []
        for k, n in enumerate(_TET_NORMALS):
            R_rot = J4RingCell._get_ring_rotation(n)
            self.basis_particles.append(
                BasisParticle(
                    geometry=proto,
                    fractional_offset=self.offset_ratio * n,
                    sublattice_id=f"hex_{k}",
                    orientation_fn=lambda idx, c, ctx, R_m=R_rot: R_m,
                    metadata={"loop_index": k, "normal": n.tolist()},
                )
            )

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 8.482300164692441

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        R_major = a * self.radius_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        particles = []
        for i, bp in enumerate(self.basis_particles):
            geom = _build_hex_loop_geometry(radius=R_major)
            d_vec = bp.fractional_offset * a
            center = orig + site_frame @ d_vec

            R_local = bp.orientation_fn(grid_index, center, ctx)
            R_world = site_frame @ R_local

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = center

            p = InterlinkedParticle(
                particle_id=int(id_start + i),
                geometry=geom,
                transform=T,
                sublattice_id=bp.sublattice_id,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "radius": R_major,
                    "unit_cell_pitch": a,
                    "cluster_site": orig.tolist(),
                    "cluster_id": id_start // 4,
                    **bp.metadata,
                },
            )
            particles.append(p)
        return particles


# =============================================================================
# 11. C-6-CO & B-8-CO: Cuboctahedral Families
# =============================================================================

def _build_cuboctahedral_geometry(size: float) -> ParticleGeometry:
    """Generate 12-vertex / 24-strut cuboctahedron wireframe centered at origin.

    The 12 cuboctahedron vertices are all permutations of (±s, ±s, 0), giving:
      - Edge length:      s * sqrt(2)
      - Bounding radius:  s * sqrt(2)

    Note: do NOT apply a 1/sqrt(2) pre-factor here. The earlier bug that did so
    shrunk vertex positions to (±s/√2, ±s/√2, 0), making the bounding radius
    equal to just s, which caused zero contact with pcu / bcc neighbors.
    """
    s = float(size)
    raw = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            raw.append([x * s, y * s, 0.0])
            raw.append([x * s, 0.0, y * s])
            raw.append([0.0, x * s, y * s])
    nodes = np.asarray(raw, dtype=np.float64)
    _, idx = np.unique(np.round(nodes, decimals=8), axis=0, return_index=True)
    nodes = nodes[np.sort(idx)]
    edge_len = s * np.sqrt(2.0)
    struts = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(np.linalg.norm(nodes[i] - nodes[j]), edge_len, atol=1e-4 * s):
                struts.append([i, j])
    strut_arr = np.array(struts, dtype=np.int64)
    bounding_radius = s * np.sqrt(2.0)
    return ParticleGeometry(
        nodes=nodes,
        struts=strut_arr,
        bounding_radius=bounding_radius,
        geometry_type="CO",
        metadata={"size": s, "edge_length": edge_len, "bounding_radius": bounding_radius},
    )


def _get_bcc_twist_rotation() -> np.ndarray:
    """60° relative twist about [1, 1, 1] for B-8-CO / C-6-CO alternating sublattices."""
    u = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    theta = np.pi / 3.0  # 60 degrees
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@InterlinkedRegistry.register("c6co")
@InterlinkedRegistry.register("c-6-co")
class C6COCell(InterlinkedCellBase):
    """
    Simple Cubic Cuboctahedron (C-6-CO) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    ** EXPERIMENTAL — GEOMETRICALLY FRUSTRATED **

    The cuboctahedron has square faces with inscribed circle radius = s/√2 and square
    cross-section (profile) along ⟨100⟩ bond axes of side s√2. Both are exactly equal,
    leaving zero margin for threading. Furthermore, the 60° [1,1,1] twist intended to
    avoid coplanar strut collisions cannot simultaneously satisfy all six ⟨100⟩ bond axes
    because 3D rotations do not commute — this is the same rotational frustration
    documented for C-6-TT in docs/POLYCATED_ARCHITECTED_MATERIALS.md §3.1.

    As a result, C-6-CO does NOT produce reliable topological catenation on the pcu
    parent network. It is retained in the registry for research purposes and for
    the Zhou et al. Science 2025 reference, but should NOT be used for production
    lattices without a corrected threading geometry.

    For 6-connected Simple Cubic catenation, use C-6-TT (Truncated Tetrahedron)
    which threads through hexagonal rather than square faces and avoids this frustration.

    Parameters
    ----------
    size_ratio : float
        Ratio of cuboctahedron size (arm length s) to unit cell pitch a.
        With the corrected geometry (bounding_radius = s*sqrt(2)), reasonable
        packing requires size_ratio ≤ 1/(2*sqrt(2)) ≈ 0.354 to avoid self-overlap.
    """
    name: str = "C-6-CO"
    family: str = "pam_polyhedra"
    parent_network: str = "pcu"
    coordination_number: int = 6
    preferred_seeding_type: str = "cartesian"  # auto-selected by generator when seeding_type not explicitly set
    is_experimental: bool = True  # Geometrically frustrated on pcu — see docstring

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    DEFAULT_SIZE_RATIO: float = 0.3400  # Corrected for fixed geometry (bounding_radius = s*sqrt(2))
    CLEARANCE_KAPPA: float = 0.2217    # Empirical neighbor-proximity kappa at sr=0.34

    def __init__(self, size_ratio: float = 0.3400):
        self.size_ratio = float(size_ratio)
        proto_a = _build_cuboctahedral_geometry(size=1.0)
        R_twist = _get_bcc_twist_rotation()
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto_a,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "A"},
            ),
            BasisParticle(
                geometry=proto_a,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="B",
                orientation_fn=lambda idx, c, ctx, R_m=R_twist: R_m,
                metadata={"sublattice": "B"},
            ),
        ]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 9.047786842338603

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        s = a * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        parity = (grid_index[0] + grid_index[1] + grid_index[2]) % 2
        sub = ctx.get("sublattice", "A" if parity == 0 else "B")

        proto = _build_cuboctahedral_geometry(size=s)
        bp = self.basis_particles[0] if sub == "A" else self.basis_particles[1]
        R_local = bp.orientation_fn(grid_index, orig, ctx)
        R_world = site_frame @ R_local

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_world
        T[:3, 3] = orig

        p = InterlinkedParticle(
            particle_id=id_start,
            geometry=proto,
            transform=T,
            sublattice_id=sub,
            cell_index=tuple(grid_index),
            metadata={
                "cell_name": self.name,
                "size": s,
                "unit_cell_pitch": a,
                "sublattice": sub,
                "parity": parity,
            },
        )
        return [p]


@InterlinkedRegistry.register("b8co")
@InterlinkedRegistry.register("b-8-co")
class B8COCell(InterlinkedCellBase):
    """
    Body-Centered Cubic Cuboctahedron (B-8-CO) PAM Unit Cell.
    Zhou et al., Science 387, 269–277 (2025).

    8-coordinated catenation on BCC network along body diagonals <111> through
    triangular faces. Corner site A (R=I) and body-center site B ((1/2,1/2,1/2) a)
    with 60° relative twist about [1, 1, 1].
    """
    name: str = "B-8-CO"
    family: str = "pam_polyhedra"
    parent_network: str = "bcc"
    coordination_number: int = 8
    preferred_seeding_type: str = "bcc"  # auto-selected by generator when seeding_type not explicitly set

    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]

    DEFAULT_SIZE_RATIO: float = 0.4000  # size / pitch; calibrated for BCC body-diagonal catenation
    CLEARANCE_KAPPA: float = 0.0577

    def __init__(self, size_ratio: float = 0.4000):
        self.size_ratio = float(size_ratio)
        proto = _build_cuboctahedral_geometry(size=1.0)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"sublattice": "A"},
            ),
            BasisParticle(
                geometry=proto,
                fractional_offset=np.array([0.5, 0.5, 0.5], dtype=np.float64),
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
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    SOLID_VOLUME_COEFFICIENT: float = 15.833626974092555

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """Determine (pitch a, strut_diameter d) from target clearance and solid fraction."""
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        s = a * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        sub_req = ctx.get("sublattice", None)
        proto = _build_cuboctahedral_geometry(size=s)
        particles = []

        if sub_req in ("A", "B"):
            bp = self.basis_particles[0] if sub_req == "A" else self.basis_particles[1]
            R_local = bp.orientation_fn(grid_index, orig, ctx)
            R_world = site_frame @ R_local
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_world
            T[:3, 3] = orig
            p = InterlinkedParticle(
                particle_id=id_start,
                geometry=proto,
                transform=T,
                sublattice_id=sub_req,
                cell_index=tuple(grid_index),
                metadata={
                    "cell_name": self.name,
                    "size": s,
                    "unit_cell_pitch": a,
                    "sublattice": sub_req,
                },
            )
            particles.append(p)
        else:
            for i, bp in enumerate(self.basis_particles):
                proto = _build_cuboctahedral_geometry(size=s)
                shift = site_frame @ (bp.fractional_offset * a)
                pos = orig + shift
                R_local = bp.orientation_fn(grid_index, pos, ctx)
                R_world = site_frame @ R_local
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R_world
                T[:3, 3] = pos
                p = InterlinkedParticle(
                    particle_id=id_start + i,
                    geometry=proto,
                    transform=T,
                    sublattice_id=bp.sublattice_id,
                    cell_index=tuple(grid_index),
                    metadata={
                        "cell_name": self.name,
                        "size": s,
                        "unit_cell_pitch": a,
                        "sublattice": bp.sublattice_id,
                    },
                )
                particles.append(p)
        return particles


# =============================================================================
# 14. Custom Wireframe Cell (Arbitrary & Ad-Hoc Wireframes)
# =============================================================================

@InterlinkedRegistry.register("custom")
@InterlinkedRegistry.register("custom-wireframe")
@InterlinkedRegistry.register("custom_wireframe")
class CustomWireframeCell(InterlinkedCellBase):
    """
    Arbitrary user-defined or ad-hoc wireframe interlinked unit cell.
    Zhou et al. / modular PAM protocol.

    Accepts:
      - nodes: (N, 3) float array of vertex coordinates, or Sequence of (x, y, z)
      - struts: (M, 2) int array of strut edge indices, or Sequence of (u, v)
      - OR recipe: dictionary or JSON/YAML structure containing:
            'nodes': [[x, y, z], ...],
            'struts': [[u, v], ...],
            optional 'name', 'parent_network', 'coordination_number', 'size_ratio',
            'preferred_seeding_type', 'clearance_kappa', 'face_cycles', etc.
    """
    name: str = "Custom-Wireframe"
    family: str = "custom"
    parent_network: str = "pcu"
    coordination_number: int = 6
    preferred_seeding_type: str = "cartesian"
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    CLEARANCE_KAPPA: float = 0.25
    SOLID_VOLUME_COEFFICIENT: float = 8.0

    def __init__(
        self,
        nodes: np.ndarray | Sequence[Sequence[float]] | None = None,
        struts: np.ndarray | Sequence[Sequence[int]] | None = None,
        recipe: dict[str, Any] | None = None,
        name: str | None = None,
        parent_network: str | None = None,
        coordination_number: int | None = None,
        preferred_seeding_type: str | None = None,
        size_ratio: float = 1.0,
        clearance_kappa: float | None = None,
        solid_volume_coefficient: float | None = None,
        neighbor_offsets: list[tuple[int, int, int]] | None = None,
        face_cycles: list[list[int]] | None = None,
    ):
        if isinstance(nodes, (tuple, list)) and len(nodes) == 2 and struts is None:
            nodes, struts = nodes[0], nodes[1]
        elif isinstance(nodes, dict) and recipe is None:
            recipe = nodes
            nodes = None

        if recipe is not None:
            raw_nodes = recipe.get("nodes", nodes)
            raw_struts = recipe.get("struts", struts)
            name = recipe.get("name", name)
            parent_network = recipe.get("parent_network", parent_network)
            coordination_number = recipe.get("coordination_number", coordination_number)
            preferred_seeding_type = recipe.get("preferred_seeding_type", preferred_seeding_type)
            size_ratio = recipe.get("size_ratio", size_ratio)
            clearance_kappa = recipe.get("clearance_kappa", clearance_kappa)
            solid_volume_coefficient = recipe.get("solid_volume_coefficient", solid_volume_coefficient)
            neighbor_offsets = recipe.get("neighbor_offsets", neighbor_offsets)
            face_cycles = recipe.get("face_cycles", face_cycles)
        else:
            raw_nodes = nodes
            raw_struts = struts

        if raw_nodes is None:
            # Default fallback: regular octahedron (6 nodes, 12 struts)
            raw_nodes = np.array([
                [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            ], dtype=np.float64)
            raw_struts = np.array([
                [0, 2], [0, 3], [0, 4], [0, 5],
                [1, 2], [1, 3], [1, 4], [1, 5],
                [2, 4], [2, 5], [3, 4], [3, 5],
            ], dtype=np.int64)

        if name:
            self.name = str(name)
        if parent_network:
            self.parent_network = str(parent_network).lower().strip()
            if self.parent_network in ("dia", "diamond"):
                self.preferred_seeding_type = "diamond"
            elif self.parent_network in ("bcc", "body_centered"):
                self.preferred_seeding_type = "bcc"
            else:
                self.preferred_seeding_type = "cartesian"
        if preferred_seeding_type:
            self.preferred_seeding_type = str(preferred_seeding_type)
        if coordination_number is not None:
            self.coordination_number = int(coordination_number)
        if clearance_kappa is not None:
            self.CLEARANCE_KAPPA = float(clearance_kappa)
        if solid_volume_coefficient is not None:
            self.SOLID_VOLUME_COEFFICIENT = float(solid_volume_coefficient)
        if neighbor_offsets is not None:
            self.neighbor_catenation_offsets = list(neighbor_offsets)
        self.size_ratio = float(size_ratio)
        self._explicit_face_cycles = list(face_cycles) if face_cycles is not None else None

        # Canonicalize nodes
        arr_nodes = np.asarray(raw_nodes, dtype=np.float64).copy()
        if len(arr_nodes) > 0:
            arr_nodes = arr_nodes - arr_nodes.mean(axis=0)
            r_max = float(np.max(np.linalg.norm(arr_nodes, axis=1)))
            if r_max > 1e-8:
                arr_nodes = arr_nodes / r_max
        self._canonical_nodes = arr_nodes
        self._canonical_struts = np.asarray(raw_struts, dtype=np.int64).copy()

        proto = ParticleGeometry(
            nodes=self._canonical_nodes,
            struts=self._canonical_struts,
            bounding_radius=1.0,
            geometry_type=self.name,
            metadata={"cell_name": self.name, "canonical": True},
        )
        self.basis_particles = [
            BasisParticle(
                geometry=proto,
                fractional_offset=np.zeros(3, dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"role": "custom_link"},
            )
        ]

    def get_face_cycles(self) -> list[list[int]]:
        """Return fundamental face cycles of the unit cell particle prototype."""
        if self._explicit_face_cycles is not None:
            return self._explicit_face_cycles
        return super().get_face_cycles()

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        a = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        tc = float(target_clearance)
        d = float(strut_diameter)
        return float((tc + d) / self.CLEARANCE_KAPPA)

    def resolve_dimensions(
        self,
        target_clearance: float,
        target_solid_fraction: float,
        **kwargs: Any,
    ) -> tuple[float, float]:
        from .dimensioning import PAMDimensioner
        res = PAMDimensioner.solve_from_clearance_and_vf(self, target_clearance, target_solid_fraction, **kwargs)
        return (res.pitch, res.wire_diameter)

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        ctx = context or {}
        a = float(cell_pitch)
        s = a * self.size_ratio
        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        site_frame = ctx.get("site_frame", np.eye(3, dtype=np.float64))

        proto_nodes = self._canonical_nodes * s
        proto = ParticleGeometry(
            nodes=proto_nodes,
            struts=self._canonical_struts,
            bounding_radius=s,
            geometry_type=self.name,
            metadata={"cell_name": self.name, "size": s, "unit_cell_pitch": a},
        )
        sub_req = ctx.get("sublattice", "A")
        bp = self.basis_particles[0]
        R_local = bp.orientation_fn(grid_index, orig, ctx)
        R_world = site_frame @ R_local
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_world
        T[:3, 3] = orig
        p = InterlinkedParticle(
            particle_id=id_start,
            geometry=proto,
            transform=T,
            sublattice_id=sub_req,
            cell_index=tuple(grid_index),
            metadata={
                "cell_name": self.name,
                "size": s,
                "unit_cell_pitch": a,
                "sublattice": sub_req,
            },
        )
        return [p]


