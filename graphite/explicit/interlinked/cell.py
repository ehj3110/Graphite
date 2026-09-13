"""
Graphite Explicit Interlinked — Declarative Cell Protocol & Registry

Defines:
- BasisParticle: Specification of a constituent particle within a unit cell repeat basis.
- InterlinkedCell: Universal protocol for interlinked, chainmail, and polycatenated unit cells.
- InterlinkedRegistry: Central class decorator and catalog for modular interlinked cell registration.
- C6TTCell: Pilot reference implementation of C-6-TT (Truncated Tetrahedron on Simple Cubic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable
import numpy as np

from .particle import ParticleGeometry, InterlinkedParticle


# Type signature for local orientation evaluation:
# (grid_index: (i,j,k), site_center: np.ndarray, context: dict) -> SO(3) matrix (3,3)
OrientationFn = Callable[[tuple[int, int, int], np.ndarray, dict[str, Any]], np.ndarray]


def _default_identity_orientation(
    grid_index: tuple[int, int, int],
    site_center: np.ndarray,
    context: dict[str, Any],
) -> np.ndarray:
    """Default identity SO(3) rotation matrix."""
    return np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class BasisParticle:
    """
    A constituent particle within the unit cell's crystallographic asymmetric basis.

    Attributes:
        geometry: Canonical unmutated ParticleGeometry prototype centered at local origin.
        fractional_offset: Relative coordinates [u, v, w] in [0, 1)^3 within the unit cell.
        sublattice_id: Sublattice or role tag (e.g. 'A', 'B', 'horizontal', 'vertical').
        orientation_fn: Callable evaluating local SO(3) orientation from (grid_index, site_center, context).
        metadata: Additional basis metadata.
    """
    geometry: ParticleGeometry
    fractional_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sublattice_id: str = ""
    orientation_fn: OrientationFn = _default_identity_orientation
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self,
            "fractional_offset",
            np.asarray(self.fractional_offset, dtype=np.float64).reshape(3),
        )


@runtime_checkable
class InterlinkedCell(Protocol):
    """
    Universal protocol for all interlinked metamaterial unit cells.
    """
    name: str
    family: str
    parent_network: str
    basis_particles: list[BasisParticle]
    coordination_number: int
    neighbor_catenation_offsets: list[tuple[int, int, int]]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """
        Calculate expected minimum physical clearance (mm) for given pitch and strut thickness.
        Positive indicates non-contact clearance; negative indicates collision.
        """
        ...

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """
        Inverse solver: determine the minimum unit cell pitch (mm) required
        to achieve target_clearance with given strut_diameter.
        """
        ...

    def instantiate_site(
        self,
        grid_index: tuple[int, int, int],
        site_origin: np.ndarray,
        cell_pitch: float,
        id_start: int = 0,
        context: dict[str, Any] | None = None,
    ) -> list[InterlinkedParticle]:
        """
        Instantiate and place all BasisParticles for a specific spatial lattice site.
        """
        ...


class InterlinkedRegistry:
    """
    Central catalog and factory for interlinked metamaterial unit cells.
    """
    _registry: dict[str, type[InterlinkedCell]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an InterlinkedCell class."""
        def decorator(cell_cls: type[InterlinkedCell]):
            key = name.strip().lower()
            cls._registry[key] = cell_cls
            return cell_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[InterlinkedCell]:
        """Retrieve registered cell class by name (case-insensitive)."""
        key = name.strip().lower().replace("_", "-")
        if key not in cls._registry:
            # Try raw lowercase
            key_raw = name.strip().lower()
            if key_raw in cls._registry:
                return cls._registry[key_raw]
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(
                f"Unknown interlinked cell '{name}'. Available cells: {available}"
            )
        return cls._registry[key]

    @classmethod
    def list_cells(cls) -> list[dict[str, Any]]:
        """Return metadata summary for all registered cells."""
        summary = []
        for key, cell_cls in sorted(cls._registry.items()):
            inst = cell_cls() if callable(cell_cls) else None
            summary.append({
                "key": key,
                "name": getattr(inst, "name", key),
                "family": getattr(inst, "family", "unknown"),
                "parent_network": getattr(inst, "parent_network", "unknown"),
                "coordination_number": getattr(inst, "coordination_number", 0),
            })
        return summary

    @classmethod
    def clear(cls) -> None:
        """Clear registry (primarily for isolated test fixtures)."""
        cls._registry.clear()


# =============================================================================
# Pilot Cell Implementation: C-6-TT (Truncated Tetrahedron on Simple Cubic)
# =============================================================================

def _build_truncated_tetrahedron_geometry(size: float = 1.0) -> ParticleGeometry:
    """
    Generate local ParticleGeometry for a Truncated Tetrahedron centered at [0, 0, 0].
    Vertices: 12. Struts: 18.
    """
    s = float(size)
    scale = s / (3.0 * np.sqrt(2.0))
    raw_nodes = []
    for s1, s2, s3 in [
        (+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)
    ]:
        raw_nodes.append([3 * s1 * scale, 1 * s2 * scale, 1 * s3 * scale])
        raw_nodes.append([1 * s1 * scale, 3 * s2 * scale, 1 * s3 * scale])
        raw_nodes.append([1 * s1 * scale, 1 * s2 * scale, 3 * s3 * scale])

    nodes = np.round(np.array(raw_nodes, dtype=np.float64), decimals=8)
    nodes = np.unique(nodes, axis=0)

    edge_len = 2.0 * np.sqrt(2.0) * scale
    struts: list[tuple[int, int]] = []
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = float(np.linalg.norm(nodes[i] - nodes[j]))
            if abs(dist - edge_len) < 1e-4 * max(1.0, edge_len):
                struts.append((i, j))

    struts_arr = np.array(struts, dtype=np.int64)
    bounding_radius = float(np.max(np.linalg.norm(nodes, axis=1)))

    return ParticleGeometry(
        nodes=nodes,
        struts=struts_arr,
        bounding_radius=bounding_radius,
        geometry_type="TT",
        metadata={
            "size": s,
            "edge_length": float(edge_len),
            "num_vertices": 12,
            "num_struts": 18,
        },
    )


@InterlinkedRegistry.register("c6tt")
@InterlinkedRegistry.register("c-6-tt")
class C6TTCell:
    """
    Simple Cubic Truncated Tetrahedron (C-6-TT) PAM Unit Cell.
    Zhou et al., Science 2025. 6-fold face catenation on Simple Cubic network.
    """
    name: str = "C-6-TT"
    family: str = "pam_polyhedra"
    parent_network: str = "pcu"
    coordination_number: int = 6
    neighbor_catenation_offsets: list[tuple[int, int, int]] = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    # Validated empirical clearance factor when unit_cell_pitch a0 = 1.25 * cage_size:
    # Delta = kappa * a0 - D_strut, with kappa ≈ 0.131370
    CLEARANCE_KAPPA: float = 0.131370
    DEFAULT_SIZE_RATIO: float = 0.80  # cage_size / a0 = 1 / 1.25

    def __init__(self, size_ratio: float = 0.80):
        self.size_ratio = float(size_ratio)
        # Canonical prototype for unit size (s=1.0)
        self._proto_geom = _build_truncated_tetrahedron_geometry(size=1.0)
        self.basis_particles: list[BasisParticle] = [
            BasisParticle(
                geometry=self._proto_geom,
                fractional_offset=np.zeros(3, dtype=np.float64),
                sublattice_id="A",
                orientation_fn=_default_identity_orientation,
                metadata={"role": "center_cage"},
            )
        ]

    def forward_clearance(
        self,
        unit_cell_pitch: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """
        Evaluate surface clearance for C-6-TT.
        Delta = kappa * a0 - D_strut
        """
        a0 = float(unit_cell_pitch)
        d = float(strut_diameter)
        return float(self.CLEARANCE_KAPPA * a0 - d)

    def resolve_pitch(
        self,
        target_clearance: float,
        strut_diameter: float,
        **kwargs: Any,
    ) -> float:
        """
        Invert clearance relationship to solve minimum pitch:
        a0 = (target_clearance + strut_diameter) / kappa
        """
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
        Instantiate the C-6-TT basis particle scaled to size = cell_pitch * size_ratio.
        """
        ctx = context or {}
        a0 = float(cell_pitch)
        s = a0 * self.size_ratio

        # Scale local geometry to actual cage size
        scaled_geom = _build_truncated_tetrahedron_geometry(size=s)

        orig = np.asarray(site_origin, dtype=np.float64).reshape(3)
        bp = self.basis_particles[0]
        offset = bp.fractional_offset * a0
        center = orig + offset

        R = bp.orientation_fn(grid_index, center, ctx)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = center

        p = InterlinkedParticle(
            particle_id=int(id_start),
            geometry=scaled_geom,
            transform=T,
            sublattice_id=bp.sublattice_id,
            cell_index=tuple(grid_index),
            metadata={
                "cell_name": self.name,
                "size": s,
                "unit_cell_pitch": a0,
                **bp.metadata,
            },
        )
        return [p]
