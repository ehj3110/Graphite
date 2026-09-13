"""
Graphite Explicit Interlinked — Canonical Particle Abstraction

Defines:
- ParticleGeometry: Lightweight, frozen specification of canonical particle wireframe/surface geometry at local origin [0, 0, 0].
- InterlinkedParticle: Placed, identified particle instance in SE(3) with lazy global coordinate evaluation and bridge methods to legacy PAMParticle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .pams import PAMParticle


@dataclass(frozen=True)
class ParticleGeometry:
    """
    Canonical, unmutated geometry of a discrete unbonded particle centered at [0, 0, 0].

    Attributes:
        nodes: (V, 3) float64 nodal coordinates in local particle space.
        struts: (E, 2) int64 edge connectivity in local index space (0 ... V-1).
        bounding_radius: Exact outer bounding sphere radius enclosing all local nodes.
        geometry_type: Semantic identifier (e.g. 'TT', 'TET', 'CO', 'OCT', 'ring', 'custom').
        metadata: Immutable dictionary of geometric parameters (edge length, size, etc.).
    """
    nodes: np.ndarray
    struts: np.ndarray
    bounding_radius: float
    geometry_type: str = "wireframe"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "nodes", np.asarray(self.nodes, dtype=np.float64))
        object.__setattr__(self, "struts", np.asarray(self.struts, dtype=np.int64))
        object.__setattr__(self, "bounding_radius", float(self.bounding_radius))


@dataclass
class InterlinkedParticle:
    """
    A discrete unbonded particle placed within an interlinked metamaterial assembly.

    Maintains local canonical geometry and an SE(3) rigid transformation matrix (R | t).
    Global coordinates are evaluated lazily via matrix-vector multiplication.
    """
    particle_id: int
    geometry: ParticleGeometry
    transform: np.ndarray  # (4, 4) float64 homogeneous matrix in SE(3)
    sublattice_id: str = ""
    cell_index: tuple[int, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.particle_id = int(self.particle_id)
        self.transform = np.asarray(self.transform, dtype=np.float64)
        if self.transform.shape != (4, 4):
            raise ValueError(f"transform must have shape (4, 4), got {self.transform.shape}")

    @property
    def local_nodes(self) -> np.ndarray:
        """Local coordinates centered at particle origin."""
        return self.geometry.nodes

    @property
    def struts(self) -> np.ndarray:
        """Local strut connectivity (E, 2) indexing 0 ... V-1."""
        return self.geometry.struts

    @property
    def geometry_type(self) -> str:
        """Particle geometry type tag."""
        return self.geometry.geometry_type

    @property
    def bounding_radius(self) -> float:
        """Local bounding sphere radius in mm."""
        return self.geometry.bounding_radius

    @property
    def center(self) -> np.ndarray:
        """Global center coordinates in mm (Cartesian translation vector t)."""
        return self.transform[:3, 3].copy()

    @center.setter
    def center(self, new_center: np.ndarray | Sequence[float]) -> None:
        self.transform[:3, 3] = np.asarray(new_center, dtype=np.float64).reshape(3)

    @property
    def rotation(self) -> np.ndarray:
        """Global orientation SO(3) rotation matrix (3, 3)."""
        return self.transform[:3, :3].copy()

    @rotation.setter
    def rotation(self, new_rot: np.ndarray) -> None:
        R = np.asarray(new_rot, dtype=np.float64).reshape(3, 3)
        self.transform[:3, :3] = R

    def global_nodes(self) -> np.ndarray:
        """
        Compute global coordinates for all particle nodes on the fly.
        x_global = (R @ x_local.T).T + t
        """
        R = self.transform[:3, :3]
        t = self.transform[:3, 3]
        return (R @ self.geometry.nodes.T).T + t

    def global_bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Return (center, bounding_radius) in global coordinate space."""
        return self.center, self.bounding_radius

    def __iter__(self):
        """Allows direct tuple unpacking: nodes, struts = particle."""
        yield self.global_nodes()
        yield self.struts

    def copy_transformed(self, new_id: int, new_transform: np.ndarray) -> InterlinkedParticle:
        """Clone particle with new ID and transformation matrix."""
        return InterlinkedParticle(
            particle_id=int(new_id),
            geometry=self.geometry,
            transform=np.asarray(new_transform, dtype=np.float64).copy(),
            sublattice_id=self.sublattice_id,
            cell_index=self.cell_index,
            metadata=dict(self.metadata),
        )

    def to_pam(self) -> PAMParticle:
        """
        Convert to legacy PAMParticle with baked global coordinates for backwards compatibility.
        """
        from .pams import PAMParticle
        return PAMParticle(
            particle_id=self.particle_id,
            nodes=self.global_nodes(),
            struts=self.struts.copy(),
            center=self.center,
            geometry_type=self.geometry_type,
            metadata={
                **self.metadata,
                "sublattice": self.sublattice_id,
                "cell_index": list(self.cell_index),
                "bounding_radius": self.bounding_radius,
            },
        )

    @classmethod
    def from_pam(
        cls,
        pam: PAMParticle,
        particle_id: int | None = None,
        sublattice_id: str = "",
    ) -> InterlinkedParticle:
        """
        Create an InterlinkedParticle from a legacy PAMParticle by isolating its local frame.
        """
        pid = pam.particle_id if particle_id is None else int(particle_id)
        c = np.asarray(pam.center, dtype=np.float64).reshape(3)
        local_nodes = np.asarray(pam.nodes, dtype=np.float64) - c
        radius = float(np.max(np.linalg.norm(local_nodes, axis=1))) if len(local_nodes) > 0 else 0.0

        geom = ParticleGeometry(
            nodes=local_nodes,
            struts=np.asarray(pam.struts, dtype=np.int64).copy(),
            bounding_radius=radius,
            geometry_type=pam.geometry_type,
            metadata=dict(pam.metadata),
        )

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = c

        sub_id = sublattice_id or str(pam.metadata.get("sublattice", ""))
        return cls(
            particle_id=pid,
            geometry=geom,
            transform=T,
            sublattice_id=sub_id,
            metadata=dict(pam.metadata),
        )
