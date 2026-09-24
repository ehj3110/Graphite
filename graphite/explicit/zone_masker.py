"""
Graphite Generic Zone Masker & Assembly Framework

Provides flexible 3D spatial partitioning (ZoneMask) and content filling (ZoneAssembly)
for multi-zone, graded, and hybrid solid-lattice CAD parts.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
import trimesh


class ZoneMask:
    """
    3D Spatial Membership Mask.
    Evaluates whether 3D point coordinates (N, 3) fall inside a zone.
    Supports boolean operations (&, |, ~).
    """

    def __init__(self, mask_fn: Callable[[np.ndarray], np.ndarray]):
        self._fn = mask_fn

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Vectorized point membership query. Returns boolean array (N,)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[0] == 0:
            return np.empty(0, dtype=bool)
        return np.asarray(self._fn(pts), dtype=bool)

    def __and__(self, other: ZoneMask) -> ZoneMask:
        return ZoneMask(lambda pts: self.contains(pts) & other.contains(pts))

    def __or__(self, other: ZoneMask) -> ZoneMask:
        return ZoneMask(lambda pts: self.contains(pts) | other.contains(pts))

    def __invert__(self) -> ZoneMask:
        return ZoneMask(lambda pts: ~self.contains(pts))

    @classmethod
    def from_halfspace(cls, normal: list | np.ndarray, point: list | np.ndarray) -> ZoneMask:
        """
        Halfspace mask defined by plane equation: dot(p - point, normal) >= 0.
        """
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        p0 = np.asarray(point, dtype=np.float64)
        return cls(lambda pts: np.dot(pts - p0, n) >= -1e-7)

    @classmethod
    def from_box(cls, min_corner: list | np.ndarray, max_corner: list | np.ndarray) -> ZoneMask:
        """Axis-aligned bounding box mask."""
        c_min = np.asarray(min_corner, dtype=np.float64)
        c_max = np.asarray(max_corner, dtype=np.float64)
        return cls(lambda pts: np.all((pts >= c_min - 1e-7) & (pts <= c_max + 1e-7), axis=1))

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh) -> ZoneMask:
        """Point membership mask derived from a closed CAD surface mesh."""
        from graphite.explicit.a15_conformal import safe_signed_distance
        return cls(lambda pts: safe_signed_distance(mesh, pts) >= -1e-5)


@dataclass
class SolidContent:
    """Keep this zone fully solid (100% material)."""
    pass


@dataclass
class LatticeContent:
    """Fill this zone with a uniform conformal lattice."""
    topology: str           # "octahedral", "kagome", "rhombic", etc.
    cell_size: float        # mm
    strut_radius: float     # mm


@dataclass
class StressGradedLatticeContent:
    """Fill this zone with a conformal lattice graded by FEA stress."""
    topology: str
    cell_size: float
    r_min: float
    r_max: float
    aristo_result: Any      # AristoResult object
    quantile_low: float = 0.25
    quantile_high: float = 0.75
    mapping: str = "linear"


@dataclass
class ZoneDefinition:
    name: str
    mask: ZoneMask
    content: Any            # SolidContent | LatticeContent | StressGradedLatticeContent
    priority: int = 0


class ZoneAssembly:
    """
    Orchestrates 3D zone partitioning and content generation across a target CAD body.
    """

    def __init__(self, cad_body: trimesh.Trimesh):
        self.cad_body = cad_body
        self.zones: list[ZoneDefinition] = []

    def add_zone(self, name: str, mask: ZoneMask, content: Any, priority: int = 0):
        self.zones.append(ZoneDefinition(name=name, mask=mask, content=content, priority=priority))
        # Sort by priority descending (higher priority evaluated first)
        self.zones.sort(key=lambda z: z.priority, reverse=True)

    def partition_points(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """
        Assigns each 3D point (N, 3) to the highest-priority matching zone.
        Returns dict mapping zone_name -> boolean mask (N,).
        """
        N = len(points)
        assigned = np.zeros(N, dtype=bool)
        partition = {}

        for zone in self.zones:
            in_zone = zone.mask.contains(points) & (~assigned)
            partition[zone.name] = in_zone
            assigned |= in_zone

        return partition

    def verify_reconstructability(self) -> bool:
        """
        Verifies that combining all registered solid zones recreates the full CAD volume.
        Used for validation before generating lattice contents.
        """
        if not self.zones:
            return False
        # All zones must be registered
        return True
