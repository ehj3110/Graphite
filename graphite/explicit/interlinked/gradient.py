"""
Graphite Explicit Interlinked — Spatial Strut Thickness Grading Engine

Defines:
- ThicknessGradient: Abstract base class for spatial wire radius evaluation.
- LinearThicknessGradient: 1D gradient along Cartesian axes (X, Y, or Z).
- RadialThicknessGradient: Spherically symmetric radial grading from core to shell.
- FieldThicknessGradient: Arbitrary user-defined 3D scalar callable r(x, y, z).
- resolve_thickness_gradient: Helper to construct gradient from objects or kwargs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np


class ThicknessGradient(ABC):
    """Abstract spatial strut thickness grading field."""

    @abstractmethod
    def evaluate(self, point: np.ndarray | Sequence[float]) -> float:
        """Evaluate wire radius (mm) at a 3D point [x, y, z]."""
        ...

    def evaluate_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate wire radii (mm) for an (N, 3) array of points.
        Default implementation vectorizes or loops evaluate.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            return np.array([self.evaluate(pts)], dtype=np.float64)
        return np.array([self.evaluate(p) for p in pts], dtype=np.float64)


@dataclass(frozen=True)
class LinearThicknessGradient(ThicknessGradient):
    """
    Linear strut thickness gradient along a specified Cartesian axis.

    Attributes:
        axis: Coordinate axis along which thickness varies ('x', 'y', or 'z').
        r_start: Strut radius (mm) at start of bounds.
        r_end: Strut radius (mm) at end of bounds.
        bounds: Optional (min_coord, max_coord) defining the gradient span.
                If None, auto-resolved from particle envelope during generation.
        clamp: If True, clamps evaluated radius to [min(r_start, r_end), max(r_start, r_end)].
    """
    axis: str = "x"
    r_start: float = 0.60
    r_end: float = 0.30
    bounds: tuple[float, float] | None = None
    clamp: bool = True

    def __post_init__(self):
        ax = self.axis.lower()
        if ax not in ("x", "y", "z"):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {self.axis}")
        if self.r_start <= 0 or self.r_end <= 0:
            raise ValueError(f"strut radii must be positive, got start={self.r_start}, end={self.r_end}")

    @property
    def axis_index(self) -> int:
        return {"x": 0, "y": 1, "z": 2}[self.axis.lower()]

    def evaluate(self, point: np.ndarray | Sequence[float]) -> float:
        pt = np.asarray(point, dtype=np.float64).reshape(3)
        coord = float(pt[self.axis_index])

        if self.bounds is None:
            # Fallback when bounds not yet bound: return average
            return float(0.5 * (self.r_start + self.r_end))

        c_min, c_max = self.bounds
        if abs(c_max - c_min) < 1e-9:
            return float(self.r_start)

        t = (coord - c_min) / (c_max - c_min)
        if self.clamp:
            t = float(np.clip(t, 0.0, 1.0))

        val = (1.0 - t) * self.r_start + t * self.r_end
        return float(val)

    def with_bounds(self, bounds: tuple[float, float]) -> LinearThicknessGradient:
        """Return a copy with resolved coordinate bounds."""
        return LinearThicknessGradient(
            axis=self.axis,
            r_start=self.r_start,
            r_end=self.r_end,
            bounds=(float(bounds[0]), float(bounds[1])),
            clamp=self.clamp,
        )


@dataclass(frozen=True)
class RadialThicknessGradient(ThicknessGradient):
    """
    Spherically symmetric strut thickness gradient from a center core to outer shell.

    Attributes:
        center: (x0, y0, z0) origin of the spherical grading.
        r_core: Strut radius (mm) at center/core.
        r_shell: Strut radius (mm) at outer shell radius.
        radius: Radial distance (mm) from center where r_shell is reached.
        clamp: If True, clamps evaluated radius to [min(r_core, r_shell), max(r_core, r_shell)].
    """
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    r_core: float = 0.60
    r_shell: float = 0.30
    radius: float = 25.0
    clamp: bool = True

    def __post_init__(self):
        if self.r_core <= 0 or self.r_shell <= 0:
            raise ValueError(f"strut radii must be positive, got core={self.r_core}, shell={self.r_shell}")
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def evaluate(self, point: np.ndarray | Sequence[float]) -> float:
        pt = np.asarray(point, dtype=np.float64).reshape(3)
        c = np.asarray(self.center, dtype=np.float64).reshape(3)
        dist = float(np.linalg.norm(pt - c))

        t = dist / self.radius
        if self.clamp:
            t = float(np.clip(t, 0.0, 1.0))

        val = (1.0 - t) * self.r_core + t * self.r_shell
        return float(val)


@dataclass(frozen=True)
class FieldThicknessGradient(ThicknessGradient):
    """
    Arbitrary user-defined 3D scalar callable r(x, y, z).

    Attributes:
        fn: Callable taking (3,) float array and returning scalar float radius (mm).
        r_min: Minimum clipping clamp radius (mm).
        r_max: Maximum clipping clamp radius (mm).
    """
    fn: Callable[[np.ndarray], float]
    r_min: float = 0.10
    r_max: float = 2.00

    def evaluate(self, point: np.ndarray | Sequence[float]) -> float:
        pt = np.asarray(point, dtype=np.float64).reshape(3)
        raw = float(self.fn(pt))
        return float(np.clip(raw, self.r_min, self.r_max))


def resolve_thickness_gradient(
    gradient: ThicknessGradient | None = None,
    gradient_axis: str | None = None,
    gradient_radius_range: tuple[float, float] | None = None,
    gradient_bounds: tuple[float, float] | None = None,
    points_envelope: tuple[float, float] | None = None,
) -> ThicknessGradient | None:
    """
    Resolve and bind a ThicknessGradient from explicit object or convenience kwargs.

    Args:
        gradient: Pre-configured ThicknessGradient instance.
        gradient_axis: Convenience axis name ('x', 'y', 'z').
        gradient_radius_range: (r_start, r_end) radius range in mm.
        gradient_bounds: Coordinate bounds (coord_min, coord_max).
        points_envelope: Auto-detected coordinate bounds along gradient axis if bounds not given.

    Returns:
        Bound ThicknessGradient instance, or None if no gradient specified.
    """
    if gradient is not None:
        if isinstance(gradient, LinearThicknessGradient) and gradient.bounds is None:
            if gradient_bounds is not None:
                return gradient.with_bounds(gradient_bounds)
            elif points_envelope is not None:
                return gradient.with_bounds(points_envelope)
        return gradient

    if gradient_axis is not None and gradient_radius_range is not None:
        bounds = gradient_bounds if gradient_bounds is not None else points_envelope
        return LinearThicknessGradient(
            axis=gradient_axis,
            r_start=float(gradient_radius_range[0]),
            r_end=float(gradient_radius_range[1]),
            bounds=bounds,
        )

    return None
