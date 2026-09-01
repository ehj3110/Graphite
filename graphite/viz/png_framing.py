"""
Standard framing for Graphite PyVista PNG exports.

All off-screen PNGs should frame the model so it occupies the center
``CONTENT_FILL_FRACTION`` of the viewport (default 75%). Load arrows use a
fixed *screen* size — not world units and not tied to force or stress magnitude.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pyvista as pv

DEFAULT_WINDOW_SIZE: tuple[int, int] = (1024, 768)
CONTENT_FILL_FRACTION: float = 0.75
LOAD_ARROW_SCREEN_FRACTION: float = 0.10
MAX_LOAD_FACE_ARROWS: int = 36


def create_offscreen_plotter(
    window_size: Sequence[int] = DEFAULT_WINDOW_SIZE,
    *,
    background: str = "white",
) -> pv.Plotter:
    """Create a plotter with Graphite's default PNG window size."""
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.background_color = background
    return plotter


def viewport_world_height(plotter: pv.Plotter) -> float:
    """
    Visible world-space height at the camera focal plane.

    Used to size decorative overlays (e.g. load arrows) as a fixed fraction of
    the image, independent of model scale or load magnitude.
    """
    cam = plotter.camera
    if cam.parallel_projection:
        return float(2.0 * cam.parallel_scale)
    position = np.asarray(cam.position, dtype=np.float64)
    focal = np.asarray(cam.focal_point, dtype=np.float64)
    distance = float(np.linalg.norm(position - focal))
    half_angle = np.deg2rad(float(cam.view_angle) * 0.5)
    return float(2.0 * distance * np.tan(half_angle))


def frame_plotter_content(
    plotter: pv.Plotter,
    bounds: Sequence[float] | np.ndarray | None,
    *,
    fill_fraction: float = CONTENT_FILL_FRACTION,
    view_isometric: bool = False,
) -> None:
    """
    Frame ``bounds`` in the plotter so the content fills ``fill_fraction`` of
    the viewport (centered), leaving equal margin on all sides.
    """
    if view_isometric:
        plotter.view_isometric()
    if bounds is not None:
        plotter.reset_camera(bounds=tuple(float(v) for v in bounds))
    else:
        plotter.reset_camera()
    if abs(fill_fraction - 1.0) > 1e-9:
        plotter.camera.zoom(float(fill_fraction))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal ``(u, v)`` spanning the plane perpendicular to ``normal``."""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(n, u)
    return u, v


def subsample_load_face_sites(
    face_centroids: np.ndarray,
    load_direction: Sequence[float],
    *,
    max_arrows: int = MAX_LOAD_FACE_ARROWS,
) -> np.ndarray:
    """
    Pick arrow anchor points spread over the loaded surface.

    Uses a 2D grid in the plane perpendicular to ``load_direction``. Each
    occupied grid cell contributes one arrow at the mean of its face centroids.
    """
    points = np.asarray(face_centroids, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.shape[0] <= max(1, max_arrows):
        return points

    direction = np.asarray(load_direction, dtype=np.float64)
    u, v = _plane_basis(direction)
    local = np.column_stack([points @ u, points @ v])

    lo = local.min(axis=0)
    hi = local.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    aspect = float(span[0] / span[1])
    n_v = max(1, int(np.ceil(np.sqrt(max_arrows / max(aspect, 1e-9)))))
    n_u = max(1, int(np.ceil(max_arrows / n_v)))

    cell_u = np.floor((local[:, 0] - lo[0]) / span[0] * n_u).astype(np.int64)
    cell_v = np.floor((local[:, 1] - lo[1]) / span[1] * n_v).astype(np.int64)
    cell_u = np.clip(cell_u, 0, n_u - 1)
    cell_v = np.clip(cell_v, 0, n_v - 1)

    sites: list[np.ndarray] = []
    for iu in range(n_u):
        for iv in range(n_v):
            mask = (cell_u == iu) & (cell_v == iv)
            if mask.any():
                sites.append(points[mask].mean(axis=0))
    return np.asarray(sites, dtype=np.float64)


def add_fixed_load_arrows(
    plotter: pv.Plotter,
    base_points: np.ndarray,
    direction: Sequence[float],
    *,
    color: str = "red",
    screen_fraction: float = LOAD_ARROW_SCREEN_FRACTION,
    tip_gap_fraction: float = 0.15,
) -> None:
    """
    Add fixed-size load arrows whose tips hover just outside the loaded surface.

    ``base_points`` are on the loaded face. Arrow tails sit farther outward;
    tips end at ``base + outward * tip_gap`` (not embedded in the surface).
    """
    surface_pts = np.asarray(base_points, dtype=np.float64)
    if surface_pts.size == 0:
        return

    direction_arr = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction_arr))
    if norm < 1e-12:
        return
    unit = direction_arr / norm
    outward = -unit

    length = float(screen_fraction) * viewport_world_height(plotter)
    tip_gap = float(tip_gap_fraction) * length
    tails = surface_pts + outward * (tip_gap + length)
    directions = np.tile(unit, (len(surface_pts), 1))

    plotter.add_arrows(tails, directions, mag=length, color=color)


def add_fixed_load_arrow(
    plotter: pv.Plotter,
    base_point: Sequence[float],
    direction: Sequence[float],
    *,
    color: str = "red",
    screen_fraction: float = LOAD_ARROW_SCREEN_FRACTION,
) -> None:
    """Add a single fixed-size load arrow (see :func:`add_fixed_load_arrows`)."""
    add_fixed_load_arrows(
        plotter,
        np.asarray([base_point], dtype=np.float64),
        direction,
        color=color,
        screen_fraction=screen_fraction,
    )
