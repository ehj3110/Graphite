from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.meshing_backends import extract_isosurface


def _sphere_field(n: int = 40, radius: float = 0.7) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float]]:
    axis = np.linspace(-1.0, 1.0, n)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.sqrt(X**2 + Y**2 + Z**2) - radius
    spacing = (float(axis[1] - axis[0]),) * 3
    origin = (float(axis[0]), float(axis[0]), float(axis[0]))
    return field.astype(np.float32), spacing, origin


def test_marching_cubes_backend_returns_mesh() -> None:
    field, spacing, origin = _sphere_field()
    res = extract_isosurface(field, spacing=spacing, origin=origin, backend="marching_cubes")
    assert res.backend_used == "marching_cubes"
    assert len(res.mesh.faces) > 0
    assert res.mesh.faces.shape[1] == 3


def test_pyvista_backend_parity_or_fallback() -> None:
    field, spacing, origin = _sphere_field()
    res = extract_isosurface(field, spacing=spacing, origin=origin, backend="pyvista_flying_edges")
    # If pyvista is unavailable or fails, fallback should be explicit and non-crashing.
    assert res.backend_used in {"pyvista_flying_edges", "marching_cubes"}
    assert len(res.mesh.faces) > 0
    if res.backend_used == "marching_cubes":
        assert res.fallback_used
        assert res.fallback_reason is not None


def test_enforce_watertight_does_not_crash() -> None:
    field, spacing, origin = _sphere_field(n=32, radius=0.6)
    base = extract_isosurface(
        field,
        spacing=spacing,
        origin=origin,
        backend="marching_cubes",
        enforce_watertight=False,
    )
    res = extract_isosurface(
        field,
        spacing=spacing,
        origin=origin,
        backend="marching_cubes",
        enforce_watertight=True,
    )
    assert len(res.mesh.vertices) > 0
    assert len(res.mesh.faces) > 0
    base_extent = np.max(base.mesh.extents)
    hard_extent = np.max(res.mesh.extents)
    assert 0.5 * base_extent <= hard_extent <= 2.0 * base_extent

