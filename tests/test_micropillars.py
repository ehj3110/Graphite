from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.micropillars import (
    MicropillarConfig,
    generate_micropillars,
    sample_pillar_anchors,
)


def _simple_cube(size: float = 2.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[size, size, size])


def test_sample_pillar_anchors() -> None:
    cube = _simple_cube(size=2.0)
    cfg = MicropillarConfig(spacing_mm=0.500, max_pillars=100)
    pts, norms = sample_pillar_anchors(cube, cfg)

    assert len(pts) > 0
    assert len(pts) == len(norms)
    assert pts.shape[1] == 3
    assert norms.shape[1] == 3
    # Verify unit normals
    lengths = np.linalg.norm(norms, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_sample_pillar_anchors_filter_printable() -> None:
    cube = _simple_cube(size=2.0)
    # Filter only self-supporting pillars (60 to 90 deg from horizontal)
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        filter_printable=True,
        min_angle_from_horizontal_deg=60.0,
        max_angle_from_horizontal_deg=90.0,
    )
    pts, norms = sample_pillar_anchors(cube, cfg)
    abs_nz = np.abs(norms[:, 2])
    angles = np.rad2deg(np.arcsin(np.clip(abs_nz, 0.0, 1.0)))
    assert np.all(angles >= 60.0 - 1e-4)
    assert np.all(angles <= 90.0 + 1e-4)


def test_generate_micropillars_cube() -> None:
    cube = _simple_cube(size=2.0)
    initial_faces = len(cube.faces)
    initial_extents = cube.extents.copy()

    cfg = MicropillarConfig(
        diameter_mm=0.100,
        height_mm=0.400,
        spacing_mm=0.500,
        circular_segments=8,
    )
    result = generate_micropillars(cube, cfg)

    assert len(result.faces) > initial_faces
    assert result.is_watertight
    # Bounding box should expand outward by approximately height_mm
    assert np.all(result.extents >= initial_extents)
    assert np.all(result.extents <= initial_extents + 2.0 * cfg.height_mm + 0.05)


def test_generate_micropillars_separate_forest() -> None:
    cube = _simple_cube(size=2.0)
    cfg = MicropillarConfig(
        diameter_mm=0.100,
        height_mm=0.400,
        spacing_mm=0.500,
    )
    forest = generate_micropillars(cube, cfg, return_separate_forest=True)
    assert len(forest.faces) > 0
    # Face count should be multiple of cylinder faces (8-sided cylinder = 8*2 sides + 2*8 caps = 32 tris)
    assert len(forest.vertices) > 0
