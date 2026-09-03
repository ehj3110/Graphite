from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.micropillars import (
    MicropillarConfig,
    generate_micropillars,
    sample_pillar_anchors,
)


def _simple_box(size: float = 4.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=[size, size, size])


def test_poisson_disk_spacing_enforces_clearance() -> None:
    box = _simple_box(size=4.0)
    min_clearance = 0.250  # 250 um minimum clearance
    cfg = MicropillarConfig(
        spacing_mm=0.250,
        min_spacing_mm=min_clearance,
        distribution="poisson_disk",
    )
    pts, norms = sample_pillar_anchors(box, cfg)

    assert len(pts) > 10
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_observed_dist = dists[:, 1].min()
    # Every pair must be at least min_clearance apart (with tiny floating point tolerance)
    assert min_observed_dist >= min_clearance - 1e-4


def test_printability_filter_enforces_overhang_angle() -> None:
    box = _simple_box(size=4.0)
    # Self-supporting vertical to near-vertical angles: 60 to 90 degrees from horizontal
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        filter_printable=True,
        min_angle_from_horizontal_deg=60.0,
        max_angle_from_horizontal_deg=90.0,
    )
    pts, norms = sample_pillar_anchors(box, cfg)

    assert len(pts) > 0
    # Lateral faces have nz = 0 (0 deg) -> rejected. Only top/bottom (90 deg) kept!
    abs_nz = np.clip(np.abs(norms[:, 2]), 0.0, 1.0)
    angles_deg = np.rad2deg(np.arcsin(abs_nz))
    assert np.all(angles_deg >= 60.0 - 1e-3)
    assert np.all(angles_deg <= 90.0 + 1e-3)


def test_tangential_relaxation_tightens_spacing() -> None:
    box = _simple_box(size=4.0)
    cfg = MicropillarConfig(
        spacing_mm=0.250,
        distribution="relaxed",
    )
    pts, norms = sample_pillar_anchors(box, cfg)
    assert len(pts) > 10
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    # Ensure minimum distance is maintained after relaxation
    assert dists[:, 1].min() >= 0.15


def test_selected_faces_top_only() -> None:
    box = _simple_box(size=4.0)  # bounds [-2, 2] in X, Y, Z
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        selected_faces=("+z",),
    )
    pts, norms = sample_pillar_anchors(box, cfg)

    assert len(pts) > 0
    # All points must be on the top face (z >= 1.9)
    assert np.all(pts[:, 2] >= 1.90)
    # All normals must be clamped strictly upward [0, 0, 1]
    assert np.allclose(norms, [0.0, 0.0, 1.0], atol=1e-3)


def test_selected_faces_multiple_combinations() -> None:
    box = _simple_box(size=4.0)
    # Select +X and -Z faces
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        selected_faces=("+x", "-z"),
    )
    pts, norms = sample_pillar_anchors(box, cfg)

    assert len(pts) > 0
    is_px = (pts[:, 0] >= 1.90) & np.isclose(norms[:, 0], 1.0, atol=1e-2)
    is_nz = (pts[:, 2] <= -1.90) & np.isclose(norms[:, 2], -1.0, atol=1e-2)
    # Every point must be either on +X or on -Z
    assert np.all(is_px | is_nz)


def test_location_internal_only_excludes_outer_faces() -> None:
    box = _simple_box(size=4.0)
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        location="internal_only",
    )
    pts, norms = sample_pillar_anchors(box, cfg)
    # On a plain solid box, there are NO internal surfaces (all faces are outer faces)
    # So internal_only must return 0 points
    assert len(pts) == 0


def test_generate_micropillars_with_selected_face_union() -> None:
    box = _simple_box(size=4.0)
    cfg = MicropillarConfig(
        diameter_mm=0.080,
        height_mm=0.300,
        spacing_mm=0.400,
        selected_faces=("top",),  # synonym for +z
    )
    result = generate_micropillars(box, cfg)

    assert result.is_watertight
    # Bounding box should only expand in +Z, not in -Z, +X, -X, etc.
    assert result.bounds[1, 2] >= 2.0 + 0.25  # top expanded by ~0.300 mm
    assert np.isclose(result.bounds[0, 2], -2.0, atol=0.02)  # bottom unchanged


def test_selected_faces_cylinder_top_and_bottom() -> None:
    cyl = trimesh.creation.cylinder(radius=2.5, height=2.0)
    cfg = MicropillarConfig(
        spacing_mm=0.300,
        boundary_type="cylinder",
        selected_faces=("top", "bottom"),
    )
    pts, norms = sample_pillar_anchors(cyl, cfg)
    assert len(pts) > 0
    # Points should be either on top (z >= 0.9) or bottom (z <= -0.9)
    assert np.all((pts[:, 2] >= 0.9) | (pts[:, 2] <= -0.9))
    # Normals should be clamped strictly to +/- Z
    assert np.all(np.isclose(np.abs(norms[:, 2]), 1.0, atol=1e-3))
