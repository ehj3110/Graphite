"""Tests for piecewise cross-hatch woodpile box single-pass generation."""

from __future__ import annotations

from graphite.implicit.piecewise_woodpile import woodpile_piecewise_box_single_pass
from graphite.math.woodpile_anchor import compute_woodpile_xy_origin_box


def test_box_origin_center_void_places_void_at_cube_center():
    ox, oy, meta = compute_woodpile_xy_origin_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        pore_mm=0.2,
        mode="center_void",
        origin_x_mm=0.0,
        origin_y_mm=0.0,
    )
    assert abs(ox - 0.7) < 1e-9
    assert abs(oy - 0.7) < 1e-9
    assert meta["frame"] == "axis_aligned_box"


def test_piecewise_woodpile_box_single_pass_coarse():
    mesh, report = woodpile_piecewise_box_single_pass(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=[0.0, 0.5, 1.0],
        pore_mm=[0.2, 0.4],
        resolution_mm=0.05,
        true_woodpile=False,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    assert report["combine_method"] == "single_pass_implicit"
    assert report["lattice_type"] == "cross_hatch"
    assert len(report["slabs"]) == 2
    assert report["slabs"][0]["swap_xy"] is False
    assert report["slabs"][1]["swap_xy"] is False
    assert len(mesh.faces) > 0
    assert mesh.volume > 0
