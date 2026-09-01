"""Tests for element-slice cross-section fill."""

from __future__ import annotations

import numpy as np

from graphite.aristo.cross_section_viz import element_slice_field, element_slice_soft_field


def test_element_slice_averages_overlapping_tets():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2, 3], [1, 4, 5, 3]], dtype=np.int64)
    element_vm = np.array([2.0, 18.0], dtype=np.float64)

    _a, _b, field, material = element_slice_field(
        nodes,
        elements,
        element_vm,
        plane="xz",
        x_center=0.5,
        y_center=0.25,
        z_center=0.0,
        x_half_thickness=0.5,
        y_half_thickness=0.5,
        z_half_thickness=0.5,
        n_a=40,
        n_b=40,
        domain_clip_mm=(0.0, 1.0, 0.0, 1.0),
    )

    inside = material > 0.0
    assert np.any(inside)
    assert np.nanmin(field[inside]) >= 2.0 - 1e-9
    assert np.nanmax(field[inside]) <= 18.0 + 1e-9


def test_element_slice_soft_blurs_inside_material():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=np.int64)
    element_vm = np.array([2.0, 18.0], dtype=np.float64)

    _a, _b, sharp, mat = element_slice_field(
        nodes,
        elements,
        element_vm,
        plane="xz",
        x_center=0.5,
        y_center=0.25,
        z_center=0.0,
        x_half_thickness=0.5,
        y_half_thickness=0.5,
        z_half_thickness=0.5,
        n_a=80,
        n_b=80,
        domain_clip_mm=(0.0, 1.0, 0.0, 1.0),
    )
    _a, _b, soft, _mat = element_slice_soft_field(
        nodes,
        elements,
        element_vm,
        plane="xz",
        x_center=0.5,
        y_center=0.25,
        z_center=0.0,
        x_half_thickness=0.5,
        y_half_thickness=0.5,
        z_half_thickness=0.5,
        n_a=80,
        n_b=80,
        domain_clip_mm=(0.0, 1.0, 0.0, 1.0),
        blur_radius_mm=0.08,
    )

    inside = mat > 0.0
    soft_inside = _mat > 0.0
    assert np.all(np.isfinite(soft[soft_inside]))
    assert np.all(np.isnan(soft[~soft_inside]))
    assert float(np.nanstd(soft[inside])) < float(np.nanstd(sharp[inside]))


def test_element_slice_fills_tet_interior_and_leaves_void_white():
    # Unit tet in the first octant; slice through mid-plane y=0.25.
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    element_vm = np.array([12.0], dtype=np.float64)

    _a, _b, field, material = element_slice_field(
        nodes,
        elements,
        element_vm,
        plane="xz",
        x_center=0.0,
        y_center=0.25,
        z_center=0.0,
        x_half_thickness=0.5,
        y_half_thickness=0.5,
        z_half_thickness=0.5,
        n_a=80,
        n_b=80,
        domain_clip_mm=(0.0, 1.0, 0.0, 1.0),
    )

    assert np.any(material > 0.0)
    assert np.all(np.isfinite(field[material > 0.0]))
    assert np.all(field[material > 0.0] == 12.0)
    assert np.all(np.isnan(field[material <= 0.0]))

    # Corner of domain outside tet should remain void.
    assert np.isnan(field[0, 0])
