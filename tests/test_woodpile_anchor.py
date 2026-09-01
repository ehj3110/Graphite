"""Tests for woodpile phase anchoring in bounded cylinders."""

from __future__ import annotations

from graphite.math.woodpile_anchor import (
    compute_woodpile_xy_origin,
    is_woodpile_solid_at_axis,
    verify_piecewise_interface_layers_perpendicular,
)


def test_default_origin_puts_solid_at_center():
    assert is_woodpile_solid_at_axis(coord_mm=0.0, pore_mm=0.8, origin_mm=0.0)


def test_center_void_origin_puts_void_at_center():
    ox, oy, meta = compute_woodpile_xy_origin(radius_mm=1.0, pore_mm=0.8, mode="center_void")
    assert ox == oy == 0.8
    assert meta["center_solid_at_origin"] is False


def test_edge_solid_reaches_cylinder_wall():
    ox, oy, meta = compute_woodpile_xy_origin(radius_mm=1.0, pore_mm=0.8, mode="edge_solid")
    assert abs(ox - 0.6) < 1e-9
    assert meta["edge_solid_at_plus_x"] is True
    assert meta["edge_solid_at_minus_x"] is True
    assert meta["center_solid_at_origin"] is False


def test_alternate_band_orientation_uses_global_layer_continuity():
    from graphite.math.woodpile_anchor import compute_band_orientation

    _, _, m0 = compute_band_orientation(0, prev_swap_xy=False, alternate_band_orientation=True)
    _, _, m1 = compute_band_orientation(1, prev_swap_xy=False, alternate_band_orientation=True)
    assert m0["orientation_mode"] == "global_layer_continuity"
    assert m0["swap_xy"] is False
    assert m1["swap_xy"] is False


def test_verify_piecewise_interface_layers_perpendicular_p139_p277():
    slabs = [
        {
            "band_index": 0,
            "z0_mm": 0.0,
            "swap_xy": False,
            "flip_layer_parity": False,
            "layer_index_offset": 0,
            "layers": [
                {"idx": 0, "global_layer_idx": 0, "strut_axis": "x"},
                {"idx": 1, "global_layer_idx": 1, "strut_axis": "y"},
                {"idx": 2, "global_layer_idx": 2, "strut_axis": "x"},
                {"idx": 3, "global_layer_idx": 3, "strut_axis": "y"},
            ],
        },
        {
            "band_index": 1,
            "z0_mm": 0.5,
            "swap_xy": False,
            "flip_layer_parity": False,
            "layer_index_offset": 4,
            "layers": [
                {"idx": 0, "global_layer_idx": 4, "strut_axis": "x"},
                {"idx": 1, "global_layer_idx": 5, "strut_axis": "y"},
            ],
        },
    ]
    qc = verify_piecewise_interface_layers_perpendicular(
        slabs, alternate_band_orientation=True
    )
    assert qc["ok"]
    assert qc["checks"][0]["axis_below"] == "Y"
    assert qc["checks"][0]["axis_above"] == "X"
    assert qc["checks"][0]["perpendicular"]


def test_dominant_strut_axis_uses_band_z_origin_and_global_offset():
    from graphite.math.woodpile_anchor import dominant_strut_axis_at_z

    assert (
        dominant_strut_axis_at_z(
            0.75,
            0.2771,
            z_layer_origin_mm=0.5,
            layer_index_offset=4,
        )
        == "X"
    )
    assert (
        dominant_strut_axis_at_z(
            0.49,
            0.1386,
            z_layer_origin_mm=0.0,
            layer_index_offset=0,
        )
        == "Y"
    )


def test_box_center_void_origin_at_unit_cube_center():
    from graphite.math.woodpile_anchor import compute_woodpile_xy_origin_box

    ox, oy, meta = compute_woodpile_xy_origin_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        pore_mm=0.2,
        mode="center_void",
    )
    assert abs(ox - 0.7) < 1e-9
    assert abs(oy - 0.7) < 1e-9
    assert meta["centered_origin_x_mm"] == 0.2
