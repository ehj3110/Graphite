"""Phase 0–1 tests for explicit woodpile extrusion."""

from __future__ import annotations

from graphite.explicit.woodpile_extrude import (
    crosshatch_layer_plan,
    generate_crosshatch_box,
    generate_crosshatch_cylinder,
    generate_piecewise_crosshatch_box,
    generate_piecewise_crosshatch_cylinder,
    generate_single_layer_box,
    mid_plane_solid_fraction,
    strut_centers_in_extent,
    true_woodpile_phase_shift_mm,
    verify_cylinder_clip,
)


def test_strut_centers_1mm_box_p200():
    centers = strut_centers_in_extent(0.0, 1.0, pore_mm=0.2, origin_mm=0.0)
    assert centers == [0.0, 0.4, 0.8]


def test_single_x_layer_watertight_and_bar_count():
    mesh, report = generate_single_layer_box(
        strut_axis="x",
        pore_mm=0.2,
        width_x_mm=1.0,
        depth_y_mm=1.0,
    )
    assert report["n_bars"] == 3
    assert report["transverse_centers_mm"] == [0.0, 0.4, 0.8]
    assert report["watertight"]
    assert mesh.is_watertight
    bounds = mesh.bounds
    assert abs(bounds[1, 2] - bounds[0, 2] - 0.2) < 1e-3
    assert abs(bounds[1, 0] - bounds[0, 0] - 1.0) < 1e-3


def test_crosshatch_layer_plan_1mm_p200():
    plan = crosshatch_layer_plan(1.0, 0.2, origin_z_mm=0.0)
    assert len(plan) == 5
    assert plan[0] == (0, 0.0, 0.2)
    assert plan[-1][1] == 0.8


def test_crosshatch_box_watertight_and_mid_y_solid_fraction():
    mesh, report = generate_crosshatch_box(
        pore_mm=0.2,
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
    )
    assert report["n_layers"] == 5
    assert report["layer_axes"] == ["x", "y", "x", "y", "x"]
    assert report["watertight"]
    assert mesh.is_watertight
    sf = mid_plane_solid_fraction(mesh, center_mm=0.5, n_samples=128)
    assert 0.65 <= sf <= 0.85


def test_piecewise_crosshatch_box_p139_p277_watertight():
    z_breaks = [0.0, 0.5, 1.0]
    pores = [0.1386, 0.2771]
    mesh, report = generate_piecewise_crosshatch_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    assert report["watertight"]
    assert mesh.is_watertight
    assert len(report["slabs"]) == 2
    assert report["slabs"][0]["pore_mm"] == pores[0]
    assert report["slabs"][1]["pore_mm"] == pores[1]
    assert report["slabs"][1]["swap_xy"] is False
    assert report["interface_layer_qc"]["ok"]
    iface = report["interface_layer_qc"]["checks"][0]
    assert iface["z_interface_mm"] == 0.5
    assert iface["axis_below"] == "Y"
    assert iface["axis_above"] == "X"
    assert iface["perpendicular"]
    assert report["n_layers"] > 0
    bounds = mesh.bounds
    assert abs(bounds[1, 2] - bounds[0, 2] - 1.0) < 0.02


def test_piecewise_extrude_matches_implicit_band_rotation():
    """Extrude and implicit piecewise reports share swap_xy / ⊥ layer-0 hatch."""
    from graphite.implicit.piecewise_woodpile import woodpile_piecewise_box_single_pass
    from graphite.math.woodpile_anchor import (
        verify_piecewise_interface_layers_perpendicular,
    )

    z_breaks = [0.0, 0.5, 1.0]
    pores = [0.1386, 0.2771]
    kwargs = dict(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    _, rep_ext = generate_piecewise_crosshatch_box(**kwargs)
    _, rep_imp = woodpile_piecewise_box_single_pass(
        resolution_mm=0.05,
        true_woodpile=False,
        **kwargs,
    )
    qc_ext = rep_ext["interface_layer_qc"]
    qc_imp = verify_piecewise_interface_layers_perpendicular(
        rep_imp["slabs"],
        alternate_band_orientation=True,
    )
    assert qc_ext["ok"] and qc_imp["ok"]
    assert qc_ext["checks"][0]["perpendicular"]
    for slab_ext, slab_imp in zip(rep_ext["slabs"], rep_imp["slabs"], strict=True):
        assert slab_ext["layer_index_offset"] == slab_imp["layer_index_offset"]
        assert (
            slab_ext["layers"][-1]["strut_axis"]
            == slab_imp["layers"][-1]["strut_axis"]
        )
        assert (
            slab_ext["layers"][0]["strut_axis"]
            == slab_imp["layers"][0]["strut_axis"]
        )


def test_true_woodpile_phase_shifts():
    pore = 0.8
    assert true_woodpile_phase_shift_mm(0, pore) == (0.0, 0.0)
    assert true_woodpile_phase_shift_mm(1, pore) == (0.0, 0.0)
    assert true_woodpile_phase_shift_mm(2, pore) == (0.0, pore)
    assert true_woodpile_phase_shift_mm(3, pore) == (pore, 0.0)


def test_piecewise_cylinder_default_spec_watertight_and_clip():
    z_breaks = [0.0, 2.23, 2.31, 2.7]
    pores = [0.8, 0.4, 0.2]
    mesh, report = generate_piecewise_crosshatch_cylinder(
        radius_mm=1.0,
        height_mm=2.7,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    assert report["watertight"]
    assert mesh.is_watertight
    assert len(report["slabs"]) == 3
    assert report["clip_qc"]["ok"]
    assert report["interface_layer_qc"]["ok"]
    assert len(report["interface_layer_qc"]["checks"]) == 2


def test_true_woodpile_cylinder_watertight():
    mesh, report = generate_crosshatch_cylinder(
        pore_mm=0.8,
        radius_mm=1.0,
        height_mm=2.7,
        true_woodpile=True,
    )
    assert report["true_woodpile"]
    assert report["watertight"]
    assert mesh.is_watertight
    clip = verify_cylinder_clip(mesh, center_x_mm=0.0, center_y_mm=0.0, radius_mm=1.0)
    assert clip["ok"]


def test_piecewise_cylinder_extrude_matches_implicit_band_layers():
    from graphite.implicit.piecewise_woodpile import woodpile_piecewise_cylinder_single_pass

    z_breaks = [0.0, 2.23, 2.31, 2.7]
    pores = [0.8, 0.4, 0.2]
    kwargs = dict(
        radius_mm=1.0,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    _, rep_ext = generate_piecewise_crosshatch_cylinder(
        height_mm=2.7,
        **kwargs,
    )
    _, rep_imp = woodpile_piecewise_cylinder_single_pass(
        resolution_mm=0.05,
        true_woodpile=False,
        **kwargs,
    )
    for slab_ext, slab_imp in zip(rep_ext["slabs"], rep_imp["slabs"], strict=True):
        assert slab_ext["layer_index_offset"] == slab_imp["layer_index_offset"]
        assert (
            slab_ext["layers"][0]["strut_axis"]
            == slab_imp["layers"][0]["strut_axis"]
        )
