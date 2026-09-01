"""Tests for ``build_piecewise_woodpile_mesh`` generator dispatch."""

from __future__ import annotations

import warnings

import pytest

from graphite.implicit.woodpile_input import (
    WoodpileImplicitSpec,
    WoodpileLatticeSpec,
    build_piecewise_woodpile_mesh,
    default_stem_from_spec,
    repair_implicit_woodpile_mesh,
    repair_woodpile_mesh,
)


def test_build_piecewise_woodpile_extrude_box():
    spec = WoodpileLatticeSpec(
        domain="box",
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=[0.0, 0.5, 1.0],
        pore_mm=[0.1386, 0.2771],
        anchor_mode="center_void",
        alternate_band_orientation=True,
        generator="extrude",
        repair_mesh=False,
    )
    mesh, report = build_piecewise_woodpile_mesh(spec)
    assert report["generator"] == "extrude"
    assert report["combine_method"] == "extrude"
    assert report["watertight"]
    assert mesh.is_watertight
    assert len(mesh.faces) < 5000
    assert report["interface_layer_qc"]["ok"]
    assert "lattice_spec" in report


def test_default_generator_is_extrude():
    spec = WoodpileLatticeSpec()
    assert spec.generator == "extrude"


def test_default_stem_extrude_suffix():
    spec = WoodpileLatticeSpec(
        domain="cylinder",
        z_breaks_mm=[0.0, 2.7],
        pore_mm=[0.8],
    )
    stem = default_stem_from_spec(spec)
    assert stem.endswith("_extrude")


def test_implicit_stem_has_no_extrude_suffix():
    spec = WoodpileLatticeSpec(
        domain="cylinder",
        generator="implicit",
        z_breaks_mm=[0.0, 2.7],
        pore_mm=[0.8],
    )
    stem = default_stem_from_spec(spec)
    assert not stem.endswith("_extrude")


def test_extrude_rejects_invert_solids():
    spec = WoodpileLatticeSpec(
        domain="box",
        generator="extrude",
        invert_solids=True,
        z_breaks_mm=[0.0, 1.0],
        pore_mm=[0.2],
    )
    with pytest.raises(ValueError, match="invert_solids"):
        build_piecewise_woodpile_mesh(spec)


def test_woodpile_implicit_spec_deprecated():
    with pytest.warns(DeprecationWarning, match="WoodpileImplicitSpec"):
        spec = WoodpileImplicitSpec(
            domain="box",
            z_breaks_mm=[0.0, 1.0],
            pore_mm=[0.2],
        )
    assert isinstance(spec, WoodpileLatticeSpec)


def test_repair_implicit_woodpile_mesh_deprecated():
    import trimesh

    mesh = trimesh.creation.box()
    with pytest.warns(DeprecationWarning, match="repair_implicit_woodpile_mesh"):
        repaired = repair_implicit_woodpile_mesh(mesh)
    assert repair_woodpile_mesh(mesh).vertices.shape == repaired.vertices.shape
