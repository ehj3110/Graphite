"""Tests for nodal stress recovery (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_solver import run_aristo
from graphite.aristo.stress_postprocess import (
    build_stress_fields,
    element_to_nodal_von_mises,
    nodal_to_element_mean,
)


def test_uniform_bar_nodal_matches_element():
    """Constant element stress → nodal average equals that constant."""
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    vm = np.array([42.0])
    vols = np.array([1.0])
    nodal, support = element_to_nodal_von_mises(elements, vm, vols)
    assert support.all()
    assert np.allclose(nodal, 42.0)


def test_nodal_to_element_mean():
    nodal = np.array([1.0, 2.0, 3.0, 4.0])
    elements = np.array([[0, 1, 2, 3]])
    assert nodal_to_element_mean(nodal, elements)[0] == pytest.approx(2.5)


def test_build_stress_fields_nodal_max_normalization():
    elements = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    vm = np.array([10.0, 30.0])
    vols = np.array([1.0, 1.0])
    mask = np.ones(2, dtype=bool)
    cfg = AristoConfig(stress_representation="nodal_averaged", fea_mesh_resolution=1.0)
    out = build_stress_fields(vm, elements, vols, mask, cfg)
    assert out["stress_field_mode"] == "nodal_averaged"
    assert out["max_von_mises_raw"] == pytest.approx(30.0)
    assert out["max_von_mises_element_raw"] == pytest.approx(30.0)
    assert out["von_mises_nodal_norm"].max() == pytest.approx(1.0)


def test_run_aristo_nodal_averaged_by_default():
    mesh = trimesh.creation.box([2.0, 2.0, 2.0])
    cfg = AristoConfig(
        fea_mesh_resolution=1.0,
        load_magnitude=0.1,
        bc_z_band_fraction=0.05,
    )
    result = run_aristo(mesh, cfg)
    assert result.stress_field_mode == "nodal_averaged"
    assert result.von_mises_nodal_raw.shape[0] == result.fea_nodes.shape[0]
    assert result.max_von_mises_raw <= result.max_von_mises_element_raw + 1e-9
    assert "stress_nodal" in result.mesh_quality_report
