"""Tests for Aristo mesh quality gates (Phase 0/1)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_solver import run_aristo
from graphite.aristo.mesh_quality import (
    build_quality_mask,
    element_aspect_ratios,
    element_volumes,
    mesh_has_giant_elements,
    resolve_max_tet_edge,
    resolve_min_tet_volume,
    stress_percentiles,
)


def _needle_tet_nodes() -> tuple[np.ndarray, np.ndarray]:
    """Extremely flat tet (high aspect ratio, tiny volume)."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-3, 0.0],
            [0.5, 0.0, 1e-3],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return nodes, elements


def test_element_volumes_unit_tet_positive():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elems = np.array([[0, 1, 2, 3]])
    vol = element_volumes(nodes, elems)[0]
    assert vol == pytest.approx(1.0 / 6.0)


def test_needle_tet_fails_quality_mask():
    nodes, elems = _needle_tet_nodes()
    cfg = AristoConfig(
        fea_mesh_resolution=1.0,
        min_tet_volume_fraction=0.01,
        max_tet_aspect_ratio=20.0,
    )
    mask, vols, aspects, _ = build_quality_mask(nodes, elems, cfg)
    assert not mask[0]
    assert aspects[0] > cfg.max_tet_aspect_ratio


def test_giant_tet_rejected_by_edge_cap():
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
    cfg = AristoConfig(fea_mesh_resolution=0.15, max_tet_edge_factor=4.0)
    mask, _, _, edges = build_quality_mask(nodes, elems, cfg)
    assert edges[0] > resolve_max_tet_edge(cfg) or not mask[0]


def test_resolve_min_tet_volume_uses_median_fraction():
    nodes, elems = _needle_tet_nodes()
    vols = np.abs(element_volumes(nodes, elems))
    vols = np.array([1.0, 0.5, 0.25, 0.1])
    cfg = AristoConfig(min_tet_volume_fraction=0.1)
    v_min = resolve_min_tet_volume(vols, cfg)
    assert v_min == pytest.approx(0.1 * np.median(vols))


def test_stress_percentiles_ignore_masked():
    vm = np.array([1.0, 2.0, 100.0, 3.0])
    mask = np.array([True, True, False, True])
    p = stress_percentiles(vm, mask)
    assert p["max"] == pytest.approx(3.0)
    assert p["p99"] <= 3.0


def test_run_aristo_returns_quality_fields():
    mesh = trimesh.creation.box([2.0, 2.0, 2.0])
    cfg = AristoConfig(
        fea_mesh_resolution=1.0,
        load_magnitude=0.1,
        fea_quality_mode="quick",
        bc_z_band_fraction=0.05,
    )
    result = run_aristo(mesh, cfg)
    assert result.quality_mask.shape == result.fea_elements.shape[:1]
    assert result.element_volumes.shape == result.quality_mask.shape
    assert "poor_fraction" in result.mesh_quality_report
    assert result.max_von_mises_element_raw >= result.max_von_mises_raw
