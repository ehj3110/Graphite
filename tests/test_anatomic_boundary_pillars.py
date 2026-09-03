import pytest
import trimesh
import numpy as np
from graphite.implicit.micropillars import MicropillarConfig, segment_cad_boundary, sample_pillar_anchors

def test_segment_cad_boundary():
    mesh = trimesh.load("test_parts/SkullCutout_OriginToZero.stl")
    masks = segment_cad_boundary(mesh)
    
    assert "top" in masks
    assert "bottom" in masks
    assert "sides" in masks
    
    assert np.any(masks["top"])
    assert np.any(masks["bottom"])
    assert np.any(masks["sides"])
    
    total_true = np.sum(masks["top"]) + np.sum(masks["bottom"]) + np.sum(masks["sides"])
    assert total_true == len(mesh.faces)

def test_sample_pillar_anchors_z_aligned():
    boundary = trimesh.load("test_parts/SkullCutout_OriginToZero.stl")
    mesh = boundary.copy()
    
    config = MicropillarConfig(
        spacing_mm=2.0,
        selected_faces=("top", "bottom"),
        orientation="z_aligned",
        boundary_mesh=boundary,
        max_boundary_dist_mm=10.0
    )
    pts, norms = sample_pillar_anchors(mesh, config)
    
    assert len(pts) > 0
    for n in norms:
        is_up = np.allclose(n, [0, 0, 1])
        is_down = np.allclose(n, [0, 0, -1])
        assert is_up or is_down

def test_sample_pillar_anchors_local_normal():
    boundary = trimesh.load("test_parts/SkullCutout_OriginToZero.stl")
    mesh = boundary.copy()
    
    config = MicropillarConfig(
        spacing_mm=2.0,
        selected_faces=("top", "bottom"),
        orientation="local_normal",
        boundary_mesh=boundary,
        max_boundary_dist_mm=10.0
    )
    pts, norms = sample_pillar_anchors(mesh, config)
    
    assert len(pts) > 0
    assert len(norms) == len(pts)
