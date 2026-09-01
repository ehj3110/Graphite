"""STL surface clean helpers (Open3D + trimesh backends)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.aristo.stl_surface_clean import (
    HAS_OPEN3D,
    _cap_face_vertex_indices,
    clean_stl_surface,
    surface_clean_backend,
    trimesh_to_o3d,
)


def _flat_needle_cap_mesh() -> trimesh.Trimesh:
    """Two caps with one long needle triangle on the bottom plane."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 2],
            [2, 5, 3],
            [2, 3, 0],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_surface_clean_backend_matches_open3d_flag():
    if HAS_OPEN3D:
        assert surface_clean_backend() == "open3d"
    else:
        assert surface_clean_backend() in ("open3d_subprocess", "trimesh")


@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D not installed")
def test_trimesh_to_o3d_roundtrip():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    o3d_mesh = trimesh_to_o3d(mesh)
    assert len(o3d_mesh.vertices) > 0
    assert len(o3d_mesh.triangles) > 0


def test_clean_reduces_needle_max_edge():
    mesh = _flat_needle_cap_mesh()
    before = float(mesh.edges_unique_length.max())
    cleaned = clean_stl_surface(mesh, target_edge_mm=0.15)
    after = float(cleaned.edges_unique_length.max())
    assert after <= before + 1e-6
    assert len(cleaned.faces) > 0


def test_trimesh_repair_pipeline_runs_on_box():
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    cleaned = clean_stl_surface(mesh, target_edge_mm=0.15)
    assert len(cleaned.faces) > 0
    assert cleaned.euler_number is not None


def test_cap_vertices_remain_planar_after_clean():
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 0.2))
    z_min_in = float(mesh.bounds[0][2])
    z_max_in = float(mesh.bounds[1][2])
    cleaned = clean_stl_surface(mesh, target_edge_mm=0.15)
    bottom_verts, top_verts = _cap_face_vertex_indices(cleaned)
    bottom_z = cleaned.vertices[bottom_verts, 2]
    top_z = cleaned.vertices[top_verts, 2]
    assert bottom_z.max() == pytest.approx(z_min_in, abs=1e-6)
    assert bottom_z.min() == pytest.approx(z_min_in, abs=1e-6)
    assert top_z.max() == pytest.approx(z_max_in, abs=1e-6)
    assert top_z.min() == pytest.approx(z_max_in, abs=1e-6)
