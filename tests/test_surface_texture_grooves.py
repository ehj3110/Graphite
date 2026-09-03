from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.surface_textures import (
    SurfaceTextureConfig,
    apply_surface_texture,
    subdivide_for_texture,
)


def _simple_sphere(radius: float = 2.0, subdivisions: int = 2) -> trimesh.Trimesh:
    """Helper to create a small test sphere."""
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def test_subdivide_for_texture() -> None:
    mesh = _simple_sphere(radius=2.0, subdivisions=1)
    initial_faces = len(mesh.faces)
    initial_max_edge = float(np.max(mesh.edges_unique_length))

    target_edge = initial_max_edge / 2.5
    subdivided = subdivide_for_texture(mesh, target_edge_length_mm=target_edge, max_faces=10_000)

    assert len(subdivided.faces) > initial_faces
    assert float(np.max(subdivided.edges_unique_length)) <= target_edge * 1.05


def test_subdivide_max_faces_guard() -> None:
    mesh = _simple_sphere(radius=2.0, subdivisions=2)
    # Set cap lower than what full subdivision would produce
    cap = len(mesh.faces) * 2
    guarded = subdivide_for_texture(mesh, target_edge_length_mm=0.001, max_faces=cap)
    assert len(guarded.faces) <= cap * 2  # within guard factor


def test_apply_surface_texture_none() -> None:
    mesh = _simple_sphere(radius=2.0)
    cfg = SurfaceTextureConfig(texture_type="none")
    result = apply_surface_texture(mesh, cfg)
    assert np.allclose(result.vertices, mesh.vertices)
    assert len(result.faces) == len(mesh.faces)


def test_apply_surface_texture_microgrooves() -> None:
    mesh = _simple_sphere(radius=2.0, subdivisions=2)
    cfg = SurfaceTextureConfig(
        texture_type="microgrooves",
        amplitude_mm=0.025,  # 25 um
        wavelength_mm=0.100, # 100 um
        direction=(0, 0, 1),
        profile="sine",
        displacement_mode="centered",
    )
    result = apply_surface_texture(mesh, cfg)

    assert len(result.faces) > len(mesh.faces)
    assert not np.any(np.isnan(result.vertices))
    # Vertex coordinates must have changed due to displacement
    assert not np.allclose(result.vertices[:len(mesh.vertices)], mesh.vertices)
    # Extent should expand by approximately the amplitude (+/- 25 um)
    ext_orig = np.max(mesh.extents)
    ext_tex = np.max(result.extents)
    assert abs(ext_tex - ext_orig) < 0.1  # within 100 um of original extents


def test_displacement_modes() -> None:
    mesh = _simple_sphere(radius=3.0, subdivisions=2)
    amp = 0.050

    # Subdivide base to compare vertex by vertex
    sub_base = subdivide_for_texture(mesh, target_edge_length_mm=0.200 / 2.0)

    cfg_emboss = SurfaceTextureConfig(
        texture_type="microgrooves",
        amplitude_mm=amp,
        wavelength_mm=0.200,
        displacement_mode="emboss",
        repair_after_displacement=False,
    )
    res_emboss = apply_surface_texture(mesh, cfg_emboss)
    diff_emboss = res_emboss.vertices - sub_base.vertices
    disp_emboss = np.sum(diff_emboss * sub_base.vertex_normals, axis=1)
    # Emboss displacement along normal must be non-negative in [0, amp]
    assert np.all(disp_emboss >= -1e-6)
    assert float(np.max(disp_emboss)) <= amp + 1e-6

    cfg_engrave = SurfaceTextureConfig(
        texture_type="microgrooves",
        amplitude_mm=amp,
        wavelength_mm=0.200,
        displacement_mode="engrave",
        repair_after_displacement=False,
    )
    res_engrave = apply_surface_texture(mesh, cfg_engrave)
    diff_engrave = res_engrave.vertices - sub_base.vertices
    disp_engrave = np.sum(diff_engrave * sub_base.vertex_normals, axis=1)
    # Engrave displacement along normal must be non-positive in [-amp, 0]
    assert np.all(disp_engrave <= 1e-6)
    assert float(np.min(disp_engrave)) >= -amp - 1e-6


def test_triplanar_microgrooves() -> None:
    mesh = _simple_sphere(radius=2.0, subdivisions=2)
    cfg = SurfaceTextureConfig(
        texture_type="microgrooves",
        amplitude_mm=0.025,
        wavelength_mm=0.100,
        use_triplanar=True,
    )
    result = apply_surface_texture(mesh, cfg)
    assert len(result.faces) > len(mesh.faces)
    assert not np.any(np.isnan(result.vertices))
