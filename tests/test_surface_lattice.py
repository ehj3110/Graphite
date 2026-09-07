# -*- coding: utf-8 -*-
"""
Tests for the unified 2D and Surface-Conformal Lattice Engine (graphite.explicit.surface_lattice).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import trimesh
import manifold3d as m3d

from graphite.explicit import (
    generate_surface_lattice,
    generate_tetrachiral_cell,
    generate_trichiral_cell,
)
from graphite.explicit.surface_lattice.unit_cells import tessellate_chiral_domain
from graphite.explicit.surface_lattice.cad_fixtures import inspect_cylinder_fixture

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASE_RING_1TO2 = WORKSPACE_ROOT / "test_parts" / "BaseRing_1to2.STL"


def test_unit_cells():
    """Verify chiral unit-cell generation parameters and topology counts."""
    nodes_sq, struts_sq, meta_sq = generate_tetrachiral_cell(L=8.0, r=1.5)
    assert meta_sq["D_pitch"] > 8.0
    assert len(struts_sq) == 20
    assert len(nodes_sq) == 24

    nodes_tri, struts_tri, meta_tri = generate_trichiral_cell(L=8.0, r=1.5)
    assert meta_tri["D_pitch"] > 8.0
    assert len(struts_tri) == 19
    assert len(nodes_tri) == 22


def test_tessellate_chiral_domain():
    """Verify 2D domain tessellation with periodic closure."""
    c_mid = 100.0
    height = 30.0
    segs, meta = tessellate_chiral_domain(
        topology="tetra",
        domain_width=c_mid,
        domain_height=height,
        n_circumferential=8,
        r_node=2.0,
    )
    assert len(segs) > 50
    assert meta["n_circumferential"] == 8
    assert np.isclose(meta["pitch_D"], 100.0 / 8.0)


def test_generate_surface_lattice_plate():
    """Verify Mode A: Flat plate extrusion (orthogonal Z-extrusion)."""
    mesh = generate_surface_lattice(
        surface="plate",
        pattern="tetra_chiral",
        width=40.0,
        height=40.0,
        thickness=3.0,
        n_circumferential=4,
        r_node=2.0,
        strut_w=1.6,
    )
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0
    assert mesh.is_watertight
    assert np.isclose(mesh.extents[2], 3.0, atol=0.2)
    assert mesh.volume > 0.0


def test_generate_surface_lattice_cylinder():
    """Verify Mode B: Cylindrical prism extrusion (conformal sleeve)."""
    rin = 19.05
    rout = 22.05
    height = 20.0
    mesh = generate_surface_lattice(
        surface="cylinder",
        pattern="tri_chiral",
        r_in=rin,
        r_out=rout,
        height=height,
        n_circumferential=8,
        r_node=2.0,
        strut_w=1.8,
    )
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0

    # Verify 2-manifold status using manifold3d
    m_mesh = m3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    )
    man = m3d.Manifold(m_mesh)
    assert man.status() == m3d.Error.NoError
    assert man.volume() > 0.0


@pytest.mark.skipif(not BASE_RING_1TO2.exists(), reason="BaseRing_1to2.STL not found in test_parts")
def test_generate_surface_lattice_cad_fixture():
    """Verify Method B: Auto-detecting CAD fixture and fusing collar rims."""
    fixture_info = inspect_cylinder_fixture(BASE_RING_1TO2)
    assert np.isclose(fixture_info["r_in"], 19.05, atol=1.0)
    assert np.isclose(fixture_info["r_out"], 25.4, atol=1.0)

    # Generate full ring with auto-fused collar rims
    full_ring = generate_surface_lattice(
        surface="cylinder",
        pattern="tetra_chiral",
        cad_fixture=BASE_RING_1TO2,
        height=19.05,
        y_base=6.65,
        n_circumferential=10,
        r_node=2.2,
        strut_w=2.0,
    )
    assert isinstance(full_ring, trimesh.Trimesh)
    assert len(full_ring.faces) > 0

    m_ring = m3d.Mesh(
        vert_properties=np.asarray(full_ring.vertices, dtype=np.float32),
        tri_verts=np.asarray(full_ring.faces, dtype=np.uint32),
    )
    man = m3d.Manifold(m_ring)
    assert man.status() == m3d.Error.NoError
    assert man.volume() > 0.0


def test_generate_surface_lattice_mesh_surface_dual():
    """Verify Mode C: Direct surface element mapping on a 3D sphere mesh."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=25.0)
    sphere.apply_translation([25.0, 25.0, 25.0])

    cage = generate_surface_lattice(
        surface="mesh",
        pattern="Rhombic",
        surface_mesh=sphere,
        sphere_center=[25.0, 25.0, 25.0],
        trim_radius=25.0 - 0.25,
        thickness=3.0,
        strut_w=1.2,
    )
    assert isinstance(cage, trimesh.Trimesh)
    assert len(cage.faces) > 0
    assert cage.volume > 0.0
