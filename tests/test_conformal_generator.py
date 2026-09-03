"""
Tests for the wired-in Graphite explicit engine.

Covers:
  - Sizing solver (unchanged, previously passing)
  - Mesh repair (unchanged, previously passing)
  - generate_a15_conformal_lattice  (routes to a15_conformal.py — existing)
  - generate_conformal_lattice with lattice_type='A15' (same underlying code)
  - generate_conformal_lattice with lattice_type='SC'  (new SC engine)
"""

from __future__ import annotations
import os
import pytest
import numpy as np
import trimesh

from graphite.explicit import (
    solve_sizing,
    repair_cad_mesh,
    generate_a15_conformal_lattice,
    generate_conformal_lattice,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere_stl(tmp_path, radius=5.0) -> str:
    """Save a sphere STL and return its path."""
    s = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    path = str(tmp_path / "sphere.stl")
    s.export(path)
    return path


# ---------------------------------------------------------------------------
# Sizing Solver (unchanged — must keep passing)
# ---------------------------------------------------------------------------

def test_sizing_solver_strut_radius():
    res = solve_sizing(solid_fraction=0.10, cell_size=2.0)
    assert "strut_radius" in res
    assert res["strut_radius"] > 0


def test_sizing_solver_solid_fraction():
    res1 = solve_sizing(solid_fraction=0.10, cell_size=2.0)
    r = res1["strut_radius"]
    res2 = solve_sizing(strut_radius=r, cell_size=2.0)
    assert np.isclose(res2["solid_fraction"], 0.10, atol=1e-3)


def test_sizing_solver_cell_size():
    res1 = solve_sizing(solid_fraction=0.10, cell_size=2.0)
    r = res1["strut_radius"]
    res3 = solve_sizing(solid_fraction=0.10, strut_radius=r)
    assert np.isclose(res3["cell_size"], 2.0, atol=1e-3)


def test_sizing_solver_warning():
    with pytest.warns(UserWarning, match="exceeds the printable limit of 20%"):
        solve_sizing(solid_fraction=0.25, cell_size=2.0)


# ---------------------------------------------------------------------------
# Mesh Repair (unchanged — must keep passing)
# ---------------------------------------------------------------------------

def test_mesh_repair():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    faces = sphere.faces[:-1]
    broken = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    assert not broken.is_watertight
    repaired = repair_cad_mesh(broken)
    assert repaired.is_watertight


# ---------------------------------------------------------------------------
# A15 Conformal Lattice — routes to a15_conformal.py (existing, tested)
# ---------------------------------------------------------------------------

def test_a15_conformal_generation(tmp_path):
    """A15 pipeline with skip_sweep=True must return topology stats."""
    cad_path = _make_sphere_stl(tmp_path, radius=5.0)
    res = generate_a15_conformal_lattice(
        cad_filepath=cad_path,
        cell_size=8.0,
        strut_radius=0.5,
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
    )
    assert "nodes_count" in res
    assert res["nodes_count"] > 0
    assert "struts_count" in res
    assert res["struts_count"] > 0
    assert "cyan_struts_count" in res
    assert "red_struts_count" in res


def test_a15_has_cyan_struts(tmp_path):
    """A15 surface dual (cyan struts) must be non-empty for a non-trivial mesh."""
    cad_path = _make_sphere_stl(tmp_path, radius=5.0)
    res = generate_a15_conformal_lattice(
        cad_filepath=cad_path,
        cell_size=8.0,
        strut_radius=0.5,
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
    )
    assert res["cyan_struts_count"] > 0, "A15 surface dual must generate boundary struts"


# ---------------------------------------------------------------------------
# Unified generate_conformal_lattice — A15 & SC paths
# ---------------------------------------------------------------------------

def test_unified_lattice_a15(tmp_path):
    """generate_conformal_lattice(lattice_type='A15') must route to a15_conformal.py."""
    cad_path = _make_sphere_stl(tmp_path, radius=5.0)
    res = generate_conformal_lattice(
        cad_filepath=cad_path,
        cell_size=8.0,
        strut_radius=0.5,
        lattice_type="A15",
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
    )
    assert res["nodes_count"] > 0
    assert res["struts_count"] > 0


def test_unified_lattice_sc(tmp_path):
    """generate_conformal_lattice(lattice_type='SC') must route to conformal_generator.py."""
    cad_path = _make_sphere_stl(tmp_path, radius=5.0)
    res = generate_conformal_lattice(
        cad_filepath=cad_path,
        cell_size=4.0,
        strut_radius=0.5,
        lattice_type="SC",
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
        volume_fraction_threshold=0.5,
    )
    assert res["nodes_count"] > 0
    assert res["struts_count"] > 0
    assert "cyan_struts_count" in res
    assert "red_struts_count" in res


def test_conformal_lattice_invalid_type(tmp_path):
    """An unknown lattice_type must raise an error."""
    cad_path = _make_sphere_stl(tmp_path, radius=5.0)
    with pytest.raises((ValueError, KeyError, TypeError)):
        generate_conformal_lattice(
            cad_filepath=cad_path,
            cell_size=8.0,
            strut_radius=0.5,
            lattice_type="BCC",
            export_dir=str(tmp_path / "out"),
            skip_sweep=True,
        )


# ---------------------------------------------------------------------------
# Mode (Conformal vs Boolean) and trimesh.Trimesh input tests
# ---------------------------------------------------------------------------

def test_boolean_mode_and_trimesh_input_a15(tmp_path):
    """Test A15 generation with mode='boolean' and trimesh.Trimesh object input."""
    # Create the mesh object
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    
    res = generate_conformal_lattice(
        cad_filepath=sphere,
        cell_size=8.0,
        strut_radius=0.5,
        lattice_type="A15",
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
        mode="boolean",
    )
    assert res["nodes_count"] > 0
    assert res["struts_count"] > 0
    # In boolean mode, skin dual struts are not extracted
    assert res["cyan_struts_count"] == 0
    # All struts are red
    assert res["red_struts_count"] == res["struts_count"]


def test_boolean_mode_and_trimesh_input_sc(tmp_path):
    """Test SC generation with mode='boolean' and trimesh.Trimesh object input."""
    # Create the mesh object
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    
    res = generate_conformal_lattice(
        cad_filepath=sphere,
        cell_size=4.0,
        strut_radius=0.5,
        lattice_type="SC",
        export_dir=str(tmp_path / "out"),
        skip_sweep=True,
        mode="boolean",
        volume_fraction_threshold=0.5,
    )
    assert res["nodes_count"] > 0
    assert res["struts_count"] > 0
    # In boolean mode, skin dual struts are not extracted
    assert res["cyan_struts_count"] == 0
    # All struts are red
    assert res["red_struts_count"] == res["struts_count"]

