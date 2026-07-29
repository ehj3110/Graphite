"""
Tests for A15 Conformal Lattice engine integration.
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
)


def test_sizing_solver():
    # 1. Given solid_fraction & cell_size, solve for strut_radius
    res1 = solve_sizing(solid_fraction=0.10, cell_size=2.0)
    assert "strut_radius" in res1
    r = res1["strut_radius"]
    assert r > 0

    # 2. Given strut_radius & cell_size, solve for solid_fraction
    res2 = solve_sizing(strut_radius=r, cell_size=2.0)
    assert np.isclose(res2["solid_fraction"], 0.10)

    # 3. Given solid_fraction & strut_radius, solve for cell_size
    res3 = solve_sizing(solid_fraction=0.10, strut_radius=r)
    assert np.isclose(res3["cell_size"], 2.0)


def test_sizing_solver_warning():
    # Verify that solid fractions above 20% emit a printable limit warning
    with pytest.warns(UserWarning, match="exceeds the printable limit of 20%"):
        solve_sizing(solid_fraction=0.25, cell_size=2.0)


def test_mesh_repair():
    # Create a non-watertight mesh (e.g. sphere with missing face)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    faces = sphere.faces[:-1]  # remove one face to make a hole
    broken_sphere = trimesh.Trimesh(vertices=sphere.vertices, faces=faces, process=False)
    assert not broken_sphere.is_watertight

    repaired = repair_cad_mesh(broken_sphere)
    assert repaired.is_watertight


def test_a15_conformal_generation():
    # Run conformal generation on Part2_Adapter.STL
    cad_path = r"c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\test_parts\Part2_Adapter.STL"
    assert os.path.exists(cad_path)

    # Skip actual geometry sweep to keep the test fast
    res = generate_a15_conformal_lattice(
        cad_filepath=cad_path,
        cell_size=8.0,
        strut_radius=0.5,
        export_dir="scratch/test_output",
        export_debug_stls=False,
        skip_sweep=True,
    )

    assert "nodes_count" in res
    assert res["nodes_count"] > 0
    assert "struts_count" in res
    assert res["struts_count"] > 0
    assert "cyan_struts_count" in res
    assert "red_struts_count" in res
