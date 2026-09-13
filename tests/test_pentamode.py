"""
Unit tests for Continuous Pentamode (Meta-Fluid) Network Generator.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import trimesh

from graphite.generators.pentamode import (
    generate_pentamode_lattice,
    generate_diamond_cubic_graph,
    bicone_radius,
    LatticeGraph,
    ImplicitField,
)


class TestPentamodeKinematics:
    """Test diamond cubic crystallography, coordination number, and bond angles."""

    def test_diamond_cubic_coordination_and_valency(self):
        bounds = ((0.0, 0.0, 0.0), (30.0, 30.0, 30.0))
        a = 10.0
        nodes, struts = generate_diamond_cubic_graph(bounds=bounds, unit_cell_size=a)

        assert len(nodes) > 0
        assert len(struts) > 0

        # Calculate node degrees
        degrees = np.zeros(len(nodes), dtype=int)
        for u, v in struts:
            degrees[u] += 1
            degrees[v] += 1

        # Check internal nodes (at least 1 unit cell away from outer boundaries)
        interior_mask = (
            (nodes[:, 0] >= a) & (nodes[:, 0] <= 30.0 - a) &
            (nodes[:, 1] >= a) & (nodes[:, 1] <= 30.0 - a) &
            (nodes[:, 2] >= a) & (nodes[:, 2] <= 30.0 - a)
        )
        interior_degrees = degrees[interior_mask]
        assert len(interior_degrees) > 0

        # Every single interior node in diamond cubic must have coordination Z = 4
        unique_degrees = np.unique(interior_degrees)
        assert list(unique_degrees) == [4], f"Expected Z=4, got degrees: {unique_degrees}"

    def test_diamond_bond_angles(self):
        bounds = ((0.0, 0.0, 0.0), (30.0, 30.0, 30.0))
        a = 10.0
        nodes, struts = generate_diamond_cubic_graph(bounds=bounds, unit_cell_size=a)

        # Adjacency list
        adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
        for u, v in struts:
            adj[u].append(v)
            adj[v].append(u)

        # Find nodes with degree 4 well inside boundary
        interior_mask = (
            (nodes[:, 0] >= a) & (nodes[:, 0] <= 30.0 - a) &
            (nodes[:, 1] >= a) & (nodes[:, 1] <= 30.0 - a) &
            (nodes[:, 2] >= a) & (nodes[:, 2] <= 30.0 - a)
        )
        interior_indices = np.where(interior_mask)[0]

        expected_angle = np.degrees(np.arccos(-1.0 / 3.0))  # ~109.4712 deg

        for idx in interior_indices:
            neighbors = adj[idx]
            if len(neighbors) != 4:
                continue
            center = nodes[idx]
            # Compute all 6 pairwise angles between the 4 bond vectors
            vecs = [nodes[n] - center for n in neighbors]
            vecs = [v / np.linalg.norm(v) for v in vecs]
            for i in range(4):
                for j in range(i + 1, 4):
                    dot = np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0)
                    angle = np.degrees(np.arccos(dot))
                    assert angle == pytest.approx(expected_angle, abs=0.2)

    def test_bicone_profile_math(self):
        r_min = 0.25
        r_max = 1.10
        # Ends must match r_min
        assert bicone_radius(0.0, r_min, r_max) == pytest.approx(r_min)
        assert bicone_radius(1.0, r_min, r_max) == pytest.approx(r_min)
        # Midspan must match r_max
        assert bicone_radius(0.5, r_min, r_max) == pytest.approx(r_max)
        # Monotonic increase on [0, 0.5]
        t_samples = np.linspace(0.0, 0.5, 11)
        r_samples = bicone_radius(t_samples, r_min, r_max)
        assert np.all(np.diff(r_samples) >= 0)


class TestPentamodeOutputs:
    """Test graph, mesh, hierarchical sub-trusses, and implicit SDF output modes."""

    def test_pentamode_graph_output(self):
        bounds = ((0.0, 0.0, 0.0), (15.0, 15.0, 15.0))
        res = generate_pentamode_lattice(
            bounds=bounds,
            unit_cell_size=10.0,
            r_min=0.30,
            r_max=1.00,
            output_format="graph",
        )
        assert isinstance(res, LatticeGraph)
        assert len(res.nodes) > 0
        assert len(res.struts) > 0
        assert res.metadata["r_min"] == 0.30
        assert res.metadata["r_max"] == 1.00

    def test_pentamode_hierarchical_subtruss_tetrahedral(self):
        bounds = ((0.0, 0.0, 0.0), (15.0, 15.0, 15.0))
        res = generate_pentamode_lattice(
            bounds=bounds,
            unit_cell_size=10.0,
            r_min=0.30,
            r_max=1.00,
            hierarchical=True,
            sub_element_type="tetrahedral",
            output_format="graph",
        )
        assert isinstance(res, LatticeGraph)
        # Hierarchical infill expands strut and node counts
        assert len(res.nodes) > 100
        assert len(res.struts) > 200

        # Verify watertight mesh conversion
        mesh = res.to_trimesh(add_spheres=True)
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume > 0.0

    def test_pentamode_hierarchical_subtruss_c15(self):
        bounds = ((0.0, 0.0, 0.0), (15.0, 15.0, 15.0))
        res = generate_pentamode_lattice(
            bounds=bounds,
            unit_cell_size=10.0,
            r_min=0.30,
            r_max=1.00,
            hierarchical=True,
            sub_element_type="c15_subtruss",
            output_format="graph",
        )
        assert isinstance(res, LatticeGraph)
        assert len(res.nodes) > 100
        assert len(res.struts) > 200
        assert res.metadata["sub_element_type"] == "c15_subtruss"

    def test_pentamode_solid_mesh_watertight(self):
        bounds = ((0.0, 0.0, 0.0), (15.0, 15.0, 15.0))
        mesh = generate_pentamode_lattice(
            bounds=bounds,
            unit_cell_size=10.0,
            r_min=0.35,
            r_max=0.90,
            output_format="mesh",
        )
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.is_watertight
        assert mesh.volume > 100.0

    def test_pentamode_implicit_sdf_and_isosurface(self):
        bounds = ((0.0, 0.0, 0.0), (12.0, 12.0, 12.0))
        sdf = generate_pentamode_lattice(
            bounds=bounds,
            unit_cell_size=10.0,
            r_min=0.40,
            r_max=1.20,
            output_format="implicit_sdf",
            grid_resolution=28,
        )
        assert isinstance(sdf, ImplicitField)
        assert sdf.field.shape == (28, 28, 28)
        assert sdf.field.min() < 0.0  # Solid interior exists
        assert sdf.field.max() > 0.0  # Void exterior exists

        # Extract isosurface mesh
        iso_mesh = sdf.to_trimesh(level=0.0)
        # Slender pentamode in a 12mm box has low volume fraction (~1%), volume ~15 mm^3
        assert isinstance(iso_mesh, trimesh.Trimesh)
        assert len(iso_mesh.faces) > 500
        assert iso_mesh.volume > 10.0

    def test_lattice_graph_inp_export(self):
        graph = LatticeGraph(
            nodes=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
            struts=np.array([[0, 1]], dtype=np.int64),
            radii=0.5,
        )
        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            graph.export_inp(tmp_path, job_name="TEST_PENTAMODE")
            content = tmp_path.read_text(encoding="utf-8")
            assert "*HEADING" in content
            assert "*NODE" in content
            assert "*ELEMENT, TYPE=B31" in content
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
