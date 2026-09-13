"""
Unit tests for transversely isotropic hexagonal pentamode (AB translation stagger).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
import trimesh

from graphite.generators.hexagonal_pentamode import (
    ROLE_BASAL,
    ROLE_VERTICAL,
    ab_layer_shift,
    generate_hexagonal_pentamode_graph,
    generate_hexagonal_pentamode_lattice,
    generate_transverse_hexagonal_pentamode,
    hexagonal_pentamode_cell,
)
from graphite.generators.pentamode import LatticeGraph, ImplicitField, bicone_radius


class TestABCell:
    def test_no_inplane_rotation(self):
        """Layer A and Layer B share identical in-plane strut orientations."""
        a, c = 1.0, 1.5
        nodes, edges, roles = generate_transverse_hexagonal_pentamode(nx=2, ny=2, nz=1, a=a, c=c)
        basal = edges[roles == ROLE_BASAL]
        z = 0.5 * (nodes[basal[:, 0], 2] + nodes[basal[:, 1], 2])
        a_edges = basal[np.abs(z - 0.0) < 1e-6]
        b_edges = basal[np.abs(z - c) < 1e-6]
        assert len(a_edges) > 0 and len(b_edges) > 0

        def folded_angles(eidx: np.ndarray) -> np.ndarray:
            d = nodes[eidx[:, 1], :2] - nodes[eidx[:, 0], :2]
            ang = np.arctan2(d[:, 1], d[:, 0])
            return np.sort(np.unique(np.round((ang % (np.pi / 3.0)), 5)))

        assert np.allclose(folded_angles(a_edges), folded_angles(b_edges))

    def test_ab_lateral_shift(self):
        a, c = 2.0, 3.0
        delta = ab_layer_shift(a, c)
        assert delta[0] == pytest.approx(0.5 * np.sqrt(3.0) * a)
        assert delta[1] == pytest.approx(0.5 * a)
        assert delta[2] == pytest.approx(c)

        nodes, _edges, _roles = generate_transverse_hexagonal_pentamode(nx=1, ny=1, nz=1, a=a, c=c)
        layer_a = nodes[np.abs(nodes[:, 2]) < 1e-6][:, :2]
        layer_b = nodes[np.abs(nodes[:, 2] - c) < 1e-6][:, :2]
        # Every B vertex equals some A vertex + delta_xy
        shift = delta[:2]
        for pb in layer_b:
            diffs = np.linalg.norm(layer_a + shift - pb, axis=1)
            assert np.min(diffs) < 1e-3

    def test_hub_degree_z4(self):
        nodes, edges, roles = generate_transverse_hexagonal_pentamode(nx=2, ny=2, nz=1, a=1.0, c=1.5)
        degrees = np.zeros(len(nodes), dtype=int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        incident: dict[int, list[int]] = defaultdict(list)
        for (u, v), role in zip(edges, roles, strict=True):
            incident[int(u)].append(int(role))
            incident[int(v)].append(int(role))
        hubs = [
            i
            for i, rs in incident.items()
            if rs and all(r == ROLE_VERTICAL for r in rs)
        ]
        assert len(hubs) > 0
        assert all(degrees[i] == 4 for i in hubs)


class TestCoordination:
    def test_interior_z4_no_overcoordination(self):
        nodes, edges, roles = generate_transverse_hexagonal_pentamode(
            nx=4, ny=4, nz=2, a=5.0, c=7.5
        )
        degrees = np.zeros(len(nodes), dtype=int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        incident: dict[int, list[int]] = defaultdict(list)
        for (u, v), role in zip(edges, roles, strict=True):
            incident[int(u)].append(int(role))
            incident[int(v)].append(int(role))

        # Global: no parasitic over-coordination
        assert int(degrees.max()) <= 4

        # All hubs Z=4
        hubs = [
            i
            for i, rs in incident.items()
            if rs and all(r == ROLE_VERTICAL for r in rs)
        ]
        assert len(hubs) > 0
        assert all(degrees[i] == 4 for i in hubs)

        # Ring nodes: at most one vertical; full honeycomb verts with a vertical are Z=4
        for i, rs in incident.items():
            n_basal = sum(1 for r in rs if r == ROLE_BASAL)
            n_vert = sum(1 for r in rs if r == ROLE_VERTICAL)
            if n_basal == 0:
                continue  # pure hub
            if n_vert:
                assert n_vert == 1
                assert degrees[i] == n_basal + 1
                assert degrees[i] <= 4
            if n_basal == 3 and n_vert == 1:
                assert degrees[i] == 4



class TestAPI:
    def test_hexagonal_pentamode_cell(self):
        nodes, edges = hexagonal_pentamode_cell(a=1.0, c=1.5)
        assert len(nodes) > 0
        assert len(edges) > 0

    def test_graph_nx_ny_nz(self):
        nodes, struts, roles = generate_hexagonal_pentamode_graph(nx=3, ny=3, nz=2, a=5.0, c=7.5)
        assert len(nodes) > 0
        assert np.any(roles == ROLE_VERTICAL)
        assert np.any(roles == ROLE_BASAL)

    def test_lattice_graph_mesh(self):
        graph = generate_hexagonal_pentamode_lattice(
            a=5.0, c=7.5, r_min=0.28, r_max=0.85, r_basal=0.22,
            output_format="graph", nx=2, ny=2, nz=1,
        )
        assert isinstance(graph, LatticeGraph)
        assert graph.metadata["stagger"] == "AB_translation"

        mesh = generate_hexagonal_pentamode_lattice(
            a=5.0, c=7.5, r_min=0.30, r_max=0.80, r_basal=0.22,
            output_format="mesh", nx=2, ny=2, nz=1,
        )
        assert isinstance(mesh, trimesh.Trimesh)
        assert mesh.volume > 0.0
        assert len(mesh.faces) > 100

    def test_bicone_and_sdf(self):
        assert bicone_radius(0.0, 0.25, 1.0) == pytest.approx(0.25)
        assert bicone_radius(0.5, 0.25, 1.0) == pytest.approx(1.0)
        sdf = generate_hexagonal_pentamode_lattice(
            a=5.0, c=7.5, r_min=0.3, r_max=0.8, r_basal=0.22,
            output_format="implicit_sdf", grid_resolution=20, nx=2, ny=2, nz=1,
        )
        assert isinstance(sdf, ImplicitField)
        assert sdf.field.min() < 0.0
        assert sdf.field.max() > 0.0
