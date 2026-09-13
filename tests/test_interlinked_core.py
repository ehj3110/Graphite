"""
Unit tests for the canonical InterlinkedParticle, ParticleGeometry,
InterlinkedCell protocol, InterlinkedRegistry, and C6TTCell pilot.
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    ParticleGeometry,
    InterlinkedParticle,
    BasisParticle,
    InterlinkedCell,
    InterlinkedRegistry,
    C6TTCell,
    PAMParticle,
    pam_particles_to_meshes,
    generate_truncated_tetrahedron_particle,
)


class TestParticleGeometryAndInterlinkedParticle:
    def test_particle_geometry_immutability_and_attributes(self):
        nodes = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
        struts = np.array([[0, 1]], dtype=np.int64)
        geom = ParticleGeometry(
            nodes=nodes,
            struts=struts,
            bounding_radius=1.0,
            geometry_type="test_rod",
            metadata={"prop": 42},
        )
        assert geom.nodes.shape == (2, 3)
        assert geom.struts.shape == (1, 2)
        assert geom.bounding_radius == 1.0
        assert geom.geometry_type == "test_rod"
        assert geom.metadata["prop"] == 42

    def test_interlinked_particle_lazy_transformation(self):
        # 2 vertices symmetric about origin: [-1, 0, 0] and [+1, 0, 0]
        nodes = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        struts = np.array([[0, 1]], dtype=np.int64)
        geom = ParticleGeometry(nodes=nodes, struts=struts, bounding_radius=1.0)

        # 90-degree Z-rotation + translation [10, 20, 30]
        R_z_90 = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)
        t = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_z_90
        T[:3, 3] = t

        p = InterlinkedParticle(particle_id=7, geometry=geom, transform=T, sublattice_id="A")

        # Local nodes remain unmutated [-1, 0, 0] and [1, 0, 0]
        np.testing.assert_allclose(p.local_nodes, nodes)
        np.testing.assert_allclose(p.center, [10.0, 20.0, 30.0])

        # Global nodes: R_z_90 @ [-1, 0, 0] = [0, -1, 0] + [10, 20, 30] = [10, 19, 30]
        #               R_z_90 @ [ 1, 0, 0] = [0,  1, 0] + [10, 20, 30] = [10, 21, 30]
        expected_global = np.array([[10.0, 19.0, 30.0], [10.0, 21.0, 30.0]])
        np.testing.assert_allclose(p.global_nodes(), expected_global)

        # Test tuple unpacking
        unpacked_nodes, unpacked_struts = p
        np.testing.assert_allclose(unpacked_nodes, expected_global)
        np.testing.assert_array_equal(unpacked_struts, struts)

        # Test center setter
        p.center = [0.0, 0.0, 0.0]
        np.testing.assert_allclose(p.global_nodes(), [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(p.local_nodes, nodes)

    def test_bridge_conversion_to_and_from_pam(self):
        # Generate legacy PAM particle
        pam = generate_truncated_tetrahedron_particle(size=10.0, center=(5.0, 5.0, 5.0), particle_id=3)

        # Convert to InterlinkedParticle via classmethod and instance method
        interlinked = InterlinkedParticle.from_pam(pam)
        assert interlinked.particle_id == 3
        np.testing.assert_allclose(interlinked.center, [5.0, 5.0, 5.0])
        np.testing.assert_allclose(interlinked.global_nodes(), pam.nodes)

        # Local nodes must be centered at [0, 0, 0]
        np.testing.assert_allclose(np.mean(interlinked.local_nodes, axis=0), [0.0, 0.0, 0.0], atol=1e-12)

        # Convert via pam.to_interlinked()
        interlinked2 = pam.to_interlinked()
        np.testing.assert_allclose(interlinked2.global_nodes(), pam.nodes)

        # Convert back to PAMParticle
        pam_back = interlinked.to_pam()
        assert pam_back.particle_id == 3
        np.testing.assert_allclose(pam_back.nodes, pam.nodes)
        np.testing.assert_allclose(pam_back.center, pam.center)
        np.testing.assert_array_equal(pam_back.struts, pam.struts)


class TestInterlinkedRegistryAndC6TTCell:
    def test_registry_lookup_and_listing(self):
        # C6TTCell should be registered under "c6tt" and "c-6-tt"
        cell_cls1 = InterlinkedRegistry.get("c6tt")
        cell_cls2 = InterlinkedRegistry.get("C-6-TT")
        assert cell_cls1 is cell_cls2
        assert issubclass(cell_cls1, C6TTCell)

        available = InterlinkedRegistry.list_cells()
        keys = [entry["key"] for entry in available]
        assert "c6tt" in keys or "c-6-tt" in keys

    def test_c6tt_protocol_conformance(self):
        cell = C6TTCell()
        assert isinstance(cell, InterlinkedCell)
        assert cell.name == "C-6-TT"
        assert cell.family == "pam_polyhedra"
        assert cell.parent_network == "pcu"
        assert cell.coordination_number == 6
        assert len(cell.neighbor_catenation_offsets) == 6
        assert (1, 0, 0) in cell.neighbor_catenation_offsets
        assert (-1, 0, 0) in cell.neighbor_catenation_offsets

    def test_c6tt_clearance_and_pitch_inversion(self):
        cell = C6TTCell()
        a0 = 12.7
        D = 1.25
        # Forward clearance: Delta = 0.13137 * 12.7 - 1.25 ≈ 0.4184 mm
        clr = cell.forward_clearance(unit_cell_pitch=a0, strut_diameter=D)
        assert np.isclose(clr, 0.13137 * 12.7 - 1.25, atol=1e-4)
        assert clr > 0.40

        # Inversion: resolve pitch for target 0.418399 mm
        solved_pitch = cell.resolve_pitch(target_clearance=clr, strut_diameter=D)
        assert np.isclose(solved_pitch, a0, atol=1e-4)

    def test_c6tt_instantiate_site_and_solidification(self):
        cell = C6TTCell(size_ratio=0.80)
        a0 = 10.0
        particles = cell.instantiate_site(
            grid_index=(1, 2, 3),
            site_origin=np.array([10.0, 20.0, 30.0]),
            cell_pitch=a0,
            id_start=100,
        )
        assert len(particles) == 1
        p = particles[0]
        assert p.particle_id == 100
        np.testing.assert_allclose(p.center, [10.0, 20.0, 30.0])
        assert p.cell_index == (1, 2, 3)
        assert p.local_nodes.shape == (12, 3)
        assert p.struts.shape == (18, 2)
        assert p.geometry_type == "TT"

        # Check that bridge to PAM solid generation creates a watertight mesh
        pam_obj = p.to_pam()
        meshes = pam_particles_to_meshes([pam_obj], strut_radius=0.40, circular_segments=16)
        assert len(meshes) == 1
        assert meshes[0].is_watertight
        assert meshes[0].volume > 0
