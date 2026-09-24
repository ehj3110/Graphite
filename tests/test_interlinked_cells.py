"""
Unit tests for the modular InterlinkedCell implementations (Phase 2):
- D4TetCell (Diamond Tetrahedron, n=4, bipartite A/B dual basis)
- J4OctCell (Square-Planar Octahedron Cross, n=4, 45-deg relative twist)
- European4in1Cell (2D European 4-in-1 chainmail, n=4, alternating tilt)
- JapaneseKusariCell (2D Japanese Kusari chainmail, n=4, flat + arch rings)
- NasaSpaceFabricCell (NASA JPL Space Fabric, n=6, spiral hook arms)
"""

from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    InterlinkedRegistry,
    InterlinkedCell,
    D4TetCell,
    J4OctCell,
    European4in1Cell,
    JapaneseKusariCell,
    NasaSpaceFabricCell,
    pam_particles_to_meshes,
    get_available_support_recipes,
    export_seed_cell_stl,
    apply_support_recipe,
    create_custom_supported_particle,
    ParticleGeometry,
    InterlinkedParticle,
)


class TestD4TetCell:
    def test_registry_lookup(self):
        cls1 = InterlinkedRegistry.get("d4tet")
        cls2 = InterlinkedRegistry.get("d-4-tet")
        assert cls1 is cls2
        assert issubclass(cls1, D4TetCell)

    def test_protocol_attributes(self):
        cell = D4TetCell()
        assert isinstance(cell, InterlinkedCell)
        assert cell.name == "D-4-TET"
        assert cell.family == "pam_polyhedra"
        assert cell.parent_network == "dia"
        assert cell.coordination_number == 4
        assert len(cell.neighbor_catenation_offsets) == 4
        assert len(cell.basis_particles) == 2

    def test_bipartite_crystallographic_dual_basis(self):
        cell = D4TetCell()
        bp_a, bp_b = cell.basis_particles
        assert bp_a.sublattice_id == "A"
        assert bp_b.sublattice_id == "B"
        np.testing.assert_allclose(bp_a.fractional_offset, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(bp_b.fractional_offset, [0.25, 0.25, 0.25])

        # Dual inversion: nodes of B should be -nodes of A
        np.testing.assert_allclose(bp_b.geometry.nodes, -bp_a.geometry.nodes)

    def test_clearance_and_inversion(self):
        cell = D4TetCell()
        a_conv = 11.55  # mm conventional cell size
        D_strut = 0.75  # mm strut diameter
        clr = cell.forward_clearance(unit_cell_pitch=a_conv, strut_diameter=D_strut)
        # Delta = 0.235 * 11.55 - 0.75 = 2.714 - 0.75 ≈ 1.964 mm
        assert clr > 0.40

        # Inversion
        solved = cell.resolve_pitch(target_clearance=clr, strut_diameter=D_strut)
        assert np.isclose(solved, a_conv, atol=1e-4)

    def test_instantiate_site(self):
        cell = D4TetCell()
        parts = cell.instantiate_site(
            grid_index=(0, 0, 0),
            site_origin=np.zeros(3),
            cell_pitch=12.0,
            id_start=0,
        )
        assert len(parts) == 2
        pa, pb = parts
        assert pa.sublattice_id == "A"
        assert pb.sublattice_id == "B"
        np.testing.assert_allclose(pa.center, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pb.center, [3.0, 3.0, 3.0])  # 0.25 * 12.0


class TestJ4OctCell:
    def test_registry_lookup(self):
        cls1 = InterlinkedRegistry.get("j4oct")
        cls2 = InterlinkedRegistry.get("j-4-oct")
        assert cls1 is cls2
        assert issubclass(cls1, J4OctCell)

    def test_protocol_attributes(self):
        cell = J4OctCell()
        assert isinstance(cell, InterlinkedCell)
        assert cell.name == "J-4-OCT"
        assert cell.coordination_number == 4
        assert len(cell.neighbor_catenation_offsets) == 4

    def test_checkerboard_orientation(self):
        cell = J4OctCell()
        # Even parity: identity
        parts_even = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0, id_start=0)
        np.testing.assert_allclose(parts_even[0].rotation, np.eye(3), atol=1e-8)

        # Odd parity along X: 45-deg rotation about X
        parts_odd_x = cell.instantiate_site((1, 0, 0), np.array([10.0, 0.0, 0.0]), 10.0, id_start=1)
        c45 = np.cos(np.pi / 4.0)
        s45 = np.sin(np.pi / 4.0)
        expected_Rx = np.array([[1.0, 0.0, 0.0], [0.0, c45, -s45], [0.0, s45, c45]])
        np.testing.assert_allclose(parts_odd_x[0].rotation, expected_Rx, atol=1e-8)

        # Odd parity along Y: 45-deg rotation about Y
        parts_odd_y = cell.instantiate_site((0, 1, 0), np.array([0.0, 10.0, 0.0]), 10.0, id_start=2)
        expected_Ry = np.array([[c45, 0.0, s45], [0.0, 1.0, 0.0], [-s45, 0.0, c45]])
        np.testing.assert_allclose(parts_odd_y[0].rotation, expected_Ry, atol=1e-8)


class TestEuropean4in1Cell:
    def test_registry_lookup(self):
        cls1 = InterlinkedRegistry.get("european_4in1")
        cls2 = InterlinkedRegistry.get("european-4in1")
        assert cls1 is cls2
        assert issubclass(cls1, European4in1Cell)

    def test_alternating_tilt(self):
        cell = European4in1Cell(tilt_angle_deg=28.0)
        # (0, 0, 0) has even parity -> positive tilt about [1, 1, 0]/sqrt(2)
        p00 = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0, id_start=0)[0]
        # (1, 0, 0) has odd parity -> negative tilt about [1, 1, 0]/sqrt(2)
        p10 = cell.instantiate_site((1, 0, 0), np.array([10.0, 0.0, 0.0]), 10.0, id_start=1)[0]

        k = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        theta = np.radians(28.0)
        expected_R_pos = np.eye(3)*np.cos(theta) + K*np.sin(theta) + np.outer(k, k)*(1 - np.cos(theta))
        expected_R_neg = np.eye(3)*np.cos(-theta) + K*np.sin(-theta) + np.outer(k, k)*(1 - np.cos(-theta))

        np.testing.assert_allclose(p00.rotation, expected_R_pos, atol=1e-8)
        np.testing.assert_allclose(p10.rotation, expected_R_neg, atol=1e-8)

    def test_clearance_and_inversion(self):
        cell = European4in1Cell(radius_ratio=0.65, tilt_angle_deg=28.0)
        pitch = 10.0
        strut_d = 0.80
        clr = cell.forward_clearance(pitch, strut_d)
        # Delta = 0.1142 * 10.0 - 0.80 = 1.142 - 0.80 = 0.342 mm
        assert clr > 0.30

        solved_pitch = cell.resolve_pitch(target_clearance=clr, strut_diameter=strut_d)
        assert np.isclose(solved_pitch, pitch, atol=1e-4)

    def test_multi_ring_mesh_clearance_no_collision(self):
        # 3x3 array of European 4-in-1 must have zero collisions
        from graphite.explicit.interlinked.clearance import check_ring_clearance
        from graphite.explicit.interlinked.patterns import Ring

        cell = European4in1Cell(radius_ratio=0.65, tilt_angle_deg=28.0)
        pitch = 10.0
        rings = []
        for i in range(3):
            for j in range(3):
                p = cell.instantiate_site((i, j, 0), np.array([i * pitch, j * pitch, 0.0]), pitch)[0]
                n = p.rotation @ [0.0, 0.0, 1.0]
                rings.append(Ring(
                    center=p.center,
                    normal=n,
                    radius=p.geometry.bounding_radius,
                    wire_radius=0.40,
                    nodes=p.global_nodes(),
                    struts=p.struts,
                ))
        valid, min_clr, viol = check_ring_clearance(rings, min_clearance=0.30)
        assert valid
        assert min_clr >= 0.30
        assert len(viol) == 0


class TestJapaneseKusariCell:
    def test_registry_lookup(self):
        cls1 = InterlinkedRegistry.get("japanese_kusari")
        cls2 = InterlinkedRegistry.get("kusari")
        assert cls1 is cls2
        assert issubclass(cls1, JapaneseKusariCell)

    def test_orthogonal_tri_particle_basis(self):
        cell = JapaneseKusariCell()
        assert len(cell.basis_particles) == 3
        tags = [bp.sublattice_id for bp in cell.basis_particles]
        assert "flat" in tags
        assert "arch_x" in tags
        assert "arch_y" in tags

        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0, id_start=0)
        assert len(parts) == 3
        # Flat ring is in XY plane (normal along Z)
        p_flat = parts[0]
        assert np.allclose(p_flat.global_nodes()[:, 2], 0.0)

        # Arch-X ring is in XZ plane (normal along Y)
        p_arch_x = parts[1]
        assert np.allclose(p_arch_x.global_nodes()[:, 1], 0.0)

        # Arch-Y ring is in YZ plane (normal along X)
        p_arch_y = parts[2]
        assert np.allclose(p_arch_y.global_nodes()[:, 0], 0.0)


class TestNasaSpaceFabricCell:
    def test_registry_lookup(self):
        cls1 = InterlinkedRegistry.get("nasa_hexagon")
        cls2 = InterlinkedRegistry.get("nasa_space_fabric")
        assert cls1 is cls2
        assert issubclass(cls1, NasaSpaceFabricCell)

    def test_attributes(self):
        cell = NasaSpaceFabricCell()
        assert cell.coordination_number == 6
        assert len(cell.neighbor_catenation_offsets) == 6
        assert cell.parent_network == "hex_2d"

        parts = cell.instantiate_site((0, 0, 0), np.zeros(3), 12.75, id_start=0)
        assert len(parts) == 1
        p = parts[0]
        assert p.particle_id == 0
        assert p.geometry_type == "nasa_hexagon_proxy"


class TestPAMSupportRecipes:
    def test_get_available_support_recipes(self):
        recipes = get_available_support_recipes("c6tt")
        assert isinstance(recipes, list)
        assert len(recipes) >= 3
        assert "None (Unprinted / Free)" in recipes
        assert "Vertical Pin Bridges" in recipes
        assert "Base Plate Breakaway Pins" in recipes

    def test_export_seed_cell_stl(self, tmp_path):
        out_stl = tmp_path / "c6tt_seed.stl"
        res_path = export_seed_cell_stl("c6tt", pitch=10.0, wire_radius=0.4, output_path=out_stl)
        assert res_path == out_stl
        assert out_stl.is_file()
        assert out_stl.stat().st_size > 0

    def test_apply_support_recipe_vertical_pin_bridges(self):
        cell = InterlinkedRegistry.get("c6tt")()
        pts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        proto = pts[0].geometry
        orig_node_count = len(proto.nodes)
        orig_strut_count = len(proto.struts)

        supported = apply_support_recipe(proto, "Vertical Pin Bridges", wire_radius=0.4)
        assert isinstance(supported, ParticleGeometry)
        assert len(supported.nodes) > orig_node_count
        assert len(supported.struts) > orig_strut_count
        assert supported.metadata.get("pin_diameter") == 0.25
        assert supported.metadata.get("pin_radius") == 0.125
        assert len(supported.metadata.get("pin_struts", [])) > 0
        assert "solid_mesh" in supported.metadata
        solid = supported.metadata["solid_mesh"]
        assert solid.is_watertight

    def test_apply_support_recipe_none_and_unrecognized(self):
        cell = InterlinkedRegistry.get("c6tt")()
        pts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        proto = pts[0].geometry

        # "None" returns unchanged
        res_none = apply_support_recipe(proto, "None (Unprinted / Free)", wire_radius=0.4)
        assert len(res_none.nodes) == len(proto.nodes)
        assert len(res_none.struts) == len(proto.struts)

        # Unrecognized recipe returns unchanged
        res_unknown = apply_support_recipe(proto, "Nonexistent Recipe", wire_radius=0.4)
        assert len(res_unknown.nodes) == len(proto.nodes)
        assert len(res_unknown.struts) == len(proto.struts)

    def test_create_custom_supported_particle(self):
        import trimesh
        from graphite.explicit.interlinked.generator import _particles_to_combined_mesh

        cell = InterlinkedRegistry.get("c6tt")()
        pts = cell.instantiate_site((0, 0, 0), np.zeros(3), 10.0)
        base_p = pts[0]

        custom_mesh = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
        custom_p = create_custom_supported_particle(custom_mesh, base_p)

        assert isinstance(custom_p, InterlinkedParticle)
        assert custom_p.particle_id == base_p.particle_id
        assert custom_p.geometry.metadata.get("solid_mesh") is custom_mesh

        # Verify seamless instancing into combined mesh
        combined = _particles_to_combined_mesh([custom_p], wire_radius=0.4)
        assert len(combined.vertices) == len(custom_mesh.vertices)
        assert len(combined.faces) == len(custom_mesh.faces)
        assert combined.is_watertight


class TestCADAutoRepair:
    def test_load_cad_mesh_primitive_clean(self):
        from graphite.ui.surface_preview import load_cad_mesh
        mesh = load_cad_mesh("primitive", primitive_shape="Cube", size=15.0)
        assert mesh.is_watertight
        assert len(mesh.faces) == 12

    def test_load_cad_mesh_broken_auto_repair(self, tmp_path):
        import trimesh
        from graphite.ui.surface_preview import load_cad_mesh

        box = trimesh.creation.box()
        raw_verts = box.vertices[box.faces].reshape(-1, 3)
        raw_faces = np.arange(len(box.faces) * 3).reshape(-1, 3)
        # Invert winding of first triangle
        raw_faces[0] = raw_faces[0, [0, 2, 1]]
        broken = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
        assert not broken.is_watertight

        bad_stl = tmp_path / "broken.stl"
        broken.export(str(bad_stl))

        repaired = load_cad_mesh("file", file_path=bad_stl)
        assert repaired.is_watertight
        assert len(repaired.vertices) == 8
