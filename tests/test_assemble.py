"""
Unit tests for graphite.explicit.interlinked.assemble multi-zone engine (Phase 3).
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pytest
import trimesh

from graphite.explicit.interlinked import (
    ZoneSpec,
    InterfaceSpec,
    assemble,
    InterlinkedConfig,
    generate_interlinked_lattice,
    C6TTCell,
    D4CubeCell,
    D4TetCell,
    CustomWireframeCell,
)


class TestMultiZoneAssemble:
    """Test deterministic multi-zone assembly and permissive diagnosis."""

    def test_assemble_two_zones_basic(self):
        """Assemble two adjacent C-6-TT blocks along [1, 0, 0]."""
        zone_a = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=10.0,
            grid_size=(2, 2, 1),
            name="Zone_A",
            wire_radius=0.40,
        )
        zone_b = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=10.0,
            grid_size=(2, 2, 1),
            name="Zone_B",
            wire_radius=0.40,
        )
        iface = InterfaceSpec(
            normal=np.array([1.0, 0.0, 0.0]),
            gap=10.0,
            shift=np.zeros(3),
            twist_deg=0.0,
        )

        res = assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.40)
        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert len(res.particles) == 8  # 4 particles in A + 4 in B
        assert res.metadata["num_zones"] == 2
        assert res.metadata["particles_per_zone"] == [4, 4]
        assert "Zone_0" in res.metadata["status_counts"]
        assert "Zone_1" in res.metadata["status_counts"]

    def test_assemble_custom_interface_params(self):
        """Assemble with habit plane '111', gap, and 60-degree azimuthal twist."""
        zone_a = ZoneSpec(
            cell="d4cube",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="111",
            name="Cube_111",
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="111",
            name="Tet_111",
        )
        iface = InterfaceSpec(
            normal="111",
            gap=4.33,
            shift=np.array([0.0, 0.5, 0.0]),
            twist_deg=60.0,
        )

        res = assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.35)
        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert len(res.particles) == 30  # 15 per zone on 111 supercell
        # Check that particles in Zone B received the twist rotation and translation
        p_b = res.particles[15]
        assert p_b.metadata["zone_index"] == 1
        assert p_b.metadata["zone_name"] == "Tet_111"

    def test_permissive_behavior_infeasible_pair(self):
        """Permissive standard: severe physical collision builds watertight mesh and diagnoses WARN, never raises."""
        zone_a = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=10.0,
            grid_size=(1, 1, 1),
            name="A",
            wire_radius=0.50,
        )
        zone_b = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=10.0,
            grid_size=(1, 1, 1),
            name="B",
            wire_radius=0.50,
        )
        # Zero gap -> overlapping particles at the exact same location
        iface = InterfaceSpec(normal="001", gap=0.0)

        res = assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.50, strict=False)
        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert res.clearance_valid is False
        assert res.min_clearance < 0.0
        assert res.metadata["status_counts"]["colliding"] > 0

    def test_strict_mode_raises(self):
        """In CI or validation pipelines, strict=True converts clearance failure into ValueError."""
        zone_a = ZoneSpec(cell="c6tt", network="pcu", pitch=10.0, grid_size=(1, 1, 1), wire_radius=0.50)
        zone_b = ZoneSpec(cell="c6tt", network="pcu", pitch=10.0, grid_size=(1, 1, 1), wire_radius=0.50)
        iface = InterfaceSpec(normal="001", gap=0.0)

        with pytest.raises(ValueError, match="Multi-zone clearance violation under strict=True"):
            assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.50, strict=True)

    def test_transition_to_cell_routes_through_assemble(self):
        """Verify generator.transition_to_cell executes through assemble without EpitaxialIncompatibilityError."""
        cfg = InterlinkedConfig(
            cell="d4cube",
            transition_to_cell="d4tet",
            pitch=10.0,
            grid_size=(1, 1, 1),
            habit_plane="111",
            wire_radius=0.35,
            strict=False,
        )
        res = generate_interlinked_lattice(cfg)
        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert len(res.particles) == 30
        assert res.metadata["num_zones"] == 2
        assert res.metadata["particles_per_zone"] == [15, 15]

    def test_color_coded_3mf_export(self, tmp_path):
        """Verify 3MF export contains distinct zone and status color group assignments."""
        zone_a = ZoneSpec(cell="c6tt", network="pcu", pitch=10.0, grid_size=(1, 1, 1), name="ZoneA")
        zone_b = ZoneSpec(cell="d4tet", network="dia", pitch=10.0, grid_size=(1, 1, 1), name="ZoneB")
        iface = InterfaceSpec(normal="001", gap=6.0)

        res = assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.40)
        out_3mf = tmp_path / "assembly_test.3mf"
        res.export_3mf(out_3mf, color_by_status=True)
        assert out_3mf.exists()

        # Verify 3MF archive XML structure
        with zipfile.ZipFile(out_3mf, "r") as zf:
            xml_str = zf.read("3D/3dmodel.model").decode("utf-8")
            assert "m:colorgroup" in xml_str
            assert "pid=" in xml_str
            assert "pindex=" in xml_str

        # Load into trimesh
        scene = trimesh.load(str(out_3mf), file_type="3mf")
        assert scene is not None

    def test_continuous_mode_111_interlinks(self):
        """Verify continuous mode on {111} achieves true cross-boundary catenation with >= 150 um clearance."""
        zone_a = ZoneSpec(
            cell="d4cube",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="111",
            name="ZoneA_Cube",
            wire_radius=0.35,
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="111",
            name="ZoneB_Tet",
            wire_radius=0.35,
        )
        iface = InterfaceSpec(normal="111", twist_deg=60.0)

        res = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="continuous",
            strict=True,
        )

        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert res.clearance_valid is True
        assert res.min_clearance >= 0.150
        assert res.metadata["boundary_linked_count"] == 3
        assert res.metadata["inter_zone_linked"] is True
        assert res.metadata["particles_per_zone"] == [15, 12]

    def test_inter_zone_unlinked_strict_raises(self):
        """Strict mode raises ValueError when zones are disjoint in space (0 boundary links)."""
        zone_a = ZoneSpec(cell="c6tt", network="pcu", pitch=10.0, grid_size=(1, 1, 1), name="A")
        zone_b = ZoneSpec(cell="c6tt", network="pcu", pitch=10.0, grid_size=(1, 1, 1), name="B")
        # 50 mm gap -> completely disjoint in air
        iface = InterfaceSpec(normal="001", gap=50.0)

        with pytest.raises(ValueError, match="Multi-zone inter-zone catenation violation"):
            assemble(zones=[zone_a, zone_b], interfaces=[iface], wire_radius=0.35, strict=True)

    def test_c6tt_to_d4tet_100_interlinks(self):
        """Verify C-6-TT to D-4-TET on {100} achieves true boundary catenation and >= 150 um clearance."""
        a_sc = 9.10
        a_dia = np.sqrt(2.0) * a_sc
        zone_a = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=a_sc,
            grid_size=(2, 2, 1),
            frame="100",
            name="ZoneA_C6TT",
            wire_radius=0.35,
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=a_dia,
            grid_size=(1, 1, 1),
            frame="100",
            name="ZoneB_D4Tet",
            wire_radius=0.35,
        )
        iface = InterfaceSpec(
            normal="001",
            gap=7.0,
            twist_deg=45.0,
        )

        res = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="discrete",
            strict=True,
        )

        assert res.mesh is not None
        assert res.mesh.is_watertight
        assert res.clearance_valid is True
        assert res.min_clearance >= 0.150
        assert res.metadata["boundary_linked_count"] == 4
        assert res.metadata["inter_zone_linked"] is True

    def test_insufficient_linking_detection_and_culling(self):
        """
        Phase 5 Quality Gate:
        1. Without culling (cull_dangling=False), detects orphaned boundary link (inter >= 1, intra == 0).
        2. With culling (cull_dangling=True), eliminates orphaned link, yielding 3 fully anchored boundary links.
        """
        a_sc = 9.10
        a_dia = np.sqrt(2.0) * a_sc
        zone_a = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=a_sc,
            grid_size=(2, 2, 1),
            frame="100",
            name="ZoneA_C6TT",
            wire_radius=0.35,
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=a_dia,
            grid_size=(1, 1, 1),
            frame="100",
            name="ZoneB_D4Tet",
            wire_radius=0.35,
        )
        iface = InterfaceSpec(
            normal="001",
            gap=7.0,
            twist_deg=45.0,
        )

        # 1. Unculled run: flags orphaned boundary particle
        res_unculled = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="discrete",
            strict=False,
            cull_dangling=False,
        )
        insuf_rep = res_unculled.metadata["insufficient_linking_report"]
        assert insuf_rep.is_valid is False
        assert insuf_rep.num_orphaned_boundary_particles == 1
        assert "Zone_1_orphaned" in res_unculled.metadata["status_counts"]
        assert res_unculled.metadata["status_counts"]["Zone_1_orphaned"] == 1

        # 2. Culled run: eliminates orphaned particle, all links anchored
        res_culled = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="discrete",
            strict=True,
            cull_dangling=True,
        )
        assert res_culled.mesh.is_watertight
        assert res_culled.clearance_valid is True
        assert res_culled.min_clearance >= 0.150
        assert res_culled.metadata["insufficient_linking_report"].is_valid is True
        assert res_culled.metadata["orphaned_boundary_count"] == 0
        assert res_culled.metadata["boundary_linked_count"] == 3
        assert "Zone_1_orphaned" not in res_culled.metadata["status_counts"]

    def test_connectivity_single_component_c6tt_d4tet(self):
        """Verify C-6-TT to D-4-TET under cull_dangling forms a single connected assembly."""
        a_sc = 9.10
        a_dia = np.sqrt(2.0) * a_sc
        zone_a = ZoneSpec(
            cell="c6tt",
            network="pcu",
            pitch=a_sc,
            grid_size=(2, 2, 1),
            frame="100",
            name="ZoneA_C6TT",
            wire_radius=0.35,
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=a_dia,
            grid_size=(1, 1, 1),
            frame="100",
            name="ZoneB_D4Tet",
            wire_radius=0.35,
        )
        iface = InterfaceSpec(
            normal="001",
            gap=7.0,
            twist_deg=45.0,
        )

        res = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="discrete",
            strict=True,
            cull_dangling=True,
        )
        conn_rep = res.metadata["connectivity_report"]
        assert conn_rep.is_single_component is True
        assert conn_rep.num_components == 1
        assert res.metadata["is_single_component"] is True

    def test_assemble_tags_zone_detached_particles(self):
        """Verify that particles disconnected from the largest component are tagged as Zone_k_detached."""
        # Two disconnected zones (e.g. Coupon 2 configuration on incompatible habit plane)
        zone_a = ZoneSpec(
            cell="d4cube",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="100",
            name="ZoneA_D4Cube",
            wire_radius=0.35,
        )
        zone_b = ZoneSpec(
            cell="d4tet",
            network="dia",
            pitch=10.0,
            grid_size=(1, 1, 1),
            frame="100",
            name="ZoneB_D4Tet",
            wire_radius=0.35,
        )
        iface = InterfaceSpec(normal="100", gap=5.0)

        res = assemble(
            zones=[zone_a, zone_b],
            interfaces=[iface],
            wire_radius=0.35,
            min_clearance=0.15,
            mode="discrete",
            strict=False,
        )

        assert res.metadata["num_components"] > 1
        status_counts = res.metadata["status_counts"]
        # In Coupon 2, Zone B is unlinked to Zone A, so its particles must be tagged Zone_1_detached
        assert "Zone_1_detached" in status_counts
        assert status_counts["Zone_1_detached"] > 0
        detached_particles = [p for p in res.particles if p.metadata.get("status") == "Zone_1_detached"]
        assert len(detached_particles) == status_counts["Zone_1_detached"]



