"""
Unit tests for Interlocking Auxetic and Kinematic Assembly Generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.generators.interlocking import (
    generate_interlocking_auxetic_sheet,
    combine_interlocking_meshes,
    verify_interlocking_clearance,
)


class TestInterlockingAuxetics:
    """Test discrete non-welded kinematic unit cell assemblies and clearance guarantees."""

    def test_reentrant_bowtie_mesh_count_and_watertight(self):
        nx, ny = 3, 2
        meshes = generate_interlocking_auxetic_sheet(
            dimensions=(nx, ny),
            cell_pitch=10.0,
            clearance_gap=0.40,
            cell_topology="reentrant_bowtie",
        )

        assert isinstance(meshes, list)
        assert len(meshes) == nx * ny
        for i, m in enumerate(meshes):
            assert isinstance(m, trimesh.Trimesh), f"Item {i} is not a Trimesh"
            assert m.is_watertight, f"Link {i} is not watertight"
            assert m.volume > 0.0

    def test_reentrant_bowtie_clearance_and_zero_collision(self):
        # 3x3 grid of interlocking 3D auxetic bowties
        meshes = generate_interlocking_auxetic_sheet(
            dimensions=(3, 3),
            cell_pitch=10.0,
            clearance_gap=0.40,
            cell_topology="reentrant_bowtie",
        )

        report = verify_interlocking_clearance(meshes, max_neighbor_distance=15.0)

        assert report["num_meshes"] == 9
        assert report["num_adjacent_pairs_checked"] >= 12
        # Zero boolean intersection volume between all adjacent links
        assert report["intersection_count"] == 0
        # Positive physical clearance
        assert report["min_clearance_mm"] > 0.30
        assert report["clearance_valid"] is True

    def test_combine_interlocking_meshes(self):
        meshes = generate_interlocking_auxetic_sheet(
            dimensions=(2, 2),
            cell_pitch=10.0,
            clearance_gap=0.40,
            cell_topology="reentrant_bowtie",
        )
        combined = combine_interlocking_meshes(meshes)
        assert isinstance(combined, trimesh.Trimesh)
        assert len(combined.faces) == sum(len(m.faces) for m in meshes)
        # Bounding box covers full 2x2 grid
        extents = combined.extents
        assert extents[0] > 10.0
        assert extents[1] > 10.0

    def test_interlocking_hook_array_topology(self):
        meshes = generate_interlocking_auxetic_sheet(
            dimensions=(2, 2),
            cell_pitch=12.75,
            clearance_gap=0.40,
            cell_topology="hook_array",
        )
        assert len(meshes) == 4
        for m in meshes:
            assert m.is_watertight

    def test_interlocking_chainmail_topologies(self):
        # European 4-in-1
        meshes_euro = generate_interlocking_auxetic_sheet(
            dimensions=(2, 2),
            cell_pitch=10.0,
            clearance_gap=0.30,
            cell_topology="european_ring",
        )
        assert len(meshes_euro) == 4
        for m in meshes_euro:
            assert m.is_watertight

        # Kusari
        meshes_kusari = generate_interlocking_auxetic_sheet(
            dimensions=(2, 2),
            cell_pitch=10.0,
            clearance_gap=0.30,
            cell_topology="kusari_ring",
        )
        assert len(meshes_kusari) > 0
        for m in meshes_kusari:
            assert m.is_watertight

    def test_reentrant_bowtie_3d_cube_clearance(self):
        # 2x2x2 volumetric grid (8 discrete, topologically interlocked links)
        meshes = generate_interlocking_auxetic_sheet(
            dimensions=(2, 2, 2),
            cell_pitch=10.0,
            clearance_gap=0.40,
            cell_topology="reentrant_bowtie",
        )
        assert len(meshes) == 8
        for m in meshes:
            assert m.is_watertight
            assert m.volume > 0.0

        report = verify_interlocking_clearance(meshes, max_neighbor_distance=15.0)
        assert report["num_meshes"] == 8
        assert report["num_adjacent_pairs_checked"] >= 12
        assert report["intersection_count"] == 0
        assert report["min_clearance_mm"] > 0.35
        assert report["clearance_valid"] is True

    def test_invalid_topology_raises_error(self):
        with pytest.raises(ValueError, match="Unknown cell_topology"):
            generate_interlocking_auxetic_sheet(
                dimensions=(2, 2),
                cell_pitch=10.0,
                clearance_gap=0.40,
                cell_topology="non_existent_topology",
            )
