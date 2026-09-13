"""
Unit tests for graphite.explicit.interlinked module.

Tests cover:
    - Ring geometry discretization, closure, and properties.
    - European 4-in-1 pattern, clearance verification, and Gauss linking numbers.
    - Japanese Kusari pattern, clearance verification, and Gauss linking numbers.
    - 2x2x2 Cube interlinked ring structure.
    - Volumetric Kusari 3D lattice.
    - SDF inset culling (verifying surviving rings are 100% intact with zero cuts).
    - Analytical Lipschitz gradient clamping.
    - Watertight Manifold3D print-in-place mesh generation.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from graphite.explicit.interlinked import (
    Ring,
    InterlinkedConfig,
    InterlinkedLatticeResult,
    generate_ring,
    generate_european_4in1_rings,
    generate_kusari_rings,
    generate_cubic_8ring,
    generate_volumetric_kusari_rings,
    generate_interlinked_lattice,
    check_ring_clearance,
    compute_pairwise_ring_clearances,
    gauss_linking_number,
    clamp_lipschitz_gradients,
    cull_rings_by_sdf,
    cull_rings_by_mesh,
    seed_planar_grid,
    seed_surface_conformal_frames,
)


class TestRingBasics:
    """Test fundamental Ring creation, discretization, and geometry."""

    def test_ring_creation_and_closure(self):
        center = np.array([10.0, 20.0, 30.0])
        normal = np.array([0.0, 0.0, 1.0])
        radius = 5.0
        wire_radius = 0.4
        num_segments = 24

        ring = generate_ring(
            center=center,
            normal=normal,
            radius=radius,
            wire_radius=wire_radius,
            num_segments=num_segments,
            tag="test_ring",
            cell_index=(0, 0, 0),
        )

        assert isinstance(ring, Ring)
        assert ring.nodes.shape == (num_segments, 3)
        assert ring.struts.shape == (num_segments, 2)
        assert ring.outer_radius == pytest.approx(radius + wire_radius)
        assert ring.inner_radius == pytest.approx(radius - wire_radius)
        assert ring.outer_diameter == pytest.approx(2.0 * (radius + wire_radius))
        assert ring.inner_diameter == pytest.approx(2.0 * (radius - wire_radius))

        # Check planarity: all nodes lie in the plane perpendicular to normal
        dots = np.dot(ring.nodes - center, normal)
        np.testing.assert_allclose(dots, 0.0, atol=1e-10)

        # Check circularity: all nodes at radius distance from center
        dists = np.linalg.norm(ring.nodes - center, axis=-1)
        np.testing.assert_allclose(dists, radius, atol=1e-10)

        # Check closure: each vertex has valency 2
        flat_struts = ring.struts.ravel()
        counts = np.bincount(flat_struts, minlength=num_segments)
        assert np.all(counts == 2)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            generate_ring([0, 0, 0], [0, 0, 1], radius=-1.0, wire_radius=0.4)
        with pytest.raises(ValueError):
            generate_ring([0, 0, 0], [0, 0, 1], radius=5.0, wire_radius=-0.1)
        with pytest.raises(ValueError):
            generate_ring([0, 0, 0], [0, 0, 1], radius=5.0, wire_radius=0.4, num_segments=2)


class TestEuropean4in1:
    """Test European 4-in-1 maille pattern, clearances, and topological linking."""

    def test_european_3x3_clearance_and_links(self):
        rings = generate_european_4in1_rings(
            grid_size=(3, 3, 1),
            pitch=10.0,
            radius_ratio=0.65,
            wire_radius=0.40,
            tilt_angle_deg=28.0,
            num_segments=24,
        )
        assert len(rings) == 9

        is_valid, min_clr, violations = check_ring_clearance(rings, min_clearance=0.30)
        assert is_valid, f"Clearance violations found: {violations}"
        assert min_clr >= 0.30

        # Verify topological links exist
        min_clr_eval, records = compute_pairwise_ring_clearances(rings)
        linked = [rec for rec in records if abs(rec["linking_number"]) > 0]
        assert len(linked) >= 12, f"Expected at least 12 linked pairs in 3x3, got {len(linked)}"

    def test_european_5x5_positive_clearance(self):
        rings = generate_european_4in1_rings(
            grid_size=(5, 5, 1),
            pitch=10.0,
            radius_ratio=0.65,
            wire_radius=0.40,
            tilt_angle_deg=28.0,
            num_segments=24,
        )
        assert len(rings) == 25
        is_valid, min_clr, violations = check_ring_clearance(rings, min_clearance=0.30)
        assert is_valid
        assert min_clr >= 0.30


class TestJapaneseKusari:
    """Test Japanese Kusari maille pattern, clearances, and topological linking."""

    def test_kusari_counts_and_clearances(self):
        nx, ny = 3, 3
        rings = generate_kusari_rings(
            grid_size=(nx, ny, 1),
            pitch=10.0,
            flat_radius=3.6,
            arch_radius=4.1,
            wire_radius=0.35,
            num_segments=24,
        )
        # Flat: nx * ny = 9. Arch X: (nx - 1) * ny = 6. Arch Y: nx * (ny - 1) = 6. Total: 21
        assert len(rings) == 9 + 6 + 6

        is_valid, min_clr, violations = check_ring_clearance(rings, min_clearance=0.30)
        assert is_valid, f"Kusari clearance violations: {violations}"
        assert min_clr >= 0.40

        # Verify linking: arch rings link with adjacent flat rings
        _, records = compute_pairwise_ring_clearances(rings)
        linked = [rec for rec in records if abs(rec["linking_number"]) > 0]
        # In a 3x3 grid, each X arch links 2 flat rings (6 * 2 = 12 links),
        # each Y arch links 2 flat rings (6 * 2 = 12 links). Total: 24 links
        assert len(linked) == 24


class TestCubicInterlinked:
    """Test 3D cube of 8 interlinked rings and volumetric Kusari."""

    def test_cubic_8ring_clearance_and_links(self):
        rings = generate_cubic_8ring(
            pitch=8.0,
            radius=5.10,
            wire_radius=0.35,
            num_segments=24,
        )
        assert len(rings) == 8

        is_valid, min_clr, violations = check_ring_clearance(rings, min_clearance=0.30)
        assert is_valid, f"Cube clearance violations: {violations}"
        assert min_clr >= 0.35

        # Check that edges form a connected linking cycle
        _, records = compute_pairwise_ring_clearances(rings)
        linked = [rec for rec in records if abs(rec["linking_number"]) > 0]
        assert len(linked) == 6

    def test_volumetric_kusari_3d(self):
        rings = generate_volumetric_kusari_rings(
            grid_size=(2, 2, 2),
            pitch=10.0,
            flat_radius=3.6,
            arch_radius=4.1,
            arch_z_radius=5.4,
            wire_radius=0.30,
            num_segments=24,
        )
        assert len(rings) > 8
        min_clr, records = compute_pairwise_ring_clearances(rings)
        assert min_clr > 0.0, f"Expected positive clearance, got {min_clr}"


class TestInsetCulling:
    """Test Inset Culling: guaranteeing 100% whole, complete, uncut rings."""

    def test_inset_culling_sdf(self):
        # Generate 5x5 grid of rings centered around (0, 0, 0)
        rings = generate_european_4in1_rings(
            grid_size=(5, 5, 1),
            pitch=10.0,
            origin=(-20.0, -20.0, 0.0),
        )

        # Spherical SDF centered at origin: SDF(x) = ||x|| - R_sphere
        # Negative inside sphere, positive outside.
        r_sphere = 25.0

        def sphere_sdf(pts: np.ndarray) -> np.ndarray:
            return np.linalg.norm(pts, axis=-1) - r_sphere

        margin = 0.5
        surviving, culled = cull_rings_by_sdf(rings, sphere_sdf, margin=margin)

        assert len(surviving) > 0
        assert len(culled) > 0
        assert len(surviving) + len(culled) == len(rings)

        # Every surviving ring must strictly satisfy SDF <= -(R_outer + margin)
        for ring in surviving:
            sdf_val = sphere_sdf(ring.center.reshape(1, 3))[0]
            assert sdf_val <= -(ring.outer_radius + margin)
            # Crucial: surviving rings must retain full original vertex and strut count!
            assert len(ring.nodes) == 24
            assert len(ring.struts) == 24

        # Every culled ring must violate the threshold
        for ring in culled:
            sdf_val = sphere_sdf(ring.center.reshape(1, 3))[0]
            assert sdf_val > -(ring.outer_radius + margin)

    def test_inset_culling_mesh(self):
        # Create a box boundary mesh [0, 50] x [0, 50] x [-20, 20]
        box = trimesh.creation.box(extents=[50.0, 50.0, 40.0])
        box.apply_translation([25.0, 25.0, 0.0])

        rings = generate_european_4in1_rings(
            grid_size=(6, 6, 1),
            pitch=10.0,
            origin=(0.0, 0.0, 0.0),
        )

        surviving, culled = cull_rings_by_mesh(rings, box, margin=0.5)
        assert len(surviving) > 0
        # Perimeter rings should be culled because they are too close to the box edges
        assert len(culled) > 0
        assert len(surviving) + len(culled) == len(rings)


class TestLipschitzClamp:
    """Test analytical Lipschitz gradient clamp."""

    def test_lipschitz_clamp_1d(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        # Extreme step: 1.0 -> 10.0 -> 1.0
        values = np.array([1.0, 10.0, 1.0])
        max_grad = 2.0  # Max change per unit distance is 2.0

        clamped = clamp_lipschitz_gradients(values, coords, max_gradient=max_grad)

        # Difference between adjacent points must be <= max_grad * 1.0 = 2.0
        diffs = np.abs(np.diff(clamped))
        assert np.all(diffs <= max_grad + 1e-5)
        # Peak at center should be reduced
        assert clamped[1] < 10.0


class TestGeometryGeneration:
    """Test watertight Manifold3D print-in-place mesh generation."""

    def test_european_lattice_mesh(self):
        cfg = InterlinkedConfig(
            pattern="european_4in1",
            grid_size=(2, 2, 1),
            pitch=10.0,
            radius_ratio=0.65,
            wire_radius=0.40,
            num_ring_segments=16,
            add_spheres=True,
        )
        res = generate_interlinked_lattice(cfg)

        assert isinstance(res, InterlinkedLatticeResult)
        assert res.num_rings == 4
        assert res.num_nodes == 4 * 16
        assert res.num_struts == 4 * 16
        assert res.mesh.is_watertight
        assert res.volume > 0.0
        assert res.clearance_valid
        assert res.bounds.shape == (2, 3)

    def test_kusari_lattice_mesh(self):
        cfg = InterlinkedConfig(
            pattern="kusari",
            grid_size=(2, 2, 1),
            pitch=10.0,
            flat_radius=3.6,
            arch_radius=4.1,
            wire_radius=0.35,
            num_ring_segments=16,
            add_spheres=True,
        )
        res = generate_interlinked_lattice(cfg)

        # 4 flat + 2 arch_x + 2 arch_y = 8 rings
        assert res.num_rings == 8
        assert res.mesh.is_watertight
        assert res.volume > 0.0
        assert res.clearance_valid

    def test_cube_lattice_mesh(self):
        cfg = InterlinkedConfig(
            pattern="cubic_8ring",
            pitch=8.0,
            wire_radius=0.35,
            num_ring_segments=16,
            add_spheres=True,
        )
        res = generate_interlinked_lattice(cfg)

        assert res.num_rings == 8
        assert res.mesh.is_watertight
        assert res.volume > 0.0
        assert res.clearance_valid
