"""
Unit Tests for Epitaxially-Constrained Continuous Basis Morphing (Idea 3)
and Multi-Zone Buffer Superlattices (Idea 2).
"""

import numpy as np
import pytest

from graphite.explicit.interlinked import (
    InterlinkedConfig,
    generate_interlinked_lattice,
    build_morphing_tetrahedron_geometry,
    EpitaxialMorphConfig,
    EpitaxialBasisMorpher,
    generate_epitaxial_morph_lattice,
    LatticeZoneConfig,
    SuperlatticeConfig,
    generate_epitaxial_superlattice,
    LinearThicknessGradient,
)


def test_build_morphing_tetrahedron_geometry_c6():
    """Verify tau = 1/3 produces Archimedean Truncated Tetrahedron (12 verts, 18 struts)."""
    geom = build_morphing_tetrahedron_geometry(size=8.0, tau=1.0 / 3.0)
    assert len(geom.nodes) == 12
    assert len(geom.struts) == 18
    assert geom.geometry_type == "TT_MORPH"
    assert np.isclose(geom.metadata["tau"], 1.0 / 3.0)


def test_build_morphing_tetrahedron_geometry_d4():
    """Verify tau = 0 produces Platonic Regular Tetrahedron (4 verts, 6 struts)."""
    geom = build_morphing_tetrahedron_geometry(size=8.0, tau=0.0)
    assert len(geom.nodes) == 4
    assert len(geom.struts) == 6
    assert geom.geometry_type == "TET"
    assert np.isclose(geom.metadata["tau"], 0.0)


def test_build_morphing_tetrahedron_geometry_intermediate():
    """Verify intermediate tau values produce valid 12-vert, 18-strut polyhedra."""
    for tau in [0.25, 0.1667, 0.0833]:
        geom = build_morphing_tetrahedron_geometry(size=8.0, tau=tau)
        assert len(geom.nodes) == 12
        assert len(geom.struts) == 18
        assert geom.bounding_radius > 0.0


def test_epitaxial_basis_morpher_states():
    """Verify EpitaxialBasisMorpher computes correct geodesic states across t in [0, 1]."""
    morpher = EpitaxialBasisMorpher(pitch_a=10.0, pitch_b=14.142)

    # State at t = 0 (Pure Phase A)
    s0 = morpher.evaluate_state(0.0)
    assert np.isclose(s0["tau"], 1.0 / 3.0)
    assert np.isclose(s0["theta_deg"], 0.0)
    assert np.allclose(s0["rotation_matrix"], np.eye(3))

    # State at t = 0.5 (Hybrid)
    s_mid = morpher.evaluate_state(0.5)
    assert np.isclose(s_mid["tau"], 1.0 / 6.0)
    assert np.isclose(s_mid["theta_deg"], 22.5)

    # State at t = 1.0 (Phase B orientation)
    s1 = morpher.evaluate_state(1.0)
    assert np.isclose(s1["tau"], 0.0)
    assert np.isclose(s1["theta_deg"], 45.0)


def test_generate_epitaxial_morph_lattice_basic():
    """Verify continuous morph assembly generation with 1 intermediate morph layer."""
    cfg = EpitaxialMorphConfig(
        cell_a="c6tt",
        cell_b="d4tet",
        grid_size_a=(1, 2, 2),
        grid_size_b=(1, 2, 2),
        num_morph_layers=1,
        pitch=10.0,
        wire_radius=0.35,
        min_clearance=0.150,
    )
    res = generate_epitaxial_morph_lattice(cfg, check_clearance=True)
    assert len(res.particles) > 0
    assert res.clearance_valid is True
    assert res.min_clearance >= 0.150
    assert res.volume > 0.0
    assert res.metadata["pipeline"] == "epitaxial_continuous_morph"


def test_generate_epitaxial_morph_lattice_multilayers():
    """Verify continuous morph assembly generation with 2 intermediate morph layers."""
    cfg = EpitaxialMorphConfig(
        cell_a="c6tt",
        cell_b="d4tet",
        grid_size_a=(1, 2, 2),
        grid_size_b=(1, 2, 2),
        num_morph_layers=2,
        pitch=10.0,
        wire_radius=0.35,
        min_clearance=0.150,
    )
    res = generate_epitaxial_morph_lattice(cfg, check_clearance=True)
    assert res.clearance_valid is True
    assert res.min_clearance >= 0.150
    assert res.volume > 0.0


def test_morph_lattice_with_thickness_gradient():
    """Verify continuous morph assembly combined with spatial thickness grading."""
    cfg = EpitaxialMorphConfig(
        cell_a="c6tt",
        cell_b="d4tet",
        grid_size_a=(1, 2, 2),
        grid_size_b=(1, 2, 2),
        num_morph_layers=1,
        pitch=10.0,
        wire_radius=0.35,
        min_clearance=0.100,
        thickness_gradient=LinearThicknessGradient(axis="x", r_start=0.40, r_end=0.25),
    )
    res = generate_epitaxial_morph_lattice(cfg, check_clearance=False)
    radii = [p.effective_wire_radius() for p in res.particles]
    assert min(radii) < max(radii)
    assert np.isclose(max(radii), 0.40, atol=0.05)
    assert np.isclose(min(radii), 0.25, atol=0.05)


def test_superlattice_basic_two_zone():
    """Verify multi-zone buffer superlattice generation between C-6-TT and D-4-TET."""
    cfg = SuperlatticeConfig(
        zones=[
            LatticeZoneConfig(cell="c6tt", num_layers=1, pitch=10.0),
            LatticeZoneConfig(cell="d4tet", num_layers=1, pitch=14.142),
        ],
        grid_transverse=(2, 2),
        wire_radius=0.35,
        min_clearance=0.150,
    )
    res = generate_epitaxial_superlattice(cfg, check_clearance=True)
    assert res.metadata["num_zones"] == 2
    assert res.clearance_valid is True
    assert res.min_clearance >= 0.150
    assert res.volume > 0.0


def test_superlattice_with_thickness_gradient():
    """Verify multi-zone buffer superlattice combined with spatial thickness grading."""
    cfg = SuperlatticeConfig(
        zones=[
            LatticeZoneConfig(cell="c6tt", num_layers=1, pitch=10.0),
            LatticeZoneConfig(cell="d4tet", num_layers=1, pitch=14.142),
        ],
        grid_transverse=(2, 2),
        wire_radius=0.35,
        min_clearance=0.100,
        thickness_gradient=LinearThicknessGradient(axis="x", r_start=0.45, r_end=0.25),
    )
    res = generate_epitaxial_superlattice(cfg, check_clearance=False)
    radii = [p.effective_wire_radius() for p in res.particles]
    assert min(radii) < max(radii)


def test_generator_dispatch_morph_and_superlattice():
    """Verify generate_interlinked_lattice dispatches properly to morph and superlattice."""
    # 1. Morph dispatch via InterlinkedConfig
    cfg_morph = InterlinkedConfig(
        cell="c6tt",
        transition_to_cell="d4tet",
        transition_mode="morph",
        num_morph_layers=1,
        grid_size=(1, 2, 2),
        pitch=10.0,
        wire_radius=0.35,
        min_clearance=0.150,
    )
    res_morph = generate_interlinked_lattice(cfg_morph, check_clearance=True)
    assert res_morph.metadata["pipeline"] == "epitaxial_continuous_morph"
    assert res_morph.clearance_valid is True

    # 2. Superlattice dispatch via InterlinkedConfig.zones
    cfg_sl = InterlinkedConfig(
        zones=[
            LatticeZoneConfig(cell="c6tt", num_layers=1, pitch=10.0),
            LatticeZoneConfig(cell="d4tet", num_layers=1, pitch=14.142),
        ],
        grid_size=(1, 2, 2),
        wire_radius=0.35,
        min_clearance=0.150,
    )
    res_sl = generate_interlinked_lattice(cfg_sl, check_clearance=True)
    assert res_sl.metadata["pipeline"] == "epitaxial_superlattice"
    assert res_sl.clearance_valid is True
