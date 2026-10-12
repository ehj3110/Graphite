"""
Tests for Universal Topological Catenation & Link Checker (catenation.py)
"""

import numpy as np
import pytest

from graphite.explicit.interlinked.particle import InterlinkedParticle, ParticleGeometry
from graphite.explicit.interlinked.cells import _build_ring_geometry, _build_tet_geometry, _build_oct_geometry
from graphite.explicit.interlinked.catenation import (
    to_spanning_solid,
    verify_particle_linking,
    audit_lattice_catenation,
)
from graphite.explicit.interlinked.pams import (
    generate_d4tet_interlocked_pair,
    generate_j4oct_interlocked_pair,
)


def test_hopf_linked_rings():
    """Two perpendicular rings through each other's center must be topologically linked."""
    r_geom = _build_ring_geometry(major_radius=5.0, num_segments=32)
    
    # Ring 1 in XY plane at z=0
    p1 = InterlinkedParticle(
        particle_id=0,
        geometry=r_geom,
        transform=np.eye(4),
    )
    
    # Ring 2 in XZ plane at x=2.5 (rotated 90 deg about X, piercing Ring 1's interior)
    T2 = np.eye(4)
    T2[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    T2[:3, 3] = np.array([2.5, 0.0, 0.0])
    
    p2 = InterlinkedParticle(
        particle_id=1,
        geometry=r_geom,
        transform=T2,
    )
    
    res = verify_particle_linking(p1, p2, strut_radius=0.35)
    assert res.is_linked, "Hopf linked rings must report is_linked=True"
    assert res.solid_intersection_volume > 0.0
    assert not res.colliding


def test_concentric_coplanar_rings_not_linked():
    """A smaller ring floating inside a larger ring in the same plane must NOT be linked."""
    r_geom_large = _build_ring_geometry(major_radius=8.0, num_segments=32)
    r_geom_small = _build_ring_geometry(major_radius=3.0, num_segments=32)
    
    p_large = InterlinkedParticle(particle_id=0, geometry=r_geom_large, transform=np.eye(4))
    p_small = InterlinkedParticle(particle_id=1, geometry=r_geom_small, transform=np.eye(4))
    
    res = verify_particle_linking(p_large, p_small, strut_radius=0.35)
    # The solids overlap, but the curves do NOT link!
    assert not res.is_linked, "Concentric rings must NOT be reported as linked!"


def test_separated_particles_not_linked():
    """Widely separated particles must have zero solid intersection and report not linked."""
    r_geom = _build_ring_geometry(major_radius=5.0, num_segments=32)
    p1 = InterlinkedParticle(particle_id=0, geometry=r_geom, transform=np.eye(4))
    
    T2 = np.eye(4)
    T2[:3, 3] = np.array([50.0, 0.0, 0.0])
    p2 = InterlinkedParticle(particle_id=1, geometry=r_geom, transform=T2)
    
    res = verify_particle_linking(p1, p2, strut_radius=0.35)
    assert not res.is_linked
    assert res.solid_intersection_volume == 0.0


def test_d4tet_catenated_pair():
    """Diamond tetrahedron catenated pair must be linked with positive clearance."""
    pam_res = generate_d4tet_interlocked_pair(edge_length=10.0, strut_radius=0.40)
    pA = InterlinkedParticle.from_pam(pam_res.particles[0])
    pB = InterlinkedParticle.from_pam(pam_res.particles[1])
    
    res = verify_particle_linking(pA, pB, strut_radius=0.40)
    assert res.is_linked, "D-4-TET catenated pair must be linked"
    assert res.solid_intersection_volume > 0.0
    assert res.min_surface_clearance > 0.0


def test_j4oct_catenated_pair():
    """J-4-OCT tip-to-tip 45-deg catenated pair must be linked."""
    pam_res = generate_j4oct_interlocked_pair(size=6.0, strut_radius=0.35)
    pA = InterlinkedParticle.from_pam(pam_res.particles[0])
    pB = InterlinkedParticle.from_pam(pam_res.particles[1])
    
    res = verify_particle_linking(pA, pB, strut_radius=0.35)
    assert res.is_linked, "J-4-OCT catenated pair must be linked"
    assert res.solid_intersection_volume > 0.0
    assert res.min_surface_clearance > 0.0
