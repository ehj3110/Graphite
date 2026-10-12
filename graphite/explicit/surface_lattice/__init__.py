# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Subpackage.

Unified 2D and surface-conformal lattice engine for:
- Flat plates & test coupons (orthogonal Z-extrusion)
- Cylinders, sleeves & napkin rings (radial prism extrusion)
- Direct surface element mapping & surface duals (surface-normal sweeping)
- Automated CAD fixture extraction and collar rim fusion
"""
from __future__ import annotations

from graphite.explicit.surface_lattice.unit_cells import (
    generate_tetrachiral_cell,
    generate_trichiral_cell,
    tessellate_chiral_domain,
    tessellate_reentrant_domain,
    get_triangle_segments,
    get_square_segments,
    dedupe_2d_segments,
    tessellate_graded_cell_domain,
    TRI_TOPOLOGIES,
    SQ_TOPOLOGIES,
)
from graphite.explicit.surface_lattice.grading import (
    eval_linear_gradient,
    eval_radial_gradient,
    eval_attractor_distance,
    stiffness_to_strut_width_bending,
    stiffness_to_strut_width_stretching,
    width_for_constant_solid_fraction,
    compute_sine_curvature,
    SineCurvatureMapper,
    compute_profile_curvature,
    ProfileCurvatureMapper,
    eval_sine_compensated_theta,
    eval_sine_compensated_strut_width,
    solve_reentrant_dimensions_from_pore,
    analyze_pore_maximal_inscribed_circles,
)
from graphite.explicit.surface_lattice.face_operators import (
    apply_surface_pattern_to_mesh,
    segments_from_tri_face,
    segments_from_quad_face,
    segments_to_nodes_struts,
)
from graphite.explicit.surface_lattice.sweepers import (
    sweep_planar_2d,
    sweep_cylindrical_prisms,
    sweep_surface_skin,
)
from graphite.explicit.surface_lattice.cad_fixtures import (
    inspect_cylinder_fixture,
    carve_and_fuse_collar_rims,
)
from graphite.explicit.surface_lattice.pipeline import (
    generate_surface_lattice,
)

__all__ = [
    "generate_surface_lattice",
    "generate_tetrachiral_cell",
    "generate_trichiral_cell",
    "tessellate_chiral_domain",
    "tessellate_reentrant_domain",
    "get_triangle_segments",
    "get_square_segments",
    "tessellate_graded_cell_domain",
    "dedupe_2d_segments",
    "TRI_TOPOLOGIES",
    "SQ_TOPOLOGIES",
    "eval_linear_gradient",
    "eval_radial_gradient",
    "eval_attractor_distance",
    "stiffness_to_strut_width_bending",
    "stiffness_to_strut_width_stretching",
    "width_for_constant_solid_fraction",
    "compute_sine_curvature",
    "SineCurvatureMapper",
    "compute_profile_curvature",
    "ProfileCurvatureMapper",
    "eval_sine_compensated_theta",
    "eval_sine_compensated_strut_width",
    "solve_reentrant_dimensions_from_pore",
    "analyze_pore_maximal_inscribed_circles",
    "apply_surface_pattern_to_mesh",
    "segments_from_tri_face",
    "segments_from_quad_face",
    "segments_to_nodes_struts",
    "sweep_planar_2d",
    "sweep_cylindrical_prisms",
    "sweep_surface_skin",
    "inspect_cylinder_fixture",
    "carve_and_fuse_collar_rims",
]
