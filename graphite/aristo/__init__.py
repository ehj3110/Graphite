"""
Aristo — FEA/FEM Analysis Module for the Graphite Pipeline

Named after Aristo Werke, the German slide-rule manufacturer
(a tongue-in-cheek nod, since Abaqus is already taken).

Aristo provides linear-elastic finite element analysis of solid parts and
lattice structures using a custom vectorized P1 tetrahedron direct stiffness
method built on scipy.sparse — zero new dependencies beyond the existing
Graphite tech stack (gmsh, scipy, numpy, trimesh).

Future direction (documented intent, not implemented): migrate assembly and
standard FEM forms to scikit-FEM while keeping Gmsh meshing and lattice-specific
BC/quality-gate logic. See docs/ARISTO.md § "Future direction: scikit-FEM".

Pipeline Position
-----------------
Aristo can be inserted at two optional points in the Graphite pipeline:

  Pre-lattice pass:
    load_and_verify_mesh → [run_aristo] → generate_conformal_scaffold
    → generate_topology → generate_geometry

  Post-lattice pass:
    generate_geometry → [run_aristo] → export_stl

Both passes are fully opt-in. If neither is enabled, the pipeline is
unchanged from its existing behavior.

Quick Start
-----------
    from graphite.aristo.implicit_input import AristoImplicitSpec, build_piecewise_splitp_boundary_mesh
    from graphite.aristo import AristoConfig, run_aristo

    mesh, _meta = build_piecewise_splitp_boundary_mesh(AristoImplicitSpec())
    config = AristoConfig(fea_input_mode="implicit", fea_mesh_resolution=0.005)
    result = run_aristo(mesh, config)

    # Developer STL path:
    # mesh = trimesh.load("my_part.stl")
    # config = AristoConfig(fea_input_mode="stl", fea_gmsh_mesh_mode="tpms")

Public API
----------
    AristoConfig   — configuration dataclass (material, mesh, load, constraints)
    AristoResult   — NamedTuple with displacement, Von Mises, and mesh arrays
    run_aristo     — main entry point; returns AristoResult
    MATERIAL_PRESETS — dict of (E, ν) tuples for common materials

Module Structure
----------------
    graphite/aristo/
        __init__.py          ← this file (public API)
        aristo_config.py     ← AristoConfig dataclass + MATERIAL_PRESETS
        aristo_solver.py     ← run_aristo, AristoResult, gmsh mesh gen, solve
        stiffness_assembly.py← K assembly, B matrices, Von Mises (vectorized)

    Planned (Phase 1b–2):
        boundary_detection.py← proper face-normal BC auto-detection
        stress_mapper.py     ← Von Mises → graded solid_fraction_field / radius_map
"""

import os

from graphite.aristo.aristo_config import AristoConfig, MATERIAL_PRESETS

if os.environ.get("ARISTO_SURFACE_CLEAN_WORKER", "").strip() != "1":
    from graphite.aristo.aristo_solver import AristoResult, run_aristo
    from graphite.aristo.stress_mapper import stress_to_vf_gradient, stress_to_strut_radius_map
else:
    AristoResult = None
    run_aristo = None
    stress_to_vf_gradient = None
    stress_to_strut_radius_map = None

__all__ = [
    "AristoConfig",
    "AristoResult",
    "MATERIAL_PRESETS",
    "run_aristo",
    "stress_to_vf_gradient",
    "stress_to_strut_radius_map",
]
