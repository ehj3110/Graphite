"""
Graphite FEA and Homogenization Subsystem.

Provides micro-scale unit-cell homogenization, tensor surrogate modeling,
and two-scale continuum bridging for macro-scale FEA in Aristo.
"""

from __future__ import annotations

from graphite.fea.aristo_bridge import (
    MacroMesh,
    TwoScaleFEAResult,
    apply_surface_traction,
    assemble_anisotropic_global_K,
    create_box_continuum_mesh,
    evaluate_surrogate_elasticity,
    export_two_scale_result_vtk,
    find_boundary_nodes_by_plane,
    generate_macro_continuum_mesh,
    map_grading_field_to_centroids,
    recover_element_and_nodal_stresses,
    run_two_scale_macro_fea,
)
from graphite.fea.homogenization import (
    EngineeringConstants,
    HomogenizationResult,
    RVEGridConfig,
    build_voxel_c3d8_stiffness,
    homogenize_octet_cell,
    homogenize_strut_cell,
    homogenize_tpms_cell,
    homogenize_voxel_rve,
)
from graphite.fea.stress_adaptation import (
    OptimizationIterationRecord,
    StressAdaptationConfig,
    TwoScaleOptimizationResult,
    apply_volume_bisection_scaling,
    build_neighborhood_filter,
    export_optimization_result_vtk,
    optimize_lattice_density_fsd,
    realize_optimized_strut_lattice,
    realize_optimized_tpms_lattice,
    render_optimization_summary_png,
)
from graphite.fea.surrogate import (
    MaterialTensorSurrogate,
    SurrogateCalibrationPoint,
    build_octet_homogenization_surrogate,
    build_strut_homogenization_surrogate,
    build_tpms_homogenization_surrogate,
)

__all__ = [
    "EngineeringConstants",
    "HomogenizationResult",
    "MacroMesh",
    "MaterialTensorSurrogate",
    "OptimizationIterationRecord",
    "RVEGridConfig",
    "StressAdaptationConfig",
    "SurrogateCalibrationPoint",
    "TwoScaleFEAResult",
    "TwoScaleOptimizationResult",
    "apply_surface_traction",
    "apply_volume_bisection_scaling",
    "assemble_anisotropic_global_K",
    "build_neighborhood_filter",
    "build_octet_homogenization_surrogate",
    "build_strut_homogenization_surrogate",
    "build_tpms_homogenization_surrogate",
    "build_voxel_c3d8_stiffness",
    "create_box_continuum_mesh",
    "evaluate_surrogate_elasticity",
    "export_optimization_result_vtk",
    "export_two_scale_result_vtk",
    "find_boundary_nodes_by_plane",
    "generate_macro_continuum_mesh",
    "homogenize_octet_cell",
    "homogenize_strut_cell",
    "homogenize_tpms_cell",
    "homogenize_voxel_rve",
    "map_grading_field_to_centroids",
    "optimize_lattice_density_fsd",
    "realize_optimized_strut_lattice",
    "realize_optimized_tpms_lattice",
    "recover_element_and_nodal_stresses",
    "render_optimization_summary_png",
    "run_two_scale_macro_fea",
]


