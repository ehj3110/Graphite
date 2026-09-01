"""
Implicit lattice engines (TPMS, SDF-based conformal lattices).

Engines from `Implicit_Lattice_Exploration` will be progressively
wrapped and migrated into this namespace.
"""

from graphite.implicit.pore_metrics import (
    CrossSectionalPoreSizeResult,
    EffectivePoreSizeResult,
    MaxInscribedSphereResult,
    WallThicknessResolutionResult,
    ZGradedPoreMetricsResult,
    compute_cross_sectional_pore_size,
    compute_effective_pore_size,
    compute_max_inscribed_sphere_pore_size,
    compute_pore_metrics_for_z_graded,
    recommend_resolution_from_wall_thickness,
)
from graphite.implicit.calibration import (
    CalibrationConfig,
    CalibrationGradientResult,
    CalibrationIteration,
    CalibrationPointResult,
    calibrate_tau_at_fixed_period,
    calibrate_period_for_pore_at_wall,
    calibrate_period_for_pore_at_sf,
    calibrate_tpms_gradient_profile,
    calibrate_tpms_point,
    calibration_result_to_dict,
    load_calibration_seed_table,
    save_calibration_seed_table,
    update_seed_table_with_result,
)
from graphite.implicit.tpms_parameter_lut import (
    STREAMLIT_TPMS_TYPES,
    adaptive_calibration_config,
    build_tpms_parameter_lut,
    load_tpms_parameter_lut,
    lookup_tpms_parameters,
    save_tpms_parameter_lut,
)
from graphite.implicit.piecewise_bands import (
    cumulative_phase_w_from_l_profile,
    splitp_piecewise_box_single_pass,
    splitp_piecewise_cylinder_single_pass,
)
from graphite.implicit.piecewise_woodpile import (
    woodpile_piecewise_box_single_pass,
    woodpile_piecewise_cylinder_single_pass,
    woodpile_piecewise_cylinder_union,
)
from graphite.implicit.woodpile_input import (
    WoodpileLatticeSpec,
    WoodpileImplicitSpec,
    build_piecewise_woodpile_mesh,
    repair_woodpile_mesh,
)
from graphite.implicit.meshing_backends import (
    IsosurfaceExtractionResult,
    extract_isosurface,
    extract_isosurface_from_image_data,
)

__all__ = [
    "CrossSectionalPoreSizeResult",
    "EffectivePoreSizeResult",
    "MaxInscribedSphereResult",
    "WallThicknessResolutionResult",
    "ZGradedPoreMetricsResult",
    "compute_cross_sectional_pore_size",
    "compute_effective_pore_size",
    "compute_max_inscribed_sphere_pore_size",
    "compute_pore_metrics_for_z_graded",
    "recommend_resolution_from_wall_thickness",
    "CalibrationConfig",
    "CalibrationGradientResult",
    "CalibrationIteration",
    "CalibrationPointResult",
    "calibrate_tpms_gradient_profile",
    "calibrate_tpms_point",
    "calibrate_tau_at_fixed_period",
    "calibrate_period_for_pore_at_wall",
    "calibrate_period_for_pore_at_sf",
    "calibration_result_to_dict",
    "load_calibration_seed_table",
    "save_calibration_seed_table",
    "update_seed_table_with_result",
    "STREAMLIT_TPMS_TYPES",
    "adaptive_calibration_config",
    "build_tpms_parameter_lut",
    "load_tpms_parameter_lut",
    "lookup_tpms_parameters",
    "save_tpms_parameter_lut",
    "IsosurfaceExtractionResult",
    "extract_isosurface",
    "extract_isosurface_from_image_data",
    "cumulative_phase_w_from_l_profile",
    "splitp_piecewise_box_single_pass",
    "splitp_piecewise_cylinder_single_pass",
    "woodpile_piecewise_cylinder_union",
    "woodpile_piecewise_cylinder_single_pass",
    "woodpile_piecewise_box_single_pass",
    "WoodpileLatticeSpec",
    "WoodpileImplicitSpec",
    "build_piecewise_woodpile_mesh",
    "repair_woodpile_mesh",
]

