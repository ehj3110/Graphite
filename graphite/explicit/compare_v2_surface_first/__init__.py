"""SC conformal compare V2 — surface-first dual + volumetric trim + gated stitch."""

from .generate import generate_compare_v2
from .stitch import (
    StitchReport,
    angle_from_inward_normal_deg,
    angle_from_vertical_deg,
    horizontal_reach,
    orphan_cull_volume_graph,
    passes_angle_gate,
    passes_approach_gate,
    passes_distance_gate,
    stitch_cut_to_dual,
    unit_cell_length_horizontal,
)

__all__ = [
    "generate_compare_v2",
    "StitchReport",
    "angle_from_inward_normal_deg",
    "angle_from_vertical_deg",
    "horizontal_reach",
    "orphan_cull_volume_graph",
    "passes_angle_gate",
    "passes_approach_gate",
    "passes_distance_gate",
    "stitch_cut_to_dual",
    "unit_cell_length_horizontal",
]
