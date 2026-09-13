from __future__ import annotations

# Route legacy GMSH-based hex scaffold calls to the archived location
from graphite.legacy_gmsh.hex_scaffold_module import (
    generate_conformed_hex_scaffold,
    generate_cropped_hex_scaffold,
    synthesize_vf_gated_hex_volume_and_surface_dual,
    _generate_bbox_hex_grid,
)

from graphite.explicit.lofted_scaffold import (
    compute_equal_phase_stations,
    generate_brute_force_fixed_grid_hex_scaffold,
    generate_lofted_hex_scaffold,
    synthesize_lofted_lattice,
    synthesize_lofted_multilattice,
)

__all__ = [
    "generate_conformed_hex_scaffold",
    "generate_cropped_hex_scaffold",
    "synthesize_vf_gated_hex_volume_and_surface_dual",
    "_generate_bbox_hex_grid",
    "generate_lofted_hex_scaffold",
    "synthesize_lofted_lattice",
    "synthesize_lofted_multilattice",
    "compute_equal_phase_stations",
    "generate_brute_force_fixed_grid_hex_scaffold",
]
