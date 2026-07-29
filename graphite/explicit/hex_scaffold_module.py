from __future__ import annotations

# Route legacy GMSH-based hex scaffold calls to the archived location
from graphite.legacy_gmsh.hex_scaffold_module import (
    generate_conformed_hex_scaffold,
    generate_cropped_hex_scaffold,
    synthesize_vf_gated_hex_volume_and_surface_dual,
    _generate_bbox_hex_grid,
)

__all__ = [
    "generate_conformed_hex_scaffold",
    "generate_cropped_hex_scaffold",
    "synthesize_vf_gated_hex_volume_and_surface_dual",
    "_generate_bbox_hex_grid",
]
