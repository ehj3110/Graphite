"""SC conformal comparison V3 — hybrid dual shrinkwrap + morph-to-dual."""

from __future__ import annotations

from graphite.explicit.compare_v3_hybrid.pipeline import (
    generate_compare_v3,
    remove_volume_surface_struts,
    write_compare_v3_report,
)
from graphite.explicit.compare_v3_hybrid.valence import (
    apply_valence_gate,
    local_max_node_degree,
)

__all__ = [
    "generate_compare_v3",
    "write_compare_v3_report",
    "remove_volume_surface_struts",
    "apply_valence_gate",
    "local_max_node_degree",
]
