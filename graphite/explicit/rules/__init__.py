"""Rule registry and local tetrahedral micro-rules for explicit topology generation."""

from .local_tet_rules import (
    apply_icosahedral_rule,
    apply_kagome_rule,
    apply_rhombic_rule,
    apply_voronoi_rule,
)
from .tet_topology_rules import (
    TOPOLOGY_RULES,
    TopologyRule,
    get_supported_topology_names,
    get_topology_rule,
    normalize_topology_name,
    register_topology_rule,
    unregister_topology_rule,
)

__all__ = [
    "TopologyRule",
    "TOPOLOGY_RULES",
    "get_supported_topology_names",
    "get_topology_rule",
    "normalize_topology_name",
    "register_topology_rule",
    "unregister_topology_rule",
    "apply_voronoi_rule",
    "apply_kagome_rule",
    "apply_icosahedral_rule",
    "apply_rhombic_rule",
]
