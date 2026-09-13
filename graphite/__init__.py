"""
Graphite: core lattice and TPMS toolkit.

Phase 1 exposes shared math utilities; higher-level
geometry and engine modules will be migrated here over time.
"""

from graphite.generators import (
    generate_pentamode_lattice,
    generate_interlocking_auxetic_sheet,
    combine_interlocking_meshes,
    verify_interlocking_clearance,
    LatticeGraph,
    ImplicitField,
)

__all__ = [
    "generate_pentamode_lattice",
    "generate_interlocking_auxetic_sheet",
    "combine_interlocking_meshes",
    "verify_interlocking_clearance",
    "LatticeGraph",
    "ImplicitField",
]
