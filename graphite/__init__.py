"""
Graphite: core lattice and TPMS toolkit.

Phase 1 exposes shared math utilities; higher-level
geometry and engine modules will be migrated here over time.
"""

try:
    from graphite.generators import (
        generate_pentamode_lattice,
        generate_interlocking_auxetic_sheet,
        combine_interlocking_meshes,
        verify_interlocking_clearance,
        LatticeGraph,
        ImplicitField,
    )
except ImportError:
    generate_pentamode_lattice = None
    generate_interlocking_auxetic_sheet = None
    combine_interlocking_meshes = None
    verify_interlocking_clearance = None
    LatticeGraph = None
    ImplicitField = None

__all__ = [
    "generate_pentamode_lattice",
    "generate_interlocking_auxetic_sheet",
    "combine_interlocking_meshes",
    "verify_interlocking_clearance",
    "LatticeGraph",
    "ImplicitField",
]

