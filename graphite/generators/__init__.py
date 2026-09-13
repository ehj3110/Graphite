"""
Graphite Advanced Generators Package

Exposes high-compliance and kinematic metamaterial lattice generators:
- Transversely isotropic hexagonal pentamode (Z=4 hubs, basal rings, biconical pillars)
- Continuous Pentamode (Meta-Fluid) Networks (Milton-Cherkaev diamond cubic with bicones & sub-trusses)
- Interlocking Contact Assemblies (3D re-entrant auxetics, hook arrays, chainmail)
"""

from .pentamode import (
    generate_pentamode_lattice,
    generate_diamond_cubic_graph,
    bicone_radius,
    LatticeGraph,
    ImplicitField,
)
from .hexagonal_pentamode import (
    hexagonal_pentamode_cell,
    generate_hexagonal_pentamode_graph,
    generate_hexagonal_pentamode_lattice,
    generate_transverse_hexagonal_pentamode,
    export_hex_pentamode_review,
    ab_layer_shift,
    TOP_RING_PHASE,
)
from .interlocking import (
    generate_interlocking_auxetic_sheet,
    combine_interlocking_meshes,
    verify_interlocking_clearance,
)
from .rotating_auxetics import generate_rotating_squares_lattice
from .plate_lattice import generate_plate_lattice, calibrate_plate_thickness

__all__ = [
    "generate_pentamode_lattice",
    "generate_diamond_cubic_graph",
    "bicone_radius",
    "LatticeGraph",
    "ImplicitField",
    "hexagonal_pentamode_cell",
    "generate_hexagonal_pentamode_graph",
    "generate_hexagonal_pentamode_lattice",
    "generate_transverse_hexagonal_pentamode",
    "export_hex_pentamode_review",
    "ab_layer_shift",
    "TOP_RING_PHASE",
    "generate_interlocking_auxetic_sheet",
    "combine_interlocking_meshes",
    "verify_interlocking_clearance",
    "generate_rotating_squares_lattice",
    "generate_plate_lattice",
    "calibrate_plate_thickness",
]

