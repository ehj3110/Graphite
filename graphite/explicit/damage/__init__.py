"""
Graphite Explicit — Damage-Programmable Metamaterials Engine (Gao et al., Nature Communications 2024).

Implements:
- Microfiber-reinforced Body-Centered Cubic (BCC) cell hierarchy (T0, T1, T2, T3).
- Spatial partitioning into Guiding (S_g), Correction (S_c), and Background (S_b) functional zones.
- Pre-programmed 3D crack path steering and additive fracture toughening mechanisms:
  Crack Bowing (CB), Crack Deflection (CD), Crack Shielding (PS/NS), and Reinforcement Bridging.
- Clean mitered truss solidification via `generate_geometry(clean_miter=True)`.
"""

from __future__ import annotations

from .dp_cells import (
    DPBCCCell,
    DPLatticeResult,
    generate_bcc_base_cell,
    generate_dp_cell,
    generate_damage_programmable_lattice,
    calculate_fracture_energy,
)

__all__ = [
    "DPBCCCell",
    "DPLatticeResult",
    "generate_bcc_base_cell",
    "generate_dp_cell",
    "generate_damage_programmable_lattice",
    "calculate_fracture_energy",
]
