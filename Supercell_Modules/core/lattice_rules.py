"""
Backward compatibility: re-export tet rules from Universal_Lattice_Engine.

The canonical implementation lives in Universal_Lattice_Engine/core/tet_rules.py.
This shim preserves imports for Supercell_Modules/core/factory.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Universal_Lattice_Engine.core.tet_rules import (
    apply_icosahedral_rule,
    apply_kagome_rule,
    apply_rhombic_rule,
    apply_voronoi_rule,
)

__all__ = [
    "apply_voronoi_rule",
    "apply_kagome_rule",
    "apply_icosahedral_rule",
    "apply_rhombic_rule",
]
