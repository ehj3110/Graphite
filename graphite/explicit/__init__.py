"""
Explicit - Fast engine (GMSH-free as of V2).

Fast, stochastic (random), macro-isotropic lattice pipeline for organic
geometry and medical implants.

Tet mesh: A15 Conformal Kagome Dual
  generate_a15_conformal_lattice (a15_conformal.py - existing, tested)

Hex mesh: SC Conformal Octahedral Dual
  generate_conformed_hex_scaffold (conformal_generator.py - new SC engine)

Legacy GMSH scaffolders are preserved in graphite/legacy_gmsh/ for archival reference.
"""

from __future__ import annotations

from .health import (
    check_explicit_health,
    missing_dependencies,
    require_explicit_dependencies,
)


# ---------------------------------------------------------------------------
# A15 Tetrahedral (Kagome dual) - routes to existing, tested a15_conformal.py
# ---------------------------------------------------------------------------

def generate_a15_conformal_lattice(*args, **kwargs):
    """A15 Kagome conformal lattice (GMSH-free). Routes to a15_conformal.py."""
    from .a15_conformal import generate_a15_conformal_lattice as _impl
    return _impl(*args, **kwargs)


def generate_conformal_scaffold(*args, **kwargs):
    """GMSH-free conformed A15 tetrahedral background grid generator.

    Routes to a15_conformal.generate_a15_conformal_lattice with skip_sweep=True,
    which returns topology data without STL export.
    """
    from .a15_conformal import generate_a15_conformal_lattice as _impl
    # Forward with skip_sweep so it returns quickly; caller can use the result dict
    kwargs.setdefault("skip_sweep", True)
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# SC Hexahedral (Octahedral dual) - new engine in conformal_generator.py
# ---------------------------------------------------------------------------

def generate_conformal_lattice(*args, **kwargs):
    """Unified GMSH-free conformal lattice generator.

    lattice_type='A15' -> routes to A15 Kagome
    lattice_type='SC'  -> routes to SC Octahedral
    """
    lt = kwargs.get("lattice_type", "A15")
    if lt == "A15":
        from .a15_conformal import generate_a15_conformal_lattice as _impl
        kwargs.pop("lattice_type", None)
        return _impl(*args, **kwargs)
    elif lt == "SC":
        from .conformal_generator import generate_conformal_lattice as _impl
        return _impl(*args, **kwargs)
    else:
        raise ValueError(f"Unknown lattice type: '{lt}'. Supported: 'A15', 'SC'")


# ---------------------------------------------------------------------------
# Topology, Geometry, Sizing, Repair - unchanged
# ---------------------------------------------------------------------------

def generate_topology(*args, **kwargs):
    """Lazy import wrapper for topology generation."""
    from .topology_module import generate_topology as _impl
    return _impl(*args, **kwargs)


def generate_geometry(*args, **kwargs):
    """Lazy import wrapper for geometry generation."""
    from .geometry_module import generate_geometry as _impl
    return _impl(*args, **kwargs)


def generate_hex_topology(*args, **kwargs):
    """Lazy import wrapper for hexahedral topology generation."""
    from .hex_topology_module import generate_hex_topology as _impl
    return _impl(*args, **kwargs)


def solve_sizing(*args, **kwargs):
    """Lazy import wrapper for sizing solver."""
    from .sizing_solver import solve_sizing as _impl
    return _impl(*args, **kwargs)


def repair_cad_mesh(*args, **kwargs):
    """Lazy import wrapper for CAD mesh repair."""
    from .mesh_repair import repair_cad_mesh as _impl
    return _impl(*args, **kwargs)


__all__ = [
    # Primary A15 API (routes to battle-tested a15_conformal.py)
    "generate_a15_conformal_lattice",
    "generate_conformal_scaffold",
    # Unified wrapper
    "generate_conformal_lattice",
    # Shared modules
    "generate_topology",
    "generate_hex_topology",
    "generate_geometry",
    "solve_sizing",
    "repair_cad_mesh",
    # Health
    "check_explicit_health",
    "missing_dependencies",
    "require_explicit_dependencies",
]
