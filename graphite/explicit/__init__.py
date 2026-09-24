"""
Explicit - Fast engine (GMSH-free as of V2).

Fast, stochastic (random), macro-isotropic lattice pipeline for organic
geometry and medical implants.

Tet mesh: A15 Conformal Kagome Dual
  generate_a15_conformal_lattice (a15_conformal.py - existing, tested)

Hex mesh: Modular SC conformal (any hex rule via ``rule_name``)
  generate_conformal_lattice(..., lattice_type='SC', rule_name=...)

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
# SC Hexahedral (Nodal Conformation + Planar Slicing Surface Dual)
# ---------------------------------------------------------------------------

def generate_conformal_lattice(*args, **kwargs):
    """Unified GMSH-free conformal lattice generator.

    lattice_type='A15' -> routes to A15 Kagome
    lattice_type='SC'  -> canonical SC Nodal Conformation + Planar Slicing Surface Dual (default)
                          pass legacy=True to access archived conformal_generator morph
    """
    lt = kwargs.get("lattice_type", "SC")
    legacy = kwargs.pop("legacy", False)
    if legacy and lt == "SC":
        from .conformal_generator import generate_conformal_lattice as _legacy_impl
        return _legacy_impl(*args, **kwargs)

    if lt == "A15":
        from .a15_conformal import generate_a15_conformal_lattice as _impl
        kwargs.pop("lattice_type", None)
        return _impl(*args, **kwargs)
    elif lt == "SC":
        from .nodal_conformation import generate_sc_conformal_lattice as _impl
        return _impl(*args, **kwargs)
    else:
        raise ValueError(f"Unknown lattice type: '{lt}'. Supported: 'A15', 'SC'")


def generate_sc_conformal_lattice(*args, **kwargs):
    """Canonical SC Nodal Conformation + Planar Slicing Surface Dual generator."""
    from .nodal_conformation import generate_sc_conformal_lattice as _impl
    return _impl(*args, **kwargs)


def generate_legacy_conformal_lattice(*args, **kwargs):
    """Archived legacy hex-cage morph generator (conformal_generator.py)."""
    from .conformal_generator import generate_conformal_lattice as _impl
    return _impl(*args, **kwargs)


# ---------------------------------------------------------------------------
# SC Nodal Conformation (Cartesian SC + Node-Plane Trim + Universal Dual)
# ---------------------------------------------------------------------------

def generate_nodal_conformation(*args, **kwargs):
    """SC Nodal Conformation generator (node-plane trim + universal surface dual)."""
    from .nodal_conformation import generate_nodal_conformation as _impl
    return _impl(*args, **kwargs)


def deform_outside_nodes(*args, **kwargs):
    """Snap outside volume nodes to closest points on CAD boundary."""
    from .nodal_conformation import deform_outside_nodes as _impl
    return _impl(*args, **kwargs)


def weld_combined_lattice(*args, **kwargs):
    """Weld conformed volume lattice and surface dual into unified graph."""
    from .nodal_conformation import weld_combined_lattice as _impl
    return _impl(*args, **kwargs)


def build_planar_slicing_surface_dual(*args, **kwargs):
    """Planar Slicing Contour Sweep surface dual builder."""
    from .planar_surface_sweep import build_planar_slicing_surface_dual as _impl
    return _impl(*args, **kwargs)


def PlanarSweepConfig(*args, **kwargs):
    """Configuration dataclass for the Planar Slicing Contour Sweep."""
    from .planar_surface_sweep import PlanarSweepConfig as _impl
    return _impl(*args, **kwargs)



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


from .geometry_module import (
    affine_rows_from_R_t,
    build_clean_miter_truss,
    manifold_cylinder_between,
    manifold_to_trimesh,
    rotation_align_local_z_to_unit,
    trimesh_to_manifold,
)


from .chiral_cell import (
    generate_tetrachiral_cell,
    generate_trichiral_cell,
)


from .surface_lattice import (
    generate_surface_lattice,
)


def generate_interlinked_lattice(*args, **kwargs):
    """Lazy import wrapper for interlinked ring lattice generation."""
    from .interlinked import generate_interlinked_lattice as _impl
    return _impl(*args, **kwargs)


def InterlinkedConfig(*args, **kwargs):
    """Lazy import wrapper for InterlinkedConfig."""
    from .interlinked import InterlinkedConfig as _impl
    return _impl(*args, **kwargs)


def generate_lofted_hex_scaffold(*args, **kwargs):
    """Generate a boundary-conforming structured hexahedral scaffold lofted along a spine axis."""
    from .lofted_scaffold import generate_lofted_hex_scaffold as _impl
    return _impl(*args, **kwargs)


def synthesize_lofted_lattice(*args, **kwargs):
    """Synthesize explicit strut lattice topology onto deformed lofted hex elements."""
    from .lofted_scaffold import synthesize_lofted_lattice as _impl
    return _impl(*args, **kwargs)


def synthesize_lofted_multilattice(*args, **kwargs):
    """Synthesize multi-lattice topology across lofted hex layers with interface pyramids."""
    from .lofted_scaffold import synthesize_lofted_multilattice as _impl
    return _impl(*args, **kwargs)


def compute_equal_phase_stations(*args, **kwargs):
    """Compute 1D station coordinates along a spine using analytical equal-phase integration."""
    from .lofted_scaffold import compute_equal_phase_stations as _impl
    return _impl(*args, **kwargs)


__all__ = [
    # Primary A15 API (routes to battle-tested a15_conformal.py)
    "generate_a15_conformal_lattice",
    "generate_conformal_scaffold",
    # Unified wrapper
    "generate_conformal_lattice",
    "generate_sc_conformal_lattice",
    "generate_legacy_conformal_lattice",
    # SC Nodal Conformation & Universal Dual
    "generate_nodal_conformation",
    "deform_outside_nodes",
    "weld_combined_lattice",
    "build_planar_slicing_surface_dual",
    "PlanarSweepConfig",
    # Lofted explicit hex
    "generate_lofted_hex_scaffold",
    "synthesize_lofted_lattice",
    "synthesize_lofted_multilattice",
    "compute_equal_phase_stations",
    # 2D & Surface Conformal Lattices
    "generate_surface_lattice",
    # Interlinked print-in-place lattices
    "generate_interlinked_lattice",
    "InterlinkedConfig",
    # Chiral unit cells
    "generate_tetrachiral_cell",
    "generate_trichiral_cell",
    # Shared modules
    "generate_topology",
    "generate_hex_topology",
    "generate_geometry",
    "solve_sizing",
    "repair_cad_mesh",
    # Manifold3D Geometry Primitives & Helpers
    "build_clean_miter_truss",
    "manifold_cylinder_between",
    "rotation_align_local_z_to_unit",
    "affine_rows_from_R_t",
    "trimesh_to_manifold",
    "manifold_to_trimesh",
    # Health
    "check_explicit_health",
    "missing_dependencies",
    "require_explicit_dependencies",
]
