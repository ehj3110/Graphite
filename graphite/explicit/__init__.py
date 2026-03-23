"""
Explicit - Fast (Delaunay) engine.

Fast, stochastic (random), macro-isotropic lattice pipeline for organic
geometry and medical implants. This module targets robust, high-throughput
conformal generation using GMSH Delaunay tetrahedralization plus strut-based
post-processing.
"""

from .scaffold_module import generate_conformal_scaffold
from .topology_module import generate_topology
from .geometry_module import generate_geometry

__all__ = ["generate_conformal_scaffold", "generate_topology", "generate_geometry"]