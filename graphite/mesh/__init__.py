"""
Graphite Mesh - High-performance mesh extraction and surface post-processing.
"""

from graphite.mesh.extraction import extract_isosurface_flying_edges
from graphite.mesh.smoothing import compute_mean_curvature, smooth_mesh_taubin

extract_isosurface = extract_isosurface_flying_edges

__all__ = [
    "compute_mean_curvature",
    "extract_isosurface",
    "extract_isosurface_flying_edges",
    "smooth_mesh_taubin",
]

