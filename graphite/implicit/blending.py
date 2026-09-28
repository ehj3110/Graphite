"""
Implicit Field Blending Utilities.

Provides high-level functions for smoothly blending implicit lattices with CAD
boundaries and solid skins using C^1/C^2 smooth Boolean operators.
"""

from __future__ import annotations

import numpy as np
from graphite.math.boolean import smooth_max, smooth_min


def blend_lattice_with_skin(
    lattice_field: np.ndarray,
    cad_sdf: np.ndarray,
    skin_thickness: float,
    blend_radius: float = 0.0,
    method: str = "polynomial",
) -> np.ndarray:
    """
    Smoothly blend an interior implicit lattice field with an exterior CAD skin.

    Parameters
    ----------
    lattice_field : np.ndarray
        Implicit scalar field of the lattice architecture (solid is <= 0).
    cad_sdf : np.ndarray
        Signed Distance Field of the CAD boundary (inside domain is <= 0).
    skin_thickness : float
        Physical thickness of the solid outer skin in mm. Must be >= 0.
    blend_radius : float, optional
        Radius of the smooth fillet at the lattice-skin junction in mm.
        Default is 0.0 (sharp Boolean).
    method : str, optional
        Smooth blending formulation: 'polynomial', 'circular', or 'exponential'.
        Default is "polynomial".

    Returns
    -------
    np.ndarray
        The blended implicit scalar field (solid is <= 0).
    """
    if lattice_field.shape != cad_sdf.shape:
        raise ValueError(
            f"Shape mismatch: lattice_field {lattice_field.shape} vs cad_sdf {cad_sdf.shape}"
        )
    if skin_thickness < 0.0:
        raise ValueError(f"skin_thickness must be non-negative, got {skin_thickness}")
    if blend_radius < 0.0:
        raise ValueError(f"blend_radius must be non-negative, got {blend_radius}")

    # Clip lattice inside CAD volume
    if blend_radius > 0.0:
        core_sdf = smooth_max(lattice_field, cad_sdf, r=blend_radius, method=method)
    else:
        core_sdf = np.maximum(lattice_field, cad_sdf)

    if skin_thickness <= 0.0:
        return core_sdf

    # Exterior skin definition: cad_sdf <= 0 AND distance_from_surface <= skin_thickness
    # In EDT distance: cad_sdf <= 0 and -cad_sdf <= skin_thickness  <=>  -cad_sdf - skin_thickness <= 0
    if blend_radius > 0.0:
        skin_sdf = smooth_max(
            cad_sdf, -cad_sdf - skin_thickness, r=blend_radius, method=method
        )
        # Union core and skin with smooth fillet
        return smooth_min(core_sdf, skin_sdf, r=blend_radius, method=method)
    else:
        skin_sdf = np.maximum(cad_sdf, -cad_sdf - skin_thickness)
        return np.minimum(core_sdf, skin_sdf)
