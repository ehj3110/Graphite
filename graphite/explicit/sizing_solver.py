"""
Graphite Sizing Solver — Variable Calibration for A15 Kagome Lattice

Given any two of (solid_fraction, strut_radius, cell_size), solves for the third.
"""

from __future__ import annotations
import numpy as np
import warnings

# Total strut length for one A15 Kagome unit cell of cell_size = 1.0
UNIT_CELL_STRUT_LENGTH_CONSTANT = 73.36174010646175


def solve_sizing(
    solid_fraction: float | None = None,
    strut_radius: float | None = None,
    cell_size: float | None = None,
) -> dict[str, float]:
    """
    Solves for the missing variable among solid_fraction, strut_radius, and cell_size.
    Exactly two variables must be provided.

    Formula:
        solid_fraction * cell_size^3 = pi * strut_radius^2 * (c * cell_size)
        where c = UNIT_CELL_STRUT_LENGTH_CONSTANT = 73.36174010646175

    Simplifies to:
        solid_fraction * cell_size^2 = pi * strut_radius^2 * c
    """
    provided = [v is not None for v in (solid_fraction, strut_radius, cell_size)]
    if sum(provided) != 2:
        raise ValueError(
            f"Exactly two of (solid_fraction, strut_radius, cell_size) must be provided. "
            f"Got: solid_fraction={solid_fraction}, strut_radius={strut_radius}, cell_size={cell_size}"
        )

    c = UNIT_CELL_STRUT_LENGTH_CONSTANT
    resolved_sf = 0.0

    if solid_fraction is None:
        # Solve for solid_fraction
        if strut_radius <= 0 or cell_size <= 0:
            raise ValueError("All physical dimensions must be positive.")
        val = np.pi * c * (strut_radius / cell_size) ** 2
        resolved_sf = float(val)
        result = {
            "solid_fraction": resolved_sf,
            "strut_radius": float(strut_radius),
            "cell_size": float(cell_size),
        }

    elif strut_radius is None:
        # Solve for strut_radius
        if solid_fraction <= 0 or cell_size <= 0:
            raise ValueError("All physical dimensions and fractions must be positive.")
        val = cell_size * np.sqrt(solid_fraction / (np.pi * c))
        resolved_sf = float(solid_fraction)
        result = {
            "solid_fraction": resolved_sf,
            "strut_radius": float(val),
            "cell_size": float(cell_size),
        }

    else:
        # Solve for cell_size
        if solid_fraction <= 0 or strut_radius <= 0:
            raise ValueError("All physical dimensions and fractions must be positive.")
        val = strut_radius * np.sqrt((np.pi * c) / solid_fraction)
        resolved_sf = float(solid_fraction)
        result = {
            "solid_fraction": resolved_sf,
            "strut_radius": float(strut_radius),
            "cell_size": float(val),
        }

    # Emit warning if solid fraction is above 20% limit
    if resolved_sf > 0.20:
        warnings.warn(
            f"Solid fraction {resolved_sf * 100:.1f}% exceeds the printable limit of 20%. "
            f"High solid fraction lattices are difficult to print and may print as solid blocks.",
            UserWarning,
            stacklevel=2,
        )

    return result
