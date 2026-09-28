"""
Smooth Boolean Operators (R-Functions) for Implicit Field Blending.

Provides C^1 and C^2 continuous smooth minimum, maximum, and difference operators
for level-set fields where solid material is defined by {x | field(x) <= 0}.
"""

from __future__ import annotations

import numpy as np


def smooth_min(
    a: float | np.ndarray,
    b: float | np.ndarray,
    r: float,
    method: str = "polynomial",
) -> float | np.ndarray:
    """
    Smooth union operator (smin) for level-set fields where solid is field <= 0.

    Parameters
    ----------
    a : float or np.ndarray
        First field values.
    b : float or np.ndarray
        Second field values.
    r : float
        Blending radius / smoothing parameter. Must be non-negative.
        When r <= 1e-6, exact np.minimum(a, b) is returned.
    method : str, optional
        Blending formula:
        - "polynomial": Quadratic Bézier / Inigo Quilez smooth minimum (C^1).
        - "circular": Exact circular arc fillet (HG_SDF / Quilez C^1).
        - "exponential": LogSumExp formulation (C^inf).
        Default is "polynomial".

    Returns
    -------
    float or np.ndarray
        Smooth minimum of a and b.
    """
    if r < 0.0:
        raise ValueError(f"blend_radius must be non-negative, got {r}")

    if r <= 1e-6:
        return np.minimum(a, b)

    method = method.lower()
    if method == "polynomial":
        diff = b - a
        h = np.clip(0.5 + 0.5 * (diff / r), 0.0, 1.0)
        return a * h + b * (1.0 - h) - r * h * (1.0 - h)

    elif method == "circular":
        diff = np.abs(a - b)
        h = np.clip((r - diff) / r, 0.0, 1.0)
        correction = r * 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - h * h, 0.0)))
        return np.minimum(a, b) - correction

    elif method == "exponential":
        min_ab = np.minimum(a, b)
        diff = np.abs(a - b)
        return min_ab - r * np.log1p(np.exp(-diff / r))

    else:
        raise ValueError(
            f"Unknown smoothing method '{method}'. Valid options are: "
            f"'polynomial', 'circular', 'exponential'."
        )


def smooth_max(
    a: float | np.ndarray,
    b: float | np.ndarray,
    r: float,
    method: str = "polynomial",
) -> float | np.ndarray:
    """
    Smooth intersection operator (smax) for level-set fields where solid is field <= 0.

    By De Morgan's duality:
        smax(a, b, r) = -smin(-a, -b, r)

    Parameters
    ----------
    a : float or np.ndarray
        First field values.
    b : float or np.ndarray
        Second field values.
    r : float
        Blending radius / smoothing parameter. Must be non-negative.
        When r <= 1e-6, exact np.maximum(a, b) is returned.
    method : str, optional
        Blending formula: 'polynomial', 'circular', or 'exponential'.
        Default is "polynomial".

    Returns
    -------
    float or np.ndarray
        Smooth maximum of a and b.
    """
    if r < 0.0:
        raise ValueError(f"blend_radius must be non-negative, got {r}")

    if r <= 1e-6:
        return np.maximum(a, b)

    method = method.lower()
    if method == "polynomial":
        diff = a - b
        h = np.clip(0.5 + 0.5 * (diff / r), 0.0, 1.0)
        return a * h + b * (1.0 - h) + r * h * (1.0 - h)

    elif method == "circular":
        diff = np.abs(a - b)
        h = np.clip((r - diff) / r, 0.0, 1.0)
        correction = r * 0.5 * (1.0 - np.sqrt(np.maximum(1.0 - h * h, 0.0)))
        return np.maximum(a, b) + correction

    elif method == "exponential":
        max_ab = np.maximum(a, b)
        diff = np.abs(a - b)
        return max_ab + r * np.log1p(np.exp(-diff / r))

    else:
        raise ValueError(
            f"Unknown smoothing method '{method}'. Valid options are: "
            f"'polynomial', 'circular', 'exponential'."
        )


def smooth_difference(
    a: float | np.ndarray,
    b: float | np.ndarray,
    r: float,
    method: str = "polynomial",
) -> float | np.ndarray:
    """
    Smooth difference operator (A \\ B) for level-set fields where solid is field <= 0.

    Computes smooth intersection of A and the complement of B (~B is -b <= 0):
        smooth_difference(a, b, r) = smooth_max(a, -b, r)

    Parameters
    ----------
    a : float or np.ndarray
        Field of base geometry A.
    b : float or np.ndarray
        Field of geometry B to subtract.
    r : float
        Blending radius / smoothing parameter. Must be non-negative.
    method : str, optional
        Blending formula: 'polynomial', 'circular', or 'exponential'.
        Default is "polynomial".

    Returns
    -------
    float or np.ndarray
        Smooth difference of A and B.
    """
    return smooth_max(a, -b, r=r, method=method)
