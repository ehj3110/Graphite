"""
Map between TPMS density controls: solid fraction (calibrated) vs physical wall thickness.

Sheet TPMS solids use |F| <= tau. For small tau, physical wall thickness w (mm) is
approximated by w ≈ 2*tau/k with k = 2*pi/L, hence tau ≈ w*k/2.
"""
from __future__ import annotations

import numpy as np


def period_mm_from_sizing(
    *,
    pore_size_mm: float | None,
    unit_cell_size_mm: float | None,
    solid_fraction_for_pore_mapping: float = 0.33,
) -> float:
    """Resolve lattice period L (mm) from Step 3 sizing inputs."""
    if unit_cell_size_mm is not None:
        return max(float(unit_cell_size_mm), 1e-6)
    if pore_size_mm is not None:
        phi = float(np.clip(solid_fraction_for_pore_mapping, 0.01, 0.99))
        return max(float(pore_size_mm) / (1.0 - 1.15 * phi), 1e-6)
    raise ValueError("Must provide pore_size_mm or unit_cell_size_mm")


def tau_from_wall_thickness_mm(
    wall_thickness_mm: float | np.ndarray,
    period_L_mm: float | np.ndarray,
    *,
    tau_cap: float | None = None,
) -> np.ndarray:
    """
    Convert physical wall thickness (mm) to TPMS iso threshold tau.

    Parameters
    ----------
    wall_thickness_mm : float or ndarray
        Target strut / sheet wall thickness in millimeters.
    period_L_mm : float or ndarray
        Local unit-cell period L in millimeters (same broadcast shape as wall).
    tau_cap : float, optional
        Upper bound on tau (e.g. from |F| distribution). By default unbounded except >0.
    """
    k = 2.0 * np.pi / np.maximum(np.asarray(period_L_mm, dtype=np.float64), 1e-6)
    w = np.asarray(wall_thickness_mm, dtype=np.float64)
    tau = w * k / 2.0
    tau = np.maximum(tau, 1e-6)
    if tau_cap is not None:
        tau = np.minimum(tau, float(tau_cap))
    return tau


def solid_fraction_from_wall_thickness_mm(
    wall_thickness_mm: float,
    period_L_mm: float,
    abs_values_sorted: np.ndarray,
) -> float:
    """
    Approximate a solid fraction that maps to the same tau under quantile calibration.

    Used for display / legacy paths that still accept solid_fraction.
    """
    tau = float(tau_from_wall_thickness_mm(wall_thickness_mm, period_L_mm, tau_cap=None).ravel()[0])
    n = int(len(abs_values_sorted))
    if n < 2:
        return 0.33
    pos = float(np.searchsorted(abs_values_sorted, tau, side="right")) / float(n - 1)
    return float(np.clip(pos, 0.0, 1.0))
