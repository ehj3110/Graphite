"""
Graphite Math - Woodpile Equations

This module contains the mathematical definitions for evaluating implicit 
woodpile structures, supporting both simple cross-hatch and true alternating 
woodpile geometries.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from graphite.math.woodpile_anchor import count_crosshatch_layers_in_height


def evaluate_woodpile(
    X,
    Y,
    Z,
    pore_size: float,
    true_woodpile: bool,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
    z_layer_origin: float = 0.0,
    layer_index_offset: int = 0,
):
    """
    Vectorized implicit woodpile field.

    Field sign convention:
    - field <= 0 : inside beam (solid)
    - field > 0  : outside beam (void)

    Parameters
    ----------
    X : ndarray
        The X-coordinate grid.
    Y : ndarray
        The Y-coordinate grid.
    Z : ndarray
        The Z-coordinate grid.
    pore_size : float
        The side length of the square pores in the transverse plane.
    true_woodpile : bool
        If True, layers alternate X, Y, X (shifted), Y (shifted). If False, 
        generates a simple cross-hatch (X, Y, X, Y).
    origin_x, origin_y : float
        Translate the transverse grid (mm). Use ``graphite.math.woodpile_anchor``
        helpers for edge-solid / center-void anchoring in bounded parts.
    flip_layer_parity : bool
        If True, swap which Z parity uses X vs Y struts.
    swap_xy : bool
        If True, exchange X and Y before evaluating waves — a true 90° in-plane
        rotation of the cross-hatch (used between piecewise Z bands).
    z_layer_origin : float
        Z reference for layer thickness stacking within a band (mm).
    layer_index_offset : int
        Added to the local layer index so piecewise bands can continue X/Y
        parity across grade interfaces (printability: next layer ⊥ prior layer).

    Returns
    -------
    ndarray
        The evaluated woodpile scalar field.
    """
    pitch = 2.0 * pore_size
    layer_idx = int(layer_index_offset) + np.floor(
        (Z - float(z_layer_origin)) / pore_size
    ).astype(int)

    X_in = Y if swap_xy else X
    Y_in = X if swap_xy else Y
    X_eff = X_in - float(origin_x)
    Y_eff = Y_in - float(origin_y)
    if true_woodpile:
        shift_x_mask = (layer_idx % 4) == 3
        shift_y_mask = (layer_idx % 4) == 2
        X_eff = X_eff + np.where(shift_x_mask, pore_size, 0.0)
        Y_eff = Y_eff + np.where(shift_y_mask, pore_size, 0.0)

    wave_X = np.abs(np.mod(X_eff + pore_size, pitch) - pore_size) - (pore_size / 2.0)
    wave_Y = np.abs(np.mod(Y_eff + pore_size, pitch) - pore_size) - (pore_size / 2.0)
    if flip_layer_parity:
        field = np.where(layer_idx % 2 == 0, wave_X, wave_Y)
    else:
        field = np.where(layer_idx % 2 == 0, wave_Y, wave_X)
    return field


def evaluate_woodpile_piecewise_cylinder(
    X,
    Y,
    Z,
    *,
    z_breaks_mm: Sequence[float],
    pore_mm: Sequence[float],
    origin_x_mm: Sequence[float],
    origin_y_mm: Sequence[float],
    swap_xy: Sequence[bool],
    flip_layer_parity: Sequence[bool],
    true_woodpile: bool,
    continuous_layer_index: bool = True,
) -> np.ndarray:
    """
    Piecewise woodpile on one grid: each Z band uses its own pore and anchor.

    When ``continuous_layer_index`` is True (default), Z-layer parity continues
    across grade interfaces so the first layer above each break is ⊥ to the last
    layer below (printability).
    """
    z_breaks = [float(z) for z in z_breaks_mm]
    pores = [float(p) for p in pore_mm]
    n = len(pores)
    if len(z_breaks) != n + 1:
        raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")

    field = np.full(np.shape(X), np.inf, dtype=np.float64)
    layer_offset = 0
    for i in range(n):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        if i < n - 1:
            mask = (Z >= z0 - 1e-12) & (Z < z1)
        else:
            mask = (Z >= z0 - 1e-12) & (Z <= z1 + 1e-12)
        if not np.any(mask):
            continue
        band_field = evaluate_woodpile(
            X,
            Y,
            Z,
            pore_size=pores[i],
            true_woodpile=true_woodpile,
            origin_x=float(origin_x_mm[i]),
            origin_y=float(origin_y_mm[i]),
            flip_layer_parity=bool(flip_layer_parity[i]),
            swap_xy=bool(swap_xy[i]),
            z_layer_origin=z0,
            layer_index_offset=layer_offset if continuous_layer_index else 0,
        )
        field = np.where(mask, band_field, field)
        if continuous_layer_index:
            layer_offset += count_crosshatch_layers_in_height(z1 - z0, pores[i])
    return field
