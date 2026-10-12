"""
Implicit Field Blending Utilities.

Provides high-level functions for smoothly blending implicit lattices with CAD
boundaries, solid skins, and transitioning between distinct TPMS lattices using
metric-normalized Signed Distance Fields (SDFs).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from graphite.implicit.normalization import (
    TPMSNormalizedFieldConfig,
    evaluate_normalized_tpms_mid_sdf,
)
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


def compute_transition_weights(
    coord: np.ndarray,
    start_val: float,
    end_val: float,
    kernel: Literal["linear", "smoothstep", "smootherstep"] = "smoothstep",
) -> np.ndarray:
    """
    Compute smooth spatial transition weight field alpha in [0, 1] across an interval.

    Parameters
    ----------
    coord : np.ndarray
        Spatial coordinate array (e.g. Z values across the grid).
    start_val : float
        Coordinate position where alpha = 0.0 begins.
    end_val : float
        Coordinate position where alpha = 1.0 ends.
    kernel : {'linear', 'smoothstep', 'smootherstep'}, optional
        Smoothing kernel:
        - 'linear': C^0 linear ramp.
        - 'smoothstep': C^1 Hermite polynomial (3*xi^2 - 2*xi^3).
        - 'smootherstep': C^2 Perlin polynomial (6*xi^5 - 15*xi^4 + 10*xi^3).
        Default is 'smoothstep'.

    Returns
    -------
    np.ndarray
        Weight array alpha with values strictly in [0.0, 1.0].
    """
    delta = float(end_val - start_val)
    if abs(delta) < 1e-12:
        return np.where(coord >= start_val, 1.0, 0.0)

    xi = np.clip((coord - start_val) / delta, 0.0, 1.0)

    k_name = kernel.strip().lower()
    if k_name == "linear":
        return xi
    elif k_name == "smoothstep":
        return 3.0 * xi * xi - 2.0 * xi * xi * xi
    elif k_name == "smootherstep":
        return 6.0 * (xi**5) - 15.0 * (xi**4) + 10.0 * (xi**3)
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Valid options: 'linear', 'smoothstep', 'smootherstep'."
        )


def blend_normalized_sdfs(
    sdf_a: np.ndarray,
    sdf_b: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate two metric normalized signed distance fields.

    Parameters
    ----------
    sdf_a : np.ndarray
        First normalized SDF (weight = 1 - alpha).
    sdf_b : np.ndarray
        Second normalized SDF (weight = alpha).
    alpha : np.ndarray
        Weight array in [0, 1].

    Returns
    -------
    np.ndarray
        Interpolated signed distance field: (1 - alpha) * sdf_a + alpha * sdf_b.
    """
    if sdf_a.shape != sdf_b.shape:
        raise ValueError(f"Shape mismatch: sdf_a {sdf_a.shape} vs sdf_b {sdf_b.shape}")
    return (1.0 - alpha) * sdf_a + alpha * sdf_b


def blend_tpms_lattices(
    config_a: TPMSNormalizedFieldConfig,
    config_b: TPMSNormalizedFieldConfig,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    spacing: tuple[float, float, float] | Sequence[float] | float,
    transition_axis: Literal["X", "Y", "Z"] = "Z",
    transition_range: tuple[float, float] = (4.0, 6.0),
    kernel: Literal["linear", "smoothstep", "smootherstep"] = "smoothstep",
) -> np.ndarray:
    """
    End-to-end evaluation and blending of two TPMS lattices across a transition zone.

    Order of operations:
    - For Sheet TPMS (is_sheet=True):
      1. Evaluates metric mid-surface SDFs for lattice A and B with clamping to prevent
         field explosion at extrema: phi_mid_a, phi_mid_b.
      2. Linearly blends the mid-surfaces: phi_mid = (1 - alpha) * phi_a + alpha * phi_b.
      3. Linearly interpolates target physical wall thickness: t = (1 - alpha) * t_a + alpha * t_b.
      4. Forms the metric solid sheet: Phi_solid = |phi_mid| - t / 2.0.
    - For Skeletal TPMS (is_sheet=False):
      Evaluates normalized skeletal SDFs and blends directly: Phi_solid = (1 - alpha) * phi_a + alpha * phi_b.

    Parameters
    ----------
    config_a : TPMSNormalizedFieldConfig
        Configuration for the starting lattice.
    config_b : TPMSNormalizedFieldConfig
        Configuration for the ending lattice.
    X, Y, Z : np.ndarray
        Cartesian coordinate meshgrids in millimeters.
    spacing : tuple of floats or float
        Voxel resolution in millimeters.
    transition_axis : {'X', 'Y', 'Z'}, optional
        Spatial axis driving the transition weight. Default is 'Z'.
    transition_range : tuple of (float, float), optional
        (start_coord, end_coord) defining the interface zone in mm. Default is (4.0, 6.0).
    kernel : {'linear', 'smoothstep', 'smootherstep'}, optional
        Transition interpolation kernel. Default is 'smoothstep'.

    Returns
    -------
    np.ndarray
        Solid field where solid material is defined by {x | Phi(x) <= 0}.
    """
    axis = transition_axis.strip().upper()
    if axis == "X":
        coord = X
    elif axis == "Y":
        coord = Y
    elif axis == "Z":
        coord = Z
    else:
        raise ValueError(f"Unknown transition_axis '{transition_axis}'. Must be 'X', 'Y', or 'Z'.")

    z_start, z_end = float(transition_range[0]), float(transition_range[1])
    alpha = compute_transition_weights(coord, z_start, z_end, kernel=kernel)

    # Evaluate mid-surface normalized SDFs
    phi_mid_a = evaluate_normalized_tpms_mid_sdf(config_a, X, Y, Z, spacing)
    phi_mid_b = evaluate_normalized_tpms_mid_sdf(config_b, X, Y, Z, spacing)

    # Blend mid-surfaces
    phi_mid_blend = blend_normalized_sdfs(phi_mid_a, phi_mid_b, alpha)

    if config_a.is_sheet or config_b.is_sheet:
        t_a = float(config_a.wall_thickness)
        t_b = float(config_b.wall_thickness)
        t_blend = (1.0 - alpha) * t_a + alpha * t_b
        solid_field = np.abs(phi_mid_blend) - (t_blend / 2.0)
    else:
        solid_field = phi_mid_blend

    return solid_field


def blend_multi_knot_tpms_sdfs(
    lattice_type: str,
    axis_knots: Sequence[float],
    period_knots: Sequence[float],
    thickness_knots: Sequence[float],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    spacing: tuple[float, float, float] | Sequence[float] | float,
    transition_axis: Literal["X", "Y", "Z"] = "Z",
    kernel: Literal["linear", "smoothstep", "smootherstep"] = "smoothstep",
    is_sheet: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate and blend multi-knot metric-normalized TPMS signed distance fields.

    Parameters
    ----------
    lattice_type : str
        TPMS architecture name (e.g. 'Gyroid', 'Split-P', 'Schwarz-P').
    axis_knots : Sequence of float
        Spatial knot coordinates along transition_axis in millimeters.
    period_knots : Sequence of float
        Unit cell spatial period L at each knot in millimeters.
    thickness_knots : Sequence of float
        Target physical wall thickness at each knot in millimeters.
    X, Y, Z : np.ndarray
        Coordinate grids in millimeters.
    spacing : tuple of floats or float
        Voxel resolution in millimeters.
    transition_axis : {'X', 'Y', 'Z'}, optional
        Coordinate axis driving the multi-knot transition. Default is 'Z'.
    kernel : {'linear', 'smoothstep', 'smootherstep'}, optional
        Smoothing interpolation kernel. Default is 'smoothstep'.
    is_sheet : bool, optional
        If True, returns sheet solid field (|phi| - t/2 <= 0).
        If False, returns skeletal network field (phi <= 0). Default is True.

    Returns
    -------
    solid_field : np.ndarray
        Final solid scalar field (solid material is <= 0).
    phi_mid : np.ndarray
        Blended metric mid-surface signed distance field.
    t_field : np.ndarray
        Interpolated physical wall thickness field in millimeters.
    """
    num_knots = len(axis_knots)
    if len(period_knots) != num_knots or len(thickness_knots) != num_knots:
        raise ValueError(
            f"Knot count mismatch: axis_knots ({num_knots}), period_knots ({len(period_knots)}), "
            f"thickness_knots ({len(thickness_knots)})."
        )
    if num_knots < 2:
        raise ValueError("Must provide at least 2 control knots.")

    axis = transition_axis.strip().upper()
    if axis == "X":
        coord = X
    elif axis == "Y":
        coord = Y
    elif axis == "Z":
        coord = Z
    else:
        raise ValueError(f"Unknown transition_axis '{transition_axis}'. Must be 'X', 'Y', or 'Z'.")

    # Pre-evaluate unique distinct normalized mid-surface SDFs
    unique_L = sorted(list(set(period_knots)))
    mid_sdfs: dict[float, np.ndarray] = {}
    for L_val in unique_L:
        cfg = TPMSNormalizedFieldConfig(
            lattice_type=lattice_type,
            unit_cell_size=float(L_val),
            wall_thickness=0.05,
            is_sheet=is_sheet,
            clip_distance=float(L_val) / 2.0,
        )
        mid_sdfs[float(L_val)] = evaluate_normalized_tpms_mid_sdf(cfg, X, Y, Z, spacing)

    phi_mid = np.zeros_like(coord, dtype=np.float64)
    t_field = np.zeros_like(coord, dtype=np.float64)

    # Below bottom knot
    mask_below = coord <= axis_knots[0]
    phi_mid[mask_below] = mid_sdfs[float(period_knots[0])][mask_below]
    t_field[mask_below] = float(thickness_knots[0])

    # Above top knot
    mask_above = coord >= axis_knots[-1]
    phi_mid[mask_above] = mid_sdfs[float(period_knots[-1])][mask_above]
    t_field[mask_above] = float(thickness_knots[-1])

    # Interior intervals
    for i in range(num_knots - 1):
        c_start, c_end = float(axis_knots[i]), float(axis_knots[i + 1])
        L_start, L_end = float(period_knots[i]), float(period_knots[i + 1])
        t_start, t_end = float(thickness_knots[i]), float(thickness_knots[i + 1])

        mask = (coord >= c_start) & (coord < c_end)
        if not np.any(mask):
            continue

        if np.isclose(c_start, c_end) or np.isclose(L_start, L_end):
            phi_mid[mask] = mid_sdfs[L_start][mask]
            t_field[mask] = t_start
        else:
            alpha = compute_transition_weights(coord[mask], c_start, c_end, kernel=kernel)
            phi_a = mid_sdfs[L_start][mask]
            phi_b = mid_sdfs[L_end][mask]

            phi_mid[mask] = (1.0 - alpha) * phi_a + alpha * phi_b
            t_field[mask] = (1.0 - alpha) * t_start + alpha * t_end

    if is_sheet:
        solid_field = np.abs(phi_mid) - (t_field / 2.0)
    else:
        solid_field = phi_mid

    return solid_field, phi_mid, t_field

