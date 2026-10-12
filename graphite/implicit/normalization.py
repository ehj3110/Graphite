"""
Graphite Implicit Engine - SDF Normalization & Analytical Gradients.

Converts raw trigonometric TPMS scalar fields into metric-accurate Signed Distance
Fields (SDFs) satisfying ||grad phi|| approx 1 near the zero-crossing, enabling
robust blending, exact physical wall thickness control, and topological integrity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from graphite.math.tpms import evaluate_tpms


@dataclass(frozen=True)
class TPMSNormalizedFieldConfig:
    """
    Configuration for evaluating a metric-normalized TPMS Signed Distance Field.

    Parameters
    ----------
    lattice_type : str
        TPMS architecture name (e.g. 'Gyroid', 'Schwarz-P', 'Diamond').
    unit_cell_size : float
        Physical unit cell period L in millimeters.
    wall_thickness : float
        Physical target wall thickness in millimeters for sheet lattices.
    is_sheet : bool
        If True, represents a thickened sheet slab (|phi| - t/2 <= 0).
        If False, represents a skeletal network (phi <= 0).
    epsilon : float
        Regularization parameter to avoid division by zero at extrema (default 1e-7).
    clip_distance : float or None
        Metric distance bound to clamp normalized SDF. Defaults to L / 2.0 if None.
    """

    lattice_type: str = "Gyroid"
    unit_cell_size: float = 4.0
    wall_thickness: float = 0.5
    is_sheet: bool = True
    epsilon: float = 1e-7
    clip_distance: float | None = None


def compute_tpms_gradient_analytical(
    lattice_type: str,
    k: float,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute exact closed-form spatial gradient (dF/dx, dF/dy, dF/dz) for supported TPMS.

    Parameters
    ----------
    lattice_type : str
        TPMS family name ('Gyroid', 'Schwarz-P', 'Diamond').
    k : float
        Spatial wavenumber (2 * pi / unit_cell_size) in rad/mm.
    X, Y, Z : np.ndarray
        Cartesian coordinate meshgrids in world millimeters.

    Returns
    -------
    tuple of 3 ndarrays
        The gradient components (dF/dX, dF/dY, dF/dZ) in mm^-1.

    Raises
    ------
    NotImplementedError
        If an analytical gradient has not been implemented for the specified lattice type.
    """
    l_type = lattice_type.strip().lower()
    kx, ky, kz = k * X, k * Y, k * Z

    if l_type == "gyroid":
        sin_kx, cos_kx = np.sin(kx), np.cos(kx)
        sin_ky, cos_ky = np.sin(ky), np.cos(ky)
        sin_kz, cos_kz = np.sin(kz), np.cos(kz)

        dF_dx = k * (cos_kx * cos_ky - sin_kz * sin_kx)
        dF_dy = k * (cos_ky * cos_kz - sin_kx * sin_ky)
        dF_dz = k * (cos_kz * cos_kx - sin_ky * sin_kz)
        return dF_dx, dF_dy, dF_dz

    elif l_type in ("schwarz-p", "schwarz primitive", "schwarz", "primitive"):
        dF_dx = -k * np.sin(kx)
        dF_dy = -k * np.sin(ky)
        dF_dz = -k * np.sin(kz)
        return dF_dx, dF_dy, dF_dz

    elif l_type in ("diamond", "schwarz_d"):
        sin_kx, cos_kx = np.sin(kx), np.cos(kx)
        sin_ky, cos_ky = np.sin(ky), np.cos(ky)
        sin_kz, cos_kz = np.sin(kz), np.cos(kz)

        dF_dx = k * (
            cos_kx * (sin_ky * sin_kz + cos_ky * cos_kz)
            - sin_kx * (sin_ky * cos_kz + cos_ky * sin_kz)
        )
        dF_dy = k * (
            cos_ky * (sin_kx * sin_kz + cos_kx * cos_kz)
            - sin_ky * (sin_kx * cos_kz + cos_kx * sin_kz)
        )
        dF_dz = k * (
            cos_kz * (sin_kx * sin_ky + cos_kx * cos_ky)
            - sin_kz * (sin_kx * cos_ky + cos_kx * sin_ky)
        )
        return dF_dx, dF_dy, dF_dz

    elif l_type in ("schwarz-diamond", "schwarz diamond"):
        sin_kx, cos_kx = np.sin(kx), np.cos(kx)
        sin_ky, cos_ky = np.sin(ky), np.cos(ky)
        sin_kz, cos_kz = np.sin(kz), np.cos(kz)

        dF_dx = -k * (sin_kx * cos_ky * cos_kz + cos_kx * sin_ky * sin_kz)
        dF_dy = -k * (cos_kx * sin_ky * cos_kz + sin_kx * cos_ky * sin_kz)
        dF_dz = -k * (cos_kx * cos_ky * sin_kz + sin_kx * sin_ky * cos_kz)
        return dF_dx, dF_dy, dF_dz

    elif l_type in ("split-p", "split_p"):
        sin_kx, cos_kx = np.sin(kx), np.cos(kx)
        sin_ky, cos_ky = np.sin(ky), np.cos(ky)
        sin_kz, cos_kz = np.sin(kz), np.cos(kz)

        sin_2kx, cos_2kx = np.sin(2.0 * kx), np.cos(2.0 * kx)
        sin_2ky, cos_2ky = np.sin(2.0 * ky), np.cos(2.0 * ky)
        sin_2kz, cos_2kz = np.sin(2.0 * kz), np.cos(2.0 * kz)

        dt1_dx = k * (
            2.0 * cos_2kx * sin_kz * cos_ky
            + sin_2ky * cos_kx * cos_kz
            - sin_2kz * sin_ky * sin_kx
        )
        dt2_dx = -2.0 * k * sin_2kx * (cos_2ky + cos_2kz)
        dt3_dx = -2.0 * k * sin_2kx
        dF_dx = 1.1 * dt1_dx - 0.2 * dt2_dx - 0.4 * dt3_dx

        dt1_dy = k * (
            -sin_2kx * sin_kz * sin_ky
            + 2.0 * cos_2ky * sin_kx * cos_kz
            + sin_2kz * cos_ky * cos_kx
        )
        dt2_dy = -2.0 * k * sin_2ky * (cos_2kx + cos_2kz)
        dt3_dy = -2.0 * k * sin_2ky
        dF_dy = 1.1 * dt1_dy - 0.2 * dt2_dy - 0.4 * dt3_dy

        dt1_dz = k * (
            sin_2kx * cos_kz * cos_ky
            - sin_2ky * sin_kx * sin_kz
            + 2.0 * cos_2kz * sin_ky * cos_kx
        )
        dt2_dz = -2.0 * k * sin_2kz * (cos_2ky + cos_2kx)
        dt3_dz = -2.0 * k * sin_2kz
        dF_dz = 1.1 * dt1_dz - 0.2 * dt2_dz - 0.4 * dt3_dz

        return dF_dx, dF_dy, dF_dz

    else:
        raise NotImplementedError(
            f"Analytical gradient not implemented for lattice type '{lattice_type}'. "
            "Use compute_field_gradient_numerical as fallback."
        )


def compute_field_gradient_numerical(
    field: np.ndarray,
    spacing: tuple[float, float, float] | Sequence[float] | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spatial gradient (dF/dx, dF/dy, dF/dz) using second-order central differences.

    Parameters
    ----------
    field : np.ndarray
        3D scalar field array (indexing='ij').
    spacing : tuple of floats or float
        Voxel spacing (dx, dy, dz) in millimeters.

    Returns
    -------
    tuple of 3 ndarrays
        The numerical gradient components (dF/dx, dF/dy, dF/dz).
    """
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got {field.ndim}D array")

    if isinstance(spacing, (int, float)):
        dx = dy = dz = float(spacing)
    else:
        dx, dy, dz = float(spacing[0]), float(spacing[1]), float(spacing[2])

    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError(f"Spacing components must be positive, got ({dx}, {dy}, {dz})")

    # np.gradient for 3D array along axis 0, 1, 2
    gx, gy, gz = np.gradient(field, dx, dy, dz, edge_order=2)
    return gx, gy, gz


def normalize_tpms_sdf(
    field: np.ndarray,
    spacing: tuple[float, float, float] | Sequence[float] | float,
    grad: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    clip_distance: float | None = None,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Normalize a raw trigonometric scalar field F(x) into an approximate metric SDF.

    Uses first-order distance normalization:
        phi_tilde(x) = F(x) / sqrt(||grad F(x)||^2 + epsilon)

    Parameters
    ----------
    field : np.ndarray
        Raw trigonometric level-set field (F = 0 at mid-surface).
    spacing : tuple of floats or float
        Voxel grid resolution (dx, dy, dz) in millimeters.
    grad : tuple of 3 ndarrays, optional
        Pre-computed spatial gradient components (dF/dx, dF/dy, dF/dz).
        If None, computed numerically using compute_field_gradient_numerical.
    clip_distance : float, optional
        Metric distance threshold to clamp the normalized SDF.
        If provided, clamps output to [-clip_distance, +clip_distance] to prevent
        field blowup at extrema where ||grad F|| -> 0.
    epsilon : float, optional
        Regularization constant to prevent division by zero near extrema (default 1e-7).

    Returns
    -------
    np.ndarray
        Metric-normalized signed distance field with ||grad phi|| approx 1.
    """
    if grad is None:
        gx, gy, gz = compute_field_gradient_numerical(field, spacing)
    else:
        gx, gy, gz = grad

    grad_sq = gx * gx + gy * gy + gz * gz
    denom = np.sqrt(grad_sq + float(epsilon))

    norm_sdf = field / denom

    if clip_distance is not None:
        c = abs(float(clip_distance))
        norm_sdf = np.clip(norm_sdf, -c, c)

    return norm_sdf


def evaluate_normalized_tpms_mid_sdf(
    config: TPMSNormalizedFieldConfig,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    spacing: tuple[float, float, float] | Sequence[float] | float,
) -> np.ndarray:
    """
    Evaluate and normalize the mid-surface Signed Distance Field for a TPMS configuration.

    Parameters
    ----------
    config : TPMSNormalizedFieldConfig
        Unit cell and lattice configuration.
    X, Y, Z : np.ndarray
        Cartesian coordinate meshgrids in millimeters.
    spacing : tuple of floats or float
        Voxel resolution in millimeters.

    Returns
    -------
    np.ndarray
        Metric-accurate mid-surface signed distance field phi_mid (phi = 0 at mid-surface).
    """
    L = float(config.unit_cell_size)
    if L <= 0.0:
        raise ValueError(f"unit_cell_size must be positive, got {L}")

    k = 2.0 * np.pi / L
    raw_field = evaluate_tpms(config.lattice_type, k, X, Y, Z)

    # Attempt analytical gradient first; fallback to numerical
    try:
        grad = compute_tpms_gradient_analytical(config.lattice_type, k, X, Y, Z)
    except NotImplementedError:
        grad = compute_field_gradient_numerical(raw_field, spacing)

    clip_dist = config.clip_distance if config.clip_distance is not None else (L / 2.0)

    norm_sdf = normalize_tpms_sdf(
        field=raw_field,
        spacing=spacing,
        grad=grad,
        clip_distance=clip_dist,
        epsilon=config.epsilon,
    )
    return norm_sdf


def create_thickened_shell_sdf(
    normalized_mid_sdf: np.ndarray,
    wall_thickness_mm: float | np.ndarray,
) -> np.ndarray:
    """
    Construct a solid sheet representation from a mid-surface SDF.

    Parameters
    ----------
    normalized_mid_sdf : np.ndarray
        Metric signed distance field to the mid-surface (mm).
    wall_thickness_mm : float or np.ndarray
        Physical wall thickness t in millimeters.

    Returns
    -------
    np.ndarray
        The thickened solid sheet field:
            Phi_shell = |normalized_mid_sdf| - t / 2.0
        where solid is defined by Phi_shell <= 0.
    """
    half_t = np.asarray(wall_thickness_mm, dtype=np.float64) / 2.0
    return np.abs(normalized_mid_sdf) - half_t
