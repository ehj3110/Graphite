"""
Piecewise-constant unit-cell grading along Z (hard band transitions).

Builds Split-P (and similar TPMS) implicit fields with discrete L(z) and τ(z)
bands. Prefer **single-pass** full-domain EDT for volume-mesh-safe STLs; optional
per-band union remains for legacy / comparison paths.

See docs/PIECEWISE_PRISM_LATTICE_GENERATION.md and docs/IMPLICIT_TO_VOLUME_MESHING.md.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from graphite.geometry.masking import (
    axis_aligned_box_grid,
    axis_aligned_box_sdf,
    voxelize_mesh_and_edt,
)
from graphite.math.tpms import evaluate_tpms_phase


def cumulative_phase_w_from_l_profile(
    height_mm: float,
    z_breaks_mm: Sequence[float],
    L_mm: Sequence[float],
    *,
    n_samples: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Jacobian-integrated axial phase W(z) for piecewise-constant L bands.

    ``z_breaks`` has length ``len(L_mm) + 1`` (monotone from 0 to height).
    """
    z_breaks = [float(z) for z in z_breaks_mm]
    L_vals = [float(L) for L in L_mm]
    if len(z_breaks) != len(L_vals) + 1:
        raise ValueError("z_breaks must have len(L_mm) + 1 entries.")

    z_dense = np.linspace(0.0, float(height_mm), int(n_samples))
    L_dense = np.full_like(z_dense, L_vals[-1])
    for i, L in enumerate(L_vals):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        if i < len(L_vals) - 1:
            mask = (z_dense >= z0 - 1e-12) & (z_dense < z1)
        else:
            mask = (z_dense >= z0 - 1e-12) & (z_dense <= z1 + 1e-12)
        L_dense[mask] = L

    omega_dense = 2.0 * np.pi / np.maximum(L_dense, 1e-6)
    W_dense = np.zeros_like(z_dense)
    if len(z_dense) > 1:
        W_dense[1:] = np.cumsum(
            0.5 * (omega_dense[1:] + omega_dense[:-1]) * np.diff(z_dense)
        )
    return z_dense, W_dense


def _assign_band_fields(
    Z: np.ndarray,
    z_breaks_mm: Sequence[float],
    L_mm: Sequence[float],
    tau: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise L/τ on Z. Values below the first break use band 0; above the last use the final band."""
    z_breaks = [float(z) for z in z_breaks_mm]
    L_voxel = np.full_like(Z, float(L_mm[0]))
    tau_voxel = np.full_like(Z, float(tau[0]))
    for i, (L, t) in enumerate(zip(L_mm, tau, strict=True)):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        if i < len(L_mm) - 1:
            mask = (Z >= z0 - 1e-12) & (Z < z1)
        else:
            mask = (Z >= z0 - 1e-12) & (Z <= z1 + 1e-12)
        L_voxel[mask] = float(L)
        tau_voxel[mask] = float(t)
    above = Z > z_breaks[-1] + 1e-12
    L_voxel[above] = float(L_mm[-1])
    tau_voxel[above] = float(tau[-1])
    return L_voxel, tau_voxel


def _assign_band_scalar_fields(
    Z: np.ndarray,
    z_breaks_mm: Sequence[float],
    values: Sequence[float],
) -> np.ndarray:
    """Assign a piecewise-constant scalar field (e.g. per-band XY phase origin)."""
    z_breaks = [float(z) for z in z_breaks_mm]
    out = np.full_like(Z, float(values[0]))
    for i, value in enumerate(values):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        if i < len(values) - 1:
            mask = (Z >= z0 - 1e-12) & (Z < z1)
        else:
            mask = (Z >= z0 - 1e-12) & (Z <= z1 + 1e-12)
        out[mask] = float(value)
    out[Z > z_breaks[-1] + 1e-12] = float(values[-1])
    return out


def splitp_piecewise_cylinder_single_pass(
    *,
    radius_mm: float,
    height_mm: float,
    resolution_mm: float,
    z_breaks_mm: Sequence[float],
    L_mm: Sequence[float],
    tau: Sequence[float],
    lattice_type: str = "split-p",
) -> tuple[trimesh.Trimesh, dict]:
    """
    One full-cylinder EDT field with piecewise L/τ — recommended for Gmsh volume mesh.
    """
    if len(z_breaks_mm) != len(L_mm) + 1 or len(L_mm) != len(tau):
        raise ValueError("z_breaks, L_mm, and tau length mismatch.")

    z_dense, W_dense = cumulative_phase_w_from_l_profile(height_mm, z_breaks_mm, L_mm)
    boundary = trimesh.creation.cylinder(
        radius=float(radius_mm), height=float(height_mm), sections=96
    )
    boundary.apply_translation([0.0, 0.0, 0.5 * float(height_mm)])

    X, Y, Z, cad_sdf, padded_min_bound, _max_bound, _nx, _ny, _nz = (
        voxelize_mesh_and_edt(boundary, float(resolution_mm))
    )
    L_voxel, tau_voxel = _assign_band_fields(Z, z_breaks_mm, L_mm, tau)
    omega = 2.0 * np.pi / np.maximum(L_voxel, 1e-6)
    W_phase = np.interp(Z, z_dense, W_dense)
    U = X * omega
    V = Y * omega
    F = evaluate_tpms_phase(lattice_type, U, V, W_phase)
    final_field = np.maximum(np.abs(F) - tau_voxel, cad_sdf)

    verts, faces, _n, _v = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution_mm, resolution_mm, resolution_mm),
    )
    verts = verts + padded_min_bound
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    meta = {
        "pipeline": "full-cylinder EDT + piecewise L/tau; single pass",
        "combine_method": "single_pass_implicit",
        "z_breaks_mm": list(z_breaks_mm),
        "L_mm": list(L_mm),
        "tau": list(tau),
        "n_bands": len(L_mm),
        "lattice_type": lattice_type,
    }
    return mesh_out, meta


def splitp_piecewise_box_single_pass(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    height_mm: float,
    resolution_mm: float,
    z_breaks_mm: Sequence[float],
    L_mm: Sequence[float],
    tau: Sequence[float],
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
    lattice_type: str = "split-p",
    band_phase_origin_x_mm: Sequence[float] | None = None,
    band_phase_origin_y_mm: Sequence[float] | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Axis-aligned box domain with piecewise L/τ along Z (analytic flat caps).

    ``band_phase_origin_*`` are per-band lateral phase shifts (mm) applied as
    ``U = (X - phase_x) * omega``, ``V = (Y - phase_y) * omega``. Use half the
    bottom-band unit cell (``L_bottom / 4``) to stagger the smaller-pore region.
    """
    if len(z_breaks_mm) != len(L_mm) + 1 or len(L_mm) != len(tau):
        raise ValueError("z_breaks, L_mm, and tau length mismatch.")

    n_bands = len(L_mm)
    phase_x = (
        [0.0] * n_bands
        if band_phase_origin_x_mm is None
        else [float(v) for v in band_phase_origin_x_mm]
    )
    phase_y = (
        [0.0] * n_bands
        if band_phase_origin_y_mm is None
        else [float(v) for v in band_phase_origin_y_mm]
    )
    if len(phase_x) != n_bands or len(phase_y) != n_bands:
        raise ValueError("band_phase_origin_* must have one entry per band.")

    ox, oy, oz = float(origin_x), float(origin_y), float(origin_z)
    wx, wy, hz = float(width_x_mm), float(depth_y_mm), float(height_mm)
    res = float(resolution_mm)

    z_dense, W_dense = cumulative_phase_w_from_l_profile(height_mm, z_breaks_mm, L_mm)
    X, Y, Z, grid_origin, spacing = axis_aligned_box_grid(
        wx, wy, hz, res, origin_x=ox, origin_y=oy, origin_z=oz
    )
    box_sdf = axis_aligned_box_sdf(
        X,
        Y,
        Z,
        origin_x=ox,
        origin_y=oy,
        origin_z=oz,
        width_x_mm=wx,
        depth_y_mm=wy,
        height_z_mm=hz,
    )
    L_voxel, tau_voxel = _assign_band_fields(Z, z_breaks_mm, L_mm, tau)
    phase_x_voxel = _assign_band_scalar_fields(Z, z_breaks_mm, phase_x)
    phase_y_voxel = _assign_band_scalar_fields(Z, z_breaks_mm, phase_y)
    omega = 2.0 * np.pi / np.maximum(L_voxel, 1e-6)
    W_phase = np.interp(Z, z_dense, W_dense)
    U = (X - phase_x_voxel) * omega
    V = (Y - phase_y_voxel) * omega
    F = evaluate_tpms_phase(lattice_type, U, V, W_phase)
    final_field = np.maximum(np.abs(F) - tau_voxel, box_sdf)

    verts, faces, _n, _v = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=spacing,
    )
    verts = verts + grid_origin
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    meta = {
        "pipeline": "analytic box SDF + piecewise L/tau; single pass",
        "boundary_sdf": "analytic_box",
        "combine_method": "single_pass_implicit",
        "z_breaks_mm": list(z_breaks_mm),
        "L_mm": list(L_mm),
        "tau": list(tau),
        "n_bands": len(L_mm),
        "lattice_type": lattice_type,
        "origin_mm": [ox, oy, oz],
        "extents_mm": [wx, wy, hz],
        "band_phase_origin_x_mm": phase_x,
        "band_phase_origin_y_mm": phase_y,
    }
    return mesh_out, meta
