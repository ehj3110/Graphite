"""
Graphite Implicit Engine - Pore Metrics

This module provides tools for geometric analysis of TPMS lattices, specifically 
focusing on computing maximum inscribed sphere (MIS) pore sizes and estimating 
strut/wall thicknesses to recommend safe voxel resolutions for meshing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt as edt
import trimesh

from graphite.math.tpms import evaluate_tpms


@dataclass(frozen=True)
class EffectivePoreSizeResult:
    effective_global_mm: float
    effective_p10_mm: float
    effective_p50_mm: float
    effective_p90_mm: float
    effective_by_z_mm: list[float]
    z_bin_centers_mm: list[float]
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class CrossSectionalPoreSizeResult:
    pore_size_input_mm: float
    solid_fraction: float
    cross_section_2d_mm: float | None
    cross_section_3d_mm: float | None
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class ZGradedPoreMetricsResult:
    z_samples_mm: list[float]
    cross_section_2d_mm: list[float | None]
    cross_section_3d_mm: list[float | None]
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class WallThicknessResolutionResult:
    estimated_wall_thickness_mm: float
    recommended_resolution_mm: float
    multiplier: float
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class MaxInscribedSphereResult:
    max_diameter_mm: float
    center_ijk: tuple[int, int, int] | None
    confidence: str
    notes: list[str]


def _confidence_from_ratio(max_diameter_mm: float, resolution_mm: float) -> tuple[str, list[str]]:
    notes: list[str] = []
    voxels_per_diameter = max_diameter_mm / max(resolution_mm, 1e-9)
    if voxels_per_diameter < 2.0:
        notes.append("Very coarse sampling: diameter < 2 voxels.")
        return "low", notes
    if voxels_per_diameter < 3.0:
        notes.append("Coarse sampling: diameter < 3 voxels.")
        return "medium", notes
    return "high", notes


def _safe_percentile(values: np.ndarray, pct: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, pct))


def recommend_resolution_from_wall_thickness(
    mesh_or_path: trimesh.Trimesh | str,
    multiplier: float = 0.5,
    probe_resolution_mm: float = 0.05,
    thickness_percentile: float = 10.0,
    min_resolution_mm: float = 0.005,
    max_resolution_mm: float = 1.0,
) -> WallThicknessResolutionResult:
    """
    Estimate wall thickness from a provisional voxelization and recommend a resolution.

    The recommendation follows:
        recommended_resolution_mm = multiplier * estimated_wall_thickness_mm

    Default multiplier=0.5 means at least two voxels across estimated wall thickness.

    Parameters
    ----------
    mesh_or_path : trimesh.Trimesh or str
        The boundary mesh or path to an STL file.
    multiplier : float, optional
        Target ratio of resolution to wall thickness, by default 0.5.
    probe_resolution_mm : float, optional
        Resolution of the temporary voxelization used for estimation, by default 0.05.
    thickness_percentile : float, optional
        The percentile of the EDT diameter distribution to use as the estimated 
        wall thickness, by default 10.0 (conservative).
    min_resolution_mm : float, optional
        Hard minimum limit for the recommended resolution, by default 0.005.
    max_resolution_mm : float, optional
        Hard maximum limit for the recommended resolution, by default 1.0.

    Returns
    -------
    WallThicknessResolutionResult
        A dataclass containing the estimated thickness, recommended resolution, 
        and confidence metrics.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")
    if probe_resolution_mm <= 0:
        raise ValueError("probe_resolution_mm must be > 0")
    if not (0.0 < thickness_percentile <= 100.0):
        raise ValueError("thickness_percentile must be within (0, 100]")
    if min_resolution_mm <= 0 or max_resolution_mm <= 0:
        raise ValueError("resolution clamps must be > 0")
    if min_resolution_mm > max_resolution_mm:
        raise ValueError("min_resolution_mm must be <= max_resolution_mm")

    mesh = trimesh.load(str(mesh_or_path)) if isinstance(mesh_or_path, str) else mesh_or_path
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    vox = mesh.voxelized(pitch=probe_resolution_mm).fill()
    solid = np.asarray(vox.matrix, dtype=bool)
    dist = edt(solid) * probe_resolution_mm
    diameters = 2.0 * dist[solid]
    if diameters.size == 0:
        return WallThicknessResolutionResult(
            estimated_wall_thickness_mm=0.0,
            recommended_resolution_mm=max_resolution_mm,
            multiplier=multiplier,
            confidence="low",
            notes=["Could not estimate thickness from voxelized solid mask."],
        )

    est = float(np.percentile(diameters, thickness_percentile))
    raw_recommended = multiplier * est
    recommended = float(np.clip(raw_recommended, min_resolution_mm, max_resolution_mm))
    confidence, notes = _confidence_from_ratio(est, probe_resolution_mm)
    if recommended != raw_recommended:
        notes.append("Recommended resolution was clamped by min/max limits.")

    return WallThicknessResolutionResult(
        estimated_wall_thickness_mm=est,
        recommended_resolution_mm=recommended,
        multiplier=multiplier,
        confidence=confidence,
        notes=notes,
    )


def compute_effective_pore_size(
    void_mask: np.ndarray,
    resolution_mm: float,
    z_bins: int = 12,
) -> EffectivePoreSizeResult:
    """
    Measure effective pore size from realized void geometry.

    Parameters
    ----------
    void_mask : ndarray
        3D boolean array where True indicates void voxels.
    resolution_mm : float
        Voxel pitch in mm.
    z_bins : int, optional
        Number of bins for Z-profile of effective pore size, by default 12.

    Returns
    -------
    EffectivePoreSizeResult
        A dataclass containing global and Z-binned effective pore size metrics 
        (e.g., p10, p50, p90 percentiles).
    """
    if void_mask.ndim != 3:
        raise ValueError("void_mask must be a 3D boolean array")
    if resolution_mm <= 0:
        raise ValueError("resolution_mm must be > 0")

    void_mask = np.asarray(void_mask, dtype=bool)
    distances = edt(void_mask) * resolution_mm
    diameters = 2.0 * distances[void_mask]

    effective_global_mm = float(np.max(diameters)) if diameters.size else 0.0
    effective_p10_mm = _safe_percentile(diameters, 10.0)
    effective_p50_mm = _safe_percentile(diameters, 50.0)
    effective_p90_mm = _safe_percentile(diameters, 90.0)

    nz = void_mask.shape[2]
    z_edges = np.linspace(0, nz, max(int(z_bins), 1) + 1, dtype=int)
    effective_by_z_mm: list[float] = []
    z_bin_centers_mm: list[float] = []
    for idx in range(len(z_edges) - 1):
        z0 = z_edges[idx]
        z1 = z_edges[idx + 1]
        if z1 <= z0:
            continue
        slab = distances[:, :, z0:z1]
        slab_mask = void_mask[:, :, z0:z1]
        slab_values = 2.0 * slab[slab_mask]
        effective_by_z_mm.append(float(np.max(slab_values)) if slab_values.size else 0.0)
        z_center = 0.5 * ((z0 + z1 - 1) * resolution_mm)
        z_bin_centers_mm.append(float(z_center))

    confidence, notes = _confidence_from_ratio(effective_global_mm, resolution_mm)
    if diameters.size == 0:
        notes.append("No void voxels were found in the provided mask.")

    return EffectivePoreSizeResult(
        effective_global_mm=effective_global_mm,
        effective_p10_mm=effective_p10_mm,
        effective_p50_mm=effective_p50_mm,
        effective_p90_mm=effective_p90_mm,
        effective_by_z_mm=effective_by_z_mm,
        z_bin_centers_mm=z_bin_centers_mm,
        confidence=confidence,
        notes=notes,
    )


def compute_max_inscribed_sphere_pore_size(
    void_mask: np.ndarray,
    resolution_mm: float,
    boundary_guard_mm: float = 0.0,
    require_sphere_within_domain: bool = True,
) -> MaxInscribedSphereResult:
    """
    Compute pore size as maximal inscribed sphere diameter in the void.

    Boundary handling:
    - `boundary_guard_mm` excludes sphere centers near domain boundaries.
    - `require_sphere_within_domain=True` also rejects candidates where
      sphere radius would extend beyond the sampled domain bounds.

    Parameters
    ----------
    void_mask : ndarray
        3D boolean array where True indicates void voxels.
    resolution_mm : float
        Voxel pitch in mm.
    boundary_guard_mm : float, optional
        Distance from the grid boundary to exclude sphere centers, by default 0.0.
    require_sphere_within_domain : bool, optional
        If True, rejects any sphere whose radius exceeds its distance to the 
        domain boundary, by default True.

    Returns
    -------
    MaxInscribedSphereResult
        A dataclass containing the maximum diameter and the IJK center index.
    """
    if void_mask.ndim != 3:
        raise ValueError("void_mask must be a 3D boolean array")
    if resolution_mm <= 0:
        raise ValueError("resolution_mm must be > 0")
    if boundary_guard_mm < 0:
        raise ValueError("boundary_guard_mm must be >= 0")

    void_mask = np.asarray(void_mask, dtype=bool)
    if not np.any(void_mask):
        return MaxInscribedSphereResult(
            max_diameter_mm=0.0,
            center_ijk=None,
            confidence="low",
            notes=["No void voxels found."],
        )

    d_void = edt(void_mask) * resolution_mm
    nx, ny, nz = void_mask.shape
    ii, jj, kk = np.indices(void_mask.shape)

    d_boundary = np.minimum.reduce(
        [
            ii.astype(float) * resolution_mm,
            (nx - 1 - ii).astype(float) * resolution_mm,
            jj.astype(float) * resolution_mm,
            (ny - 1 - jj).astype(float) * resolution_mm,
            kk.astype(float) * resolution_mm,
            (nz - 1 - kk).astype(float) * resolution_mm,
        ]
    )

    candidate = void_mask & (d_boundary >= boundary_guard_mm)
    if require_sphere_within_domain:
        candidate &= d_void <= d_boundary

    notes: list[str] = []
    if not np.any(candidate):
        return MaxInscribedSphereResult(
            max_diameter_mm=0.0,
            center_ijk=None,
            confidence="low",
            notes=[
                "No valid MIS candidate remained after boundary filtering. "
                "Try reducing boundary_guard_mm."
            ],
        )

    radii = np.where(candidate, d_void, -1.0)
    flat_idx = int(np.argmax(radii))
    max_radius = float(radii.flat[flat_idx])
    i, j, k = np.unravel_index(flat_idx, radii.shape)

    conf, conf_notes = _confidence_from_ratio(2.0 * max_radius, resolution_mm)
    notes.extend(conf_notes)
    if boundary_guard_mm > 0:
        notes.append(f"Applied boundary_guard_mm={boundary_guard_mm:.4f}.")
    if require_sphere_within_domain:
        notes.append("Rejected spheres that would extend beyond domain bounds.")

    return MaxInscribedSphereResult(
        max_diameter_mm=2.0 * max_radius,
        center_ijk=(int(i), int(j), int(k)),
        confidence=conf,
        notes=notes,
    )


def _parent_cell_void_mask(
    lattice_type: str,
    pore_size_mm: float,
    solid_fraction: float,
    resolution_mm: float,
) -> np.ndarray:
    if pore_size_mm <= 0:
        raise ValueError("pore_size_mm must be > 0")
    if resolution_mm <= 0:
        raise ValueError("resolution_mm must be > 0")
    if not (0.0 < solid_fraction < 1.0):
        raise ValueError("solid_fraction must be within (0, 1)")

    n = max(int(np.ceil(pore_size_mm / resolution_mm)), 8)
    axis = np.linspace(0.0, pore_size_mm, n, endpoint=False)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    k = (2.0 * np.pi) / pore_size_mm
    F = evaluate_tpms(lattice_type, k, X, Y, Z)
    threshold = abs(1.5 * (2.0 * solid_fraction - 1.0))
    solid_mask = np.abs(F) <= threshold
    return ~solid_mask


def compute_cross_sectional_pore_size(
    lattice_type: str,
    pore_size_mm: float,
    solid_fraction: float,
    unit_cell_resolution_mm: float = 0.02,
    compute_2d: bool = True,
    compute_3d: bool = True,
) -> CrossSectionalPoreSizeResult:
    """
    Compute cross-sectional pore size from a local parent unit cell.

    Returns both 2D (mid-plane inscribed circle) and 3D (inscribed sphere)
    variants when requested.

    Parameters
    ----------
    lattice_type : str
        TPMS equation type (e.g., 'Gyroid').
    pore_size_mm : float
        Target theoretical unit cell size (L).
    solid_fraction : float
        Target solid volume fraction threshold.
    unit_cell_resolution_mm : float, optional
        Voxel resolution for evaluating the unit cell, by default 0.02.
    compute_2d : bool, optional
        Whether to compute the 2D mid-plane inscribed circle, by default True.
    compute_3d : bool, optional
        Whether to compute the 3D inscribed sphere, by default True.

    Returns
    -------
    CrossSectionalPoreSizeResult
        A dataclass containing the computed 2D and 3D cross-sectional metrics.
    """
    if not compute_2d and not compute_3d:
        raise ValueError("At least one of compute_2d/compute_3d must be True")

    void_mask = _parent_cell_void_mask(
        lattice_type=lattice_type,
        pore_size_mm=pore_size_mm,
        solid_fraction=solid_fraction,
        resolution_mm=unit_cell_resolution_mm,
    )

    cross_section_2d_mm: float | None = None
    cross_section_3d_mm: float | None = None

    if compute_2d:
        z_mid = void_mask.shape[2] // 2
        slice_mask = void_mask[:, :, z_mid]
        d2 = edt(slice_mask) * unit_cell_resolution_mm
        cross_section_2d_mm = float(2.0 * np.max(d2)) if np.any(slice_mask) else 0.0

    if compute_3d:
        d3 = edt(void_mask) * unit_cell_resolution_mm
        cross_section_3d_mm = float(2.0 * np.max(d3)) if np.any(void_mask) else 0.0

    ref_diameter = pore_size_mm
    confidence, notes = _confidence_from_ratio(ref_diameter, unit_cell_resolution_mm)

    return CrossSectionalPoreSizeResult(
        pore_size_input_mm=pore_size_mm,
        solid_fraction=solid_fraction,
        cross_section_2d_mm=cross_section_2d_mm,
        cross_section_3d_mm=cross_section_3d_mm,
        confidence=confidence,
        notes=notes,
    )


def compute_pore_metrics_for_z_graded(
    lattice_type: str,
    z_samples_mm: np.ndarray | list[float],
    pore_sizes_mm: np.ndarray | list[float],
    solid_fraction: float | np.ndarray | list[float],
    unit_cell_resolution_mm: float = 0.02,
) -> ZGradedPoreMetricsResult:
    """
    Compute cross-sectional pore metrics over sampled Z values for a Z-graded design.

    Parameters
    ----------
    lattice_type : str
        TPMS equation type.
    z_samples_mm : ndarray or list of float
        Z-coordinates to sample the metrics at.
    pore_sizes_mm : ndarray or list of float
        Target pore sizes corresponding to each Z-sample.
    solid_fraction : float or ndarray or list of float
        Target solid volume fraction at each Z-sample.
    unit_cell_resolution_mm : float, optional
        Voxel resolution for local unit cell evaluation, by default 0.02.

    Returns
    -------
    ZGradedPoreMetricsResult
        A dataclass containing profiles of 2D and 3D cross-sectional pore sizes.
    """
    z = np.asarray(z_samples_mm, dtype=float).ravel()
    p = np.asarray(pore_sizes_mm, dtype=float).ravel()
    if z.size != p.size:
        raise ValueError("z_samples_mm and pore_sizes_mm must have the same length")
    if z.size == 0:
        raise ValueError("z_samples_mm must not be empty")

    if np.isscalar(solid_fraction):
        sf_arr = np.full_like(z, float(solid_fraction), dtype=float)
    else:
        sf_arr = np.asarray(solid_fraction, dtype=float).ravel()
        if sf_arr.size != z.size:
            raise ValueError("solid_fraction array must match z_samples_mm length")

    cross_2d: list[float | None] = []
    cross_3d: list[float | None] = []
    confidences: list[str] = []
    notes: list[str] = []

    for pore_val, sf_val in zip(p, sf_arr):
        result = compute_cross_sectional_pore_size(
            lattice_type=lattice_type,
            pore_size_mm=float(pore_val),
            solid_fraction=float(sf_val),
            unit_cell_resolution_mm=unit_cell_resolution_mm,
            compute_2d=True,
            compute_3d=True,
        )
        cross_2d.append(result.cross_section_2d_mm)
        cross_3d.append(result.cross_section_3d_mm)
        confidences.append(result.confidence)
        notes.extend(result.notes)

    if "low" in confidences:
        overall_conf = "low"
    elif "medium" in confidences:
        overall_conf = "medium"
    else:
        overall_conf = "high"

    unique_notes = sorted(set(notes))
    return ZGradedPoreMetricsResult(
        z_samples_mm=[float(v) for v in z],
        cross_section_2d_mm=cross_2d,
        cross_section_3d_mm=cross_3d,
        confidence=overall_conf,
        notes=unique_notes,
    )

