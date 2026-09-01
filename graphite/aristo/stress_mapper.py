"""
Aristo Stress Mapper — Von Mises to Gradient Field Adapter

Maps FEA stress results from fine tetrahedral meshes onto Graphite's
target domains (implicit voxel grids or explicit struts).

Phase 1 focus: Implicit TPMS engine (solid fraction gradient).
Enforces hard manufacturing constraints:
  - Solid fraction (Vf) capped at 50%
  - Implicit unit-cell size floor of 1.0 mm

Author: Graphite / Aristo Project
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_solver import AristoResult


def stress_to_vf_gradient(
    aristo_result: AristoResult,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    config: AristoConfig,
) -> np.ndarray:
    """
    Rasterizes von Mises stress onto a voxel grid → solid fraction field.

    Uses nodal stress (normalized) when ``stress_field_mode == nodal_averaged``,
    otherwise nearest element centroid with element-normalized stress.
    """
    fea_nodes = aristo_result.fea_nodes
    grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    if (
        aristo_result.stress_field_mode == "nodal_averaged"
        and aristo_result.von_mises_nodal_norm.size == fea_nodes.shape[0]
    ):
        tree = cKDTree(fea_nodes)
        _dists, nearest = tree.query(grid_points, k=1)
        voxel_stress = aristo_result.von_mises_nodal_norm[nearest]
    else:
        fea_elems = aristo_result.fea_elements
        centroids = fea_nodes[fea_elems].mean(axis=1)
        stress = aristo_result.von_mises
        if (
            aristo_result.quality_mask.size == stress.size
            and aristo_result.quality_mask.any()
        ):
            valid = aristo_result.quality_mask
            centroids = centroids[valid]
            stress = stress[valid]
        tree = cKDTree(centroids)
        _dists, nearest = tree.query(grid_points, k=1)
        voxel_stress = stress[nearest]

    vf_floor = config.vf_floor
    vf_cap = min(config.vf_cap, 0.50)
    voxel_vf = vf_floor + (vf_cap - vf_floor) * voxel_stress
    return voxel_vf.reshape(X.shape).astype(np.float32)


def stress_to_strut_radius_map(
    aristo_result: AristoResult,
    strut_midpoints: np.ndarray,
    r_min: float,
    r_max: float,
    quantile_low: float = 0.25,
    quantile_high: float = 0.75,
    mapping: str = "linear",
    zone_mask: Any | None = None,
) -> np.ndarray:
    """
    Maps FEA Von Mises stress to per-strut radii for explicit wireframe lattices.

    Parameters
    ----------
    aristo_result : AristoResult
        FEA result container from run_aristo().
    strut_midpoints : np.ndarray
        (S, 3) array of 3D midpoint coordinates for each strut segment.
    r_min : float
        Minimum strut radius (mm) applied at low stress regions.
    r_max : float
        Maximum strut radius (mm) applied at high stress regions.
    quantile_low : float
        Lower stress quantile threshold for r_min clamping (default: 0.25).
    quantile_high : float
        Upper stress quantile threshold for r_max clamping (default: 0.75).
    mapping : str
        "linear" (continuous smooth ramp) or "step" (4-bin discrete step function).
    zone_mask : ZoneMask | None
        Optional ZoneMask filter. When provided, stress percentiles (quantiles)
        are computed ONLY using FEA nodes/elements falling inside the zone.

    Returns
    -------
    radii : np.ndarray
        (S,) array of float64 strut radii in mm.
    """
    midpoints = np.asarray(strut_midpoints, dtype=np.float64)
    if midpoints.size == 0:
        return np.empty(0, dtype=np.float64)

    fea_nodes = aristo_result.fea_nodes

    if (
        aristo_result.stress_field_mode == "nodal_averaged"
        and aristo_result.von_mises_nodal_raw.size == fea_nodes.shape[0]
    ):
        if zone_mask is not None:
            in_zone = zone_mask.contains(fea_nodes)
            if not in_zone.any():
                in_zone = np.ones(fea_nodes.shape[0], dtype=bool)
            target_nodes = fea_nodes[in_zone]
            target_stress = aristo_result.von_mises_nodal_raw[in_zone]
        else:
            target_nodes = fea_nodes
            target_stress = aristo_result.von_mises_nodal_raw

        tree = cKDTree(target_nodes)
        _dists, nearest = tree.query(midpoints, k=1)
        raw_stress = target_stress[nearest]
        quantile_stress = target_stress
    else:
        fea_elems = aristo_result.fea_elements
        centroids = fea_nodes[fea_elems].mean(axis=1)
        stress = aristo_result.von_mises
        if (
            aristo_result.quality_mask.size == stress.size
            and aristo_result.quality_mask.any()
        ):
            valid = aristo_result.quality_mask
            centroids = centroids[valid]
            stress = stress[valid]

        if zone_mask is not None:
            in_zone = zone_mask.contains(centroids)
            if not in_zone.any():
                in_zone = np.ones(centroids.shape[0], dtype=bool)
            target_centroids = centroids[in_zone]
            target_stress = stress[in_zone]
        else:
            target_centroids = centroids
            target_stress = stress

        tree = cKDTree(target_centroids)
        _dists, nearest = tree.query(midpoints, k=1)
        raw_stress = target_stress[nearest]
        quantile_stress = target_stress

    if mapping == "step":
        q25, q50, q75 = np.quantile(quantile_stress, [0.25, 0.50, 0.75])
        step_delta = (r_max - r_min) / 3.0
        radii = np.where(
            raw_stress < q25,
            r_min,
            np.where(
                raw_stress < q50,
                r_min + step_delta,
                np.where(raw_stress < q75, r_min + 2.0 * step_delta, r_max),
            ),
        )
    else:
        # Linear mode (default)
        q_low = float(np.quantile(quantile_stress, quantile_low))
        q_high = float(np.quantile(quantile_stress, quantile_high))
        if q_high > q_low:
            norm_stress = np.clip((raw_stress - q_low) / (q_high - q_low), 0.0, 1.0)
        else:
            norm_stress = np.zeros_like(raw_stress)
        radii = r_min + (r_max - r_min) * norm_stress

    return radii.astype(np.float64)

