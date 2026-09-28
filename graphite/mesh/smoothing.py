"""
Graphite Mesh - Curvature Smoothing and Surface Fairing Engine

This module implements volume-preserving Taubin smoothing to relax TPMS and
spinodal metamaterial level-set surfaces toward constant mean curvature (CMC)
while strictly preventing the catastrophic shrinkage associated with standard
Laplacian filters.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import scipy.sparse as sp
import trimesh

logger = logging.getLogger(__name__)


def compute_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute vertex mean curvature H on a surface mesh.

    Uses PyVista/VTK surface curvature analysis if available, with a discrete
    Laplace-Beltrami fallback operator: H = 0.5 * dot(L(x), n).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input surface mesh.

    Returns
    -------
    np.ndarray, shape (n_vertices,), dtype float64
        Mean curvature evaluated at each vertex.
    """
    try:
        import pyvista as pv

        pv_mesh = pv.wrap(mesh)
        curv = pv_mesh.curvature("mean")
        if curv is not None and len(curv) == len(mesh.vertices):
            return np.asarray(curv, dtype=np.float64)
    except Exception:
        pass

    # Discrete Laplace-Beltrami mean curvature fallback
    try:
        import trimesh.smoothing

        lap = trimesh.smoothing.laplacian_calculation(mesh)
    except Exception:
        lap = _build_normalized_laplacian(mesh)

    V = mesh.vertices.view(np.ndarray)
    lap_v = lap.dot(V) - V
    normals = mesh.vertex_normals
    # Projected mean curvature scalar
    h = 0.5 * np.sum(lap_v * normals, axis=1)
    return np.asarray(h, dtype=np.float64)


def _build_normalized_laplacian(mesh: trimesh.Trimesh) -> sp.csr_matrix:
    """Construct degree-normalized sparse adjacency graph Laplacian operator."""
    edges = mesh.edges
    n_verts = len(mesh.vertices)
    if n_verts == 0 or len(edges) == 0:
        return sp.csr_matrix((n_verts, n_verts), dtype=np.float64)

    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(row), dtype=np.float64)
    adj = sp.coo_matrix((data, (row, col)), shape=(n_verts, n_verts)).tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0  # guard isolated vertices
    inv_deg = sp.diags(1.0 / deg)
    return inv_deg.dot(adj)


def _fallback_taubin_smooth(
    mesh: trimesh.Trimesh,
    iterations: int,
    lamb: float,
    nu_mag: float,
) -> None:
    """Explicit sparse-adjacency Laplacian Taubin filter."""
    lap = _build_normalized_laplacian(mesh)
    V = mesh.vertices.view(np.ndarray).astype(np.float64)

    for index in range(iterations):
        dot = lap.dot(V) - V
        if index % 2 == 0:
            V += lamb * dot
        else:
            V -= nu_mag * dot

    mesh.vertices = V


def smooth_mesh_taubin(
    mesh: trimesh.Trimesh,
    iterations: int = 15,
    lamb: float = 0.5,
    nu: float = -0.53,
    inplace: bool = False,
) -> trimesh.Trimesh:
    """
    Apply volume-preserving Taubin smoothing to a surface mesh.

    Taubin smoothing uses an alternating two-step filter (shrinkage followed by
    expansion) to eliminate high-frequency voxel terracing and curvature spikes
    while strictly preserving macro volume:
        x'  = x  + lambda * L(x)    (lambda > 0, shrinking pass)
        x'' = x' + mu * L(x')       (mu < -lambda < 0, anti-shrinking dilation pass)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangular surface mesh.
    iterations : int, optional
        Number of alternating filter cycles, by default 15. If an odd integer is
        provided, it is rounded up to the next even integer to ensure complete
        pairs of shrink and dilate passes.
    lamb : float, optional
        Positive pass step scale lambda (0 < lambda < 1), by default 0.5.
    nu : float, optional
        Negative deflation step scale mu (mu < -lambda), by default -0.53.
        Both negative convention (e.g. -0.53) and positive magnitude (0.53)
        are supported.
    inplace : bool, optional
        If True, modifies the input mesh in-place. If False (default), operates
        on a copy.

    Returns
    -------
    trimesh.Trimesh
        Smoothed surface mesh with preserved volume and relaxed curvature.

    Raises
    ------
    ValueError
        If lamb is outside (0, 1) or iterations <= 0.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be > 0, got {iterations}")
    if not (0.0 < lamb < 1.0):
        raise ValueError(f"lamb must be in the open interval (0, 1), got {lamb}")

    nu_mag = abs(float(nu))
    if nu_mag <= lamb:
        warnings.warn(
            f"Taubin stability condition |nu| > lamb not met (|nu|={nu_mag}, lamb={lamb}); "
            "mesh may experience residual shrinkage.",
            UserWarning,
            stacklevel=2,
        )

    # Ensure alternating filter runs complete cycles (shrink + dilate pairs)
    # so the mesh does not end on an uncompensated shrinkage pass.
    n_steps = iterations if iterations % 2 == 0 else iterations + 1

    target_mesh = mesh if inplace else mesh.copy()

    # Measure initial volume for drift detection
    v_initial = 0.0
    try:
        v_initial = abs(float(target_mesh.volume))
    except Exception:
        pass

    # Apply smoothing
    smoothed = False
    try:
        import trimesh.smoothing

        trimesh.smoothing.filter_taubin(
            target_mesh,
            lamb=float(lamb),
            nu=float(nu_mag),
            iterations=int(n_steps),
        )
        smoothed = True
    except Exception as exc:
        logger.debug(f"trimesh.smoothing.filter_taubin unavailable ({exc}); using sparse Laplacian kernel.")

    if not smoothed:
        _fallback_taubin_smooth(
            target_mesh,
            iterations=int(n_steps),
            lamb=float(lamb),
            nu_mag=float(nu_mag),
        )

    # Clean topology and fix normals
    target_mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(target_mesh)
    try:
        if float(target_mesh.volume) < 0.0:
            target_mesh.invert()
            trimesh.repair.fix_normals(target_mesh)
    except Exception:
        pass

    # Check and report volume drift
    if v_initial > 0.0:
        try:
            v_final = abs(float(target_mesh.volume))
            drift = abs(v_final - v_initial) / v_initial
            target_mesh.metadata["taubin_volume_drift"] = drift
            if drift >= 0.005:
                logger.info(
                    f"Taubin smoothing: volume drift |ΔV/V0| = {drift * 100:.3f}% "
                    f"over {n_steps} steps."
                )
        except Exception:
            pass

    return target_mesh
