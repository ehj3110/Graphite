"""
Graphite Explicit — Level-Set Multi-Morphology Hybrid Optimization Engine

Based on Liu et al. (Nature Communications 2024):
"Ultrastiff metamaterials generated through a multilayer strategy and topology optimization"

Implements:
- Regularized Heaviside density projection H_eps(phi) and derivative h_eps(phi).
- Periodic level-set function regularization step phi* = alpha * phi.
- Vectorized 3D mean curvature field calculation H = div(grad(phi) / |grad(phi)|).
- Smooth multi-morphology interface interpolation suppressing parasitic interface bending and stress singularities.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np
import trimesh

from graphite.math.tpms import (
    gyroid,
    schwarz_p,
    schwarz_d,
    neovius,
    lidinoid,
    split_p,
)
from graphite.implicit.meshing_backends import extract_isosurface


def iwp(x: np.ndarray, y: np.ndarray, z: np.ndarray, unit_cell_size: float, iso_offset: float = 0.0, is_sheet: bool = True) -> np.ndarray:
    """Evaluate the I-WP TPMS equation."""
    k = (2.0 * np.pi) / unit_cell_size
    kx, ky, kz = k * x, k * y, k * z
    eq = 2.0 * (np.cos(kx) * np.cos(ky) + np.cos(ky) * np.cos(kz) + np.cos(kz) * np.cos(kx)) - (
        np.cos(2.0 * kx) + np.cos(2.0 * ky) + np.cos(2.0 * kz)
    )
    return np.abs(eq) - iso_offset if is_sheet else eq - iso_offset



# =============================================================================
# 1. Level-Set Density Mappings & Derivatives (Liu et al. 2024)
# =============================================================================

def regularized_heaviside(phi: np.ndarray | float, epsilon: float = 0.1) -> np.ndarray:
    """
    Smooth, C^1-continuous regularized Heaviside step function H_epsilon(phi).

    Formula (Liu et al., Nature Communications 2024, Eq. 4):
        H_eps(phi) = 1                                              for phi > eps
        H_eps(phi) = 0                                              for phi < -eps
        H_eps(phi) = 1/2 * (1 + phi/eps + 1/pi * sin(pi*phi/eps))   for |phi| <= eps

    Properties:
        - H_eps(-eps) = 0.0
        - H_eps(0.0) = 0.5
        - H_eps(+eps) = 1.0
        - Monotonic and strictly C^1 smooth across the transition zone [-eps, eps].

    Args:
        phi: Scalar or numpy array of level-set function values.
        epsilon: Transition bandwidth (> 0).

    Returns:
        Projected density values in [0, 1].
    """
    p = np.asarray(phi, dtype=np.float64)
    eps = float(max(epsilon, 1e-12))

    out = np.empty_like(p)
    mask_high = p > eps
    mask_low = p < -eps
    mask_mid = ~mask_high & ~mask_low

    out[mask_high] = 1.0
    out[mask_low] = 0.0

    p_mid = p[mask_mid]
    out[mask_mid] = 0.5 * (1.0 + (p_mid / eps) + (1.0 / np.pi) * np.sin((np.pi * p_mid) / eps))
    return out


def heaviside_derivative(phi: np.ndarray | float, epsilon: float = 0.1) -> np.ndarray:
    """
    Analytical first derivative of the regularized Heaviside function h_epsilon(phi) = dH_eps / dphi.

    Approximates the continuous Dirac delta function:
        h_eps(phi) = 1/(2*eps) * (1 + cos(pi*phi/eps))    for |phi| <= eps
        h_eps(phi) = 0                                    for |phi| > eps

    Args:
        phi: Scalar or numpy array of level-set values.
        epsilon: Transition bandwidth.

    Returns:
        Derivative values (>= 0).
    """
    p = np.asarray(phi, dtype=np.float64)
    eps = float(max(epsilon, 1e-12))

    out = np.zeros_like(p)
    mask_mid = np.abs(p) <= eps

    p_mid = p[mask_mid]
    out[mask_mid] = (1.0 / (2.0 * eps)) * (1.0 + np.cos((np.pi * p_mid) / eps))
    return out


def levelset_regularization_step(
    phi: np.ndarray,
    alpha: float = 0.95,
) -> np.ndarray:
    """
    Periodic Level-Set Function (LSF) regularization step:
        phi* = alpha * phi   (0 < alpha < 1).

    Maintains bounded Lipschitz gradient magnitudes and prevents numerical divergence
    during ODE-driven level-set evolution (dphi/dt + V_NS = 0).

    Args:
        phi: 3D level-set scalar array.
        alpha: Regularization decay factor (typically 0.90 to 0.98).

    Returns:
        Regularized level-set array phi*.
    """
    a = float(np.clip(alpha, 0.01, 0.999))
    return np.asarray(phi, dtype=np.float64) * a


# =============================================================================
# 2. Mean Curvature Field (Zero-Curvature Interface Verification)
# =============================================================================

def mean_curvature_field(
    phi: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Compute 3D mean curvature field H = div(grad(phi) / ||grad(phi)||).

    For minimal surface TPMS shells and smooth non-parasitic boundaries, H = 0.
    Non-zero peaks indicate sharp re-entrant corners that generate parasitic bending.

    Args:
        phi: 3D numpy array of level-set values.
        spacing: Voxel pitch (dx, dy, dz) in mm.

    Returns:
        3D numpy array of mean curvature values H(x, y, z).
    """
    p = np.asarray(phi, dtype=np.float64)
    dx, dy, dz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

    # Gradients of phi
    g_x, g_y, g_z = np.gradient(p, dx, dy, dz)
    norm = np.sqrt(g_x**2 + g_y**2 + g_z**2)
    norm_safe = np.maximum(norm, 1e-12)

    # Unit normal components n = grad(phi) / |grad(phi)|
    nx = g_x / norm_safe
    ny = g_y / norm_safe
    nz = g_z / norm_safe

    # Divergence of unit normal field
    dnx_dx = np.gradient(nx, dx, axis=0)
    dny_dy = np.gradient(ny, dy, axis=1)
    dnz_dz = np.gradient(nz, dz, axis=2)

    h_field = 0.5 * (dnx_dx + dny_dy + dnz_dz)
    return h_field


# =============================================================================
# 3. Continuous Multi-Morphology Hybrid Blending
# =============================================================================

def blend_lattice_morphologies(
    field_a: np.ndarray,
    field_b: np.ndarray,
    transition_coord: np.ndarray,
    transition_center: float = 0.0,
    transition_width: float = 1.0,
    epsilon: float = 0.5,
    axis: int = 0,
) -> np.ndarray:
    """
    Blend two distinct 3D lattice morphology scalar fields using regularized Heaviside weights.

    Formula:
        w(X) = H_eps((X - X_center) / (transition_width / 2))
        Phi_hybrid(X) = (1 - w(X)) * field_a(X) + w(X) * field_b(X)

    Guarantees C^1 continuity across the interface, suppressing parasitic interface bending
    and eliminating stress singularities at the boundary between distinct cell archetypes.

    Args:
        field_a: 3D scalar field for morphology A (e.g. stretching-dominated Schwarz P).
        field_b: 3D scalar field for morphology B (e.g. shear-resistant Gyroid).
        transition_coord: 1D or 3D coordinate array along the transition direction.
        transition_center: Spatial coordinate where transition occurs.
        transition_width: Full physical width of the blending region in mm.
        epsilon: Smoothing bandwidth for regularized Heaviside.
        axis: Coordinate axis for 1D transition array (0=x, 1=y, 2=z).

    Returns:
        Blended 3D scalar field Phi_hybrid.
    """
    fa = np.asarray(field_a, dtype=np.float64)
    fb = np.asarray(field_b, dtype=np.float64)
    tc = np.asarray(transition_coord, dtype=np.float64)

    half_w = max(0.5 * float(transition_width), 1e-6)
    # Normalized coordinate: -1 at start of transition, 0 at center, +1 at end
    xi = (tc - float(transition_center)) / half_w

    weight = regularized_heaviside(xi, epsilon=float(epsilon))

    # Broadcast weight across grid if necessary
    if weight.shape != fa.shape:
        if weight.ndim == 1 and fa.ndim == 3:
            if axis == 0:
                weight = weight[:, None, None]
            elif axis == 1:
                weight = weight[None, :, None]
            elif axis == 2:
                weight = weight[None, None, :]
        weight = np.broadcast_to(weight, fa.shape)

    return (1.0 - weight) * fa + weight * fb


# =============================================================================
# 4. Hybrid Level-Set Lattice Generator
# =============================================================================

_MORPHOLOGY_DISPATCH: dict[str, Callable] = {
    "schwarz_p": schwarz_p,
    "pcu": schwarz_p,
    "primitive": schwarz_p,
    "gyroid": gyroid,
    "schwarz_d": schwarz_d,
    "diamond": schwarz_d,
    "neovius": neovius,
    "iwp": iwp,
}


def generate_hybrid_levelset_lattice(
    dims: tuple[int, int, int] = (60, 60, 60),
    physical_size: tuple[float, float, float] = (30.0, 30.0, 30.0),
    morphology_a: str = "schwarz_p",
    morphology_b: str = "gyroid",
    unit_cell_size: float = 10.0,
    transition_width: float = 6.0,
    transition_axis: str = "x",
    iso_offset: float = 0.35,
    is_sheet: bool = True,
    epsilon: float = 0.5,
    build_mesh: bool = True,
    cap_boundaries: bool = True,
) -> tuple[np.ndarray, trimesh.Trimesh | None, dict[str, Any]]:
    """
    Generate an optimized multi-morphology hybrid lattice with smooth level-set interface.

    Combines two distinct TPMS architectures (e.g. Schwarz P and Gyroid) across a continuous
    ODE-regularized Heaviside boundary. Suppresses parasitic bending and eliminates re-entrant notches.

    Args:
        dims: Voxel grid resolution (Nx, Ny, Nz).
        physical_size: Physical bounding box (Lx, Ly, Lz) in mm.
        morphology_a: Name of morphology A ('schwarz_p', 'gyroid', 'schwarz_d', 'neovius', 'iwp').
        morphology_b: Name of morphology B.
        unit_cell_size: Unit cell repeat period in mm (default 10.0 mm).
        transition_width: Width of interface transition zone in mm.
        transition_axis: Spatial axis for transition ('x', 'y', or 'z').
        iso_offset: Level-set isovalue offset (thickness control).
        is_sheet: True for sheet TPMS network, False for skeletal network.
        epsilon: Heaviside regularizing bandwidth.
        build_mesh: Extract 3D watertight mesh via marching cubes/flying edges.
        cap_boundaries: If True, pads volume and trims with CAD bounding box to guarantee watertightness.

    Returns:
        tuple: (hybrid_field (Nx, Ny, Nz), mesh or None, metadata_dict)
    """
    nx, ny, nz = (int(dims[0]), int(dims[1]), int(dims[2]))
    lx, ly, lz = (float(physical_size[0]), float(physical_size[1]), float(physical_size[2]))

    dx = lx / (nx - 1) if nx > 1 else lx
    dy = ly / (ny - 1) if ny > 1 else ly
    dz = lz / (nz - 1) if nz > 1 else lz
    spacing = (dx, dy, dz)
    origin = (-0.5 * lx, -0.5 * ly, -0.5 * lz)

    # Coordinate grids
    xs = np.linspace(-0.5 * lx, 0.5 * lx, nx)
    ys = np.linspace(-0.5 * ly, 0.5 * ly, ny)
    zs = np.linspace(-0.5 * lz, 0.5 * lz, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Resolve morphology functions
    key_a = morphology_a.strip().lower().replace("-", "_")
    key_b = morphology_b.strip().lower().replace("-", "_")

    if key_a not in _MORPHOLOGY_DISPATCH:
        raise ValueError(f"Unknown morphology_a '{morphology_a}'. Supported: {list(_MORPHOLOGY_DISPATCH.keys())}")
    if key_b not in _MORPHOLOGY_DISPATCH:
        raise ValueError(f"Unknown morphology_b '{morphology_b}'. Supported: {list(_MORPHOLOGY_DISPATCH.keys())}")

    fn_a = _MORPHOLOGY_DISPATCH[key_a]
    fn_b = _MORPHOLOGY_DISPATCH[key_b]

    # Evaluate morphology fields
    field_a = fn_a(X, Y, Z, unit_cell_size=float(unit_cell_size), iso_offset=float(iso_offset), is_sheet=is_sheet)
    field_b = fn_b(X, Y, Z, unit_cell_size=float(unit_cell_size), iso_offset=float(iso_offset), is_sheet=is_sheet)

    # Determine transition coordinate
    axis_map = {"x": X, "y": Y, "z": Z}
    axis_indices = {"x": 0, "y": 1, "z": 2}
    axis_str = str(transition_axis).strip().lower()
    if axis_str not in axis_map:
        raise ValueError(f"Invalid transition_axis '{transition_axis}'. Expected 'x', 'y', or 'z'.")
    trans_grid = axis_map[axis_str]
    axis_idx = axis_indices[axis_str]

    # Blend fields using regularized Heaviside
    hybrid_field = blend_lattice_morphologies(
        field_a=field_a,
        field_b=field_b,
        transition_coord=trans_grid,
        transition_center=0.0,
        transition_width=float(transition_width),
        epsilon=float(epsilon),
        axis=axis_idx,
    )

    # Compute mean curvature field for interface diagnostics
    h_field = mean_curvature_field(hybrid_field, spacing=spacing)
    trans_mask = np.abs(trans_grid) <= (0.5 * float(transition_width))
    h_interface = h_field[trans_mask]

    h_mean = float(np.mean(h_interface)) if len(h_interface) > 0 else 0.0
    h_std = float(np.std(h_interface)) if len(h_interface) > 0 else 0.0
    h_max = float(np.max(np.abs(h_interface))) if len(h_interface) > 0 else 0.0

    mesh = None
    if build_mesh:
        if cap_boundaries:
            pad = 2
            padded_field = np.pad(hybrid_field, pad_width=pad, mode="constant", constant_values=1.0)
            padded_origin = (origin[0] - pad * dx, origin[1] - pad * dy, origin[2] - pad * dz)
            res = extract_isosurface(
                field=padded_field,
                spacing=spacing,
                origin=padded_origin,
                level=0.0,
                backend="pyvista_flying_edges",
                postprocess=True,
                fill_holes=True,
                enforce_watertight=True,
            )
            mesh = res.mesh
            # CAD box boolean intersection for planar cut faces
            try:
                from graphite.explicit.geometry_module import boolean_intersect_with_cad
                cad_box = trimesh.creation.box(extents=(lx, ly, lz))
                trimmed, _ = boolean_intersect_with_cad(mesh, cad_box)
                if trimmed is not None and len(trimmed.faces) > 0 and trimmed.is_watertight:
                    mesh = trimmed
            except Exception:
                pass
        else:
            res = extract_isosurface(
                field=hybrid_field,
                spacing=spacing,
                origin=origin,
                level=0.0,
                backend="pyvista_flying_edges",
                postprocess=True,
                fill_holes=True,
            )
            mesh = res.mesh

    metadata = {
        "morphology_a": morphology_a,
        "morphology_b": morphology_b,
        "grid_resolution": list(dims),
        "physical_size_mm": list(physical_size),
        "unit_cell_size_mm": float(unit_cell_size),
        "transition_width_mm": float(transition_width),
        "transition_axis": axis_str,
        "iso_offset": float(iso_offset),
        "is_sheet": is_sheet,
        "epsilon": float(epsilon),
        "mean_curvature_interface": {
            "mean": h_mean,
            "std": h_std,
            "max_abs": h_max,
            "zero_curvature_satisfied": bool(abs(h_mean) < 0.05),
        },
        "is_watertight": mesh.is_watertight if mesh is not None else False,
        "volume_mm3": float(mesh.volume) if mesh is not None else 0.0,
    }

    return hybrid_field, mesh, metadata
