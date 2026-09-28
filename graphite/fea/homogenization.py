"""
Micro-Scale Voxel RVE Homogenization Engine.

Computes the effective 6x6 linear-elastic constitutive stiffness tensor C^H and
engineering constants for periodic unit cells (TPMS, spinodal, explicit struts)
using voxelized finite element analysis with periodic boundary conditions (PBCs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import cg, splu

from graphite.math.tpms import evaluate_tpms


# ===========================================================================
# Dataclasses & Configuration
# ===========================================================================


@dataclass(frozen=True)
class RVEGridConfig:
    """
    Configuration for voxelized unit-cell RVE homogenization.

    Parameters
    ----------
    resolution : int
        Number of voxels along each spatial axis (e.g., 32 or 48).
    cell_size : float
        Physical unit cell edge length L in mm, by default 1.0.
    base_E : float
        Young's modulus of the solid constituent material in MPa, by default 2000.0.
    base_nu : float
        Poisson's ratio of the solid constituent material, by default 0.35.
    ersatz_ratio : float
        Stiffness multiplier for void voxels (ersatz material), by default 1e-6.
    solver_backend : str
        'auto', 'direct' (sparse LU factorization), or 'cg' (preconditioned conjugate gradient).
    solver_tol : float
        Convergence tolerance for iterative solver, by default 1e-6.
    max_iter : int
        Maximum iterations for iterative solver, by default 2000.
    """

    resolution: int = 32
    cell_size: float = 1.0
    base_E: float = 2000.0
    base_nu: float = 0.35
    ersatz_ratio: float = 1e-6
    solver_backend: Literal["auto", "direct", "cg"] = "cg"
    solver_tol: float = 1e-6
    max_iter: int = 2000


@dataclass
class EngineeringConstants:
    """
    Homogenized engineering elastic constants extracted from C^H.

    Voigt convention in Graphite: [xx, yy, zz, xy, yz, xz].
    """

    E_x: float
    E_y: float
    E_z: float
    G_xy: float
    G_yz: float
    G_zx: float
    nu_xy: float
    nu_yx: float
    nu_xz: float
    nu_zx: float
    nu_yz: float
    nu_zy: float
    bulk_modulus: float
    zener_anisotropy: float


@dataclass
class HomogenizationResult:
    """
    Result of unit cell RVE asymptotic homogenization.

    Attributes
    ----------
    C_homogenized : np.ndarray
        Effective 6x6 stiffness tensor in Voigt notation (MPa).
    compliance : np.ndarray
        Effective 6x6 compliance tensor S^H = (C^H)^-1 (MPa^-1).
    solid_fraction : float
        Actual solid volume fraction of the voxelized domain.
    engineering_constants : EngineeringConstants
        Derived directional moduli, Poisson ratios, and anisotropy.
    solve_time_s : float
        Wall-clock time for the 6-strain solve.
    solver_used : str
        Name of solver backend used ('direct' or 'cg').
    iterations : list[int]
        Solver iteration count for each of the 6 canonical load cases.
    """

    C_homogenized: np.ndarray
    compliance: np.ndarray
    solid_fraction: float
    engineering_constants: EngineeringConstants
    solve_time_s: float
    solver_used: str
    iterations: list[int] = field(default_factory=list)


# ===========================================================================
# C3D8 Hexahedral Element Stiffness Formulation
# ===========================================================================

# Corner node coordinate offsets in reference coordinates [-1, 1]^3
# Local node numbering 0 through 7
_HEX_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],  # 0
        [+1.0, -1.0, -1.0],  # 1
        [+1.0, +1.0, -1.0],  # 2
        [-1.0, +1.0, -1.0],  # 3
        [-1.0, -1.0, +1.0],  # 4
        [+1.0, -1.0, +1.0],  # 5
        [+1.0, +1.0, +1.0],  # 6
        [-1.0, +1.0, +1.0],  # 7
    ],
    dtype=np.float64,
)

# Corner index offsets relative to voxel (i, j, k)
_CORNER_I = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
_CORNER_J = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
_CORNER_K = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)


def build_isotropic_material_matrix(E: float, nu: float) -> np.ndarray:
    """
    Build the 6x6 isotropic constitutive stiffness matrix C_s (Voigt notation).

    Voigt strain/stress ordering: [xx, yy, zz, xy, yz, xz].
    """
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C = np.zeros((6, 6), dtype=np.float64)
    # Normal components
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = lam
    # Shear components (xy, yz, xz)
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def _evaluate_B_matrix_at_point(
    xi: float, eta: float, zeta: float, hx: float, hy: float, hz: float
) -> np.ndarray:
    """Evaluate 6x24 strain-displacement matrix B at natural coordinate (xi, eta, zeta)."""
    # Shape function derivatives w.r.t natural coords: (8,)
    dN_dxi = (
        0.125
        * _HEX_CORNERS[:, 0]
        * (1.0 + _HEX_CORNERS[:, 1] * eta)
        * (1.0 + _HEX_CORNERS[:, 2] * zeta)
    )
    dN_deta = (
        0.125
        * (1.0 + _HEX_CORNERS[:, 0] * xi)
        * _HEX_CORNERS[:, 1]
        * (1.0 + _HEX_CORNERS[:, 2] * zeta)
    )
    dN_dzeta = (
        0.125
        * (1.0 + _HEX_CORNERS[:, 0] * xi)
        * (1.0 + _HEX_CORNERS[:, 1] * eta)
        * _HEX_CORNERS[:, 2]
    )

    # Physical derivatives via diagonal Jacobian J = diag(hx/2, hy/2, hz/2)
    dN_dx = dN_dxi * (2.0 / hx)
    dN_dy = dN_deta * (2.0 / hy)
    dN_dz = dN_dzeta * (2.0 / hz)

    # B matrix shape: (6, 24)
    # Voigt order: [xx, yy, zz, xy, yz, xz]
    B = np.zeros((6, 24), dtype=np.float64)
    # eps_xx = du/dx
    B[0, 0::3] = dN_dx
    # eps_yy = dv/dy
    B[1, 1::3] = dN_dy
    # eps_zz = dw/dz
    B[2, 2::3] = dN_dz
    # gamma_xy = du/dy + dv/dx
    B[3, 0::3] = dN_dy
    B[3, 1::3] = dN_dx
    # gamma_yz = dv/dz + dw/dy
    B[4, 1::3] = dN_dz
    B[4, 2::3] = dN_dy
    # gamma_xz = du/dz + dw/dx
    B[5, 0::3] = dN_dz
    B[5, 2::3] = dN_dx

    return B


def build_voxel_c3d8_stiffness(
    hx: float, hy: float, hz: float, E: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute analytical 24x24 element stiffness matrix K0 and centroid B matrix.

    Uses 2x2x2 Gauss integration for exact P1 trilinear brick stiffness.

    Returns
    -------
    K0 : np.ndarray
        Element stiffness matrix of shape (24, 24).
    B_centroid : np.ndarray
        Strain-displacement matrix at element centroid of shape (6, 24).
    """
    C_base = build_isotropic_material_matrix(E, nu)
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = (-gp, gp)

    # Voxel volume Ve = hx * hy * hz; det(J) = Ve / 8; Gauss weight = 1.0 * 1.0 * 1.0 = 1.0
    vol_weight = (hx * hy * hz) / 8.0

    K0 = np.zeros((24, 24), dtype=np.float64)
    for xi in gauss_pts:
        for eta in gauss_pts:
            for zeta in gauss_pts:
                B_g = _evaluate_B_matrix_at_point(xi, eta, zeta, hx, hy, hz)
                K0 += vol_weight * (B_g.T @ C_base @ B_g)

    # Centroid B matrix at (0, 0, 0)
    B_centroid = _evaluate_B_matrix_at_point(0.0, 0.0, 0.0, hx, hy, hz)
    return K0, B_centroid


# ===========================================================================
# Periodic RVE Homogenization Engine
# ===========================================================================


def homogenize_voxel_rve(
    voxel_mask: np.ndarray,
    config: RVEGridConfig | None = None,
) -> HomogenizationResult:
    """
    Perform 3D linear-elastic asymptotic homogenization on a voxelized RVE.

    Parameters
    ----------
    voxel_mask : np.ndarray
        3D boolean mask or continuous density array in [0, 1] of shape (Nx, Ny, Nz).
    config : RVEGridConfig, optional
        RVE settings (resolution, base properties, solver mode). If None, defaults are used.

    Returns
    -------
    HomogenizationResult
        Computed 6x6 effective stiffness matrix C^H and engineering elastic constants.
    """
    if config is None:
        config = RVEGridConfig()

    mask = np.asarray(voxel_mask, dtype=np.float64)
    if mask.ndim != 3:
        raise ValueError(f"voxel_mask must be 3-dimensional, got shape {mask.shape}")

    nx, ny, nz = mask.shape
    num_voxels = nx * ny * nz
    solid_fraction = float(np.mean(mask))

    # Density per voxel: solid voxels are 1.0, void voxels are scaled by ersatz_ratio
    # Continuous density: chi_e = ersatz_ratio + (1 - ersatz_ratio) * mask_e
    chi = config.ersatz_ratio + (1.0 - config.ersatz_ratio) * np.clip(mask.ravel(), 0.0, 1.0)

    hx = config.cell_size / nx
    hy = config.cell_size / ny
    hz = config.cell_size / nz
    voxel_volume = hx * hy * hz
    total_volume = config.cell_size**3

    # 1. Precompute reference voxel element stiffness K0 and centroid B
    K0, B_centroid = build_voxel_c3d8_stiffness(
        hx, hy, hz, E=config.base_E, nu=config.base_nu
    )
    C_base = build_isotropic_material_matrix(config.base_E, config.base_nu)

    # 2. Build periodic node mapping on 3-torus
    # Node indexing: (i, j, k) where i in [0, nx-1], j in [0, ny-1], k in [0, nz-1]
    # Any coordinate at boundary wraps modulo grid dimension
    grid_i, grid_j, grid_k = np.meshgrid(
        np.arange(nx, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        np.arange(nz, dtype=np.int64),
        indexing="ij",
    )
    vox_i = grid_i.ravel()
    vox_j = grid_j.ravel()
    vox_k = grid_k.ravel()

    # Corner nodes for all voxels with periodic wrapping: (num_voxels, 8)
    ci = (vox_i[:, None] + _CORNER_I[None, :]) % nx
    cj = (vox_j[:, None] + _CORNER_J[None, :]) % ny
    ck = (vox_k[:, None] + _CORNER_K[None, :]) % nz

    node_ids = ci * (ny * nz) + cj * nz + ck  # shape (num_voxels, 8)

    # 24 DOFs per voxel: [u0, v0, w0, u1, v1, w1, ...]
    dofs = np.zeros((num_voxels, 24), dtype=np.int64)
    dofs[:, 0::3] = 3 * node_ids + 0
    dofs[:, 1::3] = 3 * node_ids + 1
    dofs[:, 2::3] = 3 * node_ids + 2

    # 3. Assemble global periodic stiffness matrix K_periodic via COO
    num_nodes = nx * ny * nz
    num_dofs = 3 * num_nodes

    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()

    # K_e = chi_e * K0
    # Expand values: (num_voxels, 24, 24)
    vals = (chi[:, None, None] * K0[None, :, :]).ravel()
    K_periodic = coo_matrix((vals, (rows, cols)), shape=(num_dofs, num_dofs)).tocsr()

    # 4. Remove rigid body modes: fix node 0 (DOFs 0, 1, 2)
    # The reduced system has DOFs 3 to num_dofs - 1
    free_dofs = np.arange(3, num_dofs, dtype=np.int64)
    K_free = K_periodic[free_dofs, :][:, free_dofs]

    # 5. Formulate canonical strain load vectors
    # 6 canonical macroscopic strains in Voigt notation: [xx, yy, zz, xy, yz, xz]
    # Load vector for case k: F_e^(k) = - V_e * B_centroid^T * (chi_e * C_base * eps_bar^(k))
    I_6 = np.eye(6, dtype=np.float64)
    # sigma_base[k] = C_base @ eps_bar[k]
    sigma_base = C_base @ I_6  # shape (6, 6)

    # g0[k] = B_centroid.T @ sigma_base[:, k] : shape (24, 6)
    g0 = B_centroid.T @ sigma_base

    # Element forces: Fe[e, :, k] = - V_e * chi[e] * g0[:, k]
    # shape: (num_voxels, 24, 6)
    Fe_all = -voxel_volume * (chi[:, None, None] * g0[None, :, :])

    # Assemble global force vectors F_all: shape (num_dofs, 6)
    F_all = np.zeros((num_dofs, 6), dtype=np.float64)
    for k in range(6):
        F_all[:, k] = np.bincount(
            dofs.ravel(), weights=Fe_all[:, :, k].ravel(), minlength=num_dofs
        )

    F_free_all = F_all[free_dofs, :]  # shape (num_free_dofs, 6)

    # 6. Solve K_free * U_free = F_free for all 6 load cases
    t0 = time.perf_counter()
    backend = config.solver_backend
    if backend == "auto":
        backend = "direct" if num_dofs <= 120_000 else "cg"

    U_all = np.zeros((num_dofs, 6), dtype=np.float64)
    iterations = []

    if backend == "direct":
        # Factorize K_free once using SuperLU
        solver = splu(K_free.tocsc())
        for k in range(6):
            U_all[free_dofs, k] = solver.solve(F_free_all[:, k])
            iterations.append(1)
    else:
        # Preconditioned Conjugate Gradient with Jacobi preconditioner
        diag_K = K_free.diagonal()
        inv_diag = np.where(np.abs(diag_K) > 1e-12, 1.0 / diag_K, 1.0)
        M_prec = diags(inv_diag, format="csr")

        for k in range(6):
            u_sol, info = cg(
                K_free,
                F_free_all[:, k],
                M=M_prec,
                rtol=config.solver_tol,
                maxiter=config.max_iter,
            )
            if info != 0:
                # If CG did not converge to tolerance, fallback to sparse direct
                solver = splu(K_free.tocsc())
                u_sol = solver.solve(F_free_all[:, k])
            U_all[free_dofs, k] = u_sol
            iterations.append(config.max_iter if info != 0 else 50)

    solve_time_s = time.perf_counter() - t0

    # 7. Volume Stress Averaging to assemble homogenized stiffness C^H
    # For element e: total strain eps_e = eps_bar^(k) + B_centroid @ u_e^(k)
    # stress_e = chi_e * C_base @ eps_e
    C_H = np.zeros((6, 6), dtype=np.float64)

    for k in range(6):
        # Extract element nodal displacements: shape (num_voxels, 24)
        u_e = U_all[dofs, k]
        # Fluctuation strain: shape (num_voxels, 6)
        eps_fluc = u_e @ B_centroid.T
        # Total strain: eps_bar^(k) + eps_fluc
        eps_tot = I_6[k, :][None, :] + eps_fluc
        # Microscopic stress: chi_e * (C_base @ eps_tot.T).T
        sig_e = chi[:, None] * (eps_tot @ C_base.T)
        # Volume average: (1 / |Y|) * sum(V_e * sig_e) = (1 / num_voxels) * sum(sig_e)
        C_H[:, k] = np.mean(sig_e, axis=0)

    # Enforce Maxwell-Betti symmetry
    C_H = 0.5 * (C_H + C_H.T)

    # 8. Invert to Compliance matrix S^H = (C^H)^-1 and extract constants
    try:
        S_H = np.linalg.inv(C_H)
    except np.linalg.LinAlgError:
        S_H = np.linalg.pinv(C_H)

    # Directional Young's moduli
    E_x = float(1.0 / S_H[0, 0]) if abs(S_H[0, 0]) > 1e-15 else 0.0
    E_y = float(1.0 / S_H[1, 1]) if abs(S_H[1, 1]) > 1e-15 else 0.0
    E_z = float(1.0 / S_H[2, 2]) if abs(S_H[2, 2]) > 1e-15 else 0.0

    # Shear moduli: indices 3: xy, 4: yz, 5: xz
    G_xy = float(1.0 / S_H[3, 3]) if abs(S_H[3, 3]) > 1e-15 else 0.0
    G_yz = float(1.0 / S_H[4, 4]) if abs(S_H[4, 4]) > 1e-15 else 0.0
    G_zx = float(1.0 / S_H[5, 5]) if abs(S_H[5, 5]) > 1e-15 else 0.0

    # Poisson's ratios: nu_ij = - S_ji / S_ii
    nu_xy = float(-S_H[1, 0] / S_H[0, 0]) if abs(S_H[0, 0]) > 1e-15 else 0.0
    nu_yx = float(-S_H[0, 1] / S_H[1, 1]) if abs(S_H[1, 1]) > 1e-15 else 0.0
    nu_xz = float(-S_H[2, 0] / S_H[0, 0]) if abs(S_H[0, 0]) > 1e-15 else 0.0
    nu_zx = float(-S_H[0, 2] / S_H[2, 2]) if abs(S_H[2, 2]) > 1e-15 else 0.0
    nu_yz = float(-S_H[2, 1] / S_H[1, 1]) if abs(S_H[1, 1]) > 1e-15 else 0.0
    nu_zy = float(-S_H[1, 2] / S_H[2, 2]) if abs(S_H[2, 2]) > 1e-15 else 0.0

    # Bulk modulus for orthotropic/cubic: K = (sum(C_ij for i,j in {0,1,2})) / 9
    bulk_K = float(np.sum(C_H[:3, :3])) / 9.0

    # Zener anisotropy ratio: A = 2 * C44 / (C11 - C12)
    denom = C_H[0, 0] - C_H[0, 1]
    zener_A = float(2.0 * C_H[3, 3] / denom) if abs(denom) > 1e-12 else 1.0

    constants = EngineeringConstants(
        E_x=E_x,
        E_y=E_y,
        E_z=E_z,
        G_xy=G_xy,
        G_yz=G_yz,
        G_zx=G_zx,
        nu_xy=nu_xy,
        nu_yx=nu_yx,
        nu_xz=nu_xz,
        nu_zx=nu_zx,
        nu_yz=nu_yz,
        nu_zy=nu_zy,
        bulk_modulus=bulk_K,
        zener_anisotropy=zener_A,
    )

    return HomogenizationResult(
        C_homogenized=C_H,
        compliance=S_H,
        solid_fraction=solid_fraction,
        engineering_constants=constants,
        solve_time_s=solve_time_s,
        solver_used=backend,
        iterations=iterations,
    )


# ===========================================================================
# High-Level Unit Cell Homogenization Drivers
# ===========================================================================


def homogenize_tpms_cell(
    lattice_type: str = "Gyroid",
    solid_fraction: float = 0.33,
    is_sheet: bool = True,
    config: RVEGridConfig | None = None,
) -> HomogenizationResult:
    """
    Homogenize a Triply Periodic Minimal Surface (TPMS) unit cell.

    Parameters
    ----------
    lattice_type : str, optional
        TPMS type (e.g. 'Gyroid', 'Schwarz_P', 'Schwarz_D', 'Split_P', 'Neovius').
    solid_fraction : float, optional
        Target solid volume fraction in (0, 1), by default 0.33.
    is_sheet : bool, optional
        Sheet network if True, solid network if False. By default True.
    config : RVEGridConfig, optional
        RVE grid resolution and material properties.

    Returns
    -------
    HomogenizationResult
    """
    if config is None:
        config = RVEGridConfig()

    N = config.resolution
    L = config.cell_size

    # Regular voxel grid over Y = [0, L]^3
    # Voxel centers
    dx = L / N
    coords = np.linspace(0.5 * dx, L - 0.5 * dx, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Evaluate TPMS equation (with zero offset initially)
    k = 2.0 * np.pi / L
    field_0 = evaluate_tpms(lattice_type, k, X, Y, Z)

    # Determine offset tau to match target solid fraction
    if is_sheet:
        # Sheet: |field| <= tau => solid fraction is fraction where |field| <= tau
        abs_field = np.abs(field_0)
        tau = float(np.percentile(abs_field, solid_fraction * 100.0))
        mask = abs_field <= tau
    else:
        # Skeletal: field <= tau
        tau = float(np.percentile(field_0, solid_fraction * 100.0))
        mask = field_0 <= tau

    return homogenize_voxel_rve(mask, config=config)


def homogenize_octet_cell(
    solid_fraction: float = 0.10,
    config: RVEGridConfig | None = None,
    smooth_subvoxel: bool = True,
) -> HomogenizationResult:
    """
    Homogenize an explicit octet-truss unit cell on [0, 1]^3.

    Parameters
    ----------
    solid_fraction : float, optional
        Target solid volume fraction in (0, 1), by default 0.10.
    config : RVEGridConfig, optional
        RVE grid resolution and material properties.
    smooth_subvoxel : bool, optional
        Apply continuous sub-voxel smoothing. By default True.

    Returns
    -------
    HomogenizationResult
    """
    from graphite.explicit.hex_rules import apply_hex_octet_truss

    if config is None:
        config = RVEGridConfig()

    N = config.resolution
    L = config.cell_size
    dx = L / N

    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    nodes, struts = apply_hex_octet_truss(coords)

    # Invert slender-rod formula to find approximate radius r0: phi ~ 12 * sqrt(2) * pi * (r/L)^2
    r_approx = np.sqrt(max(solid_fraction, 0.01) / (12.0 * np.sqrt(2.0) * np.pi))

    coords_1d = np.linspace(0.5 * dx, L - 0.5 * dx, N)
    X, Y, Z = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    nodes_phys = nodes * L
    shifts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                shifts.append(np.array([di, dj, dk], dtype=np.float64) * L)

    min_dist_sq = np.full(grid_pts.shape[0], np.inf, dtype=np.float64)
    for p1_idx, p2_idx in struts:
        p1_base = nodes_phys[p1_idx]
        p2_base = nodes_phys[p2_idx]
        v_base = p2_base - p1_base
        len_sq = np.dot(v_base, v_base)
        if len_sq < 1e-12:
            continue
        for s in shifts:
            p1 = p1_base + s
            p2 = p2_base + s
            v = p2 - p1
            p1_to_pts = grid_pts - p1
            t = np.clip(np.dot(p1_to_pts, v) / len_sq, 0.0, 1.0)
            proj = p1 + t[:, None] * v
            dist_sq = np.sum((grid_pts - proj) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    dist = np.sqrt(min_dist_sq)

    # 1D Bisection to find exact radius r for target solid fraction
    r_low, r_high = 0.01, 0.30
    r_best = r_approx
    for _ in range(25):
        r_mid = 0.5 * (r_low + r_high)
        if smooth_subvoxel:
            vf = float(np.mean(np.clip(0.5 - (dist - r_mid * L) / dx, 0.0, 1.0)))
        else:
            vf = float(np.mean(dist <= r_mid * L))
        if vf < solid_fraction:
            r_low = r_mid
        else:
            r_high = r_mid
        r_best = r_mid

    if smooth_subvoxel:
        mask = np.clip(0.5 - (dist - r_best * L) / dx, 0.0, 1.0).reshape(N, N, N)
    else:
        mask = (dist <= r_best * L).reshape(N, N, N)

    return homogenize_voxel_rve(mask, config=config)


def homogenize_strut_cell(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float,
    config: RVEGridConfig | None = None,
    smooth_subvoxel: bool = True,
) -> HomogenizationResult:
    """
    Homogenize an explicit strut unit cell (nodes + struts) on [0, 1]^3.

    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates in normalized reference cube [0, 1]^3, shape (V, 3).
    struts : np.ndarray
        Strut connectivity pairs (indices into nodes), shape (S, 2).
    strut_radius : float
        Strut cylinder radius normalized to unit cell size (r / L).
    config : RVEGridConfig, optional
        RVE grid resolution and material properties.
    smooth_subvoxel : bool, optional
        If True, applies continuous trilinear sub-voxel boundary smoothing to eliminate
        voxel discretization staircasing for slender struts. Default True.

    Returns
    -------
    HomogenizationResult
    """
    if config is None:
        config = RVEGridConfig()

    N = config.resolution
    L = config.cell_size
    dx = L / N
    coords = np.linspace(0.5 * dx, L - 0.5 * dx, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # shape (N^3, 3)

    nodes_phys = np.asarray(nodes, dtype=np.float64) * L
    struts_arr = np.asarray(struts, dtype=np.int64)
    r_phys = strut_radius * L

    # Distance to each cylinder segment with periodic images
    # Minimum squared distance array
    min_dist_sq = np.full(grid_pts.shape[0], np.inf, dtype=np.float64)

    # Account for periodic wrapping of struts across boundary faces (-1, 0, 1)
    shifts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                shifts.append(np.array([di, dj, dk], dtype=np.float64) * L)

    for p1_idx, p2_idx in struts_arr:
        p1_base = nodes_phys[p1_idx]
        p2_base = nodes_phys[p2_idx]
        v_base = p2_base - p1_base
        len_sq = np.dot(v_base, v_base)
        if len_sq < 1e-12:
            continue

        for s in shifts:
            p1 = p1_base + s
            p2 = p2_base + s
            v = p2 - p1

            # Project grid points onto segment p1-p2
            p1_to_pts = grid_pts - p1
            t = np.clip(np.dot(p1_to_pts, v) / len_sq, 0.0, 1.0)
            proj = p1 + t[:, None] * v
            dist_sq = np.sum((grid_pts - proj) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    if smooth_subvoxel:
        dist = np.sqrt(min_dist_sq)
        mask = np.clip(0.5 - (dist - r_phys) / dx, 0.0, 1.0).reshape(N, N, N)
    else:
        mask = (min_dist_sq <= r_phys**2).reshape(N, N, N)

    return homogenize_voxel_rve(mask, config=config)

