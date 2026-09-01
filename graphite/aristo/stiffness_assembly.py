"""
Aristo Stiffness Assembly — Vectorized P1 Tetrahedral FEM

Assembles the global stiffness matrix K for a linear-elastic solid
using P1 (constant-strain) tetrahedral elements. All element-level
operations are fully vectorized over the element batch via numpy
broadcasting and einsum — no Python loops over elements.

Physical units: mm (length), MPa = N/mm² (stress/modulus), N (force).

Math Reference (Voigt notation, 3D isotropic elasticity)
---------------------------------------------------------
Shape functions for a 4-node P1 tetrahedron (reference coords ξ,η,ζ):
    N₁ = 1 − ξ − η − ζ,  N₂ = ξ,  N₃ = η,  N₄ = ζ

Jacobian (maps reference → physical):
    J[m] = [[p₂−p₁], [p₃−p₁], [p₄−p₁]]   shape (3,3)

Shape function gradients in physical coords:
    ∇N_phys[m] = dN_ref @ J_inv[m]          shape (4,3)
    (where dN_ref[i,:] = ∂N_i/∂(ξ,η,ζ) = constant)

Strain-displacement matrix (constant inside P1 element):
    B[m]  shape (6, 12),  Voigt order: [εxx, εyy, εzz, γxy, γyz, γxz]

Elasticity tensor (isotropic):
    λ = Eν / ((1+ν)(1−2ν)),  μ = E / (2(1+ν))
    C = diag([λ+2μ, λ+2μ, λ+2μ, μ, μ, μ]) + λ off-diagonal (3×3 block)

Element stiffness (exact for P1 — no Gauss quadrature needed):
    K_e[m] = Vₑ · Bᵀ C B    shape (12, 12)

Global assembly via COO scatter:
    K[dof_i, dof_j] += K_e[m, i_local, j_local]

DOF ordering for one element: [u₁ₓ, u₁ᵧ, u₁_z, u₂ₓ, ..., u₄_z]

Author: Graphite / Aristo Project
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

# ---------------------------------------------------------------------------
# Reference shape function gradients in (ξ, η, ζ) space
# Row i → (∂Nᵢ/∂ξ, ∂Nᵢ/∂η, ∂Nᵢ/∂ζ)
# N₁=1−ξ−η−ζ → [−1,−1,−1],  N₂=ξ → [1,0,0],  N₃=η → [0,1,0],  N₄=ζ → [0,0,1]
# ---------------------------------------------------------------------------
_DN_REF: np.ndarray = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0,  0.0,  0.0],
        [0.0,  1.0,  0.0],
        [0.0,  0.0,  1.0],
    ],
    dtype=np.float64,
)  # shape (4, 3)


# ===========================================================================
# Public API
# ===========================================================================


def build_elasticity_tensor(E: float, nu: float) -> np.ndarray:
    """
    Build the 6×6 isotropic linear-elastic constitutive matrix C (Voigt notation).

    Voigt strain/stress order: [xx, yy, zz, xy, yz, xz]

    Parameters
    ----------
    E : float
        Young's modulus (MPa).
    nu : float
        Poisson's ratio.

    Returns
    -------
    C : ndarray, shape (6, 6), dtype float64
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C = np.zeros((6, 6), dtype=np.float64)
    # Normal-stress diagonal
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    # Normal-stress coupling
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    # Shear diagonal (γ = 2ε convention for Voigt engineering shear strains)
    C[3, 3] = C[4, 4] = C[5, 5] = mu

    return C


def assemble_global_K(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Assemble the global stiffness matrix K for a P1 tetrahedral mesh.

    Fully vectorized — no Python loops over elements. Uses COO accumulation
    then converts to CSR for downstream solving.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Global node coordinates (mm).
    elements : ndarray, shape (M, 4)
        Tetrahedral connectivity, zero-based node indices.
    E : float
        Young's modulus (MPa).
    nu : float
        Poisson's ratio.

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (3N, 3N)
        Symmetric global stiffness matrix in CSR format.
    volumes : ndarray, shape (M,)
        Element volumes (mm³).

    Raises
    ------
    ValueError
        If any element has non-positive volume (degenerate or inverted tet).
    """
    N = nodes.shape[0]
    M = elements.shape[0]
    n_dof = 3 * N

    C = build_elasticity_tensor(E, nu)
    _, dN_phys, volumes = _compute_jacobians_and_grads(nodes, elements)

    # ---------- B matrices: (M, 6, 12) ----------
    B = _build_B_matrices(dN_phys)

    # ---------- K_e = Vₑ · Bᵀ C B (chunked COO assembly) ----------
    # Avoid allocating (M, 12, 12) for large lattice meshes (~2M+ tets).
    chunk = 10_000
    dof_offsets = np.arange(3, dtype=np.int64)
    K = None

    for start in range(0, M, chunk):
        end = min(start + chunk, M)
        B_chunk = B[start:end]
        vol_chunk = volumes[start:end]
        elems_chunk = elements[start:end]
        n_chunk = end - start

        CB = np.einsum("ij,mjk->mik", C, B_chunk)
        BtCB = np.einsum("mji,mjk->mik", B_chunk, CB)
        K_e = vol_chunk[:, np.newaxis, np.newaxis] * BtCB

        node_dofs = elems_chunk[:, :, np.newaxis] * 3 + dof_offsets
        global_dofs = node_dofs.reshape(n_chunk, 12)
        rows_coo = np.repeat(global_dofs, 12, axis=1).ravel()
        cols_coo = np.tile(global_dofs, (1, 12)).ravel()
        K_chunk = coo_matrix(
            (K_e.ravel(), (rows_coo, cols_coo)),
            shape=(n_dof, n_dof),
        )
        K = K_chunk if K is None else K + K_chunk

    if K is None:
        K = coo_matrix((n_dof, n_dof))
    K = K.tocsr()

    return K, volumes


def apply_dirichlet_bcs(
    K: csr_matrix,
    F: np.ndarray,
    fixed_dofs: np.ndarray,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Apply homogeneous Dirichlet BCs by static condensation (elimination).

    Extracts the reduced system for free DOFs only:
        K_free · u_free = F_free
    where u_free are the unconstrained displacements (fixed DOFs = 0).

    This is more numerically stable than the penalty method and produces
    a smaller, better-conditioned system.

    Parameters
    ----------
    K : csr_matrix, shape (3N, 3N)
        Global stiffness matrix.
    F : ndarray, shape (3N,)
        Global force vector.
    fixed_dofs : ndarray of int
        Global DOF indices with prescribed zero displacement.

    Returns
    -------
    K_free : csr_matrix, shape (n_free, n_free)
        Reduced stiffness matrix for unconstrained DOFs.
    F_free : ndarray, shape (n_free,)
        Reduced force vector.
    free_dofs : ndarray, shape (n_free,)
        Global DOF indices of the unconstrained DOFs (for solution mapping).

    Raises
    ------
    ValueError
        If no free DOFs remain after applying constraints.
    """
    n_dof = K.shape[0]

    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[fixed_dofs] = False
    free_dofs = np.where(free_mask)[0]

    if free_dofs.size == 0:
        raise ValueError(
            "No free DOFs remain after applying Dirichlet BCs — "
            "the system is fully constrained. Check fixed_face_ids."
        )

    # Row extraction (efficient on CSR), column extraction via CSC
    K_rows = K[free_dofs, :]
    K_free = K_rows.tocsc()[:, free_dofs].tocsr()
    F_free = F[free_dofs]

    return K_free, F_free, free_dofs


def compute_element_stresses(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    E: float,
    nu: float,
) -> np.ndarray:
    """
    Compute per-element Von Mises stress from the nodal displacement vector.

    Fully vectorized via einsum — no Python loops over elements.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)
    u : ndarray, shape (3N,)
        Global nodal displacement vector (mm units if E in MPa).
    E : float
        Young's modulus (MPa).
    nu : float
        Poisson's ratio.

    Returns
    -------
    von_mises : ndarray, shape (M,)
        Von Mises stress per element (MPa, raw physical units).
    """
    M = elements.shape[0]
    C = build_elasticity_tensor(E, nu)
    _, dN_phys, _ = _compute_jacobians_and_grads(nodes, elements)
    B = _build_B_matrices(dN_phys)  # (M, 6, 12)

    # ---------- Gather element displacements ----------
    # node_dofs: (M, 4, 3) → global_dofs: (M, 12)
    node_dofs = elements[:, :, np.newaxis] * 3 + np.arange(3, dtype=np.int64)
    global_dofs = node_dofs.reshape(M, 12)
    u_e = u[global_dofs]  # (M, 12)

    # ---------- Strains: εₘ = B[m] @ u_e[m] ----------
    eps = np.einsum("mij,mj->mi", B, u_e)   # (M, 6)

    # ---------- Stresses: σₘ = C @ εₘ ----------
    sigma = np.einsum("ij,mj->mi", C, eps)  # (M, 6)

    # ---------- Von Mises scalar ----------
    # σ_VM = √(½[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)²] + 3(τxy²+τyz²+τxz²))
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    txy, tyz, txz = sigma[:, 3], sigma[:, 4], sigma[:, 5]

    vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy ** 2 + tyz ** 2 + txz ** 2)
    )
    return vm  # (M,)


# ===========================================================================
# Private helpers
# ===========================================================================


def _compute_jacobians_and_grads(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Jacobian inverses, physical shape function gradients, and
    element volumes for all P1 tets. Fully vectorized.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)

    Returns
    -------
    J_inv : ndarray, shape (M, 3, 3)
    dN_phys : ndarray, shape (M, 4, 3)
        Physical-space shape function gradients.
        dN_phys[m, i, :] = (∂Nᵢ/∂x, ∂Nᵢ/∂y, ∂Nᵢ/∂z) for element m.
    volumes : ndarray, shape (M,)
        Element volumes (mm³).

    Raises
    ------
    ValueError
        If any element has non-positive volume.
    """
    # Gather node coordinates per element: (M, 4, 3)
    coords = nodes[elements]

    # Jacobian: J[m] = [[p₂−p₁], [p₃−p₁], [p₄−p₁]]  shape (M, 3, 3)
    J = np.stack(
        [
            coords[:, 1] - coords[:, 0],
            coords[:, 2] - coords[:, 0],
            coords[:, 3] - coords[:, 0],
        ],
        axis=1,
    )  # (M, 3, 3)

    det_J = np.linalg.det(J)    # (M,)
    volumes = det_J / 6.0       # (M,)

    n_bad = int(np.sum(volumes <= 0.0))
    if n_bad > 0:
        raise ValueError(
            f"{n_bad} tetrahedral element(s) have non-positive volume "
            "(degenerate or inverted). Ensure the input mesh is watertight "
            "with consistent winding order before running FEA."
        )

    J_inv = np.linalg.inv(J)  # (M, 3, 3)

    # Physical gradients: dN_phys[m,i,j] = Σₖ dN_ref[i,k] · J_inv[m,k,j]
    # einsum: 'ik, mkj → mij'  (i=node index, k=ref-coord, j=phys-coord)
    dN_phys = np.einsum("ik,mkj->mij", _DN_REF, J_inv)  # (M, 4, 3)

    return J_inv, dN_phys, volumes


def _build_B_matrices(dN_phys: np.ndarray) -> np.ndarray:
    """
    Build strain-displacement B matrices from physical shape function gradients.

    DOF ordering per element: [u₁ₓ, u₁ᵧ, u₁_z, u₂ₓ, u₂ᵧ, u₂_z, u₃ₓ, ..., u₄_z]
    Voigt strain ordering:    [εxx, εyy, εzz, γxy, γyz, γxz]

    The loop over 4 nodes (range(4)) is intentional and not an anti-pattern —
    it has only 4 iterations and the inner operations are fully vectorized (M,).

    Parameters
    ----------
    dN_phys : ndarray, shape (M, 4, 3)

    Returns
    -------
    B : ndarray, shape (M, 6, 12)
    """
    M = dN_phys.shape[0]
    B = np.zeros((M, 6, 12), dtype=np.float64)

    for i in range(4):
        c = i * 3  # starting DOF column for node i
        bx = dN_phys[:, i, 0]  # ∂Nᵢ/∂x  (M,)
        by = dN_phys[:, i, 1]  # ∂Nᵢ/∂y  (M,)
        bz = dN_phys[:, i, 2]  # ∂Nᵢ/∂z  (M,)

        B[:, 0, c + 0] = bx        # εxx = ∂u/∂x
        B[:, 1, c + 1] = by        # εyy = ∂v/∂y
        B[:, 2, c + 2] = bz        # εzz = ∂w/∂z
        B[:, 3, c + 0] = by        # γxy row: ∂u/∂y
        B[:, 3, c + 1] = bx        #        + ∂v/∂x
        B[:, 4, c + 1] = bz        # γyz row: ∂v/∂z
        B[:, 4, c + 2] = by        #        + ∂w/∂y
        B[:, 5, c + 0] = bz        # γxz row: ∂u/∂z
        B[:, 5, c + 2] = bx        #        + ∂w/∂x

    return B  # (M, 6, 12)
