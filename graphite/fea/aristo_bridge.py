"""
Aristo Continuum Bridge for Two-Scale Homogenized Lattice FEA.

Bridges offline micro-scale unit cell surrogates (MaterialTensorSurrogate)
with macro-scale continuum FEA in Aristo. Meshes macro CAD envelopes into
coarse continuum elements, maps spatial grading fields to element centroids,
assembles anisotropic global stiffness matrices, applies boundary conditions,
and recovers stress and strain fields without micro-scale mesh explosion.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.linear_solver import sparse_direct_solve
from graphite.aristo.stiffness_assembly import (
    _build_B_matrices,
    _compute_jacobians_and_grads,
)
from graphite.fea.surrogate import MaterialTensorSurrogate


@dataclass
class MacroMesh:
    """
    Coarse continuum volume mesh for macro-scale structural FEA.

    Attributes
    ----------
    nodes : np.ndarray
        Global node coordinates, shape (N, 3), dtype float64 (mm).
    elements : np.ndarray
        Element connectivity, shape (M, 4) for tet4 or (M, 8) for hex8, dtype int64.
    elem_type : Literal["tet4", "hex8"]
        Element formulation.
    element_centroids : np.ndarray
        Centroid coordinate of each element, shape (M, 3).
    element_volumes : np.ndarray
        Physical volume of each element in mm³, shape (M,).
    """

    nodes: np.ndarray
    elements: np.ndarray
    elem_type: Literal["tet4", "hex8"] = "tet4"
    element_centroids: np.ndarray = field(init=False)
    element_volumes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)

        coords = self.nodes[self.elements]  # shape (M, num_nodes, 3)
        self.element_centroids = np.mean(coords, axis=1)

        if self.elem_type == "tet4":
            J = np.stack(
                [
                    coords[:, 1] - coords[:, 0],
                    coords[:, 2] - coords[:, 0],
                    coords[:, 3] - coords[:, 0],
                ],
                axis=1,
            )
            det_J = np.linalg.det(J)
            vols = det_J / 6.0

            # Automatically fix any inverted elements by swapping nodes 1 and 2
            inverted = vols < 0.0
            if np.any(inverted):
                self.elements[inverted, 1], self.elements[inverted, 2] = (
                    self.elements[inverted, 2].copy(),
                    self.elements[inverted, 1].copy(),
                )
                coords = self.nodes[self.elements]
                J = np.stack(
                    [
                        coords[:, 1] - coords[:, 0],
                        coords[:, 2] - coords[:, 0],
                        coords[:, 3] - coords[:, 0],
                    ],
                    axis=1,
                )
                vols = np.linalg.det(J) / 6.0

            self.element_volumes = vols
        elif self.elem_type == "hex8":
            # Sum volume of 6 sub-tets per hex
            n0, n1, n2, n3, n4, n5, n6, n7 = [coords[:, i] for i in range(8)]

            def _tet_v(a, b, c, d):
                return (
                    np.abs(np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)))
                    / 6.0
                )

            self.element_volumes = (
                _tet_v(n0, n1, n2, n6)
                + _tet_v(n0, n2, n3, n6)
                + _tet_v(n0, n3, n7, n6)
                + _tet_v(n0, n7, n4, n6)
                + _tet_v(n0, n4, n5, n6)
                + _tet_v(n0, n5, n1, n6)
            )


@dataclass
class TwoScaleFEAResult:
    """
    Result container for two-scale macro continuum FEA.

    Attributes
    ----------
    mesh : MacroMesh
        Macro-continuum mesh.
    displacements : np.ndarray
        Nodal displacements (mm), shape (N, 3).
    element_strains : np.ndarray
        Voigt strains [xx, yy, zz, xy, yz, xz] per element, shape (M, 6).
    element_stresses : np.ndarray
        Voigt stresses (MPa) per element, shape (M, 6).
    element_von_mises : np.ndarray
        Von Mises stress per element (MPa), shape (M,).
    nodal_von_mises : np.ndarray
        Volume-weighted projected Von Mises stress per node (MPa), shape (N,).
    compliance_energy : float
        Total strain energy / compliance 0.5 * u^T F (mJ).
    max_displacement : float
        Maximum resultant nodal displacement magnitude (mm).
    max_von_mises : float
        Maximum element Von Mises stress (MPa).
    assembly_time_s : float
        Wall-clock time for stiffness assembly (s).
    solve_time_s : float
        Wall-clock time for sparse direct linear solve (s).
    solver_used : str
        Linear solver backend used ('scipy' or 'pypardiso').
    grading_values : np.ndarray
        Evaluated grading parameter at element centroids, shape (M,).
    """

    mesh: MacroMesh
    displacements: np.ndarray
    element_strains: np.ndarray
    element_stresses: np.ndarray
    element_von_mises: np.ndarray
    nodal_von_mises: np.ndarray
    compliance_energy: float
    max_displacement: float
    max_von_mises: float
    assembly_time_s: float
    solve_time_s: float
    solver_used: str
    grading_values: np.ndarray


# ===========================================================================
# Mesh Generators
# ===========================================================================


def create_box_continuum_mesh(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (10.0, 10.0, 10.0),
    ),
    subdivisions: tuple[int, int, int] = (10, 10, 10),
    elem_type: Literal["tet4", "hex8"] = "tet4",
) -> MacroMesh:
    """
    Create a structured box continuum volume mesh.

    Parameters
    ----------
    bounds : tuple
        ((x_min, y_min, z_min), (x_max, y_max, z_max)) in mm.
    subdivisions : tuple of int
        (nx, ny, nz) grid cells along each axis.
    elem_type : str
        'tet4' (6 tets per box cell) or 'hex8' (8-node brick).

    Returns
    -------
    MacroMesh
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounds
    nx, ny, nz = subdivisions

    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    z = np.linspace(z_min, z_max, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def _idx(i: int, j: int, k: int) -> int:
        return i * ((ny + 1) * (nz + 1)) + j * (nz + 1) + k

    if elem_type == "hex8":
        hexes = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    hexes.append(
                        [
                            _idx(i, j, k),
                            _idx(i + 1, j, k),
                            _idx(i + 1, j + 1, k),
                            _idx(i, j + 1, k),
                            _idx(i, j, k + 1),
                            _idx(i + 1, j, k + 1),
                            _idx(i + 1, j + 1, k + 1),
                            _idx(i, j + 1, k + 1),
                        ]
                    )
        elements = np.array(hexes, dtype=np.int64)
        return MacroMesh(nodes=nodes, elements=elements, elem_type="hex8")

    elif elem_type == "tet4":
        tets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n0 = _idx(i, j, k)
                    n1 = _idx(i + 1, j, k)
                    n2 = _idx(i + 1, j + 1, k)
                    n3 = _idx(i, j + 1, k)
                    n4 = _idx(i, j, k + 1)
                    n5 = _idx(i + 1, j, k + 1)
                    n6 = _idx(i + 1, j + 1, k + 1)
                    n7 = _idx(i, j + 1, k + 1)

                    # 6 tets per cube (Kuhn triangulation)
                    tets.extend(
                        [
                            [n0, n1, n2, n6],
                            [n0, n2, n3, n6],
                            [n0, n3, n7, n6],
                            [n0, n7, n4, n6],
                            [n0, n4, n5, n6],
                            [n0, n5, n1, n6],
                        ]
                    )
        elements = np.array(tets, dtype=np.int64)
        return MacroMesh(nodes=nodes, elements=elements, elem_type="tet4")
    else:
        raise ValueError(f"Unsupported elem_type: {elem_type}")


def generate_macro_continuum_mesh(
    cad_mesh: Any,
    target_element_size: float = 2.0,
    mesh_algorithm: int = 1,
) -> MacroMesh:
    """
    Generate a 3D linear tetrahedral macro-mesh for an arbitrary closed CAD surface using Gmsh.

    Parameters
    ----------
    cad_mesh : trimesh.Trimesh, str, or Path
        Closed watertight surface envelope.
    target_element_size : float
        Characteristic element size h (mm).
    mesh_algorithm : int
        Gmsh 3D mesh algorithm (1: Delaunay, 4: Frontal, 7: MMG3D).

    Returns
    -------
    MacroMesh
    """
    import gmsh
    import trimesh

    temp_path: Path | None = None
    if isinstance(cad_mesh, (str, Path)):
        stl_path = Path(cad_mesh).resolve()
    elif isinstance(cad_mesh, trimesh.Trimesh):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            temp_path = Path(f.name)
        cad_mesh.export(str(temp_path))
        stl_path = temp_path
    else:
        raise TypeError(f"Expected trimesh.Trimesh or Path, got {type(cad_mesh)}")

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cad_macro_continuum")
        gmsh.merge(str(stl_path))

        # Classify surfaces and form volume loop
        gmsh.model.mesh.classifySurfaces(
            angle=np.deg2rad(45.0), boundary=True, forReparametrization=True
        )
        gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()

        surfaces = gmsh.model.getEntities(2)
        surf_tags = [s[1] for s in surfaces]
        if not surf_tags:
            raise RuntimeError("Gmsh could not classify surface patches from CAD mesh.")

        loop = gmsh.model.geo.addSurfaceLoop(surf_tags)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        # Set mesh sizing
        gmsh.option.setNumber("Mesh.Algorithm3D", mesh_algorithm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_element_size * 0.7)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_element_size * 1.3)
        gmsh.model.mesh.generate(3)

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = node_coords.reshape(-1, 3)

        # Extract 4-node tets (element type 4 in Gmsh)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
        tet_idx = -1
        for i, etype in enumerate(elem_types):
            if etype == 4:
                tet_idx = i
                break

        if tet_idx == -1 or len(elem_node_tags[tet_idx]) == 0:
            raise RuntimeError("Gmsh failed to generate 3D tetrahedral elements.")

        raw_tets = elem_node_tags[tet_idx].reshape(-1, 4)

        # Map non-contiguous Gmsh node tags to contiguous 0-based indices
        tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
        tets = np.vectorize(tag_to_idx.get)(raw_tets).astype(np.int64)

        return MacroMesh(nodes=nodes, elements=tets, elem_type="tet4")
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


# ===========================================================================
# Field & Surrogate Mapping
# ===========================================================================


def map_grading_field_to_centroids(
    mesh: MacroMesh,
    grading_source: Callable[[np.ndarray], np.ndarray] | float | np.ndarray,
) -> np.ndarray:
    """
    Evaluate a spatial grading function or parameter at macro-element centroids.

    Parameters
    ----------
    mesh : MacroMesh
    grading_source : callable, float, or np.ndarray
        - Callable f(coords) -> values where coords is (M, 3).
        - Float for uniform field.
        - Array of shape (M,) for precomputed element values.

    Returns
    -------
    np.ndarray, shape (M,)
    """
    M = mesh.elements.shape[0]
    if callable(grading_source):
        vals = grading_source(mesh.element_centroids)
        return np.asarray(vals, dtype=np.float64).ravel()
    elif np.isscalar(grading_source):
        return np.full(M, float(grading_source), dtype=np.float64)
    else:
        arr = np.asarray(grading_source, dtype=np.float64).ravel()
        if arr.shape[0] != M:
            raise ValueError(
                f"Grading array length {arr.shape[0]} does not match element count {M}"
            )
        return arr


def evaluate_surrogate_elasticity(
    mesh: MacroMesh,
    surrogate: MaterialTensorSurrogate,
    param_values: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the homogenized elasticity tensor C^H for each macro-element via the surrogate.

    Parameters
    ----------
    mesh : MacroMesh
    surrogate : MaterialTensorSurrogate
    param_values : np.ndarray, shape (M,)

    Returns
    -------
    np.ndarray, shape (M, 6, 6)
    """
    return surrogate.evaluate_material_tensor(param_values)


# ===========================================================================
# Global Stiffness Matrix Assembly
# ===========================================================================


def assemble_anisotropic_global_K(
    mesh: MacroMesh,
    C_elements: np.ndarray,
    chunk_size: int = 10000,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Assemble the global stiffness matrix K for a P1 tetrahedral mesh with element-wise anisotropic stiffness tensors.

    Parameters
    ----------
    mesh : MacroMesh
    C_elements : np.ndarray
        Shape (M, 6, 6) or (6, 6) if uniform across all elements.
    chunk_size : int, optional
        Chunk size for vectorized COO assembly to bound memory.

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (3N, 3N)
    volumes : np.ndarray, shape (M,)
    """
    if mesh.elem_type != "tet4":
        raise NotImplementedError(
            f"Anisotropic assembly currently supports 'tet4', got {mesh.elem_type}"
        )

    N = mesh.nodes.shape[0]
    M = mesh.elements.shape[0]
    n_dof = 3 * N

    if C_elements.ndim == 2:
        C_elements = np.broadcast_to(C_elements[np.newaxis, :, :], (M, 6, 6))

    _, dN_phys, volumes = _compute_jacobians_and_grads(mesh.nodes, mesh.elements)
    B = _build_B_matrices(dN_phys)  # shape (M, 6, 12)

    dof_offsets = np.arange(3, dtype=np.int64)
    K = None

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        B_chunk = B[start:end]
        vol_chunk = volumes[start:end]
        elems_chunk = mesh.elements[start:end]
        C_chunk = C_elements[start:end]
        n_chunk = end - start

        # CB: (n_chunk, 6, 12) = C_chunk @ B_chunk
        CB = np.einsum("mij,mjk->mik", C_chunk, B_chunk)
        # BtCB: (n_chunk, 12, 12) = B_chunk^T @ CB
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
    return K.tocsr(), volumes


# ===========================================================================
# Boundary Conditions & Loading
# ===========================================================================


def find_boundary_nodes_by_plane(
    mesh: MacroMesh,
    axis: int,
    value: float,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Find node indices lying on or near a coordinate plane x[axis] == value.

    Parameters
    ----------
    mesh : MacroMesh
    axis : int
        0 for X, 1 for Y, 2 for Z.
    value : float
        Plane coordinate.
    tol : float
        Distance tolerance in mm.

    Returns
    -------
    np.ndarray of int
    """
    coords = mesh.nodes[:, axis]
    return np.where(np.abs(coords - value) <= tol)[0]


def apply_surface_traction(
    mesh: MacroMesh,
    node_indices: np.ndarray,
    total_force: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """
    Distribute a total force vector (Fx, Fy, Fz) across specified boundary nodes using area weighting.

    Parameters
    ----------
    mesh : MacroMesh
    node_indices : np.ndarray
        Indices of surface nodes.
    total_force : tuple or ndarray of float
        Total resultant force (N) along (x, y, z).

    Returns
    -------
    np.ndarray, shape (3 * N,)
        Global force vector F.
    """
    N = mesh.nodes.shape[0]
    F = np.zeros(3 * N, dtype=np.float64)
    if len(node_indices) == 0:
        return F

    force_vec = np.asarray(total_force, dtype=np.float64)
    node_set = set(np.asarray(node_indices, dtype=np.int64))

    # Compute tributary areas for nodes from surface faces
    if mesh.elem_type == "tet4":
        tet_faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        node_areas = {n: 0.0 for n in node_set}
        total_surf_area = 0.0
        for elem in mesh.elements:
            for f in tet_faces:
                fn = [elem[idx] for idx in f]
                if fn[0] in node_set and fn[1] in node_set and fn[2] in node_set:
                    p0, p1, p2 = mesh.nodes[fn]
                    area = 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
                    total_surf_area += area
                    for n in fn:
                        node_areas[n] += area / 3.0

        if total_surf_area > 1e-12:
            for n, a in node_areas.items():
                w = a / total_surf_area
                for comp in range(3):
                    F[n * 3 + comp] = force_vec[comp] * w
            return F

    # Fallback to uniform per-node distribution
    f_per_node = force_vec / len(node_indices)
    for comp in range(3):
        F[node_indices * 3 + comp] = f_per_node[comp]
    return F


# ===========================================================================
# Stress & Strain Field Recovery
# ===========================================================================


def recover_element_and_nodal_stresses(
    mesh: MacroMesh,
    C_elements: np.ndarray,
    u: np.ndarray,
    volumes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover element strains, element stresses, element Von Mises, and volume-weighted nodal Von Mises.

    Parameters
    ----------
    mesh : MacroMesh
    C_elements : np.ndarray, shape (M, 6, 6)
    u : np.ndarray, shape (3 * N,)
        Global nodal displacements.
    volumes : np.ndarray, shape (M,)

    Returns
    -------
    eps : np.ndarray, shape (M, 6)
    sigma : np.ndarray, shape (M, 6)
    element_vm : np.ndarray, shape (M,)
    nodal_vm : np.ndarray, shape (N,)
    """
    N = mesh.nodes.shape[0]
    M = mesh.elements.shape[0]

    if C_elements.ndim == 2:
        C_elements = np.broadcast_to(C_elements[np.newaxis, :, :], (M, 6, 6))

    _, dN_phys, _ = _compute_jacobians_and_grads(mesh.nodes, mesh.elements)
    B = _build_B_matrices(dN_phys)

    # Gather element displacements
    node_dofs = mesh.elements[:, :, np.newaxis] * 3 + np.arange(3, dtype=np.int64)
    global_dofs = node_dofs.reshape(M, 12)
    u_e = u[global_dofs]  # shape (M, 12)

    # Element strains: eps[m] = B[m] @ u_e[m]
    eps = np.einsum("mij,mj->mi", B, u_e)  # (M, 6)

    # Element stresses: sigma[m] = C[m] @ eps[m]
    sigma = np.einsum("mij,mj->mi", C_elements, eps)  # (M, 6)

    # Von Mises: sxx, syy, szz, txy, tyz, txz
    sxx, syy, szz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    txy, tyz, txz = sigma[:, 3], sigma[:, 4], sigma[:, 5]

    element_vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )

    # Volume-weighted nodal stress projection
    nodal_vm_weighted = np.zeros(N, dtype=np.float64)
    nodal_vol_sum = np.zeros(N, dtype=np.float64)

    for i in range(4):
        np.add.at(nodal_vm_weighted, mesh.elements[:, i], volumes * element_vm)
        np.add.at(nodal_vol_sum, mesh.elements[:, i], volumes)

    nodal_vol_sum = np.maximum(nodal_vol_sum, 1e-15)
    nodal_vm = nodal_vm_weighted / nodal_vol_sum

    return eps, sigma, element_vm, nodal_vm


# ===========================================================================
# Master Two-Scale FEA Orchestrator
# ===========================================================================


def run_two_scale_macro_fea(
    mesh: MacroMesh,
    surrogate: MaterialTensorSurrogate,
    grading_source: Callable[[np.ndarray], np.ndarray] | float | np.ndarray,
    fixed_nodes: np.ndarray,
    forces: np.ndarray,
    fixed_components: tuple[int, ...] = (0, 1, 2),
    config: AristoConfig | None = None,
) -> TwoScaleFEAResult:
    """
    Execute two-scale macro FEA on a continuum mesh using a homogenized constitutive surrogate.

    Parameters
    ----------
    mesh : MacroMesh
        The continuum macro-mesh.
    surrogate : MaterialTensorSurrogate
        The constitutive tensor surrogate mapping grading values to C^H.
    grading_source : callable, float, or np.ndarray
        Spatial grading parameter function or values.
    fixed_nodes : np.ndarray
        Node indices with prescribed Dirichlet displacement = 0.
    forces : np.ndarray, shape (3 * N,)
        Global applied force vector F (N).
    fixed_components : tuple of int
        Which displacement components to fix (0: x, 1: y, 2: z).
    config : AristoConfig, optional
        Solver configuration.

    Returns
    -------
    TwoScaleFEAResult
    """
    N = mesh.nodes.shape[0]
    n_dof = 3 * N

    # 1. Map grading field to element centroids
    param_values = map_grading_field_to_centroids(mesh, grading_source)

    # 2. Evaluate surrogate material tensor for all elements
    t_start = time.perf_counter()
    C_elements = evaluate_surrogate_elasticity(mesh, surrogate, param_values)

    # 3. Assemble global anisotropic stiffness matrix
    K, volumes = assemble_anisotropic_global_K(mesh, C_elements)
    t_assembly = time.perf_counter() - t_start

    # 4. Apply Dirichlet BCs via elimination (static condensation)
    dof_mask = np.ones(n_dof, dtype=bool)
    fixed_dof_list = []
    for comp in fixed_components:
        fixed_dof_list.append(fixed_nodes * 3 + comp)
    if fixed_dof_list:
        all_fixed_dofs = np.unique(np.concatenate(fixed_dof_list))
        dof_mask[all_fixed_dofs] = False
    else:
        all_fixed_dofs = np.array([], dtype=np.int64)

    free_dofs = np.where(dof_mask)[0]
    if len(free_dofs) == 0:
        raise ValueError("All DOFs are constrained. No free DOFs to solve.")

    K_free = K[free_dofs, :][:, free_dofs]
    F_free = forces[free_dofs]

    # 5. Direct sparse solve
    t_solve_start = time.perf_counter()
    u_free, solver_used = sparse_direct_solve(K_free, F_free, config=config)
    t_solve = time.perf_counter() - t_solve_start

    # 6. Reconstruct full displacement field
    u = np.zeros(n_dof, dtype=np.float64)
    u[free_dofs] = u_free
    displacements = u.reshape(N, 3)

    # 7. Recover strains, stresses, and Von Mises
    eps, sigma, element_vm, nodal_vm = recover_element_and_nodal_stresses(
        mesh, C_elements, u, volumes
    )

    # 8. Post-processing metrics
    compliance = float(0.5 * np.dot(u, forces))
    disp_norms = np.linalg.norm(displacements, axis=1)
    max_disp = float(np.max(disp_norms))
    max_vm = float(np.max(element_vm))

    return TwoScaleFEAResult(
        mesh=mesh,
        displacements=displacements,
        element_strains=eps,
        element_stresses=sigma,
        element_von_mises=element_vm,
        nodal_von_mises=nodal_vm,
        compliance_energy=compliance,
        max_displacement=max_disp,
        max_von_mises=max_vm,
        assembly_time_s=t_assembly,
        solve_time_s=t_solve,
        solver_used=solver_used,
        grading_values=param_values,
    )


# ===========================================================================
# Visualization & VTK Export
# ===========================================================================


def export_two_scale_result_vtk(
    result: TwoScaleFEAResult,
    filepath: str | Path,
) -> Path:
    """
    Export the two-scale macro FEA result to a VTK Unstructured Grid (.vtu).

    Parameters
    ----------
    result : TwoScaleFEAResult
    filepath : str or Path
        Target file path (.vtu or .vtk).

    Returns
    -------
    Path
    """
    import pyvista as pv

    out_p = Path(filepath).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    mesh = result.mesh
    M = mesh.elements.shape[0]

    if mesh.elem_type == "tet4":
        cells = np.column_stack([np.full(M, 4, dtype=np.int64), mesh.elements]).ravel()
        cell_types = np.full(M, pv.CellType.TETRA, dtype=np.uint8)
    elif mesh.elem_type == "hex8":
        cells = np.column_stack([np.full(M, 8, dtype=np.int64), mesh.elements]).ravel()
        cell_types = np.full(M, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported elem_type: {mesh.elem_type}")

    grid = pv.UnstructuredGrid(cells, cell_types, mesh.nodes)

    # Point Data
    grid.point_data["displacement_mm"] = result.displacements
    grid.point_data["displacement_mag_mm"] = np.linalg.norm(
        result.displacements, axis=1
    )
    grid.point_data["von_mises_nodal_MPa"] = result.nodal_von_mises

    # Cell Data
    grid.cell_data["von_mises_elem_MPa"] = result.element_von_mises
    grid.cell_data["grading_field"] = result.grading_values
    grid.cell_data["stress_xx_MPa"] = result.element_stresses[:, 0]
    grid.cell_data["stress_yy_MPa"] = result.element_stresses[:, 1]
    grid.cell_data["stress_zz_MPa"] = result.element_stresses[:, 2]

    grid.save(str(out_p))
    return out_p
