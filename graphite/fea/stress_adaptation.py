"""
Graphite FEA Subsystem - Closed-Loop Stress-Adaptive Topology Optimization (Path A).

Implements Fully Stressed Design (FSD) with move limits, volume fraction bisection,
and spatial neighborhood filtering. Iteratively drives element relative densities
until local stresses equilibrate with the allowable material limit without open-loop error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from graphite.aristo.aristo_config import AristoConfig
from graphite.fea.aristo_bridge import (
    MacroMesh,
    TwoScaleFEAResult,
    run_two_scale_macro_fea,
)
from graphite.fea.surrogate import MaterialTensorSurrogate


@dataclass
class StressAdaptationConfig:
    """
    Configuration for closed-loop stress-adaptive lattice optimization.

    Attributes
    ----------
    target_stress : float
        Allowable Von Mises stress target (MPa) for fully stressed design.
    target_volume_fraction : float | None
        Optional global target volume fraction. If specified, a 1D bisection
        scaling step is applied each iteration to conserve total material volume.
    relaxation_eta : float
        Damping exponent for the stress ratio (phi * (sigma / sigma_target)^eta).
        Typically 0.20 to 0.40 to guarantee monotonic convergence. Default 0.35.
    move_limit : float
        Maximum allowed change in relative density |dphi| per iteration. Default 0.10.
    min_density : float
        Minimum allowable element relative density. Default 0.10.
    max_density : float
        Maximum allowable element relative density. Default 0.60.
    max_iterations : int
        Maximum number of optimization iterations. Default 30.
    convergence_tol : float
        Convergence threshold on max|dphi| between consecutive iterations. Default 1e-3.
    filter_radius : float
        Spatial neighborhood filter radius (mm). If > 0, applies distance-weighted
        density smoothing to eliminate checkerboarding. Default 0.0 (disabled).
    """

    target_stress: float = 50.0
    target_volume_fraction: float | None = None
    relaxation_eta: float = 0.35
    move_limit: float = 0.10
    min_density: float = 0.10
    max_density: float = 0.60
    max_iterations: int = 30
    convergence_tol: float = 1e-3
    filter_radius: float = 0.0


@dataclass
class OptimizationIterationRecord:
    """Telemetry record for a single optimization iteration."""

    iteration: int
    compliance: float
    max_von_mises: float
    mean_von_mises: float
    volume_fraction: float
    max_delta_phi: float
    solve_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TwoScaleOptimizationResult:
    """
    Results of closed-loop two-scale stress-adaptive lattice optimization.

    Attributes
    ----------
    mesh : MacroMesh
        Macro continuum mesh.
    optimal_densities : np.ndarray
        Optimal converged relative density per element, shape (M,).
    initial_densities : np.ndarray
        Initial relative density per element, shape (M,).
    final_fea_result : TwoScaleFEAResult
        FEA displacement, strain, and stress state under optimal densities.
    history : list[OptimizationIterationRecord]
        Convergence telemetry per iteration.
    total_time_s : float
        Total wall-clock runtime for the optimization loop (seconds).
    iterations_completed : int
        Number of iterations executed.
    converged : bool
        True if max|dphi| < convergence_tol before reaching max_iterations.
    config : StressAdaptationConfig
        Configuration used for optimization.
    """

    mesh: MacroMesh
    optimal_densities: np.ndarray
    initial_densities: np.ndarray
    final_fea_result: TwoScaleFEAResult
    history: list[OptimizationIterationRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    iterations_completed: int = 0
    converged: bool = False
    config: StressAdaptationConfig = field(default_factory=StressAdaptationConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return {
            "converged": self.converged,
            "iterations_completed": self.iterations_completed,
            "total_time_s": self.total_time_s,
            "initial_volume_fraction": float(
                np.sum(self.mesh.element_volumes * self.initial_densities)
                / np.sum(self.mesh.element_volumes)
            ),
            "final_volume_fraction": float(
                np.sum(self.mesh.element_volumes * self.optimal_densities)
                / np.sum(self.mesh.element_volumes)
            ),
            "final_compliance": float(self.final_fea_result.compliance_energy),
            "final_max_von_mises": float(self.final_fea_result.max_von_mises),
            "history": [rec.to_dict() for rec in self.history],
        }

    def save(self, filepath: str | Path) -> None:
        """Save optimization summary to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ===========================================================================
# Algorithmic Operators
# ===========================================================================


def build_neighborhood_filter(mesh: MacroMesh, filter_radius: float) -> sp.csr_matrix:
    """
    Construct a sparse distance-weighted neighborhood smoothing matrix.

    W_ij = max(0, 1 - ||x_i - x_j|| / R) * V_j
    Normalized such that row sums equal 1.

    Parameters
    ----------
    mesh : MacroMesh
        Macro-continuum mesh with centroids and element_volumes.
    filter_radius : float
        Smoothing filter radius (mm).

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-normalized smoothing operator of shape (M, M).
    """
    if filter_radius <= 0.0:
        return sp.eye(mesh.elements.shape[0], format="csr")

    centroids = mesh.element_centroids
    volumes = mesh.element_volumes
    M = centroids.shape[0]

    tree = cKDTree(centroids)
    pairs = tree.query_pairs(r=filter_radius, output_type="ndarray")

    rows = []
    cols = []
    data = []

    # Diagonal self-weights
    rows.extend(range(M))
    cols.extend(range(M))
    data.extend([1.0 * volumes[i] for i in range(M)])

    # Off-diagonal neighbor pairs
    if len(pairs) > 0:
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        dist = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
        w = (1.0 - dist / filter_radius)

        # i -> j
        rows.extend(i_idx)
        cols.extend(j_idx)
        data.extend(w * volumes[j_idx])

        # j -> i
        rows.extend(j_idx)
        cols.extend(i_idx)
        data.extend(w * volumes[i_idx])

    W = sp.csr_matrix((data, (rows, cols)), shape=(M, M), dtype=np.float64)

    # Normalize rows
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0.0] = 1.0
    inv_diag = sp.diags(1.0 / row_sums, format="csr")
    W_norm = inv_diag @ W

    return W_norm.tocsr()


def apply_volume_bisection_scaling(
    densities: np.ndarray,
    element_volumes: np.ndarray,
    target_volume_fraction: float,
    min_density: float,
    max_density: float,
    max_bisection_iter: int = 25,
) -> np.ndarray:
    """
    Scale densities using 1D bisection to strictly satisfy target volume fraction.

    Finds multiplier lambda > 0 such that:
        sum(V_m * clamp(lambda * phi_m, min_density, max_density)) = target_vf * sum(V_m)

    Parameters
    ----------
    densities : np.ndarray
        Unscaled proposed relative densities, shape (M,).
    element_volumes : np.ndarray
        Element volumes, shape (M,).
    target_volume_fraction : float
        Target global volume fraction in [min_density, max_density].
    min_density : float
        Lower density bound.
    max_density : float
        Upper density bound.
    max_bisection_iter : int, optional
        Maximum bisection steps, by default 25.

    Returns
    -------
    np.ndarray
        Scaled and clamped relative densities, shape (M,).
    """
    total_volume = np.sum(element_volumes)
    target_material_vol = target_volume_fraction * total_volume

    l_low = 1e-4
    l_high = 1e4

    def _eval_vol(lam: float) -> float:
        clamped = np.clip(lam * densities, min_density, max_density)
        return float(np.sum(element_volumes * clamped))

    # Fast bisection
    for _ in range(max_bisection_iter):
        l_mid = 0.5 * (l_low + l_high)
        vol_mid = _eval_vol(l_mid)
        if vol_mid < target_material_vol:
            l_low = l_mid
        else:
            l_high = l_mid

    l_opt = 0.5 * (l_low + l_high)
    return np.clip(l_opt * densities, min_density, max_density)


# ===========================================================================
# Main Closed-Loop Optimization Driver
# ===========================================================================


def optimize_lattice_density_fsd(
    mesh: MacroMesh,
    surrogate: MaterialTensorSurrogate,
    fixed_nodes: np.ndarray,
    forces: np.ndarray,
    config: StressAdaptationConfig | None = None,
    initial_densities: np.ndarray | float | None = None,
    fixed_components: tuple[int, ...] = (0, 1, 2),
    aristo_config: AristoConfig | None = None,
) -> TwoScaleOptimizationResult:
    """
    Execute closed-loop stress-adaptive Fully Stressed Design (FSD) optimization.

    Parameters
    ----------
    mesh : MacroMesh
        Macro-continuum mesh (Tet4 or Hex8).
    surrogate : MaterialTensorSurrogate
        Calibrated constitutive tensor surrogate.
    fixed_nodes : np.ndarray
        Node indices with prescribed zero-displacement boundary conditions.
    forces : np.ndarray
        External nodal force vector, shape (N, 3) or (3*N,).
    config : StressAdaptationConfig, optional
        Optimization hyperparameters. If None, default settings are used.
    initial_densities : np.ndarray or float, optional
        Starting relative density field. If None, defaults to target_volume_fraction
        or the midpoint of (min_density, max_density).
    fixed_components : tuple of int, optional
        Displacement components to fix (0=X, 1=Y, 2=Z), by default (0, 1, 2).
    aristo_config : AristoConfig, optional
        FEA linear solver configuration.

    Returns
    -------
    TwoScaleOptimizationResult
        Object containing converged density field, final FEA result, and telemetry history.
    """
    if config is None:
        config = StressAdaptationConfig()

    M = mesh.elements.shape[0]
    element_volumes = mesh.element_volumes

    # 1. Initialize density field
    if initial_densities is None:
        if config.target_volume_fraction is not None:
            phi_init = np.full(M, config.target_volume_fraction, dtype=np.float64)
        else:
            phi_init = np.full(M, 0.5 * (config.min_density + config.max_density), dtype=np.float64)
    elif np.isscalar(initial_densities):
        phi_init = np.full(M, float(initial_densities), dtype=np.float64)
    else:
        phi_init = np.asarray(initial_densities, dtype=np.float64).copy()

    phi_curr = np.clip(phi_init, config.min_density, config.max_density)

    # 2. Build spatial filter if requested
    W_filter = None
    if config.filter_radius > 0.0:
        W_filter = build_neighborhood_filter(mesh, config.filter_radius)

    history: list[OptimizationIterationRecord] = []
    t_start = time.perf_counter()
    converged = False
    last_fea_result: TwoScaleFEAResult | None = None

    # 3. Iteration loop
    for it in range(config.max_iterations):
        # a. Run macro FEA with current density field
        fea_res = run_two_scale_macro_fea(
            mesh=mesh,
            surrogate=surrogate,
            grading_source=phi_curr,
            fixed_nodes=fixed_nodes,
            forces=forces,
            fixed_components=fixed_components,
            config=aristo_config,
        )
        last_fea_result = fea_res

        # b. Recover stresses & compute metrics
        vm_elem = fea_res.element_von_mises
        max_vm = float(np.max(vm_elem))
        mean_vm = float(np.mean(vm_elem))
        comp = float(fea_res.compliance_energy)
        curr_vf = float(np.sum(element_volumes * phi_curr) / np.sum(element_volumes))

        # c. Check convergence
        max_dphi = 0.0
        if it > 0 and len(history) > 0:
            max_dphi = float(np.max(np.abs(phi_curr - phi_prev)))
            if max_dphi < config.convergence_tol:
                converged = True
                record = OptimizationIterationRecord(
                    iteration=it,
                    compliance=comp,
                    max_von_mises=max_vm,
                    mean_von_mises=mean_vm,
                    volume_fraction=curr_vf,
                    max_delta_phi=max_dphi,
                    solve_time_s=fea_res.solve_time_s,
                )
                history.append(record)
                break

        record = OptimizationIterationRecord(
            iteration=it,
            compliance=comp,
            max_von_mises=max_vm,
            mean_von_mises=mean_vm,
            volume_fraction=curr_vf,
            max_delta_phi=max_dphi,
            solve_time_s=fea_res.solve_time_s,
        )
        history.append(record)

        if it == config.max_iterations - 1:
            break

        phi_prev = phi_curr.copy()

        # d. Fully Stressed Design (FSD) update
        # ratio = (sigma_VM / sigma_target) ^ eta
        vm_safe = np.maximum(vm_elem, 1e-6)
        stress_ratio = (vm_safe / config.target_stress) ** config.relaxation_eta
        # Clamp ratio to prevent explosive jumps
        stress_ratio = np.clip(stress_ratio, 0.20, 5.0)

        phi_proposed = phi_curr * stress_ratio

        # e. Apply move limit: |phi_new - phi_curr| <= move_limit
        lower_bound = np.maximum(config.min_density, phi_curr - config.move_limit)
        upper_bound = np.minimum(config.max_density, phi_curr + config.move_limit)
        phi_bounded = np.clip(phi_proposed, lower_bound, upper_bound)

        # f. Apply volume fraction conservation if requested
        if config.target_volume_fraction is not None:
            phi_scaled = apply_volume_bisection_scaling(
                densities=phi_bounded,
                element_volumes=element_volumes,
                target_volume_fraction=config.target_volume_fraction,
                min_density=config.min_density,
                max_density=config.max_density,
            )
        else:
            phi_scaled = phi_bounded

        # g. Apply spatial neighborhood filter if active
        if W_filter is not None:
            phi_filtered = W_filter.dot(phi_scaled)
            phi_curr = np.clip(phi_filtered, config.min_density, config.max_density)
        else:
            phi_curr = phi_scaled

    t_total = time.perf_counter() - t_start

    assert last_fea_result is not None

    return TwoScaleOptimizationResult(
        mesh=mesh,
        optimal_densities=phi_curr,
        initial_densities=phi_init,
        final_fea_result=last_fea_result,
        history=history,
        total_time_s=t_total,
        iterations_completed=len(history),
        converged=converged,
        config=config,
    )


# ===========================================================================
# Visualization & VTK Export
# ===========================================================================


def export_optimization_result_vtk(
    result: TwoScaleOptimizationResult,
    filepath: str | Path,
) -> None:
    """
    Export optimization results and fields to a ParaView/PyVista VTU file.

    Parameters
    ----------
    result : TwoScaleOptimizationResult
        The result object returned by optimize_lattice_density_fsd.
    filepath : str or Path
        Destination path ending in .vtu.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required to export optimization results to VTU.") from exc

    mesh = result.mesh
    M = mesh.elements.shape[0]
    N = mesh.nodes.shape[0]

    if mesh.elem_type == "tet4":
        cell_type = pv.CellType.TETRA
    elif mesh.elem_type == "hex8":
        cell_type = pv.CellType.HEXAHEDRON
    else:
        raise ValueError(f"Unsupported elem_type: {mesh.elem_type}")

    n_nodes_per_elem = mesh.elements.shape[1]
    cells = np.empty((M, n_nodes_per_elem + 1), dtype=np.int64)
    cells[:, 0] = n_nodes_per_elem
    cells[:, 1:] = mesh.elements
    cell_types = np.full(M, cell_type, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, mesh.nodes)

    # Element Data
    grid.cell_data["optimal_density"] = result.optimal_densities
    grid.cell_data["initial_density"] = result.initial_densities
    grid.cell_data["density_change"] = result.optimal_densities - result.initial_densities
    grid.cell_data["element_von_mises_MPa"] = result.final_fea_result.element_von_mises

    # Nodal Data
    u = result.final_fea_result.displacements
    grid.point_data["displacements_mm"] = u
    grid.point_data["displacement_magnitude_mm"] = np.linalg.norm(u, axis=1)
    grid.point_data["nodal_von_mises_MPa"] = result.final_fea_result.nodal_von_mises

    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(dest))


def render_optimization_summary_png(
    result: TwoScaleOptimizationResult,
    out_path_3d: str | Path,
    out_path_convergence: str | Path | None = None,
    window_size: tuple[int, int] = (1600, 600),
) -> tuple[Path, Path | None]:
    """
    Render publication-quality PNG images of 3D fields and convergence history.

    Parameters
    ----------
    result : TwoScaleOptimizationResult
        Optimization result container.
    out_path_3d : str or Path
        Target filepath for the 3D side-by-side render PNG (Density + Stress).
    out_path_convergence : str or Path, optional
        Target filepath for the 3-panel convergence plot PNG.
    window_size : tuple of int, optional
        PyVista window resolution in pixels, default (1600, 600).

    Returns
    -------
    tuple of (Path, Path or None)
        Paths to the generated PNG files.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("pyvista is required to render 3D optimization summary PNG.") from exc

    dest_3d = Path(out_path_3d)
    dest_3d.parent.mkdir(parents=True, exist_ok=True)

    mesh = result.mesh
    M = mesh.elements.shape[0]
    cell_type = pv.CellType.TETRA if mesh.elem_type == "tet4" else pv.CellType.HEXAHEDRON
    n_nodes_per_elem = mesh.elements.shape[1]

    cells = np.empty((M, n_nodes_per_elem + 1), dtype=np.int64)
    cells[:, 0] = n_nodes_per_elem
    cells[:, 1:] = mesh.elements
    cell_types = np.full(M, cell_type, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, mesh.nodes)
    grid.cell_data["optimal_density"] = result.optimal_densities
    grid.cell_data["element_von_mises_MPa"] = result.final_fea_result.element_von_mises

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=window_size)

    # Subplot 0: Density
    plotter.subplot(0, 0)
    plotter.add_text("Optimal Relative Density phi*(x)", font_size=11, color="black")
    plotter.add_mesh(
        grid,
        scalars="optimal_density",
        cmap="viridis",
        show_edges=True,
        edge_color="#555555",
        line_width=0.4,
        scalar_bar_args={"title": "Relative Density", "color": "black", "vertical": True},
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.1)
    plotter.set_background("white")

    # Subplot 1: Von Mises Stress
    plotter.subplot(0, 1)
    plotter.add_text("Equilibrated Von Mises Stress (MPa)", font_size=11, color="black")
    plotter.add_mesh(
        grid,
        scalars="element_von_mises_MPa",
        cmap="turbo",
        show_edges=True,
        edge_color="#555555",
        line_width=0.4,
        scalar_bar_args={"title": "Von Mises (MPa)", "color": "black", "vertical": True},
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.1)
    plotter.set_background("white")

    plotter.screenshot(str(dest_3d))
    plotter.close()

    dest_conv = None
    if out_path_convergence is not None and len(result.history) > 0:
        import matplotlib.pyplot as plt

        dest_conv = Path(out_path_convergence)
        dest_conv.parent.mkdir(parents=True, exist_ok=True)

        history = result.history
        iters = [h.iteration for h in history]
        comp = [h.compliance for h in history]
        mean_vm = [h.mean_von_mises for h in history]
        max_vm = [h.max_von_mises for h in history]
        dphi = [h.max_delta_phi for h in history]

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

        # Compliance
        axs[0].plot(iters, comp, "b-o", lw=2, markersize=5)
        axs[0].set_xlabel("Iteration", fontweight="bold")
        axs[0].set_ylabel("Strain Energy / Compliance (mJ)", fontweight="bold")
        axs[0].set_title("Compliance Minimization", fontweight="bold")
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # Stress
        axs[1].plot(iters, mean_vm, "g-s", lw=2, markersize=5, label="Mean Von Mises")
        axs[1].plot(iters, max_vm, "r--", lw=1.5, label="Peak Von Mises")
        axs[1].axhline(result.config.target_stress, color="k", linestyle=":", label=f"Target ({result.config.target_stress:.1f} MPa)")
        axs[1].set_xlabel("Iteration", fontweight="bold")
        axs[1].set_ylabel("Stress (MPa)", fontweight="bold")
        axs[1].set_title("Stress Redistribution", fontweight="bold")
        axs[1].legend(frameon=True)
        axs[1].grid(True, linestyle="--", alpha=0.6)

        # Delta Phi
        if len(iters) > 1:
            axs[2].plot(iters[1:], dphi[1:], "m-^", lw=2, markersize=5)
        else:
            axs[2].plot(iters, dphi, "m-^", lw=2, markersize=5)
        axs[2].set_xlabel("Iteration", fontweight="bold")
        axs[2].set_ylabel("max |Delta phi|", fontweight="bold")
        axs[2].set_title("Density Convergence Step Size", fontweight="bold")
        axs[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(str(dest_conv))
        plt.close()

    return dest_3d, dest_conv


def realize_optimized_tpms_lattice(
    result: TwoScaleOptimizationResult,
    lattice_type: str = "Gyroid",
    cell_size: float = 10.0,
    resolution: tuple[int, int, int] = (120, 30, 30),
    out_stl: str | Path | None = None,
    taubin_iterations: int = 15,
) -> Any:
    """
    Synthesize a physical, watertight, 3D printable TPMS mesh from converged optimal densities.

    Parameters
    ----------
    result : TwoScaleOptimizationResult
        Optimization result containing optimal element densities and macro-mesh.
    lattice_type : str
        TPMS architecture ('Gyroid', 'Schwarz_P', 'Diamond'). Default 'Gyroid'.
    cell_size : float
        Unit cell period in mm. Default 10.0.
    resolution : tuple of int
        Voxel grid resolution (nx, ny, nz). Default (120, 30, 30).
    out_stl : str or Path, optional
        Destination filepath for the exported STL mesh.
    taubin_iterations : int
        Number of Taubin non-shrinking smoothing iterations. Default 15.

    Returns
    -------
    trimesh.Trimesh
        The smoothed, watertight 3D printable lattice surface mesh.
    """
    from scipy.interpolate import NearestNDInterpolator
    from skimage.measure import marching_cubes
    import trimesh
    from graphite.math.tpms import evaluate_tpms
    from graphite.mesh.smoothing import smooth_mesh_taubin

    mesh = result.mesh
    nodes = mesh.nodes
    min_b = np.min(nodes, axis=0)
    max_b = np.max(nodes, axis=0)

    nx, ny, nz = resolution
    x = np.linspace(min_b[0], max_b[0], nx)
    y = np.linspace(min_b[1], max_b[1], ny)
    z = np.linspace(min_b[2], max_b[2], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Spatial interpolator for density
    interp_phi = NearestNDInterpolator(mesh.element_centroids, result.optimal_densities)
    phi_grid = interp_phi(X, Y, Z)

    # TPMS Minimal surface evaluation
    k = 2.0 * np.pi / cell_size
    F_tpms = evaluate_tpms(lattice_type, k, X, Y, Z)

    # Sheet TPMS: solid where |F| <= tau(phi)
    tau_grid = 1.15 * phi_grid
    sdf_sheet = np.abs(F_tpms) - tau_grid

    # Bounding envelope constraint
    sdf_box = np.maximum.reduce([
        min_b[0] - X, X - max_b[0],
        min_b[1] - Y, Y - max_b[1],
        min_b[2] - Z, Z - max_b[2],
    ])
    sdf_part = np.maximum(sdf_sheet, sdf_box)

    spacing = (
        (max_b[0] - min_b[0]) / (nx - 1),
        (max_b[1] - min_b[1]) / (ny - 1),
        (max_b[2] - min_b[2]) / (nz - 1),
    )
    verts, faces, _, _ = marching_cubes(sdf_part, level=0.0, spacing=spacing)
    verts += min_b

    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Apply Taubin non-shrinking smoothing
    if taubin_iterations > 0:
        tri_mesh = smooth_mesh_taubin(tri_mesh, iterations=taubin_iterations, lamb=0.5, nu=-0.53)

    if out_stl is not None:
        stl_path = Path(out_stl)
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        tri_mesh.export(str(stl_path))

    return tri_mesh


def realize_optimized_strut_lattice(
    result: TwoScaleOptimizationResult,
    rule_name: str = "octet",
    cell_size: float = 2.0,
    out_stl: str | Path | None = None,
    clean_miter: bool = True,
    cad_mesh: Any | None = None,
    circular_segments: int = 12,
) -> Any:
    """
    Synthesize an explicit strut lattice mesh with clean mitered joints from converged densities.

    Parameters
    ----------
    result : TwoScaleOptimizationResult
        Optimization result containing optimal element densities and macro-mesh.
    rule_name : str
        Topology rule ('octet', 'star', 'grid', 'cross'). Default 'octet'.
    cell_size : float
        Unit cell period in mm. Default 2.0.
    out_stl : str or Path, optional
        Destination filepath for the exported STL mesh.
    clean_miter : bool
        If True, builds clean mitered truss joints with bisector-plane cut strut ends.
    cad_mesh : trimesh.Trimesh, optional
        Optional bounding CAD mesh to cull outside cells. If None, uses macro-mesh bounding box.
    circular_segments : int
        Radial resolution of strut cylinders. Default 12.

    Returns
    -------
    trimesh.Trimesh
    """
    from scipy.interpolate import NearestNDInterpolator
    from scipy.spatial import cKDTree
    import trimesh
    from graphite.explicit.geometry_module import build_clean_miter_truss
    from graphite.explicit.hex_topology_module import get_hex_topology_rule

    macro_nodes = result.mesh.nodes
    min_b = np.min(macro_nodes, axis=0)
    max_b = np.max(macro_nodes, axis=0)

    # 1. Grid of cell coordinates
    xs = np.arange(min_b[0], max_b[0] - 1e-5, cell_size)
    ys = np.arange(min_b[1], max_b[1] - 1e-5, cell_size)
    zs = np.arange(min_b[2], max_b[2] - 1e-5, cell_size)

    rule = get_hex_topology_rule(rule_name)
    builder = rule.builder

    # Density interpolator
    interp_phi = NearestNDInterpolator(result.mesh.element_centroids, result.optimal_densities)

    # Gather cells
    all_cell_corners = []
    all_cell_densities = []

    grid_coords = []
    for x in xs:
        for y in ys:
            for z in zs:
                grid_coords.append([x, y, z])
    grid_coords = np.array(grid_coords, dtype=np.float64)
    centroids = grid_coords + 0.5 * cell_size

    if cad_mesh is not None:
        try:
            mask_inside = np.asarray(cad_mesh.contains(centroids), dtype=bool)
        except Exception:
            mask_inside = np.ones(len(centroids), dtype=bool)
        active_origins = grid_coords[mask_inside]
        active_centroids = centroids[mask_inside]
    else:
        active_origins = grid_coords
        active_centroids = centroids

    for (x, y, z), c in zip(active_origins, active_centroids):
        corners = np.array([
            [x, y, z],
            [x + cell_size, y, z],
            [x + cell_size, y + cell_size, z],
            [x, y + cell_size, z],
            [x, y, z + cell_size],
            [x + cell_size, y, z + cell_size],
            [x + cell_size, y + cell_size, z + cell_size],
            [x, y + cell_size, z + cell_size],
        ], dtype=np.float64)
        all_cell_corners.append(corners)
        phi_c = float(interp_phi(c[0], c[1], c[2]))
        all_cell_densities.append(phi_c)


    if len(all_cell_corners) == 0:
        raise ValueError("No cells were generated inside the specified geometry bounds.")

    # 2. Build local topologies and weld shared nodes
    raw_nodes = []
    raw_struts = []
    strut_densities = []
    node_offset = 0

    for corners, phi_val in zip(all_cell_corners, all_cell_densities):
        c_nodes, c_struts = builder(corners)
        raw_nodes.append(c_nodes)
        raw_struts.append(c_struts + node_offset)
        strut_densities.extend([phi_val] * len(c_struts))
        node_offset += len(c_nodes)

    stacked_nodes = np.vstack(raw_nodes)
    stacked_struts = np.vstack(raw_struts)
    stacked_densities = np.array(strut_densities, dtype=np.float64)

    # Weld coincident nodes
    tree = cKDTree(stacked_nodes)
    _, cluster_ids = tree.query(stacked_nodes, distance_upper_bound=1e-4)

    unique_clusters, inverse_indices = np.unique(cluster_ids, return_inverse=True)
    unique_nodes = np.zeros((len(unique_clusters), 3), dtype=np.float64)
    for new_idx, old_idx in enumerate(unique_clusters):
        unique_nodes[new_idx] = stacked_nodes[old_idx]

    remapped_struts = inverse_indices[stacked_struts]

    # Remove self-loops and duplicate edges
    remapped_struts = np.sort(remapped_struts, axis=1)
    non_degenerate = remapped_struts[:, 0] != remapped_struts[:, 1]
    filtered_struts = remapped_struts[non_degenerate]
    filtered_densities = stacked_densities[non_degenerate]

    # Unique edges with density averaging
    edge_dict: dict[tuple[int, int], list[float]] = {}
    for (u, v), d in zip(filtered_struts, filtered_densities):
        key = (int(u), int(v))
        if key not in edge_dict:
            edge_dict[key] = []
        edge_dict[key].append(d)

    final_struts = []
    final_radii = []

    # Map density to radius: for octet, phi ~ 12*sqrt(2)*pi*(r/L)^2 => r/L = sqrt(phi / (12*sqrt(2)*pi))
    for (u, v), d_list in edge_dict.items():
        phi_avg = float(np.mean(d_list))
        # Radius normalized to cell_size
        r_rel = np.sqrt(max(phi_avg, 0.02) / (12.0 * np.sqrt(2.0) * np.pi))
        r_phys = r_rel * cell_size
        final_struts.append([u, v])
        final_radii.append(r_phys)

    final_struts_arr = np.array(final_struts, dtype=np.int64)
    final_radii_arr = np.array(final_radii, dtype=np.float64)

    # 3. Solidify with Clean Mitered Truss
    if clean_miter:
        tri_mesh = build_clean_miter_truss(
            unique_nodes,
            final_struts_arr,
            final_radii_arr,
            circular_segments=circular_segments,
        )
    else:
        # Fallback cylinder union
        parts = []
        for (u, v), r in zip(final_struts_arr, final_radii_arr):
            p1, p2 = unique_nodes[u], unique_nodes[v]
            cyl = trimesh.creation.cylinder(radius=r, segment=[p1, p2], sections=circular_segments)
            parts.append(cyl)
        tri_mesh = trimesh.util.concatenate(parts)

    if out_stl is not None:
        stl_path = Path(out_stl)
        stl_path.parent.mkdir(parents=True, exist_ok=True)
        tri_mesh.export(str(stl_path))

    return tri_mesh


