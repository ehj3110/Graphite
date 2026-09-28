"""
Aristo Adapt — Two-Scale Homogenized Continuum FEA & Stress-Adaptive Lattice Optimization.

This module integrates micro-scale homogenization, surrogate tensor modeling,
macro continuum FEA, and closed-loop Fully Stressed Design (FSD) into the
core public API of Aristo.

Unlike legacy heuristic stress-to-relative-density remapping (which meshed
full micro-lattices and exhausted workstation RAM), Aristo Adapt solves on a
coarse macro continuum mesh using the homogenized anisotropic elasticity tensor
C^H(phi), updating density fields in closed-loop in seconds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import trimesh

from graphite.aristo.aristo_log import aristo_log
from graphite.fea.aristo_bridge import (
    MacroMesh,
    TwoScaleFEAResult,
    apply_surface_traction,
    assemble_anisotropic_global_K,
    create_box_continuum_mesh,
    evaluate_surrogate_elasticity,
    export_two_scale_result_vtk,
    find_boundary_nodes_by_plane,
    generate_macro_continuum_mesh,
    map_grading_field_to_centroids,
    recover_element_and_nodal_stresses,
    run_two_scale_macro_fea,
)
from graphite.fea.homogenization import (
    EngineeringConstants,
    HomogenizationResult,
    RVEGridConfig,
    build_voxel_c3d8_stiffness,
    homogenize_octet_cell,
    homogenize_strut_cell,
    homogenize_tpms_cell,
    homogenize_voxel_rve,
)
from graphite.fea.stress_adaptation import (
    OptimizationIterationRecord,
    StressAdaptationConfig,
    TwoScaleOptimizationResult,
    apply_volume_bisection_scaling,
    build_neighborhood_filter,
    export_optimization_result_vtk,
    optimize_lattice_density_fsd,
    realize_optimized_strut_lattice,
    realize_optimized_tpms_lattice,
    render_optimization_summary_png,
)
from graphite.fea.surrogate import (
    MaterialTensorSurrogate,
    SurrogateCalibrationPoint,
    build_octet_homogenization_surrogate,
    build_strut_homogenization_surrogate,
    build_tpms_homogenization_surrogate,
)

# Public Aliases
AristoAdaptConfig = StressAdaptationConfig
AristoAdaptResult = TwoScaleOptimizationResult


def run_aristo_adaptive(
    part: trimesh.Trimesh | MacroMesh | tuple[tuple[float, float, float], tuple[float, float, float]],
    config: AristoAdaptConfig | None = None,
    *,
    lattice_type: Literal["octet", "gyroid", "diamond", "split_p", "primitive"] | str = "octet",
    surrogate: MaterialTensorSurrogate | None = None,
    cell_size_mm: float = 2.0,
    base_E_MPa: float = 2000.0,
    base_nu: float = 0.35,
    elem_type: Literal["tet4", "hex8"] = "tet4",
    subdivisions: tuple[int, int, int] | None = None,
    macro_resolution_mm: float | None = None,
    fixed_face: Literal["-x", "+x", "-y", "+y", "-z", "+z"] | None = "-x",
    load_face: Literal["-x", "+x", "-y", "+y", "-z", "+z"] | None = "+x",
    fixed_nodes: np.ndarray | None = None,
    applied_forces: dict[int, np.ndarray] | None = None,
    total_force_N: float | None = 1000.0,
    load_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    realize_lattice: bool = False,
    clean_miter: bool = True,
    output_stl: str | Path | None = None,
    output_vtu: str | Path | None = None,
    output_png: str | Path | None = None,
    logger: Callable[[str], None] | None = None,
) -> AristoAdaptResult:
    """
    Run closed-loop two-scale stress-adaptive lattice optimization in Aristo.

    Parameters
    ----------
    part : trimesh.Trimesh or MacroMesh or tuple of bounds
        The design envelope. Can be a watertight CAD Trimesh, an existing MacroMesh,
        or a ((xmin, ymin, zmin), (xmax, ymax, zmax)) bounding box.
    config : AristoAdaptConfig, optional
        Optimization hyperparameters (target stress, volume fraction, move limits).
        If None, default AristoAdaptConfig() is used.
    lattice_type : str
        Unit cell topology ('octet', 'gyroid', 'diamond', etc.). Default 'octet'.
    surrogate : MaterialTensorSurrogate, optional
        Pre-calibrated homogenization surrogate. If None, builds or fits one
        for the chosen lattice_type automatically.
    cell_size_mm : float
        Unit cell edge length in mm. Default 2.0 mm.
    base_E_MPa : float
        Base constituent solid Young's modulus in MPa. Default 2000.0 (resin).
    base_nu : float
        Base constituent Poisson's ratio. Default 0.35.
    elem_type : {'tet4', 'hex8'}
        Macro continuum element type. Default 'tet4'.
    subdivisions : tuple of (int, int, int), optional
        Grid subdivisions if generating box mesh from bounds.
    macro_resolution_mm : float, optional
        Characteristic element edge length in mm for CAD meshing.
    fixed_face : {'-x', '+x', '-y', '+y', '-z', '+z'}, optional
        Convenience face selector to clamp. Ignored if fixed_nodes is provided.
    load_face : {'-x', '+x', '-y', '+y', '-z', '+z'}, optional
        Convenience face selector for traction load. Ignored if applied_forces is provided.
    fixed_nodes : np.ndarray, optional
        Explicit array of clamped node indices.
    applied_forces : dict[int, np.ndarray], optional
        Explicit map of {node_idx: force_vector_3D} in N.
    total_force_N : float, optional
        Total resultant load in N distributed over load_face.
    load_direction : tuple of float
        Direction vector for applied load, default (1.0, 0.0, 0.0).
    realize_lattice : bool
        If True, synthesizes the physical 3D lattice mesh from optimal densities.
    clean_miter : bool
        If True and realizing struts, uses clean mitered truss joints (bisector cut).
    output_stl : str or Path, optional
        Optional file path to save the realized physical lattice STL.
    output_vtu : str or Path, optional
        Optional file path to save the macro FEA ParaView VTU result.
    output_png : str or Path, optional
        Optional file path to save the 3D optimization summary PNG.
    logger : Callable[[str], None], optional
        Logging callback. If None, uses aristo_log.

    Returns
    -------
    AristoAdaptResult
        Optimization result container containing optimal density field,
        iteration history, final FEA stress/displacement results, and mesh.
    """
    log_fn = logger or aristo_log
    log_fn(f"[Aristo Adapt] Initializing two-scale lattice optimization (type='{lattice_type}')...")

    # 1. Resolve Macro Continuum Mesh
    cad_mesh: trimesh.Trimesh | None = None
    if isinstance(part, MacroMesh):
        macro_mesh = part
    elif isinstance(part, trimesh.Trimesh):
        cad_mesh = part
        target_h = macro_resolution_mm or (max(cad_mesh.extents) / 20.0)
        macro_mesh = generate_macro_continuum_mesh(cad_mesh, target_edge_length=target_h, elem_type=elem_type)
    elif isinstance(part, (tuple, list)) and len(part) == 2:
        bounds = (tuple(part[0]), tuple(part[1]))
        if subdivisions is None:
            ext = np.array(bounds[1]) - np.array(bounds[0])
            subdivs = tuple(max(2, int(round(s / (cell_size_mm * 2.0)))) for s in ext)
        else:
            subdivs = subdivisions
        macro_mesh = create_box_continuum_mesh(bounds=bounds, subdivisions=subdivs, elem_type=elem_type)
    else:
        raise TypeError(f"Unsupported part type: {type(part)}")

    log_fn(
        f"[Aristo Adapt] Macro mesh ready: {macro_mesh.nodes.shape[0]} nodes, "
        f"{macro_mesh.elements.shape[0]} {macro_mesh.elem_type} elements."
    )

    # 2. Resolve Material Surrogate
    if surrogate is None:
        log_fn(f"[Aristo Adapt] Building homogenization surrogate for '{lattice_type}'...")
        rve_cfg = RVEGridConfig(
            resolution=32,
            cell_size=cell_size_mm,
            base_E=base_E_MPa,
            base_nu=base_nu,
            solver_backend="cg",
        )
        if lattice_type.lower() == "octet":
            surrogate = build_octet_homogenization_surrogate(
                solid_fractions=(0.08, 0.12, 0.18, 0.26, 0.36),
                rve_config=rve_cfg,
            )
        elif lattice_type.lower() in ("gyroid", "diamond", "split_p", "primitive"):
            surrogate = build_tpms_homogenization_surrogate(
                lattice_type=lattice_type.lower(),
                solid_fractions=(0.08, 0.15, 0.25, 0.35, 0.50),
                rve_config=rve_cfg,
            )
        else:
            from graphite.explicit.hex_topology_module import get_hex_topology_rule
            rule = get_hex_topology_rule(lattice_type.lower())
            unit_box = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=np.float64)
            r_nodes, r_struts = rule.builder(unit_box)
            surrogate = build_strut_homogenization_surrogate(
                nodes=r_nodes,
                struts=r_struts,
                radii=(0.05, 0.08, 0.12, 0.16, 0.20),
                rve_config=rve_cfg,
            )


    # 3. Resolve Boundary Conditions
    nodes = macro_mesh.nodes
    min_b = np.min(nodes, axis=0)
    max_b = np.max(nodes, axis=0)

    face_map = {
        "-x": (0, min_b[0], -1.0),
        "+x": (0, max_b[0], 1.0),
        "-y": (1, min_b[1], -1.0),
        "+y": (1, max_b[1], 1.0),
        "-z": (2, min_b[2], -1.0),
        "+z": (2, max_b[2], 1.0),
    }

    if fixed_nodes is None:
        if fixed_face is not None and fixed_face in face_map:
            axis, val, _ = face_map[fixed_face]
            tol = (max_b[axis] - min_b[axis]) * 0.05
            fixed_nodes = find_boundary_nodes_by_plane(macro_mesh, axis=axis, value=val, tol=tol)
        else:
            raise ValueError("No fixed boundary conditions specified (provide fixed_nodes or fixed_face).")

    N = macro_mesh.nodes.shape[0]
    forces_vec = np.zeros(3 * N, dtype=np.float64)

    if applied_forces is not None:
        if isinstance(applied_forces, np.ndarray):
            forces_vec = applied_forces.ravel()
        elif isinstance(applied_forces, dict):
            for n_idx, f_vec in applied_forces.items():
                forces_vec[int(n_idx) * 3 : int(n_idx) * 3 + 3] = np.asarray(f_vec, dtype=np.float64)
    else:
        if load_face is not None and load_face in face_map:
            axis, val, _ = face_map[load_face]
            tol = (max_b[axis] - min_b[axis]) * 0.05
            loaded_nodes = find_boundary_nodes_by_plane(macro_mesh, axis=axis, value=val, tol=tol)
            if len(loaded_nodes) > 0:
                ld = np.asarray(load_direction, dtype=np.float64)
                ld /= max(np.linalg.norm(ld), 1e-12)
                f_per_node = (total_force_N or 1000.0) / len(loaded_nodes) * ld
                for n_idx in loaded_nodes:
                    forces_vec[int(n_idx) * 3 : int(n_idx) * 3 + 3] = f_per_node
        else:
            raise ValueError("No load boundary conditions specified (provide applied_forces or load_face).")

    # 4. Closed-Loop FSD Optimization
    opt_config = config or AristoAdaptConfig()
    log_fn(
        f"[Aristo Adapt] Starting FSD optimization: target_stress={opt_config.target_stress} MPa, "
        f"target_vf={opt_config.target_volume_fraction}, move_limit={opt_config.move_limit}..."
    )

    result = optimize_lattice_density_fsd(
        mesh=macro_mesh,
        surrogate=surrogate,
        fixed_nodes=fixed_nodes,
        forces=forces_vec,
        config=opt_config,
    )


    # 5. Export VTU & PNG if requested
    if output_vtu is not None:
        export_optimization_result_vtk(result, output_vtu)
        log_fn(f"[Aristo Adapt] Exported optimization VTU: {output_vtu}")

    if output_png is not None:
        render_optimization_summary_png(result, out_path_3d=output_png)
        log_fn(f"[Aristo Adapt] Rendered optimization PNG: {output_png}")

    # 6. Physical Lattice Realization
    if realize_lattice or output_stl is not None:
        log_fn(f"[Aristo Adapt] Synthesizing physical 3D lattice (clean_miter={clean_miter})...")
        if lattice_type.lower() in ("gyroid", "diamond", "split_p", "primitive"):
            realized_mesh = realize_optimized_tpms_lattice(
                result=result,
                lattice_type=lattice_type.lower(),
                cell_size=cell_size_mm,
                out_stl=output_stl,
            )
        else:
            realized_mesh = realize_optimized_strut_lattice(
                result=result,
                rule_name=lattice_type.lower(),
                cell_size=cell_size_mm,
                out_stl=output_stl,
                clean_miter=clean_miter,
                cad_mesh=cad_mesh,
            )
        if output_stl is not None:
            log_fn(f"[Aristo Adapt] Saved realized lattice STL: {output_stl}")

    log_fn(f"[Aristo Adapt] Two-scale optimization complete. Converged: {result.converged}.")
    return result


__all__ = [
    "AristoAdaptConfig",
    "AristoAdaptResult",
    "EngineeringConstants",
    "HomogenizationResult",
    "MacroMesh",
    "MaterialTensorSurrogate",
    "OptimizationIterationRecord",
    "RVEGridConfig",
    "StressAdaptationConfig",
    "SurrogateCalibrationPoint",
    "TwoScaleFEAResult",
    "TwoScaleOptimizationResult",
    "apply_surface_traction",
    "apply_volume_bisection_scaling",
    "assemble_anisotropic_global_K",
    "build_neighborhood_filter",
    "build_octet_homogenization_surrogate",
    "build_strut_homogenization_surrogate",
    "build_tpms_homogenization_surrogate",
    "build_voxel_c3d8_stiffness",
    "create_box_continuum_mesh",
    "evaluate_surrogate_elasticity",
    "export_optimization_result_vtk",
    "export_two_scale_result_vtk",
    "find_boundary_nodes_by_plane",
    "generate_macro_continuum_mesh",
    "homogenize_octet_cell",
    "homogenize_strut_cell",
    "homogenize_tpms_cell",
    "homogenize_voxel_rve",
    "map_grading_field_to_centroids",
    "optimize_lattice_density_fsd",
    "realize_optimized_strut_lattice",
    "realize_optimized_tpms_lattice",
    "recover_element_and_nodal_stresses",
    "render_optimization_summary_png",
    "run_aristo_adaptive",
    "run_two_scale_macro_fea",
]
