"""
Aristo Solver — FEA Entry Point for the Graphite Pipeline

This module is the main entry point for the Aristo FEA module. It:
  1. Generates a dedicated fine-resolution tetrahedral FEA mesh via gmsh
     (separate from the lattice scaffold — different resolutions are needed).
  2. Auto-detects boundary conditions (fixed support + load face) from mesh geometry.
  3. Assembles the global stiffness matrix K via stiffness_assembly.py.
  4. Builds the nodal force vector from surface pressure on loaded faces.
  5. Solves the reduced linear system K_free · u_free = F_free (SciPy or MKL PARDISO).
  6. Reconstructs the full displacement field and computes per-element Von Mises stress.
  7. Returns an AristoResult with normalized stress, raw metrics, and mesh data.

Phase 1 Scope (current):
  - Uniaxial load case only (single load direction vector, Phase 1).
  - BC detection via spatial heuristic (closest-face-to-extremum percentile).
  - Phase 1b (boundary_detection.py) will replace the heuristic with proper
    normal-vector-based face picker and support for user-provided face IDs.
  - Phase 2 will add multi-axis loads and the graphical face picker.

gmsh Usage:
  Follows the same "merge bypass" strategy as scaffold_module.py:
    export STL → gmsh.merge → classifySurfaces → createGeometry/createTopology
    → define volume → generate mesh → extract nodes/elements/surface.
  A separate gmsh call is made for each Aristo run (initialize/finalize pair).

Author: Graphite / Aristo Project
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
from typing import NamedTuple

import numpy as np
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.linear_solver import resolve_linear_solver_key, sparse_direct_solve
from graphite.aristo.stiffness_assembly import (
    apply_dirichlet_bcs,
    assemble_global_K,
    compute_element_stresses,
)
from graphite.aristo.boundary_detection import (
    detect_boundary_masks,
    detect_boundary_masks_flat_top_load,
    detect_boundary_masks_flat_top_vertex_plane,
    detect_boundary_masks_z_band,
    map_surfaces_to_fea,
)
from graphite.aristo.mesh_quality import (
    build_mesh_quality_report,
    build_quality_mask,
    element_volumes,
    mesh_has_giant_elements,
    stress_percentiles,
)
from graphite.aristo.aristo_log import aristo_log, aristo_stage
from graphite.aristo.stress_postprocess import (
    build_stress_fields,
    element_to_nodal_von_mises,
)


# ===========================================================================
# Public result type
# ===========================================================================


class AristoResult(NamedTuple):
    """
    Output of a single Aristo FEA pass.

    Fields
    ------
    displacement : np.ndarray
        Nodal displacement vectors (N, 3).
    von_mises : np.ndarray
        Per-element normalized stress (M,); denominator is nodal peak when
        ``stress_field_mode == 'nodal_averaged'``.
    von_mises_nodal_raw : np.ndarray
        Nodal von Mises stress (N,) in MPa (volume-weighted patch average).
    von_mises_nodal_norm : np.ndarray
        Nodal von Mises normalized by ``max_von_mises_raw`` (nodal peak).
    stress_field_mode : str
        ``nodal_averaged`` or ``element_raw``.
    fea_nodes : np.ndarray
        FEA mesh coordinates (N, 3).
    fea_elements : np.ndarray
        Tetrahedral connectivity (M, 4).
    fea_surface_faces : np.ndarray
        Surface face connectivity (K, 3).
    max_displacement : float
        Peak displacement (mm).
    max_von_mises_raw : float
        Peak stress (MPa): nodal max in ``nodal_averaged`` mode, else valid
        element max.
    max_von_mises_element_raw : float
        Peak element stress including poor tets (debug).
    hotspot_fraction : float
        Proportion of part in high-stress (valid elements only).
    fixed_nodes : np.ndarray
        Indices of constrained nodes.
    load_centroids : np.ndarray
        Centroids of surfaces receiving load.
    element_volumes : np.ndarray
        Per-tet signed volumes (mm³).
    quality_mask : np.ndarray
        True for tets included in stress post-processing.
    mesh_quality_report : dict
        Volume/aspect/stress diagnostics (JSON-serializable).
    """

    displacement: np.ndarray
    von_mises: np.ndarray
    fea_nodes: np.ndarray
    fea_elements: np.ndarray
    fea_surface_faces: np.ndarray
    max_displacement: float
    max_von_mises_raw: float
    hotspot_fraction: float
    fixed_nodes: np.ndarray
    load_nodes: np.ndarray
    load_centroids: np.ndarray
    element_volumes: np.ndarray
    quality_mask: np.ndarray
    mesh_quality_report: dict
    max_von_mises_element_raw: float
    von_mises_nodal_raw: np.ndarray
    von_mises_nodal_norm: np.ndarray
    stress_field_mode: str


def _detect_bc_masks(
    fea_nodes: np.ndarray,
    surface_faces: np.ndarray,
    d_hat: np.ndarray,
    config: AristoConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    band = (
        config.bc_z_band_fraction
        if config.bc_z_band_fraction is not None
        else 0.01
    )
    if config.bc_load_mode == "flat_top_vertex_plane":
        fixed_mask, load_mask, meta = detect_boundary_masks_flat_top_vertex_plane(
            fea_nodes,
            surface_faces,
            d_hat,
            band,
            min_vertices_at_plane=config.bc_top_min_vertices_at_plane,
            top_normal_z_min=config.bc_top_normal_z_min,
            bottom_normal_z_min=config.bc_bottom_normal_z_min,
        )
        return fixed_mask, load_mask, meta
    if config.bc_load_mode == "flat_top":
        fixed_mask, load_mask = detect_boundary_masks_flat_top_load(
            fea_nodes,
            surface_faces,
            d_hat,
            band,
            z_tolerance_mm=config.bc_top_z_tolerance_mm,
            top_normal_z_min=config.bc_top_normal_z_min,
        )
        return fixed_mask, load_mask, {}
    if config.bc_z_band_fraction is not None:
        fixed_mask, load_mask = detect_boundary_masks_z_band(
            fea_nodes,
            surface_faces,
            d_hat,
            config.bc_z_band_fraction,
        )
        return fixed_mask, load_mask, {}
    fixed_mask, load_mask = detect_boundary_masks(fea_nodes, surface_faces, d_hat)
    return fixed_mask, load_mask, {}


# ===========================================================================
# Public entry point
# ===========================================================================


def run_aristo(
    mesh: trimesh.Trimesh,
    config: AristoConfig,
    pass_label: str = "pre_lattice",
) -> AristoResult:
    """
    Run a complete Aristo FEA analysis on a solid trimesh boundary.

    Pipeline:
        mesh → generate_fea_mesh → auto_detect_bcs → assemble_K
             → build_F → apply_bcs → spsolve → von_mises → AristoResult

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Watertight boundary mesh of the solid part.  Must satisfy
        `mesh.is_watertight == True` for correct volume meshing.
    config : AristoConfig
        FEA configuration (material, mesh resolution, load direction, etc.).
    pass_label : str
        'pre_lattice' or 'post_lattice'. Stored in AristoResult for
        downstream identification.

    Returns
    -------
    AristoResult

    Raises
    ------
    TypeError
        If mesh is not a trimesh.Trimesh instance.
    ValueError
        If the mesh is not watertight, or gmsh fails to produce tets.
    RuntimeError
        If the linear system solve fails (e.g. singular K due to no BCs).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"mesh must be trimesh.Trimesh, got {type(mesh)}.")
    if not mesh.is_watertight:
        warnings.warn(
            "Input mesh is not watertight. gmsh may produce incomplete or "
            "incorrect tetrahedral meshes. Run graphite.repair first.",
            stacklevel=2,
        )

    aristo_log(
        f"run_aristo pass={pass_label!r} quality={config.fea_quality_mode!r} "
        f"gmsh_mode={config.fea_gmsh_mesh_mode or os.environ.get('ARISTO_GMSH_FEA_MESH_MODE', 'single_surface')!r} "
        f"h={config.fea_mesh_resolution} mm solver={resolve_linear_solver_key(config)!r}"
    )

    # ------------------------------------------------------------------
    # 1. Generate dedicated fine FEA mesh (separate from lattice scaffold)
    # ------------------------------------------------------------------
    with aristo_stage("fea_mesh_generation"):
        fea_nodes, fea_elements, surface_faces, remesh_attempts = _generate_fea_mesh(
            mesh, config=config
        )
    with aristo_stage("connected_component_filter"):
        fea_nodes, fea_elements, surface_faces = _retain_largest_connected_component(
            fea_nodes, fea_elements, surface_faces
        )
    N = fea_nodes.shape[0]
    n_dof = 3 * N
    aristo_log(
        f"mesh ready: nodes={N:,} tets={len(fea_elements):,} "
        f"surface_tris={len(surface_faces):,} remesh_attempts={remesh_attempts}"
    )

    # ------------------------------------------------------------------
    # 2. Auto-detect boundary conditions (Phase 1 heuristic)
    # ------------------------------------------------------------------
    with aristo_stage("boundary_detection"):
        load_dir = np.asarray(config.load_direction, dtype=np.float64)
        load_dir_norm = np.linalg.norm(load_dir)
        if load_dir_norm < 1e-12:
            raise ValueError("load_direction must be a non-zero vector.")
        d_hat = load_dir / load_dir_norm

        bc_load_meta: dict = {}
        if config.fixed_face_ids or config.load_face_ids:
            fea_surface_ids = map_surfaces_to_fea(mesh, fea_nodes, surface_faces)

            fixed_face_mask = np.zeros(surface_faces.shape[0], dtype=bool)
            for fid in config.fixed_face_ids:
                fixed_face_mask |= fea_surface_ids == fid

            load_face_mask = np.zeros(surface_faces.shape[0], dtype=bool)
            for fid in config.load_face_ids:
                load_face_mask |= fea_surface_ids == fid

            if not fixed_face_mask.any():
                warnings.warn(
                    "Explicit fixed_face_ids resulted in zero FEA faces. Using heuristic.",
                    stacklevel=2,
                )
                fixed_face_mask, _, _ = _detect_bc_masks(
                    fea_nodes, surface_faces, d_hat, config
                )

            if not load_face_mask.any():
                warnings.warn(
                    "Explicit load_face_ids resulted in zero FEA faces. Using heuristic.",
                    stacklevel=2,
                )
                _, load_face_mask, bc_load_meta = _detect_bc_masks(
                    fea_nodes, surface_faces, d_hat, config
                )
        else:
            fixed_face_mask, load_face_mask, bc_load_meta = _detect_bc_masks(
                fea_nodes, surface_faces, d_hat, config
            )

        if bc_load_meta:
            aristo_log(
                "BC vertex planes: "
                f"z_floor={bc_load_meta.get('z_floor_mm', 0):.9f} mm "
                f"fixed_faces={bc_load_meta.get('n_fixed_faces', 0):,} "
                f"excluded_upward_on_floor="
                f"{bc_load_meta.get('n_faces_on_floor_excluded_upward_normal', 0):,} | "
                f"z_cap={bc_load_meta.get('z_cap_mm', 0):.9f} mm "
                f"load_faces={bc_load_meta.get('n_load_faces', 0):,} "
                f"excluded_downward_on_cap="
                f"{bc_load_meta.get('n_faces_on_cap_excluded_downward_normal', 0):,}"
            )

        fixed_nodes = np.unique(surface_faces[fixed_face_mask].ravel())
        fixed_dofs = (fixed_nodes[:, np.newaxis] * 3 + np.arange(3)).ravel()
        fixed_dofs = fixed_dofs.astype(np.int64)

        n_fixed_faces = int(fixed_face_mask.sum())
        n_load_faces = int(load_face_mask.sum())
        if n_fixed_faces == 0:
            raise ValueError(
                "BC auto-detection found zero fixed faces. "
                "Try a different load_direction or provide fixed_face_ids explicitly."
            )
        if n_load_faces == 0:
            raise ValueError(
                "BC auto-detection found zero loaded faces. "
                "Try a different load_direction or provide load_face_ids explicitly."
            )
    aristo_log(
        f"BCs: fixed_faces={n_fixed_faces:,} load_faces={n_load_faces:,} "
        f"fixed_nodes={len(fixed_nodes):,}"
    )

    # ------------------------------------------------------------------
    # 3. Assemble global stiffness matrix K
    # ------------------------------------------------------------------
    with aristo_stage("stiffness_assembly"):
        K, _volumes = assemble_global_K(
            fea_nodes, fea_elements, config.youngs_modulus, config.poisson_ratio
        )
    aristo_log(f"K assembled: {K.shape[0]:,} DOF, nnz={K.nnz:,}")

    # ------------------------------------------------------------------
    # 4. Build global force vector F
    # ------------------------------------------------------------------
    with aristo_stage("force_vector"):
        F, load_centroids, load_nodes = _build_force_vector(
            n_dof,
            fea_nodes,
            surface_faces,
            load_face_mask,
            d_hat,
            config.load_magnitude,
            config.load_mode,
        )

    # ------------------------------------------------------------------
    # 5. Apply Dirichlet BCs (static condensation) and solve
    # ------------------------------------------------------------------
    with aristo_stage("linear_solve"):
        K_free, F_free, free_dofs = apply_dirichlet_bcs(K, F, fixed_dofs)
        aristo_log(
            f"solve system: free_dofs={len(free_dofs):,} "
            f"K_free nnz={K_free.nnz:,}"
        )
        t_solve0 = time.perf_counter()
        try:
            u_free, linear_solver_backend = sparse_direct_solve(K_free, F_free, config)
        except Exception as exc:
            raise RuntimeError(
                f"Sparse linear solve failed: {exc}. "
                "The stiffness matrix may be singular — check boundary conditions."
            ) from exc
        linear_solve_sec = time.perf_counter() - t_solve0
        aristo_log(
            f"linear solve backend={linear_solver_backend!r} time={linear_solve_sec:.2f}s"
        )

        if not np.all(np.isfinite(u_free)):
            raise RuntimeError(
                "Displacement solution contains NaN/inf. "
                "The system may be ill-conditioned or the mesh degenerate."
            )

    # ------------------------------------------------------------------
    # 6. Reconstruct full displacement vector
    # ------------------------------------------------------------------
    u_full = np.zeros(n_dof, dtype=np.float64)
    u_full[free_dofs] = u_free

    # ------------------------------------------------------------------
    # 7. Compute Von Mises stress per element
    # ------------------------------------------------------------------
    with aristo_stage("element_stresses"):
        vm_raw = compute_element_stresses(
            fea_nodes, fea_elements, u_full, config.youngs_modulus, config.poisson_ratio
        )

    # ------------------------------------------------------------------
    # 8. Mesh quality gate + stress post-processing (Phase 0/1)
    # ------------------------------------------------------------------
    with aristo_stage("stress_postprocess"):
        elem_vols = element_volumes(fea_nodes, fea_elements)
        quality_mask, _, _, _ = build_quality_mask(fea_nodes, fea_elements, config)
    n_poor = int(np.sum(~quality_mask))
    if n_poor:
        warnings.warn(
            f"[Aristo] {n_poor} element(s) ({100.0 * n_poor / max(len(quality_mask), 1):.2f}%) "
            "excluded from stress post-processing (volume/aspect quality gate).",
            stacklevel=2,
        )

    stress = build_stress_fields(
        vm_raw, fea_elements, elem_vols, quality_mask, config
    )

    disp_xyz = u_full.reshape(N, 3)
    max_disp = float(np.linalg.norm(disp_xyz, axis=1).max())

    mesh_report = build_mesh_quality_report(
        fea_nodes,
        fea_elements,
        config,
        vm_raw=vm_raw,
        volumes=elem_vols,
        quality_mask=quality_mask,
        remesh_attempts=remesh_attempts,
    )
    mesh_report["linear_solver"] = linear_solver_backend
    mesh_report["linear_solve_sec"] = float(linear_solve_sec)
    if bc_load_meta:
        mesh_report["bc_load_detection"] = bc_load_meta
    nodal_raw = stress["von_mises_nodal_raw"]
    if (
        config.stress_representation == "nodal_averaged"
        and isinstance(nodal_raw, np.ndarray)
        and nodal_raw.size
    ):
        _, nodal_support = element_to_nodal_von_mises(
            fea_elements, vm_raw, elem_vols, quality_mask
        )
        mesh_report["stress_nodal"] = stress_percentiles(nodal_raw, nodal_support)
        mesh_report["stress_field_mode"] = config.stress_representation

    return AristoResult(
        displacement=disp_xyz,
        von_mises=stress["von_mises"],
        fea_nodes=fea_nodes,
        fea_elements=fea_elements,
        fea_surface_faces=surface_faces,
        max_displacement=max_disp,
        max_von_mises_raw=float(stress["max_von_mises_raw"]),
        hotspot_fraction=float(stress["hotspot_fraction"]),
        fixed_nodes=fixed_nodes,
        load_nodes=load_nodes,
        load_centroids=load_centroids,
        element_volumes=elem_vols,
        quality_mask=quality_mask,
        mesh_quality_report=mesh_report,
        max_von_mises_element_raw=float(stress["max_von_mises_element_raw"]),
        von_mises_nodal_raw=stress["von_mises_nodal_raw"],
        von_mises_nodal_norm=stress["von_mises_nodal_norm"],
        stress_field_mode=str(stress["stress_field_mode"]),
    )


# ===========================================================================
# Private helpers
# ===========================================================================


def _sanitize_trimesh_for_gmsh(
    mesh: trimesh.Trimesh,
    *,
    allow_face_removal: bool,
) -> trimesh.Trimesh:
    """
    Light cleanup before gmsh discrete injection.

    ``remove_degenerate_faces`` changes triangle count; skip it when
    ``allow_face_removal`` is False so caller face indices (e.g. BC seeds) stay valid.
    """
    if mesh.is_watertight:
        return mesh.copy()

    m = mesh.copy()
    try:
        m.merge_vertices()
    except Exception:
        pass
    if allow_face_removal:
        try:
            m.remove_degenerate_faces()
        except Exception:
            pass
    try:
        m.remove_unreferenced_vertices()
    except Exception:
        pass
    return m


def _gmsh_geometry_tolerance_for_mesh(mesh: trimesh.Trimesh) -> float:
    """Match ``scaffold_module`` heuristic for ``Geometry.Tolerance``."""
    lo = np.min(mesh.vertices, axis=0)
    hi = np.max(mesh.vertices, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return max(1e-9, min(1e-2, diag * 1e-5))


def _aristo_fea_mesh_strategies() -> list[dict]:
    """
    Ordered fallbacks for ``gmsh.model.mesh.generate(3)``.

    "Singular matrix 3x3" often comes from the Delaunay mesher or Netgen
    optimization on thin / irregular discrete boundaries; HXT and disabling
    optimizers or the Distance/MathEval background field usually recovers.
    """
    return [
        {
            "name": "delaunay+bg+netgen",
            "algorithm_3d": 1,
            "use_bg_mesh": True,
            "optimize": 1,
            "optimize_netgen": 1,
            "smoothing": 20,
        },
        {
            "name": "hxt+bg+netgen",
            "algorithm_3d": 10,
            "use_bg_mesh": True,
            "optimize": 1,
            "optimize_netgen": 1,
            "smoothing": 20,
        },
        {
            "name": "hxt+bg+nonetgen",
            "algorithm_3d": 10,
            "use_bg_mesh": True,
            "optimize": 1,
            "optimize_netgen": 0,
            "smoothing": 10,
        },
        {
            "name": "hxt+uniform",
            "algorithm_3d": 10,
            "use_bg_mesh": False,
            "optimize": 1,
            "optimize_netgen": 0,
            "smoothing": 10,
        },
        {
            "name": "hxt+uniform+minimal_opt",
            "algorithm_3d": 10,
            "use_bg_mesh": False,
            "optimize": 0,
            "optimize_netgen": 0,
            "smoothing": 0,
        },
        {
            "name": "delaunay+uniform+minimal_opt",
            "algorithm_3d": 1,
            "use_bg_mesh": False,
            "optimize": 0,
            "optimize_netgen": 0,
            "smoothing": 0,
        },
    ]


def _gmsh_remove_mesh_fields() -> None:
    """
    Drop sizing fields before trying another ``generate(3)`` strategy.

    Do **not** call ``mesh.clear()`` here: it strips the injected discrete
    boundary and breaks subsequent surface recovery (self-edges / hung meshing).
    """
    import gmsh

    remove_all = getattr(gmsh.model.mesh.field, "removeAll", None)
    if remove_all is not None:
        try:
            remove_all()
        except Exception:
            pass

    # Fallback to individual tag removal
    for tag in [1, 2, 3, 4]:
        try:
            gmsh.model.mesh.field.remove(tag)
        except Exception:
            pass


def _gmsh_classify_angle_rad() -> float:
    """
    Dihedral threshold (radians) for ``gmsh.model.mesh.classifySurfaces``.

    Default 90° — porous lattices split into dozens of patches at 45°, which
    breaks the full Aristo volume-mesh pipeline. Override with
    ``ARISTO_GMSH_CLASSIFY_ANGLE_DEG`` (e.g. ``45`` for legacy CAD parts).
    """
    env = os.environ.get("ARISTO_GMSH_CLASSIFY_ANGLE_DEG", "").strip()
    if env:
        return float(np.deg2rad(float(env)))
    return math.pi / 2.0  # 90°


def _gmsh_fea_mesh_mode(config: AristoConfig | None = None) -> str:
    """
    Volume-mesh strategy for ``_generate_fea_mesh``.

    ``single_surface`` (default): one discrete shell, no ``classifySurfaces``.
    ``tpms`` / ``lattice``: classifySurfaces + explicit volume + curvature sizing.
    ``classify``: legacy CAD path with surface classification + fallback strategies.
    ``single_only``: single-surface only, no classify fallback.
    """
    if config is not None and config.fea_gmsh_mesh_mode:
        key = config.fea_gmsh_mesh_mode.strip().lower()
    else:
        key = os.environ.get(
            "ARISTO_GMSH_FEA_MESH_MODE", "single_surface"
        ).strip().lower()
    if key in ("tpms", "lattice"):
        return "tpms"
    if key in ("sdf", "sdf_mc", "voxel"):
        return "sdf"
    if key in ("single", "single_surface", "single-only", "single_only"):
        return "single_only" if key in ("single-only", "single_only") else "single_surface"
    if key == "classify":
        return "classify"
    return "single_surface"


def _extract_gmsh_tet_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull nodes, linear tets (type 4), and surface tris from the current gmsh model."""
    import gmsh

    tet_conn_raw: np.ndarray | None = None
    elem_types_3d, _, elem_conn_3d = gmsh.model.mesh.getElements(dim=3)
    for etype, econn in zip(elem_types_3d, elem_conn_3d):
        if int(etype) == 4:
            n_elems = len(econn) // 4
            tet_conn_raw = econn.astype(np.int64).reshape(n_elems, 4)
            break

    if tet_conn_raw is None:
        raise ValueError(
            "[Aristo] gmsh did not produce linear tetrahedral elements. "
            "This may happen if fea_mesh_resolution is too large relative "
            "to the part size, or if the mesh has geometry issues."
        )

    invalid = np.any(tet_conn_raw <= 0, axis=1)
    if invalid.any():
        n_bad = int(np.sum(invalid))
        warnings.warn(
            f"[Aristo] Dropping {n_bad} tet(s) with invalid gmsh node tags.",
            stacklevel=2,
        )
        tet_conn_raw = tet_conn_raw[~invalid]

    surf_parts: list[np.ndarray] = []
    elem_types_2d, _, elem_conn_2d = gmsh.model.mesh.getElements(dim=2)
    for etype, econn in zip(elem_types_2d, elem_conn_2d):
        if int(etype) == 2:
            n_elems = len(econn) // 3
            surf_parts.append(econn.astype(np.int64).reshape(n_elems, 3))

    surface_faces_raw = (
        np.vstack(surf_parts) if surf_parts else np.empty((0, 3), dtype=np.int64)
    )
    if surface_faces_raw.size:
        good_tri = np.all(surface_faces_raw > 0, axis=1)
        if not np.all(good_tri):
            surface_faces_raw = surface_faces_raw[good_tri]

    tet_conn = tet_conn_raw
    used_tags = np.unique(
        np.concatenate([tet_conn.ravel(), surface_faces_raw.ravel()])
        if surface_faces_raw.size
        else tet_conn.ravel()
    )
    used_tags = used_tags[used_tags > 0]

    node_tags_out, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags_out = node_tags_out.astype(np.int64)
    node_coords = node_coords.reshape(-1, 3)
    known = set(node_tags_out.tolist())
    missing = [int(t) for t in used_tags if int(t) not in known]

    if not missing:
        max_tag = int(node_tags_out.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        tag_to_idx[node_tags_out] = np.arange(len(node_tags_out), dtype=np.int64)
        tet_conn = tag_to_idx[tet_conn]
        if np.any(tet_conn < 0):
            raise RuntimeError(
                "[Aristo] Unmapped node tags in tet elements."
            )
        if surface_faces_raw.size:
            surface_faces = tag_to_idx[surface_faces_raw]
            if np.any(surface_faces < 0):
                raise RuntimeError(
                    "[Aristo] Unmapped node tags in surface elements."
                )
        else:
            surface_faces = surface_faces_raw
    else:
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes(-1, -1, True)
        all_tags = all_tags.astype(np.int64)
        node_coords = all_coords.reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(all_tags.tolist())}
        for tag in missing:
            if tag not in tag_to_idx:
                _, coord, _ = gmsh.model.mesh.getNode(tag)
                tag_to_idx[tag] = len(node_coords)
                node_coords = np.vstack(
                    [node_coords, np.asarray(coord, dtype=np.float64).reshape(1, 3)]
                )
        tet_conn = np.vectorize(tag_to_idx.__getitem__, otypes=[np.int64])(tet_conn)
        surface_faces = (
            np.vectorize(tag_to_idx.__getitem__, otypes=[np.int64])(surface_faces_raw)
            if surface_faces_raw.size
            else surface_faces_raw
        )
    node_coords, tet_conn = _repair_tet_connectivity(node_coords, tet_conn)
    return node_coords, tet_conn, surface_faces


def _tet_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    return element_volumes(nodes, elements)


def _retain_largest_connected_component(
    nodes: np.ndarray,
    elements: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop floating node islands left by invalid-tet removal on discrete TPMS meshes.

    Unconstrained debris components make ``K`` singular.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = nodes.shape[0]
    if elements.size == 0 or n == 0:
        return nodes, elements, surface_faces

    rows: list[int] = []
    cols: list[int] = []
    for tet in elements:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(tet[i]), int(tet[j])
                rows.extend((a, b))
                cols.extend((b, a))
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(adj, directed=False)
    if n_comp <= 1:
        return nodes, elements, surface_faces

    keep_label = int(np.argmax(np.bincount(labels)))
    node_mask = labels == keep_label
    elem_mask = node_mask[elements].all(axis=1)
    n_drop = int(np.sum(~elem_mask))
    n_orphan_nodes = int(np.sum(~node_mask))
    warnings.warn(
        f"[Aristo] Removed {n_drop} tet(s) and {n_orphan_nodes} node(s) in "
        f"{n_comp - 1} disconnected mesh component(s).",
        stacklevel=2,
    )

    new_idx = np.full(n, -1, dtype=np.int64)
    new_idx[node_mask] = np.arange(int(node_mask.sum()), dtype=np.int64)
    nodes_out = nodes[node_mask]
    elems_out = new_idx[elements[elem_mask]]
    if surface_faces.size:
        surf_mask = node_mask[surface_faces].all(axis=1)
        surf_out = new_idx[surface_faces[surf_mask]]
    else:
        surf_out = surface_faces
    return nodes_out, elems_out, surf_out


def _repair_tet_connectivity(
    nodes: np.ndarray,
    elements: np.ndarray,
    *,
    min_volume: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip inverted gmsh tets and drop near-degenerate elements.

    Porous STL volume meshes occasionally emit a tiny fraction of inverted
    tets; swapping two node indices fixes most without remeshing.
    """
    if elements.size == 0:
        return nodes, elements

    elems = np.asarray(elements, dtype=np.int64).copy()
    volumes = _tet_volumes(nodes, elems)
    bad = volumes <= 0.0
    if bad.any():
        for i, j in ((1, 2), (1, 3), (2, 3)):
            if not bad.any():
                break
            idx = np.flatnonzero(bad)
            swapped = elems[idx].copy()
            swapped[:, i], swapped[:, j] = swapped[:, j], swapped[:, i]
            elems[idx] = swapped
            volumes[idx] = _tet_volumes(nodes, elems[idx])
            bad = volumes <= 0.0

    good = volumes > min_volume
    n_drop = int(np.size(good) - np.count_nonzero(good))
    if n_drop:
        warnings.warn(
            f"[Aristo] Dropped {n_drop} near-degenerate tet(s) after orientation repair.",
            stacklevel=2,
        )
        elems = elems[good]

    n_inverted = int(np.sum(_tet_volumes(nodes, elems) <= 0.0)) if elems.size else 0
    if n_inverted:
        raise RuntimeError(
            f"[Aristo] {n_inverted} inverted tet(s) remain after orientation repair."
        )
    return nodes, elems


def _generate_fea_mesh_single_surface(
    mesh_work: trimesh.Trimesh,
    fea_mesh_resolution: float,
    *,
    silent: bool = True,
    optimize_volume: bool = False,
    h_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Volume mesh by treating the boundary as **one** discrete surface (no classifySurfaces).

    Preferred for porous lattices and other STL soups where CAD surface partitioning
    produces empty patches or PLC errors.
    """
    import gmsh

    gmsh_initialized = False
    h = float(fea_mesh_resolution) * float(h_scale)
    try:
        gmsh.initialize()
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0 if silent else 1)
        gmsh.model.add("aristo_fea_single_surface")

        tol_env = os.environ.get("ARISTO_GMSH_GEOMETRY_TOLERANCE", "").strip()
        try:
            if tol_env:
                gmsh.option.setNumber("Geometry.Tolerance", float(tol_env))
        except Exception:
            pass

        surf_tag = gmsh.model.addDiscreteEntity(2)
        verts = np.asarray(mesh_work.vertices, dtype=np.float64)
        faces = np.asarray(mesh_work.faces, dtype=np.int64)
        node_tags = np.arange(1, len(verts) + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, surf_tag, node_tags, verts.ravel(), np.empty(0))
        elem_tags = np.arange(1, len(faces) + 1, dtype=np.int64)
        gmsh.model.mesh.addElementsByType(surf_tag, 2, elem_tags, (faces + 1).ravel())
        if not mesh_work.is_watertight:
            gmsh.model.mesh.removeDuplicateNodes()

        sl = gmsh.model.geo.addSurfaceLoop([surf_tag])
        gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h * 2.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        algo_override = os.environ.get("ARISTO_GMSH_ALGORITHM3D", "").strip()
        gmsh.option.setNumber(
            "Mesh.Algorithm3D",
            int(algo_override) if algo_override else 1,
        )
        if optimize_volume:
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        else:
            gmsh.option.setNumber("Mesh.Optimize", 0)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.model.mesh.generate(3)
        if optimize_volume:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                try:
                    gmsh.model.mesh.optimize("Laplace")
                except Exception:
                    pass
        return _extract_gmsh_tet_mesh()
    finally:
        if gmsh_initialized:
            try:
                gmsh.finalize()
            except Exception:
                pass


def _mesh_work_surface_fn(config: AristoConfig):
    """Return the primary surface/volume mesh generator for ``config``."""
    if _gmsh_fea_mesh_mode(config) == "tpms":
        from graphite.aristo.gmsh_lattice_mesh import generate_lattice_fea_mesh_from_stl

        def _tpms(mesh_work, h, *, h_scale=1.0, optimize_volume=False):
            del optimize_volume
            from graphite.aristo.aristo_log import gmsh_terminal_enabled

            return generate_lattice_fea_mesh_from_stl(
                mesh_work,
                h,
                config=config,
                silent=not gmsh_terminal_enabled(),
                h_scale=h_scale,
            )

        return _tpms
    return _generate_fea_mesh_single_surface


def _generate_fea_mesh_thorough(
    mesh_work: trimesh.Trimesh,
    config: AristoConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Retry volume meshing with progressively tighter sizing until the poor-tet
    fraction falls below ``config.thorough_max_poor_fraction``.

    Netgen optimize is **off** by default — on discrete lattice STLs it can
    collapse the mesh into giant diagonal tets.
    """
    mesh_fn = _mesh_work_surface_fn(config)
    use_tpms = _gmsh_fea_mesh_mode(config) == "tpms"
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    best_score: tuple[float, float, float] | None = None
    attempts = 0

    for attempt in range(config.thorough_remesh_max_attempts):
        attempts = attempt + 1
        h_scale = 0.85 ** attempt
        h_eff = config.fea_mesh_resolution * h_scale
        with aristo_stage(
            f"thorough_remesh attempt {attempts}/{config.thorough_remesh_max_attempts} "
            f"(h_eff={h_eff:.4f} mm)"
        ):
            if use_tpms:
                nodes, elems, surf = mesh_fn(
                    mesh_work, config.fea_mesh_resolution, h_scale=h_scale
                )
            else:
                nodes, elems, surf = mesh_fn(
                    mesh_work,
                    config.fea_mesh_resolution,
                    silent=True,
                    optimize_volume=config.thorough_use_netgen_optimize,
                    h_scale=h_scale,
                )
        has_giants, max_edge, vol_ratio = mesh_has_giant_elements(nodes, elems, config)
        if has_giants:
            warnings.warn(
                f"[Aristo] thorough attempt {attempts}: rejected mesh with "
                f"max_edge={max_edge:.3f} mm or vol/median={vol_ratio:.1f} "
                "(giant elements — common with Netgen on lattice STLs).",
                stacklevel=2,
            )
            continue

        mask, _, _, _ = build_quality_mask(nodes, elems, config)
        poor_frac = float(np.mean(~mask))
        score = (poor_frac, max_edge, vol_ratio)
        aristo_log(
            f"thorough attempt {attempts}: nodes={len(nodes):,} tets={len(elems):,} "
            f"poor_frac={poor_frac:.4f} max_edge={max_edge:.4f} mm giants={has_giants}"
        )
        if best_score is None or score < best_score:
            best = (nodes, elems, surf)
            best_score = score
        if poor_frac <= config.thorough_max_poor_fraction:
            break

    if best is None:
        warnings.warn(
            "[Aristo] thorough remesh found no giant-free mesh; falling back to "
            "single quick-style pass.",
            stacklevel=2,
        )
        if use_tpms:
            nodes, elems, surf = mesh_fn(mesh_work, config.fea_mesh_resolution)
        else:
            nodes, elems, surf = mesh_fn(
                mesh_work,
                config.fea_mesh_resolution,
                silent=True,
                optimize_volume=False,
            )
        return nodes, elems, surf, attempts

    assert best_score is not None
    if best_score[0] > config.thorough_max_poor_fraction:
        warnings.warn(
            f"[Aristo] thorough remesh stopped after {attempts} attempt(s) with "
            f"poor_fraction={best_score[0]:.4f} (target "
            f"{config.thorough_max_poor_fraction:.4f}).",
            stacklevel=2,
        )

    nodes, elems, surf = best
    return nodes, elems, surf, attempts


def _generate_fea_mesh(
    mesh: trimesh.Trimesh,
    fea_mesh_resolution: float | None = None,
    force_uniform: bool = False,
    seed_face_indices: list[int] | None = None,
    *,
    config: AristoConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Generate a fine tetrahedral mesh for FEA using gmsh.

    Default strategy is **single-surface** (one discrete boundary shell, no
    ``classifySurfaces``). That path is reliable for porous lattices. On failure,
    or when ``ARISTO_GMSH_FEA_MESH_MODE=classify``, falls back to the legacy
    classifySurfaces + multi-strategy pipeline (better for some CAD parts with
    per-surface sizing seeds).

    Set ``ARISTO_GMSH_FEA_MESH_MODE=single_only`` to disable the classify fallback.

    Returns ``(nodes, elements, surface_faces, remesh_attempts)``.
    """
    if config is None:
        if fea_mesh_resolution is None:
            raise ValueError("Provide config or fea_mesh_resolution.")
        config = AristoConfig(fea_mesh_resolution=fea_mesh_resolution)
    elif fea_mesh_resolution is not None:
        raise ValueError("Pass fea_mesh_resolution or config, not both.")

    allow_fr = not bool(seed_face_indices)
    mesh_work = _sanitize_trimesh_for_gmsh(mesh, allow_face_removal=allow_fr)

    from graphite.aristo.stl_surface_clean import maybe_clean_stl_surface

    input_stl = getattr(config, "fea_input_stl_path", None)
    mesh_work = maybe_clean_stl_surface(
        mesh_work,
        config.fea_mesh_resolution,
        config=config,
        input_stl_path=input_stl,
    )

    if config.fea_quality_mode == "thorough":
        return _generate_fea_mesh_thorough(mesh_work, config)

    mode = _gmsh_fea_mesh_mode(config)
    h = config.fea_mesh_resolution

    if mode == "tpms":
        from graphite.aristo.gmsh_lattice_mesh import generate_lattice_fea_mesh_from_stl

        nodes, elems, surf = generate_lattice_fea_mesh_from_stl(
            mesh_work, h, config=config, silent=True
        )
        return nodes, elems, surf, 1

    if mode == "sdf":
        from graphite.aristo.sdf_lattice_mesh import mesh_solid_via_sdf

        nodes, elems, surf = mesh_solid_via_sdf(
            mesh_work, h, config=config, silent=True
        )
        return nodes, elems, surf, 1

    if mode != "classify":
        try:
            nodes, elems, surf = _generate_fea_mesh_single_surface(
                mesh_work, h, silent=True
            )
            return nodes, elems, surf, 1
        except Exception as exc:
            if mode == "single_only":
                raise
            warnings.warn(
                f"[Aristo] single-surface gmsh meshing failed ({exc}); "
                "retrying with classifySurfaces pipeline.",
                stacklevel=2,
            )

    try:
        nodes, elems, surf = _generate_fea_mesh_classify_surfaces(
            mesh,
            h,
            force_uniform=force_uniform,
            seed_face_indices=seed_face_indices,
            mesh_work=mesh_work,
        )
        return nodes, elems, surf, 1
    except Exception as exc_classify:
        warnings.warn(
            f"[Aristo] classifySurfaces meshing also failed ({exc_classify}); "
            "falling back to voxel SDF/Marching Cubes pipeline.",
            stacklevel=2,
        )
        from graphite.aristo.sdf_lattice_mesh import mesh_solid_via_sdf

        nodes, elems, surf = mesh_solid_via_sdf(
            mesh_work, h, config=config, silent=True
        )
        return nodes, elems, surf, 1


def _generate_fea_mesh_classify_surfaces(
    mesh: trimesh.Trimesh,
    fea_mesh_resolution: float,
    force_uniform: bool = False,
    seed_face_indices: list[int] | None = None,
    mesh_work: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy FEA volume mesh: ``classifySurfaces`` + optional background-field strategies.

    Used for CAD-like parts when single-surface meshing fails or when
    ``ARISTO_GMSH_FEA_MESH_MODE=classify``.
    """
    import gmsh  # imported inside function to isolate gmsh state

    gmsh_initialized = False

    import signal
    try:
        # Streamlit runs in a thread, preventing signal handling
        original_signal = signal.signal
        signal.signal = lambda *args, **kwargs: None
    except Exception:
        pass

    try:
        # ---- Initialize gmsh ----
        gmsh.initialize()
        gmsh_initialized = True
        
        try:
            signal.signal = original_signal
        except Exception:
            pass
        gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output for progress
        gmsh.model.add("aristo_fea")

        # Sanitize boundary: merging vertices helps gmsh; defer degenerate-face
        # removal when seed_face_indices must stay aligned with the input mesh.
        if mesh_work is None:
            allow_fr = not bool(seed_face_indices)
            mesh_work = _sanitize_trimesh_for_gmsh(mesh, allow_face_removal=allow_fr)

        tol_env = os.environ.get("ARISTO_GMSH_GEOMETRY_TOLERANCE", "").strip()
        try:
            if tol_env:
                gmsh.option.setNumber("Geometry.Tolerance", float(tol_env))
            else:
                gmsh.option.setNumber(
                    "Geometry.Tolerance",
                    _gmsh_geometry_tolerance_for_mesh(mesh_work),
                )
        except Exception:
            pass

        # ---- Inject surface mesh directly (avoids gmsh.merge STL issues) ----
        # Add a discrete surface entity and seed it with the trimesh geometry.
        surf_tag = gmsh.model.addDiscreteEntity(2)

        verts = np.asarray(mesh_work.vertices, dtype=np.float64)
        faces = np.asarray(mesh_work.faces, dtype=np.int64)
        n_verts = verts.shape[0]
        n_faces = faces.shape[0]

        # Node tags are 1-based in gmsh
        node_tags = np.arange(1, n_verts + 1, dtype=np.int64)
        coords_flat = verts.ravel()
        params_flat = np.empty(0, dtype=np.float64)  # no parametric coords
        gmsh.model.mesh.addNodes(2, surf_tag, node_tags, coords_flat, params_flat)

        # Element type 2 = 3-node linear triangle; tags are 1-based
        elem_tags = np.arange(1, n_faces + 1, dtype=np.int64)
        conn_flat = (faces + 1).ravel()  # convert to 1-based node tags
        gmsh.model.mesh.addElementsByType(surf_tag, 2, elem_tags, conn_flat)

        # ---- Recover CAD topology from surface mesh ----
        if not mesh_work.is_watertight:
            gmsh.model.mesh.removeDuplicateNodes()
            gmsh.model.mesh.removeDuplicateElements()
        classify_angle = _gmsh_classify_angle_rad()
        # forReparametrization=False avoids gmsh spinning on near-degenerate
        # surface partitions ("parametrized triangles are too small"). Opt in
        # with ARISTO_GMSH_CLASSIFY_FOR_REPARAM=1 if an old workflow needs it.
        for_reparam = os.environ.get(
            "ARISTO_GMSH_CLASSIFY_FOR_REPARAM", ""
        ).strip().lower() in ("1", "true", "yes")
        gmsh.model.mesh.classifySurfaces(
            classify_angle, True, for_reparam, math.pi
        )
        try:
            gmsh.model.mesh.createGeometry()
        except Exception as geo_err:
            warnings.warn(
                f"[Aristo] createGeometry() failed ({geo_err}); "
                "falling back to createTopology() (discrete BRep).",
                stacklevel=3,
            )
            try:
                gmsh.model.mesh.createTopology(True, True)
            except Exception as topo_err:
                raise RuntimeError(
                    f"[Aristo] Could not build CAD from mesh: "
                    f"createGeometry failed ({geo_err}), "
                    f"createTopology also failed ({topo_err}). "
                    "Ensure the mesh is watertight and manifold."
                ) from topo_err

        # ---- Define volume if not already created by createGeometry ----
        existing_vols = gmsh.model.getEntities(3)
        if not existing_vols:
            surfs = gmsh.model.getEntities(2)
            if not surfs:
                raise RuntimeError(
                    "[Aristo] No surfaces found after classifySurfaces. "
                    "The mesh may be empty or non-manifold."
                )
            sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()

        # ---- Seed surfaces for optional Distance-field refinement ----
        all_surfs = gmsh.model.getEntities(2)
        seed_tags: list[int] = []

        if seed_face_indices:
            selected_centroids = mesh_work.triangles_center[seed_face_indices]
            for dim, tag in all_surfs:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                for c in selected_centroids:
                    if (
                        bbox[0] - 1e-3 <= c[0] <= bbox[3] + 1e-3
                        and bbox[1] - 1e-3 <= c[1] <= bbox[4] + 1e-3
                        and bbox[2] - 1e-3 <= c[2] <= bbox[5] + 1e-3
                    ):
                        seed_tags.append(tag)
                        break

        if not seed_tags:
            seed_tags = [s[1] for s in all_surfs]
            warnings.warn(
                "[Aristo] No specific seed faces identified; falling back to full skin seeding.",
                stacklevel=3,
            )

        strategies = _aristo_fea_mesh_strategies()
        if force_uniform:
            strategies = [s for s in strategies if not s["use_bg_mesh"]]
            if not strategies:
                strategies = _aristo_fea_mesh_strategies()

        algo_override = os.environ.get("ARISTO_GMSH_ALGORITHM3D", "").strip()
        if algo_override:
            try:
                strategies[0]["algorithm_3d"] = int(algo_override)
            except ValueError:
                pass

        gen_err: Exception | None = None
        for attempt, strat in enumerate(strategies):
            if attempt > 0:
                _gmsh_remove_mesh_fields()

            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

            if strat["use_bg_mesh"]:
                gmsh.model.mesh.field.add("Distance", 1)
                gmsh.model.mesh.field.setNumbers(1, "SurfacesList", seed_tags)
                gmsh.model.mesh.field.setNumber(1, "Sampling", 100)
                h0 = float(fea_mesh_resolution)
                slope = 0.2
                h_max = float(fea_mesh_resolution) * 2.0
                math_expr = f"Min({h0} + {slope} * F1, {h_max})"
                gmsh.model.mesh.field.add("MathEval", 2)
                gmsh.model.mesh.field.setString(2, "F", math_expr)
                gmsh.model.mesh.field.setAsBackgroundMesh(2)
            else:
                h0 = float(fea_mesh_resolution)
                h_max = h0 * 2.0
                gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
                gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)

            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", strat["algorithm_3d"])
            gmsh.option.setNumber("Mesh.Optimize", strat["optimize"])
            gmsh.option.setNumber("Mesh.OptimizeNetgen", strat["optimize_netgen"])
            gmsh.option.setNumber("Mesh.Smoothing", strat["smoothing"])

            try:
                gmsh.model.mesh.generate(3)
                gen_err = None
                break
            except Exception as exc:
                gen_err = exc
                warnings.warn(
                    f"[Aristo] gmsh 3D mesh strategy {strat['name']!r} failed "
                    f"({exc}); retrying with a fallback.",
                    stacklevel=3,
                )

        if gen_err is not None:
            raise RuntimeError(
                "[Aristo] gmsh could not build a volume mesh after all fallback "
                f"strategies. Last error: {gen_err}"
            ) from gen_err

        return _extract_gmsh_tet_mesh()

    finally:
        if gmsh_initialized:
            try:
                gmsh.finalize()
            except Exception:
                pass


def _build_force_vector(
    n_dof: int,
    nodes: np.ndarray,
    surface_faces: np.ndarray,
    load_face_mask: np.ndarray,
    load_direction: np.ndarray,
    load_magnitude: float,
    load_mode: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build global nodal force vector from surface pressure on loaded faces.

    Includes optional normal-based filtering for tension/compression half-surfaces.

    Parameters
    ----------
    ...
    load_mode : str
        'full', 'tension_half', or 'compression_half'.

    Returns
    -------
    F : ndarray, shape (n_dof,)
    load_centroids : ndarray, shape (L, 3)
        Centroids of faces that actually received load.
    load_nodes : ndarray, shape (P,)
        Unique node indices belonging to the loaded faces.
    """
    F = np.zeros(n_dof, dtype=np.float64)

    loaded_faces_idx = np.where(load_face_mask)[0]
    if len(loaded_faces_idx) == 0:
        return F, np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    loaded_faces = surface_faces[loaded_faces_idx]
    verts = nodes[loaded_faces]                    # (Kl, 3, 3)

    # Compute triangle properties
    e1 = verts[:, 1] - verts[:, 0]
    e2 = verts[:, 2] - verts[:, 0]
    cross_prod = np.cross(e1, e2)
    areas = np.linalg.norm(cross_prod, axis=1) / 2.0
    centroids = np.mean(verts, axis=1)

    # Apply normal filtering for half-surface loads
    mask_to_apply = np.ones(len(loaded_faces), dtype=bool)
    act_cents = []
    if load_mode != "full":
        # Unit outward normal (assuming CCW faces from internal mesh)
        norms = cross_prod / (np.linalg.norm(cross_prod, axis=1)[:, np.newaxis] + 1e-12)
        # Pulling DOWN (-Z) on a hole: Interior normals point inward.
        # If pin pulls DOWN, bottom faces have normal with -Z component.
        # dot(norm, load_dir) > 0 means same direction.
        cos_theta = np.sum(norms * load_direction, axis=1)
        
        if load_mode == "tension_half":
            mask_to_apply = cos_theta > 0
        elif load_mode == "compression_half":
            mask_to_apply = cos_theta < 0
        
        loaded_faces = loaded_faces[mask_to_apply]
        areas = areas[mask_to_apply]
        for idx, val in enumerate(mask_to_apply):
            if val:
                act_cents.append(centroids[idx])
    else:
        act_cents = list(centroids)
        
    if len(loaded_faces) == 0:
        return F, np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

    # Force per node per face: pressure × area / 3 × direction
    f_per_node = (load_magnitude * areas / 3.0)[:, np.newaxis] * load_direction

    # Scatter to global F
    for local_i in range(3):
        node_inds = loaded_faces[:, local_i]
        np.add.at(F, node_inds * 3 + 0, f_per_node[:, 0])
        np.add.at(F, node_inds * 3 + 1, f_per_node[:, 1])
        np.add.at(F, node_inds * 3 + 2, f_per_node[:, 2])

    return F, np.array(act_cents) if act_cents else np.zeros((0, 3)), np.unique(loaded_faces)
