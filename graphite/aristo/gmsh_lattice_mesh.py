"""
GMSH volume meshing for TPMS / porous lattice solids (STL or STEP).

Addresses common lattice mesh failures:
  1. STL → classifySurfaces + createGeometry before 3D mesh
  2. Explicit surface loops / volume entities (no implicit auto-fill)
  3. Curvature-based sizing for thin walls
  4. STEP → OpenCASCADE import + healShapes

Use via ``ARISTO_GMSH_FEA_MESH_MODE=tpms`` or ``generate_lattice_fea_mesh()``.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import trimesh

from graphite.aristo.aristo_config import AristoConfig
from graphite.aristo.aristo_log import aristo_log, aristo_stage, gmsh_terminal_enabled

# Fixed cap-intersection merge distance (mm); do not scale with target h.
_BOUNDARY_MERGE_TOL_MM = 1e-4


def _classify_angle_rad() -> float:
    env = os.environ.get("ARISTO_GMSH_CLASSIFY_ANGLE_DEG", "").strip()
    if env:
        return float(np.deg2rad(float(env)))
    return math.pi / 2.0  # 90° — fewer patches than 45° on lattice soup


def _geometry_tolerance(mesh: trimesh.Trimesh) -> float:
    lo = np.min(mesh.vertices, axis=0)
    hi = np.max(mesh.vertices, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return max(1e-9, min(1e-2, diag * 1e-5))


def _estimate_wall_pitch(mesh: trimesh.Trimesh) -> float:
    """Rough characteristic wall spacing from bbox / face density."""
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    # Heuristic: more faces per volume → thinner features
    density = len(mesh.faces) / max(mesh.volume, 1e-9)
    if density > 1e5:
        return max(diag / 200.0, 0.02)
    return max(diag / 100.0, 0.05)


def _adaptive_curvature_params(
    config: AristoConfig | None,
) -> tuple[float, float, int]:
    """Resolve (char_length_min, char_length_max, min_elements_per_two_pi) in mm."""
    cfg = config or AristoConfig()
    min_env = os.environ.get("ARISTO_GMSH_CHAR_LENGTH_MIN", "").strip()
    max_env = os.environ.get("ARISTO_GMSH_CHAR_LENGTH_MAX", "").strip()
    pi_env = os.environ.get("ARISTO_GMSH_MIN_ELEMENTS_PER_TWO_PI", "").strip()
    if not pi_env:
        pi_env = os.environ.get("ARISTO_GMSH_ELEMENTS_PER_TWO_PI", "").strip()
    h_min = float(min_env) if min_env else float(cfg.fea_gmsh_char_length_min)
    h_max = float(max_env) if max_env else float(cfg.fea_gmsh_char_length_max)
    n_pi = int(pi_env) if pi_env else int(cfg.fea_gmsh_min_elements_per_two_pi)
    return h_min, h_max, n_pi


def _use_adaptive_curvature_sizing(config: AristoConfig | None) -> bool:
    """Adaptive CharacteristicLength* sizing (default) vs legacy h-scaled MeshSize*."""
    legacy = os.environ.get("ARISTO_GMSH_LEGACY_CURVATURE_SIZING", "").strip().lower()
    if legacy in ("1", "true", "yes"):
        return False
    if config is None:
        return True
    return bool(config.fea_gmsh_adaptive_curvature)


def _disable_uniform_mesh_size_constraints() -> None:
    """Clear fixed ``Mesh.MeshSize*`` fields so curvature drives element size."""
    import gmsh

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)


def _apply_adaptive_curvature_sizing_options(
    config: AristoConfig | None = None,
    *,
    h_scale: float = 1.0,
) -> None:
    """Curvature-adaptive lattice sizing via Gmsh characteristic length."""
    import gmsh

    h_min, h_max, n_pi = _adaptive_curvature_params(config)
    scale = float(h_scale)
    _disable_uniform_mesh_size_constraints()
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min * scale)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max * scale)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", float(n_pi))
    aristo_log(
        "adaptive sizing: CharacteristicLengthFromCurvature=1 "
        f"cl_min={h_min * scale:.4f} cl_max={h_max * scale:.4f} elem/2pi={n_pi}"
    )


def _apply_curvature_sizing_options(
    h_target: float,
    *,
    elements_per_two_pi: int,
    wall_pitch: float,
) -> None:
    """Legacy h-scaled sizing (``ARISTO_GMSH_LEGACY_CURVATURE_SIZING=1``)."""
    import gmsh

    h_min = min(h_target * 0.35, wall_pitch / 3.0)
    h_max = h_target * 2.0
    _disable_uniform_mesh_size_constraints()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", float(elements_per_two_pi))
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_min)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)


def _classify_only_meshing(config: AristoConfig | None) -> bool:
    """When True, never fall back to single-surface (no classify) meshing."""
    if config is not None and config.fea_gmsh_classify_only:
        return True
    env = os.environ.get("ARISTO_GMSH_CLASSIFY_ONLY", "").strip().lower()
    return env in ("1", "true", "yes")


def _apply_boundary_merge_options() -> None:
    """
    Fixed geometric / Delaunay PLC tolerance at lattice–cap boolean intersections.

    Merges only microscopic knife-edge nodes without collapsing valid skin geometry.
    """
    import gmsh

    tol = _BOUNDARY_MERGE_TOL_MM
    gmsh.option.setNumber("Geometry.Tolerance", tol)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", tol)


def _cap_surface_dim_tags() -> list[tuple[int, int]]:
    """
    Classified TPMS slabs: one large lattice skin + small planar cap patches.

    Knife-edge nodes live on caps; merging on the full skin creates overlapping facets.
    """
    import gmsh

    surfaces = [(dim, tag) for dim, tag in gmsh.model.getEntities(2) if dim == 2]
    if len(surfaces) <= 1:
        return surfaces

    def _bbox_volume(dim: int, tag: int) -> float:
        bb = gmsh.model.getBoundingBox(dim, tag)
        return float((bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2]))

    skin_tag = max(surfaces, key=lambda dt: _bbox_volume(dt[0], dt[1]))[1]
    return [(2, tag) for dim, tag in surfaces if tag != skin_tag]


def _cleanup_boundary_nodes_before_volume_mesh() -> None:
    """Collapse knife-edge cap nodes within ``_BOUNDARY_MERGE_TOL_MM``."""
    import gmsh

    tol = _BOUNDARY_MERGE_TOL_MM
    gmsh.option.setNumber("Geometry.Tolerance", tol)
    # gmsh Python API: merge distance follows Geometry.Tolerance (no tol kwarg here).
    targets = _cap_surface_dim_tags()
    if targets:
        for dim_tag in targets:
            gmsh.model.mesh.removeDuplicateNodes([dim_tag])
            gmsh.model.mesh.removeDuplicateElements([dim_tag])
    else:
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.removeDuplicateElements()


def _apply_uniform_sizing_options(h_target: float) -> None:
    import gmsh

    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", h_target * 0.5)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h_target * 2.0)


def _apply_strict_uniform_sizing_options(h: float) -> None:
    """Single global characteristic length (no curvature / CharacteristicLength*)."""
    import gmsh

    _disable_uniform_mesh_size_constraints()
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", float(h))
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(h))


def _apply_aggressive_optimization_options() -> None:
    """Laplace-only post-generation smoothing (Netgen unsafe on discrete STL lattices)."""
    import gmsh

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.3)


def _linear_mesh_health_before_order_elevation() -> tuple[int, int]:
    """
    After ``generate(3)``, before ``setOrder(2)``.

    Returns ``(n_illegal_gmsh, n_inverted_by_volume)`` for linear (type 4) tets.
    """
    import gmsh

    from graphite.aristo.mesh_quality import element_volumes

    n_illegal = 0
    n_inverted = 0

    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=3)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags = node_tags.astype(np.int64)
    coords = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags.tolist())}

    for etype, etags, enodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
        if int(etype) != 4:
            continue
        etags_arr = np.asarray(etags, dtype=np.int64)
        if etags_arr.size == 0:
            continue
        try:
            qualities = gmsh.model.mesh.getElementQualities(etags_arr.tolist(), "minSICN")
            n_illegal += int(np.sum(np.asarray(qualities, dtype=np.float64) <= 0.0))
        except Exception:
            pass

        conn = np.asarray(enodes, dtype=np.int64).reshape(len(etags_arr), 4)
        idx = np.vectorize(tag_to_idx.__getitem__, otypes=[np.int64])(conn)
        vols = element_volumes(coords, idx)
        n_inverted += int(np.sum(vols <= 0.0))

    return n_illegal, n_inverted


def _count_open_boundary_edges() -> int:
    """Surface edges adjacent to only one triangle (non-manifold / open shell)."""
    import gmsh

    try:
        edges = gmsh.model.getBoundary(
            gmsh.model.getEntities(2), combined=False, oriented=False, recursive=False
        )
    except Exception:
        return -1
    # getBoundary on (2,s) returns (1,edge) with orientation; duplicates = interior
    edge_tags: list[int] = [tag for dim, tag in edges if dim == 1]
    if not edge_tags:
        return 0
    unique, counts = np.unique(np.asarray(edge_tags), return_counts=True)
    return int(np.sum(counts == 1))


def _largest_volume_tag() -> tuple[int, int] | None:
    import gmsh

    vols = gmsh.model.getEntities(3)
    if not vols:
        return None
    best: tuple[int, int] | None = None
    best_vol = -1.0
    for dim, tag in vols:
        bb = gmsh.model.getBoundingBox(dim, tag)
        size = float((bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2]))
        if size > best_vol:
            best_vol = size
            best = (dim, tag)
    return best


def _define_explicit_volume_from_surfaces() -> None:
    """
    Build one ``SurfaceLoop`` + ``Volume`` from classified surface patches.

    Skips if ``createGeometry`` already produced volume entities.
    """
    import gmsh

    if gmsh.model.getEntities(3):
        return

    surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
    if not surfaces:
        raise RuntimeError(
            "[Aristo/TPMS] No surface entities after classifySurfaces."
        )

    # Single exterior solid: one loop over all oriented exterior patches
    sl = gmsh.model.geo.addSurfaceLoop(surfaces)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()


def _inject_discrete_surface(mesh_work: trimesh.Trimesh) -> int:
    """Seed one discrete surface entity from trimesh (1-based tags)."""
    import gmsh

    surf_tag = gmsh.model.addDiscreteEntity(2)
    verts = np.asarray(mesh_work.vertices, dtype=np.float64)
    faces = np.asarray(mesh_work.faces, dtype=np.int64)
    node_tags = np.arange(1, len(verts) + 1, dtype=np.int64)
    gmsh.model.mesh.addNodes(2, surf_tag, node_tags, verts.ravel(), np.empty(0))
    elem_tags = np.arange(1, len(faces) + 1, dtype=np.int64)
    gmsh.model.mesh.addElementsByType(surf_tag, 2, elem_tags, (faces + 1).ravel())
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.removeDuplicateElements()
    return surf_tag


def _classify_and_create_geometry(*, prefer_topology: bool = False) -> None:
    """
    Partition the discrete STL into analytic patches, then build a BRep.

    For dense lattice soups, ``createGeometry()`` re-meshes 2D patches and can
    collapse the skin to a handful of nodes (PLC failure). Default for TPMS is
    ``createTopology`` which keeps the imported triangles. Set
    ``ARISTO_GMSH_CLASSIFY_FOR_REPARAM=1`` to opt into full re-parametrization.
    """
    import gmsh

    angle = _classify_angle_rad()
    for_reparam = os.environ.get(
        "ARISTO_GMSH_CLASSIFY_FOR_REPARAM", ""
    ).strip().lower() in ("1", "true", "yes")
    gmsh.model.mesh.classifySurfaces(angle, True, for_reparam, math.pi)

    if prefer_topology and not for_reparam:
        gmsh.model.mesh.createTopology(True, True)
        return

    try:
        gmsh.model.mesh.createGeometry()
    except Exception as geo_err:
        warnings.warn(
            f"[Aristo/TPMS] createGeometry failed ({geo_err}); using createTopology.",
            stacklevel=2,
        )
        gmsh.model.mesh.createTopology(True, True)


def _generate_single_surface_fallback(
    mesh_work: trimesh.Trimesh,
    h: float,
    use_curvature: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One discrete shell + explicit volume — reliable for dense lattice STLs when
    classify/createGeometry collapses the skin mesh.
    """
    import gmsh

    gmsh.option.setNumber(
        "Geometry.Tolerance",
        _geometry_tolerance(mesh_work),
    )
    surf_tag = _inject_discrete_surface(mesh_work)
    sl = gmsh.model.geo.addSurfaceLoop([surf_tag])
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    wall_pitch = _estimate_wall_pitch(mesh_work)
    if use_curvature:
        # Curvature needs parametric patches; on discrete shells use pitch-based min h.
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(h * 0.35, wall_pitch / 4.0))
        gmsh.option.setNumber("Mesh.MeshSizeMax", h * 2.0)
    else:
        _apply_uniform_sizing_options(h)

    _set_meshing_algorithm_options()
    _apply_boundary_merge_options()
    _cleanup_boundary_nodes_before_volume_mesh()
    gmsh.model.mesh.generate(3)

    from graphite.aristo.aristo_solver import _extract_gmsh_tet_mesh

    return _extract_gmsh_tet_mesh()


def _set_meshing_algorithm_options(*, aggressive_optimize: bool = False) -> None:
    import gmsh

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 2D
    algo_env = os.environ.get("ARISTO_GMSH_ALGORITHM3D", "").strip()
    algo_3d = int(algo_env) if algo_env else 1  # Delaunay — HXT crashes on V4 classify
    gmsh.option.setNumber("Mesh.Algorithm3D", algo_3d)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    if aggressive_optimize:
        _apply_aggressive_optimization_options()


def _extract_gmsh_volume_mesh(mesh_order: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull nodes and volume elements from the current gmsh model.

    ``mesh_order`` 1 → linear tets (type 4); 2 → quadratic tets (type 11).
    """
    import gmsh

    target_type = 11 if mesh_order >= 2 else 4
    nodes_per_cell = 10 if mesh_order >= 2 else 4

    vol_conn_raw: np.ndarray | None = None
    elem_types_3d, _, elem_conn_3d = gmsh.model.mesh.getElements(dim=3)
    for etype, econn in zip(elem_types_3d, elem_conn_3d):
        if int(etype) == target_type:
            n_elems = len(econn) // nodes_per_cell
            vol_conn_raw = econn.astype(np.int64).reshape(n_elems, nodes_per_cell)
            break

    if vol_conn_raw is None:
        raise ValueError(
            f"[Aristo/TPMS] gmsh did not produce order-{mesh_order} tetrahedra "
            f"(expected element type {target_type})."
        )

    surf_parts: list[np.ndarray] = []
    surf_type = 9 if mesh_order >= 2 else 2
    surf_nodes = 6 if mesh_order >= 2 else 3
    elem_types_2d, _, elem_conn_2d = gmsh.model.mesh.getElements(dim=2)
    for etype, econn in zip(elem_types_2d, elem_conn_2d):
        if int(etype) == surf_type:
            n_elems = len(econn) // surf_nodes
            surf_parts.append(econn.astype(np.int64).reshape(n_elems, surf_nodes))
        elif mesh_order < 2 and int(etype) == 2:
            n_elems = len(econn) // 3
            surf_parts.append(econn.astype(np.int64).reshape(n_elems, 3))

    surface_faces_raw = (
        np.vstack(surf_parts) if surf_parts else np.empty((0, 3), dtype=np.int64)
    )

    used_tags = np.unique(
        np.concatenate([vol_conn_raw.ravel(), surface_faces_raw.ravel()])
        if surface_faces_raw.size
        else vol_conn_raw.ravel()
    )
    used_tags = used_tags[used_tags > 0]

    node_tags_out, node_coords, _ = gmsh.model.mesh.getNodes()
    node_tags_out = node_tags_out.astype(np.int64)
    node_coords = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags_out.tolist())}

    for tag in used_tags:
        if int(tag) not in tag_to_idx:
            _, coord, _ = gmsh.model.mesh.getNode(int(tag))
            tag_to_idx[int(tag)] = len(node_coords)
            node_coords = np.vstack(
                [node_coords, np.asarray(coord, dtype=np.float64).reshape(1, 3)]
            )

    vol_conn = np.vectorize(tag_to_idx.__getitem__, otypes=[np.int64])(vol_conn_raw)
    if surface_faces_raw.size:
        if surface_faces_raw.shape[1] > 3:
            surface_faces = surface_faces_raw[:, :3]
            surface_faces = np.vectorize(
                tag_to_idx.__getitem__, otypes=[np.int64]
            )(surface_faces)
        else:
            surface_faces = np.vectorize(
                tag_to_idx.__getitem__, otypes=[np.int64]
            )(surface_faces_raw)
    else:
        surface_faces = surface_faces_raw

    if mesh_order < 2:
        from graphite.aristo.aristo_solver import _repair_tet_connectivity

        node_coords, vol_conn = _repair_tet_connectivity(node_coords, vol_conn)

    return node_coords, vol_conn, surface_faces


def generate_lattice_fea_mesh_from_stl(
    mesh_work: trimesh.Trimesh,
    fea_mesh_resolution: float,
    *,
    config: AristoConfig | None = None,
    silent: bool = True,
    use_curvature: bool = True,
    h_scale: float = 1.0,
    strict_uniform: bool = False,
    mesh_order: int = 1,
    mesh_meta: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TPMS/lattice volume mesh from a watertight STL (trimesh).

    Pipeline: discrete surface → classifySurfaces → createTopology → explicit
    volume → 3D tet mesh (optional strict uniform h, optional quadratic order).
    """
    import gmsh

    if not mesh_work.is_watertight:
        warnings.warn(
            "[Aristo/TPMS] Input STL is not watertight; volume mesh may leak.",
            stacklevel=2,
        )

    h = float(fea_mesh_resolution) * float(h_scale)
    cfg = config or AristoConfig(fea_mesh_resolution=h)
    elements_per_two_pi = int(
        os.environ.get("ARISTO_GMSH_ELEMENTS_PER_TWO_PI", "32")
    )
    wall_pitch = _estimate_wall_pitch(mesh_work)
    adaptive = (
        not strict_uniform
        and use_curvature
        and _use_adaptive_curvature_sizing(cfg)
    )

    gmsh_initialized = False
    show_gmsh = (not silent) or gmsh_terminal_enabled()
    if strict_uniform:
        aristo_log(
            f"tpms mesh strict uniform h={h:.4f} mm order={mesh_order} "
            f"classify-only={_classify_only_meshing(cfg)} gmsh_terminal={show_gmsh}"
        )
    elif adaptive:
        h_min, h_max, n_pi = _adaptive_curvature_params(cfg)
        aristo_log(
            f"tpms mesh adaptive curvature classify-only={_classify_only_meshing(cfg)} "
            f"cl_min={h_min * h_scale:.4f} cl_max={h_max * h_scale:.4f} "
            f"elem/2pi={n_pi} gmsh_terminal={show_gmsh}"
        )
    else:
        aristo_log(
            f"tpms mesh h={h:.4f} mm curvature={use_curvature} "
            f"gmsh_terminal={show_gmsh}"
        )
    try:
        with aristo_stage("gmsh_initialize"):
            gmsh.initialize()
            gmsh_initialized = True
            gmsh.option.setNumber("General.Terminal", 1 if show_gmsh else 0)
            gmsh.model.add("aristo_tpms_stl")
            gmsh.option.setNumber(
                "Geometry.Tolerance",
                _geometry_tolerance(mesh_work),
            )

        with aristo_stage("inject_discrete_surface"):
            _inject_discrete_surface(mesh_work)
        with aristo_stage("classify_surfaces_and_topology"):
            _classify_and_create_geometry(prefer_topology=True)
        with aristo_stage("define_volume"):
            _define_explicit_volume_from_surfaces()

        n_open = _count_open_boundary_edges()
        if n_open > 0:
            warnings.warn(
                f"[Aristo/TPMS] {n_open} open boundary edge(s) after classification.",
                stacklevel=2,
            )

        with aristo_stage("mesh_sizing_options"):
            if strict_uniform:
                _apply_strict_uniform_sizing_options(h)
                _set_meshing_algorithm_options(aggressive_optimize=True)
            elif use_curvature:
                if adaptive:
                    _apply_adaptive_curvature_sizing_options(cfg, h_scale=h_scale)
                else:
                    _apply_curvature_sizing_options(
                        h,
                        elements_per_two_pi=elements_per_two_pi,
                        wall_pitch=wall_pitch,
                    )
                _set_meshing_algorithm_options()
            else:
                _apply_uniform_sizing_options(h)
                _set_meshing_algorithm_options()

        vol = _largest_volume_tag()
        if vol is not None:
            _dim, vtag = vol
            try:
                pg = gmsh.model.addPhysicalGroup(3, [vtag])
                gmsh.model.setPhysicalName(3, pg, "solid")
            except Exception:
                pass

        with aristo_stage("cap_boundary_merge_and_cleanup"):
            _apply_boundary_merge_options()
            _cleanup_boundary_nodes_before_volume_mesh()

        effective_order = int(mesh_order)
        order_elevation_aborted = False
        illegal_linear_tets = 0
        inverted_linear_tets = 0
        try:
            with aristo_stage("gmsh_generate_3d (Delaunay + Laplace)"):
                gmsh.model.mesh.generate(3)
            if effective_order >= 2:
                illegal_linear_tets, inverted_linear_tets = (
                    _linear_mesh_health_before_order_elevation()
                )
                if illegal_linear_tets > 0 or inverted_linear_tets > 0:
                    warnings.warn(
                        "[Aristo/TPMS] CRITICAL: "
                        f"{illegal_linear_tets} illegal + {inverted_linear_tets} inverted "
                        "linear tets after 3D mesh; aborting setOrder(2), "
                        "exporting linear P1.",
                        stacklevel=2,
                    )
                    effective_order = 1
                    order_elevation_aborted = True
                else:
                    with aristo_stage(f"gmsh_set_order_{effective_order}"):
                        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
                        gmsh.model.mesh.setOrder(effective_order)
        except Exception as mesh_err:
            if _classify_only_meshing(cfg):
                raise RuntimeError(
                    f"[Aristo/TPMS] classifySurfaces volume mesh failed ({mesh_err}); "
                    "single-surface fallback disabled (fea_gmsh_classify_only)."
                ) from mesh_err
            warnings.warn(
                f"[Aristo/TPMS] classified volume mesh failed ({mesh_err}); "
                "retrying single discrete surface (no classify).",
                stacklevel=2,
            )
            if gmsh_initialized:
                try:
                    gmsh.finalize()
                except Exception:
                    pass
                gmsh_initialized = False
            gmsh.initialize()
            gmsh_initialized = True
            gmsh.option.setNumber("General.Terminal", 1 if show_gmsh else 0)
            gmsh.model.add("aristo_tpms_stl_fallback")
            return _generate_single_surface_fallback(mesh_work, h, use_curvature)

        if mesh_meta is not None:
            mesh_meta.update(
                {
                    "requested_mesh_order": int(mesh_order),
                    "effective_mesh_order": effective_order,
                    "order_elevation_aborted": order_elevation_aborted,
                    "illegal_linear_tets": illegal_linear_tets,
                    "inverted_linear_tets": inverted_linear_tets,
                    "algorithm_3d": int(
                        os.environ.get("ARISTO_GMSH_ALGORITHM3D", "").strip() or "1"
                    ),
                    "optimize_netgen": 0,
                }
            )

        with aristo_stage("extract_tet_mesh"):
            return _extract_gmsh_volume_mesh(effective_order)
    finally:
        if gmsh_initialized:
            try:
                gmsh.finalize()
            except Exception:
                pass


def generate_lattice_fea_mesh_from_step(
    step_path: str | Path,
    fea_mesh_resolution: float,
    *,
    silent: bool = True,
    use_curvature: bool = True,
    heal_tolerance: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TPMS/lattice volume mesh from STEP/BREP via OpenCASCADE + healShapes.
    """
    import gmsh

    path = Path(step_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    h = float(fea_mesh_resolution)
    elements_per_two_pi = int(
        os.environ.get("ARISTO_GMSH_ELEMENTS_PER_TWO_PI", "32")
    )

    gmsh_initialized = False
    try:
        gmsh.initialize()
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0 if silent else 1)
        gmsh.model.add("aristo_tpms_step")

        gmsh.model.occ.importShapes(str(path))
        gmsh.model.occ.healShapes(
            tolerance=heal_tolerance,
            fixDegenerated=True,
            fixSmallEdges=True,
            fixSmallFaces=True,
            sewFaces=True,
            makeSolids=True,
        )
        gmsh.model.occ.synchronize()

        vols = gmsh.model.getEntities(3)
        if not vols:
            raise RuntimeError("[Aristo/TPMS] No volumes after STEP import + heal.")

        step_cfg = AristoConfig(fea_mesh_resolution=h)
        if use_curvature:
            if _use_adaptive_curvature_sizing(step_cfg):
                _apply_adaptive_curvature_sizing_options(step_cfg)
            else:
                _apply_curvature_sizing_options(
                    h,
                    elements_per_two_pi=elements_per_two_pi,
                    wall_pitch=h * 0.5,
                )
        else:
            _apply_uniform_sizing_options(h)

        _set_meshing_algorithm_options()
        _apply_boundary_merge_options()
        _cleanup_boundary_nodes_before_volume_mesh()
        gmsh.model.mesh.generate(3)

        from graphite.aristo.aristo_solver import _extract_gmsh_tet_mesh

        return _extract_gmsh_tet_mesh()
    finally:
        if gmsh_initialized:
            try:
                gmsh.finalize()
            except Exception:
                pass


def generate_lattice_fea_mesh(
    mesh: trimesh.Trimesh,
    fea_mesh_resolution: float,
    *,
    config: AristoConfig | None = None,
    step_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch STL (trimesh) or STEP (OCC) lattice meshing."""
    if step_path is not None:
        return generate_lattice_fea_mesh_from_step(
            step_path, fea_mesh_resolution, silent=True
        )
    return generate_lattice_fea_mesh_from_stl(
        mesh, fea_mesh_resolution, config=config, silent=True
    )
