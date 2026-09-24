"""
Carbon-style lightweight 3D surface preview for Graphite.

Renders base CAD as an opaque studio body and paints the TPMS wall footprint
onto the exterior boundary in vibrant cyan (#0FA4AF) with a Blender-style dark build-plate
floor grid underneath. Supports shelling preview (core, skin, combined) and arbitrary CAD geometries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import numpy as np
import pyvista as pv
import trimesh
from graphite.geometry.primitives import generate_primitive
from graphite.implicit.density_control import tau_from_wall_thickness_mm
from graphite.math.tpms import evaluate_tpms
from graphite.math.woodpile import evaluate_woodpile

SUPPORTED_TPMS_TYPES: tuple[str, ...] = (
    "Gyroid",
    "Diamond",
    "Schwarz-P",
    "Schwarz-Diamond",
    "Neovius",
    "Lidinoid",
    "Split-P",
    "Woodpile",
    "Cross-Hatch",
)


def build_primitive_mesh(shape: str, size: float | tuple[float, float, float]) -> trimesh.Trimesh:
    """Create a primitive CAD trimesh."""
    return generate_primitive(shape, size)


def load_cad_mesh(
    source_type: str,
    primitive_shape: str = "Cube",
    size: float = 20.0,
    file_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """
    Load or generate a CAD mesh from a primitive shape or an external file.

    Parameters
    ----------
    source_type : str
        Source type identifier: "Primitive" or "File" / "STL" / "Upload".
    primitive_shape : str, optional
        Shape name for primitives ("Cube", "Cylinder", "Sphere", "Toros", "Torus"), by default "Cube".
    size : float, optional
        Dimension for primitive generation in mm, by default 20.0.
    file_path : str | Path | None, optional
        Filepath to STL/OBJ/PLY/mesh when source_type is "File" or "STL".

    Returns
    -------
    trimesh.Trimesh
        The loaded or generated triangular CAD mesh.
    """
    st = str(source_type).strip().lower()
    if st in ("file", "stl", "upload", "custom", "path") or (
        file_path is not None and st not in ("primitive", "shape")
    ):
        if file_path is None:
            raise ValueError("file_path must be provided when source_type is 'File' or 'STL'")
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"CAD mesh file not found: {p}")
        mesh = trimesh.load(str(p), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load a valid triangle mesh from {p}")
    else:
        mesh = build_primitive_mesh(primitive_shape, size)

    # Always-on CAD health check & auto-repair triage
    has_non_manifold = False
    if len(mesh.faces) > 0 and len(mesh.edges_unique_inverse) > 0:
        counts = np.bincount(mesh.edges_unique_inverse)
        has_non_manifold = bool(np.any(counts > 2))

    if not mesh.is_watertight or len(mesh.faces) == 0 or has_non_manifold:
        from graphite.repair.repair_suite import repair_trimesh_gentle
        repaired_mesh = repair_trimesh_gentle(mesh)
        if repaired_mesh is not None and len(repaired_mesh.faces) > 0:
            mesh = repaired_mesh

    return mesh


def create_floor_grid(
    bounds: tuple[float, float, float, float, float, float] | list[float] | np.ndarray,
    padding: float = 12.0,
    grid_spacing: float = 5.0,
) -> pv.PolyData:
    """Create a build-plate floor grid plane beneath the CAD part (Carbon Design Engine style)."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    wx = max(xmax - xmin + 2.0 * padding, 30.0)
    wy = max(ymax - ymin + 2.0 * padding, 30.0)
    size_x = float(np.ceil(wx / grid_spacing) * grid_spacing)
    size_y = float(np.ceil(wy / grid_spacing) * grid_spacing)
    i_res = max(int(round(size_x / grid_spacing)), 2)
    j_res = max(int(round(size_y / grid_spacing)), 2)
    z_floor = float(zmin) - 0.05

    return pv.Plane(
        center=(cx, cy, z_floor),
        direction=(0, 0, 1),
        i_size=size_x,
        j_size=size_y,
        i_resolution=i_res,
        j_resolution=j_res,
    )


def _evaluate_surface_field(
    pts: np.ndarray,
    lattice_type: str,
    L: float,
    tau_nominal: float,
    enable_grading: bool,
    grading_axis: str,
    grading_mode: str,
    doubling_interval_mm: float,
    min_solid_fraction: float,
    max_solid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate field and threshold at points. Returns (field_vals, thresh_vals)
    such that solid is defined by field_vals <= thresh_vals.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    l_type = lattice_type.strip().lower()

    if l_type in ("woodpile", "cross-hatch"):
        true_wood = (l_type == "woodpile")
        F = evaluate_woodpile(x, y, z, pore_size=L, true_woodpile=true_wood)
        return F, np.zeros_like(F)

    # TPMS Uniform
    if not enable_grading:
        k = 2.0 * np.pi / L
        F = evaluate_tpms(lattice_type, k, x, y, z)
        return np.abs(F), np.full_like(F, tau_nominal)

    # Graded TPMS
    axis_upper = grading_axis.upper()
    if axis_upper in ("RADIAL", "CYLINDRICAL"):
        s = np.sqrt(x**2 + y**2)
    else:
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        ax = axis_map.get(axis_upper, 2)
        s = pts[:, ax]

    s_min = float(np.min(s))
    s_max = float(np.max(s))
    span = max(s_max - s_min, 1e-6)
    W = np.clip((s - s_min) / span, 0.0, 1.0)
    W = 3.0 * W**2 - 2.0 * W**3

    if grading_mode.lower() in ("solid_fraction", "sf"):
        tau_min = min_solid_fraction / 1.15
        tau_max = max_solid_fraction / 1.15
        tau_field = tau_min + W * (tau_max - tau_min)
        k = 2.0 * np.pi / L
        F = evaluate_tpms(lattice_type, k, x, y, z)
        return np.abs(F), tau_field
    else:
        D = max(float(doubling_interval_mm), 0.1)
        c = np.log(2.0) / D
        k0 = 2.0 * np.pi / L
        s_rel = np.maximum(s - s_min, 0.0)
        theta = (k0 / c) * (1.0 - np.exp(-c * s_rel))
        k_inst = k0 * np.exp(-c * s_rel)

        if axis_upper in ("RADIAL", "CYLINDRICAL"):
            coords = [x * (k_inst / k0), y * (k_inst / k0), z * k0]
            F = evaluate_tpms(lattice_type, 1.0, coords[0], coords[1], coords[2])
        else:
            coords = [x * k_inst, y * k_inst, z * k_inst]
            coords[ax] = theta
            F = evaluate_tpms(lattice_type, 1.0, coords[0], coords[1], coords[2])

        return np.abs(F), np.full_like(F, tau_nominal)


def extract_surface_tpms_walls(
    cad_mesh: trimesh.Trimesh,
    lattice_type: str = "Gyroid",
    unit_cell_size: float = 5.0,
    solid_fraction: float = 0.30,
    wall_thickness_mm: float | None = None,
    tau: float | None = None,
    enable_grading: bool = False,
    grading_axis: str = "Z",
    grading_mode: str = "cell_size",
    doubling_interval_mm: float = 5.0,
    min_solid_fraction: float = 0.15,
    max_solid_fraction: float = 0.45,
    export_mode: Literal["core", "skin", "combined"] = "core",
    shell_thickness: float = 2.0,
    cutaway: bool = False,
) -> tuple[pv.PolyData, pv.PolyData | None]:
    """
    Subdivide CAD surface adaptively and clip using exact linear interpolation along edges
    to produce smooth, unbroken, non-jagged TPMS / Woodpile wall boundaries in vibrant cyan (#0FA4AF).

    Supports arbitrary CAD meshes, woodpiles, radial grading, and shelling preview:
    - 'core': previews the CAD mesh and exterior surface lattice wall footprint.
    - 'skin': previews the solid outer shell boundary (no interior wall footprint).
    - 'combined': previews the outer shell boundary and interior lattice walls at the core interface.
    """
    if isinstance(cad_mesh, trimesh.Scene):
        cad_mesh = cad_mesh.dump(concatenate=True)

    if len(cad_mesh.vertices) == 0:
        empty = pv.PolyData()
        return empty, None

    # Resolve threshold tau and estimate physical wall thickness
    L = max(float(unit_cell_size), 0.1)
    if tau is not None:
        tau_val = float(tau)
        w_est = (tau_val / 1.15) * L / np.pi
    elif wall_thickness_mm is not None and float(wall_thickness_mm) > 0:
        w_est = float(wall_thickness_mm)
        tau_val = float(tau_from_wall_thickness_mm(w_est, L).ravel()[0])
    else:
        phi = float(np.clip(float(solid_fraction), 0.01, 0.95))
        tau_val = phi / 1.15
        w_est = (tau_val / 1.15) * L / np.pi

    # Adapt edge size to wall thickness and part size so arbitrary meshes subdivide smoothly
    diag = float(np.linalg.norm(cad_mesh.bounds[1] - cad_mesh.bounds[0])) if len(cad_mesh.vertices) > 0 else 20.0
    min_edge = max(0.20, diag / 75.0)
    target_edge = float(np.clip(max(w_est * 0.75, min_edge), 0.20, 2.0))

    mesh_dense = cad_mesh.copy()
    if len(mesh_dense.faces) < 200_000:
        try:
            mesh_dense = mesh_dense.subdivide_to_size(target_edge)
        except Exception:
            pass

    pv_cad = pv.wrap(mesh_dense)
    mode = str(export_mode).strip().lower()

    # If skin only, preview outer boundary shell with no TPMS walls
    if mode == "skin":
        return pv_cad, None

    # If combined, evaluate interior walls at the inner shell interface
    if mode == "combined":
        eff_shell = min(max(float(shell_thickness), 0.0), diag * 0.4)
        mesh_inner = mesh_dense.copy()
        if len(mesh_inner.vertex_normals) == len(mesh_inner.vertices):
            mesh_inner.vertices = mesh_inner.vertices - eff_shell * mesh_inner.vertex_normals

        pv_inner = pv.wrap(mesh_inner)
        f_vals, thresh_vals = _evaluate_surface_field(
            pv_inner.points,
            lattice_type=lattice_type,
            L=L,
            tau_nominal=tau_val,
            enable_grading=enable_grading,
            grading_axis=grading_axis,
            grading_mode=grading_mode,
            doubling_interval_mm=doubling_interval_mm,
            min_solid_fraction=min_solid_fraction,
            max_solid_fraction=max_solid_fraction,
        )
        pv_inner.point_data["level_diff"] = f_vals - thresh_vals

        try:
            clipped = pv_inner.clip_scalar(scalars="level_diff", value=0.0, invert=True)
            if clipped.n_points == 0:
                clipped = None
            else:
                clipped = clipped.compute_normals(cell_normals=False, point_normals=True)
                clipped.points = clipped.points + 0.02 * clipped.point_data["Normals"]
        except Exception:
            clipped = None

        if cutaway:
            cx = float((pv_cad.bounds[0] + pv_cad.bounds[1]) / 2.0)
            pv_cad = pv_cad.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
            if clipped is not None:
                clipped = clipped.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)

        return pv_cad, clipped

    # Default 'core' mode: evaluate field directly on exterior CAD surface
    f_vals, thresh_vals = _evaluate_surface_field(
        pv_cad.points,
        lattice_type=lattice_type,
        L=L,
        tau_nominal=tau_val,
        enable_grading=enable_grading,
        grading_axis=grading_axis,
        grading_mode=grading_mode,
        doubling_interval_mm=doubling_interval_mm,
        min_solid_fraction=min_solid_fraction,
        max_solid_fraction=max_solid_fraction,
    )
    pv_cad.point_data["level_diff"] = f_vals - thresh_vals

    try:
        clipped = pv_cad.clip_scalar(scalars="level_diff", value=0.0, invert=True)
        if clipped.n_points == 0:
            clipped = None
        else:
            clipped = clipped.compute_normals(cell_normals=False, point_normals=True)
            clipped.points = clipped.points + 0.02 * clipped.point_data["Normals"]
    except Exception:
        clipped = None

    if cutaway:
        cx = float((pv_cad.bounds[0] + pv_cad.bounds[1]) / 2.0)
        pv_cad = pv_cad.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
        if clipped is not None:
            clipped = clipped.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)

    return pv_cad, clipped


def generate_surface_tpms_preview(
    shape: str = "Cube",
    size: float = 20.0,
    lattice_type: str = "Gyroid",
    unit_cell_size: float = 5.0,
    solid_fraction: float = 0.30,
    wall_thickness_mm: float | None = None,
    tau: float | None = None,
    enable_grading: bool = False,
    grading_axis: str = "Z",
    grading_mode: str = "cell_size",
    doubling_interval_mm: float = 5.0,
    min_solid_fraction: float = 0.15,
    max_solid_fraction: float = 0.45,
    export_mode: Literal["core", "skin", "combined"] = "core",
    shell_thickness: float = 2.0,
    cad_mesh: trimesh.Trimesh | None = None,
    cad_file: str | Path | None = None,
    cutaway: bool = False,
) -> tuple[pv.PolyData, pv.PolyData | None]:
    """Generate opaque base CAD PolyData and vibrant cyan wall footprint PolyData."""
    if cad_mesh is not None:
        mesh = cad_mesh
    elif cad_file is not None:
        mesh = load_cad_mesh("file", file_path=cad_file)
    else:
        mesh = load_cad_mesh("primitive", primitive_shape=shape, size=size)

    return extract_surface_tpms_walls(
        cad_mesh=mesh,
        lattice_type=lattice_type,
        unit_cell_size=unit_cell_size,
        solid_fraction=solid_fraction,
        wall_thickness_mm=wall_thickness_mm,
        tau=tau,
        enable_grading=enable_grading,
        grading_axis=grading_axis,
        grading_mode=grading_mode,
        doubling_interval_mm=doubling_interval_mm,
        min_solid_fraction=min_solid_fraction,
        max_solid_fraction=max_solid_fraction,
        export_mode=export_mode,
        shell_thickness=shell_thickness,
        cutaway=cutaway,
    )


def extract_fast_sc_surface_dual(
    cad_mesh: trimesh.Trimesh,
    cell_size: float,
    rule_name: str = "octahedral",
    mode: str = "conformal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract conformal or boolean surface dual chords for Modular SC Hex grids.
    Uses canonical SC Nodal Conformation (Universal Role Surface Dual) with sub-second responsiveness.

    Returns
    -------
    dual_nodes : (N, 3) np.ndarray
        Coordinates of surface dual nodes on the CAD boundary.
    dual_struts : (M, 2) np.ndarray
        Strut indices indexing into dual_nodes.
    boundary_nodes : (B, 3) np.ndarray
        Boundary anchor nodes.
    """
    from graphite.explicit.nodal_conformation import generate_nodal_conformation

    nc = generate_nodal_conformation(
        cad_mesh,
        cell_size=float(cell_size),
        rule_name=rule_name,
        surface_dual_mode="topology_only",
    )
    dual_pts = nc.dual_nodes_projected if len(nc.dual_nodes_projected) > 0 else nc.dual_nodes
    return dual_pts, nc.dual_struts, dual_pts


def extract_fast_a15_surface_dual(
    cad_mesh: trimesh.Trimesh,
    cell_size: float,
    mode: str = "conformal",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract conformal surface dual chords for A15 Kagome lattice.
    Uses vectorized unique face culling and skips internal volume graph construction,
    internal BFS depth tagging, and relaxation for fast preview.

    Returns
    -------
    dual_nodes : (N, 3) np.ndarray
        Coordinates of surface dual nodes on the CAD boundary.
    dual_struts : (M, 2) np.ndarray
        Strut indices indexing into dual_nodes.
    boundary_nodes : (B, 3) np.ndarray
        Boundary anchor nodes.
    """
    from graphite.explicit.proven_topologies import generate_background_grid
    from graphite.explicit.a15_conformal import (
        FACE_TRIPLETS,
        safe_signed_distance,
        project_to_cad_surface,
    )

    # 1. Background grid
    tet_nodes, tets = generate_background_grid("A15", cad_mesh.bounds, float(cell_size))

    if len(tets) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    # 2. Vectorized unique face extraction
    all_faces = np.sort(tets[:, FACE_TRIPLETS], axis=-1).reshape(-1, 3)
    unique_faces, inv_map = np.unique(all_faces, axis=0, return_inverse=True)
    unique_centroids = tet_nodes[unique_faces].mean(axis=1)

    # 3. Query signed distance only once per unique face
    s_dists_unique = safe_signed_distance(cad_mesh, unique_centroids)
    s_dists_all = s_dists_unique[inv_map].reshape(len(tets), 4)

    # 4. Cull tetrahedra
    if mode == "boolean":
        kept_mask = np.any(s_dists_all >= -1e-5, axis=1)
    else:
        kept_mask = np.all(s_dists_all >= -1e-5, axis=1)
        if not np.any(kept_mask):
            # Fallback to boolean if no tets are fully inside
            kept_mask = np.any(s_dists_all >= -1e-5, axis=1)

    surviving_tets = tets[kept_mask]
    if len(surviving_tets) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    # 5. Extract boundary faces (faces owned by exactly 1 surviving tet)
    surv_faces = np.sort(surviving_tets[:, FACE_TRIPLETS], axis=-1).reshape(-1, 3)
    u_surv, counts = np.unique(surv_faces, axis=0, return_counts=True)
    b_faces = u_surv[counts == 1]

    if len(b_faces) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    # 6. Compute boundary face centroids
    b_cents = tet_nodes[b_faces].mean(axis=1)
    if mode == "conformal":
        dual_nodes, _ = project_to_cad_surface(b_cents, cad_mesh)
    else:
        dual_nodes = b_cents

    # 7. Connect adjacent boundary faces sharing an edge to form Kagome surface dual chords
    edges_01 = b_faces[:, [0, 1]]
    edges_12 = b_faces[:, [1, 2]]
    edges_02 = b_faces[:, [0, 2]]
    all_b_edges = np.vstack([edges_01, edges_12, edges_02])
    face_owner = np.tile(np.arange(len(b_faces)), 3)

    _u_edges, inv_e, e_counts = np.unique(all_b_edges, axis=0, return_inverse=True, return_counts=True)
    shared_mask = (e_counts == 2)

    edge_to_owners: dict[int, list[int]] = {}
    for e_idx, f_idx in zip(inv_e, face_owner):
        if shared_mask[e_idx]:
            edge_to_owners.setdefault(int(e_idx), []).append(int(f_idx))

    dual_struts = np.array([owners for owners in edge_to_owners.values() if len(owners) == 2], dtype=np.int64)

    return dual_nodes, dual_struts, dual_nodes


def generate_explicit_preview(
    shape: str = "Cube",
    size: float = 20.0,
    cad_mesh: trimesh.Trimesh | None = None,
    cad_file: str | Path | None = None,
    lattice_type: str = "SC",
    rule_name: str = "octahedral",
    cell_size: float = 5.0,
    strut_radius: float = 0.40,
    mode: str = "conformal",
    cutaway: bool = False,
    surface_only: bool = True,
) -> tuple[pv.PolyData, pv.PolyData | None, pv.PolyData | None]:
    """
    Generate lightweight fast line-network wireframe preview for explicit strut lattices.
    Defaults to fast surface-dual-only extraction for sub-second responsiveness.

    Returns
    -------
    pv_cad : pv.PolyData
        Wrapped CAD mesh (opaque body).
    pv_struts : pv.PolyData | None
        Line network representing the surface dual chords.
    pv_boundary : pv.PolyData | None
        Point glyphs representing boundary/surface nodes.
    """
    if cad_mesh is not None:
        mesh = cad_mesh
    elif cad_file is not None:
        mesh = load_cad_mesh("file", file_path=cad_file)
    else:
        mesh = load_cad_mesh("primitive", primitive_shape=shape, size=size)

    is_a15 = str(lattice_type).strip().upper() == "A15"

    if surface_only:
        # Fast Surface-Dual-Only Preview path
        if is_a15:
            nodes, struts, b_nodes = extract_fast_a15_surface_dual(
                mesh, cell_size=cell_size, mode=mode
            )
        else:
            nodes, struts, b_nodes = extract_fast_sc_surface_dual(
                mesh, cell_size=cell_size, rule_name=rule_name, mode=mode
            )

        pv_cad = pv.wrap(mesh)

        if len(nodes) > 0 and len(struts) > 0:
            lines = np.column_stack([np.full(len(struts), 2, dtype=np.int64), struts]).ravel()
            pv_struts = pv.PolyData(nodes, lines=lines)
        else:
            pv_struts = None

        if len(b_nodes) > 0:
            pv_boundary = pv.PolyData(b_nodes)
        else:
            pv_boundary = None
    else:
        from graphite.explicit.a15_conformal import generate_a15_conformal_lattice
        from graphite.explicit import generate_conformal_lattice as generate_sc_conformal_lattice

        try:
            if is_a15:
                res = generate_a15_conformal_lattice(
                    mesh,
                    cell_size=float(cell_size),
                    strut_radius=float(strut_radius),
                    export_dir="output",
                    skip_sweep=True,
                    mode=mode,
                )
            else:
                res = generate_sc_conformal_lattice(
                    mesh,
                    cell_size=float(cell_size),
                    strut_radius=float(strut_radius),
                    lattice_type="SC",
                    rule_name=rule_name,
                    skip_sweep=True,
                    mode=mode,
                    volume_fraction_threshold=0.5 if mode == "conformal" else 1.0,
                )
        except Exception:
            if mode == "conformal":
                if is_a15:
                    res = generate_a15_conformal_lattice(
                        mesh,
                        cell_size=float(cell_size),
                        strut_radius=float(strut_radius),
                        export_dir="output",
                        skip_sweep=True,
                        mode="boolean",
                    )
                else:
                    res = generate_sc_conformal_lattice(
                        mesh,
                        cell_size=float(cell_size),
                        strut_radius=float(strut_radius),
                        lattice_type="SC",
                        rule_name=rule_name,
                        skip_sweep=True,
                        mode="boolean",
                    )
            else:
                raise

        nodes = res.get("nodes_relaxed")
        if nodes is None:
            nodes = res.get("nodes_3d", np.empty((0, 3), dtype=np.float64))

        red_struts = res.get("red_struts", np.empty((0, 2), dtype=np.int64))
        cyan_struts = res.get("cyan_struts", np.empty((0, 2), dtype=np.int64))

        if len(red_struts) > 0 and len(cyan_struts) > 0:
            all_struts = np.vstack([red_struts, cyan_struts])
        elif len(cyan_struts) > 0:
            all_struts = cyan_struts
        elif len(red_struts) > 0:
            all_struts = red_struts
        else:
            all_struts = np.empty((0, 2), dtype=np.int64)

        pv_cad = pv.wrap(mesh)

        if len(nodes) > 0 and len(all_struts) > 0:
            lines = np.column_stack([np.full(len(all_struts), 2, dtype=np.int64), all_struts]).ravel()
            pv_struts = pv.PolyData(nodes, lines=lines)
        else:
            pv_struts = None

        if len(nodes) > 0 and len(cyan_struts) > 0:
            b_ids = np.unique(cyan_struts)
            pv_boundary = pv.PolyData(nodes[b_ids])
        else:
            pv_boundary = None

    if cutaway:
        cx = float((pv_cad.bounds[0] + pv_cad.bounds[1]) / 2.0)
        pv_cad = pv_cad.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
        if pv_struts is not None:
            pv_struts = pv_struts.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
        if pv_boundary is not None:
            pv_boundary = pv_boundary.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)

    return pv_cad, pv_struts, pv_boundary


def generate_interlinked_preview(
    shape: str = "Cube",
    size: float = 20.0,
    cad_file: str | Path | None = None,
    cell_type: str = "c6tt",
    seeding_type: str = "cartesian",
    pitch: float = 10.0,
    wire_radius: float = 0.40,
    min_clearance: float = 0.35,
    cull_margin: float = 0.50,
    cylinder_radius: float = 15.0,
    cylinder_height: float = 25.0,
    cutaway: bool = False,
    num_segments: int = 16,
) -> tuple[pv.PolyData, pv.PolyData | None, pv.PolyData | None]:
    """
    Generate interactive preview for explicit interlinked metamaterials.

    Implements Policy A (Strict Inset Culling): overlays periodic cell sites
    over the CAD boundary and drops any candidate particle whose nodes or bounding
    volume fall outside the CAD boundary (minus cull_margin), guaranteeing 100% whole,
    uncut particles inside the part.

    Returns
    -------
    tuple[pv.PolyData, pv.PolyData | None, pv.PolyData | None]
        (pv_cad, pv_seed, pv_connecting)
        pv_cad: CAD boundary envelope.
        pv_seed: Central seed cell/particle mesh (rendered 100% solid/opaque).
        pv_connecting: Connecting/neighboring particles mesh (rendered 90% transparent gray).
    """
    from graphite.explicit.interlinked import InterlinkedConfig, generate_interlinked_lattice
    from graphite.explicit.interlinked.generator import _particles_to_combined_mesh

    mesh = load_cad_mesh("File" if cad_file else "Primitive", shape, size, cad_file)
    pv_cad = pv.wrap(mesh)

    seeding_mode = str(seeding_type).strip().lower()
    pv_seed: pv.PolyData | None = None
    pv_connecting: pv.PolyData | None = None

    try:
        if seeding_mode in ("cylindrical", "cylinder", "cylindrical_wrap"):
            cfg = InterlinkedConfig(
                cell=cell_type,
                seeding_type="cylindrical",
                cylinder_radius=float(cylinder_radius),
                cylinder_height=float(cylinder_height),
                pitch=float(pitch),
                wire_radius=float(wire_radius),
                min_clearance=float(min_clearance),
                num_ring_segments=int(num_segments),
            )
            res = generate_interlinked_lattice(cfg, check_clearance=False)
        else:
            min_b, max_b = mesh.bounds[0], mesh.bounds[1]
            p = float(pitch)
            grid_size = np.maximum(np.ceil((max_b - min_b) / p).astype(int) + 1, 1)
            origin = min_b + (max_b - min_b - (grid_size - 1) * p) / 2.0

            cfg = InterlinkedConfig(
                cell=cell_type,
                seeding_type="cartesian",
                grid_size=tuple(grid_size),
                pitch=p,
                wire_radius=float(wire_radius),
                min_clearance=float(min_clearance),
                cull_margin=float(cull_margin),
                origin=tuple(origin),
                num_ring_segments=int(num_segments),
            )
            res = generate_interlinked_lattice(cfg, boundary_mesh=mesh, check_clearance=False)

        particles = getattr(res, "particles", [])
        if particles and len(particles) > 0:
            centers = np.array([p_obj.center for p_obj in particles], dtype=np.float64)
            centroid = np.mean(centers, axis=0)
            dists = np.linalg.norm(centers - centroid, axis=1)
            min_idx = int(np.argmin(dists))
            seed_cell_idx = particles[min_idx].cell_index

            if seed_cell_idx:
                seed_particles = [p_obj for p_obj in particles if p_obj.cell_index == seed_cell_idx]
                connecting_particles = [p_obj for p_obj in particles if p_obj.cell_index != seed_cell_idx]
            else:
                seed_particles = [particles[min_idx]]
                connecting_particles = [p_obj for i, p_obj in enumerate(particles) if i != min_idx]

            if len(seed_particles) == len(particles) and len(particles) > 1:
                seed_particles = [particles[min_idx]]
                connecting_particles = [p_obj for i, p_obj in enumerate(particles) if i != min_idx]

            m_seed = _particles_to_combined_mesh(
                seed_particles,
                wire_radius=float(wire_radius),
                circular_segments=int(num_segments),
            )
            if m_seed is not None and len(m_seed.vertices) > 0:
                pv_seed = pv.wrap(m_seed)

            if connecting_particles:
                m_conn = _particles_to_combined_mesh(
                    connecting_particles,
                    wire_radius=float(wire_radius),
                    circular_segments=int(num_segments),
                )
                if m_conn is not None and len(m_conn.vertices) > 0:
                    pv_connecting = pv.wrap(m_conn)
        elif res.mesh is not None and len(res.mesh.vertices) > 0:
            pv_seed = pv.wrap(res.mesh)
    except Exception:
        pv_seed = None
        pv_connecting = None

    if cutaway:
        cx = float((pv_cad.bounds[0] + pv_cad.bounds[1]) / 2.0)
        pv_cad = pv_cad.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
        if pv_seed is not None:
            pv_seed = pv_seed.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)
        if pv_connecting is not None:
            pv_connecting = pv_connecting.clip(normal=(1, 0, 0), origin=(cx, 0, 0), invert=False)

    return pv_cad, pv_seed, pv_connecting


def generate_auxetic_preview(
    surface: str = "plate",
    pattern: str = "tetra_chiral",
    width: float = 50.0,
    height: float = 50.0,
    thickness: float = 2.0,
    r_in: float = 15.0,
    r_out: float = 17.5,
    n_circumferential: int = 6,
    r_node: float = 2.0,
    strut_w: float = 1.0,
    square_side: float = 10.0,
    rotation_angle_deg: float = 30.0,
    hinge_radius: float = 0.45,
    cutaway: bool = False,
    cutaway_axis: str = "X",
    cutaway_pos: float = 50.0,
    cutaway_invert: bool = False,
    **kwargs: Any,
) -> pv.PolyData:
    """
    Generate an interactive 3D preview for mechanical auxetic and metamaterial lattices.

    Parameters
    ----------
    surface : str
        Surface geometry: "plate" (flat planar sheet) or "cylinder" / "tube" (seamless sleeve).
    pattern : str
        Metamaterial pattern: "tetra_chiral", "tri_chiral", "anti_tetra_chiral",
        "anti_tri_chiral", "reentrant", "rotating_squares", or "pentamode".
    width : float
        Sheet width or tube length in mm.
    height : float
        Sheet height in mm.
    thickness : float
        Plate or strut thickness in mm.
    r_in : float
        Inner cylinder radius in mm (for cylinder/tube).
    r_out : float
        Outer cylinder radius in mm (for cylinder/tube).
    n_circumferential : int
        Number of periodic cell columns around cylinder circumference or sheet width.
    r_node : float
        Circular node radius in mm for chiral patterns.
    strut_w : float
        Chiral ligament or strut width in mm.
    square_side : float
        Side length of rigid square plates in mm (for rotating squares).
    rotation_angle_deg : float
        Deployment angle in degrees (for rotating squares).
    hinge_radius : float
        Living hinge radius in mm (for rotating squares).
    cutaway : bool
        If True, clips the preview with a dynamic cutting plane.
    cutaway_axis : str
        Cutaway plane normal axis: "X", "Y", or "Z".
    cutaway_pos : float
        Percentage along bounding box extent (0.0 to 100.0%).
    cutaway_invert : bool
        If True, flips the clipped half.

    Returns
    -------
    pv.PolyData
        PyVista PolyData mesh of the auxetic metamaterial.
    """
    pat = str(pattern).strip().lower()
    surf = str(surface).strip().lower()

    if pat == "rotating_squares":
        from graphite.generators.rotating_auxetics import generate_rotating_squares_lattice

        nx = max(int(width / square_side), 2)
        ny = max(int(height / square_side), 2)
        mesh = generate_rotating_squares_lattice(
            (nx, ny),
            square_side=float(square_side),
            plate_thickness=float(thickness),
            hinge_radius=float(hinge_radius),
            rotation_angle_deg=float(rotation_angle_deg),
        )

    elif pat in ("tetra_chiral", "tri_chiral", "anti_tetra_chiral", "anti_tri_chiral", "reentrant"):
        from graphite.explicit.surface_lattice import generate_surface_lattice

        if surf in ("cylinder", "tube"):
            mesh_or_path = generate_surface_lattice(
                surface="cylinder",
                pattern=pat,
                r_in=float(r_in),
                r_out=float(r_out),
                height=float(width),
                n_circumferential=int(n_circumferential),
                r_node=float(r_node),
                strut_w=float(strut_w),
                export_stl=False,
            )
        else:
            mesh_or_path = generate_surface_lattice(
                surface="plate",
                pattern=pat,
                width=float(width),
                height=float(height),
                thickness=float(thickness),
                n_circumferential=int(n_circumferential),
                r_node=float(r_node),
                strut_w=float(strut_w),
                export_stl=False,
            )

        if isinstance(mesh_or_path, (str, Path)):
            mesh = trimesh.load(str(mesh_or_path), force="mesh")
        else:
            mesh = mesh_or_path

    elif pat == "pentamode":
        from graphite.generators.pentamode import generate_pentamode_lattice

        unit_cell_sz = float(kwargs.get("unit_cell_size", 10.0))
        r_min = float(kwargs.get("r_min", 0.35))
        r_max = float(kwargs.get("r_max", 0.90))
        mesh = generate_pentamode_lattice(
            bounds=((0.0, 0.0, 0.0), (float(width), float(height), float(thickness))),
            unit_cell_size=unit_cell_sz,
            r_min=r_min,
            r_max=r_max,
            output_format="mesh",
        )

    else:
        raise ValueError(
            f"Unsupported auxetic pattern '{pattern}'. Supported: 'tetra_chiral', 'tri_chiral', "
            f"'anti_tetra_chiral', 'anti_tri_chiral', 'reentrant', 'rotating_squares', 'pentamode'."
        )

    pv_auxetic = pv.wrap(mesh)

    if cutaway:
        axis_str = str(cutaway_axis).strip().upper()
        axis_idx = 0 if axis_str == "X" else (1 if axis_str == "Y" else 2)
        bounds = pv_auxetic.bounds
        p_val = bounds[2 * axis_idx] + (bounds[2 * axis_idx + 1] - bounds[2 * axis_idx]) * (float(cutaway_pos) / 100.0)
        normal = (1.0, 0.0, 0.0) if axis_str == "X" else ((0.0, 1.0, 0.0) if axis_str == "Y" else (0.0, 0.0, 1.0))
        origin = (p_val, 0.0, 0.0) if axis_str == "X" else ((0.0, p_val, 0.0) if axis_str == "Y" else (0.0, 0.0, p_val))
        pv_auxetic = pv_auxetic.clip(normal=normal, origin=origin, invert=bool(cutaway_invert))

    return pv_auxetic





