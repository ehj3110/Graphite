"""
Graphite Conformal Lattice Engine V2 — Unified Conformal Generator

Supports both:
- A15: Conformed Tetrahedral template grid + Kagome dual routing.
- SC: Conformed Simple Cubic template grid + Octahedral dual routing.

No GMSH dependency. 100% GMSH-free watertight meshing.
"""

from __future__ import annotations
import os
import time
import tempfile
import warnings
from pathlib import Path
from collections import defaultdict, deque
from itertools import combinations
from typing import NamedTuple, Callable

import numpy as np
import trimesh

from .mesh_repair import repair_cad_mesh
from .sizing_solver import solve_sizing
from .geometry_module import (
    boolean_intersect_with_cad,
    generate_rectangular_surface_cage,
    union_lattice_with_spherical_joints,
    union_solid_meshes,
)


class ScaffoldResult(NamedTuple):
    """
    Structured return type for conformed background grid scaffolds.
    """
    nodes: np.ndarray
    elements: np.ndarray
    surface_faces: np.ndarray
    element_order: int = 1


# Face definitions
TET_FACE_TRIPLETS = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
HEX_FACE_QUADS = np.array([
    [0, 1, 2, 3],  # Bottom (Z=0)
    [4, 5, 6, 7],  # Top (Z=1)
    [0, 1, 5, 4],  # Front (Y=0)
    [1, 2, 6, 5],  # Right (X=1)
    [2, 3, 7, 6],  # Back (Y=1)
    [3, 0, 4, 7],  # Left (X=0)
], dtype=np.int64)

# Adjacent face pairs (sharing an edge)
TET_ADJACENT_FACES = list(combinations(range(4), 2))
HEX_ADJACENT_FACES = [
    (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 3), (2, 5),
    (4, 3), (4, 5)
]

LATTICE_CONFIGS = {
    "A15": {
        "faces": TET_FACE_TRIPLETS,
        "adjacent_faces": TET_ADJACENT_FACES,
        "unit_cell_multiplier": 4.0,
        "boundary_valency_cutoff": 3,
        "clique_size": 4,
    },
    "SC": {
        "faces": HEX_FACE_QUADS,
        "adjacent_faces": HEX_ADJACENT_FACES,
        "unit_cell_multiplier": 1.0,
        "boundary_valency_cutoff": 4,
        "clique_size": 8,
    }
}


def _pt_key(pt: np.ndarray) -> tuple[int, int, int]:
    return (int(np.round(pt[0] * 2000)), int(np.round(pt[1] * 2000)), int(np.round(pt[2] * 2000)))


def _coord_face_key(nodes: np.ndarray, cell: np.ndarray, face_indices: np.ndarray) -> frozenset[tuple[int, int, int]]:
    face_nodes = cell[face_indices]
    coords = nodes[face_nodes]
    return frozenset(_pt_key(c) for c in coords)


def safe_signed_distance(cad_mesh: trimesh.Trimesh, points: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """
    Computes signed distance of points to the CAD mesh using ray-casting.
    Negative means inside, positive means outside.
    """
    n_pts = len(points)
    s_dists = np.zeros(n_pts, dtype=np.float64)
    proximity = trimesh.proximity.ProximityQuery(cad_mesh)

    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        chunk = points[start:end]
        dists = proximity.signed_distance(chunk)
        # trimesh signed_distance returns positive for inside, negative for outside.
        # We invert it: negative inside, positive outside.
        s_dists[start:end] = -dists

    return s_dists


def project_to_cad_surface(nodes: np.ndarray, cad_mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects nodes to the closest points on the CAD mesh surface.
    Returns (closest_points, is_inside).
    """
    proximity = trimesh.proximity.ProximityQuery(cad_mesh)
    closest_pts, _, _ = proximity.on_surface(nodes)
    # Check inside status
    inside = proximity.signed_distance(nodes) >= -1e-5
    return closest_pts, inside


def apply_sdf_ironing(
    nodes_3d: np.ndarray,
    struts: np.ndarray,
    boundary_node_ids: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    valency_cutoff: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """
    Applies the valency-based boundary snapping (SDF Ironing) rule.
    """
    # Count node valencies
    degrees = np.zeros(len(nodes_3d), dtype=np.int32)
    for u, v in struts:
        degrees[u] += 1
        degrees[v] += 1

    # Project boundary nodes to CAD surface
    boundary_coords = nodes_3d[boundary_node_ids]
    closest_pts, is_inside = project_to_cad_surface(boundary_coords, cad_mesh)

    nodes_ironed = nodes_3d.copy()
    conformed_mask = np.zeros(len(nodes_3d), dtype=bool)
    conformed_nodes_map = {}

    for local_i, idx in enumerate(boundary_node_ids):
        inside = is_inside[local_i]
        valency = degrees[idx]

        conformed = False
        if not inside:
            # Outside: always snap
            conformed = True
        else:
            # Inside: snap only if valency <= cutoff
            conformed = (valency <= valency_cutoff)

        if conformed:
            snap_pt = closest_pts[local_i]
            nodes_ironed[idx] = snap_pt
            conformed_mask[idx] = True
            conformed_nodes_map[idx] = snap_pt

    return nodes_ironed, conformed_mask, conformed_nodes_map


def apply_depth_gated_relaxation(
    nodes_ironed: np.ndarray,
    struts: np.ndarray,
    node_depths: np.ndarray,
    iterations: int = 15,
    alpha: float = 0.5,
    max_depth: int | None = None,
    layer1_weight: float = 1.0,
) -> np.ndarray:
    """
    Apply constrained Laplacian smoothing after boundary projection.

    Depth 0 is the conformed surface and remains fixed; all deeper nodes move at
    full strength (``layer1_weight`` can damp the first layer if needed). Deep
    interior nodes with symmetric neighborhoods self-limit — their neighbor
    average equals their own position — so no rigid-core cap is required. If
    ``max_depth`` is set, nodes deeper than that are frozen anyway.

    The purpose is *radial* redistribution: projected boundary nodes compress or
    stretch the outermost cell layer, and relaxation lets successive interior
    layers absorb that displacement instead of trapping it at the surface.
    """
    nodes_relaxed = nodes_ironed.copy()
    adj = defaultdict(list)
    for u, v in struts:
        adj[u].append(v)
        adj[v].append(u)

    # Depth 0 (boundary conformed): fixed. Depth 1: layer1_weight (default full).
    # Depth >= 2: full relaxation, optionally capped by max_depth.
    weights = np.zeros(len(nodes_relaxed), dtype=np.float64)
    for i in range(len(nodes_relaxed)):
        d = node_depths[i]
        if d <= 0:
            weights[i] = 0.0
        elif max_depth is not None and d > int(max_depth):
            weights[i] = 0.0
        elif d == 1:
            weights[i] = float(layer1_weight)
        else:
            weights[i] = 1.0

    for _ in range(iterations):
        next_nodes = nodes_relaxed.copy()
        for idx in range(len(nodes_relaxed)):
            w = weights[idx]
            if w == 0.0 or len(adj[idx]) == 0:
                continue
            neighbors = adj[idx]
            avg = np.mean(nodes_relaxed[neighbors], axis=0)
            next_nodes[idx] = (1.0 - alpha * w) * nodes_relaxed[idx] + (alpha * w) * avg
        nodes_relaxed = next_nodes

    return nodes_relaxed


def sweep_to_manifold(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """
    Helper to generate watertight solid cylinders using trimesh.
    """
    cylinders = []
    for u, v in struts:
        p0 = nodes[u]
        p1 = nodes[v]
        vec = p1 - p0
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        # Align cylinder to connection vector
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        direction = vec / length
        rotation_matrix = _rotation_matrix_from_z(direction)
        translation = 0.5 * (p0 + p1)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        cyl.apply_transform(transform)
        cylinders.append(cyl)

    if not cylinders:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(cylinders)


def sweep_struts_concat(nodes: np.ndarray, struts: np.ndarray, radius: float = 0.15) -> trimesh.Trimesh:
    """
    Helper to sweep lines to solid cylinders.
    """
    return sweep_to_manifold(nodes, struts, radius)


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    v = np.cross(z_axis, vec)
    c = np.dot(z_axis, vec)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c < 0:
            # Opposite direction
            return -np.eye(3)
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def _generate_sc_modular_lattice(
    cad_filepath: str | trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    strut_radius: float,
    export_dir: str,
    skin_only: bool,
    skin_output_name: str | None,
    skip_sweep: bool,
    signed_distance_fn: Callable[[np.ndarray], np.ndarray] | None,
    mode: str,
    volume_fraction_threshold: float,
    rule_name: str,
    start_time: float,
    sphere_center: np.ndarray | tuple[float, float, float] | None = None,
    sphere_radius: float | None = None,
    relax_layers: int | None = None,
    relax_iterations: int = 15,
    relax_alpha: float = 0.5,
    relax_mode: str = "laplacian",
    max_projection_factor: float | None = 1.0,
    projection_mode: str = "closest",
    snap_outside_nodes: bool = True,
    cull_collapsed_hexes: bool = False,
    surface_cage_profile: str = "rectangular",
    surface_cage_width: float | None = None,
    surface_cage_thickness: float | None = None,
    surface_cage_normal_oversize: float = 0.25,
    surface_cage_project_stations: bool = True,
) -> dict[str, any]:
    """
    Modular SC path:
      VF cull → morph hex cage to CAD → stamp volume rule into deformed bricks
      → surface skin on deformed scaffold.

    Boolean mode skips cage morph / skin (debug/fast).
    """
    from graphite.explicit.conformal_core import (
        cull_hex_elements,
        generate_sc_volume_topology,
        merge_surface_skin,
        morph_hex_scaffold,
    )
    from graphite.explicit.hex_topology_module import get_hex_topology_rule

    if isinstance(cad_filepath, trimesh.Trimesh):
        cad_mesh = repair_cad_mesh(cad_filepath)
        part_name = "mesh"
    else:
        raw_mesh = trimesh.load(cad_filepath)
        print(f"\nRepairing part geometry: {cad_filepath}")
        cad_mesh = repair_cad_mesh(raw_mesh)
        part_name = Path(cad_filepath).stem

    rule = get_hex_topology_rule(rule_name)
    print(
        f"Processing conformed lattice (SC/{rule.name}, mode={mode}) for part: {part_name}"
    )
    print(f"  CAD bounds: {cad_mesh.bounds}")
    print(
        f"  Policy: conform_dofs={sorted(rule.conform_dofs)} "
        f"skin_mode={rule.skin_mode} valency_cutoff={rule.valency_cutoff}"
    )

    hex_elems, _grid_nodes, _surviving, _n_partial = cull_hex_elements(
        cad_mesh,
        cell_size,
        volume_fraction_threshold=volume_fraction_threshold,
        mode=mode,
        signed_distance_fn=signed_distance_fn,
    )

    boundary_faces_count = 0
    cyan_struts = np.empty((0, 2), dtype=np.int64)

    if mode == "conformal":
        # 1) Morph the hex cage (corners ironed; near-surface corners relaxed).
        # 2) Stamp the unit-cell rule into deformed bricks (trilinear morph of contents).
        hex_elems, scaffold_report = morph_hex_scaffold(
            hex_elems,
            cad_mesh,
            valency_cutoff=int(rule.valency_cutoff),
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            relax_layers=relax_layers,
            relax_iterations=relax_iterations,
            relax_alpha=relax_alpha,
            relax_mode=relax_mode,
            cell_size=cell_size,
            max_projection_factor=max_projection_factor,
            projection_mode=projection_mode,
            snap_outside_nodes=snap_outside_nodes,
            cull_collapsed_hexes=cull_collapsed_hexes,
        )
        boundary_faces_count = int(scaffold_report["n_exterior_iron"])
        print(
            f"  Cage morph complete ({scaffold_report.get('relax_mode', 'laplacian')}): "
            f"exposed corners projected="
            f"{scaffold_report['n_exterior_iron']}"
            f" (max disp={scaffold_report.get('max_projection_distance', 0):.4g}); "
            f"relaxed interior nodes={scaffold_report.get('n_relaxed_layer_nodes', 0)}"
            + (
                " (all layers)"
                if scaffold_report.get("relax_layers", -1) == -1
                else f" (capped at depth {scaffold_report.get('relax_layers')})"
            )
            + (
                f"; radial L_r={scaffold_report.get('radial_rest_length', 0):.3g}, "
                f"slide={scaffold_report.get('n_surface_slide', 0)}"
                if scaffold_report.get("relax_mode") == "radial_equalize"
                else ""
            )
        )

        nodes_3d, volume_struts, rule = generate_sc_volume_topology(hex_elems, rule.name)
        print(
            f"  Volume topology stamped into deformed hexes ({rule.name}): "
            f"{len(nodes_3d)} nodes, {len(volume_struts)} struts"
        )

        # Surface skin on the *deformed* scaffold (same hex cage).
        skin_nodes, cyan_struts = merge_surface_skin(hex_elems, nodes_3d, rule)
        if len(skin_nodes) > len(nodes_3d):
            n_extra = len(skin_nodes) - len(nodes_3d)
            nodes_3d = skin_nodes
            print(f"  Appended {n_extra} skin-only nodes")
        print(f"  Surface skin struts: {len(cyan_struts)}")

        # Volume nodes already sit in the morphed cage — do not re-snap interiors.
        nodes_relaxed = nodes_3d
        red_struts = volume_struts.copy()
        struts = volume_struts
    else:
        # boolean / debug: undeformed stamp only
        nodes_3d, volume_struts, rule = generate_sc_volume_topology(hex_elems, rule.name)
        print(
            f"  Volume topology ({rule.name}): {len(nodes_3d)} nodes, "
            f"{len(volume_struts)} struts"
        )
        cyan_struts = np.empty((0, 2), dtype=np.int64)
        red_struts = volume_struts.copy()
        nodes_relaxed = nodes_3d.copy()
        struts = volume_struts

    if skip_sweep:
        return {
            "nodes_count": len(nodes_relaxed),
            "struts_count": len(struts),
            "boundary_faces_count": boundary_faces_count,
            "cyan_struts_count": len(cyan_struts),
            "red_struts_count": len(red_struts),
            "elapsed_time": time.time() - start_time,
            "nodes_relaxed": nodes_relaxed,
            "cyan_struts": cyan_struts,
            "red_struts": red_struts,
            "volume_struts": np.asarray(volume_struts, dtype=np.int64),
            "skin_struts": np.asarray(cyan_struts, dtype=np.int64),
            "rule_name": rule.name,
            "mode": mode,
        }

    cage_profile = str(surface_cage_profile).strip().lower()
    if cage_profile not in ("rectangular", "cylindrical"):
        raise ValueError(
            "surface_cage_profile must be 'rectangular' or 'cylindrical'; "
            f"got {surface_cage_profile!r}"
        )
    cage_width = (
        2.0 * float(strut_radius)
        if surface_cage_width is None
        else float(surface_cage_width)
    )
    cage_thickness = (
        0.5 * cage_width
        if surface_cage_thickness is None
        else float(surface_cage_thickness)
    )

    core_manifold = None
    skin_manifold = None
    skin_raw = None
    if not skin_only and len(red_struts) > 0:
        core_manifold = sweep_to_manifold(nodes_relaxed, red_struts, radius=strut_radius)
    if len(cyan_struts) > 0:
        if cage_profile == "rectangular":
            skin_raw = generate_rectangular_surface_cage(
                nodes_relaxed,
                cyan_struts,
                cad_mesh,
                width=cage_width,
                thickness=cage_thickness,
                normal_oversize=float(surface_cage_normal_oversize),
                crop_to_boundary=False,
                project_stations=bool(surface_cage_project_stations),
            )
            skin_manifold, _ = boolean_intersect_with_cad(skin_raw, cad_mesh)
        else:
            skin_manifold = sweep_to_manifold(
                nodes_relaxed, cyan_struts, radius=1.5 * strut_radius
            )

    os.makedirs(export_dir, exist_ok=True)
    if skin_only:
        skin_fname = skin_output_name or f"{part_name}_surface_dual.stl"
        skin_path = os.path.join(export_dir, skin_fname)
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)
    else:
        combined_lattice = None
        if core_manifold is not None and skin_manifold is not None:
            if cage_profile == "rectangular":
                # Baseball cleanup recipe generalized to arbitrary CAD:
                # union cylindrical core with outward-oversized cage, then
                # intersect the complete solid with CAD for a flush interface.
                combined_lattice = union_solid_meshes([core_manifold, skin_raw])
                combined_lattice, _ = boolean_intersect_with_cad(
                    combined_lattice, cad_mesh
                )
            else:
                combined_lattice, _ = union_lattice_with_spherical_joints(
                    nodes_relaxed,
                    np.vstack([red_struts, cyan_struts]),
                    np.hstack(
                        [
                            np.full(len(red_struts), strut_radius),
                            np.full(len(cyan_struts), 1.5 * strut_radius),
                        ]
                    ),
                )
        elif core_manifold is not None:
            combined_lattice = core_manifold
        elif skin_manifold is not None:
            combined_lattice = skin_manifold

        # Boolean trim only on explicit boolean mode (debug/fast). Conformal default
        # relies on ironing + skin; optional intersect remains available for boolean.
        if combined_lattice is not None and mode == "boolean":
            combined_lattice, _ = boolean_intersect_with_cad(combined_lattice, cad_mesh)

        lattice_path = os.path.join(export_dir, f"{part_name}_conformal_lattice.stl")
        if combined_lattice is not None:
            combined_lattice.export(lattice_path)
        else:
            trimesh.Trimesh().export(lattice_path)

        skin_path = os.path.join(export_dir, f"{part_name}_boundary_skin.stl")
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)

    return {
        "nodes_count": len(nodes_relaxed),
        "struts_count": len(struts),
        "boundary_faces_count": boundary_faces_count,
        "cyan_struts_count": len(cyan_struts),
        "red_struts_count": len(red_struts),
        "elapsed_time": time.time() - start_time,
        "rule_name": rule.name,
        "mode": mode,
        "surface_cage_profile": cage_profile,
        "surface_cage_width": cage_width,
        "surface_cage_thickness": cage_thickness,
    }


def generate_conformal_lattice(
    cad_filepath: str | trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    strut_radius: float,
    lattice_type: str = "A15",
    export_dir: str = "output",
    export_debug_stls: bool = False,
    skin_only: bool = False,
    skin_output_name: str | None = None,
    skip_sweep: bool = False,
    signed_distance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    mode: str = "conformal",
    volume_fraction_threshold: float = 1.0,
    rule_name: str = "octahedral",
    sphere_center: np.ndarray | tuple[float, float, float] | None = None,
    sphere_radius: float | None = None,
    relax_layers: int | None = None,
    relax_iterations: int = 15,
    relax_alpha: float = 0.5,
    relax_mode: str = "laplacian",
    max_projection_factor: float | None = 1.0,
    projection_mode: str = "closest",
    snap_outside_nodes: bool = True,
    cull_collapsed_hexes: bool = False,
    surface_cage_profile: str = "rectangular",
    surface_cage_width: float | None = None,
    surface_cage_thickness: float | None = None,
    surface_cage_normal_oversize: float = 0.25,
    surface_cage_project_stations: bool = True,
) -> dict[str, any]:
    """
    Unified entry point to generate conformed lattices (A15 Kagome or SC hex rules)
    without using GMSH.

    Parameters
    ----------
    volume_fraction_threshold : float, optional
        Fraction of face-centroids that must be inside the CAD mesh for a cell to
        be retained. 1.0 (default) = all face-centroids inside. 0.5 = keep cells
        where at least half of face-centroids are inside.
    rule_name : str, optional
        SC hex topology rule (ignored for A15). Default ``octahedral``.
    mode : str, optional
        ``conformal`` (default): iron boundary DOFs, relax, surface skin.
        ``boolean``: volume topology only (debug/fast; no iron/relax/skin).
    sphere_center, sphere_radius : optional
        If both set (SC conformal), exposed hex corners use algebraic bidirectional
        projection onto the sphere instead of mesh closest-point queries.
    relax_layers : int or None, optional
        If set, freeze scaffold nodes deeper than this many layers. Default None:
        all interior nodes relax (deep symmetric interiors self-limit, so the
        relaxation naturally redistributes radial boundary compression inward
        without a rigid-core cap). Depth 0 (projected surface) is always fixed
        under ``relax_mode="laplacian"``.
    relax_iterations, relax_alpha : optional
        For ``laplacian``: Jacobi iteration count and step size.
        For ``radial_equalize``: spring iterations and step size (use ~300 / 0.2).
    relax_mode : str, optional
        ``laplacian`` (default) or experimental ``radial_equalize`` (sphere-only:
        equalize radial edge lengths with exposed nodes fixed after projection).
    max_projection_factor : float or None, optional
        Bound on boundary-projection travel, as a multiple of the cell dimension
        on each axis (default 1.0). Prefer ``None`` with ``cull_collapsed_hexes``
        so corners reach the surface and crushed bricks are dropped afterward.
    projection_mode : str, optional
        ``closest`` (default) or ``face_normal``. Face-normal mode raycasts along
        owning exposed-face normals so stair-step side walls beat the nearby
        floor when a node sits on both.
    snap_outside_nodes : bool, optional
        Also iron scaffold corners that sit outside the CAD even when they are
        not on an exposed face (shared edges between two partial cells).
    cull_collapsed_hexes : bool, optional
        After morph, drop hexes below ``collapse_warn_ratio`` of their original
        volume (or inverted). Use with ``max_projection_factor=None``.
    surface_cage_profile : str, optional
        SC surface-solid profile. ``rectangular`` (default) keeps volume struts
        cylindrical but sweeps the surface cage with uniform width/thickness,
        outward oversizes it, then Boolean-trims it flush to CAD.
    surface_cage_width, surface_cage_thickness : float, optional
        Uniform cage dimensions. Width defaults to the cylindrical volume-strut
        diameter, ``2 * strut_radius``; thickness defaults to half the width.
        Thickness denotes retained inward depth after CAD trimming.
    surface_cage_normal_oversize : float, optional
        Additional outward stock before the final CAD intersection.
    surface_cage_project_stations : bool, optional
        If True (legacy), loft stations are re-projected onto the CAD. If False,
        the cage follows the skin-node chords (expected when nodes are already
        on the surface) and only samples CAD normals for bar orientation.
    """
    start_time = time.time()
    if str(lattice_type).strip().upper() == "SC":
        return _generate_sc_modular_lattice(
            cad_filepath=cad_filepath,
            cell_size=cell_size,
            strut_radius=strut_radius,
            export_dir=export_dir,
            skin_only=skin_only,
            skin_output_name=skin_output_name,
            skip_sweep=skip_sweep,
            signed_distance_fn=signed_distance_fn,
            mode=mode,
            volume_fraction_threshold=volume_fraction_threshold,
            rule_name=rule_name,
            start_time=start_time,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
            relax_layers=relax_layers,
            relax_iterations=relax_iterations,
            relax_alpha=relax_alpha,
            relax_mode=relax_mode,
            max_projection_factor=max_projection_factor,
            projection_mode=projection_mode,
            snap_outside_nodes=snap_outside_nodes,
            cull_collapsed_hexes=cull_collapsed_hexes,
            surface_cage_profile=surface_cage_profile,
            surface_cage_width=surface_cage_width,
            surface_cage_thickness=surface_cage_thickness,
            surface_cage_normal_oversize=surface_cage_normal_oversize,
            surface_cage_project_stations=surface_cage_project_stations,
        )

    config = LATTICE_CONFIGS.get(lattice_type)
    if config is None:
        raise ValueError(f"Unknown lattice type: {lattice_type}. Supported: 'A15', 'SC'")

    CELL_FACES = config["faces"]
    CELL_ADJACENT_FACES = config["adjacent_faces"]
    UNIT_CELL_MULTIPLIER = config["unit_cell_multiplier"]
    valency_cutoff = config["boundary_valency_cutoff"]

    # 1. Load and repair CAD Mesh
    if isinstance(cad_filepath, trimesh.Trimesh):
        cad_mesh = repair_cad_mesh(cad_filepath)
        part_name = "mesh"
    else:
        raw_mesh = trimesh.load(cad_filepath)
        print(f"\nRepairing part geometry: {cad_filepath}")
        cad_mesh = repair_cad_mesh(raw_mesh)
        part_name = Path(cad_filepath).stem

    print(f"Processing conformed lattice ({lattice_type}) for part: {part_name}")
    print(f"  CAD bounds: {cad_mesh.bounds}")

    # 2. Bounding Box & Grid Auto-Scaling
    min_bound, max_bound = cad_mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # Generate background grid nodes and elements
    from graphite.explicit.proven_topologies import generate_background_grid
    grid_nodes, cells = generate_background_grid(lattice_type, cad_mesh.bounds, cell_size)

    print(f"  Background grid: {len(grid_nodes)} nodes, {len(cells)} cells")

    # 3. Exact Face-Centroid Culling
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in CELL_FACES:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    if signed_distance_fn is not None:
        s_dists = np.asarray(signed_distance_fn(all_centroids), dtype=np.float64)
    else:
        s_dists = safe_signed_distance(cad_mesh, all_centroids)

    kept_cells = []
    n_partial = 0
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        n_faces = len(c_dists)
        # safe_signed_distance: negative = inside, positive = outside
        n_inside = int(np.sum(c_dists <= 1e-5))
        inside_frac = n_inside / n_faces

        if mode == "boolean":
            if n_inside > 0:
                kept_cells.append(cell)
        else:
            if inside_frac >= volume_fraction_threshold:
                kept_cells.append(cell)
                if inside_frac < 1.0:
                    n_partial += 1

    surviving_cells = np.array(kept_cells)
    print(f"  Surviving cells: {len(surviving_cells)} / {len(cells)} ({n_partial} partial boundary cells retained)")
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # 4. Extract boundary faces
    boundary_faces = []
    boundary_faces_set = set()
    if mode == "conformal":
        face_counts = defaultdict(int)
        for cell in surviving_cells:
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_counts[fkey] += 1
        boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]
        boundary_faces_set = set(boundary_faces)

    # 5. Generate dual graph and map boundary nodes
    dual_coords = []
    dual_coord_to_idx = {}
    strut_set = set()
    face_to_centroid = {}

    def get_or_add_dual(coord):
        key = _pt_key(coord)
        if key not in dual_coord_to_idx:
            dual_coord_to_idx[key] = len(dual_coords)
            dual_coords.append(coord.copy())
        return dual_coord_to_idx[key]

    for cell in surviving_cells:
        face_node_ids = []
        for fv in CELL_FACES:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            n_idx = get_or_add_dual(centroid)
            face_node_ids.append(n_idx)

            if mode == "conformal":
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_to_centroid[fkey] = centroid

        for a, b in CELL_ADJACENT_FACES:
            strut_set.add((min(face_node_ids[a], face_node_ids[b]), max(face_node_ids[a], face_node_ids[b])))

    nodes_3d = np.array(dual_coords)
    struts = np.array(sorted(strut_set))

    boundary_node_ids = np.empty(0, dtype=np.int64)
    if mode == "conformal":
        boundary_nodes_list = []
        for bf in boundary_faces:
            centroid = face_to_centroid[bf]
            n_idx = dual_coord_to_idx[_pt_key(centroid)]
            boundary_nodes_list.append(n_idx)
        boundary_node_ids = np.unique(boundary_nodes_list)

    if mode == "conformal":
        # 6. Topological BFS Depth Tagging
        M_t = len(surviving_cells)
        face_to_cells = defaultdict(list)
        for hi, cell in enumerate(surviving_cells):
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                face_to_cells[fkey].append(hi)

        cell_depth = np.full(M_t, -1, dtype=np.int32)
        queue = deque()
        for bf in boundary_faces:
            if bf in face_to_cells:
                for hi in face_to_cells[bf]:
                    if cell_depth[hi] == -1:
                        cell_depth[hi] = 0
                        queue.append(hi)

        cell_neighbors = defaultdict(list)
        for fkey, owning_cells in face_to_cells.items():
            if len(owning_cells) == 2:
                u, v = owning_cells[0], owning_cells[1]
                cell_neighbors[u].append(v)
                cell_neighbors[v].append(u)

        while queue:
            curr = queue.popleft()
            d = cell_depth[curr]
            for nb in cell_neighbors[curr]:
                if cell_depth[nb] == -1:
                    cell_depth[nb] = d + 1
                    queue.append(nb)

        node_depths = np.full(len(nodes_3d), 999999, dtype=np.int32)
        for hi, cell in enumerate(surviving_cells):
            d = cell_depth[hi]
            if d == -1:
                continue
            for fv in CELL_FACES:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                centroid = face_to_centroid[fkey]
                idx = dual_coord_to_idx[_pt_key(centroid)]
                node_depths[idx] = min(node_depths[idx], d)

        node_depths[node_depths == 999999] = int(np.max(node_depths[node_depths != 999999])) + 1 if np.any(node_depths != 999999) else 0

        # 7. SDF Ironing
        nodes_ironed, conformed_mask, conformed_nodes_map = apply_sdf_ironing(
            nodes_3d, struts, boundary_node_ids, cad_mesh, valency_cutoff
        )

        # 8. Depth-Gated Relaxation
        nodes_relaxed = apply_depth_gated_relaxation(
            nodes_ironed, struts, node_depths, iterations=15, alpha=0.5
        )

        # 9. Topological Wiring
        edge_to_faces = defaultdict(list)
        for bf in boundary_faces:
            verts_list = sorted(bf)
            for a, b in combinations(verts_list, 2):
                edge_key = (a, b) if a < b else (b, a)
                edge_to_faces[edge_key].append(bf)

        cyan_struts_set = set()
        for edge, faces in edge_to_faces.items():
            if len(faces) == 2:
                fa, fb = faces[0], faces[1]
                c_a = face_to_centroid[fa]
                c_b = face_to_centroid[fb]
                idx_a = dual_coord_to_idx[_pt_key(c_a)]
                idx_b = dual_coord_to_idx[_pt_key(c_b)]
                cyan_struts_set.add((min(idx_a, idx_b), max(idx_a, idx_b)))

        cyan_struts = np.array(list(cyan_struts_set), dtype=np.int64) if cyan_struts_set else np.empty((0, 2), dtype=np.int64)

        active_nodes = set(range(len(nodes_3d))) - set(boundary_node_ids) | set(conformed_nodes_map.keys())
        red_struts_list = []
        for u, v in struts:
            if u in active_nodes and v in active_nodes:
                red_struts_list.append((u, v))
        red_struts = np.array(red_struts_list, dtype=np.int64) if red_struts_list else np.empty((0, 2), dtype=np.int64)
    else:
        # Boolean mode: no skin, no relaxation, all struts are core (red) struts
        nodes_relaxed = nodes_3d.copy()
        cyan_struts = np.empty((0, 2), dtype=np.int64)
        red_struts = struts.copy()

    if skip_sweep:
        elapsed_time = time.time() - start_time
        return {
            "nodes_count": len(nodes_3d),
            "struts_count": len(struts),
            "boundary_faces_count": len(boundary_faces),
            "cyan_struts_count": len(cyan_struts),
            "red_struts_count": len(red_struts),
            "elapsed_time": elapsed_time,
            "nodes_relaxed": nodes_relaxed,
            "cyan_struts": cyan_struts,
            "red_struts": red_struts,
            # Full octahedral graph before boundary-node filtering (fills cell interiors).
            "volume_struts": np.asarray(struts, dtype=np.int64),
        }

    # 10. Sweep and Export STLs
    core_manifold = None
    skin_manifold = None

    if not skin_only and len(red_struts) > 0:
        core_manifold = sweep_to_manifold(nodes_relaxed, red_struts, radius=strut_radius)
    if len(cyan_struts) > 0:
        skin_manifold = sweep_to_manifold(nodes_relaxed, cyan_struts, radius=1.5 * strut_radius)

    os.makedirs(export_dir, exist_ok=True)

    if skin_only:
        skin_fname = skin_output_name or f"{part_name}_surface_dual.stl"
        skin_path = os.path.join(export_dir, skin_fname)
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)
    else:
        combined_lattice = None
        if core_manifold is not None and skin_manifold is not None:
            # watertight union of struts
            combined_lattice, _ = union_lattice_with_spherical_joints(
                nodes_relaxed,
                np.vstack([red_struts, cyan_struts]),
                np.hstack([np.full(len(red_struts), strut_radius), np.full(len(cyan_struts), 1.5 * strut_radius)]),
            )
        elif core_manifold is not None:
            combined_lattice = core_manifold
        elif skin_manifold is not None:
            combined_lattice = skin_manifold

        if combined_lattice is not None:
            combined_lattice, _ = boolean_intersect_with_cad(combined_lattice, cad_mesh)

        lattice_path = os.path.join(export_dir, f"{part_name}_conformal_lattice.stl")
        if combined_lattice is not None:
            combined_lattice.export(lattice_path)
        else:
            trimesh.Trimesh().export(lattice_path)

        skin_path = os.path.join(export_dir, f"{part_name}_boundary_skin.stl")
        if skin_manifold is not None:
            skin_manifold.export(skin_path)
        else:
            trimesh.Trimesh().export(skin_path)

    elapsed_time = time.time() - start_time
    result = {
        "nodes_count": len(nodes_3d),
        "struts_count": len(struts),
        "boundary_faces_count": len(boundary_faces),
        "cyan_struts_count": len(cyan_struts),
        "red_struts_count": len(red_struts),
        "elapsed_time": elapsed_time,
    }
    return result


def generate_conformal_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    **kwargs
) -> ScaffoldResult:
    """
    GMSH-free conformed background grid generator for tetrahedral meshes (A15).
    Conforms the grid nodes to the boundary mesh and culls elements.
    """
    # 1. Bounding box & bounds scaling
    cell_size = target_element_size
    min_bound, max_bound = mesh.bounds
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    # A15 Basis & Grid Setup
    A15_BASIS = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        [0.25, 0.5, 0.0], [0.75, 0.5, 0.0],
        [0.0, 0.25, 0.5], [0.0, 0.75, 0.5],
        [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
    ], dtype=np.float64)
    BOND_CUTOFF = 0.62

    pts_list = []
    for ix_n in range(-1, 3):
        for iy_n in range(-1, 3):
            for iz_n in range(-1, 3):
                for b in A15_BASIS:
                    pts_list.append(b + np.array([ix_n, iy_n, iz_n], dtype=np.float64))
    pts = np.unique(np.round(np.vstack(pts_list), 9), axis=0)

    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    adj = {i: set() for i in range(len(pts))}
    for u, v in zip(*np.where((dist > 1e-9) & (dist <= BOND_CUTOFF))):
        if u < v:
            adj[u].add(v); adj[v].add(u)

    cliques = []
    for u in range(len(pts)):
        for v in adj[u]:
            if v <= u: continue
            uv = adj[u] & adj[v]
            for w in uv:
                if w <= v: continue
                for x in (uv & adj[w]):
                    if x > w:
                        cliques.append((u, v, w, x))

    arr = np.array(cliques, dtype=np.int64)
    centroids_frac = pts[arr].mean(axis=1)
    inside = np.all((centroids_frac >= -1e-9) & (centroids_frac <= 1.0 + 1e-9), axis=1)
    cliques_inside = arr[inside]

    node_coords_int = []
    node_coord_to_idx = {}

    def get_or_add(coord_int):
        key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
        if key not in node_coord_to_idx:
            node_coord_to_idx[key] = len(node_coords_int)
            node_coords_int.append(coord_int)
        return node_coord_to_idx[key]

    seen_cells = set()
    cells_out = []
    pts_int = np.round(pts * 4.0).astype(np.int64)

    for ix in range(min_ix, max_ix + 1):
        for iy in range(min_iy, max_iy + 1):
            for iz in range(min_iz, max_iz + 1):
                offset = np.array([ix, iy, iz], dtype=np.int64) * 4
                for cell in cliques_inside:
                    cell_int = pts_int[cell] + offset
                    ck = frozenset(tuple(p) for p in cell_int)
                    if ck in seen_cells:
                        continue
                    seen_cells.add(ck)
                    c_indices = [get_or_add(p) for p in cell_int]
                    cells_out.append(c_indices)

    grid_nodes = np.array(node_coords_int, dtype=np.float64) * (cell_size / 4.0)
    cells = np.array(cells_out, dtype=np.int64)

    # Trimming using face-centroid checks
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in TET_FACE_TRIPLETS:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    s_dists = safe_signed_distance(mesh, all_centroids)

    kept_cells = []
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        # safe_signed_distance: negative = inside, positive = outside
        if np.all(c_dists <= 1e-5):
            kept_cells.append(cell)

    surviving_cells = np.array(kept_cells)
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # Boundary faces culling
    face_counts = defaultdict(int)
    for cell in surviving_cells:
        for fv in TET_FACE_TRIPLETS:
            fkey = _coord_face_key(grid_nodes, cell, fv)
            face_counts[fkey] += 1
    boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]

    # Map face keys back to node indices
    # Each face key is a frozenset of point rounded keys. Let's find node indices matching these.
    boundary_node_ids = set()
    for bf in boundary_faces:
        # Resolve face points from frozenset of rounded coords
        for cell in surviving_cells:
            for fv in TET_FACE_TRIPLETS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    for n in cell[fv]:
                        boundary_node_ids.add(n)

    boundary_node_ids = np.array(list(boundary_node_ids), dtype=np.int64)

    # Snap boundary nodes to CAD surface
    grid_nodes_conformed = grid_nodes.copy()
    if len(boundary_node_ids) > 0:
        boundary_coords = grid_nodes[boundary_node_ids]
        closest_pts, _ = project_to_cad_surface(boundary_coords, mesh)
        for i, idx in enumerate(boundary_node_ids):
            grid_nodes_conformed[idx] = closest_pts[i]

    # Convert boundary faces back to node IDs
    surface_faces_list = []
    for bf in boundary_faces:
        face_nodes = []
        for cell in surviving_cells:
            for fv in TET_FACE_TRIPLETS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    face_nodes = list(cell[fv])
                    break
            if face_nodes:
                break
        surface_faces_list.append(face_nodes)

    surface_faces = np.array(surface_faces_list, dtype=np.int64)

    return ScaffoldResult(
        nodes=grid_nodes_conformed,
        elements=surviving_cells,
        surface_faces=surface_faces,
        element_order=1
    )


def generate_conformed_hex_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    **kwargs
) -> ScaffoldResult:
    """
    GMSH-free conformed background grid generator for hexahedral meshes (SC).
    Conforms the grid nodes to the boundary mesh and culls elements.
    """
    cell_size = target_element_size
    min_bound, max_bound = mesh.bounds

    # Align grid indices directly to origin-based multiples of cell_size
    min_ix = int(np.floor(min_bound[0] / cell_size))
    max_ix = int(np.ceil(max_bound[0] / cell_size))
    min_iy = int(np.floor(min_bound[1] / cell_size))
    max_iy = int(np.ceil(max_bound[1] / cell_size))
    min_iz = int(np.floor(min_bound[2] / cell_size))
    max_iz = int(np.ceil(max_bound[2] / cell_size))

    # Grid Setup (SC)
    node_coords_int = []
    node_coord_to_idx = {}

    def get_or_add(coord_int):
        key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
        if key not in node_coord_to_idx:
            node_coord_to_idx[key] = len(node_coords_int)
            node_coords_int.append(coord_int)
        return node_coord_to_idx[key]

    cells_out = []
    for ix in range(min_ix, max_ix + 1):
        for iy in range(min_iy, max_iy + 1):
            for iz in range(min_iz, max_iz + 1):
                voxel_corners = np.array([
                    [ix, iy, iz],
                    [ix + 1, iy, iz],
                    [ix + 1, iy + 1, iz],
                    [ix, iy + 1, iz],
                    [ix, iy, iz + 1],
                    [ix + 1, iy, iz + 1],
                    [ix + 1, iy + 1, iz + 1],
                    [ix, iy + 1, iz + 1]
                ], dtype=np.int64)
                c_indices = [get_or_add(p) for p in voxel_corners]
                cells_out.append(c_indices)

    grid_nodes = np.array(node_coords_int, dtype=np.float64) * cell_size
    cells = np.array(cells_out, dtype=np.int64)

    # Trimming using face-centroid checks
    all_centroids = []
    cell_to_centroids_indices = []
    for cell in cells:
        indices = []
        for fv in HEX_FACE_QUADS:
            verts = cell[list(fv)]
            verts_coords = grid_nodes[verts]
            centroid = verts_coords.mean(axis=0)
            indices.append(len(all_centroids))
            all_centroids.append(centroid)
        cell_to_centroids_indices.append(indices)

    all_centroids = np.array(all_centroids)
    s_dists = safe_signed_distance(mesh, all_centroids)

    kept_cells = []
    for i, cell in enumerate(cells):
        indices = cell_to_centroids_indices[i]
        c_dists = s_dists[indices]
        if np.all(c_dists <= 1e-5):
            kept_cells.append(cell)

    surviving_cells = np.array(kept_cells)
    if len(surviving_cells) == 0:
        raise ValueError("No cells survived trimming!")

    # Boundary faces culling
    face_counts = defaultdict(int)
    for cell in surviving_cells:
        for fv in HEX_FACE_QUADS:
            fkey = _coord_face_key(grid_nodes, cell, fv)
            face_counts[fkey] += 1
    boundary_faces = [fkey for fkey, count in face_counts.items() if count == 1]

    # Map face keys back to node indices
    boundary_node_ids = set()
    for bf in boundary_faces:
        for cell in surviving_cells:
            for fv in HEX_FACE_QUADS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    for n in cell[fv]:
                        boundary_node_ids.add(n)

    boundary_node_ids = np.array(list(boundary_node_ids), dtype=np.int64)

    # Snap boundary nodes to CAD surface
    grid_nodes_conformed = grid_nodes.copy()
    if len(boundary_node_ids) > 0:
        boundary_coords = grid_nodes[boundary_node_ids]
        closest_pts, _ = project_to_cad_surface(boundary_coords, mesh)
        for i, idx in enumerate(boundary_node_ids):
            grid_nodes_conformed[idx] = closest_pts[i]

    # Convert boundary faces back to node IDs
    surface_faces_list = []
    for bf in boundary_faces:
        face_nodes = []
        for cell in surviving_cells:
            for fv in HEX_FACE_QUADS:
                fkey = _coord_face_key(grid_nodes, cell, fv)
                if fkey == bf:
                    face_nodes = list(cell[fv])
                    break
            if face_nodes:
                break
        surface_faces_list.append(face_nodes)

    surface_faces = np.array(surface_faces_list, dtype=np.int64)

    return ScaffoldResult(
        nodes=grid_nodes_conformed,
        elements=surviving_cells,
        surface_faces=surface_faces,
        element_order=1
    )


# Backwards compatibility wrappers
def generate_a15_conformal_lattice(*args, **kwargs) -> dict[str, any]:
    """Wrapper to maintain compatibility with conformed A15 tests."""
    return generate_conformal_lattice(*args, **kwargs, lattice_type="A15")
