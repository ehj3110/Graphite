"""
Graphite Implicit Engine - Micropillar & Microfiber Forest Generator

This module synthesizes high-aspect-ratio cylindrical micropillars (hairs, cilia,
micro-posts) on 3D TPMS and scaffold surfaces. It samples surface anchor points
and outward normal vectors, generates exact geometric cylinders via manifold3d,
and executes high-speed CSG boolean unions to produce a single watertight solid.

Key Capabilities
----------------
1. Poisson-Disk Distribution: Blue-noise relaxation for strictly uniform spacing
   without random clumping or pairing.
2. 3D-Printability Filtering: Angle thresholding relative to the horizontal build plate
   to discard unsupported drooping hairs (e.g. <= 30 deg from horizontal).
3. Selective Surface Placement: Place hairs everywhere ("all"), on internal pore
   lumens only ("internal_only"), on outer boundary skins only ("outer_only"), or on
   user-selected primitive planar faces (+X, -X, +Y, -Y, +Z, -Z, sides, top, bottom).
4. Boundary Normal Clamping: Eliminates 45-degree corner tilt along sharp CAD boundary cuts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import manifold3d
import numpy as np
from scipy.spatial import cKDTree
import trimesh

from graphite.explicit.geometry_module import (
    manifold_cylinder_between,
    manifold_to_trimesh,
    trimesh_to_manifold,
)


@dataclass(frozen=True)
class MicropillarConfig:
    """
    Configuration parameters for explicit micropillar / hair forest synthesis.

    Attributes
    ----------
    diameter_mm : float
        Pillar diameter in millimeters. Defaults to 0.050 mm (50 um).
    height_mm : float
        Pillar length / height in millimeters. Defaults to 0.200 mm (200 um).
    spacing_mm : float
        Target center-to-center spacing between pillars in millimeters.
        Defaults to 0.200 mm (200 um).
    circular_segments : int
        Number of polygon sides for cylinder tessellation. Defaults to 8.
    embed_depth_mm : float
        Depth the pillar base penetrates into the underlying surface to ensure
        a robust, non-tangent boolean union. Defaults to 0.010 mm (10 um).
    distribution : str
        Spatial sampling mode: 'poisson_disk' (even spacing, no clumping) or
        'random' (uniform area white noise). Defaults to 'poisson_disk'.
    min_spacing_mm : float | None
        Minimum edge-to-edge or center-to-center clearance between any two hairs.
        If None and distribution='poisson_disk', defaults to 0.85 * spacing_mm.
    filter_printable : bool
        If True, discards pillars oriented at angles likely to fail 3D printing
        overhang limits. Defaults to False.
    max_angle_from_horizontal_deg : float
        When filter_printable is True, keeps only pillars oriented within this
        many degrees from the horizontal XY plane (default 30.0 deg).
    location : str
        Surface region placement: 'all' (entire surface), 'internal_only' (inside
        pore channels only), or 'outer_only' (exterior envelope only). Defaults to 'all'.
    selected_faces : tuple[str, ...] | list[str] | None
        Selective primitive faces to populate with hairs (e.g. ('+z', '-z'), ('+x',),
        ('top', 'bottom'), ('sides',)). Overrides location if specified.
    boundary_type : str
        Primitive envelope geometry: 'box' (cube), 'cylinder', 'sphere', or 'auto'.
        Defaults to 'box'.
    boundary_bounds : tuple[float, float, float, float, float, float] | None
        Explicit (xmin, xmax, ymin, ymax, zmin, zmax) bounding box. If None,
        inferred from mesh extents.
    clamp_boundary_normals : bool
        If True, clamps normal vectors on planar boundary cuts to the cardinal
        axis, eliminating 45-degree edge tilt along sharp box cuts. Defaults to True.
    max_pillars : int
        Safety threshold to prevent generating excessive geometry. Defaults to 50,000.
    seed : int
        Random seed for surface sampling reproducibility. Defaults to 42.
    """
    diameter_mm: float = 0.050
    height_mm: float = 0.200
    spacing_mm: float = 0.200
    circular_segments: int = 8
    embed_depth_mm: float = 0.010
    distribution: str = "poisson_disk"
    min_spacing_mm: float | None = None
    filter_printable: bool = False
    min_angle_from_horizontal_deg: float = 60.0
    max_angle_from_horizontal_deg: float = 90.0
    location: str = "all"
    selected_faces: tuple[str, ...] | list[str] | None = None
    boundary_type: str = "box"
    boundary_bounds: tuple[float, float, float, float, float, float] | None = None
    clamp_boundary_normals: bool = True
    max_pillars: int = 50_000
    seed: int = 42
    orientation: str = "local_normal"
    boundary_mesh: trimesh.Trimesh | None = None
    max_boundary_dist_mm: float = 0.250


def _normalize_face_name(name: str) -> str:
    """Normalize face names and synonyms to standard canonical identifiers."""
    n = name.strip().lower().replace(" ", "_")
    mapping = {
        "+z": "+z", "z+": "+z", "top": "+z", "up": "+z", "upper": "+z",
        "-z": "-z", "z-": "-z", "bottom": "-z", "down": "-z", "lower": "-z",
        "+x": "+x", "x+": "+x", "right": "+x", "east": "+x",
        "-x": "-x", "x-": "-x", "left": "-x", "west": "-x",
        "+y": "+y", "y+": "+y", "front": "+y", "north": "+y",
        "-y": "-y", "y-": "-y", "back": "-y", "south": "-y",
        "sides": "sides", "side": "sides", "curved": "sides", "circumference": "sides",
        "outer": "outer", "outside": "outer", "external": "outer",
        "internal": "internal", "inside": "internal", "pores": "internal",
    }
    return mapping.get(n, n)


def segment_cad_boundary(mesh: trimesh.Trimesh, crease_angle_deg: float = 35.0) -> dict[str, np.ndarray]:
    """
    Computes face adjacency angles on mesh.
    Splits mesh into connected components bounded by dihedral angles >= radians(crease_angle_deg).
    Labels components as "top" (mean nz > 0.4), "bottom" (mean nz < -0.4), or "sides" (mean nz ~= 0).
    For any unassigned or border faces, fall back to nz > 0.35 ("top"), nz < -0.35 ("bottom"), else "sides".
    """
    import networkx as nx

    threshold = np.radians(crease_angle_deg)
    # trimesh.graph.face_adjacency_angles is only available for edges that are in face_adjacency
    adjacency = mesh.face_adjacency
    angles = mesh.face_adjacency_angles
    
    smooth_mask = angles < threshold
    smooth_adjacency = adjacency[smooth_mask]
    
    g = nx.Graph()
    g.add_nodes_from(range(len(mesh.faces)))
    g.add_edges_from(smooth_adjacency)
    
    components = list(nx.connected_components(g))
    
    labels = np.full(len(mesh.faces), "", dtype=object)
    
    for comp in components:
        comp_idx = list(comp)
        mean_nz = np.mean(mesh.face_normals[comp_idx, 2])
        if mean_nz > 0.4:
            labels[comp_idx] = "top"
        elif mean_nz < -0.4:
            labels[comp_idx] = "bottom"
        else:
            labels[comp_idx] = "sides"
            
    # Fallback for unassigned
    unassigned = labels == ""
    if np.any(unassigned):
        nz = mesh.face_normals[unassigned, 2]
        new_labels = np.where(nz > 0.35, "top", np.where(nz < -0.35, "bottom", "sides"))
        labels[unassigned] = new_labels
        
    return {
        "top": labels == "top",
        "bottom": labels == "bottom",
        "sides": labels == "sides"
    }


def _sample_poisson_disk(
    mesh: trimesh.Trimesh,
    min_dist_mm: float,
    target_count: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample surface points with guaranteed minimum surface clearance (Poisson-disk).
    Uses KDTree spatial exclusion with surface-normal compatibility checks so that
    pillars on opposite sides of thin walls do not erroneously cancel each other out.
    """
    # Sample 4x candidates to ensure full coverage under dense packing
    oversample = max(target_count * 4, 2000)
    candidates, face_indices = trimesh.sample.sample_surface(mesh, count=oversample, seed=seed)

    # Compute candidate normals for surface-side compatibility check
    tri_vert_norms = mesh.vertex_normals[mesh.faces[face_indices]]
    cand_norms = np.mean(tri_vert_norms, axis=1)
    cand_norm_lens = np.linalg.norm(cand_norms, axis=1, keepdims=True)
    cand_norm_lens = np.where(cand_norm_lens < 1e-12, 1.0, cand_norm_lens)
    cand_norms = cand_norms / cand_norm_lens

    tree = cKDTree(candidates)
    active = np.ones(len(candidates), dtype=bool)
    kept_indices = []

    for i in range(len(candidates)):
        if not active[i]:
            continue
        kept_indices.append(i)
        if len(kept_indices) >= target_count:
            break
        # Query neighbors within clearance radius
        nearby = tree.query_ball_point(candidates[i], r=min_dist_mm)
        if len(nearby) > 1:
            # Check normal alignment: only deactivate neighbors on the SAME or contiguous surface!
            # If two points are on opposite sides of a thin sheet wall, their normals point in
            # opposite directions (dot < -0.3), so they DO NOT deactivate each other.
            dots = np.sum(cand_norms[nearby] * cand_norms[i], axis=1)
            same_surface = np.array(nearby)[dots >= -0.3]
            active[same_surface] = False

    kept_indices = np.array(kept_indices, dtype=np.int64)
    return candidates[kept_indices], face_indices[kept_indices]


def _tangential_relaxation(
    pts: np.ndarray,
    norms: np.ndarray,
    spacing_mm: float,
    iterations: int = 6,
) -> np.ndarray:
    """
    Relax surface points along local tangent planes to achieve near-optimal
    hexagonal centroidal spacing across 3D surfaces.
    """
    if len(pts) <= 3:
        return pts

    pts_curr = pts.copy()
    r_repel = 1.20 * spacing_mm

    for _ in range(iterations):
        tree = cKDTree(pts_curr)
        k_query = min(7, len(pts_curr))
        dists, indices = tree.query(pts_curr, k=k_query)
        forces = np.zeros_like(pts_curr)
        for k in range(1, indices.shape[1]):
            neighbor_idx = indices[:, k]
            disp = pts_curr - pts_curr[neighbor_idx]
            d = dists[:, k]
            d = np.where(d < 1e-6, 1e-6, d)

            # Only repel neighbors on the same side of the wall
            dots = np.sum(norms * norms[neighbor_idx], axis=1)
            weight = np.where(
                (dots > 0.3) & (d < r_repel),
                (r_repel - d) / r_repel,
                0.0,
            )
            forces += (disp / d[:, None]) * weight[:, None]

        # Tangent plane projection with bounded displacement
        forces_tan = forces - np.sum(forces * norms, axis=1, keepdims=True) * norms
        norm_tan = np.linalg.norm(forces_tan, axis=1, keepdims=True)
        step = 0.05 * spacing_mm * forces_tan / np.maximum(norm_tan, 1.0)
        pts_curr += step

    return pts_curr


def sample_pillar_anchors(
    mesh: trimesh.Trimesh,
    config: MicropillarConfig,
    boundary_mesh: trimesh.Trimesh | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample surface anchor points and outward surface normals for micropillars.
    Supports Poisson-disk (even) spacing, printability filtering, and selective
    surface/primitive-face masking.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input scaffold mesh.
    config : MicropillarConfig
        Pillar configuration settings.
    boundary_mesh : trimesh.Trimesh | None
        Optional explicit CAD boundary mesh to align and constrain pillars.

    Returns
    -------
    points : ndarray, shape (N, 3)
        3D anchor coordinates on the mesh surface.
    normals : ndarray, shape (N, 3)
        Outward unit normal vectors at each anchor point.
    """
    if config.spacing_mm <= 0:
        raise ValueError(f"spacing_mm must be > 0, got {config.spacing_mm}")

    work_mesh = mesh
    try:
        if work_mesh.is_watertight and float(work_mesh.volume) < 0.0:
            work_mesh = work_mesh.copy()
            work_mesh.invert()
    except Exception:
        pass

    total_area = float(work_mesh.area)
    target_count = int(np.round(total_area / (config.spacing_mm ** 2)))
    target_count = max(1, min(target_count, int(config.max_pillars)))

    dist_mode = config.distribution.strip().lower()
    if dist_mode in ("poisson", "poisson_disk", "blue_noise", "relaxed", "even", "uniform", "hexagonal"):
        min_dist = float(config.min_spacing_mm if config.min_spacing_mm is not None else 0.85 * config.spacing_mm)
        pts, face_indices = _sample_poisson_disk(work_mesh, min_dist_mm=min_dist, target_count=target_count, seed=config.seed)
    else:
        pts, face_indices = trimesh.sample.sample_surface(work_mesh, count=target_count, seed=config.seed)

    if len(pts) == 0:
        return pts, np.empty((0, 3), dtype=np.float64)

    # Use smoothed vertex normals across sampled triangles to ensure cylinders project straight out
    tri_vert_norms = work_mesh.vertex_normals[work_mesh.faces[face_indices]]
    norms = np.mean(tri_vert_norms, axis=1)

    # Normalize normals
    norm_lens = np.linalg.norm(norms, axis=1, keepdims=True)
    norm_lens = np.where(norm_lens < 1e-12, 1.0, norm_lens)
    norms = norms / norm_lens

    # If relaxed / even mode requested, perform tangential relaxation to make spacing as uniform as possible
    if dist_mode in ("relaxed", "even", "uniform", "hexagonal"):
        pts = _tangential_relaxation(pts, norms, spacing_mm=config.spacing_mm, iterations=6)

    b_mesh = boundary_mesh if boundary_mesh is not None else config.boundary_mesh
    if b_mesh is not None:
        masks = segment_cad_boundary(b_mesh)
        closest_pts, dists, face_ids = trimesh.proximity.closest_point(b_mesh, pts)
        
        keep = np.ones(len(pts), dtype=bool)
        if config.selected_faces:
            norm_faces = [_normalize_face_name(f) for f in config.selected_faces]
            keep.fill(False)
            
            is_top_face = masks["top"][face_ids]
            is_bottom_face = masks["bottom"][face_ids]
            is_side_face = masks["sides"][face_ids]
            valid_dist = dists <= config.max_boundary_dist_mm
            
            if "top" in norm_faces or "+z" in norm_faces:
                keep |= (is_top_face & valid_dist)
            if "bottom" in norm_faces or "-z" in norm_faces:
                keep |= (is_bottom_face & valid_dist)
            
            keep &= ~is_side_face
        
        pts = pts[keep]
        norms = norms[keep]
        face_ids = face_ids[keep]
        
        if config.orientation == "z_aligned":
            new_norms = np.zeros_like(norms)
            is_top = masks["top"][face_ids]
            is_bot = masks["bottom"][face_ids]
            new_norms[is_top] = [0.0, 0.0, 1.0]
            new_norms[is_bot] = [0.0, 0.0, -1.0]
            # fallback for anything else? retain normal or set 0? 
            # normally there wouldn't be anything else if keep &= ~is_side_face and we selected top/bottom
            norms = new_norms
        elif config.orientation == "cad_normal":
            norms = b_mesh.face_normals[face_ids]
            
        if len(pts) == 0:
            return pts, norms
            
        if config.filter_printable:
            abs_nz = np.clip(np.abs(norms[:, 2]), 0.0, 1.0)
            angles_deg = np.rad2deg(np.arcsin(abs_nz))
            keep_mask = (angles_deg >= float(config.min_angle_from_horizontal_deg) - 1e-3) & (
                angles_deg <= float(config.max_angle_from_horizontal_deg) + 1e-3
            )
            pts = pts[keep_mask]
            norms = norms[keep_mask]
            
        return pts, norms

    # Determine boundary bounds for primitive face classification
    if config.boundary_bounds is not None:
        min_b = np.array(config.boundary_bounds[:3], dtype=np.float64)
        max_b = np.array(config.boundary_bounds[3:], dtype=np.float64)
    else:
        mb = work_mesh.bounds
        min_b, max_b = mb[0], mb[1]

    tol_mm = max(0.020, 0.03 * float(np.min(max_b - min_b)))

    # Primitive boundary face masks (Box / Cylinder / Sphere)
    b_type = config.boundary_type.strip().lower()
    face_masks: dict[str, np.ndarray] = {}

    if b_type in ("box", "cube", "auto"):
        face_masks["+x"] = (pts[:, 0] >= max_b[0] - tol_mm) & (norms[:, 0] > 0.5)
        face_masks["-x"] = (pts[:, 0] <= min_b[0] + tol_mm) & (norms[:, 0] < -0.5)
        face_masks["+y"] = (pts[:, 1] >= max_b[1] - tol_mm) & (norms[:, 1] > 0.5)
        face_masks["-y"] = (pts[:, 1] <= min_b[1] + tol_mm) & (norms[:, 1] < -0.5)
        face_masks["+z"] = (pts[:, 2] >= max_b[2] - tol_mm) & (norms[:, 2] > 0.5)
        face_masks["-z"] = (pts[:, 2] <= min_b[2] + tol_mm) & (norms[:, 2] < -0.5)
        outer_mask = (
            face_masks["+x"] | face_masks["-x"] |
            face_masks["+y"] | face_masks["-y"] |
            face_masks["+z"] | face_masks["-z"]
        )
    elif b_type in ("cylinder", "cyl"):
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        radius_est = max(abs(max_b[0]), abs(min_b[0]), abs(max_b[1]), abs(min_b[1]))
        face_masks["+z"] = (pts[:, 2] >= max_b[2] - tol_mm) & (norms[:, 2] > 0.5)
        face_masks["-z"] = (pts[:, 2] <= min_b[2] + tol_mm) & (norms[:, 2] < -0.5)
        is_cap = face_masks["+z"] | face_masks["-z"]
        rad_dot = (pts[:, 0] * norms[:, 0] + pts[:, 1] * norms[:, 1]) / np.maximum(r, 1e-12)
        face_masks["sides"] = (r >= radius_est - tol_mm) & (rad_dot > 0.5) & (~is_cap)
        outer_mask = face_masks["+z"] | face_masks["-z"] | face_masks["sides"]
    elif b_type in ("sphere", "spherical"):
        center = (min_b + max_b) / 2.0
        r_sph = np.linalg.norm(pts - center, axis=1)
        r_max = np.max(r_sph)
        face_masks["outer"] = (r_sph >= r_max - tol_mm)
        outer_mask = face_masks["outer"]
    else:
        outer_mask = np.zeros(len(pts), dtype=bool)

    face_masks["outer"] = outer_mask
    face_masks["internal"] = ~outer_mask

    # Clamp boundary normals to eliminate 45-degree edge tilt on planar CAD boundary cuts
    if config.clamp_boundary_normals:
        if b_type in ("box", "cube", "auto"):
            norms[face_masks["+x"]] = np.array([1.0, 0.0, 0.0])
            norms[face_masks["-x"]] = np.array([-1.0, 0.0, 0.0])
            norms[face_masks["+y"]] = np.array([0.0, 1.0, 0.0])
            norms[face_masks["-y"]] = np.array([0.0, -1.0, 0.0])
            norms[face_masks["+z"]] = np.array([0.0, 0.0, 1.0])
            norms[face_masks["-z"]] = np.array([0.0, 0.0, -1.0])
        elif b_type in ("cylinder", "cyl"):
            side_pts = pts[face_masks["sides"]]
            if len(side_pts) > 0:
                side_r = np.sqrt(side_pts[:, 0]**2 + side_pts[:, 1]**2)
                rad_norms = np.zeros_like(side_pts)
                rad_norms[:, 0] = side_pts[:, 0] / np.maximum(side_r, 1e-12)
                rad_norms[:, 1] = side_pts[:, 1] / np.maximum(side_r, 1e-12)
                norms[face_masks["sides"]] = rad_norms
            norms[face_masks["+z"]] = np.array([0.0, 0.0, 1.0])
            norms[face_masks["-z"]] = np.array([0.0, 0.0, -1.0])

    # Filter by selected faces or location
    if config.selected_faces:
        norm_faces = [_normalize_face_name(f) for f in config.selected_faces]
        combined_mask = np.zeros(len(pts), dtype=bool)
        for nf in norm_faces:
            if nf in face_masks:
                combined_mask |= face_masks[nf]
        pts = pts[combined_mask]
        norms = norms[combined_mask]
    else:
        loc = config.location.strip().lower()
        if loc in ("outer", "outer_only", "exterior"):
            pts = pts[outer_mask]
            norms = norms[outer_mask]
        elif loc in ("internal", "internal_only", "pores", "interior"):
            internal_mask = ~outer_mask
            pts = pts[internal_mask]
            norms = norms[internal_mask]

    if len(pts) == 0:
        return pts, norms

    # 3D-Printability Overhang Filtering: keep only hairs within [min_angle, max_angle] from horizontal
    if config.filter_printable:
        abs_nz = np.clip(np.abs(norms[:, 2]), 0.0, 1.0)
        angles_deg = np.rad2deg(np.arcsin(abs_nz))
        keep_mask = (angles_deg >= float(config.min_angle_from_horizontal_deg) - 1e-3) & (
            angles_deg <= float(config.max_angle_from_horizontal_deg) + 1e-3
        )
        pts = pts[keep_mask]
        norms = norms[keep_mask]

    return pts, norms


def generate_micropillars(
    base_mesh: trimesh.Trimesh,
    config: MicropillarConfig | None = None,
    *,
    boundary_mesh: trimesh.Trimesh | None = None,
    return_separate_forest: bool = False,
) -> trimesh.Trimesh:
    """
    Synthesize an explicit micropillar forest onto a scaffold surface.

    Parameters
    ----------
    base_mesh : trimesh.Trimesh
        The base (untextured) scaffold mesh.
    config : MicropillarConfig, optional
        Micropillar configuration parameters. If None, defaults to 50 um diameter,
        200 um height, 200 um spacing, Poisson-disk distribution.
    boundary_mesh : trimesh.Trimesh | None, optional
        Optional CAD boundary mesh to align and constrain pillars.
    return_separate_forest : bool, optional
        If True, returns only the pillar forest mesh without boolean-unioning
        onto the base mesh. Defaults to False.

    Returns
    -------
    trimesh.Trimesh
        The combined watertight scaffold with micropillars.
    """
    if config is None:
        config = MicropillarConfig()

    t0 = time.perf_counter()

    # Step 1: Sample anchor points and surface normals
    pts, norms = sample_pillar_anchors(base_mesh, config, boundary_mesh=boundary_mesh)
    if len(pts) == 0:
        return base_mesh.copy()

    radius = float(config.diameter_mm) / 2.0
    embed = float(config.embed_depth_mm)
    height = float(config.height_mm)

    # Step 2: Build cylinder manifolds
    cylinders: list[manifold3d.Manifold] = []
    for p, n in zip(pts, norms):
        p_start = p - n * embed
        p_end = p + n * height
        cyl = manifold_cylinder_between(
            p_start, p_end, radius=radius, circular_segments=config.circular_segments
        )
        if cyl is not None:
            cylinders.append(cyl)

    if not cylinders:
        return base_mesh.copy()

    # Step 3: Batch compose pillar forest
    t1 = time.perf_counter()
    pillar_forest = manifold3d.Manifold.compose(cylinders)
    t_compose = time.perf_counter() - t1

    if return_separate_forest:
        return manifold_to_trimesh(pillar_forest)

    # Step 4: Union with base scaffold mesh
    t2 = time.perf_counter()
    base_man = trimesh_to_manifold(base_mesh)
    combined = base_man + pillar_forest
    t_union = time.perf_counter() - t2

    out_mesh = manifold_to_trimesh(combined)
    trimesh.repair.fix_normals(out_mesh)

    total_time = time.perf_counter() - t0
    loc_desc = f"faces={config.selected_faces}" if config.selected_faces else f"loc={config.location}"
    print(
        f"[Micropillars] Generated {len(cylinders):,} pillars ({config.distribution}, "
        f"d={config.diameter_mm*1e3:.0f}um, h={config.height_mm*1e3:.0f}um, {loc_desc}): "
        f"faces={len(out_mesh.faces):,}, watertight={out_mesh.is_watertight} in {total_time:.2f}s "
        f"(compose: {t_compose:.3f}s, union: {t_union:.3f}s)"
    )

    return out_mesh
