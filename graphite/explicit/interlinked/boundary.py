"""
Graphite Explicit Interlinked — Boundary Management Module (Phase 3)

Provides a 3-tier boundary management architecture for interlinked lattices:
1. Policy A: Particle SDF Inset Culling (cull_particles_by_sdf, cull_particles_by_mesh)
   - Guarantees 100% whole, uncut particles inside the CAD boundary envelope.
2. Policy B: Dedicated Boundary Anchors (identify_boundary_particles, tag_boundary_anchors)
   - Identifies particles on the boundary that lack full catenation coordination.
3. Policy C: Solid Perimeter Frame Welding (build_perimeter_frame_solid, fuse_perimeter_frame)
   - Generates exterior solid CAD frames (box border, cylindrical end collars)
     whose inner surfaces penetrate perimeter struts, locking them for tensile
     testing and recoater blade stability in AM.
"""

from __future__ import annotations

from typing import Callable, Sequence, Any
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.spatial import cKDTree

from .particle import InterlinkedParticle
from .cell import InterlinkedCell
from graphite.explicit.geometry_module import _manifold_to_trimesh


# =============================================================================
# 1. Policy A: Particle SDF Inset Culling
# =============================================================================

def cull_particles_by_sdf(
    particles: Sequence[InterlinkedParticle],
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    margin: float = 0.5,
    strict_nodes: bool = True,
) -> tuple[list[InterlinkedParticle], list[InterlinkedParticle]]:
    """
    Apply Inset Culling to candidate InterlinkedParticle instances against a Signed Distance Function.

    Convention:
        sdf_fn(pts) <= 0 inside the solid boundary, > 0 outside.

    Culling Rule:
        - If strict_nodes is True (default): A particle survives if and only if ALL of its
          global nodes satisfy `SDF(v) <= -margin`.
        - If strict_nodes is False: Uses conservative bounding sphere:
          `SDF(center) <= -(bounding_radius + margin)`.

    Ensures zero cut, broken, or clipped struts, preventing catenation destruction.

    Args:
        particles: Sequence of InterlinkedParticle instances.
        sdf_fn: Callable taking (N, 3) coordinates and returning (N,) signed distances.
        margin: Additional inset buffer clearance in mm.
        strict_nodes: If True, tests all global nodes. If False, tests bounding sphere.

    Returns:
        tuple: (surviving_particles, culled_particles)
    """
    if not particles:
        return [], []

    m = float(margin)
    surviving: list[InterlinkedParticle] = []
    culled: list[InterlinkedParticle] = []

    if strict_nodes:
        for p in particles:
            gnodes = p.global_nodes()
            if len(gnodes) == 0:
                # Fallback to center
                d = float(sdf_fn(p.center.reshape(1, 3))[0])
                if d <= -(p.bounding_radius + m):
                    surviving.append(p)
                else:
                    culled.append(p)
                continue

            sdf_vals = np.asarray(sdf_fn(gnodes), dtype=np.float64).reshape(-1)
            # All nodes must be strictly inside the margin
            if np.all(sdf_vals <= -m):
                surviving.append(p)
            else:
                culled.append(p)
    else:
        centers = np.array([p.center for p in particles], dtype=np.float64)
        sdf_vals = np.asarray(sdf_fn(centers), dtype=np.float64).reshape(-1)
        for i, p in enumerate(particles):
            threshold = -(p.bounding_radius + m)
            if sdf_vals[i] <= threshold:
                surviving.append(p)
            else:
                culled.append(p)

    return surviving, culled


def cull_particles_by_mesh(
    particles: Sequence[InterlinkedParticle],
    boundary_mesh: trimesh.Trimesh,
    margin: float = 0.5,
    strict_nodes: bool = True,
) -> tuple[list[InterlinkedParticle], list[InterlinkedParticle]]:
    """
    Inset-cull candidate particles against a watertight boundary mesh.

    Uses `trimesh.proximity.signed_distance` converted to canonical negative-inside convention:
        SDF(x) = -mesh_signed_distance(x)
    """
    if not particles:
        return [], []

    def _mesh_sdf(pts: np.ndarray) -> np.ndarray:
        # trimesh signed_distance: positive inside, negative outside
        sd = trimesh.proximity.signed_distance(boundary_mesh, pts)
        return -np.asarray(sd, dtype=np.float64)

    return cull_particles_by_sdf(
        particles=particles,
        sdf_fn=_mesh_sdf,
        margin=margin,
        strict_nodes=strict_nodes,
    )


# =============================================================================
# 2. Policy B: Dedicated Boundary Anchors & Tagging
# =============================================================================

def identify_boundary_particles(
    particles: Sequence[InterlinkedParticle],
    cell: InterlinkedCell | None = None,
    neighbor_radius: float | None = None,
    expected_coordination: int | None = None,
) -> list[InterlinkedParticle]:
    """
    Identify and tag particles that reside on the boundary of an interlinked assembly.

    Identifies particles that lack a complete set of catenating neighbors.
    Boundary particles are tagged in-place with `p.metadata['is_boundary'] = True`
    and `p.metadata['missing_neighbors'] = [...]`.

    Args:
        particles: Sequence of InterlinkedParticle instances.
        cell: Optional InterlinkedCell containing neighbor_catenation_offsets or coordination_number.
        neighbor_radius: Radius to search for neighbor particles when using spatial KD-tree.
        expected_coordination: Expected number of neighboring particles.

    Returns:
        List of particles identified as boundary particles.
    """
    if not particles:
        return []

    boundary_particles: list[InterlinkedParticle] = []

    # Strategy 1: Grid index lookup (if cell_indices are populated)
    has_indices = all(len(p.cell_index) >= 2 for p in particles)
    has_offsets = cell is not None and len(getattr(cell, "neighbor_catenation_offsets", [])) > 0

    if has_indices and has_offsets and cell is not None:
        # Map (cell_index, sublattice_id) -> particle
        site_map: dict[tuple[tuple[int, ...], str], InterlinkedParticle] = {
            (p.cell_index, p.sublattice_id): p for p in particles
        }
        offsets = cell.neighbor_catenation_offsets

        for p in particles:
            missing = []
            idx = np.array(p.cell_index, dtype=np.int64)
            for off in offsets:
                n_idx = tuple((idx + np.array(off[:len(idx)], dtype=np.int64)).tolist())
                # Check all possible sublattices or any partner
                found = any((n_idx, sub) in site_map for sub in ["", "A", "B", p.sublattice_id])
                if not found:
                    missing.append(tuple(off))

            if len(missing) > 0:
                p.metadata["is_boundary"] = True
                p.metadata["missing_neighbors"] = missing
                boundary_particles.append(p)
            else:
                p.metadata["is_boundary"] = False

        return boundary_particles

    # Strategy 2: Spatial KD-tree coordination fallback
    centers = np.array([p.center for p in particles], dtype=np.float64)
    kdtree = cKDTree(centers)

    if neighbor_radius is None:
        # Estimate neighbor radius from 2 nearest neighbors
        if len(particles) > 1:
            dists, _ = kdtree.query(centers, k=2)
            median_d = float(np.median(dists[:, 1]))
            neighbor_radius = median_d * 1.5
        else:
            neighbor_radius = 10.0

    exp_coord = expected_coordination
    if exp_coord is None and cell is not None:
        exp_coord = getattr(cell, "coordination_number", None)

    # If expected_coordination is not specified, use max observed coordination
    neighbor_lists = kdtree.query_ball_point(centers, r=neighbor_radius)
    counts = [len(nl) - 1 for nl in neighbor_lists]  # exclude self
    if exp_coord is None:
        exp_coord = max(counts) if counts else 0

    for i, p in enumerate(particles):
        c_count = counts[i]
        if c_count < exp_coord:
            p.metadata["is_boundary"] = True
            p.metadata["neighbor_count"] = c_count
            p.metadata["expected_coordination"] = exp_coord
            boundary_particles.append(p)
        else:
            p.metadata["is_boundary"] = False
            p.metadata["neighbor_count"] = c_count

    return boundary_particles


# =============================================================================
# 3. Policy C: Solid Perimeter Frame Welding
# =============================================================================

def build_perimeter_frame_solid(
    particles: Sequence[InterlinkedParticle],
    wall_thickness: float = 3.0,
    margin: float = 0.5,
    z_padding: float = 1.0,
    frame_shape: str = "box",
    radius: float | None = None,
    height: float | None = None,
) -> trimesh.Trimesh:
    """
    Generate an exterior solid CAD frame whose inner surface penetrates/welds with
    perimeter particle struts to form a rigid handling/tensile border.

    Supported shapes:
        - "box": Rectangular frame bordering the lattice in the XY plane.
                 Inner cutout penetrates outermost particles by `margin`.
        - "cylinder": Top and bottom annular collar rings enclosing cylindrical wraps.

    Args:
        particles: Sequence of surviving InterlinkedParticle instances.
        wall_thickness: Thickness of the outer frame wall in mm.
        margin: Penetration depth (in mm) of the frame inner boundary into the perimeter particles.
        z_padding: Vertical extension beyond the particle z-bounds in mm.
        frame_shape: "box" or "cylinder".
        radius: Cylinder radius for "cylinder" frame (optional).
        height: Cylinder height for "cylinder" frame (optional).

    Returns:
        Watertight trimesh.Trimesh of the solid perimeter frame.
    """
    if not particles:
        raise ValueError("Cannot build perimeter frame for empty particle list")

    w = float(wall_thickness)
    m = float(margin)
    zp = float(z_padding)

    # Compute bounding envelope across all global nodes
    all_nodes = [p.global_nodes() for p in particles if len(p.global_nodes()) > 0]
    if all_nodes:
        stacked = np.vstack(all_nodes)
        min_bound = np.min(stacked, axis=0)
        max_bound = np.max(stacked, axis=0)
    else:
        centers = np.array([p.center for p in particles], dtype=np.float64)
        min_bound = np.min(centers, axis=0)
        max_bound = np.max(centers, axis=0)

    if frame_shape == "box":
        xmin, ymin, zmin = min_bound
        xmax, ymax, zmax = max_bound

        # Outer dimensions
        dx_outer = (xmax - xmin) + 2.0 * w
        dy_outer = (ymax - ymin) + 2.0 * w
        dz_frame = (zmax - zmin) + 2.0 * zp

        outer_center = [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ]

        # Inner cutout dimensions:
        # Inner boundary begins inside the outer struts by margin:
        # dx_inner = (xmax - xmin) - 2.0 * m
        # dy_inner = (ymax - ymin) - 2.0 * m
        dx_inner = max(1.0, (xmax - xmin) - 2.0 * m)
        dy_inner = max(1.0, (ymax - ymin) - 2.0 * m)
        dz_cutout = dz_frame + 10.0  # Pass through entirely along Z

        m_outer = m3d.Manifold.cube(
            [dx_outer, dy_outer, dz_frame],
            center=True,
        ).translate(outer_center)

        m_inner = m3d.Manifold.cube(
            [dx_inner, dy_inner, dz_cutout],
            center=True,
        ).translate(outer_center)

        frame_solid = m_outer - m_inner
        return _manifold_to_trimesh(frame_solid)

    elif frame_shape == "cylinder":
        # Top and bottom annular collar rings for cylindrical wrap specimens
        zmin, zmax = min_bound[2], max_bound[2]
        r_nom = float(radius) if radius is not None else float(np.mean(np.linalg.norm(min_bound[:2])))
        collar_h = max(2.0, w)

        r_inner = max(0.5, r_nom - m)
        r_outer = r_nom + w

        # Bottom collar
        bot_z = zmin - zp
        c_bot_outer = m3d.Manifold.cylinder(collar_h, r_outer, r_outer, circular_segments=64)
        c_bot_inner = m3d.Manifold.cylinder(collar_h + 2.0, r_inner, r_inner, circular_segments=64).translate([0, 0, -1])
        collar_bottom = (c_bot_outer - c_bot_inner).translate([0, 0, bot_z])

        # Top collar
        top_z = zmax - collar_h + m + zp
        c_top_outer = m3d.Manifold.cylinder(collar_h, r_outer, r_outer, circular_segments=64)
        c_top_inner = m3d.Manifold.cylinder(collar_h + 2.0, r_inner, r_inner, circular_segments=64).translate([0, 0, -1])
        collar_top = (c_top_outer - c_top_inner).translate([0, 0, top_z])

        frame_solid = collar_bottom + collar_top
        return _manifold_to_trimesh(frame_solid)

    else:
        raise ValueError(f"Unsupported frame_shape: {frame_shape}. Choose 'box' or 'cylinder'.")


def fuse_perimeter_frame(
    lattice_meshes: Sequence[trimesh.Trimesh] | trimesh.Trimesh,
    frame_mesh: trimesh.Trimesh,
) -> trimesh.Trimesh:
    """
    Concatenate lattice meshes and the perimeter frame mesh into a unified multi-body assembly.

    Args:
        lattice_meshes: Sequence of particle Trimesh solids or a single combined Trimesh.
        frame_mesh: Solid boundary frame Trimesh.

    Returns:
        Unified trimesh.Trimesh containing both the frame and the lattice bodies.
    """
    if isinstance(lattice_meshes, trimesh.Trimesh):
        all_meshes = [lattice_meshes, frame_mesh]
    else:
        all_meshes = list(lattice_meshes) + [frame_mesh]

    combined = trimesh.util.concatenate(all_meshes)
    return combined
