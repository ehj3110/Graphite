"""
Graphite Explicit Interlinked — Custom Cell STL Importer and Tesselator

Loads custom kinematic unit cell STLs (e.g. nasa_fabric_hexagon.stl),
decomposes them into sub-components (plate, ring, arms), applies decoupled
component scaling (enabling thickness grading on imported meshes), and
tessellates them across hexagonal or Cartesian lattices with Inset Culling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence
import numpy as np
import trimesh
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
    _rotation_align_local_z_to_unit,
    _affine_rows_from_R_t,
)
from .conformal import cull_rings_by_sdf


class ImportedInterlinkedCell:
    """
    Encapsulates an imported multi-body kinematic unit cell mesh.
    """

    def __init__(self, stl_path: str | Path):
        self.path = Path(stl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Cell STL not found: {self.path}")

        loaded = trimesh.load_mesh(str(self.path))
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            self.full_mesh = trimesh.util.concatenate(geoms)
        else:
            self.full_mesh = loaded

        self.bounds = np.asarray(self.full_mesh.bounds, dtype=np.float64)
        self.extents = np.asarray(self.full_mesh.extents, dtype=np.float64)
        self.centroid = np.asarray(self.full_mesh.centroid, dtype=np.float64)

        # Decompose into components
        bodies = self.full_mesh.split(only_watertight=False)
        self.body_meshes = bodies
        self.body_manifolds: list[m3d.Manifold] = []

        # Classify sub-bodies
        self.plate_idx: int | None = None
        self.ring_idx: int | None = None
        self.arm_indices: list[int] = []

        max_xy = max(self.extents[0], self.extents[1])

        for idx, b in enumerate(bodies):
            m = _trimesh_to_manifold(b)
            self.body_manifolds.append(m)
            b_ext = b.extents
            euler = int(b.euler_number)
            genus = max(0, (2 - euler) // 2) if b.is_watertight else 0

            if b_ext[2] < 0.25 * max_xy and genus == 0 and self.plate_idx is None:
                self.plate_idx = idx
            elif genus == 1 and self.ring_idx is None:
                self.ring_idx = idx
            else:
                self.arm_indices.append(idx)

        # Symmetry & estimated pitch
        if len(self.arm_indices) in (6, 12):
            self.symmetry_order = 6
            self.lattice_type = "hexagonal"
        elif len(self.arm_indices) in (4, 8):
            self.symmetry_order = 4
            self.lattice_type = "cartesian"
        else:
            self.symmetry_order = 1
            self.lattice_type = "cartesian"

        # Pitch estimate: distance between tile centers in a close-packed lattice
        if self.plate_idx is not None:
            plate_b = bodies[self.plate_idx]
            r_plate = max(plate_b.extents[:2]) / 2.0
            self.estimated_pitch = float(np.sqrt(3.0) * r_plate if self.lattice_type == "hexagonal" else 2.0 * r_plate)
        else:
            self.estimated_pitch = float(max(self.extents[:2]))

    def build_instance(
        self,
        scale_xy: float = 1.0,
        scale_z: float = 1.0,
        scale_arms: float = 1.0,
        scale_plate_t: float = 1.0,
    ) -> m3d.Manifold:
        """
        Build a scaled instance of the cell with decoupled component scaling.

        Args:
            scale_xy: Overall lateral scaling factor.
            scale_z: Overall vertical height scaling factor.
            scale_arms: Independent thickness/scale multiplier for interlocking arms.
            scale_plate_t: Independent thickness multiplier for the base plate.
        """
        parts: list[m3d.Manifold] = []

        for idx, m in enumerate(self.body_manifolds):
            if idx == self.plate_idx:
                # Plate: scaled laterally and along Z by plate thickness factor
                scaled = m.scale((scale_xy, scale_xy, scale_z * scale_plate_t))
            elif idx == self.ring_idx:
                # Ring: scaled with overall cell
                scaled = m.scale((scale_xy, scale_xy, scale_z * scale_arms))
            elif idx in self.arm_indices:
                # Arms: scaled with arm multiplier
                scaled = m.scale((scale_xy * scale_arms, scale_xy * scale_arms, scale_z * scale_arms))
            else:
                scaled = m.scale((scale_xy, scale_xy, scale_z))
            parts.append(scaled)

        return m3d.Manifold.compose(parts)


def generate_hexagonal_sheet_seeds(
    num_rows: int = 4,
    num_cols: int = 4,
    pitch: float = 12.75,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    center_origin: bool = True,
) -> np.ndarray:
    """
    Generate close-packed hexagonal sheet seeds with alternating row offsets.

    Parameters:
        num_rows: Number of rows along Y.
        num_cols: Number of columns along X.
        pitch: Center-to-center spacing d between adjacent tiles.
        origin: Base center coordinate (x0, y0, z0).
        center_origin: If True, centers the entire sheet around origin.

    Returns:
        (N, 3) seed coordinates where each row j along Y is offset in X by (j % 2) * (pitch / 2).
    """
    d = float(pitch)
    dy = d * (np.sqrt(3.0) / 2.0)
    pts = []
    for r in range(num_rows):
        x_offset = (r % 2) * (d / 2.0)
        y = r * dy
        for c in range(num_cols):
            x = c * d + x_offset
            pts.append([x, y, 0.0])

    pts = np.array(pts, dtype=np.float64)
    if center_origin and len(pts) > 0:
        center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        pts -= center

    pts[:, 0] += origin[0]
    pts[:, 1] += origin[1]
    pts[:, 2] += origin[2]
    return pts


def generate_hexagonal_grid_seeds(
    num_rings_radial: int = 1,
    pitch: float = 12.12,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """
    Generate close-packed triangular/hexagonal lattice seed coordinates in concentric rosettes.

    Parameters:
        num_rings_radial: Number of concentric rings (0 = 1 center tile; 1 = 7-tile rosette; 2 = 19-tile rosette).
        pitch: Center-to-center spacing d between adjacent tiles.
        origin: Cartesian center (x0, y0, z0).

    Returns:
        (N, 3) seed coordinates.
    """
    d = float(pitch)
    pts = []

    # Axial coordinate range: -N <= q, r, s <= N where q + r + s = 0
    N = int(num_rings_radial)
    for q in range(-N, N + 1):
        for r in range(max(-N, -q - N), min(N, -q + N) + 1):
            x = d * (q + r / 2.0)
            y = d * (np.sqrt(3.0) / 2.0 * r)
            pts.append([x + origin[0], y + origin[1], origin[2]])

    return np.array(pts, dtype=np.float64)


def tessellate_imported_cell(
    cell: ImportedInterlinkedCell,
    seed_points: np.ndarray,
    scale_xy_field: float | np.ndarray = 1.0,
    scale_arms_field: float | np.ndarray = 1.0,
    boundary_mesh: trimesh.Trimesh | None = None,
    cull_margin: float = 0.50,
) -> trimesh.Trimesh:
    """
    Tessellate an imported kinematic unit cell across an array of seed points.

    Args:
        cell: ImportedInterlinkedCell instance.
        seed_points: (N, 3) grid coordinates.
        scale_xy_field: Uniform float or (N,) array of lateral scaling factors.
        scale_arms_field: Uniform float or (N,) array of arm thickness multipliers.
        boundary_mesh: Optional CAD domain for Inset Culling.
        cull_margin: Inset safety margin in mm.

    Returns:
        Watertight multi-body trimesh.Trimesh.
    """
    n_pts = len(seed_points)
    s_xy = np.broadcast_to(np.asarray(scale_xy_field, dtype=np.float64), (n_pts,))
    s_arms = np.broadcast_to(np.asarray(scale_arms_field, dtype=np.float64), (n_pts,))

    # Inset culling if boundary_mesh is provided
    keep_indices = list(range(n_pts))
    if boundary_mesh is not None:
        proximity = trimesh.proximity.ProximityQuery(boundary_mesh)
        max_extent_r = float(max(cell.extents[:2]) / 2.0)
        keep = []
        for i in range(n_pts):
            pt = seed_points[i]
            r_eff = max_extent_r * s_xy[i]
            # trimesh signed_distance returns positive inside
            dist = float(proximity.signed_distance(pt[None, :])[0])
            if dist >= (r_eff + cull_margin):
                keep.append(i)
        keep_indices = keep

    if not keep_indices:
        raise ValueError("All candidate cell instances were culled by the boundary mesh.")

    # Build and translate instances
    instances: list[m3d.Manifold] = []
    for idx in keep_indices:
        pt = seed_points[idx]
        inst = cell.build_instance(
            scale_xy=float(s_xy[idx]),
            scale_z=1.0,
            scale_arms=float(s_arms[idx]),
        )
        translated = inst.translate((float(pt[0]), float(pt[1]), float(pt[2])))
        instances.append(translated)

    composed = m3d.Manifold.compose(instances)
    return _manifold_to_trimesh(composed)
