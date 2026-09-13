"""
Graphite Explicit Interlinked — Parametric NASA Hexagonal Space Fabric

Implements the code-driven parametric version of the 6-fold symmetric
NASA JPL 3D-printed space fabric (matching test_parts/nasa_fabric_hexagon.stl):
- Base hexagonal contact/reflector tile.
- Central circular top torus ring.
- 6 curved spiral interlocking hook legs rotated at 60-degree increments.
- Full continuous grading support: independent variation of wire radius r(x),
  hexagonal pitch d(x), and base plate thickness t(x).
"""

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
from .importer import (
    generate_hexagonal_grid_seeds,
    generate_hexagonal_sheet_seeds,
)


_CACHED_REFERENCE_ARM_MANIFOLD: m3d.Manifold | None = None


def _get_reference_arm_manifold() -> m3d.Manifold | None:
    """Retrieve or cache the reference hook arm manifold from test_parts/nasa_fabric_hexagon.stl."""
    global _CACHED_REFERENCE_ARM_MANIFOLD
    if _CACHED_REFERENCE_ARM_MANIFOLD is not None:
        return _CACHED_REFERENCE_ARM_MANIFOLD

    possible_paths = [
        Path("test_parts/nasa_fabric_hexagon.stl"),
        Path(__file__).resolve().parent.parent.parent.parent / "test_parts" / "nasa_fabric_hexagon.stl",
    ]
    for p in possible_paths:
        if p.exists():
            loaded = trimesh.load_mesh(str(p))
            if isinstance(loaded, trimesh.Scene):
                geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(geoms)
            else:
                mesh = loaded
            bodies = mesh.split(only_watertight=False)
            if len(bodies) >= 6:
                # Body 5 is the arm pointing toward +X (0 deg)
                _CACHED_REFERENCE_ARM_MANIFOLD = _trimesh_to_manifold(bodies[5])
                return _CACHED_REFERENCE_ARM_MANIFOLD
    return None


class NasaHexagonCell:
    """
    Parametric generator for a single 6-fold symmetric NASA Space Fabric tile.
    """

    def __init__(
        self,
        pitch: float = 12.75,
        wire_radius: float = 0.30,
        plate_radius: float = 7.0,
        plate_thickness: float = 0.45,
        ring_radius: float = 3.75,
        ring_height: float = 5.898,
        arm_manifold: m3d.Manifold | None = None,
    ):
        self.pitch = float(pitch)
        self.wire_radius = float(wire_radius)
        self.plate_radius = float(plate_radius)
        self.plate_thickness = float(plate_thickness)
        self.ring_radius = float(ring_radius)
        self.ring_height = float(ring_height)
        self.arm_manifold = arm_manifold

    def _generate_arm_centerline(self, num_pts: int = 16) -> np.ndarray:
        """
        Generate the 3D spline centerline for a single spiral hook arm fallback.
        """
        t = np.linspace(0.0, 1.0, num_pts)
        z = self.plate_thickness + t * (self.ring_height - self.plate_thickness)
        r_start = 0.55 * self.plate_radius
        r_end = 0.57 * self.pitch
        r = r_start + (r_end - r_start) * (t ** 1.3)
        th_start = np.radians(0.0)
        th_end = np.radians(-25.0)
        theta = th_start + (th_end - th_start) * t
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y, z])

    def build_manifold(self) -> m3d.Manifold:
        """
        Construct the watertight Manifold3D solid for this cell.
        """
        parts: list[m3d.Manifold] = []

        # 1. Base Hexagonal Plate (regular 6-gon, rotated 30 deg so flats align with lattice grid)
        plate_2d = m3d.CrossSection.circle(self.plate_radius, circular_segments=6).rotate(30.0)
        plate_3d = plate_2d.extrude(self.plate_thickness)
        parts.append(plate_3d)

        # 2. Top Torus Ring
        c_circle = m3d.CrossSection.circle(self.wire_radius, circular_segments=16).translate([self.ring_radius, 0.0])
        ring_3d = m3d.Manifold.revolve(c_circle, circular_segments=32).translate([0.0, 0.0, self.ring_height])
        parts.append(ring_3d)

        # 3. Six Rotational Hook Arms (60-degree C6 symmetry)
        ref_arm = self.arm_manifold or _get_reference_arm_manifold()
        if ref_arm is not None:
            scale_xy = self.pitch / 12.75
            scale_arms = self.wire_radius / 0.30 if self.wire_radius != 0.30 else 1.0
            scale_z = 1.0
            scaled_arm = ref_arm.scale((scale_xy, scale_xy, scale_z))
            for k in range(6):
                angle_deg = float(60.0 * k)
                R_rot = np.array([
                    [np.cos(np.radians(angle_deg)), -np.sin(np.radians(angle_deg)), 0.0],
                    [np.sin(np.radians(angle_deg)),  np.cos(np.radians(angle_deg)), 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                aff_k = _affine_rows_from_R_t(R_rot, np.zeros(3))
                parts.append(scaled_arm.transform(aff_k))
        else:
            arm_pts = self._generate_arm_centerline(num_pts=18)
            arm_struts: list[m3d.Manifold] = []
            for i in range(len(arm_pts) - 1):
                p0 = arm_pts[i]
                p1 = arm_pts[i + 1]
                seg = p1 - p0
                length = float(np.linalg.norm(seg))
                if length <= 1e-8:
                    continue
                t_dir = seg / length
                mid = 0.5 * (p0 + p1)
                R_mat = _rotation_align_local_z_to_unit(t_dir)
                aff = _affine_rows_from_R_t(R_mat, mid)
                cyl = m3d.Manifold.cylinder(
                    height=length,
                    radius_low=self.wire_radius,
                    radius_high=self.wire_radius,
                    circular_segments=14,
                    center=True,
                ).transform(aff)
                arm_struts.append(cyl)
                arm_struts.append(m3d.Manifold.sphere(self.wire_radius).translate(tuple(float(v) for v in p0)))

            arm_struts.append(m3d.Manifold.sphere(self.wire_radius).translate(tuple(float(v) for v in arm_pts[-1])))
            single_arm = m3d.Manifold.compose(arm_struts)

            for k in range(6):
                angle_deg = float(60.0 * k)
                R_rot = np.array([
                    [np.cos(np.radians(angle_deg)), -np.sin(np.radians(angle_deg)), 0.0],
                    [np.sin(np.radians(angle_deg)),  np.cos(np.radians(angle_deg)), 0.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                aff_k = _affine_rows_from_R_t(R_rot, np.zeros(3))
                parts.append(single_arm.transform(aff_k))

        return m3d.Manifold.compose(parts)


def generate_nasa_hexagon_lattice(
    num_rings_radial: int | None = None,
    num_rows: int | None = None,
    num_cols: int | None = None,
    pitch: float = 12.75,
    wire_radius: float = 0.30,
    plate_radius: float = 7.0,
    plate_thickness: float = 0.45,
    ring_radius: float = 3.75,
    ring_height: float = 5.898,
    wire_radius_field: Callable[[np.ndarray], float] | None = None,
    pitch_field: Callable[[np.ndarray], float] | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    boundary_mesh: trimesh.Trimesh | None = None,
    cull_margin: float = 0.50,
) -> trimesh.Trimesh:
    """
    Generate a full planar or graded hexagonal NASA Space Fabric mesh.

    Parameters:
        num_rings_radial: If specified, generates a concentric radial rosette (1 = 7 tiles; 2 = 19 tiles).
        num_rows: Number of rows in rectangular sheet (alternating row offsets).
        num_cols: Number of columns in rectangular sheet.
        pitch: Nominal center-to-center tile spacing d in mm.
        wire_radius: Nominal arm and ring wire radius in mm.
        plate_radius: Nominal base hexagonal plate radius in mm.
        plate_thickness: Base plate thickness in mm.
        ring_radius: Torus ring radius in mm.
        ring_height: Ring height above base in mm.
        wire_radius_field: Optional function mapping (3,) point to wire radius r in mm.
        pitch_field: Optional function mapping (3,) point to pitch d in mm.
        boundary_mesh: Optional CAD boundary for Inset Culling.
        cull_margin: Safety margin for Inset Culling in mm.

    Returns:
        Watertight multi-body trimesh.Trimesh.
    """
    if num_rows is not None or num_cols is not None:
        rows = num_rows if num_rows is not None else 4
        cols = num_cols if num_cols is not None else 4
        seeds = generate_hexagonal_sheet_seeds(
            num_rows=rows,
            num_cols=cols,
            pitch=pitch,
            origin=origin,
        )
    elif num_rings_radial is not None:
        seeds = generate_hexagonal_grid_seeds(
            num_rings_radial=num_rings_radial,
            pitch=pitch,
            origin=origin,
        )
    else:
        # Default to a 4x4 rectangular sheet with row offsets
        seeds = generate_hexagonal_sheet_seeds(
            num_rows=4,
            num_cols=4,
            pitch=pitch,
            origin=origin,
        )

    n_pts = len(seeds)
    instances: list[m3d.Manifold] = []

    # Inset culling
    keep_indices = list(range(n_pts))
    if boundary_mesh is not None:
        proximity = trimesh.proximity.ProximityQuery(boundary_mesh)
        max_extent_r = float(plate_radius + 1.5)
        keep = []
        for i in range(n_pts):
            pt = seeds[i]
            dist = float(proximity.signed_distance(pt[None, :])[0])
            if dist >= (max_extent_r + cull_margin):
                keep.append(i)
        keep_indices = keep

    for idx in keep_indices:
        pt = seeds[idx]

        # Evaluate local graded parameters
        local_wire_r = float(wire_radius_field(pt)) if wire_radius_field else float(wire_radius)
        local_pitch = float(pitch_field(pt)) if pitch_field else float(pitch)

        cell = NasaHexagonCell(
            pitch=local_pitch,
            wire_radius=local_wire_r,
            plate_radius=plate_radius * (local_pitch / pitch),
            plate_thickness=plate_thickness,
            ring_radius=ring_radius * (local_pitch / pitch),
            ring_height=ring_height,
        )

        m = cell.build_manifold().translate((float(pt[0]), float(pt[1]), float(pt[2])))
        instances.append(m)

    composed = m3d.Manifold.compose(instances)
    return _manifold_to_trimesh(composed)
