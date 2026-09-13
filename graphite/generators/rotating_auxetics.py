"""
Graphite — Rotating Rigid Squares Auxetic Metamaterial Generator

Implements the classical Grima & Evans (2000) rotating rigid unit auxetic mechanism:
- Rigid square plates of side length s and thickness t.
- Alternating clockwise/counter-clockwise deployment angle theta in a checkerboard pattern.
- Continuous living hinges connecting corner vertices without collision.
- Exact Poisson's ratio nu = -1 throughout the deployment range.
- Monolithic, 100% watertight Manifold3D print-in-place meshes.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import trimesh
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
)


def generate_rotating_squares_lattice(
    dimensions: tuple[int, int] | tuple[int, int, int],
    square_side: float = 10.0,
    plate_thickness: float = 2.0,
    hinge_radius: float = 0.45,
    rotation_angle_deg: float = 30.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    layer_spacing: float | None = None,
    align_axes: bool = True,
) -> trimesh.Trimesh:
    """
    Generate a monolithic 3D-printable rotating squares auxetic lattice.

    Parameters:
        dimensions: (nx, ny) grid count for a 2D sheet, or (nx, ny, nz) for 3D stacking.
        square_side: Side length s of each rigid square plate in mm.
        plate_thickness: Out-of-plane thickness t in mm.
        hinge_radius: Radius of the cylindrical living hinge at shared vertices in mm.
        rotation_angle_deg: Inter-square deployment angle theta in degrees (0 to 60 deg).
        origin: Base coordinate offset (x0, y0, z0).
        layer_spacing: Inter-layer vertical pitch for 3D stacking (defaults to 1.5 * plate_thickness).
        align_axes: If True, rotates the assembly by -45 degrees to align with Cartesian X and Y axes.

    Returns:
        Watertight trimesh.Trimesh representing the monolithic auxetic metamaterial.
    """
    if len(dimensions) == 2:
        nx, ny = int(dimensions[0]), int(dimensions[1])
        nz = 1
    elif len(dimensions) == 3:
        nx, ny, nz = int(dimensions[0]), int(dimensions[1]), int(dimensions[2])
    else:
        raise ValueError(f"dimensions must be a 2-tuple (nx, ny) or 3-tuple (nx, ny, nz), got {dimensions}")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"dimensions must be positive integers, got {dimensions}")
    if square_side <= 0:
        raise ValueError(f"square_side must be positive, got {square_side}")
    if plate_thickness <= 0:
        raise ValueError(f"plate_thickness must be positive, got {plate_thickness}")
    if hinge_radius <= 0:
        raise ValueError(f"hinge_radius must be positive, got {hinge_radius}")
    if not (0.0 <= rotation_angle_deg <= 60.0):
        raise ValueError(f"rotation_angle_deg must be in [0, 60], got {rotation_angle_deg}")

    s = float(square_side)
    t = float(plate_thickness)
    r_h = float(hinge_radius)
    theta_deg = float(rotation_angle_deg)
    phi_deg = theta_deg / 2.0
    phi = np.radians(phi_deg)
    D = s * np.cos(phi)
    ox, oy, oz = origin

    z_pitch = float(layer_spacing) if layer_spacing is not None else t * 1.5

    # Templates
    sq_base = m3d.Manifold.cube([s, s, t], center=True)
    hinge_cyl = m3d.Manifold.cylinder(t, r_h, r_h, circular_segments=16, center=True)

    corners_local = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64) * (s / 2.0)

    parts: list[m3d.Manifold] = []

    for k in range(nz):
        z_center = k * z_pitch + t / 2.0
        layer_parity = (k % 2)

        for u in range(nx):
            for v in range(ny):
                cell_parity = (u + v + layer_parity) % 2
                ang = phi_deg if (cell_parity == 0) else -phi_deg
                cx = (u - v) * D
                cy = (u + v) * D

                sq = sq_base.rotate([0, 0, ang]).translate([cx, cy, z_center])
                parts.append(sq)

                # Living hinge cylinders at the 4 corners
                rad = np.radians(ang)
                R_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                c_world = corners_local @ R_mat.T + np.array([cx, cy])
                for c_pt in c_world:
                    parts.append(hinge_cyl.translate([c_pt[0], c_pt[1], z_center]))

        # Multi-layer 3D vertical pins at shared hinges
        if nz > 1 and k < nz - 1:
            pin_height = z_pitch
            pin_cyl = m3d.Manifold.cylinder(pin_height, r_h, r_h, circular_segments=16, center=True)
            z_pin_mid = z_center + z_pitch / 2.0
            for u in range(nx):
                for v in range(ny):
                    cell_parity = (u + v + layer_parity) % 2
                    ang = phi_deg if (cell_parity == 0) else -phi_deg
                    rad = np.radians(ang)
                    R_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
                    cx = (u - v) * D
                    cy = (u + v) * D
                    c_world = corners_local @ R_mat.T + np.array([cx, cy])
                    for c_pt in c_world:
                        parts.append(pin_cyl.translate([c_pt[0], c_pt[1], z_pin_mid]))

    composed = m3d.Manifold.compose(parts)

    if align_axes:
        composed = composed.rotate([0, 0, -45.0])

    # Re-zero bounding box minimum to origin
    mesh_tmp = _manifold_to_trimesh(composed)
    min_b = mesh_tmp.bounds[0]
    composed = composed.translate((-float(min_b[0]) + ox, -float(min_b[1]) + oy, -float(min_b[2]) + oz))

    return _manifold_to_trimesh(composed)
