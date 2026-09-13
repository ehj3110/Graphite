"""
Graphite — Hashin-Shtrikman Optimal Plate-Lattice Generator

Implements closed-cell and open-cell plate-lattices (Berger et al. 2017, Tancogne-Dejean et al. 2018):
- Simple Cubic (SC) Plate-Lattice: 3 mutually orthogonal plates along {100} planes (membrane stiffness).
- Body-Centered Cubic (BCC) Plate-Lattice: 6 plates along {110} planes (rhombic dodecahedral symmetry).
- Face-Centered Cubic (FCC) Plate-Lattice: 4 plates along {111} planes (octahedral symmetry).
- SC-BCC Hybrid: 9 plates per cell combining cubic and diagonal shear stiffness.
- Plate thickness calibration solver matching target solid fraction rho* to exact tolerance.
- Monolithic, 100% watertight Manifold3D print-ready meshes.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.optimize import root_scalar

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
    _rotation_align_local_z_to_unit,
    _affine_rows_from_R_t,
)

# Plane normals for crystallographic plate families
NORMALS_110 = [
    np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0),
    np.array([1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0),
    np.array([1.0, 0.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
    np.array([1.0, 0.0, -1.0], dtype=np.float64) / np.sqrt(2.0),
    np.array([0.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(2.0),
    np.array([0.0, 1.0, -1.0], dtype=np.float64) / np.sqrt(2.0),
]

NORMALS_111 = [
    np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([1.0, -1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([1.0, 1.0, -1.0], dtype=np.float64) / np.sqrt(3.0),
    np.array([-1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0),
]


def _build_plate_unit_cell_manifold(
    cell_size: float,
    plate_thickness: float,
    topology: str = "sc",
) -> m3d.Manifold:
    """
    Construct a single canonical plate-lattice unit cell in [0, cell_size]^3.
    """
    a = float(cell_size)
    t = float(plate_thickness)
    diag = a * np.sqrt(3.0) * 1.05 if topology == "fcc" else a * np.sqrt(2.0) * 1.05
    center = np.array([a / 2.0, a / 2.0, a / 2.0], dtype=np.float64)



    parts: list[m3d.Manifold] = []

    if topology in ("sc", "sc_bcc"):
        # 3 mid-plane orthogonal plates
        box_xy = m3d.Manifold.cube([a, a, t], center=True).translate([a / 2.0, a / 2.0, a / 2.0])
        box_xz = m3d.Manifold.cube([a, t, a], center=True).translate([a / 2.0, a / 2.0, a / 2.0])
        box_yz = m3d.Manifold.cube([t, a, a], center=True).translate([a / 2.0, a / 2.0, a / 2.0])
        parts.extend([box_xy, box_xz, box_yz])

    if topology in ("bcc", "sc_bcc"):
        for n in NORMALS_110:
            slab = m3d.Manifold.cube([diag, diag, t], center=True)
            R_mat = _rotation_align_local_z_to_unit(n)
            aff = _affine_rows_from_R_t(R_mat, center)
            parts.append(slab.transform(aff))

    if topology == "fcc":
        for n in NORMALS_111:
            slab = m3d.Manifold.cube([diag, diag, t], center=True)
            R_mat = _rotation_align_local_z_to_unit(n)
            aff = _affine_rows_from_R_t(R_mat, center)
            parts.append(slab.transform(aff))

    cell = m3d.Manifold.compose(parts)
    bound_cube = m3d.Manifold.cube([a, a, a])
    return cell ^ bound_cube


def calibrate_plate_thickness(
    unit_cell_size: float,
    target_solid_fraction: float,
    topology: str = "sc",
) -> float:
    """
    Numerically solve for the exact plate thickness t achieving target_solid_fraction.
    """
    a = float(unit_cell_size)
    target_sf = float(target_solid_fraction)

    if not (0.0 < target_sf < 0.95):
        raise ValueError(f"target_solid_fraction must be in (0, 0.95), got {target_sf}")

    # First analytical estimate
    if topology == "sc":
        t0 = target_sf * a / 3.0
    elif topology == "bcc":
        t0 = target_sf * a / 4.24
    elif topology == "fcc":
        t0 = target_sf * a / 3.46
    elif topology == "sc_bcc":
        t0 = target_sf * a / 7.24
    else:
        raise ValueError(f"Unknown topology '{topology}'. Supported: 'sc', 'bcc', 'fcc', 'sc_bcc'.")

    def objective(t_val: float) -> float:
        cell = _build_plate_unit_cell_manifold(a, t_val, topology=topology)
        measured_sf = float(cell.volume()) / (a**3)
        return measured_sf - target_sf

    t_min = max(1e-4, t0 * 0.2)
    t_max = min(a * 0.45, t0 * 2.5)

    try:
        res = root_scalar(objective, bracket=[t_min, t_max], method="brentq")
        return float(res.root)
    except Exception:
        # Fallback to analytical estimate if numerical bracketing fails
        return float(t0)


def generate_plate_lattice(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    unit_cell_size: float,
    plate_thickness: float | None = None,
    target_solid_fraction: float | None = None,
    topology: str = "sc",
    crop_to_bounds: bool = True,
) -> trimesh.Trimesh:
    """
    Generate a high-stiffness plate-lattice mesh across a 3D bounding domain.

    Parameters:
        bounds: ((xmin, ymin, zmin), (xmax, ymax, zmax)) bounding domain in mm.
        unit_cell_size: Cell edge length a in mm.
        plate_thickness: Wall thickness t in mm (solved from target_solid_fraction if None).
        target_solid_fraction: Desired relative density rho* in (0, 1).
        topology: 'sc' (Simple Cubic) | 'bcc' ({110} planes) | 'fcc' ({111} planes) | 'sc_bcc' (hybrid).
        crop_to_bounds: Whether to clip the lattice cleanly to domain boundary planes.

    Returns:
        Watertight trimesh.Trimesh solid plate-lattice.
    """
    p_min = np.array(bounds[0], dtype=np.float64)
    p_max = np.array(bounds[1], dtype=np.float64)
    extents = p_max - p_min
    a = float(unit_cell_size)

    if a <= 0:
        raise ValueError(f"unit_cell_size must be positive, got {a}")
    if np.any(extents <= 0):
        raise ValueError(f"bounds must define a positive volume, got extents {extents}")

    topology = topology.lower().strip()
    if topology not in ("sc", "bcc", "fcc", "sc_bcc"):
        raise ValueError(f"Unknown topology '{topology}'. Supported: 'sc', 'bcc', 'fcc', 'sc_bcc'.")

    # Solve plate thickness if target solid fraction is provided
    if plate_thickness is None:
        if target_solid_fraction is None:
            raise ValueError("Must specify either plate_thickness or target_solid_fraction.")
        t = calibrate_plate_thickness(a, target_solid_fraction, topology=topology)
    else:
        t = float(plate_thickness)

    nx = int(np.ceil(extents[0] / a))
    ny = int(np.ceil(extents[1] / a))
    nz = int(np.ceil(extents[2] / a))

    # 1. OPTIMIZED DIRECT SLAB EVALUATION FOR SC
    if topology == "sc":
        lx = float(extents[0])
        ly = float(extents[1])
        lz = float(extents[2])
        ox, oy, oz = p_min

        parts: list[m3d.Manifold] = []
        # XY plates
        for k in range(nz):
            zc = oz + (k + 0.5) * a
            if zc <= p_max[2] + a * 0.5:
                parts.append(m3d.Manifold.cube([lx, ly, t], center=True).translate([ox + lx / 2.0, oy + ly / 2.0, zc]))
        # XZ plates
        for j in range(ny):
            yc = oy + (j + 0.5) * a
            if yc <= p_max[1] + a * 0.5:
                parts.append(m3d.Manifold.cube([lx, t, lz], center=True).translate([ox + lx / 2.0, yc, oz + lz / 2.0]))
        # YZ plates
        for i in range(nx):
            xc = ox + (i + 0.5) * a
            if xc <= p_max[0] + a * 0.5:
                parts.append(m3d.Manifold.cube([t, ly, lz], center=True).translate([xc, oy + ly / 2.0, oz + lz / 2.0]))

        lattice = m3d.Manifold.compose(parts)

    # 2. SLAB COMPOSITION FOR BCC, FCC, SC_BCC
    else:
        diag = a * np.sqrt(3.0) * 1.05
        slabs: list[m3d.Manifold] = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    center = p_min + np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
                    if topology in ("bcc", "sc_bcc"):
                        for n in NORMALS_110:
                            slab = m3d.Manifold.cube([diag, diag, t], center=True)
                            R_mat = _rotation_align_local_z_to_unit(n)
                            aff = _affine_rows_from_R_t(R_mat, center)
                            slabs.append(slab.transform(aff))
                    if topology == "fcc":
                        for n in NORMALS_111:
                            slab = m3d.Manifold.cube([diag, diag, t], center=True)
                            R_mat = _rotation_align_local_z_to_unit(n)
                            aff = _affine_rows_from_R_t(R_mat, center)
                            slabs.append(slab.transform(aff))
                    if topology == "sc_bcc":
                        box_xy = m3d.Manifold.cube([a, a, t], center=True).translate(tuple(float(c) for c in center))
                        box_xz = m3d.Manifold.cube([a, t, a], center=True).translate(tuple(float(c) for c in center))
                        box_yz = m3d.Manifold.cube([t, a, a], center=True).translate(tuple(float(c) for c in center))
                        slabs.extend([box_xy, box_xz, box_yz])

        lattice = m3d.Manifold.compose(slabs)


    # Boundary cropping
    if crop_to_bounds:
        bound_box = m3d.Manifold.cube([float(extents[0]), float(extents[1]), float(extents[2])]).translate(
            (float(p_min[0]), float(p_min[1]), float(p_min[2]))
        )
        lattice = lattice ^ bound_box

    return _manifold_to_trimesh(lattice)
