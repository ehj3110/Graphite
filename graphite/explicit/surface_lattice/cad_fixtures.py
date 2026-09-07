# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - CAD Fixture Extraction & Collar Rim Fusion.

Enables Method B: Automatic geometric parameter extraction from CAD fixture STLs
(e.g., base rings or hollow sleeves), lattice window carving, and solid collar rim fusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import trimesh
import manifold3d as m3d

from graphite.explicit.geometry_module import _trimesh_to_manifold, _manifold_to_trimesh


def inspect_cylinder_fixture(
    cad_mesh_or_path: Union[str, Path, trimesh.Trimesh],
    axis: str = "y",
) -> dict:
    """
    Analyze an input CAD STL to extract cylindrical parameters automatically.

    Parameters
    ----------
    cad_mesh_or_path : str, Path, or trimesh.Trimesh
        Input CAD model.
    axis : str
        Cylinder axis of symmetry ("x", "y", or "z"). Default is "y".

    Returns
    -------
    dict
        Geometric specifications: axis, center, r_in, r_out, wall_thickness, height, bounds.
    """
    if isinstance(cad_mesh_or_path, (str, Path)):
        mesh = trimesh.load(str(cad_mesh_or_path))
    else:
        mesh = cad_mesh_or_path

    bounds = mesh.bounds
    center = mesh.centroid

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    radial_indices = [i for i in range(3) if i != axis_idx]

    height = float(bounds[1, axis_idx] - bounds[0, axis_idx])

    # Compute radial distance from axis
    v_rad = mesh.vertices[:, radial_indices] - center[radial_indices]
    radii = np.linalg.norm(v_rad, axis=1)

    r_min = float(np.min(radii))
    r_max = float(np.max(radii))

    # Inner bore estimate: 5th percentile, outer bore: 95th percentile
    r_in = float(np.percentile(radii, 5))
    r_out = float(np.percentile(radii, 95))

    return {
        "axis": axis.lower(),
        "center": center,
        "height": height,
        "axis_min": float(bounds[0, axis_idx]),
        "axis_max": float(bounds[1, axis_idx]),
        "r_in": r_in,
        "r_out": r_out,
        "wall_thickness": r_out - r_in,
        "r_min_bound": r_min,
        "r_max_bound": r_max,
    }


def carve_and_fuse_collar_rims(
    cad_fixture: Union[str, Path, trimesh.Trimesh],
    lattice_manifold: m3d.Manifold,
    h_lattice: float,
    y_start: float = 6.65,
    center_xy: float = 25.4,
) -> trimesh.Trimesh:
    """
    Extract solid collar rims from a base part, carve the lattice window, and fuse with lattice.

    Parameters
    ----------
    cad_fixture : str, Path, or trimesh.Trimesh
        Input CAD fixture mesh (e.g. BaseRing_1to2.STL or BaseRing_V1.STL).
    lattice_manifold : m3d.Manifold
        Generated 3D cylindrical lattice manifold.
    h_lattice : float
        Height of the carved lattice window in mm.
    y_start : float
        Vertical starting height of the lattice window in mm.
    center_xy : float
        Center coordinate for X and Z in mm.

    Returns
    -------
    trimesh.Trimesh
        Unified watertight napkin ring or sleeve with solid collar rims.
    """
    if isinstance(cad_fixture, (str, Path)):
        base_mesh = trimesh.load(str(cad_fixture))
    else:
        base_mesh = cad_fixture

    m_base = _trimesh_to_manifold(base_mesh)

    y_center = y_start + h_lattice / 2.0
    center = np.array([center_xy, y_center, center_xy], dtype=np.float64)

    # Box cutout to hollow out the middle lattice window from CAD part
    box_cut = trimesh.creation.box(extents=[200.0, h_lattice, 200.0])
    box_cut.apply_translation(center)
    m_box = _trimesh_to_manifold(box_cut)

    # Extract collar rims: CAD - box
    solid_rims = m_base - m_box

    # Fuse rims with lattice: collar rims + lattice
    full_part = solid_rims + lattice_manifold
    return _manifold_to_trimesh(full_part)
