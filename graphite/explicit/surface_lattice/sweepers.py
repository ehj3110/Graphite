# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - Solidification Sweepers.

Provides three robust solidification engines:
1. Planar 2D Sweeper: 2D polygon buffering + orthogonal Z extrusion (Plates, Coasters, Bookmarks).
2. Cylindrical Prism Sweeper: Arc subdivision + radial prism extrusion (Napkin rings, Sleeves, Stents).
3. Surface-Normal Sweeper: Rectangular bars oriented to surface normals + inset trim (Spheres, Balls, Shells).
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import trimesh
from shapely.geometry import LineString, box, Polygon
from shapely.ops import unary_union
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
    sweep_square_straight_struts,
)


def sweep_planar_2d(
    segments_2d: list[tuple[np.ndarray, np.ndarray]],
    thickness: float = 3.0,
    strut_w: float = 1.6,
    outer_boundary: Polygon | None = None,
    inner_boundary: Polygon | None = None,
    z_center: bool = True,
) -> trimesh.Trimesh:
    """
    Extrude 2D line segments orthogonally along Z into a solid watertight plate.

    Parameters
    ----------
    segments_2d : list of (ndarray, ndarray)
        2D line segments on the XY plane.
    thickness : float
        Extrusion thickness in mm along Z.
    strut_w : float
        In-plane strut width in mm.
    outer_boundary : Polygon, optional
        Optional outer clipping boundary polygon.
    inner_boundary : Polygon, optional
        Optional inner boundary polygon for creating a framed perimeter border.
    z_center : bool
        If True, centers the mesh vertically so Z in [-thickness/2, thickness/2].

    Returns
    -------
    trimesh.Trimesh
        Watertight 3D manifold with mathematically flat Z-caps.
    """
    if not segments_2d:
        return trimesh.Trimesh()

    # Buffer 2D segments into polygonal bars with square end caps (cap_style=2)
    strut_polys: list[Polygon] = []
    half_w = strut_w / 2.0
    for p1, p2 in segments_2d:
        line = LineString([p1[:2], p2[:2]])
        strut_polys.append(line.buffer(half_w, cap_style=2))

    lattice_poly = unary_union(strut_polys)

    # Apply boundary clipping and framing
    if outer_boundary is not None:
        if inner_boundary is not None:
            # Framed: lattice clipped to inner_boundary, unioned with (outer - inner) frame
            frame_poly = outer_boundary.difference(inner_boundary)
            clipped = lattice_poly.intersection(inner_boundary)
            final_poly = clipped.union(frame_poly)
        else:
            final_poly = lattice_poly.intersection(outer_boundary)
    else:
        final_poly = lattice_poly

    path = trimesh.load_path(final_poly)
    extruded = path.extrude(thickness)

    if isinstance(extruded, (list, tuple)):
        mesh = trimesh.util.concatenate(extruded)
    elif hasattr(extruded, "geometry") and isinstance(extruded.geometry, dict):
        mesh = trimesh.util.concatenate(list(extruded.geometry.values()))
    else:
        mesh = extruded

    if z_center:
        mesh.apply_translation([0.0, 0.0, -thickness / 2.0])

    return mesh


def sweep_cylindrical_prisms(
    segments_2d: list[tuple[np.ndarray, np.ndarray]],
    r_in: float,
    r_out: float,
    height: float,
    center: np.ndarray | Sequence[float] | None = None,
    y_base: float = 0.0,
    strut_w: float = 2.0,
    max_step: float = 1.0,
    add_boundary_rings: bool = True,
    circular_segments: int = 128,
) -> m3d.Manifold:
    """
    Convert 2D (u, y) segments into a surface-conforming 3D cylindrical manifold.

    Parameters
    ----------
    segments_2d : list of (ndarray, ndarray)
        2D line segments where u in [0, C_mid] and y in [y_base, y_base + height].
    r_in : float
        Inner cylinder radius in mm.
    r_out : float
        Outer cylinder radius in mm.
    height : float
        Cylinder height in mm.
    center : array_like, optional
        3D cylinder center [x, y, z]. Defaults to [r_out, y_base + height/2, r_out].
    y_base : float
        Lower vertical baseline in mm.
    strut_w : float
        In-plane strut width in mm.
    max_step : float
        Maximum arc chord step in mm for subdividing curved segments.
    add_boundary_rings : bool
        If True, adds upper and lower boundary stabilization rings.
    circular_segments : int
        Discretization segments for bounding cylinders.

    Returns
    -------
    m3d.Manifold
        Watertight 3D manifold of the cylindrical lattice.
    """
    r_mid = (r_in + r_out) / 2.0
    c_circ = 2.0 * np.pi * r_mid
    if center is None:
        c_vec = np.array([r_out, y_base + height / 2.0, r_out], dtype=np.float64)
    else:
        c_vec = np.asarray(center, dtype=np.float64)

    radial_thick = (r_out - r_in) + 0.6
    all_prisms: list[m3d.Manifold] = []

    for p1, p2 in segments_2d:
        L2d = float(np.linalg.norm(p2 - p1))
        if L2d < 1e-4:
            continue

        du = p2[0] - p1[0]
        if abs(du) > c_circ / 2.0:
            p2_u = p2[0] - c_circ if du > 0 else p2[0] + c_circ
        else:
            p2_u = p2[0]

        n_steps = max(2, int(np.ceil(L2d / max_step)))
        t_vals = np.linspace(0.0, 1.0, n_steps + 1)
        pts: list[np.ndarray] = []
        for t in t_vals:
            u = (1.0 - t) * p1[0] + t * p2_u
            y = (1.0 - t) * p1[1] + t * p2_u
            th = u / r_mid
            x = c_vec[0] + r_mid * np.cos(th)
            z = c_vec[2] + r_mid * np.sin(th)
            pts.append(np.array([x, (1.0 - t) * p1[1] + t * p2[1], z]))

        for k in range(len(pts) - 1):
            pa, pb = pts[k], pts[k + 1]
            pmid = 0.5 * (pa + pb)
            t_vec = pb - pa
            L = float(np.linalg.norm(t_vec))
            if L < 1e-6:
                continue
            t_dir = t_vec / L

            n_vec = np.array([pmid[0] - c_vec[0], 0.0, pmid[2] - c_vec[2]])
            n_len = float(np.linalg.norm(n_vec))
            if n_len < 1e-6:
                continue
            n_dir = n_vec / n_len

            b_vec = np.cross(t_dir, n_dir)
            b_len = float(np.linalg.norm(b_vec))
            if b_len < 1e-6:
                continue
            b_dir = b_vec / b_len
            n_dir = np.cross(b_dir, t_dir)

            prism = m3d.Manifold.cube([radial_thick, strut_w, L], center=True)
            R_mat = [
                [n_dir[0], b_dir[0], t_dir[0], pmid[0]],
                [n_dir[1], b_dir[1], t_dir[1], pmid[1]],
                [n_dir[2], b_dir[2], t_dir[2], pmid[2]],
            ]
            all_prisms.append(prism.transform(R_mat))

    if add_boundary_rings:
        for y in [y_base, y_base + height]:
            cyl_out = m3d.Manifold.cylinder(
                height=strut_w,
                radius_low=r_out + 0.1,
                radius_high=r_out + 0.1,
                circular_segments=circular_segments,
                center=True,
            )
            cyl_in = m3d.Manifold.cylinder(
                height=strut_w + 0.2,
                radius_low=r_in - 0.1,
                radius_high=r_in - 0.1,
                circular_segments=circular_segments,
                center=True,
            )
            ring = (cyl_out - cyl_in).transform([
                [1.0, 0.0, 0.0, c_vec[0]],
                [0.0, 0.0, -1.0, y],
                [0.0, 1.0, 0.0, c_vec[2]],
            ])
            all_prisms.append(ring)

    composed = m3d.Manifold.batch_boolean(all_prisms, m3d.OpType.Add)

    # Trim to flush inner and outer cylindrical envelope
    sleeve_out = m3d.Manifold.cylinder(
        height=height,
        radius_low=r_out,
        radius_high=r_out,
        circular_segments=circular_segments,
        center=True,
    )
    sleeve_in = m3d.Manifold.cylinder(
        height=height + 1.0,
        radius_low=r_in,
        radius_high=r_in,
        circular_segments=circular_segments,
        center=True,
    )
    y_center = y_base + height / 2.0
    sleeve = (sleeve_out - sleeve_in).transform([
        [1.0, 0.0, 0.0, c_vec[0]],
        [0.0, 0.0, -1.0, y_center],
        [0.0, 1.0, 0.0, c_vec[2]],
    ])

    return composed ^ sleeve


def sweep_surface_skin(
    mesh: trimesh.Trimesh,
    nodes: np.ndarray,
    struts: np.ndarray,
    width: float = 1.6,
    thickness: float = 5.0,
    sphere_center: Sequence[float] | None = None,
    trim_radius: float | None = None,
) -> trimesh.Trimesh:
    """
    Sweep straight rectangular struts oriented along surface normals and trim with an inset surface.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Underlying surface mesh.
    nodes : ndarray, shape (N, 3)
        3D node coordinates.
    struts : ndarray, shape (S, 2)
        Edge indices.
    width : float
        In-plane strut width in mm.
    thickness : float
        Strut thickness along the surface normal in mm.
    sphere_center : array_like, optional
        Spherical center coordinates if part is a sphere.
    trim_radius : float, optional
        Inset trimming radius (e.g. R - 0.25 mm) to flush outer joints.

    Returns
    -------
    trimesh.Trimesh
        Watertight surface cage.
    """
    center_arr = np.asarray(sphere_center, dtype=np.float64) if sphere_center is not None else None
    return sweep_square_straight_struts(
        nodes=nodes,
        struts=struts,
        cad_mesh=mesh,
        side=width,
        thickness=thickness,
        sphere_center=center_arr,
        normal_oversize=0.25,
        trim_sphere_radius=trim_radius,
    )
