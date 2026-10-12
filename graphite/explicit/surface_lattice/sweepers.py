# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - Solidification Sweepers.

Provides three robust solidification engines:
1. Planar 2D Sweeper: 2D polygon buffering + orthogonal Z extrusion (Plates, Coasters, Bookmarks).
2. Cylindrical Prism Sweeper: Arc subdivision + radial prism extrusion (Napkin rings, Sleeves, Stents).
3. Surface-Normal Sweeper: Rectangular bars oriented to surface normals + inset trim (Spheres, Balls, Shells).
"""
from __future__ import annotations

from typing import Callable, Sequence, Union
import numpy as np
import trimesh
from shapely.geometry import LineString, box, Polygon, Point
from shapely.ops import unary_union
import manifold3d as m3d

from graphite.explicit.geometry_module import (
    _trimesh_to_manifold,
    _manifold_to_trimesh,
    sweep_square_straight_struts,
)


def sweep_planar_2d(
    segments_2d: list[tuple[np.ndarray, np.ndarray]],
    thickness: float | Callable[[float, float], float] = 3.0,
    strut_w: float | np.ndarray | Sequence[float] | Callable[[float, float], float] = 1.6,
    outer_boundary: Polygon | None = None,
    inner_boundary: Polygon | None = None,
    z_center: bool = True,
    tapered: bool = True,
    number_glyph: Polygon | None = None,
    number_wall: Polygon | None = None,
) -> trimesh.Trimesh:
    """
    Extrude 2D line segments orthogonally along Z into a solid watertight plate.
    Supports uniform or spatially graded strut widths w(x, y) and out-of-plane thickness t_z(x, y).

    Parameters
    ----------
    segments_2d : list of (ndarray, ndarray)
        2D line segments on the XY plane.
    thickness : float or callable
        Extrusion thickness in mm along Z, or a callable t_z(x, y) for spatial thickness grading.
    strut_w : float, array, or callable
        In-plane strut width in mm. Can be scalar, per-strut sequence, or callable w(x, y).
    outer_boundary : Polygon, optional
        Optional outer clipping boundary polygon.
    inner_boundary : Polygon, optional
        Optional inner boundary polygon for creating a framed perimeter border.
    z_center : bool
        If True, centers the mesh vertically so Z in [-thickness/2, thickness/2].
    tapered : bool
        If True and strut_w is callable, constructs continuous linearly tapered trapezoidal
        ligaments between segment endpoints.
    number_glyph : Polygon, optional
        Optional glyph polygon to cut out from the lattice.
    number_wall : Polygon, optional
        Optional boundary ribbon wall polygon to fuse around the cutout glyph.

    Returns
    -------
    trimesh.Trimesh
        Watertight 3D manifold mesh.
    """
    if not segments_2d:
        return trimesh.Trimesh()

    # Buffer 2D segments into polygonal bars
    strut_polys: list[Polygon] = []
    is_seq = isinstance(strut_w, (list, tuple, np.ndarray))
    is_callable = callable(strut_w)

    for idx, (p1, p2) in enumerate(segments_2d):
        p1_xy = np.asarray(p1[:2], dtype=np.float64)
        p2_xy = np.asarray(p2[:2], dtype=np.float64)
        vec = p2_xy - p1_xy
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            continue

        mid_xy = 0.5 * (p1_xy + p2_xy)

        if is_callable:
            if tapered:
                w1 = float(strut_w(p1_xy[0], p1_xy[1]))
                w2 = float(strut_w(p2_xy[0], p2_xy[1]))
            else:
                w_mid = float(strut_w(mid_xy[0], mid_xy[1]))
                w1 = w2 = w_mid
        elif is_seq:
            w_val = float(strut_w[idx])
            w1 = w2 = w_val
        else:
            w_val = float(strut_w)
            w1 = w2 = w_val

        if tapered and abs(w1 - w2) > 1e-4:
            # Continuous linearly tapered trapezoid
            u_dir = vec / length
            n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)
            p1_l = p1_xy + n_dir * (w1 / 2.0)
            p1_r = p1_xy - n_dir * (w1 / 2.0)
            p2_r = p2_xy - n_dir * (w2 / 2.0)
            p2_l = p2_xy + n_dir * (w2 / 2.0)
            strut_polys.append(Polygon([p1_l, p2_l, p2_r, p1_r]))
            # Round node discs to guarantee clean seamless intersections
            strut_polys.append(Point(p1_xy).buffer(w1 / 2.0, quad_segs=8))
            strut_polys.append(Point(p2_xy).buffer(w2 / 2.0, quad_segs=8))
        else:
            avg_w = 0.5 * (w1 + w2)
            line = LineString([p1_xy, p2_xy])
            strut_polys.append(line.buffer(avg_w / 2.0, cap_style=2))

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

    # Apply number cutout and perimeter wall if requested
    if number_glyph is not None:
        final_poly = final_poly.difference(number_glyph)
        if number_wall is not None:
            final_poly = final_poly.union(number_wall)
        if outer_boundary is not None:
            final_poly = final_poly.intersection(outer_boundary)

    # Extrude solid mesh
    is_thickness_callable = callable(thickness)
    nominal_thick = 1.0 if is_thickness_callable else float(thickness)

    path = trimesh.load_path(final_poly)
    extruded = path.extrude(nominal_thick)

    if isinstance(extruded, (list, tuple)):
        mesh = trimesh.util.concatenate(extruded)
    elif hasattr(extruded, "geometry") and isinstance(extruded.geometry, dict):
        mesh = trimesh.util.concatenate(list(extruded.geometry.values()))
    else:
        mesh = extruded

    if hasattr(mesh, "to_mesh"):
        mesh = mesh.to_mesh()

    if mesh.volume < 0.0:
        mesh.faces = mesh.faces[:, [0, 2, 1]]

    if z_center and not is_thickness_callable:
        mesh.apply_translation([0.0, 0.0, -nominal_thick / 2.0])

    # Apply out-of-plane thickness grading t_z(x, y) if specified
    if is_thickness_callable:
        verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        # Nominal extrusion has z in [0, 1]
        z_norm = verts[:, 2] / nominal_thick
        # If z_center, shift z_norm to [-0.5, 0.5]
        z_frac = (z_norm - 0.5) if z_center else z_norm

        tz_vals = np.array([float(thickness(vx, vy)) for vx, vy in verts[:, :2]], dtype=np.float64)
        verts[:, 2] = z_frac * tz_vals
        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=True)

    return mesh


def sweep_cylindrical_prisms(
    segments_2d: list[tuple[np.ndarray, np.ndarray]],
    r_in: float,
    r_out: float,
    height: float,
    center: np.ndarray | Sequence[float] | None = None,
    y_base: float = 0.0,
    strut_w: float | np.ndarray | Sequence[float] | Callable[[float, float], float] = 2.0,
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
    strut_w : float, array, or callable
        In-plane strut width in mm. Can be scalar, per-strut sequence, or callable w(u, y).
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
    is_callable_w = callable(strut_w)
    is_seq_w = isinstance(strut_w, (list, tuple, np.ndarray))

    for seg_idx, (p1, p2) in enumerate(segments_2d):
        L2d = float(np.linalg.norm(p2 - p1))
        if L2d < 1e-4:
            continue

        du = p2[0] - p1[0]
        if abs(du) > c_circ / 2.0:
            p2_u = p2[0] - c_circ if du > 0 else p2[0] + c_circ
        else:
            p2_u = p2[0]

        if is_callable_w:
            w_seg = float(strut_w(0.5 * (p1[0] + p2_u), 0.5 * (p1[1] + p2[1])))
        elif is_seq_w:
            w_seg = float(strut_w[seg_idx])
        else:
            w_seg = float(strut_w)

        n_steps = max(2, int(np.ceil(L2d / max_step)))
        t_vals = np.linspace(0.0, 1.0, n_steps + 1)
        pts: list[np.ndarray] = []
        for t in t_vals:
            u = (1.0 - t) * p1[0] + t * p2_u
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

            prism = m3d.Manifold.cube([radial_thick, w_seg, L], center=True)
            R_mat = [
                [n_dir[0], b_dir[0], t_dir[0], pmid[0]],
                [n_dir[1], b_dir[1], t_dir[1], pmid[1]],
                [n_dir[2], b_dir[2], t_dir[2], pmid[2]],
            ]
            all_prisms.append(prism.transform(R_mat))

    if add_boundary_rings:
        ring_w = 2.0 if (is_callable_w or is_seq_w) else float(strut_w)
        for y in [y_base, y_base + height]:
            cyl_out = m3d.Manifold.cylinder(
                height=ring_w,
                radius_low=r_out + 0.1,
                radius_high=r_out + 0.1,
                circular_segments=circular_segments,
                center=True,
            )
            cyl_in = m3d.Manifold.cylinder(
                height=ring_w + 0.2,
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
