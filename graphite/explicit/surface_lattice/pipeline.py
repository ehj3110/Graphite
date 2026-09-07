# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - Unified Pipeline.

Main entry point for generating 2D and surface-conformal lattices across:
- Mode A: Flat Plates & Coupons (orthogonal Z-extrusion)
- Mode B: Cylinders, Sleeves & Napkin Rings (radial prism extrusion)
- Mode C: Surface Meshes & Duals (surface-normal sweeping on spheres or general shells)
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union
import numpy as np
import trimesh
import manifold3d as m3d

from graphite.explicit.surface_lattice.unit_cells import (
    tessellate_chiral_domain,
)
from graphite.explicit.surface_lattice.face_operators import (
    apply_surface_pattern_to_mesh,
)
from graphite.explicit.surface_lattice.sweepers import (
    sweep_planar_2d,
    sweep_cylindrical_prisms,
    sweep_surface_skin,
)
from graphite.explicit.surface_lattice.cad_fixtures import (
    inspect_cylinder_fixture,
    carve_and_fuse_collar_rims,
)
from graphite.explicit.geometry_module import _manifold_to_trimesh


def generate_surface_lattice(
    surface: str,
    pattern: str = "tetra_chiral",
    *,
    # Plate dimensions
    width: float | None = None,
    height: float | None = None,
    thickness: float | None = None,
    # Cylinder dimensions
    r_in: float | None = None,
    r_out: float | None = None,
    y_base: float = 0.0,
    cad_fixture: Union[str, Path, trimesh.Trimesh, None] = None,
    center_xy: float = 25.4,
    # Metamaterial parameters
    n_circumferential: int = 10,
    r_node: float = 2.0,
    strut_w: float = 1.6,
    n_circle_segs: int = 16,
    # General mesh surface
    surface_mesh: trimesh.Trimesh | None = None,
    sphere_center: Sequence[float] | None = None,
    trim_radius: float | None = None,
) -> trimesh.Trimesh:
    """
    Unified generator for 2D planar and surface-conformal lattices.

    Parameters
    ----------
    surface : str
        "plate" (flat Z-extrusion), "cylinder" (radial prism wrapping), or "mesh" (surface-normal sweep).
    pattern : str
        Topology pattern name:
        - Chiral: "tetra_chiral", "tri_chiral", "anti_tetra_chiral", "anti_tri_chiral"
        - Regular: "rhombic" (surface dual), "kelvin", "tesseract", "icosahedral", "tetrahedral", "grid"
    width : float, optional
        Plate width in mm (for surface="plate"). Default 50.0.
    height : float, optional
        Plate or cylinder height in mm. Default 50.0 (plate) or 38.1 (cylinder).
    thickness : float, optional
        Solid thickness in mm (for plate or surface skin).
    r_in : float, optional
        Inner cylinder radius in mm.
    r_out : float, optional
        Outer cylinder radius in mm.
    y_base : float, optional
        Vertical baseline for cylinder in mm. Default 0.0.
    cad_fixture : str, Path, or trimesh.Trimesh, optional
        Input CAD fixture STL to auto-extract dimensions and fuse collar rims.
    center_xy : float, optional
        Center coordinate for CAD fixture positioning. Default 25.4.
    n_circumferential : int
        Number of unit cells around cylinder circumference or plate width. Default 10.
    r_node : float
        Circular node radius in mm for chiral patterns. Default 2.0.
    strut_w : float
        In-plane strut width in mm. Default 1.6.
    n_circle_segs : int
        Resolution for circular nodes. Default 16.
    surface_mesh : trimesh.Trimesh, optional
        Input surface mesh (for surface="mesh").
    sphere_center : array_like, optional
        Center coordinate for spherical cages.
    trim_radius : float, optional
        Inset trimming radius for flushing surface joints.

    Returns
    -------
    trimesh.Trimesh
        Watertight 3D solid mesh.
    """
    surface_key = surface.lower().strip()

    # =========================================================================
    # 1. FLAT PLATE MODE (Bookmarks, Coasters, Panels)
    # =========================================================================
    if surface_key in ("plate", "planar", "coaster", "bookmark"):
        w = width if width is not None else 50.0
        h = height if height is not None else 50.0
        th = thickness if thickness is not None else 3.0

        # Generate 2D segments
        if "chiral" in pattern.lower():
            segs_2d, _ = tessellate_chiral_domain(
                topology=pattern,
                domain_width=w,
                domain_height=h,
                n_circumferential=n_circumferential,
                r_node=r_node,
                n_circle_segs=n_circle_segs,
                periodic_x=False,
            )
        else:
            # Fallback to tri_sq coaster generators for regular patterns
            from scripts.coasters.tri_sq_patterns import (
                generate_tri_coaster_segments,
                generate_sq_coaster_segments,
                unique_segments,
            )
            p_cap = pattern.capitalize()
            if p_cap in ("Grid", "Icosahedral", "Kelvin", "Tesseract"):
                raw_segs = generate_sq_coaster_segments(p_cap, w / n_circumferential, 0.0, 0.0, w, h)
            else:
                raw_segs = generate_tri_coaster_segments(p_cap, (w / n_circumferential) / np.sqrt(3), 0.0, 0.0, w, h)
            segs_2d = unique_segments(raw_segs)

        return sweep_planar_2d(segs_2d, thickness=th, strut_w=strut_w)

    # =========================================================================
    # 2. CYLINDRICAL MODE (Napkin Rings, Sleeves, Stents)
    # =========================================================================
    elif surface_key in ("cylinder", "cylindrical", "sleeve", "ring"):
        # Auto-detect parameters from CAD fixture if provided (Method B)
        if cad_fixture is not None:
            fix_info = inspect_cylinder_fixture(cad_fixture)
            rin = r_in if r_in is not None else fix_info["r_in"]
            rout = r_out if r_out is not None else min(rin + 3.0, fix_info["r_out"])
            h_lat = height if height is not None else 19.05
            y_b = y_base if y_base != 0.0 else 6.65
        else:
            rin = r_in if r_in is not None else 19.05
            rout = r_out if r_out is not None else 22.05
            h_lat = height if height is not None else 38.1
            y_b = y_base

        r_mid = (rin + rout) / 2.0
        c_mid = 2.0 * np.pi * r_mid

        # Generate periodic 2D segments
        segs_2d, _ = tessellate_chiral_domain(
            topology=pattern,
            domain_width=c_mid,
            domain_height=h_lat,
            n_circumferential=n_circumferential,
            r_node=r_node,
            n_circle_segs=n_circle_segs,
            periodic_x=True,
            y_base=y_b,
        )

        center_vec = np.array([center_xy, y_b + h_lat / 2.0, center_xy], dtype=np.float64)

        lattice_manifold = sweep_cylindrical_prisms(
            segments_2d=segs_2d,
            r_in=rin,
            r_out=rout,
            height=h_lat,
            center=center_vec,
            y_base=y_b,
            strut_w=strut_w,
            add_boundary_rings=True,
        )

        # If CAD fixture provided, fuse solid collar rims
        if cad_fixture is not None:
            return carve_and_fuse_collar_rims(
                cad_fixture=cad_fixture,
                lattice_manifold=lattice_manifold,
                h_lattice=h_lat,
                y_start=y_b,
                center_xy=center_xy,
            )

        return _manifold_to_trimesh(lattice_manifold)

    # =========================================================================
    # 3. SURFACE MESH MODE (Spheres, Baseballs, General Surface Duals)
    # =========================================================================
    elif surface_key in ("mesh", "surface", "sphere", "surface_dual"):
        if surface_mesh is None:
            # Default to sphere of radius 37 mm (standard baseball)
            surface_mesh = trimesh.creation.icosphere(subdivisions=3, radius=37.0)
            if sphere_center is None:
                sphere_center = [37.0, 37.0, 37.0]
            surface_mesh.apply_translation(sphere_center)
            if trim_radius is None:
                trim_radius = 37.0 - 0.25

        th = thickness if thickness is not None else 5.0

        nodes_3d, struts_3d = apply_surface_pattern_to_mesh(
            mesh=surface_mesh,
            pattern=pattern,
        )

        return sweep_surface_skin(
            mesh=surface_mesh,
            nodes=nodes_3d,
            struts=struts_3d,
            width=strut_w,
            thickness=th,
            sphere_center=sphere_center,
            trim_radius=trim_radius,
        )

    else:
        raise ValueError(
            f"Unknown surface '{surface}'. Supported options: 'plate', 'cylinder', 'mesh'."
        )
