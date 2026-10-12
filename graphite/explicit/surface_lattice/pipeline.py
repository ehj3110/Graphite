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
    tessellate_reentrant_domain,
    get_triangle_segments,
    get_square_segments,
    dedupe_2d_segments,
    tessellate_graded_cell_domain,
)
from graphite.explicit.surface_lattice.grading import (
    width_for_constant_solid_fraction,
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

SUPPORTED_SURFACES: tuple[str, ...] = (
    "plate",
    "planar",
    "coaster",
    "bookmark",
    "cylinder",
    "cylindrical",
    "sleeve",
    "ring",
    "tube",
    "mesh",
    "surface",
    "sphere",
    "surface_dual",
)

SUPPORTED_PATTERNS: tuple[str, ...] = (
    "reentrant",
    "re_entrant",
    "tetra_chiral",
    "tri_chiral",
    "anti_tetra_chiral",
    "anti_tri_chiral",
    "grid",
    "tetrahedral",
    "icosahedral",
    "kelvin",
    "tesseract",
    "rhombic",
)


def generate_surface_lattice(
    surface: str,
    pattern: str = "tetra_chiral",
    *,
    # Plate dimensions
    width: float | None = None,
    height: float | None = None,
    thickness: float | Callable[[float, float], float] | None = None,
    # Cylinder dimensions
    r_in: float | None = None,
    r_out: float | None = None,
    y_base: float = 0.0,
    cad_fixture: Union[str, Path, trimesh.Trimesh, None] = None,
    center_xy: float = 25.4,
    # Metamaterial / Lattice parameters
    n_circumferential: int = 10,
    r_node: float = 2.0,
    strut_w: float | np.ndarray | Sequence[float] | Callable[[float, float], float] = 1.6,
    n_circle_segs: int = 16,
    tapered: bool = True,
    # Variable cell sizing & forming compensation
    graded_cell_size: bool = False,
    cell_size_min: float = 4.0,
    cell_size_max: float = 12.0,
    gradient_axis: str = "xy",
    constant_sf: float | None = None,
    warp_power: float = 2.0,
    theta_field: float | Callable[[float], float] | Sequence[float] | None = None,
    theta_min_deg: float = 5.0,
    theta_max_deg: float = 45.0,
    sine_compensation: dict | None = None,
    profile_compensation: dict | None = None,
    curvature_compensation: dict | None = None,
    # General mesh surface
    surface_mesh: trimesh.Trimesh | None = None,
    sphere_center: Sequence[float] | None = None,
    trim_radius: float | None = None,
    export_stl: bool = False,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> trimesh.Trimesh | Path:
    """
    Unified generator for 2D planar and surface-conformal lattices.

    Parameters
    ----------
    surface : str
        "plate" (flat Z-extrusion), "cylinder" / "tube" (radial prism wrapping), or "mesh" (surface-normal sweep).
    pattern : str
        Topology pattern name:
        - Chiral: "tetra_chiral", "tri_chiral", "anti_tetra_chiral", "anti_tri_chiral"
        - Reentrant: "reentrant"
        - Regular: "rhombic" (surface dual), "kelvin", "tesseract", "icosahedral", "tetrahedral", "grid"
    width : float, optional
        Plate width in mm (for surface="plate"). Default 50.0.
    height : float, optional
        Plate or cylinder height in mm. Default 50.0 (plate) or 38.1 (cylinder).
    thickness : float or callable, optional
        Solid thickness in mm (for plate or surface skin), or callable t_z(x, y) for spatial thickness grading.
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
    strut_w : float, array, or callable
        In-plane strut width in mm. Default 1.6.
    n_circle_segs : int
        Resolution for circular nodes. Default 16.
    tapered : bool
        If True and strut_w is callable, constructs continuous linearly tapered trapezoidal struts.
    graded_cell_size : bool
        If True, generates a smoothly graded grid where local cell size L(x, y) varies across the domain.
    cell_size_min : float
        Minimum cell size in mm for graded_cell_size=True.
    cell_size_max : float
        Maximum cell size in mm for graded_cell_size=True.
    gradient_axis : str
        Grading axis for cell sizing: 'x', 'y', or 'xy'.
    constant_sf : float, optional
        If set (e.g. 0.15), automatically scales strut width w(x, y) proportionally with local L(x, y)
        to maintain mathematically constant solid fraction everywhere.
    warp_power : float
        Non-linear coordinate warping power for cell sizing.
    surface_mesh : trimesh.Trimesh, optional
        Input surface mesh (for surface="mesh").
    sphere_center : array_like, optional
        Center coordinate for spherical cages.
    trim_radius : float, optional
        Inset trimming radius for flushing surface joints.
    export_stl : bool, optional
        If True, exports generated mesh to output_path.
    output_path : str or Path, optional
        Destination filepath when export_stl is True or output_path is provided. Supports .stl, .3mf, .ply.

    Returns
    -------
    trimesh.Trimesh | Path
        Watertight 3D solid mesh, or Path if export_stl=True or output_path is provided.
    """
    surface_key = surface.lower().strip()
    pat_lower = pattern.lower().strip()

    if surface_key not in SUPPORTED_SURFACES:
        raise ValueError(
            f"Unsupported surface '{surface}'. Supported options: {sorted(SUPPORTED_SURFACES)}"
        )

    if pat_lower not in SUPPORTED_PATTERNS:
        raise ValueError(
            f"Unsupported pattern '{pattern}'. Supported options: {sorted(SUPPORTED_PATTERNS)}"
        )

    # =========================================================================
    # 1. FLAT PLATE MODE (Bookmarks, Coasters, Panels)
    # =========================================================================
    if surface_key in ("plate", "planar", "coaster", "bookmark"):
        w = float(width if width is not None else 50.0)
        h = float(height if height is not None else 50.0)
        if w <= 0.0 or h <= 0.0:
            raise ValueError(
                f"Dimensions width and height must be strictly positive, got width={w}, height={h}"
            )
        if not callable(thickness) and float(thickness if thickness is not None else 3.0) <= 0.0:
            raise ValueError(f"thickness must be strictly positive, got {thickness}")
        th = thickness if thickness is not None else 3.0
        if n_circumferential < 1:
            raise ValueError(f"n_circumferential must be >= 1, got {n_circumferential}")

        cell_pitch = w / float(n_circumferential)
        if isinstance(strut_w, (int, float)):
            if float(strut_w) <= 0.0:
                raise ValueError(f"strut_w must be strictly positive, got {strut_w}")
            if float(strut_w) >= cell_pitch:
                raise ValueError(
                    f"strut_w ({strut_w:.3f} mm) must be strictly less than cell pitch "
                    f"a ({cell_pitch:.3f} mm) to maintain open pore voids."
                )

        prof_comp = profile_compensation if profile_compensation is not None else curvature_compensation
        if prof_comp is not None:
            from graphite.explicit.surface_lattice.grading import ProfileCurvatureMapper

            if "mapper" in prof_comp:
                p_mapper = prof_comp["mapper"]
            elif "harmonics" in prof_comp:
                p_mapper = ProfileCurvatureMapper(
                    harmonics=prof_comp["harmonics"],
                    length=float(prof_comp.get("length", 140.0)),
                )
            else:
                p_mapper = ProfileCurvatureMapper.create_semirandom(
                    length=float(prof_comp.get("length", 140.0)),
                    seed=int(prof_comp.get("seed", 42)),
                )

            # If width not explicitly specified, snap to developable profile arc length
            if width is None:
                w = float(p_mapper.S_total)

            p_cell_min = float(prof_comp.get("cell_size_min", cell_size_min if cell_size_min != 4.0 else 3.5))
            p_cell_max = float(prof_comp.get("cell_size_max", cell_size_max if cell_size_max != 12.0 else 14.0))
            p_power = float(prof_comp.get("warp_power", warp_power if warp_power != 2.0 else 0.5))

            if prof_comp.get("modulate_cell_size", True):
                col_bounds = p_mapper.compute_column_boundaries(
                    cell_size_min=p_cell_min,
                    cell_size_max=p_cell_max,
                    power=p_power,
                )
                n_circumferential = len(col_bounds) - 1
            else:
                col_bounds = None

            t_sheet = float(prof_comp.get("sheet_thickness", th if not callable(th) else 2.0))
            th_nom = float(prof_comp.get("theta_nominal_deg", 20.0))
            th_min = float(prof_comp.get("theta_min_deg", theta_min_deg))
            th_max = float(prof_comp.get("theta_max_deg", theta_max_deg))
            beta = prof_comp.get("beta", None)
            gamma = float(prof_comp.get("gamma", 0.15))
            t_base = float(prof_comp.get("t_base", strut_w if isinstance(strut_w, (int, float)) else 1.0))

            if theta_field is None and prof_comp.get("modulate_theta", True):
                theta_field = lambda x: p_mapper.compensate_theta(
                    x,
                    theta_nominal_deg=th_nom,
                    sheet_thickness=t_sheet,
                    beta=beta,
                    theta_min_deg=th_min,
                    theta_max_deg=th_max,
                )

            if prof_comp.get("modulate_strut_thickness", False):
                strut_w = lambda x, y: p_mapper.compensate_strut_thickness(x, t_base=t_base, gamma=gamma)

            if pat_lower in ("reentrant", "re_entrant"):
                segs_2d, _ = tessellate_reentrant_domain(
                    domain_width=w,
                    domain_height=h,
                    n_circumferential=n_circumferential,
                    column_boundaries=col_bounds,
                    periodic_x=False,
                    theta_field=theta_field,
                    theta_min_deg=th_min,
                    theta_max_deg=th_max,
                )
                if prof_comp.get("constant_sf") is not None or constant_sf is not None:
                    phi_target = float(prof_comp.get("constant_sf", constant_sf))
                    seg_midpoints = [(p1[0] + p2[0]) / 2.0 for p1, p2 in segs_2d]
                    strut_w = np.array([
                        width_for_constant_solid_fraction(
                            p_mapper.cell_size_at_s(sm, cell_size_min=p_cell_min, cell_size_max=p_cell_max, power=p_power),
                            solid_fraction=phi_target,
                            topology="square",
                        )
                        for sm in seg_midpoints
                    ], dtype=np.float64)
            else:
                topo = "triangle" if ("tri" in pat_lower or "kagome" in pat_lower) else "square"
                segs_2d, cell_lengths, _ = tessellate_graded_cell_domain(
                    domain_width=w,
                    domain_height=h,
                    cell_size_min=p_cell_min,
                    cell_size_max=p_cell_max,
                    x_coords=col_bounds,
                    topology=topo,
                )
                if prof_comp.get("constant_sf") is not None or constant_sf is not None:
                    phi_target = float(prof_comp.get("constant_sf", constant_sf))
                    strut_w = np.array([
                        width_for_constant_solid_fraction(L_k, solid_fraction=phi_target, topology=topo)
                        for L_k in cell_lengths
                    ], dtype=np.float64)
        elif graded_cell_size:
            topo = "triangle" if ("tri" in pat_lower or "kagome" in pat_lower) else "square"
            segs_2d, cell_lengths, _ = tessellate_graded_cell_domain(
                domain_width=w,
                domain_height=h,
                cell_size_min=cell_size_min,
                cell_size_max=cell_size_max,
                gradient_axis=gradient_axis,
                topology=topo,
                warp_power=warp_power,
            )
            if constant_sf is not None:
                phi_target = float(constant_sf)
                strut_w = np.array([
                    width_for_constant_solid_fraction(L_k, solid_fraction=phi_target, topology=topo)
                    for L_k in cell_lengths
                ], dtype=np.float64)
        elif pat_lower in ("reentrant", "re_entrant"):
            if sine_compensation is not None:
                from graphite.explicit.surface_lattice.grading import SineCurvatureMapper

                amp = float(sine_compensation.get("amplitude", 0.2 * w))
                wv = float(sine_compensation.get("wavelength", w))
                t_sheet = float(sine_compensation.get("sheet_thickness", th))
                th_nom = float(sine_compensation.get("theta_nominal_deg", 20.0))
                th_min = float(sine_compensation.get("theta_min_deg", theta_min_deg))
                th_max = float(sine_compensation.get("theta_max_deg", theta_max_deg))
                beta = sine_compensation.get("beta", None)
                gamma = float(sine_compensation.get("gamma", 0.15))
                t_base = float(
                    sine_compensation.get(
                        "t_base", strut_w if isinstance(strut_w, (int, float)) else 1.0
                    )
                )

                mapper = SineCurvatureMapper(amplitude=amp, wavelength=wv)
                if theta_field is None and sine_compensation.get("modulate_theta", True):
                    theta_field = lambda x: mapper.compensate_theta(
                        x,
                        theta_nominal_deg=th_nom,
                        sheet_thickness=t_sheet,
                        beta=beta,
                        theta_min_deg=th_min,
                        theta_max_deg=th_max,
                    )
                if sine_compensation.get("modulate_strut_thickness", True):
                    strut_w = lambda x, y: mapper.compensate_strut_thickness(
                        x, t_base=t_base, gamma=gamma
                    )

            segs_2d, _ = tessellate_reentrant_domain(
                domain_width=w,
                domain_height=h,
                n_circumferential=n_circumferential,
                periodic_x=False,
                theta_field=theta_field,
                theta_min_deg=theta_min_deg,
                theta_max_deg=theta_max_deg,
            )
        elif "chiral" in pat_lower:
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
            p_cap = pattern.capitalize()
            if p_cap in ("Grid", "Icosahedral", "Kelvin", "Tesseract"):
                raw_segs = get_square_segments(p_cap, side=w / n_circumferential)
            else:
                raw_segs = get_triangle_segments(p_cap, side=(w / n_circumferential) / np.sqrt(3))
            segs_2d = dedupe_2d_segments(raw_segs)

        mesh = sweep_planar_2d(segs_2d, thickness=th, strut_w=strut_w, tapered=tapered)

    # =========================================================================
    # 2. CYLINDRICAL MODE (Napkin Rings, Sleeves, Stents)
    # =========================================================================
    elif surface_key in ("cylinder", "cylindrical", "sleeve", "ring", "tube"):
        if n_circumferential < 1:
            raise ValueError(f"n_circumferential must be >= 1, got {n_circumferential}")

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
        if pat_lower in ("reentrant", "re_entrant"):
            segs_2d, _ = tessellate_reentrant_domain(
                domain_width=c_mid,
                domain_height=h_lat,
                n_circumferential=n_circumferential,
                periodic_x=True,
                y_base=y_b,
            )
        else:
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
            mesh = carve_and_fuse_collar_rims(
                cad_fixture=cad_fixture,
                lattice_manifold=lattice_manifold,
                h_lattice=h_lat,
                y_start=y_b,
                center_xy=center_xy,
            )
        else:
            mesh = _manifold_to_trimesh(lattice_manifold)

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

        mesh = sweep_surface_skin(
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

    if export_stl or output_path is not None:
        out_dest = Path(output_path) if output_path else Path("outputs/surface_lattice.stl")
        out_dest.parent.mkdir(parents=True, exist_ok=True)
        ext = out_dest.suffix.lower().lstrip(".")
        if ext in ("stl", "ply", "3mf", "obj"):
            mesh.export(str(out_dest), file_type=ext)
        else:
            mesh.export(str(out_dest))
        return out_dest

    return mesh
