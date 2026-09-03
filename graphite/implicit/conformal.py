"""
Graphite Implicit Engine - Conformal Lattices

This module generates conformal Triply Periodic Minimal Surface (TPMS) lattices 
bounded by arbitrary STL geometries. It computes the intersection of the TPMS 
implicit field with the Signed Distance Field (SDF) of the CAD boundary, and 
can optionally generate solid outer shells for selected faces.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.density_control import tau_from_wall_thickness_mm
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.io.mesh_export import export_mesh
from graphite.math.tpms import evaluate_tpms


def _compute_L_and_k(
    pore_size: float | None, unit_cell_size: float | None, solid_fraction: float
) -> tuple[float, float]:
    if pore_size is not None:
        L = pore_size / (1.0 - 1.15 * solid_fraction)
    elif unit_cell_size is not None:
        L = unit_cell_size
    else:
        raise ValueError("Must provide either pore_size or unit_cell_size")
    k = 2.0 * np.pi / L
    return L, k


def generate_conformal_lattice(
    stl_path: str | Path,
    lattice_type: str = "Gyroid",
    resolution: float = 0.25,
    pore_size: float | None = 5.0,
    unit_cell_size: float | None = None,
    solid_fraction: float = 0.33,
    wall_thickness_mm: float | None = None,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    center_origin: bool = False,
    selected_surfaces: list[int] | None = None,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
    exact_cad_trim: bool = False,
    trim_dilation_mm: float | None = None,
) -> trimesh.Trimesh:
    """
    Generate a conformal TPMS lattice inside an input STL using EDT-based CAD SDF.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    lattice_type : str, optional
        TPMS equation type (e.g., 'Gyroid'), by default "Gyroid".
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    pore_size : float, optional
        Target maximum inscribed sphere pore diameter in mm. Mutually exclusive 
        with unit_cell_size. By default 5.0.
    unit_cell_size : float, optional
        Unit cell period L in mm (alternative to pore_size), by default None.
    solid_fraction : float, optional
        Target solid volume fraction threshold, by default 0.33.
    wall_thickness_mm : float, optional
        If set, overrides solid_fraction and sets TPMS threshold from physical
        wall thickness and period L.
    export_mode : str, optional
        'core' (lattice only), 'skin' (solid shell only), or 'combined' 
        (lattice with shell), by default "core".
    shell_thickness : float, optional
        Thickness of the generated outer shell in mm, by default 2.0.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    selected_surfaces : list of int, optional
        List of specific mesh face indices to apply the skin to. If None, applies 
        globally. By default None.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.

    Returns
    -------
    trimesh.Trimesh
        The resulting meshed and clipped conformal lattice.
    """
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, nx, ny, nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    L, k = _compute_L_and_k(pore_size, unit_cell_size, solid_fraction)
    F = evaluate_tpms(lattice_type, k, X, Y, Z)

    if wall_thickness_mm is not None:
        tau = float(tau_from_wall_thickness_mm(wall_thickness_mm, L).ravel()[0])
    else:
        tau = float(solid_fraction)
    solid_field = np.abs(F) - tau

    if exact_cad_trim:
        dilation = float(trim_dilation_mm if trim_dilation_mm is not None else 3.0 * resolution)
        effective_cad_sdf = cad_sdf - dilation
    else:
        effective_cad_sdf = cad_sdf

    core_sdf = np.maximum(solid_field, effective_cad_sdf)

    if selected_surfaces is not None and len(selected_surfaces) > 0:
        facets = mesh.facets
        valid_ids = [int(i) for i in selected_surfaces]
        max_id = len(facets) - 1
        invalid_ids = [i for i in valid_ids if i < 0 or i > max_id]
        if invalid_ids:
            raise ValueError(
                f"selected_surfaces contains invalid IDs {invalid_ids}; valid range is 0 to {max_id}"
            )

        selected_face_indices = np.hstack([facets[i] for i in valid_ids])
        sub_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[selected_face_indices],
            process=False,
        )

        num_points = max(10000, int(sub_mesh.area / (resolution**2)) * 2)
        surface_points, _ = trimesh.sample.sample_surface(sub_mesh, num_points)

        ix = np.clip(
            np.round((surface_points[:, 0] - padded_min_bound[0]) / resolution).astype(
                int
            ),
            0,
            nx - 1,
        )
        iy = np.clip(
            np.round((surface_points[:, 1] - padded_min_bound[1]) / resolution).astype(
                int
            ),
            0,
            ny - 1,
        )
        iz = np.clip(
            np.round((surface_points[:, 2] - padded_min_bound[2]) / resolution).astype(
                int
            ),
            0,
            nz - 1,
        )

        surface_mask = np.zeros((nx, ny, nz), dtype=bool)
        surface_mask[ix, iy, iz] = True
        distance_to_surface = edt(~surface_mask) * resolution

        skin_sdf = np.maximum(cad_sdf, distance_to_surface - shell_thickness)
    else:
        skin_sdf = np.maximum(cad_sdf, -cad_sdf - shell_thickness)

    if export_mode == "core":
        final_field = core_sdf
    elif export_mode == "skin":
        final_field = skin_sdf
    elif export_mode == "combined":
        final_field = np.minimum(core_sdf, skin_sdf)
    else:
        raise ValueError("export_mode must be 'core', 'skin', or 'combined'")

    t0 = time.perf_counter()
    iso = extract_isosurface(
        final_field,
        spacing=(resolution, resolution, resolution),
        origin=padded_min_bound,
        level=0.0,
        enforce_watertight=True,
    )
    mesh_out = iso.mesh
    t_mc = time.perf_counter() - t0

    if exact_cad_trim:
        t_trim = time.perf_counter()
        try:
            import manifold3d
            from graphite.explicit.geometry_module import manifold_to_trimesh, trimesh_to_manifold
            man_tpms = trimesh_to_manifold(mesh_out)
            man_cad = trimesh_to_manifold(mesh)
            if (
                man_tpms is not None
                and man_cad is not None
                and man_cad.status() == manifold3d.Error.NoError
            ):
                man_trimmed = man_tpms ^ man_cad
                mesh_out = manifold_to_trimesh(man_trimmed)
                trimesh.repair.fix_normals(mesh_out)
                t_trim_s = time.perf_counter() - t_trim
                print(
                    f"  Exact B-Rep Boolean Trim: {len(mesh_out.faces):,} faces, "
                    f"watertight={mesh_out.is_watertight} in {t_trim_s:.2f}s"
                )
        except Exception as exc:
            print(f"  Warning: exact_cad_trim failed ({exc}); retaining implicit isosurface.")

    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path is not None:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    print(
        f"Conformal {lattice_type} lattice from '{stl_path.name}': res={resolution}mm, "
        f"SF={solid_fraction:.2f}, mode={export_mode}"
    )
    print(
        f"  L={L:.3f}mm (pore={pore_size}), shell={shell_thickness:.2f}mm, "
        f"marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )
    if center_origin:
        print("  Output mesh centered at origin")
    if selected_surfaces:
        print(f"  Localized shell surfaces: {selected_surfaces}")

    return mesh_out


def generate_conformal_gyroid(
    stl_path: str | Path,
    resolution: float = 0.25,
    pore_size: float | None = 5.0,
    solid_fraction: float = 0.33,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    selected_surfaces: list[int] | None = None,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """
    Backward-compatible wrapper for the original gyroid-only API.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    pore_size : float, optional
        Target maximum inscribed sphere pore diameter in mm, by default 5.0.
    solid_fraction : float, optional
        Target solid volume fraction threshold, by default 0.33.
    export_mode : str, optional
        'core', 'skin', or 'combined', by default "core".
    shell_thickness : float, optional
        Thickness of the generated outer shell in mm, by default 2.0.
    selected_surfaces : list of int, optional
        List of specific mesh face indices to apply the skin to, by default None.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.

    Returns
    -------
    trimesh.Trimesh
        The resulting meshed and clipped conformal lattice.
    """
    return generate_conformal_lattice(
        stl_path=stl_path,
        lattice_type="Gyroid",
        resolution=resolution,
        pore_size=pore_size,
        solid_fraction=solid_fraction,
        export_mode=export_mode,
        shell_thickness=shell_thickness,
        selected_surfaces=selected_surfaces,
        output_path=output_path,
    )

