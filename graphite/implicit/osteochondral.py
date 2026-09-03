"""
Graphite Implicit Engine - Osteochondral Lattices

This module provides functionality to generate functionally graded TPMS lattices 
designed to mimic osteochondral (bone-to-cartilage) tissue structure. It allows 
continuous grading of both pore size and solid volume fraction along the Z-axis 
using integrated phase evaluation.
"""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from pathlib import Path

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.io.mesh_export import export_mesh
from graphite.math.tpms import calculate_integrated_phase, evaluate_tpms_phase


def generate_osteochondral_lattice(
    stl_path,
    lattice_type="Gyroid",
    z_heights=[0.0, 10.0],
    pore_sizes=[5.0, 5.0],
    solid_fractions=[0.33, 0.33],
    resolution=0.25,
    center_origin=False,
    output_path=None,
    export_formats=None,
):
    """
    Generate a conformal TPMS lattice with continuous Z-axis grading.

    Designed for osteochondral scaffolds where porosity and strut thickness 
    vary from the bone interface to the cartilage interface.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    lattice_type : str, optional
        TPMS equation type (e.g., 'Gyroid'), by default "Gyroid".
    z_heights : list of float, optional
        Z-axis control points (in mm) relative to the bottom of the bounding box, 
        by default [0.0, 10.0].
    pore_sizes : list of float, optional
        Target maximum inscribed sphere pore diameter (in mm) at each Z-height, 
        by default [5.0, 5.0].
    solid_fractions : list of float, optional
        Target solid volume fraction threshold at each Z-height, by default [0.33, 0.33].
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.

    Returns
    -------
    trimesh.Trimesh
        The meshed osteochondral lattice.
    """
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    # Relative Z-height from the bottom of the padded voxel grid
    Z_rel = Z - padded_min_bound[2]

    z_arr = np.array(z_heights)
    p_arr = np.array(pore_sizes)
    sf_arr = np.array(solid_fractions)

    L_grid = np.interp(Z_rel, z_arr, p_arr)
    SF_grid = np.interp(Z_rel, z_arr, sf_arr)

    L_grid = np.maximum(L_grid, 0.001)
    omega_local = 2.0 * np.pi / L_grid

    z_min = float(np.min(z_arr))
    z_max = float(np.max(z_arr))
    z_dense = np.linspace(z_min, z_max, 4096)
    w_dense = calculate_integrated_phase(z_dense, z_arr, p_arr)
    W_phase = np.interp(Z_rel, z_dense, w_dense)

    U = X * omega_local
    V = Y * omega_local
    F = evaluate_tpms_phase(lattice_type, U, V, W_phase)
    solid_field = np.abs(F) - SF_grid

    final_field = np.maximum(solid_field, cad_sdf)

    verts, faces, _normals, _values = marching_cubes(
        final_field, level=0.0, spacing=(resolution, resolution, resolution)
    )

    verts[:, 0] += padded_min_bound[0]
    verts[:, 1] += padded_min_bound[1]
    verts[:, 2] += padded_min_bound[2]

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    return mesh_out
