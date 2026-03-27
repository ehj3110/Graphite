import numpy as np
import trimesh
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.math.tpms import evaluate_tpms


def generate_osteochondral_lattice(
    stl_path,
    lattice_type="Gyroid",
    z_heights=[0.0, 10.0],
    pore_sizes=[5.0, 5.0],
    solid_fractions=[0.33, 0.33],
    resolution=0.25,
    center_origin=False,
    output_path=None,
):
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
    K_grid = 2.0 * np.pi / L_grid

    F = evaluate_tpms(lattice_type, K_grid, X, Y, Z)
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
        mesh_out.export(output_path)

    return mesh_out
