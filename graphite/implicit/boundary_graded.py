"""
Dual-EDT boundary-driven grading: distance-based layers from the Start Surface,
then smooth blending to the End Surface using shifted dual EDT weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.geometry.surface_picking import compute_face_surface_ids
from graphite.math.tpms import evaluate_tpms


def _distance_field_to_facets(
    mesh: trimesh.Trimesh,
    face_surface_ids: np.ndarray,
    logical_surface_ids: list[int],
    padded_min_bound: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    resolution: float,
) -> np.ndarray:
    """Point-cloud EDT approximation: distance (mm) to union of triangles in logical surfaces."""
    valid_ids = [int(i) for i in logical_surface_ids]
    present = np.unique(np.asarray(face_surface_ids, dtype=np.int64))
    present_set = set(present.tolist())
    bad = [i for i in valid_ids if i not in present_set]
    if bad:
        raise ValueError(
            f"surface IDs {bad} not present after feature-angle grouping; "
            f"available IDs include 0..{int(present.max()) if present.size else 'n/a'}"
        )

    mask = np.isin(face_surface_ids, valid_ids)
    selected_face_indices = np.where(mask)[0]
    if selected_face_indices.size == 0:
        raise ValueError("no mesh faces matched the selected logical surface IDs")
    sub_mesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[selected_face_indices],
        process=False,
    )

    num_points = max(10000, int(sub_mesh.area / (resolution**2)) * 2)
    surface_points, _ = trimesh.sample.sample_surface(sub_mesh, num_points)

    ix = np.clip(
        np.round((surface_points[:, 0] - padded_min_bound[0]) / resolution).astype(int),
        0,
        nx - 1,
    )
    iy = np.clip(
        np.round((surface_points[:, 1] - padded_min_bound[1]) / resolution).astype(int),
        0,
        ny - 1,
    )
    iz = np.clip(
        np.round((surface_points[:, 2] - padded_min_bound[2]) / resolution).astype(int),
        0,
        nz - 1,
    )

    surface_mask = np.zeros((nx, ny, nz), dtype=bool)
    surface_mask[ix, iy, iz] = True
    return edt(~surface_mask) * resolution


def generate_boundary_graded_lattice(
    stl_path,
    lattice_type="Gyroid",
    start_surfaces=None,
    end_surfaces=None,
    start_distances=None,
    start_pore_sizes=None,
    start_solid_fractions=None,
    end_pore_size=6.0,
    end_solid_fraction=0.15,
    resolution=0.25,
    feature_angle=45.0,
    center_origin=False,
    output_path=None,
):
    """
    Hybrid offset-boundary grading: 1D interpolation of pore size / solid fraction vs
    distance from the Start Surface, then dual-EDT smoothstep blend to End Surface values.

    Logical surface IDs (start/end) follow the same dihedral-angle grouping as the Surface
    Picker, controlled by ``feature_angle`` (degrees).
    """
    if start_surfaces is None:
        start_surfaces = [0]
    if end_surfaces is None:
        end_surfaces = [1]
    if start_distances is None:
        start_distances = [0.0, 5.0]
    if start_pore_sizes is None:
        start_pore_sizes = [2.0, 2.0]
    if start_solid_fractions is None:
        start_solid_fractions = [0.4, 0.4]

    dist_arr = np.asarray(start_distances, dtype=float).ravel()
    p_arr = np.asarray(start_pore_sizes, dtype=float).ravel()
    sf_arr = np.asarray(start_solid_fractions, dtype=float).ravel()

    if dist_arr.size != p_arr.size or dist_arr.size != sf_arr.size:
        raise ValueError(
            "start_distances, start_pore_sizes, and start_solid_fractions must have the same length"
        )
    if dist_arr.size < 1:
        raise ValueError("start_distances must contain at least one value")

    order = np.argsort(dist_arr)
    dist_arr = dist_arr[order]
    p_arr = p_arr[order]
    sf_arr = sf_arr[order]

    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    face_surface_ids = compute_face_surface_ids(mesh, feature_angle)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, nx, ny, nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    D_A = _distance_field_to_facets(
        mesh,
        face_surface_ids,
        list(start_surfaces),
        padded_min_bound,
        nx,
        ny,
        nz,
        resolution,
    )
    D_B = _distance_field_to_facets(
        mesh,
        face_surface_ids,
        list(end_surfaces),
        padded_min_bound,
        nx,
        ny,
        nz,
        resolution,
    )

    # 1D interpolation expanding out from Start Surface
    L_base = np.interp(D_A, dist_arr, p_arr)
    SF_base = np.interp(D_A, dist_arr, sf_arr)

    # Shift the Start Distance for the Dual-EDT region
    d_last = float(dist_arr[-1])
    D_A_prime = np.maximum(D_A - d_last, 0.0)

    # Dual-EDT weight bridging the final gap to Surface B
    W_raw = np.clip(D_A_prime / (D_A_prime + D_B + 1e-8), 0.0, 1.0)
    W = 3.0 * W_raw**2 - 2.0 * W_raw**3

    # Final interpolation
    L_grid = L_base * (1.0 - W) + float(end_pore_size) * W
    SF_grid = SF_base * (1.0 - W) + float(end_solid_fraction) * W

    L_grid = np.maximum(L_grid, 0.001)
    k_grid = 2.0 * np.pi / L_grid

    f = evaluate_tpms(lattice_type, k_grid, X, Y, Z)
    solid_field = np.abs(f) - SF_grid
    final_field = np.maximum(solid_field, cad_sdf)

    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )

    verts = (verts * resolution) + padded_min_bound

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(out))

    return mesh_out
