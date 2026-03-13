from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
from scipy.interpolate import RegularGridInterpolator
from skimage.measure import marching_cubes


def generate_gradient_gyroid(
    stl_path: str | Path,
    resolution: float = 0.25,
    pore_size: float = 5.0,
    min_solid_fraction: float = 0.10,
    max_solid_fraction: float = 0.80,
    gradient_type: str = "Z",
    modifier_path: str | Path | None = None,
    transition_width: float = 10.0,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a conformal gyroid with axis or modifier-based gradients."""
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    # 1) Fast voxelization at given resolution
    vox = mesh.voxelized(pitch=resolution).fill()

    # 2) Pad binary mask for watertight EDT boundary
    pad_width = 4
    inside_mask = np.pad(vox.matrix, pad_width, mode="constant", constant_values=False)

    # 3) Grid aligned to padded voxel matrix
    padded_min_bound = vox.translation - (pad_width * resolution)
    nx, ny, nz = inside_mask.shape

    x_axis = np.arange(nx, dtype=float) * resolution + padded_min_bound[0]
    y_axis = np.arange(ny, dtype=float) * resolution + padded_min_bound[1]
    z_axis = np.arange(nz, dtype=float) * resolution + padded_min_bound[2]

    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    # Gyroid math with constant cell size derived from the average solid fraction
    avg_sf = (min_solid_fraction + max_solid_fraction) / 2.0
    L = pore_size / (1.0 - 1.15 * avg_sf)
    k = 2.0 * np.pi / L

    F = (
        np.sin(k * X) * np.cos(k * Y)
        + np.sin(k * Y) * np.cos(k * Z)
        + np.sin(k * Z) * np.cos(k * X)
    )

    # Universal weight map W in [0, 1]
    W = np.zeros_like(X, dtype=np.float64)
    gradient_key = gradient_type.upper()

    if gradient_key == "MODIFIER" and modifier_path is not None:
        modifier_path = Path(modifier_path)
        mod_mesh = trimesh.load(str(modifier_path))
        if not isinstance(mod_mesh, trimesh.Trimesh):
            mod_mesh = mod_mesh.dump(concatenate=True)

        mod_vox = mod_mesh.voxelized(pitch=resolution).fill()
        mod_mask = np.pad(mod_vox.matrix, pad_width, mode="constant", constant_values=False)

        mod_inside_dist = edt(mod_mask)
        mod_outside_dist = edt(~mod_mask)
        mod_sdf = (mod_outside_dist - mod_inside_dist) * resolution

        mod_min_bound = mod_vox.translation - (pad_width * resolution)
        mod_nx, mod_ny, mod_nz = mod_mask.shape
        mod_x = np.arange(mod_nx, dtype=float) * resolution + mod_min_bound[0]
        mod_y = np.arange(mod_ny, dtype=float) * resolution + mod_min_bound[1]
        mod_z = np.arange(mod_nz, dtype=float) * resolution + mod_min_bound[2]

        interp = RegularGridInterpolator(
            (mod_x, mod_y, mod_z),
            mod_sdf,
            bounds_error=False,
            fill_value=transition_width,
        )
        points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        dist_to_mod = interp(points).reshape(X.shape)
        W = np.clip(1.0 - (dist_to_mod / transition_width), 0.0, 1.0)
    elif gradient_key in {"X", "Y", "Z"}:
        if gradient_key == "X":
            target_arr = X
        elif gradient_key == "Y":
            target_arr = Y
        else:
            target_arr = Z
        arr_min, arr_max = target_arr.min(), target_arr.max()
        W = (target_arr - arr_min) / (arr_max - arr_min)

    # Spatially varying sheet thickness using direct solid-fraction proxy
    sf_grid = min_solid_fraction + W * (max_solid_fraction - min_solid_fraction)
    t_grid = sf_grid
    solid_field = np.abs(F) - t_grid

    # Smooth CAD SDF via EDT on padded inside_mask
    inside_dist = edt(inside_mask)
    outside_dist = edt(~inside_mask)
    cad_sdf = (outside_dist - inside_dist) * resolution

    # Implicit intersection of gyroid sheet and CAD volume
    final_field = np.maximum(solid_field, cad_sdf)

    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    t_mc = time.perf_counter() - t0

    # Shift vertices back to global coordinates
    verts = (verts * resolution) + padded_min_bound

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))

    print(
        f"Gradient gyroid from '{stl_path.name}': res={resolution}mm, pore={pore_size}mm, "
        f"gradient={gradient_type}"
    )
    print(
        f"  SF range={min_solid_fraction:.2f}->{max_solid_fraction:.2f}, "
        f"L={L:.3f}mm, marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )

    return mesh_out
