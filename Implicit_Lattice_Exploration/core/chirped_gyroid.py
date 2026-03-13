from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt as edt
from skimage.measure import marching_cubes


def generate_chirped_gyroid(
    stl_path: str | Path,
    modifier_path: str | Path,
    resolution: float = 0.25,
    solid_fraction: float = 0.33,
    transition_width: float = 5.0,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a conformal chirped gyroid using a modifier-driven smoothstep field."""
    stl_path = Path(stl_path)
    modifier_path = Path(modifier_path)

    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    size_x = float(mesh.bounds[1][0] - mesh.bounds[0][0])
    L_base = size_x / 6.0
    L_mod = L_base / 2.0
    k_base = 2.0 * np.pi / L_base
    k_mod = 2.0 * np.pi / L_mod

    # Fast voxelization and EDT setup for the base mesh
    vox = mesh.voxelized(pitch=resolution).fill()
    pad_width = 4
    inside_mask = np.pad(vox.matrix, pad_width, mode="constant", constant_values=False)

    padded_min_bound = vox.translation - (pad_width * resolution)
    nx, ny, nz = inside_mask.shape

    x_axis = np.arange(nx, dtype=float) * resolution + padded_min_bound[0]
    y_axis = np.arange(ny, dtype=float) * resolution + padded_min_bound[1]
    z_axis = np.arange(nz, dtype=float) * resolution + padded_min_bound[2]
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

    inside_dist = edt(inside_mask)
    outside_dist = edt(~inside_mask)
    cad_sdf = (outside_dist - inside_dist) * resolution

    # Modifier SDF and smoothstep weight map
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

    W_linear = np.clip(1.0 - (dist_to_mod / transition_width), 0.0, 1.0)
    W = 3.0 * W_linear**2 - 2.0 * W_linear**3

    # Chirped spatial frequency field
    K_grid = k_base * (1.0 - W) + k_mod * W
    F = (
        np.sin(K_grid * X) * np.cos(K_grid * Y)
        + np.sin(K_grid * Y) * np.cos(K_grid * Z)
        + np.sin(K_grid * Z) * np.cos(K_grid * X)
    )

    solid_field = np.abs(F) - solid_fraction

    # Implicit intersection and extraction
    final_field = np.maximum(solid_field, cad_sdf)

    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    t_mc = time.perf_counter() - t0

    verts = (verts * resolution) + padded_min_bound
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))

    print(
        f"Chirped gyroid from '{stl_path.name}' with modifier '{modifier_path.name}': "
        f"res={resolution}mm, SF={solid_fraction:.2f}, smooth={transition_width}mm"
    )
    print(
        f"  L_base={L_base:.3f}mm, L_mod={L_mod:.3f}mm, "
        f"marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )

    return mesh_out
