from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt as edt
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.math.tpms import evaluate_tpms


def generate_chirped_lattice(
    stl_path: str | Path,
    lattice_type: str = "Gyroid",
    gradient_type: str = "Z",
    modifier_path: str | Path | None = None,
    resolution: float = 0.25,
    solid_fraction: float = 0.33,
    transition_width: float = 5.0,
    center_origin: bool = False,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a conformal chirped TPMS lattice using a modifier-driven smoothstep field."""
    stl_path = Path(stl_path)

    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    size_x = float(mesh.bounds[1][0] - mesh.bounds[0][0])
    L_base = size_x / 6.0
    L_mod = L_base / 2.0
    k_base = 2.0 * np.pi / L_base
    k_mod = 2.0 * np.pi / L_mod

    X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    if gradient_type.lower() == "modifier" and modifier_path:
        modifier_path = Path(modifier_path)
        mod_mesh = trimesh.load(str(modifier_path))
        if not isinstance(mod_mesh, trimesh.Trimesh):
            mod_mesh = mod_mesh.dump(concatenate=True)

        pad_width = 4
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
    else:
        # Mathematical Axis Grading
        if gradient_type == "X":
            t = X
            t_min, t_max = padded_min_bound[0], padded_max_bound[0]
        elif gradient_type == "Y":
            t = Y
            t_min, t_max = padded_min_bound[1], padded_max_bound[1]
        elif gradient_type == "Z":
            t = Z
            t_min, t_max = padded_min_bound[2], padded_max_bound[2]
        elif gradient_type == "Radial":
            # Cylindrical distance from Z-axis
            t = np.sqrt(X**2 + Y**2)
            t_min, t_max = 0.0, np.max(t)
        else:
            raise ValueError(f"Unknown gradient_type: {gradient_type}")

        # Normalize t to [0, 1] across the bounding box
        W_linear = np.clip((t - t_min) / (t_max - t_min + 1e-8), 0.0, 1.0)

        # Apply smoothstep for C1 continuity
        W = 3.0 * W_linear**2 - 2.0 * W_linear**3

    K_grid = k_base * (1.0 - W) + k_mod * W
    F = evaluate_tpms(lattice_type, K_grid, X, Y, Z)

    solid_field = np.abs(F) - solid_fraction

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

    if center_origin:
        verts -= mesh_out.centroid
        mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))

    modifier_desc = Path(modifier_path).name if modifier_path else "axis-driven"
    print(
        f"Chirped {lattice_type} lattice from '{stl_path.name}' with driver '{modifier_desc}': "
        f"res={resolution}mm, SF={solid_fraction:.2f}, smooth={transition_width}mm"
    )
    print(
        f"  L_base={L_base:.3f}mm, L_mod={L_mod:.3f}mm, "
        f"marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )
    if center_origin:
        print("  Output mesh centered at origin")

    return mesh_out

