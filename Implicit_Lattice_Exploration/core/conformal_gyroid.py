from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt as edt


def _compute_L_and_k(pore_size: float | None, unit_cell_size: float | None, solid_fraction: float) -> tuple[float, float]:
    if pore_size is not None:
        L = pore_size / (1.0 - 1.15 * solid_fraction)
    elif unit_cell_size is not None:
        L = unit_cell_size
    else:
        raise ValueError("Must provide either pore_size or unit_cell_size")
    k = 2.0 * np.pi / L
    return L, k


def generate_conformal_gyroid(
    stl_path: str | Path,
    resolution: float = 0.25,
    unit_cell_size: float | None = None,
    pore_size: float | None = 5.0,
    solid_fraction: float = 0.33,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a conformal gyroid inside an input STL using EDT-based CAD SDF."""
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

    # Gyroid math (same as basic generator)
    L, k = _compute_L_and_k(pore_size, unit_cell_size, solid_fraction)

    F = (
        np.sin(k * X) * np.cos(k * Y)
        + np.sin(k * Y) * np.cos(k * Z)
        + np.sin(k * Z) * np.cos(k * X)
    )

    t = 1.5 * (2.0 * solid_fraction - 1.0)
    solid_field = np.abs(F) - abs(t)

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

    # Shift vertices back to global coordinates (padded_min_bound in world space)
    verts = (verts * resolution) + padded_min_bound

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))

    print(f"Conformal gyroid from '{stl_path.name}': res={resolution}mm, SF={solid_fraction:.2f}")
    print(f"  L={L:.3f}mm (pore={pore_size}), marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}")

    return mesh_out

