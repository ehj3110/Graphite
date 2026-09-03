"""
Graphite Implicit Engine - Graded Lattices

This module provides functionality to generate functionally graded TPMS lattices, 
where the solid volume fraction varies spatially (e.g., thicker struts at the 
bottom transitioning to thinner struts at the top). It supports simple axis-driven 
gradients (X, Y, Z, Radial), modifier-mesh-driven gradients, and externally 
supplied solid fraction fields (e.g., from FEA analysis).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt as edt
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.io.mesh_export import export_mesh
from graphite.math.tpms import evaluate_tpms


def generate_graded_lattice(
    stl_path: str | Path,
    lattice_type: str = "Gyroid",
    gradient_type: str = "Z",
    modifier_path: str | Path | None = None,
    resolution: float = 0.25,
    pore_size: float = 5.0,
    min_solid_fraction: float = 0.10,
    max_solid_fraction: float = 0.50,
    transition_width: float = 5.0,
    center_origin: bool = False,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
    solid_fraction_field: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Generate a conformal graded TPMS lattice with axis or modifier-based gradients.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    lattice_type : str, optional
        TPMS equation type (e.g., 'Gyroid'), by default "Gyroid".
    gradient_type : str, optional
        Axis of the gradient ('X', 'Y', 'Z', 'Radial') or 'modifier', by default "Z".
    modifier_path : str or Path, optional
        Path to the modifier STL if gradient_type is 'modifier', by default None.
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    pore_size : float, optional
        Target maximum inscribed sphere pore diameter in mm, by default 5.0.
    min_solid_fraction : float, optional
        Minimum solid volume fraction (at weight 0), by default 0.10.
    max_solid_fraction : float, optional
        Maximum solid volume fraction (at weight 1), by default 0.50.
    transition_width : float, optional
        Width of the smoothstep transition region in mm (used for modifier mode), 
        by default 5.0.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.
    solid_fraction_field : ndarray, optional
        Pre-computed solid fraction field (voxel grid) to use. If provided,
        gradient_type and modifier_path are ignored. Used for Aristo FEA
        stress-informed latticing.

    Returns
    -------
    trimesh.Trimesh
        The meshed and clipped graded lattice.
    """
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    # Legacy pore-size mapping (sheet TPMS) for graded fields
    L = pore_size / 0.65
    k = 2.0 * np.pi / L

    F = evaluate_tpms(lattice_type, k, X, Y, Z)

    # Universal weight map W in [0, 1]
    if gradient_type.lower() == "modifier" and modifier_path:
        modifier_path = Path(modifier_path)
        mod_mesh = trimesh.load(str(modifier_path))
        if not isinstance(mod_mesh, trimesh.Trimesh):
            mod_mesh = mod_mesh.dump(concatenate=True)

        mod_vox = mod_mesh.voxelized(pitch=resolution).fill()
        pad_width = 4
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

    if solid_fraction_field is not None:
        # Use provided FEA-informed field (ensure shape match)
        if solid_fraction_field.shape != F.shape:
            raise ValueError(
                f"solid_fraction_field shape {solid_fraction_field.shape} "
                f"must match grid shape {F.shape}."
            )
        SF_grid = solid_fraction_field
    else:
        SF_grid = min_solid_fraction + W * (max_solid_fraction - min_solid_fraction)

    solid_field = np.abs(F) - SF_grid

    final_field = np.maximum(solid_field, cad_sdf)

    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    t_mc = time.perf_counter() - t0

    # spacing= already maps voxel indices → mm; only translate into world frame
    verts = verts + padded_min_bound

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    if center_origin:
        verts -= mesh_out.centroid
        mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    print(
        f"Gradient {lattice_type} lattice from '{stl_path.name}': res={resolution}mm, "
        f"pore={pore_size}mm, gradient={gradient_type}"
    )
    print(
        f"  SF range={min_solid_fraction:.2f}->{max_solid_fraction:.2f}, "
        f"L={L:.3f}mm, marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )
    if center_origin:
        print("  Output mesh centered at origin")

    return mesh_out

