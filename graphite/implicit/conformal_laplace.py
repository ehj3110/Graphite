"""
Harmonic (Laplace) UVW coordinates inside a masked domain via Jacobi smoothing.

World coordinates X, Y, Z are relaxed toward discrete harmonic extensions in the
interior; boundary voxels keep fixed values for Dirichlet conditions. The
resulting U, V, W are scaled and fed into the Gyroid trigonometric combination.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

# 6-neighbor (face-adjacent) average on a 3×3×3 stencil, symmetric Laplace step
_KERNEL_6 = (
    np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=np.float64,
    )
    / 6.0
)


def solve_laplace_uvw(
    mask: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    iterations: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Jacobi iteration: each step replaces (U,V,W) by the 6-neighbor mean, then
    enforces Dirichlet data on the mask boundary and outside the domain.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    U = np.array(X, dtype=np.float64, copy=True)
    V = np.array(Y, dtype=np.float64, copy=True)
    W = np.array(Z, dtype=np.float64, copy=True)

    eroded = scipy.ndimage.binary_erosion(mask)
    boundary = mask & ~eroded

    for _ in range(iterations):
        U_new = scipy.ndimage.convolve(U, _KERNEL_6, mode="nearest")
        V_new = scipy.ndimage.convolve(V, _KERNEL_6, mode="nearest")
        W_new = scipy.ndimage.convolve(W, _KERNEL_6, mode="nearest")

        fix = ~mask | boundary
        U_new[fix] = U[fix]
        V_new[fix] = V[fix]
        W_new[fix] = W[fix]

        U, V, W = U_new, V_new, W_new

    return U, V, W


def frustum_mask_and_grid(
    grid_shape: tuple[int, int, int],
    physical_size: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Centered box [-sx/2, sx/2]³ and frustum mask:
    max_r(z) = 20 - 0.3 * (Z + 20) with Z in [-20, 20] when physical_size is 40.
    """
    nx, ny, nz = grid_shape
    sx, sy, sz = physical_size
    x = np.linspace(-sx / 2.0, sx / 2.0, nx)
    y = np.linspace(-sy / 2.0, sy / 2.0, ny)
    z = np.linspace(-sz / 2.0, sz / 2.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    max_r = 20.0 - 0.3 * (Z + 20.0)
    mask = (X**2 + Y**2) <= max_r**2
    return mask, X, Y, Z


def gyroid_from_harmonic_uvw(
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    mask: np.ndarray,
    L_target: float,
) -> np.ndarray:
    omega = 2.0 * np.pi / L_target
    field = (
        np.sin(U * omega) * np.cos(V * omega)
        + np.sin(V * omega) * np.cos(W * omega)
        + np.sin(W * omega) * np.cos(U * omega)
    )
    out = np.array(field, dtype=np.float64)
    out[~mask] = 1.0
    return out


def generate_conformal_gyroid(
    grid_shape: tuple[int, int, int],
    physical_size: tuple[float, float, float],
    L_target: float,
    iterations: int = 500,
) -> np.ndarray:
    mask, X, Y, Z = frustum_mask_and_grid(grid_shape, physical_size)
    U, V, W = solve_laplace_uvw(mask, X, Y, Z, iterations=iterations)
    return gyroid_from_harmonic_uvw(U, V, W, mask, L_target)


def generate_harmonic_conformal_lattice(
    stl_path: str | Path,
    lattice_type: str = "Gyroid",
    resolution: float = 0.25,
    unit_cell_size: float = 5.0,
    solid_fraction: float = 0.33,
    wall_thickness_mm: float | None = None,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    iterations: int = 150,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
) -> trimesh.Trimesh:
    """
    Generate an experimental harmonic conformal TPMS lattice inside an arbitrary CAD mesh.

    Solves a discrete Laplace-Beltrami Dirichlet boundary value problem on the voxelized
    CAD domain to produce harmonic coordinates (U, V, W) that flow organically parallel to
    the outer boundary walls.
    """
    from pathlib import Path
    import trimesh
    from graphite.geometry.masking import voxelize_mesh_and_edt
    from graphite.implicit.density_control import tau_from_wall_thickness_mm
    from graphite.implicit.meshing_backends import extract_isosurface
    from graphite.io.mesh_export import export_mesh
    from graphite.math.tpms import evaluate_tpms

    p = Path(stl_path)
    mesh = trimesh.load(str(p), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, padded_max_bound, nx, ny, nz = voxelize_mesh_and_edt(
        mesh, resolution=resolution
    )
    mask = (cad_sdf <= 0.0)

    # Solve discrete harmonic UVW coordinates
    U, V, W = solve_laplace_uvw(mask, X, Y, Z, iterations=iterations)

    L = max(float(unit_cell_size), 0.1)
    k = 2.0 * np.pi / L

    # Evaluate TPMS on curvilinear harmonic coordinate fields
    F = evaluate_tpms(lattice_type, 1.0, U * (k / (2.0 * np.pi)), V * (k / (2.0 * np.pi)), W * (k / (2.0 * np.pi)))

    if wall_thickness_mm is not None and float(wall_thickness_mm) > 0:
        tau_val = float(tau_from_wall_thickness_mm(float(wall_thickness_mm), L).ravel()[0])
    else:
        phi = float(np.clip(float(solid_fraction), 0.01, 0.95))
        tau_val = phi / 1.15

    solid_field = np.abs(F) - tau_val
    core_sdf = np.maximum(solid_field, cad_sdf)
    skin_sdf = np.maximum(cad_sdf, -cad_sdf - float(shell_thickness))

    mode = str(export_mode).strip().lower()
    if mode == "core":
        final_field = core_sdf
    elif mode == "skin":
        final_field = skin_sdf
    elif mode == "combined":
        final_field = np.minimum(core_sdf, skin_sdf)
    else:
        final_field = core_sdf

    iso_res = extract_isosurface(
        final_field,
        spacing=(resolution, resolution, resolution),
        origin=padded_min_bound,
    )
    out_mesh = iso_res.mesh

    if output_path is not None:
        fmts = export_formats if export_formats is not None else ("stl",)
        if isinstance(fmts, str):
            fmts = (fmts,)
        export_mesh(out_mesh, output_path, formats=fmts)

    return out_mesh
