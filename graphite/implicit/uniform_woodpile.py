"""
Graphite Implicit Engine - Uniform Woodpile

This module provides functionality to generate implicit woodpile structures 
(both true woodpiles with alternating shifted layers, and simple cross-hatches). 
It evaluates the analytical woodpile field and intersects it with a CAD 
Signed Distance Field (SDF).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_cylinder_slab_and_edt, voxelize_mesh_and_edt
from graphite.io.mesh_export import export_mesh
from graphite.math.woodpile import evaluate_woodpile


def _mesh_woodpile_from_fields(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cad_sdf: np.ndarray,
    padded_min_bound: np.ndarray,
    resolution: float,
    pore_size: float,
    true_woodpile: bool,
    invert_solids: bool,
    origin_x: float,
    origin_y: float,
    flip_layer_parity: bool,
    swap_xy: bool,
    center_origin: bool,
    output_path: str | Path | None,
    export_formats: tuple[str, ...] | str | None,
    log_label: str,
) -> trimesh.Trimesh:
    woodpile_field = evaluate_woodpile(
        X,
        Y,
        Z,
        pore_size=pore_size,
        true_woodpile=true_woodpile,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        flip_layer_parity=bool(flip_layer_parity),
        swap_xy=bool(swap_xy),
    )
    if invert_solids:
        woodpile_field = -woodpile_field
    final_field = np.maximum(woodpile_field, cad_sdf)

    t0 = time.perf_counter()
    verts, faces, _normals, _values = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    dt = time.perf_counter() - t0

    verts = verts + padded_min_bound
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path is not None:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    mode = "true-woodpile" if true_woodpile else "cross-hatch"
    inv = " (inverted solids)" if invert_solids else ""
    print(
        f"Uniform woodpile ({mode}{inv}) {log_label}: pore={pore_size:.3f}mm, "
        f"res={resolution:.3f}mm, mc={dt:.2f}s, faces={len(mesh_out.faces):,}"
    )
    return mesh_out


def generate_uniform_woodpile_cylinder_slab(
    *,
    radius_mm: float,
    z0_mm: float,
    z1_mm: float,
    pore_size: float = 1.0,
    true_woodpile: bool = True,
    resolution: float = 0.05,
    center_origin: bool = False,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
    invert_solids: bool = False,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
) -> trimesh.Trimesh:
    """Woodpile inside a short Z cylinder slab (analytic EDT, no STL voxelize)."""
    X, Y, Z, cad_sdf, padded_min_bound, _max_bound, _nx, _ny, _nz = (
        voxelize_cylinder_slab_and_edt(radius_mm, z0_mm, z1_mm, resolution)
    )
    return _mesh_woodpile_from_fields(
        X=X,
        Y=Y,
        Z=Z,
        cad_sdf=cad_sdf,
        padded_min_bound=padded_min_bound,
        resolution=float(resolution),
        pore_size=float(pore_size),
        true_woodpile=bool(true_woodpile),
        invert_solids=bool(invert_solids),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        flip_layer_parity=bool(flip_layer_parity),
        swap_xy=bool(swap_xy),
        center_origin=bool(center_origin),
        output_path=output_path,
        export_formats=export_formats,
        log_label=f"cylinder r={radius_mm:g}mm z=[{z0_mm:g},{z1_mm:g}]",
    )


def generate_uniform_woodpile(
    stl_path: str | Path,
    pore_size: float = 1.0,
    true_woodpile: bool = True,
    resolution: float = 0.05,
    center_origin: bool = False,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
    invert_solids: bool = False,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
) -> trimesh.Trimesh:
    """
    Generate a conformal implicit woodpile lattice inside an input STL.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    pore_size : float, optional
        The side length of the square pores in the transverse plane, by default 1.0.
    true_woodpile : bool, optional
        If True, layers alternate X, Y, X (shifted), Y (shifted). If False, 
        generates a simple cross-hatch (X, Y, X, Y), by default True.
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.05.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.
    invert_solids : bool, optional
        If True, negate the woodpile scalar field before union with ``cad_sdf``, so
        former void becomes solid and former beams become void (complement **inside**
        the mesh clipping). Cross-hatch then exposes a pore along axes that were
        previously strut centers (e.g. mid-plane of the narrow dimension), by default False.

    Returns
    -------
    trimesh.Trimesh
        The meshed woodpile lattice.
    """
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    return _mesh_woodpile_from_fields(
        X=X,
        Y=Y,
        Z=Z,
        cad_sdf=cad_sdf,
        padded_min_bound=padded_min_bound,
        resolution=float(resolution),
        pore_size=float(pore_size),
        true_woodpile=bool(true_woodpile),
        invert_solids=bool(invert_solids),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        flip_layer_parity=bool(flip_layer_parity),
        swap_xy=bool(swap_xy),
        center_origin=bool(center_origin),
        output_path=output_path,
        export_formats=export_formats,
        log_label=f"from '{stl_path.name}'",
    )
