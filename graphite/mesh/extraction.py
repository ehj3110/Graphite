"""
Graphite Mesh - High-Performance Isosurface Extraction Engine

This module implements isosurface extraction using the multi-threaded Flying Edges
algorithm via PyVista/VTK, with an automatic fallback to scikit-image's marching cubes
if dependencies are unavailable.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

import numpy as np
import trimesh
from skimage.measure import marching_cubes

logger = logging.getLogger(__name__)


def _postprocess_extracted_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clean topology, eliminate unreferenced elements, and orient normals outward."""
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    trimesh.repair.fix_normals(mesh)
    try:
        if float(mesh.volume) < 0.0:
            mesh.invert()
            trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh


def extract_isosurface_flying_edges(
    field: np.ndarray,
    origin: tuple[float, float, float] | Sequence[float],
    spacing: tuple[float, float, float] | Sequence[float],
    level: float = 0.0,
) -> trimesh.Trimesh:
    """
    Extract a 3D isosurface mesh from a regular scalar voxel field using Flying Edges.

    Utilizes PyVista's Flying Edges 3D contouring algorithm (`vtkFlyingEdges3D`),
    which processes voxel rows in parallel with significantly higher throughput
    and lower memory overhead than classic Marching Cubes.

    If PyVista/VTK is unavailable, falls back to `skimage.measure.marching_cubes`
    with a warning.

    Parameters
    ----------
    field : np.ndarray
        3D numpy array representing the implicit scalar field (indexing='ij').
    origin : tuple of 3 floats
        World-coordinate position (ox, oy, oz) in millimeters of voxel [0, 0, 0].
    spacing : tuple of 3 floats
        Physical voxel pitch (dx, dy, dz) in millimeters.
    level : float, optional
        Isovalue contour threshold, by default 0.0.

    Returns
    -------
    trimesh.Trimesh
        Extracted surface mesh in world millimeters, with cleaned face topology
        and outward-facing normals.

    Raises
    ------
    ValueError
        If field is not a 3D array or spacing contains non-positive values.
    """
    if field.ndim != 3:
        raise ValueError(f"field must be a 3D array, got {field.ndim}D shape {field.shape}")

    spacing_tuple = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    if any(s <= 0.0 for s in spacing_tuple):
        raise ValueError(f"spacing components must be positive, got {spacing_tuple}")

    origin_tuple = (float(origin[0]), float(origin[1]), float(origin[2]))
    level_f = float(level)

    # Ensure contiguous float32 memory to avoid intermediate copies
    field_f32 = np.ascontiguousarray(field, dtype=np.float32)
    nx, ny, nz = field_f32.shape

    # Attempt primary production backend: PyVista / VTK Flying Edges
    try:
        import pyvista as pv

        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=spacing_tuple,
            origin=origin_tuple,
        )
        # In VTK ImageData, points vary fastest along X (axis 0), then Y (axis 1), then Z (axis 2).
        # Flattening with Fortran order ('F') maps NumPy indexing='ij' arrays directly to VTK point order.
        grid.point_data["values"] = field_f32.ravel(order="F")

        surf = grid.contour(
            isosurfaces=[level_f],
            scalars="values",
            method="flying_edges",
        )

        if surf.n_points == 0 or surf.n_cells == 0:
            return trimesh.Trimesh()

        # Extract surface triangles
        tri = surf.extract_surface(algorithm="geometry").triangulate()
        if tri.n_cells == 0:
            return trimesh.Trimesh()

        # PyVista/VTK triangle cell connectivity: shape (n_tri, 4), where first col is count (3)
        faces = tri.faces.reshape(-1, 4)[:, 1:4]
        mesh = trimesh.Trimesh(
            vertices=np.asarray(tri.points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=True,
        )
        return _postprocess_extracted_mesh(mesh)

    except Exception as exc:
        msg = (
            f"Flying Edges extraction backend unavailable ({exc}); "
            "falling back to skimage.measure.marching_cubes."
        )
        logger.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

        try:
            verts, faces, _normals, _values = marching_cubes(
                field_f32,
                level=level_f,
                spacing=spacing_tuple,
            )
            verts[:, 0] += origin_tuple[0]
            verts[:, 1] += origin_tuple[1]
            verts[:, 2] += origin_tuple[2]

            mesh = trimesh.Trimesh(
                vertices=verts.astype(np.float64),
                faces=faces.astype(np.int64),
                process=True,
            )
            return _postprocess_extracted_mesh(mesh)
        except RuntimeError:
            # Marching cubes did not find any zero-crossing surface
            return trimesh.Trimesh()
