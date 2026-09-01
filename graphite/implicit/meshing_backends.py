"""
Graphite Implicit Engine - Meshing Backends

This module provides a unified interface for extracting 3D isosurfaces (meshes) 
from implicit 3D scalar fields (voxel grids). It supports standard marching cubes 
as well as PyVista's Flying Edges algorithm for potentially faster extraction, 
with automatic fallback and optional post-processing to ensure watertightness.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import trimesh
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class IsosurfaceExtractionResult:
    """
    Dataclass wrapping the result of an isosurface extraction.

    Attributes
    ----------
    mesh : trimesh.Trimesh
        The extracted 3D mesh.
    backend_requested : str
        The backend that was asked to run.
    backend_used : str
        The backend that actually completed the extraction.
    fallback_used : bool
        True if the requested backend failed and a fallback was triggered.
    fallback_reason : str or None
        The error message if a fallback was triggered.
    runtime_seconds : float
        The time taken for extraction and postprocessing.
    notes : list of str
        Any diagnostic notes generated during the process.
    """
    mesh: trimesh.Trimesh
    backend_requested: str
    backend_used: str
    fallback_used: bool
    fallback_reason: str | None
    runtime_seconds: float
    notes: list[str]


def _postprocess_mesh(mesh: trimesh.Trimesh, fill_holes: bool = True) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if fill_holes:
        trimesh.repair.fill_holes(mesh)
    return mesh


def extract_isosurface_from_image_data(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    level: float = 0.0,
) -> trimesh.Trimesh:
    """
    Extract an isosurface using PyVista's Flying Edges implementation.

    Parameters
    ----------
    field : ndarray
        The 3D scalar field voxel data.
    spacing : tuple of float
        The physical size of each voxel in mm.
    origin : tuple of float
        The physical origin of the voxel grid.
    level : float, optional
        The isovalue to contour at, by default 0.0.

    Returns
    -------
    trimesh.Trimesh
        The extracted isosurface mesh.
    """
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("pyvista is required for pyvista_flying_edges backend") from exc

    nx, ny, nz = field.shape
    img = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing, origin=origin)
    # VTK expects Fortran order flattening for point_data in image grids.
    img.point_data["values"] = np.asarray(field, dtype=np.float32).ravel(order="F")
    surf = img.contour(isosurfaces=[float(level)], scalars="values")
    if surf.n_points == 0 or surf.n_cells == 0:
        raise RuntimeError("pyvista contour returned empty surface")
    tri = surf.extract_surface(algorithm="geometry").triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:4]
    mesh = trimesh.Trimesh(vertices=np.asarray(tri.points), faces=np.asarray(faces), process=True)
    return mesh


def extract_isosurface(
    field: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    *,
    backend: str = "marching_cubes",
    level: float = 0.0,
    postprocess: bool = True,
    fill_holes: bool = True,
    enforce_watertight: bool = False,
) -> IsosurfaceExtractionResult:
    """
    Extract an isosurface mesh from a 3D scalar field.

    Parameters
    ----------
    field : ndarray
        3D numpy array representing the scalar field values at each voxel.
    spacing : tuple of float
        The physical (x, y, z) dimensions of a single voxel in mm.
    origin : tuple of float
        The physical (x, y, z) coordinate of the minimum corner of the grid.
    backend : str, optional
        Algorithm to use: 'marching_cubes' (scikit-image) or 'pyvista_flying_edges', 
        by default "marching_cubes".
    level : float, optional
        The contour level to extract, by default 0.0 (for level-set zero-crossings).
    postprocess : bool, optional
        Whether to clean up the mesh (remove unreferenced vertices, fix normals, 
        merge duplicates), by default True.
    fill_holes : bool, optional
        Whether to attempt to fill non-manifold holes during postprocessing, 
        by default True.
    enforce_watertight : bool, optional
        If True and the mesh is not watertight after extraction, runs a voxel 
        rewrap pass (voxelize and remesh) to force watertightness. By default False.

    Returns
    -------
    IsosurfaceExtractionResult
        A dataclass containing the resulting `trimesh.Trimesh` and extraction metadata.

    Raises
    ------
    ValueError
        If the field is not 3D or the backend is unsupported.
    """
    if field.ndim != 3:
        raise ValueError("field must be a 3D array")
    if backend not in {"marching_cubes", "pyvista_flying_edges"}:
        raise ValueError("backend must be 'marching_cubes' or 'pyvista_flying_edges'")

    t0 = time.perf_counter()
    backend_used = backend
    fallback_used = False
    fallback_reason: str | None = None
    notes: list[str] = []

    def _mc_mesh() -> trimesh.Trimesh:
        verts, faces, _normals, _values = marching_cubes(
            np.asarray(field, dtype=np.float32),
            level=float(level),
            spacing=spacing,
        )
        verts[:, 0] += origin[0]
        verts[:, 1] += origin[1]
        verts[:, 2] += origin[2]
        return trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if backend == "pyvista_flying_edges":
        try:
            mesh = extract_isosurface_from_image_data(field, spacing=spacing, origin=origin, level=level)
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"pyvista backend failed: {exc}"
            backend_used = "marching_cubes"
            notes.append("Fell back to marching_cubes backend.")
            mesh = _mc_mesh()
    else:
        mesh = _mc_mesh()

    if postprocess:
        mesh = _postprocess_mesh(mesh, fill_holes=fill_holes)

    if enforce_watertight and not mesh.is_watertight:
        # Last-resort hardening path: voxel rewrap + marching cubes.
        pitch = min(spacing)
        vox = mesh.voxelized(pitch=pitch).fill()
        hard = vox.marching_cubes
        # Preserve world transform from voxel index-space back to mesh space.
        if hasattr(vox, "transform") and vox.transform is not None:
            hard.vertices = trimesh.transform_points(hard.vertices, vox.transform)
        hard = _postprocess_mesh(hard, fill_holes=True)
        if hard.is_watertight:
            mesh = hard
            notes.append("Applied voxel hardening pass to enforce watertight mesh.")
        else:
            notes.append("Watertight enforcement attempted but mesh remains non-watertight.")

    elapsed = time.perf_counter() - t0
    return IsosurfaceExtractionResult(
        mesh=mesh,
        backend_requested=backend,
        backend_used=backend_used,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        runtime_seconds=float(elapsed),
        notes=notes,
    )

