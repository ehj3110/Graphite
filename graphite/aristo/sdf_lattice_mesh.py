"""
SDF-based Voxel Isosurface Mesher for Aristo FEA.

Rasterizes surface meshes to a 3D Signed Distance Field (SDF), extracts a manifold surface
via Marching Cubes, and passes the clean isosurface to Gmsh.
Bypasses surface topology issues (coplanar/duplicate facets) at CAD boolean interfaces.
"""

from __future__ import annotations
import math
import time
import warnings
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes

from graphite.aristo.aristo_config import AristoConfig


def compute_voxel_sdf(
    mesh: trimesh.Trimesh,
    voxel_pitch: float,
    padding_voxels: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rasterize mesh interior to a signed distance field on a regular voxel grid.

    Uses trimesh.voxelized for fast 3D rasterization.

    Returns
    -------
    sdf_volume : np.ndarray (Nx, Ny, Nz) float32
        SDF grid in mm. Negative values = inside solid, positive = outside.
    origin : np.ndarray (3,) float64
        Min (x, y, z) corner of grid [0,0,0] in world coordinates (mm).
    pitch : float
        Voxel pitch in mm.
    """
    t0 = time.time()
    vox = mesh.voxelized(voxel_pitch)
    inside_mask = vox.matrix

    # Pad boundary voxels to ensure closed marching cubes surface
    if padding_voxels > 0:
        inside_mask = np.pad(inside_mask, padding_voxels, mode="constant", constant_values=False)

    # Compute Euclidean distance transforms
    dist_outside = distance_transform_edt(~inside_mask) * voxel_pitch
    dist_inside = distance_transform_edt(inside_mask) * voxel_pitch

    # Negative inside, positive outside
    sdf = dist_outside - dist_inside
    
    # Calculate grid origin in world coordinates
    origin = vox.translation - padding_voxels * voxel_pitch

    print(
        f"  [SDF] Voxel grid {sdf.shape} at pitch {voxel_pitch:.2f}mm "
        f"built in {time.time() - t0:.2f}s",
        flush=True
    )
    return sdf.astype(np.float32), origin, float(voxel_pitch)


def extract_isosurface(
    sdf_volume: np.ndarray,
    origin: np.ndarray,
    pitch: float,
    isovalue: float = 0.0,
) -> trimesh.Trimesh:
    """
    Extract zero-isosurface using Marching Cubes and convert to trimesh.Trimesh.
    """
    t0 = time.time()
    verts, faces, normals, values = marching_cubes(
        sdf_volume, level=isovalue, spacing=(pitch, pitch, pitch)
    )
    world_verts = verts + origin
    mc_mesh = trimesh.Trimesh(vertices=world_verts, faces=faces, process=True)
    print(
        f"  [MarchingCubes] Extracted clean surface ({len(mc_mesh.vertices)} verts, "
        f"{len(mc_mesh.faces)} faces) in {time.time() - t0:.2f}s",
        flush=True
    )
    return mc_mesh


def mesh_solid_via_sdf(
    mesh: trimesh.Trimesh,
    fea_mesh_resolution: float,
    *,
    config: AristoConfig | None = None,
    voxel_pitch: float | None = None,
    silent: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: trimesh -> SDF voxel grid -> MC isosurface -> Gmsh volume tet mesh.
    """
    h = float(fea_mesh_resolution)
    if voxel_pitch is None:
        if config is not None and getattr(config, "fea_sdf_voxel_pitch", None) is not None:
            voxel_pitch = float(config.fea_sdf_voxel_pitch)
        else:
            # Recommended default for fast, accurate lattice FEA meshing (1.0mm pitch)
            voxel_pitch = min(h / 2.0, 1.0)

    sdf, origin, pitch = compute_voxel_sdf(mesh, voxel_pitch=voxel_pitch)
    clean_surface = extract_isosurface(sdf, origin, pitch, isovalue=0.0)

    from graphite.aristo.aristo_solver import _generate_fea_mesh_single_surface

    print("  [Gmsh] Generating linear tetrahedral volume mesh from clean SDF surface...", flush=True)
    t0 = time.time()
    nodes, elems, surf = _generate_fea_mesh_single_surface(clean_surface, h, silent=silent)
    print(f"  [Gmsh] Produced {len(nodes)} nodes, {len(elems)} tets in {time.time() - t0:.2f}s", flush=True)
    return nodes, elems, surf
