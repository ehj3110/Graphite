"""
Graphite Mesh Repair Module
Provides utilities to repair CAD/STL meshes and ensure they are watertight.
"""

from __future__ import annotations
import trimesh
from trimesh import repair as trimesh_repair
import warnings


def repair_cad_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Repair a trimesh.Trimesh CAD/STL boundary mesh.
    Tries standard watertightness fixes, hole filling, normal repairs,
    and vertex merging.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")

    # 1. Basic duplicate, degenerate, and unreferenced removal via process=True
    m = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    if m.is_watertight:
        return m

    # 2. Standard trimesh hole filling and normal repair
    trimesh_repair.fill_holes(m)
    trimesh_repair.fix_normals(m)
    trimesh_repair.fix_inversion(m)
    m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)

    if m.is_watertight:
        return m

    # 3. Aggressive broken faces and vertex clustering
    trimesh_repair.broken_faces(m)
    m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)

    if not m.is_watertight:
        # Merge vertices with digits_vertex clustering
        span = float(m.extents.max())
        if span > 0:
            m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
            trimesh_repair.fill_holes(m)
            trimesh_repair.fix_normals(m)
            trimesh_repair.fix_inversion(m)
            m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)

    if not m.is_watertight:
        warnings.warn(
            "Mesh is still not watertight after all repair attempts. "
            "Downstream tetrahedral meshing may encounter errors.",
            UserWarning,
        )

    return m
