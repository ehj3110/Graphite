"""
Graphite Mesh Repair Module
Provides utilities to repair CAD/STL meshes and ensure they are watertight.
"""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from trimesh import repair as trimesh_repair


def sanitize_cad_mesh_for_sdf(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Strict pre-SDF / pre-VF sanitization (Task 23).

    Always runs hole fill + normal/winding repair before voxelization so
    inside/outside does not invert into a hollow EDT core. Does **not**
    early-return on ``is_watertight`` — watertight meshes can still have
    flipped normals.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")

    m = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=True,
    )

    # Degenerate / duplicate cleanup (API varies by trimesh version)
    try:
        m.remove_degenerate_faces()
    except AttributeError:
        m.update_faces(m.nondegenerate_faces())
    try:
        m.remove_duplicate_faces()
    except AttributeError:
        m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()

    # Mandatory repair pass (even if already watertight)
    trimesh_repair.fill_holes(m)
    try:
        trimesh_repair.fix_winding(m)
    except Exception:
        pass
    try:
        trimesh_repair.fix_inversion(m)
    except Exception:
        pass
    trimesh_repair.fix_normals(m)

    m = trimesh.Trimesh(
        vertices=np.asarray(m.vertices, dtype=np.float64),
        faces=np.asarray(m.faces, dtype=np.int64),
        process=True,
    )

    if not m.is_watertight:
        try:
            trimesh_repair.broken_faces(m)
        except Exception:
            pass
        span = float(np.max(m.extents)) if m.extents is not None else 0.0
        if span > 0.0:
            m.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=4)
        trimesh_repair.fill_holes(m)
        try:
            trimesh_repair.fix_winding(m)
        except Exception:
            pass
        try:
            trimesh_repair.fix_inversion(m)
        except Exception:
            pass
        trimesh_repair.fix_normals(m)
        m = trimesh.Trimesh(
            vertices=np.asarray(m.vertices, dtype=np.float64),
            faces=np.asarray(m.faces, dtype=np.int64),
            process=True,
        )

    # Positive volume = outward normals (required for voxel fill / EDT sign)
    try:
        vol = float(m.volume)
    except Exception:
        vol = 0.0
    if vol < 0.0:
        m.invert()
        trimesh_repair.fix_normals(m)

    if not m.is_watertight:
        warnings.warn(
            "CAD mesh is still not watertight after sanitization; "
            "VF/SDF may still misclassify interior voxels.",
            UserWarning,
        )
    return m


def repair_cad_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Repair a trimesh.Trimesh CAD/STL boundary mesh.

    Delegates to ``sanitize_cad_mesh_for_sdf`` so every caller (VF, EDT, TNP)
    gets a strict normals + holes pass before inside/outside queries.
    """
    return sanitize_cad_mesh_for_sdf(mesh)
