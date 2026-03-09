"""
Graphite Geometry Module — Strut Solid Generation and Boundary Trimming

This module converts a topology skeleton (nodes + struts) into explicit 3D
solid geometry using manifold3d CSG operations. It supports:
    - Cylinder generation for each strut segment
    - Optional stress-relief spheres at nodes
    - Batch union via Manifold.compose
    - Optional boundary intersection trim against an input boundary mesh
    - Conversion back to trimesh for export and downstream processing
"""

from __future__ import annotations

import manifold3d
import numpy as np
import trimesh


def _trimesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
    """
    Convert a trimesh.Trimesh into a manifold3d.Manifold.

    manifold3d expects:
        - vert_properties: float32 vertex coordinates, shape (N, 3)
        - tri_verts: uint32 triangle indices, shape (M, 3)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}.")

    if mesh.faces.shape[0] == 0:
        raise ValueError("Cannot convert empty trimesh (zero faces) to Manifold.")

    manifold_mesh = manifold3d.Mesh(
        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
    )
    return manifold3d.Manifold(manifold_mesh)


def _manifold_to_trimesh(manifold_mesh: manifold3d.Manifold) -> trimesh.Trimesh:
    """
    Convert manifold3d.Manifold into trimesh.Trimesh.
    """
    raw_mesh = manifold_mesh.to_mesh()
    vertices = np.asarray(raw_mesh.vert_properties, dtype=np.float64)
    faces = np.asarray(raw_mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def generate_geometry(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float,
    boundary_mesh: trimesh.Trimesh | None = None,
    add_spheres: bool = False,
) -> trimesh.Trimesh:
    """
    Generate explicit lattice geometry from topology nodes + struts.

    Args:
        nodes: (N, 3) node coordinates.
        struts: (S, 2) strut endpoint indices into `nodes`.
        strut_radius: Radius for strut cylinders.
        boundary_mesh: Optional trimesh boundary for boolean trim intersection.
        add_spheres: If True, add a sphere at every node with radius 1.2 * strut_radius.

    Returns:
        Final lattice mesh as trimesh.Trimesh.

    Raises:
        ValueError: If inputs are invalid or no geometry can be generated.
        TypeError: If boundary_mesh is provided with invalid type.
    """
    # -------------------------------------------------------------------------
    # Input normalization and validation
    # -------------------------------------------------------------------------
    nodes_np = np.asarray(nodes, dtype=np.float64)
    struts_np = np.asarray(struts, dtype=np.int64)

    if nodes_np.ndim != 2 or nodes_np.shape[1] != 3:
        raise ValueError(f"`nodes` must have shape (N, 3); got {nodes_np.shape}.")
    if struts_np.ndim != 2 or struts_np.shape[1] != 2:
        raise ValueError(f"`struts` must have shape (S, 2); got {struts_np.shape}.")
    if strut_radius <= 0:
        raise ValueError("`strut_radius` must be > 0.")
    if boundary_mesh is not None and not isinstance(boundary_mesh, trimesh.Trimesh):
        raise TypeError(
            f"`boundary_mesh` must be trimesh.Trimesh or None; got {type(boundary_mesh)}."
        )

    if struts_np.shape[0] == 0:
        raise ValueError("Cannot generate geometry: `struts` is empty.")

    # -------------------------------------------------------------------------
    # Step 1/2: Create cylinders for every strut and optional node spheres
    # -------------------------------------------------------------------------
    manifold_objects: list[manifold3d.Manifold] = []

    # Looping here is acceptable because each item is an individual primitive
    # CSG construction call; the heavy set operations are batched afterward.
    for a_idx, b_idx in struts_np:
        start = nodes_np[a_idx]
        end = nodes_np[b_idx]
        segment_vec = end - start
        segment_len = float(np.linalg.norm(segment_vec))

        # Skip degenerate segments (zero or near-zero length) to avoid invalid
        # primitive generation and downstream boolean instability.
        if segment_len <= 1e-12:
            continue

        # trimesh can generate a cylinder directly between two endpoints.
        # This avoids manual rotation matrix bookkeeping for axis alignment.
        cylinder_tm = trimesh.creation.cylinder(
            radius=float(strut_radius),
            segment=np.vstack((start, end)),
            sections=16,
        )
        manifold_objects.append(_trimesh_to_manifold(cylinder_tm))

    if add_spheres:
        fillet_radius = float(strut_radius) * 1.2
        for xyz in nodes_np:
            # manifold3d sphere primitive is centered at origin, then translated.
            sphere = manifold3d.Manifold.sphere(fillet_radius).translate(tuple(xyz))
            manifold_objects.append(sphere)

    if not manifold_objects:
        raise ValueError(
            "No valid primitives were created from the provided topology "
            "(all struts may be degenerate)."
        )

    # -------------------------------------------------------------------------
    # Step 3: Batch union all primitives
    # -------------------------------------------------------------------------
    united_lattice = manifold3d.Manifold.compose(manifold_objects)

    # -------------------------------------------------------------------------
    # Step 4: Optional boundary trim (boolean intersection)
    # -------------------------------------------------------------------------
    if boundary_mesh is not None:
        boundary_manifold = _trimesh_to_manifold(boundary_mesh)
        united_lattice = united_lattice ^ boundary_manifold

    # -------------------------------------------------------------------------
    # Step 5: Convert final manifold back to trimesh and return
    # -------------------------------------------------------------------------
    final_mesh = _manifold_to_trimesh(united_lattice)
    return final_mesh
