"""
Graphite Geometry Module — Strut Solid Generation and Boundary Trimming

This module converts a topology skeleton (nodes + struts) into explicit 3D
solid geometry using manifold3d CSG operations. It supports:
    - Cylinder generation for each strut segment (per-strut radii: r = L * k)
    - Optional stress-relief spheres at nodes
    - Batch union via Manifold.compose
    - Optional boundary intersection trim against an input boundary mesh
    - Conversion back to trimesh for export and downstream processing

Performance (Day 2):
- return_manifold=True: skip to_trimesh, return (Manifold, volume). Use
  manifold_to_trimesh() for final export. ~200x speedup for volume iteration.
- Boundary mesh must be watertight (process=True) or Boolean returns empty.
- Performance Audit: timers for cylinder creation, union, intersect, convert.
"""

from __future__ import annotations

import time

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


def manifold_to_trimesh(manifold_mesh: manifold3d.Manifold) -> trimesh.Trimesh:
    """Convert manifold3d.Manifold to trimesh.Trimesh (for final export)."""
    return _manifold_to_trimesh(manifold_mesh)


def export_lattice_to_stl(
    nodes: np.ndarray,
    struts: np.ndarray,
    thickness: float = 0.5,
    output_filename: str | None = None,
) -> None:
    """
    Build lattice geometry (spheres + cylinders) and export to STL.

    Parameters
    ----------
    nodes : ndarray, shape (N, 3)
        Node coordinates.
    struts : ndarray, shape (S, 2)
        Strut endpoint indices into nodes.
    thickness : float
        Strut diameter; node spheres use radius = thickness / 2.
    output_filename : str | None
        Output path. If None, exports to "lattice_export.stl".
    """
    if output_filename is None:
        output_filename = "lattice_export.stl"

    mesh = generate_geometry(
        nodes,
        struts,
        strut_radius=thickness / 2.0,
        boundary_mesh=None,
        add_spheres=True,
        crop_to_boundary=False,
    )
    mesh.export(output_filename)


def generate_geometry(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float | np.ndarray,
    boundary_mesh: trimesh.Trimesh | None = None,
    add_spheres: bool = False,
    crop_to_boundary: bool = True,
    return_manifold: bool = False,
) -> trimesh.Trimesh | tuple[trimesh.Trimesh, float] | tuple[manifold3d.Manifold, float]:
    """
    Generate explicit lattice geometry from topology nodes + struts.

    Args:
        nodes: (N, 3) node coordinates.
        struts: (S, 2) strut endpoint indices into `nodes`.
        strut_radius: Single radius (float) or per-strut radii (S,) array for adaptive thickness.
        boundary_mesh: Optional trimesh boundary for boolean trim intersection.
        add_spheres: If True, add a sphere at every node with radius 1.2 * max(strut_radius).
        crop_to_boundary: If True, intersect lattice with boundary (trim). If False,
            return raw cylinder union (pipe-style); volume is still computed from
            intersection when boundary_mesh is provided.
        return_manifold: If True and crop_to_boundary and boundary_mesh, skip trimesh
            conversion and return (Manifold, volume). Use manifold_to_trimesh() for
            export. Saves ~40s per call when only volume is needed.

    Returns:
        If crop_to_boundary=True: trimesh.Trimesh (cropped lattice).
        If crop_to_boundary=False and boundary_mesh is not None: tuple of
            (trimesh.Trimesh, float) — raw lattice mesh and intersection volume.
        If crop_to_boundary=False and boundary_mesh is None: trimesh.Trimesh.

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
    if struts_np.shape[0] == 0:
        raise ValueError("Cannot generate geometry: `struts` is empty.")

    radii = np.atleast_1d(np.asarray(strut_radius, dtype=np.float64))
    if radii.ndim == 0:
        radii = np.full(struts_np.shape[0], float(radii))
    elif radii.shape[0] == 1:
        radii = np.full(struts_np.shape[0], float(radii[0]))
    elif radii.shape[0] != struts_np.shape[0]:
        raise ValueError(
            f"strut_radius per-strut array length {radii.shape[0]} must match "
            f"struts {struts_np.shape[0]}."
        )
    if np.any(radii <= 0):
        raise ValueError("All strut radii must be > 0.")

    if boundary_mesh is not None and not isinstance(boundary_mesh, trimesh.Trimesh):
        raise TypeError(
            f"`boundary_mesh` must be trimesh.Trimesh or None; got {type(boundary_mesh)}."
        )

    # -------------------------------------------------------------------------
    # Step 1/2: Create cylinders for every strut and optional node spheres
    # -------------------------------------------------------------------------
    manifold_objects: list[manifold3d.Manifold] = []
    t0_cyl = time.perf_counter()

    for i, (a_idx, b_idx) in enumerate(struts_np):
        start = nodes_np[a_idx]
        end = nodes_np[b_idx]
        segment_vec = end - start
        segment_len = float(np.linalg.norm(segment_vec))

        if segment_len <= 1e-12:
            continue

        r_i = float(radii[i])
        cylinder_tm = trimesh.creation.cylinder(
            radius=r_i,
            segment=np.vstack((start, end)),
            sections=16,
        )
        manifold_objects.append(_trimesh_to_manifold(cylinder_tm))

    if add_spheres:
        fillet_radius = float(np.max(radii)) * 1.2
        for xyz in nodes_np:
            # manifold3d sphere primitive is centered at origin, then translated.
            sphere = manifold3d.Manifold.sphere(fillet_radius).translate(tuple(xyz))
            manifold_objects.append(sphere)

    if not manifold_objects:
        raise ValueError(
            "No valid primitives were created from the provided topology "
            "(all struts may be degenerate)."
        )

    t_cylinders = time.perf_counter() - t0_cyl

    # -------------------------------------------------------------------------
    # Step 3: Batch union all primitives
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    united_lattice = manifold3d.Manifold.compose(manifold_objects)
    t_union = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Step 4: Optional boundary trim (boolean intersection)
    # -------------------------------------------------------------------------
    t_intersect = 0.0
    if boundary_mesh is not None:
        t0 = time.perf_counter()
        boundary_manifold = _trimesh_to_manifold(boundary_mesh)
        if crop_to_boundary:
            united_lattice = united_lattice ^ boundary_manifold
        else:
            # For Vf: compute intersection volume; export mesh stays raw
            trimmed_manifold = united_lattice ^ boundary_manifold
            trimmed_volume = float(trimmed_manifold.volume())
        t_intersect = time.perf_counter() - t0

    # -------------------------------------------------------------------------
    # Step 5: Volume from manifold (no conversion needed for volume)
    # -------------------------------------------------------------------------
    if boundary_mesh is not None:
        if crop_to_boundary:
            manifold_volume = float(united_lattice.volume())
        else:
            manifold_volume = trimmed_volume
    else:
        manifold_volume = float(united_lattice.volume())

    # -------------------------------------------------------------------------
    # Step 6: Convert to trimesh only when needed (skip if return_manifold)
    # -------------------------------------------------------------------------
    t_convert = 0.0
    if return_manifold and crop_to_boundary and boundary_mesh is not None:
        # Skip conversion for volume-only pass; caller uses manifold_to_trimesh() for export
        t_convert = 0.0
        print("\n--- Performance Audit ---")
        print(f"  Cylinder creation:       {t_cylinders:.3f} s")
        print(f"  Manifold.compose (union): {t_union:.3f} s")
        print(f"  Manifold.intersect (clip): {t_intersect:.3f} s")
        print(f"  Manifold.to_trimesh:      {t_convert:.3f} s (skipped)")
        print(f"  Manifold.volume():       {manifold_volume:.2f}")
        return united_lattice, manifold_volume

    t0 = time.perf_counter()
    final_mesh = _manifold_to_trimesh(united_lattice)
    t_convert = time.perf_counter() - t0

    # Performance Audit
    print("\n--- Performance Audit ---")
    print(f"  Cylinder creation:       {t_cylinders:.3f} s")
    print(f"  Manifold.compose (union): {t_union:.3f} s")
    print(f"  Manifold.intersect (clip): {t_intersect:.3f} s")
    print(f"  Manifold.to_trimesh:      {t_convert:.3f} s")
    print(f"  Manifold.volume():       {manifold_volume:.2f}")
    if boundary_mesh is not None:
        return final_mesh, manifold_volume
    return final_mesh
