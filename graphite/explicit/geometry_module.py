"""
Graphite Geometry Module — Strut Solid Generation and Boundary Trimming

This module converts a topology skeleton (nodes + struts) into explicit 3D
solid geometry using manifold3d CSG operations. It supports:
    - Native ``Manifold.cylinder`` / ``Manifold.cube`` strut solids + ``transform``
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


def _rotation_align_local_z_to_unit(axis_unit: np.ndarray) -> np.ndarray:
    """
    Right-handed rotation R (3x3): local +z maps to ``axis_unit`` (third column).
    Manifold ``cylinder`` is centered on z when ``center=True``.
    """
    t = np.asarray(axis_unit, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(t))
    if n < 1e-15:
        raise ValueError("Degenerate strut axis (zero length).")
    t = t / n
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, t))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(up, t)
    un = float(np.linalg.norm(u))
    if un < 1e-12:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(up, t)
        un = float(np.linalg.norm(u))
    u = u / un
    v = np.cross(t, u)
    return np.column_stack((u, v, t))


def _affine_rows_from_R_t(R: np.ndarray, translation: np.ndarray) -> list[list[float]]:
    """3x4 row-major affine for ``Manifold.transform`` (rotation then translation)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    return [
        [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(t[0])],
        [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(t[1])],
        [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(t[2])],
    ]


def _manifold_cylinder_between(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    *,
    trim_radius: float = 0.0,
    circular_segments: int = 16,
) -> manifold3d.Manifold | None:
    """Cylinder along ``start``→``end``; optional trim of ``trim_radius`` from each end."""
    p0 = np.asarray(start, dtype=np.float64).reshape(3)
    p1 = np.asarray(end, dtype=np.float64).reshape(3)
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    if length <= 1e-12:
        return None
    t_dir = seg / length
    tr = float(trim_radius)
    if tr > 0.0 and length > 2.0 * tr:
        p0 = p0 + t_dir * tr
        p1 = p1 - t_dir * tr
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length <= 1e-12:
            return None
        t_dir = seg / length
    r = float(radius)
    mid = 0.5 * (p0 + p1)
    R = _rotation_align_local_z_to_unit(t_dir)
    cyl = manifold3d.Manifold.cylinder(
        height=length,
        radius_low=r,
        radius_high=r,
        circular_segments=int(circular_segments),
        center=True,
    )
    return cyl.transform(_affine_rows_from_R_t(R, mid))


def _append_joint_spheres(
    parts: list[manifold3d.Manifold],
    points: np.ndarray,
    radius: float,
    *,
    round_decimals: int = 4,
) -> int:
    """Union fillet spheres at unique points; returns count added."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0 or radius <= 0.0:
        return 0
    keys = np.round(pts, int(round_decimals))
    _, idx = np.unique(keys, axis=0, return_index=True)
    n = 0
    for i in idx:
        parts.append(
            manifold3d.Manifold.sphere(float(radius)).translate(
                tuple(float(x) for x in pts[int(i)])
            )
        )
        n += 1
    return n


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


# Public aliases for reusable manifold3d primitives and conversion helpers
manifold_cylinder_between = _manifold_cylinder_between
rotation_align_local_z_to_unit = _rotation_align_local_z_to_unit
affine_rows_from_R_t = _affine_rows_from_R_t
trimesh_to_manifold = _trimesh_to_manifold


def _skin_strut_ribbon_frame(
    boundary_mesh: trimesh.Trimesh,
    start: np.ndarray,
    end: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    Local frame for a skin strut ribbon: width w (tangent), thickness n (outward),
    axis t along the strut. Returns (w_unit, n_unit, t_unit, length).
    """
    t = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    L = float(np.linalg.norm(t))
    if L < 1e-12:
        return None
    t = t / L
    n_sum = np.asarray(n0, dtype=np.float64) + np.asarray(n1, dtype=np.float64)
    n_n = float(np.linalg.norm(n_sum))
    if n_n < 1e-12:
        n_avg = np.asarray(n0, dtype=np.float64)
        n_n = float(np.linalg.norm(n_avg))
        if n_n < 1e-12:
            return None
        n_avg = n_avg / n_n
    else:
        n_avg = n_sum / n_n
    # Thickness axis: outward normal projected into plane perpendicular to strut tangent.
    n_use = n_avg - float(np.dot(n_avg, t)) * t
    n_n = float(np.linalg.norm(n_use))
    if n_n < 1e-9:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(up, t))) > 0.95:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n_use = np.cross(t, up)
        n_n = float(np.linalg.norm(n_use))
        if n_n < 1e-12:
            return None
    n_use = n_use / n_n
    mid = 0.5 * (np.asarray(start, dtype=np.float64) + np.asarray(end, dtype=np.float64))
    outward_hint = mid - np.asarray(boundary_mesh.centroid, dtype=np.float64)
    if float(np.dot(n_use, outward_hint)) < 0.0:
        n_use = -n_use
    w = np.cross(n_use, t)
    w_n = float(np.linalg.norm(w))
    if w_n < 1e-12:
        return None
    w = w / w_n
    if float(np.dot(np.cross(w, n_use), t)) < 0.0:
        w = -w
    return w, n_use, t, L


def generate_surface_skin_ribbon_mesh(
    boundary_mesh: trimesh.Trimesh,
    nodes_skin: np.ndarray,
    skin_struts: np.ndarray,
    strut_radius: float,
    return_manifold: bool = False,
) -> trimesh.Trimesh | manifold3d.Manifold | None:
    """
    Sweep a flattened box profile along each skin strut (no boolean crop).

    Profile (local): width ``2 * strut_radius`` (along surface binormal w),
    thickness ``strut_radius`` (along averaged face normal), length = segment.

    Skin normals: one batched ``closest_point`` query on all segment **midpoints**
    (vectorized), then ``face_normals[triangle_ids]``. Primitives use native
    ``Manifold.cube`` + ``transform``; a single ``Manifold.compose`` unions ribbons.
    """
    pts = np.asarray(nodes_skin, dtype=np.float64)
    segs = np.asarray(skin_struts, dtype=np.int64)
    if segs.size == 0:
        if return_manifold:
            return None
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    r = float(strut_radius)
    if r <= 0:
        raise ValueError("strut_radius must be > 0 for skin ribbons.")

    ea = pts[segs[:, 0]]
    eb = pts[segs[:, 1]]
    midpoints = 0.5 * (ea + eb)
    _, _, tri_ids = trimesh.proximity.closest_point(boundary_mesh, midpoints)
    tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
    fn = np.asarray(boundary_mesh.face_normals, dtype=np.float64)
    n_mid = fn[tri_ids]

    manifolds: list[manifold3d.Manifold] = []
    for si in range(len(segs)):
        a, b = int(segs[si, 0]), int(segs[si, 1])
        start = pts[a]
        end = pts[b]
        frame = _skin_strut_ribbon_frame(
            boundary_mesh, start, end, n_mid[si], n_mid[si]
        )
        if frame is None:
            continue
        w, n_use, t_dir, L = frame
        if L <= 1e-12:
            continue
        # Box: local x = width, y = thickness, z = length (aligned with strut axis).
        R = np.column_stack((w, n_use, t_dir))
        mid = 0.5 * (start + end)
        box = manifold3d.Manifold.cube([2.0 * r, r, L], center=True)
        manifolds.append(box.transform(_affine_rows_from_R_t(R, mid)))

    if not manifolds:
        if return_manifold:
            return None
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    united = manifold3d.Manifold.compose(manifolds)
    if return_manifold:
        return united
    return _manifold_to_trimesh(united)


def generate_decoupled_core_and_ribbed_skin(
    boundary_mesh: trimesh.Trimesh,
    nodes_core: np.ndarray,
    struts_core: np.ndarray,
    nodes_skin: np.ndarray,
    skin_struts: np.ndarray,
    strut_radius: float,
    synthesis_report: dict[str, object] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """
    Route 3 decoupled export: boolean-cropped cylindrical **core** (volume + bridges)
    plus **uncropped** ribbon skin, then **manifold3d** boolean union of both solids.

    Core uses ``generate_geometry`` (native ``Manifold.cylinder`` + compose + ^ trim).
    Skin uses ``generate_surface_skin_ribbon_mesh`` (native ``Manifold.cube``, no crop).
    Final merge: ``Manifold.compose([core_manifold, skin_manifold])`` — no trimesh
    round-trip for core or skin solids.
    """
    meta: dict[str, object] = {
        "core_struts": int(len(struts_core)),
        "skin_struts": int(len(skin_struts)),
        "union_ok": False,
        "skin_primitive": "manifold3d.Manifold.cube + transform + compose",
        "core_primitive": "manifold3d.Manifold.cylinder + transform + compose, crop ^ boundary",
    }
    if synthesis_report is not None:
        for k in (
            "bridge_struts",
            "volume_struts_before_filter",
            "volume_struts_after_filter",
            "n_boundary_faces",
            "n_quad_diagonals",
        ):
            if k in synthesis_report:
                meta[k] = synthesis_report[k]
    parts_m: list[manifold3d.Manifold] = []

    if struts_core.size > 0:
        core_m, _vol = generate_geometry(
            nodes_core,
            struts_core,
            strut_radius,
            boundary_mesh=boundary_mesh,
            add_spheres=False,
            crop_to_boundary=True,
            return_manifold=True,
        )
        parts_m.append(core_m)

    if skin_struts.size > 0:
        skin_m = generate_surface_skin_ribbon_mesh(
            boundary_mesh,
            nodes_skin,
            skin_struts,
            strut_radius,
            return_manifold=True,
        )
        if skin_m is not None:
            parts_m.append(skin_m)

    if not parts_m:
        raise ValueError("Decoupled geometry: no core struts and no skin struts.")

    united = manifold3d.Manifold.compose(parts_m)
    out = _manifold_to_trimesh(united)
    meta["union_ok"] = True
    meta["watertight"] = bool(out.is_watertight)
    return out, meta


def generate_decoupled_core_and_curved_skin(
    boundary_mesh: trimesh.Trimesh,
    nodes_core: np.ndarray,
    struts_core: np.ndarray,
    nodes_skin: np.ndarray,
    skin_struts: np.ndarray,
    strut_radius: float,
    *,
    segments_per_strut: int = 8,
    synthesis_report: dict[str, object] | None = None,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """
    Cropped cylindrical core plus skin struts swept along CAD-projected polylines.

    Skin follows ``polylines_for_struts_on_surface`` (chord subdivide + project).
    """
    from .stl_surface_skin import polylines_for_struts_on_surface

    meta: dict[str, object] = {
        "core_struts": int(len(struts_core)),
        "skin_struts": int(len(skin_struts)),
        "skin_primitive": "projected_polyline_cylinders",
        "segments_per_strut": int(segments_per_strut),
    }
    if synthesis_report is not None:
        meta.update({k: synthesis_report[k] for k in synthesis_report if k not in meta})

    parts_m: list[manifold3d.Manifold] = []
    r = float(strut_radius)

    if struts_core.size > 0:
        core_m, _vol = generate_geometry(
            nodes_core,
            struts_core,
            r,
            boundary_mesh=boundary_mesh,
            add_spheres=False,
            crop_to_boundary=True,
            return_manifold=True,
        )
        parts_m.append(core_m)

    if skin_struts.size > 0:
        polylines = polylines_for_struts_on_surface(
            boundary_mesh,
            nodes_skin,
            skin_struts,
            segments_per_strut=int(segments_per_strut),
        )
        for poly in polylines:
            for k in range(len(poly) - 1):
                p1, p2 = poly[k], poly[k + 1]
                vec = p2 - p1
                length = float(np.linalg.norm(vec))
                if length < 1e-9:
                    continue
                cyl = trimesh.creation.cylinder(radius=r, height=length, sections=10)
                z = np.array([0.0, 0.0, 1.0])
                direction = vec / length
                cross = np.cross(z, direction)
                if np.linalg.norm(cross) < 1e-8:
                    cross = np.array([1.0, 0.0, 0.0])
                angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
                if angle > 1e-8:
                    mat = trimesh.transformations.rotation_matrix(angle, cross)
                    cyl.apply_transform(mat)
                cyl.apply_translation((p1 + p2) * 0.5)
                parts_m.append(
                    manifold3d.Manifold(
                        manifold3d.Mesh(
                            vert_properties=np.asarray(cyl.vertices, dtype=np.float32),
                            tri_verts=np.asarray(cyl.faces, dtype=np.uint32),
                        )
                    )
                )

    if not parts_m:
        raise ValueError("Decoupled geometry: no core or skin struts.")

    united = manifold3d.Manifold.compose(parts_m)
    out = _manifold_to_trimesh(united)
    meta["union_ok"] = True
    meta["watertight"] = bool(out.is_watertight)
    return out, meta


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
    joint_sphere_scale: float = 1.15,
    trim_strut_ends: bool | None = None,
    crop_to_boundary: bool = True,
    return_manifold: bool = False,
) -> trimesh.Trimesh | tuple[trimesh.Trimesh, float] | tuple[manifold3d.Manifold, float]:
    """
    Generate explicit lattice geometry from topology nodes + struts.

    Strut solids use native ``manifold3d.Manifold.cylinder`` (plus ``transform``),
    unioned with ``Manifold.compose``; only the optional boundary uses
    ``_trimesh_to_manifold``.

    Args:
        nodes: (N, 3) node coordinates.
        struts: (S, 2) strut endpoint indices into `nodes`.
        strut_radius: Single radius (float) or per-strut radii (S,) array for adaptive thickness.
        boundary_mesh: Optional trimesh boundary for boolean trim intersection.
        add_spheres: If True, add fillet spheres at nodes (see ``joint_sphere_scale``).
        joint_sphere_scale: Node sphere radius = scale * max(per-node incident strut radius).
        trim_strut_ends: Shorten each cylinder by one strut radius at both ends so spheres
            meet tangentially. Default True when ``add_spheres`` is True, else False.
        crop_to_boundary: If True, intersect lattice with boundary (trim). If False,
            return raw cylinder union (pipe-style); volume is still computed from
            intersection when boundary_mesh is provided.
        return_manifold: If True and crop_to_boundary and boundary_mesh, skip trimesh
            conversion and return ``(Manifold, volume)``. Use ``manifold_to_trimesh()``
            for export when a mesh is required.

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

    do_trim = bool(trim_strut_ends) if trim_strut_ends is not None else bool(add_spheres)

    # Per-node max radius for joint spheres when strut radii vary.
    node_max_r = np.zeros(nodes_np.shape[0], dtype=np.float64)
    for i, (a_idx, b_idx) in enumerate(struts_np):
        r_i = float(radii[i])
        node_max_r[int(a_idx)] = max(node_max_r[int(a_idx)], r_i)
        node_max_r[int(b_idx)] = max(node_max_r[int(b_idx)], r_i)
    if np.all(node_max_r <= 0):
        node_max_r[:] = float(np.max(radii))

    # -------------------------------------------------------------------------
    # Step 1/2: Create cylinders for every strut and optional node spheres
    # -------------------------------------------------------------------------
    manifold_objects: list[manifold3d.Manifold] = []
    t0_cyl = time.perf_counter()

    for i, (a_idx, b_idx) in enumerate(struts_np):
        r_i = float(radii[i])
        trim_r = r_i if do_trim else 0.0
        cyl = _manifold_cylinder_between(
            nodes_np[int(a_idx)],
            nodes_np[int(b_idx)],
            r_i,
            trim_radius=trim_r,
        )
        if cyl is not None:
            manifold_objects.append(cyl)

    if add_spheres:
        scale = float(joint_sphere_scale)
        for ni in range(nodes_np.shape[0]):
            r_node = float(node_max_r[ni]) * scale
            if r_node > 0.0:
                manifold_objects.append(
                    manifold3d.Manifold.sphere(r_node).translate(
                        tuple(float(x) for x in nodes_np[ni])
                    )
                )

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


def _rotation_matrix_from_z(vec: np.ndarray) -> np.ndarray:
    """Build transform that rotates +Z to vec direction."""
    length = np.linalg.norm(vec)
    if length <= 0:
        return np.eye(4)
    v = vec / length
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z, v), -1.0, 1.0))
    if axis_norm < 1e-12:
        if dot < 0:
            return trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        return np.eye(4)
    axis /= axis_norm
    angle = np.arccos(dot)
    return trimesh.transformations.rotation_matrix(angle, axis)


def _union_trimesh_list(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Pairwise manifold3d union of a list of meshes."""
    import manifold3d

    if not meshes:
        return trimesh.Trimesh()
    manifolds = []
    for mesh in meshes:
        try:
            manifolds.append(
                manifold3d.Manifold(
                    manifold3d.Mesh(
                        vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                        tri_verts=np.asarray(mesh.faces, dtype=np.uint32),
                    )
                )
            )
        except Exception:
            continue
    if not manifolds:
        return trimesh.util.concatenate(meshes)
    while len(manifolds) > 1:
        next_level = []
        for i in range(0, len(manifolds), 2):
            if i + 1 < len(manifolds):
                try:
                    next_level.append(manifolds[i] + manifolds[i + 1])
                except Exception:
                    next_level.append(manifolds[i])
            else:
                next_level.append(manifolds[i])
        manifolds = next_level
    mesh_raw = manifolds[0].to_mesh()
    verts = np.asarray(mesh_raw.vert_properties).reshape(-1, 3)
    faces = np.asarray(mesh_raw.tri_verts).reshape(-1, 3)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def keep_largest_solid_component(
    mesh: trimesh.Trimesh,
    *,
    min_volume_fraction: float = 1e-5,
    label: str = "mesh",
) -> trimesh.Trimesh:
    """
    Drop zero-volume Boolean crumbs without silently deleting real geometry.

    Boolean intersection against CAD leaves slivers where the lattice is
    tangent to the surface — a handful of triangles enclosing ~1e-8 mm^3. Those
    are discarded. Anything larger than ``min_volume_fraction`` of the biggest
    component is *kept* and reported instead, because a genuinely disconnected
    lattice island means something upstream went wrong and must not vanish
    quietly.

    The mesh is deliberately not re-processed. manifold3d already emits a clean
    indexed mesh, and trimesh's vertex welding merges near-tangent Boolean
    vertices that sit microns apart, fusing two surfaces into a non-manifold
    edge that was not there before.
    """
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    volumes = np.array([abs(float(part.volume)) for part in components])
    order = np.argsort(volumes)[::-1]
    threshold = volumes[order[0]] * float(min_volume_fraction)

    keep = [components[i] for i in order if volumes[i] >= threshold]
    dropped = [volumes[i] for i in order if volumes[i] < threshold]

    if dropped:
        print(
            f"  [{label}] discarded {len(dropped)} Boolean crumb(s), "
            f"total {sum(dropped):.6g} mm^3 "
            f"(each below {threshold:.6g} mm^3)"
        )
    if len(keep) > 1:
        print(
            f"  WARNING: [{label}] {len(keep)} disconnected solids above the "
            f"discard threshold; keeping all. Volumes: "
            + ", ".join(f"{v:.4g}" for v in sorted(volumes, reverse=True)[: len(keep)])
            + " mm^3. The lattice is not a single connected body."
        )
        return trimesh.util.concatenate(keep)
    return keep[0]


def union_solid_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Public pairwise manifold union for already-solid component meshes."""
    nonempty = [
        mesh
        for mesh in meshes
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0
    ]
    return _union_trimesh_list(nonempty)


def _cylinder_manifold_for_joint(p0: np.ndarray, p1: np.ndarray, radius: float, segments: int = 12):
    """Native manifold3d cylinder between two points."""
    import manifold3d

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None

    z = vec / length
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mid = 0.5 * (p0 + p1)

    cyl = manifold3d.Manifold.cylinder(
        height=length,
        radius_low=float(radius),
        radius_high=float(radius),
        circular_segments=int(segments),
        center=True,
    )
    affine = [
        [float(x[0]), float(y[0]), float(z[0]), float(mid[0])],
        [float(x[1]), float(y[1]), float(z[1]), float(mid[1])],
        [float(x[2]), float(y[2]), float(z[2]), float(mid[2])],
    ]
    return cyl.transform(affine)


def _smoothstep01(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _wick_end_scales(
    arc_length: float,
    *,
    scale_a: float,
    scale_b: float,
    wick_length_a: float,
    wick_length_b: float,
    n_stations: int,
) -> np.ndarray:
    """Per-station section scale: end scales ease down to 1.0 over wick lengths."""
    n = max(int(n_stations), 2)
    L = float(max(arc_length, 0.0))
    sa = float(max(scale_a, 1.0))
    sb = float(max(scale_b, 1.0))
    Lt_a = float(max(wick_length_a, 0.0))
    Lt_b = float(max(wick_length_b, 0.0))
    if L > 1e-12 and Lt_a + Lt_b > L:
        shrink = L / (Lt_a + Lt_b)
        Lt_a *= shrink
        Lt_b *= shrink
    s = np.linspace(0.0, L, n)
    out = np.ones(n, dtype=np.float64)
    for i, si in enumerate(s):
        fa = 1.0
        if Lt_a > 1e-12 and si <= Lt_a:
            e = _smoothstep01(si / Lt_a)
            fa = sa * (1.0 - e) + 1.0 * e
        fb = 1.0
        if Lt_b > 1e-12 and (L - si) <= Lt_b:
            e = _smoothstep01((L - si) / Lt_b)
            fb = sb * (1.0 - e) + 1.0 * e
        out[i] = max(fa, fb)
    return out


def _strut_manifold_with_wick(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    radius_a: float,
    radius_b: float,
    *,
    wick_length_scale: float = 1.25,
    segments: int = 12,
    n_samples: int = 12,
):
    """Cylinder with smooth end tapers from joint radii down to strut radius."""
    import manifold3d

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None
    r_nom = float(radius)
    r_a = float(max(radius_a, r_nom))
    r_b = float(max(radius_b, r_nom))
    if (
        wick_length_scale <= 0.0
        or (r_a <= r_nom * 1.0001 and r_b <= r_nom * 1.0001)
    ):
        return _cylinder_manifold_for_joint(p0, p1, r_nom, segments=segments)

    scales = _wick_end_scales(
        length,
        scale_a=r_a / r_nom,
        scale_b=r_b / r_nom,
        wick_length_a=float(wick_length_scale) * (2.0 * r_a),
        wick_length_b=float(wick_length_scale) * (2.0 * r_b),
        n_stations=max(int(n_samples), 2),
    )
    radii = r_nom * scales
    z = vec / length
    tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    s = np.linspace(0.0, length, len(radii))
    parts = []
    for i in range(len(radii) - 1):
        h = float(s[i + 1] - s[i])
        if h < 1e-9:
            continue
        mid = p0 + z * (0.5 * (s[i] + s[i + 1]))
        frustum = manifold3d.Manifold.cylinder(
            height=h,
            radius_low=float(radii[i]),
            radius_high=float(radii[i + 1]),
            circular_segments=int(segments),
            center=True,
        )
        affine = [
            [float(x[0]), float(y[0]), float(z[0]), float(mid[0])],
            [float(x[1]), float(y[1]), float(z[1]), float(mid[1])],
            [float(x[2]), float(y[2]), float(z[2]), float(mid[2])],
        ]
        parts.append(frustum.transform(affine))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    united, _ = _union_manifolds_for_joint(parts)
    return united


def _union_manifolds_for_joint(parts: list, chunk_size: int = 128):
    """Robust, fast CSG union via manifold3d.Manifold.batch_boolean."""
    import time as _time

    if not parts:
        return None, 0.0

    t0 = _time.time()
    try:
        result = manifold3d.Manifold.batch_boolean(parts, manifold3d.OpType.Add)
        return result, _time.time() - t0
    except Exception:
        pass

    try:
        result = manifold3d.Manifold.compose(parts)
        return result, _time.time() - t0
    except Exception:
        pass

    level = list(parts)
    round_i = 0
    while len(level) > 1:
        round_i += 1
        next_level = []
        n = len(level)
        for i in range(0, n, 2):
            if i + 1 < n:
                try:
                    next_level.append(level[i] + level[i + 1])
                except Exception as e:
                    print(f"    [Union] pair failed at round {round_i}: {e}")
                    next_level.append(level[i])
            else:
                next_level.append(level[i])
        level = next_level
    result = level[0]
    elapsed = _time.time() - t0
    return result, elapsed


def union_lattice_with_spherical_joints(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float | np.ndarray,
    *,
    joint_scale: float = 1.05,
    joint_radius: float | np.ndarray | None = None,
    cylinder_segments: int = 16,
    sphere_segments: int = 16,
    joint_wick: bool = False,
    joint_wick_length_scale: float = 1.25,
    wick_samples: int = 12,
) -> tuple[trimesh.Trimesh, float]:
    """
    Build cylinders for every strut + spheres at every used node, then CSG-union.

    strut_radius may be a scalar or length-n_struts array.
    Joint sphere radius defaults to strut_radius * joint_scale (1.05) for
    scalars; for per-strut radii, each used node gets
    max(incident strut radii) * joint_scale unless joint_radius is set.

    When ``joint_wick`` is True, each strut radius eases from the incident
    joint sphere radius at each end down to the nominal strut radius over
    ``joint_wick_length_scale * joint_sphere_diameter`` (short struts clamp).

    Returns (unified_mesh, union_elapsed_seconds)
    """
    import manifold3d
    import time as _time

    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    n_struts = len(struts)
    strut_r = np.asarray(strut_radius, dtype=np.float64)
    if strut_r.ndim == 0:
        strut_radii = np.full(n_struts, float(strut_r), dtype=np.float64)
        scalar_strut = True
        scalar_r = float(strut_r)
    else:
        if strut_r.shape != (n_struts,):
            raise ValueError(f"strut_radius array shape {strut_r.shape} != ({n_struts},)")
        strut_radii = strut_r
        scalar_strut = False
        scalar_r = float(np.mean(strut_radii)) if n_struts else 0.0

    t0 = _time.time()
    used_nodes: set[int] = set()
    max_incident: dict[int, float] = {}
    for i, (a, b) in enumerate(struts):
        ai, bi = int(a), int(b)
        used_nodes.add(ai)
        used_nodes.add(bi)
        r_i = float(strut_radii[i])
        max_incident[ai] = max(max_incident.get(ai, 0.0), r_i)
        max_incident[bi] = max(max_incident.get(bi, 0.0), r_i)

    if joint_radius is not None:
        jr = np.asarray(joint_radius, dtype=np.float64)
        if jr.ndim == 0:
            joint_radii = {nid: float(jr) for nid in used_nodes}
            joint_r_msg = f"{float(jr):.4f}"
        else:
            if jr.shape != (len(nodes),):
                raise ValueError(f"joint_radius array shape {jr.shape} != ({len(nodes)},)")
            joint_radii = {nid: float(jr[nid]) for nid in used_nodes}
            vals = list(joint_radii.values())
            joint_r_msg = f"[{min(vals):.4f}..{max(vals):.4f}]" if vals else "n/a"
    elif scalar_strut:
        joint_r = scalar_r * float(joint_scale)
        joint_radii = {nid: joint_r for nid in used_nodes}
        joint_r_msg = f"{joint_r:.4f}"
    else:
        joint_radii = {
            nid: float(max_incident[nid]) * float(joint_scale) for nid in used_nodes
        }
        vals = list(joint_radii.values())
        joint_r_msg = f"[{min(vals):.4f}..{max(vals):.4f}]" if vals else "n/a"

    parts = []
    for i, (a, b) in enumerate(struts):
        ai, bi = int(a), int(b)
        r_i = float(strut_radii[i])
        if joint_wick:
            cyl = _strut_manifold_with_wick(
                nodes[ai],
                nodes[bi],
                r_i,
                float(joint_radii.get(ai, r_i)),
                float(joint_radii.get(bi, r_i)),
                wick_length_scale=float(joint_wick_length_scale),
                segments=cylinder_segments,
                n_samples=int(wick_samples),
            )
        else:
            cyl = _cylinder_manifold_for_joint(
                nodes[ai], nodes[bi], r_i, segments=cylinder_segments
            )
        if cyl is not None:
            parts.append(cyl)

    if joint_scale > 0.0:
        for nid in sorted(used_nodes):
            r = joint_radii[nid]
            if r <= 0.0:
                continue
            xyz = nodes[nid]
            sph = manifold3d.Manifold.sphere(
                r, circular_segments=int(sphere_segments)
            ).translate((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            parts.append(sph)

    if scalar_strut:
        strut_msg = f"r={scalar_r:.4f} mm"
    else:
        strut_msg = f"r=[{float(strut_radii.min()):.4f}..{float(strut_radii.max()):.4f}] mm"
    wick_msg = (
        f" wick={float(joint_wick_length_scale):g}*Dj"
        if joint_wick
        else ""
    )
    print(
        f"  Lattice primitives: {len(struts)} cylinders ({strut_msg}{wick_msg}) + "
        f"{len(used_nodes)} spheres (joint_r={joint_r_msg} mm) -> {len(parts)} manifolds "
        f"(build {_time.time() - t0:.2f}s)"
    )
    if not parts:
        return trimesh.Trimesh(), 0.0

    unified, _tree_time = _union_manifolds_for_joint(parts)
    mesh = _manifold_to_trimesh(unified)
    elapsed = _time.time() - t0
    return mesh, elapsed


def boolean_intersect_with_cad(
    lattice_mesh: trimesh.Trimesh,
    cad_mesh: trimesh.Trimesh,
) -> tuple[trimesh.Trimesh, float]:
    """
    Boolean Intersection of the unified lattice with the original CAD solid.
    Returns (trimmed_mesh, intersect_elapsed_seconds).
    """
    import manifold3d
    import time as _time

    t0 = _time.time()
    m_lat = _trimesh_to_manifold(lattice_mesh)
    m_cad = _trimesh_to_manifold(cad_mesh)
    try:
        m_out = m_lat ^ m_cad
    except Exception:
        m_out = manifold3d.Manifold.batch_boolean(
            [m_lat, m_cad], manifold3d.OpType.Intersect
        )
    mesh = _manifold_to_trimesh(m_out)
    elapsed = _time.time() - t0
    return mesh, elapsed


def _square_prism_between(p0: np.ndarray, p1: np.ndarray, side: float) -> trimesh.Trimesh | None:
    """Axis-aligned square box from p0 to p1 (cross-section side x side)."""
    vec = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None
    box = trimesh.creation.box(extents=[side, side, length])
    mat = _rotation_matrix_from_z(vec)
    mat[:3, 3] = (np.asarray(p0, dtype=np.float64) + np.asarray(p1, dtype=np.float64)) * 0.5
    box.apply_transform(mat)
    return box


def _square_prism_surface_oriented(
    p0: np.ndarray,
    p1: np.ndarray,
    side: float,
    surface_normal: np.ndarray,
    *,
    thickness: float | None = None,
    normal_oversize: float = 0.0,
) -> trimesh.Trimesh | None:
    """
    Straight rectangular prism from p0 to p1.
    Local frame: Z along strut, Y = surface normal projected perp to strut, X = Y × Z.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    t = p1 - p0
    length = float(np.linalg.norm(t))
    if length < 1e-6:
        return None
    t = t / length
    n = np.asarray(surface_normal, dtype=np.float64)
    n_n = float(np.linalg.norm(n))
    if n_n < 1e-12:
        return _square_prism_between(p0, p1, side)
    n = n / n_n
    n_use = n - float(np.dot(n, t)) * t
    n_len = float(np.linalg.norm(n_use))
    if n_len < 1e-9:
        return _square_prism_between(p0, p1, side)
    n_use = n_use / n_len
    w = np.cross(n_use, t)
    w_len = float(np.linalg.norm(w))
    if w_len < 1e-12:
        return _square_prism_between(p0, p1, side)
    w = w / w_len

    width = float(side)
    thick = float(side if thickness is None else thickness)
    over = max(float(normal_oversize), 0.0)
    box = trimesh.creation.box(extents=[width, thick + over, length])
    mat = np.eye(4)
    mat[:3, 0] = w
    mat[:3, 1] = n_use
    mat[:3, 2] = t
    mid = 0.5 * (p0 + p1) + (0.5 * over) * n_use
    mat[:3, 3] = mid
    box.apply_transform(mat)
    return box


def sweep_square_surface_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    side: float = 6.35,
    *,
    n_segments: int = 16,
    sphere_center: np.ndarray | None = None,
    sphere_radius: float | None = None,
    boolean_union: bool = True,
) -> trimesh.Trimesh:
    """Sweep square-cross-section struts along surface-conforming polylines."""
    from experiments.conformal_v2.conformal_utils import conform_strut_polyline
    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    if len(struts) == 0:
        return trimesh.Trimesh()

    segment_meshes: list[trimesh.Trimesh] = []
    for a, b in struts:
        poly = conform_strut_polyline(
            nodes[a],
            nodes[b],
            cad_mesh,
            n_segments=n_segments,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
        )
        for i in range(len(poly) - 1):
            prism = _square_prism_between(poly[i], poly[i + 1], side)
            if prism is not None:
                segment_meshes.append(prism)

    if not segment_meshes:
        return trimesh.Trimesh()
    if not boolean_union:
        return trimesh.util.concatenate(segment_meshes)
    return _union_trimesh_list(segment_meshes)


def sweep_square_straight_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    side: float = 1.6,
    *,
    thickness: float | None = None,
    sphere_center: np.ndarray | None = None,
    boolean_union: bool = True,
    normal_oversize: float = 0.0,
    trim_sphere_radius: float | None = None,
    trim_sphere_subdivisions: int = 5,
) -> trimesh.Trimesh:
    """Straight rectangular struts (chord between endpoints), oriented to the surface."""
    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    if len(struts) == 0:
        return trimesh.Trimesh()

    center = None if sphere_center is None else np.asarray(sphere_center, dtype=np.float64)
    ea = nodes[struts[:, 0]]
    eb = nodes[struts[:, 1]]
    mids = 0.5 * (ea + eb)

    if center is not None:
        normals = mids - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
    else:
        _, _, tri_ids = trimesh.proximity.closest_point(cad_mesh, mids)
        normals = np.asarray(cad_mesh.face_normals, dtype=np.float64)[
            np.asarray(tri_ids, dtype=np.int64)
        ]

    segment_meshes: list[trimesh.Trimesh] = []
    for i, (a, b) in enumerate(struts):
        prism = _square_prism_surface_oriented(
            nodes[a],
            nodes[b],
            side,
            normals[i],
            thickness=thickness,
            normal_oversize=normal_oversize,
        )
        if prism is not None:
            segment_meshes.append(prism)

    if not segment_meshes:
        return trimesh.Trimesh()
    if not boolean_union:
        mesh = trimesh.util.concatenate(segment_meshes)
    else:
        mesh = _union_trimesh_list(segment_meshes)

    if trim_sphere_radius is not None:
        if center is None:
            raise ValueError("trim_sphere_radius requires sphere_center")
        trim_cad = trimesh.creation.icosphere(
            subdivisions=int(trim_sphere_subdivisions),
            radius=float(trim_sphere_radius),
        )
        trim_cad.apply_translation(center)
        if not trim_cad.is_watertight:
            trim_cad = trimesh.Trimesh(
                vertices=trim_cad.vertices, faces=trim_cad.faces, process=True
            )
        mesh, _ = boolean_intersect_with_cad(mesh, trim_cad)
    return mesh


def _canonical_surface_nodes(
    cad_mesh: trimesh.Trimesh, nodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One CAD point and face normal per input node, shared by every incident bar."""
    nodes = np.asarray(nodes, dtype=np.float64)
    if len(nodes) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty
    query = trimesh.proximity.ProximityQuery(cad_mesh)
    closest, _, tri = query.on_surface(nodes)
    fn = np.asarray(cad_mesh.face_normals, dtype=np.float64)
    n = fn[np.asarray(tri, dtype=np.int64)]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.asarray(closest, dtype=np.float64), n


def _tangent_plane_frame(
    stations: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal section frame with the CAD normal as authority.

    Polyline tangents are projected into the local tangent plane so T ⟂ N.
    Width is U = N × T. Bars that share a node therefore share N and meet
    with in-surface tangents at that exact point.
    """
    stations = np.asarray(stations, dtype=np.float64)
    n_hat = np.asarray(normals, dtype=np.float64)
    n_hat = n_hat / np.maximum(np.linalg.norm(n_hat, axis=2, keepdims=True), 1e-12)

    tangents = np.empty_like(stations)
    tangents[:, 1:-1] = stations[:, 2:] - stations[:, :-2]
    tangents[:, 0] = stations[:, 1] - stations[:, 0]
    tangents[:, -1] = stations[:, -1] - stations[:, -2]

    t_tan = tangents - np.sum(tangents * n_hat, axis=2, keepdims=True) * n_hat
    t_len = np.linalg.norm(t_tan, axis=2, keepdims=True)
    t_hat = t_tan / np.maximum(t_len, 1e-12)

    bad = t_len[..., 0] < 1e-10
    if np.any(bad):
        ref = np.zeros_like(n_hat)
        ref[..., 0] = 1.0
        alt = np.cross(n_hat, ref)
        alt_len = np.linalg.norm(alt, axis=2, keepdims=True)
        ref2 = np.zeros_like(n_hat)
        ref2[..., 1] = 1.0
        alt = np.where(alt_len > 1e-10, alt, np.cross(n_hat, ref2))
        alt = alt / np.maximum(np.linalg.norm(alt, axis=2, keepdims=True), 1e-12)
        t_hat = np.where(bad[..., None], alt, t_hat)

    u_hat = np.cross(n_hat, t_hat)
    u_hat = u_hat / np.maximum(np.linalg.norm(u_hat, axis=2, keepdims=True), 1e-12)
    # Keep a consistent side along each bar so the loft does not flip 180°.
    n_st = u_hat.shape[1]
    for si in range(1, n_st):
        flip = np.sum(u_hat[:, si] * u_hat[:, si - 1], axis=1) < 0.0
        if np.any(flip):
            u_hat[flip, si] *= -1.0
            t_hat[flip, si] *= -1.0
    t_hat = np.cross(u_hat, n_hat)
    t_hat = t_hat / np.maximum(np.linalg.norm(t_hat, axis=2, keepdims=True), 1e-12)
    return n_hat, t_hat, u_hat


def _nlerp_endpoint_normals(
    n_a: np.ndarray, n_b: np.ndarray, n_st: int
) -> np.ndarray:
    """Smooth section normals along a bar; endpoints stay exact."""
    ts = np.linspace(0.0, 1.0, int(n_st))
    n = (
        (1.0 - ts)[None, :, None] * n_a[:, None, :]
        + ts[None, :, None] * n_b[:, None, :]
    )
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
    n[:, 0, :] = n_a
    n[:, -1, :] = n_b
    return n


def _fair_polylines_min_radius(
    stations: np.ndarray, radius: float, *, max_iters: int = 40
) -> np.ndarray:
    """Pull interior samples off tight corners until Menger radius >= ``radius``.

    Endpoints stay pinned so incident bars still meet. Gentle curves with
    R already above the limit are left on the CAD-projected path.
    """
    p = np.asarray(stations, dtype=np.float64).copy()
    r_min = float(radius)
    if r_min <= 0.0 or p.ndim != 3 or p.shape[1] < 3:
        return p
    for _ in range(int(max_iters)):
        a = p[:, :-2]
        b = p[:, 1:-1]
        c = p[:, 2:]
        v0 = b - a
        v1 = c - b
        la = np.linalg.norm(v0, axis=2)
        lb = np.linalg.norm(v1, axis=2)
        lc = np.linalg.norm(c - a, axis=2)
        area2 = np.linalg.norm(np.cross(v0, v1), axis=2)
        circ_r = (la * lb * lc) / np.maximum(2.0 * area2, 1e-18)
        violate = (area2 > 1e-14) & (la > 1e-12) & (lb > 1e-12) & (circ_r < r_min)
        if not np.any(violate):
            break
        alpha = np.clip(1.0 - circ_r / r_min, 0.15, 0.65)
        mix = np.where(violate, alpha, 0.0)[..., None]
        p[:, 1:-1] = (1.0 - mix) * b + mix * (0.5 * (a + c))
    return p


def _relax_crease_nodes(
    node_pts: np.ndarray,
    struts: np.ndarray,
    node_n: np.ndarray,
    max_travel: float,
    *,
    iterations: int = 8,
    lam: float = 0.5,
    min_inward_frac: float = 0.2,
) -> np.ndarray:
    """Move nodes that sit on a sharp CAD crease toward their neighbor chord.

    Gentle convex curvature also has an inward Laplacian (the chord sits
    inside the surface). Require the inward sag to be at least
    ``min_inward_frac`` of the mean incident edge, so only real folds move.
    Travel is capped at ``max_travel``.
    """
    pts0 = np.asarray(node_pts, dtype=np.float64)
    travel = float(max_travel)
    if travel <= 0.0 or len(pts0) == 0:
        return pts0.copy()
    pts = pts0.copy()
    nrm = np.asarray(node_n, dtype=np.float64)
    adj: list[list[int]] = [[] for _ in range(len(pts))]
    for a, b in np.asarray(struts, dtype=np.int64).reshape(-1, 2):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        adj[ia].append(ib)
        adj[ib].append(ia)
    sag_frac = float(min_inward_frac)
    step_cap = travel / max(int(iterations), 1)
    for _ in range(max(int(iterations), 1)):
        new = pts.copy()
        for i, nbrs in enumerate(adj):
            if len(nbrs) < 2:
                continue
            nbr_pts = pts[np.asarray(nbrs, dtype=np.int64)]
            mean = nbr_pts.mean(axis=0)
            delta = mean - pts[i]
            inward = float(-np.dot(delta, nrm[i]))
            if inward < 1e-6:
                continue
            edge = float(np.mean(np.linalg.norm(nbr_pts - pts[i], axis=1)))
            if edge > 1e-12 and inward < sag_frac * edge:
                continue
            step = float(lam) * delta
            mag = float(np.linalg.norm(step))
            if mag < 1e-12:
                continue
            if mag > step_cap:
                step = step * (step_cap / mag)
            new[i] = pts[i] + step
        pts = new
    disp = pts - pts0
    mag = np.linalg.norm(disp, axis=1, keepdims=True)
    scale = np.minimum(1.0, travel / np.maximum(mag, 1e-12))
    return pts0 + disp * scale


def _surface_bar_paths(
    nodes: np.ndarray,
    skin_struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    n_segments: int,
    project_stations: bool,
    project_max_travel: float | None = None,
    width: float = 1.0,
    thickness: float = 1.0,
    min_turn_radius: float = 0.0,
    crease_relax: float = 0.0,
    snap_nodes: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loft centerlines with shared endpoints and a minimum turning radius.

    Nodes are snapped to CAD once (unless ``snap_nodes`` is False), then optional
    crease relaxation pulls joints off sharp edges. Interiors follow the CAD
    unless ``min_turn_radius`` > 0, which fairs tight corners inward so the
    bar is not forced to stay strictly tangent through a crease.

    Returns
    -------
    stations, normals, kept_struts, node_pts, node_n
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    skin_struts = np.asarray(skin_struts, dtype=np.int64)
    cad_pts, node_n = _canonical_surface_nodes(cad_mesh, nodes)
    path_pts = cad_pts if snap_nodes else nodes.copy()

    ea = path_pts[skin_struts[:, 0]]
    eb = path_pts[skin_struts[:, 1]]
    lengths = np.linalg.norm(eb - ea, axis=1)
    keep = lengths > 1e-9
    kept_struts = skin_struts[keep]
    ea, eb = ea[keep], eb[keep]
    if len(ea) == 0:
        empty = np.empty((0, 2, 3), dtype=np.float64)
        return empty, empty, kept_struts, path_pts, node_n

    n_a = node_n[kept_struts[:, 0]]
    n_b = node_n[kept_struts[:, 1]]
    n_st = max(int(n_segments), 1) + 1
    ts = np.linspace(0.0, 1.0, n_st)
    chord = ea[:, None, :] * (1.0 - ts)[None, :, None] + eb[:, None, :] * ts[None, :, None]

    if project_stations:
        travel = (
            float(project_max_travel)
            if project_max_travel is not None
            else max(8.0 * max(float(width), float(thickness)), 12.0)
        )
        stations, _tri_ids = _project_stations_along_normals(
            cad_mesh,
            chord,
            ea,
            eb,
            n_a,
            n_b,
            max_travel=travel,
            pin_ends=True,
        )
        stations[:, 0, :] = ea
        stations[:, -1, :] = eb
    else:
        stations = chord

    r_turn = float(min_turn_radius)
    if r_turn > 0.0:
        stations = _fair_polylines_min_radius(stations, r_turn)
        stations[:, 0, :] = ea
        stations[:, -1, :] = eb

    node_pts = path_pts
    if float(crease_relax) > 0.0:
        node_pts = _relax_crease_nodes(
            path_pts, skin_struts, node_n, float(crease_relax)
        )
        ia = kept_struts[:, 0]
        ib = kept_struts[:, 1]
        disp_a = node_pts[ia] - path_pts[ia]
        disp_b = node_pts[ib] - path_pts[ib]
        stations = (
            stations
            + (1.0 - ts)[None, :, None] * disp_a[:, None, :]
            + ts[None, :, None] * disp_b[:, None, :]
        )
        stations[:, 0, :] = node_pts[ia]
        stations[:, -1, :] = node_pts[ib]
        if r_turn > 0.0:
            stations = _fair_polylines_min_radius(stations, r_turn)
            stations[:, 0, :] = node_pts[ia]
            stations[:, -1, :] = node_pts[ib]

    normals = _nlerp_endpoint_normals(n_a, n_b, n_st)
    return stations, normals, kept_struts, node_pts, node_n


def _rounded_rect_offsets(
    half_w: float,
    n_lo: float,
    n_hi: float,
    n_arc: int,
) -> np.ndarray:
    """(K, 2) offsets in (lateral, outward-normal) for a rounded rectangle."""
    if int(n_arc) <= 0:
        return np.array(
            [[-half_w, n_lo], [half_w, n_lo], [half_w, n_hi], [-half_w, n_hi]],
            dtype=np.float64,
        )
    r = min(float(half_w), 0.5 * (float(n_hi) - float(n_lo))) * 0.99
    r = max(r, 1e-6)
    n_arc = max(int(n_arc), 1)
    cx = (
        (half_w - r, n_lo + r),
        (half_w - r, n_hi - r),
        (-half_w + r, n_hi - r),
        (-half_w + r, n_lo + r),
    )
    a0 = (-0.5 * np.pi, 0.0, 0.5 * np.pi, np.pi)
    a1 = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
    pts: list[list[float]] = []
    for (cxu, cxn), lo, hi in zip(cx, a0, a1):
        for ang in np.linspace(lo, hi, n_arc + 1, endpoint=False):
            pts.append([cxu + r * float(np.cos(ang)), cxn + r * float(np.sin(ang))])
    return np.asarray(pts, dtype=np.float64)


def _loft_profile_faces(n_stations: int, n_prof: int) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for i in range(n_stations - 1):
        a, b = n_prof * i, n_prof * (i + 1)
        for k in range(n_prof):
            k_next = (k + 1) % n_prof
            faces.append((a + k, a + k_next, b + k_next))
            faces.append((a + k, b + k_next, b + k))
    last = n_prof * (n_stations - 1)
    for k in range(1, n_prof - 1):
        faces.append((0, k, k + 1))
        faces.append((last, last + k + 1, last + k))
    return np.asarray(faces, dtype=np.int64)


def _loft_rect_bars(
    stations: np.ndarray,
    normals: np.ndarray,
    width: float,
    depth_inner: float,
    depth_outer: float,
    *,
    stabilize_frame: bool = True,
    profile_arc_samples: int = 0,
    width_scales: np.ndarray | None = None,
) -> list[trimesh.Trimesh]:
    """
    Loft a rectangular (optionally rounded) cross-section along each path.

    ``stations`` is (n_bars, n_stations, 3) of points lying on the surface and
    ``normals`` the matching outward surface normals. The section is swept with
    its inner face held ``depth_inner`` below the surface at every station, so
    wall thickness stays uniform even where the surface curves.

    ``stabilize_frame`` uses the CAD normal as the thickness axis and projects
    the path tangent into that tangent plane (so joints that share a node also
    share N, with T ⟂ N). When False, T is taken from the polyline and N is
    Gram-Schmidt'd against T instead. ``profile_arc_samples`` > 0 fillets
    the four corners.

    ``width_scales`` optional (n_bars, n_stations) multipliers on bar width
    (e.g. joint-wick swell from sphere diameter down to nominal).
    """
    stations = np.asarray(stations, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    n_bars, n_stations = stations.shape[0], stations.shape[1]
    if n_stations < 2:
        return []

    if stabilize_frame:
        n_hat, _t_hat, u_hat = _tangent_plane_frame(stations, normals)
    else:
        tangents = np.empty_like(stations)
        tangents[:, 1:-1] = stations[:, 2:] - stations[:, :-2]
        tangents[:, 0] = stations[:, 1] - stations[:, 0]
        tangents[:, -1] = stations[:, -1] - stations[:, -2]
        tangents /= np.maximum(
            np.linalg.norm(tangents, axis=2, keepdims=True), 1e-12
        )
        n_cad = normals / np.maximum(
            np.linalg.norm(normals, axis=2, keepdims=True), 1e-12
        )
        n_hat = n_cad - np.sum(n_cad * tangents, axis=2, keepdims=True) * tangents
        n_hat /= np.maximum(np.linalg.norm(n_hat, axis=2, keepdims=True), 1e-12)
        u_hat = np.cross(n_hat, tangents)
        u_hat /= np.maximum(np.linalg.norm(u_hat, axis=2, keepdims=True), 1e-12)

    half_w0 = 0.5 * float(width)
    if width_scales is None:
        scales = np.ones((n_bars, n_stations), dtype=np.float64)
    else:
        scales = np.asarray(width_scales, dtype=np.float64)
        if scales.shape != (n_bars, n_stations):
            raise ValueError(
                f"width_scales shape {scales.shape} != ({n_bars}, {n_stations})"
            )

    bars: list[trimesh.Trimesh] = []
    for i in range(n_bars):
        verts_list = []
        n_prof = None
        for j in range(n_stations):
            hw = half_w0 * float(scales[i, j])
            offsets = _rounded_rect_offsets(
                hw, -float(depth_inner), float(depth_outer), int(profile_arc_samples)
            )
            if n_prof is None:
                n_prof = int(offsets.shape[0])
            corner = (
                stations[i, j]
                + offsets[:, 0:1] * u_hat[i, j]
                + offsets[:, 1:2] * n_hat[i, j]
            )
            verts_list.append(corner)
        verts = np.stack(verts_list, axis=0).reshape(-1, 3)
        faces = _loft_profile_faces(n_stations, int(n_prof))
        bar = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.repair.fix_normals(bar)
        bars.append(bar)
    return bars


def _slot_hub_cylinder(
    center: np.ndarray,
    normal: np.ndarray,
    *,
    radius: float,
    height: float,
    sections: int = 24,
) -> trimesh.Trimesh:
    """Cylinder along the surface normal, centered on a dual node."""
    cyl = trimesh.creation.cylinder(
        radius=float(radius),
        height=float(height),
        sections=max(int(sections), 8),
    )
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-12:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / nrm
    mat = np.eye(4)
    mat[:3, :3] = _rotation_align_local_z_to_unit(n)
    mat[:3, 3] = np.asarray(center, dtype=np.float64).reshape(3)
    cyl.apply_transform(mat)
    return cyl


def _node_stress_sphere(
    center: np.ndarray,
    *,
    radius: float,
    subdivisions: int = 3,
) -> trimesh.Trimesh:
    """Sphere at a dual/core node for stress deconcentration."""
    sph = trimesh.creation.icosphere(
        subdivisions=max(int(subdivisions), 1),
        radius=float(radius),
    )
    sph.apply_translation(np.asarray(center, dtype=np.float64).reshape(3))
    return sph


def _project_stations_along_normals(
    cad_mesh: trimesh.Trimesh,
    chord: np.ndarray,
    end_a: np.ndarray,
    end_b: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray,
    *,
    max_travel: float,
    pin_ends: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Snap chord samples onto CAD along interpolated endpoint normals.

    Closest-point projection of an interior chord can jump to a nearer patch
    (floor vs wall). Raycasting along the skin normal keeps the bar on the
    same sheet as its endpoints. Origins are nudged slightly inward so points
    already on the outer surface hit the near face instead of the far side.
    """
    n_bars, n_st, _ = chord.shape
    ts = np.linspace(0.0, 1.0, n_st)
    n_a = np.asarray(n_a, dtype=np.float64)
    n_b = np.asarray(n_b, dtype=np.float64)
    n_a /= np.maximum(np.linalg.norm(n_a, axis=1, keepdims=True), 1e-12)
    n_b /= np.maximum(np.linalg.norm(n_b, axis=1, keepdims=True), 1e-12)
    sample_n = (
        (1.0 - ts)[None, :, None] * n_a[:, None, :]
        + ts[None, :, None] * n_b[:, None, :]
    )
    sample_n /= np.maximum(np.linalg.norm(sample_n, axis=2, keepdims=True), 1e-12)

    dirs = sample_n.reshape(-1, 3)
    origins = chord.reshape(-1, 3) - 1e-3 * dirs
    n_rays = len(origins)
    best_t = np.full(n_rays, np.inf, dtype=np.float64)
    best_p = chord.reshape(-1, 3).copy()
    best_tri = np.full(n_rays, -1, dtype=np.int64)
    max_t = float(max_travel)

    for sign in (1.0, -1.0):
        try:
            locs, ray_id, tri_id = cad_mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=sign * dirs,
                multiple_hits=True,
            )
        except Exception:
            continue
        if locs is None or len(locs) == 0:
            continue
        for loc, rid, tid in zip(locs, ray_id, tri_id):
            i = int(rid)
            travel = float(np.linalg.norm(loc - origins[i]))
            if travel < 1e-9 or travel > max_t:
                continue
            if travel < best_t[i]:
                best_t[i] = travel
                best_p[i] = loc
                best_tri[i] = int(tid)

    miss = ~np.isfinite(best_t)
    if np.any(miss):
        query = trimesh.proximity.ProximityQuery(cad_mesh)
        closest, _, tri = query.on_surface(origins[miss] + 1e-3 * dirs[miss])
        best_p[miss] = closest
        best_tri[miss] = np.asarray(tri, dtype=np.int64)

    stations = best_p.reshape(n_bars, n_st, 3)
    if pin_ends:
        stations[:, 0, :] = np.asarray(end_a, dtype=np.float64)
        stations[:, -1, :] = np.asarray(end_b, dtype=np.float64)
    tri = best_tri.reshape(n_bars, n_st)
    return stations, tri


def generate_rectangular_surface_cage(
    nodes: np.ndarray,
    skin_struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    width: float,
    thickness: float,
    normal_oversize: float = 0.25,
    crop_to_boundary: bool = True,
    n_segments: int = 8,
    project_stations: bool = True,
    stabilize_frame: bool = True,
    profile_arc_samples: int = 0,
    round_ends: bool = False,
    end_hub_sections: int = 24,
    hub_radius: float | None = None,
    hub_height: float | None = None,
    project_max_travel: float | None = None,
    min_turn_radius: float = 0.0,
    crease_relax: float = 0.0,
    snap_nodes: bool = True,
    node_stress_spheres: bool = False,
    node_stress_sphere_scale: float = 1.1,
    node_stress_sphere_subdivisions: int = 3,
    joint_wick: bool = False,
    joint_wick_length_scale: float = 1.25,
) -> trimesh.Trimesh:
    """
    Build a uniform rectangular surface cage on arbitrary CAD.

    Volume struts are intentionally not handled here; callers keep those
    cylindrical. ``width`` is the in-surface bar width and ``thickness`` the
    inward wall depth measured from the CAD surface.

    When ``project_stations`` is True, interior loft stations are raycast onto
    CAD along interpolated endpoint face normals so each bar follows the skin
    instead of a 3D chord. Endpoints are pinned to one canonical closest-point
    per node so every incident bar meets at the exact same location, with that
    node's face normal as the shared thickness axis.

    ``min_turn_radius`` fairs interiors that wrap a CAD crease so the centerline
    is not forced to stay strictly tangent through a sharp edge. ``crease_relax``
    pulls shared nodes off those edges (inward only) by up to that distance.
    ``stabilize_frame`` orients the rectangle in the local tangent plane
    (thickness along the blended CAD normal, width ⟂ tangent).

    End padding (mutually exclusive):
      - ``node_stress_spheres``: sphere at each used node; diameter =
        ``node_stress_sphere_scale * width`` (default 1.1× width).
      - else ``round_ends``: cylinder hub, axis = surface normal, height =
        strut thickness, radius defaults to ``1.1 * (width / 2)``.

    ``joint_wick`` (with spheres) eases bar width from the sphere diameter at
    each end down to nominal width over
    ``joint_wick_length_scale * sphere_diameter`` along the loft path.
    """
    w = float(width)
    t = float(thickness)
    over = float(normal_oversize)
    if w <= 0.0:
        raise ValueError("surface cage width must be > 0")
    if t <= 0.0:
        raise ValueError("surface cage thickness must be > 0")
    if over < 0.0:
        raise ValueError("surface cage normal_oversize must be >= 0")

    nodes = np.asarray(nodes, dtype=np.float64)
    skin_struts = np.asarray(skin_struts, dtype=np.int64)
    if len(skin_struts) == 0:
        return trimesh.Trimesh()

    stations, normals, kept_struts, node_pts, node_n = _surface_bar_paths(
        nodes,
        skin_struts,
        cad_mesh,
        n_segments=max(int(n_segments), 1),
        project_stations=bool(project_stations),
        project_max_travel=project_max_travel,
        width=w,
        thickness=t,
        min_turn_radius=float(min_turn_radius),
        crease_relax=float(crease_relax),
        snap_nodes=bool(snap_nodes),
    )
    if len(stations) == 0:
        return trimesh.Trimesh()

    half_w = 0.5 * w
    hub_r = float(hub_radius) if hub_radius is not None else 1.1 * half_w
    hub_h = float(hub_height) if hub_height is not None else t
    stress_scale = float(node_stress_sphere_scale)
    stress_r = 0.5 * stress_scale * w

    width_scales = None
    if joint_wick and stress_scale > 1.0 + 1e-9:
        n_bars, n_st = stations.shape[0], stations.shape[1]
        width_scales = np.ones((n_bars, n_st), dtype=np.float64)
        wick_scale = float(joint_wick_length_scale)
        sphere_dia = stress_scale * w
        for bi in range(n_bars):
            seg = stations[bi, 1:] - stations[bi, :-1]
            seg_len = np.linalg.norm(seg, axis=1)
            arc = float(np.sum(seg_len))
            width_scales[bi] = _wick_end_scales(
                arc,
                scale_a=stress_scale,
                scale_b=stress_scale,
                wick_length_a=wick_scale * sphere_dia,
                wick_length_b=wick_scale * sphere_dia,
                n_stations=n_st,
            )

    bars = _loft_rect_bars(
        stations,
        normals,
        w,
        t,
        over,
        stabilize_frame=bool(stabilize_frame),
        profile_arc_samples=int(profile_arc_samples),
        width_scales=width_scales,
    )
    used = np.unique(kept_struts.reshape(-1))
    if node_stress_spheres and stress_r > 1e-9:
        used_pts = node_pts[used]
        for c in used_pts:
            bars.append(
                _node_stress_sphere(
                    c,
                    radius=stress_r,
                    subdivisions=int(node_stress_sphere_subdivisions),
                )
            )
    elif round_ends and hub_r > 1e-9 and hub_h > 1e-9:
        used_pts = node_pts[used]
        used_n = node_n[used]
        # Hub fills the inward wall; outer cap sits on the CAD surface.
        centers = used_pts - (0.5 * hub_h) * used_n
        n_sec = max(int(end_hub_sections), 8)
        for c, nrm in zip(centers, used_n):
            bars.append(
                _slot_hub_cylinder(c, nrm, radius=hub_r, height=hub_h, sections=n_sec)
            )
    if not bars:
        return trimesh.Trimesh()

    raw = _union_trimesh_list(bars)
    if not crop_to_boundary or len(raw.faces) == 0:
        return raw
    cropped, _ = boolean_intersect_with_cad(raw, cad_mesh)
    return cropped


def trim_mesh_with_inset_sphere(
    mesh: trimesh.Trimesh,
    *,
    sphere_center: np.ndarray,
    design_radius: float,
    inset_mm: float = 0.25,
    subdivisions: int = 5,
) -> tuple[trimesh.Trimesh, float]:
    """Boolean ∩ mesh with a sphere of radius design_radius - inset_mm."""
    r_trim = float(design_radius) - float(inset_mm)
    if r_trim <= 0.0:
        raise ValueError(f"trim radius non-positive: {r_trim}")
    center = np.asarray(sphere_center, dtype=np.float64)
    cad = trimesh.creation.icosphere(subdivisions=int(subdivisions), radius=r_trim)
    cad.apply_translation(center)
    if not cad.is_watertight:
        cad = trimesh.Trimesh(vertices=cad.vertices, faces=cad.faces, process=True)
    return boolean_intersect_with_cad(mesh, cad)

