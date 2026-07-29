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


def _union_manifolds_for_joint(parts: list, chunk_size: int = 128):
    """Robust CSG union via pairwise manifold3d Add."""
    import time as _time

    if not parts:
        return None, 0.0

    t0 = _time.time()
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
    cylinder_segments: int = 8,
    sphere_segments: int = 8,
) -> tuple[trimesh.Trimesh, float]:
    """
    Build cylinders for every strut + spheres at every used node, then CSG-union.

    strut_radius may be a scalar or length-n_struts array.
    Joint sphere radius defaults to strut_radius * joint_scale (1.05) for
    scalars; for per-strut radii, each used node gets
    max(incident strut radii) * joint_scale unless joint_radius is set.

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
    parts = []
    used_nodes: set[int] = set()
    max_incident: dict[int, float] = {}
    for i, (a, b) in enumerate(struts):
        ai, bi = int(a), int(b)
        used_nodes.add(ai)
        used_nodes.add(bi)
        r_i = float(strut_radii[i])
        max_incident[ai] = max(max_incident.get(ai, 0.0), r_i)
        max_incident[bi] = max(max_incident.get(bi, 0.0), r_i)
        cyl = _cylinder_manifold_for_joint(nodes[ai], nodes[bi], r_i, segments=cylinder_segments)
        if cyl is not None:
            parts.append(cyl)

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

    for nid in sorted(used_nodes):
        xyz = nodes[nid]
        sph = manifold3d.Manifold.sphere(
            joint_radii[nid], circular_segments=int(sphere_segments)
        ).translate((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        parts.append(sph)

    if scalar_strut:
        strut_msg = f"r={scalar_r:.4f} mm"
    else:
        strut_msg = f"r=[{float(strut_radii.min()):.4f}..{float(strut_radii.max()):.4f}] mm"
    print(
        f"  Lattice primitives: {len(struts)} cylinders ({strut_msg}) + "
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

