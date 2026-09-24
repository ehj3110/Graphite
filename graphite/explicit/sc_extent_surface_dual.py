"""
Surface dual UV net via direct projection of the extent-trimmed octahedral graph.

1. Empty UV canvas: exposed Cartesian faces of VF-kept SC cells (BFS unwrap).
2. Project surviving trimmed-lattice nodes onto those faces (outer plane or
   mid-plane when the outer face-center was culled) — no synthetic nodes.
3. Native same-face struts from the trimmed strut list.
4. Topological manifold stitches across shared 3D Cartesian edges (Task 22).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.hex_rules import _HEX_FACES, _hex_face_centers
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.sc_axis_cull_octahedral import (
    HALF_EXTENT_MAX,
    CellMaterialExtents,
    face_keep_mask_from_extents,
    generate_extent_trimmed_octahedral,
    measure_material_extents,
)
from graphite.explicit.sc_boundary_states import (
    build_edt_sdf_field,
    estimate_hex_volume_fractions,
)
from graphite.explicit.sc_contextual_surface import (
    ROLE_FULL,
    ROLE_HALF_CUT,
    ROLE_HALF_SIDE,
    face_plot_size,
)
from graphite.explicit.sc_quantized_surface import (
    _corner_key,
    _edge_key,
    _face_key,
    _outward_face_normal,
)
from graphite.explicit.sc_surface_net import (
    NetFace,
    NetUnfoldResult,
    bfs_spanning_tree,
    build_face_hinge_graph,
    select_root_face,
    unfold_faces_integer_grid,
)
from graphite.explicit.sc_topological_dual import (
    TopoFace,
    TopoRoutedStrut,
    TopologicalDualResult,
    _edge_midpoint_key,
    _route_pair_across_edge,
    build_edge_map,
)

_FACE_EXTENT_ATTR = (
    "neg_z",
    "pos_z",
    "neg_y",
    "pos_y",
    "neg_x",
    "pos_x",
)


def _extent_for_face(ext: CellMaterialExtents, face_i: int) -> float:
    return float(getattr(ext, _FACE_EXTENT_ATTR[int(face_i)]))


def _cut_axis_from_face(face_i: int) -> int:
    if face_i in (0, 1):
        return 2
    if face_i in (2, 3):
        return 1
    return 0


@dataclass
class ExtentCellRecord:
    hex_i: int
    corners: np.ndarray
    extents: CellMaterialExtents
    keep_mask: np.ndarray


@dataclass
class ExtentSurfaceBundle:
    cells: list[ExtentCellRecord]
    lattice_nodes: np.ndarray
    lattice_struts: np.ndarray
    net: NetUnfoldResult
    topo: TopologicalDualResult
    report: dict = field(default_factory=dict)


def collect_extent_cells(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    half_max: float = HALF_EXTENT_MAX,
    resolution: float | None = None,
    sdf_field=None,
) -> tuple[list[ExtentCellRecord], object]:
    """VF-candidate cells with measured extents + outer-node keep masks."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    vf = estimate_hex_volume_fractions(
        cad,
        elems,
        samples_per_axis=int(samples_per_axis),
        resolution=resolution,
        sdf_field=sdf_field,
    )
    if sdf_field is None:
        res = float(vf.voxel_resolution) if vf.voxel_resolution > 0 else 0.5
        sdf_field = build_edt_sdf_field(cad, res)

    out: list[ExtentCellRecord] = []
    for hi in np.flatnonzero(vf.volume_fractions > float(empty_vf_max)):
        corners = elems[int(hi)]
        ext = measure_material_extents(
            corners,
            sdf_field,
            samples_per_axis=int(extent_samples_per_axis),
        )
        if not ext.has_material:
            continue
        keep = face_keep_mask_from_extents(ext, half_max=half_max)
        if not np.any(keep):
            continue
        out.append(
            ExtentCellRecord(
                hex_i=int(hi),
                corners=np.asarray(corners, dtype=np.float64),
                extents=ext,
                keep_mask=np.asarray(keep, dtype=bool),
            )
        )
    return out, sdf_field


def _plane_frame(corners_3d: np.ndarray, normal: np.ndarray):
    """Orthonormal (origin, e1, e2, n) for a face quad."""
    c = np.asarray(corners_3d, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    n = n / max(nn, 1e-14)
    origin = c.mean(axis=0)
    e1 = c[1] - c[0]
    if float(np.linalg.norm(e1)) < 1e-14:
        e1 = c[2] - c[0]
    e1 = e1 - np.dot(e1, n) * n
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-14)
    e2 = np.cross(n, e1)
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-14)
    return origin, e1, e2, n


def _point_in_quad_plane(
    p: np.ndarray,
    corners_3d: np.ndarray,
    normal: np.ndarray,
    *,
    plane_tol: float,
    inplane_pad: float = 0.05,
) -> bool:
    """True if ``p`` lies on the face plane and inside the quad (with pad)."""
    origin, e1, e2, n = _plane_frame(corners_3d, normal)
    d = np.asarray(p, dtype=np.float64) - origin
    if abs(float(np.dot(d, n))) > float(plane_tol):
        return False
    # Bbox of quad in (e1,e2)
    uv = np.array(
        [[np.dot(c - origin, e1), np.dot(c - origin, e2)] for c in corners_3d],
        dtype=np.float64,
    )
    u, v = float(np.dot(d, e1)), float(np.dot(d, e2))
    umin, umax = float(uv[:, 0].min()), float(uv[:, 0].max())
    vmin, vmax = float(uv[:, 1].min()), float(uv[:, 1].max())
    span_u = max(umax - umin, 1e-14)
    span_v = max(vmax - vmin, 1e-14)
    pad_u = float(inplane_pad) * span_u
    pad_v = float(inplane_pad) * span_v
    return (umin - pad_u) <= u <= (umax + pad_u) and (vmin - pad_v) <= v <= (
        vmax + pad_v
    )


def midplane_quad_for_face(corners: np.ndarray, face_i: int) -> np.ndarray:
    """
    Mid-plane square parallel to ``face_i`` (through cell centroid).

    Corners are the midpoints of the four cell edges parallel to the face
    normal — equivalently the four orthogonal face-centers as a quad when
    ordered around the mid-plane.
    """
    c = np.asarray(corners, dtype=np.float64)
    fcs = _hex_face_centers(c)
    axis = _cut_axis_from_face(int(face_i))
    orth = [i for i in range(6) if _cut_axis_from_face(i) != axis]
    # Order around mid-plane by angle in the plane
    centroid = c.mean(axis=0)
    n = _outward_face_normal(c, face_i)
    _, e1, e2, _ = _plane_frame(
        np.array([fcs[i] for i in orth], dtype=np.float64), n
    )
    pts = [fcs[i] for i in orth]
    ang = [
        float(np.arctan2(np.dot(p - centroid, e2), np.dot(p - centroid, e1)))
        for p in pts
    ]
    order = np.argsort(ang)
    return np.asarray([pts[i] for i in order], dtype=np.float64)


def infer_face_role_and_size(
    cell: ExtentCellRecord,
    face_i: int,
    node_gids: list[int],
    *,
    on_midplane: bool,
) -> tuple[str, tuple[float, float]]:
    """
    Size/role labels from projected geometry (not synthetic node inventing).

    - Mid-plane tile with ≥2 nodes → CUT 1×1
    - Outer tile, this face culled orthogonals in cell → SIDE 1×0.5
    - Else → Full 1×1
    """
    keep = cell.keep_mask
    n = len(node_gids)
    if on_midplane and n >= 2:
        return ROLE_HALF_CUT, face_plot_size(ROLE_HALF_CUT)
    # SIDE: this outer face kept, but cell has a culled direction orthogonal
    # to this face (half leaf wall), OR single node on a half-extent face.
    axis = _cut_axis_from_face(int(face_i))
    has_half_cut_axis = any(
        (not bool(keep[j])) and _cut_axis_from_face(j) != axis for j in range(6)
    )
    this_culled = not bool(keep[int(face_i)])
    if (not on_midplane) and n >= 1 and (has_half_cut_axis or this_culled):
        # Thin wall strip when neighboring mid-plane cut exists
        if has_half_cut_axis and bool(keep[int(face_i)]):
            return ROLE_HALF_SIDE, face_plot_size(ROLE_HALF_SIDE)
        if this_culled and n == 1:
            return ROLE_HALF_SIDE, face_plot_size(ROLE_HALF_SIDE)
    return ROLE_FULL, face_plot_size(ROLE_FULL)


def project_nodes_onto_face(
    cell: ExtentCellRecord,
    face_i: int,
    lattice_nodes: np.ndarray,
    *,
    plane_tol: float,
) -> tuple[list[int], bool]:
    """
    Surviving lattice node indices that belong on this exposed face tile.

    1) Prefer nodes on the outer Cartesian face plane.
    2) If the outer face-center was culled, fall back to nodes on the
       mid-plane parallel to this face (restores CUT diamonds).
    """
    c = cell.corners
    face = _HEX_FACES[int(face_i)]
    corners_3d = np.array([c[i] for i in face], dtype=np.float64)
    normal = _outward_face_normal(c, face_i)
    pts = np.asarray(lattice_nodes, dtype=np.float64)

    on_outer: list[int] = []
    for gid, p in enumerate(pts):
        if _point_in_quad_plane(p, corners_3d, normal, plane_tol=plane_tol):
            on_outer.append(gid)
    if on_outer:
        return on_outer, False

    # Mid-plane fallback only when this outer FC was culled
    if bool(cell.keep_mask[int(face_i)]):
        return [], False

    mid_quad = midplane_quad_for_face(c, face_i)
    on_mid: list[int] = []
    for gid, p in enumerate(pts):
        if _point_in_quad_plane(p, mid_quad, normal, plane_tol=plane_tol):
            on_mid.append(gid)
    return on_mid, True


def native_struts_on_face(
    node_gids: list[int],
    lattice_struts: np.ndarray,
) -> list[tuple[int, int]]:
    """Local index pairs for struts whose both endpoints lie on this face."""
    if len(node_gids) < 2:
        return []
    gid_to_local = {g: i for i, g in enumerate(node_gids)}
    out: list[tuple[int, int]] = []
    for a, b in np.asarray(lattice_struts, dtype=np.int64):
        ia, ib = gid_to_local.get(int(a)), gid_to_local.get(int(b))
        if ia is None or ib is None or ia == ib:
            continue
        out.append((ia, ib) if ia < ib else (ib, ia))
    return sorted(set(out))


def build_projection_exposed_faces(
    cells: list[ExtentCellRecord],
    lattice_nodes: np.ndarray,
    lattice_struts: np.ndarray,
    *,
    round_decimals: int = 6,
    plane_tol_frac: float = 0.08,
) -> tuple[list[NetFace], list[TopoFace]]:
    """
    Empty SC exposed-face canvas, populated only by projected lattice nodes.
    """
    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for cell in cells:
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(cell.corners, face, round_decimals)
            face_owners.setdefault(fkey, []).append((cell.hex_i, fi))

    net_faces: list[NetFace] = []
    topo_faces: list[TopoFace] = []
    pts = np.asarray(lattice_nodes, dtype=np.float64)

    for cell in cells:
        c = cell.corners
        # Characteristic length for plane tolerance
        extents = c.max(axis=0) - c.min(axis=0)
        plane_tol = float(plane_tol_frac) * float(np.min(extents))
        fcs = _hex_face_centers(c)

        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(c, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue

            gids, on_mid = project_nodes_onto_face(
                cell, fi, pts, plane_tol=plane_tol
            )
            if not gids:
                continue

            role, size = infer_face_role_and_size(
                cell, fi, gids, on_midplane=on_mid
            )
            corners_3d = np.array([c[i] for i in face], dtype=np.float64)
            # CUT tiles: center at mid-plane for UV basis; keep outer corners
            # for hinge edge keys (shared with SIDE walls).
            if role == ROLE_HALF_CUT:
                center_3d = c.mean(axis=0)
            elif role == ROLE_HALF_SIDE:
                center_3d = pts[gids[0]].copy()
            else:
                center_3d = fcs[fi]

            corner_keys = tuple(
                _corner_key(corners_3d[i], round_decimals) for i in range(4)
            )
            edge_keys = tuple(
                _edge_key(corners_3d[a], corners_3d[b], round_decimals)
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            native = native_struts_on_face(gids, lattice_struts)
            nodes_3d = pts[np.asarray(gids, dtype=np.int64)]

            fid = len(net_faces)
            net_faces.append(
                NetFace(
                    face_id=fid,
                    hex_i=cell.hex_i,
                    face_i=fi,
                    corners_3d=corners_3d,
                    center_3d=np.asarray(center_3d, dtype=np.float64),
                    normal=_outward_face_normal(c, fi),
                    primary_gid=int(gids[0]),
                    corner_keys=corner_keys,
                    edge_keys=edge_keys,
                    role=role,
                    size_uv=size,
                    gids=tuple(int(g) for g in gids),
                )
            )
            edges = tuple(
                (corners_3d[a].copy(), corners_3d[b].copy())
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            topo_faces.append(
                TopoFace(
                    face_id=fid,
                    hex_i=cell.hex_i,
                    face_i=fi,
                    rule="extent_projected",
                    role=role,
                    corners_3d=corners_3d,
                    edges_3d=edges,
                    normal=_outward_face_normal(c, fi),
                    nodes_3d=np.asarray(nodes_3d, dtype=np.float64).reshape(-1, 3),
                    native_locals=tuple(native),
                )
            )
    return net_faces, topo_faces


def compute_extent_uv_unfold(net_faces: list[NetFace]) -> NetUnfoldResult:
    """BFS spanning-tree proportional unwrap of exposed faces."""
    if not net_faces:
        return NetUnfoldResult(
            faces=[],
            hinge_adj={},
            hinges=set(),
            cuts=set(),
            root_id=-1,
            unfolds={},
            node_uv={},
            node_face={},
            report={"n_faces": 0},
        )

    adj, _ = build_face_hinge_graph(net_faces)
    root_id = select_root_face(net_faces, adj)
    hinges, parent = bfs_spanning_tree(adj, root_id)
    cuts: set[tuple[int, int]] = set()
    for u, nbrs in adj.items():
        for v in nbrs:
            if u >= v:
                continue
            if (u, v) not in hinges:
                cuts.add((u, v))

    unfolds = unfold_faces_integer_grid(net_faces, parent, root_id)
    n_full = sum(1 for f in net_faces if f.role == ROLE_FULL)
    n_side = sum(1 for f in net_faces if f.role == ROLE_HALF_SIDE)
    n_cut = sum(1 for f in net_faces if f.role == ROLE_HALF_CUT)

    grids = list(unfolds.values())
    if grids:
        u0 = min(uf.grid_ij[0] for uf in grids)
        v0 = min(uf.grid_ij[1] for uf in grids)
        u1 = max(uf.grid_ij[0] + uf.size_uv[0] for uf in grids)
        v1 = max(uf.grid_ij[1] + uf.size_uv[1] for uf in grids)
        span_u, span_v = u1 - u0, v1 - v0
    else:
        span_u = span_v = 0.0

    return NetUnfoldResult(
        faces=net_faces,
        hinge_adj=adj,
        hinges=hinges,
        cuts=cuts,
        root_id=root_id,
        unfolds=unfolds,
        node_uv={},
        node_face={},
        report={
            "n_faces": len(net_faces),
            "n_hinges": len(hinges),
            "n_cuts": len(cuts),
            "n_unfolded": len(unfolds),
            "root_id": root_id,
            "n_full": n_full,
            "n_half_side": n_side,
            "n_half_cut": n_cut,
            "grid_span_u": float(span_u),
            "grid_span_v": float(span_v),
            "projection": True,
        },
    )


def route_projection_surface_dual(
    faces: list[TopoFace],
    *,
    edge_decimals: int = 5,
) -> TopologicalDualResult:
    """Task-22 edge_map stitches between projected face node sets."""
    edge_map = build_edge_map(faces, decimals=edge_decimals)
    by_id = {f.face_id: f for f in faces}
    edge_seg: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=edge_decimals)
            edge_seg.setdefault(key, (p0, p1))

    native: list[tuple[int, int, int, int]] = []
    for f in faces:
        for i, j in f.native_locals:
            native.append((f.face_id, int(i), f.face_id, int(j)))

    routed: list[TopoRoutedStrut] = []
    seen: set[tuple[int, int, int, int]] = set()
    n_shared = 0
    for key, fids in edge_map.items():
        if len(fids) != 2:
            continue
        n_shared += 1
        fa, fb = by_id[fids[0]], by_id[fids[1]]
        p0, p1 = edge_seg[key]
        for s in _route_pair_across_edge(fa, fb, p0, p1, key):
            a = (s.face_a, s.node_a, s.face_b, s.node_b)
            b = (s.face_b, s.node_b, s.face_a, s.node_a)
            if a in seen or b in seen:
                continue
            seen.add(a)
            routed.append(s)

    report = {
        "n_faces": len(faces),
        "n_edge_keys": len(edge_map),
        "n_shared_edges": n_shared,
        "n_boundary_edges": sum(1 for v in edge_map.values() if len(v) == 1),
        "n_native_struts": len(native),
        "n_routed_struts": len(routed),
        "n_full": sum(1 for f in faces if f.role == ROLE_FULL),
        "n_half_cut": sum(1 for f in faces if f.role == ROLE_HALF_CUT),
        "n_half_side": sum(1 for f in faces if f.role == ROLE_HALF_SIDE),
        "n_half_apex": 0,
    }
    return TopologicalDualResult(
        faces=faces,
        edge_map=edge_map,
        native_struts=native,
        routed_struts=routed,
        report=report,
    )


def map_topo_nodes_to_uv(
    net: NetUnfoldResult,
    topo: TopologicalDualResult,
) -> dict[int, list[np.ndarray]]:
    """Project each face's real 3D lattice nodes into UV (no synthetic points)."""
    from graphite.explicit.sc_surface_net import (
        _map_point_uv,
        _rect_edge_mids_uv,
        side_cut_edge_and_midpoint_uv,
    )

    unfold_by_id = dict(net.unfolds)
    out: dict[int, list[np.ndarray]] = {}

    for tf in topo.faces:
        uf = unfold_by_id.get(tf.face_id)
        if uf is None:
            continue
        gu, gv = float(uf.grid_ij[0]), float(uf.grid_ij[1])
        w, h = float(uf.size_uv[0]), float(uf.size_uv[1])
        pts3 = np.asarray(tf.nodes_3d, dtype=np.float64).reshape(-1, 3)

        if tf.role == ROLE_HALF_CUT and len(pts3) >= 2:
            # Map mid-plane nodes via angular order → UV edge mids / polygon
            mapped = [_map_point_uv(uf, p) for p in pts3]
            # If outer-face basis collapses mid-plane points, use edge mids
            arr = np.asarray(mapped, dtype=np.float64)
            span = float(arr.max(axis=0).max() - arr.min(axis=0).min()) if len(arr) else 0.0
            # Also check if all map near center
            ctr = np.array([gu + 0.5 * w, gv + 0.5 * h])
            if len(arr) >= 4 and (
                span < 0.15 or float(np.mean(np.linalg.norm(arr - ctr, axis=1))) < 0.12
            ):
                mids = _rect_edge_mids_uv(gu, gv, w, h)
                # Assign each 3D node to nearest UV mid by face-tangent angle
                origin = pts3.mean(axis=0)
                n = np.asarray(tf.normal, dtype=np.float64)
                n = n / max(float(np.linalg.norm(n)), 1e-14)
                # Build tangent frame
                ref = pts3[0] - origin
                ref = ref - np.dot(ref, n) * n
                if float(np.linalg.norm(ref)) < 1e-14:
                    ref = np.array([1.0, 0.0, 0.0]) - n[0] * n
                e1 = ref / max(float(np.linalg.norm(ref)), 1e-14)
                e2 = np.cross(n, e1)
                angs = [
                    float(np.arctan2(np.dot(p - origin, e2), np.dot(p - origin, e1)))
                    for p in pts3
                ]
                order = list(np.argsort(angs))
                # Rotate so first maps stably; place on consecutive edge mids
                placed = [None] * len(pts3)
                for k, oi in enumerate(order):
                    placed[oi] = mids[k % 4]
                out[tf.face_id] = [np.asarray(p, dtype=np.float64) for p in placed]
            else:
                out[tf.face_id] = mapped
            continue

        if tf.role == ROLE_HALF_SIDE and len(pts3) == 1:
            mapped = _map_point_uv(uf, pts3[0])
            (_a, _b), mid = side_cut_edge_and_midpoint_uv(uf)
            if (
                gu - 1e-6 <= mapped[0] <= gu + w + 1e-6
                and gv - 1e-6 <= mapped[1] <= gv + h + 1e-6
            ):
                # Snap to cut edge if close
                if float(np.linalg.norm(mapped - mid)) < 0.35:
                    out[tf.face_id] = [mid]
                else:
                    out[tf.face_id] = [mapped]
            else:
                out[tf.face_id] = [mid]
            continue

        if tf.role == ROLE_FULL and len(pts3) == 1:
            out[tf.face_id] = [
                np.array([gu + 0.5 * w, gv + 0.5 * h], dtype=np.float64)
            ]
            continue

        out[tf.face_id] = [_map_point_uv(uf, p) for p in pts3]
    return out


def compute_extent_surface_dual_bundle(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    half_max: float = HALF_EXTENT_MAX,
    round_decimals: int = 6,
    lattice_nodes: np.ndarray | None = None,
    lattice_struts: np.ndarray | None = None,
) -> ExtentSurfaceBundle:
    """
    Project the extent-trimmed octahedral graph onto the SC exposed-face UV net.

    If ``lattice_nodes`` / ``lattice_struts`` are omitted, regenerates them with
    ``generate_extent_trimmed_octahedral`` (same graph as the Task-28 STL).
    """
    cells, sdf = collect_extent_cells(
        cad_mesh,
        hex_elems,
        empty_vf_max=empty_vf_max,
        samples_per_axis=samples_per_axis,
        extent_samples_per_axis=extent_samples_per_axis,
        half_max=half_max,
    )

    if lattice_nodes is None or lattice_struts is None:
        lattice_nodes, lattice_struts, cull_report = generate_extent_trimmed_octahedral(
            cad_mesh,
            hex_elems,
            empty_vf_max=empty_vf_max,
            samples_per_axis=samples_per_axis,
            extent_samples_per_axis=extent_samples_per_axis,
            half_max=half_max,
            round_decimals=round_decimals,
            sdf_field=sdf,
        )
    else:
        cull_report = None
        lattice_nodes = np.asarray(lattice_nodes, dtype=np.float64)
        lattice_struts = np.asarray(lattice_struts, dtype=np.int64)

    net_faces, topo_faces = build_projection_exposed_faces(
        cells,
        lattice_nodes,
        lattice_struts,
        round_decimals=round_decimals,
    )
    net = compute_extent_uv_unfold(net_faces)
    topo = route_projection_surface_dual(topo_faces)

    report = {
        "n_cells": len(cells),
        "n_lattice_nodes": int(len(lattice_nodes)),
        "n_lattice_struts": int(len(lattice_struts)),
        "n_faces": len(net_faces),
        "n_hinges": net.report.get("n_hinges"),
        "n_cuts": net.report.get("n_cuts"),
        "n_full": net.report.get("n_full"),
        "n_half_side": net.report.get("n_half_side"),
        "n_half_cut": net.report.get("n_half_cut"),
        "n_shared_edges": topo.report.get("n_shared_edges"),
        "n_routed_struts": topo.report.get("n_routed_struts"),
        "n_native_struts": topo.report.get("n_native_struts"),
        "half_max": float(half_max),
        "projection": True,
    }
    if cull_report is not None:
        report["cull_hex_kept"] = cull_report.n_hex_kept
        report["cull_nodes_kept"] = cull_report.n_nodes_kept
    return ExtentSurfaceBundle(
        cells=cells,
        lattice_nodes=lattice_nodes,
        lattice_struts=lattice_struts,
        net=net,
        topo=topo,
        report=report,
    )


# Back-compat aliases used by older tests during transition
def classify_extent_face_role(extent: float, *, half_max: float = HALF_EXTENT_MAX) -> str:
    e = float(extent)
    if e > float(half_max):
        return ROLE_FULL
    if e > 0.0:
        return ROLE_HALF_SIDE
    return ROLE_HALF_CUT


def midplane_cut_edge_midpoint(corners: np.ndarray, face_i: int) -> np.ndarray:
    return np.asarray(_hex_face_centers(corners)[int(face_i)], dtype=np.float64)


def midplane_diamond_for_cut_face(corners: np.ndarray, face_i: int) -> np.ndarray:
    return midplane_quad_for_face(corners, face_i)
