"""
Simplified surface dual from surviving exposed face-centers.

Surface nodes
  Exposed Cartesian faces whose outer face-center survived extent cull.

Surface dual struts (union of):
  1. Any core lattice strut whose both endpoints are surface nodes
     (incl. same-cell octahedral edges / mid-plane diamonds on the skin).
  2. Inter-cell stitches: two surface faces that share a 3D Cartesian edge
     but belong to different cells (added if not already a core edge).

No UV net, no CUT/SIDE dual roles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.hex_rules import _HEX_FACES, _hex_face_centers
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.sc_axis_cull_octahedral import (
    EXTENT_POLICY_MIDPLANE_05,
    HALF_EXTENT_MAX,
    face_keep_mask_from_extents,
    generate_extent_trimmed_octahedral,
    measure_material_extents,
)
from graphite.explicit.sc_boundary_states import (
    build_edt_sdf_field,
    estimate_hex_volume_fractions,
)
from graphite.explicit.sc_quantized_surface import _corner_key, _face_key
from graphite.explicit.sc_topological_dual import _edge_midpoint_key, _nodes_near_edge

_EDGE_DECIMALS = 5


@dataclass
class SimpleDualFace:
    face_id: int
    hex_i: int
    face_i: int
    center_3d: np.ndarray
    corners_3d: np.ndarray
    edges_3d: tuple[tuple[np.ndarray, np.ndarray], ...]
    node_gid: int  # index into lattice_nodes


@dataclass
class SimpleSurfaceDualResult:
    nodes: np.ndarray
    struts: np.ndarray
    faces: list[SimpleDualFace]
    report: dict = field(default_factory=dict)


def _collect_kept_cells(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float,
    samples_per_axis: int,
    extent_samples_per_axis: int,
    half_max: float,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    sdf_field=None,
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], object]:
    """Return list of (hex_i, corners, keep_mask) for VF candidates with material."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    vf = estimate_hex_volume_fractions(
        cad,
        elems,
        samples_per_axis=int(samples_per_axis),
        sdf_field=sdf_field,
    )
    if sdf_field is None:
        res = float(vf.voxel_resolution) if vf.voxel_resolution > 0 else 0.5
        sdf_field = build_edt_sdf_field(cad, res)

    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for hi in np.flatnonzero(vf.volume_fractions > float(empty_vf_max)):
        corners = np.asarray(elems[int(hi)], dtype=np.float64)
        ext = measure_material_extents(
            corners, sdf_field, samples_per_axis=int(extent_samples_per_axis)
        )
        if not ext.has_material:
            continue
        keep = face_keep_mask_from_extents(
            ext, half_max=half_max, extent_policy=extent_policy
        )
        if not np.any(keep):
            continue
        out.append((int(hi), corners, np.asarray(keep, dtype=bool)))
    return out, sdf_field


def build_simple_fc_surface_dual(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    half_max: float = HALF_EXTENT_MAX,
    round_decimals: int = 6,
    edge_decimals: int = _EDGE_DECIMALS,
    lattice_nodes: np.ndarray | None = None,
    lattice_struts: np.ndarray | None = None,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    prune_loose_external: bool = False,
    include_core_surface_struts: bool = True,
) -> SimpleSurfaceDualResult:
    """
    Surface dual = all struts connecting two surface nodes.

    Surface nodes = exposed-face centers that survived extent cull.
    Includes core lattice edges between those nodes plus inter-cell stitches.
    """
    cells, sdf = _collect_kept_cells(
        cad_mesh,
        hex_elems,
        empty_vf_max=empty_vf_max,
        samples_per_axis=samples_per_axis,
        extent_samples_per_axis=extent_samples_per_axis,
        half_max=half_max,
        extent_policy=extent_policy,
    )

    if lattice_nodes is None or lattice_struts is None:
        lattice_nodes, lattice_struts, _ = generate_extent_trimmed_octahedral(
            cad_mesh,
            hex_elems,
            empty_vf_max=empty_vf_max,
            samples_per_axis=samples_per_axis,
            extent_samples_per_axis=extent_samples_per_axis,
            half_max=half_max,
            round_decimals=round_decimals,
            sdf_field=sdf,
            extent_policy=extent_policy,
            prune_loose_external=prune_loose_external,
        )
    else:
        lattice_nodes = np.asarray(lattice_nodes, dtype=np.float64)
        lattice_struts = np.asarray(lattice_struts, dtype=np.int64)

    node_key_to_gid = {
        _corner_key(p, round_decimals): i
        for i, p in enumerate(lattice_nodes)
    }
    core_edges = {
        (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        for a, b in np.asarray(lattice_struts, dtype=np.int64).reshape(-1, 2)
    }

    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for hi, corners, _keep in cells:
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, round_decimals)
            face_owners.setdefault(fkey, []).append((hi, fi))

    faces: list[SimpleDualFace] = []
    surface_gids: set[int] = set()
    for hi, corners, keep in cells:
        fcs = _hex_face_centers(corners)
        for fi, face in enumerate(_HEX_FACES):
            if not bool(keep[fi]):
                continue
            fkey = _face_key(corners, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            ctr = fcs[fi]
            gid = node_key_to_gid.get(_corner_key(ctr, round_decimals))
            if gid is None:
                continue
            surface_gids.add(int(gid))
            corners_3d = np.array([corners[i] for i in face], dtype=np.float64)
            edges = tuple(
                (corners_3d[a].copy(), corners_3d[b].copy())
                for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))
            )
            faces.append(
                SimpleDualFace(
                    face_id=len(faces),
                    hex_i=hi,
                    face_i=fi,
                    center_3d=np.asarray(ctr, dtype=np.float64),
                    corners_3d=corners_3d,
                    edges_3d=edges,
                    node_gid=int(gid),
                )
            )

    dual_strut_set: set[tuple[int, int]] = set()
    n_from_core = 0
    if include_core_surface_struts:
        for a, b in core_edges:
            if a in surface_gids and b in surface_gids:
                dual_strut_set.add((a, b) if a < b else (b, a))
                n_from_core += 1

    edge_map: dict[tuple[float, float, float], list[int]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=edge_decimals)
            edge_map.setdefault(key, []).append(int(f.face_id))
    for k, ids in list(edge_map.items()):
        edge_map[k] = list(dict.fromkeys(ids))

    by_id = {f.face_id: f for f in faces}
    n_shared = 0
    n_stitches_added = 0
    n_skipped_same_cell = 0
    n_skipped_same_node = 0
    n_already_core = 0

    for _key, fids in edge_map.items():
        if len(fids) != 2:
            continue
        n_shared += 1
        fa, fb = by_id[fids[0]], by_id[fids[1]]
        if fa.hex_i == fb.hex_i:
            n_skipped_same_cell += 1
            continue
        ga, gb = int(fa.node_gid), int(fb.node_gid)
        if ga == gb:
            n_skipped_same_node += 1
            continue
        pair = (ga, gb) if ga < gb else (gb, ga)
        if pair in dual_strut_set:
            n_already_core += 1
            continue
        dual_strut_set.add(pair)
        n_stitches_added += 1

    used: set[int] = set()
    for a, b in dual_strut_set:
        used.add(a)
        used.add(b)
    used |= surface_gids

    old_to_new = {old: i for i, old in enumerate(sorted(used))}
    nodes = (
        np.asarray([lattice_nodes[old] for old in sorted(used)], dtype=np.float64)
        if used
        else np.empty((0, 3), dtype=np.float64)
    )
    struts = (
        np.array(
            sorted(
                (old_to_new[a], old_to_new[b])
                if old_to_new[a] < old_to_new[b]
                else (old_to_new[b], old_to_new[a])
                for a, b in dual_strut_set
            ),
            dtype=np.int64,
        )
        if dual_strut_set
        else np.empty((0, 2), dtype=np.int64)
    )

    report = {
        "n_cells": len(cells),
        "n_surface_nodes": int(len(surface_gids)),
        "n_candidate_fc_faces": len(faces),
        "n_shared_edges": n_shared,
        "n_core_surface_struts": int(n_from_core),
        "n_stitches_added": int(n_stitches_added),
        "n_skipped_same_cell": n_skipped_same_cell,
        "n_skipped_same_node": n_skipped_same_node,
        "n_already_core": n_already_core,
        "n_dual_struts": int(len(struts)),
        "n_dual_nodes": int(len(nodes)),
        "n_lattice_nodes": int(len(lattice_nodes)),
        "n_lattice_struts": int(len(lattice_struts)),
        "rule": "any_strut_between_two_surface_nodes_plus_intercell_stitches",
        "include_core_surface_struts": bool(include_core_surface_struts),
    }
    return SimpleSurfaceDualResult(nodes=nodes, struts=struts, faces=faces, report=report)


# ---------------------------------------------------------------------------
# Geometric surface dual: nodes on exposed SC faces (outer or mid-plane cut)
# ---------------------------------------------------------------------------

# Face index → (axis, is_max): 0=-Z, 1=+Z, 2=-Y, 3=+Y, 4=-X, 5=+X
_FACE_AXIS_MAX = (
    (2, False),
    (2, True),
    (1, False),
    (1, True),
    (0, False),
    (0, True),
)


@dataclass
class GeometricSurfaceFace:
    """Exposed SC face quad used only for occupancy / adjacency, not lattice type."""

    face_id: int
    hex_i: int
    face_i: int
    is_midplane: bool
    corners_3d: np.ndarray
    edges_3d: tuple[tuple[np.ndarray, np.ndarray], ...]
    node_gids: list[int]


@dataclass
class GeometricSurfaceDualResult:
    """Core lattice plus dual graph (extracted surface struts + added stitches)."""

    lattice_nodes: np.ndarray
    lattice_struts: np.ndarray
    dual_nodes: np.ndarray
    dual_struts: np.ndarray
    faces: list[GeometricSurfaceFace]
    surface_gids: set[int]
    report: dict = field(default_factory=dict)


def geometric_face_project_normal(face: GeometricSurfaceFace) -> np.ndarray:
    """Unit Cartesian outward of the exposed outer face, or of the culled face."""
    axis, is_max = _FACE_AXIS_MAX[int(face.face_i)]
    n = np.zeros(3, dtype=np.float64)
    n[int(axis)] = 1.0 if is_max else -1.0
    return n


def _append_unique_normal(bucket: list[np.ndarray], n: np.ndarray) -> None:
    vec = np.asarray(n, dtype=np.float64).reshape(3)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-14:
        return
    vec = vec / nrm
    for existing in bucket:
        if float(np.dot(existing, vec)) > 0.999:
            return
    bucket.append(vec)


def ownership_normals_for_geometric_dual(
    result: GeometricSurfaceDualResult,
) -> dict[int, list[np.ndarray]]:
    """Map dual-node index → unique owning-face outward normals.

    Dual node ``i`` is lattice node ``sorted(surface_gids)[i]``. A node that
    lies on several exposed quads (quarter-cell mid-planes, outer+cut, …)
    collects one normal per distinct direction. Projection blends those
    directions with cell-aspect weights (12×12×4 → 3×3×1).
    """
    gid_to_dual = {int(g): i for i, g in enumerate(sorted(result.surface_gids))}
    ownership: dict[int, list[np.ndarray]] = {
        i: [] for i in range(len(result.dual_nodes))
    }

    for f in result.faces:
        n = geometric_face_project_normal(f)
        for gid in f.node_gids:
            di = gid_to_dual.get(int(gid))
            if di is None:
                continue
            _append_unique_normal(ownership[di], n)
    return ownership


def axis_aligned_face_quad(
    corners: np.ndarray,
    face_i: int,
    *,
    midplane: bool = False,
) -> np.ndarray:
    """Four corners of an SC face (outer) or the parallel mid-plane cut quad."""
    c = np.asarray(corners, dtype=np.float64)
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    mid = 0.5 * (mn + mx)
    axis, is_max = _FACE_AXIS_MAX[int(face_i)]
    val = float(mid[axis]) if midplane else float(mx[axis] if is_max else mn[axis])
    a0, a1 = (i for i in range(3) if i != axis)
    lo0, hi0 = float(mn[a0]), float(mx[a0])
    lo1, hi1 = float(mn[a1]), float(mx[a1])
    # Winding in (a0, a1): (lo,lo), (hi,lo), (hi,hi), (lo,hi)
    uv = ((lo0, lo1), (hi0, lo1), (hi0, hi1), (lo0, hi1))
    out = np.zeros((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(uv):
        out[i, axis] = val
        out[i, a0] = u
        out[i, a1] = v
    return out


def point_on_quad(
    p: np.ndarray,
    corners_3d: np.ndarray,
    *,
    plane_eps: float,
    inplane_eps: float = 0.0,
) -> bool:
    """True if ``p`` lies on the axis-aligned rectangle ``corners_3d``."""
    c = np.asarray(corners_3d, dtype=np.float64)
    q = np.asarray(p, dtype=np.float64).reshape(3)
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    span = mx - mn
    axis = int(np.argmin(span))
    plane_val = float(c[:, axis].mean())
    if abs(float(q[axis]) - plane_val) > float(plane_eps):
        return False
    pad = float(inplane_eps)
    for i in range(3):
        if i == axis:
            continue
        if float(q[i]) < float(mn[i]) - pad or float(q[i]) > float(mx[i]) + pad:
            return False
    return True


def _point_to_seg_dist(p: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    v = p1 - p0
    w = p - p0
    c1 = float(np.dot(w, v))
    if c1 <= 0:
        return float(np.linalg.norm(p - p0))
    c2 = float(np.dot(v, v))
    if c2 <= c1:
        return float(np.linalg.norm(p - p1))
    b = c1 / c2
    pb = p0 + b * v
    return float(np.linalg.norm(p - pb))


def route_geometric_face_stitch(
    nodes_a: np.ndarray,
    nodes_b: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    near_tol_frac: float = 0.15,
    allow_far: bool = False,
) -> list[tuple[int, int]]:
    """
    Pair surface nodes across one shared Cartesian edge.

    Rules:
      1. Prefer nodes that lie on / very near the shared edge.
      2. If a face has exactly one node (typical outer face-center), that
         node may represent the whole face even if it is not on the edge.
      3. If allow_far is True (e.g. for cut faces), fall back to closest node(s).
    """
    a = np.asarray(nodes_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(nodes_b, dtype=np.float64).reshape(-1, 3)
    na = _nodes_near_edge(a, p0, p1, tol_frac=near_tol_frac)
    nb = _nodes_near_edge(b, p0, p1, tol_frac=near_tol_frac)

    if not na:
        if len(a) == 1:
            na = [0]
        elif allow_far and len(a) > 0:
            dists_a = [_point_to_seg_dist(p, p0, p1) for p in a]
            min_d_a = min(dists_a)
            na = [i for i, d in enumerate(dists_a) if abs(d - min_d_a) < 1e-4]

    if not nb:
        if len(b) == 1:
            nb = [0]
        elif allow_far and len(b) > 0:
            dists_b = [_point_to_seg_dist(p, p0, p1) for p in b]
            min_d_b = min(dists_b)
            nb = [i for i, d in enumerate(dists_b) if abs(d - min_d_b) < 1e-4]

    if not na or not nb:
        return []

    pairs: list[tuple[int, int]] = []
    if len(na) == 1 and len(nb) == 1:
        ia, ib = na[0], nb[0]
        if float(np.linalg.norm(a[ia] - b[ib])) < 1e-9:
            return []
        return [(ia, ib)]

    for ia in na:
        for ib in nb:
            if float(np.linalg.norm(a[ia] - b[ib])) < 1e-9:
                continue
            pairs.append((ia, ib))
    return pairs


def _quad_edges(corners_3d: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    c = np.asarray(corners_3d, dtype=np.float64)
    return tuple((c[a].copy(), c[b].copy()) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)))


def build_geometric_surface_dual(
    cad_mesh: trimesh.Trimesh,
    hex_elems: np.ndarray,
    *,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    half_max: float = HALF_EXTENT_MAX,
    round_decimals: int = 6,
    edge_decimals: int = _EDGE_DECIMALS,
    lattice_nodes: np.ndarray | None = None,
    lattice_struts: np.ndarray | None = None,
    extent_policy: str = EXTENT_POLICY_MIDPLANE_05,
    prune_loose_external: bool = False,
    plane_eps_frac: float = 1e-4,
) -> GeometricSurfaceDualResult:
    """
    Surface nodes = trimmed lattice nodes that lie on an exposed SC face.

    Exposed face quads:
      - outer Cartesian face if that direction kept (extent full)
      - mid-plane cut parallel to the face if that outer was culled

    Dual struts = core edges with both ends surface nodes, plus inter-cell
    stitches across shared 3D Cartesian edges of those quads.
    """
    cells, sdf = _collect_kept_cells(
        cad_mesh,
        hex_elems,
        empty_vf_max=empty_vf_max,
        samples_per_axis=samples_per_axis,
        extent_samples_per_axis=extent_samples_per_axis,
        half_max=half_max,
        extent_policy=extent_policy,
    )

    if lattice_nodes is None or lattice_struts is None:
        lattice_nodes, lattice_struts, _ = generate_extent_trimmed_octahedral(
            cad_mesh,
            hex_elems,
            empty_vf_max=empty_vf_max,
            samples_per_axis=samples_per_axis,
            extent_samples_per_axis=extent_samples_per_axis,
            half_max=half_max,
            round_decimals=round_decimals,
            sdf_field=sdf,
            extent_policy=extent_policy,
            prune_loose_external=prune_loose_external,
        )
    else:
        lattice_nodes = np.asarray(lattice_nodes, dtype=np.float64)
        lattice_struts = np.asarray(lattice_struts, dtype=np.int64)

    pts = lattice_nodes
    core_edges = {
        (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        for a, b in np.asarray(lattice_struts, dtype=np.int64).reshape(-1, 2)
    }

    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for hi, corners, _keep in cells:
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, round_decimals)
            face_owners.setdefault(fkey, []).append((hi, fi))

    faces: list[GeometricSurfaceFace] = []
    for hi, corners, keep in cells:
        extents = np.asarray(corners, dtype=np.float64).max(axis=0) - np.asarray(
            corners, dtype=np.float64
        ).min(axis=0)
        plane_eps = max(float(plane_eps_frac) * float(np.min(extents)), 1e-6)
        for fi, face in enumerate(_HEX_FACES):
            fkey = _face_key(corners, face, round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            is_mid = not bool(keep[fi])
            quad = axis_aligned_face_quad(corners, fi, midplane=is_mid)
            gids = [
                i
                for i, p in enumerate(pts)
                if point_on_quad(p, quad, plane_eps=plane_eps)
            ]
            if not gids and not is_mid:
                continue
            if not gids:
                continue
            faces.append(
                GeometricSurfaceFace(
                    face_id=len(faces),
                    hex_i=hi,
                    face_i=fi,
                    is_midplane=is_mid,
                    corners_3d=quad,
                    edges_3d=_quad_edges(quad),
                    node_gids=gids,
                )
            )

    surface_gids: set[int] = set()
    for f in faces:
        surface_gids.update(int(g) for g in f.node_gids)

    dual_strut_set: set[tuple[int, int]] = set()
    n_from_core = 0
    for a, b in core_edges:
        if a in surface_gids and b in surface_gids:
            dual_strut_set.add((a, b) if a < b else (b, a))
            n_from_core += 1

    edge_map: dict[tuple[float, float, float], list[int]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=edge_decimals)
            edge_map.setdefault(key, []).append(int(f.face_id))
    for k, ids in list(edge_map.items()):
        edge_map[k] = list(dict.fromkeys(ids))

    by_id = {f.face_id: f for f in faces}

    n_shared = 0
    n_stitches_added = 0
    n_skipped_same_cell = 0
    n_skipped_same_node = 0
    n_already_core = 0
    n_skipped_no_edge_node = 0
    edge_seg: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray]] = {}
    for f in faces:
        for p0, p1 in f.edges_3d:
            key = _edge_midpoint_key(p0, p1, decimals=edge_decimals)
            edge_seg.setdefault(key, (p0, p1))

    for key, fids in edge_map.items():
        if len(fids) != 2:
            continue
        n_shared += 1
        fa, fb = by_id[fids[0]], by_id[fids[1]]
        if fa.hex_i == fb.hex_i:
            n_skipped_same_cell += 1
            continue
        p0, p1 = edge_seg[key]
        nodes_a = (
            pts[np.asarray(fa.node_gids, dtype=np.int64)]
            if fa.node_gids
            else np.empty((0, 3), dtype=np.float64)
        )
        nodes_b = (
            pts[np.asarray(fb.node_gids, dtype=np.int64)]
            if fb.node_gids
            else np.empty((0, 3), dtype=np.float64)
        )
        local_pairs = route_geometric_face_stitch(nodes_a, nodes_b, p0, p1)
        if not local_pairs:
            n_skipped_no_edge_node += 1
            continue
        for ia, ib in local_pairs:
            ga = int(fa.node_gids[int(ia)])
            gb = int(fb.node_gids[int(ib)])
            if ga == gb:
                n_skipped_same_node += 1
                continue
            pair = (ga, gb) if ga < gb else (gb, ga)
            if pair in dual_strut_set:
                n_already_core += 1
                continue
            dual_strut_set.add(pair)
            n_stitches_added += 1

    used: set[int] = set(surface_gids)
    for a, b in dual_strut_set:
        used.add(a)
        used.add(b)
    old_to_new = {old: i for i, old in enumerate(sorted(used))}
    dual_nodes = (
        np.asarray([pts[old] for old in sorted(used)], dtype=np.float64)
        if used
        else np.empty((0, 3), dtype=np.float64)
    )
    dual_struts = (
        np.array(
            sorted(
                (old_to_new[a], old_to_new[b])
                if old_to_new[a] < old_to_new[b]
                else (old_to_new[b], old_to_new[a])
                for a, b in dual_strut_set
            ),
            dtype=np.int64,
        )
        if dual_strut_set
        else np.empty((0, 2), dtype=np.int64)
    )

    report = {
        "n_cells": len(cells),
        "n_surface_nodes": int(len(surface_gids)),
        "n_surface_faces": len(faces),
        "n_midplane_faces": sum(1 for f in faces if f.is_midplane),
        "n_outer_faces": sum(1 for f in faces if not f.is_midplane),
        "n_shared_edges": n_shared,
        "n_core_surface_struts": int(n_from_core),
        "n_stitches_added": int(n_stitches_added),
        "n_skipped_same_cell": n_skipped_same_cell,
        "n_skipped_same_node": n_skipped_same_node,
        "n_skipped_no_edge_node": n_skipped_no_edge_node,
        "n_already_core": n_already_core,
        "n_dual_struts": int(len(dual_struts)),
        "n_dual_nodes": int(len(dual_nodes)),
        "n_lattice_nodes": int(len(lattice_nodes)),
        "n_lattice_struts": int(len(lattice_struts)),
        "rule": "geometric_on_exposed_sc_face_plus_intercell_stitches",
    }
    return GeometricSurfaceDualResult(
        lattice_nodes=lattice_nodes,
        lattice_struts=lattice_struts,
        dual_nodes=dual_nodes,
        dual_struts=dual_struts,
        faces=faces,
        surface_gids=surface_gids,
        report=report,
    )
