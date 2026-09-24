"""
Shared SC conformal core: hex scaffold cull, DOF ironing helpers, depth tagging,
node identity maps, and surface-skin recipes.

Used by ``conformal_generator.generate_conformal_lattice`` for all SC hex rules.
Boolean intersect with CAD is intentionally not part of this core (debug/fast only).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Callable

import numpy as np
import trimesh

from graphite.explicit.hex_rules import _HEX_EDGES, _HEX_FACES
from graphite.explicit.hex_surface_dual import (
    HEX_FACES,
    generate_hex_surface_dual_cage,
    hex_node_ids_from_elements,
)
from graphite.explicit.hex_topology_module import (
    CONFORM_DOF_CORNERS,
    CONFORM_DOF_FACE_CENTROIDS,
    SKIN_MODE_CORNER_EDGE_CAGE,
    SKIN_MODE_FACE_CENTROID_DUAL,
    SKIN_MODE_FACE_LOCAL_RULE,
    SKIN_MODE_KELVIN_FACE_BRIDGE,
    SKIN_MODE_NONE,
    HexTopologyRule,
    generate_hex_topology,
    get_hex_topology_rule,
)
from graphite.explicit.proven_topologies import generate_background_grid

# Local face index -> (axis, value) on the unit cube [0,1]³ for hex8 ordering.
_HEX_FACE_PLANE: tuple[tuple[str, float], ...] = (
    ("w", 0.0),
    ("w", 1.0),
    ("v", 0.0),
    ("v", 1.0),
    ("u", 0.0),
    ("u", 1.0),
)


def _round_key(pt: np.ndarray, decimals: int = 6) -> tuple[float, float, float]:
    r = np.round(np.asarray(pt, dtype=np.float64), decimals)
    return (float(r[0]), float(r[1]), float(r[2]))


def _pt_key_fine(pt: np.ndarray) -> tuple[int, int, int]:
    return (
        int(np.round(pt[0] * 2000)),
        int(np.round(pt[1] * 2000)),
        int(np.round(pt[2] * 2000)),
    )


def safe_signed_distance(
    cad_mesh: trimesh.Trimesh,
    points: np.ndarray,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Negative = inside, positive = outside (inverted trimesh convention)."""
    n_pts = len(points)
    s_dists = np.zeros(n_pts, dtype=np.float64)
    proximity = trimesh.proximity.ProximityQuery(cad_mesh)
    for start in range(0, n_pts, chunk_size):
        end = min(start + chunk_size, n_pts)
        dists = proximity.signed_distance(points[start:end])
        s_dists[start:end] = -dists
    return s_dists


def snap_outside_corners_to_surface(
    nodes: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    outside_eps: float = 1e-3,
) -> tuple[np.ndarray, int]:
    """
    Closest-point project any corner with ``safe_signed_distance > outside_eps``.

    Prefer ``continue_outside_along_directions`` when a face-normal (or other
    directed) projection already chose a travel axis — closest-point can fold
    stair-step meshes by yanking neighbors onto different surface patches.
    """
    pts = np.asarray(nodes, dtype=np.float64).copy()
    if len(pts) == 0:
        return pts, 0
    sd = safe_signed_distance(cad_mesh, pts)
    outside = np.flatnonzero(sd > float(outside_eps))
    if outside.size == 0:
        return pts, 0
    closest, _, _ = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(pts[outside])
    pts[outside] = closest
    return pts, int(outside.size)


def continue_outside_along_directions(
    nodes: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    directions: np.ndarray,
    node_ids: np.ndarray,
    *,
    outside_eps: float = 1e-3,
    ray_length: float = 50.0,
    fallback_targets: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """
    Finish still-outside corners by raycasting along a chosen travel direction.

    ``directions`` / ``fallback_targets`` are indexed like ``node_ids`` (the
    subset being finished — typically iron DOFs). From each outside node's
    current position, cast forward along its unit direction and take the
    nearest hit within ``ray_length``. If the ray misses and
    ``fallback_targets`` is given, use that target; otherwise leave the node.
    """
    pts = np.asarray(nodes, dtype=np.float64).copy()
    dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    ids = np.asarray(node_ids, dtype=np.int32).reshape(-1)
    if len(dirs) != len(ids):
        raise ValueError("directions and node_ids must have the same length")
    if len(ids) == 0:
        return pts, 0

    sd = safe_signed_distance(cad_mesh, pts[ids])
    local_out = np.flatnonzero(sd > float(outside_eps))
    if local_out.size == 0:
        return pts, 0

    origins: list[np.ndarray] = []
    ray_dirs: list[np.ndarray] = []
    meta: list[int] = []  # index into local_out
    for j, loc_i in enumerate(local_out):
        d = dirs[int(loc_i)]
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            continue
        unit = d / nrm
        origins.append(pts[int(ids[int(loc_i)])] - 1e-4 * unit)
        ray_dirs.append(unit)
        meta.append(int(loc_i))

    hits: dict[int, tuple[float, np.ndarray]] = {}
    if origins:
        try:
            locations, index_ray, _tri = cad_mesh.ray.intersects_location(
                ray_origins=np.asarray(origins, dtype=np.float64),
                ray_directions=np.asarray(ray_dirs, dtype=np.float64),
                multiple_hits=True,
            )
        except Exception:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.int64)

        for loc, ray_i in zip(locations, index_ray):
            loc_i = meta[int(ray_i)]
            nid = int(ids[loc_i])
            travel = float(np.linalg.norm(loc - pts[nid]))
            if travel > float(ray_length) or travel < 1e-9:
                continue
            prev = hits.get(loc_i)
            if prev is None or travel < prev[0]:
                hits[loc_i] = (travel, np.asarray(loc, dtype=np.float64))

    n_moved = 0
    fb = None if fallback_targets is None else np.asarray(fallback_targets, dtype=np.float64)
    for loc_i in local_out:
        loc_i = int(loc_i)
        nid = int(ids[loc_i])
        if loc_i in hits:
            pts[nid] = hits[loc_i][1]
            n_moved += 1
        elif fb is not None:
            pts[nid] = fb[loc_i]
            n_moved += 1
    return pts, n_moved


def cull_hex_elements(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    volume_fraction_threshold: float = 0.5,
    mode: str = "conformal",
    signed_distance_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    VF-gate an SC background hex grid against ``cad_mesh``.

    Returns hex_elems (N,8,3), grid_nodes, surviving_cells (N,8) indices, n_partial.
    """
    grid_nodes, cells = generate_background_grid("SC", cad_mesh.bounds, cell_size)

    all_centroids: list[np.ndarray] = []
    cell_to_centroid_idx: list[list[int]] = []
    for cell in cells:
        idxs: list[int] = []
        corners = grid_nodes[cell]
        for fv in _HEX_FACES:
            idxs.append(len(all_centroids))
            all_centroids.append(corners[list(fv)].mean(axis=0))
        cell_to_centroid_idx.append(idxs)

    cents = np.asarray(all_centroids, dtype=np.float64)
    if signed_distance_fn is not None:
        s_dists = np.asarray(signed_distance_fn(cents), dtype=np.float64)
    else:
        s_dists = safe_signed_distance(cad_mesh, cents)

    kept_cells: list[np.ndarray] = []
    kept_elems: list[np.ndarray] = []
    n_partial = 0
    for i, cell in enumerate(cells):
        c_dists = s_dists[cell_to_centroid_idx[i]]
        n_inside = int(np.sum(c_dists <= 1e-5))
        inside_frac = n_inside / max(len(c_dists), 1)
        keep = False
        if mode == "boolean":
            keep = n_inside > 0
        elif inside_frac >= volume_fraction_threshold:
            keep = True
            if inside_frac < 1.0:
                n_partial += 1
        if keep:
            kept_cells.append(cell)
            kept_elems.append(grid_nodes[cell])

    if not kept_elems:
        raise ValueError("No hex cells survived trimming!")

    hex_elems = np.asarray(kept_elems, dtype=np.float64)
    surviving_cells = np.asarray(kept_cells, dtype=np.int64)
    print(
        f"  Surviving cells: {len(hex_elems)} / {len(cells)} "
        f"({n_partial} partial boundary cells retained)"
    )
    return hex_elems, grid_nodes, surviving_cells, n_partial


def classify_boundary_from_hex_elems(
    hex_elems: np.ndarray,
    round_decimals: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """scaffold_nodes, hex_node_ids (N,8), boundary_quads (F,4), exterior corner ids."""
    scaffold_nodes, hex_ids = hex_node_ids_from_elements(
        hex_elems, round_decimals=round_decimals
    )
    boundary_quads = find_exposed_faces(hex_ids)
    exterior_corners = exposed_node_ids_from_faces(boundary_quads)
    return scaffold_nodes, hex_ids, boundary_quads, exterior_corners


# ---------------------------------------------------------------------------
# Bidirectional boundary morph (face-adjacency exposed nodes → surface)
# ---------------------------------------------------------------------------

# (6, 4) local corner indices for vectorized face gather.
_HEX_FACES_IDX = np.asarray(HEX_FACES, dtype=np.intp)


def find_exposed_faces(hex_node_ids: np.ndarray) -> np.ndarray:
    """
    Faces owned by exactly one retained hex (exposed / exterior).

    Ownership is by shared corner-ID set — **not** node valency. Stair-step
    re-entrant corners (valency 5–6) are still captured when they sit on an
    exposed face.

    Parameters
    ----------
    hex_node_ids : (N, 8) int
        Global corner IDs per retained hex.

    Returns
    -------
    exposed_faces : (F, 4) int32
        Ordered corner IDs for each exposed quad (local HEX_FACES winding).
    """
    ids = np.asarray(hex_node_ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[1] != 8:
        raise ValueError(f"hex_node_ids must be (N, 8); got {ids.shape}")
    if ids.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int32)

    # (N, 6, 4) → (N*6, 4)
    face_corners = ids[:, _HEX_FACES_IDX].reshape(-1, 4)
    keys = np.sort(face_corners, axis=1)
    _uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    exposed_mask = counts[inverse] == 1
    if not np.any(exposed_mask):
        return np.empty((0, 4), dtype=np.int32)
    return face_corners[exposed_mask].astype(np.int32, copy=False)


def exposed_node_ids_from_faces(exposed_faces: np.ndarray) -> np.ndarray:
    """Unique node IDs appearing on any exposed face."""
    faces = np.asarray(exposed_faces)
    if faces.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.unique(faces.ravel()).astype(np.int32)


def exposed_faces_with_outward_normals(
    hex_node_ids: np.ndarray,
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exposed faces plus outward unit normals (pointing out of their owner hex).

    Returns
    -------
    exposed_faces : (F, 4) int32
    normals : (F, 3) float64
    """
    ids = np.asarray(hex_node_ids, dtype=np.int32)
    nodes = np.asarray(nodes, dtype=np.float64)
    if ids.ndim != 2 or ids.shape[1] != 8:
        raise ValueError(f"hex_node_ids must be (N, 8); got {ids.shape}")
    if ids.shape[0] == 0:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )

    face_corners = ids[:, _HEX_FACES_IDX].reshape(-1, 4)
    hex_of_face = np.repeat(np.arange(len(ids), dtype=np.int32), 6)
    keys = np.sort(face_corners, axis=1)
    _uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    exposed_mask = counts[inverse] == 1
    if not np.any(exposed_mask):
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )

    faces = face_corners[exposed_mask].astype(np.int32, copy=False)
    owners = hex_of_face[exposed_mask]
    corners = nodes[faces]
    e1 = corners[:, 1] - corners[:, 0]
    e2 = corners[:, 3] - corners[:, 0]
    normals = np.cross(e1, e2)
    face_centroids = corners.mean(axis=1)
    hex_centroids = nodes[ids[owners]].mean(axis=1)
    # Flip so the normal points away from the owning hex centre.
    inward = np.einsum("ij,ij->i", normals, hex_centroids - face_centroids)
    normals[inward > 0.0] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)
    return faces, normals


def project_points_face_normal_aware(
    points: np.ndarray,
    node_ids: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    exposed_faces: np.ndarray,
    face_normals: np.ndarray,
    *,
    ray_length: float = 50.0,
) -> np.ndarray:
    """
    Project exposed scaffold nodes using their owning exposed-face normals.

    Closest-point projection alone snaps a bottom-edge stair-step node to the
    floor because the floor is nearer than the side wall. This path instead:

    1. For every exposed face a node sits on, cast rays along ± the face's
       outward normal and keep the *nearest* hit for that face.
    2. If the node belongs to several faces (floor + stair side), keep the
       face-candidate with the *largest* travel. The side wall usually wins
       over the nearby floor, which is the stair-step lateral snap.
    3. Fall back to ordinary closest-point if no ray hits.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    node_ids = np.asarray(node_ids, dtype=np.int32).reshape(-1)
    faces = np.asarray(exposed_faces, dtype=np.int32)
    normals = np.asarray(face_normals, dtype=np.float64)
    if len(points) != len(node_ids):
        raise ValueError("points and node_ids must have the same length")
    if len(points) == 0:
        return points.copy()

    id_to_local = {int(nid): i for i, nid in enumerate(node_ids)}

    # One candidate per (node, owning face): nearest hit along ± face normal.
    face_hits: dict[tuple[int, int], tuple[float, np.ndarray]] = {}

    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []  # (local_index, face_index)
    for face_i, (face, normal) in enumerate(zip(faces, normals)):
        for nid in face:
            local = id_to_local.get(int(nid))
            if local is None:
                continue
            for sign in (1.0, -1.0):
                origins.append(points[local])
                directions.append(sign * normal)
                meta.append((local, face_i))

    if origins:
        origins_arr = np.asarray(origins, dtype=np.float64)
        dirs_arr = np.asarray(directions, dtype=np.float64)
        origins_arr = origins_arr - 1e-4 * dirs_arr
        try:
            locations, index_ray, _tri = cad_mesh.ray.intersects_location(
                ray_origins=origins_arr,
                ray_directions=dirs_arr,
                multiple_hits=True,
            )
        except Exception:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.int64)

        for loc, ray_i in zip(locations, index_ray):
            local, face_i = meta[int(ray_i)]
            travel = float(np.linalg.norm(loc - points[local]))
            if travel > float(ray_length) or travel < 1e-9:
                continue
            key = (local, face_i)
            prev = face_hits.get(key)
            if prev is None or travel < prev[0]:
                face_hits[key] = (travel, np.asarray(loc, dtype=np.float64))

    # Collapse per-face nearest hits → per-node candidates.
    candidates: dict[int, list[tuple[float, np.ndarray]]] = {
        i: [] for i in range(len(points))
    }
    for (local, _face_i), (travel, loc) in face_hits.items():
        candidates[local].append((travel, loc))

    need_fallback = [i for i in range(len(points)) if not candidates[i]]
    if need_fallback:
        fallback_pts, _, _ = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(
            points[need_fallback]
        )
        for local, pt in zip(need_fallback, fallback_pts):
            pt = np.asarray(pt, dtype=np.float64)
            travel = float(np.linalg.norm(pt - points[local]))
            candidates[local].append((travel, pt))

    projected = points.copy()
    for i in range(len(points)):
        cands = candidates[i]
        if not cands:
            continue
        # Single-face → that hit. Multi-face → farthest face-candidate so a
        # stair side wall beats the nearby floor.
        projected[i] = cands[int(np.argmax([t for t, _ in cands]))][1]

    return projected


def redirect_interior_pull_to_exposed_normals(
    points: np.ndarray,
    targets: np.ndarray,
    node_ids: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    exposed_faces: np.ndarray,
    face_normals: np.ndarray,
    *,
    signed_distance: np.ndarray | None = None,
    min_cos: float = 0.35,
    ray_length: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep interior-side snaps inside the cone of a corner's own exposed faces.

    A corner in a stair-step re-entrant notch keeps hexes both above and below
    it, so its only exposed faces are lateral. Closest-point still aims it
    straight down at the floor, because the floor is nearer than the side wall.
    The corner is then dragged through a brick it still belongs to, the travel
    clamp strands it mid-brick, and the strut above it stretches while the one
    below it shrinks.

    Any corner that is still *inside* the CAD and whose target displacement is
    not aligned (``min_cos``) with one of its own exposed-face outward normals
    gets re-aimed: raycast along each of those normals and keep the *nearest*
    hit. Nearest, not farthest, so a corner never overshoots laterally the way
    ``projection_mode="face_normal"`` does. Corners already outside the CAD are
    left alone — their inward closest-point pull is correct by construction.

    Returns the amended targets and the local indices that were re-aimed.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.asarray(targets, dtype=np.float64).reshape(-1, 3).copy()
    ids = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    faces = np.asarray(exposed_faces, dtype=np.int64)
    normals = np.asarray(face_normals, dtype=np.float64)
    if not (len(pts) == len(out) == len(ids)):
        raise ValueError("points, targets and node_ids must have the same length")
    if len(pts) == 0 or faces.size == 0:
        return out, np.empty(0, dtype=np.int64)

    id_to_local = {int(nid): i for i, nid in enumerate(ids)}
    per_node: dict[int, list[np.ndarray]] = {}
    for face, normal in zip(faces, normals):
        for nid in face:
            local = id_to_local.get(int(nid))
            if local is not None:
                per_node.setdefault(local, []).append(normal)

    sd = (
        safe_signed_distance(cad_mesh, pts)
        if signed_distance is None
        else np.asarray(signed_distance, dtype=np.float64).reshape(-1)
    )

    redirected: list[int] = []
    for local, normal_list in per_node.items():
        if sd[local] >= -1e-6:
            continue
        disp = out[local] - pts[local]
        travel = float(np.linalg.norm(disp))
        if travel < 1e-9:
            continue
        dirs = np.unique(np.round(np.asarray(normal_list, dtype=np.float64), 6), axis=0)
        unit = disp / travel
        if float(np.max(dirs @ unit)) >= float(min_cos):
            continue

        best: tuple[float, np.ndarray] | None = None
        for d in dirs:
            origins = (pts[local] - 1e-4 * d).reshape(1, 3)
            try:
                hits, _idx, _tri = cad_mesh.ray.intersects_location(
                    ray_origins=origins,
                    ray_directions=d.reshape(1, 3),
                    multiple_hits=False,
                )
            except Exception:
                continue
            for hit in hits:
                t = float(np.linalg.norm(hit - pts[local]))
                if t < 1e-9 or t > float(ray_length):
                    continue
                if best is None or t < best[0]:
                    best = (t, np.asarray(hit, dtype=np.float64))
        if best is not None:
            out[local] = best[1]
            redirected.append(local)

    return out, np.asarray(sorted(redirected), dtype=np.int64)


def surface_adjacency_from_exposed_faces(
    exposed_faces: np.ndarray,
    n_nodes: int,
) -> list[list[int]]:
    """Undirected neighbour lists from exposed quad edges (skin graph only)."""
    faces = np.asarray(exposed_faces, dtype=np.int64)
    adj: list[list[int]] = [[] for _ in range(int(n_nodes))]
    if faces.size == 0:
        return adj
    seen: set[tuple[int, int]] = set()
    for face in faces:
        corners = [int(c) for c in face]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in seen:
                continue
            seen.add(edge)
            if 0 <= a < n_nodes and 0 <= b < n_nodes:
                adj[a].append(b)
                adj[b].append(a)
    return adj


def cad_sharp_edge_segments(
    cad_mesh: trimesh.Trimesh,
    *,
    dihedral_deg: float = 60.0,
) -> np.ndarray:
    """
    Return ``(E, 2, 3)`` segments for CAD edges whose adjacent-face dihedral
    exceeds ``dihedral_deg`` (feature creases / corners).
    """
    mesh = cad_mesh
    if (
        not hasattr(mesh, "face_adjacency")
        or len(getattr(mesh, "face_adjacency", [])) == 0
    ):
        return np.empty((0, 2, 3), dtype=np.float64)
    angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)
    sharp = angles >= np.deg2rad(float(dihedral_deg))
    if not np.any(sharp):
        return np.empty((0, 2, 3), dtype=np.float64)
    edge_ids = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[sharp]
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    return verts[edge_ids]


def point_to_segments_distance(
    points: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Minimum Euclidean distance from each point to any segment."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    segs = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    if len(pts) == 0:
        return np.empty(0, dtype=np.float64)
    if len(segs) == 0:
        return np.full(len(pts), np.inf, dtype=np.float64)

    a = segs[:, 0][None, :, :]
    b = segs[:, 1][None, :, :]
    p = pts[:, None, :]
    ab = b - a
    ab2 = np.einsum("ije,ije->ij", ab, ab)
    t = np.einsum("ije,ije->ij", p - a, ab) / np.maximum(ab2, 1e-18)
    t = np.clip(t, 0.0, 1.0)
    closest = a + ab * t[:, :, None]
    dist = np.linalg.norm(p - closest, axis=2)
    return dist.min(axis=1)


def classify_surface_relax_roles(
    points: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    sharp_segments: np.ndarray | None = None,
    dihedral_deg: float = 60.0,
    feature_proximity: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Role per surface node: 0 = free (tangent plane), 1 = crease (slide along
    feature), 2 = pinned (near a sharp corner / multi-feature).

    MVP: nodes within ``feature_proximity`` of a sharp CAD edge are pinned so
    they cannot slide around a floor/wall crease. All other nodes are free in
    the local CAD tangent plane. Crease-sliding (role 1) is reserved for a
    later refinement; the pin is enough to protect stair-step gates.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    roles = np.zeros(len(pts), dtype=np.int32)
    if len(pts) == 0:
        return roles, np.zeros((0, 3), dtype=np.float64)

    closest, _dist, tri_id = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(pts)
    normals = np.asarray(cad_mesh.face_normals, dtype=np.float64)[
        np.asarray(tri_id, dtype=np.int64)
    ]
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nn, 1e-12)

    segs = (
        cad_sharp_edge_segments(cad_mesh, dihedral_deg=dihedral_deg)
        if sharp_segments is None
        else np.asarray(sharp_segments, dtype=np.float64)
    )
    if len(segs):
        d_feat = point_to_segments_distance(pts, segs)
        roles[d_feat <= float(feature_proximity)] = 2

    _ = closest  # proximity also validates the query path
    return roles, normals


def skin_edge_length_stats(
    nodes: np.ndarray,
    exposed_faces: np.ndarray,
) -> dict[str, float]:
    """Median / mean / std / CV of unique exposed-face edge lengths."""
    faces = np.asarray(exposed_faces, dtype=np.int64)
    pts = np.asarray(nodes, dtype=np.float64)
    if faces.size == 0 or len(pts) == 0:
        return {
            "skin_edge_median": 0.0,
            "skin_edge_mean": 0.0,
            "skin_edge_std": 0.0,
            "skin_edge_cv": 0.0,
            "n_skin_edges": 0,
        }
    seen: set[tuple[int, int]] = set()
    lengths: list[float] = []
    for face in faces:
        corners = [int(c) for c in face]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in seen:
                continue
            seen.add(edge)
            lengths.append(float(np.linalg.norm(pts[b] - pts[a])))
    arr = np.asarray(lengths, dtype=np.float64)
    med = float(np.median(arr)) if arr.size else 0.0
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    return {
        "skin_edge_median": med,
        "skin_edge_mean": mean,
        "skin_edge_std": std,
        "skin_edge_cv": (std / med) if med > 1e-12 else 0.0,
        "n_skin_edges": int(arr.size),
    }


def relax_surface_nodes_tangent(
    nodes: np.ndarray,
    iron_ids: np.ndarray,
    exposed_faces: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    iterations: int = 20,
    alpha: float = 0.5,
    max_travel: float = 2.0,
    dihedral_deg: float = 60.0,
    feature_proximity: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Even out iron-node spacing on the CAD surface (Ideas 1–3).

    1. Build the skin adjacency from exposed faces.
    2. Pin nodes near sharp CAD features so they cannot slide onto another patch.
    3. Push/pull free iron nodes along skin edges toward the median skin edge
       length (edge-length springs — avoids open-mesh Laplacian shrinkage).
    4. Keep only the tangential component of each step (CAD normal).
    5. Re-snap to the CAD with closest-point after each iteration; reject steps
       that flip the CAD normal (patch jump).
    6. Cap cumulative travel from the start pose so a node cannot migrate far
       from its home cell.
    """
    out = np.asarray(nodes, dtype=np.float64).copy()
    iron = np.asarray(iron_ids, dtype=np.int64).reshape(-1)
    report = {
        "surface_relax_iterations": int(iterations),
        "surface_relax_alpha": float(alpha),
        "surface_relax_max_travel": float(max_travel),
        "n_surface_free": 0,
        "n_surface_pinned": 0,
        "n_surface_moved": 0,
        "max_surface_step": 0.0,
        "max_surface_travel": 0.0,
        "skin_edge_target": 0.0,
    }
    if iron.size == 0 or int(iterations) <= 0:
        return out, report

    start = out[iron].copy()
    roles, _normals0 = classify_surface_relax_roles(
        start,
        cad_mesh,
        dihedral_deg=dihedral_deg,
        feature_proximity=feature_proximity,
    )
    free_mask = roles == 0
    report["n_surface_free"] = int(np.count_nonzero(free_mask))
    report["n_surface_pinned"] = int(np.count_nonzero(~free_mask))
    if not np.any(free_mask):
        return out, report

    adj = surface_adjacency_from_exposed_faces(exposed_faces, len(out))
    iron_set = set(int(i) for i in iron)
    stats0 = skin_edge_length_stats(out, exposed_faces)
    L_target = float(stats0["skin_edge_median"])
    report["skin_edge_target"] = L_target
    if L_target < 1e-9:
        return out, report

    free_gids = {int(iron[i]) for i in range(len(iron)) if free_mask[i]}
    max_step = 0.0
    n_moved = 0
    pq = trimesh.proximity.ProximityQuery(cad_mesh)

    for _ in range(int(iterations)):
        cur = out[iron]
        _closest, _d, tri_id = pq.on_surface(cur)
        normals = np.asarray(cad_mesh.face_normals, dtype=np.float64)[
            np.asarray(tri_id, dtype=np.int64)
        ]
        nn = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(nn, 1e-12)

        proposed = cur.copy()
        for local, gid in enumerate(iron):
            gid = int(gid)
            if not free_mask[local]:
                continue
            nbrs = [n for n in adj[gid] if n in iron_set]
            if not nbrs:
                continue
            force = np.zeros(3, dtype=np.float64)
            for nb in nbrs:
                edge = out[nb] - cur[local]
                length = float(np.linalg.norm(edge))
                if length < 1e-12:
                    continue
                # Move this node so the edge approaches L_target. If the
                # neighbour is pinned / non-free, take the full correction;
                # if both are free, split the correction.
                corr = (length - L_target) * (edge / length)
                if nb in free_gids:
                    corr *= 0.5
                force += corr
            force /= float(len(nbrs))
            # Tangential only.
            n = normals[local]
            force = force - float(np.dot(force, n)) * n
            candidate = cur[local] + float(alpha) * force
            travel = candidate - start[local]
            tnorm = float(np.linalg.norm(travel))
            if tnorm > float(max_travel) and tnorm > 1e-12:
                candidate = start[local] + travel * (float(max_travel) / tnorm)
            proposed[local] = candidate
            sn = float(np.linalg.norm(candidate - cur[local]))
            if sn > 1e-9:
                n_moved += 1
                max_step = max(max_step, sn)

        free_local = np.flatnonzero(free_mask)
        if free_local.size:
            snapped, _, tri2 = pq.on_surface(proposed[free_local])
            new_normals = np.asarray(cad_mesh.face_normals, dtype=np.float64)[
                np.asarray(tri2, dtype=np.int64)
            ]
            nn2 = np.linalg.norm(new_normals, axis=1, keepdims=True)
            new_normals = new_normals / np.maximum(nn2, 1e-12)
            old_n = normals[free_local]
            flip = np.einsum("ij,ij->i", old_n, new_normals) < 0.25
            keep = snapped.copy()
            if np.any(flip):
                keep[flip] = cur[free_local[flip]]
            proposed[free_local] = keep
        out[iron] = proposed

    travel = np.linalg.norm(out[iron] - start, axis=1)
    report["n_surface_moved"] = int(n_moved)
    report["max_surface_step"] = float(max_step)
    report["max_surface_travel"] = float(travel.max()) if travel.size else 0.0
    return out, report


def project_points_onto_sphere(
    points: np.ndarray,
    center: np.ndarray | tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """
    Algebraic nearest-point projection onto sphere ``||x - C|| = R``.

    Bidirectional: points inside move outward; points outside move inward.
    Vectorized over an ``(M, 3)`` point cloud.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r <= 0.0:
        raise ValueError("sphere radius must be > 0")

    delta = pts - c
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    # Degenerate: point at center → arbitrary radial direction
    degenerate = norms.ravel() < 1e-15
    if np.any(degenerate):
        delta = delta.copy()
        norms = norms.copy()
        delta[degenerate] = np.array([1.0, 0.0, 0.0])
        norms[degenerate] = 1.0
    return c + delta * (r / norms)


def bidirectional_morph_hex_to_sphere(
    hex_elems: np.ndarray,
    *,
    center: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float,
    round_decimals: int = 8,
) -> tuple[np.ndarray, dict]:
    """
    VF-retained hex cage → expose faces by adjacency → snap all exposed
    corners onto the sphere (no valency gate).

    After a 50% VF cull, travel distance is typically ≲ 0.5 * cell size.

    Returns
    -------
    deformed_hexes : (N, 8, 3)
    report : dict with exposed face/node counts and displacement stats
    """
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must be (N, 8, 3); got {elems.shape}")

    scaffold, hex_ids = hex_node_ids_from_elements(elems, round_decimals=round_decimals)
    exposed_faces = find_exposed_faces(hex_ids)
    exposed_ids = exposed_node_ids_from_faces(exposed_faces)

    nodes = scaffold.copy()
    disp = 0.0
    if exposed_ids.size:
        before = nodes[exposed_ids]
        after = project_points_onto_sphere(before, center, radius)
        nodes[exposed_ids] = after
        disp = float(np.linalg.norm(after - before, axis=1).max()) if len(before) else 0.0

    deformed = nodes[hex_ids]
    report = {
        "n_exposed_faces": int(exposed_faces.shape[0]),
        "n_exposed_nodes": int(exposed_ids.size),
        "max_projection_distance": disp,
        "exposed_faces": exposed_faces,
        "exposed_node_ids": exposed_ids,
        "scaffold_nodes": nodes,
        "hex_corner_ids": hex_ids,
    }
    return deformed, report


# Six-tet decomposition of a hex8 in standard corner ordering. Verified to sum
# to exactly 1.0 on the unit cube.
_HEX_TETS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 3, 7),
    (0, 1, 7, 4),
    (1, 2, 3, 7),
    (1, 2, 7, 6),
    (1, 4, 5, 7),
    (1, 5, 6, 7),
)


def hex_volumes(hex_elems: np.ndarray) -> np.ndarray:
    """Signed volume of each ``(N, 8, 3)`` hex brick."""
    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must be (N, 8, 3); got {elems.shape}")
    total = np.zeros(len(elems), dtype=np.float64)
    for a, b, c, d in _HEX_TETS:
        e1 = elems[:, b] - elems[:, a]
        e2 = elems[:, c] - elems[:, a]
        e3 = elems[:, d] - elems[:, a]
        total += np.einsum("ij,ij->i", np.cross(e1, e2), e3) / 6.0
    return total


def resolve_cell_dims(
    cell_size: float | tuple[float, float, float] | np.ndarray | None,
) -> np.ndarray | None:
    """Normalise a scalar or per-axis cell size into a ``(3,)`` array."""
    if cell_size is None:
        return None
    dims = np.asarray(cell_size, dtype=np.float64)
    if dims.ndim == 0:
        dims = np.full(3, float(dims), dtype=np.float64)
    if dims.shape != (3,) or np.any(dims <= 0.0):
        raise ValueError(
            "cell_size must be a positive scalar or three positive dimensions"
        )
    return dims


def clamp_projection_travel(
    before: np.ndarray,
    after: np.ndarray,
    cell_dims: np.ndarray,
    max_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Limit how far a boundary projection may drag a scaffold corner.

    Each displacement is scaled so no axis component exceeds
    ``max_factor * cell_dims[axis]``. Direction is preserved, so nodes still
    travel toward the surface — just not far enough to crush the brick behind
    them. The per-axis budget matters for anisotropic cells: a 12 x 12 x 4 mm
    cell can absorb a 6 mm lateral pull but not a 6 mm vertical one.

    Returns the clamped points and the per-node scale factor applied.
    """
    before = np.asarray(before, dtype=np.float64).reshape(-1, 3)
    after = np.asarray(after, dtype=np.float64).reshape(-1, 3)
    budget = np.asarray(cell_dims, dtype=np.float64).reshape(3) * float(max_factor)
    if np.any(budget <= 0.0):
        raise ValueError("projection budget must be positive")

    disp = after - before
    magnitude = np.abs(disp)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_axis = np.where(
            magnitude > 1e-12, budget[None, :] / np.maximum(magnitude, 1e-12), np.inf
        )
    scale = np.minimum(per_axis.min(axis=1), 1.0)
    return before + disp * scale[:, None], scale


def _hex_corner_jacobians(nodes: np.ndarray, hex_ids: np.ndarray) -> np.ndarray:
    """Scalar triple product of local edge frame at corner 0 of each hex."""
    corners = nodes[np.asarray(hex_ids, dtype=np.intp)]
    e01 = corners[:, 1] - corners[:, 0]
    e03 = corners[:, 3] - corners[:, 0]
    e04 = corners[:, 4] - corners[:, 0]
    return np.einsum("ij,ij->i", np.cross(e01, e03), e04)


def apply_radial_equalization_relaxation(
    nodes: np.ndarray,
    struts: np.ndarray,
    exposed_ids: np.ndarray,
    *,
    center: np.ndarray | tuple[float, float, float],
    radius: float,
    cell_size: float,
    node_depths: np.ndarray | None = None,
    hex_ids: np.ndarray | None = None,
    iterations: int = 300,
    step: float = 0.2,
    radial_weight: float = 1.0,
    tangential_weight: float = 0.15,
    radial_dot_threshold: float = 0.5,
    travel_cap_factor: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Equalize mostly-radial scaffold edge lengths with the surface hard-fixed.

    Experimental sphere-only smoother. Exposed nodes remain exactly where the
    bidirectional projection left them (no tangential slide). Interior nodes
    are free within a travel cap of ``travel_cap_factor * cell_size`` from the
    start pose.

    Radial edges (``|ê · r̂| >= radial_dot_threshold`` at the midpoint) spring
    toward a shared rest length ``L_r = R / max(depths)``. Tangential edges get
    a weak spring toward the current mean tangential length (anti-bunching).
    """
    pos = np.asarray(nodes, dtype=np.float64).copy()
    struts = np.asarray(struts, dtype=np.int64).reshape(-1, 2)
    exposed = np.asarray(exposed_ids, dtype=np.int64).ravel()
    c = np.asarray(center, dtype=np.float64).reshape(3)
    r = float(radius)
    if r <= 0.0:
        raise ValueError("sphere radius must be > 0")
    if cell_size <= 0.0:
        raise ValueError("cell_size must be > 0")
    if struts.size == 0:
        return pos, {
            "mean_radial_length": 0.0,
            "radial_length_std": 0.0,
            "n_surface_slide": 0,
            "max_interior_travel": 0.0,
            "radial_rest_length": 0.0,
            "n_radial_edges": 0,
        }

    start = pos.copy()
    travel_cap = float(travel_cap_factor) * float(cell_size)
    is_exposed = np.zeros(len(pos), dtype=bool)
    if exposed.size:
        is_exposed[exposed] = True
    interior = ~is_exposed

    if node_depths is None:
        max_depth = 1
    else:
        depths = np.asarray(node_depths)
        max_depth = int(max(1, int(depths.max()) if depths.size else 1))
    l_r = r / float(max_depth)

    u_idx = struts[:, 0]
    v_idx = struts[:, 1]
    jac_eps = 1e-14
    start_jac = None
    if hex_ids is not None and len(hex_ids):
        start_jac = _hex_corner_jacobians(start, hex_ids)

    for _ in range(int(iterations)):
        prev = pos.copy()
        mid = 0.5 * (pos[u_idx] + pos[v_idx])
        d = pos[v_idx] - pos[u_idx]
        length = np.linalg.norm(d, axis=1)
        safe_len = np.maximum(length, 1e-12)
        unit = d / safe_len[:, None]

        radial_hat = mid - c
        radial_norm = np.linalg.norm(radial_hat, axis=1, keepdims=True)
        radial_hat = radial_hat / np.maximum(radial_norm, 1e-12)
        abs_dot = np.abs(np.einsum("ij,ij->i", unit, radial_hat.reshape(-1, 3)))
        is_radial = abs_dot >= float(radial_dot_threshold)
        is_tangential = ~is_radial

        rest = np.where(is_radial, l_r, 0.0)
        if np.any(is_tangential):
            mean_tang = float(length[is_tangential].mean())
            rest = np.where(is_tangential, mean_tang, rest)

        weight = np.where(is_radial, float(radial_weight), float(tangential_weight))
        force_mag = (weight * (rest - length))[:, None] * unit
        acc = np.zeros_like(pos)
        np.add.at(acc, u_idx, -force_mag)
        np.add.at(acc, v_idx, force_mag)
        counts = np.zeros(len(pos), dtype=np.float64)
        np.add.at(counts, u_idx, 1.0)
        np.add.at(counts, v_idx, 1.0)
        acc /= np.maximum(counts, 1.0)[:, None]

        # Only interior nodes move; exposed stay at the projected pose.
        pos[interior] = pos[interior] + float(step) * acc[interior]
        if exposed.size:
            pos[exposed] = start[exposed]

        if np.any(interior) and travel_cap > 0.0:
            delta = pos[interior] - start[interior]
            dist = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, travel_cap / np.maximum(dist, 1e-15))
            pos[interior] = start[interior] + delta * scale

        if start_jac is not None:
            jac = _hex_corner_jacobians(pos, hex_ids)
            # Reject the step if any hex that started non-inverted is now inverted.
            flipped = (start_jac > jac_eps) & (jac < jac_eps)
            if np.any(flipped):
                pos = prev

    # Stats on final radial edges
    mid = 0.5 * (pos[u_idx] + pos[v_idx])
    d = pos[v_idx] - pos[u_idx]
    length = np.linalg.norm(d, axis=1)
    safe_len = np.maximum(length, 1e-12)
    unit = d / safe_len[:, None]
    radial_hat = mid - c
    radial_norm = np.linalg.norm(radial_hat, axis=1, keepdims=True)
    radial_hat = radial_hat / np.maximum(radial_norm, 1e-12)
    abs_dot = np.abs(np.einsum("ij,ij->i", unit, radial_hat.reshape(-1, 3)))
    is_radial = abs_dot >= float(radial_dot_threshold)
    radial_lengths = length[is_radial] if np.any(is_radial) else np.array([0.0])

    max_interior_travel = 0.0
    if np.any(interior):
        max_interior_travel = float(
            np.linalg.norm(pos[interior] - start[interior], axis=1).max()
        )

    report = {
        "mean_radial_length": float(radial_lengths.mean()),
        "radial_length_std": float(radial_lengths.std()) if radial_lengths.size else 0.0,
        "n_surface_slide": 0,
        "max_interior_travel": max_interior_travel,
        "radial_rest_length": float(l_r),
        "n_radial_edges": int(np.count_nonzero(is_radial)),
    }
    return pos, report


def build_lattice_node_index(
    nodes: np.ndarray,
    round_decimals: int = 6,
) -> dict[tuple[float, float, float], int]:
    out: dict[tuple[float, float, float], int] = {}
    for i, pt in enumerate(np.asarray(nodes, dtype=np.float64)):
        out[_round_key(pt, round_decimals)] = int(i)
    return out


def resolve_boundary_iron_ids(
    hex_elems: np.ndarray,
    lattice_nodes: np.ndarray,
    rule: HexTopologyRule,
    round_decimals: int = 6,
    match_tol: float | None = None,
) -> np.ndarray:
    """Lattice node indices that are exterior DOFs per ``rule.conform_dofs``."""
    from scipy.spatial import cKDTree

    scaffold_nodes, _hex_ids, boundary_quads, exterior_corner_ids = (
        classify_boundary_from_hex_elems(hex_elems, round_decimals=max(round_decimals, 6))
    )
    nodes = np.asarray(lattice_nodes, dtype=np.float64)
    if len(nodes) == 0:
        return np.empty(0, dtype=np.int64)

    # Default match tolerance: ~5% of mean hex edge length
    if match_tol is None:
        sample = hex_elems[: min(20, len(hex_elems))]
        edge_lens = []
        for elem in sample:
            edge_lens.append(np.linalg.norm(elem[1] - elem[0]))
        match_tol = float(np.mean(edge_lens) * 0.05) if edge_lens else 1e-3

    tree = cKDTree(nodes)
    node_index = build_lattice_node_index(nodes, round_decimals=round_decimals)
    iron: set[int] = set()

    def _nearest(pt: np.ndarray) -> int | None:
        d, idx = tree.query(np.asarray(pt, dtype=np.float64), k=1)
        if float(d) <= match_tol:
            return int(idx)
        return node_index.get(_round_key(pt, round_decimals))

    if CONFORM_DOF_CORNERS in rule.conform_dofs:
        for cid in exterior_corner_ids:
            hit = _nearest(scaffold_nodes[int(cid)])
            if hit is not None:
                iron.add(hit)

    if CONFORM_DOF_FACE_CENTROIDS in rule.conform_dofs:
        for quad in boundary_quads:
            centroid = scaffold_nodes[quad].mean(axis=0)
            hit = _nearest(centroid)
            if hit is not None:
                iron.add(hit)

    return np.array(sorted(iron), dtype=np.int64)


def compute_node_depths_from_boundary(
    n_nodes: int,
    struts: np.ndarray,
    boundary_node_ids: np.ndarray,
) -> np.ndarray:
    """Graph BFS depth from iron/boundary nodes (depth 0)."""
    depths = np.full(n_nodes, -1, dtype=np.int32)
    if n_nodes == 0:
        return depths
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in np.asarray(struts, dtype=np.int64):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    q: deque[int] = deque()
    for bid in np.asarray(boundary_node_ids, dtype=np.int64).ravel():
        b = int(bid)
        if 0 <= b < n_nodes and depths[b] < 0:
            depths[b] = 0
            q.append(b)

    while q:
        cur = q.popleft()
        d = int(depths[cur])
        for nb in adj[cur]:
            if depths[nb] < 0:
                depths[nb] = d + 1
                q.append(nb)

    if np.any(depths < 0):
        max_d = int(depths[depths >= 0].max()) if np.any(depths >= 0) else 0
        depths[depths < 0] = max_d + 1
    return depths


def compute_hex_cell_face_centroid_depths(
    hex_elems: np.ndarray,
    lattice_nodes: np.ndarray,
    struts: np.ndarray,
    boundary_node_ids: np.ndarray,
    round_decimals: int = 6,
) -> np.ndarray:
    """Hex-adjacency depth for face-centroid lattices; else graph BFS."""
    n = len(lattice_nodes)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    node_index = build_lattice_node_index(lattice_nodes, round_decimals=round_decimals)
    face_to_cells: dict[frozenset, list[int]] = defaultdict(list)
    cell_face_node_ids: list[list[int]] = []

    for hi, coords in enumerate(hex_elems):
        face_ids: list[int] = []
        for fv in _HEX_FACES:
            centroid = coords[list(fv)].mean(axis=0)
            key = _round_key(centroid, round_decimals)
            idx = node_index.get(key)
            if idx is None:
                fine = _pt_key_fine(centroid)
                idx = next(
                    (i for i, pt in enumerate(lattice_nodes) if _pt_key_fine(pt) == fine),
                    None,
                )
            if idx is None:
                face_ids = []
                break
            face_ids.append(int(idx))
            fkey = frozenset(_round_key(coords[j], round_decimals) for j in fv)
            face_to_cells[fkey].append(hi)
        cell_face_node_ids.append(face_ids)

    if any(len(f) != 6 for f in cell_face_node_ids):
        return compute_node_depths_from_boundary(n, struts, boundary_node_ids)

    M = len(hex_elems)
    cell_depth = np.full(M, -1, dtype=np.int32)
    boundary_faces = {fk for fk, owners in face_to_cells.items() if len(owners) == 1}
    q: deque[int] = deque()
    for fk in boundary_faces:
        for hi in face_to_cells[fk]:
            if cell_depth[hi] < 0:
                cell_depth[hi] = 0
                q.append(hi)

    neighbors: dict[int, list[int]] = defaultdict(list)
    for owners in face_to_cells.values():
        if len(owners) == 2:
            a, b = owners
            neighbors[a].append(b)
            neighbors[b].append(a)

    while q:
        cur = q.popleft()
        d = int(cell_depth[cur])
        for nb in neighbors[cur]:
            if cell_depth[nb] < 0:
                cell_depth[nb] = d + 1
                q.append(nb)

    node_depths = np.full(n, 999999, dtype=np.int32)
    for hi, face_ids in enumerate(cell_face_node_ids):
        d = int(cell_depth[hi])
        if d < 0:
            continue
        for idx in face_ids:
            node_depths[idx] = min(node_depths[idx], d)

    if np.any(node_depths != 999999):
        fill = int(node_depths[node_depths != 999999].max()) + 1
    else:
        fill = 0
    node_depths[node_depths == 999999] = fill

    for bid in np.asarray(boundary_node_ids, dtype=np.int64).ravel():
        b = int(bid)
        if 0 <= b < n:
            node_depths[b] = 0
    return node_depths


def _kelvin_ref_face_uv_nodes(
    *,
    L: float = 1.0,
    plane_eps: float = 1e-6,
    uv_decimals: int = 6,
) -> list[list[tuple[tuple[float, float], np.ndarray]]]:
    """
    Per hex-face: Kelvin reference nodes on that unit-cube plane with 2D UV keys.

    UV is the two free parametric axes (u,v,w in [0,1]) in axis-alphabetical order
    excluding the fixed plane axis, so opposite faces share the same UV layout.
    """
    from graphite.explicit.kelvin_cell import generate_kelvin_cell

    ref, _ = generate_kelvin_cell(float(L))
    uvw = np.asarray(ref, dtype=np.float64) / float(L) + 0.5
    out: list[list[tuple[tuple[float, float], np.ndarray]]] = []
    for axis, value in _HEX_FACE_PLANE:
        ax_i = {"u": 0, "v": 1, "w": 2}[axis]
        free = [i for i in range(3) if i != ax_i]
        face_nodes: list[tuple[tuple[float, float], np.ndarray]] = []
        for pt, q in zip(ref, uvw):
            if abs(float(q[ax_i]) - float(value)) > plane_eps:
                continue
            uv = (
                round(float(q[free[0]]), uv_decimals),
                round(float(q[free[1]]), uv_decimals),
            )
            face_nodes.append((uv, np.asarray(pt, dtype=np.float64)))
        out.append(face_nodes)
    return out


def _append_kelvin_face_bridges(
    hex_elems: np.ndarray,
    get_or_add: Callable[[np.ndarray], int],
    strut_set: set[tuple[int, int]],
    *,
    round_decimals: int = 6,
) -> None:
    """
    Kelvin surface dual: bridge corresponding face nodes on shared *exterior*
    side walls of neighboring cells.

    Example (neighbors along +Z): top node on the +X face of the lower cell ↔
    bottom node on the +X face of the cell above; likewise left/right for ±Y.
    Interior shared-face nodes already weld under tiling and are not re-linked.
    """
    from graphite.explicit.kelvin_cell import _hex8_trilinear

    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.size == 0:
        return

    ref_faces = _kelvin_ref_face_uv_nodes(L=1.0)
    axis_index = {"u": 0, "v": 1, "w": 2}

    flat = elems.reshape(-1, 3)
    rounded = np.round(flat, max(round_decimals, 6))
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    corner_ids = inverse.reshape(-1, 8)

    face_owners: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for hi, corners in enumerate(corner_ids):
        for fi, face in enumerate(_HEX_FACES):
            key = tuple(sorted(int(corners[j]) for j in face))
            face_owners[key].append((hi, fi))

    exterior: set[tuple[int, int]] = {
        (hi, fi)
        for owners in face_owners.values()
        if len(owners) == 1
        for hi, fi in owners
    }

    def mapped_face_nodes(
        hi: int, fi: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        c = elems[hi]
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for _uv, pref in ref_faces[fi]:
            uvw = np.asarray(pref, dtype=np.float64) + 0.5
            world = _hex8_trilinear(c, float(uvw[0]), float(uvw[1]), float(uvw[2]))
            out.append((uvw, world))
        return out

    for owners in face_owners.values():
        if len(owners) != 2:
            continue
        (h0, f0), (h1, f1) = owners
        axis0, val0 = _HEX_FACE_PLANE[f0]
        axis1, val1 = _HEX_FACE_PLANE[f1]
        if axis0 != axis1:
            continue
        neighbor_axis = axis0
        # Hex owning the high-value interface face is on the -neighbor side.
        if val0 > val1:
            h_lo, f_shared_lo = h0, f0
            h_hi, f_shared_hi = h1, f1
        else:
            h_lo, f_shared_lo = h1, f1
            h_hi, f_shared_hi = h0, f0

        for side in range(6):
            if side in (f_shared_lo, f_shared_hi):
                continue
            if (h_lo, side) not in exterior or (h_hi, side) not in exterior:
                continue
            side_axis, _ = _HEX_FACE_PLANE[side]
            free = [a for a in ("u", "v", "w") if a != side_axis]
            if neighbor_axis not in free:
                continue
            match_axis = free[0] if free[1] == neighbor_axis else free[1]
            mi = axis_index[match_axis]
            ni = axis_index[neighbor_axis]

            def by_match(hi: int) -> dict[float, list[tuple[float, np.ndarray]]]:
                groups: dict[float, list[tuple[float, np.ndarray]]] = defaultdict(list)
                for uvw, world in mapped_face_nodes(hi, side):
                    groups[round(float(uvw[mi]), 6)].append((float(uvw[ni]), world))
                return groups

            g_lo = by_match(h_lo)
            g_hi = by_match(h_hi)
            for key in set(g_lo) & set(g_hi):
                # Only the near-interface pair: high-neighbor node on the lower
                # cell ↔ low-neighbor node on the upper cell (skip mid-edge singles).
                n_lo_t = max(g_lo[key], key=lambda t: t[0])
                n_hi_t = min(g_hi[key], key=lambda t: t[0])
                if n_lo_t[0] <= 0.5 + 1e-6 or n_hi_t[0] >= 0.5 - 1e-6:
                    continue
                a = get_or_add(n_lo_t[1])
                b = get_or_add(n_hi_t[1])
                if a != b:
                    strut_set.add((min(a, b), max(a, b)))


def merge_surface_skin(
    hex_elems: np.ndarray,
    lattice_nodes: np.ndarray,
    rule: HexTopologyRule,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Append any skin-only nodes and return (nodes, skin_struts).

    skin_mode:
      face_centroid_dual — hex_surface_dual cage
      corner_edge_cage — exterior hex edges (grid / tesseract)
      face_local_rule — face centroid + spokes to corners (star / octet)
      kelvin_face_bridge — corresponding Kelvin nodes on shared exterior side walls
      none — empty
    """
    mode = rule.skin_mode
    if mode == SKIN_MODE_NONE:
        return (
            np.asarray(lattice_nodes, dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
        )

    scaffold_nodes, _hex_ids, boundary_quads, _ = classify_boundary_from_hex_elems(
        hex_elems, round_decimals=max(round_decimals, 6)
    )
    node_list = [np.asarray(lattice_nodes[i], dtype=np.float64).copy() for i in range(len(lattice_nodes))]
    index = build_lattice_node_index(np.asarray(node_list), round_decimals=round_decimals)

    def get_or_add(pt: np.ndarray) -> int:
        key = _round_key(pt, round_decimals)
        if key in index:
            return index[key]
        fine = _pt_key_fine(pt)
        for i, existing in enumerate(node_list):
            if _pt_key_fine(existing) == fine:
                index[key] = i
                return i
        idx = len(node_list)
        node_list.append(np.asarray(pt, dtype=np.float64).copy())
        index[key] = idx
        return idx

    strut_set: set[tuple[int, int]] = set()

    if mode == SKIN_MODE_FACE_CENTROID_DUAL:
        if boundary_quads.size == 0:
            return np.asarray(node_list, dtype=np.float64), np.empty((0, 2), dtype=np.int64)
        cage_nodes, cage_struts = generate_hex_surface_dual_cage(
            scaffold_nodes,
            boundary_quads,
            coplanar_cos_threshold=0.0,
            include_isolated_closure=True,
            include_corner_closure=True,
        )
        cage_to_global = [get_or_add(cage_nodes[i]) for i in range(len(cage_nodes))]
        for a, b in cage_struts:
            ga, gb = cage_to_global[int(a)], cage_to_global[int(b)]
            if ga != gb:
                strut_set.add((min(ga, gb), max(ga, gb)))

    elif mode == SKIN_MODE_CORNER_EDGE_CAGE:
        for quad in boundary_quads:
            for i in range(4):
                a = get_or_add(scaffold_nodes[int(quad[i])])
                b = get_or_add(scaffold_nodes[int(quad[(i + 1) % 4])])
                if a != b:
                    strut_set.add((min(a, b), max(a, b)))

    elif mode == SKIN_MODE_FACE_LOCAL_RULE:
        # Corner lattices (star): spokes through face centroid, not opposite-corner diagonals.
        # Octet hybrid: same spokes + perimeter edges on the exterior face.
        both = (
            CONFORM_DOF_CORNERS in rule.conform_dofs
            and CONFORM_DOF_FACE_CENTROIDS in rule.conform_dofs
        )
        for quad in boundary_quads:
            corners = [get_or_add(scaffold_nodes[int(quad[i])]) for i in range(4)]
            fci = get_or_add(scaffold_nodes[quad].mean(axis=0))
            for c in corners:
                if c != fci:
                    strut_set.add((min(c, fci), max(c, fci)))
            if both:
                for i in range(4):
                    a, b = corners[i], corners[(i + 1) % 4]
                    if a != b:
                        strut_set.add((min(a, b), max(a, b)))

    elif mode == SKIN_MODE_KELVIN_FACE_BRIDGE:
        _append_kelvin_face_bridges(
            hex_elems, get_or_add, strut_set, round_decimals=round_decimals
        )

    else:
        raise ValueError(f"Unhandled skin_mode {mode!r}")

    out_nodes = np.asarray(node_list, dtype=np.float64)
    out_struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return out_nodes, out_struts


def morph_hex_scaffold(
    hex_elems: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    valency_cutoff: int = 4,
    relax_iterations: int = 15,
    relax_alpha: float = 0.5,
    round_decimals: int = 8,
    sphere_center: np.ndarray | tuple[float, float, float] | None = None,
    sphere_radius: float | None = None,
    relax_layers: int | None = None,
    relax_mode: str = "laplacian",
    cell_size: float | tuple[float, float, float] | np.ndarray | None = None,
    max_projection_factor: float | None = 1.0,
    collapse_warn_ratio: float = 0.10,
    projection_mode: str = "closest",
    snap_outside_nodes: bool = True,
    cull_collapsed_hexes: bool = False,
    stair_step_normal_gate: bool = False,
    stair_step_gate_factor: float = 1.0,
    surface_relax_iterations: int = 0,
    surface_relax_alpha: float = 0.5,
    surface_relax_max_travel_factor: float = 0.35,
    surface_feature_dihedral_deg: float = 60.0,
    surface_feature_proximity_factor: float = 0.15,
) -> tuple[np.ndarray, dict]:
    """
    Conform the SC hex *cage* to the CAD, then return morphed hex bricks.

    1. Unique hex-corner scaffold
    2. Identify exposed faces (owned by exactly one hex) → unique exposed nodes
       (face adjacency, **not** valency — captures stair-step interior corners)
    3. Optionally union in every scaffold node that still sits *outside* the CAD
       (shared edges between two partial cells never appear on an exposed face,
       so without this they stay stranded outside the part)
    4. Bidirectional project every iron node onto the surface:
       - algebraic sphere if ``sphere_center`` / ``sphere_radius`` are set
       - ``projection_mode="closest"`` (default): nearest CAD point
       - ``projection_mode="face_normal"``: raycast along owning exposed-face
         normals; multi-face nodes take the farthest face-candidate so a stair
         side wall beats the nearby floor
    5. Relax interior scaffold nodes:
       - ``relax_mode="laplacian"`` (default): Jacobi Laplacian with iron
         nodes fixed at depth 0. Pass ``relax_layers`` to freeze nodes deeper
         than that many layers.
       - ``relax_mode="radial_equalize"`` (experimental, sphere-only): equalize
         radial edge lengths with exposed nodes fixed at their projected pose.
    6. Optionally drop hexes whose post-morph volume collapses
       (``cull_collapsed_hexes``) instead of clamping projection travel mid-flight
    7. Write deformed corners back into ``(N, 8, 3)`` hex elements

    ``valency_cutoff`` is retained for API compatibility but is **not** used to
    filter which exposed nodes project (all exposed nodes snap).

    ``max_projection_factor`` bounds each projection to that multiple of the
    cell dimension on every axis. Prefer ``None`` together with
    ``cull_collapsed_hexes=True``: full surface snap, then discard crushed
    bricks. The clamp leaves outside corners stranded when they need more
    travel than one half-cell; with ``snap_outside_nodes`` those corners get a
    residual closest-point finish onto the CAD.
    ``cell_size`` may be a scalar or per-axis triple; per-axis is required for
    the bound to respect anisotropy.

    ``stair_step_normal_gate`` re-aims interior corners whose closest CAD point
    sits behind a face they do not expose — the stair-step notch corner that
    otherwise snaps down to the floor instead of out to the side wall (see
    ``redirect_interior_pull_to_exposed_normals``). Re-aimed corners get
    ``stair_step_gate_factor`` of travel budget instead of
    ``max_projection_factor``, since an axis-aligned normal ray cannot wander
    laterally onto a neighbouring surface patch.

    ``surface_relax_iterations`` > 0 runs a tangent-constrained Laplacian on
    iron nodes after projection (and before interior relax), with sharp CAD
    features pinned. See ``relax_surface_nodes_tangent``.
    """
    # Local import avoids circular import with conformal_generator at module load.
    from graphite.explicit.conformal_generator import (
        apply_depth_gated_relaxation,
        project_to_cad_surface,
    )

    mode = str(relax_mode).strip().lower()
    if mode not in ("laplacian", "radial_equalize"):
        raise ValueError(
            f"relax_mode must be 'laplacian' or 'radial_equalize'; got {relax_mode!r}"
        )
    proj_mode = str(projection_mode).strip().lower()
    if proj_mode not in ("closest", "face_normal"):
        raise ValueError(
            "projection_mode must be 'closest' or 'face_normal'; "
            f"got {projection_mode!r}"
        )

    elems = np.asarray(hex_elems, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elems must be (N, 8, 3); got {elems.shape}")
    if relax_layers is not None and int(relax_layers) < 0:
        raise ValueError("relax_layers must be >= 0 or None")

    scaffold_nodes, hex_ids = hex_node_ids_from_elements(elems, round_decimals=round_decimals)
    exposed_faces = find_exposed_faces(hex_ids)
    exposed_ids = exposed_node_ids_from_faces(exposed_faces)
    iron_ids = exposed_ids.copy()
    n_outside_extra = 0

    # Hex edge graph among scaffold corners
    edge_set: set[tuple[int, int]] = set()
    for elem in hex_ids:
        for a, b in _HEX_EDGES:
            ia, ib = int(elem[a]), int(elem[b])
            if ia != ib:
                edge_set.add((min(ia, ib), max(ia, ib)))
    scaffold_struts = (
        np.array(sorted(edge_set), dtype=np.int64)
        if edge_set
        else np.empty((0, 2), dtype=np.int64)
    )

    use_sphere = sphere_center is not None and sphere_radius is not None
    if mode == "radial_equalize" and not use_sphere:
        raise ValueError(
            "relax_mode='radial_equalize' requires sphere_center and sphere_radius"
        )

    # Estimate cell size from undeformed scaffold edges if not provided.
    if cell_size is None and scaffold_struts.size:
        edge_lens = np.linalg.norm(
            scaffold_nodes[scaffold_struts[:, 1]] - scaffold_nodes[scaffold_struts[:, 0]],
            axis=1,
        )
        cell_size = float(np.median(edge_lens)) if edge_lens.size else 1.0
    elif cell_size is None:
        cell_size = 1.0
    cell_dims = resolve_cell_dims(cell_size)
    cell_scalar = float(np.min(cell_dims))

    # Shared edges between two partial cells never appear on an exposed face,
    # yet can sit outside the CAD. Force those corners into the iron set.
    if snap_outside_nodes and not use_sphere and len(scaffold_nodes):
        # Use the same sign convention as VF cull: safe_signed_distance inverts
        # raw trimesh SDF so positive = outside. (Raw trimesh on many solids is
        # the opposite; using it here previously ironed interior midplane nodes.)
        sd = safe_signed_distance(cad_mesh, scaffold_nodes)
        outside_ids = np.flatnonzero(sd > 1e-3).astype(np.int32)
        if outside_ids.size:
            before_set = set(int(i) for i in iron_ids)
            extra = np.array(
                [i for i in outside_ids if int(i) not in before_set], dtype=np.int32
            )
            n_outside_extra = int(len(extra))
            if n_outside_extra:
                iron_ids = (
                    np.unique(np.concatenate([iron_ids, extra]))
                    if iron_ids.size
                    else extra
                )

    print(
        f"  Hex scaffold morph: {len(scaffold_nodes)} corners, "
        f"{len(iron_ids)} iron DOFs "
        f"({len(exposed_ids)} exposed"
        + (f" + {n_outside_extra} outside-only" if n_outside_extra else "")
        + f"), {len(exposed_faces)} exposed faces, {len(scaffold_struts)} edges"
        + (f", sphere R={float(sphere_radius):.4g}" if use_sphere else "")
        + f", relax_mode={mode}, projection_mode={proj_mode}"
    )

    nodes_ironed = scaffold_nodes.copy()
    max_disp = 0.0
    max_disp_raw = 0.0
    n_clamped = 0
    n_stair_gate = 0
    if iron_ids.size > 0:
        before = nodes_ironed[iron_ids]
        if use_sphere:
            after = project_points_onto_sphere(before, sphere_center, float(sphere_radius))
        elif proj_mode == "face_normal":
            exposed_faces, face_normals = exposed_faces_with_outward_normals(
                hex_ids, scaffold_nodes
            )
            after = project_points_face_normal_aware(
                before,
                iron_ids,
                cad_mesh,
                exposed_faces,
                face_normals,
                ray_length=float(np.max(cell_dims) * 4.0),
            )
        else:
            after, _inside = project_to_cad_surface(before, cad_mesh)

        gated_local = np.empty(0, dtype=np.int64)
        if stair_step_normal_gate and not use_sphere:
            gate_faces, gate_normals = exposed_faces_with_outward_normals(
                hex_ids, scaffold_nodes
            )
            after, gated_local = redirect_interior_pull_to_exposed_normals(
                before,
                after,
                iron_ids,
                cad_mesh,
                gate_faces,
                gate_normals,
                ray_length=float(np.max(cell_dims) * 4.0),
            )
            n_stair_gate = int(gated_local.size)
            if n_stair_gate:
                print(
                    f"  Stair-step gate: re-aimed {n_stair_gate} interior "
                    f"corner(s) from the nearest face onto an exposed-face normal"
                )

        max_disp_raw = float(np.linalg.norm(after - before, axis=1).max())
        if max_projection_factor is not None:
            unclamped = after
            after, scale = clamp_projection_travel(
                before, unclamped, cell_dims, float(max_projection_factor)
            )
            if gated_local.size:
                gate_factor = max(
                    float(max_projection_factor), float(stair_step_gate_factor)
                )
                after[gated_local], scale[gated_local] = clamp_projection_travel(
                    before[gated_local], unclamped[gated_local], cell_dims, gate_factor
                )
            n_clamped = int(np.count_nonzero(scale < 1.0 - 1e-9))
        nodes_ironed[iron_ids] = after
        max_disp = float(np.linalg.norm(after - before, axis=1).max())
        _ = valency_cutoff  # API compat; exposed-face set replaces valency gating

    if n_clamped:
        print(
            f"  Projection bound: clamped {n_clamped}/{len(iron_ids)} corners "
            f"to {max_projection_factor:g} x cell "
            f"(max travel {max_disp_raw:.4g} -> {max_disp:.4g} mm)"
        )

    # Clamp can leave ironed corners short of the CAD. Finish any still-outside
    # corners with closest-point before relax uses them as anchors.
    n_residual_snap = 0
    if snap_outside_nodes and not use_sphere and len(nodes_ironed):
        nodes_ironed, n_residual_snap = snap_outside_corners_to_surface(
            nodes_ironed, cad_mesh
        )
        if n_residual_snap:
            print(
                f"  Snap outside: closest-point finished {n_residual_snap} "
                f"corner(s) still outside after projection"
            )

    surface_report: dict = {
        "surface_relax_iterations": int(surface_relax_iterations),
        "n_surface_free": 0,
        "n_surface_pinned": 0,
        "n_surface_moved": 0,
        "max_surface_step": 0.0,
        "max_surface_travel": 0.0,
        "skin_edge_cv_before": 0.0,
        "skin_edge_cv_after": 0.0,
    }
    if (
        int(surface_relax_iterations) > 0
        and not use_sphere
        and iron_ids.size > 0
        and exposed_faces.size > 0
    ):
        stats_before = skin_edge_length_stats(nodes_ironed, exposed_faces)
        surface_report["skin_edge_cv_before"] = float(stats_before["skin_edge_cv"])
        max_travel = float(surface_relax_max_travel_factor) * float(np.min(cell_dims))
        feature_prox = float(surface_feature_proximity_factor) * float(np.min(cell_dims))
        nodes_ironed, srep = relax_surface_nodes_tangent(
            nodes_ironed,
            iron_ids,
            exposed_faces,
            cad_mesh,
            iterations=int(surface_relax_iterations),
            alpha=float(surface_relax_alpha),
            max_travel=max_travel,
            dihedral_deg=float(surface_feature_dihedral_deg),
            feature_proximity=feature_prox,
        )
        surface_report.update(srep)
        stats_after = skin_edge_length_stats(nodes_ironed, exposed_faces)
        surface_report["skin_edge_cv_after"] = float(stats_after["skin_edge_cv"])
        surface_report["skin_edge_median_after"] = float(stats_after["skin_edge_median"])
        surface_report["skin_edge_std_after"] = float(stats_after["skin_edge_std"])
        print(
            f"  Surface relax: {surface_report['n_surface_free']} free / "
            f"{surface_report['n_surface_pinned']} pinned, "
            f"max travel {surface_report['max_surface_travel']:.3f} mm, "
            f"skin CV {surface_report['skin_edge_cv_before']:.3f} -> "
            f"{surface_report['skin_edge_cv_after']:.3f}"
        )

    depths = compute_node_depths_from_boundary(
        len(scaffold_nodes), scaffold_struts, iron_ids
    )
    for bid in iron_ids:
        b = int(bid)
        if 0 <= b < len(depths):
            depths[b] = 0

    radial_report: dict = {}
    if mode == "radial_equalize":
        # For radial mode, relax_iterations defaults (15) are too few; callers
        # should pass ~300. relax_alpha is reused as the spring step size.
        nodes_relaxed, radial_report = apply_radial_equalization_relaxation(
            nodes_ironed,
            scaffold_struts,
            iron_ids,
            center=sphere_center,
            radius=float(sphere_radius),
            cell_size=cell_scalar,
            node_depths=depths,
            hex_ids=hex_ids,
            iterations=int(relax_iterations),
            step=float(relax_alpha),
        )
        max_depth = None
        n_relaxed = int(np.count_nonzero(depths > 0))
    else:
        max_depth = None if relax_layers is None else int(relax_layers)
        nodes_relaxed = apply_depth_gated_relaxation(
            nodes_ironed,
            scaffold_struts,
            depths,
            iterations=int(relax_iterations),
            alpha=float(relax_alpha),
            max_depth=max_depth,
        )
        if max_depth is None:
            n_relaxed = int(np.count_nonzero(depths > 0))
        else:
            n_relaxed = int(np.count_nonzero((depths > 0) & (depths <= max_depth)))

    n_post_relax_snap = 0
    if snap_outside_nodes and not use_sphere and len(nodes_relaxed):
        nodes_relaxed, n_post_relax_snap = snap_outside_corners_to_surface(
            nodes_relaxed, cad_mesh
        )
        if n_post_relax_snap:
            print(
                f"  Snap outside: closest-point finished {n_post_relax_snap} "
                f"corner(s) outside after relax"
            )

    deformed = nodes_relaxed[hex_ids]

    # A brick that survives the cull but loses nearly all of its volume in the
    # morph produces struts shorter than their own diameter, which fuse into a
    # blob and read as a hole in the lattice. Either warn, or drop them when
    # the caller opts into the quality gate (preferred over mid-flight clamp).
    vol_before = hex_volumes(scaffold_nodes[hex_ids])
    vol_after = hex_volumes(deformed)
    volume_ratio = np.abs(vol_after) / np.maximum(np.abs(vol_before), 1e-12)
    sign_ok = np.sign(vol_after) == np.sign(vol_before)
    collapsed_mask = volume_ratio < float(collapse_warn_ratio)
    inverted_mask = ~sign_ok
    n_collapsed = int(np.count_nonzero(collapsed_mask))
    n_inverted = int(np.count_nonzero(inverted_mask))
    n_culled = 0
    if cull_collapsed_hexes:
        keep = (~collapsed_mask) & sign_ok
        n_culled = int(np.count_nonzero(~keep))
        if n_culled:
            print(
                f"  Quality gate: dropped {n_culled} crushed/inverted hex(es) "
                f"(threshold {100.0 * collapse_warn_ratio:g}% volume; "
                f"min retained "
                f"{100.0 * float(volume_ratio[keep].min()) if np.any(keep) else 0.0:.1f}%)"
            )
            deformed = deformed[keep]
            hex_ids = hex_ids[keep]
            volume_ratio = volume_ratio[keep]
            if len(deformed) == 0:
                raise ValueError(
                    "Quality gate removed every hex; loosen collapse_warn_ratio "
                    "or check the CAD / cell size"
                )
        n_collapsed = 0
        n_inverted = 0
    elif n_collapsed or n_inverted:
        print(
            f"  WARNING: {n_collapsed} hex(es) below "
            f"{100.0 * collapse_warn_ratio:g}% of original volume "
            f"(min {100.0 * float(volume_ratio.min()):.1f}%), "
            f"{n_inverted} inverted"
        )

    report = {
        "n_scaffold_corners": int(len(scaffold_nodes)),
        "n_exterior_iron": int(len(iron_ids)),
        "n_exposed_iron": int(len(exposed_ids)),
        "n_outside_extra_iron": int(n_outside_extra),
        "n_residual_outside_snap": int(n_residual_snap),
        "n_post_relax_outside_snap": int(n_post_relax_snap),
        "n_exposed_faces": int(exposed_faces.shape[0]),
        "n_scaffold_edges": int(len(scaffold_struts)),
        "max_projection_distance": max_disp,
        "max_projection_distance_unclamped": max_disp_raw,
        "n_projection_clamped": n_clamped,
        "n_stair_step_gate_redirects": n_stair_gate,
        "stair_step_normal_gate": bool(stair_step_normal_gate),
        "max_projection_factor": max_projection_factor,
        "cell_dims": cell_dims,
        "min_hex_volume_ratio": float(volume_ratio.min()) if len(volume_ratio) else 1.0,
        "n_collapsed_hexes": n_collapsed,
        "n_inverted_hexes": n_inverted,
        "n_culled_hexes": n_culled,
        "cull_collapsed_hexes": bool(cull_collapsed_hexes),
        "snap_outside_nodes": bool(snap_outside_nodes),
        "relax_layers": -1 if max_depth is None else max_depth,
        "n_relaxed_layer_nodes": n_relaxed,
        "scaffold_nodes_relaxed": nodes_relaxed,
        "hex_corner_ids": hex_ids,
        "exterior_corner_ids": iron_ids,
        "exposed_faces": exposed_faces,
        "sphere_projection": bool(use_sphere),
        "relax_mode": mode,
        "projection_mode": proj_mode,
    }
    report.update(radial_report)
    report.update(surface_report)
    return deformed, report


def generate_sc_volume_topology(
    hex_elems: np.ndarray,
    rule_name: str,
    round_decimals: int = 6,
    cell_tags: list | np.ndarray | None = None,
    classification=None,
) -> tuple[np.ndarray, np.ndarray, HexTopologyRule]:
    rule = get_hex_topology_rule(rule_name)
    nodes, struts = generate_hex_topology(
        hex_elems,
        rule_name=rule.name,
        round_decimals=round_decimals,
        cell_tags=cell_tags,
        classification=classification,
        full_rule_name=rule.name,
    )
    return nodes, struts, rule
