"""Independent surface dual placed surface-natively (V2).

Dual nodes = exposed faces of the VF-culled SC hex shell, projected onto CAD.
Dual edges follow boundary quad adjacency (or exposed-corner skeleton).

Projection modes
----------------
- ``exposed_normal`` (legacy): raycast along the hex face outward normal.
- ``target_normal`` (Blender-style Target Normal Project): snap to the nearest
  surface point whose (face) normal aims at / away from the source point —
  i.e. solve ``V = P + t * n(P)`` on candidate triangles. Avoids the crease
  jumping of Euclidean closest-point.
- ``closest``: trimesh ``on_surface`` nearest point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.conformal_core import (
    classify_boundary_from_hex_elems,
)
from graphite.explicit.hex_surface_dual import (
    get_boundary_quad_adjacency,
    boundary_quad_centroids,
)


@dataclass
class DualReport:
    n_exposed_faces: int = 0
    n_dual_nodes: int = 0
    n_dual_struts: int = 0
    n_projected: int = 0
    n_projection_fallback_closest: int = 0
    placement: str = "face_centroid_along_exposed_normal"
    projection_mode: str = "exposed_normal"
    skin_snap_mode: str = "all"
    n_skin_simple: int = 0
    n_skin_stair: int = 0
    n_skin_corner: int = 0
    n_snapped: int = 0
    n_snap_skipped_stair: int = 0
    # Pre-projection dual origins (same length/order as dual_nodes), for identity stitch.
    cartesian_origins: np.ndarray | None = None
    # Per dual node: "simple" | "stair" | "corner" (exposed_corners mode).
    skin_classes: list[str] = field(default_factory=list)


@dataclass
class SkinNodeClassification:
    """Node-level skin labels from unique exposed-face normals."""

    scaffold_ids: np.ndarray  # (S,) int — scaffold corner ids that lie on skin
    classes: np.ndarray  # (S,) object/str — simple|stair|corner
    n_unique_normals: np.ndarray  # (S,) int
    n_incident_faces: np.ndarray  # (S,) int
    normal_sums: np.ndarray  # (S, 3) — sum of incident face normals (for travel dir)
    # Per skin node: (K_i, 3) unit world-axis directions from quantized exposed normals.
    allowed_axes: list[np.ndarray] = field(default_factory=list)

    @property
    def n_simple(self) -> int:
        return int(np.count_nonzero(self.classes == "simple"))

    @property
    def n_stair(self) -> int:
        return int(np.count_nonzero(self.classes == "stair"))

    @property
    def n_corner(self) -> int:
        return int(np.count_nonzero(self.classes == "corner"))


def _normal_to_axis(n: np.ndarray) -> np.ndarray:
    """Snap a face normal to the nearest signed world axis (±X/±Y/±Z)."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    i = int(np.argmax(np.abs(n)))
    axis = np.zeros(3, dtype=np.float64)
    axis[i] = 1.0 if n[i] >= 0.0 else -1.0
    return axis


def classify_skin_nodes(
    boundary_quads: np.ndarray,
    face_normals: np.ndarray,
    *,
    quantize: int = 6,
) -> SkinNodeClassification:
    """
    Classify every node on the exposed-face complex by unique face normals.

    - ``simple``: 1 unique outward normal (plateau / single wall), including
      2×2 patch centers that only see +Z (or one axis).
    - ``stair``: 2 unique normals (edge between faces).
    - ``corner``: 3+ unique normals.

    Also records ``allowed_axes``: unique world-axis directions the node may
    travel along when projecting (from those exposed normals).
    """
    quads = np.asarray(boundary_quads, dtype=np.int64)
    normals = np.asarray(face_normals, dtype=np.float64)
    if len(quads) == 0:
        return SkinNodeClassification(
            scaffold_ids=np.empty(0, dtype=np.int64),
            classes=np.empty(0, dtype=object),
            n_unique_normals=np.empty(0, dtype=np.int64),
            n_incident_faces=np.empty(0, dtype=np.int64),
            normal_sums=np.empty((0, 3), dtype=np.float64),
            allowed_axes=[],
        )

    nkey = np.round(normals, int(quantize))
    per_node_keys: dict[int, set[tuple[float, float, float]]] = {}
    per_node_count: dict[int, int] = {}
    per_node_sum: dict[int, np.ndarray] = {}
    for face, key, n in zip(quads, nkey, normals):
        kt = (float(key[0]), float(key[1]), float(key[2]))
        for cid in face:
            cid = int(cid)
            if cid not in per_node_keys:
                per_node_keys[cid] = set()
                per_node_count[cid] = 0
                per_node_sum[cid] = np.zeros(3, dtype=np.float64)
            per_node_keys[cid].add(kt)
            per_node_count[cid] += 1
            per_node_sum[cid] += n

    ids = np.asarray(sorted(per_node_keys.keys()), dtype=np.int64)
    classes = np.empty(len(ids), dtype=object)
    n_unique = np.zeros(len(ids), dtype=np.int64)
    n_faces = np.zeros(len(ids), dtype=np.int64)
    sums = np.zeros((len(ids), 3), dtype=np.float64)
    allowed_axes: list[np.ndarray] = []
    for i, cid in enumerate(ids):
        keys = per_node_keys[int(cid)]
        nu = len(keys)
        n_unique[i] = nu
        n_faces[i] = per_node_count[int(cid)]
        sums[i] = per_node_sum[int(cid)]
        if nu <= 1:
            classes[i] = "simple"
        elif nu == 2:
            classes[i] = "stair"
        else:
            classes[i] = "corner"
        axes = []
        seen: set[tuple[float, float, float]] = set()
        for kt in keys:
            ax = _normal_to_axis(np.asarray(kt, dtype=np.float64))
            ak = (float(ax[0]), float(ax[1]), float(ax[2]))
            if ak not in seen:
                seen.add(ak)
                axes.append(ax)
        allowed_axes.append(
            np.asarray(axes, dtype=np.float64)
            if axes
            else np.zeros((0, 3), dtype=np.float64)
        )
    return SkinNodeClassification(
        scaffold_ids=ids,
        classes=classes,
        n_unique_normals=n_unique,
        n_incident_faces=n_faces,
        normal_sums=sums,
        allowed_axes=allowed_axes,
    )


def _quad_outward_normals(nodes: np.ndarray, quads: np.ndarray, hex_centroid: np.ndarray) -> np.ndarray:
    """Unit normals for quads, flipped so they point away from a reference interior point."""
    corners = nodes[quads]
    p0, p1, p2, p3 = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    n = np.cross(p1 - p0, p3 - p0) + np.cross(p3 - p0, p2 - p1)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norms, 1e-12)
    cents = corners.mean(axis=1)
    # Flip if pointing toward hex interior reference.
    to_out = cents - hex_centroid.reshape(1, 3)
    flip = np.sum(n * to_out, axis=1) < 0.0
    n[flip] *= -1.0
    return n


def _point_in_triangle(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    eps: float = 1e-8,
) -> bool:
    """Barycentric inside test (inclusive) on the triangle plane."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-18:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def _tnp_on_triangle(
    v: np.ndarray,
    tri: np.ndarray,
    n: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """
    Flat-face Target Normal Project on one triangle.

    Solve ``V = P + t * n`` with ``P`` on the triangle plane and inside the
    triangle. ``n`` is the triangle normal (either orientation is fine).
    Returns ``(|t|, P)`` or None if the foot falls outside the triangle.
    """
    n = np.asarray(n, dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn < 1e-12:
        return None
    n = n / nn
    a, b, c = tri[0], tri[1], tri[2]
    # Plane through a with normal n: P = V - t n, (P - a)·n = 0.
    t = float(np.dot(v - a, n))
    p = v - t * n
    if not _point_in_triangle(p, a, b, c):
        return None
    return abs(t), p


def _vertex_face_adjacency(mesh: trimesh.Trimesh) -> list[list[int]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n_v = int(len(mesh.vertices))
    adj: list[list[int]] = [[] for _ in range(n_v)]
    for fi, face in enumerate(faces):
        for vid in face:
            adj[int(vid)].append(int(fi))
    return adj


def target_normal_project(
    points: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    ring_hops: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Blender-style Target Normal Project onto ``cad_mesh``.

    For each source point V, search nearby triangles (starting from the
    Euclidean nearest face, expanding by ``ring_hops`` of vertex–face
    adjacency) for a point P on a triangle such that the triangle normal
    aims at V: ``V = P + t * n``. Prefer the solution with smallest ``|t|``.
    Fallback: closest surface point.

    Returns
    -------
    projected : (N, 3)
    hit_mask : (N,) bool — True if a TNP triangle solution was found
    """
    pts = np.asarray(points, dtype=np.float64)
    out = pts.copy()
    hit = np.zeros(len(pts), dtype=bool)
    if len(pts) == 0:
        return out, hit

    mesh = cad_mesh
    if not hasattr(mesh, "triangles") or mesh.triangles is None:
        mesh = mesh.copy()
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    v_adj = _vertex_face_adjacency(mesh)

    query = trimesh.proximity.ProximityQuery(mesh)
    _closest, _dist, tri_ids = query.on_surface(pts)
    tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)

    for i, v in enumerate(pts):
        seed = int(tri_ids[i])
        if seed < 0 or seed >= len(triangles):
            out[i] = _closest[i]
            continue

        # Expand face neighborhood around the nearest triangle.
        face_set: set[int] = {seed}
        frontier = {seed}
        for _ in range(max(0, int(ring_hops))):
            nxt: set[int] = set()
            for fi in frontier:
                for vid in faces[fi]:
                    for fj in v_adj[int(vid)]:
                        if fj not in face_set:
                            face_set.add(fj)
                            nxt.add(fj)
            frontier = nxt
            if not frontier:
                break

        best: tuple[float, np.ndarray] | None = None
        for fi in face_set:
            sol = _tnp_on_triangle(v, triangles[fi], normals[fi])
            if sol is None:
                continue
            if best is None or sol[0] < best[0]:
                best = sol

        if best is not None:
            out[i] = best[1]
            hit[i] = True
        else:
            out[i] = _closest[i]
            hit[i] = False

    return out, hit


def project_axis_constrained(
    origins: np.ndarray,
    allowed_axes: list[np.ndarray],
    cad_mesh: trimesh.Trimesh,
    *,
    cell_size: float | tuple[float, float, float] | np.ndarray = 1.0,
    ray_length: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project each origin onto CAD in the subspace of its exposed-face axes.

    ``allowed_axes[i]`` is ``(K_i, 3)`` of outward unit ±X/±Y/±Z directions.

    - **One horizontal axis** (side wall): raycast along ± that axis.
    - **Roof/floor involved** (±Z among exposed axes): raycast along ±Z only.
      Mixing Z with a vertical-riser normal builds a diagonal into the stair
      notch and pulls top nodes backwards; Z-only avoids that.
    - **Two horizontal axes** (vertical corner): cell-size-weighted XY diagonal.

    No unconstrained closest-point fallback.
    """
    pts = np.asarray(origins, dtype=np.float64)
    out = pts.copy()
    hit = np.zeros(len(pts), dtype=bool)
    if len(pts) == 0:
        return out, hit
    if len(allowed_axes) != len(pts):
        raise ValueError("allowed_axes must align with origins")

    cs = np.asarray(cell_size, dtype=np.float64).ravel()
    if cs.size == 1:
        cell_xyz = np.full(3, float(cs[0]), dtype=np.float64)
    elif cs.size == 3:
        cell_xyz = cs.astype(np.float64, copy=False)
    else:
        raise ValueError(f"cell_size must be scalar or length-3; got shape {cs.shape}")

    origins_list: list[np.ndarray] = []
    dirs_list: list[np.ndarray] = []
    meta: list[int] = []  # point index

    for i, axes in enumerate(allowed_axes):
        axes_arr = np.asarray(axes, dtype=np.float64).reshape(-1, 3)
        if len(axes_arr) == 0:
            continue

        units: list[np.ndarray] = []
        for ax in axes_arr:
            nrm = float(np.linalg.norm(ax))
            if nrm < 1e-12:
                continue
            units.append(ax / nrm)

        if not units:
            continue

        # If a roof/floor axis (±Z) is present, do NOT mix in vertical-riser
        # horizontals — that diagonal points into the stair notch and pulls
        # top nodes "backwards". Top/bottom: travel in Z only.
        # Pure sidewalls (XY only): single axis or cell-weighted XY diagonal.
        z_units = [u for u in units if abs(float(u[2])) > 0.5]
        h_units = [u for u in units if abs(float(u[2])) <= 0.5]

        if z_units:
            # Prefer a single Z sense if both somehow appear; use the first.
            travel_dirs = [z_units[0]]
        elif len(h_units) == 1:
            travel_dirs = [h_units[0]]
        elif len(h_units) >= 2:
            diag = np.zeros(3, dtype=np.float64)
            for u in h_units:
                ax_i = int(np.argmax(np.abs(u)))
                diag += u * float(cell_xyz[ax_i])
            dn = float(np.linalg.norm(diag))
            travel_dirs = [diag / dn] if dn > 1e-12 else h_units
        else:
            travel_dirs = units

        for unit in travel_dirs:
            for sign in (1.0, -1.0):
                origins_list.append(pts[i] - 1e-3 * sign * unit)
                dirs_list.append(sign * unit)
                meta.append(i)

    best: dict[int, tuple[float, np.ndarray]] = {}
    if origins_list:
        try:
            locations, index_ray, _ = cad_mesh.ray.intersects_location(
                ray_origins=np.asarray(origins_list, dtype=np.float64),
                ray_directions=np.asarray(dirs_list, dtype=np.float64),
                multiple_hits=True,
            )
        except Exception:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.int64)

        for loc, ray_i in zip(locations, index_ray):
            pi = meta[int(ray_i)]
            travel = float(np.linalg.norm(loc - pts[pi]))
            if travel < 1e-9 or travel > float(ray_length):
                continue
            prev = best.get(pi)
            if prev is None or travel < prev[0]:
                best[pi] = (travel, np.asarray(loc, dtype=np.float64))

    for i in range(len(pts)):
        if i in best:
            out[i] = best[i][1]
            hit[i] = True
    return out, hit


def _project_along_rays(
    origins: np.ndarray,
    directions: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    ray_length: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Raycast each origin along ±direction; prefer the hit that lands on/near
    the surface in the outward direction. Fallback: closest point.
    """
    pts = np.asarray(origins, dtype=np.float64).copy()
    dirs = np.asarray(directions, dtype=np.float64)
    hit_mask = np.zeros(len(pts), dtype=bool)
    if len(pts) == 0:
        return pts, hit_mask

    # Cast both ways from a slight inset along the normal.
    origins_list = []
    dirs_list = []
    meta = []  # (point_index, sign)
    for i in range(pts.shape[0]):
        d = dirs[i]
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            continue
        unit = d / nrm
        for sign in (1.0, -1.0):
            origins_list.append(pts[i] - 1e-3 * sign * unit)
            dirs_list.append(sign * unit)
            meta.append((i, sign))

    best: dict[int, tuple[float, np.ndarray]] = {}
    if origins_list:
        try:
            locations, index_ray, _ = cad_mesh.ray.intersects_location(
                ray_origins=np.asarray(origins_list, dtype=np.float64),
                ray_directions=np.asarray(dirs_list, dtype=np.float64),
                multiple_hits=True,
            )
        except Exception:
            locations = np.empty((0, 3), dtype=np.float64)
            index_ray = np.empty(0, dtype=np.int64)

        for loc, ray_i in zip(locations, index_ray):
            pi, sign = meta[int(ray_i)]
            travel = float(np.linalg.norm(loc - pts[pi]))
            if travel < 1e-9 or travel > float(ray_length):
                continue
            # Prefer outward (sign=+1) hits; among ties, nearer.
            rank = (0 if sign > 0 else 1, travel)
            prev = best.get(pi)
            if prev is None or rank < prev[0]:
                best[pi] = (rank, np.asarray(loc, dtype=np.float64))

    query = trimesh.proximity.ProximityQuery(cad_mesh)
    n_fallback = 0
    for i in range(len(pts)):
        if i in best:
            pts[i] = best[i][1]
            hit_mask[i] = True
        else:
            closest, _, _ = query.on_surface(pts[i].reshape(1, 3))
            pts[i] = closest[0]
            hit_mask[i] = False
            n_fallback += 1
    return pts, hit_mask


def build_surface_native_dual(
    hex_elems: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    dual_mode: str = "face_centroid",
    projection_mode: str = "exposed_normal",
    skin_snap_mode: str = "all",
    cell_size: float | tuple[float, float, float] | np.ndarray | None = None,
    round_decimals: int = 6,
    ray_length: float = 80.0,
    tnp_ring_hops: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DualReport]:
    """
    Build independent dual nodes/struts from VF-exposed faces, placed on CAD.

    ``dual_mode``:
      - ``"face_centroid"``: one dual node per exposed face at the face centroid.
        Dual edges follow face–face adjacency.
      - ``"exposed_corners"``: unique corners of exposed faces; dual edges are
        the exposed quad edges.

    ``projection_mode``:
      - ``"exposed_normal"``: raycast along hex exposed-face normals (legacy).
      - ``"target_normal"``: Blender Target Normal Project onto CAD.
      - ``"closest"``: Euclidean closest surface point.
      - ``"axis_constrained"``: single-axis nodes ray along that axis; multi-axis
        (stair) nodes ray along the cell-size-weighted diagonal of their
        exposed axes (node-tangent projection).

    ``cell_size``:
      Used by ``axis_constrained`` multi-axis diagonals (default 1,1,1 → 45°
      for cubic). Prefer the same (sx, sy, sz) used to cull the hex grid.

    ``skin_snap_mode`` (node policy; strongest with ``exposed_corners``):
      - ``"all"``: project every dual origin (legacy).
      - ``"simple_only"``: project only ``skin_simple`` nodes.
    """
    mode = str(dual_mode).strip().lower()
    if mode not in ("face_centroid", "exposed_corners"):
        raise ValueError(
            f"dual_mode must be 'face_centroid' or 'exposed_corners'; got {dual_mode!r}"
        )
    proj = str(projection_mode).strip().lower()
    if proj not in ("exposed_normal", "target_normal", "closest", "axis_constrained"):
        raise ValueError(
            f"projection_mode must be 'exposed_normal', 'target_normal', "
            f"'closest', or 'axis_constrained'; got {projection_mode!r}"
        )
    snap_mode = str(skin_snap_mode).strip().lower()
    if snap_mode not in ("all", "simple_only"):
        raise ValueError(
            f"skin_snap_mode must be 'all' or 'simple_only'; got {skin_snap_mode!r}"
        )

    report = DualReport(projection_mode=proj, skin_snap_mode=snap_mode)
    scaffold, hex_ids, boundary_quads, _ = classify_boundary_from_hex_elems(
        hex_elems, round_decimals=max(round_decimals, 6)
    )
    report.n_exposed_faces = int(len(boundary_quads))
    if boundary_quads.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            report,
        )

    hex_cents = np.asarray(hex_elems, dtype=np.float64).mean(axis=1)
    interior_ref = hex_cents.mean(axis=0)
    face_normals = _quad_outward_normals(scaffold, boundary_quads, interior_ref)
    skin = classify_skin_nodes(boundary_quads, face_normals)
    report.n_skin_simple = skin.n_simple
    report.n_skin_stair = skin.n_stair
    report.n_skin_corner = skin.n_corner
    skin_by_scaffold = {
        int(sid): str(cls) for sid, cls in zip(skin.scaffold_ids, skin.classes)
    }
    axes_by_scaffold = {
        int(sid): axes
        for sid, axes in zip(skin.scaffold_ids, skin.allowed_axes)
    }

    if mode == "face_centroid":
        origins = boundary_quad_centroids(scaffold, boundary_quads)
        normals = face_normals
        pairs, _ = get_boundary_quad_adjacency(boundary_quads)
        strut_set: set[tuple[int, int]] = {
            (int(min(i, j)), int(max(i, j))) for i, j in pairs if int(i) != int(j)
        }
        skin_classes: list[str] = []
        node_axes: list[np.ndarray] = []
        for face, fn in zip(boundary_quads, face_normals):
            counts = {"simple": 0, "stair": 0, "corner": 0}
            for cid in face:
                counts[skin_by_scaffold[int(cid)]] += 1
            skin_classes.append(max(counts, key=counts.get))  # type: ignore[arg-type]
            node_axes.append(_normal_to_axis(fn).reshape(1, 3))
        report.placement = f"face_centroid_{proj}_{snap_mode}"
    else:
        corner_normal_sum: dict[int, np.ndarray] = {}
        for face, n in zip(boundary_quads, face_normals):
            for cid in face:
                cid = int(cid)
                if cid not in corner_normal_sum:
                    corner_normal_sum[cid] = np.zeros(3, dtype=np.float64)
                corner_normal_sum[cid] += n
        corner_ids = sorted(corner_normal_sum.keys())
        id_to_local = {cid: i for i, cid in enumerate(corner_ids)}
        origins = scaffold[np.asarray(corner_ids, dtype=np.int64)].copy()
        normals = np.zeros((len(corner_ids), 3), dtype=np.float64)
        skin_classes = []
        node_axes = []
        for cid in corner_ids:
            s = corner_normal_sum[cid]
            nn = float(np.linalg.norm(s))
            normals[id_to_local[cid]] = (
                s / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
            )
            skin_classes.append(skin_by_scaffold[int(cid)])
            node_axes.append(axes_by_scaffold[int(cid)])
        strut_set = set()
        for face in boundary_quads:
            corners = [int(c) for c in face]
            for a, b in zip(corners, corners[1:] + corners[:1]):
                if a == b:
                    continue
                la, lb = id_to_local[a], id_to_local[b]
                strut_set.add((min(la, lb), max(la, lb)))
        report.placement = f"exposed_corners_{proj}_{snap_mode}"

    report.skin_classes = list(skin_classes)
    report.cartesian_origins = np.asarray(origins, dtype=np.float64).copy()

    snap_mask = np.ones(len(origins), dtype=bool)
    if snap_mode == "simple_only":
        snap_mask = np.asarray([c == "simple" for c in skin_classes], dtype=bool)
    report.n_snapped = int(np.count_nonzero(snap_mask))
    report.n_snap_skipped_stair = int(np.count_nonzero(~snap_mask))

    placed = np.asarray(origins, dtype=np.float64).copy()
    hit_mask = np.zeros(len(origins), dtype=bool)
    if np.any(snap_mask):
        if proj == "target_normal":
            snapped, hits = target_normal_project(
                origins[snap_mask], cad_mesh, ring_hops=int(tnp_ring_hops)
            )
        elif proj == "closest":
            closest, _, _ = trimesh.proximity.ProximityQuery(cad_mesh).on_surface(
                origins[snap_mask]
            )
            snapped = np.asarray(closest, dtype=np.float64)
            hits = np.ones(len(snapped), dtype=bool)
        elif proj == "axis_constrained":
            axes_sub = [node_axes[i] for i in np.flatnonzero(snap_mask)]
            cs = (1.0, 1.0, 1.0) if cell_size is None else cell_size
            snapped, hits = project_axis_constrained(
                origins[snap_mask],
                axes_sub,
                cad_mesh,
                cell_size=cs,
                ray_length=ray_length,
            )
        else:
            snapped, hits = _project_along_rays(
                origins[snap_mask],
                normals[snap_mask],
                cad_mesh,
                ray_length=ray_length,
            )
        placed[snap_mask] = snapped
        hit_mask[snap_mask] = hits

    report.n_projected = int(np.count_nonzero(hit_mask))
    report.n_projection_fallback_closest = int(np.count_nonzero(snap_mask & ~hit_mask))

    dual_nodes = placed
    dual_struts = (
        np.asarray(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    report.n_dual_nodes = int(len(dual_nodes))
    report.n_dual_struts = int(len(dual_struts))
    _ = hex_ids
    return dual_nodes, dual_struts, normals.astype(np.float64, copy=False), report
