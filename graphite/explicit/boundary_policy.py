"""
Graphite Explicit Engine - Boundary Conformation and Smoothing

This module contains algorithms for snapping hexahedral scaffold nodes to a 
target boundary mesh (STL) and performing Laplacian smoothing on internal nodes
while preventing hexahedral element inversion.

It implements a Jacobian-gated line search and tiered boundary snapping 
strategies (pull in, stretch out) to ensure the resulting scaffold remains 
a valid conformal mapping of the original volume.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from graphite.geometry.masking import voxelize_mesh_and_edt

_HEX_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)

_CORNER_NEIGHBORS = {
    0: (1, 3, 4),
    1: (0, 2, 5),
    2: (1, 3, 6),
    3: (0, 2, 7),
    4: (0, 5, 7),
    5: (1, 4, 6),
    6: (2, 5, 7),
    7: (3, 4, 6),
}

_LINE_SEARCH_ALPHAS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)


def _corner_jacobian_proxy(hex_coords: np.ndarray, corner: int) -> float:
    n1, n2, n3 = _CORNER_NEIGHBORS[int(corner)]
    v0 = hex_coords[int(corner)]
    e1 = hex_coords[n1] - v0
    e2 = hex_coords[n2] - v0
    e3 = hex_coords[n3] - v0
    return float(np.linalg.det(np.column_stack((e1, e2, e3))))


def _hex_min_corner_jacobian(hex_coords: np.ndarray) -> float:
    return float(min(_corner_jacobian_proxy(hex_coords, c) for c in range(8)))


def _validate_hex_inversion_local(
    original_hex_coords: np.ndarray,
    proposed_hex_coords: np.ndarray,
    local_corner_idx: int,
    min_det_ratio: float = 0.2,
    min_abs_det: float = 1e-10,
) -> bool:
    base_det = _corner_jacobian_proxy(original_hex_coords, local_corner_idx)
    prop_det = _corner_jacobian_proxy(proposed_hex_coords, local_corner_idx)
    if abs(base_det) < min_abs_det:
        return False
    if abs(prop_det) < max(min_abs_det, min_det_ratio * abs(base_det)):
        return False
    if np.sign(prop_det) != np.sign(base_det):
        return False
    return True


def _hex_corners_valid_after_move(
    reference_hex_coords: np.ndarray,
    proposed_hex_coords: np.ndarray,
    *,
    min_jacobian_proxy: float,
) -> bool:
    """All eight corners must keep orientation and |det| >= threshold."""
    eps = float(min_jacobian_proxy)
    for c in range(8):
        base_det = _corner_jacobian_proxy(reference_hex_coords, c)
        prop_det = _corner_jacobian_proxy(proposed_hex_coords, c)
        if abs(base_det) < 1e-12:
            return False
        if np.sign(prop_det) != np.sign(base_det):
            return False
        if abs(prop_det) < eps:
            return False
    return True


def _build_hex_incident_index(
    hex_node_ids: np.ndarray,
    keep_hex_mask: np.ndarray | None,
) -> dict[int, list[int]]:
    incident: dict[int, list[int]] = {}
    for hi, elem in enumerate(hex_node_ids):
        if keep_hex_mask is not None and not bool(keep_hex_mask[hi]):
            continue
        for gid in elem:
            incident.setdefault(int(gid), []).append(int(hi))
    return incident


def _line_search_node_move(
    nodes: np.ndarray,
    gid: int,
    target: np.ndarray,
    hex_node_ids: np.ndarray,
    reference_hex_coords: np.ndarray,
    incident_hexes: list[int],
    *,
    min_jacobian_proxy: float,
    line_search_alphas: tuple[float, ...] = _LINE_SEARCH_ALPHAS,
) -> tuple[float, bool]:
    """
    Move node toward ``target`` along v = target - current with Jacobian-gated line search.

  Returns:
      (alpha_applied, moved): largest alpha in ``line_search_alphas`` that keeps all
      incident hex corners valid, or (0, False) if no step is safe.
    """
    current = nodes[int(gid)]
    displacement = np.asarray(target, dtype=np.float64) - current
    if np.linalg.norm(displacement) < 1e-14:
        return 1.0, False

    for alpha in line_search_alphas:
        trial_nodes = nodes.copy()
        trial_nodes[int(gid)] = current + float(alpha) * displacement
        ok = True
        for hi in incident_hexes:
            elem = hex_node_ids[hi]
            ref_hex = reference_hex_coords[int(hi)]
            prop_hex = trial_nodes[elem]
            if not _hex_corners_valid_after_move(
                ref_hex,
                prop_hex,
                min_jacobian_proxy=min_jacobian_proxy,
            ):
                ok = False
                break
        if ok:
            nodes[int(gid)] = trial_nodes[int(gid)]
            return float(alpha), True
    return 0.0, False


def closest_points_with_fallback(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    try:
        closest, distances, _ = trimesh.proximity.closest_point(mesh, pts)
        return np.asarray(closest, dtype=np.float64), np.asarray(distances, dtype=np.float64)
    except Exception:
        tree = cKDTree(mesh.vertices)
        distances, idx = tree.query(pts)
        return np.asarray(mesh.vertices[idx], dtype=np.float64), np.asarray(distances, dtype=np.float64)


def closest_points_with_normals(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest surface point and outward face normal at each query."""
    pts = np.asarray(points, dtype=np.float64)
    try:
        closest, _, tri_id = trimesh.proximity.closest_point(mesh, pts)
        closest = np.asarray(closest, dtype=np.float64)
        tri_id = np.asarray(tri_id, dtype=np.int64)
        normals = np.asarray(mesh.face_normals[tri_id], dtype=np.float64)
    except Exception:
        closest, _ = closest_points_with_fallback(mesh, pts)
        tree = cKDTree(mesh.vertices)
        _, idx = tree.query(pts)
        normals = np.asarray(mesh.vertex_normals[idx], dtype=np.float64)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    return closest, normals / lengths


def build_edt_sdf_sampler(mesh: trimesh.Trimesh, resolution: float):
    _, _, _, cad_sdf, padded_min_bound, _, nx, ny, nz = voxelize_mesh_and_edt(mesh, resolution)

    def sample(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        ind = np.round((pts - padded_min_bound) / resolution).astype(int)
        ind[:, 0] = np.clip(ind[:, 0], 0, nx - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, ny - 1)
        ind[:, 2] = np.clip(ind[:, 2], 0, nz - 1)
        return cad_sdf[ind[:, 0], ind[:, 1], ind[:, 2]]

    return sample


def compress_graph_to_kept_struts(
    nodes: np.ndarray, struts: np.ndarray, keep_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    kept = np.asarray(struts, dtype=np.int32)[np.asarray(keep_mask, dtype=bool)]
    if len(kept) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    used = np.unique(kept.ravel())
    mapping = np.full(len(nodes), -1, dtype=np.int32)
    mapping[used] = np.arange(len(used))
    return np.asarray(nodes, dtype=np.float64)[used], mapping[kept]


def apply_tiered_boundary_policy(
    nodes: np.ndarray,
    closest_points: np.ndarray,
    node_sdf: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    eps_pull: float,
    band_inner: float,
    beta_soft: float = 0.45,
    alpha_stretch: float = 0.28,
    hex_node_ids: np.ndarray | None = None,
    keep_hex_mask: np.ndarray | None = None,
    reference_hex_coords: np.ndarray | None = None,
    min_jacobian_proxy: float = 0.05,
    line_search_alphas: tuple[float, ...] = _LINE_SEARCH_ALPHAS,
) -> tuple[np.ndarray, dict[str, int]]:
    n = np.asarray(nodes, dtype=np.float64).copy()
    sdf = np.asarray(node_sdf, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool)
    cpts = np.asarray(closest_points, dtype=np.float64)

    outside_hard = sdf > float(eps_pull)
    outside_soft = (sdf > 0.0) & (~outside_hard)
    inside_near = (sdf <= 0.0) & (sdf >= -float(band_inner))

    pull_in = eligible & outside_hard
    pull_soft = eligible & outside_soft
    stretch_out = eligible & inside_near

    use_jacobian_gate = (
        hex_node_ids is not None
        and reference_hex_coords is not None
        and len(hex_node_ids) > 0
    )

    counts = {
        "conform_pull_in_full": 0,
        "conform_pull_in_soft": 0,
        "conform_stretch_out": 0,
        "conform_line_search_partial": 0,
        "conform_line_search_rejected": 0,
    }

    if not use_jacobian_gate:
        if np.any(pull_in):
            n[pull_in] = cpts[pull_in]
        if np.any(pull_soft):
            n[pull_soft] = n[pull_soft] * (1.0 - beta_soft) + cpts[pull_soft] * beta_soft
        if np.any(stretch_out):
            n[stretch_out] = n[stretch_out] * (1.0 - alpha_stretch) + cpts[stretch_out] * alpha_stretch
        counts["conform_pull_in_full"] = int(np.sum(pull_in))
        counts["conform_pull_in_soft"] = int(np.sum(pull_soft))
        counts["conform_stretch_out"] = int(np.sum(stretch_out))
        return n, counts

    hex_ids = np.asarray(hex_node_ids, dtype=np.int32)
    ref_hex = np.asarray(reference_hex_coords, dtype=np.float64)
    incident = _build_hex_incident_index(hex_ids, keep_hex_mask)

    def _targets_for_mask(mask: np.ndarray, blend: float) -> None:
        gids = np.where(mask)[0]
        for gid in gids:
            gid = int(gid)
            hi_list = incident.get(gid, [])
            if not hi_list:
                continue
            target = (
                cpts[gid]
                if blend >= 1.0 - 1e-12
                else n[gid] * (1.0 - blend) + cpts[gid] * blend
            )
            alpha, moved = _line_search_node_move(
                n,
                gid,
                target,
                hex_ids,
                ref_hex,
                hi_list,
                min_jacobian_proxy=min_jacobian_proxy,
                line_search_alphas=line_search_alphas,
            )
            if not moved:
                counts["conform_line_search_rejected"] += 1
            elif alpha < 1.0 - 1e-12:
                counts["conform_line_search_partial"] += 1

    if np.any(pull_in):
        _targets_for_mask(pull_in, 1.0)
        counts["conform_pull_in_full"] = int(np.sum(pull_in))
    if np.any(pull_soft):
        _targets_for_mask(pull_soft, beta_soft)
        counts["conform_pull_in_soft"] = int(np.sum(pull_soft))
    if np.any(stretch_out):
        _targets_for_mask(stretch_out, alpha_stretch)
        counts["conform_stretch_out"] = int(np.sum(stretch_out))

    return n, counts


def apply_gated_node_targets(
    nodes: np.ndarray,
    node_targets: np.ndarray,
    node_mask: np.ndarray,
    hex_node_ids: np.ndarray,
    reference_hex_coords: np.ndarray,
    keep_hex_mask: np.ndarray | None = None,
    *,
    min_jacobian_proxy: float = 0.05,
    line_search_alphas: tuple[float, ...] = _LINE_SEARCH_ALPHAS,
) -> dict[str, int]:
    """Jacobian-gated line search toward per-node targets (e.g. after neighbor stretch)."""
    n = np.asarray(nodes, dtype=np.float64)
    targets = np.asarray(node_targets, dtype=np.float64)
    mask = np.asarray(node_mask, dtype=bool)
    hex_ids = np.asarray(hex_node_ids, dtype=np.int32)
    ref_hex = np.asarray(reference_hex_coords, dtype=np.float64)
    incident = _build_hex_incident_index(hex_ids, keep_hex_mask)
    partial = 0
    rejected = 0
    for gid in np.where(mask)[0]:
        gid = int(gid)
        hi_list = incident.get(gid, [])
        if not hi_list:
            continue
        alpha, moved = _line_search_node_move(
            n,
            gid,
            targets[gid],
            hex_ids,
            ref_hex,
            hi_list,
            min_jacobian_proxy=min_jacobian_proxy,
            line_search_alphas=line_search_alphas,
        )
        if not moved:
            rejected += 1
        elif alpha < 1.0 - 1e-12:
            partial += 1
    return {
        "conform_line_search_partial": int(partial),
        "conform_line_search_rejected": int(rejected),
    }


def calculate_hex_volume_fractions(
    hex_nodes: np.ndarray,
    sdf_sampler,
) -> np.ndarray:
    """
    Estimate per-hex inside volume fraction using a 3x3x3 sub-grid sampler.

    Args:
        hex_nodes: (N, 8, 3) coordinates for each hex's corners.
        sdf_sampler: callable(points[N,3]) -> sdf values (<=0 is inside).

    Returns:
        vf: (N,) fraction in [0,1], computed as inside_count / 27.
    """
    hn = np.asarray(hex_nodes, dtype=np.float64)
    if hn.ndim != 3 or hn.shape[1:] != (8, 3):
        raise ValueError(f"hex_nodes must have shape (N, 8, 3); got {hn.shape}.")
    n_hex = hn.shape[0]
    if n_hex == 0:
        return np.empty((0,), dtype=np.float64)

    mins = np.min(hn, axis=1)  # (N,3)
    maxs = np.max(hn, axis=1)  # (N,3)
    t = np.linspace(0.0, 1.0, 3, dtype=np.float64)
    gx, gy, gz = np.meshgrid(t, t, t, indexing="ij")
    coeff = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))  # (27,3)

    span = (maxs - mins)[:, None, :]  # (N,1,3)
    base = mins[:, None, :]  # (N,1,3)
    points = base + coeff[None, :, :] * span  # (N,27,3)
    sdf = np.asarray(sdf_sampler(points.reshape(-1, 3)), dtype=np.float64).reshape(n_hex, 27)
    inside = sdf <= 0.0
    return np.mean(inside, axis=1)


def apply_laplacian_smoothing(
    nodes: np.ndarray,
    hexes: np.ndarray,
    frozen_mask: np.ndarray,
    iterations: int = 5,
    alpha: float = 0.5,
    *,
    boundary_mask: np.ndarray | None = None,
    mesh: trimesh.Trimesh | None = None,
    surface_normals: np.ndarray | None = None,
    tangential_boundary: bool = False,
    min_det_ratio: float = 0.2,
) -> np.ndarray:
    """
    Apply Laplacian smoothing to unfrozen nodes while preventing hex inversion.

    Frozen nodes never move. Each proposed move is accepted only if every 
    incident hex corner passes a local Jacobian determinant check 
    (`_validate_hex_inversion_local`).

    Parameters
    ----------
    nodes : ndarray
        (N, 3) array of node coordinates.
    hexes : ndarray
        (M, 8) array of hexahedral element connectivities.
    frozen_mask : ndarray
        (N,) boolean array; True for nodes that should not be moved.
    iterations : int, optional
        Number of smoothing passes, by default 5.
    alpha : float, optional
        Relaxation factor (0.0 to 1.0) for the smoothing update, by default 0.5.
    boundary_mask : ndarray, optional
        (N,) boolean array marking boundary nodes.
    mesh : trimesh.Trimesh, optional
        Target surface mesh, used if tangential sliding is enabled.
    surface_normals : ndarray, optional
        (N, 3) pre-computed normals for boundary nodes.
    tangential_boundary : bool, optional
        If True, unfrozen boundary nodes slide tangentially along the mesh surface.
    min_det_ratio : float, optional
        Minimum allowed ratio of the proposed Jacobian determinant to the 
        reference determinant (to prevent severe distortion), by default 0.2.

    Returns
    -------
    ndarray
        (N, 3) array of smoothed node coordinates.
    """
    pts = np.asarray(nodes, dtype=np.float64).copy()
    elems = np.asarray(hexes, dtype=np.int32)
    frozen = np.asarray(frozen_mask, dtype=bool)
    if elems.ndim != 2 or elems.shape[1] != 8:
        raise ValueError(f"hexes must have shape (N, 8); got {elems.shape}.")
    if len(pts) != len(frozen):
        raise ValueError("nodes and frozen_mask length mismatch.")
    if iterations <= 0 or alpha <= 0.0:
        return pts
    alpha = float(min(1.0, max(0.0, alpha)))

    boundary = (
        np.asarray(boundary_mask, dtype=bool)
        if boundary_mask is not None
        else np.zeros(len(pts), dtype=bool)
    )
    use_tangential = bool(
        tangential_boundary
        and np.any(boundary)
        and (mesh is not None or surface_normals is not None)
    )
    if use_tangential and surface_normals is None and mesh is not None:
        _, surface_normals = closest_points_with_normals(mesh, pts)
    if surface_normals is not None:
        surface_normals = np.asarray(surface_normals, dtype=np.float64)
        lengths = np.linalg.norm(surface_normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        surface_normals = surface_normals / lengths

    reference_hex_coords = pts[elems].copy()

    neighbors: dict[int, set[int]] = {i: set() for i in range(len(pts))}
    incident: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(pts))}
    for hi, elem in enumerate(elems):
        for a, b in _HEX_EDGES:
            ga = int(elem[a])
            gb = int(elem[b])
            neighbors[ga].add(gb)
            neighbors[gb].add(ga)
        for li, gid in enumerate(elem):
            incident[int(gid)].append((int(hi), int(li)))

    for _ in range(int(iterations)):
        base_pts = pts.copy()
        for gid in range(len(pts)):
            is_boundary = bool(boundary[gid])
            if frozen[gid] and not (use_tangential and is_boundary):
                continue
            neigh = sorted(neighbors.get(gid, set()))
            if not neigh:
                continue
            nbr_mean = np.mean(base_pts[np.asarray(neigh, dtype=np.int32)], axis=0)
            delta = nbr_mean - base_pts[gid]
            if use_tangential and is_boundary and surface_normals is not None:
                nrm = surface_normals[gid]
                delta = delta - float(np.dot(delta, nrm)) * nrm
            if np.linalg.norm(delta) < 1e-14:
                continue
            proposal = base_pts[gid] + alpha * delta

            trial_pts = pts.copy()
            trial_pts[gid] = proposal
            valid = True
            for hi, local_corner in incident.get(gid, []):
                elem = elems[hi]
                ref_hex = reference_hex_coords[hi]
                prop_hex = trial_pts[elem]
                if not _validate_hex_inversion_local(
                    ref_hex,
                    prop_hex,
                    int(local_corner),
                    min_det_ratio=float(min_det_ratio),
                ):
                    valid = False
                    break
            if valid:
                pts[gid] = proposal
    return pts
