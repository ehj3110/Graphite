"""Axis-raycast Boolean crop and perimeter skin engine for Grid lattices (Carbon-style).

Crops axis-aligned SC grid lines to exact CAD boundary pierces, retains Cartesian
interior nodes, and connects boundary pierces along cell face perimeters in
cyclical order, eliminating crossing 'X' diagonals.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import trimesh

from graphite.explicit.conformal_core import safe_signed_distance
from graphite.explicit.geometry_module import union_lattice_with_spherical_joints
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf


def _connect_face_pierces(face_entries: list[tuple[float, int]]) -> list[tuple[int, int]]:
    """Connect pierce nodes cyclically around a 2D quad boundary without diagonals.

    face_entries: list of (perimeter_parameter_s, gid) where s in [0, 4)
    represents position along the 4 edges of the quad in counter-clockwise order:
      - [0, 1): Bottom edge
      - [1, 2): Right edge
      - [2, 3): Top edge (reversed)
      - [3, 4): Left edge (reversed)
    """
    if len(face_entries) < 2:
        return []
    # Deduplicate by gid, taking minimum s if duplicate
    by_gid: dict[int, float] = {}
    for s, gid in face_entries:
        if gid not in by_gid or s < by_gid[gid]:
            by_gid[gid] = s
    sorted_pts = sorted(by_gid.items(), key=lambda item: item[1])
    n = len(sorted_pts)
    if n == 2:
        ga, gb = sorted_pts[0][0], sorted_pts[1][0]
        return [(min(ga, gb), max(ga, gb))] if ga != gb else []
    edges: list[tuple[int, int]] = []
    for k in range(n):
        ga = sorted_pts[k][0]
        gb = sorted_pts[(k + 1) % n][0]
        if ga != gb:
            edges.append((min(ga, gb), max(ga, gb)))
    return edges


def _is_inside_or_on_surface(
    cad: trimesh.Trimesh,
    pt: np.ndarray,
    tol: float = 1e-4,
) -> bool:
    """True if point is strictly inside or lies on the CAD surface boundary."""
    if bool(cad.contains([pt])[0]):
        return True
    return bool(safe_signed_distance(cad, np.asarray([pt]))[0] <= tol)


def generate_grid_boolean_lattice(
    cad_mesh: trimesh.Trimesh,
    cell_size: tuple[float, float, float] | np.ndarray = (25.4, 25.4, 25.4),
    origin_offset: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    *,
    prune_floor_shortcuts: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate boundary-cropped Cartesian Grid lattice with cyclical perimeter surface dual.

    Returns:
        nodes: (V, 3) float64 - all nodes (Cartesian interior + exact surface pierces)
        vol_struts: (E_vol, 2) int64 - Cartesian interior struts terminating at pierces
        surf_struts: (E_surf, 2) int64 - boundary skin struts connecting pierces
        surf_nodes: (V_surf,) int64 - indices of surface pierce nodes
    """
    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    if isinstance(cad, trimesh.Scene):
        cad = trimesh.util.concatenate(tuple(cad.geometry.values()))

    bounds = np.asarray(cad.bounds, dtype=np.float64)
    c_sz = np.asarray(cell_size, dtype=np.float64)
    off = np.asarray(origin_offset, dtype=np.float64)
    dx, dy, dz = c_sz[0], c_sz[1], c_sz[2]

    # Background grid coordinate arrays
    x_min = np.floor((bounds[0, 0] - off[0]) / c_sz[0]) * c_sz[0] + off[0] - c_sz[0]
    x_max = np.ceil((bounds[1, 0] - off[0]) / c_sz[0]) * c_sz[0] + off[0] + c_sz[0]
    xs = np.arange(x_min, x_max + c_sz[0] * 0.5, c_sz[0])

    y_min = np.floor((bounds[0, 1] - off[1]) / c_sz[1]) * c_sz[1] + off[1] - c_sz[1]
    y_max = np.ceil((bounds[1, 1] - off[1]) / c_sz[1]) * c_sz[1] + off[1] + c_sz[1]
    ys = np.arange(y_min, y_max + c_sz[1] * 0.5, c_sz[1])

    z_min = np.floor((bounds[0, 2] - off[2]) / c_sz[2]) * c_sz[2] + off[2] - c_sz[2]
    z_max = np.ceil((bounds[1, 2] - off[2]) / c_sz[2]) * c_sz[2] + off[2] + c_sz[2]
    zs = np.arange(z_min, z_max + c_sz[2] * 0.5, c_sz[2])

    ray_query = cad.ray

    nodes_list: list[np.ndarray] = []
    node_to_id: dict[tuple[float, float, float], int] = {}

    def get_node_id(pt) -> int:
        k = tuple(np.round(np.asarray(pt, dtype=np.float64), 4).tolist())
        if k not in node_to_id:
            node_to_id[k] = len(nodes_list)
            nodes_list.append(np.asarray(pt, dtype=np.float64))
        return node_to_id[k]

    vol_struts: set[tuple[int, int]] = set()

    # Store pierce points by edge key:
    # ('X', j, k, i) -> list of (x_coord, gid) for edge between xs[i] and xs[i+1] at (y[j], z[k])
    # ('Y', i, k, j) -> list of (y_coord, gid) for edge between ys[j] and ys[j+1] at (x[i], z[k])
    # ('Z', i, j, k) -> list of (z_coord, gid) for edge between zs[k] and zs[k+1] at (x[i], y[j])
    edge_pierces: dict[tuple, list[tuple[float, int]]] = {}
    surface_gids: set[int] = set()

    # 1. Clip along X-lines
    p_start_x = xs[0] - 10.0
    for j, y in enumerate(ys):
        for k, z in enumerate(zs):
            ray_org = np.array([p_start_x, y, z])
            ray_dir = np.array([1.0, 0.0, 0.0])
            locs, _, _ = ray_query.intersects_location(
                ray_origins=[ray_org],
                ray_directions=[ray_dir],
            )
            if len(locs) == 0:
                continue
            hit_xs = sorted(loc[0] for loc in locs)

            intervals: list[tuple[float, float]] = []
            for h_idx in range(len(hit_xs) - 1):
                p_mid = np.array([0.5 * (hit_xs[h_idx] + hit_xs[h_idx + 1]), y, z])
                if _is_inside_or_on_surface(cad, p_mid):
                    intervals.append((hit_xs[h_idx], hit_xs[h_idx + 1]))

            for x_in, x_out in intervals:
                if x_out - x_in < 1e-4:
                    continue
                gid_in = get_node_id([x_in, y, z])
                gid_out = get_node_id([x_out, y, z])
                surface_gids.add(gid_in)
                surface_gids.add(gid_out)

                i_in = int(np.floor((x_in - xs[0]) / c_sz[0]))
                i_out = int(np.floor((x_out - xs[0]) / c_sz[0]))
                if 0 <= i_in < len(xs) - 1:
                    edge_pierces.setdefault(("X", j, k, i_in), []).append((x_in, gid_in))
                if 0 <= i_out < len(xs) - 1:
                    edge_pierces.setdefault(("X", j, k, i_out), []).append((x_out, gid_out))

                int_xs = [x for x in xs if x_in + 1e-4 < x < x_out - 1e-4]
                chain = [gid_in] + [get_node_id([x, y, z]) for x in int_xs] + [gid_out]
                for c_idx in range(len(chain) - 1):
                    ga, gb = chain[c_idx], chain[c_idx + 1]
                    if ga != gb:
                        vol_struts.add((min(ga, gb), max(ga, gb)))

    # 2. Clip along Y-lines
    p_start_y = ys[0] - 10.0
    for i, x in enumerate(xs):
        for k, z in enumerate(zs):
            ray_org = np.array([x, p_start_y, z])
            ray_dir = np.array([0.0, 1.0, 0.0])
            locs, _, _ = ray_query.intersects_location(
                ray_origins=[ray_org],
                ray_directions=[ray_dir],
            )
            if len(locs) == 0:
                continue
            hit_ys = sorted(loc[1] for loc in locs)

            intervals = []
            for h_idx in range(len(hit_ys) - 1):
                p_mid = np.array([x, 0.5 * (hit_ys[h_idx] + hit_ys[h_idx + 1]), z])
                if _is_inside_or_on_surface(cad, p_mid):
                    intervals.append((hit_ys[h_idx], hit_ys[h_idx + 1]))

            for y_in, y_out in intervals:
                if y_out - y_in < 1e-4:
                    continue
                gid_in = get_node_id([x, y_in, z])
                gid_out = get_node_id([x, y_out, z])
                surface_gids.add(gid_in)
                surface_gids.add(gid_out)

                j_in = int(np.floor((y_in - ys[0]) / c_sz[1]))
                j_out = int(np.floor((y_out - ys[0]) / c_sz[1]))
                if 0 <= j_in < len(ys) - 1:
                    edge_pierces.setdefault(("Y", i, k, j_in), []).append((y_in, gid_in))
                if 0 <= j_out < len(ys) - 1:
                    edge_pierces.setdefault(("Y", i, k, j_out), []).append((y_out, gid_out))

                int_ys = [y for y in ys if y_in + 1e-4 < y < y_out - 1e-4]
                chain = [gid_in] + [get_node_id([x, y, z]) for y in int_ys] + [gid_out]
                for c_idx in range(len(chain) - 1):
                    ga, gb = chain[c_idx], chain[c_idx + 1]
                    if ga != gb:
                        vol_struts.add((min(ga, gb), max(ga, gb)))

    # 3. Clip along Z-lines
    p_start_z = zs[0] - 10.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            ray_org = np.array([x, y, p_start_z])
            ray_dir = np.array([0.0, 0.0, 1.0])
            locs, _, _ = ray_query.intersects_location(
                ray_origins=[ray_org],
                ray_directions=[ray_dir],
            )
            if len(locs) == 0:
                continue
            hit_zs = sorted(loc[2] for loc in locs)

            intervals = []
            for h_idx in range(len(hit_zs) - 1):
                p_mid = np.array([x, y, 0.5 * (hit_zs[h_idx] + hit_zs[h_idx + 1])])
                if _is_inside_or_on_surface(cad, p_mid):
                    intervals.append((hit_zs[h_idx], hit_zs[h_idx + 1]))

            for z_in, z_out in intervals:
                if z_out - z_in < 1e-4:
                    continue
                gid_in = get_node_id([x, y, z_in])
                gid_out = get_node_id([x, y, z_out])
                surface_gids.add(gid_in)
                surface_gids.add(gid_out)

                k_in = int(np.floor((z_in - zs[0]) / c_sz[2]))
                k_out = int(np.floor((z_out - zs[0]) / c_sz[2]))
                if 0 <= k_in < len(zs) - 1:
                    edge_pierces.setdefault(("Z", i, j, k_in), []).append((z_in, gid_in))
                if 0 <= k_out < len(zs) - 1:
                    edge_pierces.setdefault(("Z", i, j, k_out), []).append((z_out, gid_out))

                int_zs = [z for z in zs if z_in + 1e-4 < z < z_out - 1e-4]
                chain = [gid_in] + [get_node_id([x, y, z]) for z in int_zs] + [gid_out]
                for c_idx in range(len(chain) - 1):
                    ga, gb = chain[c_idx], chain[c_idx + 1]
                    if ga != gb:
                        vol_struts.add((min(ga, gb), max(ga, gb)))

    nodes = np.asarray(nodes_list, dtype=np.float64)
    vol_struts_arr = np.asarray(sorted(vol_struts), dtype=np.int64)

    # 4. Surface Dual Construction (cyclical perimeter walk per boundary face, no crossing diagonals)
    surf_struts: set[tuple[int, int]] = set()

    # Face type A: XY-faces at constant Z = zs[k]
    for k in range(len(zs)):
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                face_entries: list[tuple[float, int]] = []
                # Bottom X-edge at y_j: s in [0, 1)
                for x, gid in edge_pierces.get(("X", j, k, i), []):
                    face_entries.append((0.0 + (x - xs[i]) / dx, gid))
                # Right Y-edge at x_{i+1}: s in [1, 2)
                for y, gid in edge_pierces.get(("Y", i + 1, k, j), []):
                    face_entries.append((1.0 + (y - ys[j]) / dy, gid))
                # Top X-edge at y_{j+1}: s in [2, 3) (reversed)
                for x, gid in edge_pierces.get(("X", j + 1, k, i), []):
                    face_entries.append((2.0 + (xs[i + 1] - x) / dx, gid))
                # Left Y-edge at x_i: s in [3, 4) (reversed)
                for y, gid in edge_pierces.get(("Y", i, k, j), []):
                    face_entries.append((3.0 + (ys[j + 1] - y) / dy, gid))

                for ga, gb in _connect_face_pierces(face_entries):
                    surf_struts.add((ga, gb))

    # Face type B: XZ-faces at constant Y = ys[j]
    for j in range(len(ys)):
        for i in range(len(xs) - 1):
            for k in range(len(zs) - 1):
                face_entries = []
                # Bottom X-edge at z_k: s in [0, 1)
                for x, gid in edge_pierces.get(("X", j, k, i), []):
                    face_entries.append((0.0 + (x - xs[i]) / dx, gid))
                # Right Z-edge at x_{i+1}: s in [1, 2)
                for z, gid in edge_pierces.get(("Z", i + 1, j, k), []):
                    face_entries.append((1.0 + (z - zs[k]) / dz, gid))
                # Top X-edge at z_{k+1}: s in [2, 3) (reversed)
                for x, gid in edge_pierces.get(("X", j, k + 1, i), []):
                    face_entries.append((2.0 + (xs[i + 1] - x) / dx, gid))
                # Left Z-edge at x_i: s in [3, 4) (reversed)
                for z, gid in edge_pierces.get(("Z", i, j, k), []):
                    face_entries.append((3.0 + (zs[k + 1] - z) / dz, gid))

                for ga, gb in _connect_face_pierces(face_entries):
                    surf_struts.add((ga, gb))

    # Face type C: YZ-faces at constant X = xs[i]
    for i in range(len(xs)):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                face_entries = []
                # Bottom Y-edge at z_k: s in [0, 1)
                for y, gid in edge_pierces.get(("Y", i, k, j), []):
                    face_entries.append((0.0 + (y - ys[j]) / dy, gid))
                # Right Z-edge at y_{j+1}: s in [1, 2)
                for z, gid in edge_pierces.get(("Z", i, j + 1, k), []):
                    face_entries.append((1.0 + (z - zs[k]) / dz, gid))
                # Top Y-edge at z_{k+1}: s in [2, 3) (reversed)
                for y, gid in edge_pierces.get(("Y", i, k + 1, j), []):
                    face_entries.append((2.0 + (ys[j + 1] - y) / dy, gid))
                # Left Z-edge at y_j: s in [3, 4) (reversed)
                for z, gid in edge_pierces.get(("Z", i, j, k), []):
                    face_entries.append((3.0 + (zs[k + 1] - z) / dz, gid))

                for ga, gb in _connect_face_pierces(face_entries):
                    surf_struts.add((ga, gb))

    # Eliminate any surface struts that duplicate volume struts
    vol_set = {(min(a, b), max(a, b)) for a, b in vol_struts_arr}
    filtered_surf = {(a, b) for a, b in surf_struts if (a, b) not in vol_set}

    # Prune rogue floor-to-wall diagonal shortcuts across interior bays
    if prune_floor_shortcuts and filtered_surf:
        z_floor = float(np.min(bounds[0, 2]))
        dz_thresh = 0.5 * dz
        dxy_thresh = 0.25 * min(dx, dy)
        valid_surf = set()
        for ga, gb in filtered_surf:
            pa, pb = nodes[ga], nodes[gb]
            z_min = min(pa[2], pb[2])
            z_max = max(pa[2], pb[2])
            dxy = float(np.linalg.norm(pa[:2] - pb[:2]))
            if z_min <= z_floor + 0.2 and (z_max - z_min) > dz_thresh and dxy > dxy_thresh:
                continue
            valid_surf.add((ga, gb))
        filtered_surf = valid_surf

    surf_struts_arr = (
        np.asarray(sorted(filtered_surf), dtype=np.int64)
        if filtered_surf
        else np.empty((0, 2), dtype=np.int64)
    )
    surf_nodes_arr = np.asarray(sorted(surface_gids), dtype=np.int64)

    return nodes, vol_struts_arr, surf_struts_arr, surf_nodes_arr


# Backwards compatibility alias
boolean_grid_and_surface_dual = generate_grid_boolean_lattice


if __name__ == "__main__":
    cad_path = Path("test_parts") / "BookendLatticeSection_Sloped.STL"
    cad = trimesh.load(cad_path, force="mesh")
    nodes, vol_s, surf_s, surf_n = generate_grid_boolean_lattice(cad, cell_size=(25.4, 25.4, 25.4))
    print(
        f"Refined Grid Boolean: {len(nodes)} total nodes, {len(vol_s)} volume struts, "
        f"{len(surf_s)} surface dual struts, {len(surf_n)} surface nodes"
    )
