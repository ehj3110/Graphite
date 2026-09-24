"""
Graphite Explicit Engine - Lofted Hexahedral Scaffold Generation

This module provides GMSH-free explicit hexahedral scaffolding along a designated
spine axis (X, Y, or Z). It samples CAD cross-sections along the spine to create
a boundary-conforming structured hex grid, supports 1D equal-phase height grading,
and synthesizes single-rule or multi-lattice topologies with interface transition pyramids.
"""

from __future__ import annotations

from typing import Any, Callable, Literal
import numpy as np
import trimesh

from graphite.explicit.hex_rules import _HEX_FACES
from graphite.explicit.hex_topology_module import (
    _builder_kwargs_for,
    generate_hex_topology,
    get_hex_topology_rule,
)


def _mesh_section_extents_on_plane(
    mesh: trimesh.Trimesh,
    axis: int,
    value: float,
    out_axes: tuple[int, int],
    *,
    fallback_lo: np.ndarray,
    fallback_hi: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Intersect mesh triangles with an axis-aligned plane and compute bounding extents.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Watertight or closed boundary mesh.
    axis : int
        Spine axis index (0=X, 1=Y, 2=Z) where plane is fixed at `value`.
    value : float
        Plane coordinate along `axis`.
    out_axes : tuple[int, int]
        Indices of transverse axes to measure extents on.
    fallback_lo, fallback_hi : np.ndarray
        Bounding extents fallback if plane does not intersect triangles.

    Returns
    -------
    tuple[float, float, float, float]
        (u_min, u_max, v_min, v_max) bounds in the transverse plane.
    """
    ax = int(axis)
    oa, ob = int(out_axes[0]), int(out_axes[1])
    coord = float(value)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)

    # Filter triangles whose spine coordinate range encompasses coord
    v_spine = triangles[:, :, ax]
    v_min = v_spine.min(axis=1)
    v_max = v_spine.max(axis=1)
    mask = (v_min <= coord) & (coord <= v_max)
    candidates = triangles[mask]

    if len(candidates) == 0:
        return (
            float(fallback_lo[oa]),
            float(fallback_hi[oa]),
            float(fallback_lo[ob]),
            float(fallback_hi[ob]),
        )

    # Intersect candidate triangle edges with plane
    pts: list[np.ndarray] = []
    edges = [(0, 1), (1, 2), (2, 0)]
    for tri in candidates:
        for e0, e1 in edges:
            a, b = tri[e0], tri[e1]
            da = float(a[ax] - coord)
            db = float(b[ax] - coord)
            if da * db <= 0.0 and abs(float(a[ax] - b[ax])) > 1e-12:
                t = (coord - float(a[ax])) / (float(b[ax]) - float(a[ax]))
                t = np.clip(t, 0.0, 1.0)
                pts.append(a + t * (b - a))

    if len(pts) < 2:
        return (
            float(fallback_lo[oa]),
            float(fallback_hi[oa]),
            float(fallback_lo[ob]),
            float(fallback_hi[ob]),
        )

    p = np.vstack(pts)
    u_min, u_max = float(p[:, oa].min()), float(p[:, oa].max())
    v_min_val, v_max_val = float(p[:, ob].min()), float(p[:, ob].max())

    # Avoid zero-thickness degenerate slices
    if abs(u_max - u_min) < 1e-9:
        u_min = float(fallback_lo[oa])
        u_max = float(fallback_hi[oa])
    if abs(v_max_val - v_min_val) < 1e-9:
        v_min_val = float(fallback_lo[ob])
        v_max_val = float(fallback_hi[ob])

    return u_min, u_max, v_min_val, v_max_val


def _structured_hex_elements_from_node_grid(
    nodes: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    """
    Assemble (nx*ny*nz, 8, 3) hex elements from structured (nx+1, ny+1, nz+1, 3) nodes.
    Maintains positive right-handed Jacobian element orientation.
    """
    n_hex = int(nx * ny * nz)
    hexes = np.zeros((n_hex, 8, 3), dtype=np.float64)
    h = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                hexes[h, 0] = nodes[i, j, k]
                hexes[h, 1] = nodes[i + 1, j, k]
                hexes[h, 2] = nodes[i + 1, j + 1, k]
                hexes[h, 3] = nodes[i, j + 1, k]
                hexes[h, 4] = nodes[i, j, k + 1]
                hexes[h, 5] = nodes[i + 1, j, k + 1]
                hexes[h, 6] = nodes[i + 1, j + 1, k + 1]
                hexes[h, 7] = nodes[i, j + 1, k + 1]
                h += 1
    return hexes


def _fixed_grid_shell_and_core_hex_ids(
    nx: int, ny: int, nz: int
) -> tuple[np.ndarray, np.ndarray]:
    """Identify indices of internal core vs exterior boundary shell hexes."""
    core: list[int] = []
    shell: list[int] = []
    h = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                is_shell = (
                    i == 0
                    or i == nx - 1
                    or j == 0
                    or j == ny - 1
                    or k == 0
                    or k == nz - 1
                )
                if is_shell:
                    shell.append(h)
                else:
                    core.append(h)
                h += 1
    return np.array(core, dtype=np.int32), np.array(shell, dtype=np.int32)


def compute_equal_phase_stations(
    s_min: float,
    s_max: float,
    control_points: list[tuple[float, float]] | np.ndarray,
    *,
    target_n_stations: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Compute 1D station coordinates along a spine using analytical equal-phase integration.

    Given target cell height profile L(s) defined by piecewise-linear control points,
    evaluates the phase integral:
        W(s) = \\int_{s_min}^s (2\\pi / L(t)) dt
    and places station boundaries at equal phase increments \\Delta W = 2\\pi / target_N.

    Parameters
    ----------
    s_min, s_max : float
        Domain boundaries along the spine axis.
    control_points : list of (position, target_L_mm)
        Control points defining the target unit cell height L along the spine.
    target_n_stations : int, optional
        Target number of subdivisions along the spine. If None, automatically
        determined by the integrated phase count round(total_integral).

    Returns
    -------
    stations : ndarray of shape (N + 1,)
        Monotonically increasing station coordinates from s_min to s_max.
    n_subdivisions : int
        Number of cell layers N along the spine.
    """
    if s_min >= s_max:
        raise ValueError(f"s_min ({s_min}) must be strictly less than s_max ({s_max}).")

    cps = sorted([(float(p), float(l)) for p, l in control_points], key=lambda x: x[0])
    for p, l in cps:
        if l <= 0:
            raise ValueError(f"Target cell height L must be strictly positive; got {l} at s={p}.")

    # Clamp and extend control points to cover [s_min, s_max]
    pts: list[tuple[float, float]] = []
    if cps[0][0] > s_min:
        pts.append((s_min, cps[0][1]))
    for p, l in cps:
        if s_min <= p <= s_max:
            pts.append((p, l))
    if len(pts) == 0:
        pts = [(s_min, cps[0][1]), (s_max, cps[-1][1])]
    else:
        if pts[0][0] > s_min:
            pts.insert(0, (s_min, pts[0][1]))
        if pts[-1][0] < s_max:
            pts.append((s_max, pts[-1][1]))

    # Analytical piecewise-linear integral of 1 / L(s)
    cum_i = [0.0]
    for i in range(len(pts) - 1):
        a, la = pts[i]
        b, lb = pts[i + 1]
        ds = b - a
        if ds <= 0.0:
            continue
        m = (lb - la) / ds
        if abs(m) < 1e-12:
            di = ds / la
        else:
            di = (1.0 / m) * np.log(lb / la)
        cum_i.append(cum_i[-1] + di)

    total_i = cum_i[-1]
    if total_i <= 0.0:
        raise ValueError("Total phase integral must be positive.")

    if target_n_stations is not None:
        n = max(1, int(target_n_stations))
    else:
        n = max(1, int(round(total_i)))

    stations = [float(s_min)]
    for k in range(1, n):
        target_i = (float(k) / float(n)) * total_i
        # Find containing subinterval
        idx = max(0, min(len(cum_i) - 2, int(np.searchsorted(cum_i, target_i)) - 1))
        a, la = pts[idx]
        b, lb = pts[idx + 1]
        ds = b - a
        m = (lb - la) / ds
        di = target_i - cum_i[idx]
        if abs(m) < 1e-12:
            sk = a + la * di
        else:
            sk = a + (la / m) * (np.exp(m * di) - 1.0)
        stations.append(float(sk))
    stations.append(float(s_max))

    return np.array(stations, dtype=np.float64), n


def generate_lofted_hex_scaffold(
    mesh: trimesh.Trimesh,
    *,
    nx: int = 8,
    ny: int = 4,
    nz: int = 4,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    station_coords: np.ndarray | list[float] | None = None,
    station_control_points: list[tuple[float, float]] | np.ndarray | None = None,
    snap_surface_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Generate a boundary-conforming structured hexahedral scaffold lofted along a spine axis.

    This engine slices the part along the chosen spine axis at stations, extracts
    the transverse cross-sectional extents on the CAD mesh, and lofts a structured
    `nx × ny × nz` grid of hexahedra without needing GMSH. Supports 1D equal-phase
    height grading via `station_control_points`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Target solid CAD mesh.
    nx, ny, nz : int
        Number of grid subdivisions along X, Y, and Z.
    spine_axis : {'x', 'y', 'z'}, default 'z'
        The primary lofting / grading direction along which cross-sections follow geometry.
    station_coords : array-like, optional
        Custom 1D station coordinates along the spine.
    station_control_points : list of (position_mm, target_L_mm), optional
        1D control points defining target unit cell height along the spine.
        Uses equal-phase integration to position station boundaries.
    snap_surface_nodes : bool, default False
        If True, project exterior boundary nodes to the closest point on the mesh.

    Returns
    -------
    volume_hexes : ndarray, shape (N, 8, 3)
        All deformed hexahedral elements.
    skin_hexes : ndarray, shape (N, 8, 3)
        Scaffold elements available for surface dual extraction.
    report : dict
        Metadata report detailing element counts, pitch, spine parameters, and shell IDs.
    """
    nx_i, ny_i, nz_i = int(nx), int(ny), int(nz)
    if min(nx_i, ny_i, nz_i) < 1:
        raise ValueError(f"nx, ny, nz must each be >= 1; got ({nx_i}, {ny_i}, {nz_i}).")

    axis_str = str(spine_axis).strip().lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis_str not in axis_map:
        raise ValueError(f"spine_axis must be 'x', 'y', or 'z'; got {spine_axis!r}.")
    spine_ax = axis_map[axis_str]

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]

    # Transverse axes
    all_axes = [0, 1, 2]
    transverse_axes = tuple(a for a in all_axes if a != spine_ax)
    oa, ob = transverse_axes[0], transverse_axes[1]

    # Station coordinates along spine
    n_stations = [nx_i, ny_i, nz_i][spine_ax]
    is_graded = False
    if station_coords is not None:
        stations = np.asarray(station_coords, dtype=np.float64).ravel()
        if len(stations) != n_stations + 1:
            raise ValueError(
                f"station_coords length ({len(stations)}) must equal spine subdivisions + 1 ({n_stations + 1})."
            )
    elif station_control_points is not None:
        stations, n_calc = compute_equal_phase_stations(
            lo[spine_ax], hi[spine_ax], station_control_points, target_n_stations=n_stations
        )
        is_graded = True
    else:
        stations = np.linspace(lo[spine_ax], hi[spine_ax], n_stations + 1)

    nodes = np.zeros((nx_i + 1, ny_i + 1, nz_i + 1, 3), dtype=np.float64)

    if spine_ax == 2:  # spine = Z (transverse = X, Y)
        for k in range(nz_i + 1):
            zv = float(stations[k])
            x0, x1, y0, y1 = _mesh_section_extents_on_plane(
                mesh, 2, zv, (0, 1), fallback_lo=lo, fallback_hi=hi
            )
            for j in range(ny_i + 1):
                yv = y0 + float(j) * (y1 - y0) / float(ny_i)
                for i in range(nx_i + 1):
                    xv = x0 + float(i) * (x1 - x0) / float(nx_i)
                    nodes[i, j, k] = [xv, yv, zv]

    elif spine_ax == 1:  # spine = Y (transverse = X, Z)
        for j in range(ny_i + 1):
            yv = float(stations[j])
            x0, x1, z0, z1 = _mesh_section_extents_on_plane(
                mesh, 1, yv, (0, 2), fallback_lo=lo, fallback_hi=hi
            )
            for k in range(nz_i + 1):
                zv = z0 + float(k) * (z1 - z0) / float(nz_i)
                for i in range(nx_i + 1):
                    xv = x0 + float(i) * (x1 - x0) / float(nx_i)
                    nodes[i, j, k] = [xv, yv, zv]

    else:  # spine = X (transverse = Y, Z)
        for i in range(nx_i + 1):
            xv = float(stations[i])
            y0, y1, z0, z1 = _mesh_section_extents_on_plane(
                mesh, 0, xv, (1, 2), fallback_lo=lo, fallback_hi=hi
            )
            for k in range(nz_i + 1):
                zv = z0 + float(k) * (z1 - z0) / float(nz_i)
                for j in range(ny_i + 1):
                    yv = y0 + float(j) * (y1 - y0) / float(ny_i)
                    nodes[i, j, k] = [xv, yv, zv]

    # Optional boundary surface snapping
    if snap_surface_nodes:
        for i in range(nx_i + 1):
            for j in range(ny_i + 1):
                for k in range(nz_i + 1):
                    is_boundary = (
                        i in (0, nx_i) or j in (0, ny_i) or k in (0, nz_i)
                    )
                    if not is_boundary:
                        continue
                    pt = nodes[i, j, k].reshape(1, 3)
                    snapped, _, _ = trimesh.proximity.closest_point(mesh, pt)
                    s = np.asarray(snapped[0], dtype=np.float64)
                    # Snap transverse coordinates while preserving station position
                    nodes[i, j, k, oa] = float(s[oa])
                    nodes[i, j, k, ob] = float(s[ob])

    all_hexes = _structured_hex_elements_from_node_grid(nodes, nx_i, ny_i, nz_i)
    core_ids, shell_ids = _fixed_grid_shell_and_core_hex_ids(nx_i, ny_i, nz_i)
    n_hex = int(len(all_hexes))
    all_gids = np.arange(n_hex, dtype=np.int32)

    report: dict[str, Any] = {
        "engine": "lofted_hex_scaffold",
        "spine_axis": axis_str,
        "grid_nx": nx_i,
        "grid_ny": ny_i,
        "grid_nz": nz_i,
        "n_hexes_total": n_hex,
        "volume_hexes": n_hex,
        "skin_hexes": n_hex,
        "volume_interior_only_hexes": int(len(core_ids)),
        "skin_shell_hexes": int(len(shell_ids)),
        "spine_stations": stations,
        "is_equal_phase_graded": bool(is_graded),
        "snap_surface_nodes": bool(snap_surface_nodes),
        "volume_global_hex_ids": all_gids,
        "skin_global_hex_ids": all_gids,
    }

    return all_hexes, all_hexes.copy(), report


_F_ONLY_RULES = frozenset({"octahedral", "hex_face_dual"})
_C_ONLY_RULES = frozenset({"grid", "tesseract"})


def synthesize_lofted_multilattice(
    hex_elements: np.ndarray,
    rule_schedule: list[str] | dict[int, str] | Callable[[int], str],
    *,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    grid_shape: tuple[int, int, int] | None = None,
    insert_interface_pyramids: bool = True,
    topology_round_decimals: int = 6,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Synthesize a multi-lattice structure across lofted hex layers with interface pyramids.

    Assigns different cellular rules along spine station layers and automatically
    injects 4-strut transition pyramids at heterogeneous F <-> C boundaries.

    Parameters
    ----------
    hex_elements : ndarray, shape (N, 8, 3)
        Lofted hex brick elements.
    rule_schedule : list of str, dict, or callable
        Mapping from spine layer index (0 to N_spine - 1) to registered rule name.
    spine_axis : {'x', 'y', 'z'}, default 'z'
        The spine axis along which layers are indexed.
    grid_shape : tuple of (nx, ny, nz), optional
        Grid dimensions. If omitted, inferred from element count.
    insert_interface_pyramids : bool, default True
        If True, injects 4 transition pyramid struts at interfaces between
        face-center rules (e.g. octahedral) and corner rules (e.g. grid).
    topology_round_decimals : int, default 6
        Coordinate precision for node welding.
    **kwargs : Any
        Additional arguments passed to rule builders.

    Returns
    -------
    nodes : ndarray, shape (V, 3)
        Welded global lattice nodes.
    struts : ndarray, shape (E, 2)
        Lattice strut connectivity including interface transition pyramids.
    report : dict
        Multi-lattice synthesis report and interface statistics.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")
    n_hex = elems.shape[0]

    axis_str = str(spine_axis).strip().lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis_str not in axis_map:
        raise ValueError(f"spine_axis must be 'x', 'y', or 'z'; got {spine_axis!r}.")
    spine_ax = axis_map[axis_str]

    if grid_shape is None:
        raise ValueError("grid_shape=(nx, ny, nz) must be provided for multi-lattice synthesis.")
    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    if nx * ny * nz != n_hex:
        raise ValueError(f"grid_shape ({nx}, {ny}, {nz}) product {nx*ny*nz} != n_hex {n_hex}.")

    n_spine = [nx, ny, nz][spine_ax]

    # Resolve rule per layer
    def get_layer_rule(layer_idx: int) -> str:
        if isinstance(rule_schedule, (list, tuple)):
            idx = min(max(0, int(layer_idx)), len(rule_schedule) - 1)
            return str(rule_schedule[idx])
        elif isinstance(rule_schedule, dict):
            return str(rule_schedule.get(int(layer_idx), "octahedral"))
        elif callable(rule_schedule):
            return str(rule_schedule(int(layer_idx)))
        return str(rule_schedule)

    layer_rules = [get_layer_rule(l) for l in range(n_spine)]

    # 1. Stamp each element with its layer rule and weld nodes
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()

    for h in range(n_hex):
        k = h // (nx * ny)
        rem = h % (nx * ny)
        j = rem // nx
        i = rem % nx
        layer_idx = [i, j, k][spine_ax]
        rule_name = layer_rules[layer_idx]

        rule = get_hex_topology_rule(rule_name)
        rule_kwargs = _builder_kwargs_for(rule.builder, kwargs)
        elem = elems[h]
        local_nodes, local_struts = rule.builder(elem, **rule_kwargs)

        local_to_global: list[int] = []
        for pt in np.asarray(local_nodes, dtype=np.float64):
            key = tuple(np.round(pt, topology_round_decimals).tolist())
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(pt.copy())
            local_to_global.append(idx)

        for a, b in np.asarray(local_struts, dtype=np.int64).reshape(-1, 2):
            ga, gb = local_to_global[int(a)], local_to_global[int(b)]
            if ga != gb:
                strut_set.add((min(ga, gb), max(ga, gb)))

    # 2. Automated Interface Transition Pyramids between adjacent layers
    n_pyramids = 0
    n_pyramid_struts = 0
    if insert_interface_pyramids and n_spine > 1:
        # Interface face index on the lower-index element:
        # Face 1 (+Z) for spine=Z; Face 3 (+Y) for spine=Y; Face 5 (+X) for spine=X
        face_interface_map = {2: 1, 1: 3, 0: 5}
        pos_face_idx = face_interface_map[spine_ax]
        face_corner_indices = _HEX_FACES[pos_face_idx]

        for l in range(n_spine - 1):
            rule_a = layer_rules[l].lower()
            rule_b = layer_rules[l + 1].lower()

            needs_pyramid = (
                (rule_a in _F_ONLY_RULES and rule_b in _C_ONLY_RULES)
                or (rule_b in _F_ONLY_RULES and rule_a in _C_ONLY_RULES)
            )
            if not needs_pyramid:
                continue

            # Iterate over all transverse columns at this interface
            if spine_ax == 2:  # spine = Z (iterate over i, j)
                for j in range(ny):
                    for i in range(nx):
                        h = (l * ny + j) * nx + i
                        elem = elems[h]
                        corners = elem[list(face_corner_indices)]
                        fc = corners.mean(axis=0)

                        # Lookup or register face center and corners
                        key_f = tuple(np.round(fc, topology_round_decimals).tolist())
                        if key_f not in node_map:
                            node_map[key_f] = len(nodes_list)
                            nodes_list.append(fc.copy())
                        gid_f = node_map[key_f]

                        for c in corners:
                            key_c = tuple(np.round(c, topology_round_decimals).tolist())
                            if key_c not in node_map:
                                node_map[key_c] = len(nodes_list)
                                nodes_list.append(c.copy())
                            gid_c = node_map[key_c]
                            if gid_f != gid_c:
                                edge = (min(gid_f, gid_c), max(gid_f, gid_c))
                                if edge not in strut_set:
                                    strut_set.add(edge)
                                    n_pyramid_struts += 1
                        n_pyramids += 1

            elif spine_ax == 1:  # spine = Y (iterate over i, k)
                for k in range(nz):
                    for i in range(nx):
                        h = (k * ny + l) * nx + i
                        elem = elems[h]
                        corners = elem[list(face_corner_indices)]
                        fc = corners.mean(axis=0)

                        key_f = tuple(np.round(fc, topology_round_decimals).tolist())
                        if key_f not in node_map:
                            node_map[key_f] = len(nodes_list)
                            nodes_list.append(fc.copy())
                        gid_f = node_map[key_f]

                        for c in corners:
                            key_c = tuple(np.round(c, topology_round_decimals).tolist())
                            if key_c not in node_map:
                                node_map[key_c] = len(nodes_list)
                                nodes_list.append(c.copy())
                            gid_c = node_map[key_c]
                            if gid_f != gid_c:
                                edge = (min(gid_f, gid_c), max(gid_f, gid_c))
                                if edge not in strut_set:
                                    strut_set.add(edge)
                                    n_pyramid_struts += 1
                        n_pyramids += 1

            else:  # spine = X (iterate over j, k)
                for k in range(nz):
                    for j in range(ny):
                        h = (k * ny + j) * nx + l
                        elem = elems[h]
                        corners = elem[list(face_corner_indices)]
                        fc = corners.mean(axis=0)

                        key_f = tuple(np.round(fc, topology_round_decimals).tolist())
                        if key_f not in node_map:
                            node_map[key_f] = len(nodes_list)
                            nodes_list.append(fc.copy())
                        gid_f = node_map[key_f]

                        for c in corners:
                            key_c = tuple(np.round(c, topology_round_decimals).tolist())
                            if key_c not in node_map:
                                node_map[key_c] = len(nodes_list)
                                nodes_list.append(c.copy())
                            gid_c = node_map[key_c]
                            if gid_f != gid_c:
                                edge = (min(gid_f, gid_c), max(gid_f, gid_c))
                                if edge not in strut_set:
                                    strut_set.add(edge)
                                    n_pyramid_struts += 1
                        n_pyramids += 1

    final_nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    final_struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )

    report: dict[str, Any] = {
        "engine": "lofted_multilattice",
        "spine_axis": axis_str,
        "grid_shape": (nx, ny, nz),
        "n_hexes": n_hex,
        "layer_rules": layer_rules,
        "n_nodes": int(len(final_nodes)),
        "n_struts": int(len(final_struts)),
        "n_transition_pyramids": n_pyramids,
        "n_pyramid_struts": n_pyramid_struts,
        "round_decimals": int(topology_round_decimals),
    }

    return final_nodes, final_struts, report


def synthesize_lofted_lattice(
    hex_elements: np.ndarray,
    *,
    rule_name: str | list[str] | dict[int, str] | Callable[[int], str] = "octahedral",
    rule_schedule: list[str] | dict[int, str] | Callable[[int], str] | None = None,
    spine_axis: Literal["x", "y", "z", "X", "Y", "Z"] = "z",
    grid_shape: tuple[int, int, int] | None = None,
    insert_interface_pyramids: bool = True,
    topology_round_decimals: int = 6,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Synthesize explicit strut lattice topology onto deformed lofted hex elements.

    Supports uniform cellular rules or multi-lattice layer schedules with automated
    interface transition pyramids.

    Parameters
    ----------
    hex_elements : ndarray, shape (N, 8, 3)
        Lofted hex brick elements.
    rule_name : str, list, dict, or callable, default 'octahedral'
        Name of registered hex cellular rule, or multi-lattice schedule.
    rule_schedule : list, dict, or callable, optional
        Explicit layer rule schedule for multi-lattice synthesis.
    spine_axis : {'x', 'y', 'z'}, default 'z'
        The spine axis along which layers are indexed.
    grid_shape : tuple of (nx, ny, nz), optional
        Grid dimensions (required for multi-lattice synthesis).
    insert_interface_pyramids : bool, default True
        If True, injects 4 transition pyramid struts at interfaces between
        face-center rules (e.g. octahedral) and corner rules (e.g. grid).
    topology_round_decimals : int, default 6
        Coordinate precision for node welding.
    **kwargs : Any
        Additional parameters passed to unit cell rule builders.

    Returns
    -------
    nodes : ndarray, shape (V, 3)
        Welded global lattice nodes.
    struts : ndarray, shape (E, 2)
        Lattice strut connectivity.
    report : dict
        Synthesis statistics and metadata.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")

    sched = rule_schedule if rule_schedule is not None else rule_name
    is_multi = isinstance(sched, (list, tuple, dict)) or callable(sched)

    if is_multi:
        return synthesize_lofted_multilattice(
            elems,
            rule_schedule=sched,  # type: ignore
            spine_axis=spine_axis,
            grid_shape=grid_shape,
            insert_interface_pyramids=insert_interface_pyramids,
            topology_round_decimals=topology_round_decimals,
            **kwargs,
        )

    # Uniform single-rule synthesis
    nodes, struts = generate_hex_topology(
        elems,
        rule_name=str(rule_name),
        round_decimals=int(topology_round_decimals),
        **kwargs,
    )

    report: dict[str, Any] = {
        "rule_name": str(rule_name),
        "n_hexes": int(elems.shape[0]),
        "n_nodes": int(len(nodes)),
        "n_struts": int(len(struts)),
        "round_decimals": int(topology_round_decimals),
    }

    return nodes, struts, report


def generate_brute_force_fixed_grid_hex_scaffold(
    mesh: trimesh.Trimesh,
    *,
    nx: int = 8,
    ny: int = 4,
    nz: int = 4,
    taper_along: str = "z",
    snap_surface_nodes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Backwards-compatible wrapper routing to `generate_lofted_hex_scaffold`.
    """
    return generate_lofted_hex_scaffold(
        mesh,
        nx=nx,
        ny=ny,
        nz=nz,
        spine_axis=taper_along,  # type: ignore
        snap_surface_nodes=snap_surface_nodes,
    )
