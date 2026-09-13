"""
Transversely isotropic hexagonal pentamode generator.

ABAB-stacked planar hexagonal honeycombs with identical strut orientations.
Layer B is staggered by lateral translation only (no in-plane rotation):
    Delta r = (sqrt(3)/2 * a, 1/2 * a, c)

Midplane hubs connect 2 Layer-A nodes + 2 Layer-B nodes (hub Z=4).
Honeycomb vertices have 3 in-plane struts + 1 inclined branch (Z=4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import manifold3d as m3d
import numpy as np
import trimesh

from graphite.explicit.geometry_module import (
    _affine_rows_from_R_t,
    _manifold_to_trimesh,
    _rotation_align_local_z_to_unit,
)
from graphite.generators.pentamode import (
    ImplicitField,
    LatticeGraph,
    bicone_radius,
)

# Strut role codes stored in LatticeGraph.metadata["strut_roles"]
ROLE_BASAL = 0
ROLE_VERTICAL = 1

# Deprecated: rotation stagger removed. Kept so old imports do not break.
TOP_RING_PHASE = 0.0


def ab_layer_shift(a: float, c: float = 0.0) -> np.ndarray:
    """Lateral (+ optional z) AB shift of Layer B relative to Layer A."""
    a = float(a)
    return np.array([0.5 * np.sqrt(3.0) * a, 0.5 * a, float(c)], dtype=np.float64)


def _primitive_vectors(a: float) -> tuple[np.ndarray, np.ndarray]:
    """2D primitive translation vectors for the hexagonal honeycomb tiling."""
    a = float(a)
    v1 = np.array([np.sqrt(3.0) * a, 0.0, 0.0], dtype=np.float64)
    v2 = np.array([0.5 * np.sqrt(3.0) * a, 1.5 * a, 0.0], dtype=np.float64)
    return v1, v2


def generate_transverse_hexagonal_pentamode(
    nx: int = 3,
    ny: int = 3,
    nz: int = 1,
    a: float = 1.0,
    c: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate an ABAB-stacked, non-intersecting hexagonal pentamode lattice.

    Parameters:
        nx, ny: Number of in-plane unit-cell repetitions.
        nz: Number of interlayer pitches along z (nz=1 → layers at 0 and c).
        a: Hexagonal beam length (honeycomb nearest-neighbor spacing / circumradius).
        c: Layer height along the impact axis (z).

    Returns:
        nodes: (N, 3) float64 unique vertex coordinates
        edges: (M, 2) int64 undirected edge indices
        roles: (M,) int64 ROLE_BASAL or ROLE_VERTICAL
    """
    nx, ny, nz = int(nx), int(ny), int(nz)
    a, c = float(a), float(c)
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError(f"nx, ny, nz must be >= 1, got {nx}, {ny}, {nz}")
    if a <= 1e-12 or c <= 1e-12:
        raise ValueError(f"a and c must be positive, got a={a}, c={c}")

    shift_xy = ab_layer_shift(a, 0.0)[:2]
    # Hex-center lattice for edge-sharing rings (circumradius a)
    c1 = np.array([1.5 * a, 0.5 * np.sqrt(3.0) * a], dtype=np.float64)
    c2 = np.array([0.0, np.sqrt(3.0) * a], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)  # identical every layer

    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []
    roles: list[int] = []
    node_lookup: dict[tuple[float, float, float], int] = {}

    def get_or_add_node(pt: np.ndarray) -> int:
        key = (round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4))
        if key not in node_lookup:
            idx = len(nodes)
            node_lookup[key] = idx
            nodes.append(np.asarray(pt, dtype=np.float64).reshape(3))
            return idx
        return node_lookup[key]

    def _layer_xy_offset(L: int) -> np.ndarray:
        return shift_xy if (L % 2) else np.zeros(2, dtype=np.float64)

    # --- Basal honeycombs (identical orientation; AB lateral shift on odd planes) ---
    for L in range(nz + 1):
        z = float(L) * c
        off = _layer_xy_offset(L)
        for ix in range(-1, nx + 1):
            for iy in range(-1, ny + 1):
                center = off + ix * c1 + iy * c2
                idxs = []
                for k in range(6):
                    pt = np.array(
                        [
                            center[0] + a * np.cos(angles[k]),
                            center[1] + a * np.sin(angles[k]),
                            z,
                        ],
                        dtype=np.float64,
                    )
                    idxs.append(get_or_add_node(pt))
                for k in range(6):
                    edges.append((idxs[k], idxs[(k + 1) % 6]))
                    roles.append(ROLE_BASAL)

    nodes_arr = np.vstack(nodes) if nodes else np.zeros((0, 3), dtype=np.float64)
    if len(nodes_arr) == 0:
        return nodes_arr, np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)

    basal_uniq: dict[tuple[int, int], None] = {}
    for (u, v), role in zip(edges, roles, strict=True):
        if role != ROLE_BASAL:
            continue
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key[0] != key[1]:
            basal_uniq[key] = None
    basal_edges = list(basal_uniq.keys())

    key_to_idx = {
        (round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)): i
        for i, p in enumerate(nodes_arr)
    }

    def _edge_abs_angle(u: int, v: int) -> float:
        d = nodes_arr[v, :2] - nodes_arr[u, :2]
        ang = float(np.arctan2(d[1], d[0])) % np.pi
        return min(ang, np.pi - ang)

    candidates = [e for e in basal_edges if _edge_abs_angle(*e) <= np.deg2rad(25.0)]
    by_z: dict[float, list[tuple[int, int]]] = {}
    for u, v in candidates:
        z_key = float(np.round(0.5 * (nodes_arr[u, 2] + nodes_arr[v, 2]), 6))
        by_z.setdefault(z_key, []).append((u, v))

    matched: list[tuple[int, int]] = []
    for _z, elist in by_z.items():
        used: set[int] = set()
        for u, v in sorted(
            elist,
            key=lambda e: (
                round(float(nodes_arr[e[0], 0] + nodes_arr[e[1], 0]), 4),
                round(float(nodes_arr[e[0], 1] + nodes_arr[e[1], 1]), 4),
            ),
        ):
            if u in used or v in used:
                continue
            matched.append((u, v))
            used.add(u)
            used.add(v)

    def _half(u: int, v: int) -> int:
        mid = 0.5 * (nodes_arr[u, :2] + nodes_arr[v, :2])
        qi = int(np.floor(mid[0] / (1.5 * a) + 1e-9))
        qj = int(np.floor(mid[1] / (np.sqrt(3.0) * a) + 1e-9))
        return (qi + qj) % 2

    hub_nodes: list[np.ndarray] = []
    hub_edges: list[tuple[int, int]] = []
    hub_roles: list[int] = []
    hub_offset = len(nodes_arr)

    for L in range(nz):
        z_lo = float(L) * c
        z_hi = float(L + 1) * c
        z_hub = (float(L) + 0.5) * c
        off_step = shift_xy if (L % 2 == 0) else -shift_xy
        half = 0 if nz == 1 else (L % 2)

        for u, v in matched:
            if abs(nodes_arr[u, 2] - z_lo) > 1e-4:
                continue
            if nz > 1 and _half(u, v) != half:
                continue

            def _partner(idx: int, _off=off_step, _zhi=z_hi) -> int | None:
                p = nodes_arr[idx]
                key = (
                    round(float(p[0] + _off[0]), 4),
                    round(float(p[1] + _off[1]), 4),
                    round(float(_zhi), 4),
                )
                return key_to_idx.get(key)

            pu, pv = _partner(u), _partner(v)
            if pu is None or pv is None:
                continue
            hub_xy = 0.25 * (
                nodes_arr[u, :2]
                + nodes_arr[v, :2]
                + nodes_arr[pu, :2]
                + nodes_arr[pv, :2]
            )
            hub_nodes.append(np.array([hub_xy[0], hub_xy[1], z_hub], dtype=np.float64))
            h = hub_offset + len(hub_nodes) - 1
            for q in (u, v, pu, pv):
                hub_edges.append((q, h))
                hub_roles.append(ROLE_VERTICAL)

    if hub_nodes:
        nodes_arr = np.vstack([nodes_arr, np.vstack(hub_nodes)])
        all_edge_map: dict[tuple[int, int], int] = {e: ROLE_BASAL for e in basal_edges}
        for (u, v), role in zip(hub_edges, hub_roles, strict=True):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            all_edge_map[key] = int(role)
    else:
        all_edge_map = {e: ROLE_BASAL for e in basal_edges}

    edge_list = list(all_edge_map.keys())
    role_list = [all_edge_map[e] for e in edge_list]
    return (
        nodes_arr,
        np.array(edge_list, dtype=np.int64),
        np.array(role_list, dtype=np.int64),
    )



def hexagonal_pentamode_cell(
    a: float = 1.0,
    c: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single AB unit (nx=ny=nz=1): Layer A at z=0, Layer B translated at z=c,
    two midplane hubs. No in-plane rotation.
    """
    nodes, edges, _roles = generate_transverse_hexagonal_pentamode(
        nx=1, ny=1, nz=1, a=a, c=c
    )
    return nodes, edges


def generate_hexagonal_pentamode_graph(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    a: float = 1.0,
    c: float = 1.5,
    *,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an ABAB hexagonal pentamode graph.

    Prefer ``nx, ny, nz`` (cell counts). If omitted, infer counts from ``bounds``
    using the hexagonal primitive vectors.
    """
    a, c = float(a), float(c)
    v1, v2 = _primitive_vectors(a)

    if nx is None or ny is None or nz is None:
        if bounds is None:
            nx = nx or 3
            ny = ny or 3
            nz = nz or 1
        else:
            (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
            # Cover the XY box with primitive cells
            span_x = max(xmax - xmin, a)
            span_y = max(ymax - ymin, a)
            span_z = max(zmax - zmin, c)
            nx = nx or max(1, int(np.ceil(span_x / (np.sqrt(3.0) * a))))
            ny = ny or max(1, int(np.ceil(span_y / (1.5 * a))))
            nz = nz or max(1, int(np.round(span_z / c)))

    nodes, edges, roles = generate_transverse_hexagonal_pentamode(
        nx=int(nx), ny=int(ny), nz=int(nz), a=a, c=c
    )

    if bounds is not None and len(edges) > 0:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
        # Translate so lattice origin sits at bounds minimum
        nodes = nodes + np.array([xmin, ymin, zmin], dtype=np.float64)
        mid = 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])
        tol = 1e-4
        inside = (
            (mid[:, 0] >= xmin - tol)
            & (mid[:, 0] <= xmax + tol)
            & (mid[:, 1] >= ymin - tol)
            & (mid[:, 1] <= ymax + tol)
            & (mid[:, 2] >= zmin - tol)
            & (mid[:, 2] <= zmax + tol)
        )
        edges = edges[inside]
        roles = roles[inside]
        if len(edges) == 0:
            return (
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 2), dtype=np.int64),
                np.zeros(0, dtype=np.int64),
            )
        used = np.unique(edges)
        remap = {int(old): new for new, old in enumerate(used)}
        nodes = nodes[used]
        edges = np.array([[remap[int(u)], remap[int(v)]] for u, v in edges], dtype=np.int64)

    return nodes, edges, roles


def _build_hex_pentamode_mesh(
    nodes: np.ndarray,
    struts: np.ndarray,
    roles: np.ndarray,
    r_min: float,
    r_max: float,
    r_basal: float,
    *,
    crop_to_bounds: bool = False,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    n_bicone_segments: int = 10,
) -> trimesh.Trimesh:
    """Manifold mesh: constant basal cylinders + vertical bicones + nodal spheres."""
    parts: list[m3d.Manifold] = []

    for (u, v), role in zip(struts, roles, strict=True):
        pA = nodes[int(u)]
        pB = nodes[int(v)]
        seg = pB - pA
        L = float(np.linalg.norm(seg))
        if L <= 1e-8:
            continue
        z_unit = seg / L
        R_mat = _rotation_align_local_z_to_unit(z_unit)

        if int(role) == ROLE_BASAL:
            mid = 0.5 * (pA + pB)
            aff = _affine_rows_from_R_t(R_mat, mid)
            cyl = m3d.Manifold.cylinder(
                height=L * 1.02,
                radius_low=float(r_basal),
                radius_high=float(r_basal),
                circular_segments=16,
                center=True,
            ).transform(aff)
            parts.append(cyl)
        else:
            for k in range(n_bicone_segments):
                t0 = k / float(n_bicone_segments)
                t1 = (k + 1) / float(n_bicone_segments)
                r0 = float(bicone_radius(t0, r_min, r_max))
                r1 = float(bicone_radius(t1, r_min, r_max))
                pos0 = pA + t0 * seg
                pos1 = pA + t1 * seg
                mid = 0.5 * (pos0 + pos1)
                h_seg = (L / float(n_bicone_segments)) * 1.05
                aff = _affine_rows_from_R_t(R_mat, mid)
                frustum = m3d.Manifold.cylinder(
                    height=h_seg,
                    radius_low=r0,
                    radius_high=r1,
                    circular_segments=16,
                    center=True,
                ).transform(aff)
                parts.append(frustum)

    degrees = np.zeros(len(nodes), dtype=np.int64)
    for u, v in struts:
        degrees[int(u)] += 1
        degrees[int(v)] += 1

    vertical_nodes: set[int] = set()
    for (u, v), role in zip(struts, roles, strict=True):
        if int(role) == ROLE_VERTICAL:
            vertical_nodes.add(int(u))
            vertical_nodes.add(int(v))

    for idx, pt in enumerate(nodes):
        if degrees[idx] == 0:
            continue
        if degrees[idx] >= 4 and idx in vertical_nodes:
            rad = float(r_min)
        elif idx in vertical_nodes:
            rad = float(max(r_min, r_basal))
        else:
            rad = float(r_basal)
        sphere = m3d.Manifold.sphere(rad * 1.05, circular_segments=16).translate(
            tuple(float(c) for c in pt)
        )
        parts.append(sphere)

    if not parts:
        raise ValueError("No solid parts generated for hexagonal pentamode mesh")

    composed = m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)
    if crop_to_bounds and bounds is not None:
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
        box = m3d.Manifold.cube([xmax - xmin, ymax - ymin, zmax - zmin]).translate(
            (xmin, ymin, zmin)
        )
        composed = composed ^ box
    return _manifold_to_trimesh(composed)


def _build_hex_pentamode_sdf(
    nodes: np.ndarray,
    struts: np.ndarray,
    roles: np.ndarray,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
    r_min: float,
    r_max: float,
    r_basal: float,
    grid_resolution: int,
) -> ImplicitField:
    """Exact min distance to basal cylinders and vertical bicones (no blend fill)."""
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    res = int(grid_resolution)
    xs = np.linspace(xmin, xmax, res)
    ys = np.linspace(ymin, ymax, res)
    zs = np.linspace(zmin, zmax, res)
    spacing = (
        float((xmax - xmin) / max(res - 1, 1)),
        float((ymax - ymin) / max(res - 1, 1)),
        float((zmax - zmin) / max(res - 1, 1)),
    )
    origin = (float(xmin), float(ymin), float(zmin))
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    min_sdf = np.full(len(grid_pts), 1e6, dtype=np.float64)

    for (u, v), role in zip(struts, roles, strict=True):
        pA = nodes[int(u)]
        pB = nodes[int(v)]
        seg = pB - pA
        L = float(np.linalg.norm(seg))
        if L <= 1e-8:
            continue
        u_dir = seg / L
        diff = grid_pts - pA
        proj = np.dot(diff, u_dir)
        t = np.clip(proj / L, 0.0, 1.0)
        closest = pA + t[:, None] * seg
        dist_to_axis = np.linalg.norm(grid_pts - closest, axis=1)
        if int(role) == ROLE_BASAL:
            r_local = float(r_basal)
        else:
            r_local = bicone_radius(t, r_min=r_min, r_max=r_max)
        min_sdf = np.minimum(min_sdf, dist_to_axis - r_local)

    field = min_sdf.reshape((res, res, res)).astype(np.float32)
    return ImplicitField(
        field=field,
        origin=origin,
        spacing=spacing,
        metadata={"topology": "hexagonal_pentamode"},
    )


def generate_hexagonal_pentamode_lattice(
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    a: float = 5.0,
    c: float = 7.5,
    r_min: float = 0.25,
    r_max: float = 0.80,
    r_basal: float = 0.20,
    output_format: Literal["graph", "implicit_sdf", "mesh"] = "graph",
    grid_resolution: int = 64,
    crop_to_bounds: bool = False,
    *,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
) -> LatticeGraph | ImplicitField | trimesh.Trimesh:
    """
    Generate a transversely isotropic hexagonal pentamode lattice (AB translation stagger).

    Prefer ``nx, ny, nz``. If only ``bounds`` is given, cell counts are inferred.
    """
    nodes, struts, roles = generate_hexagonal_pentamode_graph(
        bounds=bounds, a=a, c=c, nx=nx, ny=ny, nz=nz
    )
    if len(struts) == 0:
        raise ValueError(
            f"No hexagonal pentamode struts for bounds={bounds}, nx={nx}, ny={ny}, nz={nz}, a={a}, c={c}"
        )

    # Bounds for SDF / crop: use provided box or axis-aligned node extents
    if bounds is None:
        pad = float(a) * 0.05
        lo = nodes.min(axis=0) - pad
        hi = nodes.max(axis=0) + pad
        bounds = ((float(lo[0]), float(lo[1]), float(lo[2])),
                  (float(hi[0]), float(hi[1]), float(hi[2])))

    if output_format == "graph":
        return LatticeGraph(
            nodes=nodes,
            struts=struts,
            radii=np.where(roles == ROLE_VERTICAL, float(r_min), float(r_basal)).astype(
                np.float64
            ),
            metadata={
                "topology": "hexagonal_pentamode",
                "stagger": "AB_translation",
                "a": float(a),
                "c": float(c),
                "r_min": float(r_min),
                "r_max": float(r_max),
                "r_basal": float(r_basal),
                "strut_roles": roles,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            },
        )

    if output_format == "implicit_sdf":
        return _build_hex_pentamode_sdf(
            nodes,
            struts,
            roles,
            bounds=bounds,
            r_min=r_min,
            r_max=r_max,
            r_basal=r_basal,
            grid_resolution=grid_resolution,
        )

    if output_format == "mesh":
        return _build_hex_pentamode_mesh(
            nodes,
            struts,
            roles,
            r_min=r_min,
            r_max=r_max,
            r_basal=r_basal,
            crop_to_bounds=crop_to_bounds,
            bounds=bounds,
        )

    raise ValueError(
        f"Unknown output_format '{output_format}'. Expected 'graph', 'implicit_sdf', or 'mesh'."
    )


def export_hex_pentamode_review(
    out_dir: str | Path | None = None,
) -> dict:
    """Write review STLs under outputs/hexagonal_pentamode/ (unit + 3x3x2)."""
    import json
    from collections import defaultdict

    root = Path(__file__).resolve().parents[2]
    out = Path(out_dir) if out_dir is not None else root / "outputs" / "hexagonal_pentamode"
    out.mkdir(parents=True, exist_ok=True)

    a, c = 5.0, 7.5
    r_min, r_max, r_basal = 0.28, 0.85, 0.22

    cell_nodes, cell_edges, cell_roles = generate_transverse_hexagonal_pentamode(
        nx=1, ny=1, nz=1, a=a, c=c
    )
    unit_mesh = _build_hex_pentamode_mesh(
        cell_nodes, cell_edges, cell_roles, r_min, r_max, r_basal
    )
    unit_path = out / "hex_pentamode_unit_cell.stl"
    unit_mesh.export(unit_path)

    nx, ny, nz = 3, 3, 2
    block_mesh = generate_hexagonal_pentamode_lattice(
        a=a,
        c=c,
        r_min=r_min,
        r_max=r_max,
        r_basal=r_basal,
        output_format="mesh",
        nx=nx,
        ny=ny,
        nz=nz,
    )
    block_path = out / "hex_pentamode_3x3x2.stl"
    block_mesh.export(block_path)
    (out / "hex_pentamode_block.stl").write_bytes(block_path.read_bytes())

    graph = generate_hexagonal_pentamode_lattice(
        a=a,
        c=c,
        r_min=r_min,
        r_max=r_max,
        r_basal=r_basal,
        output_format="graph",
        nx=nx,
        ny=ny,
        nz=nz,
    )
    assert isinstance(graph, LatticeGraph)
    roles = graph.metadata["strut_roles"]
    degrees = np.zeros(len(graph.nodes), dtype=np.int64)
    for u, v in graph.struts:
        degrees[int(u)] += 1
        degrees[int(v)] += 1

    incident: dict[int, list[int]] = defaultdict(list)
    for (u, v), role in zip(graph.struts, roles, strict=True):
        incident[int(u)].append(int(role))
        incident[int(v)].append(int(role))
    hub_degrees = [
        int(degrees[i])
        for i, rs in incident.items()
        if rs and all(r == ROLE_VERTICAL for r in rs) and degrees[i] == 4
    ]
    ring_degrees = [
        int(degrees[i])
        for i, rs in incident.items()
        if any(r == ROLE_BASAL for r in rs) and any(r == ROLE_VERTICAL for r in rs)
    ]

    report = {
        "stagger": "AB_translation",
        "delta_r": ab_layer_shift(a, c).tolist(),
        "a_mm": a,
        "c_mm": c,
        "r_min_mm": r_min,
        "r_max_mm": r_max,
        "r_basal_mm": r_basal,
        "unit_cell": {
            "nodes": int(len(cell_nodes)),
            "edges": int(len(cell_edges)),
            "stl": str(unit_path.name),
            "watertight": bool(unit_mesh.is_watertight),
            "volume_mm3": float(unit_mesh.volume),
        },
        "block": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "nodes": int(len(graph.nodes)),
            "struts": int(len(graph.struts)),
            "n_basal": int(np.sum(roles == ROLE_BASAL)),
            "n_vertical": int(np.sum(roles == ROLE_VERTICAL)),
            "hub_degrees_unique": sorted(set(hub_degrees)),
            "n_hubs": len(hub_degrees),
            "ring_degrees_unique": sorted(set(ring_degrees)),
            "stl": str(block_path.name),
            "watertight": bool(block_mesh.is_watertight),
            "volume_mm3": float(block_mesh.volume),
        },
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["paths"] = {
        "unit_cell_stl": str(unit_path),
        "block_stl": str(block_path),
        "report": str(report_path),
    }
    return report


if __name__ == "__main__":
    import json as _json

    info = export_hex_pentamode_review()
    print(_json.dumps(info, indent=2))
