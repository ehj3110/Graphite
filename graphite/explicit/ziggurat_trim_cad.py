"""Corner-inset half-cell ziggurat CAD for trim / dual interface tests.

Half-cell voxels (a, b, c) are filled when::

    a >= 1, b >= 1, c >= 1
    a < 1 + W(c), b < 1 + W(c)
    W(c) = (2 * n_hex - 2) - (c - 1)

That insets the solid 0.5 cell from −X/−Y/−Z (skins on midplanes) and
shrinks the +X/+Y plan by 0.5 cell per 0.5 cell of +Z (stairs + sidewall
steps). Hex field is ``n_hex³`` unit cells starting at the origin.

Not every one of the 27 (H−/Full/H+)³ combinations appears — receding
+X/+Y drops far-corner types at height — but full, half, quarter, and
eighth all do, with horizontal, stair, and sidewall neighbor changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from graphite.explicit.geometry_module import union_solid_meshes
from graphite.explicit.hex_rules import apply_hex_grid
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf


def _as_cell_dims(cell_size) -> np.ndarray:
    cell_dims = np.asarray(cell_size, dtype=np.float64)
    if cell_dims.ndim == 0:
        cell_dims = np.full(3, float(cell_dims), dtype=np.float64)
    if cell_dims.shape != (3,) or np.any(cell_dims <= 0.0):
        raise ValueError("cell_size must be a positive scalar or three positive dimensions")
    return cell_dims


def hex_corners(ijk: tuple[int, int, int], cell_dims) -> np.ndarray:
    """SC hex corners for integer index ``(i, j, k)`` (same winding as hex_rules)."""
    cd = _as_cell_dims(cell_dims)
    o = np.asarray(ijk, dtype=np.float64) * cd
    e = cd
    return np.array(
        [
            o + [0, 0, 0],
            o + [e[0], 0, 0],
            o + [e[0], e[1], 0],
            o + [0, e[1], 0],
            o + [0, 0, e[2]],
            o + [e[0], 0, e[2]],
            o + [e[0], e[1], e[2]],
            o + [0, e[1], e[2]],
        ],
        dtype=np.float64,
    )


def weld_graphs(graphs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    parts = [np.asarray(n, dtype=np.float64).reshape(-1, 3) for n, _s in graphs]
    stacked = np.vstack(parts)
    keys = np.round(stacked, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    offset = 0
    edges: list[tuple[int, int]] = []
    for n, s in graphs:
        for a, b in np.asarray(s, dtype=np.int64).reshape(-1, 2):
            ia, ib = int(inv[offset + int(a)]), int(inv[offset + int(b)])
            if ia == ib:
                continue
            edges.append((ia, ib) if ia < ib else (ib, ia))
        offset += len(n)
    struts = (
        np.array(sorted(set(edges)), dtype=np.int64)
        if edges
        else np.empty((0, 2), dtype=np.int64)
    )
    return uniq.astype(np.float64), struts


@dataclass(frozen=True)
class ZigguratSpec:
    """``n_hex`` unit cells per axis; half-cell stairs with a 1-half-cell inset."""

    n_hex: int = 4
    cell_dims: tuple[float, float, float] = (4.0, 4.0, 4.0)

    def __post_init__(self) -> None:
        if int(self.n_hex) < 3:
            raise ValueError("n_hex must be >= 3 to get full, half, quarter, and eighth")
        object.__setattr__(self, "n_hex", int(self.n_hex))
        object.__setattr__(self, "cell_dims", tuple(float(x) for x in _as_cell_dims(self.cell_dims)))

    @property
    def n_half(self) -> int:
        return 2 * self.n_hex

    @property
    def w0(self) -> int:
        """Half-cell plan width at the first filled Z layer (c=1)."""
        return self.n_half - 2

    def width_at(self, c: int) -> int:
        """Plan width in half-cells at half-index ``c`` (0-based)."""
        if int(c) < 1:
            return 0
        w = self.w0 - (int(c) - 1)
        return int(w) if w >= 1 else 0

    def octant_filled(self, a: int, b: int, c: int) -> bool:
        w = self.width_at(int(c))
        if w < 1:
            return False
        return int(a) >= 1 and int(b) >= 1 and int(a) < 1 + w and int(b) < 1 + w

    def filled_slabs(self) -> list[tuple[int, int]]:
        """``(c, W)`` for each non-empty Z half-layer."""
        out: list[tuple[int, int]] = []
        for c in range(self.n_half):
            w = self.width_at(c)
            if w >= 1:
                out.append((c, w))
        return out

    def half_size(self) -> np.ndarray:
        return np.asarray(self.cell_dims, dtype=np.float64) * 0.5

    def octant_origin(self, a: int, b: int, c: int) -> np.ndarray:
        h = self.half_size()
        return np.array([a, b, c], dtype=np.float64) * h

    def slab_world_box(self, c: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Origin and extents (mm) of the XY slab at half-index ``c``."""
        h = self.half_size()
        origin = np.array([1.0, 1.0, float(c)], dtype=np.float64) * h
        extents = np.array([float(w), float(w), 1.0], dtype=np.float64) * h
        return origin, extents

    def hex_filled_octants(self, i: int, j: int, k: int) -> list[tuple[int, int, int]]:
        """Local octant indices ``(da, db, dc)`` in {0,1} that contain solid."""
        out: list[tuple[int, int, int]] = []
        for dc in (0, 1):
            for db in (0, 1):
                for da in (0, 1):
                    if self.octant_filled(2 * int(i) + da, 2 * int(j) + db, 2 * int(k) + dc):
                        out.append((da, db, dc))
        return out

    def hex_ijk(self, hex_i: int) -> tuple[int, int, int]:
        n = self.n_hex
        i = int(hex_i) % n
        j = (int(hex_i) // n) % n
        k = int(hex_i) // (n * n)
        return i, j, k

    def iter_occupied_hexes(self) -> list[tuple[int, int, int]]:
        n = self.n_hex
        return [
            (i, j, k)
            for k in range(n)
            for j in range(n)
            for i in range(n)
            if self.hex_filled_octants(i, j, k)
        ]


def occupancy_inventory(spec: ZigguratSpec) -> dict:
    """Count hexes by filled-octant count (8=full, 4=half-rect, 2=quarter, 1=eighth)."""
    by_n: dict[int, int] = {}
    n_empty = 0
    n_occ = 0
    for k in range(spec.n_hex):
        for j in range(spec.n_hex):
            for i in range(spec.n_hex):
                n = len(spec.hex_filled_octants(i, j, k))
                if n == 0:
                    n_empty += 1
                    continue
                n_occ += 1
                by_n[n] = by_n.get(n, 0) + 1
    return {
        "n_hex_field": spec.n_hex**3,
        "n_hex_occupied": n_occ,
        "n_hex_empty": n_empty,
        "n_octants_filled": by_n,
        "n_full_8": int(by_n.get(8, 0)),
        "n_halfish_4": int(by_n.get(4, 0)),
        "n_quarterish_2": int(by_n.get(2, 0)),
        "n_eighth_1": int(by_n.get(1, 0)),
        "n_other": int(n_occ - by_n.get(8, 0) - by_n.get(4, 0) - by_n.get(2, 0) - by_n.get(1, 0)),
    }


def kept_planes_for_hex(
    spec: ZigguratSpec, i: int, j: int, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact {0, 0.5, 1} keep sets from filled octants (empty hex → empty arrays)."""
    filled = spec.hex_filled_octants(i, j, k)
    empty = np.zeros(0, dtype=np.float64)
    if not filled:
        return (empty, empty, empty)
    planes: list[np.ndarray] = []
    for axis in range(3):
        halves = {int(p[axis]) for p in filled}
        kp: list[float] = []
        if 0 in halves:
            kp.extend([0.0, 0.5])
        if 1 in halves:
            kp.extend([0.5, 1.0])
        planes.append(np.unique(np.asarray(kp, dtype=np.float64)))
    return (planes[0], planes[1], planes[2])


def kept_planes_per_hex(spec: ZigguratSpec) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = spec.n_hex
    return [kept_planes_for_hex(spec, i, j, k) for k in range(n) for j in range(n) for i in range(n)]


def _node_hits_filled_octant(
    uvw: np.ndarray,
    spec: ZigguratSpec,
    i: int,
    j: int,
    k: int,
    *,
    atol: float = 1e-6,
) -> bool:
    u = np.asarray(uvw, dtype=np.float64).reshape(3)
    halves: list[list[int]] = []
    for a in range(3):
        opts: list[int] = []
        if float(u[a]) <= 0.5 + atol:
            opts.append(0)
        if float(u[a]) >= 0.5 - atol:
            opts.append(1)
        if not opts:
            return False
        halves.append(opts)
    for da in halves[0]:
        for db in halves[1]:
            for dc in halves[2]:
                if spec.octant_filled(2 * i + da, 2 * j + db, 2 * k + dc):
                    return True
    return False


def stamp_rule_on_ziggurat(
    spec: ZigguratSpec,
    rule_name: str,
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stamp ``rule_name`` and keep nodes whose UVW octant is filled (exact, no SDF)."""
    from graphite.explicit.hex_topology_module import get_hex_topology_rule
    from graphite.explicit.sc_role_surface_dual import hex_uvw

    rule = get_hex_topology_rule(str(rule_name).strip().lower())
    graphs: list[tuple[np.ndarray, np.ndarray]] = []
    n_stamped = 0
    n_hex_kept = 0
    cd = spec.cell_dims
    occupied = spec.iter_occupied_hexes()
    for i, j, k in occupied:
        corners = hex_corners((i, j, k), cd)
        local_n, local_s = rule.builder(corners)
        local_n = np.asarray(local_n, dtype=np.float64).reshape(-1, 3)
        local_s = np.asarray(local_s, dtype=np.int64).reshape(-1, 2)
        n_stamped += len(local_n)
        keep = np.array(
            [_node_hits_filled_octant(hex_uvw(corners, p), spec, i, j, k) for p in local_n],
            dtype=bool,
        )
        if not np.any(keep):
            continue
        n_hex_kept += 1
        old_to_new = -np.ones(len(local_n), dtype=np.int64)
        kept_idx = np.flatnonzero(keep)
        old_to_new[kept_idx] = np.arange(len(kept_idx))
        edges: list[tuple[int, int]] = []
        for a, b in local_s:
            ia, ib = int(old_to_new[int(a)]), int(old_to_new[int(b)])
            if ia < 0 or ib < 0 or ia == ib:
                continue
            edges.append((ia, ib) if ia < ib else (ib, ia))
        kept_s = (
            np.array(sorted(set(edges)), dtype=np.int64)
            if edges
            else np.empty((0, 2), dtype=np.int64)
        )
        graphs.append((local_n[kept_idx], kept_s))
    if not graphs:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {"n_hex_kept": 0, "n_nodes_stamped": n_stamped, "n_nodes_kept": 0},
        )
    nodes, struts = weld_graphs(graphs)
    return nodes, struts, {
        "n_hex_kept": int(n_hex_kept),
        "n_nodes_stamped": int(n_stamped),
        "n_nodes_kept": int(len(nodes)),
        "n_struts_kept": int(len(struts)),
        "mode": "exact_octant_occupancy",
    }


def build_ziggurat_hex_elems(spec: ZigguratSpec) -> np.ndarray:
    n = spec.n_hex
    cd = spec.cell_dims
    return np.stack(
        [hex_corners((i, j, k), cd) for k in range(n) for j in range(n) for i in range(n)],
        axis=0,
    )


def build_ziggurat_cad(spec: ZigguratSpec) -> trimesh.Trimesh:
    """Watertight solid: one box per filled Z half-layer, then CSG union."""
    meshes: list[trimesh.Trimesh] = []
    for c, w in spec.filled_slabs():
        origin, extents = spec.slab_world_box(c, w)
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(origin + 0.5 * extents)
        meshes.append(box)
    if not meshes:
        raise RuntimeError("ziggurat CAD is empty")
    cad = union_solid_meshes(meshes) if len(meshes) > 1 else meshes[0]
    return sanitize_cad_mesh_for_sdf(cad)


def build_unit_cell_grid(spec: ZigguratSpec) -> tuple[np.ndarray, np.ndarray]:
    elems = build_ziggurat_hex_elems(spec)
    return weld_graphs([apply_hex_grid(corners) for corners in elems])


def build_occupied_octant_grid(spec: ZigguratSpec) -> tuple[np.ndarray, np.ndarray]:
    graphs: list[tuple[np.ndarray, np.ndarray]] = []
    h = spec.half_size()
    for c, w in spec.filled_slabs():
        for a in range(1, 1 + w):
            for b in range(1, 1 + w):
                graphs.append(apply_hex_grid(hex_corners((a, b, c), h)))
    return weld_graphs(graphs)


def build_partial_hex_cages(spec: ZigguratSpec) -> tuple[np.ndarray, np.ndarray]:
    """SC grid cage of each occupied hex's filled octants (half/quarter/eighth boxes)."""
    return build_occupied_octant_grid(spec)


def build_background_hex_field(spec: ZigguratSpec) -> tuple[np.ndarray, np.ndarray]:
    """Full unit-hex cages for the whole n_hex³ field (includes empty cells)."""
    return build_unit_cell_grid(spec)
