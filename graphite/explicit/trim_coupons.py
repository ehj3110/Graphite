"""Packed trim-transition coupons: exact octant occupancy, not SDF bleed.

Four models cover the 11-case catalog (T kept, no gap) on isotropic cells:

* A — full/half welds, stairs, and a half–half flush
* B — isolated quarter (no shared face; 2-edge diamond)
* C — full → quarter two-riser, then quarter–quarter flush
* D — half → quarter add-cut, T-junction, and quarter–quarter stair

Axes: +X right, −X left, +Y far, −Y close, +Z top, −Z bottom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from graphite.explicit.geometry_module import union_solid_meshes
from graphite.explicit.hex_rules import apply_hex_grid
from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.ziggurat_trim_cad import (
    kept_planes_for_hex,
    stamp_rule_on_ziggurat,
    weld_graphs,
    hex_corners,
    _as_cell_dims,
)


_UNICODE_MINUS = "\u2212"


def _norm_type(name: str) -> str:
    key = str(name).strip().upper().replace(_UNICODE_MINUS, "-").replace(" ", "")
    aliases = {
        "FULL": "F",
        "Q": "HX-HZ-",
        "QXZ": "HX-HZ-",
        "QZX": "HX-HZ-",
        "QP": "HX-HZ+",
        "QT": "HX-HY-",
        "QXY": "HX-HY-",
    }
    return aliases.get(key, key)


def local_octants_for_type(name: str) -> frozenset[tuple[int, int, int]]:
    """Local (da, db, dc) in {0,1}³ occupied by a named hex type."""
    t = _norm_type(name)
    all8 = frozenset((da, db, dc) for da in (0, 1) for db in (0, 1) for dc in (0, 1))
    if t == "F":
        return all8
    if t == "HX-":
        return frozenset(p for p in all8 if p[0] == 0)
    if t == "HX+":
        return frozenset(p for p in all8 if p[0] == 1)
    if t == "HY-":
        return frozenset(p for p in all8 if p[1] == 0)
    if t == "HY+":
        return frozenset(p for p in all8 if p[1] == 1)
    if t == "HZ-":
        return frozenset(p for p in all8 if p[2] == 0)
    if t == "HZ+":
        return frozenset(p for p in all8 if p[2] == 1)
    signed: list[str] = []
    i = 0
    while i < len(t):
        if t[i : i + 2] in {"HX", "HY", "HZ"} and i + 2 < len(t) and t[i + 2] in "+-":
            signed.append(t[i : i + 3])
            i += 3
        else:
            break
    if len(signed) >= 2 and i == len(t):
        keep = all8
        for code in signed:
            keep = keep & local_octants_for_type(code)
        if not keep:
            raise ValueError(f"empty occupancy for type {name!r}")
        return frozenset(keep)
    raise ValueError(f"unknown hex type {name!r}")


@dataclass(frozen=True)
class HexPlacement:
    ijk: tuple[int, int, int]
    hex_type: str
    note: str = ""

    @property
    def type_key(self) -> str:
        return _norm_type(self.hex_type)


@dataclass(frozen=True)
class CouponSpec:
    """Sparse hex placements with exact filled octants. Duck-types ziggurat occupancy."""

    name: str
    placements: tuple[HexPlacement, ...]
    cell_dims: tuple[float, float, float] = (4.0, 4.0, 4.0)
    title: str = ""
    catalog: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_dims", tuple(float(x) for x in _as_cell_dims(self.cell_dims)))
        seen: set[tuple[int, int, int]] = set()
        for p in self.placements:
            if p.ijk in seen:
                raise ValueError(f"duplicate hex at {p.ijk} in {self.name}")
            seen.add(p.ijk)
            local_octants_for_type(p.hex_type)

    @property
    def n_hex(self) -> int:
        if not self.placements:
            return 1
        return 1 + max(max(int(v) for v in p.ijk) for p in self.placements)

    def _type_at(self, i: int, j: int, k: int) -> str | None:
        key = (int(i), int(j), int(k))
        for p in self.placements:
            if p.ijk == key:
                return p.type_key
        return None

    def octant_filled(self, a: int, b: int, c: int) -> bool:
        i, j, k = int(a) // 2, int(b) // 2, int(c) // 2
        if min(i, j, k, a, b, c) < 0:
            return False
        typ = self._type_at(i, j, k)
        if typ is None:
            return False
        da, db, dc = int(a) % 2, int(b) % 2, int(c) % 2
        return (da, db, dc) in local_octants_for_type(typ)

    def hex_filled_octants(self, i: int, j: int, k: int) -> list[tuple[int, int, int]]:
        typ = self._type_at(i, j, k)
        if typ is None:
            return []
        return sorted(local_octants_for_type(typ))

    def iter_occupied_hexes(self) -> list[tuple[int, int, int]]:
        return [p.ijk for p in self.placements]

    def half_size(self) -> np.ndarray:
        return np.asarray(self.cell_dims, dtype=np.float64) * 0.5

    def filled_octants(self) -> list[tuple[int, int, int]]:
        out: list[tuple[int, int, int]] = []
        for p in self.placements:
            i, j, k = p.ijk
            for da, db, dc in local_octants_for_type(p.hex_type):
                out.append((2 * i + da, 2 * j + db, 2 * k + dc))
        return out


def placement_world_box(
    spec: CouponSpec, placement: HexPlacement
) -> tuple[np.ndarray, np.ndarray]:
    """Origin and extents (mm) of the axis-aligned box for one hex type."""
    h = spec.half_size()
    i, j, k = placement.ijk
    octs = local_octants_for_type(placement.hex_type)
    lo = np.array(
        [min(p[0] for p in octs), min(p[1] for p in octs), min(p[2] for p in octs)],
        dtype=np.float64,
    )
    hi = np.array(
        [max(p[0] for p in octs) + 1, max(p[1] for p in octs) + 1, max(p[2] for p in octs) + 1],
        dtype=np.float64,
    )
    origin = (np.array([2 * i, 2 * j, 2 * k], dtype=np.float64) + lo) * h
    extents = (hi - lo) * h
    return origin, extents


def build_coupon_cad(spec: CouponSpec) -> trimesh.Trimesh:
    """Watertight solid: one box per placed hex type, then CSG union."""
    meshes: list[trimesh.Trimesh] = []
    for p in spec.placements:
        origin, extents = placement_world_box(spec, p)
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(origin + 0.5 * extents)
        meshes.append(box)
    if not meshes:
        raise RuntimeError(f"coupon CAD is empty: {spec.name}")
    cad = union_solid_meshes(meshes) if len(meshes) > 1 else meshes[0]
    return sanitize_cad_mesh_for_sdf(cad)


def build_coupon_hex_elems(spec: CouponSpec) -> np.ndarray:
    cd = spec.cell_dims
    return np.stack([hex_corners(p.ijk, cd) for p in spec.placements], axis=0)


def kept_planes_occupied(spec: CouponSpec) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [kept_planes_for_hex(spec, *p.ijk) for p in spec.placements]


def build_partial_hex_cages(spec: CouponSpec) -> tuple[np.ndarray, np.ndarray]:
    """SC grid cage of each occupied hex's filled octants (half/quarter boxes)."""
    h = spec.half_size()
    graphs = [apply_hex_grid(hex_corners((a, b, c), h)) for a, b, c in spec.filled_octants()]
    return weld_graphs(graphs)


def stamp_coupon(spec: CouponSpec, rule_name: str = "octahedral"):
    return stamp_rule_on_ziggurat(spec, rule_name)


def _cell(i: int, j: int, k: int, typ: str, note: str) -> HexPlacement:
    return HexPlacement((int(i), int(j), int(k)), typ, note)


def model_a_halves(cell_dims=(4.0, 4.0, 4.0)) -> CouponSpec:
    catalog = """\
Model A — halves (one full cell, three half interfaces)
  Axes: +X right, +Y far, +Z top. Cell 4 mm isotropic.

              [HX-]                 (1,0,0)  F|HX- weld on +X
     [F] — [HZ-] — [HZ-] — [HZ+]    along +Y
    (0,0)  (0,1)   (0,2)   (0,3)
              stair    flush     stair

  Cases: F|HX- weld, F|HZ- stair, HZ-|HZ- flush, HZ-|HZ+ half-stair.
"""
    return CouponSpec(
        name="model_a_halves",
        title="Halves: weld, stair, flush, half-stair",
        catalog=catalog,
        cell_dims=tuple(cell_dims),
        placements=(
            _cell(0, 0, 0, "F", "full"),
            _cell(1, 0, 0, "HX-", "F|HX- weld +X"),
            _cell(0, 1, 0, "HZ-", "F|HZ- stair +Y"),
            _cell(0, 2, 0, "HZ-", "HZ-|HZ- flush +Y"),
            _cell(0, 3, 0, "HZ+", "HZ-|HZ+ stair +Y"),
        ),
    )


def model_b_isolated_quarter(cell_dims=(4.0, 4.0, 4.0)) -> CouponSpec:
    catalog = """\
Model B — isolated quarter HX- HZ-
  Must not share a face with any other hex (2-edge diamond test).
  Occupies x in [0, 0.5], z in [0, 0.5], full Y.
"""
    return CouponSpec(
        name="model_b_isolated_quarter",
        title="Isolated quarter HX- HZ-",
        catalog=catalog,
        cell_dims=tuple(cell_dims),
        placements=(_cell(0, 0, 0, "HX-HZ-", "isolated 2-edge diamond"),),
    )


def model_c_full_quarter(cell_dims=(4.0, 4.0, 4.0)) -> CouponSpec:
    catalog = """\
Model C — full → quarter + quarter–quarter flush
     [F] — [Q] — [Q]     along +Y
    (0,0)  (0,1)  (0,2)
  Q = HX- HZ- (two-riser into F, then flush Q|Q bar).
"""
    return CouponSpec(
        name="model_c_full_quarter",
        title="Full to quarter two-riser, then Q|Q flush",
        catalog=catalog,
        cell_dims=tuple(cell_dims),
        placements=(
            _cell(0, 0, 0, "F", "full"),
            _cell(0, 1, 0, "HX-HZ-", "F|Q two-riser +Y"),
            _cell(0, 2, 0, "HX-HZ-", "Q|Q flush +Y"),
        ),
    )


def model_d_half_quarters(cell_dims=(4.0, 4.0, 4.0)) -> CouponSpec:
    catalog = """\
Model D — half → quarters (T kept). Extra-contact-free packing:

         [QT]                 (0,2,0)  HX- HY-   T on +Y of HZ-
     [HZ-] — [Q]              (0,1,0)  HZ-  — (1,1,0) HX- HZ-  add-cut +X
              [Q']            (1,0,0)  HX- HZ+   Q|Q' stair -Y of Q

  QT does not share a Z-cut with HZ- (the T). Q' is opposite Z polarity
  of Q, join perpendicular to that cut (shared Y face, both occupy it).
"""
    return CouponSpec(
        name="model_d_half_quarters",
        title="Half to quarters: add-cut, T, Q|Q stair",
        catalog=catalog,
        cell_dims=tuple(cell_dims),
        placements=(
            _cell(0, 1, 0, "HZ-", "half XY, trimmed top"),
            _cell(1, 1, 0, "HX-HZ-", "HZ-|Q add-cut +X"),
            _cell(0, 2, 0, "HX-HY-", "T: QT on +Y of HZ-"),
            _cell(1, 0, 0, "HX-HZ+", "Q|Q' stair -Y"),
        ),
    )


def all_coupon_models(cell_dims=(4.0, 4.0, 4.0)) -> tuple[CouponSpec, ...]:
    cd = tuple(cell_dims)
    return (
        model_a_halves(cd),
        model_b_isolated_quarter(cd),
        model_c_full_quarter(cd),
        model_d_half_quarters(cd),
    )


def face_neighbors(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int] | None:
    """If a,b share a face, return (axis, shared_world_plane_index). Else None."""
    da = [int(b[i]) - int(a[i]) for i in range(3)]
    if sum(abs(v) for v in da) != 1:
        return None
    axis = next(i for i, v in enumerate(da) if v != 0)
    plane = max(int(a[axis]), int(b[axis]))
    return axis, plane
