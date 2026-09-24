"""Derive per-axis node planes for SC hex topology rules.

The wrist-rest / quantized extent-trim policy should be driven by where
nodes actually sit in the unit cell, not by hand-written half-octahedra.

Example (octahedral face centers): planes {0, 0.5, 1} → mid thresholds
{0.25, 0.75}, which is exactly the historical 25/75 tiered cull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphite.explicit.hex_topology_module import (
    HexTopologyRule,
    get_hex_topology_rule,
)

# Standard SC hex corner order used by hex builders (unit cube [0, 1]^3).
_UNIT_HEX_CORNERS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class NodePlanePolicy:
    """Axis-aligned node fractions and midplane decision thresholds."""

    rule_name: str
    planes_xyz: tuple[np.ndarray, np.ndarray, np.ndarray]
    thresholds_xyz: tuple[np.ndarray, np.ndarray, np.ndarray]
    n_nodes: int

    def as_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "n_nodes_unit_cell": self.n_nodes,
            "planes": {
                "x": self.planes_xyz[0].tolist(),
                "y": self.planes_xyz[1].tolist(),
                "z": self.planes_xyz[2].tolist(),
            },
            "mid_thresholds": {
                "x": self.thresholds_xyz[0].tolist(),
                "y": self.thresholds_xyz[1].tolist(),
                "z": self.thresholds_xyz[2].tolist(),
            },
        }


def _unique_sorted_fractions(values: np.ndarray, *, decimals: int = 6) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return np.zeros(0, dtype=np.float64)
    # Clamp tiny numeric drift outside [0, 1] from builders.
    v = np.clip(v, -1e-9, 1.0 + 1e-9)
    rounded = np.round(v, int(decimals))
    uniq = np.unique(rounded)
    return uniq[(uniq >= -1e-9) & (uniq <= 1.0 + 1e-9)]


def mid_thresholds(planes: np.ndarray) -> np.ndarray:
    """Decision planes halfway between consecutive node fractions."""
    p = np.asarray(planes, dtype=np.float64).reshape(-1)
    if p.size < 2:
        return np.zeros(0, dtype=np.float64)
    return 0.5 * (p[:-1] + p[1:])


def high_plane_index(hi: float, planes: np.ndarray, thresholds: np.ndarray) -> int:
    """
    Outermost plane index supported by material max ``hi`` (from 0 toward 1).

    Recovers octahedral 25/75: hi < 0.25 → -1; 0.25..0.75 → mid (1); > 0.75 → outer.
    """
    p = np.asarray(planes, dtype=np.float64).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return -1
    if t.size == 0:
        return 0 if float(hi) >= float(p[0]) - 1e-12 else -1
    if float(hi) < float(t[0]):
        # Did not reach the first interior cut; keep the last plane at or below hi
        # (octahedral: hi < 0.25 keeps plane 0, not an empty set).
        for i in range(int(p.size) - 1, -1, -1):
            if float(p[i]) <= float(hi) + 1e-12:
                return i
        return -1
    for j in range(int(t.size) - 1):
        if float(hi) <= float(t[j + 1]):
            return j + 1
    return int(p.size) - 1


def low_plane_index(lo: float, planes: np.ndarray, thresholds: np.ndarray) -> int:
    """Outermost plane index supported by material min ``lo`` (from 1 toward 0)."""
    p = np.asarray(planes, dtype=np.float64).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    n = int(p.size) - 1
    if p.size == 0:
        return 0
    if t.size == 0:
        return n if float(lo) <= float(p[-1]) + 1e-12 else n + 1
    mirrored = high_plane_index(1.0 - float(lo), p, t)
    if mirrored < 0:
        return n + 1
    return n - mirrored


def kept_planes_for_interval(
    lo: float,
    hi: float,
    planes: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Node-plane fractions kept on one axis for material interval [lo, hi]."""
    p = np.asarray(planes, dtype=np.float64).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if p.size == 0 or float(hi) < float(lo) - 1e-12:
        return np.zeros(0, dtype=np.float64)
    i0 = low_plane_index(lo, p, t)
    i1 = high_plane_index(hi, p, t)
    if i0 > i1 or i0 < 0 or i1 >= p.size:
        return np.zeros(0, dtype=np.float64)
    return p[i0 : i1 + 1].copy()


def interval_indices_for_frac(frac: float, planes: np.ndarray, *, atol: float = 1e-9) -> tuple[int, ...]:
    """Box indices along one axis that contain ``frac`` (inclusive on plane hits)."""
    p = np.asarray(planes, dtype=np.float64).reshape(-1)
    u = float(frac)
    ids: list[int] = []
    for i in range(int(p.size) - 1):
        if float(p[i]) - atol <= u <= float(p[i + 1]) + atol:
            ids.append(i)
    return tuple(ids)


def incident_box_indices(
    uvw: np.ndarray,
    planes_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    atol: float = 1e-5,
) -> tuple[tuple[int, int, int], ...]:
    """
    Node-plane boxes incident to a node at unit-cell ``uvw``.

    Snaps each coordinate to the nearest plane, then takes adjacent intervals
    (one at an outer plane, two at an interior plane).
    """
    from itertools import product

    per_axis: list[tuple[int, ...]] = []
    pt = np.asarray(uvw, dtype=np.float64).reshape(3)
    for a in range(3):
        p = np.asarray(planes_xyz[a], dtype=np.float64).reshape(-1)
        if p.size < 2:
            per_axis.append(tuple())
            continue
        k = int(np.argmin(np.abs(p - float(pt[a]))))
        if abs(float(p[k]) - float(pt[a])) > atol:
            ids = interval_indices_for_frac(float(pt[a]), p, atol=atol)
        else:
            ids_list: list[int] = []
            if k > 0:
                ids_list.append(k - 1)
            if k < int(p.size) - 1:
                ids_list.append(k)
            ids = tuple(ids_list)
        per_axis.append(ids)
    if any(len(ids) == 0 for ids in per_axis):
        return tuple()
    return tuple((int(i), int(j), int(k)) for i, j, k in product(*per_axis))


def unit_cell_nodes(rule: HexTopologyRule | str) -> np.ndarray:
    """Stamp one rule into the unit cube; return node XYZ in [0, 1]^3."""
    if isinstance(rule, str):
        rule = get_hex_topology_rule(rule)
    nodes, _struts = rule.builder(_UNIT_HEX_CORNERS.copy())
    return np.asarray(nodes, dtype=np.float64).reshape(-1, 3)


def node_plane_policy(rule_name: str, *, decimals: int = 6) -> NodePlanePolicy:
    """
    Derive per-axis node planes and mid thresholds for ``rule_name``.

    Material extent along an axis is compared to mid thresholds to decide
    which node planes remain (generalization of octahedral 25/75 trim).
    """
    nodes = unit_cell_nodes(rule_name)
    planes = tuple(_unique_sorted_fractions(nodes[:, a], decimals=decimals) for a in range(3))
    thresholds = tuple(mid_thresholds(p) for p in planes)
    return NodePlanePolicy(
        rule_name=str(rule_name),
        planes_xyz=planes,  # type: ignore[arg-type]
        thresholds_xyz=thresholds,  # type: ignore[arg-type]
        n_nodes=int(len(nodes)),
    )


def trim_plane_policy(rule_name: str, *, decimals: int = 6) -> NodePlanePolicy:
    """
    Planes used for trim *decisions* (may be coarser than stamped nodes).

    Kelvin drops 0.25/0.75 as cut planes (octahedral 0 / 0.5 / 1).
    Tesseract still lists 0.25/0.75 for node matching, but interval keep is
    special-cased in ``kept_trim_planes``.
    """
    rule = str(rule_name).strip().lower()
    base = node_plane_policy(rule, decimals=decimals)
    if rule == "kelvin":
        p = _KELVIN_TRIM_PLANES
        t = _KELVIN_TRIM_THRESHOLDS
        return NodePlanePolicy(
            rule_name=rule,
            planes_xyz=(p, p, p),
            thresholds_xyz=(t, t, t),
            n_nodes=base.n_nodes,
        )
    return base


def tesseract_high_index(hi: float) -> int:
    """
    0 .. <0.125 → plane 0; 0.125..0.75 → plane 0.25; >0.75 → full (plane 1).
    """
    p = _TESSERACT_PLANES
    h = float(hi)
    if h < _TESSERACT_FIRST_CUT:
        for i in range(int(p.size) - 1, -1, -1):
            if float(p[i]) <= h + 1e-12:
                return i
        return -1
    if h <= _TESSERACT_FULL_MIN:
        return 1
    return int(p.size) - 1


def tesseract_low_index(lo: float) -> int:
    n = int(_TESSERACT_PLANES.size) - 1
    mirrored = tesseract_high_index(1.0 - float(lo))
    if mirrored < 0:
        return n + 1
    return n - mirrored


def kept_trim_planes(
    lo: float,
    hi: float,
    rule_name: str,
    *,
    decimals: int = 6,
) -> np.ndarray:
    """Kept trim-plane fractions on one axis for material interval [lo, hi]."""
    rule = str(rule_name).strip().lower()
    if rule == "tesseract":
        p = _TESSERACT_PLANES
        if float(hi) < float(lo) - 1e-12:
            return np.zeros(0, dtype=np.float64)
        i0 = tesseract_low_index(lo)
        i1 = tesseract_high_index(hi)
        if i0 > i1 or i0 < 0 or i1 >= p.size:
            return np.zeros(0, dtype=np.float64)
        return p[i0 : i1 + 1].copy()
    pol = trim_plane_policy(rule, decimals=decimals)
    axis = pol.planes_xyz[0]
    thr = pol.thresholds_xyz[0]
    return kept_planes_for_interval(lo, hi, axis, thr)


def frac_on_kept_trim_planes(
    frac: float,
    kept: np.ndarray,
    rule_name: str,
    *,
    atol: float = 1e-5,
) -> bool:
    """
    Whether a stamped node fraction is allowed given kept *trim* planes.

    Kelvin 0.25 lives in the [0, 0.5] block (needs both 0 and 0.5 kept);
    0.75 needs both 0.5 and 1.
    """
    if kept.size == 0:
        return False
    rule = str(rule_name).strip().lower()
    f = float(frac)

    def _has(val: float) -> bool:
        return bool(np.any(np.abs(kept - val) <= atol))

    if rule == "kelvin":
        if abs(f - 0.25) <= atol:
            return _has(0.0) and _has(0.5)
        if abs(f - 0.75) <= atol:
            return _has(0.5) and _has(1.0)
    return _has(f)


# Kelvin: nodes exist at 0.25/0.75, but trim decisions use octahedral planes only.
_KELVIN_TRIM_PLANES = np.array([0.0, 0.5, 1.0], dtype=np.float64)
_KELVIN_TRIM_THRESHOLDS = np.array([0.25, 0.75], dtype=np.float64)

# Tesseract: never stop at 0.75 as a half-cut. 0.25 is the only inner shell;
# once material passes 0.75 the whole cell is kept.
_TESSERACT_PLANES = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float64)
_TESSERACT_FIRST_CUT = 0.125
_TESSERACT_FULL_MIN = 0.75
WRIST_REST_SC_MATRIX_RULES: tuple[str, ...] = (
    "grid",
    "octahedral",
    "star",
    "octet",
    "kelvin",
    "tesseract",
    "hex_face_dual",
)
