"""
Phase anchoring for woodpile / cross-hatch lattices in bounded cylinders.

Default implicit woodpile anchors strut centers at X=0 and Y=0. On small
diameters this leaves a single central strut (e.g. Ø2 mm with 800 µm pitch).

Piecewise bands should choose per-band XY origins and optionally flip layer
parity so band interfaces do not leave aligned dangling struts.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np

WoodpileAnchorMode = Literal["edge_solid", "center_void", "default"]


def woodpile_strut_center_positions(*, pore_mm: float, origin_mm: float) -> list[float]:
    """Strut center lines along one axis (before cylindrical clip)."""
    pitch = 2.0 * float(pore_mm)
    if pitch <= 0.0:
        return []
    # Centers where mod(x - origin + pore, pitch) == pore  →  x = k * pitch + origin
    return [float(origin_mm) + float(k) * pitch for k in range(-6, 7)]


def strut_reaches_radius(*, radius_mm: float, pore_mm: float, origin_mm: float) -> bool:
    """True if some strut half-width reaches the cylindrical wall at ±radius."""
    r = abs(float(radius_mm))
    p = float(pore_mm)
    pitch = 2.0 * p
    origin = float(origin_mm)
    if pitch <= 0.0:
        return False
    # Nearest strut center to +radius
    k = round((r - origin) / pitch)
    center = origin + k * pitch
    return abs(r - center) <= (p / 2.0) + 1e-6


def is_woodpile_solid_at_axis(
    *,
    coord_mm: float,
    pore_mm: float,
    origin_mm: float,
) -> bool:
    """Solid along one transverse axis (strut runs parallel to the other axis)."""
    pitch = 2.0 * float(pore_mm)
    x_eff = float(coord_mm) - float(origin_mm)
    wave = abs((x_eff + pore_mm) % pitch - pore_mm) - (pore_mm / 2.0)
    return bool(wave <= 0.0)


def compute_woodpile_xy_origin(
    *,
    radius_mm: float,
    pore_mm: float,
    mode: WoodpileAnchorMode = "edge_solid",
) -> tuple[float, float, dict]:
    """
    Choose XY translation for ``evaluate_woodpile(..., origin_x=..., origin_y=...)``.

    edge_solid
        Place a strut outer face on the cylinder wall at x = ±radius (and y = ±radius).
        origin = radius - pore/2 when the part is wide enough.
    center_void
        Shift by one pore so the origin is void — yields symmetric ±struts instead of
        a lone central strut when diameter ≈ 2–3 pitches.
    default
        No shift (legacy behaviour).
    """
    r = float(radius_mm)
    p = float(pore_mm)
    pitch = 2.0 * p
    meta: dict = {"anchor_mode": mode, "radius_mm": r, "pore_mm": p}

    if mode == "default":
        meta["origin_x_mm"] = 0.0
        meta["origin_y_mm"] = 0.0
        meta["note"] = "legacy origin-centered grid"
        return 0.0, 0.0, meta

    if mode == "center_void":
        ox = oy = p
        meta["origin_x_mm"] = ox
        meta["origin_y_mm"] = oy
        meta["center_solid_at_origin"] = is_woodpile_solid_at_axis(
            coord_mm=0.0, pore_mm=p, origin_mm=ox
        )
        return ox, oy, meta

    # edge_solid
    if r < p / 2.0:
        meta["fallback"] = "center_void"
        meta["fallback_reason"] = "radius < pore/2; cannot seat edge strut"
        return compute_woodpile_xy_origin(radius_mm=r, pore_mm=p, mode="center_void")

    ox = oy = r - p / 2.0
    meta["origin_x_mm"] = ox
    meta["origin_y_mm"] = oy
    meta["edge_solid_at_plus_x"] = strut_reaches_radius(
        radius_mm=r, pore_mm=p, origin_mm=ox
    )
    meta["edge_solid_at_minus_x"] = strut_reaches_radius(
        radius_mm=r, pore_mm=p, origin_mm=ox
    )
    meta["center_solid_at_origin"] = is_woodpile_solid_at_axis(coord_mm=0.0, pore_mm=p, origin_mm=ox)
    meta["strut_centers_x"] = woodpile_strut_center_positions(pore_mm=p, origin_mm=ox)
    return ox, oy, meta


def compute_woodpile_xy_origin_box(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    pore_mm: float,
    mode: WoodpileAnchorMode = "center_void",
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> tuple[float, float, dict]:
    """
    XY phase origin for an axis-aligned box ``[ox, ox+width] × [oy, oy+depth]``.

    Cylinder anchoring assumes the lattice center is at X=Y=0. Box domains use
    corner origins (e.g. ``[0, 1]³`` mm); this shifts anchored coordinates to
    the box center before applying ``compute_woodpile_xy_origin``.
    """
    half_wx = float(width_x_mm) / 2.0
    half_wy = float(depth_y_mm) / 2.0
    ox_c, _, meta_x = compute_woodpile_xy_origin(
        radius_mm=half_wx, pore_mm=float(pore_mm), mode=mode
    )
    _, oy_c, meta_y = compute_woodpile_xy_origin(
        radius_mm=half_wy, pore_mm=float(pore_mm), mode=mode
    )
    ox = float(origin_x_mm) + half_wx + ox_c
    oy = float(origin_y_mm) + half_wy + oy_c
    meta = {
        "frame": "axis_aligned_box",
        "anchor_mode": mode,
        "width_x_mm": float(width_x_mm),
        "depth_y_mm": float(depth_y_mm),
        "origin_x_mm": float(origin_x_mm),
        "origin_y_mm": float(origin_y_mm),
        "half_width_x_mm": half_wx,
        "half_width_y_mm": half_wy,
        "centered_origin_x_mm": ox_c,
        "centered_origin_y_mm": oy_c,
        "origin_x_mm_box": ox,
        "origin_y_mm_box": oy,
        "anchor_x": meta_x,
        "anchor_y": meta_y,
    }
    return ox, oy, meta


def should_flip_layer_parity_for_band(
    band_index: int,
    *,
    alternate_band_orientation: bool,
) -> bool:
    """Odd-index bands swap X/Y layer assignment (legacy; prefer ``swap_xy``)."""
    if not alternate_band_orientation:
        return False
    return int(band_index) % 2 == 1


def count_crosshatch_layers_in_height(height_mm: float, pore_mm: float) -> int:
    """Number of Z layers in a band (partial top layer counts as one)."""
    pore = float(pore_mm)
    hz = float(height_mm)
    if pore <= 0.0 or hz <= 0.0:
        return 0
    n = 0
    z = 0.0
    while z < hz - 1e-9:
        z += pore
        n += 1
    return n


def dominant_strut_axis_at_z(
    z_mm: float,
    pore_mm: float,
    *,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
    z_layer_origin_mm: float = 0.0,
    layer_index_offset: int = 0,
) -> str:
    """
    Strut run direction at height ``z_mm`` for one cross-hatch band.

    Returns ``'X'`` (struts run along +X) or ``'Y'`` (along +Y).
  """
    z_local = float(z_mm) - float(z_layer_origin_mm)
    layer = int(layer_index_offset) + int(np.floor(z_local / float(pore_mm) + 1e-9))
    return strut_run_axis_at_layer(
        layer,
        flip_layer_parity=flip_layer_parity,
        swap_xy=swap_xy,
    )


def strut_run_axis_at_layer(
    layer_idx: int,
    *,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
) -> Literal["X", "Y"]:
    """
    Cross-hatch strut run direction for one Z layer index inside a band.

    Even layers → struts along +X; odd → along +Y. ``swap_xy`` rotates the hatch
    90° in-plane (piecewise band toggle).
    """
    even_layer = (int(layer_idx) % 2) == 0
    if flip_layer_parity:
        even_layer = not even_layer
    along_x = even_layer
    if swap_xy:
        along_x = not along_x
    return "X" if along_x else "Y"


def verify_piecewise_interface_layers_perpendicular(
    slabs: Sequence[dict[str, Any]],
    *,
    alternate_band_orientation: bool,
) -> dict[str, Any]:
    """
    QC: at each grade interface, the first layer of the upper band is ⊥ to the
    last layer of the lower band (printability — new struts land on solid).

    Requires ``global_layer_idx`` on each layer dict (extrude) or computes from
  ``layer_index_offset`` + local ``idx``.
    """
    checks: list[dict[str, Any]] = []
    ok = True

    def _layer_axis(slab: dict[str, Any], layer: dict[str, Any]) -> str:
        if "strut_axis" in layer:
            return str(layer["strut_axis"]).upper()
        gidx = int(
            layer.get(
                "global_layer_idx",
                int(slab.get("layer_index_offset", 0)) + int(layer.get("idx", 0)),
            )
        )
        return strut_run_axis_at_layer(
            gidx,
            flip_layer_parity=bool(slab.get("flip_layer_parity", False)),
            swap_xy=bool(slab.get("swap_xy", False)),
        )

    for i in range(len(slabs) - 1):
        lower = slabs[i]
        upper = slabs[i + 1]
        z_if = float(upper.get("z0_mm", 0.0))
        last_layer = lower["layers"][-1]
        first_layer = upper["layers"][0]
        axis_below = _layer_axis(lower, last_layer)
        axis_above = _layer_axis(upper, first_layer)
        perp_ok = axis_below != axis_above
        if alternate_band_orientation and not perp_ok:
            ok = False
        checks.append(
            {
                "z_interface_mm": z_if,
                "lower_band_index": int(lower.get("band_index", i)),
                "upper_band_index": int(upper.get("band_index", i + 1)),
                "axis_below": axis_below,
                "axis_above": axis_above,
                "perpendicular": perp_ok,
                "lower_global_layer_idx": int(
                    last_layer.get(
                        "global_layer_idx",
                        int(lower.get("layer_index_offset", 0))
                        + int(last_layer.get("idx", 0)),
                    )
                ),
                "upper_global_layer_idx": int(
                    first_layer.get(
                        "global_layer_idx",
                        int(upper.get("layer_index_offset", 0))
                        + int(first_layer.get("idx", 0)),
                    )
                ),
            }
        )

    return {
        "ok": ok,
        "alternate_band_orientation": bool(alternate_band_orientation),
        "checks": checks,
    }


def verify_piecewise_band_hatch_rotation(
    slabs: Sequence[dict[str, Any]],
    *,
    alternate_band_orientation: bool,
) -> dict[str, Any]:
    """Deprecated alias — use ``verify_piecewise_interface_layers_perpendicular``."""
    return verify_piecewise_interface_layers_perpendicular(
        slabs,
        alternate_band_orientation=alternate_band_orientation,
    )


def compute_band_orientation(
    band_index: int,
    *,
    prev_swap_xy: bool,
    alternate_band_orientation: bool,
) -> tuple[bool, bool, dict]:
    """
    Choose ``(swap_xy, flip_layer_parity)`` for a piecewise band.

    When ``alternate_band_orientation`` is True, piecewise stacks use **global
    layer-index continuity** across grade interfaces (see
    ``evaluate_woodpile(..., layer_index_offset=...)``): the first Z layer of a
    new grade is always ⊥ to the last layer of the previous grade for
    printability. Per-band ``center_void`` origins supply XY phase shift.

    ``swap_xy`` is no longer toggled per band — it broke interface ⊥ when layer
    parity restarted at each ``z0``.
    """
    meta: dict = {
        "band_index": int(band_index),
        "orientation_mode": (
            "global_layer_continuity"
            if alternate_band_orientation
            else "local_layer_restart"
        ),
        "swap_xy": False,
        "flip_layer_parity": False,
    }
    if alternate_band_orientation and band_index > 0:
        meta["prev_swap_xy"] = bool(prev_swap_xy)
    return False, False, meta
