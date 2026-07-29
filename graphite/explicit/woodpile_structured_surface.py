"""
Structured surface mesh for extruded woodpile lattices.

Samples the union of axis-aligned bar boxes on a global grid (target edge length ``h``)
and emits boundary faces where solid occupancy changes. The result is a watertight
triangle soup suitable for gmsh ``single_surface`` volume fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import trimesh
from shapely.geometry import Polygon

from graphite.explicit.woodpile_extrude import (
    bar_polygons_single_layer,
    crosshatch_layer_plan,
    strut_axis_for_crosshatch_layer,
    true_woodpile_phase_shift_mm,
)
from graphite.implicit.woodpile_input import WoodpileLatticeSpec
from graphite.math.woodpile_anchor import (
    WoodpileAnchorMode,
    compute_band_orientation,
    compute_woodpile_xy_origin_box,
)


@dataclass(frozen=True)
class BarBox:
    """Axis-aligned strut bounds in millimetres."""

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float

    @classmethod
    def from_polygon_z_span(
        cls, poly: Polygon, *, z0: float, z1: float
    ) -> BarBox | None:
        if poly.is_empty:
            return None
        minx, miny, maxx, maxy = poly.bounds
        if maxx <= minx or maxy <= miny or z1 <= z0:
            return None
        return cls(minx, maxx, miny, maxy, float(z0), float(z1))


def _grid_coords_aligned(
    lo: float, hi: float, edge_length_mm: float, *, origin: float = 0.0
) -> np.ndarray:
    """Axis samples aligned to a global lattice; endpoints ``lo``/``hi`` are always included."""
    lo_f, hi_f, h = float(lo), float(hi), float(edge_length_mm)
    if hi_f <= lo_f + 1e-12:
        return np.array([lo_f], dtype=float)
    if h <= 0.0:
        raise ValueError("edge_length_mm must be positive.")

    k = int(np.floor((lo_f - origin) / h + 1e-9))
    pts: list[float] = []
    while True:
        t = origin + k * h
        if t > hi_f + 1e-9:
            break
        if t >= lo_f - 1e-9:
            pts.append(t)
        k += 1

    if not pts:
        return np.array([lo_f, hi_f], dtype=float)

    if abs(pts[0] - lo_f) > 1e-9:
        pts.insert(0, lo_f)
    if abs(pts[-1] - hi_f) > 1e-9:
        pts.append(hi_f)
    return np.asarray(pts, dtype=float)


def _grid_coords_with_boundaries(
    lo: float,
    hi: float,
    edge_length_mm: float,
    boundary_pts: Iterable[float],
    *,
    origin: float = 0.0,
) -> np.ndarray:
    """Aligned lattice samples plus explicit feature coordinates (bar faces)."""
    coords = set(_grid_coords_aligned(lo, hi, edge_length_mm, origin=origin).tolist())
    lo_f, hi_f = float(lo), float(hi)
    for raw in boundary_pts:
        pt = float(raw)
        if lo_f - 1e-9 <= pt <= hi_f + 1e-9:
            coords.add(pt)
    if not coords:
        return np.array([lo_f, hi_f], dtype=float)
    return np.asarray(sorted(coords), dtype=float)


class _VertexWelder:
    """Merge vertices that coincide within a coordinate tolerance."""

    def __init__(self, tol_mm: float = 1e-6) -> None:
        self._tol = float(tol_mm)
        self._scale = 1.0 / self._tol
        self._verts: list[tuple[float, float, float]] = []
        self._index: dict[tuple[int, int, int], int] = {}

    def add(self, x: float, y: float, z: float) -> int:
        key = (
            int(round(float(x) * self._scale)),
            int(round(float(y) * self._scale)),
            int(round(float(z) * self._scale)),
        )
        idx = self._index.get(key)
        if idx is not None:
            return idx
        idx = len(self._verts)
        self._verts.append((float(x), float(y), float(z)))
        self._index[key] = idx
        return idx

    def vertices(self) -> np.ndarray:
        if not self._verts:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(self._verts, dtype=float)


def _add_quad(
    welder: _VertexWelder,
    faces: list[tuple[int, int, int]],
    corners: Iterable[tuple[float, float, float]],
) -> None:
    c = list(corners)
    if len(c) != 4:
        raise ValueError("quad face requires four corners")
    i0 = welder.add(*c[0])
    i1 = welder.add(*c[1])
    i2 = welder.add(*c[2])
    i3 = welder.add(*c[3])
    faces.append((i0, i1, i2))
    faces.append((i0, i2, i3))


_SOLID_PROBE_EPS_MM = 1e-6


def _point_inside_any_bar(
    point: tuple[float, float, float], bars: list[BarBox]
) -> bool:
    x, y, z = point
    for bar in bars:
        if bar.x0 <= x <= bar.x1 and bar.y0 <= y <= bar.y1 and bar.z0 <= z <= bar.z1:
            return True
    return False


def _emit_union_grid_surface(
    bars: list[BarBox],
    *,
    edge_length_mm: float,
    welder: _VertexWelder,
    faces: list[tuple[int, int, int]],
) -> None:
    """Emit axis-aligned boundary quads on a global grid over the bar union."""
    x0 = min(b.x0 for b in bars)
    x1 = max(b.x1 for b in bars)
    y0 = min(b.y0 for b in bars)
    y1 = max(b.y1 for b in bars)
    z0 = min(b.z0 for b in bars)
    z1 = max(b.z1 for b in bars)

    x_lines = _grid_coords_with_boundaries(
        x0,
        x1,
        edge_length_mm,
        [val for bar in bars for val in (bar.x0, bar.x1)],
    )
    y_lines = _grid_coords_with_boundaries(
        y0,
        y1,
        edge_length_mm,
        [val for bar in bars for val in (bar.y0, bar.y1)],
    )
    z_lines = _grid_coords_with_boundaries(
        z0,
        z1,
        edge_length_mm,
        [val for bar in bars for val in (bar.z0, bar.z1)],
    )
    eps = _SOLID_PROBE_EPS_MM

    for z in z_lines:
        for j in range(len(y_lines) - 1):
            ya, yb = float(y_lines[j]), float(y_lines[j + 1])
            cy = 0.5 * (ya + yb)
            for i in range(len(x_lines) - 1):
                xa, xb = float(x_lines[i]), float(x_lines[i + 1])
                cx = 0.5 * (xa + xb)
                below = _point_inside_any_bar((cx, cy, z - eps), bars)
                above = _point_inside_any_bar((cx, cy, z + eps), bars)
                if below and not above:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (xa, yb, z),
                            (xa, ya, z),
                            (xb, ya, z),
                            (xb, yb, z),
                        ),
                    )
                elif above and not below:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (xa, ya, z),
                            (xa, yb, z),
                            (xb, yb, z),
                            (xb, ya, z),
                        ),
                    )

    for x in x_lines:
        for k in range(len(z_lines) - 1):
            za, zb = float(z_lines[k]), float(z_lines[k + 1])
            cz = 0.5 * (za + zb)
            for j in range(len(y_lines) - 1):
                ya, yb = float(y_lines[j]), float(y_lines[j + 1])
                cy = 0.5 * (ya + yb)
                left = _point_inside_any_bar((x - eps, cy, cz), bars)
                right = _point_inside_any_bar((x + eps, cy, cz), bars)
                if left and not right:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (x, ya, za),
                            (x, ya, zb),
                            (x, yb, zb),
                            (x, yb, za),
                        ),
                    )
                elif right and not left:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (x, ya, za),
                            (x, yb, za),
                            (x, yb, zb),
                            (x, ya, zb),
                        ),
                    )

    for y in y_lines:
        for k in range(len(z_lines) - 1):
            za, zb = float(z_lines[k]), float(z_lines[k + 1])
            cz = 0.5 * (za + zb)
            for i in range(len(x_lines) - 1):
                xa, xb = float(x_lines[i]), float(x_lines[i + 1])
                cx = 0.5 * (xa + xb)
                back = _point_inside_any_bar((cx, y - eps, cz), bars)
                front = _point_inside_any_bar((cx, y + eps, cz), bars)
                if back and not front:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (xa, y, za),
                            (xb, y, za),
                            (xb, y, zb),
                            (xa, y, zb),
                        ),
                    )
                elif front and not back:
                    _add_quad(
                        welder,
                        faces,
                        (
                            (xb, y, za),
                            (xa, y, za),
                            (xa, y, zb),
                            (xb, y, zb),
                        ),
                    )


def structured_surface_mesh_from_bars(
    bars: Iterable[BarBox],
    *,
    edge_length_mm: float,
    weld_tol_mm: float = 1e-6,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Build a uniform surface mesh over a list of axis-aligned bar boxes."""
    h = float(edge_length_mm)
    if h <= 0.0:
        raise ValueError("edge_length_mm must be positive.")

    bar_list = list(bars)
    if not bar_list:
        raise ValueError("structured_surface_mesh_from_bars requires at least one bar.")

    welder = _VertexWelder(tol_mm=weld_tol_mm)
    raw_faces: list[tuple[int, int, int]] = []
    _emit_union_grid_surface(
        bar_list, edge_length_mm=h, welder=welder, faces=raw_faces
    )

    verts_arr = welder.vertices()
    mesh = trimesh.Trimesh(
        vertices=verts_arr, faces=np.asarray(raw_faces, dtype=np.int64)
    )
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.fix_normals()

    edge_lengths = mesh.edges_unique_length
    report: dict[str, Any] = {
        "generator": "woodpile_structured_surface",
        "edge_length_mm": h,
        "n_bars": len(bar_list),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
    }
    if edge_lengths.size:
        report["edge_length_mm_min"] = float(edge_lengths.min())
        report["edge_length_mm_median"] = float(np.median(edge_lengths))
        report["edge_length_mm_max"] = float(edge_lengths.max())
    return mesh, report


def enumerate_woodpile_bar_boxes(spec: WoodpileLatticeSpec) -> list[BarBox]:
    """
    List axis-aligned bar boxes for a piecewise woodpile spec (box domain).

    Cylinder domains are not supported yet because circular clipping produces
    non-rectangular footprints.
    """
    spec.validate()
    if spec.domain != "box":
        raise NotImplementedError(
            "Structured surface mesh is supported for box domains only; "
            f"got domain={spec.domain!r}."
        )

    z_breaks = [float(z) for z in spec.z_breaks_mm]
    pores = [float(p) for p in spec.pore_mm]
    ox, oy, _oz = float(spec.origin_x_mm), float(spec.origin_y_mm), float(spec.origin_z_mm)
    mode: WoodpileAnchorMode = spec.anchor_mode

    wx, wy, _hz = float(spec.width_x_mm), float(spec.depth_y_mm), float(spec.height_mm)
    x_lo, x_hi = ox, ox + wx
    y_lo, y_hi = oy, oy + wy

    bars: list[BarBox] = []
    prev_swap_xy = False
    layer_offset = 0

    for band_idx, pore in enumerate(pores):
        z0_band, z1_band = z_breaks[band_idx], z_breaks[band_idx + 1]
        band_h = z1_band - z0_band
        if band_h <= 0.0:
            raise ValueError(f"Non-positive band height: z0={z0_band}, z1={z1_band}")

        lox, loy, _anchor = compute_woodpile_xy_origin_box(
            width_x_mm=wx,
            depth_y_mm=wy,
            pore_mm=pore,
            mode=mode,
            origin_x_mm=ox,
            origin_y_mm=oy,
        )

        swap_xy, flip, _orient = compute_band_orientation(
            band_idx,
            prev_swap_xy=prev_swap_xy,
            alternate_band_orientation=spec.alternate_band_orientation,
        )
        prev_swap_xy = swap_xy
        band_layer_offset = layer_offset if spec.alternate_band_orientation else 0

        plan = crosshatch_layer_plan(band_h, pore, origin_z_mm=z0_band)
        for idx, z_layer, lh in plan:
            global_idx = band_layer_offset + idx
            axis = strut_axis_for_crosshatch_layer(
                global_idx, flip_layer_parity=flip, swap_xy=swap_xy
            )
            lox_layer, loy_layer = float(lox), float(loy)
            if spec.true_woodpile:
                dx, dy = true_woodpile_phase_shift_mm(global_idx, pore)
                lox_layer += dx
                loy_layer += dy

            polys = bar_polygons_single_layer(
                strut_axis=axis,
                pore_mm=pore,
                x_lo=x_lo,
                x_hi=x_hi,
                y_lo=y_lo,
                y_hi=y_hi,
                origin_x_mm=lox_layer,
                origin_y_mm=loy_layer,
            )

            for poly in polys:
                bar = BarBox.from_polygon_z_span(poly, z0=z_layer, z1=z_layer + lh)
                if bar is not None:
                    bars.append(bar)

        if spec.alternate_band_orientation:
            layer_offset += len(plan)

    return bars


def structured_surface_mesh_from_spec(
    spec: WoodpileLatticeSpec,
    *,
    edge_length_mm: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Structured surface mesh for a :class:`WoodpileLatticeSpec` (extrude backend)."""
    bars = enumerate_woodpile_bar_boxes(spec)
    mesh, report = structured_surface_mesh_from_bars(
        bars, edge_length_mm=float(edge_length_mm)
    )
    report["domain"] = spec.domain
    report["n_layers"] = len({(b.z0, b.z1) for b in bars})
    return mesh, report
