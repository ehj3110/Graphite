"""
2D bar-grid extrusion for cross-hatch / woodpile lattices.

Builds axis-aligned rectangular struts by unioning bar polygons in the
transverse plane and extruding along Z. Matches implicit woodpile conventions:
pitch = 2 × pore, layer thickness = pore, strut width = pore.

See ``docs/coasters/2d_lattice_extrusion_documentation.md`` for the related
coaster workflow; this module targets piecewise woodpile/cross-hatch parts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from graphite.explicit.geometry_module import _manifold_to_trimesh, _trimesh_to_manifold
from graphite.math.woodpile_anchor import strut_run_axis_at_layer

StrutAxis = Literal["x", "y"]


def strut_centers_in_extent(
    extent_lo: float,
    extent_hi: float,
    *,
    pore_mm: float,
    origin_mm: float = 0.0,
) -> list[float]:
    """Strut center coordinates along one transverse axis inside ``[extent_lo, extent_hi]``."""
    pitch = 2.0 * float(pore_mm)
    half = float(pore_mm) / 2.0
    origin = float(origin_mm)
    lo = float(extent_lo)
    hi = float(extent_hi)
    if pitch <= 0.0 or hi <= lo:
        return []

    k_start = int(np.floor((lo + half - origin) / pitch)) - 1
    centers: list[float] = []
    k = k_start
    while True:
        center = origin + k * pitch
        if center - half > hi:
            break
        if center + half >= lo:
            centers.append(center)
        k += 1
    return centers


def bar_polygons_single_layer(
    *,
    strut_axis: StrutAxis,
    pore_mm: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
) -> list[Polygon]:
    """
    Axis-aligned bar rectangles in the XY plane for one Z layer.

    ``strut_axis='x'``: bars run along +X (fixed Y centers).
    ``strut_axis='y'``: bars run along +Y (fixed X centers).
    """
    half = float(pore_mm) / 2.0
    polys: list[Polygon] = []

    if strut_axis == "x":
        for cy in strut_centers_in_extent(
            y_lo, y_hi, pore_mm=pore_mm, origin_mm=origin_y_mm
        ):
            y0 = max(y_lo, cy - half)
            y1 = min(y_hi, cy + half)
            if y1 > y0:
                polys.append(box(x_lo, y0, x_hi, y1))
    elif strut_axis == "y":
        for cx in strut_centers_in_extent(
            x_lo, x_hi, pore_mm=pore_mm, origin_mm=origin_x_mm
        ):
            x0 = max(x_lo, cx - half)
            x1 = min(x_hi, cx + half)
            if x1 > x0:
                polys.append(box(x0, y_lo, x1, y_hi))
    else:
        raise ValueError(f"strut_axis must be 'x' or 'y', got {strut_axis!r}")
    return polys


def union_bar_polygons(polys: list[Polygon]) -> Polygon | MultiPolygon:
    if not polys:
        return Polygon()
    return unary_union(polys)


def extrude_planar_footprint(
    footprint: Polygon | MultiPolygon,
    *,
    z0_mm: float,
    height_mm: float,
) -> trimesh.Trimesh:
    """Extrude a 2D footprint from ``z0_mm`` to ``z0_mm + height_mm``."""
    if footprint.is_empty:
        raise ValueError("Cannot extrude empty footprint.")

    meshes: list[trimesh.Trimesh] = []
    if isinstance(footprint, MultiPolygon):
        parts = list(footprint.geoms)
    else:
        parts = [footprint]

    for part in parts:
        if part.is_empty or part.area <= 0.0:
            continue
        path = trimesh.load_path(part)
        ext = path.extrude(float(height_mm))
        if isinstance(ext, (list, tuple)):
            slab = trimesh.util.concatenate(ext)
        elif hasattr(ext, "geometry") and isinstance(ext.geometry, dict):
            slab = trimesh.util.concatenate(list(ext.geometry.values()))
        else:
            slab = ext
        slab.apply_translation([0.0, 0.0, float(z0_mm)])
        meshes.append(slab)

    if not meshes:
        raise ValueError("Extrusion produced no geometry.")
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def generate_single_layer_box(
    *,
    strut_axis: StrutAxis,
    pore_mm: float,
    width_x_mm: float = 1.0,
    depth_y_mm: float = 1.0,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    origin_z_mm: float = 0.0,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """
    One woodpile layer in an axis-aligned box (Phase 0 spike).

    Layer height equals ``pore_mm``; footprint is ``[ox, ox+width] × [oy, oy+depth]``.
    """
    ox, oy, oz = float(origin_x_mm), float(origin_y_mm), float(origin_z_mm)
    wx, wy = float(width_x_mm), float(depth_y_mm)
    pore = float(pore_mm)
    if pore <= 0.0 or wx <= 0.0 or wy <= 0.0:
        raise ValueError("pore_mm, width_x_mm, and depth_y_mm must be positive.")

    polys = bar_polygons_single_layer(
        strut_axis=strut_axis,
        pore_mm=pore,
        x_lo=ox,
        x_hi=ox + wx,
        y_lo=oy,
        y_hi=oy + wy,
        origin_x_mm=ox,
        origin_y_mm=oy,
    )
    footprint = union_bar_polygons(polys)
    mesh = extrude_planar_footprint(footprint, z0_mm=oz, height_mm=pore)

    transverse_centers = (
        strut_centers_in_extent(oy, oy + wy, pore_mm=pore, origin_mm=oy)
        if strut_axis == "x"
        else strut_centers_in_extent(ox, ox + wx, pore_mm=pore, origin_mm=ox)
    )
    report: dict[str, Any] = {
        "generator": "woodpile_extrude",
        "strut_axis": strut_axis,
        "pore_mm": pore,
        "pitch_mm": 2.0 * pore,
        "layer_height_mm": pore,
        "box_mm": [wx, wy, pore],
        "origin_mm": [ox, oy, oz],
        "n_bars": len(polys),
        "transverse_centers_mm": transverse_centers,
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
    }
    return mesh, report


def write_mesh_report(
    mesh: trimesh.Trimesh,
    stl_path: Path,
    report: dict[str, Any],
) -> Path:
    """Export STL and sidecar JSON report."""
    stl_path = Path(stl_path)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)
    report = dict(report)
    report["stl"] = str(stl_path.resolve())
    json_path = stl_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return json_path


def true_woodpile_phase_shift_mm(layer_idx: int, pore_mm: float) -> tuple[float, float]:
    """XY lattice shift for true woodpile layers (matches ``evaluate_woodpile``)."""
    pore = float(pore_mm)
    idx = int(layer_idx)
    dx = pore if (idx % 4) == 3 else 0.0
    dy = pore if (idx % 4) == 2 else 0.0
    return dx, dy


def strut_axis_for_crosshatch_layer(
    layer_idx: int,
    *,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
) -> StrutAxis:
    """
    Cross-hatch layer orientation (``true_woodpile=False`` implicit convention).

    Even Z layers → struts along +X; odd layers → struts along +Y.
    Delegates to ``strut_run_axis_at_layer`` in ``woodpile_anchor``.
    """
    run = strut_run_axis_at_layer(
        layer_idx,
        flip_layer_parity=flip_layer_parity,
        swap_xy=swap_xy,
    )
    return "x" if run == "X" else "y"


def crosshatch_layer_plan(
    height_mm: float,
    pore_mm: float,
    *,
    origin_z_mm: float = 0.0,
) -> list[tuple[int, float, float]]:
    """``(layer_idx, z0_mm, layer_height_mm)`` from floor to domain top."""
    pore = float(pore_mm)
    hz = float(height_mm)
    oz = float(origin_z_mm)
    if pore <= 0.0 or hz <= 0.0:
        return []
    layers: list[tuple[int, float, float]] = []
    z = oz
    idx = 0
    top = oz + hz
    while z < top - 1e-9:
        layer_h = min(pore, top - z)
        layers.append((idx, z, layer_h))
        z += pore
        idx += 1
    return layers


def clip_footprint_to_circle(
    footprint: Polygon | MultiPolygon,
    *,
    center_x_mm: float,
    center_y_mm: float,
    radius_mm: float,
    circle_resolution: int = 96,
) -> Polygon | MultiPolygon:
    disk = Point(float(center_x_mm), float(center_y_mm)).buffer(
        float(radius_mm), resolution=int(circle_resolution)
    )
    clipped = footprint.intersection(disk)
    if clipped.is_empty:
        return Polygon()
    return clipped


def _layer_mesh_box(
    *,
    layer_idx: int,
    z0_mm: float,
    layer_height_mm: float,
    pore_mm: float,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    lattice_origin_x_mm: float,
    lattice_origin_y_mm: float,
    flip_layer_parity: bool,
    swap_xy: bool,
    clip_circle: tuple[float, float, float] | None,
    true_woodpile: bool = False,
) -> trimesh.Trimesh:
    axis = strut_axis_for_crosshatch_layer(
        layer_idx, flip_layer_parity=flip_layer_parity, swap_xy=swap_xy
    )
    lox = float(lattice_origin_x_mm)
    loy = float(lattice_origin_y_mm)
    if true_woodpile:
        dx, dy = true_woodpile_phase_shift_mm(layer_idx, pore_mm)
        lox += dx
        loy += dy
    polys = bar_polygons_single_layer(
        strut_axis=axis,
        pore_mm=pore_mm,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        origin_x_mm=lox,
        origin_y_mm=loy,
    )
    footprint = union_bar_polygons(polys)
    if clip_circle is not None:
        cx, cy, r = clip_circle
        footprint = clip_footprint_to_circle(
            footprint, center_x_mm=cx, center_y_mm=cy, radius_mm=r
        )
    return extrude_planar_footprint(footprint, z0_mm=z0_mm, height_mm=layer_height_mm)


def union_trimesh_meshes(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Manifold3d boolean union of watertight extrusion slabs."""
    if not meshes:
        raise ValueError("union_trimesh_meshes requires at least one mesh.")
    if len(meshes) == 1:
        return meshes[0]
    acc = _trimesh_to_manifold(meshes[0])
    for mesh in meshes[1:]:
        acc = acc + _trimesh_to_manifold(mesh)
    return _manifold_to_trimesh(acc)


def generate_crosshatch_box(
    *,
    pore_mm: float,
    width_x_mm: float = 1.0,
    depth_y_mm: float = 1.0,
    height_mm: float = 1.0,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    origin_z_mm: float = 0.0,
    lattice_origin_x_mm: float | None = None,
    lattice_origin_y_mm: float | None = None,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
    true_woodpile: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Full cross-hatch stack in an axis-aligned box (uniform pore)."""
    ox, oy, oz = float(origin_x_mm), float(origin_y_mm), float(origin_z_mm)
    lox = float(lattice_origin_x_mm) if lattice_origin_x_mm is not None else ox
    loy = float(lattice_origin_y_mm) if lattice_origin_y_mm is not None else oy
    wx, wy, hz = float(width_x_mm), float(depth_y_mm), float(height_mm)
    pore = float(pore_mm)
    plan = crosshatch_layer_plan(hz, pore, origin_z_mm=oz)
    if not plan:
        raise ValueError("Empty cross-hatch layer plan.")

    layer_meshes = [
        _layer_mesh_box(
            layer_idx=idx,
            z0_mm=z0,
            layer_height_mm=lh,
            pore_mm=pore,
            x_lo=ox,
            x_hi=ox + wx,
            y_lo=oy,
            y_hi=oy + wy,
            lattice_origin_x_mm=lox,
            lattice_origin_y_mm=loy,
            flip_layer_parity=flip_layer_parity,
            swap_xy=swap_xy,
            clip_circle=None,
            true_woodpile=true_woodpile,
        )
        for idx, z0, lh in plan
    ]
    mesh = union_trimesh_meshes(layer_meshes)
    layer_axes = [
        strut_axis_for_crosshatch_layer(
            idx, flip_layer_parity=flip_layer_parity, swap_xy=swap_xy
        )
        for idx, _, _ in plan
    ]
    report: dict[str, Any] = {
        "generator": "woodpile_extrude",
        "mode": "crosshatch",
        "true_woodpile": bool(true_woodpile),
        "pore_mm": pore,
        "pitch_mm": 2.0 * pore,
        "domain_mm": [wx, wy, hz],
        "origin_mm": [ox, oy, oz],
        "lattice_origin_mm": [lox, loy],
        "n_layers": len(plan),
        "layer_axes": layer_axes,
        "layer_plan": [{"idx": i, "z0_mm": z, "height_mm": h} for i, z, h in plan],
        "flip_layer_parity": bool(flip_layer_parity),
        "swap_xy": bool(swap_xy),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
    }
    return mesh, report


def generate_crosshatch_cylinder(
    *,
    pore_mm: float,
    radius_mm: float,
    height_mm: float,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    origin_z_mm: float = 0.0,
    lattice_origin_x_mm: float | None = None,
    lattice_origin_y_mm: float | None = None,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
    true_woodpile: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Full cross-hatch stack clipped to a right circular cylinder in XY."""
    r = float(radius_mm)
    hz = float(height_mm)
    pore = float(pore_mm)
    ox, oy, oz = float(origin_x_mm), float(origin_y_mm), float(origin_z_mm)
    lox = float(lattice_origin_x_mm) if lattice_origin_x_mm is not None else ox
    loy = float(lattice_origin_y_mm) if lattice_origin_y_mm is not None else oy
    plan = crosshatch_layer_plan(hz, pore, origin_z_mm=oz)
    x_lo, x_hi = ox - r, ox + r
    y_lo, y_hi = oy - r, oy + r
    clip = (ox, oy, r)

    layer_meshes = [
        _layer_mesh_box(
            layer_idx=idx,
            z0_mm=z0,
            layer_height_mm=lh,
            pore_mm=pore,
            x_lo=x_lo,
            x_hi=x_hi,
            y_lo=y_lo,
            y_hi=y_hi,
            lattice_origin_x_mm=lox,
            lattice_origin_y_mm=loy,
            flip_layer_parity=flip_layer_parity,
            swap_xy=swap_xy,
            clip_circle=clip,
            true_woodpile=true_woodpile,
        )
        for idx, z0, lh in plan
    ]
    mesh = union_trimesh_meshes(layer_meshes)
    layer_axes = [
        strut_axis_for_crosshatch_layer(
            idx, flip_layer_parity=flip_layer_parity, swap_xy=swap_xy
        )
        for idx, _, _ in plan
    ]
    report: dict[str, Any] = {
        "generator": "woodpile_extrude",
        "mode": "crosshatch",
        "true_woodpile": bool(true_woodpile),
        "pore_mm": pore,
        "pitch_mm": 2.0 * pore,
        "radius_mm": r,
        "height_mm": hz,
        "origin_mm": [ox, oy, oz],
        "lattice_origin_mm": [lox, loy],
        "n_layers": len(plan),
        "layer_axes": layer_axes,
        "layer_plan": [{"idx": i, "z0_mm": z, "height_mm": h} for i, z, h in plan],
        "flip_layer_parity": bool(flip_layer_parity),
        "swap_xy": bool(swap_xy),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
    }
    return mesh, report


def _piecewise_crosshatch_stack(
    *,
    z_breaks_mm: list[float],
    pore_mm: list[float],
    origin_x_mm: float,
    origin_y_mm: float,
    origin_z_mm: float,
    anchor_mode: str,
    alternate_band_orientation: bool,
    true_woodpile: bool,
    domain: Literal["box", "cylinder"],
    width_x_mm: float,
    depth_y_mm: float,
    height_mm: float,
    radius_mm: float | None,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Shared piecewise cross-hatch builder for box and cylinder domains."""
    from graphite.math.woodpile_anchor import (
        WoodpileAnchorMode,
        compute_band_orientation,
        compute_woodpile_xy_origin,
        compute_woodpile_xy_origin_box,
        verify_piecewise_interface_layers_perpendicular,
    )

    z_breaks = [float(z) for z in z_breaks_mm]
    pores = [float(p) for p in pore_mm]
    if len(z_breaks) != len(pores) + 1:
        raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")

    ox, oy, oz = float(origin_x_mm), float(origin_y_mm), float(origin_z_mm)
    mode: WoodpileAnchorMode = anchor_mode  # type: ignore[assignment]

    if domain == "box":
        wx, wy, hz = float(width_x_mm), float(depth_y_mm), float(height_mm)
        x_lo, x_hi = ox, ox + wx
        y_lo, y_hi = oy, oy + wy
        clip_circle = None
    elif domain == "cylinder":
        if radius_mm is None or radius_mm <= 0.0:
            raise ValueError("radius_mm must be positive for cylinder domain.")
        r = float(radius_mm)
        hz = float(height_mm)
        x_lo, x_hi = ox - r, ox + r
        y_lo, y_hi = oy - r, oy + r
        clip_circle = (ox, oy, r)
    else:
        raise ValueError(f"domain must be 'box' or 'cylinder', got {domain!r}")

    layer_meshes: list[trimesh.Trimesh] = []
    slab_metas: list[dict[str, Any]] = []
    prev_swap_xy = False
    layer_offset = 0
    for band_idx, pore in enumerate(pores):
        z0, z1 = z_breaks[band_idx], z_breaks[band_idx + 1]
        band_h = z1 - z0
        if band_h <= 0.0:
            raise ValueError(f"Non-positive band height: z0={z0}, z1={z1}")

        if domain == "box":
            lox, loy, anchor_meta = compute_woodpile_xy_origin_box(
                width_x_mm=wx,
                depth_y_mm=wy,
                pore_mm=pore,
                mode=mode,
                origin_x_mm=ox,
                origin_y_mm=oy,
            )
        else:
            lox, loy, anchor_meta = compute_woodpile_xy_origin(
                radius_mm=r,
                pore_mm=pore,
                mode=mode,
            )

        swap_xy, flip, orient_meta = compute_band_orientation(
            band_idx,
            prev_swap_xy=prev_swap_xy,
            alternate_band_orientation=alternate_band_orientation,
        )
        prev_swap_xy = swap_xy
        band_layer_offset = layer_offset if alternate_band_orientation else 0

        plan = crosshatch_layer_plan(band_h, pore, origin_z_mm=z0)
        band_layers: list[dict[str, Any]] = []
        for idx, z_layer, lh in plan:
            global_idx = band_layer_offset + idx
            layer_meshes.append(
                _layer_mesh_box(
                    layer_idx=global_idx,
                    z0_mm=z_layer,
                    layer_height_mm=lh,
                    pore_mm=pore,
                    x_lo=x_lo,
                    x_hi=x_hi,
                    y_lo=y_lo,
                    y_hi=y_hi,
                    lattice_origin_x_mm=lox,
                    lattice_origin_y_mm=loy,
                    flip_layer_parity=flip,
                    swap_xy=swap_xy,
                    clip_circle=clip_circle,
                    true_woodpile=true_woodpile,
                )
            )
            band_layers.append(
                {
                    "idx": idx,
                    "global_layer_idx": global_idx,
                    "z0_mm": z_layer,
                    "height_mm": lh,
                    "strut_axis": strut_axis_for_crosshatch_layer(
                        global_idx, flip_layer_parity=flip, swap_xy=swap_xy
                    ),
                    "phase_shift_mm": list(
                        true_woodpile_phase_shift_mm(global_idx, pore)
                    )
                    if true_woodpile
                    else [0.0, 0.0],
                }
            )

        slab_metas.append(
            {
                "band_index": band_idx,
                "z0_mm": z0,
                "z1_mm": z1,
                "pore_mm": pore,
                "origin_x_mm": lox,
                "origin_y_mm": loy,
                "swap_xy": bool(swap_xy),
                "flip_layer_parity": bool(flip),
                "z_layer_origin_mm": z0,
                "layer_index_offset": band_layer_offset,
                "n_layers": len(plan),
                "anchor": anchor_meta,
                "orientation": orient_meta,
                "layers": band_layers,
            }
        )
        if alternate_band_orientation:
            layer_offset += len(plan)

    mesh = union_trimesh_meshes(layer_meshes)
    rotation_qc = verify_piecewise_interface_layers_perpendicular(
        slab_metas,
        alternate_band_orientation=alternate_band_orientation,
    )
    report: dict[str, Any] = {
        "generator": "woodpile_extrude",
        "mode": "piecewise_crosshatch",
        "domain": domain,
        "true_woodpile": bool(true_woodpile),
        "origin_mm": [ox, oy, oz],
        "z_breaks_mm": z_breaks,
        "pore_mm": pores,
        "anchor_mode": anchor_mode,
        "alternate_band_orientation": bool(alternate_band_orientation),
        "interface_layer_qc": rotation_qc,
        "slabs": slab_metas,
        "n_layers": sum(len(s["layers"]) for s in slab_metas),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume) if mesh.is_volume else None,
    }
    if domain == "box":
        report["domain_mm"] = [wx, wy, hz]
    else:
        report["radius_mm"] = r
        report["height_mm"] = hz
        report["clip_qc"] = verify_cylinder_clip(mesh, center_x_mm=ox, center_y_mm=oy, radius_mm=r)
    if not rotation_qc["ok"]:
        raise ValueError(
            "Piecewise interface layer QC failed: "
            f"{rotation_qc['checks']}"
        )
    return mesh, report


def verify_cylinder_clip(
    mesh: trimesh.Trimesh,
    *,
    center_x_mm: float,
    center_y_mm: float,
    radius_mm: float,
    tolerance_mm: float = 0.05,
) -> dict[str, Any]:
    """Check that all mesh vertices lie inside the cylinder wall (+ tolerance)."""
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        return {"ok": False, "max_radial_overshoot_mm": None, "n_vertices": 0}
    dx = verts[:, 0] - float(center_x_mm)
    dy = verts[:, 1] - float(center_y_mm)
    radial = np.sqrt(dx * dx + dy * dy)
    overshoot = float(np.max(radial) - float(radius_mm))
    return {
        "ok": overshoot <= float(tolerance_mm),
        "max_radial_overshoot_mm": overshoot,
        "tolerance_mm": float(tolerance_mm),
        "n_vertices": int(len(verts)),
    }


def generate_piecewise_crosshatch_box(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    height_mm: float,
    z_breaks_mm: list[float],
    pore_mm: list[float],
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    origin_z_mm: float = 0.0,
    anchor_mode: str = "center_void",
    alternate_band_orientation: bool = True,
    true_woodpile: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """
    Piecewise cross-hatch box: per-band pore, anchor, and global layer continuity.

    Mirrors ``woodpile_piecewise_box_single_pass`` band metadata without marching cubes.
    """
    return _piecewise_crosshatch_stack(
        z_breaks_mm=z_breaks_mm,
        pore_mm=pore_mm,
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
        origin_z_mm=origin_z_mm,
        anchor_mode=anchor_mode,
        alternate_band_orientation=alternate_band_orientation,
        true_woodpile=true_woodpile,
        domain="box",
        width_x_mm=width_x_mm,
        depth_y_mm=depth_y_mm,
        height_mm=height_mm,
        radius_mm=None,
    )


def generate_piecewise_crosshatch_cylinder(
    *,
    radius_mm: float,
    height_mm: float,
    z_breaks_mm: list[float],
    pore_mm: list[float],
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    origin_z_mm: float = 0.0,
    anchor_mode: str = "center_void",
    alternate_band_orientation: bool = True,
    true_woodpile: bool = False,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """
    Piecewise cross-hatch cylinder: per-band pore, anchor, clip, global layer continuity.

    Mirrors ``woodpile_piecewise_cylinder_single_pass`` band metadata without marching cubes.
    """
    return _piecewise_crosshatch_stack(
        z_breaks_mm=z_breaks_mm,
        pore_mm=pore_mm,
        origin_x_mm=origin_x_mm,
        origin_y_mm=origin_y_mm,
        origin_z_mm=origin_z_mm,
        anchor_mode=anchor_mode,
        alternate_band_orientation=alternate_band_orientation,
        true_woodpile=true_woodpile,
        domain="cylinder",
        width_x_mm=0.0,
        depth_y_mm=0.0,
        height_mm=height_mm,
        radius_mm=radius_mm,
    )


def mid_plane_solid_fraction(
    mesh: trimesh.Trimesh,
    *,
    plane: Literal["xz", "xy", "yz"] = "xz",
    center_mm: float = 0.5,
    half_thickness_mm: float = 0.012,
    n_samples: int = 200,
    x_range: tuple[float, float] = (0.0, 1.0),
    z_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Monte-free grid sample of solid fraction in a thin mid-plane slab."""
    a_lin = np.linspace(x_range[0], x_range[1], int(n_samples))
    b_lin = np.linspace(z_range[0], z_range[1], int(n_samples))
    if plane == "xz":
        aa, bb = np.meshgrid(a_lin, b_lin, indexing="ij")
        y0, y1 = center_mm - half_thickness_mm, center_mm + half_thickness_mm
        pts = []
        for y in (y0, center_mm, y1):
            pts.append(
                np.column_stack([aa.ravel(), np.full(aa.size, y), bb.ravel()])
            )
        pts_arr = np.vstack(pts)
        inside = mesh.contains(pts_arr).reshape(3, -1)
        solid = inside.any(axis=0).reshape(aa.shape)
        return float(solid.mean())
    raise ValueError(f"plane {plane!r} not implemented for solid fraction QC")
