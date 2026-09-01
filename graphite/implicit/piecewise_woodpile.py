"""
Piecewise Z-band woodpile.

**Preferred:** ``woodpile_piecewise_cylinder_single_pass`` — one full-cylinder EDT
field with per-band pore, phase anchor, and ``swap_xy`` orientation (see
``docs/PIECEWISE_PRISM_LATTICE_GENERATION.md`` §8).

**Legacy:** per-slab ``generate_uniform_woodpile_cylinder_slab`` + manifold3d union
(orientation toggles may not be visible — union carries prior-band struts forward).
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from graphite.explicit.geometry_module import _trimesh_to_manifold, manifold_to_trimesh
from graphite.geometry.masking import (
    axis_aligned_box_grid,
    axis_aligned_box_sdf,
    voxelize_cylinder_slab_and_edt,
)
from graphite.implicit.uniform_woodpile import generate_uniform_woodpile_cylinder_slab
from graphite.math.woodpile import evaluate_woodpile_piecewise_cylinder
from graphite.math.woodpile_anchor import (
    WoodpileAnchorMode,
    compute_band_orientation,
    compute_woodpile_xy_origin,
    compute_woodpile_xy_origin_box,
    count_crosshatch_layers_in_height,
    strut_run_axis_at_layer,
)


def boolean_union_slab_meshes(slabs: list[trimesh.Trimesh]) -> tuple[trimesh.Trimesh, str]:
    """Union slab solids with manifold3d; concatenate on failure."""
    if not slabs:
        raise ValueError("Expected at least one slab mesh.")
    try:
        manifolds = [_trimesh_to_manifold(s) for s in slabs]
        u = manifolds[0]
        for m in manifolds[1:]:
            u = u + m
        out = manifold_to_trimesh(u)
        out.remove_unreferenced_vertices()
        out = trimesh.Trimesh(vertices=out.vertices, faces=out.faces, process=True)
        return out, "manifold3d_boolean_union"
    except Exception:
        return trimesh.util.concatenate(slabs), "trimesh_concatenate_fallback"


def finalize_implicit_union_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Post-union cleanup. A successful manifold3d union is often already watertight;
    avoid trimesh repair steps that can introduce non-manifold edges.
    """
    if mesh.is_watertight:
        return mesh

    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    return m


def mesh_woodpile_cylinder_slab(
    *,
    z0_mm: float,
    z1_mm: float,
    pore_mm: float,
    radius_mm: float,
    resolution_mm: float,
    true_woodpile: bool,
    invert_solids: bool = False,
    origin_x_mm: float = 0.0,
    origin_y_mm: float = 0.0,
    flip_layer_parity: bool = False,
    swap_xy: bool = False,
) -> tuple[trimesh.Trimesh, dict]:
    """One Z band: short cylinder control surface + ``generate_uniform_woodpile``."""
    h_slab = float(z1_mm) - float(z0_mm)
    if h_slab <= 0.0:
        raise ValueError(f"Non-positive slab height: z0={z0_mm}, z1={z1_mm}")

    mesh_out = generate_uniform_woodpile_cylinder_slab(
        radius_mm=float(radius_mm),
        z0_mm=float(z0_mm),
        z1_mm=float(z1_mm),
        pore_size=float(pore_mm),
        true_woodpile=bool(true_woodpile),
        resolution=float(resolution_mm),
        center_origin=False,
        output_path=None,
        invert_solids=bool(invert_solids),
        origin_x=float(origin_x_mm),
        origin_y=float(origin_y_mm),
        flip_layer_parity=bool(flip_layer_parity),
        swap_xy=bool(swap_xy),
    )

    meta = {
        "z0_mm": float(z0_mm),
        "z1_mm": float(z1_mm),
        "thickness_mm": float(h_slab),
        "pore_mm": float(pore_mm),
        "origin_x_mm": float(origin_x_mm),
        "origin_y_mm": float(origin_y_mm),
        "flip_layer_parity": bool(flip_layer_parity),
        "swap_xy": bool(swap_xy),
        "watertight_trimesh": bool(mesh_out.is_watertight),
        "faces": int(len(mesh_out.faces)),
    }
    return mesh_out, meta


def _band_parameters(
    *,
    radius_mm: float,
    z_breaks: list[float],
    pores: list[float],
    anchor_mode: WoodpileAnchorMode,
    alternate_band_orientation: bool,
) -> tuple[list[float], list[float], list[bool], list[bool], list[dict]]:
    origins_x: list[float] = []
    origins_y: list[float] = []
    swaps: list[bool] = []
    flips: list[bool] = []
    slab_metas: list[dict] = []
    prev_swap_xy = False
    layer_offset = 0
    for i, pore in enumerate(pores):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        ox, oy, anchor_meta = compute_woodpile_xy_origin(
            radius_mm=float(radius_mm),
            pore_mm=float(pore),
            mode=anchor_mode,
        )
        swap_xy, flip, orient_meta = compute_band_orientation(
            i,
            prev_swap_xy=prev_swap_xy,
            alternate_band_orientation=alternate_band_orientation,
        )
        prev_swap_xy = swap_xy
        origins_x.append(ox)
        origins_y.append(oy)
        swaps.append(swap_xy)
        flips.append(flip)
        band_offset = layer_offset if alternate_band_orientation else 0
        n_layers = count_crosshatch_layers_in_height(z1 - z0, float(pore))
        layers = [
            {
                "idx": local_idx,
                "global_layer_idx": band_offset + local_idx,
                "strut_axis": strut_run_axis_at_layer(
                    band_offset + local_idx,
                    flip_layer_parity=flip,
                    swap_xy=swap_xy,
                ).lower(),
            }
            for local_idx in range(n_layers)
        ]
        slab_metas.append(
            {
                "band_index": int(i),
                "z0_mm": z0,
                "z1_mm": z1,
                "pore_mm": float(pore),
                "origin_x_mm": ox,
                "origin_y_mm": oy,
                "swap_xy": swap_xy,
                "flip_layer_parity": flip,
                "z_layer_origin_mm": z0,
                "layer_index_offset": band_offset,
                "n_layers": n_layers,
                "layers": layers,
                "anchor": anchor_meta,
                "orientation": orient_meta,
            }
        )
        if alternate_band_orientation:
            layer_offset += n_layers
    return origins_x, origins_y, swaps, flips, slab_metas


def _band_parameters_box(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    origin_x_mm: float,
    origin_y_mm: float,
    z_breaks: list[float],
    pores: list[float],
    anchor_mode: WoodpileAnchorMode,
    alternate_band_orientation: bool,
) -> tuple[list[float], list[float], list[bool], list[bool], list[dict]]:
    origins_x: list[float] = []
    origins_y: list[float] = []
    swaps: list[bool] = []
    flips: list[bool] = []
    slab_metas: list[dict] = []
    prev_swap_xy = False
    layer_offset = 0
    for i, pore in enumerate(pores):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        ox, oy, anchor_meta = compute_woodpile_xy_origin_box(
            width_x_mm=float(width_x_mm),
            depth_y_mm=float(depth_y_mm),
            pore_mm=float(pore),
            mode=anchor_mode,
            origin_x_mm=float(origin_x_mm),
            origin_y_mm=float(origin_y_mm),
        )
        swap_xy, flip, orient_meta = compute_band_orientation(
            i,
            prev_swap_xy=prev_swap_xy,
            alternate_band_orientation=alternate_band_orientation,
        )
        prev_swap_xy = swap_xy
        origins_x.append(ox)
        origins_y.append(oy)
        swaps.append(swap_xy)
        flips.append(flip)
        band_offset = layer_offset if alternate_band_orientation else 0
        n_layers = count_crosshatch_layers_in_height(z1 - z0, float(pore))
        layers = [
            {
                "idx": local_idx,
                "global_layer_idx": band_offset + local_idx,
                "strut_axis": strut_run_axis_at_layer(
                    band_offset + local_idx,
                    flip_layer_parity=flip,
                    swap_xy=swap_xy,
                ).lower(),
            }
            for local_idx in range(n_layers)
        ]
        slab_metas.append(
            {
                "band_index": int(i),
                "z0_mm": z0,
                "z1_mm": z1,
                "pore_mm": float(pore),
                "origin_x_mm": ox,
                "origin_y_mm": oy,
                "swap_xy": swap_xy,
                "flip_layer_parity": flip,
                "z_layer_origin_mm": z0,
                "layer_index_offset": band_offset,
                "n_layers": n_layers,
                "layers": layers,
                "anchor": anchor_meta,
                "orientation": orient_meta,
            }
        )
        if alternate_band_orientation:
            layer_offset += n_layers
    return origins_x, origins_y, swaps, flips, slab_metas


def woodpile_piecewise_cylinder_single_pass(
    *,
    radius_mm: float,
    z_breaks_mm: Sequence[float],
    pore_mm: Sequence[float],
    resolution_mm: float,
    true_woodpile: bool = True,
    invert_solids: bool = False,
    anchor_mode: WoodpileAnchorMode = "center_void",
    alternate_band_orientation: bool = True,
) -> tuple[trimesh.Trimesh, dict]:
    """
    One full-cylinder field with piecewise woodpile per Z band (no boolean union).

    Each band restarts layer parity at ``z0`` and only contributes solid in its
    Z span — avoids prior-band struts persisting through union.
    """
    z_breaks = [float(z) for z in z_breaks_mm]
    pores = [float(p) for p in pore_mm]
    if len(z_breaks) != len(pores) + 1:
        raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")

    height_mm = z_breaks[-1] - z_breaks[0]
    ox, oy, swaps, flips, slab_metas = _band_parameters(
        radius_mm=float(radius_mm),
        z_breaks=z_breaks,
        pores=pores,
        anchor_mode=anchor_mode,
        alternate_band_orientation=alternate_band_orientation,
    )

    X, Y, Z, cad_sdf, padded_min_bound, _max_bound, _nx, _ny, _nz = (
        voxelize_cylinder_slab_and_edt(
            float(radius_mm), z_breaks[0], z_breaks[-1], float(resolution_mm)
        )
    )

    woodpile_field = evaluate_woodpile_piecewise_cylinder(
        X,
        Y,
        Z,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        origin_x_mm=ox,
        origin_y_mm=oy,
        swap_xy=swaps,
        flip_layer_parity=flips,
        true_woodpile=bool(true_woodpile),
        continuous_layer_index=bool(alternate_band_orientation),
    )
    if invert_solids:
        woodpile_field = -woodpile_field
    final_field = np.maximum(woodpile_field, cad_sdf)

    t0 = time.perf_counter()
    verts, faces, _n, _v = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(float(resolution_mm),) * 3,
    )
    mc_s = time.perf_counter() - t0
    verts = verts + padded_min_bound
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    mode = "true_woodpile" if true_woodpile else "cross_hatch"
    report: dict = {
        "pipeline": "full-cylinder EDT + piecewise woodpile; single pass",
        "combine_method": "single_pass_implicit",
        "lattice_type": mode,
        "z_breaks_mm": z_breaks,
        "pore_mm": pores,
        "height_mm": float(height_mm),
        "radius_mm": float(radius_mm),
        "resolution_mm": float(resolution_mm),
        "invert_solids": bool(invert_solids),
        "anchor_mode": anchor_mode,
        "alternate_band_orientation": bool(alternate_band_orientation),
        "slabs": slab_metas,
        "marching_cubes_seconds": float(mc_s),
        "faces": int(len(mesh_out.faces)),
        "vertices": int(len(mesh_out.vertices)),
        "watertight": bool(mesh_out.is_watertight),
        "volume_mm3": float(mesh_out.volume) if mesh_out.is_volume else None,
    }
    print(
        f"Piecewise woodpile single-pass ({mode}): r={radius_mm:g}mm h={height_mm:g}mm, "
        f"res={resolution_mm:g}mm, mc={mc_s:.2f}s, faces={len(mesh_out.faces):,}"
    )
    return mesh_out, report


def woodpile_piecewise_box_single_pass(
    *,
    width_x_mm: float,
    depth_y_mm: float,
    height_mm: float,
    z_breaks_mm: Sequence[float],
    pore_mm: Sequence[float],
    resolution_mm: float,
    true_woodpile: bool = False,
    invert_solids: bool = False,
    anchor_mode: WoodpileAnchorMode = "center_void",
    alternate_band_orientation: bool = True,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    origin_z: float = 0.0,
) -> tuple[trimesh.Trimesh, dict]:
    """
    One full box field with piecewise woodpile per Z band (no boolean union).

    Domain: axis-aligned box with corner at ``(origin_x, origin_y, origin_z)``.
    Caps from analytic box SDF (same recipe as piecewise Split-P box).
    """
    z_breaks = [float(z) for z in z_breaks_mm]
    pores = [float(p) for p in pore_mm]
    if len(z_breaks) != len(pores) + 1:
        raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")

    ox, oy, oz = float(origin_x), float(origin_y), float(origin_z)
    wx, wy, hz = float(width_x_mm), float(depth_y_mm), float(height_mm)
    res = float(resolution_mm)

    ox_list, oy_list, swaps, flips, slab_metas = _band_parameters_box(
        width_x_mm=wx,
        depth_y_mm=wy,
        origin_x_mm=ox,
        origin_y_mm=oy,
        z_breaks=z_breaks,
        pores=pores,
        anchor_mode=anchor_mode,
        alternate_band_orientation=alternate_band_orientation,
    )

    X, Y, Z, grid_origin, spacing = axis_aligned_box_grid(
        wx, wy, hz, res, origin_x=ox, origin_y=oy, origin_z=oz
    )
    box_sdf = axis_aligned_box_sdf(
        X,
        Y,
        Z,
        origin_x=ox,
        origin_y=oy,
        origin_z=oz,
        width_x_mm=wx,
        depth_y_mm=wy,
        height_z_mm=hz,
    )

    woodpile_field = evaluate_woodpile_piecewise_cylinder(
        X,
        Y,
        Z,
        z_breaks_mm=z_breaks,
        pore_mm=pores,
        origin_x_mm=ox_list,
        origin_y_mm=oy_list,
        swap_xy=swaps,
        flip_layer_parity=flips,
        true_woodpile=bool(true_woodpile),
        continuous_layer_index=bool(alternate_band_orientation),
    )
    if invert_solids:
        woodpile_field = -woodpile_field
    final_field = np.maximum(woodpile_field, box_sdf)

    t0 = time.perf_counter()
    verts, faces, _n, _v = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=spacing,
    )
    mc_s = time.perf_counter() - t0
    verts = verts + grid_origin
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    mode = "true_woodpile" if true_woodpile else "cross_hatch"
    report: dict = {
        "pipeline": "analytic box SDF + piecewise woodpile; single pass",
        "boundary_sdf": "analytic_box",
        "combine_method": "single_pass_implicit",
        "lattice_type": mode,
        "z_breaks_mm": z_breaks,
        "pore_mm": pores,
        "height_mm": float(hz),
        "width_x_mm": float(wx),
        "depth_y_mm": float(wy),
        "origin_mm": [ox, oy, oz],
        "resolution_mm": float(res),
        "invert_solids": bool(invert_solids),
        "anchor_mode": anchor_mode,
        "alternate_band_orientation": bool(alternate_band_orientation),
        "slabs": slab_metas,
        "marching_cubes_seconds": float(mc_s),
        "faces": int(len(mesh_out.faces)),
        "vertices": int(len(mesh_out.vertices)),
        "watertight": bool(mesh_out.is_watertight),
        "volume_mm3": float(mesh_out.volume) if mesh_out.is_volume else None,
    }
    print(
        f"Piecewise woodpile box single-pass ({mode}): "
        f"{wx:g}x{wy:g}x{hz:g} mm, res={res:g} mm, mc={mc_s:.2f}s, "
        f"faces={len(mesh_out.faces):,}"
    )
    return mesh_out, report


def woodpile_piecewise_cylinder_union(
    *,
    radius_mm: float,
    z_breaks_mm: Sequence[float],
    pore_mm: Sequence[float],
    resolution_mm: float,
    true_woodpile: bool = True,
    invert_solids: bool = False,
    repair_mesh: bool = True,
    anchor_mode: WoodpileAnchorMode = "center_void",
    alternate_band_orientation: bool = True,
) -> tuple[trimesh.Trimesh, dict]:
    """
    Piecewise woodpile in a cylinder by Z control-surface bands.

    .. deprecated::
        Per-band ``generate_uniform_woodpile`` + manifold union. Prefer
        ``woodpile_piecewise_cylinder_single_pass`` or ``generator='extrude'``.

    ``z_breaks_mm`` has length ``len(pore_mm) + 1`` (monotone, starting at 0).
    """
    import warnings

    warnings.warn(
        "woodpile_piecewise_cylinder_union is deprecated; use "
        "woodpile_piecewise_cylinder_single_pass or generator='extrude'",
        DeprecationWarning,
        stacklevel=2,
    )
    z_breaks = [float(z) for z in z_breaks_mm]
    pores = [float(p) for p in pore_mm]
    if len(z_breaks) != len(pores) + 1:
        raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")

    slabs: list[trimesh.Trimesh] = []
    slab_metas: list[dict] = []
    prev_swap_xy = False
    for i, pore in enumerate(pores):
        z0, z1 = z_breaks[i], z_breaks[i + 1]
        ox, oy, anchor_meta = compute_woodpile_xy_origin(
            radius_mm=float(radius_mm),
            pore_mm=float(pore),
            mode=anchor_mode,
        )
        swap_xy, flip, orient_meta = compute_band_orientation(
            i,
            prev_swap_xy=prev_swap_xy,
            alternate_band_orientation=alternate_band_orientation,
        )
        prev_swap_xy = swap_xy
        mesh_s, meta = mesh_woodpile_cylinder_slab(
            z0_mm=z0,
            z1_mm=z1,
            pore_mm=pore,
            radius_mm=float(radius_mm),
            resolution_mm=float(resolution_mm),
            true_woodpile=bool(true_woodpile),
            invert_solids=bool(invert_solids),
            origin_x_mm=ox,
            origin_y_mm=oy,
            flip_layer_parity=flip,
            swap_xy=swap_xy,
        )
        meta["anchor"] = anchor_meta
        meta["orientation"] = orient_meta
        meta["band_index"] = int(i)
        slabs.append(mesh_s)
        slab_metas.append(meta)

    mesh_u, combine_method = boolean_union_slab_meshes(slabs)
    watertight_before = bool(mesh_u.is_watertight)
    mesh_out = finalize_implicit_union_mesh(mesh_u) if repair_mesh else mesh_u

    height_mm = z_breaks[-1] - z_breaks[0]
    mode = "true_woodpile" if true_woodpile else "cross_hatch"
    report: dict = {
        "pipeline": "per-Z-band cylinder control surface + generate_uniform_woodpile + manifold union",
        "lattice_type": mode,
        "combine_method": combine_method,
        "z_breaks_mm": z_breaks,
        "pore_mm": pores,
        "height_mm": float(height_mm),
        "radius_mm": float(radius_mm),
        "resolution_mm": float(resolution_mm),
        "invert_solids": bool(invert_solids),
        "anchor_mode": anchor_mode,
        "alternate_band_orientation": bool(alternate_band_orientation),
        "slabs": slab_metas,
        "watertight_union_before_repair": watertight_before,
        "watertight_union_after_repair": bool(mesh_out.is_watertight),
        "faces": int(len(mesh_out.faces)),
        "vertices": int(len(mesh_out.vertices)),
        "volume_mm3": float(mesh_out.volume) if mesh_out.is_volume else None,
    }
    return mesh_out, report
