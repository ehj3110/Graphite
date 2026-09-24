"""
Graphite Implicit Engine - Field-Driven Lattices

This module generates conformal TPMS lattices that are modulated by 1D, 2D, or 
3D scalar fields. It supports grading both the unit cell size (frequency/phase) 
and the solid volume fraction (threshold) across Cartesian, Cylindrical, and 
Spherical coordinate systems using user-defined control points.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.density_control import tau_from_wall_thickness_mm
from graphite.io.mesh_export import export_mesh
from graphite.math.tpms import calculate_integrated_phase, evaluate_tpms_phase


COORDINATE_OPTIONS = (
    "Cartesian",
    "Cylindrical Radius",
    "Spherical Radius",
)


def parse_control_points(raw: object, *, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Parse control points from either a Streamlit data_editor result or a simple list.

    Expected logical schema is ``position_mm`` and ``value``. Invalid rows are skipped.

    Parameters
    ----------
    raw : object
        The raw input data, either a list of tuples, list of dicts, or a dictionary 
        (e.g., from Streamlit).
    default : list of tuple of float
        Fallback control points if parsing fails or input is empty.

    Returns
    -------
    list of tuple of float
        A sorted list of (position, value) tuples.
    """
    if raw is None:
        return list(default)

    rows: list[object]
    if isinstance(raw, dict):
        if "edited_rows" in raw:
            return list(default)
        rows = list(raw.values())
    elif isinstance(raw, (list, tuple)):
        rows = list(raw)
    else:
        return list(default)

    out: list[tuple[float, float]] = []
    for row in rows:
        try:
            if isinstance(row, dict):
                pos = row.get("position_mm", row.get("Position (mm)", row.get("position")))
                val = row.get("value", row.get("Value", row.get("target")))
            else:
                pos, val = row  # type: ignore[misc]
            out.append((float(pos), float(val)))
        except Exception:
            continue
    if not out:
        return list(default)
    out.sort(key=lambda p: p[0])
    return out


def control_points_to_arrays(control_points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    if len(control_points) < 1:
        raise ValueError("control_points must contain at least one point")
    pts = np.asarray(control_points, dtype=np.float64)
    order = np.argsort(pts[:, 0])
    x = pts[order, 0]
    y = pts[order, 1]
    return x, y


def build_coordinate_field(
    coordinate: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    padded_min_bound: np.ndarray,
    center: np.ndarray,
) -> tuple[np.ndarray, str]:
    coord = str(coordinate)
    if coord == "Cartesian X":
        return X - float(padded_min_bound[0]), "x"
    if coord == "Cartesian Y":
        return Y - float(padded_min_bound[1]), "y"
    if coord == "Cartesian Z":
        return Z - float(padded_min_bound[2]), "z"
    if coord == "Cartesian":
        return Z - float(padded_min_bound[2]), "z"
    if coord == "Cylindrical Radius":
        return np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2), "cylindrical"
    if coord == "Spherical Radius":
        return np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2), "spherical"
    raise ValueError(f"Unsupported coordinate system: {coordinate}")


def preview_slice_axis(coordinate: str) -> str:
    if coordinate == "Cartesian X":
        return "x"
    if coordinate == "Cartesian Y":
        return "y"
    return "z"


def interpolate_profile(
    coordinate_values: np.ndarray,
    control_points: list[tuple[float, float]],
    *,
    min_value: float | None = None,
) -> np.ndarray:
    cp_x, cp_y = control_points_to_arrays(control_points)
    values = np.interp(coordinate_values, cp_x, cp_y)
    if min_value is not None:
        values = np.maximum(values, float(min_value))
    return values


def _axis_positions(
    axis_index: int,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    min_bound: np.ndarray,
) -> np.ndarray:
    grids = (X, Y, Z)
    return grids[axis_index] - float(min_bound[axis_index])


def _axis_control_points(
    control_points_by_axis: dict[str, list[tuple[float, float]]] | None,
    axis: str,
    default: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not isinstance(control_points_by_axis, dict):
        return default
    points = control_points_by_axis.get(axis) or control_points_by_axis.get(axis.upper())
    return points if points else default


def _integrated_axis_phase(
    axis_values: np.ndarray,
    control_points: list[tuple[float, float]],
    *,
    origin_position: float,
) -> np.ndarray:
    cp_x, cp_l = control_points_to_arrays(control_points)
    dense_min = min(float(np.min(axis_values)), float(cp_x[0]), float(origin_position))
    dense_max = max(float(np.max(axis_values)), float(cp_x[-1]), float(origin_position))
    dense = np.linspace(dense_min, dense_max, 8192, dtype=np.float64)
    phase_dense = calculate_integrated_phase(dense, cp_x, cp_l)
    phase = np.interp(axis_values, dense, phase_dense)
    origin_phase = float(np.interp(float(origin_position), dense, phase_dense))
    return phase - origin_phase


def build_cartesian_integrated_phase_coordinates(
    control_points_by_axis: dict[str, list[tuple[float, float]]] | None,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    min_bound: np.ndarray,
    origin: np.ndarray,
    base_unit_cell_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axis_values = _axis_positions(axis_index, X, Y, Z, min_bound)
        max_axis = float(np.max(axis_values))
        default = [(0.0, float(base_unit_cell_size)), (max_axis, float(base_unit_cell_size))]
        points = _axis_control_points(control_points_by_axis, axis_name, default)
        origin_position = float(origin[axis_index] - min_bound[axis_index])
        phases.append(
            _integrated_axis_phase(
                axis_values,
                points,
                origin_position=origin_position,
            )
        )
    return phases[0], phases[1], phases[2]


def build_cartesian_profile_grid(
    control_points_by_axis: dict[str, list[tuple[float, float]]] | None,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    min_bound: np.ndarray,
    base_value: float,
    min_value: float | None = None,
) -> np.ndarray:
    active_profiles: list[np.ndarray] = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        axis_values = _axis_positions(axis_index, X, Y, Z, min_bound)
        max_axis = float(np.max(axis_values))
        default = [(0.0, float(base_value)), (max_axis, float(base_value))]
        points = _axis_control_points(control_points_by_axis, axis_name, default)
        _cp_x, cp_y = control_points_to_arrays(points)
        profile = interpolate_profile(axis_values, points)
        if np.max(np.abs(cp_y - float(base_value))) > 1e-9:
            active_profiles.append(profile)

    if active_profiles:
        deltas = [profile - float(base_value) for profile in active_profiles]
        values = float(base_value) + np.mean(deltas, axis=0)
    else:
        values = np.full_like(X, float(base_value), dtype=np.float64)
    if min_value is not None:
        values = np.maximum(values, float(min_value))
    return values


def build_integrated_phase_coordinates(
    coordinate: str,
    coord_kind: str,
    coordinate_values: np.ndarray,
    L_grid: np.ndarray,
    unit_cell_control_points: list[tuple[float, float]],
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build TPMS phase coordinates. Unit-cell variation always uses integrated phase.
    """
    cp_x, cp_l = control_points_to_arrays(unit_cell_control_points)
    dense_min = min(float(np.min(coordinate_values)), float(cp_x[0]))
    dense_max = max(float(np.max(coordinate_values)), float(cp_x[-1]))
    dense = np.linspace(dense_min, dense_max, 8192, dtype=np.float64)
    phase_dense = calculate_integrated_phase(dense, cp_x, cp_l)
    phase = np.interp(coordinate_values, dense, phase_dense)
    omega = 2.0 * np.pi / np.maximum(L_grid, 1e-6)

    Xc = X - float(center[0])
    Yc = Y - float(center[1])
    Zc = Z - float(center[2])

    if coord_kind == "x":
        return phase, Yc * omega, Zc * omega
    if coord_kind == "y":
        return Xc * omega, phase, Zc * omega
    if coord_kind == "z":
        return Xc * omega, Yc * omega, phase
    if coord_kind == "cylindrical":
        rho = np.maximum(coordinate_values, 1e-9)
        scale = phase / rho
        return Xc * scale, Yc * scale, Zc * omega
    if coord_kind == "spherical":
        r = np.maximum(coordinate_values, 1e-9)
        scale = phase / r
        return Xc * scale, Yc * scale, Zc * scale
    raise ValueError(f"Unsupported coordinate kind: {coord_kind}")


def _sample_abs_tpms_distribution(lattice_type: str, *, samples_per_axis: int = 96) -> np.ndarray:
    axis = np.linspace(0.0, 2.0 * np.pi, int(samples_per_axis), endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    values.sort()
    return values


def _tau_from_solid_fraction(abs_values_sorted: np.ndarray, target_sf: np.ndarray | float) -> np.ndarray:
    sf = np.asarray(target_sf, dtype=np.float64)
    q = np.clip(sf, 0.0, 1.0)
    flat_q = q.ravel()
    n = int(len(abs_values_sorted))
    pos = flat_q * float(n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    w = pos - lo
    flat_tau = abs_values_sorted[lo] * (1.0 - w) + abs_values_sorted[hi] * w
    return flat_tau.reshape(q.shape)


def _build_tau_grid(
    *,
    lattice_type: str,
    density_mode: str,
    density_grid: np.ndarray,
    L_grid: np.ndarray,
    calibrate_solid_fraction: bool,
) -> np.ndarray:
    """Resolve per-voxel TPMS threshold from solid fraction or wall thickness."""
    mode = str(density_mode).lower().replace(" ", "_")
    if mode in {"wall_thickness", "wall_thickness_mm"}:
        tau_cap = float(np.max(_sample_abs_tpms_distribution(lattice_type))) * 0.98
        return tau_from_wall_thickness_mm(density_grid, L_grid, tau_cap=tau_cap)
    if calibrate_solid_fraction:
        return _tau_from_solid_fraction(_sample_abs_tpms_distribution(lattice_type), density_grid)
    return np.asarray(density_grid, dtype=np.float64)


def _contains_points(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    try:
        return mesh.contains(points)
    except Exception:
        return trimesh.proximity.signed_distance(mesh, points) >= 0.0


def _slice_inside_mask(
    mesh: trimesh.Trimesh,
    *,
    axis_index: int,
    image_axes: list[int],
    a_axis: np.ndarray,
    b_axis: np.ndarray,
    slice_position: float,
) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    normal[axis_index] = 1.0
    origin = np.zeros(3, dtype=np.float64)
    origin[axis_index] = float(slice_position)

    try:
        segments = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=normal,
            plane_origin=origin,
        )
    except Exception:
        segments = np.empty((0, 2, 3), dtype=np.float64)

    if len(segments) == 0:
        A, B = np.meshgrid(a_axis, b_axis, indexing="ij")
        coords = [
            np.full_like(A, origin[0]),
            np.full_like(A, origin[1]),
            np.full_like(A, origin[2]),
        ]
        coords[axis_index] = np.full_like(A, slice_position)
        coords[image_axes[0]] = A
        coords[image_axes[1]] = B
        points = np.column_stack([coords[0].ravel(), coords[1].ravel(), coords[2].ravel()])
        return _contains_points(mesh, points).reshape(A.shape)

    seg = np.asarray(segments, dtype=np.float64)
    x0 = seg[:, 0, image_axes[0]]
    y0 = seg[:, 0, image_axes[1]]
    x1 = seg[:, 1, image_axes[0]]
    y1 = seg[:, 1, image_axes[1]]
    dy = y1 - y0
    valid = np.abs(dy) > 1e-12
    x0, y0, x1, y1, dy = x0[valid], y0[valid], x1[valid], y1[valid], dy[valid]

    mask = np.zeros((len(a_axis), len(b_axis)), dtype=bool)
    for j, b in enumerate(b_axis):
        crosses = ((y0 <= b) & (y1 > b)) | ((y1 <= b) & (y0 > b))
        if not np.any(crosses):
            continue
        xs = x0[crosses] + (b - y0[crosses]) * (x1[crosses] - x0[crosses]) / dy[crosses]
        xs.sort()
        if len(xs) < 2:
            continue
        for left, right in zip(xs[0::2], xs[1::2]):
            start = int(np.searchsorted(a_axis, left, side="left"))
            end = int(np.searchsorted(a_axis, right, side="right"))
            if end > start:
                mask[start:end, j] = True
    return mask


def generate_field_preview_image(
    mesh: trimesh.Trimesh,
    *,
    lattice_type: str = "Gyroid",
    preview_resolution: float = 0.01,
    base_unit_cell_size: float = 5.0,
    base_solid_fraction: float = 0.33,
    grade_unit_cell: bool = False,
    unit_cell_coordinate: str = "Cartesian",
    unit_cell_control_points: list[tuple[float, float]] | None = None,
    unit_cell_cartesian_control_points: dict[str, list[tuple[float, float]]] | None = None,
    grade_solid_fraction: bool = False,
    solid_fraction_coordinate: str = "Cartesian",
    solid_fraction_control_points: list[tuple[float, float]] | None = None,
    solid_fraction_cartesian_control_points: dict[str, list[tuple[float, float]]] | None = None,
    calibrate_solid_fraction: bool = True,
    density_mode: str = "solid_fraction",
    base_wall_thickness_mm: float = 0.5,
    field_origin: np.ndarray | None = None,
    max_pixels: int = 700,
) -> tuple[np.ndarray, dict[str, float | str | int]]:
    """
    Render a binary center slice preview: black background, white lattice pixels.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    min_bound, max_bound = bounds
    extents = np.maximum(max_bound - min_bound, 1e-6)
    origin = (
        np.asarray(field_origin, dtype=np.float64)
        if field_origin is not None
        else np.asarray(mesh.center_mass, dtype=np.float64)
    )

    primary_coordinate = unit_cell_coordinate if grade_unit_cell else solid_fraction_coordinate
    slice_axis = preview_slice_axis(primary_coordinate)
    axis_index = {"x": 0, "y": 1, "z": 2}[slice_axis]
    image_axes = [idx for idx in range(3) if idx != axis_index]

    ranges = []
    for idx in image_axes:
        pad = 0.10 * extents[idx]
        ranges.append((min_bound[idx] - pad, max_bound[idx] + pad))

    pixel_size = max(float(preview_resolution), 1e-6)
    n0 = int(np.ceil((ranges[0][1] - ranges[0][0]) / pixel_size)) + 1
    n1 = int(np.ceil((ranges[1][1] - ranges[1][0]) / pixel_size)) + 1
    scale = max(n0 / max_pixels, n1 / max_pixels, 1.0)
    if scale > 1.0:
        pixel_size *= scale
        n0 = int(np.ceil((ranges[0][1] - ranges[0][0]) / pixel_size)) + 1
        n1 = int(np.ceil((ranges[1][1] - ranges[1][0]) / pixel_size)) + 1

    a = np.linspace(ranges[0][0], ranges[0][1], n0, dtype=np.float64)
    b = np.linspace(ranges[1][0], ranges[1][1], n1, dtype=np.float64)
    A, B = np.meshgrid(a, b, indexing="ij")

    coords = [np.full_like(A, origin[0]), np.full_like(A, origin[1]), np.full_like(A, origin[2])]
    coords[axis_index] = np.full_like(A, origin[axis_index])
    coords[image_axes[0]] = A
    coords[image_axes[1]] = B
    X, Y, Z = coords

    coordinate_min_bound = min_bound
    base_density = float(base_wall_thickness_mm if str(density_mode).lower().startswith("wall") else base_solid_fraction)
    if grade_unit_cell and unit_cell_coordinate == "Cartesian":
        U, V, W = build_cartesian_integrated_phase_coordinates(
            unit_cell_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            origin=origin,
            base_unit_cell_size=float(base_unit_cell_size),
        )
        L_grid = build_cartesian_profile_grid(
            unit_cell_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            base_value=float(base_unit_cell_size),
            min_value=1e-3,
        )
    elif grade_unit_cell:
        default_uc = [(0.0, float(base_unit_cell_size)), (float(np.max(extents)), float(base_unit_cell_size))]
        uc_points = unit_cell_control_points or default_uc
        uc_coord, uc_kind = build_coordinate_field(
            unit_cell_coordinate,
            X,
            Y,
            Z,
            padded_min_bound=coordinate_min_bound,
            center=origin,
        )
        L_grid = interpolate_profile(uc_coord, uc_points, min_value=1e-3)
        U, V, W = build_integrated_phase_coordinates(
            unit_cell_coordinate,
            uc_kind,
            uc_coord,
            L_grid,
            uc_points,
            X,
            Y,
            Z,
            center=origin,
        )
    else:
        L_grid = np.full_like(X, float(base_unit_cell_size), dtype=np.float64)
        omega = 2.0 * np.pi / max(float(base_unit_cell_size), 1e-6)
        U = (X - origin[0]) * omega
        V = (Y - origin[1]) * omega
        W = (Z - origin[2]) * omega

    F = evaluate_tpms_phase(lattice_type, U, V, W)

    density_min = 1e-3 if str(density_mode).lower().startswith("wall") else 0.0
    if grade_solid_fraction and solid_fraction_coordinate == "Cartesian":
        density_grid = build_cartesian_profile_grid(
            solid_fraction_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            base_value=base_density,
            min_value=density_min,
        )
    elif grade_solid_fraction:
        default_density = [(0.0, base_density), (float(np.max(extents)), base_density)]
        density_points = solid_fraction_control_points or default_density
        density_coord, _sf_kind = build_coordinate_field(
            solid_fraction_coordinate,
            X,
            Y,
            Z,
            padded_min_bound=coordinate_min_bound,
            center=origin,
        )
        density_grid = interpolate_profile(density_coord, density_points, min_value=density_min)
    else:
        density_grid = np.full_like(X, base_density, dtype=np.float64)

    tau_grid = _build_tau_grid(
        lattice_type=lattice_type,
        density_mode=density_mode,
        density_grid=density_grid,
        L_grid=L_grid,
        calibrate_solid_fraction=calibrate_solid_fraction,
    )

    inside = _slice_inside_mask(
        mesh,
        axis_index=axis_index,
        image_axes=image_axes,
        a_axis=a,
        b_axis=b,
        slice_position=float(origin[axis_index]),
    )
    lattice = (np.abs(F) - tau_grid <= 0.0) & inside
    image = np.where(lattice, 255, 0).astype(np.uint8)
    image = np.flipud(image.T)

    metadata: dict[str, float | str | int] = {
        "slice_axis": slice_axis.upper(),
        "slice_position_mm": float(origin[axis_index]),
        "pixel_size_mm": float(pixel_size),
        "width_px": int(image.shape[1]),
        "height_px": int(image.shape[0]),
    }
    return image, metadata


def generate_field_driven_lattice(
    stl_path: str | Path,
    *,
    lattice_type: str = "Gyroid",
    resolution: float = 0.25,
    base_unit_cell_size: float = 5.0,
    base_solid_fraction: float = 0.33,
    grade_unit_cell: bool = False,
    unit_cell_coordinate: str = "Cartesian",
    unit_cell_control_points: list[tuple[float, float]] | None = None,
    unit_cell_cartesian_control_points: dict[str, list[tuple[float, float]]] | None = None,
    grade_solid_fraction: bool = False,
    solid_fraction_coordinate: str = "Cartesian",
    solid_fraction_control_points: list[tuple[float, float]] | None = None,
    solid_fraction_cartesian_control_points: dict[str, list[tuple[float, float]]] | None = None,
    calibrate_solid_fraction: bool = True,
    density_mode: str = "solid_fraction",
    base_wall_thickness_mm: float = 0.5,
    field_origin: np.ndarray | None = None,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    center_origin: bool = False,
    output_path: str | Path | None = None,
    export_formats: tuple[str, ...] | str | None = None,
) -> trimesh.Trimesh:
    """
    Generate a conformal TPMS lattice graded by spatial control fields.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    lattice_type : str, optional
        TPMS equation type (e.g., 'Gyroid'), by default "Gyroid".
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    base_unit_cell_size : float, optional
        Default unit cell size (L) in mm, by default 5.0.
    base_solid_fraction : float, optional
        Default solid volume fraction threshold, by default 0.33.
    grade_unit_cell : bool, optional
        Whether to enable spatial grading of the unit cell size, by default False.
    unit_cell_coordinate : str, optional
        Coordinate system for unit cell grading ('Cartesian', 'Cylindrical Radius', 
        'Spherical Radius'), by default "Cartesian".
    unit_cell_control_points : list of tuple of float, optional
        (position, value) pairs for 1D grading, by default None.
    unit_cell_cartesian_control_points : dict, optional
        Dictionary mapping axes ('x', 'y', 'z') to their respective control points 
        for 3D Cartesian grading, by default None.
    grade_solid_fraction : bool, optional
        Whether to enable spatial grading of the solid fraction, by default False.
    solid_fraction_coordinate : str, optional
        Coordinate system for solid fraction grading, by default "Cartesian".
    solid_fraction_control_points : list of tuple of float, optional
        (position, value) pairs for 1D grading, by default None.
    solid_fraction_cartesian_control_points : dict, optional
        Dictionary mapping axes to control points for 3D grading, by default None.
    calibrate_solid_fraction : bool, optional
        If True, maps solid fraction targets through a sampled TPMS distribution 
        for accurate volume control. If False, treats values as direct level-set 
        thresholds (tau), by default True. Ignored when ``density_mode`` is wall thickness.
    density_mode : str, optional
        ``solid_fraction`` or ``wall_thickness``. Wall thickness uses physical mm and
        local period ``L_grid`` to set tau directly.
    base_wall_thickness_mm : float, optional
        Base wall thickness when ``density_mode`` is wall thickness, by default 0.5.
    field_origin : ndarray, optional
        (3,) coordinate array defining the origin of the gradient fields. If None, 
        uses the mesh center of mass. By default None.
    export_mode : str, optional
        'core', 'skin', or 'combined', by default "core".
    shell_thickness : float, optional
        Thickness of the generated outer shell in mm, by default 2.0.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.

    Returns
    -------
    trimesh.Trimesh
        The meshed field-driven lattice.
    """
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, float(resolution)
    )
    origin = (
        np.asarray(field_origin, dtype=np.float64)
        if field_origin is not None
        else np.asarray(mesh.center_mass, dtype=np.float64)
    )
    coordinate_min_bound = np.asarray(mesh.bounds[0], dtype=np.float64)
    base_density = float(base_wall_thickness_mm if str(density_mode).lower().startswith("wall") else base_solid_fraction)

    if grade_unit_cell and unit_cell_coordinate == "Cartesian":
        U, V, W = build_cartesian_integrated_phase_coordinates(
            unit_cell_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            origin=origin,
            base_unit_cell_size=float(base_unit_cell_size),
        )
        L_grid = build_cartesian_profile_grid(
            unit_cell_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            base_value=float(base_unit_cell_size),
            min_value=1e-3,
        )
    elif grade_unit_cell:
        default_uc = [(0.0, float(base_unit_cell_size)), (float(np.max(mesh.extents)), float(base_unit_cell_size))]
        uc_points = unit_cell_control_points or default_uc
        uc_coord, uc_kind = build_coordinate_field(
            unit_cell_coordinate,
            X,
            Y,
            Z,
            padded_min_bound=coordinate_min_bound,
            center=origin,
        )
        L_grid = interpolate_profile(uc_coord, uc_points, min_value=1e-3)
        U, V, W = build_integrated_phase_coordinates(
            unit_cell_coordinate,
            uc_kind,
            uc_coord,
            L_grid,
            uc_points,
            X,
            Y,
            Z,
            center=origin,
        )
    else:
        L_grid = np.full_like(X, float(base_unit_cell_size), dtype=np.float64)
        omega = 2.0 * np.pi / max(float(base_unit_cell_size), 1e-6)
        U = (X - origin[0]) * omega
        V = (Y - origin[1]) * omega
        W = (Z - origin[2]) * omega

    F = evaluate_tpms_phase(lattice_type, U, V, W)

    density_min = 1e-3 if str(density_mode).lower().startswith("wall") else 0.0
    if grade_solid_fraction and solid_fraction_coordinate == "Cartesian":
        density_grid = build_cartesian_profile_grid(
            solid_fraction_cartesian_control_points,
            X,
            Y,
            Z,
            min_bound=coordinate_min_bound,
            base_value=base_density,
            min_value=density_min,
        )
    elif grade_solid_fraction:
        default_density = [(0.0, base_density), (float(np.max(mesh.extents)), base_density)]
        density_points = solid_fraction_control_points or default_density
        density_coord, _sf_kind = build_coordinate_field(
            solid_fraction_coordinate,
            X,
            Y,
            Z,
            padded_min_bound=coordinate_min_bound,
            center=origin,
        )
        density_grid = interpolate_profile(density_coord, density_points, min_value=density_min)
    else:
        density_grid = np.full_like(X, base_density, dtype=np.float64)

    tau_grid = _build_tau_grid(
        lattice_type=lattice_type,
        density_mode=density_mode,
        density_grid=density_grid,
        L_grid=L_grid,
        calibrate_solid_fraction=calibrate_solid_fraction,
    )

    core_sdf = np.maximum(np.abs(F) - tau_grid, cad_sdf)
    skin_sdf = np.maximum(cad_sdf, -cad_sdf - float(shell_thickness))
    mode = str(export_mode).lower()
    if mode == "core":
        final_field = core_sdf
    elif mode == "skin":
        final_field = skin_sdf
    elif mode == "combined":
        final_field = np.minimum(core_sdf, skin_sdf)
    else:
        raise ValueError("export_mode must be 'core', 'skin', or 'combined'")

    t0 = time.perf_counter()
    verts, faces, _normals, _values = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(float(resolution), float(resolution), float(resolution)),
    )
    t_mc = time.perf_counter() - t0
    verts = verts + padded_min_bound
    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path is not None:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    print(
        f"Field-driven {lattice_type}: res={resolution}mm, "
        f"unit_cell={'graded' if grade_unit_cell else base_unit_cell_size}, "
        f"density={density_mode}{' graded' if grade_solid_fraction else ''}, "
        f"marching_cubes={t_mc:.2f}s, faces={len(mesh_out.faces):,}"
    )
    return mesh_out
