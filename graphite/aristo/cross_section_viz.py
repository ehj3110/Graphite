"""
Cross-section stress plots from Aristo VTU nodal fields.

Fill modes:
- ``element-slice``: rasterize tet faces cut by the slice plane; void = white.
- ``element-slice-soft``: element-slice + masked Gaussian blur (radius in mm).
- ``voronoi-sharp`` (default): true 2D Voronoi cells, clipped to scaffold wall mask.
- ``voronoi-soft``: sharp Voronoi + light Gaussian (half default blur radius).
- ``voronoi-bounded``: Voronoi nearest-neighbor fill clipped to an STL slice (sharp edges).
- ``voronoi-bounded-soft``: bounded fill + light Gaussian inside the geometry mask.
- ``blur-heatmap``: Gaussian neighborhood average of nodal values (blur radius in mm).
- ``voronoi-disk``: legacy nearest-neighbor + radial disk cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.spatial import ConvexHull, cKDTree

NODAL_VM_KEY = "von_mises_nodal_MPa"
ELEMENT_VM_KEY = "von_mises_element_MPa"
ELEMENT_QUALITY_KEY = "quality_ok"
FILL_MODE_ELEMENT_SLICE = "element-slice"
FILL_MODE_ELEMENT_SLICE_SOFT = "element-slice-soft"
FILL_MODE_VORONOI_SHARP = "voronoi-sharp"
FILL_MODE_VORONOI_SOFT = "voronoi-soft"
FILL_MODE_VORONOI_BOUNDED = "voronoi-bounded"
FILL_MODE_VORONOI_BOUNDED_SOFT = "voronoi-bounded-soft"
FILL_MODE_BLUR_HEATMAP = "blur-heatmap"
FILL_MODE_VORONOI_DISK = "voronoi-disk"
FILL_MODES = (
    FILL_MODE_ELEMENT_SLICE,
    FILL_MODE_ELEMENT_SLICE_SOFT,
    FILL_MODE_VORONOI_SHARP,
    FILL_MODE_VORONOI_SOFT,
    FILL_MODE_VORONOI_BOUNDED,
    FILL_MODE_VORONOI_BOUNDED_SOFT,
    FILL_MODE_BLUR_HEATMAP,
    FILL_MODE_VORONOI_DISK,
)
DEFAULT_RASTER_PIXELS = 1200
DEFAULT_FIGURE_DPI = 300
DEFAULT_BLUR_RADIUS_MM = 0.018
DEFAULT_SOFT_BLUR_RADIUS_MM = 0.015


def load_nodal_von_mises(vtu_path: Path) -> tuple[np.ndarray, np.ndarray]:
    grid = pv.read(str(vtu_path))
    pts = np.asarray(grid.points, dtype=np.float64)
    if NODAL_VM_KEY not in grid.point_data:
        raise KeyError(f"{NODAL_VM_KEY} missing in {vtu_path}")
    vm = np.asarray(grid.point_data[NODAL_VM_KEY], dtype=np.float64)
    return pts, vm


def load_tet_von_mises(
    vtu_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load node coordinates, tet connectivity, element von Mises, and optional quality mask."""
    grid = pv.read(str(vtu_path))
    nodes = np.asarray(grid.points, dtype=np.float64)
    if ELEMENT_VM_KEY not in grid.cell_data:
        raise KeyError(f"{ELEMENT_VM_KEY} missing in {vtu_path}")
    cells = np.asarray(grid.cells, dtype=np.int64).reshape(-1, 5)
    if cells.shape[1] != 5 or not np.all(cells[:, 0] == 4):
        raise ValueError(f"Expected linear tetrahedra in {vtu_path}")
    elements = cells[:, 1:5]
    element_vm = np.asarray(grid.cell_data[ELEMENT_VM_KEY], dtype=np.float64)
    if elements.shape[0] != element_vm.shape[0]:
        raise ValueError("Tet connectivity / element stress length mismatch.")
    quality_ok: np.ndarray | None = None
    if ELEMENT_QUALITY_KEY in grid.cell_data:
        quality_ok = np.asarray(grid.cell_data[ELEMENT_QUALITY_KEY], dtype=bool)
    return nodes, elements, element_vm, quality_ok


def _plane_axis_indices(plane: str) -> tuple[int, int, int]:
    """Return in-plane axes ``(a, b, slice)`` for ``xz``, ``xy``, or ``yz``."""
    if plane == "xz":
        return 0, 2, 1
    if plane == "xy":
        return 0, 1, 2
    if plane == "yz":
        return 1, 2, 0
    raise ValueError(f"Unknown plane {plane!r}")


def _slice_center_and_half(
    plane: str,
    *,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
) -> tuple[float, float]:
    if plane == "xz":
        return float(y_center), float(y_half_thickness)
    if plane == "xy":
        return float(z_center), float(z_half_thickness)
    if plane == "yz":
        return float(x_center), float(x_half_thickness)
    raise ValueError(f"Unknown plane {plane!r}")


def _tet_plane_polygon_ab(
    verts: np.ndarray,
    *,
    slice_axis: int,
    slice_coord: float,
    a_axis: int,
    b_axis: int,
    eps: float = 1e-9,
) -> np.ndarray | None:
    """Intersection polygon of a tet with a plane, projected to AB coordinates."""
    coords = verts[:, slice_axis]
    lo, hi = float(coords.min()), float(coords.max())
    if slice_coord < lo - eps or slice_coord > hi + eps:
        return None

    points_3d: list[np.ndarray] = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        ci, cj = float(coords[i]), float(coords[j])
        vi, vj = verts[i], verts[j]
        if abs(ci - slice_coord) <= eps:
            points_3d.append(vi)
        if abs(cj - slice_coord) <= eps:
            points_3d.append(vj)
        if (ci - slice_coord) * (cj - slice_coord) < -eps * eps:
            t = (slice_coord - ci) / (cj - ci)
            points_3d.append(vi + t * (vj - vi))

    if not points_3d:
        return None

    pts_ab = np.asarray([p[[a_axis, b_axis]] for p in points_3d], dtype=np.float64)
    pts_ab = np.unique(np.round(pts_ab, 12), axis=0)
    if pts_ab.shape[0] < 3:
        return None
    if pts_ab.shape[0] == 3:
        return pts_ab
    hull = ConvexHull(pts_ab)
    return pts_ab[hull.vertices]


def element_slice_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_vm: np.ndarray,
    *,
    plane: str,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
    quality_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Color each raster pixel from the tet that contains the slice point.

    Pixels with no intersecting tet remain NaN (white). Uses the mid-slab plane
    coordinate and tets whose vertices overlap the slab thickness.
    """
    a_axis, b_axis, slice_axis = _plane_axis_indices(plane)
    slice_coord, slice_half = _slice_center_and_half(
        plane,
        x_center=x_center,
        y_center=y_center,
        z_center=z_center,
        x_half_thickness=x_half_thickness,
        y_half_thickness=y_half_thickness,
        z_half_thickness=z_half_thickness,
    )

    tet_coords = nodes[elements][:, :, slice_axis]
    in_slab = (tet_coords.min(axis=1) <= slice_coord + slice_half + 1e-9) & (
        tet_coords.max(axis=1) >= slice_coord - slice_half - 1e-9
    )
    tet_ids = np.nonzero(in_slab)[0]
    if tet_ids.size == 0:
        a_lin = np.linspace(0.0, 1.0, n_a)
        b_lin = np.linspace(0.0, 1.0, n_b)
        empty = np.full((n_b, n_a), np.nan, dtype=np.float64)
        return *np.meshgrid(a_lin, b_lin), empty, empty

    if domain_clip_mm is not None:
        a_lo, a_hi, b_lo, b_hi = domain_clip_mm
        a_lin = np.linspace(float(a_lo), float(a_hi), n_a)
        b_lin = np.linspace(float(b_lo), float(b_hi), n_b)
    else:
        verts_ab = nodes[:, [a_axis, b_axis]]
        pad_a = 0.04 * max(float(verts_ab[:, 0].max() - verts_ab[:, 0].min()), 1e-6)
        pad_b = 0.04 * max(float(verts_ab[:, 1].max() - verts_ab[:, 1].min()), 1e-6)
        a_lin = np.linspace(float(verts_ab[:, 0].min() - pad_a), float(verts_ab[:, 0].max() + pad_a), n_a)
        b_lin = np.linspace(float(verts_ab[:, 1].min() - pad_b), float(verts_ab[:, 1].max() + pad_b), n_b)

    A, B = np.meshgrid(a_lin, b_lin)
    field_sum = np.zeros((n_b, n_a), dtype=np.float64)
    field_count = np.zeros((n_b, n_a), dtype=np.float64)

    for tet_id in tet_ids:
        if quality_mask is not None and not bool(quality_mask[int(tet_id)]):
            continue
        poly_ab = _tet_plane_polygon_ab(
            nodes[elements[int(tet_id)]],
            slice_axis=slice_axis,
            slice_coord=slice_coord,
            a_axis=a_axis,
            b_axis=b_axis,
        )
        if poly_ab is None or poly_ab.shape[0] < 3:
            continue

        a_min, a_max = float(poly_ab[:, 0].min()), float(poly_ab[:, 0].max())
        b_min, b_max = float(poly_ab[:, 1].min()), float(poly_ab[:, 1].max())
        ia0 = max(int(np.searchsorted(a_lin, a_min, side="left")) - 1, 0)
        ia1 = min(int(np.searchsorted(a_lin, a_max, side="right")) + 1, n_a)
        ib0 = max(int(np.searchsorted(b_lin, b_min, side="left")) - 1, 0)
        ib1 = min(int(np.searchsorted(b_lin, b_max, side="right")) + 1, n_b)
        if ia1 <= ia0 or ib1 <= ib0:
            continue

        sub_a = a_lin[ia0:ia1]
        sub_b = b_lin[ib0:ib1]
        sub_a_grid, sub_b_grid = np.meshgrid(sub_a, sub_b)
        sub_pts = np.column_stack([sub_a_grid.ravel(), sub_b_grid.ravel()])
        inside = MplPath(poly_ab).contains_points(sub_pts).reshape(ib1 - ib0, ia1 - ia0)
        if not inside.any():
            continue
        stress = float(element_vm[int(tet_id)])
        sum_block = field_sum[ib0:ib1, ia0:ia1]
        count_block = field_count[ib0:ib1, ia0:ia1]
        sum_block[inside] += stress
        count_block[inside] += 1.0
        field_sum[ib0:ib1, ia0:ia1] = sum_block
        field_count[ib0:ib1, ia0:ia1] = count_block

    covered = field_count > 0.0
    field = np.full((n_b, n_a), np.nan, dtype=np.float64)
    field[covered] = field_sum[covered] / field_count[covered]
    material = covered.astype(np.float64)

    if domain_clip_mm is not None:
        field, material = _apply_domain_clip(A, B, field, material, domain_clip_mm)

    return A, B, field, material


def _close_slice_raster_gaps(
    field: np.ndarray,
    material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill 1-pixel cracks between adjacent tet footprints before blur."""
    filled = material > 0.0
    if not np.any(filled):
        return field, material

    dilated = binary_dilation(filled, iterations=1)
    gaps = dilated & ~filled
    if not np.any(gaps):
        return field, material

    _, indices = distance_transform_edt(~filled, return_distances=True, return_indices=True)
    out_field = field.copy()
    out_material = material.copy()
    out_field[gaps] = field[indices[0][gaps], indices[1][gaps]]
    out_material[gaps] = 1.0
    return out_field, out_material


def _masked_gaussian_blur_field(
    A: np.ndarray,
    B: np.ndarray,
    field: np.ndarray,
    material: np.ndarray,
    *,
    blur_radius_mm: float,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked Gaussian blur confined to the material region."""
    if blur_radius_mm <= 0.0 or not np.any(material > 0.0):
        return A, B, field, material

    n_b, n_a = field.shape
    pixel_a = float(A.max() - A.min()) / max(n_a - 1, 1)
    pixel_b = float(B.max() - B.min()) / max(n_b - 1, 1)
    sigma_a = float(blur_radius_mm) / max(pixel_a, 1e-12)
    sigma_b = float(blur_radius_mm) / max(pixel_b, 1e-12)

    mat_f = material.astype(np.float64)
    weighted = np.where(material > 0.0, field, 0.0)
    num = gaussian_filter(weighted, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)
    den = gaussian_filter(mat_f, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)

    blurred = np.full_like(field, np.nan, dtype=np.float64)
    inside = material > 0.0
    valid = inside & (den > 1e-12)
    blurred[valid] = num[valid] / den[valid]
    if domain_clip_mm is not None:
        blurred, material = _apply_domain_clip(A, B, blurred, material, domain_clip_mm)
    return A, B, blurred, material


def element_slice_soft_field(
    nodes: np.ndarray,
    elements: np.ndarray,
    element_vm: np.ndarray,
    *,
    plane: str,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
    quality_mask: np.ndarray | None = None,
    blur_radius_mm: float = DEFAULT_SOFT_BLUR_RADIUS_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Element-slice raster with masked Gaussian blur.

    Blur sigma is specified in millimeters (physical distance), then converted to
    pixels from the raster spacing. Overlapping tets at a pixel are averaged before blur.
    """
    A, B, field, material = element_slice_field(
        nodes,
        elements,
        element_vm,
        plane=plane,
        x_center=x_center,
        y_center=y_center,
        z_center=z_center,
        x_half_thickness=x_half_thickness,
        y_half_thickness=y_half_thickness,
        z_half_thickness=z_half_thickness,
        n_a=n_a,
        n_b=n_b,
        domain_clip_mm=domain_clip_mm,
        quality_mask=quality_mask,
    )
    field, material = _close_slice_raster_gaps(field, material)
    return _masked_gaussian_blur_field(
        A,
        B,
        field,
        material,
        blur_radius_mm=float(blur_radius_mm),
        domain_clip_mm=domain_clip_mm,
    )


def global_nodal_peak_from_report(report_path: Path | None) -> float | None:
    if report_path is None or not report_path.is_file():
        return None
    data = json.loads(report_path.read_text(encoding="utf-8"))
    peak = float(data.get("max_von_mises_nodal_MPa", np.nan))
    return peak if np.isfinite(peak) else None


def dedupe_in_plane_max(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    round_decimals: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.round(np.column_stack([coord_a, coord_b]), round_decimals)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros(uniq.shape[0], dtype=np.float64)
    np.maximum.at(out, inverse, values)
    return uniq[:, 0], uniq[:, 1], out


def extract_slice_nodes(
    pts: np.ndarray,
    vm: np.ndarray,
    *,
    plane: str,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if plane == "xz":
        mask = np.abs(pts[:, 1] - y_center) <= y_half_thickness
        return pts[mask, 0], pts[mask, 2], vm[mask]
    if plane == "xy":
        mask = np.abs(pts[:, 2] - z_center) <= z_half_thickness
        return pts[mask, 0], pts[mask, 1], vm[mask]
    if plane == "yz":
        mask = np.abs(pts[:, 0] - x_center) <= x_half_thickness
        return pts[mask, 1], pts[mask, 2], vm[mask]
    raise ValueError(f"Unknown plane {plane!r}")


def _scaffold_material_mask(
    A: np.ndarray,
    B: np.ndarray,
    seeds: np.ndarray,
    *,
    void_distance_mm: float,
    voronoi_scale: float,
) -> np.ndarray:
    """True where a raster point lies inside the scaffold wall envelope (not void)."""
    tree = cKDTree(seeds)
    grid_pts = np.column_stack([A.ravel(), B.ravel()])
    dist, idx = tree.query(grid_pts, k=1)
    n_b, n_a = A.shape
    dist = dist.reshape(n_b, n_a)
    idx = idx.reshape(n_b, n_a)

    cap = np.full(seeds.shape[0], float(void_distance_mm), dtype=np.float64)
    if voronoi_scale > 0.0:
        nn_dist, _ = tree.query(seeds, k=2)
        cap = np.minimum(cap, float(voronoi_scale) * nn_dist[:, 1])

    return dist <= cap[idx]


def _apply_domain_clip(
    A: np.ndarray,
    B: np.ndarray,
    field: np.ndarray,
    material: np.ndarray,
    domain_clip_mm: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Mask raster pixels outside ``(a_min, a_max, b_min, b_max)``."""
    a_lo, a_hi, b_lo, b_hi = domain_clip_mm
    outside = (
        (A < float(a_lo) - 1e-12)
        | (A > float(a_hi) + 1e-12)
        | (B < float(b_lo) - 1e-12)
        | (B > float(b_hi) + 1e-12)
    )
    field_out = field.copy()
    material_out = material.copy()
    field_out[outside] = np.nan
    material_out[outside] = 0.0
    return field_out, material_out


def _raster_axis_limits(
    A: np.ndarray,
    B: np.ndarray,
    *,
    domain_clip_mm: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if domain_clip_mm is not None:
        return domain_clip_mm
    pad_a = 0.02 * max(float(A.max() - A.min()), 1e-6)
    pad_b = 0.02 * max(float(B.max() - B.min()), 1e-6)
    return (
        float(A.min()) - pad_a,
        float(A.max()) + pad_a,
        float(B.min()) - pad_b,
        float(B.max()) + pad_b,
    )


def _build_raster_grid(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    *,
    n_a: int,
    n_b: int,
    bbox_padding_fraction: float = 0.04,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if domain_clip_mm is not None:
        a_lo, a_hi, b_lo, b_hi = domain_clip_mm
        a_lin = np.linspace(float(a_lo), float(a_hi), n_a)
        b_lin = np.linspace(float(b_lo), float(b_hi), n_b)
    else:
        a_min, a_max = float(coord_a.min()), float(coord_a.max())
        b_min, b_max = float(coord_b.min()), float(coord_b.max())
        pad_a = bbox_padding_fraction * max(a_max - a_min, 1e-6)
        pad_b = bbox_padding_fraction * max(b_max - b_min, 1e-6)
        a_lin = np.linspace(a_min - pad_a, a_max + pad_a, n_a)
        b_lin = np.linspace(b_min - pad_b, b_max + pad_b, n_b)
    return np.meshgrid(a_lin, b_lin)


def raster_stl_slice_solid_mask(
    stl_path: Path,
    *,
    plane: str,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
    a_lin: np.ndarray,
    b_lin: np.ndarray,
    n_slab_samples: int = 7,
    voxel_target_n: int = 128,
) -> np.ndarray:
    """
    Rasterize solid geometry in a thin slab as a boolean mask on ``(a_lin, b_lin)``.

    Voxelizes the STL at modest resolution (default 128³ max) and unions slab
    voxels, then resamples to the plot raster with nearest-neighbor interpolation
    so strut edges stay axis-aligned.
    """
    from scipy.interpolate import RegularGridInterpolator

    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    n_b = int(len(b_lin))
    n_a = int(len(a_lin))
    pitch = float(mesh.extents.max()) / max(int(voxel_target_n), 8)
    try:
        vox = mesh.voxelized(pitch=pitch, method="subdivide").fill()
    except MemoryError:
        vox = mesh.voxelized(pitch=pitch, method="ray")
        try:
            vox = vox.fill()
        except Exception:
            pass
    matrix = np.asarray(vox.matrix, dtype=bool)
    origin = np.asarray(vox.translation, dtype=np.float64)
    nx, ny, nz = matrix.shape
    x_axis = origin[0] + np.arange(nx, dtype=np.float64) * pitch
    y_axis = origin[1] + np.arange(ny, dtype=np.float64) * pitch
    z_axis = origin[2] + np.arange(nz, dtype=np.float64) * pitch

    if plane == "xz":
        y0 = float(y_center) - float(y_half_thickness)
        y1 = float(y_center) + float(y_half_thickness)
        j0 = int(np.searchsorted(y_axis, y0, side="left"))
        j1 = int(np.searchsorted(y_axis, y1, side="right"))
        j0 = max(j0, 0)
        j1 = min(max(j1, j0 + 1), ny)
        slab2d = matrix[:, j0:j1, :].any(axis=1)
        src_a, src_b = x_axis, z_axis
    elif plane == "xy":
        z0 = float(z_center) - float(z_half_thickness)
        z1 = float(z_center) + float(z_half_thickness)
        k0 = int(np.searchsorted(z_axis, z0, side="left"))
        k1 = int(np.searchsorted(z_axis, z1, side="right"))
        k0 = max(k0, 0)
        k1 = min(max(k1, k0 + 1), nz)
        slab2d = matrix[:, :, k0:k1].any(axis=2)
        src_a, src_b = x_axis, y_axis
    elif plane == "yz":
        x0 = float(x_center) - float(x_half_thickness)
        x1 = float(x_center) + float(x_half_thickness)
        i0 = int(np.searchsorted(x_axis, x0, side="left"))
        i1 = int(np.searchsorted(x_axis, x1, side="right"))
        i0 = max(i0, 0)
        i1 = min(max(i1, i0 + 1), nx)
        slab2d = matrix[i0:i1, :, :].any(axis=0)
        src_a, src_b = y_axis, z_axis
    else:
        raise ValueError(f"Unknown plane {plane!r}")

    interpolator = RegularGridInterpolator(
        (src_a, src_b),
        slab2d.astype(np.float64),
        bounds_error=False,
        fill_value=0.0,
        method="nearest",
    )
    A, B = np.meshgrid(a_lin, b_lin)
    samples = np.column_stack([A.ravel(), B.ravel()])
    return interpolator(samples).reshape(n_b, n_a) > 0.5


def voronoi_bounded_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    solid_mask: np.ndarray,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    bbox_padding_fraction: float = 0.04,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Voronoi nearest-neighbor nodal fill, hard-clipped to a geometry mask.

    Unlike ``voronoi_sharp_field``, does not use ``void_distance_mm``; the STL
    (or other) slice mask supplies the scaffold boundary so strut edges stay sharp
    while interiors are fully filled.
    """
    if coord_a.size == 0:
        empty = np.full((n_b, n_a), np.nan, dtype=np.float64)
        a_lin = np.linspace(0.0, 1.0, n_a)
        b_lin = np.linspace(0.0, 1.0, n_b)
        return *np.meshgrid(a_lin, b_lin), empty, empty

    A, B = _build_raster_grid(
        coord_a,
        coord_b,
        n_a=n_a,
        n_b=n_b,
        bbox_padding_fraction=bbox_padding_fraction,
        domain_clip_mm=domain_clip_mm,
    )
    if solid_mask.shape != A.shape:
        raise ValueError(
            f"solid_mask shape {solid_mask.shape} must match raster grid {A.shape}"
        )

    seeds = np.column_stack([coord_a, coord_b]).astype(np.float64)
    grid_pts = np.column_stack([A.ravel(), B.ravel()])
    _, idx = cKDTree(seeds).query(grid_pts, k=1)
    idx = idx.reshape(n_b, n_a)

    material = solid_mask.astype(bool, copy=False)
    field = values[idx].astype(np.float64, copy=True)
    field[~material] = np.nan

    if domain_clip_mm is not None:
        field, material = _apply_domain_clip(A, B, field, material, domain_clip_mm)

    return A, B, field, material.astype(np.float64)


def voronoi_bounded_soft_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    solid_mask: np.ndarray,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    blur_radius_mm: float = DEFAULT_SOFT_BLUR_RADIUS_MM,
    bbox_padding_fraction: float = 0.04,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bounded Voronoi fill with masked Gaussian smoothing inside the geometry."""
    A, B, field, material = voronoi_bounded_field(
        coord_a,
        coord_b,
        values,
        solid_mask=solid_mask,
        n_a=n_a,
        n_b=n_b,
        bbox_padding_fraction=bbox_padding_fraction,
        domain_clip_mm=domain_clip_mm,
    )
    if coord_a.size == 0 or blur_radius_mm <= 0.0:
        return A, B, field, material

    pixel_a = float(A.max() - A.min()) / max(n_a - 1, 1)
    pixel_b = float(B.max() - B.min()) / max(n_b - 1, 1)
    sigma_a = float(blur_radius_mm) / max(pixel_a, 1e-12)
    sigma_b = float(blur_radius_mm) / max(pixel_b, 1e-12)

    mat_f = material.astype(np.float64)
    weighted = np.where(material, field, 0.0)
    num = gaussian_filter(weighted, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)
    den = gaussian_filter(mat_f, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)

    blurred = np.full_like(field, np.nan, dtype=np.float64)
    inside = material > 0.0
    valid = inside & (den > 1e-12)
    blurred[valid] = num[valid] / den[valid]
    if domain_clip_mm is not None:
        blurred, material = _apply_domain_clip(A, B, blurred, material, domain_clip_mm)
    return A, B, blurred, material


def voronoi_sharp_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    void_distance_mm: float = 0.018,
    voronoi_scale: float = 0.0,
    bbox_padding_fraction: float = 0.04,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    True 2D Voronoi fill with hard scaffold wall mask (void = NaN / white).

    Each seed owns a Voronoi cell with straight bisector boundaries. Pixels are
    colored only where they lie inside **both** the cell and the scaffold wall
    envelope (distance to nearest seed ≤ ``void_distance_mm``).
    """
    if coord_a.size == 0:
        empty = np.full((n_b, n_a), np.nan, dtype=np.float64)
        a_lin = np.linspace(0.0, 1.0, n_a)
        b_lin = np.linspace(0.0, 1.0, n_b)
        return *np.meshgrid(a_lin, b_lin), empty, empty

    seeds = np.column_stack([coord_a, coord_b]).astype(np.float64)

    if domain_clip_mm is not None:
        a_lo, a_hi, b_lo, b_hi = domain_clip_mm
        a_lin = np.linspace(float(a_lo), float(a_hi), n_a)
        b_lin = np.linspace(float(b_lo), float(b_hi), n_b)
    else:
        a_min, a_max = float(coord_a.min()), float(coord_a.max())
        b_min, b_max = float(coord_b.min()), float(coord_b.max())
        pad_a = bbox_padding_fraction * max(a_max - a_min, 1e-6)
        pad_b = bbox_padding_fraction * max(b_max - b_min, 1e-6)
        a_lin = np.linspace(a_min - pad_a, a_max + pad_a, n_a)
        b_lin = np.linspace(b_min - pad_b, b_max + pad_b, n_b)
    A, B = np.meshgrid(a_lin, b_lin)
    grid_pts = np.column_stack([A.ravel(), B.ravel()])

    tree = cKDTree(seeds)
    dist, idx = tree.query(grid_pts, k=1)
    dist = dist.reshape(n_b, n_a)
    idx = idx.reshape(n_b, n_a)

    # Hard scaffold wall: inside iff distance to nearest seed ≤ void_distance_mm.
    material = dist <= float(void_distance_mm)

    # Nearest-seed assignment = Voronoi cell ownership on the raster grid.
    field = values[idx].astype(np.float64, copy=True)
    field[~material] = np.nan

    if domain_clip_mm is not None:
        field, material = _apply_domain_clip(A, B, field, material, domain_clip_mm)

    return A, B, field, material.astype(np.float64)


def blur_heatmap_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    void_distance_mm: float = 0.018,
    voronoi_scale: float = 0.0,
    blur_radius_mm: float = DEFAULT_BLUR_RADIUS_MM,
    bbox_padding_fraction: float = 0.04,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Neighborhood-averaged heatmap with hard scaffold wall mask (void = white).

    Starts from the sharp Voronoi nodal field, then applies a masked Gaussian
    blur so each pixel is a distance-weighted average of nearby nodal values
    within ``blur_radius_mm`` (Gaussian sigma).
    """
    A, B, field, material = voronoi_sharp_field(
        coord_a,
        coord_b,
        values,
        n_a=n_a,
        n_b=n_b,
        void_distance_mm=void_distance_mm,
        voronoi_scale=voronoi_scale,
        bbox_padding_fraction=bbox_padding_fraction,
        domain_clip_mm=domain_clip_mm,
    )
    if coord_a.size == 0 or blur_radius_mm <= 0.0:
        return A, B, field, material

    pixel_a = float(A.max() - A.min()) / max(n_a - 1, 1)
    pixel_b = float(B.max() - B.min()) / max(n_b - 1, 1)
    sigma_a = float(blur_radius_mm) / max(pixel_a, 1e-12)
    sigma_b = float(blur_radius_mm) / max(pixel_b, 1e-12)

    mat_f = material.astype(np.float64)
    weighted = np.where(material, field, 0.0)
    num = gaussian_filter(weighted, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)
    den = gaussian_filter(mat_f, sigma=[sigma_b, sigma_a], mode="constant", cval=0.0)

    blurred = np.full_like(field, np.nan, dtype=np.float64)
    inside = material > 0.0
    valid = inside & (den > 1e-12)
    blurred[valid] = num[valid] / den[valid]
    if domain_clip_mm is not None:
        blurred, material = _apply_domain_clip(A, B, blurred, material, domain_clip_mm)
    return A, B, blurred, material


def raster_cross_section_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    fill_mode: str = FILL_MODE_VORONOI_SHARP,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    void_distance_mm: float = 0.018,
    voronoi_scale: float = 0.0,
    blur_radius_mm: float = DEFAULT_BLUR_RADIUS_MM,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
    solid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch raster fill mode."""
    mode = fill_mode.strip().lower()
    if mode == FILL_MODE_VORONOI_BOUNDED:
        if solid_mask is None:
            raise ValueError("solid_mask is required for fill_mode='voronoi-bounded'")
        return voronoi_bounded_field(
            coord_a,
            coord_b,
            values,
            solid_mask=solid_mask,
            n_a=n_a,
            n_b=n_b,
            domain_clip_mm=domain_clip_mm,
        )
    if mode == FILL_MODE_VORONOI_BOUNDED_SOFT:
        if solid_mask is None:
            raise ValueError("solid_mask is required for fill_mode='voronoi-bounded-soft'")
        return voronoi_bounded_soft_field(
            coord_a,
            coord_b,
            values,
            solid_mask=solid_mask,
            n_a=n_a,
            n_b=n_b,
            blur_radius_mm=DEFAULT_SOFT_BLUR_RADIUS_MM,
            domain_clip_mm=domain_clip_mm,
        )
    if mode == FILL_MODE_VORONOI_SHARP:
        return voronoi_sharp_field(
            coord_a,
            coord_b,
            values,
            n_a=n_a,
            n_b=n_b,
            void_distance_mm=void_distance_mm,
            voronoi_scale=voronoi_scale,
            domain_clip_mm=domain_clip_mm,
        )
    if mode == FILL_MODE_VORONOI_SOFT:
        return blur_heatmap_field(
            coord_a,
            coord_b,
            values,
            n_a=n_a,
            n_b=n_b,
            void_distance_mm=void_distance_mm,
            voronoi_scale=voronoi_scale,
            blur_radius_mm=DEFAULT_SOFT_BLUR_RADIUS_MM,
            domain_clip_mm=domain_clip_mm,
        )
    if mode == FILL_MODE_BLUR_HEATMAP:
        return blur_heatmap_field(
            coord_a,
            coord_b,
            values,
            n_a=n_a,
            n_b=n_b,
            void_distance_mm=void_distance_mm,
            voronoi_scale=voronoi_scale,
            blur_radius_mm=blur_radius_mm,
            domain_clip_mm=domain_clip_mm,
        )
    if mode == FILL_MODE_VORONOI_DISK:
        return voronoi_solid_field(
            coord_a,
            coord_b,
            values,
            n_a=n_a,
            n_b=n_b,
            void_distance_mm=void_distance_mm,
            voronoi_scale=voronoi_scale,
        )
    raise ValueError(f"fill_mode must be one of {FILL_MODES}, got {fill_mode!r}")


def voronoi_solid_field(
    coord_a: np.ndarray,
    coord_b: np.ndarray,
    values: np.ndarray,
    *,
    n_a: int = DEFAULT_RASTER_PIXELS,
    n_b: int = DEFAULT_RASTER_PIXELS,
    void_distance_mm: float = 0.018,
    voronoi_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if coord_a.size == 0:
        empty = np.full((n_b, n_a), np.nan, dtype=np.float64)
        a_lin = np.linspace(0.0, 1.0, n_a)
        b_lin = np.linspace(0.0, 1.0, n_b)
        return *np.meshgrid(a_lin, b_lin), empty, empty

    seeds = np.column_stack([coord_a, coord_b])
    tree = cKDTree(seeds)
    nn_dist, _ = tree.query(seeds, k=2)
    cap = np.full(seeds.shape[0], float(void_distance_mm), dtype=np.float64)
    if voronoi_scale > 0.0:
        cap = np.minimum(cap, float(voronoi_scale) * nn_dist[:, 1])

    a_min, a_max = float(coord_a.min()), float(coord_a.max())
    b_min, b_max = float(coord_b.min()), float(coord_b.max())
    a_lin = np.linspace(a_min, a_max, n_a)
    b_lin = np.linspace(b_min, b_max, n_b)
    A, B = np.meshgrid(a_lin, b_lin)
    dist, idx = tree.query(np.column_stack([A.ravel(), B.ravel()]), k=1)
    dist = dist.reshape(n_b, n_a)
    idx = idx.reshape(n_b, n_a)

    field = values[idx].astype(np.float64, copy=True)
    material = dist <= cap[idx]
    field[~material] = np.nan
    return A, B, field, material.astype(np.float64)


def plot_cross_section_comparison(
    cases: tuple[dict, ...],
    *,
    plane: str,
    x_center: float,
    y_center: float,
    z_center: float,
    x_half_thickness: float,
    y_half_thickness: float,
    z_half_thickness: float,
    output_path: Path,
    cmap: str = "turbo",
    void_distance_mm: float = 0.018,
    voronoi_scale: float = 0.0,
    fill_mode: str = FILL_MODE_VORONOI_SHARP,
    draw_boundary: bool = True,
    raster_pixels: int = DEFAULT_RASTER_PIXELS,
    dpi: int = DEFAULT_FIGURE_DPI,
    blur_radius_mm: float = DEFAULT_BLUR_RADIUS_MM,
    axis_limits: tuple[float, float, float, float] | None = None,
    domain_clip_mm: tuple[float, float, float, float] | None = None,
    suptitle: str | None = None,
) -> None:
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    if plane == "xz":
        xlabel, ylabel = "X (mm)", "Z (mm)"
        slice_type = f"Vertical cross-section (XZ @ Y = {y_center:.3f} mm)"
    elif plane == "xy":
        xlabel, ylabel = "X (mm)", "Y (mm)"
        slice_type = f"Horizontal cross-section (XY @ Z = {z_center:.3f} mm)"
    elif plane == "yz":
        xlabel, ylabel = "Y (mm)", "Z (mm)"
        slice_type = f"Vertical cross-section (YZ @ X = {x_center:.3f} mm)"
    else:
        raise ValueError(f"Unknown plane {plane!r}")

    for ax, case in zip(axes, cases, strict=True):
        vtu_path = Path(case["vtu"])
        mode = fill_mode.strip().lower()
        slice_peak = float("nan")

        if mode in (FILL_MODE_ELEMENT_SLICE, FILL_MODE_ELEMENT_SLICE_SOFT):
            nodes, elements, element_vm, _quality_ok = load_tet_von_mises(vtu_path)
            soft_blur_mm = (
                blur_radius_mm
                if blur_radius_mm != DEFAULT_BLUR_RADIUS_MM
                else DEFAULT_SOFT_BLUR_RADIUS_MM
            )

            if mode == FILL_MODE_ELEMENT_SLICE_SOFT:
                A, B, field, material = element_slice_soft_field(
                    nodes,
                    elements,
                    element_vm,
                    plane=plane,
                    x_center=x_center,
                    y_center=y_center,
                    z_center=z_center,
                    x_half_thickness=x_half_thickness,
                    y_half_thickness=y_half_thickness,
                    z_half_thickness=z_half_thickness,
                    n_a=raster_pixels,
                    n_b=raster_pixels,
                    domain_clip_mm=domain_clip_mm,
                    blur_radius_mm=soft_blur_mm,
                )
            else:
                A, B, field, material = element_slice_field(
                    nodes,
                    elements,
                    element_vm,
                    plane=plane,
                    x_center=x_center,
                    y_center=y_center,
                    z_center=z_center,
                    x_half_thickness=x_half_thickness,
                    y_half_thickness=y_half_thickness,
                    z_half_thickness=z_half_thickness,
                    n_a=raster_pixels,
                    n_b=raster_pixels,
                    domain_clip_mm=domain_clip_mm,
                )
            if np.any(np.isfinite(field)):
                slice_peak = float(np.nanmax(field))
            else:
                ax.set_title(f"{case['label']}\n(no elements in slice)")
                ax.axis("off")
                continue
        else:
            pts, vm = load_nodal_von_mises(vtu_path)
            a, b, vals = extract_slice_nodes(
                pts,
                vm,
                plane=plane,
                x_center=x_center,
                y_center=y_center,
                z_center=z_center,
                x_half_thickness=x_half_thickness,
                y_half_thickness=y_half_thickness,
                z_half_thickness=z_half_thickness,
            )
            if vals.size == 0:
                ax.set_title(f"{case['label']}\n(no nodes in slice)")
                ax.axis("off")
                continue

            a, b, vals = dedupe_in_plane_max(a, b, vals)
            slice_peak = float(np.nanmax(vals))

            solid_mask = case.get("solid_mask")
            if mode in (
                FILL_MODE_VORONOI_BOUNDED,
                FILL_MODE_VORONOI_BOUNDED_SOFT,
            ):
                if solid_mask is None:
                    stl_path = case.get("stl")
                    if stl_path is None:
                        raise ValueError(
                            f"Case {case['label']!r}: fill_mode={fill_mode!r} requires "
                            "'stl' or precomputed 'solid_mask' in the case dict."
                        )
                    A_grid, B_grid = _build_raster_grid(
                        a,
                        b,
                        n_a=raster_pixels,
                        n_b=raster_pixels,
                        domain_clip_mm=domain_clip_mm,
                    )
                    solid_mask = raster_stl_slice_solid_mask(
                        Path(stl_path),
                        plane=plane,
                        x_center=x_center,
                        y_center=y_center,
                        z_center=z_center,
                        x_half_thickness=x_half_thickness,
                        y_half_thickness=y_half_thickness,
                        z_half_thickness=z_half_thickness,
                        a_lin=A_grid[0, :],
                        b_lin=B_grid[:, 0],
                    )

            A, B, field, material = raster_cross_section_field(
                a,
                b,
                vals,
                fill_mode=fill_mode,
                n_a=raster_pixels,
                n_b=raster_pixels,
                void_distance_mm=void_distance_mm,
                voronoi_scale=voronoi_scale,
                blur_radius_mm=blur_radius_mm,
                domain_clip_mm=domain_clip_mm,
                solid_mask=solid_mask,
            )
        if mode in (
            FILL_MODE_BLUR_HEATMAP,
            FILL_MODE_VORONOI_SOFT,
            FILL_MODE_VORONOI_BOUNDED_SOFT,
            FILL_MODE_ELEMENT_SLICE_SOFT,
        ):
            display_peak = (
                float(np.nanmax(field))
                if np.any(np.isfinite(field))
                else slice_peak
            )
        else:
            display_peak = slice_peak

        masked = np.ma.masked_invalid(field / (display_peak + 1e-15))
        mode_lc = fill_mode.strip().lower()
        if mode_lc in (
            FILL_MODE_BLUR_HEATMAP,
            FILL_MODE_VORONOI_SOFT,
            FILL_MODE_VORONOI_BOUNDED_SOFT,
        ):
            interp = "bilinear"
        else:
            interp = "nearest"
        if axis_limits is not None:
            x_lo, x_hi, y_lo, y_hi = axis_limits
        else:
            x_lo, x_hi, y_lo, y_hi = _raster_axis_limits(
                A, B, domain_clip_mm=domain_clip_mm
            )
        ax.imshow(
            masked,
            origin="lower",
            extent=(x_lo, x_hi, y_lo, y_hi),
            aspect="equal",
            cmap=cmap_obj,
            norm=Normalize(vmin=0.0, vmax=1.0),
            interpolation=interp,
        )
        if draw_boundary and np.any(material > 0.0):
            mode_lc = fill_mode.strip().lower()
            if mode_lc not in (FILL_MODE_ELEMENT_SLICE, FILL_MODE_ELEMENT_SLICE_SOFT):
                ax.contour(
                    A,
                    B,
                    material,
                    levels=[0.5],
                    colors=("0.35",),
                    linewidths=0.2,
                    alpha=0.8,
                )

        ax.set_facecolor("white")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"{case['label']}\n"
            f"Slice peak $\\sigma_{{vm}}$ = {slice_peak:.2f} MPa",
            fontsize=10,
        )
        sm = ScalarMappable(cmap=cmap_obj, norm=Normalize(vmin=0.0, vmax=1.0))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=r"Normalized $\sigma_{vm}$")

    if suptitle is None:
        suptitle = f"Normalized von Mises - {slice_type}"
    fig.suptitle(suptitle, fontsize=12, y=1.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
