"""Pure-VTK 3D Studio Viewport & 2D Cross-Section Slice Inspector for
Graphite Implicit Studio.

Zero PyVista dependency: uses VTK directly (vtkPolyData, vtkPolyDataNormals,
vtkClipPolyData, vtkPlane) paired with NumPy/Trimesh evaluation from
``graphite.ui.implicit_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates
import trimesh
import vtk
from vtk.util import numpy_support

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.blending import blend_lattice_with_skin
from graphite.implicit.micropillars import MicropillarConfig, sample_pillar_anchors
from graphite.ui.implicit_state import (
    PREVIEW_MODES,
    ImplicitStudioState,
    evaluate_grading_profile_1d,
    evaluate_implicit_solid_field,
)

# Small LRU cache of (mesh_id, n_faces, bounds_key, spacing) -> (X, Y, Z, cad_sdf, min_bound)
_PREVIEW_SDF_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
# Cache of lightweight CAD SDF volumes for sub-15ms 2D slice evaluation
_SLICE_SDF_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, float]] = {}
# Cache of subdivided exterior CAD surfaces for fast 2D surface representation
_SURFACE_SUBDIV_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}


def trimesh_to_vtk_polydata(mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
    """Convert a ``trimesh.Trimesh`` into a ``vtkPolyData`` via vectorized arrays."""
    poly = vtk.vtkPolyData()
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)

    if len(verts) == 0 or len(faces) == 0:
        return poly

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_support.numpy_to_vtk(verts, deep=True))
    poly.SetPoints(vtk_pts)

    n_faces = len(faces)
    cells_np = np.empty((n_faces, 4), dtype=np.int64)
    cells_np[:, 0] = 3
    cells_np[:, 1:4] = faces
    vtk_cells = vtk.vtkCellArray()
    id_arr = numpy_support.numpy_to_vtkIdTypeArray(cells_np.ravel(), deep=True)
    vtk_cells.SetCells(n_faces, id_arr)
    poly.SetPolys(vtk_cells)
    return poly


def _compute_preview_spacing(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
    *,
    max_voxels: int = 350_000,
) -> float:
    """Choose coarse Flying-Edges preview pitch keeping interactive 3D viewport responsive."""
    prof = evaluate_grading_profile_1d(state, cad_mesh, n_samples=40)
    min_w = float(np.min(prof["w_mm"])) if len(prof["w_mm"]) else state.wall_thickness_mm
    min_L = float(np.min(prof["L_mm"])) if len(prof["L_mm"]) else state.unit_cell_size_mm
    ideal = max(min(min_w / 1.8, min_L / 10.0, 0.24), 0.12)

    ext = np.maximum(np.asarray(cad_mesh.bounds[1] - cad_mesh.bounds[0], float), 0.5)
    # Include 4-voxel EDT padding per side in voxel estimate
    res = float(ideal)
    for _ in range(8):
        nx = int(np.ceil(ext[0] / res)) + 9
        ny = int(np.ceil(ext[1] / res)) + 9
        nz = int(np.ceil(ext[2] / res)) + 9
        if nx * ny * nz <= max_voxels:
            break
        res *= (nx * ny * nz / float(max_voxels)) ** (1.0 / 3.0) * 1.04
    return round(float(res), 4)


def _get_cached_slice_cad_sdf(
    cad_mesh: trimesh.Trimesh,
    *,
    max_voxels: int = 420_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute and cache a compact 3D CAD SDF once per CAD mesh for instant 2D slicing."""
    b = np.asarray(cad_mesh.bounds, dtype=float)
    b_key = tuple(round(float(v), 4) for v in b.ravel())
    key = (len(cad_mesh.vertices), len(cad_mesh.faces), b_key, int(max_voxels))
    hit = _SLICE_SDF_CACHE.get(key)
    if hit is not None:
        return hit

    ext = np.maximum(b[1] - b[0], 0.5)
    res = float(np.clip(np.min(ext) / 64.0, 0.04, 0.50))
    for _ in range(8):
        nx = int(np.ceil(ext[0] / res)) + 9
        ny = int(np.ceil(ext[1] / res)) + 9
        nz = int(np.ceil(ext[2] / res)) + 9
        if nx * ny * nz <= max_voxels:
            break
        res *= (nx * ny * nz / float(max_voxels)) ** (1.0 / 3.0) * 1.04
    res = round(float(res), 4)

    _X, _Y, _Z, cad_sdf, padded_min_bound, _, _, _, _ = voxelize_mesh_and_edt(
        cad_mesh, res
    )
    if len(_SLICE_SDF_CACHE) >= 6:
        _SLICE_SDF_CACHE.clear()
    _SLICE_SDF_CACHE[key] = (
        np.ascontiguousarray(cad_sdf, dtype=np.float32),
        np.asarray(padded_min_bound, dtype=np.float64),
        float(res),
    )
    return _SLICE_SDF_CACHE[key]


def _get_cached_preview_cad_sdf(
    cad_mesh: trimesh.Trimesh,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = np.asarray(cad_mesh.bounds, dtype=float)
    b_key = tuple(round(float(v), 4) for v in b.ravel())
    key = (len(cad_mesh.vertices), len(cad_mesh.faces), b_key, round(float(spacing), 4))
    hit = _PREVIEW_SDF_CACHE.get(key)
    if hit is not None:
        return hit

    X, Y, Z, cad_sdf, padded_min_bound, _, _, _, _ = voxelize_mesh_and_edt(
        cad_mesh, float(spacing)
    )
    if len(_PREVIEW_SDF_CACHE) >= 6:
        _PREVIEW_SDF_CACHE.clear()
    _PREVIEW_SDF_CACHE[key] = (X, Y, Z, cad_sdf, padded_min_bound)
    return _PREVIEW_SDF_CACHE[key]


def _get_cached_subdivided_surface(
    cad_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a conformal, edge-length-bounded CAD exterior mesh for ultra-sharp 2D surface contours.

    Selectively bisects only edges exceeding ``max_edge`` so tiny fillet triangles stay
    untouched while spanning flat-wall CAD triangles refine smoothly down to sub-250 um.
    """
    b = np.asarray(cad_mesh.bounds, dtype=float)
    b_key = tuple(round(float(v), 4) for v in b.ravel())
    key = (len(cad_mesh.vertices), len(cad_mesh.faces), b_key)
    hit = _SURFACE_SUBDIV_CACHE.get(key)
    if hit is not None:
        return hit

    verts = np.asarray(cad_mesh.vertices, dtype=float)
    faces = np.asarray(cad_mesh.faces, dtype=np.int64)
    diag = float(np.linalg.norm(b[1] - b[0]))
    max_edge = max(diag / 135.0, 0.16)
    max_faces = 460_000

    for _ in range(10):
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        u_edges, inv = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        edge_len = np.linalg.norm(verts[u_edges[:, 1]] - verts[u_edges[:, 0]], axis=1)
        need = edge_len > max_edge
        if not np.any(need):
            break

        nf = len(faces)
        f0, f1, f2 = need[inv[:nf]], need[inv[nf : 2 * nf]], need[inv[2 * nf :]]
        cnt = f0.astype(np.int8) + f1.astype(np.int8) + f2.astype(np.int8)
        est_faces = (
            nf
            + int(np.sum(cnt == 1))
            + 2 * int(np.sum(cnt == 2))
            + 3 * int(np.sum(cnt == 3))
        )
        if est_faces > max_faces:
            break

        split_ids = np.flatnonzero(need)
        mid_pts = 0.5 * (verts[u_edges[split_ids, 0]] + verts[u_edges[split_ids, 1]])
        base_v = len(verts)
        verts = np.vstack([verts, mid_pts])
        mid_map = np.full(len(u_edges), -1, dtype=np.int64)
        mid_map[split_ids] = np.arange(base_v, base_v + len(split_ids), dtype=np.int64)
        m0, m1, m2 = mid_map[inv[:nf]], mid_map[inv[nf : 2 * nf]], mid_map[inv[2 * nf :]]
        m_tup = (m0, m1, m2)

        chunks: list[np.ndarray] = []
        if np.any(cnt == 0):
            chunks.append(faces[cnt == 0])
        for rot, c_mask in ((0, f0), (1, f1), (2, f2)):
            sel = (cnt == 1) & c_mask
            if np.any(sel):
                fr = faces[sel]
                mk = m_tup[rot][sel]
                a, b_v, c = fr[:, rot], fr[:, (rot + 1) % 3], fr[:, (rot + 2) % 3]
                chunks.append(
                    np.vstack([np.column_stack([a, mk, c]), np.column_stack([mk, b_v, c])])
                )
        for rot, miss_mask in ((0, ~f1), (1, ~f2), (2, ~f0)):
            sel = (cnt == 2) & miss_mask
            if np.any(sel):
                fr = faces[sel]
                ma = m_tup[rot][sel]
                mb = m_tup[(rot + 2) % 3][sel]
                a, b_v, c = fr[:, rot], fr[:, (rot + 1) % 3], fr[:, (rot + 2) % 3]
                chunks.append(
                    np.vstack(
                        [
                            np.column_stack([a, ma, mb]),
                            np.column_stack([ma, b_v, mb]),
                            np.column_stack([b_v, c, mb]),
                        ]
                    )
                )
        sel3 = cnt == 3
        if np.any(sel3):
            f3 = faces[sel3]
            a, b_v, c = f3[:, 0], f3[:, 1], f3[:, 2]
            x0, x1, x2 = m0[sel3], m1[sel3], m2[sel3]
            chunks.append(
                np.vstack(
                    [
                        np.column_stack([a, x0, x2]),
                        np.column_stack([x0, b_v, x1]),
                        np.column_stack([x2, x1, c]),
                        np.column_stack([x0, x1, x2]),
                    ]
                )
            )
        faces = np.vstack(chunks)

    if len(_SURFACE_SUBDIV_CACHE) >= 6:
        _SURFACE_SUBDIV_CACHE.clear()
    _SURFACE_SUBDIV_CACHE[key] = (
        np.ascontiguousarray(verts, dtype=np.float64),
        np.ascontiguousarray(faces, dtype=np.int64),
    )
    return _SURFACE_SUBDIV_CACHE[key]


def _tag_poly_closed_3d(poly: vtk.vtkPolyData, is_closed_3d: bool) -> vtk.vtkPolyData:
    flag = vtk.vtkIntArray()
    flag.SetName("is_closed_3d")
    flag.InsertNextValue(1 if is_closed_3d else 0)
    poly.GetFieldData().RemoveArray("is_closed_3d")
    poly.GetFieldData().AddArray(flag)
    return poly


def build_2d_surface_footprint_polydata(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
) -> vtk.vtkPolyData:
    """Extract ultra-sharp 2D lattice surface patches clipped directly onto the CAD skin."""
    if len(cad_mesh.faces) == 0 or cad_mesh.bounds is None:
        return vtk.vtkPolyData()

    if state.export_mode == "skin":
        return _tag_poly_closed_3d(trimesh_to_vtk_polydata(cad_mesh), False)

    verts, faces = _get_cached_subdivided_surface(cad_mesh)
    sub_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    poly = trimesh_to_vtk_polydata(sub_mesh)

    vals = evaluate_implicit_solid_field(
        state,
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        cad_mesh,
        spacing=0.05,
    )
    vtk_vals = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(vals, dtype=np.float64), deep=True
    )
    vtk_vals.SetName("solid_sdf")
    poly.GetPointData().SetScalars(vtk_vals)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(poly)
    clipper.SetValue(0.0)
    clipper.InsideOutOn()
    clipper.Update()
    return _tag_poly_closed_3d(clipper.GetOutput(), False)


def build_surface_footprint_polydata(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
    *,
    mode: str | None = None,
) -> vtk.vtkPolyData:
    """Extract either ultra-sharp 2D CAD-surface patches or coarse 3D Flying-Edges isosurface."""
    if len(cad_mesh.faces) == 0 or cad_mesh.bounds is None:
        return vtk.vtkPolyData()

    eff_mode = mode if mode is not None else getattr(state, "preview_mode", PREVIEW_MODES[0])
    if eff_mode not in ("3D Flying Edges (Coarse)", "3D Flying Edges (Fine)"):
        return build_2d_surface_footprint_polydata(state, cad_mesh)

    spacing = _compute_preview_spacing(state, cad_mesh)
    X, Y, Z, cad_sdf, padded_min_bound = _get_cached_preview_cad_sdf(cad_mesh, spacing)

    if state.export_mode == "skin":
        combined_field = np.maximum(
            cad_sdf, -cad_sdf - float(state.shell_thickness_mm)
        )
    else:
        lattice_field = evaluate_implicit_solid_field(
            state, X, Y, Z, cad_mesh, spacing=spacing
        )
        skin_t = (
            float(state.shell_thickness_mm)
            if state.export_mode == "combined"
            else 0.0
        )
        combined_field = blend_lattice_with_skin(
            lattice_field=lattice_field,
            cad_sdf=cad_sdf,
            skin_thickness=skin_t,
            blend_radius=float(state.blend_radius_mm),
            method=state.blend_method,
        )

    # Negate SDF (solid is <= 0 in Graphite) because vtkFlyingEdges3D uses positive=inside winding
    field_f32 = np.ascontiguousarray(-combined_field, dtype=np.float32)
    nx, ny, nz = field_f32.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(float(spacing), float(spacing), float(spacing))
    img.SetOrigin(
        float(padded_min_bound[0]),
        float(padded_min_bound[1]),
        float(padded_min_bound[2]),
    )
    vtk_arr = numpy_support.numpy_to_vtk(field_f32.ravel(order="F"), deep=True)
    vtk_arr.SetName("values")
    img.GetPointData().SetScalars(vtk_arr)

    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOn()
    fe.Update()
    return _tag_poly_closed_3d(fe.GetOutput(), True)


def vtk_polydata_to_trimesh(poly: vtk.vtkPolyData) -> trimesh.Trimesh:
    """Convert a triangle ``vtkPolyData`` back into a lightweight ``trimesh.Trimesh``."""
    if poly is None or poly.GetNumberOfCells() == 0 or poly.GetNumberOfPoints() == 0:
        return trimesh.Trimesh()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    tpoly = tri.GetOutput()
    verts = numpy_support.vtk_to_numpy(tpoly.GetPoints().GetData())
    conn = numpy_support.vtk_to_numpy(tpoly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(conn, dtype=np.int64), process=False)


def build_micropillar_preview_polydata(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
    *,
    footprint_poly: vtk.vtkPolyData | None = None,
    max_pillars: int = 240,
) -> vtk.vtkPolyData:
    """Build lightweight oriented cylinder glyphs previewing micropillar anchors."""
    if not state.enable_micropillars or cad_mesh is None or len(cad_mesh.faces) == 0:
        return vtk.vtkPolyData()

    target_mesh = cad_mesh
    if footprint_poly is not None and footprint_poly.GetNumberOfCells() > 0:
        candidate = vtk_polydata_to_trimesh(footprint_poly)
        if len(candidate.faces) > 0:
            target_mesh = candidate

    # Keep preview legible even for sub-100 um pillars by enforcing tiny visual minimum thickness
    diam_mm = max(float(state.pillar_diameter_um) * 1e-3, 0.04)
    height_mm = max(float(state.pillar_height_um) * 1e-3, 0.08)
    spacing_mm = max(float(state.pillar_spacing_um) * 1e-3, diam_mm * 1.1)

    cfg = MicropillarConfig(
        diameter_mm=diam_mm,
        height_mm=height_mm,
        spacing_mm=spacing_mm,
        distribution=state.pillar_distribution,
        location=state.pillar_location,
        filter_printable=bool(state.pillar_filter_printable),
        max_pillars=int(max_pillars),
        boundary_mesh=cad_mesh,
    )
    pts, norms = sample_pillar_anchors(target_mesh, cfg, boundary_mesh=cad_mesh)
    if len(pts) == 0:
        return vtk.vtkPolyData()
    if len(pts) > max_pillars:
        idx = np.linspace(0, len(pts) - 1, max_pillars, dtype=int)
        pts = pts[idx]
        norms = norms[idx]

    # Vectorized 8-sided cylinder prism assembly for sub-10ms interactive preview
    n_sides = 8
    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)[None, :]
    sin_a = np.sin(angles)[None, :]
    radius = 0.5 * diam_mm

    n_pillars = len(pts)
    norms = norms / np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-12)
    ref = np.where(np.abs(norms[:, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
    t1 = np.cross(norms, ref)
    t1 = t1 / np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(norms, t1)

    offsets = radius * (
        t1[:, None, :] * cos_a[:, :, None] + t2[:, None, :] * sin_a[:, :, None]
    )  # (N, 8, 3)
    base_ring = pts[:, None, :] + offsets
    top_ring = base_ring + (norms * height_mm)[:, None, :]
    verts = np.concatenate([base_ring, top_ring], axis=1).reshape(-1, 3)

    faces_list: list[np.ndarray] = []
    stride = 2 * n_sides
    base_idx = (np.arange(n_pillars, dtype=np.int64) * stride)[:, None]
    for k in range(n_sides):
        kn = (k + 1) % n_sides
        b0, b1 = base_idx + k, base_idx + kn
        t0, t1_i = base_idx + n_sides + k, base_idx + n_sides + kn
        faces_list.append(np.hstack([b0, b1, t1_i]))
        faces_list.append(np.hstack([b0, t1_i, t0]))
    for k in range(1, n_sides - 1):
        faces_list.append(np.hstack([base_idx + n_sides, base_idx + n_sides + k, base_idx + n_sides + k + 1]))

    all_faces = np.vstack(faces_list)
    pillar_mesh = trimesh.Trimesh(vertices=verts, faces=all_faces, process=False)
    return trimesh_to_vtk_polydata(pillar_mesh)


class ImplicitVtkScene:
    """Manages persistent VTK actors, dark studio lighting, and cutaway slicing."""

    def __init__(self, renderer: vtk.vtkRenderer) -> None:
        self.renderer = renderer
        self.renderer.SetBackground(0.020, 0.028, 0.045)  # #05070b Midnight Slate
        self.renderer.SetBackground2(0.043, 0.067, 0.118)  # #0b111e gradient
        self.renderer.GradientBackgroundOn()

        self._bounds: tuple[float, float, float, float, float, float] = (-10, 10, -10, 10, -10, 10)
        self._cad_opacity: float = 1.0

        # Shared cutaway clipping plane & raw unclipped polydata caches
        self._clip_plane = vtk.vtkPlane()
        self._clip_enabled = False
        self._clip_axis = "Z"
        self._clip_pct = 50.0
        self._cad_raw_poly: vtk.vtkPolyData | None = None
        self._cad_cap_poly: vtk.vtkPolyData | None = None
        self._footprint_raw_poly: vtk.vtkPolyData | None = None
        self._footprint_is_closed_3d: bool = False
        self._mesh_raw_poly: vtk.vtkPolyData | None = None

        # Prevent z-fighting when 2D surface lattice patches sit on opaque gray CAD
        vtk.vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

        # Actors
        self._grid_actor: vtk.vtkActor | None = None
        self._cad_mapper, self._cad_actor = self._make_shaded_actor(
            color=(0.36, 0.42, 0.50), opacity=self._cad_opacity
        )
        self._cad_actor.GetProperty().SetSpecular(0.16)
        self._cad_actor.GetProperty().SetSpecularPower(16.0)

        self._footprint_mapper, self._footprint_actor = self._make_shaded_actor(
            color=(0.0, 0.96, 0.83), opacity=1.0  # #00f5d4 glowing cyan (opaque depth writes)
        )
        self._footprint_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2.0, -2.0)
        self._footprint_actor.GetProperty().SetAmbient(0.32)
        self._footprint_actor.GetProperty().SetDiffuse(0.75)

        self._mesh_mapper, self._mesh_actor = self._make_shaded_actor(
            color=(0.22, 0.74, 0.97), opacity=1.0  # #38bdf8 sky-blue lattice
        )
        self._mesh_actor.VisibilityOff()

        self._pillar_mapper, self._pillar_actor = self._make_shaded_actor(
            color=(0.98, 0.75, 0.14), opacity=0.95  # #fbbf24 amber gold pillars
        )
        self._pillar_actor.VisibilityOff()

        self._cut_cap_plane = vtk.vtkPlaneSource()
        self._cut_cap_mapper = vtk.vtkPolyDataMapper()
        self._cut_cap_mapper.ScalarVisibilityOff()
        self._cut_cap_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-1.0, -1.0)
        self._cut_cap_tex = vtk.vtkTexture()
        self._cut_cap_tex.InterpolateOff()
        self._cut_cap_tex.EdgeClampOn()
        self._cut_cap_actor = vtk.vtkActor()
        self._cut_cap_actor.SetMapper(self._cut_cap_mapper)
        self._cut_cap_actor.SetTexture(self._cut_cap_tex)
        self._cut_cap_actor.GetProperty().LightingOff()
        self._cut_cap_actor.PickableOff()
        self._cut_cap_actor.VisibilityOff()

        self._cut_frame_mapper = vtk.vtkPolyDataMapper()
        self._cut_frame_actor = vtk.vtkActor()
        self._cut_frame_actor.SetMapper(self._cut_frame_mapper)
        self._cut_frame_actor.GetProperty().SetColor(0.22, 0.74, 0.97)
        self._cut_frame_actor.GetProperty().SetLineWidth(1.6)
        self._cut_frame_actor.GetProperty().SetOpacity(0.85)
        self._cut_frame_actor.PickableOff()
        self._cut_frame_actor.VisibilityOff()

        self._corner_hud = vtk.vtkTextActor()
        self._corner_hud.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self._corner_hud.SetPosition(0.985, 0.015)
        tp = self._corner_hud.GetTextProperty()
        tp.SetColor(0.58, 0.64, 0.72)
        tp.SetFontFamilyToCourier()
        tp.SetFontSize(12)
        tp.SetJustificationToRight()
        tp.SetVerticalJustificationToBottom()
        tp.SetShadow(False)
        self._axes_widget: vtk.vtkOrientationMarkerWidget | None = None

        self.renderer.AddActor(self._cad_actor)
        self.renderer.AddActor(self._footprint_actor)
        self.renderer.AddActor(self._mesh_actor)
        self.renderer.AddActor(self._pillar_actor)
        self.renderer.AddActor(self._cut_cap_actor)
        self.renderer.AddActor(self._cut_frame_actor)
        self.renderer.AddActor2D(self._corner_hud)

        self._setup_studio_lights()

    def attach_interactor(self, interactor: vtk.vtkRenderWindowInteractor) -> None:
        """Configure trackball camera dragging and attach a Midnight-Slate orientation triad."""
        if interactor is None:
            return
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        if self._axes_widget is not None:
            return
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.035)
        axes.SetConeRadius(0.14)
        axes.SetNormalizedTipLength(0.28, 0.28, 0.28)
        for caption in (
            axes.GetXAxisCaptionActor2D(),
            axes.GetYAxisCaptionActor2D(),
            axes.GetZAxisCaptionActor2D(),
        ):
            caption.GetCaptionTextProperty().SetFontSize(11)
            caption.GetCaptionTextProperty().SetShadow(False)
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(interactor)
        widget.SetViewport(0.0, 0.0, 0.16, 0.18)
        widget.SetEnabled(1)
        widget.InteractiveOff()
        self._axes_widget = widget

    def update_corner_hud(self, text: str) -> None:
        self._corner_hud.SetInput(text)

    def _setup_studio_lights(self) -> None:
        self.renderer.RemoveAllLights()
        key = vtk.vtkLight()
        key.SetPosition(1.5, 1.2, 2.0)
        key.SetIntensity(0.90)
        key.SetLightTypeToCameraLight()
        fill = vtk.vtkLight()
        fill.SetPosition(-1.5, -0.8, 0.8)
        fill.SetIntensity(0.45)
        fill.SetColor(0.75, 0.88, 1.0)
        fill.SetLightTypeToCameraLight()
        rim = vtk.vtkLight()
        rim.SetPosition(0.0, 1.0, -1.8)
        rim.SetIntensity(0.35)
        rim.SetLightTypeToCameraLight()
        self.renderer.AddLight(key)
        self.renderer.AddLight(fill)
        self.renderer.AddLight(rim)

    @staticmethod
    def _make_shaded_actor(
        color: tuple[float, float, float],
        opacity: float,
    ) -> tuple[vtk.vtkPolyDataMapper, vtk.vtkActor]:
        mapper = vtk.vtkPolyDataMapper()
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color)
        prop.SetOpacity(opacity)
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(24.0)
        return mapper, actor

    @staticmethod
    def _with_normals(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
        if poly.GetNumberOfCells() == 0:
            return poly
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(poly)
        norms.ComputePointNormalsOn()
        norms.SplittingOn()
        norms.SetFeatureAngle(50.0)
        norms.Update()
        return norms.GetOutput()

    def _rebuild_floor_grid(self) -> None:
        if self._grid_actor is not None:
            self.renderer.RemoveActor(self._grid_actor)
            self._grid_actor = None

        xmin, xmax, ymin, ymax, zmin, _ = self._bounds
        span = max(xmax - xmin, ymax - ymin, 10.0)
        half = 0.9 * span
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        z_floor = zmin - 0.05 * span

        n_lines = 16
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        idx = 0
        for i in range(n_lines + 1):
            t = -half + (2.0 * half) * (i / n_lines)
            pts.InsertNextPoint(cx + t, cy - half, z_floor)
            pts.InsertNextPoint(cx + t, cy + half, z_floor)
            line1 = vtk.vtkLine()
            line1.GetPointIds().SetId(0, idx)
            line1.GetPointIds().SetId(1, idx + 1)
            lines.InsertNextCell(line1)
            idx += 2

            pts.InsertNextPoint(cx - half, cy + t, z_floor)
            pts.InsertNextPoint(cx + half, cy + t, z_floor)
            line2 = vtk.vtkLine()
            line2.GetPointIds().SetId(0, idx)
            line2.GetPointIds().SetId(1, idx + 1)
            lines.InsertNextCell(line2)
            idx += 2

        grid_poly = vtk.vtkPolyData()
        grid_poly.SetPoints(pts)
        grid_poly.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(grid_poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.15, 0.22, 0.34)
        actor.GetProperty().SetOpacity(0.55)
        actor.GetProperty().SetLineWidth(1.0)
        actor.PickableOff()
        self._grid_actor = actor
        self.renderer.AddActor(actor)

    def set_cad_mesh(self, mesh: trimesh.Trimesh, *, reset_cam: bool = True) -> None:
        if mesh.is_watertight and float(mesh.volume) < 0.0:
            mesh = mesh.copy()
            mesh.invert()
        poly = self._with_normals(trimesh_to_vtk_polydata(mesh))
        self._cad_raw_poly = poly
        self._cad_mapper.SetInputData(poly)
        if mesh.bounds is not None:
            b0, b1 = mesh.bounds[0], mesh.bounds[1]
            self._bounds = (float(b0[0]), float(b1[0]), float(b0[1]), float(b1[1]), float(b0[2]), float(b1[2]))
        self._rebuild_floor_grid()
        self._apply_clipping()
        if reset_cam:
            self.set_camera_preset("iso")

    def set_surface_footprint(self, poly: vtk.vtkPolyData) -> None:
        flag_arr = poly.GetFieldData().GetArray("is_closed_3d") if poly is not None else None
        self._footprint_is_closed_3d = bool(flag_arr is not None and flag_arr.GetTuple1(0) > 0.5)
        if poly.GetPointData().GetNormals() is not None:
            norm_poly = poly
        else:
            norm_poly = self._with_normals(poly)
        self._footprint_raw_poly = norm_poly
        self._footprint_mapper.SetInputData(norm_poly)
        self._apply_clipping()

    def set_footprint_visible(self, visible: bool) -> None:
        self._footprint_actor.SetVisibility(1 if visible else 0)

    def set_cad_opacity(self, opacity: float) -> None:
        self._cad_opacity = float(np.clip(opacity, 0.0, 1.0))
        self._cad_actor.SetVisibility(1 if self._cad_opacity > 0.01 else 0)
        self._cad_actor.GetProperty().SetOpacity(self._cad_opacity)

    def set_generated_mesh(self, mesh: trimesh.Trimesh | None, *, visible: bool) -> None:
        if mesh is None or len(mesh.faces) == 0:
            self._mesh_raw_poly = None
            self._mesh_mapper.SetInputData(vtk.vtkPolyData())
            self._mesh_actor.VisibilityOff()
            return
        if mesh.is_watertight and float(mesh.volume) < 0.0:
            mesh = mesh.copy()
            mesh.invert()
        poly = self._with_normals(trimesh_to_vtk_polydata(mesh))
        self._mesh_raw_poly = poly
        self._mesh_mapper.SetInputData(poly)
        self._mesh_actor.SetVisibility(1 if visible else 0)
        self._apply_clipping()

    def set_generated_mesh_visible(self, visible: bool) -> None:
        has_data = (
            self._mesh_raw_poly is not None
            and self._mesh_raw_poly.GetNumberOfCells() > 0
        )
        self._mesh_actor.SetVisibility(1 if (visible and has_data) else 0)

    def set_micropillar_preview(self, poly: vtk.vtkPolyData | None) -> None:
        if poly is None or poly.GetNumberOfCells() == 0:
            self._pillar_mapper.SetInputData(vtk.vtkPolyData())
            self._pillar_actor.VisibilityOff()
            return
        self._pillar_mapper.SetInputData(self._with_normals(poly))
        self._pillar_actor.VisibilityOn()
        self._apply_clipping()

    def set_cutaway(
        self,
        *,
        enabled: bool,
        axis: str = "Z",
        pct: float = 50.0,
        interactive: bool = False,
    ) -> None:
        self._clip_enabled = bool(enabled)
        self._clip_axis = axis.upper() if axis else "Z"
        self._clip_pct = float(np.clip(pct, 0.0, 100.0))
        self._apply_clipping(interactive=interactive)

    @staticmethod
    def _clip_closed_solid(
        raw_poly: vtk.vtkPolyData,
        plane: vtk.vtkPlane,
        *,
        base_color: tuple[float, float, float] | None = None,
        clip_color: tuple[float, float, float] | None = None,
        extract_cap: bool = False,
    ) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData | None]:
        """Clip a closed 3D mesh with manifold solid cap faces (no hollow open walls)."""
        planes = vtk.vtkPlaneCollection()
        planes.AddItem(plane)
        clip = vtk.vtkClipClosedSurface()
        clip.SetInputData(raw_poly)
        clip.SetClippingPlanes(planes)
        clip.GenerateFacesOn()
        if base_color is not None and clip_color is not None:
            clip.SetScalarModeToColors()
            clip.SetBaseColor(*base_color)
            clip.SetClipColor(*clip_color)
        elif extract_cap:
            clip.SetScalarModeToLabels()
        else:
            clip.SetScalarModeToNone()

        norms = vtk.vtkPolyDataNormals()
        norms.SetInputConnection(clip.GetOutputPort())
        norms.ComputePointNormalsOn()
        norms.SplittingOn()
        norms.SetFeatureAngle(45.0)
        norms.Update()
        clipped_poly = norms.GetOutput()

        cap_poly: vtk.vtkPolyData | None = None
        if extract_cap and clipped_poly.GetNumberOfCells() > 0:
            thresh = vtk.vtkThreshold()
            thresh.SetInputConnection(clip.GetOutputPort())
            thresh.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Labels"
            )
            thresh.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
            thresh.SetLowerThreshold(0.5)
            thresh.SetUpperThreshold(1.5)
            surf = vtk.vtkDataSetSurfaceFilter()
            surf.SetInputConnection(thresh.GetOutputPort())
            surf.Update()
            cap_poly = surf.GetOutput()
        return clipped_poly, cap_poly

    def _apply_clipping(self, *, interactive: bool = False) -> None:
        for mapper in (self._cad_mapper, self._footprint_mapper, self._mesh_mapper, self._pillar_mapper):
            mapper.RemoveAllClippingPlanes()
        if not self._clip_enabled:
            self._cad_cap_poly = None
            if self._cad_raw_poly is not None:
                self._cad_mapper.SetInputData(self._cad_raw_poly)
                self._cad_mapper.ScalarVisibilityOff()
            if self._footprint_raw_poly is not None:
                self._footprint_mapper.SetInputData(self._footprint_raw_poly)
                self._footprint_mapper.ScalarVisibilityOff()
            if self._mesh_raw_poly is not None:
                self._mesh_mapper.SetInputData(self._mesh_raw_poly)
                self._mesh_mapper.ScalarVisibilityOff()
            self._cut_frame_actor.VisibilityOff()
            self._cut_cap_actor.VisibilityOff()
            return

        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds
        pad = 0.04 * max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
        frac = float(np.clip(self._clip_pct, 2.0, 98.0)) / 100.0
        if self._clip_axis == "X":
            val = xmin + frac * (xmax - xmin)
            origin = (val, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax))
            normal = (-1.0, 0.0, 0.0)
            corners = [
                (val, ymin - pad, zmin - pad),
                (val, ymax + pad, zmin - pad),
                (val, ymax + pad, zmax + pad),
                (val, ymin - pad, zmax + pad),
            ]
        elif self._clip_axis == "Y":
            val = ymin + frac * (ymax - ymin)
            origin = (0.5 * (xmin + xmax), val, 0.5 * (zmin + zmax))
            normal = (0.0, -1.0, 0.0)
            corners = [
                (xmin - pad, val, zmin - pad),
                (xmax + pad, val, zmin - pad),
                (xmax + pad, val, zmax + pad),
                (xmin - pad, val, zmax + pad),
            ]
        else:
            val = zmin + frac * (zmax - zmin)
            origin = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax), val)
            normal = (0.0, 0.0, -1.0)
            corners = [
                (xmin - pad, ymin - pad, val),
                (xmax + pad, ymin - pad, val),
                (xmax + pad, ymax + pad, val),
                (xmin - pad, ymax + pad, val),
            ]
        self._clip_plane.SetOrigin(*origin)
        self._clip_plane.SetNormal(*normal)

        if interactive:
            # Hardware GPU clip planes during live slider scrubbing (zero CPU cap overhead)
            self._cad_cap_poly = None
            self._cut_cap_actor.VisibilityOff()
            if self._cad_raw_poly is not None:
                self._cad_mapper.SetInputData(self._cad_raw_poly)
                self._cad_mapper.ScalarVisibilityOff()
                self._cad_mapper.AddClippingPlane(self._clip_plane)
            if self._footprint_raw_poly is not None:
                self._footprint_mapper.SetInputData(self._footprint_raw_poly)
                self._footprint_mapper.ScalarVisibilityOff()
                self._footprint_mapper.AddClippingPlane(self._clip_plane)
            if self._mesh_raw_poly is not None:
                self._mesh_mapper.SetInputData(self._mesh_raw_poly)
                self._mesh_mapper.ScalarVisibilityOff()
                self._mesh_mapper.AddClippingPlane(self._clip_plane)
            self._pillar_mapper.AddClippingPlane(self._clip_plane)
        else:
            # 1. Seal CAD volume cleanly with solid planar cap (+ extract vector cap polygon)
            if self._cad_raw_poly is not None and self._cad_raw_poly.GetNumberOfCells() > 0:
                cad_clipped, cap_poly = self._clip_closed_solid(
                    self._cad_raw_poly, self._clip_plane, extract_cap=True
                )
                self._cad_mapper.SetInputData(cad_clipped)
                self._cad_mapper.ScalarVisibilityOff()
                self._cad_cap_poly = cap_poly
            else:
                self._cad_cap_poly = None

            # 2. Seal 3D Flying Edges isosurface cleanly; keep 2D CAD-skin patches plane-clipped
            if self._footprint_raw_poly is not None and self._footprint_raw_poly.GetNumberOfCells() > 0:
                if self._footprint_is_closed_3d:
                    fp_clipped, _ = self._clip_closed_solid(
                        self._footprint_raw_poly,
                        self._clip_plane,
                        base_color=(0.0, 0.96, 0.83),
                        clip_color=(0.22, 0.74, 0.97),
                    )
                    self._footprint_mapper.SetInputData(fp_clipped)
                    self._footprint_mapper.SetScalarModeToUseCellData()
                    self._footprint_mapper.SetColorModeToDirectScalars()
                    self._footprint_mapper.ScalarVisibilityOn()
                else:
                    self._footprint_mapper.SetInputData(self._footprint_raw_poly)
                    self._footprint_mapper.ScalarVisibilityOff()
                    self._footprint_mapper.AddClippingPlane(self._clip_plane)

            # 3. Seal optional generated 3D STL mesh cleanly
            if self._mesh_raw_poly is not None and self._mesh_raw_poly.GetNumberOfCells() > 0:
                mesh_clipped, _ = self._clip_closed_solid(
                    self._mesh_raw_poly,
                    self._clip_plane,
                    base_color=(0.22, 0.74, 0.97),
                    clip_color=(0.0, 0.96, 0.83),
                )
                self._mesh_mapper.SetInputData(mesh_clipped)
                self._mesh_mapper.SetScalarModeToUseCellData()
                self._mesh_mapper.SetColorModeToDirectScalars()
                self._mesh_mapper.ScalarVisibilityOn()

            self._pillar_mapper.AddClippingPlane(self._clip_plane)

        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for c in corners:
            pts.InsertNextPoint(*c)
        for i in range(4):
            ln = vtk.vtkLine()
            ln.GetPointIds().SetId(0, i)
            ln.GetPointIds().SetId(1, (i + 1) % 4)
            lines.InsertNextCell(ln)
        frame_poly = vtk.vtkPolyData()
        frame_poly.SetPoints(pts)
        frame_poly.SetLines(lines)
        self._cut_frame_mapper.SetInputData(frame_poly)
        self._cut_frame_actor.VisibilityOn()

    def set_cutaway_cap(
        self,
        state: ImplicitStudioState | None,
        cad_mesh: trimesh.Trimesh | None,
    ) -> None:
        """Paint an ultra-sharp 2D cross-section texture sealed flush on the cutaway plane."""
        if (
            not self._clip_enabled
            or state is None
            or cad_mesh is None
            or cad_mesh.bounds is None
            or len(cad_mesh.faces) == 0
            or getattr(state, "preview_mode", PREVIEW_MODES[0]) in ("3D Flying Edges (Coarse)", "3D Flying Edges (Fine)")
        ):
            self._cut_cap_actor.VisibilityOff()
            return

        res = compute_2d_slice_image(
            state,
            cad_mesh,
            axis=self._clip_axis,
            slice_pos_pct=self._clip_pct,
            pixel_size_mm=0.05,
            max_pixels=240,
            n_ray_dirs=1,
        )
        nv, nu, _ = res.rgb_image.shape
        b0 = np.asarray(cad_mesh.bounds[0], dtype=float)
        b1 = np.asarray(cad_mesh.bounds[1], dtype=float)
        s_val = float(res.slice_coord_mm)
        ax = res.axis

        has_vector_cap = (
            self._cad_cap_poly is not None
            and self._cad_cap_poly.GetNumberOfCells() > 0
            and self._cad_cap_poly.GetNumberOfPoints() > 0
        )

        rgb = res.rgb_image.copy()
        if has_vector_cap and res.inside_mask is not None and np.any(res.inside_mask):
            # Extrapolate boundary inside colors outward so vector-clipped edges have zero dark halo
            from scipy.ndimage import distance_transform_edt as edt

            _, indices = edt(~res.inside_mask, return_indices=True)
            rgb = rgb[indices[0], indices[1]]

        rgba = np.empty((nv, nu, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        if has_vector_cap or res.inside_mask is None:
            rgba[:, :, 3] = 255
        else:
            rgba[:, :, 3] = np.where(res.inside_mask, 255, 0).astype(np.uint8)
        rgba_vtk = np.ascontiguousarray(rgba[::-1, :, :])

        img = vtk.vtkImageData()
        img.SetDimensions(nu, nv, 1)
        vtk_arr = numpy_support.numpy_to_vtk(rgba_vtk.reshape(-1, 4), deep=True)
        img.GetPointData().SetScalars(vtk_arr)
        self._cut_cap_tex.SetInputData(img)

        if has_vector_cap:
            assert self._cad_cap_poly is not None
            cap_poly = vtk.vtkPolyData()
            cap_poly.DeepCopy(self._cad_cap_poly)
            pts_np = numpy_support.vtk_to_numpy(cap_poly.GetPoints().GetData())
            if ax == "X":
                u_coords, v_coords = pts_np[:, 1], pts_np[:, 2]
                u0, u1 = float(b0[1]), float(b1[1])
                v0, v1 = float(b0[2]), float(b1[2])
            elif ax == "Y":
                u_coords, v_coords = pts_np[:, 0], pts_np[:, 2]
                u0, u1 = float(b0[0]), float(b1[0])
                v0, v1 = float(b0[2]), float(b1[2])
            else:
                u_coords, v_coords = pts_np[:, 0], pts_np[:, 1]
                u0, u1 = float(b0[0]), float(b1[0])
                v0, v1 = float(b0[1]), float(b1[1])
            u_norm = (u_coords - u0) / max(u1 - u0, 1e-6)
            v_norm = (v_coords - v0) / max(v1 - v0, 1e-6)
            tcoords = np.ascontiguousarray(np.column_stack([u_norm, v_norm]), dtype=np.float32)
            vtk_tc = numpy_support.numpy_to_vtk(tcoords, deep=True)
            vtk_tc.SetName("TextureCoordinates")
            cap_poly.GetPointData().SetTCoords(vtk_tc)
            self._cut_cap_mapper.SetInputData(cap_poly)
        else:
            if ax == "X":
                self._cut_cap_plane.SetOrigin(s_val, float(b0[1]), float(b0[2]))
                self._cut_cap_plane.SetPoint1(s_val, float(b1[1]), float(b0[2]))
                self._cut_cap_plane.SetPoint2(s_val, float(b0[1]), float(b1[2]))
            elif ax == "Y":
                self._cut_cap_plane.SetOrigin(float(b0[0]), s_val, float(b0[2]))
                self._cut_cap_plane.SetPoint1(float(b1[0]), s_val, float(b0[2]))
                self._cut_cap_plane.SetPoint2(float(b0[0]), s_val, float(b1[2]))
            else:
                self._cut_cap_plane.SetOrigin(float(b0[0]), float(b0[1]), s_val)
                self._cut_cap_plane.SetPoint1(float(b1[0]), float(b0[1]), s_val)
                self._cut_cap_plane.SetPoint2(float(b0[0]), float(b1[1]), s_val)
            self._cut_cap_plane.Update()
            self._cut_cap_mapper.SetInputData(self._cut_cap_plane.GetOutput())
        self._cut_cap_actor.VisibilityOn()

    def set_camera_preset(self, preset: str) -> None:
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds
        cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        diag = max(np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2), 5.0)
        dist = 2.6 * diag

        cam = self.renderer.GetActiveCamera()
        cam.SetViewAngle(30.0)
        cam.SetFocalPoint(cx, cy, cz)
        p = preset.lower()
        if p == "top":
            cam.SetPosition(cx, cy, cz + dist)
            cam.SetViewUp(0.0, 1.0, 0.0)
        elif p == "front":
            cam.SetPosition(cx, cy - dist, cz)
            cam.SetViewUp(0.0, 0.0, 1.0)
        elif p == "side":
            cam.SetPosition(cx + dist, cy, cz)
            cam.SetViewUp(0.0, 0.0, 1.0)
        else:
            # Isometric studio angle with balanced margin around part bounds
            d = dist / np.sqrt(3.0)
            cam.SetPosition(cx + d, cy - d, cz + 0.85 * d)
            cam.SetViewUp(0.0, 0.0, 1.0)

        pad = 0.14 * diag
        self.renderer.ResetCamera(
            xmin - pad,
            xmax + pad,
            ymin - pad,
            ymax + pad,
            zmin - pad,
            zmax + pad,
        )
        self.renderer.ResetCameraClippingRange()


@dataclass
class SlicePreviewResult:
    """Rendered 2D cross-section slice image and metrics."""

    rgb_image: np.ndarray  # shape (H, W, 3), dtype=uint8
    axis: str
    slice_coord_mm: float
    width_mm: float
    height_mm: float
    pixel_size_mm: float
    solid_area_fraction: float
    inside_mask: np.ndarray | None = None


def compute_2d_slice_image(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
    *,
    axis: str = "Z",
    slice_pos_pct: float = 50.0,
    pixel_size_mm: float = 0.05,
    max_pixels: int = 380,
    n_ray_dirs: int = 3,
) -> SlicePreviewResult:
    """Compute a crisp high-res 2D cross-section bitmap of the implicit lattice.

    Uses Midnight Slate palette (`#0b111e` exterior, `#172236` pore void,
    `#00f5d4` solid TPMS walls, `#38bdf8` outer skin band).
    """
    b0 = np.asarray(cad_mesh.bounds[0], dtype=float)
    b1 = np.asarray(cad_mesh.bounds[1], dtype=float)
    frac = float(np.clip(slice_pos_pct, 2.0, 98.0)) / 100.0
    ax = axis.upper()

    if ax == "X":
        s_val = float(b0[0] + frac * (b1[0] - b0[0]))
        u0, u1 = float(b0[1]), float(b1[1])
        v0, v1 = float(b0[2]), float(b1[2])
    elif ax == "Y":
        s_val = float(b0[1] + frac * (b1[1] - b0[1]))
        u0, u1 = float(b0[0]), float(b1[0])
        v0, v1 = float(b0[2]), float(b1[2])
    else:
        ax = "Z"
        s_val = float(b0[2] + frac * (b1[2] - b0[2]))
        u0, u1 = float(b0[0]), float(b1[0])
        v0, v1 = float(b0[1]), float(b1[1])

    w_mm = max(u1 - u0, 0.5)
    h_mm = max(v1 - v0, 0.5)
    eff_px = max(float(pixel_size_mm), max(w_mm, h_mm) / float(max_pixels), 0.01)

    nu = max(int(np.ceil(w_mm / eff_px)), 16)
    nv = max(int(np.ceil(h_mm / eff_px)), 16)
    u = np.linspace(u0, u1, nu)
    v = np.linspace(v0, v1, nv)
    # Flip v vertically so top of image corresponds to +v
    U, V = np.meshgrid(u, v[::-1], indexing="xy")

    if ax == "X":
        X = np.full_like(U, s_val)
        Y, Z = U, V
    elif ax == "Y":
        Y = np.full_like(U, s_val)
        X, Z = U, V
    else:
        Z = np.full_like(U, s_val)
        X, Y = U, V

    _ = n_ray_dirs
    cad_sdf_vol, padded_min, sdf_spacing = _get_cached_slice_cad_sdf(cad_mesh)
    ix_coords = (X.ravel() - float(padded_min[0])) / sdf_spacing
    iy_coords = (Y.ravel() - float(padded_min[1])) / sdf_spacing
    iz_coords = (Z.ravel() - float(padded_min[2])) / sdf_spacing
    cad_sdf_2d = map_coordinates(
        cad_sdf_vol,
        np.vstack([ix_coords, iy_coords, iz_coords]),
        order=1,
        mode="nearest",
    ).reshape(nv, nu)
    inside = cad_sdf_2d <= 0.0

    solid_field = evaluate_implicit_solid_field(
        state, X, Y, Z, cad_mesh, spacing=eff_px
    )
    solid_mask = inside & (solid_field <= 0.0)

    # Edge detection for crisp lattice wall contour highlight
    iso_edge = np.zeros_like(solid_mask, dtype=bool)
    iso_edge[1:-1, 1:-1] = (
        solid_mask[1:-1, 1:-1]
        & (
            (~solid_mask[:-2, 1:-1])
            | (~solid_mask[2:, 1:-1])
            | (~solid_mask[1:-1, :-2])
            | (~solid_mask[1:-1, 2:])
        )
    )

    rgb = np.zeros((nv, nu, 3), dtype=np.uint8)
    rgb[:, :] = (11, 17, 30)  # #0b111e exterior
    rgb[inside] = (23, 34, 54)  # #172236 CAD void

    if state.export_mode in ("skin", "combined"):
        skin_mask = inside & (-cad_sdf_2d <= float(state.shell_thickness_mm))
        if state.export_mode == "skin":
            solid_mask = skin_mask
            rgb[skin_mask] = (56, 189, 248)  # #38bdf8 sky-blue shell
        else:
            rgb[solid_mask] = (0, 245, 212)
            rgb[skin_mask] = (56, 189, 248)
            solid_mask = solid_mask | skin_mask
    else:
        rgb[solid_mask] = (0, 245, 212)  # #00f5d4 mint cyan

    rgb[iso_edge] = (236, 254, 255)  # crisp bright border

    inside_count = int(np.sum(inside))
    solid_frac = float(np.sum(solid_mask) / inside_count) if inside_count > 0 else 0.0

    return SlicePreviewResult(
        rgb_image=rgb,
        axis=ax,
        slice_coord_mm=round(s_val, 3),
        width_mm=round(w_mm, 3),
        height_mm=round(h_mm, 3),
        pixel_size_mm=round(eff_px, 4),
        solid_area_fraction=round(solid_frac, 4),
        inside_mask=inside,
    )
