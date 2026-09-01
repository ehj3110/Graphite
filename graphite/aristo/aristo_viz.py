"""
Aristo Visualization — PyVista Renders for FEA Results

Handles rendering of linear tetrahedral meshes with stress/displacement
scalar fields. Supports off-screen rendering for PNG export.

Author: Graphite / Aristo Project
"""

from __future__ import annotations
import os
from typing import Sequence

import numpy as np
import pyvista as pv
from graphite.aristo.aristo_solver import AristoResult
from graphite.aristo.stress_postprocess import nodal_to_element_mean
from graphite.geometry.surface_picking import compute_face_surface_ids
from graphite.viz.png_framing import (
    add_fixed_load_arrows,
    create_offscreen_plotter,
    frame_plotter_content,
    subsample_load_face_sites,
)
import trimesh

# (subplot label, stress percentile threshold)
DEFAULT_VON_MISES_ISO_LEVELS: tuple[tuple[str, float], ...] = (
    ("Top 10%", 90.0),
    ("Top 25%", 75.0),
    ("Top 50%", 50.0),
    ("Top 75%", 25.0),
)


def _uses_nodal_stress(result: AristoResult) -> bool:
    return (
        result.stress_field_mode == "nodal_averaged"
        and result.von_mises_nodal_raw.size == result.fea_nodes.shape[0]
        and result.von_mises_nodal_raw.any()
    )


def _nodal_stress_mpa(result: AristoResult) -> np.ndarray:
    return np.asarray(result.von_mises_nodal_raw, dtype=np.float64)


def _element_stress_mpa_for_iso(result: AristoResult) -> np.ndarray:
    """Per-tet stress (MPa) for iso solidification — nodal mean when available."""
    if _uses_nodal_stress(result):
        return nodal_to_element_mean(_nodal_stress_mpa(result), result.fea_elements)
    return result.von_mises * result.max_von_mises_raw


def _build_fea_stress_grid(
    result: AristoResult,
) -> tuple[pv.UnstructuredGrid, pv.PolyData]:
    """Linear-tet grid with von Mises (MPa) on nodes or cells and exterior shell."""
    nodes = result.fea_nodes
    elements = result.fea_elements
    m = elements.shape[0]
    cells = np.column_stack([np.full(m, 4), elements]).ravel()
    cell_types = np.full(m, 10, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    scalar_name = "Von Mises (MPa)"

    if _uses_nodal_stress(result):
        grid.point_data[scalar_name] = _nodal_stress_mpa(result)
        grid.set_active_scalars(scalar_name, preference="point")
    else:
        grid.cell_data[scalar_name] = result.von_mises * result.max_von_mises_raw
        grid.set_active_scalars(scalar_name, preference="cell")

    surface = grid.extract_surface(algorithm="dataset_surface")
    return grid, surface


def plot_aristo_result(
    result: AristoResult,
    output_path: str | None = None,
    scalar_key: str = "von_mises",
    show_edges: bool = False,
    use_log_scale: bool = False,
    cmap: str = "turbo",
    off_screen: bool = True,
    isomap_percentile: float | None = None,
    load_direction: Sequence[float] | None = None,
) -> str | None:
    """
    Render AristoResult using PyVista.

    Parameters
    ----------
    result : AristoResult
    output_path : str, optional
        Path to save PNG. If None, and off_screen is True, no image is saved.
    scalar_key : str
        'von_mises' (normalized) or 'displacement_magnitude'.
    show_edges : bool
        If True, draw element edges (tets).
    use_log_scale : bool
        If True, applies log10 to scalars for high dynamic range visualization.
    cmap : str
        Colormap name (e.g., 'turbo', 'viridis', 'magma').
    off_screen : bool
        Whether to render in the background (default True for UI/scripts).

    Returns
    -------
    output_path : str or None
    """
    if scalar_key == "von_mises":
        title = "Normalized Von Mises Stress"
        grid, _ = _build_fea_stress_grid(result)
        if _uses_nodal_stress(result):
            grid.point_data[title] = result.von_mises_nodal_norm
            grid.set_active_scalars(title, preference="point")
        else:
            grid.cell_data[title] = result.von_mises
            grid = grid.cell_data_to_point_data()
            grid.set_active_scalars(title)
    elif scalar_key == "displacement_magnitude":
        title = "Displacement (mm)"
        nodes = result.fea_nodes
        elements = result.fea_elements
        m = elements.shape[0]
        cells = np.column_stack([np.full(m, 4), elements]).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(m, 10, dtype=np.uint8), nodes)
        grid.point_data[title] = np.linalg.norm(result.displacement, axis=1)
        grid.set_active_scalars(title)
    else:
        raise ValueError(f"Unknown scalar_key: {scalar_key}")

    if use_log_scale and scalar_key == "von_mises":
        active = grid.active_scalars
        log_name = title + " (log)"
        assoc = grid.active_scalars_info.get("association", "point")
        if assoc == "point":
            grid.point_data[log_name] = np.log10(active + 1e-12)
        else:
            grid.cell_data[log_name] = np.log10(active + 1e-12)
        grid.set_active_scalars(log_name)

    # 3. Plotting
    plotter = (
        create_offscreen_plotter()
        if off_screen
        else pv.Plotter(off_screen=False, window_size=[1024, 768])
    )
    if not off_screen:
        plotter.background_color = "white"
    
    # Add axes (bottom left)
    plotter.add_axes()
    
    # Auto-position vertical colorbar for readability
    s_args = {
        "title": title, 
        "color": "black", 
        "vertical": True,
        "title_font_size": 16,
        "label_font_size": 14,
    }

    # Extract the exterior surface from the tetrahedral volume mesh
    # This prevents dense internal edges and Eye Dome Lighting from rendering the part black
    surface_mesh = grid.extract_surface(algorithm="dataset_surface")

    if isomap_percentile is not None:
        # Plot Isomap (internal contour)
        active_scalars = grid.active_scalars
        iso_val = np.percentile(active_scalars, isomap_percentile)
        
        # Create an isosurface at the specific value
        isomap = grid.contour([iso_val])
        
        # Plot the main body as a "ghost" boundary
        plotter.add_mesh(
            surface_mesh,
            color="lightgrey",
            opacity=0.15,
            show_edges=False,
            smooth_shading=True,
        )
        
        # Plot the internal stress isomap tightly
        plotter.add_mesh(
            isomap,
            cmap=cmap,
            show_edges=False,
            opacity=1.0,
            scalar_bar_args=s_args,
            smooth_shading=True,
        )
        plotter.add_text(
            f"{isomap_percentile}th Percentile Isomap (Val: {iso_val:.3f})", 
            position="upper_left", 
            color="black", 
            font_size=12
        )
    else:
        # Add main standard heatmap mesh
        plotter.add_mesh(
            surface_mesh,
            cmap=cmap,
            show_edges=show_edges,
            edge_color="black",
            line_width=0.5,
            opacity=1.0,
            scalar_bar_args=s_args,
            smooth_shading=True, # Smooth the stress gradients
        )

    # 4. Add Boundary Condition Markers
    # --- Constraints: Semi-transparent blue tint ---
    if result.fixed_nodes.size > 0:
        # Find all nodes for each face, check if they are in the fixed_node set
        fixed_node_set = set(result.fixed_nodes)
        fixed_faces = []
        for face in result.fea_surface_faces:
            if all(n in fixed_node_set for n in face):
                fixed_faces.append([3, face[0], face[1], face[2]])
        
        if fixed_faces:
            fixed_surf = pv.PolyData(result.fea_nodes, np.hstack(fixed_faces))
            plotter.add_mesh(fixed_surf, color="blue", opacity=0.3, label="Fixed Support")

    frame_plotter_content(plotter, surface_mesh.bounds)

    # --- Loading: fixed-size red arrows (tips hover outside loaded face) ---
    if result.load_centroids.size > 0:
        load_dir = (
            np.asarray(load_direction, dtype=np.float64)
            if load_direction is not None
            else np.array([0.0, 0.0, -1.0], dtype=np.float64)
        )
        load_sites = subsample_load_face_sites(result.load_centroids, load_dir)
        add_fixed_load_arrows(plotter, load_sites, load_dir)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plotter.screenshot(output_path)
    
    if not off_screen:
        plotter.show()
    
    plotter.close()
    
    return output_path


def plot_aristo_von_mises_isosurfaces(
    result: AristoResult,
    output_path: str,
    *,
    iso_levels: Sequence[tuple[str, float]] = DEFAULT_VON_MISES_ISO_LEVELS,
    cmap: str = "turbo",
    window_size: Sequence[int] = (1600, 1200),
    off_screen: bool = True,
) -> str:
    """
    Four-panel figure of solidified high-stress tet regions.

    Percentile thresholds use the **nodal** stress field when available; tets
    are solidified when the mean nodal stress on their four vertices meets the
    threshold. A faint exterior ghost shows the full part envelope.
    """
    grid, surface_mesh = _build_fea_stress_grid(result)
    scalar_name = "Von Mises (MPa)"
    element_vm = _element_stress_mpa_for_iso(result)
    quality_mask = (
        result.quality_mask
        if result.quality_mask.size == element_vm.size
        else np.ones(element_vm.shape[0], dtype=bool)
    )
    vm_min = float(element_vm.min())
    vm_max = float(element_vm.max())
    bounds = surface_mesh.bounds

    n_panels = len(iso_levels)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    plotter = pv.Plotter(
        shape=(n_rows, n_cols),
        off_screen=off_screen,
        window_size=list(window_size),
    )
    plotter.background_color = "white"

    for idx, (label, percentile) in enumerate(iso_levels):
        row, col = divmod(idx, n_cols)
        plotter.subplot(row, col)
        if _uses_nodal_stress(result):
            nodal_vm = _nodal_stress_mpa(result)
            supported = nodal_vm > 0.0
            sample = nodal_vm[supported] if supported.any() else nodal_vm
            iso_val = float(np.percentile(sample, percentile))
        else:
            vm_for_pct = element_vm[quality_mask] if quality_mask.any() else element_vm
            iso_val = float(np.percentile(vm_for_pct, percentile))
        cell_ids = np.flatnonzero((element_vm >= iso_val) & quality_mask)

        plotter.add_mesh(
            surface_mesh,
            color="lightgrey",
            opacity=0.10,
            show_edges=False,
            smooth_shading=True,
        )
        if cell_ids.size:
            solid = grid.extract_cells(cell_ids)
            plotter.add_mesh(
                solid,
                scalars=scalar_name,
                preference="cell",
                cmap=cmap,
                clim=[vm_min, vm_max],
                show_edges=False,
                opacity=1.0,
                smooth_shading=True,
            )
        plotter.add_text(
            f"{label}  (P{percentile:.0f} = {iso_val:.2f} MPa, "
            f"{cell_ids.size:,} tets)",
            position="upper_left",
            color="black",
            font_size=11,
        )
        frame_plotter_content(plotter, bounds, view_isometric=(idx == 0))

    plotter.link_views()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plotter.screenshot(output_path)
    plotter.close()
    return output_path


def export_aristo_paraview(
    result: AristoResult,
    output_path: str,
    *,
    element_vm_mpa: np.ndarray | None = None,
    load_face_mask: np.ndarray | None = None,
) -> str:
    """
    Export FEA mesh + diagnostics to VTK/VTU for ParaView inspection.

    Point data: nodal von Mises (MPa), normalized nodal stress, displacement.
    Cell data: element von Mises (MPa if provided), quality_mask, volume, aspect,
    max_edge, on_load_face flag.
    """
    from graphite.aristo.mesh_quality import (
        element_aspect_ratios,
        element_max_edge_lengths,
        element_volumes,
    )

    grid, _ = _build_fea_stress_grid(result)
    nodes = result.fea_nodes
    elements = result.fea_elements
    n_cells = elements.shape[0]

    grid.point_data["von_mises_nodal_MPa"] = _nodal_stress_mpa(result)
    grid.point_data["von_mises_nodal_norm"] = np.asarray(
        result.von_mises_nodal_norm, dtype=np.float64
    )
    grid.point_data["displacement_mm"] = np.asarray(result.displacement, dtype=np.float64)

    abs_vols = np.abs(element_volumes(nodes, elements))
    aspects = element_aspect_ratios(nodes, elements)
    max_edges = element_max_edge_lengths(nodes, elements)
    quality = (
        result.quality_mask.astype(np.int8)
        if result.quality_mask.size == n_cells
        else np.ones(n_cells, dtype=np.int8)
    )

    if element_vm_mpa is not None:
        grid.cell_data["von_mises_element_MPa"] = np.asarray(
            element_vm_mpa, dtype=np.float64
        )
    grid.cell_data["quality_ok"] = quality
    grid.cell_data["tet_volume_mm3"] = abs_vols
    grid.cell_data["aspect_ratio"] = aspects
    grid.cell_data["max_edge_mm"] = max_edges

    if load_face_mask is not None and load_face_mask.size:
        load_nodes = set(np.unique(result.fea_surface_faces[load_face_mask].ravel()))
        on_load = np.array(
            [any(int(n) in load_nodes for n in row) for row in elements],
            dtype=np.int8,
        )
        grid.cell_data["touches_load_face"] = on_load

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid.save(output_path)
    return output_path


def plot_surface_picker(
    mesh: trimesh.Trimesh,
    output_path: str,
    feature_angle: float = 45.0,
    cmap: str = "tab20",
) -> str:
    """
    Render logical surface IDs on a trimesh and save to PNG.
    Used for reference in the Aristo UI.
    """
    surface_ids = compute_face_surface_ids(mesh, feature_angle)
    
    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data["Surface ID"] = surface_ids

    plotter = create_offscreen_plotter()

    s_args = {
        "title": "Surface ID",
        "color": "black",
        "vertical": True,
        "title_font_size": 16,
        "label_font_size": 14,
        "fmt": "%.0f",  # Integer format
    }

    plotter.add_mesh(
        pv_mesh,
        scalars="Surface ID",
        cmap=cmap,
        line_width=0.5,
        categories=True,
        scalar_bar_args=s_args,
        show_edges=False,  # Disable surface triangulation view
    )
    
    # Add feature edges (silhouette/creases) for clarity instead of full mesh
    plotter.add_mesh(pv_mesh.extract_feature_edges(feature_angle=feature_angle), color="black", line_width=1)
    
    # Add axes
    plotter.add_axes()
    
    # Add labels at centroids for each unique surface
    unique_ids = np.unique(surface_ids)
    for sid in unique_ids:
        # Find all faces with this ID
        mask = (surface_ids == sid)
        # Find the center of these faces
        centers = mesh.triangles_center[mask]
        avg_center = centers.mean(axis=0)
        
        # Add a floating label
        plotter.add_point_labels(
            [avg_center], 
            [f"ID {sid}"],
            font_size=20,
            text_color="black",
            shape=None,
            always_visible=True,
            point_size=0,
        )

    frame_plotter_content(plotter, pv_mesh.bounds)
    plotter.screenshot(output_path)
    plotter.close()

    return output_path
