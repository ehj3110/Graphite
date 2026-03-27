"""Surface visualization and dihedral-angle-based logical face grouping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from trimesh.graph import connected_components


def compute_face_surface_ids(mesh: trimesh.Trimesh, feature_angle: float) -> np.ndarray:
    """
    Assign each triangle a logical Surface ID by grouping adjacent faces whose
    dihedral angle is *smaller* than ``feature_angle`` (degrees).

    Adjacent faces with angle >= threshold form a crease / separate surfaces.
    Isolated triangles (not merged into any pair+) get unique IDs after the groups.
    """
    angle_rad = np.radians(float(feature_angle))
    n_faces = len(mesh.faces)
    surface_ids = np.full(n_faces, -1, dtype=np.int64)

    if n_faces == 0:
        return surface_ids.astype(int)

    if len(mesh.face_adjacency) == 0:
        surface_ids[:] = np.arange(n_faces, dtype=np.int64)
        return surface_ids.astype(int)

    # Same idea as trimesh's smooth_shade: merge when angle between normals is small
    angle_ok = mesh.face_adjacency_angles < angle_rad
    adjacency = mesh.face_adjacency[angle_ok]
    facets = connected_components(
        adjacency, nodes=np.arange(n_faces, dtype=np.int64), min_len=2
    )

    for i, facet in enumerate(facets):
        surface_ids[np.asarray(facet, dtype=np.int64)] = i

    unassigned = surface_ids < 0
    num_unassigned = int(np.sum(unassigned))
    if num_unassigned > 0:
        surface_ids[unassigned] = np.arange(
            len(facets), len(facets) + num_unassigned, dtype=np.int64
        )

    return surface_ids.astype(int)


def visualize_surfaces(stl_path, feature_angle=45.0):
    stl_path = Path(stl_path)
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    surface_ids = compute_face_surface_ids(mesh, feature_angle)
    n_surfaces = int(np.max(surface_ids)) + 1 if surface_ids.size else 0
    print(f"Feature angle {feature_angle}°: {n_surfaces} logical surface(s), {len(mesh.faces)} triangles.")

    pv_mesh = pv.wrap(mesh)
    pv_mesh.cell_data["Surface ID"] = surface_ids

    p = pv.Plotter(title="Graphite Surface Picker")
    p.add_mesh(
        pv_mesh,
        scalars="Surface ID",
        cmap="tab20",
        show_edges=True,
        edge_color="black",
    )

    def pick_callback(_picked_point, picker):
        if picker is None:
            return
        cell_id = int(picker.GetCellId())
        if cell_id < 0 or cell_id >= pv_mesh.n_cells:
            return
        surf_id = int(pv_mesh.cell_data["Surface ID"][cell_id])
        print(f"Selected Cell ID: {cell_id} | Surface ID: {surf_id}")
        p.remove_actor("selection_label", render=False)
        p.add_text(
            f"Selected Surface ID: {surf_id}",
            name="selection_label",
            position="bottom_left",
            color="black",
            font_size=16,
        )

    p.enable_surface_point_picking(
        callback=pick_callback,
        picker="cell",
        use_picker=True,
        left_clicking=True,
        show_point=False,
        show_message="Click a surface to see its ID. Press 'q' to close.",
    )
    print("\nClose the 3D window to continue...")
    p.show()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    visualize_surfaces(root / "test_parts" / "20mm_cube.stl")
