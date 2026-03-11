"""
Supercell Grid Generator — Cartesian node grid and STL trimming.

Creates uniform 3D grids and trims nodes/struts to a boundary mesh using
trimesh point containment (SDF/ray-casting).
"""

from __future__ import annotations

import numpy as np
import trimesh


def generate_cartesian_nodes(
    bounding_box: tuple[tuple[float, float, float], tuple[float, float, float]],
    cell_size: float,
) -> np.ndarray:
    """
    Create a uniform 3D grid of nodes within a bounding box.

    Args:
        bounding_box: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        cell_size: Spacing between grid points in each dimension.

    Returns:
        (N, 3) array of node coordinates.
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    nx = max(1, int(np.ceil((x_max - x_min) / cell_size)) + 1)
    ny = max(1, int(np.ceil((y_max - y_min) / cell_size)) + 1)
    nz = max(1, int(np.ceil((z_max - z_min) / cell_size)) + 1)

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return np.asarray(nodes, dtype=np.float64)


def trim_to_stl(
    nodes: np.ndarray,
    struts: np.ndarray,
    stl_mesh: trimesh.Trimesh,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove nodes and struts completely outside the STL boundary.

    Uses trimesh.contains() (ray-casting) for point-in-mesh detection.
    A strut is kept if at least one endpoint is inside or on the boundary.
    Orphan nodes (no remaining struts) are removed.

    Args:
        nodes: (N, 3) node coordinates.
        struts: (S, 2) strut index pairs into nodes.
        stl_mesh: Watertight boundary mesh (trimesh).
        eps: Small tolerance for boundary points (points within eps of surface
            may be treated as inside; trimesh may vary).

    Returns:
        (nodes_trimmed, struts_trimmed) with updated indices.
    """
    if not stl_mesh.is_watertight:
        raise ValueError("trimesh.contains() requires a watertight mesh.")

    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)

    if nodes.shape[0] == 0 or struts.shape[0] == 0:
        return nodes, struts

    # Point containment: inside = True, outside = False
    inside = np.asarray(stl_mesh.contains(nodes), dtype=bool)

    # Strut is kept if at least one endpoint is inside
    a_inside = inside[struts[:, 0]]
    b_inside = inside[struts[:, 1]]
    strut_keep = a_inside | b_inside

    struts_keep = struts[strut_keep]

    # Nodes referenced by kept struts
    used_nodes = np.unique(struts_keep.ravel())
    nodes_trimmed = nodes[used_nodes]

    # Remap strut indices to new compact node array
    old_to_new = np.full(nodes.shape[0], -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(used_nodes):
        old_to_new[old_idx] = new_idx

    struts_remapped = np.column_stack(
        [old_to_new[struts_keep[:, 0]], old_to_new[struts_keep[:, 1]]]
    )

    struts_trimmed = np.sort(struts_remapped, axis=1)
    struts_trimmed = np.unique(struts_trimmed, axis=0)

    # Remove degenerate struts (same node twice)
    keep = struts_trimmed[:, 0] != struts_trimmed[:, 1]
    struts_trimmed = struts_trimmed[keep]

    return nodes_trimmed, struts_trimmed
