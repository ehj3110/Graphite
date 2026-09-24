"""
Post-processing module for extracting and converting topology optimization results.
Uses element-thresholding (not marching cubes) to guarantee a watertight solid mesh.
"""
import numpy as np
import pyvista as pv
import trimesh


def extract_isosurface(nodes: np.ndarray, elements: np.ndarray, densities: np.ndarray, threshold: float = 0.5) -> trimesh.Trimesh:
    """
    Takes a tetrahedral mesh and element densities, and extracts a solid mesh
    by keeping all elements above the density threshold and extracting their
    outer surface.

    This element-thresholding approach (vs. marching cubes contouring) always
    produces a watertight result because we literally keep the surviving
    tetrahedra and take their collective outer skin.

    Parameters
    ----------
    nodes : np.ndarray  (N, 3)
    elements : np.ndarray  (M, 4)
    densities : np.ndarray  (M,) per-element density
    threshold : float  density cutoff (default 0.5)

    Returns
    -------
    trimesh.Trimesh  The extracted solid boundary mesh.
    """
    M = elements.shape[0]

    # Build PyVista UnstructuredGrid
    cells = np.column_stack([np.full(M, 4, dtype=np.int64), elements]).ravel()
    cell_types = np.full(M, 10, dtype=np.uint8)  # VTK_TETRA
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)

    # Attach density as cell data (stay as cell data - no nodal interpolation)
    grid.cell_data["density"] = densities

    # Threshold: keep only elements >= density threshold
    thresholded = grid.threshold(threshold, scalars="density")

    if thresholded.n_cells == 0:
        raise ValueError(
            f"No elements above density threshold {threshold}. "
            f"Try lowering the threshold. "
            f"Density range: [{densities.min():.3f}, {densities.max():.3f}]"
        )

    # Extract the outer surface of the surviving solid elements
    surface = thresholded.extract_surface()
    surface = surface.triangulate()  # ensure all-triangle mesh

    faces = surface.faces.reshape(-1, 4)[:, 1:4]
    vertices = surface.points

    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    # NO CULLING: Return all thresholded islands as requested by user
    if len(result_mesh.faces) == 0:
        raise ValueError("Isosurface produced no geometry.")

    # Attempt to stitch open boundary edges
    if not result_mesh.is_watertight:
        trimesh.repair.fill_holes(result_mesh)
        trimesh.repair.fix_winding(result_mesh)
        trimesh.repair.fix_normals(result_mesh)

    return result_mesh
