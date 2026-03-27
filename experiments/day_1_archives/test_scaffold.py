"""
Graphite Scaffold Module — Test Script

This script validates scaffold generation by:
  1) Creating an in-memory watertight box boundary (20 x 20 x 20)
  2) Generating a conformal tetrahedral scaffold via GMSH
  3) Printing output array shapes
  4) Rendering the tetrahedral mesh as a 3D wireframe in PyVista

Run from project root:
    python test_scaffold.py
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh

from graphite.explicit.scaffold_module import ScaffoldResult, generate_conformal_scaffold


def _build_tet_unstructured_grid(nodes: np.ndarray, elements: np.ndarray) -> pv.UnstructuredGrid:
    """
    Build a PyVista UnstructuredGrid containing tetrahedral cells.

    PyVista/VTK expects a packed cell layout:
      [4, n0, n1, n2, n3, 4, n0, n1, n2, n3, ...]
    where each leading 4 denotes tetra cell arity.
    """
    if elements.size == 0:
        raise ValueError("Cannot build visualization grid: `elements` is empty.")

    num_tets = elements.shape[0]
    cell_sizes = np.full((num_tets, 1), 4, dtype=np.int64)
    packed_cells = np.hstack((cell_sizes, elements.astype(np.int64))).ravel()

    # VTK tetra cell type ID is 10.
    cell_types = np.full(num_tets, 10, dtype=np.uint8)

    return pv.UnstructuredGrid(packed_cells, cell_types, nodes.astype(np.float64))


def run_scaffold_test() -> None:
    """
    Execute the scaffold test workflow and open a wireframe plot.
    """
    print("=" * 60)
    print("Graphite Scaffold Module — Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Build in-memory boundary mesh
    # -------------------------------------------------------------------------
    box_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    # -------------------------------------------------------------------------
    # Step 2: Generate tetrahedral scaffold with target size = 5.0
    # -------------------------------------------------------------------------
    result: ScaffoldResult = generate_conformal_scaffold(
        mesh=box_mesh,
        target_element_size=10.0,
    )

    # -------------------------------------------------------------------------
    # Step 3: Print output dimensions
    # -------------------------------------------------------------------------
    print(f"nodes shape:         {result.nodes.shape}")
    print(f"elements shape:      {result.elements.shape}")
    print(f"surface_faces shape: {result.surface_faces.shape}")

    # -------------------------------------------------------------------------
    # Step 4: Visualize tetrahedral scaffold as wireframe
    # -------------------------------------------------------------------------
    grid = _build_tet_unstructured_grid(result.nodes, result.elements)

    plotter = pv.Plotter()
    plotter.add_title("Graphite Scaffold: Tetrahedral Wireframe")
    plotter.add_mesh(
        grid,
        style="wireframe",
        color="dodgerblue",
        line_width=1.0,
        show_edges=True,
    )
    plotter.show_grid()
    plotter.add_axes()
    plotter.show()

    print("=" * 60)


if __name__ == "__main__":
    run_scaffold_test()
