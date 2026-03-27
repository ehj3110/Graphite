"""
Graphite Topology Module — Test Script

End-to-end test chain:
    Module 1 (I/O):       load_and_verify_mesh
    Module 2 (Scaffold):  generate_conformal_scaffold
    Module 3 (Topology):  generate_topology

Workflow:
    1) Create in-memory 20x20x20 box boundary (mock input)
    2) Route through Module 1 by exporting/loading temp STL
    3) Generate tetrahedral scaffold (Module 2)
    4) Generate vertex-to-centroid topology + surface cage (Module 3)
    5) Print final node/strut counts
    6) Plot the 1D line skeleton in PyVista

Run from project root:
    python test_topology.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from io_module import load_and_verify_mesh
from graphite.explicit.scaffold_module import generate_conformal_scaffold
from graphite.explicit.topology_module import generate_topology


def _build_polyline_mesh(nodes: np.ndarray, struts: np.ndarray) -> pv.PolyData:
    """
    Build a PyVista line mesh from strut index pairs.

    PyVista line format uses a packed "lines" array:
        [2, p0, p1, 2, p2, p3, ...]
    where each leading 2 indicates one 2-point line segment.
    """
    if struts.size == 0:
        raise ValueError("Cannot visualize topology: `struts` is empty.")

    # Vectorized packing of line cells: prepend a column of 2s to each pair.
    num_struts = struts.shape[0]
    segment_sizes = np.full((num_struts, 1), 2, dtype=np.int64)
    packed_lines = np.hstack((segment_sizes, struts.astype(np.int64))).ravel()

    line_mesh = pv.PolyData(nodes.astype(np.float64))
    line_mesh.lines = packed_lines
    return line_mesh


def run_topology_test() -> None:
    """
    Execute end-to-end topology generation test and render line skeleton.
    """
    print("=" * 60)
    print("Graphite Topology Module — Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1) Create mock box boundary in memory
    # -------------------------------------------------------------------------
    mock_boundary = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    # -------------------------------------------------------------------------
    # 2) Route through Module 1 using temporary STL
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_stl_path = Path(tmp.name)
        mock_boundary.export(str(tmp_stl_path))

    try:
        io_result = load_and_verify_mesh(tmp_stl_path)
    finally:
        tmp_stl_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # 3) Generate conformal tetra scaffold (Module 2)
    # -------------------------------------------------------------------------
    scaffold = generate_conformal_scaffold(
        mesh=io_result.mesh,
        target_element_size=5.0,
    )

    # -------------------------------------------------------------------------
    # 4) Generate topology (Module 3)
    # -------------------------------------------------------------------------
    topology_nodes, struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type="vertex_to_centroid",
    )

    # -------------------------------------------------------------------------
    # 5) Print final topology sizes
    # -------------------------------------------------------------------------
    print(f"Final nodes:  {topology_nodes.shape[0]}")
    print(f"Final struts: {struts.shape[0]}")

    # -------------------------------------------------------------------------
    # 6) Plot 1D line skeleton in PyVista
    # -------------------------------------------------------------------------
    skeleton = _build_polyline_mesh(topology_nodes, struts)

    plotter = pv.Plotter()
    plotter.add_title("Graphite Topology: 1D Strut Skeleton")
    plotter.add_mesh(
        skeleton,
        color="gold",
        line_width=1.5,
        render_lines_as_tubes=False,
    )
    plotter.show_grid()
    plotter.add_axes()
    plotter.show()

    print("=" * 60)


if __name__ == "__main__":
    run_topology_test()
