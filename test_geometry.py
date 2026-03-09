"""
Graphite Geometry Module — Test Script

End-to-end chain for initial geometry validation:
    Module 1: io_module.py
    Module 2: scaffold_module.py
    Module 3: topology_module.py
    Module 4: geometry_module.py

Test configuration (as requested):
    - Mock boundary: 20x20x20 box
    - Scaffold target element size: 10.0 (reduced complexity)
    - Strut radius: 0.5
    - Boundary trim enabled using original mock box

Outputs:
    - Prints final volume and watertight status
    - Exports final lattice to test_lattice.stl
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh

from geometry_module import generate_geometry
from io_module import load_and_verify_mesh
from scaffold_module import generate_conformal_scaffold
from topology_module import generate_topology


def run_geometry_test() -> None:
    """
    Execute full Module 1 -> 4 pipeline and export resulting lattice STL.
    """
    print("=" * 60)
    print("Graphite Geometry Module — Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Create mock boundary box in memory
    # -------------------------------------------------------------------------
    mock_box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    # -------------------------------------------------------------------------
    # Step 2: Route through Module 1 (requires file path)
    # -------------------------------------------------------------------------
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        mock_box.export(str(tmp_path))

    try:
        io_result = load_and_verify_mesh(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Step 3: Generate scaffold with low complexity target
    # -------------------------------------------------------------------------
    scaffold = generate_conformal_scaffold(
        mesh=io_result.mesh,
        target_element_size=3,  # Critical request: keep strut count low
    )

    # -------------------------------------------------------------------------
    # Step 4: Generate topology graph
    # -------------------------------------------------------------------------
    topo_nodes, topo_struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type="vertex_to_centroid",
    )

    # -------------------------------------------------------------------------
    # Step 5: Generate solid geometry and trim to boundary
    # -------------------------------------------------------------------------
    lattice_mesh = generate_geometry(
        nodes=topo_nodes,
        struts=topo_struts,
        strut_radius=0.25,
        boundary_mesh=mock_box,   # Trigger boundary trim
        add_spheres=False,
    )

    # -------------------------------------------------------------------------
    # Step 6: Report + export
    # -------------------------------------------------------------------------
    print(f"Final lattice volume: {float(lattice_mesh.volume):.4f}")
    print(f"Watertight: {bool(lattice_mesh.is_watertight)}")

    output_path = Path("test_lattice_0p25small.stl")
    lattice_mesh.export(str(output_path))
    print(f"Exported: {output_path.resolve()}")

    print("=" * 60)


if __name__ == "__main__":
    run_geometry_test()
