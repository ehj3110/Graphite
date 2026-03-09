"""
Graphite Solver Module — Test Script

Test objective:
    Solve for strut radius that achieves target solid fraction (20%) in a
    20x20x20 mock boundary box using the conformal lattice pipeline and
    Vertex-to-Centroid (BCC) topology.

Outputs:
    - Target volume
    - Achieved volume
    - Final strut radius
    - Iteration count
    - Exported STL: test_solved_lattice.stl
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from solver import optimize_lattice_fraction


def run_solver_test() -> None:
    """
    Execute solver test on a mock cubic boundary.
    """
    print("=" * 60)
    print("Graphite Solver Module — Test")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Build mock boundary
    # -------------------------------------------------------------------------
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    # Requested target configuration
    target_vf = 0.20
    target_element_size = 8.0
    target_volume = float(boundary_mesh.volume) * target_vf

    # -------------------------------------------------------------------------
    # Run optimization
    # -------------------------------------------------------------------------
    result = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=target_vf,
        target_element_size=target_element_size,
        topology_type="vertex_to_centroid",
    )

    # -------------------------------------------------------------------------
    # Report results
    # -------------------------------------------------------------------------
    print(f"Target volume:      {target_volume:.4f}")
    print(f"Achieved volume:    {result.volume:.4f}")
    print(f"Final strut radius: {result.radius:.6f}")
    print(f"Iterations used:    {result.iterations}")

    # -------------------------------------------------------------------------
    # Export solved lattice
    # -------------------------------------------------------------------------
    output_path = Path("test_solved_lattice.stl")
    result.mesh.export(str(output_path))
    print(f"Exported:           {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    run_solver_test()
