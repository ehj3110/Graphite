"""
Graphite Final Conformal Test — 12.5% Solid Fraction Target

End-to-end test using the verified solver and geometry modules:
    - Module 1 (I/O):       Mock 20x20x20 box boundary
    - Module 2 (Scaffold):  generate_conformal_scaffold
    - Module 3 (Topology):  generate_topology (Vertex-to-Centroid BCC)
    - Module 4 (Geometry):  generate_geometry (via solver)
    - Solver:               optimize_lattice_fraction (bisection over strut radius)

Parameters:
    - target_element_size = 5.0
    - target_vf = 0.125 (12.5% solid fraction)

Outputs:
    - Final strut radius
    - Achieved volume
    - Number of bisection iterations
    - Exported STL: conformal_tet_12_5.stl

Run from project root:
    python test_final_conformal.py
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from solver import optimize_lattice_fraction


def run_final_conformal_test() -> None:
    """
    Execute 12.5% target conformal lattice generation and export.
    """
    print("=" * 60)
    print("Graphite Final Conformal — 12.5% Target")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1) Create 20x20x20 mock box boundary
    # -------------------------------------------------------------------------
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    # -------------------------------------------------------------------------
    # 2) Run bisection solver to find strut radius for target solid fraction
    # -------------------------------------------------------------------------
    target_vf = 0.125
    target_element_size = 5.0

    result = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=target_vf,
        target_element_size=target_element_size,
        topology_type="rhombic",
    )

    # -------------------------------------------------------------------------
    # 3) Print final radius, achieved volume, and iterations
    # -------------------------------------------------------------------------
    target_volume = float(boundary_mesh.volume) * target_vf
    print(f"Target volume:      {target_volume:.4f}")
    print(f"Achieved volume:    {result.volume:.4f}")
    print(f"Final strut radius: {result.radius:.6f}")
    print(f"Iterations used:    {result.iterations}")

    # -------------------------------------------------------------------------
    # 4) Export final watertight mesh
    # -------------------------------------------------------------------------
    output_path = Path("conformal_tet_12_5.stl")
    result.mesh.export(str(output_path))
    print(f"Exported:            {output_path.resolve()}")

    print("=" * 60)


if __name__ == "__main__":
    run_final_conformal_test()
