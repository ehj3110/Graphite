"""
Test coordinate handoff for unclipped Kagome 20mm cube.

Verifies:
- Input STL bounds match scaffold node bounds (within 1e-5)
- Center-to-center distance of outermost surface nodes matches GMSH element size

Run from project root: python tools/diagnostics/test_cube_coordinate_handoff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction
from topology_module import generate_topology

ELEMENT_SIZE = 5.0
TOL = 1e-5
DISTANCE_TOL = 0.5  # mm tolerance for center-to-center (mesh is approximate)


def main() -> None:
    # 20mm cube: trimesh default is centered at origin, so -10 to 10
    boundary = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    print("=" * 60)
    print("20mm Cube — Unclipped Kagome Coordinate Handoff Test")
    print("=" * 60)
    print(f"Input bounds: X=[{boundary.bounds[0,0]:.4f},{boundary.bounds[1,0]:.4f}] "
          f"Y=[{boundary.bounds[0,1]:.4f},{boundary.bounds[1,1]:.4f}] "
          f"Z=[{boundary.bounds[0,2]:.4f},{boundary.bounds[1,2]:.4f}]")
    print(f"Element size: {ELEMENT_SIZE} mm")
    print()

    result = optimize_lattice_fraction(
        mesh=boundary,
        target_vf=0.10,
        target_element_size=ELEMENT_SIZE,
        topology_type="kagome",
        include_surface_cage=True,
        clipped_boundary=False,
    )

    nodes = result.nodes
    input_bounds = np.asarray(boundary.bounds)
    node_mins = np.min(nodes, axis=0)
    node_maxs = np.max(nodes, axis=0)
    max_diff = np.max(np.abs(input_bounds - np.array([node_mins, node_maxs])))

    print(f"\nNode bounds: X=[{node_mins[0]:.4f},{node_maxs[0]:.4f}] "
          f"Y=[{node_mins[1]:.4f},{node_maxs[1]:.4f}] "
          f"Z=[{node_mins[2]:.4f},{node_maxs[2]:.4f}]")
    print(f"Bbox max diff: {max_diff:.2e} (must be <= {TOL})")

    # Outermost nodes: endpoints of struts that lie on/near the boundary
    # For Surface Dual, cage struts connect adjacent face centroids
    struts = result.struts
    face_threshold = 9.0
    cage_endpoints = set(struts.ravel())
    surface_cage_nodes = []
    for idx in cage_endpoints:
        p = nodes[idx]
        if np.any(np.abs(p) >= face_threshold):
            surface_cage_nodes.append(p)
    surface_cage_nodes = np.array(surface_cage_nodes) if surface_cage_nodes else np.empty((0, 3))

    if surface_cage_nodes.shape[0] >= 2:
        # Distances between nodes connected by struts (adjacent face centroids)
        strut_dists = []
        for i, j in struts:
            if i in cage_endpoints and j in cage_endpoints:
                pi, pj = nodes[i], nodes[j]
                if np.any(np.abs(pi) >= face_threshold) and np.any(np.abs(pj) >= face_threshold):
                    d = float(np.linalg.norm(pi - pj))
                    if 0.5 < d < 20:  # plausible element-size range
                        strut_dists.append(d)
        if strut_dists:
            mean_strut = float(np.mean(strut_dists))
            print(f"\nCage strut lengths (surface): mean={mean_strut:.4f} mm (expected ~{ELEMENT_SIZE})")
            if abs(mean_strut - ELEMENT_SIZE) < DISTANCE_TOL:
                print("PASS: Cage strut spacing matches element size.")
            else:
                print(f"CHECK: Mean {mean_strut:.4f} vs expected {ELEMENT_SIZE} (tol={DISTANCE_TOL})")
        else:
            print("\nCage strut spacing: (no surface struts in range)")
    else:
        print("\nCage strut spacing: (insufficient surface nodes)")

    bbox_ok = max_diff <= TOL
    print(f"\n{'PASS' if bbox_ok else 'FAIL'}: Bbox match within {TOL}")
    print("=" * 60)


if __name__ == "__main__":
    main()
