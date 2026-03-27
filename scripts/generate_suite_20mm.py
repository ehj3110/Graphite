"""
Graphite Lattice Suite — 20mm cube, 10% Vf, 5mm element size.

Iterates through rhombic, icosahedral, voronoi, kagome.
Uses one-shot solver. Voronoi and Kagome use Surface Dual (centroid-to-centroid).
Rhombic and icosahedral use standard vertex/edge logic.

Output: Suite_{Type}_10pct.stl
"""

from __future__ import annotations

import time
from pathlib import Path

import trimesh

from graphite.explicit.scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction_from_topology
from graphite.explicit.topology_module import generate_topology

TYPES = ["rhombic", "icosahedral", "voronoi", "kagome"]
TARGET_VF = 0.10
ELEMENT_SIZE = 5.0


def main() -> None:
    print("=" * 70)
    print("Graphite Lattice Suite — 20mm Cube, 10% Vf, 5mm Element")
    print("=" * 70)

    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    total_volume = float(boundary_mesh.volume)

    scaffold = generate_conformal_scaffold(
        mesh=boundary_mesh,
        target_element_size=ELEMENT_SIZE,
    )

    results = []

    for topo_type in TYPES:
        print(f"\n--- {topo_type.upper()} ---")
        t0 = time.perf_counter()

        topo_nodes, topo_struts = generate_topology(
            nodes=scaffold.nodes,
            elements=scaffold.elements,
            surface_faces=scaffold.surface_faces,
            topology_type=topo_type,
            include_surface_cage=True,
            target_element_size=ELEMENT_SIZE,
        )

        result = optimize_lattice_fraction_from_topology(
            mesh=boundary_mesh,
            target_vf=TARGET_VF,
            nodes=topo_nodes,
            struts=topo_struts,
        )

        elapsed = time.perf_counter() - t0
        achieved_vf = result.volume / total_volume

        out_name = f"Suite_{topo_type}_10pct.stl"
        result.mesh.export(out_name)
        print(f"Exported: {out_name}")

        results.append(
            {
                "type": topo_type,
                "radius": result.radius,
                "vf": achieved_vf,
                "time": elapsed,
            }
        )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Type':<12} {'Radius (mm)':<14} {'Final Vf':<12} {'Time (s)':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['type']:<12} {r['radius']:<14.6f} {r['vf']:<12.4%} {r['time']:<10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
