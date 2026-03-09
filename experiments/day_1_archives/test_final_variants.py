"""
Graphite Final Variants — Audit Box + Full Benchmark

Runs all 4 topology variants with cached scaffold and audit-box solver path.
Also benchmarks audit vs full solver for Voronoi and exports:
    - voronoi_full.stl
    - voronoi_internal_only.stl
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh

from scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction_from_topology
from topology_module import count_connected_components, generate_topology

VARIANTS = [
    #("rhombic", "variant_rhombic_12_5.stl"),
    #("voronoi", "variant_voronoi_12_5.stl"),
    #("kagome", "variant_kagome_12_5.stl"),
    ("icosahedral", "variant_icosahedral_12_5_small.stl"),
]


def _solve_variant(
    *,
    mesh: trimesh.Trimesh,
    nodes: np.ndarray,
    elements: np.ndarray,
    surface_faces: np.ndarray,
    topo_type: str,
    include_surface_cage: bool,
    representative_volume_check: bool,
) -> tuple[float, object]:
    topo_nodes, topo_struts = generate_topology(
        nodes=nodes,
        elements=elements,
        surface_faces=surface_faces,
        type=topo_type,
        include_surface_cage=include_surface_cage,
    )

    t0 = time.perf_counter()
    result = optimize_lattice_fraction_from_topology(
        mesh=mesh,
        target_vf=0.125,
        nodes=topo_nodes,
        struts=topo_struts,
        tol=0.025,
        max_iter=15,
        representative_volume_check=representative_volume_check,
        representative_volume_fraction=0.4,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result


def run_final_variants() -> None:
    print("=" * 60)
    print("Graphite Final Variants — Audit Solver")
    print("=" * 60)

    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    target_element_size = 2.0

    # Call GMSH exactly once and cache scaffold in-memory.
    scaffold = generate_conformal_scaffold(
        mesh=boundary_mesh,
        target_element_size=target_element_size,
    )

    # Run 4 variants through audit-box solver.
    for topo_type, output_name in VARIANTS:
        print(f"\n{topo_type.upper()}")
        elapsed, result = _solve_variant(
            mesh=boundary_mesh,
            nodes=scaffold.nodes,
            elements=scaffold.elements,
            surface_faces=scaffold.surface_faces,
            topo_type=topo_type,
            include_surface_cage=True,
            representative_volume_check=True,
        )
        print(f"  Smart Seed Radius: {result.seed_radius:.6f}")
        print(f"  Final Iterations:  {result.iterations}")
        print(f"  Audit Solve Time:  {elapsed:.2f}s")

        n_comp = count_connected_components(result.struts, result.nodes.shape[0])
        print("  Connectivity:      100% Connected" if n_comp == 1 else f"  Connectivity:      {n_comp} components")

        out_path = Path(output_name)
        result.mesh.export(str(out_path))
        print(f"  Exported:          {out_path.resolve()}")

    # Voronoi diagnostic exports:
    # 1) full with boundary struts
    # 2) internal only with boundary struts disabled
    print("\nVORONOI DIAGNOSTIC")
    _, vor_full = _solve_variant(
        mesh=boundary_mesh,
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topo_type="voronoi",
        include_surface_cage=True,
        representative_volume_check=True,
    )
    _, vor_internal = _solve_variant(
        mesh=boundary_mesh,
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topo_type="voronoi",
        include_surface_cage=False,
        representative_volume_check=True,
    )
    full_path = Path("voronoi_full.stl")
    internal_path = Path("voronoi_internal_only.stl")
    vor_full.mesh.export(str(full_path))
    vor_internal.mesh.export(str(internal_path))
    print(f"  Exported:          {full_path.resolve()}")
    print(f"  Exported:          {internal_path.resolve()}")

    # Benchmark audit solver vs full solver (Voronoi representative case)
    t_audit, _ = _solve_variant(
        mesh=boundary_mesh,
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topo_type="voronoi",
        include_surface_cage=True,
        representative_volume_check=True,
    )
    t_full, _ = _solve_variant(
        mesh=boundary_mesh,
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topo_type="voronoi",
        include_surface_cage=True,
        representative_volume_check=False,
    )
    print("\nBENCHMARK")
    print(f"  Audit Solver Time: {t_audit:.2f}s")
    print(f"  Full Solver Time:  {t_full:.2f}s")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_final_variants()
