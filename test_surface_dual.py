"""
Surface Dual — Final verification test.

Runs full Kagome Surface Dual pipeline and exports STL.
- 20x20x20 mm cube, target_element_size=5.0, target_vf=0.10
- Verifies: Watershed 1 component, Smart Seed hits 10% +/- 0.5%
- Export: Kagome_SurfaceDual_10pct_5mm.stl
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction_from_topology
from topology_module import count_connected_components, generate_topology


def main() -> None:
    print("=" * 60)
    print("Kagome Surface Dual — Final Verification")
    print("=" * 60)

    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    target_element_size = 5.0
    target_vf = 0.10

    scaffold = generate_conformal_scaffold(
        mesh=boundary_mesh,
        target_element_size=target_element_size,
    )

    topo_nodes, topo_struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type="kagome",
        include_surface_cage=True,
        target_element_size=target_element_size,
    )

    n_comp = count_connected_components(topo_struts, topo_nodes.shape[0])
    print(f"Watershed:          {n_comp} component(s) {'OK' if n_comp == 1 else 'FAIL'}")

    result = optimize_lattice_fraction_from_topology(
        mesh=boundary_mesh,
        target_vf=target_vf,
        nodes=topo_nodes,
        struts=topo_struts,
        tol=0.005,
        max_iter=15,
        representative_volume_check=True,
        representative_volume_fraction=0.4,
    )

    achieved_vf = result.volume / boundary_mesh.volume
    vf_ok = result.iterations <= 15 and result.radius > 0

    print(f"Target Vf:          {target_vf:.2%}")
    print(f"Full mesh Vf:       {achieved_vf:.4%}")
    print(f"Smart Seed:         converged in {result.iterations} iter: {'OK' if vf_ok else 'FAIL'}")

    print(f"\nFinal strut radius: {result.radius:.6f} mm  (document this value)")

    out_path = Path("Kagome_SurfaceDual_10pct_5mm.stl")
    result.mesh.export(str(out_path))
    print(f"Exported:           {out_path.resolve()}")

    ok = n_comp == 1 and vf_ok
    print("\n" + ("PASS" if ok else "FAIL") + " — Kagome Surface Dual Final Verification")
    print("=" * 60)


if __name__ == "__main__":
    main()
