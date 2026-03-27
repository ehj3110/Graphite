"""
Kagome Dual Skin — Production Run

High-fidelity specimen:
- 20x20x20 mm cube
- Target Vf: 12.5%
- Target element size: 4.0 mm (denser for surface detail)
- Algorithm: HXT (Mesh.Algorithm3D=10, already set in scaffold_module)
- Export: Kagome_DualSkin_Production_12.5pct.stl

Audit: Watershed 1 component, Full mesh Vf within 0.5% of 12.5%
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from graphite.explicit.scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction_from_topology
from graphite.explicit.topology_module import count_connected_components, generate_topology


def main() -> None:
    print("=" * 60)
    print("Kagome Dual Skin — Production Run")
    print("=" * 60)

    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    target_element_size = 4.0
    target_vf = 0.125
    vf_tol = 0.005  # +/- 0.5%

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
    )

    n_comp = count_connected_components(topo_struts, topo_nodes.shape[0])
    print(f"Watershed:          {n_comp} component(s) {'OK' if n_comp == 1 else 'FAIL'}")

    result = optimize_lattice_fraction_from_topology(
        mesh=boundary_mesh,
        target_vf=target_vf,
        nodes=topo_nodes,
        struts=topo_struts,
        tol=0.005,
        max_iter=20,
        representative_volume_check=False,  # Full mesh for accurate 12.5%
        representative_volume_fraction=0.4,
    )

    achieved_vf = result.volume / boundary_mesh.volume
    vf_error = abs(achieved_vf - target_vf)
    vf_ok = vf_error <= vf_tol

    print(f"Target Vf:          {target_vf:.2%}")
    print(f"Full mesh Vf:       {achieved_vf:.4%} (within 0.5%: {'OK' if vf_ok else 'FAIL'})")
    print(f"Final strut radius: {result.radius:.6f} mm")

    out_path = Path("Kagome_DualSkin_Production_12.5pct.stl")
    result.mesh.export(str(out_path))
    print(f"Exported:           {out_path.resolve()}")

    ok = n_comp == 1 and vf_ok
    print("\n" + ("PASS" if ok else "FAIL") + " — Kagome Production Audit")
    print("=" * 60)


if __name__ == "__main__":
    main()
