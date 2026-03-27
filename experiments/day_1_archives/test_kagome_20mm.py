"""
Quick 20mm Kagome test — verify Y-Skin revert.
- 20×20×20 box, target_element_size=5.0, target_vf=0.10
- topology_type=kagome, include_surface_cage=True
- Check: manifold + single connected component
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from graphite.explicit.scaffold_module import generate_conformal_scaffold
from solver import optimize_lattice_fraction_from_topology
from graphite.explicit.topology_module import count_connected_components, generate_topology


def main() -> None:
    print("=" * 60)
    print("Kagome 20mm Quick Test — Y-Skin Revert Verification")
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
    )
    result = optimize_lattice_fraction_from_topology(
        mesh=boundary_mesh,
        target_vf=target_vf,
        nodes=topo_nodes,
        struts=topo_struts,
        tol=0.025,
        max_iter=15,
        representative_volume_check=True,
        representative_volume_fraction=0.4,
    )

    n_comp = count_connected_components(result.struts, result.nodes.shape[0])
    is_watertight = result.mesh.is_watertight
    is_empty = result.mesh.is_empty

    print(f"Nodes:           {result.nodes.shape[0]}")
    print(f"Struts:          {result.struts.shape[0]}")
    print(f"Connected:       {n_comp} component(s) {'OK' if n_comp == 1 else 'FAIL'}")
    print(f"Watertight:      {is_watertight} {'OK' if is_watertight else 'FAIL'}")
    print(f"Empty:           {is_empty} {'FAIL (bad)' if is_empty else 'OK'}")

    out_path = Path("test_kagome_20mm.stl")
    result.mesh.export(str(out_path))
    print(f"Exported:        {out_path.resolve()}")

    ok = n_comp == 1 and is_watertight and not is_empty
    print("\n" + ("PASS" if ok else "FAIL") + " — Y-Skin revert verification")
    print("=" * 60)


if __name__ == "__main__":
    main()
