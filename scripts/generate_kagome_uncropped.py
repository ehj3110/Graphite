"""
Generate Kagome_Uncropped_10pct.stl — Pipe-style (uncropped) export.

20mm cube, 5mm element size, 10% Vf. Surface struts remain full cylinders
with Smart Inset (sits inside 20mm footprint; rounded tubular junctions).
"""

from __future__ import annotations

import trimesh

from solver import optimize_lattice_fraction


def main() -> None:
    boundary_mesh = trimesh.creation.box(extents=[20.0, 20.0, 20.0])

    result = optimize_lattice_fraction(
        mesh=boundary_mesh,
        target_vf=0.10,
        target_element_size=5.0,
        topology_type="kagome",
        include_surface_cage=True,
        clipped_boundary=False,
    )

    out_path = "Kagome_Uncropped_10pct.stl"
    result.mesh.export(out_path)
    print(f"Exported: {out_path}")
    print(f"Volume (intersection): {result.volume:.4f}")
    print(f"Vf: {result.volume / boundary_mesh.volume:.4%}")


if __name__ == "__main__":
    main()
