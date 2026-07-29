"""
Torture Test Execution Script for Part2_Adapter.STL

Generates 5 conformal lattice permutations across cell sizes and topologies:
1. A15 Kagome, cell_size = D / 2
2. A15 Kagome, cell_size = D / 2.6
3. A15 Kagome, cell_size = D / 1.25238
4. A15 Rhombic, cell_size = D / 3
5. SC Octahedral, cell_size = D / 3

Renders isometric PNG previews for quick review.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphite.explicit.mesh_repair import repair_cad_mesh

def generate_conformal_lattice_unified(
    cad_filepath: str,
    cell_size: float,
    strut_radius: float,
    background_grid: str,
    topology_type: str,
    export_dir: str,
    skip_sweep: bool = False,
) -> dict:
    if topology_type == "rhombic":
        raw = trimesh.load(cad_filepath)
        cad_mesh = repair_cad_mesh(raw)
        from graphite.explicit.conformal_generator import generate_conformal_scaffold
        from graphite.explicit.topology_module import generate_topology
        from graphite.explicit.geometry_module import generate_geometry

        scaffold = generate_conformal_scaffold(cad_mesh, cell_size)
        nodes, struts = generate_topology(
            nodes=scaffold.nodes,
            elements=scaffold.elements,
            surface_faces=scaffold.surface_faces,
            topology_type="rhombic",
        )
        
        if not skip_sweep:
            lattice_mesh, _ = generate_geometry(
                nodes=nodes,
                struts=struts,
                strut_radius=strut_radius,
                boundary_mesh=cad_mesh,
                crop_to_boundary=True,
            )
            os.makedirs(export_dir, exist_ok=True)
            lattice_path = os.path.join(export_dir, "Part2_Adapter_conformal_lattice.stl")
            lattice_mesh.export(lattice_path)
            
        return {
            "nodes_count": len(nodes),
            "struts_count": len(struts),
        }
    else:
        from graphite.explicit import generate_conformal_lattice
        lattice_type = "SC" if topology_type == "octahedral" else "A15"
        return generate_conformal_lattice(
            cad_filepath=cad_filepath,
            cell_size=cell_size,
            strut_radius=strut_radius,
            lattice_type=lattice_type,
            export_dir=export_dir,
            skip_sweep=skip_sweep,
        )


def render_mesh_png(stl_path: str, png_path: str, title: str):
    """Render isometric preview PNG of an STL mesh with equal 3D aspect ratio."""
    if not os.path.exists(stl_path):
        print(f"Skipping render: {stl_path} does not exist.")
        return

    mesh = trimesh.load(stl_path)
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=12, fontweight='bold')

    if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
        poly3d = Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolors='#2b5c8f',
            edgecolors='#1a3654',
            linewidths=0.2,
            alpha=0.9
        )
        ax.add_collection3d(poly3d)

        # Equalize 3D aspect ratio so rendering does not stretch along any axis
        bounds = mesh.bounds
        max_range = np.array([
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2]
        ]).max() / 2.0

        mid_x = (bounds[1][0] + bounds[0][0]) * 0.5
        mid_y = (bounds[1][1] + bounds[0][1]) * 0.5
        mid_z = (bounds[1][2] + bounds[0][2]) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=25, azim=135)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG preview: {png_path}")


def run_torture_test():
    cad_path = r"c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\test_parts\Part2_Adapter.STL"
    if not os.path.exists(cad_path):
        raise FileNotFoundError(f"Missing test geometry: {cad_path}")

    raw = trimesh.load(cad_path)
    D = float(np.min(raw.extents))  # Smallest dimension
    print(f"Part2_Adapter extents: {raw.extents}, smallest dimension D = {D:.4f} mm\n")

    cases = [
        {
            "name": "Case2_A15_Kagome_D_div_2.6",
            "background_grid": "A15",
            "topology_type": "kagome",
            "cell_size": D / 2.6,
            "title": f"Case 2: A15 Kagome (Cell = {D/2.6:.2f}mm)"
        },
    ]

    out_dir = r"c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\test_parts\conformal_outputs"
    os.makedirs(out_dir, exist_ok=True)

    summary = []

    for c in cases:
        print(f"==================================================")
        print(f"Running {c['title']}...")
        t0 = time.time()
        res = generate_conformal_lattice_unified(
            cad_filepath=cad_path,
            cell_size=c["cell_size"],
            strut_radius=0.5,
            background_grid=c["background_grid"],
            topology_type=c["topology_type"],
            export_dir=out_dir,
            skip_sweep=False,
        )
        t_elapsed = time.time() - t0
        print(f"Completed in {t_elapsed:.2f}s: {res['nodes_count']} nodes, {res['struts_count']} struts.")

        # Rename output STL to case name
        default_stl = os.path.join(out_dir, "Part2_Adapter_conformal_lattice.stl")
        case_stl = os.path.join(out_dir, f"{c['name']}.stl")
        if os.path.exists(default_stl):
            if os.path.exists(case_stl):
                os.remove(case_stl)
            os.rename(default_stl, case_stl)

        # Render PNG
        png_path = os.path.join(out_dir, f"{c['name']}.png")
        render_mesh_png(case_stl, png_path, c["title"])

        summary.append({
            "case": c["name"],
            "grid": c["background_grid"],
            "topology": c["topology_type"],
            "cell_size": c["cell_size"],
            "nodes": res["nodes_count"],
            "struts": res["struts_count"],
            "time_s": t_elapsed,
            "stl": case_stl,
            "png": png_path,
        })

    print("\n==================================================")
    print("TORTURE TEST SUMMARY:")
    for s in summary:
        print(f"  {s['case']}: {s['nodes']} nodes, {s['struts']} struts, {s['time_s']:.1f}s")
    print("==================================================")


if __name__ == "__main__":
    run_torture_test()
