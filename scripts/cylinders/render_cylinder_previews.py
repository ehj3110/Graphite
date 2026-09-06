# -*- coding: utf-8 -*-
"""
Render visual preview figures for the cylindrical lattice napkin rings.
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "cylinders"

def render_preview(mesh_path: Path, out_png: Path, title: str) -> None:
    mesh = trimesh.load(str(mesh_path))
    faces = mesh.faces
    vertices = mesh.vertices
    if len(faces) > 5000:
        mesh_vis = mesh.simplify_quadric_decimation(face_count=4500)
        faces = mesh_vis.faces
        vertices = mesh_vis.vertices
        
    fig = plt.figure(figsize=(12, 6))
    
    # 1. 3D Perspective View
    ax1 = fig.add_subplot(121, projection='3d')
    tri_verts = vertices[faces]
    poly = Poly3DCollection(tri_verts, alpha=0.9, edgecolor='k', linewidths=0.2)
    poly.set_facecolor([0.25, 0.55, 0.85, 0.9])
    ax1.add_collection3d(poly)
    
    bounds = mesh.bounds
    max_range = np.array([
        bounds[1, 0] - bounds[0, 0],
        bounds[1, 1] - bounds[0, 1],
        bounds[1, 2] - bounds[0, 2]
    ]).max() / 2.0
    mid = bounds.mean(axis=0)
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y - Height (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.view_init(elev=25, azim=45)
    ax1.set_title(f"{title}\n3D Perspective", fontsize=11, fontweight='bold')
    
    # 2. Side View / Slice
    ax2 = fig.add_subplot(122)
    center_z = mid[2]
    slice_2d = mesh.section(plane_origin=[mid[0], mid[1], center_z], plane_normal=[0, 0, 1])
    if slice_2d:
        for entity in slice_2d.entities:
            pts = slice_2d.vertices[entity.points]
            ax2.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1.5)
        ax2.set_aspect('equal')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_title("Midplane Cross-Section (X-Y Slice)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("X (mm)")
        ax2.set_ylabel("Y - Height (mm)")
    else:
        ax2.text(0.5, 0.5, "Slice not available", ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close(fig)
    print(f"Saved render: {out_png.name}")

def render_pv_preview(mesh_path: Path, out_png: Path) -> None:
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
        pv.set_plot_theme('document')
        mesh = pv.read(str(mesh_path))
        plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
        plotter.background_color = 'white'
        plotter.add_mesh(mesh, color='#38bdf8', show_edges=False, smooth_shading=True)
        plotter.camera_position = 'iso'
        plotter.camera.elevation += 10
        plotter.camera.azimuth += 15
        plotter.screenshot(str(out_png))
        plotter.close()
        print(f"Saved PyVista render: {out_png.name}")
    except Exception as e:
        print(f"PyVista render failed: {e}")


def main() -> None:
    models = [
        ("NapkinRing_1p5inch_A15.stl", "napkin_ring_1p5inch_a15_pv.png", "1.5-inch Napkin Ring — A15 (1.0 Cell Tall, M=1, N=4)"),
        ("NapkinRing_1p5inch_C15.stl", "napkin_ring_1p5inch_c15_pv.png", "1.5-inch Napkin Ring — C15 (1.0 Cell Tall, M=1, N=4) [FIXED]"),
        ("NapkinRing_2inch_A15.stl", "napkin_ring_2inch_a15_pv.png", "2.0-inch Napkin Ring — A15 (1.5 Cells Tall, M=1.5, N=5)"),
        ("NapkinRing_2inch_C15.stl", "napkin_ring_2inch_c15_pv.png", "2.0-inch Napkin Ring — C15 (1.5 Cells Tall, M=1.5, N=5) [FIXED]"),
    ]
    for stl_name, png_name, title in models:
        stl_path = OUTPUT_DIR / stl_name
        png_path = OUTPUT_DIR / png_name
        if stl_path.exists():
            render_pv_preview(stl_path, png_path)

if __name__ == "__main__":
    main()
