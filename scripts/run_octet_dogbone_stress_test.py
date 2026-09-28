"""
ASTM Dogbone Octet-Truss Stress-Adaptive Lattice Optimization & Physical Realization.

Executes:
1. Octet-truss homogenization parameter sweep and surrogate calibration.
2. ASTM Dogbone continuum mesh generation with 3 cells across thickness (T = 6.0 mm, L_cell = 2.0 mm).
3. Closed-loop Fully Stressed Design (FSD) optimization starting at 10% solid fraction (phi_0 = 0.10).
4. Automated 3D and convergence PNG rendering.
5. Physical 3D lattice realization with Clean Mitered Joints (clean_miter=True).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

repo_root = Path(r"c:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from shapely.geometry import Polygon
import trimesh

from graphite.fea.aristo_bridge import (
    apply_surface_traction,
    find_boundary_nodes_by_plane,
    generate_macro_continuum_mesh,
)
from graphite.fea.homogenization import RVEGridConfig
from graphite.fea.stress_adaptation import (
    StressAdaptationConfig,
    export_optimization_result_vtk,
    optimize_lattice_density_fsd,
    realize_optimized_strut_lattice,
)
from graphite.fea.surrogate import (
    MaterialTensorSurrogate,
    build_octet_homogenization_surrogate,
)

pv.OFF_SCREEN = True
output_dir = Path("outputs/fea")
output_dir.mkdir(parents=True, exist_ok=True)

artifact_dir = Path(r"C:\Users\ehunt\.gemini\antigravity\brain\a3f21a84-b61a-4ab1-86d1-c3b26bbbd6a6")
artifact_dir.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("  GRAPHITE TWO-SCALE OPTIMIZATION: ASTM DOGBONE OCTET-TRUSS STRESS TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Calibrate Octet-Truss Homogenization Surrogate
    # -------------------------------------------------------------------------
    surrogate_path = output_dir / "octet_strut_surrogate.json"
    if surrogate_path.exists():
        print(f"\n[1/5] Loading existing octet surrogate: {surrogate_path}")
        octet_surrogate = MaterialTensorSurrogate.load(str(surrogate_path))
    else:
        print("\n[1/5] Calibrating Octet-Truss Homogenization Surrogate across phi in [0.08, 0.36]...")
        t0_cal = time.perf_counter()
        rve_cfg = RVEGridConfig(resolution=32, cell_size=1.0, base_E=2000.0, base_nu=0.35, solver_backend="cg")
        octet_surrogate = build_octet_homogenization_surrogate(
            solid_fractions=(0.08, 0.12, 0.18, 0.26, 0.36),
            rve_config=rve_cfg,
            fitting_method="pchip",
        )
        t_cal = time.perf_counter() - t0_cal
        octet_surrogate.save(str(surrogate_path))
        print(f"      Surrogate calibrated and saved in {t_cal:.2f} s -> {surrogate_path}")

    for pt in octet_surrogate.sample_points:
        c = pt.constants
        print(f"      phi = {pt.solid_fraction:.4f}: E_x = {c.E_x:6.1f} MPa, G_xy = {c.G_xy:6.1f} MPa, nu_xy = {c.nu_xy:5.3f}, A = {c.zener_anisotropy:5.3f}")

    # -------------------------------------------------------------------------
    # Step 2: Build ASTM Dogbone CAD & Macro Continuum Mesh
    # -------------------------------------------------------------------------
    print("\n[2/5] Creating ASTM Dogbone Specimen (T = 6.0 mm, L = 80.0 mm)...")
    # Profile: Length 80 mm (x in [-40, 40]), Grips 16 mm (y in [-8, 8]), Gauge 8 mm (y in [-4, 4])
    x_pts, y_pts = [], []
    x_pts.extend(np.linspace(-40.0, -22.0, 10))
    y_pts.extend([8.0] * 10)
    x_trans_left = np.linspace(-22.0, -12.0, 20)[1:-1]
    y_trans_left = 6.0 + 2.0 * np.cos(np.pi * (x_trans_left - (-22.0)) / 10.0)
    x_pts.extend(x_trans_left)
    y_pts.extend(y_trans_left)
    x_pts.extend(np.linspace(-12.0, 12.0, 25))
    y_pts.extend([4.0] * 25)
    x_trans_right = np.linspace(12.0, 22.0, 20)[1:-1]
    y_trans_right = 6.0 - 2.0 * np.cos(np.pi * (x_trans_right - 12.0) / 10.0)
    x_pts.extend(x_trans_right)
    y_pts.extend(y_trans_right)
    x_pts.extend(np.linspace(22.0, 40.0, 10))
    y_pts.extend([8.0] * 10)

    upper_xy = np.column_stack([x_pts, y_pts])
    lower_xy = np.column_stack([x_pts[::-1], -np.array(y_pts)[::-1]])
    poly = Polygon(np.vstack([upper_xy, lower_xy]))

    # Thickness T = 6.0 mm -> exactly 3 cells across thickness with L_cell = 2.0 mm!
    dogbone_cad = trimesh.creation.extrude_polygon(poly, height=6.0)
    dogbone_cad.apply_translation([0.0, 0.0, -3.0])  # z in [-3.0, 3.0]
    cad_stl_path = output_dir / "astm_dogbone_cad.stl"
    dogbone_cad.export(str(cad_stl_path))
    print(f"      CAD volume: {dogbone_cad.volume:.1f} mm^3 | Bounds: {dogbone_cad.bounds.tolist()}")

    # Discretize continuum mesh
    macro_mesh = generate_macro_continuum_mesh(dogbone_cad, target_element_size=1.6)
    print(f"      Continuum Mesh: {macro_mesh.elements.shape[0]} tet4 elements, {macro_mesh.nodes.shape[0]} nodes")

    # -------------------------------------------------------------------------
    # Step 3: Closed-Loop FSD Stress Adaptation Starting at 10% Solid Fraction
    # -------------------------------------------------------------------------
    print("\n[3/5] Setting up Boundary Conditions & Running Closed-Loop FSD (phi_0 = 0.10)...")
    # Left grip clamped: x <= -28.0
    left_grip_nodes = np.where(macro_mesh.nodes[:, 0] <= -28.0)[0]

    # Right end tension load: total Fx = 1200.0 N at x >= 39.0
    right_face_nodes = find_boundary_nodes_by_plane(macro_mesh, axis=0, value=40.0, tol=1.0)
    f_ext = apply_surface_traction(macro_mesh, right_face_nodes, total_force=np.array([1200.0, 0.0, 0.0]))

    # Target volume fraction: 0.15 (conserves total mass, forces material redistribution into gauge)
    # Target stress: 25.0 MPa, starting relative density = 0.10 (10% solid fraction)
    config = StressAdaptationConfig(
        target_stress=25.0,
        target_volume_fraction=0.15,
        relaxation_eta=0.35,
        move_limit=0.08,
        min_density=0.08,
        max_density=0.36,
        max_iterations=15,
        convergence_tol=1e-3,
        filter_radius=2.5,
    )

    t0_opt = time.perf_counter()
    opt_result = optimize_lattice_density_fsd(
        mesh=macro_mesh,
        surrogate=octet_surrogate,
        fixed_nodes=left_grip_nodes,
        forces=f_ext,
        config=config,
        initial_densities=0.10,  # 10% solid fraction to start!
    )
    t_opt = time.perf_counter() - t0_opt
    c_m = macro_mesh.element_centroids
    gauge_mask = np.abs(c_m[:, 0]) <= 12.0
    grip_mask = np.abs(c_m[:, 0]) >= 25.0
    phi_opt = opt_result.optimal_densities

    print(f"      Optimization completed in {t_opt:.2f} s over {opt_result.iterations_completed} iterations.")
    print(f"      Initial Compliance: {opt_result.history[0].compliance:.2f} mJ -> Final: {opt_result.history[-1].compliance:.2f} mJ")
    print(f"      Density Range: min = {np.min(phi_opt):.4f}, max = {np.max(phi_opt):.4f}")
    print(f"      Mean Gauge Density: {np.mean(phi_opt[gauge_mask]):.4f} | Mean Grip Density: {np.mean(phi_opt[grip_mask]):.4f}")
    print(f"      Initial Max VM: {opt_result.history[0].max_von_mises:.2f} MPa -> Final: {opt_result.history[-1].max_von_mises:.2f} MPa")
    print(f"      Initial Mean VM: {opt_result.history[0].mean_von_mises:.2f} MPa -> Final: {opt_result.history[-1].mean_von_mises:.2f} MPa")


    # Export VTU & JSON
    vtu_path = output_dir / "octet_dogbone_opt_result.vtu"
    json_path = output_dir / "octet_dogbone_opt_summary.json"
    export_optimization_result_vtk(opt_result, vtu_path)
    opt_result.save(json_path)
    print(f"      Saved VTU -> {vtu_path}")
    print(f"      Saved JSON -> {json_path}")

    # -------------------------------------------------------------------------
    # Step 4: Render Visual Results Directly to PNGs
    # -------------------------------------------------------------------------
    print("\n[4/5] Rendering 3D Optimization and Convergence PNGs...")
    vtu_grid = pv.read(str(vtu_path))

    # 3D Side-by-Side Plot (Optimal Density + Equilibrated Stress)
    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 600))

    # Subplot 0: Density
    plotter.subplot(0, 0)
    plotter.add_text(
        "ASTM Dogbone: Optimal Octet Density phi*(x)\nInitial phi_0 = 10% | Fully Stressed Design",
        font_size=11,
        color="black",
    )
    plotter.add_mesh(
        vtu_grid,
        scalars="optimal_density",
        cmap="viridis",
        clim=[0.08, 0.42],
        show_edges=True,
        edge_color="#444444",
        line_width=0.4,
        scalar_bar_args={
            "title": "Relative Density phi",
            "color": "black",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
        },
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.set_background("white")

    # Subplot 1: Von Mises Stress
    plotter.subplot(0, 1)
    plotter.add_text(
        "ASTM Dogbone: Equilibrated Von Mises Stress (MPa)\nTarget Stress = 30.0 MPa | Tensile Pull Fx = 800 N",
        font_size=11,
        color="black",
    )
    plotter.add_mesh(
        vtu_grid,
        scalars="element_von_mises_MPa",
        cmap="turbo",
        clim=[0.0, 55.0],
        show_edges=True,
        edge_color="#444444",
        line_width=0.4,
        scalar_bar_args={
            "title": "Von Mises (MPa)",
            "color": "black",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
        },
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.set_background("white")

    png_3d_path = output_dir / "octet_dogbone_opt_3d.png"
    plotter.screenshot(str(png_3d_path))
    shutil.copy2(png_3d_path, artifact_dir / "octet_dogbone_opt_3d.png")
    plotter.close()
    print(f"      Rendered 3D Optimization PNG -> {png_3d_path}")

    # 3-Panel Convergence Plot
    history = opt_result.history
    iters = [h.iteration for h in history]
    comp = [h.compliance for h in history]
    mean_vm = [h.mean_von_mises for h in history]
    max_vm = [h.max_von_mises for h in history]
    dphi = [h.max_delta_phi for h in history]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Compliance
    comp_reduction = (1.0 - comp[-1] / comp[0]) * 100.0
    axs[0].plot(iters, comp, "b-o", lw=2, markersize=5)
    axs[0].set_xlabel("Iteration", fontweight="bold")
    axs[0].set_ylabel("Compliance (mJ)", fontweight="bold")
    axs[0].set_title(f"Compliance Evolution ({comp_reduction:+.1f}%)", fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # Stress
    axs[1].plot(iters, mean_vm, "g-s", lw=2, markersize=5, label="Mean Von Mises")
    axs[1].plot(iters, max_vm, "r--", lw=1.5, label="Peak Von Mises")
    axs[1].axhline(config.target_stress, color="k", linestyle=":", label=f"Target ({config.target_stress:.1f} MPa)")
    axs[1].set_xlabel("Iteration", fontweight="bold")
    axs[1].set_ylabel("Stress (MPa)", fontweight="bold")
    axs[1].set_title("Stress Equilibration (FSD)", fontweight="bold")
    axs[1].legend(frameon=True)
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # Max Delta Phi
    axs[2].plot(iters[1:] if len(iters) > 1 else iters, dphi[1:] if len(iters) > 1 else dphi, "m-^", lw=2, markersize=5)
    axs[2].set_xlabel("Iteration", fontweight="bold")
    axs[2].set_ylabel("max |Delta phi|", fontweight="bold")
    axs[2].set_title("Density Convergence Step Size", fontweight="bold")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    conv_png_path = output_dir / "octet_dogbone_opt_convergence.png"
    plt.savefig(str(conv_png_path))
    shutil.copy2(conv_png_path, artifact_dir / "octet_dogbone_opt_convergence.png")
    plt.close()
    print(f"      Rendered Convergence PNG -> {conv_png_path}")

    # -------------------------------------------------------------------------
    # Step 5: Physical 3D Octet Lattice Realization with Clean Mitered Joints
    # -------------------------------------------------------------------------
    print("\n[5/5] Synthesizing Physical 3D Octet Lattice (Cell Size = 2.0 mm, 3 Cells Across Thickness)...")
    print("      Using Graphite Truss Joint Standard: clean_miter=True (no spherical bulges)...")
    t0_mesh = time.perf_counter()
    lattice_stl_path = output_dir / "octet_dogbone_physical_lattice.stl"

    physical_mesh = realize_optimized_strut_lattice(
        result=opt_result,
        rule_name="octet",
        cell_size=2.0,  # 6.0 mm / 3 = 2.0 mm!
        out_stl=lattice_stl_path,
        clean_miter=True,
        cad_mesh=dogbone_cad,
        circular_segments=12,
    )
    t_mesh = time.perf_counter() - t0_mesh
    print(f"      Physical Lattice Generated in {t_mesh:.2f} s:")
    print(f"      Vertices: {len(physical_mesh.vertices):,}, Faces: {len(physical_mesh.faces):,}, Watertight: {physical_mesh.is_watertight}")
    print(f"      Saved STL -> {lattice_stl_path}")

    # Render Physical Lattice 3D PNG
    print("      Rendering Physical Lattice 3D PNG with Strut Diameter Heatmap...")
    from scipy.interpolate import NearestNDInterpolator
    interp_mesh = NearestNDInterpolator(macro_mesh.element_centroids, opt_result.optimal_densities)
    vert_densities = interp_mesh(physical_mesh.vertices)

    pv_lattice = pv.wrap(physical_mesh)
    pv_lattice.point_data["relative_density"] = vert_densities

    # Map density to physical strut diameter: for octet, r = L * sqrt(phi / (12*sqrt(2)*pi)), diameter = 2*r
    vert_diameters = 2.0 * 2.0 * np.sqrt(vert_densities / (12.0 * np.sqrt(2.0) * np.pi))
    pv_lattice.point_data["strut_diameter_mm"] = vert_diameters

    min_dia = float(np.min(vert_diameters))
    max_dia = float(np.max(vert_diameters))
    print(f"      Strut Diameter Range: {min_dia:.3f} mm (grips) to {max_dia:.3f} mm (gauge) [{max_dia/min_dia:.2f}x thickness variation]")

    p_lat = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1600, 600))

    # Overview Subplot
    p_lat.subplot(0, 0)
    p_lat.add_text(
        f"ASTM Dogbone: Physical 3D Octet Lattice\nClean Mitered Truss | Diameters: {min_dia:.2f} mm -> {max_dia:.2f} mm",
        font_size=11,
        color="black",
    )
    p_lat.add_mesh(
        pv_lattice,
        scalars="strut_diameter_mm",
        cmap="turbo",
        clim=[min_dia, max_dia],
        show_edges=True,
        edge_color="#333333",
        line_width=0.2,
        specular=0.4,
        scalar_bar_args={
            "title": "Strut Diameter (mm)",
            "color": "black",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
        },
    )
    p_lat.view_isometric()
    p_lat.camera.zoom(1.15)
    p_lat.set_background("white")

    # Gauge Section Zoom Subplot
    p_lat.subplot(0, 1)
    p_lat.add_text(
        f"Gauge Section Zoom: Variable Strut Diameters\nGauge Struts ({max_dia:.2f} mm) vs Grip Struts ({min_dia:.2f} mm)",
        font_size=11,
        color="black",
    )
    p_lat.add_mesh(
        pv_lattice,
        scalars="strut_diameter_mm",
        cmap="turbo",
        clim=[min_dia, max_dia],
        show_edges=True,
        edge_color="#222222",
        line_width=0.3,
        specular=0.5,
        show_scalar_bar=False,
    )
    p_lat.camera_position = [(0.0, -25.0, 18.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    p_lat.camera.zoom(1.8)
    p_lat.set_background("white")

    lattice_png_path = output_dir / "octet_dogbone_physical_lattice_3d.png"
    p_lat.screenshot(str(lattice_png_path))
    shutil.copy2(lattice_png_path, artifact_dir / "octet_dogbone_physical_lattice_3d.png")
    p_lat.close()
    print(f"      Rendered Physical Lattice PNG -> {lattice_png_path}")


    print("\n" + "=" * 80)
    print("  OCTET DOGBONE STRESS TEST COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved under {output_dir}/ and synced to UI brain directory.")
    print("=" * 80)


if __name__ == "__main__":
    main()
