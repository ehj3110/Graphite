"""Test low-resolution chord blanks + double-sided Boolean trim on mouse wrist rest."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import trimesh
import manifold3d

from graphite.explicit.conformal_generator import generate_conformal_lattice
from graphite.explicit.geometry_module import (
    generate_rectangular_surface_cage,
    keep_largest_solid_component,
    manifold_to_trimesh,
    trimesh_to_manifold,
    union_lattice_with_spherical_joints,
)
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.meshing_backends import extract_isosurface


def main() -> None:
    cad_path = REPO_ROOT / "test_parts" / "Mouse wrist rest v1.stl"
    out_dir = REPO_ROOT / "outputs" / "wrist_rest_low_res_trim"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOW-RES CHORD BLANKS + DOUBLE-SIDED BOOLEAN TRIM TEST")
    print("=" * 80)
    print(f"CAD File             : {cad_path.name}")
    print("Target Skin Depth    : 1.2 mm (Decoupled from outer boundary, matching strut dia)")
    print("Strut Width          : 1.2 mm")
    print("Core Strut Diameter  : 1.2 mm")
    print("=" * 80)

    cad = trimesh.load(str(cad_path), force="mesh")
    cell_size = (12.0, 12.0, 4.0)
    strut_radius = 0.6
    cage_width = 1.2
    target_skin_thickness = 1.2  # 1.2 mm full thickness

    # 1. Scaffold Generation
    print("\n[1/4] Generating conformal scaffold...")
    t0 = time.perf_counter()
    result = generate_conformal_lattice(
        cad_filepath=str(cad_path),
        cell_size=cell_size,
        strut_radius=strut_radius,
        lattice_type="SC",
        rule_name="octahedral",
        export_dir=str(out_dir),
        mode="conformal",
        skip_sweep=True,
        volume_fraction_threshold=0.5,
        relax_mode="laplacian",
        relax_iterations=200,
        relax_alpha=0.5,
        max_projection_factor=0.5,
        surface_cage_profile="rectangular",
        surface_cage_width=cage_width,
        surface_cage_thickness=target_skin_thickness,
    )
    nodes = np.asarray(result["nodes_relaxed"], dtype=np.float64)
    volume_struts = np.asarray(result["volume_struts"], dtype=np.int64)
    skin_struts = np.asarray(result["skin_struts"], dtype=np.int64)
    print(f"Scaffold ready in {time.perf_counter()-t0:.2f}s: {len(nodes)} nodes, {len(skin_struts)} skin struts")

    # 2. Extract Inset CAD Surface at level = -1.2 mm
    print("\n[2/4] Computing CAD SDF and extracting 1.2 mm inset shell...")
    t_sdf = time.perf_counter()
    sdf_res = 0.25  # 250 um resolution for high accuracy
    _, _, _, cad_sdf, origin, _, _, _, _ = voxelize_mesh_and_edt(
        cad, resolution=sdf_res, pad_width=4
    )
    iso_inset = extract_isosurface(
        cad_sdf,
        spacing=(sdf_res, sdf_res, sdf_res),
        origin=origin,
        level=-target_skin_thickness,
        enforce_watertight=True,
    )
    cad_inset_mesh = iso_inset.mesh
    print(f"Inset CAD extracted in {time.perf_counter()-t_sdf:.2f}s: {len(cad_inset_mesh.faces):,} faces")

    man_cad = trimesh_to_manifold(cad)
    man_inset = trimesh_to_manifold(cad_inset_mesh)

    # 3. Test 3 Approaches to Surface Strut Blanks
    experiments = [
        {
            "name": "High-Res Lofts (n_seg=8, projected)",
            "n_segments": 8,
            "project_stations": True,
            "thickness": target_skin_thickness + 0.6,
            "normal_oversize": 0.40,
            "id": "high_res_8seg",
        },
        {
            "name": "2-Segment Chords (n_seg=2, mid-projected)",
            "n_segments": 2,
            "project_stations": True,
            "thickness": target_skin_thickness + 0.8,
            "normal_oversize": 0.60,
            "id": "mid_res_2seg",
        },
        {
            "name": "Straight Prisms (n_seg=1, no station projection)",
            "n_segments": 1,
            "project_stations": False,
            "thickness": target_skin_thickness + 1.2,
            "normal_oversize": 1.00,
            "id": "straight_1seg",
        },
    ]

    results = []

    for exp in experiments:
        print(f"\n--- Testing: {exp['name']} ---")
        t_loft = time.perf_counter()
        raw_cage = generate_rectangular_surface_cage(
            nodes,
            skin_struts,
            cad,
            width=cage_width,
            thickness=exp["thickness"],
            normal_oversize=exp["normal_oversize"],
            n_segments=exp["n_segments"],
            project_stations=exp["project_stations"],
            crop_to_boundary=False,
        )
        t_loft_s = time.perf_counter() - t_loft
        raw_faces = len(raw_cage.faces)
        print(f"  Raw cage lofted: {raw_faces:,} faces in {t_loft_s:.2f}s")

        t_trim = time.perf_counter()
        man_raw = trimesh_to_manifold(raw_cage)
        # Double-sided trim: (raw ∩ CAD_outer) \ CAD_inset
        man_trimmed = (man_raw ^ man_cad) - man_inset
        clean_dual = manifold_to_trimesh(man_trimmed)
        trimesh.repair.fix_normals(clean_dual)
        clean_dual = keep_largest_solid_component(clean_dual, label=exp["id"])
        t_trim_s = time.perf_counter() - t_trim

        stl_path = out_dir / f"wrist_rest_dual_{exp['id']}.stl"
        clean_dual.export(str(stl_path))

        print(f"  Trimmed dual: {len(clean_dual.faces):,} faces, volume={clean_dual.volume:.2f} mm^3 in {t_trim_s:.2f}s (watertight={clean_dual.is_watertight})")

        results.append({
            **exp,
            "raw_faces": raw_faces,
            "loft_time": t_loft_s,
            "trim_time": t_trim_s,
            "final_faces": len(clean_dual.faces),
            "volume": clean_dual.volume,
            "watertight": clean_dual.is_watertight,
            "mesh": clean_dual,
            "path": stl_path,
        })

    # 4. Build Core Lattice and Combine with the Straight-Prism Dual (Case 3)
    print("\n[3/4] Building core volume lattice and fusing with 1-segment dual...")
    t_core = time.perf_counter()
    core, _ = union_lattice_with_spherical_joints(
        nodes,
        volume_struts,
        strut_radius,
        joint_scale=1.05,
        cylinder_segments=12,
        sphere_segments=12,
    )
    man_core = trimesh_to_manifold(core)
    man_core_clipped = man_core ^ man_cad
    man_comb_straight = trimesh_to_manifold(results[2]["mesh"]) + man_core_clipped
    comb_mesh = manifold_to_trimesh(man_comb_straight)
    trimesh.repair.fix_normals(comb_mesh)
    comb_mesh = keep_largest_solid_component(comb_mesh, label="straight_comb")
    comb_path = out_dir / "wrist_rest_octahedral_straight_chord_assembly.stl"
    comb_mesh.export(str(comb_path))
    print(f"Assembly fused in {time.perf_counter()-t_core:.2f}s: {len(comb_mesh.faces):,} faces, watertight={comb_mesh.is_watertight}")

    # 5. Cross-Section Slice Analysis
    print("\n[4/4] Generating cross-section analysis across YZ plane (x=0)...")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, sharey=True)

    for idx, res in enumerate(results):
        sl = res["mesh"].section(plane_origin=[0, 0, 7], plane_normal=[1, 0, 0])
        if sl is not None:
            for entity in sl.discrete:
                axes[idx].plot(entity[:, 1], entity[:, 2], lw=1.2)
        axes[idx].set_title(f"{res['name']} — Volume: {res['volume']:.1f} mm³, Faces: {res['final_faces']:,}", fontsize=10, fontweight="bold")
        axes[idx].set_ylabel("Z (mm)")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_aspect("equal")

    sl_comb = comb_mesh.section(plane_origin=[0, 0, 7], plane_normal=[1, 0, 0])
    if sl_comb is not None:
        for entity in sl_comb.discrete:
            axes[3].plot(entity[:, 1], entity[:, 2], color="forestgreen", lw=1.2)
    axes[3].set_title("Full Assembly (Straight-Chord Dual + Internal Octahedral Core)", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Y (mm)")
    axes[3].set_ylabel("Z (mm)")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_aspect("equal")

    plt.tight_layout()
    plot_path = out_dir / "wrist_rest_low_res_chord_comparison.png"
    plt.savefig(str(plot_path), dpi=200)
    print(f"Saved cross-section comparison to {plot_path}")

    print("\n" + "=" * 80)
    print("CHORD RESOLUTION VS. DOUBLE-TRIM BENCHMARK SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"{r['name']:<40} | Raw Faces: {r['raw_faces']:>7,} | Final Faces: {r['final_faces']:>7,} | Vol: {r['volume']:>7.1f} mm³ | Watertight: {r['watertight']}")
    print(f"{'Combined Assembly (Straight Chords)':<40} | Final Faces: {len(comb_mesh.faces):>7,} | Vol: {comb_mesh.volume:>7.1f} mm³ | Watertight: {comb_mesh.is_watertight}")
    print("=" * 80)


if __name__ == "__main__":
    main()
