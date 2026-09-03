"""Test double-sided Boolean trim on mouse wrist rest octahedral surface dual."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    out_dir = REPO_ROOT / "outputs" / "wrist_rest_double_trim"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("Testing Double-Sided Boolean Trim on Mouse Wrist Rest Surface Dual")
    print("=" * 75)
    print(f"CAD File             : {cad_path.name}")
    print("Cell Size (XYZ)      : (12.0, 12.0, 4.0) mm")
    print("Lattice Rule         : Octahedral")
    print("Strut Diameter       : 1.2 mm (Radius: 0.6 mm)")
    print("Surface Cage Width   : 1.2 mm")
    print("Target Skin Depth    : 0.6 mm (Uniform Shell Thickness)")
    print("=" * 75)

    cad = trimesh.load(str(cad_path), force="mesh")
    print(f"Loaded CAD: {len(cad.faces):,} faces, extents={np.round(cad.extents, 2)} mm")

    cell_size = (12.0, 12.0, 4.0)
    strut_radius = 0.6
    cage_width = 1.2
    target_skin_thickness = 0.6  # mm

    t0 = time.perf_counter()
    print("\n[1/5] Generating conformal octahedral scaffold...")
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
    print(f"Scaffold generated in {time.perf_counter()-t0:.2f}s")

    nodes = np.asarray(result["nodes_relaxed"], dtype=np.float64)
    volume_struts = np.asarray(result["volume_struts"], dtype=np.int64)
    skin_struts = np.asarray(result["skin_struts"], dtype=np.int64)
    print(f"Nodes: {len(nodes):,}, Volume Struts: {len(volume_struts):,}, Skin Struts: {len(skin_struts):,}")

    # 2. Build Core Lattice
    t1 = time.perf_counter()
    print("\n[2/5] Building core volume lattice with spherical joints...")
    core, _ = union_lattice_with_spherical_joints(
        nodes,
        volume_struts,
        strut_radius,
        joint_scale=1.05,
        cylinder_segments=12,
        sphere_segments=12,
    )
    print(f"Core lattice built in {time.perf_counter()-t1:.2f}s: {len(core.faces):,} faces")

    # 3. Build Oversized Raw Rectangular Surface Cage
    # We deliberately give it extra normal depth so both the inner and outer cuts have material to slice
    t2 = time.perf_counter()
    print("\n[3/5] Loftying oversized surface cage...")
    raw_cage = generate_rectangular_surface_cage(
        nodes,
        skin_struts,
        cad,
        width=cage_width,
        thickness=target_skin_thickness * 2.0,  # 1.2 mm deep into volume
        normal_oversize=0.30,                  # 0.30 mm extending outside CAD
        crop_to_boundary=False,
    )
    print(f"Raw cage lofted in {time.perf_counter()-t2:.2f}s: {len(raw_cage.faces):,} faces")

    # 4. Generate Inset CAD Surface via SDF Level Set
    t3 = time.perf_counter()
    print("\n[4/5] Computing CAD SDF and extracting inset shell boundary...")
    sdf_res = 0.30  # 300 um resolution for fast, smooth inset boundary
    _, _, _, cad_sdf, origin, _, _, _, _ = voxelize_mesh_and_edt(
        cad, resolution=sdf_res, pad_width=4
    )
    # Inside CAD is negative; inset boundary is at level = -target_skin_thickness
    iso_inset = extract_isosurface(
        cad_sdf,
        spacing=(sdf_res, sdf_res, sdf_res),
        origin=origin,
        level=-target_skin_thickness,
        enforce_watertight=True,
    )
    cad_inset_mesh = iso_inset.mesh
    print(f"Inset CAD extracted in {time.perf_counter()-t3:.2f}s: {len(cad_inset_mesh.faces):,} faces, watertight={cad_inset_mesh.is_watertight}")

    # 5. Manifold3D Double-Sided Boolean Trim
    t4 = time.perf_counter()
    print("\n[5/5] Executing Manifold3D Double-Sided Boolean CSG...")
    man_cad = trimesh_to_manifold(cad)
    man_inset = trimesh_to_manifold(cad_inset_mesh)
    man_raw_cage = trimesh_to_manifold(raw_cage)
    man_core = trimesh_to_manifold(core)

    # A) Traditional single-sided trim for comparison: raw_cage ^ cad
    t_single = time.perf_counter()
    man_single_cage = man_raw_cage ^ man_cad
    single_cage = manifold_to_trimesh(man_single_cage)
    print(f"  -> Single-sided cage trimmed in {time.perf_counter()-t_single:.2f}s: {len(single_cage.faces):,} faces, watertight={single_cage.is_watertight}")

    # B) Double-sided trim: (raw_cage ^ cad) - cad_inset
    t_double = time.perf_counter()
    man_double_cage = man_single_cage - man_inset
    double_cage = manifold_to_trimesh(man_double_cage)
    trimesh.repair.fix_normals(double_cage)
    double_cage = keep_largest_solid_component(double_cage, label="double_sided_cage")
    print(f"  -> Double-sided cage trimmed in {time.perf_counter()-t_double:.2f}s: {len(double_cage.faces):,} faces, watertight={double_cage.is_watertight}")

    # C) Combined part: Clean double-sided cage + (core ^ cad)
    # The core lattice is clipped to the outer CAD, but NOT to the inset CAD!
    t_comb = time.perf_counter()
    man_core_clipped = man_core ^ man_cad
    man_combined = man_double_cage + man_core_clipped
    combined_mesh = manifold_to_trimesh(man_combined)
    trimesh.repair.fix_normals(combined_mesh)
    combined_mesh = keep_largest_solid_component(combined_mesh, label="combined_lattice")
    print(f"  -> Combined lattice fused in {time.perf_counter()-t_comb:.2f}s: {len(combined_mesh.faces):,} faces, watertight={combined_mesh.is_watertight}")

    # Export all 3 variants for inspection
    out_single = out_dir / "wrist_rest_surface_dual_single_sided.stl"
    out_double = out_dir / "wrist_rest_surface_dual_double_sided_clean.stl"
    out_combined = out_dir / "wrist_rest_octahedral_double_trimmed_assembly.stl"

    single_cage.export(str(out_single))
    double_cage.export(str(out_double))
    combined_mesh.export(str(out_combined))

    print("\n" + "=" * 75)
    print("DOUBLE-SIDED BOOLEAN TRIM COMPARISON RESULTS")
    print("=" * 75)
    print(f"Single-Sided Dual (Legacy) : {out_single.name}")
    print(f"  Faces: {len(single_cage.faces):,}, Volume: {single_cage.volume:.2f} mm^3, Watertight: {single_cage.is_watertight}")
    print(f"Double-Sided Dual (Clean)  : {out_double.name}")
    print(f"  Faces: {len(double_cage.faces):,}, Volume: {double_cage.volume:.2f} mm^3, Watertight: {double_cage.is_watertight}")
    print(f"Combined Assembly          : {out_combined.name}")
    print(f"  Faces: {len(combined_mesh.faces):,}, Volume: {combined_mesh.volume:.2f} mm^3, Watertight: {combined_mesh.is_watertight}")
    print(f"Total Workflow Time        : {time.perf_counter()-t0:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
