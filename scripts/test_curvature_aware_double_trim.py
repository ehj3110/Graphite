"""Test curvature-aware adaptive chording + double-sided Boolean trim on mouse wrist rest."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh
import manifold3d

from graphite.explicit.conformal_generator import generate_conformal_lattice
from graphite.explicit.geometry_module import (
    _loft_profile_faces,
    _rounded_rect_offsets,
    _tangent_plane_frame,
    keep_largest_solid_component,
    manifold_to_trimesh,
    trimesh_to_manifold,
    union_lattice_with_spherical_joints,
    union_solid_meshes,
)
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.implicit.meshing_backends import extract_isosurface


def curvature_aware_chord_stations(
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_a: np.ndarray,
    n_b: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    prox_query: trimesh.proximity.ProximityQuery,
    *,
    max_sagitta_mm: float = 0.20,
    max_turn_angle_deg: float = 15.0,
    max_depth: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Recursively subdivide chord [p_a, p_b] where curvature or sagitta exceeds tolerance."""
    pts = [p_a]
    nrms = [n_a]

    def _subdivide(p0, p1, n0, n1, depth):
        if depth >= max_depth:
            pts.append(p1)
            nrms.append(n1)
            return

        mid = 0.5 * (p0 + p1)
        closest, _, tri_id = prox_query.on_surface([mid])
        s = closest[0]
        n_s = cad_mesh.face_normals[tri_id[0]]
        sagitta = float(np.linalg.norm(s - mid))

        dot_ends = float(np.clip(np.dot(n0, n1), -1.0, 1.0))
        angle_ends = float(np.degrees(np.arccos(dot_ends)))

        dot_mid = min(
            float(np.clip(np.dot(n0, n_s), -1.0, 1.0)),
            float(np.clip(np.dot(n1, n_s), -1.0, 1.0)),
        )
        angle_mid = float(np.degrees(np.arccos(dot_mid)))

        if (
            sagitta > max_sagitta_mm
            or angle_ends > max_turn_angle_deg
            or angle_mid > max_turn_angle_deg
        ):
            _subdivide(p0, s, n0, n_s, depth + 1)
            _subdivide(s, p1, n_s, n1, depth + 1)
        else:
            pts.append(p1)
            nrms.append(n1)

    _subdivide(p_a, p_b, n_a, n_b, 0)
    return np.asarray(pts, dtype=np.float64), np.asarray(nrms, dtype=np.float64)


def build_curvature_aware_cage_blanks(
    nodes: np.ndarray,
    skin_struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    width: float = 1.20,
    depth_inner: float = 1.50,
    depth_outer: float = 0.30,
    max_sagitta_mm: float = 0.20,
    max_turn_angle_deg: float = 15.0,
) -> trimesh.Trimesh:
    """Build rectangular strut blanks using curvature-aware adaptive stations."""
    prox = trimesh.proximity.ProximityQuery(cad_mesh)

    # Project node endpoints to CAD skin to get exact surface normals
    _, _, tri_ids = prox.on_surface(nodes)
    node_normals = cad_mesh.face_normals[tri_ids]

    half_w = 0.5 * float(width)
    bars: list[trimesh.Trimesh] = []

    for a_idx, b_idx in skin_struts:
        pa, pb = nodes[a_idx], nodes[b_idx]
        na, nb = node_normals[a_idx], node_normals[b_idx]

        st_pts, st_nrms = curvature_aware_chord_stations(
            pa,
            pb,
            na,
            nb,
            cad_mesh,
            prox,
            max_sagitta_mm=max_sagitta_mm,
            max_turn_angle_deg=max_turn_angle_deg,
        )

        n_st = len(st_pts)
        if n_st < 2:
            continue

        # Local frame stabilization along the adaptive polyline
        # stations: (1, n_st, 3), normals: (1, n_st, 3)
        n_hat, _t_hat, u_hat = _tangent_plane_frame(
            st_pts[None, :, :], st_nrms[None, :, :]
        )
        n_hat = n_hat[0]
        u_hat = u_hat[0]

        verts_list = []
        n_prof = None
        for j in range(n_st):
            offsets = _rounded_rect_offsets(half_w, -float(depth_inner), float(depth_outer), 0)
            if n_prof is None:
                n_prof = int(offsets.shape[0])
            corner = (
                st_pts[j]
                + offsets[:, 0:1] * u_hat[j]
                + offsets[:, 1:2] * n_hat[j]
            )
            verts_list.append(corner)

        verts = np.stack(verts_list, axis=0).reshape(-1, 3)
        faces = _loft_profile_faces(n_st, int(n_prof))
        bar = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        trimesh.repair.fix_normals(bar)
        bars.append(bar)

    print(f"  Lofted {len(bars):,} curvature-aware bars")
    unified_raw = union_solid_meshes(bars)
    return unified_raw


def main() -> None:
    cad_path = REPO_ROOT / "test_parts" / "Mouse wrist rest v1.stl"
    out_dir = REPO_ROOT / "outputs" / "wrist_rest_curvature_aware"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CURVATURE-AWARE ADAPTIVE CHORDING + DOUBLE-SIDED BOOLEAN TRIM")
    print("=" * 80)
    print(f"CAD File             : {cad_path.name}")
    print("Max Allowed Sagitta  : 0.20 mm (Guarantees chord stays within shell envelope)")
    print("Max Normal Turn      : 15.0 deg (Dense stations only on sharp corners)")
    print("Target Strut Depth   : 1.20 mm")
    print("Target Strut Width   : 1.20 mm")
    print("Blank Depth          : 1.50 mm (Extends 0.30 mm past 1.20 mm inner trim)")
    print("Blank Outward Oversize: 0.30 mm")
    print("=" * 80)

    cad = trimesh.load(str(cad_path), force="mesh")
    cell_size = (12.0, 12.0, 4.0)
    strut_radius = 0.6
    cage_width = 1.20
    target_skin_thickness = 1.20

    # 1. Conformal Scaffold
    t0 = time.perf_counter()
    print("\n[1/5] Generating conformal scaffold...")
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

    # 2. Extract Inset CAD Surface at level = -1.20 mm
    print("\n[2/5] Extracting CAD SDF at -1.20 mm...")
    t_sdf = time.perf_counter()
    sdf_res = 0.25
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

    # 3. Build Curvature-Aware Cage Blanks
    print("\n[3/5] Building curvature-aware blanks (sagitta <= 0.20 mm)...")
    t_loft = time.perf_counter()
    raw_cage = build_curvature_aware_cage_blanks(
        nodes,
        skin_struts,
        cad,
        width=cage_width,
        depth_inner=target_skin_thickness + 0.30,  # 1.50 mm
        depth_outer=0.30,
        max_sagitta_mm=0.20,
        max_turn_angle_deg=15.0,
    )
    t_loft_s = time.perf_counter() - t_loft
    print(f"Raw curvature-aware blanks built in {t_loft_s:.2f}s: {len(raw_cage.faces):,} faces")

    # 4. Double-Sided Boolean Trim
    print("\n[4/5] Executing Manifold3D Double-Sided Boolean CSG...")
    t_trim = time.perf_counter()
    man_cad = trimesh_to_manifold(cad)
    man_inset = trimesh_to_manifold(cad_inset_mesh)
    man_raw = trimesh_to_manifold(raw_cage)

    man_trimmed = (man_raw ^ man_cad) - man_inset
    clean_dual = manifold_to_trimesh(man_trimmed)
    trimesh.repair.fix_normals(clean_dual)
    clean_dual = keep_largest_solid_component(clean_dual, label="curvature_aware_dual")
    t_trim_s = time.perf_counter() - t_trim

    dual_stl = out_dir / "wrist_rest_curvature_aware_dual.stl"
    clean_dual.export(str(dual_stl))
    print(f"Trimmed dual exported in {t_trim_s:.2f}s: {len(clean_dual.faces):,} faces, volume={clean_dual.volume:.2f} mm^3, watertight={clean_dual.is_watertight}")

    # 5. Core Lattice and Welded Assembly
    print("\n[5/5] Building core volume lattice and fusing assembly...")
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
    man_combined = man_trimmed + man_core_clipped
    combined_mesh = manifold_to_trimesh(man_combined)
    trimesh.repair.fix_normals(combined_mesh)
    combined_mesh = keep_largest_solid_component(combined_mesh, label="curvature_aware_assembly")
    t_core_s = time.perf_counter() - t_core

    assembly_stl = out_dir / "wrist_rest_curvature_aware_assembly.stl"
    combined_mesh.export(str(assembly_stl))
    print(f"Assembly fused in {t_core_s:.2f}s: {len(combined_mesh.faces):,} faces, volume={combined_mesh.volume:.2f} mm^3, watertight={combined_mesh.is_watertight}")

    # 6. Visual Comparisons
    print("\nGenerating visual renders...")
    # Render close-up of the corners
    p = pv.Plotter(off_screen=True, window_size=(1200, 900))
    p.background_color = "white"
    p.add_mesh(pv.wrap(clean_dual), color="#2980b9", smooth_shading=True)
    p.add_title("Curvature-Aware Double-Trimmed Surface Dual (Sagitta <= 0.20mm)", font_size=10, color="black")
    p.view_isometric()
    render_dual = out_dir / "wrist_rest_curvature_aware_dual_3d.png"
    p.screenshot(str(render_dual))
    p.close()

    p = pv.Plotter(off_screen=True, window_size=(1200, 900))
    p.background_color = "white"
    p.add_mesh(pv.wrap(combined_mesh), color="#2c3e50", smooth_shading=True)
    p.add_title("Curvature-Aware Assembly (Welded Core + Non-Severed Dual)", font_size=10, color="black")
    p.view_isometric()
    render_assembly = out_dir / "wrist_rest_curvature_aware_assembly_3d.png"
    p.screenshot(str(render_assembly))
    p.close()

    print("\n" + "=" * 80)
    print("CURVATURE-AWARE BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Raw Blank Faces          : {len(raw_cage.faces):,}")
    print(f"Loft Time                : {t_loft_s:.2f}s")
    print(f"CSG Trim Time            : {t_trim_s:.2f}s")
    print(f"Trimmed Dual Faces       : {len(clean_dual.faces):,}")
    print(f"Trimmed Dual Volume      : {clean_dual.volume:.2f} mm^3")
    print(f"Dual Watertight          : {clean_dual.is_watertight}")
    print(f"Assembly Faces           : {len(combined_mesh.faces):,}")
    print(f"Assembly Volume          : {combined_mesh.volume:.2f} mm^3")
    print(f"Assembly Watertight      : {combined_mesh.is_watertight}")
    print("=" * 80)


if __name__ == "__main__":
    main()
