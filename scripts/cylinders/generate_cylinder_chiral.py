# -*- coding: utf-8 -*-
"""
Generate 3D cylindrical chiral and tetra-chiral explicit lattices and assemble them
into 3D printable Napkin Rings with watertight CAD rims.

Topologies based on Prall & Lakes (1997) and Spadoni et al. (2006):
1. Tetra-Chiral: 4 tangent ligaments per circular node on a square lattice
2. Tri-Chiral: 3 tangent ligaments per circular node on a triangular/hexagonal lattice
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import box, LineString
import manifold3d as m3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.geometry_module import _trimesh_to_manifold, _manifold_to_trimesh
from scripts.cylinders.generate_cylinder_lattice import (
    R_IN_1P5,
    WALL_T_1P5,
    R_OUT_1P5,
    H_1P5,
    Y_START_1P5,
    CIRCULAR_SEGMENTS,
    MAX_CHORD_STEP,
    BASE_RING_PATH,
    OUTPUT_DIR,
    create_sleeve_trim,
    dedupe_segments,
    build_prisms_from_2d_segments,
)

BASE_RING_1TO2_PATH = WORKSPACE_ROOT / "test_parts" / "BaseRing_1to2.STL"


def extract_custom_rims(
    base_path: Path,
    h_lattice: float,
    y_start: float = 6.65,
    center_xy: float = 25.4,
) -> tuple[m3d.Manifold, np.ndarray, float]:
    """Extract top and bottom collar rims from a base ring STL."""
    base_mesh = trimesh.load(str(base_path))
    m_base = _trimesh_to_manifold(base_mesh)
    y_center = y_start + h_lattice / 2.0
    center = np.array([center_xy, y_center, center_xy], dtype=np.float64)
    box_cut = trimesh.creation.box(extents=[150.0, h_lattice, 150.0])
    box_cut.apply_translation(center)
    m_box = _trimesh_to_manifold(box_cut)
    rims = m_base - m_box
    return rims, center, h_lattice


def generate_tetrachiral_2d_segments(
    c_mid: float,
    height: float,
    y_base: float,
    n_circumferential: int = 10,
    r_node: float = 2.2,
    n_circle_segs: int = 20,
    clip_to_domain: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Generate 2D (u, y) line segments for a periodic tetra-chiral lattice on a cylinder.
    """
    D = c_mid / float(n_circumferential)
    if D <= 2.0 * r_node:
        raise ValueError(f"Pitch D={D:.2f} mm must be > 2*r_node={2*r_node:.2f} mm.")

    L = np.sqrt(D**2 - 4.0 * r_node**2)
    alpha = np.arctan(2.0 * r_node / L)
    m_rows = int(np.ceil(height / D))

    metrics = {
        "type": "tetra_chiral",
        "c_mid": float(c_mid),
        "height": float(height),
        "y_base": float(y_base),
        "n_circumferential": int(n_circumferential),
        "m_rows": int(m_rows),
        "pitch_D_mm": float(D),
        "r_node_mm": float(r_node),
        "ligament_L_mm": float(L),
        "alpha_deg": float(np.degrees(alpha)),
    }

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segs, endpoint=False)

    for i in range(-1, n_circumferential + 2):
        for j in range(-1, m_rows + 2):
            c = np.array([i * D, y_base + j * D])

            # 1. Circular node chords
            for k in range(n_circle_segs):
                a1 = circle_angles[k]
                a2 = circle_angles[(k + 1) % n_circle_segs]
                p1 = c + r_node * np.array([np.cos(a1), np.sin(a1)])
                p2 = c + r_node * np.array([np.cos(a2), np.sin(a2)])
                raw_segments.append((p1, p2))

            # 2. Four tangent ligaments
            for k in range(4):
                th_base = -alpha + k * (np.pi / 2.0)
                p_start = c + r_node * np.array([-np.sin(th_base), np.cos(th_base)])
                p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
                raw_segments.append((p_start, p_end))

    if clip_to_domain:
        y_band = box(-D, y_base, c_mid + D, y_base + height)
        clipped: list[tuple[np.ndarray, np.ndarray]] = []
        for p0, p1 in raw_segments:
            line = LineString([p0, p1])
            cy = line.intersection(y_band)
            if not cy.is_empty:
                if cy.geom_type == "LineString":
                    clipped.append((np.array(cy.coords[0]), np.array(cy.coords[1])))
                elif cy.geom_type == "MultiLineString":
                    for l_sub in cy.geoms:
                        clipped.append((np.array(l_sub.coords[0]), np.array(l_sub.coords[1])))
        raw_segments = clipped

    unique = dedupe_segments(raw_segments, c_mid)
    metrics["n_segments"] = len(unique)
    return unique, metrics


def generate_trichiral_2d_segments(
    c_mid: float,
    height: float,
    y_base: float,
    n_circumferential: int = 10,
    r_node: float = 2.0,
    n_circle_segs: int = 20,
    clip_to_domain: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Generate 2D (u, y) line segments for a periodic tri-chiral lattice on a cylinder.
    """
    D = c_mid / float(n_circumferential)
    if D <= 2.0 * r_node:
        raise ValueError(f"Pitch D={D:.2f} mm must be > 2*r_node={2*r_node:.2f} mm.")

    L = np.sqrt(D**2 - 4.0 * r_node**2)
    phi = np.arcsin(2.0 * r_node / D)

    a1 = np.array([D, 0.0])
    a2 = np.array([D * 0.5, D * np.sqrt(3.0) / 2.0])
    m_rows = int(np.ceil(height / a2[1]))

    metrics = {
        "type": "tri_chiral",
        "c_mid": float(c_mid),
        "height": float(height),
        "y_base": float(y_base),
        "n_circumferential": int(n_circumferential),
        "m_rows": int(m_rows),
        "pitch_D_mm": float(D),
        "r_node_mm": float(r_node),
        "ligament_L_mm": float(L),
        "phi_deg": float(np.degrees(phi)),
    }

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segs, endpoint=False)

    for i in range(-2, n_circumferential + 3):
        for j in range(-2, m_rows + 3):
            c = i * a1 + j * a2 + np.array([0.0, y_base])

            # Circular chords
            for k in range(n_circle_segs):
                a1_ang = circle_angles[k]
                a2_ang = circle_angles[(k + 1) % n_circle_segs]
                p1 = c + r_node * np.array([np.cos(a1_ang), np.sin(a1_ang)])
                p2 = c + r_node * np.array([np.cos(a2_ang), np.sin(a2_ang)])
                raw_segments.append((p1, p2))

            # 3 tangent ligaments at 120-degree intervals
            for k in range(3):
                th_base = phi + k * (2.0 * np.pi / 3.0)
                tang_ang = th_base - phi
                p_start = c + r_node * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
                p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])
                raw_segments.append((p_start, p_end))

    if clip_to_domain:
        y_band = box(-D, y_base, c_mid + D, y_base + height)
        clipped: list[tuple[np.ndarray, np.ndarray]] = []
        for p0, p1 in raw_segments:
            line = LineString([p0, p1])
            cy = line.intersection(y_band)
            if not cy.is_empty:
                if cy.geom_type == "LineString":
                    clipped.append((np.array(cy.coords[0]), np.array(cy.coords[1])))
                elif cy.geom_type == "MultiLineString":
                    for l_sub in cy.geoms:
                        clipped.append((np.array(l_sub.coords[0]), np.array(l_sub.coords[1])))
        raw_segments = clipped

    unique = dedupe_segments(raw_segments, c_mid)
    metrics["n_segments"] = len(unique)
    return unique, metrics


def plot_chiral_2d_comparison(
    c_mid: float,
    height: float,
    y_base: float,
    out_png_path: Path,
) -> None:
    """Export side-by-side 2D unrolled comparison of Tetra-Chiral and Tri-Chiral lattices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), dpi=180)

    # 1. Tetra-Chiral (Square)
    segs_tetra, m_tetra = generate_tetrachiral_2d_segments(c_mid, height, y_base, n_circumferential=10)
    domain_box1 = plt.Rectangle((0.0, y_base), c_mid, height, fill=False, edgecolor="crimson", lw=1.5, ls="--")
    ax1.add_patch(domain_box1)
    for p1, p2 in segs_tetra:
        pa, pb = np.copy(p1), np.copy(p2)
        if pb[0] - pa[0] < -c_mid / 2.0:
            pb[0] += c_mid
        elif pa[0] - pb[0] < -c_mid / 2.0:
            pa[0] += c_mid
        ax1.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#1f77b4", lw=1.5)
    ax1.set_title(f"Tetra-Chiral (Square) Lattice\nN=10, D={m_tetra['pitch_D_mm']:.2f} mm, r={m_tetra['r_node_mm']:.1f} mm, L={m_tetra['ligament_L_mm']:.2f} mm",
                  fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Circumferential Coordinate u (mm)", fontsize=9)
    ax1.set_ylabel("Axial Coordinate y (mm)", fontsize=9)
    ax1.set_xlim(-0.02 * c_mid, 1.02 * c_mid)
    ax1.set_ylim(y_base - 0.05 * height, y_base + 1.05 * height)
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Tri-Chiral (Triangular)
    segs_tri, m_tri = generate_trichiral_2d_segments(c_mid, height, y_base, n_circumferential=10)
    domain_box2 = plt.Rectangle((0.0, y_base), c_mid, height, fill=False, edgecolor="crimson", lw=1.5, ls="--")
    ax2.add_patch(domain_box2)
    for p1, p2 in segs_tri:
        pa, pb = np.copy(p1), np.copy(p2)
        if pb[0] - pa[0] < -c_mid / 2.0:
            pb[0] += c_mid
        elif pa[0] - pb[0] < -c_mid / 2.0:
            pa[0] += c_mid
        ax2.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#2ca02c", lw=1.5)
    ax2.set_title(f"Tri-Chiral (Hexagonal) Lattice\nN=10, D={m_tri['pitch_D_mm']:.2f} mm, r={m_tri['r_node_mm']:.1f} mm, L={m_tri['ligament_L_mm']:.2f} mm",
                  fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Circumferential Coordinate u (mm)", fontsize=9)
    ax2.set_ylabel("Axial Coordinate y (mm)", fontsize=9)
    ax2.set_xlim(-0.02 * c_mid, 1.02 * c_mid)
    ax2.set_ylim(y_base - 0.05 * height, y_base + 1.05 * height)
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.5)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_png_path))
    plt.close(fig)
    print(f"  [Comparison Plot] Saved -> {out_png_path.name}")


def generate_chiral_cylinder_lattice(
    center: np.ndarray,
    r_in: float = R_IN_1P5,
    r_out: float = R_OUT_1P5,
    height: float = H_1P5,
    y_base: float = Y_START_1P5,
    strut_w: float = 2.0,
    n_circumferential: int = 10,
    r_node: float = 2.2,
    chiral_type: str = "tetra",
    add_boundary_rings: bool = True,
) -> tuple[m3d.Manifold, dict]:
    """Generate 3D cylindrical chiral or tetra-chiral lattice."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid

    if chiral_type.lower() == "tetra":
        segments, metrics = generate_tetrachiral_2d_segments(
            c_mid=c_mid, height=height, y_base=y_base,
            n_circumferential=n_circumferential, r_node=r_node,
        )
    elif chiral_type.lower() == "tri":
        segments, metrics = generate_trichiral_2d_segments(
            c_mid=c_mid, height=height, y_base=y_base,
            n_circumferential=n_circumferential, r_node=r_node,
        )
    else:
        raise ValueError(f"Unknown chiral_type '{chiral_type}', must be 'tetra' or 'tri'.")

    cubes = build_prisms_from_2d_segments(
        segments=segments,
        center=center,
        r_in=r_in,
        r_out=r_out,
        c_circ=c_mid,
        strut_w=strut_w,
        max_step=MAX_CHORD_STEP,
    )

    all_parts = list(cubes)
    if add_boundary_rings:
        for y in [y_base, y_base + height]:
            cyl_out = m3d.Manifold.cylinder(
                height=strut_w,
                radius_low=r_out + 0.1,
                radius_high=r_out + 0.1,
                circular_segments=CIRCULAR_SEGMENTS,
                center=True,
            )
            cyl_in = m3d.Manifold.cylinder(
                height=strut_w + 0.2,
                radius_low=r_in - 0.1,
                radius_high=r_in - 0.1,
                circular_segments=CIRCULAR_SEGMENTS,
                center=True,
            )
            ring = (cyl_out - cyl_in).transform([
                [1.0, 0.0, 0.0, center[0]],
                [0.0, 0.0, -1.0, y],
                [0.0, 1.0, 0.0, center[2]]
            ])
            all_parts.append(ring)

    composed = m3d.Manifold.batch_boolean(all_parts, m3d.OpType.Add)
    sleeve = create_sleeve_trim(center, height, r_in, r_out)
    trimmed_lattice = composed ^ sleeve
    return trimmed_lattice, metrics


def render_3d_preview(mesh_path: Path, out_png: Path, title: str) -> None:
    """Render a 2-panel figure: 3D perspective preview and X-Y midplane cross-section."""
    mesh = trimesh.load(str(mesh_path))
    faces = mesh.faces
    vertices = mesh.vertices

    fig = plt.figure(figsize=(14, 7), dpi=150)

    # 1. 3D Perspective View (Left)
    ax1 = fig.add_subplot(121, projection="3d")
    tri_verts = vertices[faces]
    poly = Poly3DCollection(tri_verts, alpha=0.92, facecolor="#2b7bba", edgecolor="#184b73", linewidths=0.1)
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
    ax1.set_title(f"{title}\n3D Perspective", fontsize=11, fontweight="bold")

    # 2. Side View / Slice (Right)
    ax2 = fig.add_subplot(122)
    center_z = mid[2]
    slice_2d = mesh.section(plane_origin=[mid[0], mid[1], center_z], plane_normal=[0, 0, 1])
    if slice_2d:
        for entity in slice_2d.entities:
            pts = slice_2d.vertices[entity.points]
            ax2.plot(pts[:, 0], pts[:, 1], "b-", linewidth=1.5)
    ax2.set_title("Midplane Cross-Section (X-Y Slice)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y - Height (mm)")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close(fig)
    print(f"  [2-Panel Render] Saved -> {out_png.name}")


def run_phase1_demo() -> None:
    """Phase 1: 2D segment generator and comparison plot."""
    print("\n" + "=" * 65)
    print("PHASE 1: CHIRAL METAMATERIAL 2D UNROLLED DIAGNOSTICS")
    print("=" * 65)
    r_mid = (R_IN_1P5 + R_OUT_1P5) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    out_cmp = OUTPUT_DIR / "chiral_variants_2d_comparison.png"
    plot_chiral_2d_comparison(c_mid=c_mid, height=H_1P5, y_base=Y_START_1P5, out_png_path=out_cmp)


def run_phase2_demo() -> None:
    """Phase 2: 3D standalone sleeves for Tetra-Chiral and Tri-Chiral."""
    print("\n" + "=" * 65)
    print("PHASE 2: 3D STANDALONE CHIRAL SLEEVES")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r_in = R_IN_1P5
    r_out = R_OUT_1P5
    height = H_1P5
    center_sa = np.array([r_out, height / 2.0, r_out], dtype=np.float64)

    types = [("tetra", "TetraChiral"), ("tri", "TriChiral")]
    for type_key, label in types:
        t0 = time.time()
        print(f"\n> Generating standalone sleeve ({label})...")
        m_sleeve, metrics = generate_chiral_cylinder_lattice(
            center=center_sa,
            r_in=r_in,
            r_out=r_out,
            height=height,
            y_base=0.0,
            strut_w=2.0,
            n_circumferential=10,
            r_node=2.2,
            chiral_type=type_key,
            add_boundary_rings=True,
        )
        mesh_sa = _manifold_to_trimesh(m_sleeve)
        out_stl = OUTPUT_DIR / f"LatticeSection_1p5inch_{label}.stl"
        mesh_sa.export(str(out_stl))
        print(f"  Exported -> {out_stl.name}")
        print(f"  Faces: {len(mesh_sa.faces):,}, Watertight: {mesh_sa.is_watertight}, Time: {time.time() - t0:.2f}s")

        out_png = OUTPUT_DIR / f"LatticeSection_1p5inch_{label}_preview.png"
        render_3d_preview(out_stl, out_png, f"Chiral Lattice Sleeve — {label}")


def run_phase3_demo() -> None:
    """Phase 3: Full napkin ring assembly with CAD collar rims."""
    print("\n" + "=" * 65)
    print("PHASE 3: FULL ASSEMBLED CHIRAL NAPKIN RINGS")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 1.5-inch Classic Ring (V1 Base)
    rims_v1, center_v1, h_v1 = extract_custom_rims(BASE_RING_PATH, h_lattice=H_1P5, y_start=Y_START_1P5)
    types = [("tetra", "TetraChiral"), ("tri", "TriChiral")]
    for type_key, label in types:
        t0 = time.time()
        print(f"\n> Assembling V1 Napkin Ring ({label})...")
        m_lat, metrics = generate_chiral_cylinder_lattice(
            center=center_v1,
            r_in=R_IN_1P5,
            r_out=R_OUT_1P5,
            height=h_v1,
            y_base=Y_START_1P5,
            strut_w=2.0,
            n_circumferential=10,
            r_node=2.2,
            chiral_type=type_key,
            add_boundary_rings=True,
        )
        full_ring = rims_v1 + m_lat
        mesh_ring = _manifold_to_trimesh(full_ring)
        out_stl = OUTPUT_DIR / f"NapkinRing_1p5inch_{label}.stl"
        mesh_ring.export(str(out_stl))
        print(f"  Exported -> {out_stl.name}")
        print(f"  Faces: {len(mesh_ring.faces):,}, Watertight: {mesh_ring.is_watertight}, Time: {time.time() - t0:.2f}s")

        out_png = OUTPUT_DIR / f"NapkinRing_1p5inch_{label}_preview.png"
        render_3d_preview(out_stl, out_png, f"Napkin Ring 1.5-in — {label}")

    # 2. 1:2 Ratio Ring (V2 Chamfered Base)
    if BASE_RING_1TO2_PATH.exists():
        h_1to2 = 19.05
        r_in_1to2 = 19.05
        wall_1to2 = 3.0
        r_out_1to2 = r_in_1to2 + wall_1to2
        rims_1to2, center_1to2, _ = extract_custom_rims(BASE_RING_1TO2_PATH, h_lattice=h_1to2, y_start=6.65)

        for type_key, label in types:
            t0 = time.time()
            print(f"\n> Assembling V2 1:2 Napkin Ring ({label})...")
            m_lat_1to2, _ = generate_chiral_cylinder_lattice(
                center=center_1to2,
                r_in=r_in_1to2,
                r_out=r_out_1to2,
                height=h_1to2,
                y_base=6.65,
                strut_w=2.0,
                n_circumferential=10,
                r_node=2.2,
                chiral_type=type_key,
                add_boundary_rings=True,
            )
            full_ring_1to2 = rims_1to2 + m_lat_1to2
            mesh_ring_1to2 = _manifold_to_trimesh(full_ring_1to2)
            out_stl_1to2 = OUTPUT_DIR / f"NapkinRing_1to2_{label}.stl"
            mesh_ring_1to2.export(str(out_stl_1to2))
            print(f"  Exported -> {out_stl_1to2.name}")
            print(f"  Faces: {len(mesh_ring_1to2.faces):,}, Watertight: {mesh_ring_1to2.is_watertight}, Time: {time.time() - t0:.2f}s")

            out_png_1to2 = OUTPUT_DIR / f"NapkinRing_1to2_{label}_preview.png"
            render_3d_preview(out_stl_1to2, out_png_1to2, f"Napkin Ring 1:2 (V2 Base) — {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cylindrical chiral napkin rings.")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 0], default=0,
                        help="Phase to execute: 1 (2D diagnostic), 2 (standalone sleeves), 3 (full rings), 0 (all)")
    args = parser.parse_args()

    if args.phase in (1, 0):
        run_phase1_demo()
    if args.phase in (2, 0):
        run_phase2_demo()
    if args.phase in (3, 0):
        run_phase3_demo()
