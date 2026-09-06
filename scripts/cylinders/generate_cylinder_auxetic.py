# -*- coding: utf-8 -*-
"""
Generate 3D cylindrical auxetic re-entrant lattices and assemble them
into 3D printable Napkin Rings with watertight CAD rims.

Based on Chen et al. (2020) "Re-entrant Auxetic Lattices with Enhanced Stiffness":
- Base: Re-entrant hexagonal cell (hourglass/bowtie columns with horizontal waist bridges)
- Type-A Variant: Reinforced with a horizontal strengthening rib across the cell waist
- Type-B Variant: Reinforced with a vertical central strengthening rib along the cell axis
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


def generate_chen_reentrant_2d_segments(
    c_mid: float,
    height: float,
    y_base: float,
    n_circumferential: int = 10,
    m_vertical: int = 3,
    w_top_ratio: float = 0.65,
    w_waist_ratio: float = 0.30,
    variant: str = "base",
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Generate 2D (u, y) line segments for the Chen et al. (2020) re-entrant lattices.

    Parameters
    ----------
    c_mid : float
        Mid-surface circumference in mm (2 * pi * R_mid).
    height : float
        Usable lattice height in mm.
    y_base : float
        Y coordinate of the bottom boundary.
    n_circumferential : int
        Number of periodic cell columns along circumference.
    m_vertical : int
        Number of cell rows along height.
    w_top_ratio : float
        Width of top/bottom caps as fraction of cell width a (default 0.65).
    w_waist_ratio : float
        Width of waist as fraction of cell width a (default 0.30).
    variant : str
        'base' (re-entrant hexagonal cell), 'type_a' (horizontal rib), or 'type_b' (vertical rib).

    Returns
    -------
    segments : list of tuple(p1, p2)
        Deduplicated 2D line segments in (u, y) space.
    metrics : dict
        Geometric parameters.
    """
    a = c_mid / float(n_circumferential)
    b = height / float(m_vertical)
    w_top = w_top_ratio * a
    w_waist = w_waist_ratio * a
    h_waist_y = b / 2.0

    # Calculate angle of slanted strut with vertical:
    delta_u = (w_top - w_waist) / 2.0
    theta_from_vertical_deg = np.degrees(np.arctan(delta_u / (b / 2.0)))
    theta_from_horizontal_deg = 90.0 - theta_from_vertical_deg

    metrics = {
        "variant": variant,
        "c_mid": float(c_mid),
        "height": float(height),
        "y_base": float(y_base),
        "n_circumferential": int(n_circumferential),
        "m_vertical": int(m_vertical),
        "a_cell_width_mm": float(a),
        "b_cell_height_mm": float(b),
        "w_top_mm": float(w_top),
        "w_waist_mm": float(w_waist),
        "theta_from_vertical_deg": float(theta_from_vertical_deg),
        "theta_from_horizontal_deg": float(theta_from_horizontal_deg),
    }

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_circumferential):
        u_c = (i + 0.5) * a
        u_tl = u_c - w_top / 2.0
        u_tr = u_c + w_top / 2.0
        u_wl = u_c - w_waist / 2.0
        u_wr = u_c + w_waist / 2.0

        for j in range(m_vertical):
            y_bot = y_base + j * b
            y_mid = y_bot + h_waist_y
            y_top = y_bot + b

            p_tl = np.array([u_tl, y_top])
            p_tr = np.array([u_tr, y_top])
            p_bl = np.array([u_tl, y_bot])
            p_br = np.array([u_tr, y_bot])
            p_wl = np.array([u_wl, y_mid])
            p_wr = np.array([u_wr, y_mid])

            # 1. Top and bottom horizontal boundary caps
            raw_segments.append((p_tl, p_tr))
            raw_segments.append((p_bl, p_br))

            # 2. Slanted re-entrant struts
            raw_segments.append((p_tl, p_wl))
            raw_segments.append((p_bl, p_wl))
            raw_segments.append((p_tr, p_wr))
            raw_segments.append((p_br, p_wr))

            # 3. Horizontal waist connector to the right adjacent cell
            u_next_c = (i + 1.5) * a
            u_next_wl = u_next_c - w_waist / 2.0
            p_next_wl = np.array([u_next_wl, y_mid])
            raw_segments.append((p_wr, p_next_wl))

            # 4. Variant-specific reinforcement struts:
            if variant == "type_a":
                # Horizontal rib across the waist
                raw_segments.append((p_wl, p_wr))
            elif variant == "type_b":
                # Vertical central rib
                p_c_bot = np.array([u_c, y_bot])
                p_c_top = np.array([u_c, y_top])
                raw_segments.append((p_c_bot, p_c_top))

    unique_segments = dedupe_segments(raw_segments, c_mid)
    metrics["n_segments"] = len(unique_segments)
    return unique_segments, metrics


def plot_chen_variants_comparison(
    c_mid: float,
    height: float,
    y_base: float,
    out_png_path: Path,
) -> None:
    """Generate side-by-side comparison of Base, Type-A, and Type-B 2D unrolled designs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=180)
    variants = ["base", "type_a", "type_b"]
    titles = [
        "(a) Re-Entrant Hexagonal Cell\n(Open Waist)",
        "(b) Type-A Variant Cell\n(Horizontal Strengthening Rib)",
        "(c) Type-B Variant Cell\n(Vertical Strengthening Rib)"
    ]

    for ax, var, title in zip(axes, variants, titles):
        segments, metrics = generate_chen_reentrant_2d_segments(
            c_mid=c_mid,
            height=height,
            y_base=y_base,
            n_circumferential=10,
            m_vertical=3,
            variant=var,
        )

        domain_box = plt.Rectangle(
            (0.0, y_base), c_mid, height,
            fill=False, edgecolor="crimson", lw=1.5, ls="--", zorder=10
        )
        ax.add_patch(domain_box)

        for p1, p2 in segments:
            pa = np.copy(p1)
            pb = np.copy(p2)
            if pb[0] - pa[0] < -c_mid / 2.0:
                pb[0] += c_mid
            elif pa[0] - pb[0] < -c_mid / 2.0:
                pa[0] += c_mid
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#1f77b4", lw=2.2, solid_capstyle="round")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Circumferential Coordinate u (mm)", fontsize=9)
        ax.set_ylabel("Axial Coordinate y (mm)", fontsize=9)
        ax.set_xlim(-0.02 * c_mid, 1.02 * c_mid)
        ax.set_ylim(y_base - 0.05 * height, y_base + 1.05 * height)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_png_path))
    plt.close(fig)
    print(f"  [Comparison Plot] Saved -> {out_png_path.name}")


def generate_auxetic_cylinder_lattice(
    center: np.ndarray,
    r_in: float = R_IN_1P5,
    r_out: float = R_OUT_1P5,
    height: float = H_1P5,
    y_base: float = Y_START_1P5,
    strut_w: float = 2.0,
    n_circumferential: int = 10,
    m_vertical: int = 3,
    variant: str = "base",
    add_boundary_rings: bool = True,
) -> tuple[m3d.Manifold, dict]:
    """Generate 3D cylindrical auxetic lattice based on the Chen et al. design."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid

    segments, metrics = generate_chen_reentrant_2d_segments(
        c_mid=c_mid,
        height=height,
        y_base=y_base,
        n_circumferential=n_circumferential,
        m_vertical=m_vertical,
        variant=variant,
    )

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
    """Phase 1: 2D segment generator and 3-variant comparison plot."""
    print("\n" + "=" * 65)
    print("PHASE 1: CHEN ET AL. RE-ENTRANT AUXETIC 2D DIAGNOSTICS")
    print("=" * 65)
    r_mid = (R_IN_1P5 + R_OUT_1P5) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    out_cmp = OUTPUT_DIR / "chen_variants_2d_comparison.png"
    plot_chen_variants_comparison(c_mid=c_mid, height=H_1P5, y_base=Y_START_1P5, out_png_path=out_cmp)


def run_phase2_demo() -> None:
    """Phase 2: 3D standalone sleeves for all 3 variants."""
    print("\n" + "=" * 65)
    print("PHASE 2: 3D STANDALONE AUXETIC SLEEVES (CHEN ET AL.)")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r_in = R_IN_1P5
    r_out = R_OUT_1P5
    height = H_1P5
    center_sa = np.array([r_out, height / 2.0, r_out], dtype=np.float64)

    variants = [("base", "Base"), ("type_a", "TypeA"), ("type_b", "TypeB")]
    for var_key, var_label in variants:
        t0 = time.time()
        print(f"\n> Generating standalone sleeve ({var_label})...")
        m_sleeve, metrics = generate_auxetic_cylinder_lattice(
            center=center_sa,
            r_in=r_in,
            r_out=r_out,
            height=height,
            y_base=0.0,
            strut_w=2.0,
            n_circumferential=10,
            m_vertical=3,
            variant=var_key,
            add_boundary_rings=True,
        )
        mesh_sa = _manifold_to_trimesh(m_sleeve)
        out_stl = OUTPUT_DIR / f"LatticeSection_1p5inch_Auxetic_{var_label}.stl"
        mesh_sa.export(str(out_stl))
        print(f"  Exported -> {out_stl.name}")
        print(f"  Faces: {len(mesh_sa.faces):,}, Watertight: {mesh_sa.is_watertight}, Time: {time.time() - t0:.2f}s")

        out_png = OUTPUT_DIR / f"LatticeSection_1p5inch_Auxetic_{var_label}_preview.png"
        render_3d_preview(out_stl, out_png, f"Auxetic Sleeve — Chen et al. {var_label}")


def run_phase3_demo() -> None:
    """Phase 3: Full napkin ring assembly with CAD collar rims."""
    print("\n" + "=" * 65)
    print("PHASE 3: FULL ASSEMBLED AUXETIC NAPKIN RINGS (CHEN ET AL.)")
    print("=" * 65)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 1.5-inch Classic Ring (V1 Base)
    rims_v1, center_v1, h_v1 = extract_custom_rims(BASE_RING_PATH, h_lattice=H_1P5, y_start=Y_START_1P5)
    variants = [("base", "Base"), ("type_a", "TypeA"), ("type_b", "TypeB")]
    for var_key, var_label in variants:
        t0 = time.time()
        print(f"\n> Assembling V1 Napkin Ring ({var_label})...")
        m_lat, metrics = generate_auxetic_cylinder_lattice(
            center=center_v1,
            r_in=R_IN_1P5,
            r_out=R_OUT_1P5,
            height=h_v1,
            y_base=Y_START_1P5,
            strut_w=2.0,
            n_circumferential=10,
            m_vertical=3,
            variant=var_key,
            add_boundary_rings=True,
        )
        full_ring = rims_v1 + m_lat
        mesh_ring = _manifold_to_trimesh(full_ring)
        out_stl = OUTPUT_DIR / f"NapkinRing_1p5inch_Auxetic_{var_label}.stl"
        mesh_ring.export(str(out_stl))
        print(f"  Exported -> {out_stl.name}")
        print(f"  Faces: {len(mesh_ring.faces):,}, Watertight: {mesh_ring.is_watertight}, Time: {time.time() - t0:.2f}s")

        out_png = OUTPUT_DIR / f"NapkinRing_1p5inch_Auxetic_{var_label}_preview.png"
        render_3d_preview(out_stl, out_png, f"Napkin Ring 1.5-in — Chen et al. {var_label}")

    # 2. 1:2 Ratio Ring (V2 Chamfered Base)
    if BASE_RING_1TO2_PATH.exists():
        h_1to2 = 19.05
        r_in_1to2 = 19.05
        wall_1to2 = 3.0
        r_out_1to2 = r_in_1to2 + wall_1to2
        rims_1to2, center_1to2, _ = extract_custom_rims(BASE_RING_1TO2_PATH, h_lattice=h_1to2, y_start=6.65)

        for var_key, var_label in [("base", "Base"), ("type_b", "TypeB")]:
            t0 = time.time()
            print(f"\n> Assembling V2 1:2 Napkin Ring ({var_label})...")
            m_lat_1to2, _ = generate_auxetic_cylinder_lattice(
                center=center_1to2,
                r_in=r_in_1to2,
                r_out=r_out_1to2,
                height=h_1to2,
                y_base=6.65,
                strut_w=2.0,
                n_circumferential=10,
                m_vertical=2,
                variant=var_key,
                add_boundary_rings=True,
            )
            full_ring_1to2 = rims_1to2 + m_lat_1to2
            mesh_ring_1to2 = _manifold_to_trimesh(full_ring_1to2)
            out_stl_1to2 = OUTPUT_DIR / f"NapkinRing_1to2_Auxetic_{var_label}.stl"
            mesh_ring_1to2.export(str(out_stl_1to2))
            print(f"  Exported -> {out_stl_1to2.name}")
            print(f"  Faces: {len(mesh_ring_1to2.faces):,}, Watertight: {mesh_ring_1to2.is_watertight}, Time: {time.time() - t0:.2f}s")

            out_png_1to2 = OUTPUT_DIR / f"NapkinRing_1to2_Auxetic_{var_label}_preview.png"
            render_3d_preview(out_stl_1to2, out_png_1to2, f"Napkin Ring 1:2 (V2 Base) — {var_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cylindrical auxetic napkin rings (Chen et al.).")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 0], default=0,
                        help="Phase to execute: 1 (2D diagnostic), 2 (standalone sleeves), 3 (full rings), 0 (all)")
    args = parser.parse_args()

    if args.phase in (1, 0):
        run_phase1_demo()
    if args.phase in (2, 0):
        run_phase2_demo()
    if args.phase in (3, 0):
        run_phase3_demo()
