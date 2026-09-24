#!/usr/bin/env python
"""
PAM Kinematic Free-Play vs. Tensile Jamming / Locking State Figure.

Illustrates the core mechanical principle of Polycatenated Architected Materials:
1. Rest State (As-Printed): Open clearance Δ = 0.64 mm, fluid-like free-play, zero internal stress.
2. Locked State (Under Tension): Strut-to-strut contact engages (Δ = 0.00 mm),
   freezing kinematic degrees of freedom into a rigid load-bearing chain.
3. Characteristic Non-Linear J-Curve: Free-play extension plateau -> Jamming transition -> Solid load-bearing.

Outputs:
  outputs/diagnostics/pam_kinematic_locking_comparison.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from graphite.explicit.interlinked.pams import (
    generate_truncated_tetrahedron_particle,
    pam_particles_to_meshes,
    _translate_particle,
    particle_pair_clearance,
    particle_strut_segments,
)
from graphite.explicit.interlinked.clearance import segment_segment_distance


def render_kinematic_state(
    particles: list,
    colors: list[str],
    contact_points: list[np.ndarray] | None = None,
    force_arrows: bool = False,
    window_size: list[int] = [1800, 750],
    camera_pos: list | str = "iso",
    zoom: float = 1.25,
) -> np.ndarray:
    """Render a multi-particle kinematic chain with optional contact spheres and force vectors."""
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = "#0a0d14"

    meshes = pam_particles_to_meshes(particles, strut_radius=0.50)

    for m, c in zip(meshes, colors):
        pdata = pv.wrap(m)
        plotter.add_mesh(
            pdata,
            color=c,
            smooth_shading=True,
            specular=0.45,
            specular_power=35,
            show_edges=False,
        )

    # Add glowing contact spheres at contact points
    if contact_points:
        for pt in contact_points:
            sphere = pv.Sphere(radius=0.85, center=pt)
            plotter.add_mesh(
                sphere,
                color="#ff0055",
                emissive=True,
                ambient=0.9,
                smooth_shading=True,
            )

    # Force arrows on outer particles
    if force_arrows:
        # Left tension arrow
        p_left = particles[0].center - np.array([6.0, 0, 0])
        arrow_l = pv.Arrow(start=p_left, direction=[-1, 0, 0], scale=7.0, tip_radius=0.22, shaft_radius=0.10)
        plotter.add_mesh(arrow_l, color="#ffdd00", ambient=0.8)

        # Right tension arrow
        p_right = particles[-1].center + np.array([6.0, 0, 0])
        arrow_r = pv.Arrow(start=p_right, direction=[1, 0, 0], scale=7.0, tip_radius=0.22, shaft_radius=0.10)
        plotter.add_mesh(arrow_r, color="#ffdd00", ambient=0.8)

    if isinstance(camera_pos, str):
        plotter.camera_position = camera_pos
    else:
        plotter.camera_position = camera_pos
    plotter.camera.zoom(zoom)

    img = plotter.screenshot()
    plotter.close()
    return img


def find_contacts_between(pA, pB, r: float = 0.50, tol: float = 0.05) -> list[np.ndarray]:
    """Find contact point locations between two touching particles."""
    sA_p0, sA_p1 = particle_strut_segments(pA)
    sB_p0, sB_p1 = particle_strut_segments(pB)
    pts = []
    for i in range(len(sA_p0)):
        for j in range(len(sB_p0)):
            d = float(segment_segment_distance(sA_p0[i:i+1], sA_p1[i:i+1], sB_p0[j:j+1], sB_p1[j:j+1]))
            if abs(d - 2.0 * r) <= tol:
                pt = (sA_p0[i] + sA_p1[i] + sB_p0[j] + sB_p1[j]) / 4.0
                pts.append(pt)
    return pts


def main() -> int:
    diag_dir = _REPO_ROOT / "outputs" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    out_png = diag_dir / "pam_kinematic_locking_comparison.png"

    print("=" * 80)
    print("Generating PAM Kinematic Free-Play vs. Tensile Jamming / Locking Figure...")
    print("=" * 80)

    # We use a 4-particle C-6-TT chain along the X-axis
    s = 10.0
    a0 = 12.50
    palette = ["#50fa7b", "#ff79c6", "#8be9fd", "#ffb86c"]

    # 1. Rest State (As-Printed)
    print("  [1/3] Building Rest State (As-Printed, Clearance Δ = 0.64 mm)...")
    parts_rest = [
        generate_truncated_tetrahedron_particle(s, center=(i * a0, 0.0, 0.0), particle_id=i)
        for i in range(4)
    ]
    cam = [(18.75, -85.0, 45.0), (18.75, 0.0, 0.0), (0.0, 0.0, 1.0)]
    img_rest = render_kinematic_state(
        parts_rest,
        palette,
        contact_points=None,
        force_arrows=False,
        camera_pos=cam,
        zoom=1.20,
    )

    # 2. Locked State (Under Uniaxial Tension)
    print("  [2/3] Building Locked State (Tensile Displacement dx = 0.642 mm per link)...")
    dx_lock = 0.6421  # displacement where clearance closes to zero
    parts_locked = []
    contact_pts = []
    for i in range(4):
        shift = np.array([i * dx_lock, 0.0, 0.0])
        p = generate_truncated_tetrahedron_particle(s, center=(i * a0, 0.0, 0.0), particle_id=i)
        p_shifted = _translate_particle(p, shift, new_id=i)
        parts_locked.append(p_shifted)

    for i in range(3):
        pts = find_contacts_between(parts_locked[i], parts_locked[i+1], r=0.50, tol=0.08)
        contact_pts.extend(pts)

    print(f"        Identified {len(contact_pts)} active kinematic locking contact points")
    img_locked = render_kinematic_state(
        parts_locked,
        palette,
        contact_points=contact_pts,
        force_arrows=True,
        camera_pos=cam,
        zoom=1.20,
    )

    # -------------------------------------------------------------------------
    # 3. Publication Layout: Visual Comparison + J-Curve Schematic
    # -------------------------------------------------------------------------
    print("  [3/3] Compositing publication figure with constitutive J-curve mechanics...")
    fig = plt.figure(figsize=(22, 12.5), facecolor="#0e1117")

    fig.suptitle(
        "Kinematic Mechanics of Polycatenated Metamaterials (PAMs): Free-Play to Tensile Jamming",
        fontsize=18,
        fontweight="bold",
        color="#ffffff",
        y=0.97,
    )

    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1.2], height_ratios=[1, 1], wspace=0.05, hspace=0.18, left=0.03, right=0.97, top=0.91, bottom=0.06)

    # --- ROW 1: Rest State ---
    ax_rest = fig.add_subplot(gs[0, 0])
    ax_rest.set_facecolor("#0a0d14")
    ax_rest.imshow(img_rest)
    ax_rest.set_xticks([])
    ax_rest.set_yticks([])
    ax_rest.set_title(
        "State I: As-Printed Equilibrium (Kinematic Free-Play & Mechanical Compliance)",
        color="#8be9fd",
        fontsize=13.5,
        fontweight="bold",
        pad=10,
    )
    for s_spine in ax_rest.spines.values():
        s_spine.set_color("#8be9fd")
        s_spine.set_linewidth(1.8)

    ax_rest.text(
        0.02,
        0.06,
        "Zero Applied Load (F = 0) | Physical Clearance: Δ = 0.64 mm everywhere\n"
        "Mechanism: Discrete unbonded cages possess finite translational & rotational rattle space (free-play degrees of freedom).",
        transform=ax_rest.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#8be9fd", alpha=0.92),
    )

    # --- ROW 2: Locked State ---
    ax_lock = fig.add_subplot(gs[1, 0])
    ax_lock.set_facecolor("#0a0d14")
    ax_lock.imshow(img_locked)
    ax_lock.set_xticks([])
    ax_lock.set_yticks([])
    ax_lock.set_title(
        "State II: Uniaxially Stretched (Geometric Jamming & Force Percolation)",
        color="#ff79c6",
        fontsize=13.5,
        fontweight="bold",
        pad=10,
    )
    for s_spine in ax_lock.spines.values():
        s_spine.set_color("#ff79c6")
        s_spine.set_linewidth(1.8)

    ax_lock.text(
        0.02,
        0.06,
        "Tensile Loading (+Fx) | Jamming Displacement: δx = 0.642 mm/cell | Minimum Clearance: Δ → 0.00 mm\n"
        "Mechanism: Opposing struts make geometric contact (highlighted in red spheres), forming a continuous force-bearing skeleton.",
        transform=ax_lock.transAxes,
        ha="left",
        va="bottom",
        color="#f1f5f9",
        fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor="#ff79c6", alpha=0.92),
    )

    # --- RIGHT COLUMN: Educational Constitutive J-Curve Plot ---
    ax_curve = fig.add_subplot(gs[:, 1])
    ax_curve.set_facecolor("#161b22")

    # Synthetic J-curve representing typical PAM constitutive response (Zhou et al., Science 2025)
    strain = np.linspace(0.0, 0.25, 300)
    strain_jam = 0.051  # 0.642 mm / 12.5 mm ≈ 5.1% free play strain
    d_strain = np.maximum(0.0, strain - strain_jam)
    stress = np.where(
        strain < strain_jam,
        0.002 * (strain / strain_jam),  # negligible free-play friction
        0.002 + 45.0 * (d_strain ** 1.35),  # rapid geometric stiffening
    )

    ax_curve.plot(strain * 100, stress, color="#50fa7b", linewidth=3.0, label="PAM Tensile Response")
    ax_curve.axvline(strain_jam * 100, color="#ff0055", linestyle="--", linewidth=1.8, label="Jamming Strain (ε_jam ≈ 5.1%)")

    # Annotate regimes
    ax_curve.fill_between(strain[strain < strain_jam] * 100, stress[strain < strain_jam], color="#8be9fd", alpha=0.15)
    ax_curve.fill_between(strain[strain >= strain_jam] * 100, stress[strain >= strain_jam], color="#ff79c6", alpha=0.15)

    ax_curve.text(
        2.5,
        1.5,
        "Regime I:\nFree-Play Plateau\n(Zero Stress,\nFluid Compliance)",
        color="#8be9fd",
        fontsize=10.0,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0e1117", edgecolor="#8be9fd", alpha=0.85),
    )

    ax_curve.text(
        14.0,
        3.0,
        "Regime II:\nGeometric Jamming\n(Strut-Strut Contact,\nHigh Tensile Modulus)",
        color="#ff79c6",
        fontsize=10.0,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0e1117", edgecolor="#ff79c6", alpha=0.90),
    )

    ax_curve.set_ylim(-0.2, 5.6)
    ax_curve.set_title("Constitutive Mechanical Response", color="#ffffff", fontsize=13, fontweight="bold", pad=12)
    ax_curve.set_xlabel("Tensile Engineering Strain ε (%)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax_curve.set_ylabel("Nominal Tensile Stress σ (MPa)", color="#e2e8f0", fontsize=11, labelpad=8)
    ax_curve.tick_params(colors="#cbd5e1", labelsize=9.5)
    ax_curve.grid(True, linestyle=":", alpha=0.3, color="#718096")
    ax_curve.legend(loc="upper left", facecolor="#0e1117", edgecolor="#2d3748", labelcolor="#f1f5f9", fontsize=9.2)

    for spine in ax_curve.spines.values():
        spine.set_color("#4a5568")
        spine.set_linewidth(1.4)

    print(f"  Saving publication figure to {out_png}...")
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[Done] Figure 4 generated: {out_png}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
