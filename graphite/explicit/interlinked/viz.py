"""
graphite/explicit/interlinked/viz.py

Headless Scientific Visualization Engine for Graphite Interlinked & PAMs.
Provides non-glare matte scientific rendering, multi-point studio lighting,
crystallographic sublattice palette mapping, and longitudinal cutaways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any
import numpy as np
import pyvista as pv
import trimesh

from .particle import InterlinkedParticle
from .generator import solidify_particle_prototype


# Curated High-Contrast Matte Palette Standard (specular=0.12, ambient=0.40, diffuse=0.65)
MATTE_PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "bg_top": "#ffffff",
        "bg_bottom": "#f1f5f9",
        "sublattice_a": "#2B5C8F",   # Deep Steel Blue
        "sublattice_b": "#D95F02",   # Vermilion Orange
        "monopartite":  "#1B9E77",   # Deep Teal / Emerald
        "text_color":   "#0f172a",   # Dark Slate
    },
    "neutral": {
        "bg_top": "#e2e8f0",
        "bg_bottom": "#cbd5e1",
        "sublattice_a": "#1e3a8a",
        "sublattice_b": "#c2410c",
        "monopartite":  "#047857",
        "text_color":   "#1e293b",
    },
    "dark": {
        "bg_top": "#1e293b",
        "bg_bottom": "#0f172a",
        "sublattice_a": "#38bdf8",   # Sky Blue
        "sublattice_b": "#fb923c",   # Warm Amber
        "monopartite":  "#34d399",   # Mint Emerald
        "text_color":   "#f8fafc",   # Light Slate
    },
}


def render_matte_gallery_specimen(
    particles: Sequence[InterlinkedParticle],
    out_png: Path | str,
    title: str = "",
    subtitle: str = "",
    cutaway: bool = False,
    theme: str = "light",
    window_size: tuple[int, int] = (1920, 1080),
) -> Path:
    """
    Render particles with matte scientific shading, soft multi-point studio lighting,
    and optional longitudinal cutaway clipping.

    Args:
        particles: Sequence of InterlinkedParticle instances.
        out_png: Output image path.
        title: Main title string.
        subtitle: Optional subtitle or metric annotations.
        cutaway: If True, hides particles whose centroids cross the midplane to expose interior chains.
        theme: 'light', 'neutral', or 'dark'.
        window_size: (width, height) pixel resolution.

    Returns:
        Path to rendered PNG.
    """
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not particles:
        raise ValueError("Cannot render empty particle list.")

    palette = MATTE_PALETTES.get(theme, MATTE_PALETTES["light"])

    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(palette["bg_top"], top=palette["bg_bottom"])

    # Determine domain bounds for cutaway
    all_centers = np.array([p.center for p in particles])
    x_mid = float(np.median(all_centers[:, 0]))

    proto_cache: dict[str, trimesh.Trimesh] = {}

    for p in particles:
        # If cutaway enabled, filter particles whose center is past the cut plane
        if cutaway and p.center[0] > x_mid + 1e-3:
            continue

        r_wire = p.effective_wire_radius(fallback=0.35)
        k = f"{p.geometry.geometry_type}_{len(p.geometry.nodes)}_{len(p.geometry.struts)}_{p.sublattice_id}_r{r_wire:.4f}"
        if k not in proto_cache:
            proto_cache[k] = solidify_particle_prototype(
                p.geometry,
                strut_radius=r_wire,
                circular_segments=16,
            )
        base_m = proto_cache[k]

        # Rigid world transform
        V_homo = np.hstack([base_m.vertices, np.ones((len(base_m.vertices), 1), dtype=np.float64)])
        V_w = (p.transform @ V_homo.T).T[:, :3]
        if np.linalg.det(p.transform[:3, :3]) < 0:
            F_w = base_m.faces[:, [0, 2, 1]]
        else:
            F_w = base_m.faces

        mesh_w = trimesh.Trimesh(vertices=V_w, faces=F_w, process=False)
        pv_m = pv.wrap(mesh_w)

        # Color assignment based on sublattice
        sub = str(p.sublattice_id).upper()
        if sub == "B":
            mesh_color = palette["sublattice_b"]
        elif sub == "A":
            mesh_color = palette["sublattice_a"]
        else:
            mesh_color = palette["monopartite"]

        # Matte scientific shading parameters (eliminates specular glare)
        plotter.add_mesh(
            pv_m,
            color=mesh_color,
            specular=0.12,
            specular_power=15,
            ambient=0.40,
            diffuse=0.65,
            smooth_shading=True,
            show_edges=False,
        )

    # Lighting: Soft 3-point directional studio lights
    plotter.enable_anti_aliasing("ssaa")
    light_key = pv.Light(position=(80, -80, 100), focal_point=(0, 0, 0), intensity=0.75)
    light_fill = pv.Light(position=(-60, 60, 50), focal_point=(0, 0, 0), intensity=0.40)
    light_back = pv.Light(position=(0, -80, -40), focal_point=(0, 0, 0), intensity=0.25)
    plotter.add_light(light_key)
    plotter.add_light(light_fill)
    plotter.add_light(light_back)

    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.azimuth += 18.0
    plotter.camera.elevation += 14.0
    plotter.reset_camera()

    # Title & Subtitle banner
    if title or subtitle:
        header = f"{title}\n{subtitle}" if subtitle else title
        plotter.add_text(
            header,
            position="upper_left",
            font_size=11,
            color=palette["text_color"],
            font="arial",
        )

    plotter.screenshot(str(out_path))
    plotter.close()
    return out_path
