# -*- coding: utf-8 -*-
"""
Generate deliverable presentation figures for the 7 coaster families.

Specifications:
- Canvas aspect ratio: 1024:765 (~1.3386), rendered at 2048 x 1530 px (2x resolution).
- High contrast: Deep solid carbon (#141416) on clean studio neutral (#F8F9FA).
- Realistic soft drop shadow under each coaster for product render elevation.
- At most 3 coasters wide, at most 3 coasters tall.
- Aesthetically pleasing, symmetric, centered layout for every category.
- TPMS drops Split-P, displaying the 8 designs (4 types x 2 slices) in a balanced 3-row layout.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union
from scipy.spatial import cKDTree, Voronoi
import shapely.plotting as spl

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphite.math.tpms import evaluate_tpms_phase
from scripts.coasters.tri_sq_patterns import (
    get_triangle_segments,
    get_square_segments,
    COASTER_TRI_R,
    COASTER_SQ_SIDE,
)

OUTPUT_DIR = REPO_ROOT / "outputs" / "Coasters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 2048
CANVAS_H = 1530
BG_COLOR = (248, 249, 250, 255)
SOLID_COLOR = "#141416"
SOLID_RGBA = (20, 20, 22, 255)

# Fonts
try:
    TITLE_FONT = ImageFont.truetype("segoeuib.ttf", 64)
    LABEL_FONT = ImageFont.truetype("segoeui.ttf", 34)
except Exception:
    try:
        TITLE_FONT = ImageFont.truetype("arialbd.ttf", 64)
        LABEL_FONT = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        TITLE_FONT = ImageFont.load_default()
        LABEL_FONT = ImageFont.load_default()


# =====================================================================
# 1. Geometry Generators (Shapely Polygons)
# =====================================================================

def segments_to_coaster_polygon(segments, strut_width=1.2, r_outer=50.0, r_inner=46.825):
    """Convert 2D line segments to a framed circular coaster polygon."""
    strut_polys = [
        LineString([p1, p2]).buffer(strut_width / 2.0, cap_style=2)
        for p1, p2 in segments
    ]
    lattice_poly = unary_union(strut_polys)
    
    outer_shape = Point(0, 0).buffer(r_outer, resolution=128)
    inner_shape = Point(0, 0).buffer(r_inner, resolution=128)
    frame_poly = outer_shape.difference(inner_shape)
    clipped_lattice = lattice_poly.intersection(inner_shape)
    return clipped_lattice.union(frame_poly)


# --- A15 Lattice ---
def _tile_basis(basis_pts, nx, ny, nz, cell_size):
    tiled_pts = []
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            for k in range(-nz, nz + 1):
                offset = np.array([i, j, k], dtype=np.float64)
                for pt in basis_pts:
                    t_pt = pt + offset - np.array([0.5, 0.5, 0.5])
                    tiled_pts.append(t_pt * cell_size)
    tiled_pts = np.vstack(tiled_pts)
    return np.unique(np.round(tiled_pts, 8), axis=0)


def generate_a15_edges(cell_size=25.0):
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75],
    ], dtype=np.float64)
    pts = _tile_basis(basis, 3, 3, 1, cell_size)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=0.62 * cell_size)
    edges = [(u, v) for u, v in pairs if np.linalg.norm(pts[u] - pts[v]) > 1e-5]
    return pts, edges


def get_a15_polygon(z_offset_frac):
    cell_size = 25.0
    pts, edges = generate_a15_edges(cell_size)
    z_limit = 2.5
    z_offset = z_offset_frac * cell_size
    pts_shifted = pts.copy()
    pts_shifted[:, 2] -= z_offset
    
    segments = []
    for u, v in edges:
        p0, p1 = pts_shifted[u], pts_shifted[v]
        z_min, z_max = min(p0[2], p1[2]), max(p0[2], p1[2])
        if z_min <= z_limit and z_max >= -z_limit:
            p0_2d, p1_2d = p0[:2], p1[:2]
            if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                segments.append((p0_2d, p1_2d))
    return segments_to_coaster_polygon(segments, strut_width=1.2)


# --- C15 Lattice ---
def generate_c15_edges(cell_size=50.0):
    fcc_trans = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ], dtype=np.float64)
    a_base = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=np.float64)
    a_basis = [((ab + t) % 1.0) for ab in a_base for t in fcc_trans]
    b_base = np.array([
        [0.625, 0.625, 0.625],
        [0.625, 0.875, 0.875],
        [0.875, 0.625, 0.875],
        [0.875, 0.875, 0.625],
    ], dtype=np.float64)
    b_basis = [((bb + t) % 1.0) for bb in b_base for t in fcc_trans]
    basis = np.vstack([np.unique(np.round(a_basis, 8), axis=0), np.unique(np.round(b_basis, 8), axis=0)])
    pts = _tile_basis(basis, 2, 2, 1, cell_size)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=0.45 * cell_size)
    edges = [(u, v) for u, v in pairs if np.linalg.norm(pts[u] - pts[v]) > 1e-5]
    return pts, edges


def get_c15_polygon(z_offset_frac):
    cell_size = 50.0
    pts, edges = generate_c15_edges(cell_size)
    z_limit = 5.0
    z_offset = z_offset_frac * cell_size
    pts_shifted = pts.copy()
    pts_shifted[:, 2] -= z_offset
    
    segments = []
    for u, v in edges:
        p0, p1 = pts_shifted[u], pts_shifted[v]
        z_min, z_max = min(p0[2], p1[2]), max(p0[2], p1[2])
        if z_min <= z_limit and z_max >= -z_limit:
            p0_2d, p1_2d = p0[:2], p1[:2]
            if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                segments.append((p0_2d, p1_2d))
    return segments_to_coaster_polygon(segments, strut_width=1.2)


# --- Voronoi ---
def get_voronoi_polygon(num_points=50, seed=42):
    extended_size = 160.0
    area_ratio = (extended_size / 100.0) ** 2
    total_points = int(round(num_points * area_ratio))
    
    np.random.seed(seed)
    pts = np.random.uniform(-extended_size / 2.0, extended_size / 2.0, size=(total_points, 2))
    vor = Voronoi(pts)
    segments = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1 = vor.vertices[ridge[0]]
            p2 = vor.vertices[ridge[1]]
            segments.append((p1, p2))
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- Explicit Tri & Sq ---
def get_tri_polygon(topo):
    side = COASTER_TRI_R * np.sqrt(3.0)
    height = side * np.sqrt(3.0) / 2.0
    segments = get_triangle_segments(topo, side, height)
    return segments_to_coaster_polygon(segments, strut_width=1.3)


def get_sq_polygon(topo):
    segments = get_square_segments(topo, COASTER_SQ_SIDE)
    return segments_to_coaster_polygon(segments, strut_width=1.3)


# --- TPMS Binary RGBA Image ---
def sample_tpms_threshold(lattice_type, target_sf=0.33):
    n = 64
    axis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    vals = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    vals.sort()
    idx = int(target_sf * len(vals))
    return float(vals[idx])


def get_tpms_image(lattice_type, z_val, size_px=500):
    cell_size = 25.0
    omega = 2.0 * np.pi / cell_size
    n = size_px
    x = np.linspace(-52.0, 52.0, n)
    y = np.linspace(-52.0, 52.0, n)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    U = X * omega
    V = Y * omega
    W = np.full_like(X, z_val * omega)
    
    tau = sample_tpms_threshold(lattice_type, 0.33)
    F = evaluate_tpms_phase(lattice_type, U, V, W)
    
    solid = (np.abs(F) <= tau) & (R <= 46.825) | ((R <= 50.0) & (R >= 46.825))
    
    rgba = np.zeros((n, n, 4), dtype=np.uint8)
    rgba[solid] = SOLID_RGBA
    return Image.fromarray(rgba, mode="RGBA")


# =====================================================================
# 2. Rendering & Drop Shadow Engine
# =====================================================================

def render_polygon_to_image(poly, size_px=500):
    """Render Shapely polygon to transparent RGBA image with anti-aliasing."""
    dpi = 120
    fig_size = size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.set_aspect("equal")
    ax.set_xlim(-53, 53)
    ax.set_ylim(-53, 53)
    ax.axis("off")
    
    spl.plot_polygon(poly, ax=ax, facecolor=SOLID_COLOR, edgecolor="none", add_points=False)
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba, mode="RGBA").resize((size_px, size_px), Image.Resampling.LANCZOS)


def add_drop_shadow(coaster_img, blur_radius=18, offset_y=16, shadow_opacity=0.35):
    """Add diffuse Gaussian drop shadow under coaster image."""
    w, h = coaster_img.size
    pad = blur_radius * 3
    large_w, large_h = w + 2 * pad, h + 2 * pad
    
    shadow_img = Image.new("RGBA", (large_w, large_h), (0, 0, 0, 0))
    alpha = coaster_img.split()[3]
    
    alpha_np = np.asarray(alpha, dtype=np.float32) * shadow_opacity
    shadow_mask = Image.fromarray(alpha_np.astype(np.uint8), mode="L")
    
    tinted = Image.new("RGBA", (w, h), (15, 20, 25, 255))
    tinted.putalpha(shadow_mask)
    
    shadow_img.paste(tinted, (pad, pad + offset_y))
    blurred = shadow_img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    composite = Image.new("RGBA", (large_w, large_h), (0, 0, 0, 0))
    composite.paste(blurred, (0, 0), blurred)
    composite.paste(coaster_img, (pad, pad), coaster_img)
    return composite, pad


# =====================================================================
# 3. Canvas Composition & Placement (Symmetric & Centered)
# =====================================================================

def compose_deliverable_canvas(
    title_text,
    items,  # list of (label, image)
    layout_rows,  # list of counts per row, e.g. [3] or [2, 2] or [3, 2] or [3, 2, 3]
    coaster_size_px=500,
    gap_x=100,
    gap_y=80,
):
    """
    Assemble the complete deliverable canvas with title, coaster drop shadows, and labels.
    Every row is centered horizontally around CANVAS_W / 2.
    The entire grid is centered vertically between the header rule and the canvas bottom.
    """
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # Draw Title
    title_bbox = draw.textbbox((0, 0), title_text, font=TITLE_FONT)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = (CANVAS_W - title_w) // 2
    title_y = 70
    draw.text((title_x, title_y), title_text, font=TITLE_FONT, fill=(15, 23, 42, 255))
    
    # Header Accent Rule
    rule_y = title_y + (title_bbox[3] - title_bbox[1]) + 20
    draw.line([(CANVAS_W // 2 - 140, rule_y), (CANVAS_W // 2 + 140, rule_y)], fill=(203, 213, 225, 255), width=3)
    
    num_rows = len(layout_rows)
    content_top = rule_y + 40
    content_bottom = CANVAS_H - 50
    available_h = content_bottom - content_top
    
    # Row unit height = coaster_size_px + label_height + padding
    label_h = 50
    row_unit_h = coaster_size_px + label_h
    total_grid_h = num_rows * row_unit_h + (num_rows - 1) * gap_y
    
    # Vertical start to center the entire grid
    grid_top_y = content_top + max(0, (available_h - total_grid_h) / 2)
    
    # Pre-render shadows for all items
    shadowed_items = []
    for label, img in items:
        scaled = img.resize((coaster_size_px, coaster_size_px), Image.Resampling.LANCZOS)
        shadowed, pad = add_drop_shadow(scaled)
        shadowed_items.append((label, shadowed, pad))
        
    item_idx = 0
    for r_idx, count_in_row in enumerate(layout_rows):
        # Y center of coasters in this row
        row_coaster_cy = grid_top_y + r_idx * (row_unit_h + gap_y) + coaster_size_px / 2.0
        
        # Calculate horizontal positions centered around CANVAS_W / 2
        total_row_w = count_in_row * coaster_size_px + (count_in_row - 1) * gap_x
        row_start_x = (CANVAS_W - total_row_w) / 2.0
        
        for c_idx in range(count_in_row):
            label, shadowed_img, pad = shadowed_items[item_idx]
            item_idx += 1
            
            coaster_cx = row_start_x + c_idx * (coaster_size_px + gap_x) + coaster_size_px / 2.0
            coaster_cy = row_coaster_cy
            
            # Paste shadowed coaster
            paste_x = int(coaster_cx - pad - coaster_size_px / 2.0)
            paste_y = int(coaster_cy - pad - coaster_size_px / 2.0)
            canvas.paste(shadowed_img, (paste_x, paste_y), shadowed_img)
            
            # Label below coaster
            lbl_bbox = draw.textbbox((0, 0), label, font=LABEL_FONT)
            lbl_w = lbl_bbox[2] - lbl_bbox[0]
            lbl_x = int(coaster_cx - lbl_w / 2.0)
            lbl_y = int(coaster_cy + coaster_size_px / 2.0 + 16)
            draw.text((lbl_x, lbl_y), label, font=LABEL_FONT, fill=(51, 65, 85, 255))
            
    return canvas.convert("RGB")


# =====================================================================
# 4. Deliverable Figure Builders
# =====================================================================

def make_a15_figure():
    print("Building A15 deliverable figure...")
    items = [
        ("A15_v1", render_polygon_to_image(get_a15_polygon(0.0), 540)),
        ("A15_v2", render_polygon_to_image(get_a15_polygon(0.125), 540)),
        ("A15_v3", render_polygon_to_image(get_a15_polygon(0.25), 540)),
    ]
    canvas = compose_deliverable_canvas(
        "A15 2D-preview (Z = 0)",
        items,
        layout_rows=[3],
        coaster_size_px=520,
        gap_x=120,
    )
    out_path = OUTPUT_DIR / "a15_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_c15_figure():
    print("Building C15 deliverable figure...")
    items = [
        ("C15_v1", render_polygon_to_image(get_c15_polygon(0.0), 540)),
        ("C15_v2", render_polygon_to_image(get_c15_polygon(0.0625), 540)),
        ("C15_v3", render_polygon_to_image(get_c15_polygon(0.125), 540)),
    ]
    canvas = compose_deliverable_canvas(
        "C15 2D-preview (Z = 0)",
        items,
        layout_rows=[3],
        coaster_size_px=520,
        gap_x=120,
    )
    out_path = OUTPUT_DIR / "c15_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_voronoi_dense_figure():
    print("Building Voronoi Dense deliverable figure...")
    items = [
        ("small_v1", render_polygon_to_image(get_voronoi_polygon(100, 42), 540)),
        ("small_v2", render_polygon_to_image(get_voronoi_polygon(100, 43), 540)),
        ("small_v3", render_polygon_to_image(get_voronoi_polygon(100, 44), 540)),
    ]
    canvas = compose_deliverable_canvas(
        "Voroni Dense 2D-preview (Z = 0)",
        items,
        layout_rows=[3],
        coaster_size_px=520,
        gap_x=120,
    )
    out_path = OUTPUT_DIR / "voronoi_dense_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_voronoi_sparse_figure():
    print("Building Voronoi Sparse deliverable figure...")
    items = [
        ("large_v1", render_polygon_to_image(get_voronoi_polygon(50, 42), 480)),
        ("large_v2", render_polygon_to_image(get_voronoi_polygon(50, 43), 480)),
        ("large_v3", render_polygon_to_image(get_voronoi_polygon(50, 44), 480)),
        ("large_v4", render_polygon_to_image(get_voronoi_polygon(50, 45), 480)),
    ]
    canvas = compose_deliverable_canvas(
        "Voroni Sparse 2D-preview (Z = 0)",
        items,
        layout_rows=[2, 2],
        coaster_size_px=460,
        gap_x=180,
        gap_y=90,
    )
    out_path = OUTPUT_DIR / "voronoi_sparse_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_explicit_square_figure():
    print("Building Explicit Square deliverable figure...")
    topos = ["Grid", "Icosahedral", "Kelvin", "Tesseract"]
    items = [
        (f"Sq_{topo}", render_polygon_to_image(get_sq_polygon(topo), 480))
        for topo in topos
    ]
    canvas = compose_deliverable_canvas(
        "Explicit Square 2D-preview (Z = 0)",
        items,
        layout_rows=[2, 2],
        coaster_size_px=460,
        gap_x=180,
        gap_y=90,
    )
    out_path = OUTPUT_DIR / "explicit_square_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_explicit_tri_figure():
    print("Building Explicit Tri deliverable figure...")
    topos = ["Tetrahedral", "Icosahedral", "Kelvin", "Tesseract", "Rhombic"]
    items = [
        (f"Tri_{topo}", render_polygon_to_image(get_tri_polygon(topo), 460))
        for topo in topos
    ]
    canvas = compose_deliverable_canvas(
        "Explicit Tri 2D-preview (Z = 0)",
        items,
        layout_rows=[3, 2],
        coaster_size_px=440,
        gap_x=110,
        gap_y=80,
    )
    out_path = OUTPUT_DIR / "explicit_tri_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_tpms_figure():
    print("Building TPMS deliverable figure (8 designs, Split-P dropped)...")
    # 4 types x 2 heights (Z = 0.0 and Z = 2.4 mm)
    items = [
        ("Gyroid (Z=0)", get_tpms_image("Gyroid", 0.0, 360)),
        ("Diamond (Z=0)", get_tpms_image("Diamond", 0.0, 360)),
        ("Lidinoid (Z=0)", get_tpms_image("Lidinoid", 0.0, 360)),
        ("Neovius (Z=0)", get_tpms_image("Neovius", 0.0, 360)),
        ("Neovius (Z=2.4)", get_tpms_image("Neovius", 2.4, 360)),
        ("Gyroid (Z=2.4)", get_tpms_image("Gyroid", 2.4, 360)),
        ("Diamond (Z=2.4)", get_tpms_image("Diamond", 2.4, 360)),
        ("Lidinoid (Z=2.4)", get_tpms_image("Lidinoid", 2.4, 360)),
    ]
    canvas = compose_deliverable_canvas(
        "TPMS 2D-preview (Z = 0 & Z = 2.4mm)",
        items,
        layout_rows=[3, 2, 3],
        coaster_size_px=330,
        gap_x=80,
        gap_y=40,
    )
    out_path = OUTPUT_DIR / "tpms_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def main():
    print("Generating all 7 high-contrast deliverable figures...")
    make_a15_figure()
    make_c15_figure()
    make_voronoi_dense_figure()
    make_voronoi_sparse_figure()
    make_explicit_square_figure()
    make_explicit_tri_figure()
    make_tpms_figure()
    print("All 7 deliverable figures generated successfully!")


if __name__ == "__main__":
    main()
