# -*- coding: utf-8 -*-
"""
Generate combined deliverable presentation figures for the 4 coaster categories.

Figures:
1. "C15/A15"   - 6 coasters (Row 1: C15_v1, C15_v2, C15_v3; Row 2: A15_v1, A15_v2, A15_v3)
2. "Voroni"    - 6 coasters (Row 1: small_v1, small_v2, small_v3; Row 2: large_v1, large_v2, large_v3; large_v4 dropped)
3. "Explicit"  - 9 coasters (3x3 grid combining all Triangle and Square lattices)
4. "TPMS"      - 8 coasters (Row 1: Z=0 Gyroid, Diamond, Lidinoid; Row 2: Neovius dual slices; Row 3: Z=2.4)

Rules:
- Canvas aspect ratio: 1024:765 (~1.3386), rendered at 2048 x 1530 px (2x resolution).
- Center-to-center spacing of each coaster equal to 1.25 x diameter (dx = dy = 1.25 * D).
- Coaster titles placed directly above each coaster, sized 3x as large.
- High contrast: Deep solid carbon (#141416) on clean studio neutral (#F8F9FA).
- Realistic soft drop shadow under each coaster for product elevation.
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
    MAIN_TITLE_FONT = ImageFont.truetype("segoeuib.ttf", 68)
except Exception:
    try:
        MAIN_TITLE_FONT = ImageFont.truetype("arialbd.ttf", 68)
    except Exception:
        MAIN_TITLE_FONT = ImageFont.load_default()


def get_coaster_title_font(text, max_w, default_size=52):
    """
    Get bold font scaled as large as possible (~3x original size)
    while ensuring the text does not exceed max_w.
    """
    size = default_size
    while size >= 24:
        for font_name in ["segoeuib.ttf", "arialbd.ttf"]:
            try:
                font = ImageFont.truetype(font_name, size)
                bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                if w <= max_w:
                    return font
                break
            except Exception:
                continue
        size -= 2
    try:
        return ImageFont.truetype("arialbd.ttf", 24)
    except Exception:
        return ImageFont.load_default()


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
    
    a_pts = _tile_basis(a_basis, 2, 2, 1, cell_size)
    b_pts = _tile_basis(b_basis, 2, 2, 1, cell_size)
    
    tree_a = cKDTree(a_pts)
    pairs_aa = tree_a.query_pairs(r=0.44 * cell_size)
    
    tree_b = cKDTree(b_pts)
    pairs_ab = tree_a.query_ball_tree(tree_b, r=0.42 * cell_size)
    
    edges_ab = []
    for u, neighbors in enumerate(pairs_ab):
        for v in neighbors:
            edges_ab.append((a_pts[u], b_pts[v]))
            
    edges_aa = [(a_pts[u], a_pts[v]) for u, v in pairs_aa]
    all_edges = edges_aa + edges_ab
    return all_edges


def get_c15_polygon(z_offset_frac):
    cell_size = 50.0
    all_edges = generate_c15_edges(cell_size)
    z_limit = 5.0
    z_offset = z_offset_frac * cell_size
    
    segments = []
    for p0_orig, p1_orig in all_edges:
        p0 = p0_orig.copy()
        p1 = p1_orig.copy()
        p0[2] -= z_offset
        p1[2] -= z_offset
        z_min, z_max = min(p0[2], p1[2]), max(p0[2], p1[2])
        if z_min <= z_limit and z_max >= -z_limit:
            p0_2d, p1_2d = p0[:2], p1[:2]
            if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                segments.append((p0_2d, p1_2d))
    return segments_to_coaster_polygon(segments, strut_width=1.2)


# --- Voronoi ---
def get_voronoi_polygon(n_pts, seed, strut_width=1.0):
    np.random.seed(seed)
    r = np.sqrt(np.random.uniform(0, 48.0**2, n_pts))
    theta = np.random.uniform(0, 2 * np.pi, n_pts)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Boundary points to bound outer cells
    b_theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    b_pts = np.column_stack([60.0 * np.cos(b_theta), 60.0 * np.sin(b_theta)])
    all_pts = np.vstack([pts, b_pts])
    
    vor = Voronoi(all_pts)
    segments = []
    for p1_idx, p2_idx in vor.ridge_vertices:
        if p1_idx >= 0 and p2_idx >= 0:
            segments.append((vor.vertices[p1_idx], vor.vertices[p2_idx]))
    return segments_to_coaster_polygon(segments, strut_width=strut_width)


# --- Explicit Tri & Sq ---
def get_tri_polygon(topo):
    side = COASTER_TRI_R * (2.0 if topo == "Tesseract" else 1.0)
    height = side * np.sqrt(3.0) / 2.0
    segments = get_triangle_segments(topo, side, height)
    return segments_to_coaster_polygon(segments, strut_width=1.3)


def get_sq_polygon(topo):
    segments = get_square_segments(topo, COASTER_SQ_SIDE)
    return segments_to_coaster_polygon(segments, strut_width=1.3)


# --- TPMS Binary RGBA Image (High Resolution Anti-Aliasing) ---
def sample_tpms_threshold(lattice_type, target_sf=0.33):
    n = 64
    axis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    vals = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    vals.sort()
    idx = int(target_sf * len(vals))
    return float(vals[idx])


def get_tpms_image(lattice_type, z_val, size_px=380):
    cell_size = 25.0
    omega = 2.0 * np.pi / cell_size
    n = max(size_px * 2, 800)
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
    raw_img = Image.fromarray(rgba, mode="RGBA")
    return raw_img.resize((size_px, size_px), Image.Resampling.LANCZOS)


# =====================================================================
# 2. Rendering & Drop Shadow Engine
# =====================================================================

def render_polygon_to_image(poly, size_px=750):
    """Render Shapely polygon to transparent RGBA image with high-res anti-aliasing."""
    dpi = 130
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


def add_drop_shadow(coaster_img, blur_radius=16, offset_y=12, shadow_opacity=0.28):
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
# 3. Canvas Composition & Placement (Center-to-Center Spacing = 1.25 x D)
# =====================================================================

def compose_deliverable_canvas(
    title_text,
    items,  # list of (label, PIL.Image)
    layout_rows,  # list of counts per row, e.g. [3, 3] or [3, 3, 3] or [3, 2, 3]
    coaster_size_px=540,
    default_label_size=56,
    pad_above=14,
):
    """
    Assemble the complete deliverable canvas.
    - Pitch = 1.25 * coaster_size_px (center-to-center spacing equal to 1.25 x diameter).
    - Coaster titles placed directly above each coaster, scaled ~3x larger.
    - Entire grid is centered vertically and each row is centered horizontally.
    """
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # 1. Main Header
    title_bbox = draw.textbbox((0, 0), title_text, font=MAIN_TITLE_FONT)
    title_x = int(round((CANVAS_W - (title_bbox[2] + title_bbox[0])) / 2.0))
    title_y = 28
    draw.text((title_x, title_y), title_text, font=MAIN_TITLE_FONT, fill=(15, 23, 42, 255))
    
    # Bottom of main title glyphs
    title_bottom = title_y + title_bbox[3]
    
    # 2. Grid Geometry Math (Center-to-Center = 1.25 * D)
    D = coaster_size_px
    pitch = 1.25 * D
    num_rows = len(layout_rows)
    
    content_top = title_bottom + 24
    content_bottom = CANVAS_H - 28
    available_h = content_bottom - content_top
    
    # Sample title text height to calculate bounding box
    sample_font = get_coaster_title_font("Sample (Z=0)", max_w=pitch * 0.9, default_size=default_label_size)
    sample_bbox = draw.textbbox((0, 0), "Sample (Z=0)", font=sample_font)
    title_text_h = sample_bbox[3] - sample_bbox[1]
    
    # Total grid vertical span: from top of Row 0 title to bottom of last row coaster
    H_total = (num_rows - 1) * pitch + D + pad_above + title_text_h
    
    # Vertically center the entire grid within available vertical content space
    y_grid_top = content_top + max(0.0, (available_h - H_total) / 2.0)
    y0 = y_grid_top + title_text_h + pad_above + D / 2.0
    
    # Shadow blur and offset scaled proportionally to diameter
    blur_rad = int(round(16 * (D / 500.0)))
    offset_y = int(round(12 * (D / 500.0)))
    
    # Pre-render shadows for all items
    shadowed_items = []
    for label, img in items:
        scaled = img.resize((D, D), Image.Resampling.LANCZOS)
        shadowed, pad = add_drop_shadow(scaled, blur_radius=blur_rad, offset_y=offset_y, shadow_opacity=0.28)
        shadowed_items.append((label, shadowed, pad))
        
    item_idx = 0
    for r_idx, count_in_row in enumerate(layout_rows):
        row_cy = y0 + r_idx * pitch
        
        # Center this row horizontally around CANVAS_W / 2 = 1024
        # Distance from first to last coaster in row = (count - 1) * pitch
        row_x0 = (CANVAS_W / 2.0) - ((count_in_row - 1) * pitch / 2.0)
        
        for c_idx in range(count_in_row):
            label, shadowed_img, pad = shadowed_items[item_idx]
            item_idx += 1
            
            coaster_cx = row_x0 + c_idx * pitch
            coaster_cy = row_cy
            
            # Coaster top edge
            coaster_top_y = coaster_cy - D / 2.0
            
            # Paste shadowed coaster
            paste_x = int(round(coaster_cx - pad - D / 2.0))
            paste_y = int(round(coaster_cy - pad - D / 2.0))
            canvas.paste(shadowed_img, (paste_x, paste_y), shadowed_img)
            
            # Coaster title directly above coaster
            lbl_font = get_coaster_title_font(label, max_w=pitch * 0.88, default_size=default_label_size)
            lbl_bbox = draw.textbbox((0, 0), label, font=lbl_font)
            lbl_w = lbl_bbox[2] - lbl_bbox[0]
            lbl_h = lbl_bbox[3] - lbl_bbox[1]
            
            # Centered horizontally over the coaster
            lbl_x = int(round(coaster_cx - (lbl_bbox[2] + lbl_bbox[0]) / 2.0))
            # Exactly pad_above pixels above coaster top edge
            lbl_y = int(round(coaster_top_y - pad_above - lbl_bbox[3]))
            
            draw.text((lbl_x, lbl_y), label, font=lbl_font, fill=(30, 41, 59, 255))
            
    return canvas.convert("RGB")


# =====================================================================
# 4. Deliverable Figure Builders
# =====================================================================

def make_c15_a15_figure():
    print("Building 'C15/A15' deliverable figure (6 coasters, center-to-center = 1.25x D)...")
    items = [
        # Row 1: C15
        ("C15_v1", render_polygon_to_image(get_c15_polygon(0.0), 750)),
        ("C15_v2", render_polygon_to_image(get_c15_polygon(0.0625), 750)),
        ("C15_v3", render_polygon_to_image(get_c15_polygon(0.125), 750)),
        # Row 2: A15
        ("A15_v1", render_polygon_to_image(get_a15_polygon(0.0), 750)),
        ("A15_v2", render_polygon_to_image(get_a15_polygon(0.125), 750)),
        ("A15_v3", render_polygon_to_image(get_a15_polygon(0.25), 750)),
    ]
    canvas = compose_deliverable_canvas(
        title_text="C15/A15",
        items=items,
        layout_rows=[3, 3],
        coaster_size_px=540,
        default_label_size=58,
        pad_above=14,
    )
    out_path = OUTPUT_DIR / "c15_a15_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_voroni_figure():
    print("Building 'Voroni' deliverable figure (6 coasters, center-to-center = 1.25x D)...")
    items = [
        # Row 1: Dense (small_v1, small_v2, small_v3)
        ("small_v1", render_polygon_to_image(get_voronoi_polygon(100, 42), 750)),
        ("small_v2", render_polygon_to_image(get_voronoi_polygon(100, 43), 750)),
        ("small_v3", render_polygon_to_image(get_voronoi_polygon(100, 44), 750)),
        # Row 2: Sparse (large_v1, large_v2, large_v3; dropping large_v4)
        ("large_v1", render_polygon_to_image(get_voronoi_polygon(50, 42), 750)),
        ("large_v2", render_polygon_to_image(get_voronoi_polygon(50, 43), 750)),
        ("large_v3", render_polygon_to_image(get_voronoi_polygon(50, 44), 750)),
    ]
    canvas = compose_deliverable_canvas(
        title_text="Voroni",
        items=items,
        layout_rows=[3, 3],
        coaster_size_px=540,
        default_label_size=58,
        pad_above=14,
    )
    out_path = OUTPUT_DIR / "voroni_previews.png"
    canvas.save(out_path, quality=95)
    # Also save as voronoi_previews.png for compatibility
    canvas.save(OUTPUT_DIR / "voronoi_previews.png", quality=95)
    print(f"  Saved: {out_path}")


def make_explicit_figure():
    print("Building 'Explicit' deliverable figure (9 coasters, 3x3 grid, center-to-center = 1.25x D)...")
    items = [
        # Row 1: Triangle Top 3
        ("Tri_Tetrahedral", render_polygon_to_image(get_tri_polygon("Tetrahedral"), 600)),
        ("Tri_Icosahedral", render_polygon_to_image(get_tri_polygon("Icosahedral"), 600)),
        ("Tri_Kelvin", render_polygon_to_image(get_tri_polygon("Kelvin"), 600)),
        # Row 2: Triangle Remaining 2 + Square 1
        ("Tri_Tesseract", render_polygon_to_image(get_tri_polygon("Tesseract"), 600)),
        ("Tri_Rhombic", render_polygon_to_image(get_tri_polygon("Rhombic"), 600)),
        ("Sq_Grid", render_polygon_to_image(get_sq_polygon("Grid"), 600)),
        # Row 3: Square Remaining 3
        ("Sq_Icosahedral", render_polygon_to_image(get_sq_polygon("Icosahedral"), 600)),
        ("Sq_Kelvin", render_polygon_to_image(get_sq_polygon("Kelvin"), 600)),
        ("Sq_Tesseract", render_polygon_to_image(get_sq_polygon("Tesseract"), 600)),
    ]
    canvas = compose_deliverable_canvas(
        title_text="Explicit",
        items=items,
        layout_rows=[3, 3, 3],
        coaster_size_px=360,
        default_label_size=46,
        pad_above=12,
    )
    out_path = OUTPUT_DIR / "explicit_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_tpms_figure():
    print("Building 'TPMS' deliverable figure (8 coasters, 3 rows, center-to-center = 1.25x D)...")
    items = [
        # Row 1: Z = 0
        ("Gyroid (Z=0)", get_tpms_image("Gyroid", 0.0, 360)),
        ("Diamond (Z=0)", get_tpms_image("Diamond", 0.0, 360)),
        ("Lidinoid (Z=0)", get_tpms_image("Lidinoid", 0.0, 360)),
        # Row 2: Neovius Z=0 and Z=2.4 centered
        ("Neovius (Z=0)", get_tpms_image("Neovius", 0.0, 360)),
        ("Neovius (Z=2.4)", get_tpms_image("Neovius", 2.4, 360)),
        # Row 3: Z = 2.4
        ("Gyroid (Z=2.4)", get_tpms_image("Gyroid", 2.4, 360)),
        ("Diamond (Z=2.4)", get_tpms_image("Diamond", 2.4, 360)),
        ("Lidinoid (Z=2.4)", get_tpms_image("Lidinoid", 2.4, 360)),
    ]
    canvas = compose_deliverable_canvas(
        title_text="TPMS",
        items=items,
        layout_rows=[3, 2, 3],
        coaster_size_px=360,
        default_label_size=46,
        pad_above=12,
    )
    out_path = OUTPUT_DIR / "tpms_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def clean_obsolete_figures():
    """Remove older uncombined preview files from outputs/Coasters."""
    obsolete = [
        "a15_previews.png",
        "c15_previews.png",
        "voronoi_dense_previews.png",
        "voronoi_sparse_previews.png",
        "explicit_square_previews.png",
        "explicit_tri_previews.png",
    ]
    for filename in obsolete:
        p = OUTPUT_DIR / filename
        if p.exists():
            p.unlink()
            print(f"  Cleaned obsolete figure: {filename}")


def main():
    print("Generating all 4 combined deliverable figures...")
    make_c15_a15_figure()
    make_voroni_figure()
    make_explicit_figure()
    make_tpms_figure()
    clean_obsolete_figures()
    print("All 4 combined figures generated successfully!")


if __name__ == "__main__":
    main()
