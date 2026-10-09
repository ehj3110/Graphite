# -*- coding: utf-8 -*-
"""
Generate combined deliverable presentation figures for the 4 coaster categories.

Figures:
1. "C15/A15"   - 6 coasters (Row 1: C15_v1, C15_v2, C15_v3; Row 2: A15_v1, A15_v2, A15_v3)
2. "Voroni"    - 6 coasters (Row 1: small_v1, small_v2, small_v3; Row 2: large_v1, large_v2, large_v3; large_v4 dropped)
3. "Explicit"  - 9 coasters (3x3 grid combining all Triangle and Square lattices)
4. "TPMS"      - 8 coasters (Row 1: Z=0 Gyroid, Diamond, Lidinoid; Row 2: Neovius dual slices; Row 3: Z=2.4)

Rules & Specifications:
- Canvas aspect ratio: 1024:765 (~1.3386), rendered at 2048 x 1530 px high resolution.
- Titles reduced by 25% (Main header: 51 pt; Coaster headers: 44 pt / 34 pt).
- Center-to-center spacing: pitch = 1.2 * diameter (dx = dy = 1.2 * D).
- 2-row layouts (C15/A15 and Voroni): Coasters sized to have exactly a 10% margin on the sides (D = 482 px, pitch = 578.4 px).
- 3-row layouts (Explicit and TPMS): Spacing = 1.2 * D, vertically centered.
- Faithful cross-sections matching the true 3D models with 1.0mm strut width and 3.175mm circular frame.
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
import shapely.plotting as spl
from scipy.spatial import cKDTree, Voronoi

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphite.math.tpms import evaluate_tpms_phase

OUTPUT_DIR = REPO_ROOT / "outputs" / "Coasters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANVAS_W = 2048
CANVAS_H = 1530
BG_COLOR = (248, 249, 250, 255)
SOLID_RGBA = (20, 20, 22, 255)

# Fonts (Reduced by 25%: 68 -> 51)
try:
    MAIN_TITLE_FONT = ImageFont.truetype("segoeuib.ttf", 51)
except Exception:
    try:
        MAIN_TITLE_FONT = ImageFont.truetype("arialbd.ttf", 51)
    except Exception:
        MAIN_TITLE_FONT = ImageFont.load_default()


def get_coaster_title_font(text, max_w, default_size=42):
    """
    Get bold font reduced by 25%, fitting within max_w.
    """
    size = default_size
    while size >= 18:
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
        return ImageFont.truetype("arialbd.ttf", 18)
    except Exception:
        return ImageFont.load_default()


# =====================================================================
# 1. Geometry Generators (Faithful to True 3D STL Models)
# =====================================================================

def unique_segments(segments, tol=1e-3):
    unique = []
    for seg in segments:
        p1, p2 = seg
        if p1[0] < p2[0] or (abs(p1[0] - p2[0]) < tol and p1[1] < p2[1]):
            s_seg = (p1, p2)
        else:
            s_seg = (p2, p1)
        duplicate = False
        for u_seg in unique:
            u_p1, u_p2 = u_seg
            if (np.linalg.norm(s_seg[0] - u_p1) < tol and np.linalg.norm(s_seg[1] - u_p2) < tol):
                duplicate = True
                break
        if not duplicate:
            unique.append(s_seg)
    return unique


def segments_to_coaster_polygon(segments, strut_width=1.0, r_outer=50.0, r_inner=46.825):
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


def tile_basis(basis_pts, nx, ny, nz, cell_size):
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


# --- C15 Lattice (Frank-Kasper Laves Phase) ---
def generate_c15_lattice(nx=2, ny=2, nz=1, cell_size=50.0):
    fcc_translations = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], dtype=np.float64)
    
    a_base = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25]
    ], dtype=np.float64)
    a_basis = [((ab + trans) % 1.0) for ab in a_base for trans in fcc_translations]
    a_basis = np.unique(np.round(a_basis, 8), axis=0)
    
    b_base = np.array([
        [0.625, 0.625, 0.625],
        [0.625, 0.875, 0.875],
        [0.875, 0.625, 0.875],
        [0.875, 0.875, 0.625]
    ], dtype=np.float64)
    b_basis = [((bb + trans) % 1.0) for bb in b_base for trans in fcc_translations]
    b_basis = np.unique(np.round(b_basis, 8), axis=0)
    
    basis = np.vstack((a_basis, b_basis))
    pts = tile_basis(basis, nx, ny, nz, cell_size)
    cutoff = 0.45 * cell_size
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=cutoff)
    edges = [(u, v) for u, v in pairs if np.linalg.norm(pts[u] - pts[v]) > 1e-5]
    return pts, edges


def get_c15_polygon(z_offset_mm):
    cell_size = 50.0
    pts, edges = generate_c15_lattice(nx=2, ny=2, nz=1, cell_size=cell_size)
    z_limit = 5.0
    pts_shifted = pts.copy()
    pts_shifted[:, 2] -= z_offset_mm
    
    segments = []
    for u, v in edges:
        p0 = pts_shifted[u]
        p1 = pts_shifted[v]
        z_min = min(p0[2], p1[2])
        z_max = max(p0[2], p1[2])
        if z_min <= z_limit and z_max >= -z_limit:
            p0_2d, p1_2d = p0[:2], p1[:2]
            if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                segments.append((p0_2d, p1_2d))
    segments = unique_segments(segments)
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- A15 Lattice ---
def generate_a15_lattice(nx=3, ny=3, nz=1, cell_size=25.0):
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.0, 0.5],
        [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0],
        [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25],
        [0.0, 0.5, 0.75]
    ], dtype=np.float64)
    pts = tile_basis(basis, nx, ny, nz, cell_size)
    cutoff = 0.62 * cell_size
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=cutoff)
    edges = [(u, v) for u, v in pairs if np.linalg.norm(pts[u] - pts[v]) > 1e-5]
    return pts, edges


def get_a15_polygon(z_offset_mm):
    cell_size = 25.0
    pts, edges = generate_a15_lattice(nx=3, ny=3, nz=1, cell_size=cell_size)
    z_limit = 2.5
    pts_shifted = pts.copy()
    pts_shifted[:, 2] -= z_offset_mm
    
    segments = []
    for u, v in edges:
        p0 = pts_shifted[u]
        p1 = pts_shifted[v]
        z_min = min(p0[2], p1[2])
        z_max = max(p0[2], p1[2])
        if z_min <= z_limit and z_max >= -z_limit:
            p0_2d, p1_2d = p0[:2], p1[:2]
            if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                segments.append((p0_2d, p1_2d))
    segments = unique_segments(segments)
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- Voronoi ---
def get_voronoi_polygon(num_points_in_coaster, seed):
    extended_size = 160.0
    area_ratio = (extended_size / 100.0) ** 2
    total_points = int(round(num_points_in_coaster * area_ratio))
    
    np.random.seed(seed)
    pts = np.random.uniform(-extended_size / 2.0, extended_size / 2.0, size=(total_points, 2))
    
    vor = Voronoi(pts)
    segments = []
    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            p1 = vor.vertices[ridge[0]]
            p2 = vor.vertices[ridge[1]]
            segments.append((p1, p2))
            
    segments = unique_segments(segments)
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- Triangle Lattices ---
def get_triangle_segments(topo, s, h):
    triangles = []
    def get_p(i, j):
        cx = i * s + (j % 2) * (s / 2)
        cy = j * h
        return np.array([cx, cy])
        
    for j in range(-7, 7):
        for i in range(-7, 7):
            p_ij = get_p(i, j)
            p_ip1_j = get_p(i+1, j)
            p_ijp1 = get_p(i, j+1)
            p_ip1_jp1 = get_p(i+1, j+1)
            if j % 2 == 0:
                triangles.append(np.array([p_ij, p_ip1_j, p_ijp1]))
                triangles.append(np.array([p_ip1_j, p_ip1_jp1, p_ijp1]))
            else:
                triangles.append(np.array([p_ij, p_ip1_j, p_ip1_jp1]))
                triangles.append(np.array([p_ij, p_ip1_jp1, p_ijp1]))
                
    segments = []
    if topo == "Tetrahedral":
        for tri in triangles:
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
    elif topo == "Icosahedral":
        for tri in triangles:
            m0 = (tri[0] + tri[1]) / 2.0
            m1 = (tri[1] + tri[2]) / 2.0
            m2 = (tri[2] + tri[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m0))
    elif topo == "Kelvin":
        for tri in triangles:
            p01_a = (2*tri[0] + tri[1]) / 3.0
            p01_b = (tri[0] + 2*tri[1]) / 3.0
            p12_a = (2*tri[1] + tri[2]) / 3.0
            p12_b = (tri[1] + 2*tri[2]) / 3.0
            p20_a = (2*tri[2] + tri[0]) / 3.0
            p20_b = (tri[2] + 2*tri[0]) / 3.0
            segments.append((p01_a, p20_b))
            segments.append((p20_b, p20_a))
            segments.append((p20_a, p12_b))
            segments.append((p12_b, p12_a))
            segments.append((p12_a, p01_b))
            segments.append((p01_b, p01_a))
    elif topo == "Tesseract":
        for tri in triangles:
            centroid = np.mean(tri, axis=0)
            inscribed = centroid + 0.5 * (tri - centroid)
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[0]))
            for i in range(3):
                segments.append((tri[i], inscribed[i]))
    elif topo == "Rhombic":
        centroids = [np.mean(tri, axis=0) for tri in triangles]
        n_tri = len(triangles)
        for i in range(n_tri):
            c_i = centroids[i]
            if np.linalg.norm(c_i) > 60:
                continue
            tri_i = triangles[i]
            for j in range(i+1, n_tri):
                c_j = centroids[j]
                if np.linalg.norm(c_j) > 60:
                    continue
                tri_j = triangles[j]
                shared = sum(1 for vi in tri_i for vj in tri_j if np.linalg.norm(vi - vj) < 1e-4)
                if shared == 2:
                    segments.append((c_i, c_j))
    return unique_segments(segments)


def get_tri_polygon(topo):
    r_tri = 12.7 if topo == "Tesseract" else 6.35
    s_tri = r_tri * np.sqrt(3.0)
    h_tri = s_tri * np.sqrt(3.0) / 2.0
    segments = get_triangle_segments(topo, s_tri, h_tri)
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- Square Lattices ---
def get_square_segments(topo, s):
    squares = []
    for row in range(-6, 7):
        for col in range(-6, 7):
            cx = col * s
            cy = row * s
            v0 = [cx - s/2, cy - s/2]
            v1 = [cx + s/2, cy - s/2]
            v2 = [cx + s/2, cy + s/2]
            v3 = [cx - s/2, cy + s/2]
            squares.append(np.array([v0, v1, v2, v3]))
            
    segments = []
    if topo == "Grid":
        for sq in squares:
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
    elif topo == "Icosahedral":
        for sq in squares:
            m0 = (sq[0] + sq[1]) / 2.0
            m1 = (sq[1] + sq[2]) / 2.0
            m2 = (sq[2] + sq[3]) / 2.0
            m3 = (sq[3] + sq[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m3))
            segments.append((m3, m0))
    elif topo == "Kelvin":
        for sq in squares:
            p01_a = (2*sq[0] + sq[1]) / 3.0
            p01_b = (sq[0] + 2*sq[1]) / 3.0
            p12_a = (2*sq[1] + sq[2]) / 3.0
            p12_b = (sq[1] + 2*sq[2]) / 3.0
            p23_a = (2*sq[2] + sq[3]) / 3.0
            p23_b = (sq[2] + 2*sq[3]) / 3.0
            p30_a = (2*sq[3] + sq[0]) / 3.0
            p30_b = (sq[3] + 2*sq[0]) / 3.0
            segments.append((p01_a, p30_b))
            segments.append((p30_b, p30_a))
            segments.append((p30_a, p23_b))
            segments.append((p23_b, p23_a))
            segments.append((p23_a, p12_b))
            segments.append((p12_b, p12_a))
            segments.append((p12_a, p01_b))
            segments.append((p01_b, p01_a))
    elif topo == "Tesseract":
        for sq in squares:
            centroid = np.mean(sq, axis=0)
            inscribed = centroid + 0.5 * (sq - centroid)
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[3]))
            segments.append((inscribed[3], inscribed[0]))
            for i in range(4):
                segments.append((sq[i], inscribed[i]))
    return unique_segments(segments)


def get_sq_polygon(topo):
    s_sq = 25.4 if topo == "Tesseract" else 12.7
    segments = get_square_segments(topo, s_sq)
    return segments_to_coaster_polygon(segments, strut_width=1.0)


# --- TPMS Geometry ---
def sample_tpms_threshold(lattice_type, target_sf=0.33):
    n = 64
    axis = np.linspace(0, 2 * np.pi, n, endpoint=False)
    U, V, W = np.meshgrid(axis, axis, axis, indexing="ij")
    vals = np.abs(evaluate_tpms_phase(lattice_type, U, V, W)).ravel()
    vals.sort()
    idx = int(target_sf * len(vals))
    return vals[idx]


def get_tpms_image(lattice_type, z_mm, target_size_px=360):
    res = 512
    dim = 50.0
    x = np.linspace(-dim, dim, res)
    y = np.linspace(-dim, dim, res)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    cell_size = 25.0
    omega = 2.0 * np.pi / cell_size
    U = X * omega
    V = Y * omega
    W = np.full_like(X, z_mm * omega)
    
    tau = sample_tpms_threshold(lattice_type, target_sf=0.33)
    F = evaluate_tpms_phase(lattice_type, U, V, W)
    
    solid_lattice = (np.abs(F) <= tau) & (R <= 46.825)
    solid_frame = (R <= 50.0) & (R >= 46.825)
    solid = solid_lattice | solid_frame
    
    img_data = np.zeros((res, res, 4), dtype=np.uint8)
    img_data[solid] = SOLID_RGBA
    return Image.fromarray(img_data, mode="RGBA").resize((target_size_px, target_size_px), Image.Resampling.LANCZOS)


# =====================================================================
# 2. Rendering & Drop Shadow
# =====================================================================

def render_polygon_to_image(poly, render_dim=750):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=render_dim / 6.0)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-50.5, 50.5)
    ax.set_ylim(-50.5, 50.5)
    
    spl.plot_polygon(poly, ax=ax, add_points=False, color="#141416", alpha=1.0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    fig.canvas.draw()
    rgba_buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba_buf.copy(), mode="RGBA")
    plt.close(fig)
    return img


def add_drop_shadow(coaster_img, blur_radius=16, offset_y=12, shadow_opacity=0.28):
    w, h = coaster_img.size
    pad = blur_radius * 2 + offset_y
    large_w, large_h = w + pad * 2, h + pad * 2
    
    shadow_img = Image.new("RGBA", (large_w, large_h), (0, 0, 0, 0))
    alpha = coaster_img.split()[3]
    tinted_alpha = alpha.point(lambda p: int(p * shadow_opacity))
    
    tinted = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    tinted.putalpha(tinted_alpha)
    
    shadow_img.paste(tinted, (pad, pad + offset_y))
    blurred = shadow_img.filter(ImageFilter.GaussianBlur(blur_radius))
    
    composite = Image.new("RGBA", (large_w, large_h), (0, 0, 0, 0))
    composite.paste(blurred, (0, 0), blurred)
    composite.paste(coaster_img, (pad, pad), coaster_img)
    return composite, pad


# =====================================================================
# 3. Canvas Composition & Placement
# =====================================================================

def compose_deliverable_canvas(
    title_text,
    items,  # list of (label, PIL.Image)
    layout_rows,  # list of counts per row
    coaster_size_px,
    default_label_size=42,
    pad_above=12,
):
    """
    Assemble the deliverable canvas.
    - Pitch = 1.2 * D (center-to-center spacing equal to 1.2 x diameter).
    - Coaster titles placed directly above each coaster.
    - Title size reduced by 25%.
    """
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # 1. Main Header (Reduced by 25%)
    title_bbox = draw.textbbox((0, 0), title_text, font=MAIN_TITLE_FONT)
    title_x = int(round((CANVAS_W - (title_bbox[2] + title_bbox[0])) / 2.0))
    title_y = 26
    draw.text((title_x, title_y), title_text, font=MAIN_TITLE_FONT, fill=(15, 23, 42, 255))
    
    title_bottom = title_y + title_bbox[3]
    
    # 2. Grid Geometry Math (Center-to-Center = 1.2 * D)
    D = coaster_size_px
    pitch = 1.2 * D
    num_rows = len(layout_rows)
    
    content_top = title_bottom + 20
    content_bottom = CANVAS_H - 24
    available_h = content_bottom - content_top
    
    # Sample title text height
    sample_font = get_coaster_title_font("Sample (Z=0)", max_w=pitch * 0.9, default_size=default_label_size)
    sample_bbox = draw.textbbox((0, 0), "Sample (Z=0)", font=sample_font)
    title_text_h = sample_bbox[3] - sample_bbox[1]
    
    # Total grid vertical span
    H_total = (num_rows - 1) * pitch + D + pad_above + title_text_h
    
    # Vertically center the entire grid
    y_grid_top = content_top + max(0.0, (available_h - H_total) / 2.0)
    y0 = y_grid_top + title_text_h + pad_above + D / 2.0
    
    # Drop shadow
    blur_rad = int(round(16 * (D / 500.0)))
    offset_y = int(round(12 * (D / 500.0)))
    
    shadowed_items = []
    for label, img in items:
        scaled = img.resize((D, D), Image.Resampling.LANCZOS)
        shadowed, pad = add_drop_shadow(scaled, blur_radius=blur_rad, offset_y=offset_y, shadow_opacity=0.28)
        shadowed_items.append((label, shadowed, pad))
        
    item_idx = 0
    for r_idx, count_in_row in enumerate(layout_rows):
        row_cy = y0 + r_idx * pitch
        # Center this row horizontally: span = (count - 1) * pitch
        row_x0 = (CANVAS_W / 2.0) - ((count_in_row - 1) * pitch / 2.0)
        
        for c_idx in range(count_in_row):
            label, shadowed_img, pad = shadowed_items[item_idx]
            item_idx += 1
            
            coaster_cx = row_x0 + c_idx * pitch
            coaster_cy = row_cy
            coaster_top_y = coaster_cy - D / 2.0
            
            # Paste coaster with shadow
            paste_x = int(round(coaster_cx - pad - D / 2.0))
            paste_y = int(round(coaster_cy - pad - D / 2.0))
            canvas.paste(shadowed_img, (paste_x, paste_y), shadowed_img)
            
            # Coaster title directly above coaster
            lbl_font = get_coaster_title_font(label, max_w=pitch * 0.88, default_size=default_label_size)
            lbl_bbox = draw.textbbox((0, 0), label, font=lbl_font)
            
            lbl_x = int(round(coaster_cx - (lbl_bbox[2] + lbl_bbox[0]) / 2.0))
            lbl_y = int(round(coaster_top_y - pad_above - lbl_bbox[3]))
            
            draw.text((lbl_x, lbl_y), label, font=lbl_font, fill=(30, 41, 59, 255))
            
    return canvas.convert("RGB")


# =====================================================================
# 4. Deliverable Figure Builders
# =====================================================================

def make_c15_a15_figure():
    print("Building 'C15/A15' deliverable figure (D=482px, 10% side margins, pitch=1.2xD)...")
    items = [
        # Row 1: C15 offsets 0.0, L/16 (3.125), L/8 (6.25)
        ("C15_v1", render_polygon_to_image(get_c15_polygon(0.0), 750)),
        ("C15_v2", render_polygon_to_image(get_c15_polygon(3.125), 750)),
        ("C15_v3", render_polygon_to_image(get_c15_polygon(6.25), 750)),
        # Row 2: A15 offsets 0.0, L/8 (3.125), L/4 (6.25)
        ("A15_v1", render_polygon_to_image(get_a15_polygon(0.0), 750)),
        ("A15_v2", render_polygon_to_image(get_a15_polygon(3.125), 750)),
        ("A15_v3", render_polygon_to_image(get_a15_polygon(6.25), 750)),
    ]
    # D = 482 px gives exactly 10% side margins (span = 2 * 1.2 * 482 + 482 = 1638.8 px = 80% of 2048)
    canvas = compose_deliverable_canvas(
        title_text="C15/A15",
        items=items,
        layout_rows=[3, 3],
        coaster_size_px=482,
        default_label_size=44,
        pad_above=12,
    )
    out_path = OUTPUT_DIR / "c15_a15_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_voroni_figure():
    print("Building 'Voroni' deliverable figure (D=482px, 10% side margins, pitch=1.2xD)...")
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
        coaster_size_px=482,
        default_label_size=44,
        pad_above=12,
    )
    out_path = OUTPUT_DIR / "voroni_previews.png"
    canvas.save(out_path, quality=95)
    canvas.save(OUTPUT_DIR / "voronoi_previews.png", quality=95)
    print(f"  Saved: {out_path}")


def make_explicit_figure():
    print("Building 'Explicit' deliverable figure (9 coasters, 3x3 grid, D=390px, pitch=1.2xD)...")
    items = [
        # Row 1: Triangle Top 3
        ("Tri_Tetrahedral", render_polygon_to_image(get_tri_polygon("Tetrahedral"), 650)),
        ("Tri_Icosahedral", render_polygon_to_image(get_tri_polygon("Icosahedral"), 650)),
        ("Tri_Kelvin", render_polygon_to_image(get_tri_polygon("Kelvin"), 650)),
        # Row 2: Triangle Remaining 2 + Square 1
        ("Tri_Tesseract", render_polygon_to_image(get_tri_polygon("Tesseract"), 650)),
        ("Tri_Rhombic", render_polygon_to_image(get_tri_polygon("Rhombic"), 650)),
        ("Sq_Grid", render_polygon_to_image(get_sq_polygon("Grid"), 650)),
        # Row 3: Square Remaining 3
        ("Sq_Icosahedral", render_polygon_to_image(get_sq_polygon("Icosahedral"), 650)),
        ("Sq_Kelvin", render_polygon_to_image(get_sq_polygon("Kelvin"), 650)),
        ("Sq_Tesseract", render_polygon_to_image(get_sq_polygon("Tesseract"), 650)),
    ]
    canvas = compose_deliverable_canvas(
        title_text="Explicit",
        items=items,
        layout_rows=[3, 3, 3],
        coaster_size_px=390,
        default_label_size=34,
        pad_above=10,
    )
    out_path = OUTPUT_DIR / "explicit_previews.png"
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")


def make_tpms_figure():
    print("Building 'TPMS' deliverable figure (8 coasters, 3 rows, D=360px, pitch=1.2xD)...")
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
        default_label_size=34,
        pad_above=10,
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
    print("Generating all 4 combined deliverable figures with exact cross sections and 1.2x pitch...")
    make_c15_a15_figure()
    make_voroni_figure()
    make_explicit_figure()
    make_tpms_figure()
    clean_obsolete_figures()
    print("All 4 combined figures generated successfully!")


if __name__ == "__main__":
    main()
