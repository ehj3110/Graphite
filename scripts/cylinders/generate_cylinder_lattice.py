# -*- coding: utf-8 -*-
"""
Generate high-resolution cylindrical explicit lattices and assemble them
into 3D printable Napkin Rings with watertight CAD rims.

Supported topologies:
1. Square Tesseract (M = 3 rows, N = 10 cols, strut_w = 2.0 mm)
2. Triangular Tesseract (M = 3 rows, N = 9 cols, strut_w = 2.0 mm)
3. 1.5-inch Voronoi Foam (3 distinct seedings: v1, v2, v3, strut_w = 2.0 mm)
4. 2.0-inch Scaled Voronoi Foam (D_in = 50.8 mm, strut_thickness = 5.33 mm, strut_w = 2.0 mm)
5. 2.0-inch A15 Crystal Lattice (2 unit cells along height, M = 2, N = 7, strut_w = 2.0 mm)
6. 2.0-inch C15 Crystal Lattice (2 unit cells along height, M = 2, N = 7, strut_w = 2.0 mm)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import box, LineString
import manifold3d as m3d

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.geometry_module import _trimesh_to_manifold, _manifold_to_trimesh

# Global CAD defaults for 1.5-inch ring
R_IN_1P5 = 19.05       # 38.1mm inner diameter
WALL_T_1P5 = 4.0       # 4.0mm wall thickness
R_OUT_1P5 = R_IN_1P5 + WALL_T_1P5 # 23.05mm
H_1P5 = 38.1
Y_START_1P5 = 6.65
CIRCULAR_SEGMENTS = 256
MAX_CHORD_STEP = 0.75

BASE_RING_PATH = WORKSPACE_ROOT / "test_parts" / "NapkingRing_BaseRing_V1.STL"
OUTPUT_DIR = WORKSPACE_ROOT / "outputs" / "cylinders"


def extract_base_ring_rims(base_path: Path, scale: float = 1.0) -> tuple[m3d.Manifold, np.ndarray, float, float, float]:
    """
    Extract top and bottom solid rims from BaseRing, scaled by `scale`.
    Returns (rims_manifold, center, r_in, r_out, height).
    """
    base_mesh = trimesh.load(str(base_path))
    if abs(scale - 1.0) > 1e-6:
        base_mesh.apply_scale(scale)
        
    m_base = _trimesh_to_manifold(base_mesh)
    
    r_in = R_IN_1P5 * scale
    wall_t = WALL_T_1P5 * scale
    r_out = r_in + wall_t
    h = H_1P5 * scale
    y_start = Y_START_1P5 * scale
    y_center = y_start + h / 2.0
    center = np.array([25.4 * scale, y_center, 25.4 * scale], dtype=np.float64)
    
    box_cut = trimesh.creation.box(extents=[150.0 * scale, h, 150.0 * scale])
    box_cut.apply_translation(center)
    m_box = _trimesh_to_manifold(box_cut)
    
    rims = m_base - m_box
    return rims, center, r_in, r_out, h


def create_sleeve_trim(center: np.ndarray, height: float, r_in: float, r_out: float, segments: int = CIRCULAR_SEGMENTS) -> m3d.Manifold:
    """Create a cylindrical sleeve manifold centered at `center` along the Y-axis."""
    cyl_out = m3d.Manifold.cylinder(height=height, radius_low=r_out, radius_high=r_out, circular_segments=segments, center=True)
    cyl_in = m3d.Manifold.cylinder(height=height + 2.0, radius_low=r_in, radius_high=r_in, circular_segments=segments, center=True)
    sleeve = cyl_out - cyl_in
    R_mat = [
        [1.0, 0.0, 0.0, center[0]],
        [0.0, 0.0, -1.0, center[1]],
        [0.0, 1.0, 0.0, center[2]]
    ]
    return sleeve.transform(R_mat)


def dedupe_segments(segs: list[tuple[np.ndarray, np.ndarray]], c_circ: float, tol: float = 1e-4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deduplicate undirected 2D segments modulo circumference."""
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    inv = 1.0 / tol
    for p1, p2 in segs:
        u1, y1 = float(p1[0]) % c_circ, float(p1[1])
        u2, y2 = float(p2[0]) % c_circ, float(p2[1])
        k1 = (int(round(u1 * inv)), int(round(y1 * inv)))
        k2 = (int(round(u2 * inv)), int(round(y2 * inv)))
        if k1 == k2:
            continue
        key = (k1, k2) if k1 < k2 else (k2, k1)
        if key not in seen:
            seen.add(key)
            unique.append((np.array([u1, y1]), np.array([u2, y2])))
    return unique


def build_prisms_from_2d_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    center: np.ndarray,
    r_in: float,
    r_out: float,
    c_circ: float,
    strut_w: float = 2.0,
    max_step: float = MAX_CHORD_STEP,
) -> list[m3d.Manifold]:
    """Convert 2D (u, y) segments into finely subdivided, surface-oriented 3D prisms."""
    r_mid = (r_in + r_out) / 2.0
    radial_thick = (r_out - r_in) + 0.6
    cubes: list[m3d.Manifold] = []
    
    for p1, p2 in segments:
        L2d = float(np.linalg.norm(p2 - p1))
        if L2d < 1e-4:
            continue
            
        du = p2[0] - p1[0]
        if abs(du) > c_circ / 2.0:
            p2_u = p2[0] - c_circ if du > 0 else p2[0] + c_circ
        else:
            p2_u = p2[0]
            
        n_steps = max(2, int(np.ceil(L2d / max_step)))
        t_vals = np.linspace(0.0, 1.0, n_steps + 1)
        pts: list[np.ndarray] = []
        for t in t_vals:
            u = (1.0 - t) * p1[0] + t * p2_u
            y = (1.0 - t) * p1[1] + t * p2[1]
            th = u / r_mid
            x = center[0] + r_mid * np.cos(th)
            z = center[2] + r_mid * np.sin(th)
            pts.append(np.array([x, y, z]))
            
        for k in range(len(pts) - 1):
            pa = pts[k]
            pb = pts[k + 1]
            pmid = 0.5 * (pa + pb)
            t_vec = pb - pa
            L = float(np.linalg.norm(t_vec))
            if L < 1e-6:
                continue
            t_dir = t_vec / L
            
            n_vec = np.array([pmid[0] - center[0], 0.0, pmid[2] - center[2]])
            n_len = float(np.linalg.norm(n_vec))
            if n_len < 1e-6:
                continue
            n_dir = n_vec / n_len
            
            b_vec = np.cross(t_dir, n_dir)
            b_len = float(np.linalg.norm(b_vec))
            if b_len < 1e-6:
                continue
            b_dir = b_vec / b_len
            n_dir = np.cross(b_dir, t_dir)
            
            cube = m3d.Manifold.cube([strut_w, radial_thick, L + 0.15], center=True)
            mat = [
                [b_dir[0], n_dir[0], t_dir[0], pmid[0]],
                [b_dir[1], n_dir[1], t_dir[1], pmid[1]],
                [b_dir[2], n_dir[2], t_dir[2], pmid[2]]
            ]
            cubes.append(cube.transform(mat))
            
    return cubes


def generate_sq_tesseract_lattice(
    center: np.ndarray,
    r_in: float = R_IN_1P5,
    r_out: float = R_OUT_1P5,
    height: float = H_1P5,
    y_base: float = Y_START_1P5,
    strut_w: float = 2.0,
    n_circumferential: int = 10,
    m_vertical: int = 3,
) -> m3d.Manifold:
    """Generate 3D cylindrical Sq_Tesseract lattice with M = 3 rows."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    s_u = c_mid / n_circumferential
    s_y = height / m_vertical
    
    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(m_vertical):
        y0 = y_base + j * s_y
        y1 = y_base + (j + 1) * s_y
        for i in range(n_circumferential):
            u0 = i * s_u
            u1 = (i + 1) * s_u
            v0 = np.array([u0, y0])
            v1 = np.array([u1, y0])
            v2 = np.array([u1, y1])
            v3 = np.array([u0, y1])
            
            centroid = 0.25 * (v0 + v1 + v2 + v3)
            w0 = centroid + 0.5 * (v0 - centroid)
            w1 = centroid + 0.5 * (v1 - centroid)
            w2 = centroid + 0.5 * (v2 - centroid)
            w3 = centroid + 0.5 * (v3 - centroid)
            
            raw_segments.extend([
                (v0, v1), (v1, v2), (v2, v3), (v3, v0),
                (w0, w1), (w1, w2), (w2, w3), (w3, w0),
                (v0, w0), (v1, w1), (v2, w2), (v3, w3)
            ])
            
    segments = dedupe_segments(raw_segments, c_mid)
    cubes = build_prisms_from_2d_segments(segments, center, r_in, r_out, c_mid, strut_w=strut_w)
    composed = m3d.Manifold.batch_boolean(cubes, m3d.OpType.Add)
    sleeve = create_sleeve_trim(center, height, r_in, r_out)
    return composed ^ sleeve


def generate_tri_tesseract_lattice(
    center: np.ndarray,
    r_in: float = R_IN_1P5,
    r_out: float = R_OUT_1P5,
    height: float = H_1P5,
    y_base: float = Y_START_1P5,
    strut_w: float = 2.0,
    n_circumferential: int = 9,
    m_vertical: int = 3,
) -> m3d.Manifold:
    """Generate 3D cylindrical Tri_Tesseract lattice with M = 3 rows."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    s_u = c_mid / n_circumferential
    h = height / m_vertical
    
    def get_p(i: int, j: int) -> np.ndarray:
        cx = (i + (j % 2) * 0.5) * s_u
        cy = y_base + j * h
        return np.array([cx, cy], dtype=np.float64)
        
    triangles: list[np.ndarray] = []
    for j in range(m_vertical):
        for i in range(n_circumferential):
            p_ij = get_p(i, j)
            p_ip1_j = get_p(i + 1, j)
            p_ijp1 = get_p(i, j + 1)
            p_ip1_jp1 = get_p(i + 1, j + 1)
            if j % 2 == 0:
                triangles.append(np.array([p_ij, p_ip1_j, p_ijp1]))
                triangles.append(np.array([p_ip1_j, p_ip1_jp1, p_ijp1]))
            else:
                triangles.append(np.array([p_ij, p_ip1_j, p_ip1_jp1]))
                triangles.append(np.array([p_ij, p_ijp1, p_ip1_jp1]))
                
    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for tri in triangles:
        v0, v1, v2 = tri[0], tri[1], tri[2]
        c = (v0 + v1 + v2) / 3.0
        w0 = c + 0.5 * (v0 - c)
        w1 = c + 0.5 * (v1 - c)
        w2 = c + 0.5 * (v2 - c)
        raw_segments.extend([
            (v0, v1), (v1, v2), (v2, v0),
            (w0, w1), (w1, w2), (w2, w0),
            (v0, w0), (v1, w1), (v2, w2)
        ])
        
    segments = dedupe_segments(raw_segments, c_mid)
    cubes = build_prisms_from_2d_segments(segments, center, r_in, r_out, c_mid, strut_w=strut_w)
    composed = m3d.Manifold.batch_boolean(cubes, m3d.OpType.Add)
    sleeve = create_sleeve_trim(center, height, r_in, r_out)
    return composed ^ sleeve


def generate_voronoi_lattice(
    center: np.ndarray,
    r_in: float,
    r_out: float,
    height: float,
    y_base: float,
    strut_w: float = 2.0,
    cols: int = 14,
    rows: int = 4,
    seed_val: int = 123,
) -> m3d.Manifold:
    """Generate 3D cylindrical Voronoi lattice with periodic boundary conditions."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    du = c_mid / cols
    dy = height / rows
    
    np.random.seed(seed_val)
    seeds: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            u = (c + 0.15 + 0.7 * np.random.rand()) * du
            y = y_base + (r + 0.15 + 0.7 * np.random.rand()) * dy
            seeds.append([u, y])
    seeds_arr = np.array(seeds)
    
    seeds_all = np.vstack([
        seeds_arr - np.array([c_mid, 0.0]),
        seeds_arr,
        seeds_arr + np.array([c_mid, 0.0])
    ])
    
    vor = Voronoi(seeds_all)
    domain = box(0.0, y_base, c_mid, y_base + height)
    
    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for p1_idx, p2_idx in vor.ridge_vertices:
        if p1_idx < 0 or p2_idx < 0:
            continue
        v1 = vor.vertices[p1_idx]
        v2 = vor.vertices[p2_idx]
        line = LineString([v1, v2])
        clipped = line.intersection(domain)
        if not clipped.is_empty:
            if clipped.geom_type == 'LineString':
                raw_segments.append((np.array(clipped.coords[0]), np.array(clipped.coords[1])))
            elif clipped.geom_type == 'MultiLineString':
                for l in clipped.geoms:
                    raw_segments.append((np.array(l.coords[0]), np.array(l.coords[1])))
                    
    segments = dedupe_segments(raw_segments, c_mid)
    cubes = build_prisms_from_2d_segments(segments, center, r_in, r_out, c_mid, strut_w=strut_w)
    
    boundary_rings: list[m3d.Manifold] = []
    for y in [y_base, y_base + height]:
        cyl_out = m3d.Manifold.cylinder(height=strut_w, radius_low=r_out + 0.1, radius_high=r_out + 0.1, circular_segments=CIRCULAR_SEGMENTS, center=True)
        cyl_in = m3d.Manifold.cylinder(height=strut_w + 0.2, radius_low=r_in - 0.1, radius_high=r_in - 0.1, circular_segments=CIRCULAR_SEGMENTS, center=True)
        ring = (cyl_out - cyl_in).transform([
            [1.0, 0.0, 0.0, center[0]],
            [0.0, 0.0, -1.0, y],
            [0.0, 1.0, 0.0, center[2]]
        ])
        boundary_rings.append(ring)
        
    all_parts = cubes + boundary_rings
    composed = m3d.Manifold.batch_boolean(all_parts, m3d.OpType.Add)
    sleeve = create_sleeve_trim(center, height, r_in, r_out)
    return composed ^ sleeve


def generate_crystal_lattice_cyl(
    basis: np.ndarray,
    cutoff: float,
    center: np.ndarray,
    r_in: float,
    r_out: float,
    height: float,
    y_base: float,
    m_vertical: float,
    n_circumferential: int,
    strut_w: float = 2.0,
    z_limit_frac: float = 0.1,
) -> m3d.Manifold:
    """Generate 3D cylindrical crystal lattice with seamless periodic circumferential wrap."""
    r_mid = (r_in + r_out) / 2.0
    c_mid = 2.0 * np.pi * r_mid
    L_x = c_mid / n_circumferential
    L_y = height / m_vertical
    num_rows = int(np.ceil(m_vertical))
    
    cell_pts = []
    for c in range(-1, n_circumferential + 2):
        for r in range(-1, num_rows + 2):
            for k in range(-1, 2):
                for b in basis:
                    cell_pts.append([c + b[0], r + b[1], k + b[2]])
    cell_pts = np.unique(np.round(cell_pts, 6), axis=0)
    
    tree = cKDTree(cell_pts)
    pairs = tree.query_pairs(r=cutoff)
    
    segs = []
    for u, v in pairs:
        p0 = cell_pts[u]
        p1 = cell_pts[v]
        if min(p0[2], p1[2]) <= z_limit_frac and max(p0[2], p1[2]) >= -z_limit_frac:
            p0_mm = np.array([p0[0] * L_x, y_base + p0[1] * L_y])
            p1_mm = np.array([p1[0] * L_x, y_base + p1[1] * L_y])
            if np.linalg.norm(p0_mm - p1_mm) > 1e-4:
                segs.append((p0_mm, p1_mm))
                
    y_band = box(-1e6, y_base, 1e6, y_base + height)
    clipped = []
    for p0, p1 in segs:
        cy = LineString([p0, p1]).intersection(y_band)
        if not cy.is_empty and cy.geom_type == 'LineString':
            coords = list(cy.coords)
            pt0, pt1 = np.array(coords[0]), np.array(coords[1])
            mid_u = 0.5 * (pt0[0] + pt1[0])
            if 0.0 <= (mid_u % c_mid) < c_mid:
                k_shift = np.floor(mid_u / c_mid)
                pt0_s = pt0 - np.array([k_shift * c_mid, 0])
                pt1_s = pt1 - np.array([k_shift * c_mid, 0])
                clipped.append((pt0_s, pt1_s))
                
    # Deduplicate
    inv = 1000.0
    seen = set()
    uniq = []
    for p1, p2 in clipped:
        u1, y1 = float(p1[0]) % c_mid, float(p1[1])
        u2, y2 = float(p2[0]) % c_mid, float(p2[1])
        k1 = (int(round(u1 * inv)), int(round(y1 * inv)))
        k2 = (int(round(u2 * inv)), int(round(y2 * inv)))
        if k1 == k2:
            continue
        k = (k1, k2) if k1 < k2 else (k2, k1)
        if k not in seen:
            seen.add(k)
            uniq.append((p1, p2))
            
    cubes = build_prisms_from_2d_segments(uniq, center, r_in, r_out, c_mid, strut_w=strut_w)
    
    rings = []
    for y in [y_base, y_base + height]:
        co = m3d.Manifold.cylinder(height=strut_w, radius_low=r_out + 0.1, radius_high=r_out + 0.1, circular_segments=CIRCULAR_SEGMENTS, center=True)
        ci = m3d.Manifold.cylinder(height=strut_w + 0.2, radius_low=r_in - 0.1, radius_high=r_in - 0.1, circular_segments=CIRCULAR_SEGMENTS, center=True)
        rings.append((co - ci).transform([[1.0, 0.0, 0.0, center[0]], [0.0, 0.0, -1.0, y], [0.0, 1.0, 0.0, center[2]]]))
        
    composed = m3d.Manifold.batch_boolean(cubes + rings, m3d.OpType.Add)
    cyl_out = m3d.Manifold.cylinder(height=height, radius_low=r_out, radius_high=r_out, circular_segments=CIRCULAR_SEGMENTS, center=True)
    cyl_in = m3d.Manifold.cylinder(height=height + 2.0, radius_low=r_in, radius_high=r_in, circular_segments=CIRCULAR_SEGMENTS, center=True)
    sleeve = (cyl_out - cyl_in).transform([[1.0, 0.0, 0.0, center[0]], [0.0, 0.0, -1.0, y_base + height / 2.0], [0.0, 1.0, 0.0, center[2]]])
    return composed ^ sleeve


def get_a15_basis() -> np.ndarray:
    return np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        [0.25, 0.0, 0.5], [0.75, 0.0, 0.5],
        [0.5, 0.25, 0.0], [0.5, 0.75, 0.0],
        [0.0, 0.5, 0.25], [0.0, 0.5, 0.75]
    ], dtype=np.float64)


def get_c15_basis() -> np.ndarray:
    fcc_translations = np.array([
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
    ], dtype=np.float64)
    a_base = np.array([[0, 0, 0], [0.25, 0.25, 0.25]], dtype=np.float64)
    a_basis = np.unique(np.round([(ab + trans) % 1.0 for ab in a_base for trans in fcc_translations], 8), axis=0)
    b_base = np.array([[0.625, 0.625, 0.625], [0.625, 0.875, 0.875], [0.875, 0.625, 0.875], [0.875, 0.875, 0.625]], dtype=np.float64)
    b_basis = np.unique(np.round([(bb + trans) % 1.0 for bb in b_base for trans in fcc_translations], 8), axis=0)
    return np.vstack((a_basis, b_basis))


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    basis_a15 = get_a15_basis()
    basis_c15 = get_c15_basis()
    
    # 1. Setup 1.5-inch ring (scale = 1.0)
    rims_1p5, center_1p5, r_in_1p5, r_out_1p5, h_1p5 = extract_base_ring_rims(BASE_RING_PATH, scale=1.0)
    y_start_1p5 = Y_START_1P5
    center_sa_1p5 = np.array([r_out_1p5, h_1p5 / 2.0, r_out_1p5], dtype=np.float64)
    
    # 2. Setup 2.0-inch ring (scale = 4/3)
    scale_2in = 4.0 / 3.0
    rims_2in, center_2in, r_in_2in, r_out_2in, h_2in = extract_base_ring_rims(BASE_RING_PATH, scale=scale_2in)
    y_start_2in = Y_START_1P5 * scale_2in
    center_sa_2in = np.array([r_out_2in, h_2in / 2.0, r_out_2in], dtype=np.float64)
    
    # Plan:
    # 1. 1.5-inch A15 (1.0 unit cell tall, M=1.0, N=4)
    # 2. 1.5-inch C15 (1.0 unit cell tall, M=1.0, N=4)
    # 3. 2.0-inch A15 (1.5 unit cells tall, M=1.5, N=5)
    # 4. 2.0-inch C15 (1.5 unit cells tall, M=1.5, N=5)
    tasks = [
        ("A15", basis_a15, 0.62, "1p5inch", rims_1p5, center_1p5, center_sa_1p5, r_in_1p5, r_out_1p5, h_1p5, y_start_1p5, 1.0, 4),
        ("C15", basis_c15, 0.45, "1p5inch", rims_1p5, center_1p5, center_sa_1p5, r_in_1p5, r_out_1p5, h_1p5, y_start_1p5, 1.0, 4),
        ("A15", basis_a15, 0.62, "2inch", rims_2in, center_2in, center_sa_2in, r_in_2in, r_out_2in, h_2in, y_start_2in, 1.5, 5),
        ("C15", basis_c15, 0.45, "2inch", rims_2in, center_2in, center_sa_2in, r_in_2in, r_out_2in, h_2in, y_start_2in, 1.5, 5),
    ]
    
    for name, basis, cutoff, size_label, rims, center, center_sa, r_in, r_out, h, y_start, M, N in tasks:
        t_task = time.time()
        print(f"\n{'='*65}\nGenerating {size_label} {name} (M={M} cells tall, N={N} cols, strut_w=2.0mm)...\n{'='*65}")
        
        # 1. Full napkin ring
        m_lat = generate_crystal_lattice_cyl(
            basis, cutoff, center, r_in, r_out, h, y_start, m_vertical=M, n_circumferential=N, strut_w=2.0
        )
        full_ring = rims + m_lat
        mesh_ring = _manifold_to_trimesh(full_ring)
        out_ring_path = OUTPUT_DIR / f"NapkinRing_{size_label}_{name}.stl"
        mesh_ring.export(str(out_ring_path))
        print(f"  Exported: {out_ring_path.name} ({len(mesh_ring.faces):,} faces, watertight={mesh_ring.is_watertight})")
        
        # 2. Standalone lattice sleeve
        m_sa = generate_crystal_lattice_cyl(
            basis, cutoff, center_sa, r_in, r_out, h, 0.0, m_vertical=M, n_circumferential=N, strut_w=2.0
        )
        mesh_sa = _manifold_to_trimesh(m_sa)
        out_sa_path = OUTPUT_DIR / f"LatticeSection_{size_label}_{name}.stl"
        mesh_sa.export(str(out_sa_path))
        print(f"  Exported: {out_sa_path.name} ({len(mesh_sa.faces):,} faces, watertight={mesh_sa.is_watertight})")
        print(f"  Completed {size_label} {name} in {time.time() - t_task:.2f} s")
        
    print(f"\nAll 4 Napkin Rings and 4 Lattice Sections generated in {time.time() - t0:.2f} s!")


if __name__ == "__main__":
    main()
