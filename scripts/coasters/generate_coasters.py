import os
import sys
import gc
import shutil
from pathlib import Path

# Add project root to path for importing graphite
ROOT_DIR = Path("c:/Users/ehunt/OneDrive/Documents/Python Scripts/Graphite").resolve()
sys.path.append(str(ROOT_DIR))

import numpy as np
import trimesh
from skimage.measure import marching_cubes
from shapely.geometry import Polygon, LineString, Point, box
from shapely.ops import unary_union
import manifold3d
from graphite.explicit.geometry_module import _trimesh_to_manifold, _manifold_to_trimesh, generate_geometry

# 2D Shape SDFs (using float32)
def circle_2d_sdf(x, y, R):
    return (np.sqrt(x**2 + y**2) - R).astype(np.float32)

def rect_2d_sdf(x, y, dx, dy):
    return np.maximum(np.abs(x) - dx, np.abs(y) - dy).astype(np.float32)

def hexagon_2d_sdf(x, y, r_in):
    d1 = np.abs(x) - r_in
    d2 = 0.5 * np.abs(x) + (np.sqrt(3)/2.0) * np.abs(y) - r_in
    return np.maximum(d1, d2).astype(np.float32)

def get_2d_sdf(shape_name, X, Y, is_inner=False):
    dim = 46.825 if is_inner else 50.0
    if shape_name == "Circle":
        return circle_2d_sdf(X, Y, dim)
    elif shape_name == "Hexagon":
        return hexagon_2d_sdf(X, Y, dim)
    elif shape_name == "Rectangle":
        return rect_2d_sdf(X, Y, dim, dim)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

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

# Phase 6: A15 and C15 Crystal Structure Tiling & Bonding
def tile_basis(basis_pts, nx, ny, nz, cell_size):
    tiled_pts = []
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            for k in range(-nz, nz + 1):
                offset = np.array([i, j, k], dtype=np.float64)
                for pt in basis_pts:
                    # Center the tiled unit cell grid symmetrically at the origin
                    t_pt = pt + offset - np.array([0.5, 0.5, 0.5])
                    tiled_pts.append(t_pt * cell_size)
    tiled_pts = np.vstack(tiled_pts)
    tiled_pts = np.unique(np.round(tiled_pts, 8), axis=0)
    return tiled_pts

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
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=cutoff)
    edges = []
    for u, v in pairs:
        dist = np.linalg.norm(pts[u] - pts[v])
        if dist > 1e-5:
            edges.append((u, v))
    return pts, np.array(edges, dtype=np.int64)

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
    a_basis = []
    for ab in a_base:
        for trans in fcc_translations:
            a_basis.append((ab + trans) % 1.0)
    a_basis = np.unique(np.round(a_basis, 8), axis=0)
    
    b_base = np.array([
        [0.625, 0.625, 0.625],
        [0.625, 0.875, 0.875],
        [0.875, 0.625, 0.875],
        [0.875, 0.875, 0.625]
    ], dtype=np.float64)
    b_basis = []
    for bb in b_base:
        for trans in fcc_translations:
            b_basis.append((bb + trans) % 1.0)
    b_basis = np.unique(np.round(b_basis, 8), axis=0)
    
    basis = np.vstack((a_basis, b_basis))
    pts = tile_basis(basis, nx, ny, nz, cell_size)
    cutoff = 0.45 * cell_size
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=cutoff)
    edges = []
    for u, v in pairs:
        dist = np.linalg.norm(pts[u] - pts[v])
        if dist > 1e-5:
            edges.append((u, v))
    return pts, np.array(edges, dtype=np.int64)

# Custom memory-safe chunked STL exporter
def export_stl_memory_safe(mesh, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    count = len(mesh.faces)
    header = b'\x00' * 80
    count_bytes = np.array([count], dtype='<u4').tobytes()
    
    dtype = np.dtype([
        ('normals', '<f4', (3,)),
        ('vertices', '<f4', (3, 3)),
        ('attributes', '<u2')
    ])
    
    chunk_size = 100000
    vertices = mesh.vertices
    faces = mesh.faces
    
    with open(file_path, 'wb') as f:
        f.write(header)
        f.write(count_bytes)
        for i in range(0, count, chunk_size):
            chunk_faces = faces[i : i + chunk_size]
            v0 = vertices[chunk_faces[:, 0]].astype(np.float32)
            v1 = vertices[chunk_faces[:, 1]].astype(np.float32)
            v2 = vertices[chunk_faces[:, 2]].astype(np.float32)
            
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            norm[norm < 1e-6] = 1.0
            chunk_norm = cross / norm
            
            chunk_tri = np.stack([v0, v1, v2], axis=1)
            
            packed = np.empty(len(chunk_tri), dtype=dtype)
            packed['normals'] = chunk_norm
            packed['vertices'] = chunk_tri
            packed['attributes'] = 0
            
            f.write(packed.tobytes())
            
            del v0, v1, v2, e1, e2, cross, norm, chunk_norm, chunk_tri, packed
            gc.collect()

def get_hexagon_polygon(r_in):
    r_out = r_in / np.cos(np.radians(30))
    pts = [
        (0.0, r_out),
        (r_in, r_out * 0.5),
        (r_in, -r_out * 0.5),
        (0.0, -r_out),
        (-r_in, -r_out * 0.5),
        (-r_in, r_out * 0.5)
    ]
    return Polygon(pts)

def run_generation():
    output_root = r"C:\Users\ehunt\OneDrive\Documents\Python Scripts\Graphite\outputs\Coasters"
    
    # Clean up old Z-named folders from the previous 3D run to keep outputs clean
    print("Cleaning up old Z-offset folders for crystal structures...")
    for z_folder in ["A15_z0.0", "A15_z0.125", "A15_z0.25", "C15_z0.0", "C15_z0.0625", "C15_z0.125"]:
        p = Path(output_root) / "Struts" / z_folder
        if p.exists():
            print(f"  Removing: {p}")
            shutil.rmtree(p)
            
    shapes = ["Circle", "Hexagon", "Rectangle"]
    modes = ["Framed", "Unframed"]
    
    # -------------------------------------------------------------
    # 3. 2D Crystal Lattices: Generate A15 and C15 (Phase 6 - Updated to 2D Extruded)
    # -------------------------------------------------------------
    crystal_lattices = [
        # (LatticeType, cell_size, Z_Offset_Fraction, folder_name, label)
        ("A15", 25.0, 0.0, "A15", "v1"),
        ("A15", 25.0, 0.125, "A15", "v2"),
        ("A15", 25.0, 0.25, "A15", "v3"),
        ("C15", 50.0, 0.0, "C15", "v1"),
        ("C15", 50.0, 0.0625, "C15", "v2"),
        ("C15", 50.0, 0.125, "C15", "v3")
    ]
    strut_width = 1.0 # 1.0mm strut width universally
    
    for lat_type, crystal_cell_size, z_offset_frac, folder_name, label in crystal_lattices:
        print(f"\nProcessing 2D Crystal lattice {lat_type} ({folder_name}/{label}) at Z_offset={z_offset_frac} via 2D Polygon Extrusion...")
        
        if lat_type == "A15":
            pts, edges = generate_a15_lattice(nx=3, ny=3, nz=1, cell_size=crystal_cell_size)
            z_limit = 2.5 # 2.5mm half-width for 25mm cell
        else: # C15 (tiled at nx=2 to optimize 50.0mm cell rendering)
            pts, edges = generate_c15_lattice(nx=2, ny=2, nz=1, cell_size=crystal_cell_size)
            z_limit = 5.0 # 5.0mm half-width for 50mm cell to keep node connections from hanging
            
        z_offset = z_offset_frac * crystal_cell_size
        pts_shifted = pts.copy()
        pts_shifted[:, 2] -= z_offset
        
        # Filter segments in Z range [-z_limit, z_limit] and project to 2D
        segments_2d = []
        for u, v in edges:
            p0 = pts_shifted[u]
            p1 = pts_shifted[v]
            z_min = min(p0[2], p1[2])
            z_max = max(p0[2], p1[2])
            if z_min <= z_limit and z_max >= -z_limit:
                p0_2d = p0[:2]
                p1_2d = p1[:2]
                if np.linalg.norm(p0_2d - p1_2d) > 1e-4:
                    segments_2d.append((p0_2d, p1_2d))
                    
        segments_2d = unique_segments(segments_2d)
        print(f"  Projected {len(segments_2d)} 2D segments from 3D lattice...")
        
        # Construct 2D polygons for the projected struts
        strut_polys = []
        for p1, p2 in segments_2d:
            line = LineString([p1, p2])
            strut_polys.append(line.buffer(strut_width / 2.0, cap_style=2))
            
        lattice_poly = unary_union(strut_polys)
        del strut_polys, segments_2d
        gc.collect()
        
        for mode in modes:
            is_framed = (mode == "Framed")
            
            for shape in shapes:
                print(f"  Generating {mode} {shape} {folder_name}/{label} coaster (polygon-based + extruded)...")
                
                if shape == "Circle":
                    outer_shape = Point(0, 0).buffer(50.0, resolution=128)
                    inner_shape = Point(0, 0).buffer(46.825, resolution=128)
                elif shape == "Rectangle":
                    outer_shape = box(-50.0, -50.0, 50.0, 50.0)
                    inner_shape = box(-46.825, -46.825, 46.825, 46.825)
                elif shape == "Hexagon":
                    outer_shape = get_hexagon_polygon(50.0)
                    inner_shape = get_hexagon_polygon(46.825)
                
                if is_framed:
                    frame_poly = outer_shape.difference(inner_shape)
                    clipped_lattice = lattice_poly.intersection(inner_shape)
                    final_poly = clipped_lattice.union(frame_poly)
                else:
                    final_poly = lattice_poly.intersection(outer_shape)
                    
                path = trimesh.load_path(final_poly)
                extruded = path.extrude(5.0)
                
                if isinstance(extruded, (list, tuple)):
                    mesh = trimesh.util.concatenate(extruded)
                elif hasattr(extruded, "geometry") and isinstance(extruded.geometry, dict):
                    mesh = trimesh.util.concatenate(list(extruded.geometry.values()))
                else:
                    mesh = extruded
                    
                mesh.apply_translation([0.0, 0.0, -2.5])
                
                folder_path = os.path.join(output_root, "Struts", folder_name, shape)
                file_name = f"{label}_{mode}_{shape}.stl"
                file_path = os.path.join(folder_path, file_name)
                
                export_stl_memory_safe(mesh, file_path)
                print(f"    Saved: {file_path}")
                
                del mesh, path
                gc.collect()

if __name__ == "__main__":
    run_generation()
