"""
Inside-Out Voxelization for Conformal Hexahedral Meshing.

Uses Centroid Expansion/Contraction: keep hexes where centroid is inside,
identify exposed boundary faces, and snap those nodes to the STL surface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from trimesh import proximity

# Standard 6 faces of an 8-node hex, 4 vertices each
_HEX_FACES: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # Bottom
    (4, 5, 6, 7),  # Top
    (0, 1, 5, 4),  # Front
    (3, 2, 6, 7),  # Back
    (0, 3, 7, 4),  # Left
    (1, 2, 6, 5),  # Right
)


def _face_key(pts: np.ndarray, face: tuple[int, int, int, int]) -> tuple:
    """Canonical key for face deduplication (sorted vertex coords)."""
    verts = [tuple(round(float(pts[i, j]), 6) for j in range(3)) for i in face]
    return tuple(sorted(verts))


def generate_conformal_hexes(
    stl_path: str | Path,
    target_size: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate conformal hexahedral elements via centroid-based voxelization.

    Keeps hexes where centroid is inside; identifies exposed faces (appear
    exactly once); returns hexes and geometric centers of exposed quads
    for explicit topological snapping downstream.

    Parameters
    ----------
    stl_path : str | Path
        Path to the STL file.
    target_size : float
        Voxel edge length (step) in mesh units.

    Returns
    -------
    kept_hexes : ndarray, shape (N, 8, 3)
        Hexahedra with exposed vertices snapped to the STL surface.
    exposed_face_centers : ndarray, shape (M, 3)
        Geometric centroids of exposed boundary quads (using snapped vertices).
    """
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    bounds = np.array(mesh.bounds)
    extents = np.array(mesh.extents)

    # Exact bounding-box subdivision (linspace)
    num_x = max(1, int(round(extents[0] / target_size)))
    num_y = max(1, int(round(extents[1] / target_size)))
    num_z = max(1, int(round(extents[2] / target_size)))

    x_vals = np.linspace(bounds[0][0], bounds[1][0], num_x + 1)
    y_vals = np.linspace(bounds[0][1], bounds[1][1], num_y + 1)
    z_vals = np.linspace(bounds[0][2], bounds[1][2], num_z + 1)

    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)
    total_grid_points = nx * ny * nz

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
    all_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def idx(i: int, j: int, k: int) -> int:
        return i * (ny * nz) + j * nz + k

    hex_corners = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ]

    # Build hexes and filter by centroid containment
    hex_list: list[np.ndarray] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = [idx(i + di, j + dj, k + dk) for di, dj, dk in hex_corners]
                pts = all_points[corners]
                centroid = pts.mean(axis=0)
                if mesh.contains(centroid.reshape(1, 3))[0]:
                    hex_list.append(pts.copy())

    if not hex_list:
        hexes = np.empty((0, 8, 3), dtype=np.float64)
        exposed_face_centers = np.empty((0, 3), dtype=np.float64)
        print()
        print("=" * 60)
        print("  CONFORMAL HEX VOXELIZER — DEBUG")
        print("=" * 60)
        print(f"  STL Name:              {stl_path.name}")
        print(f"  Total Hexes Kept:      0 (no centroids inside)")
        print("=" * 60)
        print()
        return hexes, exposed_face_centers

    hexes = np.array(hex_list, dtype=np.float64)
    n_hex = hexes.shape[0]

    # Count face occurrences
    face_counts: dict[str, int] = {}
    face_to_hex_face: dict[str, list[tuple[int, int]]] = {}

    for hi in range(n_hex):
        pts = hexes[hi]
        for fi, face in enumerate(_HEX_FACES):
            key = _face_key(pts, face)
            face_counts[key] = face_counts.get(key, 0) + 1
            if key not in face_to_hex_face:
                face_to_hex_face[key] = []
            face_to_hex_face[key].append((hi, fi))

    # Exposed faces: count == 1
    exposed_face_keys = [k for k, c in face_counts.items() if c == 1]

    # 1. Identify unique exposed nodes (vertices of boundary faces)
    node_key_to_locations: dict[str, list[tuple[int, int]]] = {}
    node_key_to_coords: dict[str, np.ndarray] = {}

    for key in exposed_face_keys:
        for hi, fi in face_to_hex_face[key]:
            face = _HEX_FACES[fi]
            for vi in face:
                pt = hexes[hi, vi]
                nk = f"{round(float(pt[0]), 6)},{round(float(pt[1]), 6)},{round(float(pt[2]), 6)}"
                if nk not in node_key_to_locations:
                    node_key_to_locations[nk] = []
                    node_key_to_coords[nk] = pt.copy()
                node_key_to_locations[nk].append((hi, vi))

    # 2. Snap exposed vertices to STL surface
    exposed_coords = np.array([node_key_to_coords[nk] for nk in node_key_to_locations], dtype=np.float64)
    n_exposed = len(exposed_coords)

    if n_exposed > 0:
        closest, _, _ = proximity.closest_point(mesh, exposed_coords)
        key_list = list(node_key_to_locations.keys())
        for idx, nk in enumerate(key_list):
            snapped = closest[idx]
            for hi, vi in node_key_to_locations[nk]:
                hexes[hi, vi] = snapped

    # 3. Compute exposed_face_centers using NEWLY SNAPPED vertex coordinates
    exposed_face_centers_list: list[np.ndarray] = []

    for key in exposed_face_keys:
        hi, fi = face_to_hex_face[key][0]
        face = _HEX_FACES[fi]
        face_pts = hexes[hi, face]  # Use snapped coordinates
        centroid = face_pts.mean(axis=0)
        exposed_face_centers_list.append(centroid)

    exposed_face_centers = (
        np.array(exposed_face_centers_list, dtype=np.float64)
        if exposed_face_centers_list
        else np.empty((0, 3), dtype=np.float64)
    )

    # Debug output
    print()
    print("=" * 60)
    print("  CONFORMAL HEX VOXELIZER — DEBUG (Centroid Rule)")
    print("=" * 60)
    print(f"  STL Name:              {stl_path.name}")
    print(f"  Bounding Box Size:     {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}")
    print(f"  Target Size:           {target_size}")
    print(f"  Total Grid Points:     {total_grid_points}")
    print(f"  Total Hexes Kept:      {n_hex}")
    print(f"  Exposed Boundary Faces: {len(exposed_face_keys)}")
    print(f"  Exposed Nodes Snapped:  {n_exposed}")
    print(f"  Exposed Face Centers:   {len(exposed_face_centers_list)}")
    print("=" * 60)
    print()

    return hexes, exposed_face_centers
