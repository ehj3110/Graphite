"""
Graphite Explicit Engine - Proven Topologies

This module generates node coordinates (seeds) for classic, mathematically 
proven lattice unit cells across a regular Cartesian bounding box.

Included topologies:
- Simple Cubic
- BCC (Body-Centered Cubic)
- FCC (Face-Centered Cubic)
- Truncated Octahedron (Kelvin cell)
- A15 (Frank-Kasper phase)
"""
import numpy as np

def _tile_fractional_points(
    base_points: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """Tile unit-cell fractional points across a regular grid."""
    tiled = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=np.float64)
                tiled.append((base_points + origin) * float(cell_size))
    if not tiled:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.vstack(tiled)
    return np.unique(np.round(pts, 10), axis=0).astype(np.float64)

def generate_simple_cubic_seeds(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """
    Generate node coordinates for a simple cubic lattice.

    Parameters
    ----------
    nx : int
        Number of unit cells along the X axis.
    ny : int
        Number of unit cells along the Y axis.
    nz : int
        Number of unit cells along the Z axis.
    cell_size : float
        The physical dimension of a single cubic unit cell.

    Returns
    -------
    ndarray
        (N, 3) array of node coordinates.
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive.")
    gx, gy, gz = np.mgrid[0 : nx + 1, 0 : ny + 1, 0 : nz + 1]
    points = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())).astype(np.float64)
    return points * float(cell_size)

def generate_bcc_seeds(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    corners = generate_simple_cubic_seeds(nx, ny, nz, cell_size)
    gx, gy, gz = np.mgrid[0:nx, 0:ny, 0:nz]
    centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())).astype(np.float64)
    centers = centers * float(cell_size) + (float(cell_size) * 0.5)
    return np.vstack((corners, centers))

def generate_fcc_seeds(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    corners = generate_simple_cubic_seeds(nx, ny, nz, cell_size)
    faces_x = _tile_fractional_points(np.array([[0.0, 0.5, 0.5]], dtype=np.float64), nx+1, ny, nz, cell_size)
    faces_y = _tile_fractional_points(np.array([[0.5, 0.0, 0.5]], dtype=np.float64), nx, ny+1, nz, cell_size)
    faces_z = _tile_fractional_points(np.array([[0.5, 0.5, 0.0]], dtype=np.float64), nx, ny, nz+1, cell_size)
    combined = np.vstack((corners, faces_x, faces_y, faces_z))
    return np.unique(np.round(combined, 10), axis=0).astype(np.float64)

def generate_truncated_oct_tet_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    base_points = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.0], [0.5, 0.5, 1.0], # XY faces
            [0.5, 0.0, 0.5], [0.5, 1.0, 0.5], # XZ faces
            [0.0, 0.5, 0.5], [1.0, 0.5, 0.5], # YZ faces
        ],
        dtype=np.float64,
    )
    return _tile_fractional_points(base_points, nx, ny, nz, cell_size)

def generate_a15_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Generate node coordinates for an A15 Frank-Kasper phase lattice.

    Parameters
    ----------
    nx : int, optional
        Number of unit cells along the X axis, by default 1.
    ny : int, optional
        Number of unit cells along the Y axis, by default 1.
    nz : int, optional
        Number of unit cells along the Z axis, by default 1.
    cell_size : float, optional
        The physical dimension of a single unit cell, by default 10.0.

    Returns
    -------
    ndarray
        (N, 3) array of node coordinates.
    """
    bcc = generate_bcc_seeds(nx, ny, nz, cell_size)
    face_offsets = np.asarray(
        [
            [0.25, 0.50, 0.00], [0.75, 0.50, 0.00],
            [0.25, 0.50, 1.00], [0.75, 0.50, 1.00],
            [0.00, 0.25, 0.50], [0.00, 0.75, 0.50],
            [1.00, 0.25, 0.50], [1.00, 0.75, 0.50],
            [0.50, 0.00, 0.25], [0.50, 0.00, 0.75],
            [0.50, 1.00, 0.25], [0.50, 1.00, 0.75],
        ],
        dtype=np.float64,
    )
    face_points = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=np.float64) * float(cell_size)
                pts = origin + face_offsets * float(cell_size)
                face_points.append(pts)
    if face_points:
        faces = np.vstack(face_points)
        all_points = np.vstack((bcc, faces))
    else:
        all_points = bcc
    return np.unique(np.round(all_points, 10), axis=0).astype(np.float64)


def generate_background_grid(
    grid_type: str,
    bounds: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate integer-space background grid nodes and elements covering `bounds`.

    Parameters
    ----------
    grid_type : str
        'A15' (tetrahedral) or 'SC' (hexahedral).
    bounds : (2, 3) ndarray
        Min and max bounding box coordinates [[min_x, min_y, min_z], [max_x, max_y, max_z]].
    cell_size : float
        Unit cell dimension.

    Returns
    -------
    grid_nodes : (N, 3) ndarray
    elements : (M, 4) or (M, 8) ndarray
    """
    gtype = str(grid_type).strip().upper()
    min_bound, max_bound = np.asarray(bounds, dtype=np.float64)
    padded_min = min_bound - 1.5 * cell_size
    padded_max = max_bound + 1.5 * cell_size

    min_ix = int(np.floor(padded_min[0] / cell_size))
    max_ix = int(np.ceil(padded_max[0] / cell_size))
    min_iy = int(np.floor(padded_min[1] / cell_size))
    max_iy = int(np.ceil(padded_max[1] / cell_size))
    min_iz = int(np.floor(padded_min[2] / cell_size))
    max_iz = int(np.ceil(padded_max[2] / cell_size))

    if gtype == "A15":
        A15_BASIS = np.array([
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
            [0.25, 0.5, 0.0], [0.75, 0.5, 0.0],
            [0.0, 0.25, 0.5], [0.0, 0.75, 0.5],
            [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
        ], dtype=np.float64)
        BOND_CUTOFF = 0.62

        pts_list = []
        for ix_n in range(-1, 3):
            for iy_n in range(-1, 3):
                for iz_n in range(-1, 3):
                    for b in A15_BASIS:
                        pts_list.append(b + np.array([ix_n, iy_n, iz_n], dtype=np.float64))
        pts = np.unique(np.round(np.vstack(pts_list), 9), axis=0)

        dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        adj = {i: set() for i in range(len(pts))}
        for u, v in zip(*np.where((dist > 1e-9) & (dist <= BOND_CUTOFF))):
            if u < v:
                adj[u].add(v); adj[v].add(u)

        cliques = []
        for u in range(len(pts)):
            for v in adj[u]:
                if v <= u: continue
                uv = adj[u] & adj[v]
                for w in uv:
                    if w <= v: continue
                    for x in (uv & adj[w]):
                        if x > w:
                            cliques.append((u, v, w, x))

        arr = np.array(cliques, dtype=np.int64)
        centroids_frac = pts[arr].mean(axis=1)
        inside = np.all((centroids_frac >= -1e-9) & (centroids_frac <= 1.0 + 1e-9), axis=1)
        cliques_inside = arr[inside]

        node_coords_int = []
        tet_node_coord_to_idx = {}

        def get_or_add(coord_int):
            key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
            if key not in tet_node_coord_to_idx:
                tet_node_coord_to_idx[key] = len(node_coords_int)
                node_coords_int.append(coord_int)
            return tet_node_coord_to_idx[key]

        seen_tets = set()
        tets_out = []
        pts_int = np.round(pts * 4.0).astype(np.int64)

        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for iz in range(min_iz, max_iz + 1):
                    off_int = np.array([ix, iy, iz], dtype=np.int64) * 4
                    global_pts_int = off_int + pts_int
                    for cl in cliques_inside:
                        tet_int = global_pts_int[cl]
                        tk = tuple(sorted([tuple(p) for p in tet_int]))
                        if tk in seen_tets:
                            continue
                        seen_tets.add(tk)
                        t_indices = [get_or_add(p) for p in tet_int]
                        tets_out.append(t_indices)

        grid_nodes = np.array(node_coords_int, dtype=np.float64) * (cell_size / 4.0)
        elements = np.array(tets_out, dtype=np.int64)
        return grid_nodes, elements

    elif gtype == "SC":
        node_coords_int = []
        node_coord_to_idx = {}

        def get_or_add(coord_int):
            key = (int(coord_int[0]), int(coord_int[1]), int(coord_int[2]))
            if key not in node_coord_to_idx:
                node_coord_to_idx[key] = len(node_coords_int)
                node_coords_int.append(coord_int)
            return node_coord_to_idx[key]

        cells_out = []
        for ix in range(min_ix, max_ix + 1):
            for iy in range(min_iy, max_iy + 1):
                for iz in range(min_iz, max_iz + 1):
                    voxel_corners = np.array([
                        [ix, iy, iz],
                        [ix + 1, iy, iz],
                        [ix + 1, iy + 1, iz],
                        [ix, iy + 1, iz],
                        [ix, iy, iz + 1],
                        [ix + 1, iy, iz + 1],
                        [ix + 1, iy + 1, iz + 1],
                        [ix, iy + 1, iz + 1]
                    ], dtype=np.int64)
                    c_indices = [get_or_add(p) for p in voxel_corners]
                    cells_out.append(c_indices)

        grid_nodes = np.array(node_coords_int, dtype=np.float64) * cell_size
        elements = np.array(cells_out, dtype=np.int64)
        return grid_nodes, elements

    else:
        raise ValueError(f"Unsupported background grid_type '{grid_type}'. Options: 'A15', 'SC'")
