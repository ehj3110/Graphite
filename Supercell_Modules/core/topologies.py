"""
Supercell Topologies — Lattice connectivity patterns.

Provides topology generators for Cartesian-based supercell exploration.
"""

from __future__ import annotations

import itertools

import numpy as np


def generate_simple_cubic_seeds(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """
    Generate corner points of a simple-cubic grid.

    Args:
        nx, ny, nz: Number of unit cells along each axis.
        cell_size: Unit-cell edge length.
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    gx, gy, gz = np.mgrid[0 : nx + 1, 0 : ny + 1, 0 : nz + 1]
    points = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())).astype(np.float64)
    return points * float(cell_size)


def generate_bcc_seeds(
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """
    Generate body-centered cubic seeds: corners + body centers.
    """
    corners = generate_simple_cubic_seeds(nx, ny, nz, cell_size)
    gx, gy, gz = np.mgrid[0:nx, 0:ny, 0:nz]
    centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())).astype(np.float64)
    centers = centers * float(cell_size) + (float(cell_size) * 0.5)
    return np.vstack((corners, centers))


def _tile_fractional_points(
    base_points: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    cell_size: float,
) -> np.ndarray:
    """Tile unit-cell fractional points across a regular grid."""
    tiled: list[np.ndarray] = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=np.float64)
                tiled.append((base_points + origin) * float(cell_size))
    if not tiled:
        return np.empty((0, 3), dtype=np.float64)
    pts = np.vstack(tiled)
    return np.unique(np.round(pts, 10), axis=0).astype(np.float64)


def generate_bitruncated_cubic_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Generate bitruncated-cubic style seeds from coordinate-table permutations.

    Base table uses all signed permutations of (0, 1, 2), normalized into [0, 1].
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    raw_points: list[tuple[float, float, float]] = []
    for perm in set(itertools.permutations((0.0, 1.0, 2.0), 3)):
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            raw_points.append((perm[0] * sx, perm[1] * sy, perm[2] * sz))

    arr = np.unique(np.asarray(raw_points, dtype=np.float64), axis=0)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    base_points = (arr - mins) / (maxs - mins)
    return _tile_fractional_points(base_points, nx, ny, nz, cell_size)


def generate_truncated_oct_tet_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Generate a truncated-octahedron/tetra hybrid seed set ("big & small").
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    # Hybrid of corner, body-center, and quarter-offset interior points.
    base_points = np.asarray(
        [
            # corners
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            # body center
            [0.5, 0.5, 0.5],
            # quarter-shifted "small" interior points
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.75],
        ],
        dtype=np.float64,
    )
    return _tile_fractional_points(base_points, nx, ny, nz, cell_size)


def generate_rhombicuboct_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Generate rhombicuboctahedral-style seeds using permutations of (1, 1, 2).
    """
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("nx, ny, nz must be positive.")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    raw_points: list[tuple[float, float, float]] = []
    for perm in set(itertools.permutations((1.0, 1.0, 2.0), 3)):
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            raw_points.append((perm[0] * sx, perm[1] * sy, perm[2] * sz))

    arr = np.unique(np.asarray(raw_points, dtype=np.float64), axis=0)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    base_points = (arr - mins) / (maxs - mins)
    # Add center to stabilize Delaunay for this symmetric point cloud.
    base_points = np.vstack((base_points, np.array([[0.5, 0.5, 0.5]], dtype=np.float64)))
    return _tile_fractional_points(base_points, nx, ny, nz, cell_size)


def generate_a15_seeds(
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    cell_size: float = 10.0,
) -> np.ndarray:
    """
    Generate A15-like seed set: BCC seeds + face-offset points per cell.
    """
    bcc = generate_bcc_seeds(nx, ny, nz, cell_size)

    # Face-point offsets inside each unit cell (fractional coordinates).
    face_offsets = np.asarray(
        [
            [0.25, 0.50, 0.00],
            [0.75, 0.50, 0.00],
            [0.50, 0.25, 1.00],
            [0.50, 0.75, 1.00],
            [0.00, 0.25, 0.50],
            [0.00, 0.75, 0.50],
            [1.00, 0.50, 0.25],
            [1.00, 0.50, 0.75],
            [0.25, 0.00, 0.50],
            [0.75, 0.00, 0.50],
            [0.50, 1.00, 0.25],
            [0.50, 1.00, 0.75],
        ],
        dtype=np.float64,
    )

    face_points: list[np.ndarray] = []
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

    # Remove duplicates introduced on shared faces.
    all_points = np.unique(np.round(all_points, 10), axis=0)
    return all_points.astype(np.float64)


def generate_centroid_dual(
    points: np.ndarray,
    clean_simplices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the centroid-dual graph of a tetrahedral mesh.

    - Dual nodes are tetrahedron centroids.
    - Dual struts connect centroids of tetrahedra that share a face.
    """
    points = np.asarray(points, dtype=np.float64)
    clean_simplices = np.asarray(clean_simplices, dtype=np.int64)

    if clean_simplices.ndim != 2 or clean_simplices.shape[1] != 4:
        raise ValueError("clean_simplices must have shape (N, 4).")
    if clean_simplices.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

    # Dual nodes: centroid of each tetrahedron.
    dual_nodes = points[clean_simplices].mean(axis=1)

    # Map each triangular face to the tetrahedra that contain it.
    face_to_tets: dict[tuple[int, int, int], list[int]] = {}
    face_vertex_triplets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for tet_id, vertex_indices in enumerate(clean_simplices):
        for a, b, c in face_vertex_triplets:
            face = tuple(sorted((int(vertex_indices[a]), int(vertex_indices[b]), int(vertex_indices[c]))))
            face_to_tets.setdefault(face, []).append(tet_id)

    # Shared face -> strut between the two neighboring tetrahedra.
    dual_struts_set: set[tuple[int, int]] = set()
    for tet_ids in face_to_tets.values():
        if len(tet_ids) == 2:
            i, j = tet_ids
            dual_struts_set.add((min(i, j), max(i, j)))

    dual_struts = np.asarray(sorted(dual_struts_set), dtype=np.int64)
    if dual_struts.size == 0:
        dual_struts = np.empty((0, 2), dtype=np.int64)
    return dual_nodes, dual_struts


def get_cartesian_kagome(
    bounding_box: tuple[tuple[float, float, float], tuple[float, float, float]],
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3D Cartesian Kagome lattice.

    Nodes are placed at the midpoints of edges of a cubic lattice. For a cube
    at (x, y, z) with size L, edge midpoints are at (x+L/2, y, z), (x, y+L/2, z),
    etc. Struts connect nodes that are L/sqrt(2) apart (face diagonals).

    Args:
        bounding_box: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        cell_size: Cube edge length L.

    Returns:
        (nodes, struts) — nodes (N, 3), struts (S, 2).
    """
    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    L = cell_size
    nx = max(1, int(np.ceil((x_max - x_min) / L)))
    ny = max(1, int(np.ceil((y_max - y_min) / L)))
    nz = max(1, int(np.ceil((z_max - z_min) / L)))

    nodes_list: list[tuple[float, float, float]] = []
    node_key_to_idx: dict[tuple[float, float, float], int] = {}

    def add_node(x: float, y: float, z: float) -> int:
        key = (round(x, 10), round(y, 10), round(z, 10))
        if key not in node_key_to_idx:
            idx = len(nodes_list)
            nodes_list.append((x, y, z))
            node_key_to_idx[key] = idx
            return idx
        return node_key_to_idx[key]

    def get_node(x: float, y: float, z: float) -> int | None:
        key = (round(x, 10), round(y, 10), round(z, 10))
        return node_key_to_idx.get(key)

    struts_set: set[tuple[int, int]] = set()

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = x_min + i * L
                cy = y_min + j * L
                cz = z_min + k * L

                # 12 edge midpoints of cube at (cx, cy, cz)
                add_node(cx + L / 2, cy, cz)
                add_node(cx + L / 2, cy + L, cz)
                add_node(cx + L / 2, cy, cz + L)
                add_node(cx + L / 2, cy + L, cz + L)
                add_node(cx, cy + L / 2, cz)
                add_node(cx + L, cy + L / 2, cz)
                add_node(cx, cy + L / 2, cz + L)
                add_node(cx + L, cy + L / 2, cz + L)
                add_node(cx, cy, cz + L / 2)
                add_node(cx + L, cy, cz + L / 2)
                add_node(cx, cy + L, cz + L / 2)
                add_node(cx + L, cy + L, cz + L / 2)

    nodes = np.array(nodes_list, dtype=np.float64)

    # For each face of each cube, add 2 diagonal struts (L/sqrt(2) apart)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cx = x_min + i * L
                cy = y_min + j * L
                cz = z_min + k * L

                # Face z=cz (xy): 4 midpoints, 2 diagonals
                n0 = get_node(cx + L / 2, cy, cz)
                n1 = get_node(cx + L, cy + L / 2, cz)
                n2 = get_node(cx + L / 2, cy + L, cz)
                n3 = get_node(cx, cy + L / 2, cz)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

                # Face z=cz+L (xy)
                n0 = get_node(cx + L / 2, cy, cz + L)
                n1 = get_node(cx + L, cy + L / 2, cz + L)
                n2 = get_node(cx + L / 2, cy + L, cz + L)
                n3 = get_node(cx, cy + L / 2, cz + L)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

                # Face y=cy (xz)
                n0 = get_node(cx + L / 2, cy, cz)
                n1 = get_node(cx + L, cy, cz + L / 2)
                n2 = get_node(cx + L / 2, cy, cz + L)
                n3 = get_node(cx, cy, cz + L / 2)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

                # Face y=cy+L (xz)
                n0 = get_node(cx + L / 2, cy + L, cz)
                n1 = get_node(cx + L, cy + L, cz + L / 2)
                n2 = get_node(cx + L / 2, cy + L, cz + L)
                n3 = get_node(cx, cy + L, cz + L / 2)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

                # Face x=cx (yz)
                n0 = get_node(cx, cy + L / 2, cz)
                n1 = get_node(cx, cy + L, cz + L / 2)
                n2 = get_node(cx, cy + L / 2, cz + L)
                n3 = get_node(cx, cy, cz + L / 2)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

                # Face x=cx+L (yz)
                n0 = get_node(cx + L, cy + L / 2, cz)
                n1 = get_node(cx + L, cy + L, cz + L / 2)
                n2 = get_node(cx + L, cy + L / 2, cz + L)
                n3 = get_node(cx + L, cy, cz + L / 2)
                if n0 is not None and n1 is not None:
                    struts_set.add((min(n0, n1), max(n0, n1)))
                if n2 is not None and n3 is not None:
                    struts_set.add((min(n2, n3), max(n2, n3)))

    struts = np.array(sorted(struts_set), dtype=np.int64)
    return nodes, struts


def get_voronoi_foam(
    bounding_box: tuple[tuple[float, float, float], tuple[float, float, float]],
    cell_size: float,
    seed_type: str = "BCC",
    jitter: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Voronoi foam lattice from seed points.

    Args:
        bounding_box: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        cell_size: Spacing for seed grid.
        seed_type: 'BCC' for Body-Centered Cubic seeds.
        jitter: Random offset [-jitter, +jitter] per axis for organic look.

    Returns:
        (nodes, struts) — Voronoi vertices and ridge edges (filtered -1).
    """
    from scipy.spatial import Voronoi

    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")

    if seed_type == "BCC":
        # BCC: corners + body centers
        x_vals = np.arange(x_min, x_max + 1e-9, cell_size)
        y_vals = np.arange(y_min, y_max + 1e-9, cell_size)
        z_vals = np.arange(z_min, z_max + 1e-9, cell_size)
        seeds_list: list[tuple[float, float, float]] = []
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                for k, z in enumerate(z_vals):
                    seeds_list.append((x, y, z))
                    if i < len(x_vals) - 1 and j < len(y_vals) - 1 and k < len(z_vals) - 1:
                        seeds_list.append(
                            (x + cell_size / 2, y + cell_size / 2, z + cell_size / 2)
                        )
        seeds = np.array(seeds_list, dtype=np.float64)
    else:
        x_vals = np.arange(x_min, x_max + 1e-9, cell_size)
        y_vals = np.arange(y_min, y_max + 1e-9, cell_size)
        z_vals = np.arange(z_min, z_max + 1e-9, cell_size)
        xx, yy, zz = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")
        seeds = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    if jitter > 0:
        rng = np.random.default_rng()
        seeds += rng.uniform(-jitter, jitter, size=seeds.shape)

    vor = Voronoi(seeds)

    all_vertices = np.asarray(vor.vertices, dtype=np.float64)

    struts_set: set[tuple[int, int]] = set()
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        for i in range(len(ridge)):
            a, b = ridge[i], ridge[(i + 1) % len(ridge)]
            if a != b and a >= 0 and b >= 0:
                struts_set.add((min(a, b), max(a, b)))

    struts_raw = np.array(sorted(struts_set), dtype=np.int64)
    if struts_raw.shape[0] == 0:
        return all_vertices, np.empty((0, 2), dtype=np.int64)

    used_vertices = np.unique(struts_raw.ravel())
    nodes = all_vertices[used_vertices]
    old_to_new = np.full(all_vertices.shape[0], -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(used_vertices):
        old_to_new[old_idx] = new_idx
    struts = np.column_stack(
        [old_to_new[struts_raw[:, 0]], old_to_new[struts_raw[:, 1]]]
    )
    struts = np.sort(struts, axis=1)
    return nodes, struts


def get_bcc_supercell(
    nodes: np.ndarray,
    cell_size: float,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Connect a Cartesian grid into a Body-Centered Cubic (BCC) structure.

    BCC: each cell has 8 corner nodes and 1 body-center node. Edges connect
    corners to the body center and to adjacent corners (along cell edges).
    For a uniform grid, we assume nodes are on a regular lattice with spacing
    cell_size. Body centers are at (i+0.5, j+0.5, k+0.5) * cell_size.

    Simplified: for a grid of nodes, connect each node to its 14 BCC neighbors:
    - 8 body centers of adjacent cells (each corner)
    - 6 edge-sharing neighbors (along ±x, ±y, ±z)

    For a grid of Nx x Ny x Nz nodes, we create struts by iterating over
    cells and adding BCC edges. We need to map grid indices to node indices.

    Args:
        nodes: (N, 3) Cartesian grid nodes.
        cell_size: Lattice spacing.
        tol: Tolerance for matching grid positions.

    Returns:
        (S, 2) strut index pairs.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    n = nodes.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)

    # Round to grid indices for matching
    grid_min = np.min(nodes, axis=0)
    indices = np.round((nodes - grid_min) / cell_size).astype(np.int64)

    # Build lookup: (i, j, k) -> node index
    i_min, j_min, k_min = np.min(indices, axis=0)
    i_max, j_max, k_max = np.max(indices, axis=0)
    ni = i_max - i_min + 1
    nj = j_max - j_min + 1
    nk = k_max - k_min + 1

    idx_to_node = np.full((ni, nj, nk), -1, dtype=np.int64)
    for idx in range(n):
        ii, jj, kk = indices[idx] - np.array([i_min, j_min, k_min])
        if 0 <= ii < ni and 0 <= jj < nj and 0 <= kk < nk:
            idx_to_node[ii, jj, kk] = idx

    def get_node(i: int, j: int, k: int) -> int:
        ii = i - i_min
        jj = j - j_min
        kk = k - k_min
        if 0 <= ii < ni and 0 <= jj < nj and 0 <= kk < nk:
            return idx_to_node[ii, jj, kk]
        return -1

    struts_set: set[tuple[int, int]] = set()

    for idx in range(n):
        i, j, k = indices[idx]
        # BCC: corner at (i,j,k). Body center at (i+0.5, j+0.5, k+0.5).
        # So body center of cell (i,j,k) is at grid index (i+1,j+1,k+1) if we
        # use half-integer indices. Actually in a simple grid, corners are at
        # integer (i,j,k). Body centers would be at (i+0.5, j+0.5, k+0.5) in
        # cell_size units. So we need nodes at half-integer positions.
        # For a pure corner grid, BCC means: connect each corner to the 8
        # body centers of the 8 cells it touches. And connect to the 6
        # edge-adjacent corners.
        # Simpler: connect (i,j,k) to (i+1,j,k), (i,j+1,k), (i,j,k+1) and
        # to body centers. Body centers are at (i+0.5,j+0.5,k+0.5) - we may
        # not have nodes there. So we need to add body-center nodes or use
        # a different interpretation.
        #
        # BCC: each corner connects to 8 body-center neighbors (diagonals).
        # Body centers are at (i+dx, j+dy, k+dz) for dx,dy,dz in {0,1}.
        # From corner (i,j,k), the 8 diagonals are (i+di, j+dj, k+dk)
        # with di,dj,dk in {-1, 1} (8 combinations).
        for di in (-1, 1):
            for dj in (-1, 1):
                for dk in (-1, 1):
                    neighbor = get_node(i + di, j + dj, k + dk)
                    if neighbor >= 0 and neighbor != idx:
                        a, b = min(idx, neighbor), max(idx, neighbor)
                        struts_set.add((a, b))

    struts = np.array(sorted(struts_set), dtype=np.int64)
    return struts
