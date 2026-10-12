"""
Graphite Explicit Interlinked — Spatial Seeding & Analytical Primitives (Phase 3)

Provides:
- Translational lattice seeders:
    - Cartesian (pcu / square_2d)
    - Staggered brick (square_staggered_2d)
    - Diamond cubic (dia)
    - Hexagonal close-pack (hex_2d)
- Analytical primitive mappings with strict particle rigidity:
    - Cylindrical wraps with exact pitch-quantization (2*pi*R = N_theta * a_theta)
    - Spherical shells with Fibonacci spiral distribution
- Universal placer:
    - instantiate_lattice_on_sites: places any InterlinkedCell on arbitrary spatial sites and frames.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .particle import InterlinkedParticle
from .cell import InterlinkedCell


# =============================================================================
# 1. Translational Parent Lattice Seeders
# =============================================================================

def seed_cartesian_lattice(
    repeats: tuple[int, int] | tuple[int, int, int],
    pitch: float = 10.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed regular Cartesian lattice sites with identity orientation frames.

    Args:
        repeats: (nx, ny) or (nx, ny, nz) cell repeat counts.
        pitch: Nominal center-to-center cell spacing in mm.
        origin: (x0, y0, z0) offset coordinates.

    Returns:
        tuple: (centers, frames, grid_indices)
            centers: (N, 3) coordinates in mm.
            frames: (N, 3, 3) identity SO(3) frames.
            grid_indices: (N, 3) integer grid indices.
    """
    nx = int(repeats[0])
    ny = int(repeats[1])
    nz = int(repeats[2]) if len(repeats) > 2 else 1
    p = float(pitch)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    xs = np.arange(nx, dtype=np.float64) * p
    ys = np.arange(ny, dtype=np.float64) * p
    zs = np.arange(nz, dtype=np.float64) * p

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = orig + np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    ix, iy, iz = np.meshgrid(
        np.arange(nx, dtype=np.int64),
        np.arange(ny, dtype=np.int64),
        np.arange(nz, dtype=np.int64),
        indexing="ij",
    )
    indices = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()])

    num_pts = len(centers)
    frames = np.tile(np.eye(3, dtype=np.float64), (num_pts, 1, 1))

    return centers, frames, indices


def seed_staggered_lattice(
    repeats: tuple[int, int] | tuple[int, int, int],
    pitch: float = 10.0,
    stagger_fraction: float = 0.5,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed 2D/3D brick-staggered lattice sites where alternating rows (j mod 2)
    are shifted along the X-axis by `stagger_fraction * pitch`.
    """
    nx = int(repeats[0])
    ny = int(repeats[1])
    nz = int(repeats[2]) if len(repeats) > 2 else 1
    p = float(pitch)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    centers_list = []
    indices_list = []

    for k in range(nz):
        for j in range(ny):
            x_shift = (stagger_fraction * p) if (j % 2 == 1) else 0.0
            for i in range(nx):
                c = orig + np.array([i * p + x_shift, j * p, k * p], dtype=np.float64)
                centers_list.append(c)
                indices_list.append([i, j, k])

    centers = np.array(centers_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)
    frames = np.tile(np.eye(3, dtype=np.float64), (len(centers), 1, 1))

    return centers, frames, indices


def seed_hexagonal_lattice(
    repeats: tuple[int, int],
    pitch: float = 10.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed 2D hexagonal / triangular close-pack lattice sites in the XY plane.
    Row spacing is sqrt(3)/2 * pitch. Alternating rows shifted by 0.5 * pitch.
    """
    nx = int(repeats[0])
    ny = int(repeats[1])
    p = float(pitch)
    dy = p * (np.sqrt(3.0) / 2.0)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    centers_list = []
    indices_list = []

    for j in range(ny):
        x_shift = (0.5 * p) if (j % 2 == 1) else 0.0
        for i in range(nx):
            c = orig + np.array([i * p + x_shift, j * dy, 0.0], dtype=np.float64)
            centers_list.append(c)
            indices_list.append([i, j, 0])

    centers = np.array(centers_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)
    frames = np.tile(np.eye(3, dtype=np.float64), (len(centers), 1, 1))

    return centers, frames, indices


def resolve_orientation_matrix(
    orientation: np.ndarray | Sequence[float] | str | None = None,
    euler_angles_deg: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Resolve an arbitrary orientation specification into a 3x3 orthonormal rotation matrix.

    Supports:
        - None: returns 3x3 identity matrix np.eye(3).
        - (3, 3) array: validates shape and returns float64 rotation matrix.
        - '111', (1, 1, 1), [1, 1, 1]: canonical crystallographic (111) orientation
          aligning [1, 1, 1]/sqrt(3) with spatial +Z:
              u_z = [1, 1, 1] / sqrt(3)
              u_x = [1, -1, 0] / sqrt(2)
              u_y = [1, 1, -2] / sqrt(6)
              R = [u_x^T; u_y^T; u_z^T]
        - (3,) unit vector: aligns [0, 0, 1] to target vector via Rodrigues rotation.
        - euler_angles_deg: intrinsic [rx, ry, rz] Euler angles in degrees (Rx @ Ry @ Rz).
    """
    if euler_angles_deg is not None:
        rad = np.radians(np.asarray(euler_angles_deg, dtype=np.float64))
        cx, cy, cz = np.cos(rad)
        sx, sy, sz = np.sin(rad)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return Rx @ Ry @ Rz

    if orientation is None:
        return np.eye(3, dtype=np.float64)

    if isinstance(orientation, str):
        s = orientation.strip().lower().replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
        if s in ("111", "[1,1,1]", "[111]", "<111>"):
            uz = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
            ux = np.array([1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
            uy = np.array([1.0, 1.0, -2.0], dtype=np.float64) / np.sqrt(6.0)
            return np.vstack([ux, uy, uz])
        raise ValueError(f"Unknown orientation string '{orientation}'. Use '111' or pass a 3x3 matrix / Euler angles.")

    arr = np.asarray(orientation, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (3,):
        if np.allclose(arr, [1, 1, 1]):
            return resolve_orientation_matrix("111")
        target = arr / float(np.linalg.norm(arr))
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.allclose(target, z):
            return np.eye(3, dtype=np.float64)
        if np.allclose(target, -z):
            return np.diag([1.0, -1.0, -1.0]).astype(np.float64)
        axis = np.cross(z, target)
        axis /= float(np.linalg.norm(axis))
        angle = np.arccos(np.clip(float(np.dot(z, target)), -1.0, 1.0))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    raise ValueError(f"Unsupported orientation specification shape: {arr.shape}")


def seed_diamond_lattice(
    repeats: tuple[int, int, int],
    conventional_cell_size: float = 16.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_matrix: np.ndarray | Sequence[float] | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Seed crystallographic Diamond cubic (dia) sites.

    Args:
        repeats: (nx, ny, nz) conventional cubic unit cell repetition counts.
        conventional_cell_size: Lattice parameter a in mm.
        origin: (x0, y0, z0) spatial origin in mm.
        rotation_matrix: Optional 3x3 rotation matrix, '111', or Euler angles to
            orient the diamond crystal arbitrarily in 3D space.

    Returns:
        tuple: (centers, frames, indices, sublattice_ids)
            sublattice_ids: list of 'A' or 'B' for each site.
    """
    from .pams import diamond_network_sites

    a = float(conventional_cell_size)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)
    raw_sites = diamond_network_sites(repeats, a)

    centers_list = []
    sublattices = []
    indices_list = []

    for idx, (pos, sub) in enumerate(raw_sites):
        centers_list.append(pos)
        sublattices.append(sub)
        cell_ijk = tuple(np.floor(pos / a).astype(np.int64).tolist())
        indices_list.append(cell_ijk)

    centers = np.array(centers_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)

    if rotation_matrix is not None:
        R = resolve_orientation_matrix(rotation_matrix)
        centers = (R @ centers.T).T + orig
        frames = np.tile(R, (len(centers), 1, 1))
    else:
        centers = centers + orig
        frames = np.tile(np.eye(3, dtype=np.float64), (len(centers), 1, 1))

    return centers, frames, indices, sublattices


def _parse_miller_plane(plane: str | Sequence[int]) -> tuple[int, int, int]:
    """Parse Miller index plane representation into an integer (h, k, l) tuple."""
    if isinstance(plane, str):
        cleaned = plane.strip().replace("{", "").replace("}", "").replace("(", "").replace(")", "").replace("<", "").replace(">", "").replace(",", " ")
        parts = cleaned.split()
        if len(parts) == 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(cleaned) == 3 and all(c.isdigit() or c == "-" for c in cleaned):
            try:
                return (int(cleaned[0]), int(cleaned[1]), int(cleaned[2]))
            except ValueError:
                pass
        if cleaned == "100":
            return (1, 0, 0)
        elif cleaned == "110":
            return (1, 1, 0)
        elif cleaned == "111":
            return (1, 1, 1)
        raise ValueError(f"Unrecognized plane specification: '{plane}'")
    else:
        return (int(plane[0]), int(plane[1]), int(plane[2]))


def orthogonal_supercell(
    network: str = "dia",
    plane: str | Sequence[int] = "100",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute orthogonal supercell orientation matrix R and normalized layer periodicities
    for a given parent crystallographic network and habit plane (hkl).

    Args:
        network: Parent network type ('pcu', 'bcc', 'dia', 'diamond').
        plane: Miller indices as string (e.g. '100', '110', '111', '{111}') or tuple (h, k, l).

    Returns:
        tuple (R, periods):
            R: (3, 3) float64 orthonormal orientation matrix mapping crystallographic coordinates
               to the orthogonal supercell coordinate system (with habit plane normal along +Z).
            periods: (3,) float64 normalized periodic layer step sizes [Lx, Ly, Lz] (for a = 1.0).
    """
    net = str(network).lower().strip()
    h, k, l = _parse_miller_plane(plane)
    hkl_abs = tuple(sorted([abs(h), abs(k), abs(l)], reverse=True))

    if hkl_abs == (1, 0, 0):
        # {100} plane: identity orientation
        R = np.eye(3, dtype=np.float64)
        periods = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return R, periods

    elif hkl_abs == (1, 1, 0):
        # {110} plane: normal along [1, 1, 0] / sqrt(2)
        ux = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        uy = np.array([1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
        uz = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
        R = np.vstack([ux, uy, uz])
        if net in ("dia", "diamond"):
            periods = np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=np.float64)
        elif net in ("bcc", "body_centered"):
            periods = np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=np.float64)
        else:  # pcu / sc
            periods = np.array([1.0, np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=np.float64)
        return R, periods

    elif hkl_abs == (1, 1, 1):
        # {111} plane: normal along [1, 1, 1] / sqrt(3)
        ux = np.array([1.0, -1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
        uy = np.array([1.0, 1.0, -2.0], dtype=np.float64) / np.sqrt(6.0)
        uz = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
        R = np.vstack([ux, uy, uz])
        if net in ("dia", "diamond"):
            periods = np.array([1.0 / np.sqrt(2.0), np.sqrt(6.0) / 4.0, np.sqrt(3.0)], dtype=np.float64)
        elif net in ("bcc", "body_centered"):
            periods = np.array([1.0 / np.sqrt(2.0), np.sqrt(6.0) / 2.0, np.sqrt(3.0)], dtype=np.float64)
        else:  # pcu / sc
            periods = np.array([np.sqrt(2.0), np.sqrt(6.0), np.sqrt(3.0)], dtype=np.float64)
        return R, periods

    else:
        # General integer (h, k, l)
        n = np.array([h, k, l], dtype=np.float64)
        uz = n / float(np.linalg.norm(n))
        # Find orthogonal in-plane vector
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(uz, ref))) > 0.90:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ux = np.cross(ref, uz)
        ux = ux / float(np.linalg.norm(ux))
        uy = np.cross(uz, ux)
        uy = uy / float(np.linalg.norm(uy))
        R = np.vstack([ux, uy, uz])
        periods = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return R, periods


def seed_orthogonal_lattice(
    repeats: tuple[int, int, int],
    pitch: float = 10.0,
    network: str = "dia",
    plane: str | Sequence[int] = "100",
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    prune_unbonded: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Seed crystallographic sites for any cubic parent network (PCU, BCC, DIA)
    tiled along orthogonal habit plane supercell axes ({100}, {110}, {111}, or (hkl)).

    Guarantees flat boundary planes with zero diagonal slicing artifacts.
    When prune_unbonded=True, removes sites with zero network bonds and keeps
    only the largest connected component of the site-bond network.
    """
    nx, ny, nz = repeats
    a = float(pitch)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)
    net = str(network).lower().strip()

    R, periods = orthogonal_supercell(network=net, plane=plane)

    # Establish basis particles within conventional unit cell
    if net in ("dia", "diamond"):
        fcc_A = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]], dtype=float)
        fcc_B = fcc_A + np.array([0.25, 0.25, 0.25], dtype=float)
        basis = [("A", p) for p in fcc_A] + [("B", p) for p in fcc_B]
    elif net in ("bcc", "body_centered"):
        basis = [("A", np.array([0.0, 0.0, 0.0], dtype=float)), ("B", np.array([0.5, 0.5, 0.5], dtype=float))]
    else:  # pcu / sc
        basis = [("A", np.array([0.0, 0.0, 0.0], dtype=float))]

    # Box bounds
    x_max = nx * (a * periods[0])
    y_max = ny * (a * periods[1])
    z_max = nz * (a * periods[2])

    max_span = max(x_max, y_max, z_max) + 2.0 * a
    n_cube = int(np.ceil(max_span / a)) + 2

    eps = 1e-3
    centers_list = []
    sublattices = []
    indices_list = []

    for ix in range(-n_cube, n_cube + 1):
        for iy in range(-n_cube, n_cube + 1):
            for iz in range(-n_cube, n_cube + 1):
                shift = np.array([ix, iy, iz], dtype=float)
                for sub_id, p in basis:
                    pt_w = R @ ((shift + p) * a)
                    if -eps <= pt_w[0] <= x_max + eps and -eps <= pt_w[1] <= y_max + eps and -eps <= pt_w[2] <= z_max + eps:
                        centers_list.append(pt_w + orig)
                        sublattices.append(sub_id)
                        indices_list.append((ix, iy, iz))

    if prune_unbonded and len(centers_list) > 0:
        from scipy.spatial import cKDTree
        from scipy.sparse import csgraph, lil_matrix

        pts = np.array(centers_list, dtype=np.float64)
        if net in ("dia", "diamond"):
            target_d = a * np.sqrt(3.0) / 4.0
        elif net in ("bcc", "body_centered"):
            target_d = a * np.sqrt(3.0) / 2.0
        else:  # pcu / sc
            target_d = a

        tol = 1e-3 * a
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=target_d + tol)
        valid_pairs = []
        for i, j in pairs:
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            if abs(dist - target_d) <= tol:
                valid_pairs.append((i, j))

        active_indices = set(range(len(pts)))
        adj: dict[int, set[int]] = {i: set() for i in active_indices}
        for i, j in valid_pairs:
            adj[i].add(j)
            adj[j].add(i)

        changed = True
        while changed:
            changed = False
            to_remove = [i for i in active_indices if len(adj[i]) == 0]
            if to_remove:
                changed = True
                for i in to_remove:
                    active_indices.remove(i)
                    del adj[i]

        if not active_indices:
            centers_list = []
            sublattices = []
            indices_list = []
        else:
            active_list = sorted(active_indices)
            idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(active_list)}
            n_act = len(active_list)
            mat = lil_matrix((n_act, n_act), dtype=bool)
            for i in active_list:
                for j in adj[i]:
                    if j in active_indices:
                        mat[idx_map[i], idx_map[j]] = True
                        mat[idx_map[j], idx_map[i]] = True

            n_comps, labels = csgraph.connected_components(mat, directed=False)
            if n_comps > 1:
                comp_sizes = np.bincount(labels)
                largest_comp = int(np.argmax(comp_sizes))
                surviving_active = [active_list[k] for k in range(n_act) if labels[k] == largest_comp]
            else:
                surviving_active = active_list

            surv_set = set(surviving_active)
            centers_list = [centers_list[i] for i in range(len(centers_list)) if i in surv_set]
            sublattices = [sublattices[i] for i in range(len(sublattices)) if i in surv_set]
            indices_list = [indices_list[i] for i in range(len(indices_list)) if i in surv_set]

    centers = np.array(centers_list, dtype=np.float64) if centers_list else np.zeros((0, 3), dtype=np.float64)
    frames = np.tile(R, (len(centers), 1, 1)) if len(centers) > 0 else np.zeros((0, 3, 3), dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64) if indices_list else np.zeros((0, 3), dtype=np.int64)
    return centers, frames, indices, sublattices


def seed_diamond_orthogonal_111(
    repeats: tuple[int, int, int],
    pitch: float = 13.5,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    prune_unbonded: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Seed crystallographic Diamond cubic (dia) sites tiled along orthogonal {111} supercell axes.
    Backward-compatible wrapper around seed_orthogonal_lattice(network='dia', plane='111').
    """
    return seed_orthogonal_lattice(
        repeats=repeats,
        pitch=pitch,
        network="dia",
        plane="111",
        origin=origin,
        prune_unbonded=prune_unbonded,
    )



def seed_bcc_lattice(
    repeats: tuple[int, int, int],
    cell_size: float = 14.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_matrix: np.ndarray | Sequence[float] | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Seed crystallographic Body-Centered Cubic (bcc) sites.

    For each unit cell (i, j, k), yields:
      - Corner site (sublattice 'A') at [i, j, k] * a
      - Body-center site (sublattice 'B') at [i + 0.5, j + 0.5, k + 0.5] * a

    Args:
        repeats: (nx, ny, nz) conventional cubic unit cell repetition counts.
        cell_size: Conventional lattice parameter a in mm.
        origin: (x0, y0, z0) spatial origin in mm.
        rotation_matrix: Optional 3x3 rotation matrix, Euler angles, or vector
            to orient the BCC crystal in 3D space.

    Returns:
        tuple: (centers, frames, indices, sublattice_ids)
            sublattice_ids: list of 'A' or 'B' for each site.
    """
    nx = int(repeats[0])
    ny = int(repeats[1])
    nz = int(repeats[2]) if len(repeats) > 2 else 1
    a = float(cell_size)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    centers_list = []
    sublattices = []
    indices_list = []

    body_center_offset = np.array([0.5, 0.5, 0.5], dtype=np.float64) * a

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c_corner = np.array([i, j, k], dtype=np.float64) * a
                c_body = c_corner + body_center_offset

                centers_list.append(c_corner)
                sublattices.append("A")
                indices_list.append([i, j, k])

                centers_list.append(c_body)
                sublattices.append("B")
                indices_list.append([i, j, k])

    centers = np.array(centers_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)

    if rotation_matrix is not None:
        R = resolve_orientation_matrix(rotation_matrix)
        centers = (R @ centers.T).T + orig
        frames = np.tile(R, (len(centers), 1, 1))
    else:
        centers = centers + orig
        frames = np.tile(np.eye(3, dtype=np.float64), (len(centers), 1, 1))

    return centers, frames, indices, sublattices


def seed_zone_lattice(
    parent_network: str,
    repeats: tuple[int, int, int],
    pitch: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_matrix: np.ndarray | Sequence[float] | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """
    Dispatch to the correct lattice seeder based on parent network.

    Args:
        parent_network: 'pcu', 'square_2d', 'dia', 'bcc', etc.
        repeats: (nx, ny, nz) cell repeat counts.
        pitch: Fundamental unit cell pitch (mm).
        origin: (x0, y0, z0) origin offset in mm.
        rotation_matrix: Optional orientation.

    Returns:
        tuple: (centers, frames, indices, sublattice_ids)
            sublattice_ids is None for single-site-per-cell networks (pcu, square_2d).
    """
    net = parent_network.lower().strip()
    if net in ("pcu", "square_2d"):
        c, f, idx = seed_cartesian_lattice(repeats, pitch, origin)
        if rotation_matrix is not None:
            R = resolve_orientation_matrix(rotation_matrix)
            orig = np.asarray(origin, dtype=np.float64).reshape(3)
            c = (R @ (c - orig).T).T + orig
            f = np.tile(R, (len(c), 1, 1))
        return c, f, idx, None
    elif net == "dia":
        c, f, idx, subs = seed_diamond_lattice(repeats, pitch, origin, rotation_matrix)
        return c, f, idx, subs
    elif net == "bcc":
        c, f, idx, subs = seed_bcc_lattice(repeats, pitch, origin, rotation_matrix)
        return c, f, idx, subs
    else:
        c, f, idx = seed_cartesian_lattice(repeats, pitch, origin)
        return c, f, idx, None


# =============================================================================
# 2. Analytical Primitive Seeding: Cylindrical Wrap & Spherical Shell
# =============================================================================

def seed_cylindrical_wrap(
    radius: float,
    height: float,
    target_pitch: float = 10.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """
    Wrap an interlinked pattern seamlessly around a cylinder of radius R and height H.

    Mandates **exact pitch-matching quantization**:
        N_theta = round(2 * pi * R / target_pitch)
        a_theta = 2 * pi * R / N_theta
        N_z = round(H / target_pitch)
        a_z = H / N_z

    This guarantees that ring 0 and ring N_theta meet with identical spacing,
    preventing any link shearing, misalignment, or collision across the 0 -> 2*pi seam.

    Returns:
        dict containing:
            'centers': (N_theta * N_z, 3) coordinates.
            'frames': (N_theta * N_z, 3, 3) orthonormal local frames [t_theta, t_z, n_radial].
            'indices': (N_theta * N_z, 3) grid indices (i_theta, j_z, 0).
            'quantized_pitch_theta': a_theta in mm.
            'quantized_pitch_z': a_z in mm.
            'num_theta': N_theta.
            'num_z': N_z.
    """
    R = float(radius)
    H = float(height)
    p = float(target_pitch)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    if R <= 0 or H <= 0 or p <= 0:
        raise ValueError("radius, height, and target_pitch must be positive")

    # Quantize circumference to integer number of cells
    N_theta = max(4, int(round(2.0 * np.pi * R / p)))
    a_theta = (2.0 * np.pi * R) / float(N_theta)

    N_z = max(1, int(round(H / p)))
    a_z = H / float(N_z)

    thetas = np.linspace(0.0, 2.0 * np.pi, N_theta, endpoint=False)
    zs = np.arange(N_z, dtype=np.float64) * a_z

    centers_list = []
    frames_list = []
    indices_list = []

    for j, z in enumerate(zs):
        for i, th in enumerate(thetas):
            c_local = np.array([R * np.cos(th), R * np.sin(th), z], dtype=np.float64)
            c_world = orig + c_local
            centers_list.append(c_world)

            # Orthonormal cylindrical surface basis:
            # t1 (azimuthal tangent): [-sin(th), cos(th), 0]
            # t2 (axial tangent):     [0, 0, 1]
            # n  (outward normal):    [cos(th), sin(th), 0]
            t1 = np.array([-np.sin(th), np.cos(th), 0.0], dtype=np.float64)
            t2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n = np.array([np.cos(th), np.sin(th), 0.0], dtype=np.float64)

            # Frame matrix mapping local XY to [t1, t2] with normal along +Z
            F = np.column_stack([t1, t2, n])
            frames_list.append(F)
            indices_list.append([i, j, 0])

    centers = np.array(centers_list, dtype=np.float64)
    frames = np.array(frames_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)

    return {
        "centers": centers,
        "frames": frames,
        "indices": indices,
        "quantized_pitch_theta": a_theta,
        "quantized_pitch_z": a_z,
        "num_theta": N_theta,
        "num_z": N_z,
        "radius": R,
        "height": H,
    }


def seed_spherical_shell(
    radius: float,
    target_pitch: float = 10.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """
    Distribute sites uniformly over a spherical shell of radius R using
    a Fibonacci golden spiral distribution.

    Maintains strict particle rigidity: each particle is positioned at the sphere
    surface and oriented with its local +Z normal aligned radially outward.
    """
    R = float(radius)
    p = float(target_pitch)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    if R <= 0 or p <= 0:
        raise ValueError("radius and target_pitch must be positive")

    # Estimated point count from surface area
    area = 4.0 * np.pi * (R ** 2)
    cell_area = p ** 2
    N = max(12, int(round(area / cell_area)))

    indices_arr = np.arange(N, dtype=np.float64)
    # Fibonacci spherical mapping
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians (~2.39996)
    y = 1.0 - (indices_arr / float(N - 1)) * 2.0  # y goes from 1 to -1
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * indices_arr

    x = np.cos(theta) * r
    z = np.sin(theta) * r

    unit_normals = np.column_stack([x, y, z])
    centers = orig + unit_normals * R

    frames_list = []
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    for n in unit_normals:
        ref = alt_up if abs(float(np.dot(n, up))) > 0.90 else up
        t1 = np.cross(ref, n)
        t1 = t1 / float(np.linalg.norm(t1))
        t2 = np.cross(n, t1)
        t2 = t2 / float(np.linalg.norm(t2))
        F = np.column_stack([t1, t2, n])
        frames_list.append(F)

    frames = np.array(frames_list, dtype=np.float64)
    indices = np.column_stack([np.arange(N, dtype=np.int64), np.zeros(N, dtype=np.int64), np.zeros(N, dtype=np.int64)])

    return {
        "centers": centers,
        "frames": frames,
        "indices": indices,
        "num_sites": N,
        "radius": R,
    }


# =============================================================================
# 3. Universal Site Placer
# =============================================================================

def instantiate_lattice_on_sites(
    cell: InterlinkedCell,
    centers: np.ndarray,
    frames: np.ndarray,
    grid_indices: np.ndarray,
    cell_pitch: float,
    id_start: int = 0,
    context: dict[str, Any] | None = None,
    sublattice_ids: Sequence[str] | None = None,
) -> list[InterlinkedParticle]:
    """
    Instantiate and place all BasisParticles of an InterlinkedCell across
    arbitrary spatial sites and surface frames.

    Combines:
        T_world = [F | c] @ [R_cell | offset]
    where F is the site surface frame (from cylindrical wrap or plane) and
    R_cell is the internal orientation function of the cell basis.

    Args:
        cell: An InterlinkedCell implementation.
        centers: (N, 3) spatial coordinates.
        frames: (N, 3, 3) orthonormal orientation frames.
        grid_indices: (N, 3) integer grid indices.
        cell_pitch: Local characteristic spacing in mm.
        id_start: Initial particle ID.
        context: Optional dictionary passed to cell orientation functions.
        sublattice_ids: Optional sequence of sublattice IDs ('A', 'B', etc.) per site.

    Returns:
        List of placed InterlinkedParticle instances.
    """
    ctx = context or {}
    all_particles: list[InterlinkedParticle] = []
    pid = int(id_start)

    num_sites = len(centers)
    for s_idx in range(num_sites):
        c_site = centers[s_idx]
        F_site = frames[s_idx]
        ijk = tuple(grid_indices[s_idx].tolist())

        site_ctx = dict(ctx)
        if sublattice_ids is not None and s_idx < len(sublattice_ids):
            site_ctx["sublattice"] = sublattice_ids[s_idx]

        # Instantiate cell basis at local site
        local_parts = cell.instantiate_site(
            grid_index=ijk,
            site_origin=np.zeros(3, dtype=np.float64),
            cell_pitch=cell_pitch,
            id_start=pid,
            context=site_ctx,
        )

        for p in local_parts:
            # Compose world transform:
            # World rotation = F_site @ p.rotation
            # World position = c_site + F_site @ p.center
            R_world = F_site @ p.rotation
            t_world = c_site + F_site @ p.center

            T_world = np.eye(4, dtype=np.float64)
            T_world[:3, :3] = R_world
            T_world[:3, 3] = t_world

            p.transform = T_world
            p.particle_id = pid
            all_particles.append(p)
            pid += 1

    return all_particles
