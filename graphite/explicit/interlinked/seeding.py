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


def seed_diamond_lattice(
    repeats: tuple[int, int, int],
    conventional_cell_size: float = 16.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Seed crystallographic Diamond cubic (dia) sites.

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
        centers_list.append(orig + pos)
        sublattices.append(sub)
        # Approximate integer cell index
        cell_ijk = tuple(np.floor(pos / a).astype(np.int64).tolist())
        indices_list.append(cell_ijk)

    centers = np.array(centers_list, dtype=np.float64)
    indices = np.array(indices_list, dtype=np.int64)
    frames = np.tile(np.eye(3, dtype=np.float64), (len(centers), 1, 1))

    return centers, frames, indices, sublattices


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
