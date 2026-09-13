"""
Graphite Explicit Interlinked — Ring Patterns Module

Parametric generation of explicit multi-body interlinked ring systems:
    - European 4-in-1: alternating tilt on square grids, looping adjacent rings.
    - Japanese Kusari: orthogonal flat XY rings and vertical XZ/YZ arch links.
    - Cubic 3D: 2x2x2 cube of 8 interlinked rings with alternating orthogonal planes.
    - Volumetric Kusari: 3D periodic lattice tiling along X, Y, and Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Ring:
    """
    Parametric closed ring representation.

    Attributes:
        center: (3,) Cartesian coordinates of the ring center in mm.
        normal: (3,) unit normal vector defining the plane of the ring.
        radius: Major radius (distance from ring center to wire centerline) in mm.
        wire_radius: Minor radius (radius of the strut wire cross-section) in mm.
        nodes: (V, 3) nodal coordinates of the discretized polygonal circle.
        struts: (S, 2) edge connectivity forming the closed loop.
        tag: Semantic identifier (e.g. 'flat', 'arch_x', 'arch_y', 'tilt_pos').
        cell_index: Discrete lattice coordinate tuple (i, j, k).
    """
    center: np.ndarray
    normal: np.ndarray
    radius: float
    wire_radius: float
    nodes: np.ndarray
    struts: np.ndarray
    tag: str = ""
    cell_index: tuple[int, ...] = field(default_factory=tuple)

    @property
    def outer_radius(self) -> float:
        """Outer boundary radius in mm (radius + wire_radius)."""
        return float(self.radius + self.wire_radius)

    @property
    def inner_radius(self) -> float:
        """Inner clearance radius in mm (radius - wire_radius)."""
        return float(self.radius - self.wire_radius)

    @property
    def outer_diameter(self) -> float:
        """Outer diameter in mm."""
        return 2.0 * self.outer_radius

    @property
    def inner_diameter(self) -> float:
        """Inner clearance hole diameter in mm."""
        return 2.0 * self.inner_radius


def _orthonormal_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors (u, v) orthonormal to normal."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        raise ValueError("Degenerate ring normal (zero length).")
    n = n / n_norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n, up))) > 0.90:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    u = np.cross(up, n)
    u_len = float(np.linalg.norm(u))
    if u_len < 1e-12:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(up, n)
        u_len = float(np.linalg.norm(u))
    u = u / u_len
    v = np.cross(n, u)
    v = v / float(np.linalg.norm(v))
    return u, v


def generate_ring(
    center: np.ndarray | list[float] | tuple[float, float, float],
    normal: np.ndarray | list[float] | tuple[float, float, float],
    radius: float,
    wire_radius: float,
    num_segments: int = 24,
    tag: str = "",
    cell_index: tuple[int, ...] = (),
) -> Ring:
    """
    Construct a discrete polygonal Ring in 3D space.

    Args:
        center: (3,) center coordinate in mm.
        normal: (3,) plane normal vector.
        radius: Major radius (centerline) in mm.
        wire_radius: Strut wire cross-section radius in mm.
        num_segments: Number of polygonal segments forming the circle.
        tag: Optional pattern classification tag.
        cell_index: Grid coordinate identifier.

    Returns:
        Ring dataclass containing nodes (num_segments, 3) and struts (num_segments, 2).
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / float(np.linalg.norm(n))
    r = float(radius)
    wr = float(wire_radius)
    n_seg = int(num_segments)

    if r <= 0:
        raise ValueError(f"Ring radius must be positive, got {r}")
    if wr <= 0:
        raise ValueError(f"Wire radius must be positive, got {wr}")
    if n_seg < 3:
        raise ValueError(f"num_segments must be >= 3, got {n_seg}")

    u, v = _orthonormal_plane_basis(n)
    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    nodes = c + r * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    idx = np.arange(n_seg, dtype=np.int64)
    struts = np.column_stack((idx, (idx + 1) % n_seg))

    return Ring(
        center=c,
        normal=n,
        radius=r,
        wire_radius=wr,
        nodes=nodes,
        struts=struts,
        tag=tag,
        cell_index=cell_index,
    )


def generate_european_4in1_rings(
    grid_size: tuple[int, int] | tuple[int, int, int] = (5, 5, 1),
    pitch: float = 10.0,
    radius_ratio: float = 0.65,
    wire_radius: float = 0.40,
    tilt_angle_deg: float = 28.0,
    num_segments: int = 24,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis_type: str = "diagonal_checker",
) -> list[Ring]:
    """
    Generate European 4-in-1 maille rings on a 2D square grid.

    On a square grid with pitch L (1:1 aspect ratio), rings have outer diameter
    ~1.38 * L, inner diameter > 2 * strut_diameter + clearance, and wire radius r.
    Rings are tilted at alternating angles around the weave axis so adjacent rings
    loop through each other in 4 directions without colliding.

    Args:
        grid_size: (nx, ny) or (nx, ny, nz) grid dimensions.
        pitch: Grid cell pitch L in mm.
        radius_ratio: Ratio of ring major radius to pitch (R / L). Default 0.65.
        wire_radius: Strut wire radius in mm. Default 0.40 mm.
        tilt_angle_deg: Weave tilt angle in degrees. Default 28.0 deg.
        num_segments: Polygon discretization segments per ring.
        origin: (x0, y0, z0) origin offset in mm.
        axis_type: Weave tilt axis pattern ('diagonal_checker' or 'row_alternating').

    Returns:
        List of Ring objects.
    """
    nx = int(grid_size[0])
    ny = int(grid_size[1])
    L = float(pitch)
    R = float(radius_ratio * L)
    wr = float(wire_radius)
    theta_rad = float(np.radians(tilt_angle_deg))
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    # Weave axis in XY plane
    if axis_type == "diagonal_checker":
        axis = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
    else:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    base_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    rings: list[Ring] = []

    for j in range(ny):
        for i in range(nx):
            center = orig + np.array([i * L, j * L, 0.0], dtype=np.float64)
            if axis_type == "diagonal_checker":
                sign = 1.0 if (i + j) % 2 == 0 else -1.0
            else:
                sign = 1.0 if j % 2 == 0 else -1.0

            angle = sign * theta_rad

            # Rodrigues rotation of base_normal around axis by angle
            k = axis
            v = base_normal
            n = (
                v * np.cos(angle)
                + np.cross(k, v) * np.sin(angle)
                + k * float(np.dot(k, v)) * (1.0 - np.cos(angle))
            )
            n = n / float(np.linalg.norm(n))

            tag = "tilt_pos" if sign > 0 else "tilt_neg"
            ring = generate_ring(
                center=center,
                normal=n,
                radius=R,
                wire_radius=wr,
                num_segments=num_segments,
                tag=tag,
                cell_index=(i, j, 0),
            )
            rings.append(ring)

    return rings


def generate_kusari_rings(
    grid_size: tuple[int, int] | tuple[int, int, int] = (5, 5, 1),
    pitch: float = 10.0,
    flat_radius: float = 3.6,
    arch_radius: float = 4.1,
    wire_radius: float = 0.35,
    num_segments: int = 24,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[Ring]:
    """
    Generate Japanese Kusari (Hira-kusari / So-kusari 4-in-1) maille rings.

    Consists of:
    - Horizontal flat rings lying in the XY plane centered at (2i * d, 2j * d, 0).
    - Vertical linking arch rings lying in the XZ plane at ((2i + 1) * d, 2j * d, 0).
    - Vertical linking arch rings lying in the YZ plane at (2i * d, (2j + 1) * d, 0).
    where d = pitch / 2.

    Each flat ring is linked with up to 4 vertical rings (North, South, East, West),
    and each vertical linking ring connects 2 adjacent flat rings.

    Args:
        grid_size: (nx, ny) or (nx, ny, nz) flat ring counts along X and Y.
        pitch: Distance between adjacent flat ring centers in mm (2 * d).
        flat_radius: Major radius of horizontal flat rings in mm.
        arch_radius: Major radius of vertical linking arch rings in mm.
        wire_radius: Strut wire radius in mm.
        num_segments: Polygon discretization segments per ring.
        origin: (x0, y0, z0) origin offset in mm.

    Returns:
        List of Ring objects.
    """
    nx = int(grid_size[0])
    ny = int(grid_size[1])
    d = float(pitch) / 2.0
    rf = float(flat_radius)
    ra = float(arch_radius)
    wr = float(wire_radius)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    rings: list[Ring] = []

    # 1. Flat rings in XY plane
    n_flat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            center = orig + np.array([2 * i * d, 2 * j * d, 0.0], dtype=np.float64)
            rings.append(
                generate_ring(
                    center=center,
                    normal=n_flat,
                    radius=rf,
                    wire_radius=wr,
                    num_segments=num_segments,
                    tag="flat",
                    cell_index=(i, j, 0),
                )
            )

    # 2. Arch X rings in XZ plane (normal along Y)
    n_arch_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for j in range(ny):
        for i in range(nx - 1):
            center = orig + np.array([(2 * i + 1) * d, 2 * j * d, 0.0], dtype=np.float64)
            rings.append(
                generate_ring(
                    center=center,
                    normal=n_arch_x,
                    radius=ra,
                    wire_radius=wr,
                    num_segments=num_segments,
                    tag="arch_x",
                    cell_index=(i, j, 0),
                )
            )

    # 3. Arch Y rings in YZ plane (normal along X)
    n_arch_y = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for j in range(ny - 1):
        for i in range(nx):
            center = orig + np.array([2 * i * d, (2 * j + 1) * d, 0.0], dtype=np.float64)
            rings.append(
                generate_ring(
                    center=center,
                    normal=n_arch_y,
                    radius=ra,
                    wire_radius=wr,
                    num_segments=num_segments,
                    tag="arch_y",
                    cell_index=(i, j, 0),
                )
            )

    return rings


def generate_cubic_8ring(
    pitch: float = 8.0,
    radius: float = 5.10,
    wire_radius: float = 0.35,
    num_segments: int = 24,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[Ring]:
    """
    Generate a 2x2x2 cube of 8 interlinked rings at the vertices of [0, L]^3.

    The 8 ring plane normals are assigned along orthogonal axes:
      assign = (X, Y, Y, X, X, Z, Z, X) for vertices ordered as (i, j, k) in {0, 1}^3.
    This creates an interlocking 3D loop along the edges of the cube with
    guaranteed clearance > 0.44 mm and zero collisions.

    Args:
        pitch: Cube side length L in mm.
        radius: Major radius of rings in mm.
        wire_radius: Strut wire radius in mm.
        num_segments: Discretization segments per ring.
        origin: (x0, y0, z0) origin offset in mm.

    Returns:
        List of 8 Ring objects forming a 2x2x2 interlinked cube.
    """
    L = float(pitch)
    r = float(radius)
    wr = float(wire_radius)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),  # 0: X
        np.array([0.0, 1.0, 0.0], dtype=np.float64),  # 1: Y
        np.array([0.0, 0.0, 1.0], dtype=np.float64),  # 2: Z
    ]

    # Vertex ordering:
    # 0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1),
    # 4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)
    assignment = (0, 1, 1, 0, 0, 2, 2, 0)

    rings: list[Ring] = []
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                center = orig + np.array([i * L, j * L, k * L], dtype=np.float64)
                axis_id = assignment[idx]
                normal = axes[axis_id]
                ring = generate_ring(
                    center=center,
                    normal=normal,
                    radius=r,
                    wire_radius=wr,
                    num_segments=num_segments,
                    tag=f"cube_ax{axis_id}",
                    cell_index=(i, j, k),
                )
                rings.append(ring)
                idx += 1

    return rings


def generate_volumetric_kusari_rings(
    grid_size: tuple[int, int, int] = (2, 2, 2),
    pitch: float = 10.0,
    flat_radius: float = 3.6,
    arch_radius: float = 4.1,
    arch_z_radius: float = 5.4,
    wire_radius: float = 0.30,
    num_segments: int = 24,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[Ring]:
    """
    Generate a 3D volumetric Kusari lattice repeating along X, Y, and Z.

    Includes:
    - Flat rings in the XY plane at (2i*d, 2j*d, 2k*d).
    - X-arch linking rings in XZ at ((2i+1)*d, 2j*d, 2k*d).
    - Y-arch linking rings in YZ at (2i*d, (2j+1)*d, 2k*d).
    - Z-arch linking rings linking layer k to k+1 at (2i*d, 2j*d, (2k+1)*d).

    Args:
        grid_size: (nx, ny, nz) cell counts.
        pitch: Cell pitch (2 * d) in mm.
        flat_radius: Major radius of XY flat rings in mm.
        arch_radius: Major radius of X and Y linking rings in mm.
        arch_z_radius: Major radius of vertical Z linking rings in mm.
        wire_radius: Strut wire radius in mm.
        num_segments: Discretization segments per ring.
        origin: (x0, y0, z0) origin offset in mm.

    Returns:
        List of Ring objects.
    """
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    d = float(pitch) / 2.0
    rf = float(flat_radius)
    ra = float(arch_radius)
    rz = float(arch_z_radius)
    wr = float(wire_radius)
    orig = np.asarray(origin, dtype=np.float64).reshape(3)

    rings: list[Ring] = []

    # 1. Flat rings in XY plane
    n_flat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                center = orig + np.array([2 * i * d, 2 * j * d, 2 * k * d], dtype=np.float64)
                rings.append(
                    generate_ring(
                        center=center,
                        normal=n_flat,
                        radius=rf,
                        wire_radius=wr,
                        num_segments=num_segments,
                        tag="vol_flat",
                        cell_index=(i, j, k),
                    )
                )

    # 2. Arch X rings in XZ plane
    n_arch_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                center = orig + np.array([(2 * i + 1) * d, 2 * j * d, 2 * k * d], dtype=np.float64)
                rings.append(
                    generate_ring(
                        center=center,
                        normal=n_arch_x,
                        radius=ra,
                        wire_radius=wr,
                        num_segments=num_segments,
                        tag="vol_arch_x",
                        cell_index=(i, j, k),
                    )
                )

    # 3. Arch Y rings in YZ plane
    n_arch_y = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                center = orig + np.array([2 * i * d, (2 * j + 1) * d, 2 * k * d], dtype=np.float64)
                rings.append(
                    generate_ring(
                        center=center,
                        normal=n_arch_y,
                        radius=ra,
                        wire_radius=wr,
                        num_segments=num_segments,
                        tag="vol_arch_y",
                        cell_index=(i, j, k),
                    )
                )

    # 4. Arch Z rings linking k to k+1
    n_arch_z = np.array([1.0, 1.0, 0.0], dtype=np.float64) / np.sqrt(2.0)
    for k in range(nz - 1):
        for j in range(ny):
            for i in range(nx):
                center = orig + np.array([2 * i * d, 2 * j * d, (2 * k + 1) * d], dtype=np.float64)
                rings.append(
                    generate_ring(
                        center=center,
                        normal=n_arch_z,
                        radius=rz,
                        wire_radius=wr,
                        num_segments=num_segments,
                        tag="vol_arch_z",
                        cell_index=(i, j, k),
                    )
                )

    return rings
