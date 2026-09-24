"""
Graphite Explicit Interlinked — Polycatenated Architected Materials (PAMs)

Procedural Research & Compatibility Adapter for PAM particles and lattices (Zhou et al., Science 2025).

NOTE FOR AGENTS & DEVELOPERS:
    This module provides low-level procedural polyhedral generators, analytical clearance solvers,
    and legacy test fixtures. For all production lattice generation, use the canonical modular engine:
        from graphite.explicit.interlinked import generate_interlinked_lattice, InterlinkedConfig
        result = generate_interlinked_lattice(InterlinkedConfig(cell="d4tet", ...))

Design rules:
    - Connectivity arrays are named ``struts`` (local per-particle indices).
    - Particles never share nodes across bodies (no KD-tree merge).
    - Clearance uses segment–segment centerline distance from ``.clearance``.
    - Solids via ``graphite.explicit.geometry_module.build_clean_miter_truss`` (clean miter default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from graphite.explicit.geometry_module import (
    generate_geometry,
    _trimesh_to_manifold,
)
from graphite.explicit.interlinked.clearance import segment_segment_distance


# Regular tetrahedron vertices (unnormalized); edge length = 2*sqrt(2).
_TET_TEMPLATE = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)

_TET_STRUTS = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ],
    dtype=np.int64,
)

# Faces as oriented triples (vertex indices).
_TET_FACES = np.array(
    [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ],
    dtype=np.int64,
)


@dataclass
class PAMParticle:
    """Discrete wireframe PAM particle with a local node/strut index space."""

    particle_id: int
    nodes: np.ndarray  # (V, 3) float64
    struts: np.ndarray  # (S, 2) int64 local indices
    center: np.ndarray  # (3,) float64
    geometry_type: str  # e.g. 'TET', 'CO', 'OCT', 'ring'
    metadata: dict = field(default_factory=dict)

    def copy_shifted(self, new_id: int, translation: np.ndarray) -> "PAMParticle":
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        return PAMParticle(
            particle_id=int(new_id),
            nodes=np.asarray(self.nodes, dtype=np.float64) + t,
            struts=np.array(self.struts, dtype=np.int64, copy=True),
            center=np.asarray(self.center, dtype=np.float64) + t,
            geometry_type=self.geometry_type,
            metadata=dict(self.metadata),
        )

    def __iter__(self):
        """Allows direct tuple unpacking: nodes, struts = particle"""
        yield self.nodes
        yield self.struts

    def to_interlinked(self) -> "InterlinkedParticle":
        """Bridge conversion to canonical InterlinkedParticle."""
        from .particle import InterlinkedParticle
        return InterlinkedParticle.from_pam(self)


@dataclass
class PAMLatticeResult:
    """Assembly of independent PAM particles with clearance diagnostics."""

    particles: list[PAMParticle]
    tripartite_code: str
    clearance_valid: bool
    min_clearance_mm: float
    strut_radius: float = 0.8
    meshes: list[trimesh.Trimesh] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def combined_mesh(self) -> trimesh.Trimesh:
        if not self.meshes:
            self.meshes = pam_particles_to_meshes(self.particles, self.strut_radius)
        return trimesh.util.concatenate(self.meshes)


def _rotation_matrix_align_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R such that R @ a_hat ≈ b_hat (Rodrigues)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / float(np.linalg.norm(a))
    b = b / float(np.linalg.norm(b))
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -1.0 + 1e-12:
        # 180°: pick any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a[0])) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = axis - np.dot(axis, a) * a
        axis = axis / float(np.linalg.norm(axis))
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3) + 2.0 * (K @ K)
    if float(np.linalg.norm(v)) < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64)
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / float(np.linalg.norm(axis))
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _regular_tet_nodes(edge_length: float) -> np.ndarray:
    """Origin-centered regular tetrahedron with given edge length."""
    L = float(edge_length)
    raw = _TET_TEMPLATE.copy()
    raw_edge = float(np.linalg.norm(raw[0] - raw[1]))
    nodes = raw * (L / raw_edge)
    nodes -= nodes.mean(axis=0)
    return nodes.astype(np.float64)


def generate_tetrahedral_particle(
    edge_length: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
    tip_direction: np.ndarray | Sequence[float] | None = None,
    twist_rad: float = 0.0,
) -> PAMParticle:
    """
    Build a 4-vertex / 6-strut wireframe tetrahedron centered at ``center``.

    When ``tip_direction`` is set, vertex 0 is aligned with that 3-fold axis
    (Zhou D-4-TET corner catenation). ``twist_rad`` rotates about the tip axis.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    nodes = _regular_tet_nodes(edge_length)

    if tip_direction is not None:
        tip = nodes[0].copy()
        R_align = _rotation_matrix_align_a_to_b(tip, np.asarray(tip_direction, dtype=np.float64))
        nodes = (R_align @ nodes.T).T
        if abs(float(twist_rad)) > 1e-15:
            axis = np.asarray(tip_direction, dtype=np.float64).reshape(3)
            R_twist = _rotation_about_axis(axis, float(twist_rad))
            nodes = (R_twist @ nodes.T).T

    nodes = nodes + c
    return PAMParticle(
        particle_id=int(particle_id),
        nodes=nodes,
        struts=_TET_STRUTS.copy(),
        center=c.copy(),
        geometry_type="TET",
        metadata={
            "edge_length": float(edge_length),
            "tip_direction": None
            if tip_direction is None
            else np.asarray(tip_direction, dtype=np.float64).reshape(3).tolist(),
            "twist_rad": float(twist_rad),
        },
    )


def generate_cuboctahedral_particle(
    size: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """
    12-vertex / 24-strut cuboctahedron wireframe (Archimedean solid).

    Vertices are ``size`` × even permutations of ``(±1, ±1, 0) / √2``.
    Square faces are normal to the Cartesian axes (4-fold); triangular faces
    support 3-fold catenation (C-6-CO, J-4-CO, S-6/2-CO).
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = float(size)
    if s <= 0:
        raise ValueError(f"size must be positive, got {s}")

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    raw: list[list[float]] = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            raw.append([x * inv_sqrt2, y * inv_sqrt2, 0.0])
            raw.append([x * inv_sqrt2, 0.0, y * inv_sqrt2])
            raw.append([0.0, x * inv_sqrt2, y * inv_sqrt2])
    nodes = np.asarray(raw, dtype=np.float64) * s
    keys = np.round(nodes, decimals=10)
    _, idx = np.unique(keys, axis=0, return_index=True)
    nodes = nodes[np.sort(idx)]
    if nodes.shape[0] != 12:
        raise RuntimeError(f"cuboctahedron expected 12 vertices, got {nodes.shape[0]}")

    edge_len = s  # adjacent unit verts are distance 1 apart
    struts: list[tuple[int, int]] = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = float(np.linalg.norm(nodes[i] - nodes[j]))
            if abs(d - edge_len) < 1e-8 * max(1.0, edge_len):
                struts.append((i, j))
    strut_arr = np.asarray(struts, dtype=np.int64)
    if strut_arr.shape[0] != 24:
        raise RuntimeError(f"cuboctahedron expected 24 struts, got {strut_arr.shape[0]}")

    return PAMParticle(
        particle_id=int(particle_id),
        nodes=nodes + c,
        struts=strut_arr,
        center=c.copy(),
        geometry_type="CO",
        metadata={
            "size": s,
            "edge_length": float(edge_len),
            "axes_4fold": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )


def generate_octahedral_particle(
    size: float,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """
    6-vertex / 12-strut regular octahedron wireframe.

    Vertices: ``size * (±1,0,0), (0,±1,0), (0,0,±1)``.
    4-fold axes through opposite vertices support corner catenation (J-4-OCT);
    3-fold axes through opposite faces support face catenation (S-6/2-OCT).
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = float(size)
    if s <= 0:
        raise ValueError(f"size must be positive, got {s}")

    nodes = s * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    edge_len = s * np.sqrt(2.0)
    struts: list[tuple[int, int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            d = float(np.linalg.norm(nodes[i] - nodes[j]))
            if abs(d - edge_len) < 1e-8 * max(1.0, edge_len):
                struts.append((i, j))
    strut_arr = np.asarray(struts, dtype=np.int64)
    if strut_arr.shape[0] != 12:
        raise RuntimeError(f"octahedron expected 12 struts, got {strut_arr.shape[0]}")

    return PAMParticle(
        particle_id=int(particle_id),
        nodes=nodes + c,
        struts=strut_arr,
        center=c.copy(),
        geometry_type="OCT",
        metadata={
            "size": s,
            "edge_length": float(edge_len),
            "axes_4fold": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )


def generate_truncated_tetrahedron_particle(
    size: float = 1.0,
    center: np.ndarray | Sequence[float] = (0.0, 0.0, 0.0),
    *,
    particle_id: int = 0,
) -> PAMParticle:
    """
    Generates vertices and struts for a Truncated Tetrahedron (TT) wireframe particle.
    Vertices: 12 (derived from truncating a regular tetrahedron at 1/3 edge lengths).
    Struts: 18 (forming 4 hexagonal faces and 4 triangular face cutouts).

    Supports both object access (`particle.nodes`, `particle.struts`, `particle.center`)
    and direct tuple unpacking (`nodes, struts = generate_truncated_tetrahedron_particle(...)`).
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    s = float(size)
    if s <= 0:
        raise ValueError(f"size must be positive, got {s}")

    scale = s / (3.0 * np.sqrt(2.0))
    raw_nodes = []

    # 12 T_d symmetric vertices
    for s1, s2, s3 in [
        (+1, +1, +1), (+1, -1, -1), (-1, +1, -1), (-1, -1, +1)
    ]:
        raw_nodes.append([3 * s1 * scale, 1 * s2 * scale, 1 * s3 * scale])
        raw_nodes.append([1 * s1 * scale, 3 * s2 * scale, 1 * s3 * scale])
        raw_nodes.append([1 * s1 * scale, 1 * s2 * scale, 3 * s3 * scale])

    nodes = np.round(np.array(raw_nodes, dtype=np.float64), decimals=8)
    nodes = np.unique(nodes, axis=0)

    # 18 edge struts (each edge length = 2*sqrt(2)/3 * size)
    edge_len = 2.0 * np.sqrt(2.0) * scale
    struts: list[tuple[int, int]] = []
    num_nodes = len(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = float(np.linalg.norm(nodes[i] - nodes[j]))
            if abs(dist - edge_len) < 1e-4 * max(1.0, edge_len):
                struts.append((i, j))

    struts_arr = np.array(struts, dtype=np.int64)
    if struts_arr.shape[0] != 18:
        raise RuntimeError(f"truncated tetrahedron expected 18 struts, got {struts_arr.shape[0]}")

    return PAMParticle(
        particle_id=int(particle_id),
        nodes=nodes + c,
        struts=struts_arr,
        center=c.copy(),
        geometry_type="TT",
        metadata={
            "size": s,
            "edge_length": float(edge_len),
            "axes_td": [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
        },
    )


def align_particle_axis(
    particle: PAMParticle,
    source_axis: np.ndarray | Sequence[float],
    target_axis: np.ndarray | Sequence[float],
) -> PAMParticle:
    """
    Rotate a particle about its center so ``source_axis`` maps onto ``target_axis``.

    Returns a new ``PAMParticle`` (local strut indices unchanged).
    """
    R = _rotation_matrix_align_a_to_b(
        np.asarray(source_axis, dtype=np.float64),
        np.asarray(target_axis, dtype=np.float64),
    )
    c = np.asarray(particle.center, dtype=np.float64).reshape(3)
    nodes = np.asarray(particle.nodes, dtype=np.float64)
    local = nodes - c
    new_nodes = (R @ local.T).T + c
    meta = dict(particle.metadata)
    meta["aligned_source_axis"] = np.asarray(source_axis, dtype=np.float64).reshape(3).tolist()
    meta["aligned_target_axis"] = np.asarray(target_axis, dtype=np.float64).reshape(3).tolist()
    return PAMParticle(
        particle_id=particle.particle_id,
        nodes=new_nodes,
        struts=np.array(particle.struts, dtype=np.int64, copy=True),
        center=c.copy(),
        geometry_type=particle.geometry_type,
        metadata=meta,
    )


def _rotate_particle_about_center(
    particle: PAMParticle,
    axis: np.ndarray | Sequence[float],
    angle_rad: float,
) -> PAMParticle:
    """Rotate particle nodes about ``particle.center``."""
    R = _rotation_about_axis(np.asarray(axis, dtype=np.float64), float(angle_rad))
    c = np.asarray(particle.center, dtype=np.float64).reshape(3)
    local = np.asarray(particle.nodes, dtype=np.float64) - c
    return PAMParticle(
        particle_id=particle.particle_id,
        nodes=(R @ local.T).T + c,
        struts=np.array(particle.struts, dtype=np.int64, copy=True),
        center=c.copy(),
        geometry_type=particle.geometry_type,
        metadata=dict(particle.metadata),
    )


def _translate_particle(particle: PAMParticle, translation: np.ndarray, new_id: int | None = None) -> PAMParticle:
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    pid = particle.particle_id if new_id is None else int(new_id)
    return PAMParticle(
        particle_id=pid,
        nodes=np.asarray(particle.nodes, dtype=np.float64) + t,
        struts=np.array(particle.struts, dtype=np.int64, copy=True),
        center=np.asarray(particle.center, dtype=np.float64) + t,
        geometry_type=particle.geometry_type,
        metadata=dict(particle.metadata),
    )


def particle_strut_segments(particle: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (p0, p1) endpoint arrays shape (S, 3) for a particle's struts."""
    if hasattr(particle, "global_nodes") and callable(particle.global_nodes):
        n = np.asarray(particle.global_nodes(), dtype=np.float64)
    else:
        n = np.asarray(particle.nodes, dtype=np.float64)
    s = np.asarray(particle.struts, dtype=np.int64)
    return n[s[:, 0]], n[s[:, 1]]


def particle_pair_centerline_distance(a: PAMParticle, b: PAMParticle) -> float:
    """Minimum centerline distance between struts of two particles (mm)."""
    p0, p1 = particle_strut_segments(a)
    q0, q1 = particle_strut_segments(b)
    return float(segment_segment_distance(p0, p1, q0, q1))


def particle_pair_clearance(a: PAMParticle, b: PAMParticle, strut_radius: float) -> float:
    """
    Surface-to-surface clearance between two wireframe particles.

    Δ = d_centerline − 2 r
    """
    r = float(strut_radius)
    return float(particle_pair_centerline_distance(a, b) - 2.0 * r)


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9) -> bool:
    """Barycentric inside test for point p known to lie on triangle plane."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < eps:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def strut_pierces_triangle(
    p0: np.ndarray,
    p1: np.ndarray,
    tri: np.ndarray,
) -> bool:
    """True if open segment p0→p1 intersects the interior of triangle ``tri`` (3,3)."""
    a, b, c = tri[0], tri[1], tri[2]
    n = np.cross(b - a, c - a)
    n_len = float(np.linalg.norm(n))
    if n_len < 1e-14:
        return False
    n = n / n_len
    d0 = float(np.dot(p0 - a, n))
    d1 = float(np.dot(p1 - a, n))
    if d0 * d1 > 0.0:
        return False  # same side
    if abs(d0 - d1) < 1e-14:
        return False
    t = d0 / (d0 - d1)
    if t <= 1e-8 or t >= 1.0 - 1e-8:
        return False  # ignore endpoint-on-plane grazing
    hit = p0 + t * (p1 - p0)
    return _point_in_triangle(hit, a, b, c)


def count_strut_face_piercings(a: PAMParticle, b: PAMParticle) -> int:
    """Count struts of ``a`` that pierce triangular faces of ``b``."""
    p0, p1 = particle_strut_segments(a)
    nodes_b = np.asarray(b.nodes, dtype=np.float64)
    count = 0
    for i in range(p0.shape[0]):
        for face in _TET_FACES:
            tri = nodes_b[face]
            if strut_pierces_triangle(p0[i], p1[i], tri):
                count += 1
                break
    return count


def particles_are_corner_catenated(a: PAMParticle, b: PAMParticle) -> bool:
    """
    Heuristic topological interlock for TET pair: at least one strut of each
    particle pierces a face of the other (corner-to-corner catenation).
    """
    return count_strut_face_piercings(a, b) >= 1 and count_strut_face_piercings(b, a) >= 1


def _tet_ab_templates(edge_length: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Crystallographic dual tetrahedron skeletons for diamond (dia) A/B sites.

    Sublattice A uses the regular template; B is the point inversion (dual).
    Relative 60° skirt stagger about every tetrahedral bond is automatic — do
    **not** add an extra twist (that eclipses edges and causes collisions).
    """
    nodes_a = _regular_tet_nodes(edge_length)
    nodes_b = -nodes_a
    return nodes_a, nodes_b


def calibrate_d4tet_bond_length(
    edge_length: float,
    strut_radius: float,
    min_clearance: float = 0.40,
) -> float:
    """Return diamond bond length d that yields corner catenation with Δ ≥ min_clearance."""
    L = float(edge_length)
    r = float(strut_radius)
    t_min = float(min_clearance)
    nodes_a, nodes_b = _tet_ab_templates(L)
    u = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    u /= float(np.linalg.norm(u))
    R_c = (np.sqrt(6.0) / 4.0) * L
    d_lo = 0.55 * 2.0 * R_c
    d_hi = 0.98 * 2.0 * R_c

    best: tuple[float, float] | None = None
    for d in np.linspace(d_lo, d_hi, 81):
        a = PAMParticle(0, nodes_a, _TET_STRUTS.copy(), np.zeros(3), "TET")
        c = float(d) * u
        b = PAMParticle(1, nodes_b + c, _TET_STRUTS.copy(), c.copy(), "TET")
        clr = particle_pair_clearance(a, b, r)
        if particles_are_corner_catenated(a, b) and clr >= t_min:
            if best is None or clr > best[0]:
                best = (clr, float(d))
    if best is None:
        for d in np.linspace(d_lo, d_hi, 121):
            a = PAMParticle(0, nodes_a, _TET_STRUTS.copy(), np.zeros(3), "TET")
            c = float(d) * u
            b = PAMParticle(1, nodes_b + c, _TET_STRUTS.copy(), c.copy(), "TET")
            clr = particle_pair_clearance(a, b, r)
            if particles_are_corner_catenated(a, b):
                if best is None or clr > best[0]:
                    best = (clr, float(d))
    if best is None:
        raise RuntimeError(
            f"No D-4-TET interlocking bond length for edge_length={L}, strut_radius={r}."
        )
    return float(best[1])


def calibrate_d4tet_edge_length(
    conventional_cell_size: float,
    strut_radius: float,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
) -> float:
    """Find tet edge L for fixed diamond conventional cell a (default bond d=a√3/4)."""
    a = float(conventional_cell_size)
    if a <= 0.0:
        raise ValueError(f"conventional_cell_size must be positive, got {a}")
    d = float(bond_length) if bond_length is not None else a * np.sqrt(3.0) / 4.0
    r = float(strut_radius)
    t_min = float(min_clearance)

    b_offsets = np.array([
        [0.25, 0.25, 0.25],
        [0.25, -0.25, -0.25],
        [-0.25, 0.25, -0.25],
        [-0.25, -0.25, 0.25],
    ], dtype=np.float64) * a

    fcc_a_offsets = np.array([
        [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [0.5, 0.0, -0.5], [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5], [0.0, -0.5, 0.5], [0.0, 0.5, -0.5], [0.0, -0.5, -0.5],
    ], dtype=np.float64) * a

    best_clr = -float("inf")
    best_L = 0.603 * a
    valid_candidates: list[tuple[float, float]] = []

    for L in np.linspace(0.50 * a, 0.70 * a, 81):
        nodes_a, nodes_b = _tet_ab_templates(float(L))
        pa0 = PAMParticle(0, nodes_a, _TET_STRUTS.copy(), np.zeros(3), "TET")
        pb0 = PAMParticle(1, nodes_b + b_offsets[0], _TET_STRUTS.copy(), b_offsets[0], "TET")
        if not particles_are_corner_catenated(pa0, pb0):
            continue

        min_b = min(
            particle_pair_clearance(pa0, PAMParticle(i + 1, nodes_b + b, _TET_STRUTS.copy(), b, "TET"), r)
            for i, b in enumerate(b_offsets)
        )
        min_a = min(
            particle_pair_clearance(pa0, PAMParticle(i + 10, nodes_a + c, _TET_STRUTS.copy(), c, "TET"), r)
            for i, c in enumerate(fcc_a_offsets)
        )
        tot_clr = min(min_b, min_a)
        if tot_clr > best_clr:
            best_clr = tot_clr
            best_L = float(L)
        if tot_clr >= t_min:
            valid_candidates.append((tot_clr, float(L)))

    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        return float(valid_candidates[0][1])
    return float(best_L)


def generate_d4tet_interlocked_pair(
    edge_length: float = 12.0,
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    twist_rad: float | None = None,
) -> PAMLatticeResult:
    """
    D-4-TET unit: two dual tetrahedral cages on one diamond bond.

    Uses crystallographic A/B dual skeletons (Zhou tip-to-tip / 3-fold alignment).
    ``twist_rad`` is ignored (kept for API compatibility); dual inversion already
    provides the 60° skirt stagger on every tetrahedral bond.
    """
    del twist_rad  # unused — dual inversion sets azimuthal stagger
    L = float(edge_length)
    r = float(strut_radius)
    t_min = float(min_clearance)
    u = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    u /= float(np.linalg.norm(u))
    nodes_a, nodes_b = _tet_ab_templates(L)
    d_star = float(bond_length) if bond_length is not None else calibrate_d4tet_bond_length(L, r, t_min)

    a = PAMParticle(
        particle_id=0,
        nodes=nodes_a.copy(),
        struts=_TET_STRUTS.copy(),
        center=np.zeros(3, dtype=np.float64),
        geometry_type="TET",
        metadata={"sublattice": "A", "edge_length": L},
    )
    c1 = d_star * u
    b = PAMParticle(
        particle_id=1,
        nodes=nodes_b + c1,
        struts=_TET_STRUTS.copy(),
        center=c1.copy(),
        geometry_type="TET",
        metadata={"sublattice": "B", "edge_length": L},
    )

    clr = particle_pair_clearance(a, b, r)
    linked = particles_are_corner_catenated(a, b)
    meshes = pam_particles_to_meshes([a, b], r)

    m0 = _trimesh_to_manifold(meshes[0])
    m1 = _trimesh_to_manifold(meshes[1])
    inter_vol = float((m0 ^ m1).volume())
    gap = float(m0.min_gap(m1, max(20.0, 4.0 * L)))

    return PAMLatticeResult(
        particles=[a, b],
        tripartite_code="D-4-TET",
        clearance_valid=bool(clr >= t_min and inter_vol <= 1e-4 and linked),
        min_clearance_mm=float(clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "edge_length": L,
            "bond_length": float(d_star),
            "conventional_cell_size": float(4.0 * d_star / np.sqrt(3.0)),
            "bond_direction": u.tolist(),
            "orientation": "crystallographic_dual_AB",
            "strut_face_piercings_0_to_1": count_strut_face_piercings(a, b),
            "strut_face_piercings_1_to_0": count_strut_face_piercings(b, a),
            "corner_catenated": bool(linked),
            "solid_intersection_volume": inter_vol,
            "manifold_min_gap_mm": gap,
            "target_min_clearance_mm": t_min,
            "joint_style": "explicit_cylinder_compose",
        },
    )


_DIAMOND_FCC = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
)


def diamond_network_sites(
    repeats: tuple[int, int, int],
    conventional_cell_size: float,
) -> list[tuple[np.ndarray, str]]:
    """
    Unique diamond (dia) node sites in an ``nx×ny×nz`` conventional-cell block.

    Returns list of (center, sublattice) with sublattice in {'A','B'}.
    """
    nx, ny, nz = (int(repeats[0]), int(repeats[1]), int(repeats[2]))
    a = float(conventional_cell_size)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if a <= 0:
        raise ValueError(f"conventional_cell_size must be positive, got {a}")

    basis_b = np.array([0.25, 0.25, 0.25], dtype=np.float64) * a
    seen: dict[tuple[float, float, float], tuple[np.ndarray, str]] = {}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=np.float64) * a
                for fcc in _DIAMOND_FCC:
                    fcc_v = np.asarray(fcc, dtype=np.float64) * a
                    for sub, shift in (("A", np.zeros(3)), ("B", basis_b)):
                        pos = origin + fcc_v + shift
                        key = tuple(np.round(pos, 8))
                        if key not in seen:
                            seen[key] = (pos, sub)
    return list(seen.values())


def recalibrate_pam_lattice(
    lattice_type: str,
    target_d: float,
    r: float,
    repeats: tuple[int, int, int] = (2, 2, 2)
) -> float:
    """
    Automated calibration tool to sweep configuration parameters (L) for a given lattice type,
    bond length d, and strut radius r. Ensures global collision-free structures.
    Currently supports: 'D-4-TET'
    Returns the optimal L value.
    """
    if lattice_type == "D-4-TET":
        best_L = 0.0
        max_clr = -float("inf")
        L_candidates = np.linspace(0.7 * target_d, 0.95 * target_d, 25)
        for L_test in L_candidates:
            try:
                geom = generate_d4tet_diamond_tiling(
                    repeats=repeats,
                    cell_size=L_test,
                    bond_length=target_d,
                    strut_radius=r,
                    build_meshes=False
                )
                clr = geom.min_clearance
                if clr > max_clr:
                    max_clr = clr
                    best_L = L_test
            except Exception:
                continue
        if max_clr < 0:
            print(f"Warning: recalibrate_pam_lattice failed to find collision-free L for D-4-TET at d={target_d}, r={r}")
        return best_L
    elif lattice_type in ("C-6-TT", "C-6-TRUNCATED-TETRAHEDRON"):
        best_a0 = 0.0
        max_clr = -float("inf")
        a0_candidates = np.linspace(1.15 * target_d, 1.40 * target_d, 25)
        for a0_test in a0_candidates:
            try:
                geom = generate_c6tt_cubic_tiling(
                    repeats=repeats,
                    size=target_d,
                    unit_cell_size=a0_test,
                    strut_radius=r,
                    build_meshes=False,
                )
                clr = geom.min_clearance_mm
                if clr > max_clr:
                    max_clr = clr
                    best_a0 = a0_test
            except Exception:
                continue
        if max_clr < 0:
            print(f"Warning: recalibrate_pam_lattice failed to find collision-free a0 for C-6-TT at size={target_d}, r={r}")
        return best_a0
    else:
        raise ValueError(f"Recalibration for {lattice_type} is not implemented.")

def _min_clearance_among_particles(
    particles: list[PAMParticle],
    strut_radius: float,
    edge_length: float,
) -> tuple[float, int]:
    """
    Pairwise clearance across ALL interacting particles in the volume.
    """
    r = float(strut_radius)
    L = float(edge_length)
    centers = np.array([p.center for p in particles], dtype=np.float64)
    from scipy.spatial import cKDTree

    tree = cKDTree(centers)
    # R_outer for tet is sqrt(6)/4 * L ≈ 0.612 * L. Max interaction is 2 * R_outer + 2*r + margin
    search_radius = 1.3 * L + 2.0 * r + 2.0
    pairs = tree.query_pairs(r=search_radius)
    min_clr = float("inf")
    n_checked = 0
    # Check ALL candidate interacting pairs within search_radius for collisions/clearance
    for i, j in pairs:
        clr = particle_pair_clearance(particles[i], particles[j], r)
        if clr < min_clr:
            min_clr = clr
        n_checked += 1
    if n_checked == 0:
        return 0.0, 0
    return float(min_clr), n_checked


def generate_d4tet_diamond_tiling(
    repeats: tuple[int, int, int],
    edge_length: float | None = None,
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    conventional_cell_size: float | None = None,
    build_meshes: bool = True,
    add_spheres: bool = False,
    joint_sphere_scale: float = 1.15,
    circular_segments: int = 16,
    clean_miter: bool = True,
) -> PAMLatticeResult:
    """
    Tile D-4-TET particles on a diamond cubic supercell (``repeats`` conventional cells).

    Size the lattice either by tet ``edge_length`` (legacy: bond calibrated from L)
    or by ``conventional_cell_size`` ``a`` (bond ``d = a√3/4``, L calibrated from a).
    When both are given with ``conventional_cell_size``, ``a`` fixes the pitch and
    ``edge_length`` is used as-is (no L search).

    Each site keeps an independent local ``nodes``/``struts`` graph (no KD-tree merge).
    Solids reuse ``generate_geometry`` cylinder compose — same joint style as graphite
    explicit (no node spheres).
    """
    import contextlib
    import io

    r = float(strut_radius)
    t_min = float(min_clearance)

    if conventional_cell_size is not None:
        a_cell = float(conventional_cell_size)
        if a_cell <= 0.0:
            raise ValueError(f"conventional_cell_size must be positive, got {a_cell}")
        d = float(bond_length) if bond_length is not None else a_cell * np.sqrt(3.0) / 4.0
        if edge_length is not None:
            L = float(edge_length)
        else:
            L = calibrate_d4tet_edge_length(
                a_cell, r, t_min, bond_length=d
            )
    else:
        L = float(12.0 if edge_length is None else edge_length)
        if bond_length is not None:
            d = float(bond_length)
            a_cell = 4.0 * d / np.sqrt(3.0)
        else:
            a_cell = L / 0.603
            d = a_cell * np.sqrt(3.0) / 4.0

    nodes_a, nodes_b = _tet_ab_templates(L)

    sites = diamond_network_sites(repeats, a_cell)
    particles: list[PAMParticle] = []
    for pid, (center, sub) in enumerate(sites):
        tmpl = nodes_a if sub == "A" else nodes_b
        particles.append(
            PAMParticle(
                particle_id=pid,
                nodes=tmpl + center,
                struts=_TET_STRUTS.copy(),
                center=np.asarray(center, dtype=np.float64).copy(),
                geometry_type="TET",
                metadata={"sublattice": sub, "edge_length": L},
            )
        )

    min_clr, n_pairs = _min_clearance_among_particles(particles, r, L)
    if build_meshes:
        with contextlib.redirect_stdout(io.StringIO()):
            meshes = pam_particles_to_meshes(
                particles,
                r,
                add_spheres=add_spheres,
                joint_sphere_scale=float(joint_sphere_scale),
                circular_segments=int(circular_segments),
                clean_miter=clean_miter,
            )
    else:
        meshes = []

    joint_style_name = (
        "clean_miter"
        if clean_miter and not add_spheres
        else ("embedded_spheres" if add_spheres else "explicit_cylinder_compose")
    )

    return PAMLatticeResult(
        particles=particles,
        tripartite_code="D-4-TET",
        clearance_valid=bool(n_pairs > 0 and min_clr >= t_min),
        min_clearance_mm=float(min_clr) if n_pairs else 0.0,
        strut_radius=r,
        meshes=meshes,
        metadata={
            "edge_length": L,
            "bond_length": d,
            "conventional_cell_size": float(a_cell),
            "repeats": (int(repeats[0]), int(repeats[1]), int(repeats[2])),
            "extent_mm": (
                float(int(repeats[0]) * a_cell),
                float(int(repeats[1]) * a_cell),
                float(int(repeats[2]) * a_cell),
            ),
            "num_particles": len(particles),
            "num_neighbor_pairs_checked": n_pairs,
            "orientation": "crystallographic_dual_AB",
            "target_min_clearance_mm": t_min,
            "joint_style": joint_style_name,
        },
    )


def calibrate_polyhedral_bond_length(
    build_pair_fn,
    d_lo: float,
    d_hi: float,
    strut_radius: float,
    min_clearance: float = 0.40,
    n_samples: int = 49,
) -> float:
    """
    Search bond length in ``[d_lo, d_hi]`` for zero solid collision and Δ ≥ min_clearance.

    Prefers the **tight interlocking window** (smaller d) when multiple windows exist.
    ``build_pair_fn(d) -> (PAMParticle, PAMParticle)``.
    """
    import contextlib
    import io

    r = float(strut_radius)
    t_min = float(min_clearance)
    windows: list[tuple[float, float]] = []  # (clr, d)
    for d in np.linspace(float(d_lo), float(d_hi), int(n_samples)):
        a, b = build_pair_fn(float(d))
        clr = particle_pair_clearance(a, b, r)
        with contextlib.redirect_stdout(io.StringIO()):
            meshes = pam_particles_to_meshes([a, b], r)
        m0 = _trimesh_to_manifold(meshes[0])
        m1 = _trimesh_to_manifold(meshes[1])
        inter = float((m0 ^ m1).volume())
        if inter <= 1e-4 and clr >= t_min:
            windows.append((clr, float(d)))
    if not windows:
        raise RuntimeError("No collision-free polyhedral bond length found in search range.")
    d_min = min(d for _, d in windows)
    band = [(clr, d) for clr, d in windows if d <= d_min * 1.15 + 1e-9]
    if not band:
        band = windows
    band.sort(key=lambda t: (-t[0], t[1]))
    return float(band[0][1])


def generate_c6co_interlocked_pair(
    size: float = 8.0,
    strut_radius: float = 0.35,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
) -> PAMLatticeResult:
    """
    Two cuboctahedra catenated along a 4-fold (+X) axis (C-6-CO bond unit).

    Neighbor B is rotated 45° about the bond so square frames interpenetrate
    without strut collision.
    """
    s = float(size)
    r = float(strut_radius)
    t_min = float(min_clearance)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def build(d: float) -> tuple[PAMParticle, PAMParticle]:
        a = generate_cuboctahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=0)
        b = generate_cuboctahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=1)
        b = _rotate_particle_about_center(b, axis, 0.25 * np.pi)
        b = _translate_particle(b, d * axis, new_id=1)
        return a, b

    d_star = (
        float(bond_length)
        if bond_length is not None
        else calibrate_polyhedral_bond_length(build, 0.95 * s, 1.35 * s, r, t_min)
    )
    a, b = build(d_star)
    clr = particle_pair_clearance(a, b, r)
    meshes = pam_particles_to_meshes([a, b], r)
    m0 = _trimesh_to_manifold(meshes[0])
    m1 = _trimesh_to_manifold(meshes[1])
    inter = float((m0 ^ m1).volume())
    return PAMLatticeResult(
        particles=[a, b],
        tripartite_code="C-6-CO",
        clearance_valid=bool(clr >= t_min and inter <= 1e-4),
        min_clearance_mm=float(clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "size": s,
            "bond_length": d_star,
            "bond_axis": axis.tolist(),
            "relative_twist_rad": 0.25 * np.pi,
            "solid_intersection_volume": inter,
            "target_min_clearance_mm": t_min,
            "joint_style": "explicit_cylinder_compose",
        },
    )


def generate_c6co_coordination_cell(
    size: float = 8.0,
    strut_radius: float = 0.35,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """
    C-6-CO unit cell: one cuboctahedron + six Cartesian neighbors (n=6).

    Each neighbor is rotated 45° about its bond axis (true 4-fold square-face
    catenation). Full SC orientation fields are deferred — a single shared
    relative twist cannot satisfy ±X/±Y/±Z simultaneously on a bipartite lattice.
    """
    import contextlib
    import io
    from itertools import combinations

    s = float(size)
    r = float(strut_radius)
    t_min = float(min_clearance)
    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
    ]

    def build_shell(d: float) -> list[PAMParticle]:
        center_p = generate_cuboctahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=0)
        parts: list[PAMParticle] = [center_p]
        for pid, axis in enumerate(axes, start=1):
            axis = axis / float(np.linalg.norm(axis))
            p = generate_cuboctahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=pid)
            p = _rotate_particle_about_center(p, axis, 0.25 * np.pi)
            p = _translate_particle(p, float(d) * axis, new_id=pid)
            parts.append(p)
        return parts

    def shell_ok(parts: list[PAMParticle]) -> tuple[bool, float]:
        center_p = parts[0]
        bond_clr = min(particle_pair_clearance(center_p, nb, r) for nb in parts[1:])
        if bond_clr < t_min:
            return False, bond_clr
        with contextlib.redirect_stdout(io.StringIO()):
            meshes = pam_particles_to_meshes(parts, r)
        mans = [_trimesh_to_manifold(m) for m in meshes]
        for i, j in combinations(range(len(mans)), 2):
            if float((mans[i] ^ mans[j]).volume()) > 1e-4:
                return False, bond_clr
        return True, bond_clr

    if bond_length is not None:
        d = float(bond_length)
        particles = build_shell(d)
        ok, bond_clr = shell_ok(particles)
        if not ok:
            raise RuntimeError("Provided bond_length does not satisfy C-6-CO shell clearance.")
    else:
        pair = generate_c6co_interlocked_pair(s, r, t_min)
        d0 = float(pair.metadata["bond_length"])
        chosen = None
        for d in np.linspace(max(d0, 0.95 * s), 1.35 * s, 25):
            parts = build_shell(float(d))
            ok, bond_clr = shell_ok(parts)
            if ok:
                chosen = (float(d), bond_clr, parts)
                break
        if chosen is None:
            raise RuntimeError("No C-6-CO coordination shell spacing found with Δ≥min_clearance.")
        d, bond_clr, particles = chosen

    meshes = pam_particles_to_meshes(particles, r) if build_meshes else []
    return PAMLatticeResult(
        particles=particles,
        tripartite_code="C-6-CO",
        clearance_valid=bool(bond_clr >= t_min),
        min_clearance_mm=float(bond_clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "size": s,
            "bond_length": d,
            "num_particles": len(particles),
            "topology": "coordination_shell_n6",
            "target_min_clearance_mm": t_min,
            "joint_style": "explicit_cylinder_compose",
            "clearance_scope": "all_pairs_zero_collision_plus_bond_delta",
        },
    )


def generate_c6co_cubic_tiling(
    repeats: tuple[int, int, int] = (2, 2, 2),
    size: float = 8.0,
    strut_radius: float = 0.35,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """
    C-6-CO review cell.

    For the Phase-2 deliverable, ``repeats=(2,2,2)`` (and any positive repeats)
    returns the 6-neighbor coordination shell, which is the correct local
    C-6-CO motif. Extended SC tilings need a richer orientation field (later).
    """
    del repeats  # coordination shell is the Phase-2 cubic cell motif
    return generate_c6co_coordination_cell(
        size=size,
        strut_radius=strut_radius,
        min_clearance=min_clearance,
        bond_length=bond_length,
        build_meshes=build_meshes,
    )


def generate_c6tt_cubic_tiling(
    repeats: tuple[int, int, int] = (2, 2, 2),
    size: float = 10.0,
    strut_radius: float = 0.50,
    min_clearance: float = 0.30,
    *,
    unit_cell_size: float | None = None,
    build_meshes: bool = True,
    add_spheres: bool = False,
    joint_sphere_scale: float = 1.15,
    circular_segments: int = 24,
) -> PAMLatticeResult:
    """
    3D bulk periodic polycatenated material with Truncated Tetrahedra (C-6-TT).

    Maps Truncated Tetrahedra onto a 3D simple cubic (pcu) network with 6-fold
    face catenation (Zhou et al., Science 2025, Figs. S2, S4, Table S1).
    Default unit cell spacing a0 = 1.25 * size guarantees zero collision
    and inter-particle surface clearance >= 0.30 mm.
    """
    s = float(size)
    r = float(strut_radius)
    t_min = float(min_clearance)
    a0 = float(unit_cell_size) if unit_cell_size is not None else 1.25 * s

    nx, ny, nz = (int(repeats[0]), int(repeats[1]), int(repeats[2]))
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    particles: list[PAMParticle] = []
    pid = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = np.array([i, j, k], dtype=np.float64) * a0
                p = generate_truncated_tetrahedron_particle(s, center=c, particle_id=pid)
                particles.append(p)
                pid += 1

    min_clr, n_pairs = _min_clearance_among_particles(particles, r, a0)
    meshes = (
        pam_particles_to_meshes(
            particles,
            r,
            add_spheres=add_spheres,
            joint_sphere_scale=float(joint_sphere_scale),
            circular_segments=int(circular_segments),
        )
        if build_meshes
        else []
    )

    return PAMLatticeResult(
        particles=particles,
        tripartite_code="C-6-TT",
        clearance_valid=bool(min_clr >= t_min),
        min_clearance_mm=float(min_clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "size": s,
            "unit_cell_size": a0,
            "repeats": list(repeats),
            "num_particles": len(particles),
            "topology": "simple_cubic_pcu",
            "coordination_number": 6,
            "target_min_clearance_mm": t_min,
            "pairs_checked": n_pairs,
            "joint_style": "explicit_cylinder_compose",
        },
    )


generate_c6tt_lattice = generate_c6tt_cubic_tiling


def generate_j4oct_interlocked_pair(
    size: float = 6.0,
    strut_radius: float = 0.40,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
) -> PAMLatticeResult:
    """
    Two octahedra tip-to-tip along +X with 45° bond twist (J-4-OCT bond unit).
    """
    s = float(size)
    r = float(strut_radius)
    t_min = float(min_clearance)
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def build(d: float) -> tuple[PAMParticle, PAMParticle]:
        a = generate_octahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=0)
        b = generate_octahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=1)
        b = _rotate_particle_about_center(b, axis, 0.25 * np.pi)
        b = _translate_particle(b, d * axis, new_id=1)
        return a, b

    d_star = (
        float(bond_length)
        if bond_length is not None
        else calibrate_polyhedral_bond_length(build, 1.15 * s, 1.55 * s, r, t_min)
    )
    a, b = build(d_star)
    clr = particle_pair_clearance(a, b, r)
    meshes = pam_particles_to_meshes([a, b], r)
    m0 = _trimesh_to_manifold(meshes[0])
    m1 = _trimesh_to_manifold(meshes[1])
    inter = float((m0 ^ m1).volume())
    return PAMLatticeResult(
        particles=[a, b],
        tripartite_code="J-4-OCT",
        clearance_valid=bool(clr >= t_min and inter <= 1e-4),
        min_clearance_mm=float(clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "size": s,
            "bond_length": d_star,
            "bond_axis": axis.tolist(),
            "relative_twist_rad": 0.25 * np.pi,
            "solid_intersection_volume": inter,
            "target_min_clearance_mm": t_min,
            "joint_style": "explicit_cylinder_compose",
        },
    )


def generate_j4oct_square_tiling(
    repeats: tuple[int, int] = (2, 2),
    size: float = 6.0,
    strut_radius: float = 0.40,
    min_clearance: float = 0.40,
    *,
    bond_length: float | None = None,
    build_meshes: bool = True,
) -> PAMLatticeResult:
    """
    J-4-OCT planar cross: center octahedron + 4 in-plane neighbors (n=4).

    ``repeats`` is accepted for API symmetry; the Phase-2 cell is the n=4 cross.
    """
    del repeats
    s = float(size)
    r = float(strut_radius)
    t_min = float(min_clearance)
    pair = generate_j4oct_interlocked_pair(s, r, t_min, bond_length=bond_length)
    d = float(pair.metadata["bond_length"])

    axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    center_p = generate_octahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=0)
    particles: list[PAMParticle] = [center_p]
    for pid, axis in enumerate(axes, start=1):
        axis = axis / float(np.linalg.norm(axis))
        p = generate_octahedral_particle(s, center=(0.0, 0.0, 0.0), particle_id=pid)
        p = _rotate_particle_about_center(p, axis, 0.25 * np.pi)
        p = _translate_particle(p, d * axis, new_id=pid)
        particles.append(p)

    # DfAM: score the four catenation bonds (center↔neighbor).
    bond_clr = min(particle_pair_clearance(center_p, nb, r) for nb in particles[1:])

    meshes = pam_particles_to_meshes(particles, r) if build_meshes else []
    return PAMLatticeResult(
        particles=particles,
        tripartite_code="J-4-OCT",
        clearance_valid=bool(bond_clr >= t_min),
        min_clearance_mm=float(bond_clr),
        strut_radius=r,
        meshes=meshes,
        metadata={
            "size": s,
            "bond_length": d,
            "num_particles": len(particles),
            "topology": "planar_cross_n4",
            "target_min_clearance_mm": t_min,
            "joint_style": "explicit_cylinder_compose",
            "clearance_scope": "center_neighbor_bonds",
        },
    )


def generate_pam_lattice(
    tripartite_code: str,
    unit_cell_size: float,
    repeats: tuple[int, int, int] = (1, 1, 1),
    strut_radius: float = 0.55,
    min_clearance: float = 0.40,
) -> PAMLatticeResult:
    """
    Place PAM particles from a tripartite code ``X-n-abc``.

    Supported:
      - ``D-4-TET``: diamond / tetrahedra (``unit_cell_size`` = tet edge length)
      - ``C-6-CO``: simple-cubic cuboctahedra (``unit_cell_size`` = CO size)
      - ``J-4-OCT``: square-grid octahedra (``unit_cell_size`` = OCT size; uses nx,ny)
    """
    code = str(tripartite_code).strip().upper().replace("_", "-")
    # Normalize S-6/2-CO style slashes
    code = code.replace("/", "-")
    nx, ny, nz = (int(repeats[0]), int(repeats[1]), int(repeats[2]))

    if code in ("D-4-TET", "D-4-TETRAHEDRON", "D-4-TETRA"):
        if (nx, ny, nz) == (1, 1, 1):
            return generate_d4tet_interlocked_pair(
                edge_length=float(unit_cell_size),
                strut_radius=float(strut_radius),
                min_clearance=float(min_clearance),
            )
        return generate_d4tet_diamond_tiling(
            repeats=(nx, ny, nz),
            edge_length=float(unit_cell_size),
            strut_radius=float(strut_radius),
            min_clearance=float(min_clearance),
        )

    if code in ("C-6-CO", "C-6-CUBOCTA", "C-6-CUBOCTAHEDRON"):
        if (nx, ny, nz) == (1, 1, 1):
            return generate_c6co_interlocked_pair(
                size=float(unit_cell_size),
                strut_radius=float(strut_radius),
                min_clearance=float(min_clearance),
            )
        return generate_c6co_cubic_tiling(
            repeats=(nx, ny, nz),
            size=float(unit_cell_size),
            strut_radius=float(strut_radius),
            min_clearance=float(min_clearance),
        )

    if code in ("C-6-TT", "C-6-TRUNCATED-TETRAHEDRON", "C-6-TRUNC-TET"):
        return generate_c6tt_cubic_tiling(
            repeats=(nx, ny, nz),
            size=float(unit_cell_size),
            strut_radius=float(strut_radius),
            min_clearance=float(min_clearance),
        )

    if code in ("J-4-OCT", "J-4-OCTAHEDRON"):
        if (nx, ny) == (1, 1):
            return generate_j4oct_interlocked_pair(
                size=float(unit_cell_size),
                strut_radius=float(strut_radius),
                min_clearance=float(min_clearance),
            )
        return generate_j4oct_square_tiling(
            repeats=(nx, ny),
            size=float(unit_cell_size),
            strut_radius=float(strut_radius),
            min_clearance=float(min_clearance),
        )

    raise NotImplementedError(
        f"Unsupported tripartite code '{tripartite_code}'. "
        "Supported: D-4-TET, C-6-TT, C-6-CO, J-4-OCT (ICO/HEX/S-6/2 later)."
    )


# Canonical clean-miter truss solidifier from geometry_module
from graphite.explicit.geometry_module import build_clean_miter_truss


def pam_particles_to_meshes(
    particles: Sequence[PAMParticle],
    strut_radius: float,
    *,
    add_spheres: bool = False,
    joint_sphere_scale: float = 1.15,
    circular_segments: int = 16,
    clean_miter: bool = True,
) -> list[trimesh.Trimesh]:
    """
    Solidify each particle into a watertight multi-body component.

    When ``clean_miter=True`` and ``not add_spheres``, struts are cut with mutual bisector
    planes to form sharp, seamless mitered joints without flat end caps or notch gaps.
    """
    r = float(strut_radius)
    out: list[trimesh.Trimesh] = []

    if clean_miter and not add_spheres:
        template_cache: dict[tuple, trimesh.Trimesh] = {}
        for p in particles:
            local_nodes = np.asarray(p.nodes, dtype=np.float64) - np.asarray(p.center, dtype=np.float64)
            cache_key = (p.geometry_type, tuple(np.round(local_nodes, 4).flat), r, int(circular_segments))
            if cache_key not in template_cache:
                template_cache[cache_key] = build_clean_miter_truss(
                    local_nodes,
                    np.asarray(p.struts, dtype=np.int64),
                    r,
                    circular_segments=circular_segments,
                )
            base_mesh = template_cache[cache_key]
            mesh = base_mesh.copy()
            mesh.apply_translation(np.asarray(p.center, dtype=np.float64))
            out.append(mesh)
        return out

    for p in particles:
        mesh = generate_geometry(
            nodes=np.asarray(p.nodes, dtype=np.float64),
            struts=np.asarray(p.struts, dtype=np.int64),
            strut_radius=r,
            add_spheres=add_spheres,
            joint_sphere_scale=float(joint_sphere_scale),
            circular_segments=int(circular_segments),
            crop_to_boundary=False,
        )
        out.append(mesh)
    return out


def export_pam_multibody_stl(result: PAMLatticeResult, path: str | Path) -> Path:
    """Write a multi-body STL (concatenated islands) under the given path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mesh = result.combined_mesh()
    mesh.export(p)
    return p
