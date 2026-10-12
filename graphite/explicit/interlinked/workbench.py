"""
graphite/explicit/interlinked/workbench.py

Permissive PAM Pair Experimental Workbench.
Philosophy: "Always Generate, Always Diagnose".

Provides lightweight, deterministic single-pair and sweep diagnostics for arbitrary
cell types, relative rotations, translations, and wire diameters without any
numerical optimization solver loops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence
import numpy as np
import trimesh

try:
    import manifold3d
except ImportError:
    manifold3d = None

from .particle import InterlinkedParticle, ParticleGeometry
from .cell import InterlinkedCell, InterlinkedRegistry
from .clearance import segment_segment_distance
from .validator import face_cycle_linking
from ..geometry_module import build_clean_miter_truss, _trimesh_to_manifold


@dataclass
class PairReport:
    """
    Diagnostic measurement report for a single candidate particle pair.

    Attributes:
        cell_a_name: Name of cell A.
        cell_b_name: Name of cell B.
        min_clearance_mm: Minimum surface-to-surface Euclidean distance (mm).
            Negative if solids overlap.
        clash_volume_mm3: Solid intersection volume (mm^3) if colliding, else 0.0.
        linking_number: Integer discrete Gauss linking number |Lk| across face cycles.
        status: Physical & topological classification:
            - 'colliding': Meshes intersect (clearance < 0, clash_volume > 0).
            - 'linked': Non-intersecting and topologically catenated (|Lk| >= 1).
            - 'entangled': Interior envelopes overlap but no closed cycle catenation (|Lk| == 0).
            - 'disjoint': No envelope penetration and clearance > 0.
        mesh_a: Optional trimesh of Particle A.
        mesh_b: Optional trimesh of Particle B.
        diagnostics: Additional timings and coordinate frames.
    """
    cell_a_name: str
    cell_b_name: str
    min_clearance_mm: float
    clash_volume_mm3: float
    linking_number: int
    status: Literal["colliding", "linked", "entangled", "disjoint"]
    mesh_a: trimesh.Trimesh | None = None
    mesh_b: trimesh.Trimesh | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _resolve_particle(
    target: InterlinkedCell | str | InterlinkedParticle,
    pitch: float = 10.0,
    sublattice: str = "A",
    particle_id: int = 0,
) -> InterlinkedParticle:
    """Ensure input is converted into an instantiated InterlinkedParticle."""
    if isinstance(target, InterlinkedParticle):
        return target
    if isinstance(target, str):
        cell_cls = InterlinkedRegistry.get(target)
        cell = cell_cls() if callable(cell_cls) else cell_cls
    else:
        cell = target

    parts = cell.instantiate_site(
        (0, 0, 0),
        np.zeros(3, dtype=np.float64),
        pitch,
        id_start=particle_id,
        context={"sublattice": sublattice},
    )
    return parts[0]


def measure_pair(
    cell_a: InterlinkedCell | str | InterlinkedParticle,
    cell_b: InterlinkedCell | str | InterlinkedParticle,
    R_rel: np.ndarray | None = None,
    t_rel: np.ndarray | None = None,
    wire_radius: float = 0.35,
    pitch_a: float = 10.0,
    pitch_b: float | None = None,
    include_meshes: bool = False,
) -> PairReport:
    """
    Test clearance, solid clash volume, and topological linking for two particles.

    Args:
        cell_a: First cell, cell tag, or particle.
        cell_b: Second cell, cell tag, or particle.
        R_rel: (3, 3) relative rotation matrix of B wrt A (default Identity).
        t_rel: (3,) relative translation vector of B wrt A (default zeros).
        wire_radius: Wire radius for both struts (mm).
        pitch_a: Pitch of cell A (mm).
        pitch_b: Pitch of cell B (mm, default pitch_a).
        include_meshes: If True, attaches Trimesh solids for both particles.

    Returns:
        PairReport with clearance, clash volume, Lk, and status.
    """
    t0 = time.perf_counter()
    p_b_pitch = float(pitch_b) if pitch_b is not None else float(pitch_a)
    pa = _resolve_particle(cell_a, pitch=pitch_a, sublattice="A", particle_id=0)
    pb = _resolve_particle(cell_b, pitch=p_b_pitch, sublattice="B", particle_id=1)

    name_a = pa.metadata.get("cell_name", getattr(cell_a, "name", str(cell_a)))
    name_b = pb.metadata.get("cell_name", getattr(cell_b, "name", str(cell_b)))

    # Apply relative transformation to B
    R_b = np.eye(3, dtype=np.float64) if R_rel is None else np.asarray(R_rel, dtype=np.float64)
    t_b = np.zeros(3, dtype=np.float64) if t_rel is None else np.asarray(t_rel, dtype=np.float64)

    T_b = np.eye(4, dtype=np.float64)
    T_b[:3, :3] = R_b
    T_b[:3, 3] = t_b
    pb.transform = T_b

    # 1. Segment-segment surface clearance
    nodes_a = pa.global_nodes()
    nodes_b = pb.global_nodes()
    struts_a = pa.struts
    struts_b = pb.struts

    p0A = nodes_a[struts_a[:, 0]]
    p1A = nodes_a[struts_a[:, 1]]
    p0B = nodes_b[struts_b[:, 0]]
    p1B = nodes_b[struts_b[:, 1]]

    d_centerline = segment_segment_distance(p0A, p1A, p0B, p1B)
    min_clearance = float(d_centerline - 2.0 * wire_radius)

    # 2. Solid Clash Volume Check
    clash_volume = 0.0
    mesh_a = None
    mesh_b = None

    needs_mesh = include_meshes or (min_clearance < 0.05 and manifold3d is not None)
    if needs_mesh:
        mesh_a = build_clean_miter_truss(nodes_a, struts_a, strut_radius=wire_radius)
        mesh_b = build_clean_miter_truss(nodes_b, struts_b, strut_radius=wire_radius)

        if min_clearance <= 0.0 and manifold3d is not None:
            m_a = _trimesh_to_manifold(mesh_a)
            m_b = _trimesh_to_manifold(mesh_b)
            inter = m_a ^ m_b
            clash_volume = float(inter.volume())

    # 3. Exact Discrete Gauss Linking Number across closed cycles
    linking_number = face_cycle_linking(pa, pb, wire_radius=wire_radius)

    # 4. Status determination
    if clash_volume > 1e-4 or min_clearance < -1e-4:
        status = "colliding"
    elif linking_number >= 1:
        status = "linked"
    else:
        # Check envelope penetration
        hull_a = trimesh.convex.convex_hull(nodes_a)
        hull_b = trimesh.convex.convex_hull(nodes_b)
        if manifold3d is not None:
            env_a = _trimesh_to_manifold(hull_a)
            env_b = _trimesh_to_manifold(hull_b)
            env_inter = env_a ^ env_b
            if float(env_inter.volume()) > 1e-4:
                status = "entangled"
            else:
                status = "disjoint"
        else:
            status = "disjoint"

    elapsed = time.perf_counter() - t0

    return PairReport(
        cell_a_name=str(name_a),
        cell_b_name=str(name_b),
        min_clearance_mm=min_clearance,
        clash_volume_mm3=clash_volume,
        linking_number=linking_number,
        status=status,
        mesh_a=mesh_a if include_meshes else None,
        mesh_b=mesh_b if include_meshes else None,
        diagnostics={
            "elapsed_s": elapsed,
            "R_rel": R_b.tolist(),
            "t_rel": t_b.tolist(),
            "wire_radius": wire_radius,
            "pitch_a": pitch_a,
            "pitch_b": p_b_pitch,
        },
    )


def sweep(
    cell_a: InterlinkedCell | str,
    cell_b: InterlinkedCell | str,
    param: Literal["gap", "twist", "shift_y", "shift_z", "wire_radius"],
    values: Sequence[float],
    base_R_rel: np.ndarray | None = None,
    base_t_rel: np.ndarray | None = None,
    habit_normal: np.ndarray | None = None,
    wire_radius: float = 0.35,
    pitch_a: float = 10.0,
    pitch_b: float | None = None,
) -> list[PairReport]:
    """
    Deterministic parameter sweep over any single geometric variable.
    No numerical optimizer, pure empirical measurement.

    Args:
        cell_a: Cell A instance or tag.
        cell_b: Cell B instance or tag.
        param: Parameter to sweep ('gap', 'twist', 'shift_y', 'shift_z', 'wire_radius').
        values: Sequence of numerical values to test.
        base_R_rel: Base (3, 3) relative rotation.
        base_t_rel: Base (3,) translation offset.
        habit_normal: Unit direction along which 'gap' translates (default [1, 0, 0]).
        wire_radius: Default wire radius (mm).
        pitch_a: Unit cell pitch of Cell A (mm).
        pitch_b: Unit cell pitch of Cell B (mm).

    Returns:
        List of PairReport records across the sweep.
    """
    R_base = np.eye(3, dtype=np.float64) if base_R_rel is None else np.asarray(base_R_rel, dtype=np.float64)
    t_base = np.zeros(3, dtype=np.float64) if base_t_rel is None else np.asarray(base_t_rel, dtype=np.float64)
    n_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64) if habit_normal is None else np.asarray(habit_normal, dtype=np.float64)
    n_dir = n_dir / np.linalg.norm(n_dir)

    results: list[PairReport] = []

    for val in values:
        v = float(val)
        R_cur = R_base.copy()
        t_cur = t_base.copy()
        r_w = float(wire_radius)

        if param == "gap":
            t_cur = t_base + n_dir * v
        elif param == "twist":
            # Rotate about normal axis
            theta = np.radians(v)
            K = np.array([
                [0, -n_dir[2], n_dir[1]],
                [n_dir[2], 0, -n_dir[0]],
                [-n_dir[1], n_dir[0], 0],
            ])
            R_twist = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
            R_cur = R_twist @ R_base
        elif param == "shift_y":
            t_cur[1] = v
        elif param == "shift_z":
            t_cur[2] = v
        elif param == "wire_radius":
            r_w = v

        rep = measure_pair(
            cell_a=cell_a,
            cell_b=cell_b,
            R_rel=R_cur,
            t_rel=t_cur,
            wire_radius=r_w,
            pitch_a=pitch_a,
            pitch_b=pitch_b,
            include_meshes=False,
        )
        rep.diagnostics["sweep_param"] = param
        rep.diagnostics["sweep_value"] = v
        results.append(rep)

    return results


# Public API alias
test_pair = measure_pair
