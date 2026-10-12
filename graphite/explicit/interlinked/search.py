"""
graphite/explicit/interlinked/search.py

Bounded Deterministic Grid Interface Sweep Engine (Phase 4).

Replaces legacy continuous numerical optimization loops (Powell / SciPy minimize)
with a deterministic, bounded coarse-to-fine parameter sweep over habit plane interface
degrees of freedom (interface gap / longitudinal offset, transverse shift, and azimuthal twist).

Key Design Principles:
1. Purely Deterministic: No random restarts, no solver divergence, no infinite iteration hangs.
2. Bounded Execution Time: Guaranteed completion within user budget (default < 1.0s).
3. Physics & Topology Aware: Evaluates physical surface clearance (segment-segment)
   and closed-loop topological catenation (strut piercings & Gauss linking).
4. Direct Assembly Integration: Can export an InterfaceSpec directly consumable by assemble().
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence, Any
import numpy as np

from .clearance import segment_segment_distance
from .particle import ParticleGeometry, InterlinkedParticle
from .cell import InterlinkedCell, InterlinkedRegistry
from .cells import CustomWireframeCell
from .pams import strut_pierces_triangle
from .epitaxy import (
    extract_triangulated_faces,
    extract_chordless_cycles,
    curve_curve_linking_number,
    get_habit_plane_rotation_matrix,
)


@dataclass(frozen=True)
class InterfaceCandidate:
    """
    A single evaluated interface candidate parameter set.

    Attributes:
        gap: Longitudinal spacing / offset along interface normal (mm).
        shift: 3D relative shift vector (dx, dy, dz) in mm.
        twist_deg: Relative azimuthal twist angle around normal in degrees.
        min_clearance: Evaluated physical surface clearance in mm.
        boundary_links: Number of catenated topological links across the interface.
        is_feasible: True if boundary_links >= 1 and min_clearance >= threshold.
        score: Unified candidate ranking score.
    """
    gap: float
    shift: np.ndarray
    twist_deg: float
    min_clearance: float
    boundary_links: int
    is_feasible: bool
    score: float


@dataclass(frozen=True)
class InterfaceSweepResult:
    """
    Result of a bounded interface parameter sweep.

    Attributes:
        best_candidate: The highest-ranking candidate, or None if none evaluated.
        candidates: Sorted list of top candidates.
        total_evaluated: Total number of parameter grid points evaluated.
        elapsed_s: Elapsed runtime in seconds.
        habit_plane: Crystallographic habit plane index.
    """
    best_candidate: InterfaceCandidate | None
    candidates: list[InterfaceCandidate] = field(default_factory=list)
    total_evaluated: int = 0
    elapsed_s: float = 0.0
    habit_plane: str = "100"

    @property
    def is_feasible(self) -> bool:
        """True if the sweep discovered at least one admissible interlinked candidate."""
        return self.best_candidate is not None and self.best_candidate.is_feasible

    def to_interface_spec(self) -> Any:
        """Convert the best candidate into an InterfaceSpec for assemble()."""
        from .assemble import InterfaceSpec
        if self.best_candidate is None:
            return InterfaceSpec(normal=self.habit_plane)
        return InterfaceSpec(
            normal=self.habit_plane,
            gap=self.best_candidate.gap,
            shift=self.best_candidate.shift,
            twist_deg=self.best_candidate.twist_deg,
        )


def _rotation_matrix_z(deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix around local Z axis by deg."""
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _resolve_geometry(cell_or_zone: Any, pitch: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract (nodes, struts, wire_radius, pitch) from a cell or ZoneSpec."""
    if hasattr(cell_or_zone, "cell"):  # ZoneSpec
        c_obj = cell_or_zone.cell
        p_val = float(cell_or_zone.pitch)
        w_r = float(cell_or_zone.wire_radius or 0.35)
    else:
        c_obj = cell_or_zone
        p_val = float(pitch)
        w_r = 0.35

    if isinstance(c_obj, str):
        c_cls = InterlinkedRegistry.get(c_obj)
        cell_inst: InterlinkedCell = c_cls() if callable(c_cls) else c_cls
    elif isinstance(c_obj, InterlinkedCell):
        cell_inst = c_obj
    else:
        raise TypeError(f"Cannot resolve geometry from {type(c_obj)}")

    parts = cell_inst.instantiate_site(
        grid_index=(0, 0, 0),
        site_origin=np.zeros(3, dtype=np.float64),
        cell_pitch=p_val,
        id_start=0,
    )
    if not parts:
        raise ValueError(f"Cell {cell_inst} produced zero particles")

    p = parts[0]
    return p.global_nodes(), p.geometry.struts, w_r, p_val


def _score_candidate(prc: int, clr: float, min_clearance: float) -> float:
    """
    Compute unified candidate score ensuring physically feasible candidates
    strictly dominate colliding or unlinked candidates.
    """
    is_feas = bool(prc >= 1 and clr >= min_clearance)
    if is_feas:
        return 10000.0 + 100.0 * prc + clr
    elif prc >= 1 and clr > 0.0:
        return 1000.0 + 10.0 * prc + clr
    elif prc >= 1:
        # Penetrated but colliding: penalize negative clearance
        return 100.0 * prc - 500.0 * abs(min(0.0, clr))
    else:
        # Unlinked / separate
        return -10000.0 + clr


def search_interface(
    zone_a: Any,
    zone_b: Any,
    pitch_a: float = 10.0,
    pitch_b: float = 10.0,
    habit_plane: str = "100",
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    gap_range: tuple[float, float] | None = None,
    gap_steps: int = 15,
    shift_range_transverse: float | None = None,
    shift_steps: int = 5,
    twist_angles: Sequence[float] | None = None,
    fine_refine: bool = True,
    top_k: int = 5,
    budget_s: float = 2.0,
) -> InterfaceSweepResult:
    """
    Execute a bounded deterministic coarse-to-fine parameter sweep for interface transition.

    Searches across:
        1. Interface gap / longitudinal offset along habit normal (mm).
        2. In-plane transverse shifts (dx, dy) in mm.
        3. Azimuthal twist angles (deg) about the habit normal.

    Args:
        zone_a: First zone specification or InterlinkedCell.
        zone_b: Second zone specification or InterlinkedCell.
        pitch_a: Unit cell pitch for zone A if not in ZoneSpec.
        pitch_b: Unit cell pitch for zone B if not in ZoneSpec.
        habit_plane: Crystallographic habit plane index (e.g. '100', '111').
        wire_radius: Fallback strut wire radius in mm.
        min_clearance: Target minimum physical surface clearance in mm.
        gap_range: Optional (gap_min, gap_max) search interval in mm.
        gap_steps: Number of sample steps along longitudinal offset axis.
        shift_range_transverse: Max transverse shift magnitude (default: 0.25 * pitch).
        shift_steps: Number of sample steps per transverse axis.
        twist_angles: Sequence of candidate twist angles in deg (default: [0, 30, 45, 60, 90]).
        fine_refine: If True, performs local grid refinement around the best candidate.
        top_k: Number of ranked candidate parameter sets to preserve.
        budget_s: Maximum allowable search runtime budget in seconds.

    Returns:
        InterfaceSweepResult containing the best candidate, ranked alternatives, and metrics.
    """
    t0 = time.perf_counter()

    # 1. Resolve geometry
    nodes_a, struts_a, r_a, p_a_eff = _resolve_geometry(zone_a, pitch_a)
    nodes_b_raw, struts_b, r_b, p_b_eff = _resolve_geometry(zone_b, pitch_b)
    eff_wire_r = max(r_a, r_b, wire_radius)

    # Align Zone B to habit plane orientation
    R_habit = get_habit_plane_rotation_matrix(habit_plane)
    nodes_b_aligned = (R_habit @ nodes_b_raw.T).T

    p0_a = nodes_a[struts_a[:, 0]]
    p1_a = nodes_a[struts_a[:, 1]]
    faces_a = extract_triangulated_faces(nodes_a, struts_a)
    cycles_a = extract_chordless_cycles(nodes_a, struts_a)

    faces_b_aligned = extract_triangulated_faces(nodes_b_aligned, struts_b)
    cycles_b_aligned = extract_chordless_cycles(nodes_b_aligned, struts_b)

    # 2. Determine Search Ranges
    p_avg = 0.5 * (float(p_a_eff) + float(p_b_eff))
    if gap_range is not None:
        g_min, g_max = gap_range
    else:
        g_min, g_max = 0.20 * p_avg, 0.85 * p_avg

    g_vals = np.linspace(g_min, g_max, max(3, gap_steps))

    s_trans = shift_range_transverse if shift_range_transverse is not None else (0.25 * min(p_a_eff, p_b_eff))
    if shift_steps > 1 and s_trans > 1e-4:
        sx_vals = np.linspace(-s_trans, s_trans, shift_steps)
        sy_vals = np.linspace(-s_trans, s_trans, shift_steps)
    else:
        sx_vals = np.array([0.0])
        sy_vals = np.array([0.0])

    if twist_angles is not None:
        twists = list(twist_angles)
    else:
        twists = [0.0, 30.0, 45.0, 60.0, 90.0]

    def _evaluate(gap: float, sx: float, sy: float, tw: float) -> tuple[float, int]:
        """Evaluate clearance and boundary catenation for a specific parameter combination."""
        R_tw = _rotation_matrix_z(tw)
        cur_nodes_b = (R_tw @ nodes_b_aligned.T).T
        shift_vec = np.array([sx, sy, gap], dtype=np.float64)
        cur_nodes_b = cur_nodes_b + shift_vec

        q0 = cur_nodes_b[struts_b[:, 0]]
        q1 = cur_nodes_b[struts_b[:, 1]]

        # Centerline segment distance & physical clearance
        d_center = segment_segment_distance(p0_a, p1_a, q0, q1)
        clr = float(d_center - 2.0 * eff_wire_r)

        # Topological catenation (piercings and cycle Gauss linking)
        prc_count = 0
        if len(faces_b_aligned) > 0:
            for i in range(len(p0_a)):
                for f in faces_b_aligned:
                    tri = cur_nodes_b[f]
                    if strut_pierces_triangle(p0_a[i], p1_a[i], tri):
                        prc_count += 1

        if len(faces_a) > 0:
            for j in range(len(q0)):
                for f in faces_a:
                    tri_a = nodes_a[f]
                    if strut_pierces_triangle(q0[j], q1[j], tri_a):
                        prc_count += 1

        # Fallback cycle Gauss linking if no planar triangular faces
        if prc_count == 0 and cycles_a and cycles_b_aligned:
            max_lk = 0.0
            for ca in cycles_a:
                sp0 = nodes_a[ca]
                sp1 = nodes_a[np.roll(ca, -1)]
                for cb in cycles_b_aligned:
                    sq0 = cur_nodes_b[cb]
                    sq1 = cur_nodes_b[np.roll(cb, -1)]
                    lk = abs(curve_curve_linking_number(sp0, sp1, sq0, sq1))
                    if lk > max_lk:
                        max_lk = lk
                    if max_lk >= 0.80:
                        break
                if max_lk >= 0.80:
                    break
            if max_lk >= 0.70:
                prc_count = int(round(max_lk))

        return clr, prc_count

    # 3. Coarse Grid Sweep
    candidates: list[InterfaceCandidate] = []
    eval_count = 0

    for tw in twists:
        for gap in g_vals:
            for sx in sx_vals:
                for sy in sy_vals:
                    eval_count += 1
                    clr, prc = _evaluate(gap, sx, sy, tw)
                    is_feas = bool(prc >= 1 and clr >= min_clearance)
                    score = _score_candidate(prc, clr, min_clearance)

                    cand = InterfaceCandidate(
                        gap=float(gap),
                        shift=np.array([sx, sy, 0.0], dtype=np.float64),
                        twist_deg=float(tw),
                        min_clearance=clr,
                        boundary_links=prc,
                        is_feasible=is_feas,
                        score=score,
                    )
                    candidates.append(cand)

                    # Check time budget
                    if (time.perf_counter() - t0) >= budget_s:
                        break
                if (time.perf_counter() - t0) >= budget_s:
                    break
            if (time.perf_counter() - t0) >= budget_s:
                break
        if (time.perf_counter() - t0) >= budget_s:
            break

    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0] if candidates else None

    # 4. Local Fine Grid Refinement
    if fine_refine and best is not None and (time.perf_counter() - t0) < budget_s:
        d_gap = (g_vals[1] - g_vals[0]) if len(g_vals) > 1 else 0.5
        d_sx = (sx_vals[1] - sx_vals[0]) if len(sx_vals) > 1 else 0.5
        d_sy = (sy_vals[1] - sy_vals[0]) if len(sy_vals) > 1 else 0.5

        fine_gaps = np.linspace(best.gap - 0.75 * d_gap, best.gap + 0.75 * d_gap, 7)
        fine_sx = np.linspace(best.shift[0] - 0.5 * d_sx, best.shift[0] + 0.5 * d_sx, 3)
        fine_sy = np.linspace(best.shift[1] - 0.5 * d_sy, best.shift[1] + 0.5 * d_sy, 3)
        fine_twists = [best.twist_deg]

        for tw in fine_twists:
            for gap in fine_gaps:
                for sx in fine_sx:
                    for sy in fine_sy:
                        eval_count += 1
                        clr, prc = _evaluate(gap, sx, sy, tw)
                        is_feas = bool(prc >= 1 and clr >= min_clearance)
                        score = _score_candidate(prc, clr, min_clearance)

                        cand = InterfaceCandidate(
                            gap=float(gap),
                            shift=np.array([sx, sy, 0.0], dtype=np.float64),
                            twist_deg=float(tw),
                            min_clearance=clr,
                            boundary_links=prc,
                            is_feasible=is_feas,
                            score=score,
                        )
                        candidates.append(cand)
                        if (time.perf_counter() - t0) >= budget_s:
                            break
                    if (time.perf_counter() - t0) >= budget_s:
                        break
                if (time.perf_counter() - t0) >= budget_s:
                    break
            if (time.perf_counter() - t0) >= budget_s:
                break

    candidates.sort(key=lambda c: c.score, reverse=True)
    best_cand = candidates[0] if candidates else None
    elapsed = time.perf_counter() - t0

    return InterfaceSweepResult(
        best_candidate=best_cand,
        candidates=candidates[:top_k],
        total_evaluated=eval_count,
        elapsed_s=elapsed,
        habit_plane=habit_plane,
    )
