"""
graphite/explicit/interlinked/legacy/solver.py

Archived Legacy Numerical Optimizer (Powell / SciPy minimize).
Moved from epitaxy.py in Phase 4.

DEPRECATION NOTICE:
Continuous numerical optimization via Powell iteration is non-deterministic,
prone to unbounded iteration times, and susceptible to local minima in disjoint
configuration spaces. Production code must use `graphite.explicit.interlinked.search_interface`
or `graphite.explicit.interlinked.assemble`.
"""

from __future__ import annotations

import warnings
from typing import Sequence, Any
import numpy as np

from ..clearance import segment_segment_distance
from ..particle import ParticleGeometry, InterlinkedParticle
from ..cell import InterlinkedCell
from ..pams import strut_pierces_triangle
from ..epitaxy import (
    InterfacePhaseOptimizationResult,
    extract_triangulated_faces,
    extract_chordless_cycles,
    curve_curve_linking_number,
    _get_strut_components,
)


def optimize_interface_phase_and_taper_powell(
    nodes_a: np.ndarray,
    struts_a: np.ndarray,
    nodes_b: np.ndarray,
    struts_b: np.ndarray,
    faces_b: np.ndarray | None = None,
    wire_radius: float = 0.35,
    min_clearance: float = 0.150,
    search_range: tuple[float, float] | None = None,
    pitch_a: float = 10.0,
    pitch_b: float = 10.0,
    enable_taper: bool = True,
    strut_taper_gamma: float = 0.70,
    cage_shrink_eta: float = 0.90,
    cell_a: InterlinkedCell | str | None = None,
    cell_b: InterlinkedCell | str | None = None,
    habit_plane: str = "100",
    rotation_b: np.ndarray | None = None,
    patch_size: tuple[int, int] = (2, 2),
) -> InterfacePhaseOptimizationResult:
    """
    Legacy 3D Interface Phase Shift & Powell Polish (Archived).
    """
    warnings.warn(
        "optimize_interface_phase_and_taper_powell is archived and deprecated. "
        "Use graphite.explicit.interlinked.search_interface for bounded deterministic parameter sweep.",
        DeprecationWarning,
        stacklevel=2,
    )

    p_avg = 0.5 * (float(pitch_a) + float(pitch_b))
    a_trans = min(float(pitch_a), float(pitch_b))
    s_range_x = search_range if search_range is not None else (0.15 * p_avg, 0.85 * p_avg)

    # 1. Setup Evaluation Geometry
    if cell_a is not None and cell_b is not None:
        from ..validator import instantiate_interface_patch
        parts_a_patch = instantiate_interface_patch(
            cell=cell_a,
            patch_size=patch_size,
            pitch=pitch_a,
            x_offset=0.0,
            transverse_offset=(0.0, 0.0),
            rotation_matrix=np.eye(3),
            zone_tag="Zone_A",
            habit_plane=habit_plane,
        )
        parts_b_patch = instantiate_interface_patch(
            cell=cell_b,
            patch_size=patch_size,
            pitch=pitch_b,
            x_offset=0.0,
            transverse_offset=(0.0, 0.0),
            rotation_matrix=rotation_b,
            zone_tag="Zone_B",
            habit_plane=habit_plane,
        )
        p0_list, p1_list = [], []
        for p in parts_a_patch:
            n_g = p.global_nodes()
            for u, v in p.geometry.struts:
                p0_list.append(n_g[u])
                p1_list.append(n_g[v])
        p0_eval = np.array(p0_list, dtype=np.float64)
        p1_eval = np.array(p1_list, dtype=np.float64)

        q0_list, q1_list = [], []
        for p in parts_b_patch:
            n_g = p.global_nodes()
            for u, v in p.geometry.struts:
                q0_list.append(n_g[u])
                q1_list.append(n_g[v])
        q0_eval = np.array(q0_list, dtype=np.float64)
        q1_eval = np.array(q1_list, dtype=np.float64)
    else:
        p0_eval = nodes_a[struts_a[:, 0]]
        p1_eval = nodes_a[struts_a[:, 1]]
        q0_eval = nodes_b[struts_b[:, 0]]
        q1_eval = nodes_b[struts_b[:, 1]]

    p0 = nodes_a[struts_a[:, 0]]
    p1 = nodes_a[struts_a[:, 1]]
    q0 = nodes_b[struts_b[:, 0]]
    q1 = nodes_b[struts_b[:, 1]]

    faces_a = extract_triangulated_faces(nodes_a, struts_a)
    if faces_b is None:
        faces_b = extract_triangulated_faces(nodes_b, struts_b)

    has_faces_b = len(faces_b) > 0
    has_faces_a = len(faces_a) > 0

    def _eval_phase(
        delta_vec: np.ndarray,
        eff_wire_r: float,
        cur_q0: np.ndarray,
        cur_q1: np.ndarray,
        cur_nodes_b: np.ndarray,
        cur_faces_b: np.ndarray,
        cur_q0_eval: np.ndarray,
        cur_q1_eval: np.ndarray,
    ) -> tuple[float, int]:
        d_arr = np.asarray(delta_vec, dtype=np.float64)
        shifted_q0_eval = cur_q0_eval + d_arr
        shifted_q1_eval = cur_q1_eval + d_arr
        d_min = segment_segment_distance(p0_eval, p1_eval, shifted_q0_eval, shifted_q1_eval)
        clr = d_min - 2.0 * eff_wire_r

        shifted_nodes_a = nodes_a - d_arr
        shifted_p0 = p0 - d_arr
        shifted_p1 = p1 - d_arr

        prc_a_to_b = 0
        if len(cur_faces_b) > 0:
            for i in range(len(shifted_p0)):
                for f in cur_faces_b:
                    tri = cur_nodes_b[f]
                    if strut_pierces_triangle(shifted_p0[i], shifted_p1[i], tri):
                        prc_a_to_b += 1

        prc_b_to_a = 0
        if has_faces_a:
            for j in range(len(cur_q0)):
                for f in faces_a:
                    tri_a = shifted_nodes_a[f]
                    if strut_pierces_triangle(cur_q0[j], cur_q1[j], tri_a):
                        prc_b_to_a += 1

        max_lk = 0.0
        cycles_a = extract_chordless_cycles(nodes_a, struts_a)
        cycles_b = extract_chordless_cycles(nodes_b, struts_b)
        for ca in cycles_a:
            sp0 = shifted_nodes_a[ca]
            sp1 = shifted_nodes_a[np.roll(ca, -1)]
            for cb in cycles_b:
                sq0 = cur_nodes_b[cb]
                sq1 = cur_nodes_b[np.roll(cb, -1)]
                lk = abs(curve_curve_linking_number(sp0, sp1, sq0, sq1))
                if lk > max_lk:
                    max_lk = lk
                if max_lk >= 0.75:
                    break
            if max_lk >= 0.75:
                break

        is_catenated = False
        if len(cur_faces_b) > 0 and has_faces_a:
            is_catenated = (prc_a_to_b > 0 and prc_b_to_a > 0) or (max_lk >= 0.75)
        elif len(cur_faces_b) > 0:
            is_catenated = (prc_a_to_b > 0) or (max_lk >= 0.75)
        elif has_faces_a:
            is_catenated = (prc_b_to_a > 0) or (max_lk >= 0.75)
        else:
            is_catenated = (max_lk >= 0.75)

        piercings = (prc_a_to_b + prc_b_to_a) if is_catenated else (1 if max_lk >= 0.75 else 0)
        return float(clr), piercings

    cur_q0 = q0
    cur_q1 = q1
    cur_nodes_b = nodes_b
    cur_faces_b = faces_b
    cur_q0_eval = q0_eval
    cur_q1_eval = q1_eval
    eff_wire_r = float(wire_radius)

    offsets_x = np.linspace(s_range_x[0], s_range_x[1], 7)
    offsets_y = np.linspace(-a_trans / 2.0, a_trans / 2.0, 7)
    dz_vals = np.linspace(-a_trans / 2.0, a_trans / 2.0, 7)

    best_delta = np.array([0.5 * (s_range_x[0] + s_range_x[1]), 0.0, 0.0], dtype=np.float64)
    best_clr = -float("inf")
    best_piercings = 0

    for dx in offsets_x:
        for dy in offsets_y:
            for dz in dz_vals:
                d_vec = np.array([dx, dy, dz], dtype=np.float64)
                clr, prc = _eval_phase(
                    d_vec, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
                )
                if prc >= 1:
                    if best_piercings == 0 or clr > best_clr:
                        best_clr = clr
                        best_delta = d_vec
                        best_piercings = prc

    # Fine Bounded Polish via Powell
    try:
        from scipy.optimize import minimize

        def loss(d_arr):
            clr, prc = _eval_phase(
                d_arr, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
            )
            penalty = 0.0 if prc >= best_piercings else 10.0 * (best_piercings - prc)
            return -clr + penalty

        bounds = [
            (s_range_x[0] * 0.85, s_range_x[1] * 1.15),
            (-a_trans * 0.6, a_trans * 0.6),
            (-a_trans * 0.6, a_trans * 0.6),
        ]
        res = minimize(
            loss,
            best_delta,
            method="Powell",
            bounds=bounds,
            options={"maxiter": 10, "ftol": 1e-3},
        )
        if res.success or res.fun < float("inf"):
            clr_pol, prc_pol = _eval_phase(
                res.x, eff_wire_r, cur_q0, cur_q1, cur_nodes_b, cur_faces_b, cur_q0_eval, cur_q1_eval
            )
            if clr_pol > best_clr and prc_pol >= best_piercings:
                best_clr = clr_pol
                best_delta = np.asarray(res.x, dtype=np.float64)
                best_piercings = prc_pol
    except Exception:
        pass

    return InterfacePhaseOptimizationResult(
        phase_shift=tuple(best_delta),
        max_clearance_mm=best_clr,
        piercings_count=best_piercings,
        is_valid=bool(best_piercings >= 1 and best_clr >= min_clearance),
        taper_applied=False,
        taper_gamma=1.0,
        shrink_eta=1.0,
    )
