"""Volume graph clip against CAD — cut / hanging nodes for V2 / V2.1 stitch.

Pragmatic approach (documented in report):
  - Classify volume nodes with ``safe_signed_distance`` (positive = outside).
  - Fully-inside struts kept as-is.
  - Crossing struts: find the **chord** surface crossing (raycast, SDF bisect
    fallback), keep the cut on the strut — never closest-point snap off-chord.
  - Optional ``cut_inset_factor`` pulls the cut toward the interior endpoint
    (legacy stitch helper; default 0 keeps cuts on the skin).
  - Fully-outside struts dropped.

Every clipped strut yields exactly one cut node linked to its interior parent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.conformal_core import safe_signed_distance


@dataclass
class ClipReport:
    n_nodes_in: int = 0
    n_struts_in: int = 0
    n_struts_kept_interior: int = 0
    n_struts_clipped: int = 0
    n_struts_dropped_outside: int = 0
    n_cut_nodes: int = 0
    cut_inset_factor: float = 0.0
    method: str = "chord_raycast_sdf_bisect"
    n_crossings_raycast: int = 0
    n_crossings_bisect: int = 0
    # Parallel arrays (length = n_cut_nodes), remapped into output node indexing.
    cut_interior_ids: list[int] = field(default_factory=list)
    cut_directions: list[list[float]] = field(default_factory=list)
    # Outside endpoint of the clipped strut (Cartesian, before drop) — identity stitch.
    cut_outside_points: list[list[float]] = field(default_factory=list)
    mean_cut_sdf: float = 0.0  # ~0 with inset=0; <0 if inset pulls inside


def _bisect_crossing_on_chord(
    p_in: np.ndarray,
    p_out: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    iters: int = 32,
) -> np.ndarray:
    """SDF zero-crossing on segment p_in (inside) → p_out (outside); stays on chord."""
    a = np.asarray(p_in, dtype=np.float64).copy()
    b = np.asarray(p_out, dtype=np.float64).copy()
    for _ in range(iters):
        mid = 0.5 * (a + b)
        sd = float(safe_signed_distance(cad_mesh, mid.reshape(1, 3))[0])
        if sd > 0.0:
            b = mid
        else:
            a = mid
    return 0.5 * (a + b)


def _raycast_crossing_on_chord(
    p_in: np.ndarray,
    p_out: np.ndarray,
    cad_mesh: trimesh.Trimesh,
) -> np.ndarray | None:
    """Nearest mesh hit along the strut chord from interior toward exterior."""
    p_in = np.asarray(p_in, dtype=np.float64)
    p_out = np.asarray(p_out, dtype=np.float64)
    seg = p_out - p_in
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-12:
        return None
    unit = seg / seg_len
    # Nudge origin slightly along the chord so we do not miss a hit at p_in.
    origin = (p_in + 1e-6 * unit).reshape(1, 3)
    try:
        locations, _index_ray, _tri = cad_mesh.ray.intersects_location(
            ray_origins=origin,
            ray_directions=unit.reshape(1, 3),
            multiple_hits=True,
        )
    except Exception:
        return None
    if locations is None or len(locations) == 0:
        return None

    best: tuple[float, np.ndarray] | None = None
    for loc in locations:
        # Parameter along chord from p_in; keep hits on the open segment.
        t = float(np.dot(loc - p_in, unit))
        if t < 1e-9 or t > seg_len + 1e-6:
            continue
        # Prefer the first exit from interior (smallest t).
        if best is None or t < best[0]:
            # Clamp onto the finite segment in case of tiny numerical overshoot.
            t_clamped = min(max(t, 0.0), seg_len)
            best = (t, p_in + t_clamped * unit)
    return None if best is None else best[1]


def chord_surface_crossing(
    p_in: np.ndarray,
    p_out: np.ndarray,
    cad_mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, str]:
    """
    Surface crossing on the strut chord.

    Prefer triangle raycast (exact mesh hit on the chord). Fall back to SDF
    bisection still constrained to the chord — never closest-point snap, which
    can jump to a neighboring patch and shorten / misplace the stub.
    """
    hit = _raycast_crossing_on_chord(p_in, p_out, cad_mesh)
    if hit is not None:
        return hit, "raycast"
    return _bisect_crossing_on_chord(p_in, p_out, cad_mesh), "bisect"


def clip_volume_graph_to_cad(
    nodes: np.ndarray,
    struts: np.ndarray,
    cad_mesh: trimesh.Trimesh,
    *,
    outside_eps: float = 1e-3,
    round_decimals: int = 6,
    cut_inset_factor: float = 0.0,
    min_stub_length: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ClipReport]:
    """
    Clip volume lattice graph to CAD interior; return cut node indices.

    Cuts land on the chord surface crossing by default. ``cut_inset_factor``
    optionally pulls each cut from the surface toward the interior endpoint by
    that fraction of the remaining stub length (legacy stitch helper).

    Returns
    -------
    out_nodes : (V', 3)
    out_struts : (E', 2)
    cut_node_ids : (C,) int indices into ``out_nodes``
    report : ClipReport
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    struts = np.asarray(struts, dtype=np.int64)
    inset = float(np.clip(cut_inset_factor, 0.0, 0.95))
    report = ClipReport(
        n_nodes_in=len(nodes),
        n_struts_in=len(struts),
        cut_inset_factor=inset,
        method=f"chord_raycast_sdf_bisect_inset_{inset:g}",
    )

    if len(nodes) == 0:
        return (
            nodes.reshape(0, 3),
            np.empty((0, 2), dtype=np.int64),
            np.empty(0, dtype=np.int64),
            report,
        )

    sd = safe_signed_distance(cad_mesh, nodes)
    inside = sd <= float(outside_eps)

    node_list = [nodes[i].copy() for i in range(len(nodes))]
    key_to_idx: dict[tuple[float, float, float], int] = {
        tuple(np.round(nodes[i], round_decimals).tolist()): i for i in range(len(nodes))
    }
    # cut_id -> (interior_id, direction unit from cut toward interior, outside point)
    cut_meta: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}

    def get_or_add(pt: np.ndarray) -> int:
        key = tuple(np.round(pt, round_decimals).tolist())
        if key in key_to_idx:
            return key_to_idx[key]
        idx = len(node_list)
        node_list.append(np.asarray(pt, dtype=np.float64).copy())
        key_to_idx[key] = idx
        return idx

    strut_set: set[tuple[int, int]] = set()

    for a, b in struts:
        ai, bi = int(a), int(b)
        a_in, b_in = bool(inside[ai]), bool(inside[bi])
        if a_in and b_in:
            strut_set.add((min(ai, bi), max(ai, bi)))
            report.n_struts_kept_interior += 1
            continue
        if (not a_in) and (not b_in):
            report.n_struts_dropped_outside += 1
            continue

        if a_in and not b_in:
            p_in, p_out, i_in = nodes[ai], nodes[bi], ai
        else:
            p_in, p_out, i_in = nodes[bi], nodes[ai], bi

        cut_pt, how = chord_surface_crossing(p_in, p_out, cad_mesh)
        if how == "raycast":
            report.n_crossings_raycast += 1
        else:
            report.n_crossings_bisect += 1

        to_in = np.asarray(p_in, dtype=np.float64) - cut_pt
        nrm = float(np.linalg.norm(to_in))
        if inset > 0.0 and nrm > 1e-12:
            cut_pt = cut_pt + inset * to_in
            # Enforce a minimum stub so inset cannot land on the interior node.
            stub = float(np.linalg.norm(np.asarray(p_in, dtype=np.float64) - cut_pt))
            if stub < float(min_stub_length) and nrm > float(min_stub_length):
                cut_pt = np.asarray(p_in, dtype=np.float64) - (
                    float(min_stub_length) * to_in / nrm
                )
        direction = to_in / nrm if nrm > 1e-12 else np.zeros(3, dtype=np.float64)

        cut_i = get_or_add(cut_pt)
        cut_meta[cut_i] = (
            int(i_in),
            direction.astype(np.float64, copy=False),
            np.asarray(p_out, dtype=np.float64).copy(),
        )
        if cut_i != i_in:
            strut_set.add((min(i_in, cut_i), max(i_in, cut_i)))
            report.n_struts_clipped += 1
        else:
            # Degenerate: cut collapsed onto interior — still count as clipped attempt.
            report.n_struts_clipped += 1

    out_nodes = np.asarray(node_list, dtype=np.float64)
    used: set[int] = set()
    for a, b in strut_set:
        used.add(int(a))
        used.add(int(b))
    used |= set(cut_meta.keys())

    keep = sorted(used)
    old_to_new = {old: new for new, old in enumerate(keep)}
    remapped_nodes = out_nodes[keep]
    remapped_struts = [
        (old_to_new[a], old_to_new[b])
        for a, b in sorted(strut_set)
        if a in old_to_new and b in old_to_new
    ]
    remapped_cuts: list[int] = []
    remapped_interiors: list[int] = []
    remapped_dirs: list[list[float]] = []
    remapped_outside: list[list[float]] = []
    for old_cut in sorted(cut_meta.keys()):
        if old_cut not in old_to_new:
            continue
        interior_old, direction, p_out = cut_meta[old_cut]
        if interior_old not in old_to_new:
            continue
        remapped_cuts.append(old_to_new[old_cut])
        remapped_interiors.append(old_to_new[interior_old])
        remapped_dirs.append([float(x) for x in direction])
        remapped_outside.append([float(x) for x in p_out])

    remapped_cut_arr = np.asarray(remapped_cuts, dtype=np.int64)
    report.n_cut_nodes = int(len(remapped_cut_arr))
    report.cut_interior_ids = remapped_interiors
    report.cut_directions = remapped_dirs
    report.cut_outside_points = remapped_outside
    if remapped_cut_arr.size:
        cut_sd = safe_signed_distance(cad_mesh, remapped_nodes[remapped_cut_arr])
        report.mean_cut_sdf = float(np.mean(cut_sd))
    strut_arr = (
        np.asarray(remapped_struts, dtype=np.int64)
        if remapped_struts
        else np.empty((0, 2), dtype=np.int64)
    )
    return remapped_nodes, strut_arr, remapped_cut_arr, report
