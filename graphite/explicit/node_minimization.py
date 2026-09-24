"""
Node Minimization — discrete SC grid phase search.

Shifts the background SC lattice by lattice-specific axial increments
(half the characteristic node spacing; octahedral → quarter cell) in
origin + ±X/±Y/±Z, scores each candidate by the fraction of tiered-trimmed
lattice nodes outside the CAD, and returns the best offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from graphite.explicit.mesh_repair import sanitize_cad_mesh_for_sdf
from graphite.explicit.proven_topologies import generate_background_grid
from graphite.explicit.sc_axis_cull_octahedral import (
    EXTENT_POLICY_TIERED_25_75,
    generate_extent_trimmed_octahedral,
)
from graphite.explicit.sc_boundary_states import build_edt_sdf_field, estimate_hex_volume_fractions

# Characteristic node spacing along a unit-cell axis, as a fraction of cell size.
# Increment = 0.5 * node_spacing_frac * cell_dims.
# Prefer deriving from ``sc_node_planes`` when possible; this table is a fallback.
NODE_SPACING_FRAC: dict[str, float] = {
    "octahedral": 0.5,  # face centers at 0 / 0.5 / 1 → spacing 0.5 → δ = 0.25 cell
    "tesseract": 0.25,  # planes 0 / 0.25 / 0.75 / 1 → min spacing 0.25 → δ = 0.125 cell
}


def _as_cell_dims(
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    cell_dims = np.asarray(cell_size, dtype=np.float64)
    if cell_dims.ndim == 0:
        cell_dims = np.full(3, float(cell_dims), dtype=np.float64)
    if cell_dims.shape != (3,) or np.any(cell_dims <= 0.0):
        raise ValueError("cell_size must be a positive scalar or three positive dimensions")
    return cell_dims


def _node_spacing_frac(lattice_rule: str) -> float:
    """Min consecutive node-plane gap on the unit cell (fraction of cell)."""
    rule = str(lattice_rule).strip().lower()
    try:
        from graphite.explicit.sc_node_planes import node_plane_policy

        pol = node_plane_policy(rule)
        gaps: list[float] = []
        for planes in pol.planes_xyz:
            if len(planes) >= 2:
                gaps.extend(float(g) for g in np.diff(planes) if float(g) > 1e-9)
        if gaps:
            return float(min(gaps))
    except Exception:
        pass
    if rule in NODE_SPACING_FRAC:
        return float(NODE_SPACING_FRAC[rule])
    raise ValueError(
        f"unknown lattice_rule {lattice_rule!r}; "
        f"known fallbacks: {sorted(NODE_SPACING_FRAC)}"
    )


def axial_increment(
    lattice_rule: str,
    cell_size: float | tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """
    Per-axis phase increment (mm) for Node Minimization.

    increment = 0.5 * (min node-plane spacing) * cell_dims
    Octahedral: spacing 0.5 → δ = 0.25 * cell_dims.
    Tesseract: spacing 0.25 → δ = 0.125 * cell_dims.
    """
    cell_dims = _as_cell_dims(cell_size)
    frac = _node_spacing_frac(lattice_rule)
    return 0.5 * frac * cell_dims


def axis6_candidate_offsets(increment: np.ndarray) -> list[np.ndarray]:
    """Origin + six axial shifts (±δx, ±δy, ±δz)."""
    d = np.asarray(increment, dtype=np.float64).reshape(3)
    out = [np.zeros(3, dtype=np.float64)]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            o = np.zeros(3, dtype=np.float64)
            o[axis] = sign * float(d[axis])
            out.append(o)
    return out


@dataclass
class CandidateScore:
    offset: np.ndarray
    outside_frac: float
    outside_count: int
    n_nodes: int
    n_hex_kept: int
    n_struts: int
    mean_outside_sdf: float = 0.0


@dataclass
class NodeMinimizationResult:
    best_offset: np.ndarray
    best_score: float
    candidates: list[CandidateScore]
    increment: np.ndarray
    lattice_rule: str
    hex_elems: np.ndarray
    lattice_nodes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    lattice_struts: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64)
    )
    report: dict = field(default_factory=dict)


def _hex_elems_at_offset(
    bounds: np.ndarray,
    cell_size: np.ndarray,
    origin_offset: np.ndarray,
) -> np.ndarray:
    grid_nodes, cells = generate_background_grid(
        "SC",
        bounds,
        cell_size,
        origin_offset=origin_offset,
    )
    return np.asarray(
        grid_nodes[np.asarray(cells, dtype=np.int64)],
        dtype=np.float64,
    )


def _score_key(c: CandidateScore) -> tuple:
    """Minimize outside_frac, then outside_count; maximize hex_kept; prefer small offset."""
    return (
        float(c.outside_frac),
        int(c.outside_count),
        -int(c.n_hex_kept),
        float(np.linalg.norm(c.offset)),
    )


def minimize_sc_grid_offset(
    cad_mesh: trimesh.Trimesh,
    cell_size: float | tuple[float, float, float] | np.ndarray,
    *,
    lattice_rule: str = "octahedral",
    extent_policy: str = EXTENT_POLICY_TIERED_25_75,
    empty_vf_max: float = 0.01,
    samples_per_axis: int = 8,
    extent_samples_per_axis: int = 10,
    search: str = "axis6",
    inside_eps: float = 0.0,
) -> NodeMinimizationResult:
    """
    Discrete SC phase search minimizing exterior trimmed-lattice nodes.

    ``search="axis6"`` evaluates origin + ±X/±Y/±Z (7 candidates).
    """
    if str(search).lower() not in ("axis6", "axis_6", "6"):
        raise ValueError(f"unsupported search mode {search!r}; use 'axis6'")

    cad = sanitize_cad_mesh_for_sdf(cad_mesh)
    if isinstance(cad, trimesh.Scene):
        cad = trimesh.util.concatenate(tuple(cad.geometry.values()))

    cell_dims = _as_cell_dims(cell_size)
    increment = axial_increment(lattice_rule, cell_dims)
    offsets = axis6_candidate_offsets(increment)
    bounds = np.asarray(cad.bounds, dtype=np.float64)

    # Shared SDF field for all candidates
    probe = _hex_elems_at_offset(bounds, cell_dims, offsets[0])
    vf0 = estimate_hex_volume_fractions(
        cad, probe, samples_per_axis=int(samples_per_axis)
    )
    res = float(vf0.voxel_resolution) if vf0.voxel_resolution > 0 else 0.5
    sdf_field = build_edt_sdf_field(cad, res)

    candidates: list[CandidateScore] = []
    best: CandidateScore | None = None
    best_hex: np.ndarray | None = None
    best_nodes: np.ndarray | None = None
    best_struts: np.ndarray | None = None

    for offset in offsets:
        hex_elems = _hex_elems_at_offset(bounds, cell_dims, offset)
        nodes, struts, cull = generate_extent_trimmed_octahedral(
            cad,
            hex_elems,
            empty_vf_max=float(empty_vf_max),
            samples_per_axis=int(samples_per_axis),
            extent_samples_per_axis=int(extent_samples_per_axis),
            extent_policy=str(extent_policy),
            sdf_field=sdf_field,
            prune_loose_external=False,
            inside_eps=float(inside_eps),
        )
        n = int(len(nodes))
        if n == 0:
            outside_count = 0
            outside_frac = 1.0  # empty lattice is worst
            mean_out = 0.0
        else:
            sdf = np.asarray(sdf_field.sample(nodes), dtype=np.float64)
            outside = sdf > float(inside_eps)
            outside_count = int(np.count_nonzero(outside))
            outside_frac = float(outside_count) / float(n)
            mean_out = (
                float(np.mean(sdf[outside])) if outside_count > 0 else 0.0
            )

        cand = CandidateScore(
            offset=np.asarray(offset, dtype=np.float64).copy(),
            outside_frac=outside_frac,
            outside_count=outside_count,
            n_nodes=n,
            n_hex_kept=int(cull.n_hex_kept),
            n_struts=int(len(struts)),
            mean_outside_sdf=mean_out,
        )
        candidates.append(cand)
        if best is None or _score_key(cand) < _score_key(best):
            best = cand
            best_hex = hex_elems
            best_nodes = nodes
            best_struts = struts

    assert best is not None and best_hex is not None
    assert best_nodes is not None and best_struts is not None

    report = {
        "search": "axis6",
        "n_candidates": len(candidates),
        "best_offset": best.offset.tolist(),
        "best_outside_frac": best.outside_frac,
        "best_outside_count": best.outside_count,
        "best_n_nodes": best.n_nodes,
        "best_n_hex_kept": best.n_hex_kept,
        "increment": increment.tolist(),
        "lattice_rule": str(lattice_rule),
        "extent_policy": str(extent_policy),
    }
    return NodeMinimizationResult(
        best_offset=best.offset.copy(),
        best_score=float(best.outside_frac),
        candidates=candidates,
        increment=increment.copy(),
        lattice_rule=str(lattice_rule),
        hex_elems=best_hex,
        lattice_nodes=best_nodes,
        lattice_struts=best_struts,
        report=report,
    )
