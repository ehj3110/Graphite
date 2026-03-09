"""
Graphite Solver Module — Solid Fraction Optimization

This module solves for a strut radius that drives lattice solid fraction toward
a requested target using bisection search.

Performance upgrades:
    - Can solve from precomputed topology (nodes + struts) so GMSH scaffold is
      generated once and reused across variants.
    - Smart radius seeding from total strut length narrows the search range.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import trimesh

from geometry_module import generate_geometry
from scaffold_module import generate_conformal_scaffold
from topology_module import generate_topology


class SolverResult(NamedTuple):
    """
    Structured output from optimization.

    Attributes:
        mesh: Final generated lattice mesh.
        radius: Final (best) strut radius.
        volume: Final (best) trimmed lattice volume.
        iterations: Number of bisection iterations executed.
        nodes: Topology node coordinates used during solve.
        struts: Topology strut index pairs used during solve.
        seed_radius: Smart-seed center radius before bisection.
    """

    mesh: trimesh.Trimesh
    radius: float
    volume: float
    iterations: int
    nodes: np.ndarray
    struts: np.ndarray
    seed_radius: float


def _validate_inputs(
    mesh: trimesh.Trimesh,
    target_vf: float,
    tol: float,
    max_iter: int,
) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"`mesh` must be trimesh.Trimesh, got {type(mesh)}.")
    if target_vf <= 0 or target_vf >= 1:
        raise ValueError("`target_vf` must be between 0 and 1 (exclusive).")
    if tol <= 0:
        raise ValueError("`tol` must be > 0.")
    if max_iter <= 0:
        raise ValueError("`max_iter` must be > 0.")
    if mesh.volume <= 0:
        raise ValueError(
            f"Boundary mesh volume must be positive, got {float(mesh.volume)}."
        )


def _compute_smart_seed_radius(
    target_volume: float,
    nodes: np.ndarray,
    struts: np.ndarray,
) -> float:
    """
    Compute seed radius from target volume and total strut length:
        r_start = sqrt(target_volume / (pi * L_total))
    """
    if struts.shape[0] == 0:
        raise ValueError("Cannot solve with empty `struts`.")
    a = nodes[struts[:, 0]]
    b = nodes[struts[:, 1]]
    seg_lengths = np.linalg.norm(b - a, axis=1)
    l_total = float(np.sum(seg_lengths))
    if l_total <= 0:
        raise ValueError("Total strut length is non-positive; invalid topology.")
    return float(math.sqrt(target_volume / (math.pi * l_total)))


def _build_audit_box(
    mesh: trimesh.Trimesh,
    representative_volume_fraction: float = 0.40,
) -> trimesh.Trimesh:
    """
    Build axis-aligned audit box centered in the input mesh bounds.

    representative_volume_fraction is the target box-volume fraction of the full
    mesh bounds volume (e.g., 0.25 -> center 25% volume).
    """
    if representative_volume_fraction <= 0 or representative_volume_fraction >= 1:
        raise ValueError("`representative_volume_fraction` must be in (0, 1).")
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    mins, maxs = bounds[0], bounds[1]
    center = 0.5 * (mins + maxs)
    extents = maxs - mins
    scale = float(representative_volume_fraction) ** (1.0 / 3.0)
    audit_extents = extents * scale
    return trimesh.creation.box(extents=audit_extents, transform=trimesh.transformations.translation_matrix(center))


def _select_audit_struts(
    nodes: np.ndarray,
    struts: np.ndarray,
    audit_bounds: np.ndarray,
) -> np.ndarray:
    """Select struts with both endpoints inside audit bounds."""
    if struts.shape[0] == 0:
        return struts
    p0 = nodes[struts[:, 0]]
    p1 = nodes[struts[:, 1]]
    lo, hi = audit_bounds[0], audit_bounds[1]
    inside0 = np.all((p0 >= lo) & (p0 <= hi), axis=1)
    inside1 = np.all((p1 >= lo) & (p1 <= hi), axis=1)
    return struts[inside0 & inside1]


def optimize_lattice_fraction_from_topology(
    mesh: trimesh.Trimesh,
    target_vf: float,
    nodes: np.ndarray,
    struts: np.ndarray,
    r_min: float = 0.1,
    r_max: float = 2.0,
    tol: float = 0.05,
    max_iter: int = 15,
    representative_volume_check: bool = False,
    representative_volume_fraction: float = 0.40,
) -> SolverResult:
    """
    Optimize lattice solid fraction using a precomputed topology.

    This function avoids scaffold/topology regeneration and is used for
    multi-variant sweeps that share one scaffold.
    """
    _validate_inputs(mesh=mesh, target_vf=target_vf, tol=tol, max_iter=max_iter)

    nodes_np = np.asarray(nodes, dtype=np.float64)
    struts_np = np.asarray(struts, dtype=np.int64)
    if nodes_np.ndim != 2 or nodes_np.shape[1] != 3:
        raise ValueError(f"`nodes` must have shape (N, 3); got {nodes_np.shape}.")
    if struts_np.ndim != 2 or struts_np.shape[1] != 2:
        raise ValueError(f"`struts` must have shape (S, 2); got {struts_np.shape}.")
    if r_min <= 0 or r_max <= 0 or r_min >= r_max:
        raise ValueError("Require 0 < r_min < r_max.")

    if representative_volume_check:
        audit_box = _build_audit_box(
            mesh=mesh,
            representative_volume_fraction=representative_volume_fraction,
        )
        eval_struts = _select_audit_struts(
            nodes=nodes_np,
            struts=struts_np,
            audit_bounds=np.asarray(audit_box.bounds, dtype=np.float64),
        )
        if eval_struts.shape[0] == 0:
            warnings.warn(
                "Representative audit selection found zero struts; falling back to full topology.",
                RuntimeWarning,
                stacklevel=2,
            )
            eval_mesh = mesh
            eval_struts = struts_np
        else:
            eval_mesh = audit_box
    else:
        eval_mesh = mesh
        eval_struts = struts_np

    target_volume = float(eval_mesh.volume) * float(target_vf)

    # Smart seed + tightened bracket
    seed_radius = _compute_smart_seed_radius(target_volume, nodes_np, eval_struts)
    lower_seed = 0.8 * seed_radius
    upper_seed = 1.2 * seed_radius
    lower = max(float(r_min), float(lower_seed))
    upper = min(float(r_max), float(upper_seed))
    if lower >= upper:
        lower, upper = float(r_min), float(r_max)

    best_mesh: trimesh.Trimesh | None = None
    best_radius = lower
    best_volume = 0.0
    best_error = float("inf")
    iterations_used = 0

    # Early-stop threshold requested by user (1.0%)
    early_stop_error = 0.01

    for iteration in range(1, max_iter + 1):
        iterations_used = iteration
        r_mid = 0.5 * (lower + upper)

        candidate_mesh = generate_geometry(
            nodes=nodes_np,
            struts=eval_struts,
            strut_radius=r_mid,
            boundary_mesh=eval_mesh,
            add_spheres=False,
        )
        current_volume = float(candidate_mesh.volume)
        error = abs(current_volume - target_volume) / target_volume

        if error < best_error:
            best_error = error
            best_mesh = candidate_mesh
            best_radius = r_mid
            best_volume = current_volume

        if error <= tol or error < early_stop_error:
            break

        if current_volume < target_volume:
            lower = r_mid
        else:
            upper = r_mid

    if best_error > tol and best_error > early_stop_error:
        warnings.warn(
            "Bisection reached max_iter without meeting tolerance. "
            f"Best relative error={best_error:.6f}, tol={tol:.6f}. "
            "Returning best mesh found.",
            RuntimeWarning,
            stacklevel=2,
        )

    if best_mesh is None:
        raise RuntimeError("Solver failed: no candidate mesh was generated.")

    final_mesh = (
        generate_geometry(
            nodes=nodes_np,
            struts=struts_np,
            strut_radius=float(best_radius),
            boundary_mesh=mesh,
            add_spheres=False,
        )
        if representative_volume_check
        else best_mesh
    )
    final_volume = float(final_mesh.volume)

    return SolverResult(
        mesh=final_mesh,
        radius=float(best_radius),
        volume=final_volume,
        iterations=int(iterations_used),
        nodes=nodes_np,
        struts=struts_np,
        seed_radius=float(seed_radius),
    )


def optimize_lattice_fraction(
    mesh: trimesh.Trimesh,
    target_vf: float,
    target_element_size: float,
    topology_type: str = "rhombic",
    r_min: float = 0.1,
    r_max: float = 2.0,
    tol: float = 0.05,
    max_iter: int = 15,
    representative_volume_check: bool = False,
    representative_volume_fraction: float = 0.40,
    include_surface_cage: bool = True,
) -> SolverResult:
    """
    Optimize lattice solid fraction by solving for strut radius via bisection.

    This high-level helper generates scaffold + topology once, then delegates to
    optimize_lattice_fraction_from_topology().
    """
    if target_element_size <= 0:
        raise ValueError("`target_element_size` must be > 0.")

    scaffold = generate_conformal_scaffold(
        mesh=mesh,
        target_element_size=float(target_element_size),
    )
    nodes, struts = generate_topology(
        nodes=scaffold.nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type=topology_type,
        include_surface_cage=include_surface_cage,
    )

    return optimize_lattice_fraction_from_topology(
        mesh=mesh,
        target_vf=target_vf,
        nodes=nodes,
        struts=struts,
        r_min=r_min,
        r_max=r_max,
        tol=tol,
        max_iter=max_iter,
        representative_volume_check=representative_volume_check,
        representative_volume_fraction=representative_volume_fraction,
    )
