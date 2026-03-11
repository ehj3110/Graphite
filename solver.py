"""
Graphite Solver Module — Solid Fraction Optimization

One-shot analytical estimator: overlapping cylinder approximation with at most
1–2 Boolean operations. Replaces the previous bisection search.

Adaptive strut thickness (Balanced Scaling):
- Per-strut radius: r_i = sqrt(L_i) * k, where k = radius-to-length ratio.
- V_est = sum(pi * (sqrt(L_i)*k)^2 * L_i) = pi * k^2 * sum(L_i^2).
- k = sqrt(target_volume / (pi * sum(L^2))), capped at K_MAX = 0.4.
- Volume scales with L^2 (not L^3), preventing large struts from hogging Vf.

Performance (Day 2):
- return_manifold=True for first pass: skip to_trimesh (~40s saved).
- Use manifold.volume() directly; convert only at final export.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import trimesh

from geometry_module import generate_geometry, manifold_to_trimesh
from scaffold_module import generate_conformal_scaffold
from topology_module import generate_topology

# Overlap correction constant for Kagome/Voronoi (cylinder junctions)
_OVERLAP_C = 2.5
# Max radius-to-length ratio; higher means target Vf may be too high for mesh density
K_MAX = 0.4
# Node overlap factor: junctions reduce effective volume. Tune for target Vf accuracy.
# 0.85 = theoretical Kagome; 0.70–0.78 often needed for clipped parts.
OVERLAP_FACTOR_KAGOME = 0.72
OVERLAP_FACTOR_VORONOI = 0.72
OVERLAP_FACTOR_DEFAULT = 0.85


def _strut_lengths(nodes: np.ndarray, struts: np.ndarray) -> np.ndarray:
    """Return length of each strut: (S,) array."""
    a = nodes[struts[:, 0]]
    b = nodes[struts[:, 1]]
    return np.linalg.norm(b - a, axis=1)


def calculate_k_one_shot(
    target_vf: float,
    nodes: np.ndarray,
    struts: np.ndarray,
    total_volume: float,
) -> float:
    """
    Analytical radius-to-length ratio (k) for balanced strut thickness.

    r_i = sqrt(L_i) * k  =>  V_est = sum(pi * (sqrt(L_i)*k)^2 * L_i) = pi * k^2 * sum(L_i^2)
    k = sqrt(target_volume / (pi * sum(L_i^2))), capped at K_MAX (0.4).
    """
    if struts.shape[0] == 0:
        raise ValueError("Cannot compute k with empty `struts`.")
    lengths = _strut_lengths(nodes, struts)
    sum_L2 = float(np.sum(lengths**2))
    if sum_L2 <= 0:
        raise ValueError("Total L^2 is non-positive.")
    target_volume = total_volume * float(target_vf)
    k0 = math.sqrt(target_volume / (math.pi * sum_L2))
    k = min(float(k0), K_MAX)
    k = max(k, 1e-6)
    return k


def calculate_k_analytical(
    target_vf: float,
    nodes: np.ndarray,
    struts: np.ndarray,
    total_volume: float,
    overlap_factor: float | None = None,
    topology_type: str = "kagome",
) -> float:
    """
    One-shot analytical k with node overlap factor. Balanced scaling: r_i = sqrt(L_i)*k.

    V_est = sum(pi * (sqrt(L_i)*k)^2 * L_i) = pi * k^2 * sum(L^2)
    Junctions reduce effective volume: V_actual ≈ V_est * overlap_factor.
    => k = sqrt(target_volume / (overlap_factor * pi * sum(L^2)))
    """
    if struts.shape[0] == 0:
        raise ValueError("Cannot compute k with empty `struts`.")
    lengths = _strut_lengths(nodes, struts)
    sum_L2 = float(np.sum(lengths**2))
    if sum_L2 <= 0:
        raise ValueError("Total L^2 is non-positive.")
    target_volume = total_volume * float(target_vf)
    if overlap_factor is not None:
        factor = overlap_factor
    elif topology_type in ("kagome", "voronoi"):
        factor = OVERLAP_FACTOR_KAGOME if topology_type == "kagome" else OVERLAP_FACTOR_VORONOI
    else:
        factor = OVERLAP_FACTOR_DEFAULT
    k0 = math.sqrt(target_volume / (factor * math.pi * sum_L2))
    k = min(float(k0), K_MAX)
    k = max(k, 1e-6)
    return k


def calculate_radius_one_shot(
    target_vf: float,
    nodes: np.ndarray,
    struts: np.ndarray,
    total_volume: float,
    overlap_c: float = _OVERLAP_C,
) -> float:
    """
    Legacy: single radius estimate. For Smart Inset representative radius.
    Balanced scaling: returns sqrt(max(L_i)) * k where k = calculate_k_one_shot(...).
    """
    k = calculate_k_one_shot(target_vf, nodes, struts, total_volume)
    lengths = _strut_lengths(nodes, struts)
    return float(math.sqrt(np.max(lengths)) * k)


class SolverResult(NamedTuple):
    """
    Structured output from optimization.

    Attributes:
        mesh: Final generated lattice mesh.
        radius: Final (best) strut radius.
        volume: Final (best) trimmed lattice volume.
        iterations: 1 or 2 (Boolean operations).
        nodes: Topology node coordinates used during solve.
        struts: Topology strut index pairs used during solve.
        seed_radius: One-shot analytical radius estimate.
    """

    mesh: trimesh.Trimesh
    radius: float
    volume: float
    iterations: int
    nodes: np.ndarray
    struts: np.ndarray
    seed_radius: float


def _validate_inputs_one_shot(
    mesh: trimesh.Trimesh,
    target_vf: float,
) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"`mesh` must be trimesh.Trimesh, got {type(mesh)}.")
    if target_vf <= 0 or target_vf >= 1:
        raise ValueError("`target_vf` must be between 0 and 1 (exclusive).")
    if mesh.volume <= 0:
        raise ValueError(
            f"Boundary mesh volume must be positive, got {float(mesh.volume)}."
        )


def _scale_boundary_for_inset(mesh: trimesh.Trimesh, radius: float) -> trimesh.Trimesh:
    """
    Create a 'Calculation Boundary' scaled down so the outermost edge of
    cylinders (radius r) stays within the original footprint.
    Scale_Factor = (Size - 2*radius) / Size.
    """
    extents = mesh.extents
    size = float(min(extents))
    if size <= 0:
        raise ValueError("Boundary mesh has zero extent.")
    scale = (size - 2.0 * radius) / size
    scale = max(0.1, min(1.0, scale))
    scaled = mesh.copy()
    centroid = mesh.centroid
    scaled.vertices = (mesh.vertices - centroid) * scale + centroid
    return scaled


def _inverse_scale_nodes_to_original(
    nodes: np.ndarray,
    original_mesh: trimesh.Trimesh,
    radius: float,
) -> np.ndarray:
    """
    Inverse-scale scaffold nodes from Smart Inset (scaled) coords back to
    original boundary coordinate system. Lattice must be in original CAD units.
    """
    extents = original_mesh.extents
    size = float(min(extents))
    scale = (size - 2.0 * radius) / size
    scale = max(0.1, min(1.0, scale))
    centroid = original_mesh.centroid
    return np.asarray((nodes - centroid) / scale + centroid, dtype=np.float64)


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
    clipped_boundary: bool = True,
    union_batch_size: int = 10,
    fast_solve: bool = False,
    topology_type: str = "kagome",
) -> SolverResult:
    """
    One-shot analytical radius + at most 1–2 Boolean operations.

    Uses overlapping cylinder approximation. If error > 2%, applies one linear
    scaling: r_final = r_guess * sqrt(target_vf / achieved_vf).

    Args:
        clipped_boundary: If True, intersect lattice with boundary (flat look).
            If False, export raw cylinder union (pipe look). Vf still uses intersection.
        union_batch_size: Batch size for recursive_batch_union. Try 5, 10, 20, 50.
        fast_solve: If True, use analytical k with overlap factor; build once, no iteration.
        topology_type: For overlap factor selection (kagome/voronoi use 0.85).
    """
    _validate_inputs_one_shot(mesh=mesh, target_vf=target_vf)

    nodes_np = np.asarray(nodes, dtype=np.float64)
    struts_np = np.asarray(struts, dtype=np.int64)
    if nodes_np.ndim != 2 or nodes_np.shape[1] != 3:
        raise ValueError(f"`nodes` must have shape (N, 3); got {nodes_np.shape}.")
    if struts_np.ndim != 2 or struts_np.shape[1] != 2:
        raise ValueError(f"`struts` must have shape (S, 2); got {struts_np.shape}.")
    if r_min <= 0 or r_max <= 0 or r_min >= r_max:
        raise ValueError("Require 0 < r_min < r_max.")

    total_volume = float(mesh.volume)
    crop = clipped_boundary
    lengths = _strut_lengths(nodes_np, struts_np)

    if fast_solve:
        # Analytical only: k with overlap factor, build once, export. No iteration.
        k_guess = calculate_k_analytical(
            target_vf=target_vf,
            nodes=nodes_np,
            struts=struts_np,
            total_volume=total_volume,
            topology_type=topology_type,
        )
        radii_1 = np.sqrt(lengths) * k_guess
        radii_1 = np.clip(radii_1, r_min, r_max)
        result = generate_geometry(
            nodes=nodes_np,
            struts=struts_np,
            strut_radius=radii_1,
            boundary_mesh=mesh,
            add_spheres=False,
            crop_to_boundary=crop,
            return_manifold=False,
            union_batch_size=union_batch_size,
        )
        if crop:
            final_mesh, final_volume = result
        else:
            final_mesh, final_volume = result
        mean_radius = float(np.mean(radii_1))
        print(f"Fast solve: k={k_guess:.6f}, Vf={final_volume / total_volume:.4%}")
        return SolverResult(
            mesh=final_mesh,
            radius=mean_radius,
            volume=float(final_volume),
            iterations=1,
            nodes=nodes_np,
            struts=struts_np,
            seed_radius=float(k_guess * math.sqrt(np.mean(lengths))),
        )

    # One-shot: balanced scaling r_i = sqrt(L_i) * k
    k_guess = calculate_k_one_shot(
        target_vf=target_vf,
        nodes=nodes_np,
        struts=struts_np,
        total_volume=total_volume,
    )
    radii_1 = np.sqrt(lengths) * k_guess
    radii_1 = np.clip(radii_1, r_min, r_max)

    # Boolean 1: union with adaptive radii (return_manifold=True for volume-only pass)
    result_1 = generate_geometry(
        nodes=nodes_np,
        struts=struts_np,
        strut_radius=radii_1,
        boundary_mesh=mesh,
        add_spheres=False,
        crop_to_boundary=crop,
        return_manifold=True,
        union_batch_size=union_batch_size,
    )
    if crop:
        manifold_1, achieved_volume = result_1
        achieved_volume = float(achieved_volume)
    else:
        lattice_1, achieved_volume = result_1
        achieved_volume = float(achieved_volume)
    achieved_vf = achieved_volume / total_volume
    error = abs(achieved_vf - target_vf)
    print(
        f"One-shot guess achieved {achieved_vf:.4%}. Target was {target_vf:.4%}. Error: {error:.2%}"
    )

    # If error > 2%, one linear scaling adjustment: k_final = k_guess * sqrt(target/achieved)
    if error > 0.02:
        if achieved_vf <= 0:
            raise ValueError(
                "Achieved Vf is zero; lattice may not intersect boundary. "
                "Check mesh coordinates or increase target element size."
            )
        k_final = k_guess * math.sqrt(target_vf / achieved_vf)
        k_final = min(k_final, K_MAX)
        radii_2 = np.sqrt(lengths) * k_final
        radii_2 = np.clip(radii_2, r_min, r_max)
        result_2 = generate_geometry(
            nodes=nodes_np,
            struts=struts_np,
            strut_radius=radii_2,
            boundary_mesh=mesh,
            add_spheres=False,
            crop_to_boundary=crop,
            return_manifold=False,
            union_batch_size=union_batch_size,
        )
        if crop:
            final_mesh, final_volume = result_2
            final_volume = float(final_volume)
        else:
            final_mesh, final_volume = result_2
            final_volume = float(final_volume)
        final_radii = radii_2
        iterations_used = 2
    else:
        # Convert manifold to trimesh only for final export (saves ~40s)
        if crop:
            final_mesh = manifold_to_trimesh(manifold_1)
        else:
            final_mesh = lattice_1
        final_volume = achieved_volume
        final_radii = radii_1
        iterations_used = 1

    # Representative radius for SolverResult (mean of adaptive radii)
    mean_radius = float(np.mean(final_radii))

    return SolverResult(
        mesh=final_mesh,
        radius=mean_radius,
        volume=float(final_volume),
        iterations=iterations_used,
        nodes=nodes_np,
        struts=struts_np,
        seed_radius=float(k_guess * math.sqrt(np.mean(lengths))),
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
    clipped_boundary: bool = True,
    algorithm_3d: int = 10,
    union_batch_size: int = 10,
    fast_solve: bool = False,
    export_quality_path: str | None = None,
) -> SolverResult:
    """
    Optimize lattice solid fraction using one-shot analytical radius.

    Generates scaffold + topology once, then delegates to
    optimize_lattice_fraction_from_topology() (1–2 Boolean ops).

    Args:
        clipped_boundary: If True, intersect lattice with boundary (flat look).
            If False, export raw cylinder union (pipe look). Rhombic and
            icosahedral always use clipped; voronoi and kagome allow Full-Pipe.
        algorithm_3d: GMSH 3D algorithm (10=HXT, 4=Netgen). Use 4 for meshes
            with self-intersecting facets.
        union_batch_size: Batch size for recursive_batch_union. Try 5, 10, 20, 50.
        fast_solve: If True, analytical k with overlap factor; build once, no iteration.
    """
    if target_element_size <= 0:
        raise ValueError("`target_element_size` must be > 0.")

    # Rhombic and icosahedral have no natural dual skin; stay clipped
    if topology_type in ("rhombic", "icosahedral"):
        clipped_boundary = True

    boundary_for_scaffold = mesh
    use_smart_inset = False
    r_guess = 0.0
    if not clipped_boundary and topology_type in ("voronoi", "kagome"):
        # Smart Inset: first pass to get representative radius, then scale boundary
        scaffold_0 = generate_conformal_scaffold(
            mesh=mesh,
            target_element_size=float(target_element_size),
            algorithm_3d=algorithm_3d,
            export_quality_path=None,  # Only export from final scaffold
        )
        nodes_0, struts_0 = generate_topology(
            nodes=scaffold_0.nodes,
            elements=scaffold_0.elements,
            surface_faces=scaffold_0.surface_faces,
            topology_type=topology_type,
            include_surface_cage=include_surface_cage,
            target_element_size=target_element_size,
        )
        r_guess = calculate_radius_one_shot(
            target_vf=target_vf,
            nodes=nodes_0,
            struts=struts_0,
            total_volume=float(mesh.volume),
        )
        r_guess = max(r_min, min(r_max, r_guess))
        boundary_for_scaffold = _scale_boundary_for_inset(mesh, r_guess)
        use_smart_inset = True

    scaffold = generate_conformal_scaffold(
        mesh=boundary_for_scaffold,
        target_element_size=float(target_element_size),
        algorithm_3d=algorithm_3d,
        export_quality_path=export_quality_path,
    )
    nodes = np.asarray(scaffold.nodes, dtype=np.float64)
    if use_smart_inset:
        nodes = _inverse_scale_nodes_to_original(nodes, mesh, r_guess)
    nodes, struts = generate_topology(
        nodes=nodes,
        elements=scaffold.elements,
        surface_faces=scaffold.surface_faces,
        topology_type=topology_type,
        include_surface_cage=include_surface_cage,
        target_element_size=target_element_size,
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
        clipped_boundary=clipped_boundary,
        union_batch_size=union_batch_size,
        fast_solve=fast_solve,
        topology_type=topology_type,
    )
