"""
Graphite Explicit Interlinked — Clearance & Topology Verification Module

Provides:
    - Fast pairwise segment-segment distance computation (vectorized broad + narrow phase).
    - Physical surface-to-surface clearance verification (delta >= min_clearance).
    - Topological Gauss linking number integral verification.
    - Analytical Lipschitz gradient clamp for spatially varying cell pitch L(x) and wire radius r(x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Any, Callable
import numpy as np
from scipy.spatial import cKDTree
import scipy.optimize

if TYPE_CHECKING:
    from .patterns import Ring
    from .particle import InterlinkedParticle
    from .cell import InterlinkedCell


def segment_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    """
    Compute minimum Euclidean distance between two sets of 3D line segments.

    Segment 1 endpoints: p1 -> p2 of shape (N, 3).
    Segment 2 endpoints: q1 -> q2 of shape (M, 3).

    Returns:
        Scalar minimum distance in mm across all N x M segment pairs.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    if p1.ndim == 1:
        p1 = p1[None, :]
        p2 = p2[None, :]
    if q1.ndim == 1:
        q1 = q1[None, :]
        q2 = q2[None, :]

    # Broadcast shapes: (N, 1, 3) and (1, M, 3)
    P1 = p1[:, None, :]
    P2 = p2[:, None, :]
    Q1 = q1[None, :, :]
    Q2 = q2[None, :, :]

    u = P2 - P1  # (N, 1, 3)
    v = Q2 - Q1  # (1, M, 3)
    w0 = P1 - Q1  # (N, M, 3)

    # Dot products broadcast to (N, M)
    a = np.sum(u * u, axis=-1)  # (N, 1)
    b = np.sum(u * v, axis=-1)  # (N, M)
    c = np.sum(v * v, axis=-1)  # (1, M)
    d = np.sum(u * w0, axis=-1)  # (N, M)
    e = np.sum(v * w0, axis=-1)  # (N, M)

    a = np.broadcast_to(a, b.shape)
    c = np.broadcast_to(c, b.shape)

    denom = a * c - b * b
    eps = 1e-12

    is_parallel = denom < eps

    denom_safe = np.where(is_parallel, 1.0, denom)
    s_unconstrained = (b * e - c * d) / denom_safe
    s = np.where(is_parallel, 0.0, np.clip(s_unconstrained, 0.0, 1.0))

    c_safe = np.where(c < eps, 1.0, c)
    t = np.clip((b * s + e) / c_safe, 0.0, 1.0)
    t = np.where(c < eps, 0.0, t)

    a_safe = np.where(a < eps, 1.0, a)
    s = np.clip((b * t - d) / a_safe, 0.0, 1.0)
    s = np.where(a < eps, 0.0, s)

    cp_p = P1 + s[:, :, None] * u
    cp_q = Q1 + t[:, :, None] * v

    dist = np.linalg.norm(cp_p - cp_q, axis=-1)
    return float(np.min(dist))


def ring_to_ring_distance(ring1: Ring, ring2: Ring) -> float:
    """Compute minimum centerline distance between two rings in mm."""
    p1 = ring1.nodes[ring1.struts[:, 0]]
    p2 = ring1.nodes[ring1.struts[:, 1]]
    q1 = ring2.nodes[ring2.struts[:, 0]]
    q2 = ring2.nodes[ring2.struts[:, 1]]
    return segment_segment_distance(p1, p2, q1, q2)


def ring_to_ring_clearance(ring1: Ring, ring2: Ring) -> float:
    """
    Compute physical surface-to-surface clearance between two solid wire rings.

    Formula:
        clearance = d_centerline - (r_wire1 + r_wire2)
    Positive value indicates non-contact clearance. Negative value indicates collision.
    """
    d = ring_to_ring_distance(ring1, ring2)
    return float(d - (ring1.wire_radius + ring2.wire_radius))


def compute_pairwise_ring_clearances(
    rings: list[Ring],
    broadphase_scale: float = 1.15,
) -> tuple[float, list[dict]]:
    """
    Evaluate all pairwise clearances across a collection of rings.

    Uses a fast bounding-sphere broad-phase test to reject non-adjacent pairs,
    followed by vectorized segment-segment distance for candidate pairs.

    Args:
        rings: List of Ring objects.
        broadphase_scale: Multiplier on (R_outer1 + R_outer2) for broad-phase candidate gating.

    Returns:
        tuple: (min_clearance, pair_records)
            where pair_records contains dicts with {ring_i, ring_j, clearance, linking_number}.
    """
    n_rings = len(rings)
    if n_rings < 2:
        return float("inf"), []

    records = []
    global_min_clr = float("inf")

    centers = np.array([r.center for r in rings], dtype=np.float64)
    outers = np.array([r.outer_radius for r in rings], dtype=np.float64)

    for i in range(n_rings):
        c_i = centers[i]
        r_out_i = outers[i]
        ring_i = rings[i]

        for j in range(i + 1, n_rings):
            c_j = centers[j]
            r_out_j = outers[j]
            ring_j = rings[j]

            dist_c = float(np.linalg.norm(c_i - c_j))
            threshold = broadphase_scale * (r_out_i + r_out_j)

            # If centers are too far, minimum possible clearance is bounded by dist_c - (r_out_i + r_out_j)
            if dist_c > threshold:
                continue

            clr = ring_to_ring_clearance(ring_i, ring_j)
            if clr < global_min_clr:
                global_min_clr = clr

            lk = gauss_linking_number(ring_i, ring_j)
            records.append({
                "ring_i": i,
                "ring_j": j,
                "tag_i": ring_i.tag,
                "tag_j": ring_j.tag,
                "clearance": clr,
                "linking_number": round(lk),
                "gauss_raw": lk,
            })

    return global_min_clr, records


def check_ring_clearance(
    rings: list[Ring],
    min_clearance: float = 0.30,
) -> tuple[bool, float, list[dict]]:
    """
    Verify that all pairwise clearances satisfy delta >= min_clearance.

    Args:
        rings: List of Ring objects.
        min_clearance: Required minimum surface-to-surface gap in mm.

    Returns:
        tuple: (is_valid, min_clearance_found, violations)
    """
    min_clr, records = compute_pairwise_ring_clearances(rings)
    violations = [rec for rec in records if rec["clearance"] < min_clearance]
    is_valid = (min_clr >= min_clearance) and (len(violations) == 0)
    return is_valid, min_clr, violations


def gauss_linking_number(ring1: Ring, ring2: Ring) -> float:
    r"""
    Compute Gauss linking number between two closed polygonal curves.

    Evaluates the double path integral:
        Lk = (1 / 4pi) * \oint_{C1} \oint_{C2} ((r1 - r2) . (dr1 x dr2)) / |r1 - r2|^3
    using 3-point Gauss-Legendre quadrature per segment pair.

    Returns:
        float representing the linking number (close to integer 0, 1, -1, etc.).
    """
    p0 = ring1.nodes[ring1.struts[:, 0]]  # (N, 3)
    p1 = ring1.nodes[ring1.struts[:, 1]]  # (N, 3)
    q0 = ring2.nodes[ring2.struts[:, 0]]  # (M, 3)
    q1 = ring2.nodes[ring2.struts[:, 1]]  # (M, 3)

    # 3-point Gauss-Legendre quadrature points & weights on [0, 1]
    g_pts = np.array([0.5 - np.sqrt(3.0 / 5.0) / 2.0, 0.5, 0.5 + np.sqrt(3.0 / 5.0) / 2.0])
    g_wts = np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0])

    u = p1 - p0  # (N, 3)
    v = q1 - q0  # (M, 3)

    u_b = u[:, None, :]  # (N, 1, 3)
    v_b = v[None, :, :]  # (1, M, 3)
    cross_uv = np.cross(u_b, v_b)  # (N, M, 3)

    total_sum = 0.0
    for si, ws in zip(g_pts, g_wts):
        ps = p0 + si * u
        ps_b = ps[:, None, :]
        for tj, wt in zip(g_pts, g_wts):
            qt = q0 + tj * v
            qt_b = qt[None, :, :]

            diff = ps_b - qt_b
            dist3 = np.maximum(np.linalg.norm(diff, axis=-1) ** 3, 1e-12)
            dot = np.sum(diff * cross_uv, axis=-1)
            total_sum += ws * wt * float(np.sum(dot / dist3))

    return float(total_sum / (4.0 * np.pi))


def clamp_lipschitz_gradients(
    values: np.ndarray,
    coordinates: np.ndarray,
    max_gradient: float,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Enforce analytical Lipschitz gradient continuity on a spatial scalar field.

    Ensures that for all points i, j:
        |values[i] - values[j]| <= max_gradient * ||coordinates[i] - coordinates[j]||
    This prevents large cell pitch L(x) or wire thickness r(x) from expanding
    faster than neighboring inner clearances can accommodate.

    Args:
        values: (N,) scalar array (e.g. cell size L or wire thickness r).
        coordinates: (N, 3) spatial coordinates of each sample point.
        max_gradient: Maximum permissible spatial slope (dimensionless or mm/mm).
        max_iterations: Maximum relaxation iterations.

    Returns:
        (N,) clamped values satisfying the Lipschitz bound.
    """
    v = np.array(values, dtype=np.float64).copy()
    coords = np.asarray(coordinates, dtype=np.float64)
    k_max = float(max_gradient)

    n_pts = len(v)
    if n_pts < 2 or k_max <= 0:
        return v

    # Pairwise distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (N, N)

    for _ in range(max_iterations):
        changed = False
        # Maximum allowed upper envelope: v[j] <= v[i] + k_max * dist[i, j]
        upper_bound = np.min(v[:, None] + k_max * dist, axis=0)
        # Minimum allowed lower envelope: v[j] >= v[i] - k_max * dist[i, j]
        lower_bound = np.max(v[:, None] - k_max * dist, axis=0)

        new_v = np.clip(v, lower_bound, upper_bound)
        if np.max(np.abs(new_v - v)) < 1e-6:
            break
        v = new_v

    return v


# =============================================================================
# Phase 4: Two-Tier Vectorized Clearance & Inversion Engine
# =============================================================================

def _extract_particle_nodes_and_struts(p: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extract (nodes, struts, center, bounding_radius) from InterlinkedParticle, PAMParticle, or Ring.
    """
    if hasattr(p, "global_nodes") and callable(p.global_nodes):
        nodes = np.asarray(p.global_nodes(), dtype=np.float64)
        struts = np.asarray(p.struts, dtype=np.int64)
        center = np.asarray(p.center, dtype=np.float64)
        r_bound = float(p.bounding_radius)
        return nodes, struts, center, r_bound
    elif hasattr(p, "nodes") and hasattr(p, "struts"):
        nodes = np.asarray(p.nodes, dtype=np.float64)
        struts = np.asarray(p.struts, dtype=np.int64)
        center = np.asarray(getattr(p, "center", nodes.mean(axis=0)), dtype=np.float64)
        r_bound = float(getattr(p, "bounding_radius", getattr(p, "outer_radius", np.max(np.linalg.norm(nodes - center, axis=1)))))
        return nodes, struts, center, r_bound
    raise TypeError(f"Unsupported particle type: {type(p)}")


def circle_circle_distance(
    center1: np.ndarray,
    normal1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    normal2: np.ndarray,
    radius2: float,
    num_samples: int = 36,
) -> float:
    """
    Compute minimum Euclidean distance between two spatial circles in R^3.

    Uses an initial discrete grid search over `num_samples` on each circle,
    followed by local continuous 2D Powell optimization for sub-micron accuracy.

    Args:
        center1, normal1, radius1: Parameters of first circle.
        center2, normal2, radius2: Parameters of second circle.
        num_samples: Angular discretization resolution per circle.

    Returns:
        Scalar minimum distance in mm between the two circular paths.
    """
    c1 = np.asarray(center1, dtype=np.float64).reshape(3)
    c2 = np.asarray(center2, dtype=np.float64).reshape(3)
    n1 = np.asarray(normal1, dtype=np.float64).reshape(3)
    n2 = np.asarray(normal2, dtype=np.float64).reshape(3)
    n1 = n1 / np.maximum(np.linalg.norm(n1), 1e-12)
    n2 = n2 / np.maximum(np.linalg.norm(n2), 1e-12)
    R1, R2 = float(radius1), float(radius2)

    # Orthonormal basis for circle 1
    up1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n1, up1))) > 0.90:
        up1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u1 = np.cross(up1, n1)
    u1 = u1 / np.linalg.norm(u1)
    v1 = np.cross(n1, u1)

    # Orthonormal basis for circle 2
    up2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(n2, up2))) > 0.90:
        up2 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u2 = np.cross(up2, n2)
    u2 = u2 / np.linalg.norm(u2)
    v2 = np.cross(n2, u2)

    thetas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    phis = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)

    pts1 = c1 + R1 * (np.cos(thetas)[:, None] * u1 + np.sin(thetas)[:, None] * v1)
    pts2 = c2 + R2 * (np.cos(phis)[:, None] * u2 + np.sin(phis)[:, None] * v2)

    diff = pts1[:, None, :] - pts2[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    min_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    th_best = thetas[min_idx[0]]
    phi_best = phis[min_idx[1]]

    # Continuous 2D refinement
    delta_angle = 2.0 * np.pi / num_samples

    def _sq_dist(angles: np.ndarray) -> float:
        th, ph = angles
        p = c1 + R1 * (np.cos(th) * u1 + np.sin(th) * v1)
        q = c2 + R2 * (np.cos(ph) * u2 + np.sin(ph) * v2)
        return float(np.sum((p - q) ** 2))

    res = scipy.optimize.minimize(
        _sq_dist,
        x0=np.array([th_best, phi_best]),
        method="Powell",
        bounds=[(th_best - delta_angle, th_best + delta_angle), (phi_best - delta_angle, phi_best + delta_angle)],
        options={"xtol": 1e-6, "ftol": 1e-8},
    )
    return float(np.sqrt(max(0.0, res.fun)))


def particle_pair_centerline_distance(
    p1: Any,
    p2: Any,
) -> float:
    """
    Compute minimum 3D Euclidean centerline distance between two particles.

    Supports InterlinkedParticle, PAMParticle, and Ring instances.
    Evaluates vectorized segment-segment distance across all struts of p1 and p2.
    """
    nodes1, struts1, _, _ = _extract_particle_nodes_and_struts(p1)
    nodes2, struts2, _, _ = _extract_particle_nodes_and_struts(p2)

    if len(struts1) == 0 or len(struts2) == 0:
        return float(np.min(np.linalg.norm(nodes1[:, None, :] - nodes2[None, :, :], axis=-1)))

    p0 = nodes1[struts1[:, 0]]
    p1_end = nodes1[struts1[:, 1]]
    q0 = nodes2[struts2[:, 0]]
    q1_end = nodes2[struts2[:, 1]]

    return segment_segment_distance(p0, p1_end, q0, q1_end)


def particle_pair_clearance(
    p1: Any,
    p2: Any,
    strut_radius: float | tuple[float, float] = 0.5,
) -> float:
    """
    Compute physical surface-to-surface clearance Delta between two particles:
        Delta = d_centerline - (r1 + r2)
    Positive indicates non-contact free play; negative indicates solid collision/penetration.
    """
    if isinstance(strut_radius, (tuple, list)):
        r1, r2 = float(strut_radius[0]), float(strut_radius[1])
    else:
        r1 = r2 = float(strut_radius)

    d = particle_pair_centerline_distance(p1, p2)
    return float(d - (r1 + r2))


def particle_linking_number(
    p1: Any,
    p2: Any,
) -> float:
    """
    Compute Gauss linking number between two closed polygonal particles.
    """
    nodes1, struts1, _, _ = _extract_particle_nodes_and_struts(p1)
    nodes2, struts2, _, _ = _extract_particle_nodes_and_struts(p2)

    from .patterns import Ring
    r1 = Ring(
        nodes=nodes1,
        struts=struts1,
        center=nodes1.mean(axis=0),
        normal=np.array([0.0, 0.0, 1.0]),
        radius=1.0,
        wire_radius=0.1,
    )
    r2 = Ring(
        nodes=nodes2,
        struts=struts2,
        center=nodes2.mean(axis=0),
        normal=np.array([0.0, 0.0, 1.0]),
        radius=1.0,
        wire_radius=0.1,
    )
    return gauss_linking_number(r1, r2)


def compute_pairwise_particle_clearances(
    particles: Sequence[Any],
    strut_radius: float | Sequence[float] = 0.5,
    min_clearance: float = 0.30,
    broadphase_margin: float = 0.5,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Two-Tier clearance verification across an assembly of particles:

    Tier 1: Broad-Phase (O(N log N))
        Builds a KD-tree over particle centroids and queries candidate pairs
        whose bounding spheres could physically interact.
        Non-interacting pairs are immediately rejected in O(log N) time.

    Tier 2: Narrow-Phase (Vectorized 3D geometry)
        Evaluates exact segment-segment 3D distance between candidate pairs.

    Returns:
        tuple: (global_min_clearance, records)
            records: list of dicts with clearance, centerline distance, and particle metadata.
    """
    n_parts = len(particles)
    if n_parts < 2:
        return float("inf"), []

    extracted = [_extract_particle_nodes_and_struts(p) for p in particles]
    centers = np.array([item[2] for item in extracted], dtype=np.float64)
    bounding_radii = np.array([item[3] for item in extracted], dtype=np.float64)

    if isinstance(strut_radius, (int, float)):
        strut_r_arr = np.full(n_parts, float(strut_radius), dtype=np.float64)
    else:
        strut_r_arr = np.asarray(strut_radius, dtype=np.float64)
        if len(strut_r_arr) != n_parts:
            raise ValueError(f"strut_radius length ({len(strut_r_arr)}) must match particle count ({n_parts})")

    # Broad-phase candidate pair query via cKDTree
    max_reach = float(np.max(bounding_radii + strut_r_arr))
    query_radius = 2.0 * max_reach + float(broadphase_margin)

    tree = cKDTree(centers)
    candidate_pairs = tree.query_pairs(r=query_radius)

    records: list[dict[str, Any]] = []
    global_min_clr = float("inf")

    for i, j in sorted(candidate_pairs):
        c_i = centers[i]
        c_j = centers[j]
        dist_c = float(np.linalg.norm(c_i - c_j))

        # Tighter particle-specific bounding sphere test
        r_bound_i = bounding_radii[i]
        r_bound_j = bounding_radii[j]
        r_wire_i = strut_r_arr[i]
        r_wire_j = strut_r_arr[j]

        threshold = r_bound_i + r_bound_j + r_wire_i + r_wire_j + float(broadphase_margin)
        if dist_c > threshold:
            continue

        d_centerline = particle_pair_centerline_distance(particles[i], particles[j])
        clr = float(d_centerline - (r_wire_i + r_wire_j))

        if clr < global_min_clr:
            global_min_clr = clr

        records.append({
            "particle_i": i,
            "particle_j": j,
            "id_i": getattr(particles[i], "particle_id", i),
            "id_j": getattr(particles[j], "particle_id", j),
            "sublattice_i": getattr(particles[i], "sublattice_id", ""),
            "sublattice_j": getattr(particles[j], "sublattice_id", ""),
            "center_distance": dist_c,
            "centerline_distance": d_centerline,
            "clearance": clr,
        })

    return global_min_clr, records


def check_particle_clearance(
    particles: Sequence[Any],
    strut_radius: float | Sequence[float] = 0.5,
    min_clearance: float = 0.30,
    broadphase_margin: float = 0.5,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """
    Verify that all pairwise clearances across particles satisfy Delta >= min_clearance.

    Args:
        particles: Sequence of InterlinkedParticle, PAMParticle, or Ring instances.
        strut_radius: Uniform or per-particle strut radius in mm.
        min_clearance: Required minimum free-play clearance in mm (default 0.30 mm).
        broadphase_margin: Additional buffer margin for KD-tree broad-phase candidate search.

    Returns:
        tuple: (is_valid, min_clearance_found, violations)
    """
    min_clr, records = compute_pairwise_particle_clearances(
        particles=particles,
        strut_radius=strut_radius,
        min_clearance=min_clearance,
        broadphase_margin=broadphase_margin,
    )
    violations = [rec for rec in records if rec["clearance"] < float(min_clearance)]
    is_valid = (min_clr >= float(min_clearance)) and (len(violations) == 0)
    return is_valid, min_clr, violations


def resolve_lattice_pitch(
    cell: InterlinkedCell,
    target_clearance: float,
    strut_diameter: float,
    **kwargs: Any,
) -> float:
    """
    Polymorphic pitch inversion: determines the required unit cell pitch (mm)
    to achieve target_clearance for a given cell type and strut diameter.

    Delegates to cell.resolve_pitch() if available; falls back to robust numerical
    root-finding via Brent's method if not implemented.
    """
    try:
        return float(cell.resolve_pitch(target_clearance, strut_diameter, **kwargs))
    except (NotImplementedError, AttributeError):
        return _numerical_resolve_pitch(cell, target_clearance, strut_diameter, **kwargs)


def _numerical_resolve_pitch(
    cell: InterlinkedCell,
    target_clearance: float,
    strut_diameter: float,
    pitch_bounds: tuple[float, float] = (0.1, 500.0),
    tol: float = 1e-5,
    **kwargs: Any,
) -> float:
    """
    Numerical root-finding pitch inversion using Brent's method.
    """
    tc = float(target_clearance)
    d = float(strut_diameter)

    def _residual(p: float) -> float:
        return float(cell.forward_clearance(p, d, **kwargs) - tc)

    p_min, p_max = float(pitch_bounds[0]), float(pitch_bounds[1])
    f_min = _residual(p_min)
    f_max = _residual(p_max)
    if f_min * f_max > 0.0:
        for factor in [0.1, 0.01, 10.0, 100.0]:
            p_min_cand = max(0.01, p_min * factor)
            p_max_cand = p_max * factor
            if _residual(p_min_cand) * _residual(p_max_cand) <= 0.0:
                p_min, p_max = p_min_cand, p_max_cand
                break
        else:
            raise ValueError(
                f"Could not bracket root for cell {getattr(cell, 'name', cell)} "
                f"with target_clearance={tc} and strut_diameter={d} in range [{p_min}, {p_max}]."
            )

    root = scipy.optimize.brentq(_residual, p_min, p_max, xtol=tol)
    return float(root)


def calibrate_cell_clearance_curve(
    cell: InterlinkedCell,
    strut_diameter: float,
    pitch_range: tuple[float, float] = (5.0, 30.0),
    num_points: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Calibrate and characterize the clearance law Delta(a_0) for an InterlinkedCell.
    Evaluates forward_clearance across a range of pitches and tests for linearity:
        Delta = kappa * a_0 + intercept

    Returns:
        dict containing slope (kappa), intercept, R^2 goodness of fit, and linearity flag.
    """
    d = float(strut_diameter)
    pitches = np.linspace(float(pitch_range[0]), float(pitch_range[1]), int(num_points))
    clearances = np.array([cell.forward_clearance(p, d, **kwargs) for p in pitches], dtype=np.float64)

    slope, intercept = np.polyfit(pitches, clearances, 1)
    residuals = clearances - (slope * pitches + intercept)
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((clearances - np.mean(clearances)) ** 2))
    r_squared = 1.0 - (ss_res / max(ss_tot, 1e-12))

    return {
        "cell_name": getattr(cell, "name", str(cell)),
        "strut_diameter_mm": d,
        "kappa": float(slope),
        "intercept_mm": float(intercept),
        "r_squared": float(r_squared),
        "is_linear": bool(r_squared > 0.999),
        "samples": list(zip(pitches.tolist(), clearances.tolist())),
    }
