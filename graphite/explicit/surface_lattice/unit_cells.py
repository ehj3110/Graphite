# -*- coding: utf-8 -*-
"""
Graphite Explicit Surface Lattice Engine - 2D Unit Cell Topologies & Tessellation.

Provides procedural generators for 2D periodic unit cells, including:
1. Chiral Metamaterials:
   - Tetra-Chiral (square lattice, 4 tangential ligaments per circular node)
   - Tri-Chiral (hexagonal/triangular lattice, 3 tangential ligaments per circular node)
   - Anti-Tetra-Chiral & Anti-Tri-Chiral variants
2. Regular & Coaster Metamaterials:
   - Grid, Tetrahedral, Icosahedral, Kelvin, Tesseract, Rhombic
3. Planar & Periodic 2D Domain Tessellation:
   - Domain tiling with closed-form circumferential seam closure for cylindrical wrapping.
"""
from __future__ import annotations

from typing import Callable, Iterable
import numpy as np


def generate_tetrachiral_cell(
    L: float = 10.0,
    r: float = 2.0,
    t: float = 1.0,
    n_circle_segments: int = 16,
    chiral: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate nodes and strut edges for a single tetra-chiral (square) unit cell.

    Parameters
    ----------
    L : float
        Length of each straight ligament in mm.
    r : float
        Radius of the central circular node in mm.
    t : float
        Nominal strut / ligament thickness in mm.
    n_circle_segments : int
        Discretization resolution for the circular node polygon.
    chiral : bool
        True for standard chiral (same-sense rotation), False for anti-chiral.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        2D node coordinates centered at (0, 0).
    struts : ndarray, shape (S, 2)
        Edge index pairs.
    metadata : dict
        Geometric properties (pitch D, rotation angle alpha).
    """
    if L <= 0.0 or r <= 0.0:
        raise ValueError("L and r must be strictly positive.")

    D = np.sqrt(L**2 + 4.0 * r**2)
    alpha = np.arctan(2.0 * r / L)

    node_list: list[np.ndarray] = []
    strut_list: list[tuple[int, int]] = []

    # 1. Circular node chords
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segments, endpoint=False)
    circle_indices = []
    for ang in circle_angles:
        idx = len(node_list)
        node_list.append(r * np.array([np.cos(ang), np.sin(ang)]))
        circle_indices.append(idx)

    for k in range(n_circle_segments):
        strut_list.append((circle_indices[k], circle_indices[(k + 1) % n_circle_segments]))

    # 2. Four tangent ligaments
    for k in range(4):
        if chiral:
            th_base = -alpha + k * (np.pi / 2.0)
            p_start = r * np.array([-np.sin(th_base), np.cos(th_base)])
            p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
        else:
            ang = k * (np.pi / 2.0)
            p_start = r * np.array([-np.sin(ang), np.cos(ang)])
            p_end = p_start + L * np.array([np.cos(ang), np.sin(ang)])

        idx_s = len(node_list)
        node_list.append(p_start)
        idx_e = len(node_list)
        node_list.append(p_end)
        strut_list.append((idx_s, idx_e))

    nodes = np.array(node_list, dtype=np.float64)
    struts = np.array(strut_list, dtype=np.int64)
    metadata = {
        "type": "tetra_chiral" if chiral else "anti_tetra_chiral",
        "L": float(L),
        "r": float(r),
        "t": float(t),
        "D_pitch": float(D),
        "alpha_deg": float(np.degrees(alpha)),
    }
    return nodes, struts, metadata


def generate_trichiral_cell(
    L: float = 10.0,
    r: float = 2.0,
    t: float = 1.0,
    n_circle_segments: int = 16,
    chiral: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate nodes and strut edges for a single tri-chiral (triangular) unit cell.

    Parameters
    ----------
    L : float
        Length of each straight ligament in mm.
    r : float
        Radius of the central circular node in mm.
    t : float
        Nominal strut / ligament thickness in mm.
    n_circle_segments : int
        Discretization resolution for the circular node polygon.
    chiral : bool
        True for standard chiral, False for anti-chiral.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        2D node coordinates centered at (0, 0).
    struts : ndarray, shape (S, 2)
        Edge index pairs.
    metadata : dict
        Geometric properties (pitch D, offset angle phi).
    """
    if L <= 0.0 or r <= 0.0:
        raise ValueError("L and r must be strictly positive.")

    D = np.sqrt(L**2 + 4.0 * r**2)
    phi = np.arcsin(2.0 * r / D)

    node_list: list[np.ndarray] = []
    strut_list: list[tuple[int, int]] = []

    # 1. Circular node chords
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segments, endpoint=False)
    circle_indices = []
    for ang in circle_angles:
        idx = len(node_list)
        node_list.append(r * np.array([np.cos(ang), np.sin(ang)]))
        circle_indices.append(idx)

    for k in range(n_circle_segments):
        strut_list.append((circle_indices[k], circle_indices[(k + 1) % n_circle_segments]))

    # 2. Three tangent ligaments spaced at 120-degree intervals
    for k in range(3):
        if chiral:
            th_base = phi + k * (2.0 * np.pi / 3.0)
            tang_ang = th_base - phi
        else:
            tang_ang = k * (2.0 * np.pi / 3.0)

        p_start = r * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
        p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])

        idx_s = len(node_list)
        node_list.append(p_start)
        idx_e = len(node_list)
        node_list.append(p_end)
        strut_list.append((idx_s, idx_e))

    nodes = np.array(node_list, dtype=np.float64)
    struts = np.array(strut_list, dtype=np.int64)
    metadata = {
        "type": "tri_chiral" if chiral else "anti_tri_chiral",
        "L": float(L),
        "r": float(r),
        "t": float(t),
        "D_pitch": float(D),
        "phi_deg": float(np.degrees(phi)),
    }
    return nodes, struts, metadata


def dedupe_2d_segments(
    segments: Iterable[tuple[np.ndarray, np.ndarray]],
    period_x: float | None = None,
    tol: float = 1e-4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deduplicate undirected 2D line segments, optionally modulo a periodic X period.
    """
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    inv = 1.0 / max(tol, 1e-12)

    for p1, p2 in segments:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        if period_x is not None and period_x > 0.0:
            x1 = x1 % period_x
            x2 = x2 % period_x

        k1 = (int(round(x1 * inv)), int(round(y1 * inv)))
        k2 = (int(round(x2 * inv)), int(round(y2 * inv)))
        if k1 == k2:
            continue

        key = (k1, k2) if k1 < k2 else (k2, k1)
        if key not in seen:
            seen.add(key)
            unique.append((np.array([x1, y1]), np.array([x2, y2])))

    return unique


def tessellate_chiral_domain(
    topology: str,
    domain_width: float,
    domain_height: float,
    n_circumferential: int = 10,
    r_node: float = 2.0,
    n_circle_segs: int = 16,
    periodic_x: bool = True,
    y_base: float = 0.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Tessellate a 2D rectangular domain with a periodic chiral lattice.

    Parameters
    ----------
    topology : str
        "tetra" or "tetra_chiral" for square basis, "tri" or "tri_chiral" for hexagonal basis.
    domain_width : float
        Width in mm (e.g. cylinder circumference C_mid = 2*pi*R_mid).
    domain_height : float
        Height in mm.
    n_circumferential : int
        Number of unit cells across the width. Pitch D = domain_width / n_circumferential.
    r_node : float
        Radius of circular nodes in mm.
    n_circle_segs : int
        Polygon chords per circular node.
    periodic_x : bool
        If True, enforces closed-form periodic basis rotation for seam-free wrapping.
    y_base : float
        Vertical offset.

    Returns
    -------
    segments : list of (ndarray, ndarray)
        Unique 2D line segments.
    metadata : dict
        Geometric parameters.
    """
    D = domain_width / float(n_circumferential)
    if D <= 2.0 * r_node:
        raise ValueError(f"Pitch D={D:.2f} mm must be strictly greater than 2*r_node={2*r_node:.2f} mm.")

    L = np.sqrt(D**2 - 4.0 * r_node**2)
    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segs, endpoint=False)

    topo_lower = topology.lower().strip()
    is_chiral = not (topo_lower.startswith("anti_") or topo_lower.startswith("anti"))
    topo_clean = topo_lower.replace("anti_", "").replace("anti", "").replace("_chiral", "").replace("chiral", "")

    if topo_clean == "tetra":
        alpha = np.arctan(2.0 * r_node / L)
        m_rows = max(1, int(round(domain_height / D)))
        metadata = {
            "topology": "tetra_chiral" if is_chiral else "anti_tetra_chiral",
            "pitch_D": float(D),
            "ligament_L": float(L),
            "r_node": float(r_node),
            "alpha_deg": float(np.degrees(alpha)),
            "n_circumferential": n_circumferential,
            "m_rows": m_rows,
        }

        # Full unit cells: node placed at center ((i + 0.5)*D, (j + 0.5)*D)
        for i in range(n_circumferential):
            for j in range(m_rows):
                c = np.array([(i + 0.5) * D, y_base + (j + 0.5) * D], dtype=np.float64)

                # 1. Complete circular node chords
                for k in range(n_circle_segs):
                    a1 = circle_angles[k]
                    a2 = circle_angles[(k + 1) % n_circle_segs]
                    p1 = c + r_node * np.array([np.cos(a1), np.sin(a1)])
                    p2 = c + r_node * np.array([np.cos(a2), np.sin(a2)])
                    raw_segments.append((p1, p2))

                # 2. Horizontal tangent ligament to neighbor (i + 1, j)
                has_next_x = (i + 1 < n_circumferential) or periodic_x
                if has_next_x:
                    if is_chiral:
                        th_base = -alpha
                        p_start = c + r_node * np.array([-np.sin(th_base), np.cos(th_base)])
                        p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
                    else:
                        p_start = c + r_node * np.array([0.0, 1.0])
                        p_end = p_start + L * np.array([1.0, 0.0])
                    raw_segments.append((p_start, p_end))

                # 3. Vertical tangent ligament to neighbor (i, j + 1)
                if j + 1 < m_rows:
                    if is_chiral:
                        th_base = -alpha + np.pi / 2.0
                        p_start = c + r_node * np.array([-np.sin(th_base), np.cos(th_base)])
                        p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
                    else:
                        p_start = c + r_node * np.array([-1.0, 0.0])
                        p_end = p_start + L * np.array([0.0, 1.0])
                    raw_segments.append((p_start, p_end))

    elif topo_clean == "tri":
        phi = np.arcsin(2.0 * r_node / D)
        a1 = np.array([D, 0.0])
        a2 = np.array([D * 0.5, D * np.sqrt(3.0) / 2.0])
        m_rows = max(1, int(round(domain_height / a2[1])))
        metadata = {
            "topology": "tri_chiral" if is_chiral else "anti_tri_chiral",
            "pitch_D": float(D),
            "ligament_L": float(L),
            "r_node": float(r_node),
            "phi_deg": float(np.degrees(phi)),
            "n_circumferential": n_circumferential,
            "m_rows": m_rows,
        }

        # Full unit cells: node placed at center
        for i in range(n_circumferential):
            for j in range(m_rows):
                c = i * a1 + j * a2 + np.array([D * 0.5, y_base + a2[1] * 0.5], dtype=np.float64)

                # 1. Complete circular node chords
                for k in range(n_circle_segs):
                    a1_ang = circle_angles[k]
                    a2_ang = circle_angles[(k + 1) % n_circle_segs]
                    p1 = c + r_node * np.array([np.cos(a1_ang), np.sin(a1_ang)])
                    p2 = c + r_node * np.array([np.cos(a2_ang), np.sin(a2_ang)])
                    raw_segments.append((p1, p2))

                # 2. Ligament 0 (tangent angle 0, along +a1 toward neighbor i+1, j)
                if (i + 1 < n_circumferential) or periodic_x:
                    if is_chiral:
                        tang_ang = 0.0
                    else:
                        tang_ang = 0.0
                    p_start = c + r_node * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
                    p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])
                    raw_segments.append((p_start, p_end))

                # 3. Ligament 1 (tangent angle 120 deg, toward neighbor i-1, j+1)
                if j + 1 < m_rows and ((i - 1 >= 0) or periodic_x):
                    if is_chiral:
                        tang_ang = 2.0 * np.pi / 3.0
                    else:
                        tang_ang = 2.0 * np.pi / 3.0
                    p_start = c + r_node * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
                    p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])
                    raw_segments.append((p_start, p_end))

                # 4. Ligament 2 (tangent angle 240 deg, toward neighbor i, j-1)
                if j - 1 >= 0:
                    if is_chiral:
                        tang_ang = 4.0 * np.pi / 3.0
                    else:
                        tang_ang = 4.0 * np.pi / 3.0
                    p_start = c + r_node * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
                    p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])
                    raw_segments.append((p_start, p_end))

    else:
        raise ValueError(f"Unsupported chiral topology '{topology}'. Expected 'tetra' or 'tri'.")

    # Deduplicate without box slicing to preserve complete full unit cells
    period = domain_width if periodic_x else None
    unique = dedupe_2d_segments(raw_segments, period_x=period)
    metadata["n_segments"] = len(unique)
    return unique, metadata


def tessellate_reentrant_domain(
    domain_width: float,
    domain_height: float,
    n_circumferential: int = 10,
    m_vertical: int | None = None,
    w_top_ratio: float = 0.65,
    w_waist_ratio: float | Callable[[float], float] = 0.30,
    variant: str = "base",
    periodic_x: bool = True,
    y_base: float = 0.0,
    theta_field: float | Callable[[float], float] | Sequence[float] | None = None,
    theta_min_deg: float = 5.0,
    theta_max_deg: float = 45.0,
    column_boundaries: Sequence[float] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Generate 2D line segments for Chen et al. (2020) re-entrant auxetic lattice on a domain.
    Supports constant, spatially variable, or curvature-compensated internal angles theta(x)
    and variable cell widths/sizing along X via column_boundaries.

    Parameters
    ----------
    domain_width : float
        Domain width in mm (or circumference 2*pi*R_mid for cylinders).
    domain_height : float
        Domain height in mm.
    n_circumferential : int
        Number of periodic cell columns across domain width (used if column_boundaries is None).
    m_vertical : int, optional
        Number of cell rows along height. If None, auto-calculated from cell aspect ratio.
    w_top_ratio : float
        Width of top/bottom caps as fraction of cell width a (default 0.65).
    w_waist_ratio : float or callable
        Width of waist as fraction of cell width a (default 0.30), or callable w_waist(x).
    variant : str
        'base' (re-entrant hexagonal cell), 'type_a' (horizontal waist rib), or 'type_b' (vertical central rib).
    periodic_x : bool
        If True, enforces closed-form periodic wrapping for cylinders.
    y_base : float
        Y coordinate offset of bottom boundary.
    theta_field : float, callable, or sequence, optional
        Internal re-entrant angle in degrees (or callable theta(x) along domain width).
        If provided, overrides w_waist_ratio via:
            delta_u = (b / 2) * tan(radians(theta))
            w_waist = w_top - 2 * delta_u
    theta_min_deg : float
        Minimum allowable re-entrant angle in degrees to prevent strut self-contact/locking (default 5.0).
    theta_max_deg : float
        Maximum allowable re-entrant angle in degrees (default 45.0).
    column_boundaries : sequence of float, optional
        Discrete cell column boundary coordinates [s_0, s_1, ..., s_M] in mm.
        If provided, enables multi-scale variable cell sizing across the domain.

    Returns
    -------
    segments : list of tuple(ndarray, ndarray)
        Deduplicated 2D line segments.
    metadata : dict
        Geometric parameters.
    """
    if column_boundaries is not None:
        bnds = np.asarray(column_boundaries, dtype=np.float64)
        if len(bnds) < 2:
            raise ValueError("column_boundaries must have at least 2 coordinate points.")
        cols = len(bnds) - 1
        domain_width = float(bnds[-1] - bnds[0])
        col_widths = [float(bnds[i + 1] - bnds[i]) for i in range(cols)]
        a_mean = float(np.mean(col_widths))
        if m_vertical is None:
            m_vertical = max(int(round(domain_height / a_mean)), 1)
        b = domain_height / float(m_vertical)
        h_waist_y = b / 2.0
        centers = [float((bnds[i] + bnds[i + 1]) / 2.0) for i in range(cols)]
    else:
        cols = int(n_circumferential)
        a_mean = domain_width / float(cols)
        if m_vertical is None:
            m_vertical = max(int(round(domain_height / a_mean)), 1)
        b = domain_height / float(m_vertical)
        h_waist_y = b / 2.0
        col_widths = [a_mean] * cols
        centers = [float((i + 0.5) * a_mean) for i in range(cols)]

    w_top_cols = [w_top_ratio * ai for ai in col_widths]

    # Precalculate per-column waist widths and angles with safety clamping
    w_waist_cols: list[float] = []
    theta_cols: list[float] = []

    for i in range(cols):
        u_c = centers[i]
        a_i = col_widths[i]
        w_top_i = w_top_cols[i]

        if theta_field is not None:
            if callable(theta_field):
                th_deg = float(theta_field(u_c))
            elif isinstance(theta_field, (list, tuple, np.ndarray)):
                th_deg = float(theta_field[i])
            else:
                th_deg = float(theta_field)
            th_clamped = float(np.clip(th_deg, theta_min_deg, theta_max_deg))
            delta_u = (b / 2.0) * np.tan(np.radians(th_clamped))
            w_w = w_top_i - 2.0 * delta_u
            # Safety bounds: keep waist positive and ensure adjacent horizontal connector space
            w_w = float(np.clip(w_w, 0.10 * a_i, 0.85 * a_i))
            theta_cols.append(th_clamped)
            w_waist_cols.append(w_w)
        elif callable(w_waist_ratio):
            ratio = float(w_waist_ratio(u_c))
            w_w = float(np.clip(ratio * a_i, 0.10 * a_i, 0.85 * a_i))
            delta_u = (w_top_i - w_w) / 2.0
            th_deg = float(np.degrees(np.arctan(delta_u / (b / 2.0))))
            theta_cols.append(th_deg)
            w_waist_cols.append(w_w)
        else:
            w_w = float(w_waist_ratio) * a_i
            delta_u = (w_top_i - w_w) / 2.0
            th_deg = float(np.degrees(np.arctan(delta_u / (b / 2.0))))
            theta_cols.append(th_deg)
            w_waist_cols.append(w_w)

    metadata = {
        "topology": "reentrant",
        "variant": variant,
        "domain_width": float(domain_width),
        "domain_height": float(domain_height),
        "y_base": float(y_base),
        "n_circumferential": int(cols),
        "m_vertical": int(m_vertical),
        "a_cell_width_mm": float(a_mean),
        "a_cell_width_min_mm": float(np.min(col_widths)),
        "a_cell_width_max_mm": float(np.max(col_widths)),
        "b_cell_height_mm": float(b),
        "w_top_min_mm": float(np.min(w_top_cols)),
        "w_top_max_mm": float(np.max(w_top_cols)),
        "w_waist_min_mm": float(np.min(w_waist_cols)),
        "w_waist_max_mm": float(np.max(w_waist_cols)),
        "theta_min_deg": float(np.min(theta_cols)),
        "theta_max_deg": float(np.max(theta_cols)),
        "theta_mean_deg": float(np.mean(theta_cols)),
    }

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(cols):
        u_c = centers[i]
        w_top_i = w_top_cols[i]
        w_waist_i = w_waist_cols[i]
        u_tl = u_c - w_top_i / 2.0
        u_tr = u_c + w_top_i / 2.0
        u_wl = u_c - w_waist_i / 2.0
        u_wr = u_c + w_waist_i / 2.0

        for j in range(m_vertical):
            y_bot = y_base + j * b
            y_mid = y_bot + h_waist_y
            y_top = y_bot + b

            p_tl = np.array([u_tl, y_top])
            p_tr = np.array([u_tr, y_top])
            p_bl = np.array([u_tl, y_bot])
            p_br = np.array([u_tr, y_bot])
            p_wl = np.array([u_wl, y_mid])
            p_wr = np.array([u_wr, y_mid])

            # 1. Top and bottom horizontal boundary caps
            raw_segments.append((p_tl, p_tr))
            raw_segments.append((p_bl, p_br))

            # 2. Slanted re-entrant struts
            raw_segments.append((p_tl, p_wl))
            raw_segments.append((p_bl, p_wl))
            raw_segments.append((p_tr, p_wr))
            raw_segments.append((p_br, p_wr))

            # 3. Horizontal waist connector to the right adjacent cell
            if periodic_x or (i < cols - 1):
                next_i = (i + 1) % cols if periodic_x else (i + 1)
                w_waist_next = w_waist_cols[next_i]
                u_next_c = centers[next_i]
                u_next_wl = u_next_c - w_waist_next / 2.0
                p_next_wl = np.array([u_next_wl, y_mid])
                raw_segments.append((p_wr, p_next_wl))

            # 4. Variant-specific reinforcement struts
            if variant == "type_a":
                raw_segments.append((p_wl, p_wr))
            elif variant == "type_b":
                p_c_bot = np.array([u_c, y_bot])
                p_c_top = np.array([u_c, y_top])
                raw_segments.append((p_c_bot, p_c_top))

    # Deduplicate without box slicing to preserve complete full unit cells
    period = domain_width if periodic_x else None
    unique = dedupe_2d_segments(raw_segments, period_x=period)
    metadata["n_segments"] = len(unique)
    return unique, metadata


# =========================================================================
# Regular 2D Triangle and Square Patterns (Internalized from tri_sq_patterns)
# =========================================================================

TRI_TOPOLOGIES = ("Tetrahedral", "Icosahedral", "Kelvin", "Tesseract", "Rhombic")
SQ_TOPOLOGIES = ("Grid", "Icosahedral", "Kelvin", "Tesseract")


def _triangle_grid(side: float, height: float, i_range: range, j_range: range) -> list[np.ndarray]:
    triangles: list[np.ndarray] = []

    def get_p(i: int, j: int) -> np.ndarray:
        cx = i * side + (j % 2) * (side / 2.0)
        cy = j * height
        return np.array([cx, cy], dtype=np.float64)

    for j in j_range:
        for i in i_range:
            p_ij = get_p(i, j)
            p_ip1_j = get_p(i + 1, j)
            p_ijp1 = get_p(i, j + 1)
            p_ip1_jp1 = get_p(i + 1, j + 1)
            if j % 2 == 0:
                triangles.append(np.array([p_ij, p_ip1_j, p_ijp1]))
                triangles.append(np.array([p_ip1_j, p_ip1_jp1, p_ijp1]))
            else:
                triangles.append(np.array([p_ij, p_ip1_j, p_ip1_jp1]))
                triangles.append(np.array([p_ij, p_ip1_jp1, p_ijp1]))
    return triangles


def get_triangle_segments(
    topo: str,
    side: float,
    height: float | None = None,
    *,
    extent_cells: int = 7,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flat triangular-cell strut patterns."""
    if height is None:
        height = side * np.sqrt(3.0) / 2.0
    i_range = range(-extent_cells, extent_cells)
    j_range = range(-extent_cells, extent_cells)
    triangles = _triangle_grid(side, height, i_range, j_range)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    topo_norm = topo.capitalize()
    if topo_norm == "Tetrahedral":
        for tri in triangles:
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
    elif topo_norm == "Icosahedral":
        for tri in triangles:
            m0 = (tri[0] + tri[1]) / 2.0
            m1 = (tri[1] + tri[2]) / 2.0
            m2 = (tri[2] + tri[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m0))
    elif topo_norm == "Kelvin":
        for tri in triangles:
            p01_a = (2 * tri[0] + tri[1]) / 3.0
            p01_b = (tri[0] + 2 * tri[1]) / 3.0
            p12_a = (2 * tri[1] + tri[2]) / 3.0
            p12_b = (tri[1] + 2 * tri[2]) / 3.0
            p20_a = (2 * tri[2] + tri[0]) / 3.0
            p20_b = (tri[2] + 2 * tri[0]) / 3.0
            segments.extend([
                (p01_a, p20_b), (p20_b, p20_a), (p20_a, p12_b),
                (p12_b, p12_a), (p12_a, p01_b), (p01_b, p01_a),
                (p01_a, p01_b), (p12_a, p12_b), (p20_a, p20_b),
            ])
    elif topo_norm == "Tesseract":
        for tri in triangles:
            centroid = np.mean(tri, axis=0)
            inscribed = centroid + 0.5 * (tri - centroid)
            segments.append((tri[0], tri[1]))
            segments.append((tri[1], tri[2]))
            segments.append((tri[2], tri[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[0]))
            for i in range(3):
                segments.append((tri[i], inscribed[i]))
    elif topo_norm == "Rhombic":
        from collections import defaultdict
        centroids = [np.mean(tri, axis=0) for tri in triangles]
        edge_to_tri: dict[tuple[tuple[int, int], tuple[int, int]], list[int]] = defaultdict(list)
        inv = 1e3
        for idx, tri in enumerate(triangles):
            for k in range(3):
                p1, p2 = tri[k], tri[(k + 1) % 3]
                k1 = (int(round(p1[0] * inv)), int(round(p1[1] * inv)))
                k2 = (int(round(p2[0] * inv)), int(round(p2[1] * inv)))
                ek = (k1, k2) if k1 < k2 else (k2, k1)
                edge_to_tri[ek].append(idx)
        for tri_indices in edge_to_tri.values():
            if len(tri_indices) == 2:
                segments.append((centroids[tri_indices[0]], centroids[tri_indices[1]]))
    else:
        raise ValueError(f"Unknown triangle topology: {topo!r}")

    return dedupe_2d_segments(segments)


def get_square_segments(
    topo: str,
    side: float,
    *,
    extent_cells: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Flat square-cell strut patterns."""
    squares: list[np.ndarray] = []
    for row in range(-extent_cells, extent_cells):
        for col in range(-extent_cells, extent_cells):
            cx = col * side
            cy = row * side
            v0 = np.array([cx - side / 2.0, cy - side / 2.0], dtype=np.float64)
            v1 = np.array([cx + side / 2.0, cy - side / 2.0], dtype=np.float64)
            v2 = np.array([cx + side / 2.0, cy + side / 2.0], dtype=np.float64)
            v3 = np.array([cx - side / 2.0, cy + side / 2.0], dtype=np.float64)
            squares.append(np.array([v0, v1, v2, v3]))

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    topo_norm = topo.capitalize()
    if topo_norm == "Grid":
        for sq in squares:
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
    elif topo_norm == "Icosahedral":
        for sq in squares:
            m0 = (sq[0] + sq[1]) / 2.0
            m1 = (sq[1] + sq[2]) / 2.0
            m2 = (sq[2] + sq[3]) / 2.0
            m3 = (sq[3] + sq[0]) / 2.0
            segments.append((m0, m1))
            segments.append((m1, m2))
            segments.append((m2, m3))
            segments.append((m3, m0))
    elif topo_norm == "Kelvin":
        for sq in squares:
            p01_a = (2 * sq[0] + sq[1]) / 3.0
            p01_b = (sq[0] + 2 * sq[1]) / 3.0
            p12_a = (2 * sq[1] + sq[2]) / 3.0
            p12_b = (sq[1] + 2 * sq[2]) / 3.0
            p23_a = (2 * sq[2] + sq[3]) / 3.0
            p23_b = (sq[2] + 2 * sq[3]) / 3.0
            p30_a = (2 * sq[3] + sq[0]) / 3.0
            p30_b = (sq[3] + 2 * sq[0]) / 3.0
            segments.extend([
                (p01_a, p30_b), (p30_b, p30_a), (p30_a, p23_b),
                (p23_b, p23_a), (p23_a, p12_b), (p12_b, p12_a),
                (p12_a, p01_b), (p01_b, p01_a), (p01_a, p01_b),
                (p12_a, p12_b), (p23_a, p23_b), (p30_a, p30_b),
            ])
    elif topo_norm == "Tesseract":
        for sq in squares:
            centroid = np.mean(sq, axis=0)
            inscribed = centroid + 0.5 * (sq - centroid)
            segments.append((sq[0], sq[1]))
            segments.append((sq[1], sq[2]))
            segments.append((sq[2], sq[3]))
            segments.append((sq[3], sq[0]))
            segments.append((inscribed[0], inscribed[1]))
            segments.append((inscribed[1], inscribed[2]))
            segments.append((inscribed[2], inscribed[3]))
            segments.append((inscribed[3], inscribed[0]))
            for i in range(4):
                segments.append((sq[i], inscribed[i]))
    else:
        raise ValueError(f"Unknown square topology: {topo!r}")

    return dedupe_2d_segments(segments)


# =========================================================================
# Variable Cell Size Grid Generator (Continuous Coordinate Warping)
# =========================================================================

def tessellate_graded_cell_domain(
    domain_width: float,
    domain_height: float,
    cell_size_min: float = 4.0,
    cell_size_max: float = 12.0,
    gradient_axis: str = "xy",
    topology: str = "square",
    warp_power: float = 2.0,
    x_coords: Sequence[float] | None = None,
    y_coords: Sequence[float] | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[float], dict]:
    """
    Generate a 2D lattice where cell size L(x, y) varies continuously across the domain,
    or follows explicit discrete coordinate series (e.g. curvature-discretized column boundaries).

    Returns
    -------
    segments : list of (ndarray, ndarray)
        2D line segments.
    local_cell_sizes : list of float
        Local cell length L_seg for each segment.
    metadata : dict
        Generation parameters.
    """
    W = float(domain_width)
    H = float(domain_height)
    l_min = float(cell_size_min)
    l_max = float(cell_size_max)

    # Build 1D coordinate series with varying spacing
    def _graded_coords(length: float, min_step: float, max_step: float, power: float) -> np.ndarray:
        # Generate non-linear parameterization u in [0, 1]
        n_est = int(np.ceil(length / (0.5 * (min_step + max_step))))
        n_steps = max(n_est, 4)
        t = np.linspace(0.0, 1.0, n_steps + 1)
        # Smooth S-curve or power warp:
        # t^p smoothly spreads or clusters coordinates
        if power == 1.0:
            coords = t * length
        else:
            # Concentrated in center or at edge
            warped_t = t ** power
            coords = warped_t * length
        return coords

    if x_coords is not None:
        xs = np.asarray(x_coords, dtype=np.float64)
    else:
        p_x = warp_power if "x" in gradient_axis.lower() else 1.0
        xs = _graded_coords(W, l_min, l_max, p_x)

    if y_coords is not None:
        ys = np.asarray(y_coords, dtype=np.float64)
    elif x_coords is not None:
        mean_step = float(np.mean(np.diff(xs)))
        ny = max(int(np.round(H / mean_step)), 2)
        ys = np.linspace(0.0, H, ny + 1)
    else:
        p_y = warp_power if "y" in gradient_axis.lower() else 1.0
        ys = _graded_coords(H, l_min, l_max, p_y)

    nx = len(xs)
    ny = len(ys)

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    cell_lengths: list[float] = []

    # Grid nodes P[i, j]
    for j in range(ny):
        for i in range(nx):
            p_cur = np.array([xs[i], ys[j]], dtype=np.float64)

            # Horizontal edge
            if i + 1 < nx:
                p_right = np.array([xs[i + 1], ys[j]], dtype=np.float64)
                raw_segments.append((p_cur, p_right))
                cell_lengths.append(float(np.linalg.norm(p_right - p_cur)))

            # Vertical edge
            if j + 1 < ny:
                p_up = np.array([xs[i], ys[j + 1]], dtype=np.float64)
                raw_segments.append((p_cur, p_up))
                cell_lengths.append(float(np.linalg.norm(p_up - p_cur)))

            # Diagonal edge for triangular grid
            if "tri" in topology.lower() and i + 1 < nx and j + 1 < ny:
                p_diag = np.array([xs[i + 1], ys[j + 1]], dtype=np.float64)
                raw_segments.append((p_cur, p_diag))
                cell_lengths.append(float(np.linalg.norm(p_diag - p_cur)))

    meta = {
        "domain_width": W,
        "domain_height": H,
        "cell_size_min": l_min,
        "cell_size_max": l_max,
        "n_segments": len(raw_segments),
        "topology": topology,
    }
    return raw_segments, cell_lengths, meta

