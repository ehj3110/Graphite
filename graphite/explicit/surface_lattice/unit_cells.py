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
        m_rows = int(np.ceil(domain_height / D))
        metadata = {
            "topology": "tetra_chiral" if is_chiral else "anti_tetra_chiral",
            "pitch_D": float(D),
            "ligament_L": float(L),
            "r_node": float(r_node),
            "alpha_deg": float(np.degrees(alpha)),
            "n_circumferential": n_circumferential,
            "m_rows": m_rows,
        }

        for i in range(-1, n_circumferential + 2):
            for j in range(-1, m_rows + 2):
                c = np.array([i * D, y_base + j * D])

                # Circle chords
                for k in range(n_circle_segs):
                    a1 = circle_angles[k]
                    a2 = circle_angles[(k + 1) % n_circle_segs]
                    p1 = c + r_node * np.array([np.cos(a1), np.sin(a1)])
                    p2 = c + r_node * np.array([np.cos(a2), np.sin(a2)])
                    raw_segments.append((p1, p2))

                # Four tangent ligaments
                for k in range(4):
                    if is_chiral:
                        th_base = -alpha + k * (np.pi / 2.0)
                        p_start = c + r_node * np.array([-np.sin(th_base), np.cos(th_base)])
                        p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
                    else:
                        ang = k * (np.pi / 2.0)
                        p_start = c + r_node * np.array([-np.sin(ang), np.cos(ang)])
                        p_end = p_start + L * np.array([np.cos(ang), np.sin(ang)])
                    raw_segments.append((p_start, p_end))

    elif topo_clean == "tri":
        phi = np.arcsin(2.0 * r_node / D)
        a1 = np.array([D, 0.0])
        a2 = np.array([D * 0.5, D * np.sqrt(3.0) / 2.0])
        m_rows = int(np.ceil(domain_height / a2[1]))
        metadata = {
            "topology": "tri_chiral" if is_chiral else "anti_tri_chiral",
            "pitch_D": float(D),
            "ligament_L": float(L),
            "r_node": float(r_node),
            "phi_deg": float(np.degrees(phi)),
            "n_circumferential": n_circumferential,
            "m_rows": m_rows,
        }

        for i in range(-2, n_circumferential + 3):
            for j in range(-2, m_rows + 3):
                c = i * a1 + j * a2 + np.array([0.0, y_base])

                # Circle chords
                for k in range(n_circle_segs):
                    a1_ang = circle_angles[k]
                    a2_ang = circle_angles[(k + 1) % n_circle_segs]
                    p1 = c + r_node * np.array([np.cos(a1_ang), np.sin(a1_ang)])
                    p2 = c + r_node * np.array([np.cos(a2_ang), np.sin(a2_ang)])
                    raw_segments.append((p1, p2))

                # Three tangent ligaments
                for k in range(3):
                    if is_chiral:
                        th_base = phi + k * (2.0 * np.pi / 3.0)
                        tang_ang = th_base - phi
                    else:
                        tang_ang = k * (2.0 * np.pi / 3.0)
                    p_start = c + r_node * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
                    p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])
                    raw_segments.append((p_start, p_end))

    else:
        raise ValueError(f"Unsupported chiral topology '{topology}'. Expected 'tetra' or 'tri'.")

    # Clip vertically to [y_base, y_base + domain_height]
    from shapely.geometry import box, LineString
    y_band = box(-D * 2, y_base, domain_width + D * 2, y_base + domain_height)
    clipped: list[tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in raw_segments:
        line = LineString([p0, p1])
        cy = line.intersection(y_band)
        if not cy.is_empty:
            if cy.geom_type == "LineString":
                clipped.append((np.array(cy.coords[0]), np.array(cy.coords[1])))
            elif cy.geom_type == "MultiLineString":
                for sub in cy.geoms:
                    clipped.append((np.array(sub.coords[0]), np.array(sub.coords[1])))

    period = domain_width if periodic_x else None
    unique = dedupe_2d_segments(clipped, period_x=period)
    metadata["n_segments"] = len(unique)
    return unique, metadata


def tessellate_reentrant_domain(
    domain_width: float,
    domain_height: float,
    n_circumferential: int = 10,
    m_vertical: int | None = None,
    w_top_ratio: float = 0.65,
    w_waist_ratio: float = 0.30,
    variant: str = "base",
    periodic_x: bool = True,
    y_base: float = 0.0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    Generate 2D line segments for Chen et al. (2020) re-entrant auxetic lattice on a domain.

    Parameters
    ----------
    domain_width : float
        Domain width in mm (or circumference 2*pi*R_mid for cylinders).
    domain_height : float
        Domain height in mm.
    n_circumferential : int
        Number of periodic cell columns across domain width.
    m_vertical : int, optional
        Number of cell rows along height. If None, auto-calculated from cell aspect ratio.
    w_top_ratio : float
        Width of top/bottom caps as fraction of cell width a (default 0.65).
    w_waist_ratio : float
        Width of waist as fraction of cell width a (default 0.30).
    variant : str
        'base' (re-entrant hexagonal cell), 'type_a' (horizontal waist rib), or 'type_b' (vertical central rib).
    periodic_x : bool
        If True, enforces closed-form periodic wrapping for cylinders.
    y_base : float
        Y coordinate offset of bottom boundary.

    Returns
    -------
    segments : list of tuple(ndarray, ndarray)
        Deduplicated 2D line segments.
    metadata : dict
        Geometric parameters.
    """
    a = domain_width / float(n_circumferential)
    if m_vertical is None:
        m_vertical = max(int(round(domain_height / a)), 1)
    b = domain_height / float(m_vertical)
    w_top = w_top_ratio * a
    w_waist = w_waist_ratio * a
    h_waist_y = b / 2.0

    delta_u = (w_top - w_waist) / 2.0
    theta_from_vertical_deg = float(np.degrees(np.arctan(delta_u / (b / 2.0))))
    theta_from_horizontal_deg = float(90.0 - theta_from_vertical_deg)

    metadata = {
        "topology": "reentrant",
        "variant": variant,
        "domain_width": float(domain_width),
        "domain_height": float(domain_height),
        "y_base": float(y_base),
        "n_circumferential": int(n_circumferential),
        "m_vertical": int(m_vertical),
        "a_cell_width_mm": float(a),
        "b_cell_height_mm": float(b),
        "w_top_mm": float(w_top),
        "w_waist_mm": float(w_waist),
        "theta_from_vertical_deg": theta_from_vertical_deg,
        "theta_from_horizontal_deg": theta_from_horizontal_deg,
    }

    raw_segments: list[tuple[np.ndarray, np.ndarray]] = []
    cols = n_circumferential if periodic_x else n_circumferential + 1

    for i in range(cols):
        u_c = (i + 0.5) * a
        u_tl = u_c - w_top / 2.0
        u_tr = u_c + w_top / 2.0
        u_wl = u_c - w_waist / 2.0
        u_wr = u_c + w_waist / 2.0

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
            if periodic_x or (i < n_circumferential - 1):
                u_next_c = (i + 1.5) * a
                u_next_wl = u_next_c - w_waist / 2.0
                p_next_wl = np.array([u_next_wl, y_mid])
                raw_segments.append((p_wr, p_next_wl))

            # 4. Variant-specific reinforcement struts
            if variant == "type_a":
                raw_segments.append((p_wl, p_wr))
            elif variant == "type_b":
                p_c_bot = np.array([u_c, y_bot])
                p_c_top = np.array([u_c, y_top])
                raw_segments.append((p_c_bot, p_c_top))

    from shapely.geometry import box, LineString
    x_min_clip = -a if periodic_x else 0.0
    x_max_clip = domain_width + a if periodic_x else domain_width
    clip_band = box(x_min_clip, y_base, x_max_clip, y_base + domain_height)
    clipped: list[tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in raw_segments:
        line = LineString([p0, p1])
        cy = line.intersection(clip_band)
        if not cy.is_empty:
            if cy.geom_type == "LineString":
                clipped.append((np.array(cy.coords[0]), np.array(cy.coords[1])))
            elif cy.geom_type == "MultiLineString":
                for sub in cy.geoms:
                    clipped.append((np.array(sub.coords[0]), np.array(sub.coords[1])))

    period = domain_width if periodic_x else None
    unique = dedupe_2d_segments(clipped, period_x=period)
    metadata["n_segments"] = len(unique)
    return unique, metadata
