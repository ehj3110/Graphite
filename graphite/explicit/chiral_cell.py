# -*- coding: utf-8 -*-
"""
Graphite Explicit Engine - Chiral and Anti-Chiral Metamaterial Unit Cells.

Implements classic 2D and 3D chiral mechanical metamaterials based on:
- Prall & Lakes (1997), "Properties of a chiral honeycomb with a poisson's ratio of -1"
- Spadoni, Ruzzene, et al. (2006), "Phononic properties of chiral and anti-chiral honeycombs"
- Alderson et al. (2010), "Elastic properties of chiral and anti-chiral honeycombs"

Topologies:
1. Tetra-Chiral (4 tangent ligaments per circular node on a square lattice)
2. Tri-Chiral (3 tangent ligaments per circular node on a hexagonal/triangular lattice)
3. Anti-Tetra-Chiral (adjacent circular nodes connected on the same side of the ligament)
4. Anti-Tri-Chiral (adjacent circular nodes connected on the same side of the ligament)
"""
from __future__ import annotations

import numpy as np


def generate_tetrachiral_cell(
    L: float = 10.0,
    r: float = 2.0,
    t: float = 1.0,
    n_circle_segments: int = 16,
    chiral: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate the nodes and strut edges for a single tetra-chiral unit cell.

    Parameters
    ----------
    L : float
        Length of each straight ligament in mm.
    r : float
        Radius of the circular central node in mm.
    t : float
        Nominal strut / ligament thickness in mm.
    n_circle_segments : int
        Number of polygonal chord segments used to discretize the circular node.
    chiral : bool
        If True, generates the standard chiral cell (adjacent nodes rotate in same sense).
        If False, generates anti-tetrachiral configuration.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        2D node coordinates.
    struts : ndarray, shape (S, 2)
        Edge index pairs connecting the nodes.
    metadata : dict
        Geometric parameters and lattice pitch D = sqrt(L^2 + 4*r^2).
    """
    if L <= 0.0 or r <= 0.0:
        raise ValueError("L and r must be strictly positive.")

    D = np.sqrt(L**2 + 4.0 * r**2)
    alpha = np.arctan(2.0 * r / L)

    node_list: list[np.ndarray] = []
    strut_list: list[tuple[int, int]] = []

    # 1. Circular central node at origin (0, 0)
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segments, endpoint=False)
    circle_node_indices = []
    for ang in circle_angles:
        pt = r * np.array([np.cos(ang), np.sin(ang)])
        idx = len(node_list)
        node_list.append(pt)
        circle_node_indices.append(idx)

    # Connect circular perimeter chords
    for k in range(n_circle_segments):
        i1 = circle_node_indices[k]
        i2 = circle_node_indices[(k + 1) % n_circle_segments]
        strut_list.append((i1, i2))

    # 2. Four tangent ligaments
    for k in range(4):
        if chiral:
            th_base = -alpha + k * (np.pi / 2.0)
            p_start = r * np.array([-np.sin(th_base), np.cos(th_base)])
            p_end = p_start + L * np.array([np.cos(th_base), np.sin(th_base)])
        else:
            # Anti-chiral: orthogonal tangent orientation
            ang = k * (np.pi / 2.0)
            p_start = r * np.array([-np.sin(ang), np.cos(ang)])
            p_end = p_start + L * np.array([np.cos(ang), np.sin(ang)])

        idx_start = len(node_list)
        node_list.append(p_start)
        idx_end = len(node_list)
        node_list.append(p_end)
        strut_list.append((idx_start, idx_end))

    nodes = np.array(node_list, dtype=np.float64)
    struts = np.array(strut_list, dtype=np.int64)
    metadata = {
        "type": "tetra_chiral" if chiral else "anti_tetra_chiral",
        "L": float(L),
        "r": float(r),
        "t": float(t),
        "D_pitch": float(D),
        "alpha_rad": float(alpha),
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
    Generate the nodes and strut edges for a single tri-chiral unit cell.

    Parameters
    ----------
    L : float
        Length of each straight ligament in mm.
    r : float
        Radius of the circular central node in mm.
    t : float
        Nominal strut / ligament thickness in mm.
    n_circle_segments : int
        Number of polygonal chord segments used to discretize the circular node.
    chiral : bool
        If True, generates standard tri-chiral cell.
        If False, generates anti-trichiral cell.

    Returns
    -------
    nodes : ndarray, shape (N, 2)
        2D node coordinates.
    struts : ndarray, shape (S, 2)
        Edge index pairs connecting the nodes.
    metadata : dict
        Geometric parameters and triangular lattice pitch D = sqrt(L^2 + 4*r^2).
    """
    if L <= 0.0 or r <= 0.0:
        raise ValueError("L and r must be strictly positive.")

    D = np.sqrt(L**2 + 4.0 * r**2)
    phi = np.arcsin(2.0 * r / D)

    node_list: list[np.ndarray] = []
    strut_list: list[tuple[int, int]] = []

    # 1. Circular central node at origin (0, 0)
    circle_angles = np.linspace(0.0, 2.0 * np.pi, n_circle_segments, endpoint=False)
    circle_node_indices = []
    for ang in circle_angles:
        pt = r * np.array([np.cos(ang), np.sin(ang)])
        idx = len(node_list)
        node_list.append(pt)
        circle_node_indices.append(idx)

    for k in range(n_circle_segments):
        i1 = circle_node_indices[k]
        i2 = circle_node_indices[(k + 1) % n_circle_segments]
        strut_list.append((i1, i2))

    # 2. Three tangent ligaments spaced at 120-degree intervals
    for k in range(3):
        if chiral:
            th_base = phi + k * (2.0 * np.pi / 3.0)
            tang_ang = th_base - phi
        else:
            tang_ang = k * (2.0 * np.pi / 3.0)

        p_start = r * np.array([-np.sin(tang_ang), np.cos(tang_ang)])
        p_end = p_start + L * np.array([np.cos(tang_ang), np.sin(tang_ang)])

        idx_start = len(node_list)
        node_list.append(p_start)
        idx_end = len(node_list)
        node_list.append(p_end)
        strut_list.append((idx_start, idx_end))

    nodes = np.array(node_list, dtype=np.float64)
    struts = np.array(strut_list, dtype=np.int64)
    metadata = {
        "type": "tri_chiral" if chiral else "anti_tri_chiral",
        "L": float(L),
        "r": float(r),
        "t": float(t),
        "D_pitch": float(D),
        "phi_rad": float(phi),
        "phi_deg": float(np.degrees(phi)),
    }
    return nodes, struts, metadata
