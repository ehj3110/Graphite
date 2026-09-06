# -*- coding: utf-8 -*-
"""Unit tests for chiral and anti-chiral mechanical metamaterial unit cells."""
from __future__ import annotations

import numpy as np
import pytest

from graphite.explicit.chiral_cell import generate_tetrachiral_cell, generate_trichiral_cell


def test_tetrachiral_cell_geometry():
    """Verify tetra-chiral unit cell topology and node/strut counts."""
    L = 10.0
    r = 2.0
    n_circ = 16
    nodes, struts, meta = generate_tetrachiral_cell(L=L, r=r, n_circle_segments=n_circ, chiral=True)

    # 16 circle vertices + 4*2 ligament vertices = 24 nodes
    assert nodes.shape == (24, 2)
    # 16 circle chord edges + 4 ligament edges = 20 struts
    assert struts.shape == (20, 2)

    # Verify pitch D = sqrt(L^2 + 4*r^2)
    expected_D = np.sqrt(L**2 + 4.0 * r**2)
    assert np.isclose(meta["D_pitch"], expected_D)

    # Verify ligament lengths
    ligament_indices = struts[n_circ:]
    for i_start, i_end in ligament_indices:
        len_i = np.linalg.norm(nodes[i_end] - nodes[i_start])
        assert np.isclose(len_i, L, atol=1e-5)


def test_trichiral_cell_geometry():
    """Verify tri-chiral unit cell topology and node/strut counts."""
    L = 10.0
    r = 2.0
    n_circ = 16
    nodes, struts, meta = generate_trichiral_cell(L=L, r=r, n_circle_segments=n_circ, chiral=True)

    # 16 circle vertices + 3*2 ligament vertices = 22 nodes
    assert nodes.shape == (22, 2)
    # 16 circle chord edges + 3 ligament edges = 19 struts
    assert struts.shape == (19, 2)

    # Verify pitch D = sqrt(L^2 + 4*r^2)
    expected_D = np.sqrt(L**2 + 4.0 * r**2)
    assert np.isclose(meta["D_pitch"], expected_D)

    # Verify ligament lengths
    ligament_indices = struts[n_circ:]
    for i_start, i_end in ligament_indices:
        len_i = np.linalg.norm(nodes[i_end] - nodes[i_start])
        assert np.isclose(len_i, L, atol=1e-5)


def test_antichiral_cell_geometry():
    """Verify anti-chiral variants generate correctly."""
    nodes_tetra, struts_tetra, meta_tetra = generate_tetrachiral_cell(chiral=False)
    assert meta_tetra["type"] == "anti_tetra_chiral"
    assert len(struts_tetra) == 20

    nodes_tri, struts_tri, meta_tri = generate_trichiral_cell(chiral=False)
    assert meta_tri["type"] == "anti_tri_chiral"
    assert len(struts_tri) == 19
