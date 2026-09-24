"""
Graphite Explicit — Custom Truss Cell Registry and Importer

Enables importing and registering custom structural unit cells into
Graphite's standard explicit lattice pipeline (hex_rules / hex_topology_module).
Custom cells are defined by canonical nodes in [0, 1]^3 and edge struts,
and are automatically trilinearly warped across conformal hex grids with
full support for variable strut thickness grading and sizing solvers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import numpy as np

from .hex_topology_module import (
    HexTopologyRule,
    _HEX_RULES_REGISTRY,
    SKIN_MODE_NONE,
    SKIN_MODE_CORNER_EDGE_CAGE,
    CONFORM_DOF_CORNERS,
)


def trilinear_warp(canonical_pts: np.ndarray, hex_corners: np.ndarray) -> np.ndarray:
    """
    Warp points from canonical unit coordinates [0, 1]^3 to deformed hex coordinates.

    Args:
        canonical_pts: (N, 3) points in [0, 1]^3.
        hex_corners: (8, 3) hex corner coordinates in canonical order:
                     0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0),
                     4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)

    Returns:
        (N, 3) physical coordinates.
    """
    pts = np.asarray(canonical_pts, dtype=np.float64)
    coords = np.asarray(hex_corners, dtype=np.float64)

    u = pts[:, 0]
    v = pts[:, 1]
    w = pts[:, 2]

    weights = np.column_stack([
        (1.0 - u) * (1.0 - v) * (1.0 - w),
        u * (1.0 - v) * (1.0 - w),
        u * v * (1.0 - w),
        (1.0 - u) * v * (1.0 - w),
        (1.0 - u) * (1.0 - v) * w,
        u * (1.0 - v) * w,
        u * v * w,
        (1.0 - u) * v * w,
    ])
    return np.dot(weights, coords)


def make_hex_builder_from_skeleton(
    canonical_nodes: np.ndarray,
    struts: np.ndarray,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Create a hex cell builder function from a canonical skeleton.
    """
    nodes_can = np.asarray(canonical_nodes, dtype=np.float64)
    struts_arr = np.asarray(struts, dtype=np.int64)

    def builder(hex_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        warped = trilinear_warp(nodes_can, hex_coords)
        return warped, struts_arr.copy()

    return builder


def register_custom_hex_truss(
    name: str,
    canonical_nodes: np.ndarray,
    struts: np.ndarray,
    cage_mode: str = "surface_dual",
    skin_mode: str = SKIN_MODE_NONE,
) -> HexTopologyRule:
    """
    Register a custom unit cell skeleton into Graphite's standard explicit engine.

    Once registered, the cell can be used directly in `generate_conformal_lattice(..., rule_name=name)`.

    Args:
        name: Unique rule name identifier (e.g. 'diamond_truss', 'x_braced_cube').
        canonical_nodes: (N, 3) coordinates normalized in [0, 1]^3.
        struts: (S, 2) edge connectivity pairs.
        cage_mode: Surface cage mode.
        skin_mode: Boundary skin mode.

    Returns:
        The registered HexTopologyRule instance.
    """
    rule_key = str(name).strip().lower()
    builder = make_hex_builder_from_skeleton(canonical_nodes, struts)

    rule = HexTopologyRule(
        name=rule_key,
        builder=builder,
        cage_mode=cage_mode,
        skin_mode=skin_mode,
        conform_dofs=frozenset({CONFORM_DOF_CORNERS}),
    )
    _HEX_RULES_REGISTRY[rule_key] = rule
    return rule


def register_custom_truss_from_json(json_path: str | Path) -> HexTopologyRule:
    """
    Load and register a custom unit cell skeleton from a JSON file.

    JSON schema:
    {
        "name": "my_cell",
        "nodes": [[x0, y0, z0], ...],
        "struts": [[u0, v0], ...]
    }
    """
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    name = data["name"]
    nodes = np.asarray(data["nodes"], dtype=np.float64)
    struts = np.asarray(data["struts"], dtype=np.int64)

    return register_custom_hex_truss(name, nodes, struts)
