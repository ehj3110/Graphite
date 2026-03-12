"""
Universal Lattice Router.

Dispatches an array of elements (tetrahedral or hexahedral) to the correct
micro-rule set based on the number of nodes per element, then performs a
global node-merge pass to produce a single connected lattice graph.

Supported element shapes:
    4-node  → tet rules: voronoi, kagome, icosahedral, rhombic
    8-node  → hex rules: grid, octahedral, star, octet
"""

from __future__ import annotations

import numpy as np

from .hex_rules import (
    apply_hex_grid,
    apply_hex_octahedral,
    apply_hex_octet_truss,
    apply_hex_star,
)
from .tet_rules import (
    apply_icosahedral_rule,
    apply_kagome_rule,
    apply_rhombic_rule,
    apply_voronoi_rule,
)

_TET_RULES = {
    "voronoi":     apply_voronoi_rule,
    "kagome":      apply_kagome_rule,
    "icosahedral": apply_icosahedral_rule,
    "rhombic":     apply_rhombic_rule,
}

_HEX_RULES = {
    "grid":       apply_hex_grid,
    "octahedral": apply_hex_octahedral,
    "star":       apply_hex_star,
    "octet":      apply_hex_octet_truss,
}

_RULE_SETS: dict[int, dict] = {
    4: _TET_RULES,
    8: _HEX_RULES,
}


def generate_universal_lattice(
    elements_coords: np.ndarray,
    rule_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a globally-merged lattice from an array of elements.

    Parameters
    ----------
    elements_coords : ndarray, shape (N, nodes_per_elem, 3)
        Each row is one element; N elements, each with ``nodes_per_elem``
        corner coordinates.  Supported shapes per element: 4 (tet) or 8 (hex).
    rule_name : str
        Micro-rule to apply.  Available rules depend on element type:
        - tet  (4 nodes): 'voronoi', 'kagome', 'icosahedral', 'rhombic'
        - hex  (8 nodes): 'grid', 'octahedral', 'star', 'octet'

    Returns
    -------
    nodes  : ndarray, shape (M, 3), float64
    struts : ndarray, shape (S, 2), int64
    """
    elements_coords = np.asarray(elements_coords, dtype=np.float64)
    if elements_coords.ndim != 3 or elements_coords.shape[2] != 3:
        raise ValueError(
            "elements_coords must have shape (N, nodes_per_elem, 3), "
            f"got {elements_coords.shape}."
        )

    nodes_per_elem: int = elements_coords.shape[1]
    rule_key = rule_name.strip().lower()

    if nodes_per_elem not in _RULE_SETS:
        raise ValueError(
            f"Unsupported element size {nodes_per_elem}. "
            f"Must be one of {sorted(_RULE_SETS)}."
        )
    rule_map = _RULE_SETS[nodes_per_elem]

    if rule_key not in rule_map:
        raise ValueError(
            f"Rule '{rule_name}' not available for {nodes_per_elem}-node elements. "
            f"Available: {sorted(rule_map)}."
        )
    rule_fn = rule_map[rule_key]

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()

    for elem_coords in elements_coords:
        local_nodes, local_struts = rule_fn(elem_coords)

        local_to_global: list[int] = []
        for node in local_nodes:
            key = (
                round(float(node[0]), 4),
                round(float(node[1]), 4),
                round(float(node[2]), 4),
            )
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(node.copy())
            local_to_global.append(idx)

        for a, b in local_struts:
            ga = local_to_global[int(a)]
            gb = local_to_global[int(b)]
            if ga != gb:
                strut_set.add((min(ga, gb), max(ga, gb)))

    nodes = (
        np.vstack(nodes_list) if nodes_list
        else np.empty((0, 3), dtype=np.float64)
    )
    struts = (
        np.array(sorted(strut_set), dtype=np.int64) if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts
