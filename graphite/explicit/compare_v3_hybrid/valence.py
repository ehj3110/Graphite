"""Per-element rule valency (k_rule) and volume→dual valence gate."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from graphite.explicit.hex_rules import apply_hex_grid, apply_hex_octahedral
from graphite.explicit.hex_topology_module import get_hex_topology_rule


def local_max_node_degree(rule_name: str) -> int:
    """
    Maximum nodal degree inside one parent element for ``rule_name``.

    Built by applying the local ``apply_hex_*`` graph on a unit cube — not
    ``HexTopologyRule.valency_cutoff`` (that field is an ironing policy knob).
    """
    name = str(rule_name).strip().lower()
    rule = get_hex_topology_rule(name)
    cube = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Prefer explicit builders so grid→3 / octahedral→4 stay stable.
    if rule.name == "octahedral":
        _nodes, struts = apply_hex_octahedral(cube)
    elif rule.name == "grid":
        _nodes, struts = apply_hex_grid(cube)
    else:
        _nodes, struts = rule.builder(cube)

    struts = np.asarray(struts, dtype=np.int64)
    if struts.size == 0:
        return 1
    deg = np.zeros(len(_nodes), dtype=np.int64)
    for a, b in struts:
        deg[int(a)] += 1
        deg[int(b)] += 1
    return int(deg.max())


def apply_valence_gate(
    candidates: list[tuple[int, int, float]],
    k_rule: int,
) -> tuple[dict[int, int], int]:
    """
    Cap volume→dual bonds per dual node at ``k_rule``.

    Parameters
    ----------
    candidates :
        ``(volume_iron_id, dual_id, distance)`` sorted preferred-first
        (typically ascending distance). Each accepted assignment is one
        volume→dual bond. Pure dual–dual edges are never passed in and do
        not consume budget.
    k_rule :
        Max accepted volume→dual bonds per dual node.

    Returns
    -------
    accepted : dict[int, int]
        volume iron id → dual id
    n_rejected : int
        Candidates rejected for exceeding the per-dual cap.
    """
    k = int(k_rule)
    if k < 1:
        raise ValueError(f"k_rule must be >= 1; got {k_rule}")

    bond_count: dict[int, int] = defaultdict(int)
    accepted: dict[int, int] = {}
    n_rejected = 0
    for iron_id, dual_id, _dist in candidates:
        iron_id = int(iron_id)
        dual_id = int(dual_id)
        if iron_id in accepted:
            continue
        if bond_count[dual_id] >= k:
            n_rejected += 1
            continue
        bond_count[dual_id] += 1
        accepted[iron_id] = dual_id
    return accepted, int(n_rejected)
