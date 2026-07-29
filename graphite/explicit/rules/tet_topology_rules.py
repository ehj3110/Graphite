"""
Vectorized topology rules for conformal tetrahedral scaffolds.

These rules operate on globally indexed arrays prepared by topology_module.py.
Each builder returns internal struts (S, 2) as integer node index pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


InternalBuilder = Callable[..., np.ndarray]


@dataclass(frozen=True)
class TopologyRule:
    """
    Definition of a specific lattice topology rule for conformal meshes.
    
    Attributes
    ----------
    name : str
        The canonical name of the topology rule.
    internal_builder : Callable
        Function that generates the internal graph struts.
    cage_mode : str
        The boundary cage construction mode used by this topology.
    overlap_factor : float | Callable[[float], float] = 0.85
        Scalar or function f(solid_fraction) -> float correcting for node junction volume.
    """
    name: str
    internal_builder: InternalBuilder
    cage_mode: str
    overlap_factor: float | Callable[[float], float] = 0.85

    def get_overlap_factor(self, solid_fraction: float = 0.10) -> float:
        if callable(self.overlap_factor):
            return float(self.overlap_factor(solid_fraction))
        return float(self.overlap_factor)


_VALID_CAGE_MODES = frozenset({"surface_cage", "surface_dual", "edge_midpoints"})


_TOPOLOGY_ALIASES: dict[str, str] = {
    "vertex_to_centroid": "rhombic",
    "bcc_vertex_conformal": "rhombic",
}


def _norm_name(name: str) -> str:
    n = name.strip().lower()
    if not n:
        raise ValueError("Topology name must be a non-empty string.")
    return n


def normalize_topology_name(name: str) -> str:
    """Normalize a topology key through alias mapping."""
    key = _norm_name(name)
    return _TOPOLOGY_ALIASES.get(key, key)


def build_rhombic_internal(*, tets_linear: np.ndarray, tet_vol_idx: np.ndarray, **_: object) -> np.ndarray:
    """Rhombic: connect each tet centroid to its four corner vertices."""
    return np.column_stack((tets_linear.reshape(-1), np.repeat(tet_vol_idx, 4)))


def build_voronoi_internal(
    *,
    uniq_face_ids: np.ndarray,
    has_two: np.ndarray,
    first_tet: np.ndarray,
    second_tet: np.ndarray,
    off_vol: int,
    off_faces: int,
    include_surface_cage: bool,
    **_: object,
) -> np.ndarray:
    """
    Voronoi adjacency rule:
    - Internal face shared by two tets: centroid-to-centroid.
    - Boundary face owned by one tet: centroid-to-face-centroid when surface cage is enabled.
    """
    boundary_face_ids = uniq_face_ids[~has_two]

    two_a = off_vol + first_tet[has_two]
    two_b = off_vol + second_tet[has_two]
    struts_internal = (
        np.column_stack((two_a, two_b))
        if two_a.size
        else np.empty((0, 2), dtype=np.int64)
    )

    if include_surface_cage:
        b_tet = off_vol + first_tet[~has_two]
        b_face = off_faces + boundary_face_ids
        struts_boundary = (
            np.column_stack((b_tet, b_face))
            if b_tet.size
            else np.empty((0, 2), dtype=np.int64)
        )
    else:
        struts_boundary = np.empty((0, 2), dtype=np.int64)

    return np.vstack((struts_internal, struts_boundary))


def build_kagome_internal(*, tet_face_idx: np.ndarray, **_: object) -> np.ndarray:
    """Kagome: complete graph on each tet's four face-centroid nodes."""
    pairs = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    a = tet_face_idx[:, pairs[:, 0]].reshape(-1)
    b = tet_face_idx[:, pairs[:, 1]].reshape(-1)
    return np.column_stack((a, b))


def build_icosahedral_internal(
    *,
    tet_edge_idx: np.ndarray,
    icosa_face_edges: np.ndarray,
    **_: object,
) -> np.ndarray:
    """Icosahedral: triangles among edge-midpoint nodes on each tet face."""
    mids = tet_edge_idx[:, icosa_face_edges]
    p = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    return mids[:, :, p].reshape(-1, 2)


TOPOLOGY_RULES: dict[str, TopologyRule] = {
    "rhombic": TopologyRule(
        name="rhombic",
        internal_builder=build_rhombic_internal,
        cage_mode="surface_cage",
        overlap_factor=0.85,
    ),
    "voronoi": TopologyRule(
        name="voronoi",
        internal_builder=build_voronoi_internal,
        cage_mode="surface_dual",
        overlap_factor=0.72,
    ),
    "kagome": TopologyRule(
        name="kagome",
        internal_builder=build_kagome_internal,
        cage_mode="surface_dual",
        overlap_factor=0.72,
    ),
    "icosahedral": TopologyRule(
        name="icosahedral",
        internal_builder=build_icosahedral_internal,
        cage_mode="edge_midpoints",
        overlap_factor=0.85,
    ),
}


def get_supported_topology_names() -> tuple[str, ...]:
    """Return stable sorted topology keys."""
    return tuple(sorted(TOPOLOGY_RULES.keys()))


def get_topology_rule(name: str) -> TopologyRule:
    """Resolve a topology name (or alias) to a registered rule."""
    key = normalize_topology_name(name)
    try:
        return TOPOLOGY_RULES[key]
    except KeyError as exc:
        supported = ", ".join(get_supported_topology_names())
        raise ValueError(
            f"Unsupported topology_type='{name}'. Supported: {supported}."
        ) from exc


def register_topology_rule(
    name: str,
    rule: TopologyRule,
    *,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    """Register a topology rule and optional aliases for plugin-like extension."""
    key = _norm_name(name)
    if not callable(rule.internal_builder):
        raise ValueError("TopologyRule.internal_builder must be callable.")
    if rule.cage_mode not in _VALID_CAGE_MODES:
        valid = ", ".join(sorted(_VALID_CAGE_MODES))
        raise ValueError(f"Invalid cage_mode '{rule.cage_mode}'. Supported: {valid}.")

    if key in TOPOLOGY_RULES and not overwrite:
        raise ValueError(
            f"Topology '{key}' is already registered. Set overwrite=True to replace it."
        )

    TOPOLOGY_RULES[key] = TopologyRule(
        name=key,
        internal_builder=rule.internal_builder,
        cage_mode=rule.cage_mode,
    )

    for alias in aliases:
        alias_key = _norm_name(alias)
        existing = _TOPOLOGY_ALIASES.get(alias_key)
        if existing is not None and existing != key and not overwrite:
            raise ValueError(
                f"Alias '{alias_key}' already maps to '{existing}'. "
                "Set overwrite=True to replace it."
            )
        _TOPOLOGY_ALIASES[alias_key] = key


def unregister_topology_rule(name: str) -> None:
    """Remove a topology rule and any aliases that point to it."""
    key = normalize_topology_name(name)
    if key not in TOPOLOGY_RULES:
        raise ValueError(f"Topology '{name}' is not registered.")

    TOPOLOGY_RULES.pop(key, None)
    alias_keys = [a for a, target in _TOPOLOGY_ALIASES.items() if target == key]
    for alias in alias_keys:
        _TOPOLOGY_ALIASES.pop(alias, None)
