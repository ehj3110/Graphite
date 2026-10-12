"""
graphite/explicit/interlinked/catalog.py

Authoritative Transition Catalog & Recommendation Engine for Polycatenated Metamaterials (PAMs).
Provides O(1) instantaneous retrieval of precomputed crystallographic transition parameters,
quality metrics (clearance, linking number, boundary components), and automated shortest-path
routing for multi-step non-frustrated transitions across habit planes {100}, {110}, {111}.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_CATALOG_FILE = Path(__file__).resolve().parent / "transition_catalog.json"
_CATALOG_CACHE: dict[str, TransitionCatalogEntry] | None = None
_GRAPH_ADJ_CACHE: dict[tuple[str, str], list[tuple[str, str]]] | None = None


@dataclass(frozen=True)
class TransitionCatalogEntry:
    """
    Standardized, precomputed entry for a PAM transition or habit plane reorientation.
    """
    key: str
    transition_type: str  # "same_cell_plane" or "inter_cell"
    cell_a: str
    cell_b: str
    plane_a: str
    plane_b: str
    habit_plane: str
    classification: str  # "DIRECT", "BUFFERED", "FRUSTRATED"
    is_feasible: bool
    pitch_a_mm: float
    pitch_b_mm: float
    pitch_ratio: float
    gap_mm: float
    gap_ratio: float
    twist_deg: float
    phase_shift: tuple[float, float, float]
    min_clearance_um: float
    boundary_links: int
    failure_reason: str = ""
    recommended_routes: tuple[str, ...] = field(default_factory=tuple)
    symmetry_notes: str = ""


def _load_catalog() -> dict[str, TransitionCatalogEntry]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    if not _CATALOG_FILE.exists():
        _CATALOG_CACHE = {}
        return _CATALOG_CACHE

    with open(_CATALOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    catalog: dict[str, TransitionCatalogEntry] = {}
    for k, v in data.items():
        phase_shift_tuple = tuple(float(x) for x in v.get("phase_shift", [0.0, 0.0, 0.0]))
        routes = tuple(str(r) for r in v.get("recommended_routes", []))
        entry = TransitionCatalogEntry(
            key=k,
            transition_type=v["transition_type"],
            cell_a=v["cell_a"].lower(),
            cell_b=v["cell_b"].lower(),
            plane_a=str(v["plane_a"]),
            plane_b=str(v["plane_b"]),
            habit_plane=str(v["habit_plane"]),
            classification=v["classification"],
            is_feasible=bool(v["is_feasible"]),
            pitch_a_mm=float(v["pitch_a_mm"]),
            pitch_b_mm=float(v["pitch_b_mm"]),
            pitch_ratio=float(v["pitch_ratio"]),
            gap_mm=float(v["gap_mm"]),
            gap_ratio=float(v["gap_ratio"]),
            twist_deg=float(v["twist_deg"]),
            phase_shift=(phase_shift_tuple[0], phase_shift_tuple[1], phase_shift_tuple[2]),
            min_clearance_um=float(v["min_clearance_um"]),
            boundary_links=int(v["boundary_links"]),
            failure_reason=v.get("failure_reason", ""),
            recommended_routes=routes,
            symmetry_notes=v.get("symmetry_notes", ""),
        )
        catalog[k] = entry

    _CATALOG_CACHE = catalog
    return _CATALOG_CACHE


def _get_direct_graph() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Returns an adjacency list of verified DIRECT transitions."""
    global _GRAPH_ADJ_CACHE
    if _GRAPH_ADJ_CACHE is not None:
        return _GRAPH_ADJ_CACHE

    catalog = _load_catalog()
    adj: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)

    for entry in catalog.values():
        if entry.classification == "DIRECT":
            u = (entry.cell_a, entry.plane_a)
            v = (entry.cell_b, entry.plane_b)
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)

    _GRAPH_ADJ_CACHE = adj
    return _GRAPH_ADJ_CACHE


def get_transition_entry(
    cell_a: str,
    cell_b: str,
    habit_plane: str | None = None,
    plane_a: str | None = None,
    plane_b: str | None = None,
) -> TransitionCatalogEntry | None:
    """
    O(1) retrieval of precomputed transition entry from the catalog.

    Args:
        cell_a: Source unit cell name (e.g. 'c6tt', 'd4tet', 'd4cube', 'b8co', 'j4oct').
        cell_b: Target unit cell name.
        habit_plane: Miller index habit plane for inter-cell transitions (e.g. '100', '110', '111').
        plane_a: Source habit plane (for same-cell plane-to-plane transitions).
        plane_b: Target habit plane (for same-cell plane-to-plane transitions).

    Returns:
        TransitionCatalogEntry if found, or None.
    """
    catalog = _load_catalog()
    ca = cell_a.lower()
    cb = cell_b.lower()

    # Same-cell plane reorientation
    if ca == cb:
        pa = plane_a or habit_plane or "100"
        pb = plane_b or habit_plane or "100"
        k1 = f"{ca}_{pa}__to__{cb}_{pb}"
        if k1 in catalog:
            return catalog[k1]
        k2 = f"{ca}_{pb}__to__{cb}_{pa}"
        if k2 in catalog:
            return catalog[k2]
        return None

    # Inter-cell heteroepitaxial transition
    hkl = habit_plane or plane_a or "100"
    k_fwd = f"{ca}_to_{cb}__{hkl}"
    if k_fwd in catalog:
        return catalog[k_fwd]

    k_rev = f"{cb}_to_{ca}__{hkl}"
    if k_rev in catalog:
        return catalog[k_rev]

    return None


def is_transition_feasible(cell_a: str, cell_b: str, habit_plane: str = "100") -> bool:
    """
    Instantaneous O(1) boolean check if a direct transition is feasible without buffer layers.
    """
    entry = get_transition_entry(cell_a, cell_b, habit_plane=habit_plane)
    return bool(entry and entry.is_feasible)


def list_feasible_transitions(cell_name: str | None = None) -> list[TransitionCatalogEntry]:
    """
    Returns all precomputed DIRECT transitions, optionally filtered by cell name.
    """
    catalog = _load_catalog()
    res = [e for e in catalog.values() if e.classification == "DIRECT"]
    if cell_name:
        cn = cell_name.lower()
        res = [e for e in res if e.cell_a == cn or e.cell_b == cn]
    return res


def recommend_transition_path(
    cell_a: str,
    cell_b: str,
    plane_a: str = "100",
    plane_b: str = "100",
    max_paths: int = 2,
) -> list[str]:
    """
    Finds the shortest sequence of verified DIRECT transitions connecting (cell_a, plane_a)
    to (cell_b, plane_b).
    """
    ca = cell_a.lower()
    cb = cell_b.lower()

    # Check direct first
    entry = get_transition_entry(ca, cb, habit_plane=plane_a, plane_a=plane_a, plane_b=plane_b)
    if entry and entry.is_feasible:
        return [f"{ca} {{{plane_a}}} -> {cb} {{{plane_b}}} (DIRECT)"]

    if entry and entry.recommended_routes:
        return list(entry.recommended_routes)

    # Perform BFS over direct transition graph
    adj = _get_direct_graph()
    start_node = (ca, plane_a)
    end_node = (cb, plane_b)

    if start_node == end_node:
        return [f"{ca} {{{plane_a}}} (Identity)"]

    queue = collections.deque([[start_node]])
    found_paths = []
    shortest_len = None

    while queue:
        path = queue.popleft()
        node = path[-1]

        if shortest_len is not None and len(path) > shortest_len + 1:
            break

        if node == end_node:
            found_paths.append(path)
            if shortest_len is None:
                shortest_len = len(path)
            if len(found_paths) >= max_paths:
                break
            continue

        for neighbor in adj.get(node, []):
            if neighbor not in path:
                queue.append(path + [neighbor])

    if found_paths:
        return [" -> ".join(f"{c} {{{h}}}" for c, h in p) for p in found_paths]

    return [f"{ca} {{{plane_a}}} -> intermediate buffer -> {cb} {{{plane_b}}}"]
