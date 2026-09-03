"""
Graphite Explicit Engine - Hexahedral Topology Synthesis

This module provides the global topology generation functions for assembling 
strut-based lattice graphs (nodes and struts) from conformal hexahedral grids.
It aggregates local topological rules (e.g. octahedral, grid, face dual) across 
a continuous field of block elements.
"""
from __future__ import annotations

import numpy as np

from .hex_rules import (
    _ADJACENT_FACE_PAIRS,
    _HEX_FACES,
    apply_hex_a15_kagome,
    apply_hex_dual,
    apply_hex_face_dual,
    apply_hex_grid,
    apply_hex_kelvin,
    apply_hex_kelvin14,
    apply_hex_octahedral,
    apply_hex_octahedral_half_neg_x,
    apply_hex_octahedral_half_neg_y,
    apply_hex_octahedral_half_neg_z,
    apply_hex_octahedral_half_pos_x,
    apply_hex_octahedral_half_pos_y,
    apply_hex_octahedral_half_pos_z,
    apply_hex_octahedral_half_x,
    apply_hex_octahedral_half_y,
    apply_hex_octahedral_half_z,
    apply_hex_octet_truss,
    apply_hex_cross,
    apply_hex_star,
    apply_hex_tesseract,
    leaf_surface_spec_for_rule,
)


from dataclasses import dataclass
from typing import Callable, Iterable

HexBuilder = Callable[..., tuple[np.ndarray, np.ndarray]]

# Conformal boundary DOF roles and surface-skin modes (SC modular conformal engine).
CONFORM_DOF_CORNERS = "corners"
CONFORM_DOF_FACE_CENTROIDS = "face_centroids"
_VALID_CONFORM_DOFS = frozenset({CONFORM_DOF_CORNERS, CONFORM_DOF_FACE_CENTROIDS})

SKIN_MODE_FACE_CENTROID_DUAL = "face_centroid_dual"
SKIN_MODE_CORNER_EDGE_CAGE = "corner_edge_cage"
SKIN_MODE_FACE_LOCAL_RULE = "face_local_rule"
SKIN_MODE_KELVIN_FACE_BRIDGE = "kelvin_face_bridge"
SKIN_MODE_NONE = "none"
_VALID_SKIN_MODES = frozenset({
    SKIN_MODE_FACE_CENTROID_DUAL,
    SKIN_MODE_CORNER_EDGE_CAGE,
    SKIN_MODE_FACE_LOCAL_RULE,
    SKIN_MODE_KELVIN_FACE_BRIDGE,
    SKIN_MODE_NONE,
})


def _normalize_conform_dofs(dofs: Iterable[str] | frozenset[str]) -> frozenset[str]:
    out = frozenset(str(d).strip().lower() for d in dofs)
    bad = out - _VALID_CONFORM_DOFS
    if bad:
        raise ValueError(f"Invalid conform_dofs {sorted(bad)}; expected subset of {sorted(_VALID_CONFORM_DOFS)}")
    if not out:
        raise ValueError("conform_dofs must be non-empty")
    return out


def _normalize_skin_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in _VALID_SKIN_MODES:
        raise ValueError(f"Invalid skin_mode {mode!r}; expected one of {sorted(_VALID_SKIN_MODES)}")
    return key


@dataclass(frozen=True)
class HexTopologyRule:
    """
    Definition of a specific hexahedral lattice topology rule.
    """
    name: str
    builder: HexBuilder
    cage_mode: str = "surface_dual"
    overlap_factor: float | Callable[[float], float] = 0.85
    # Modular SC conformal policy (source of truth for ironing + skin).
    conform_dofs: frozenset[str] = frozenset({CONFORM_DOF_FACE_CENTROIDS})
    skin_mode: str = SKIN_MODE_FACE_CENTROID_DUAL
    valency_cutoff: int = 4

    def __post_init__(self):
        object.__setattr__(self, "conform_dofs", _normalize_conform_dofs(self.conform_dofs))
        object.__setattr__(self, "skin_mode", _normalize_skin_mode(self.skin_mode))

    def get_overlap_factor(self, solid_fraction: float = 0.10) -> float:
        if callable(self.overlap_factor):
            return float(self.overlap_factor(solid_fraction))
        return float(self.overlap_factor)

_HEX_TOPOLOGY_ALIASES: dict[str, str] = {
    "kelvin_14": "kelvin14",
    "nested_cube": "tesseract",
    "hypercube": "tesseract",
}

_HEX_RULES_REGISTRY: dict[str, HexTopologyRule] = {}
_HEX_RULES = _HEX_RULES_REGISTRY

def get_hex_topology_rule(name: str) -> HexTopologyRule:
    n = str(name).strip().lower()
    canonical = _HEX_TOPOLOGY_ALIASES.get(n, n)
    if canonical not in _HEX_RULES_REGISTRY:
        supported = ", ".join(sorted(_HEX_RULES_REGISTRY.keys()))
        raise ValueError(f"Unsupported hex topology '{name}'. Options: {supported}")
    return _HEX_RULES_REGISTRY[canonical]

def register_hex_topology_rule(
    name: str,
    rule: HexTopologyRule,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> None:
    n = str(name).strip().lower()
    if n in _HEX_RULES_REGISTRY and not overwrite:
        raise ValueError(f"Hex topology rule '{n}' is already registered.")
    _HEX_RULES_REGISTRY[n] = rule
    for alias in aliases:
        a_norm = str(alias).strip().lower()
        _HEX_TOPOLOGY_ALIASES[a_norm] = n

def unregister_hex_topology_rule(name: str) -> None:
    n = str(name).strip().lower()
    canonical = _HEX_TOPOLOGY_ALIASES.get(n, n)
    if canonical in _HEX_RULES_REGISTRY:
        del _HEX_RULES_REGISTRY[canonical]

# Populate defaults: (builder, cage_mode, overlap, conform_dofs, skin_mode, valency_cutoff)
_FC = frozenset({CONFORM_DOF_FACE_CENTROIDS})
_CR = frozenset({CONFORM_DOF_CORNERS})
_BOTH = frozenset({CONFORM_DOF_CORNERS, CONFORM_DOF_FACE_CENTROIDS})

_default_rules = {
    "grid": (apply_hex_grid, "surface_cage", 0.85, _CR, SKIN_MODE_CORNER_EDGE_CAGE, 4),
    "octahedral": (apply_hex_octahedral, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4),
    # Six polarity half-cells (pos/neg × X/Y/Z). Legacy unsigned names → neg.
    "octahedral_half_neg_z": (
        apply_hex_octahedral_half_neg_z, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_pos_z": (
        apply_hex_octahedral_half_pos_z, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_neg_x": (
        apply_hex_octahedral_half_neg_x, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_pos_x": (
        apply_hex_octahedral_half_pos_x, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_neg_y": (
        apply_hex_octahedral_half_neg_y, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_pos_y": (
        apply_hex_octahedral_half_pos_y, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_z": (
        apply_hex_octahedral_half_z, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_x": (
        apply_hex_octahedral_half_x, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "octahedral_half_y": (
        apply_hex_octahedral_half_y, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4
    ),
    "star": (apply_hex_star, "surface_cage", 0.85, _CR, SKIN_MODE_FACE_LOCAL_RULE, 4),
    "octet": (apply_hex_octet_truss, "surface_cage", 0.85, _BOTH, SKIN_MODE_FACE_LOCAL_RULE, 4),
    "cross": (apply_hex_cross, "surface_cage", 0.85, _BOTH, SKIN_MODE_FACE_LOCAL_RULE, 4),
    "kelvin14": (apply_hex_kelvin14, "surface_cage", 0.85, _CR, SKIN_MODE_KELVIN_FACE_BRIDGE, 4),
    "kelvin": (apply_hex_kelvin, "surface_cage", 0.85, _CR, SKIN_MODE_KELVIN_FACE_BRIDGE, 4),
    "tesseract": (apply_hex_tesseract, "surface_cage", 0.85, _CR, SKIN_MODE_CORNER_EDGE_CAGE, 4),
    "hex_dual": (apply_hex_dual, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4),
    "hex_face_dual": (apply_hex_face_dual, "surface_dual", 0.72, _FC, SKIN_MODE_FACE_CENTROID_DUAL, 4),
    # a15_kagome: corners; skin deferred — none until shared-edge dual is wired
    "a15_kagome": (apply_hex_a15_kagome, "surface_dual", 0.72, _CR, SKIN_MODE_NONE, 3),
}

for _rname, (_rbuilder, _rcage, _roverlap, _rdofs, _rskin, _rval) in _default_rules.items():
    _HEX_RULES_REGISTRY[_rname] = HexTopologyRule(
        name=_rname,
        builder=_rbuilder,
        cage_mode=_rcage,
        overlap_factor=_roverlap,
        conform_dofs=_rdofs,
        skin_mode=_rskin,
        valency_cutoff=_rval,
    )


def list_hex_topology_rules() -> list[str]:
    """Canonical registered hex rule names (sorted)."""
    return sorted(_HEX_RULES_REGISTRY.keys())


def validate_hex_conformal_policies() -> None:
    """Raise if any registered rule is missing / invalid conformal policy fields."""
    for name, rule in _HEX_RULES_REGISTRY.items():
        _normalize_conform_dofs(rule.conform_dofs)
        _normalize_skin_mode(rule.skin_mode)
        if int(rule.valency_cutoff) < 1:
            raise ValueError(f"Rule {name!r} has invalid valency_cutoff={rule.valency_cutoff}")


# ---------------------------------------------------------------------------
# Quantized multi-rule dispatch (Full / Half_* / Empty)
# ---------------------------------------------------------------------------

# Task-1 boundary tags → registered rule names. None means skip (Empty).
_QUANTIZED_TAG_TO_RULE: dict[str, str | None] = {
    "empty": None,
    "full": "octahedral",
    # Six polarities
    "half_pos_x": "octahedral_half_pos_x",
    "half_neg_x": "octahedral_half_neg_x",
    "half_pos_y": "octahedral_half_pos_y",
    "half_neg_y": "octahedral_half_neg_y",
    "half_pos_z": "octahedral_half_pos_z",
    "half_neg_z": "octahedral_half_neg_z",
    # Legacy unsigned tags → negative polarity
    "half_x": "octahedral_half_neg_x",
    "half_y": "octahedral_half_neg_y",
    "half_z": "octahedral_half_neg_z",
    # Direct rule-name passthroughs
    "octahedral": "octahedral",
    "octahedral_half_pos_x": "octahedral_half_pos_x",
    "octahedral_half_neg_x": "octahedral_half_neg_x",
    "octahedral_half_pos_y": "octahedral_half_pos_y",
    "octahedral_half_neg_y": "octahedral_half_neg_y",
    "octahedral_half_pos_z": "octahedral_half_pos_z",
    "octahedral_half_neg_z": "octahedral_half_neg_z",
    "octahedral_half_x": "octahedral_half_neg_x",
    "octahedral_half_y": "octahedral_half_neg_y",
    "octahedral_half_z": "octahedral_half_neg_z",
}


def resolve_quantized_cell_rule(
    tag: str | None,
    *,
    full_rule_name: str = "octahedral",
) -> str | None:
    """
    Map a per-cell Quantized Topological Boundary tag to a registered rule name.

    Returns ``None`` for Empty / skipped cells.
    """
    if tag is None:
        return None
    key = str(tag).strip().lower().replace("-", "_")
    if key in ("", "none", "skip"):
        return None
    if key not in _QUANTIZED_TAG_TO_RULE:
        # Allow any already-registered rule name as an escape hatch.
        if key in _HEX_RULES_REGISTRY or key in _HEX_TOPOLOGY_ALIASES:
            return get_hex_topology_rule(key).name
        supported = ", ".join(sorted(_QUANTIZED_TAG_TO_RULE))
        raise ValueError(
            f"Unknown cell tag {tag!r}. Expected one of: {supported}, "
            f"or a registered hex rule name."
        )
    resolved = _QUANTIZED_TAG_TO_RULE[key]
    if resolved is None:
        return None
    if resolved == "octahedral" and full_rule_name != "octahedral":
        return get_hex_topology_rule(full_rule_name).name
    return resolved


def cell_tags_from_classification(classification) -> list[str | None]:
    """
    Build per-cell dispatcher tags from a Task-1 ``HexBoundaryClassification``.

    Empty → None, Full → \"Full\", Half → polarity tag
    (\"Half_Pos_X\" / \"Half_Neg_Z\" / …).
    """
    states = np.asarray(classification.states, dtype=object)
    orients = np.asarray(classification.orientations, dtype=object)
    if states.shape != orients.shape:
        raise ValueError("classification.states and orientations length mismatch")
    tags: list[str | None] = []
    for state, orient in zip(states.tolist(), orients.tolist()):
        s = str(state)
        if s == "Empty":
            tags.append(None)
        elif s == "Full":
            tags.append("Full")
        elif s == "Half":
            if orient is None:
                raise ValueError(
                    "Half cell is missing polarity; expected Half_Pos/Neg_X/Y/Z"
                )
            tags.append(str(orient))
        else:
            raise ValueError(f"Unknown fill state {state!r}")
    return tags


def _weld_stamp_local_graphs(
    local_graphs: list[tuple[np.ndarray, np.ndarray]],
    *,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-cell ``(nodes, struts)`` graphs by rounded-coordinate weld.

    Shared face-centroid nodes between Full and Half cells coalesce here —
    octahedral Half↔Full needs no extra transition bridges.
    """
    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()

    for local_nodes, local_struts in local_graphs:
        if local_nodes is None or len(local_nodes) == 0:
            continue
        local_to_global: list[int] = []
        for node in np.asarray(local_nodes, dtype=np.float64):
            key = tuple(np.round(node, round_decimals).tolist())
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(np.asarray(node, dtype=np.float64).copy())
            local_to_global.append(idx)
        if local_struts is None or len(local_struts) == 0:
            continue
        for a, b in np.asarray(local_struts, dtype=np.int64):
            ga = local_to_global[int(a)]
            gb = local_to_global[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            strut_set.add((ga, gb))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts


def _builder_kwargs_for(rule_fn: HexBuilder, kwargs: dict) -> dict:
    import inspect

    sig = inspect.signature(rule_fn)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


@dataclass
class SurfaceStampTags:
    """Global surface-dual tags after welding a multi-rule stamp (Task 15)."""

    surface_mask: np.ndarray  # (V,) bool — explicitly tagged surface nodes
    native_surface_struts: np.ndarray  # (E, 2) Half diamond perimeters only
    # (hex_i, face_i, gid) for every explicitly tagged Full exposed face center
    full_exposed_primaries: list[tuple[int, int, int]]
    # (hex_i, face_i, gid) for Half diamond corners (mid-plane face centers)
    half_diamond_primaries: list[tuple[int, int, int]]


def _corner_key_pt(pt, decimals: int) -> tuple[float, float, float]:
    return tuple(np.round(np.asarray(pt, dtype=np.float64), decimals).tolist())


def _face_key_corners(corners, face_local, decimals: int):
    return tuple(sorted(_corner_key_pt(corners[i], decimals) for i in face_local))


def generate_hex_topology_multi_tagged(
    hex_elements: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    full_rule_name: str = "octahedral",
    round_decimals: int = 6,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, SurfaceStampTags]:
    """
    Stamp Full/Half leaves and weld, returning explicit surface dual tags.

    Half: surface nodes = diamond corners (locals 1–4); native struts =
    diamond perimeter only (spokes are internal).

    Full: surface nodes = exposed face centers only; native surface struts =
    none (manifold stitcher supplies the dual).
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")
    tags = list(cell_tags)
    if len(tags) != len(elems):
        raise ValueError(
            f"cell_tags length {len(tags)} must match n_hex {len(elems)}"
        )

    retained: list[tuple[int, np.ndarray, str]] = []
    face_owners: dict[tuple, list[tuple[int, int]]] = {}
    for hi, (elem, tag) in enumerate(zip(elems, tags)):
        rule_name = resolve_quantized_cell_rule(tag, full_rule_name=full_rule_name)
        if rule_name is None:
            continue
        retained.append((hi, elem, rule_name))
        for fi, face in enumerate(_HEX_FACES):
            face_owners.setdefault(
                _face_key_corners(elem, face, round_decimals), []
            ).append((hi, fi))

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    surface_gids: set[int] = set()
    native_surface: set[tuple[int, int]] = set()
    full_exposed_primaries: list[tuple[int, int, int]] = []
    half_diamond_primaries: list[tuple[int, int, int]] = []

    for hi, elem, rule_name in retained:
        rule = get_hex_topology_rule(rule_name)
        rule_kwargs = _builder_kwargs_for(rule.builder, kwargs)
        local_nodes, local_struts = rule.builder(elem, **rule_kwargs)
        spec = leaf_surface_spec_for_rule(rule_name)
        local_nodes = np.asarray(local_nodes, dtype=np.float64)
        local_struts = np.asarray(local_struts, dtype=np.int64).reshape(-1, 2)

        local_to_global: list[int] = []
        for node in local_nodes:
            key = _corner_key_pt(node, round_decimals)
            idx = node_map.get(key)
            if idx is None:
                idx = len(nodes_list)
                node_map[key] = idx
                nodes_list.append(node.copy())
            local_to_global.append(idx)

        for a, b in local_struts:
            ga, gb = local_to_global[int(a)], local_to_global[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            strut_set.add((ga, gb))

        if spec is None:
            continue

        for li in spec.surface_node_locals:
            li = int(li)
            if li < 0 or li >= len(local_to_global):
                continue
            gid = local_to_global[li]
            surface_gids.add(gid)
            fi = (
                int(spec.local_to_face_index[li])
                if li < len(spec.local_to_face_index)
                else -1
            )
            if fi >= 0:
                half_diamond_primaries.append((hi, fi, gid))

        for a, b in spec.native_surface_strut_locals:
            ga, gb = local_to_global[int(a)], local_to_global[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            native_surface.add((ga, gb))

        for li in spec.exposed_candidate_locals:
            li = int(li)
            if li < 0 or li >= len(local_to_global):
                continue
            fi = (
                int(spec.local_to_face_index[li])
                if li < len(spec.local_to_face_index)
                else li
            )
            fkey = _face_key_corners(elem, _HEX_FACES[fi], round_decimals)
            if len(face_owners.get(fkey, [])) != 1:
                continue
            gid = local_to_global[li]
            surface_gids.add(gid)
            full_exposed_primaries.append((hi, fi, gid))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    surface_mask = np.zeros(len(nodes), dtype=bool)
    if surface_gids:
        surface_mask[list(surface_gids)] = True
    native_arr = (
        np.array(sorted(native_surface), dtype=np.int64)
        if native_surface
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts, SurfaceStampTags(
        surface_mask=surface_mask,
        native_surface_struts=native_arr,
        full_exposed_primaries=full_exposed_primaries,
        half_diamond_primaries=half_diamond_primaries,
    )


def generate_hex_topology_multi(
    hex_elements: np.ndarray,
    cell_tags: list | np.ndarray,
    *,
    full_rule_name: str = "octahedral",
    round_decimals: int = 6,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stamp mixed Full / Half rules in one pass, then weld by coordinate.

    ``cell_tags[i]`` is a Task-1 style tag (``Full``, ``Half_Z``, ``Empty``, …)
    or a registered rule name. Empty / ``None`` cells are skipped.

    Octahedral Half↔Full load paths are manifold via coincident face-center
    nodes alone — no transition pass.
    """
    nodes, struts, _tags = generate_hex_topology_multi_tagged(
        hex_elements,
        cell_tags,
        full_rule_name=full_rule_name,
        round_decimals=round_decimals,
        **kwargs,
    )
    return nodes, struts


def generate_hex_topology(
    hex_elements: np.ndarray,
    rule_name: str = "octahedral",
    round_decimals: int = 6,
    cell_tags: list | np.ndarray | None = None,
    classification=None,
    full_rule_name: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a global lattice topology (nodes and struts) from a set of hex elements.

    Uniform mode (default): applies one ``rule_name`` to every element, then
    merges coincident nodes by rounded coordinates.

    Multi-rule mode: pass ``cell_tags`` (per-cell Full / Half_Z / Empty / …) or a
    Task-1 ``classification`` object. Each retained cell is dispatched to its
    builder; Empty cells are skipped; shared nodes still weld by coordinate.

    Rules:
    - 'grid': Connects hex corners. Perfectly conformal for Route 3 conformed scaffolds.
    - 'hex_dual': Centroid-to-centroid adjacency across shared faces. Includes surface cage.
    - 'hex_face_dual': Octahedral pattern connecting centers of all 6 quad faces.
    - 'octet': Hybrid corner + face-center connectivity.
    - 'kelvin14' / 'kelvin' / 'kelvin_14': 24-node truncated octahedron (36 struts).
    - 'tesseract' / 'nested_cube' / 'hypercube': 16-node nested cube (32 struts).
    - 'star': Centroid to all 8 corners.
    - 'octahedral': 6 face centers with octahedral edge connectivity.
    - 'octahedral_half_z' / '_x' / '_y': half-cell truncations (face centers only).

    Parameters
    ----------
    hex_elements : ndarray
        (N, 8, 3) array of element corner coordinates.
    rule_name : str, optional
        String identifier for the local cell rule, by default "octahedral".
        Ignored when ``cell_tags`` or ``classification`` is provided (except as
        the Full-cell rule via ``full_rule_name``).
    round_decimals : int, optional
        Precision for coordinating merging based on floating point equality,
        by default 6.
    cell_tags : sequence, optional
        Per-cell tags for multi-rule dispatch (length N).
    classification : optional
        Task-1 ``HexBoundaryClassification``; converted to tags if ``cell_tags``
        is omitted.
    full_rule_name : str, optional
        Rule used for ``Full`` tags (default: ``rule_name`` or ``octahedral``).
    kwargs : dict
        Additional parameters forwarded to local rule functions (e.g. sdf_sampler).

    Returns
    -------
    nodes : ndarray
        (V, 3) array of unique global node coordinates.
    struts : ndarray
        (E, 2) array of global strut indices connecting the nodes.
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")

    # --- Multi-rule dispatcher path ---
    if cell_tags is not None or classification is not None:
        tags = (
            list(cell_tags)
            if cell_tags is not None
            else cell_tags_from_classification(classification)
        )
        full_name = full_rule_name if full_rule_name is not None else (
            rule_name if rule_name else "octahedral"
        )
        return generate_hex_topology_multi(
            elems,
            tags,
            full_rule_name=full_name,
            round_decimals=round_decimals,
            **kwargs,
        )

    rule = get_hex_topology_rule(rule_name)
    rule_key = rule.name
    rule_fn = rule.builder

    if rule_key == "hex_dual":
        return _generate_hex_dual_topology(elems, round_decimals=round_decimals)

    rule_kwargs = _builder_kwargs_for(rule_fn, kwargs)

    if rule_key == "a15_kagome":
        all_nodes = []
        all_edges = []
        node_offset = 0
        for elem in elems:
            local_nodes, local_struts = rule_fn(elem, **rule_kwargs)
            if len(local_nodes) == 0:
                continue
            all_nodes.append(local_nodes)
            for u, v in local_struts:
                all_edges.append((u + node_offset, v + node_offset))
            node_offset += len(local_nodes)

        if not all_nodes:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64)

        nodes = np.vstack(all_nodes)
        struts = np.array(all_edges, dtype=np.int64)

        # Determine merge tolerance based on average element size
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        sample_size = min(len(elems), 100)
        sample_lens = []
        for elem in elems[:sample_size]:
            for u, v in edge_indices:
                sample_lens.append(np.linalg.norm(elem[u] - elem[v]))
        avg_edge_len = np.mean(sample_lens) if sample_lens else 1.0
        tolerance = avg_edge_len * 0.05

        nodes, struts = merge_nodes_kdtree(nodes, struts, tolerance)
        return nodes, struts

    # Uniform single-rule stamp with coordinate weld (same path as multi-rule).
    local_graphs = []
    for elem in elems:
        local_nodes, local_struts = rule_fn(elem, **rule_kwargs)
        local_graphs.append((local_nodes, local_struts))
    return _weld_stamp_local_graphs(local_graphs, round_decimals=round_decimals)


def merge_nodes_kdtree(nodes: np.ndarray, struts: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge coincident/duplicate nodes within a specified distance tolerance using KDTree.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    pairs = tree.query_pairs(r=tolerance)
    
    parent = np.arange(len(nodes))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            
    for u, v in pairs:
        union(u, v)
        
    unique_roots = np.unique([find(i) for i in range(len(nodes))])
    root_to_new = {root: new_idx for new_idx, root in enumerate(unique_roots)}
    
    new_nodes = np.zeros((len(unique_roots), 3))
    counts = np.zeros(len(unique_roots))
    
    for i in range(len(nodes)):
        root = find(i)
        new_idx = root_to_new[root]
        new_nodes[new_idx] += nodes[i]
        counts[new_idx] += 1
        
    new_nodes = new_nodes / counts[:, None]
    
    new_struts_set = set()
    for u, v in struts:
        nu = root_to_new[find(u)]
        nv = root_to_new[find(v)]
        if nu != nv:
            new_struts_set.add((min(nu, nv), max(nu, nv)))
            
    new_struts = np.array(list(new_struts_set), dtype=np.int64) if new_struts_set else np.empty((0, 2), dtype=np.int64)
    return new_nodes, new_struts



def _generate_hex_dual_topology(
    hex_elements: np.ndarray, round_decimals: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")

    # 1. Internal centroids (one per hex cell)
    centroids = np.mean(elems, axis=1)
    nodes_list = [np.round(centroids, round_decimals).astype(np.float64)]

    # Build coordinate-aware global corner-node IDs for face deduplication.
    flat = elems.reshape(-1, 3)
    rounded = np.round(flat, round_decimals)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    elem_corner_ids = inverse.reshape(-1, 8)

    face_idx = np.array([
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]
    ], dtype=np.int32)

    face_to_owners: dict[tuple[int, int, int, int], list[int]] = {}
    
    # Track all faces and their owner(s)
    for hi, corners in enumerate(elem_corner_ids):
        for face in face_idx:
            key = tuple(sorted(int(corners[i]) for i in face))
            if key not in face_to_owners:
                face_to_owners[key] = []
            face_to_owners[key].append(hi)

    edge_set: set[tuple[int, int]] = set()
    boundary_nodes: list[np.ndarray] = []
    
    n_internal = len(centroids)
    
    for key, owners in face_to_owners.items():
        if len(owners) == 2:
            # Internal shared face: connect two centroids
            a, b = owners[0], owners[1]
            if a != b:
                edge_set.add((min(a, b), max(a, b)))
        elif len(owners) == 1:
            # Exposed boundary face: connect centroid to face center
            hi = owners[0]
            # Need actual coordinates for face center calculation
            # Use any owner to get the face coordinates
            face_nodes_coords = elems[hi][list(np.where(np.isin(elem_corner_ids[hi], key))[0])]
            # Fallback: find the face using the key in the element's local corner map
            # (Strictly, the key matches elem_corner_ids[hi][face_idx_row])
            f_center = np.mean(face_nodes_coords, axis=0)
            
            b_idx = n_internal + len(boundary_nodes)
            boundary_nodes.append(f_center)
            edge_set.add((hi, b_idx))

    # Combine internal centroids and boundary face centers
    final_nodes = np.vstack([nodes_list[0]] + boundary_nodes) if boundary_nodes else nodes_list[0]
    final_struts = np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.empty((0, 2), dtype=np.int64)
    
    return final_nodes, final_struts


def generate_hex_octahedral_volume_with_boundary_face_map(
    hex_elements: np.ndarray,
    *,
    volume_emit_mask: np.ndarray | None = None,
    round_decimals: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    """
    Octahedral volume lattice with integer ``(hex_index, face_index)`` -> global node map
    for exterior faces only.

    Args:
        hex_elements: (N, 8, 3) brick coordinates.
        volume_emit_mask: if set, only hexes with True emit interior struts; face nodes
            are still registered for boundary faces on all hexes.

    Returns:
        nodes, struts, boundary_face_to_global_node
    """
    elems = np.asarray(hex_elements, dtype=np.float64)
    if elems.ndim != 3 or elems.shape[1:] != (8, 3):
        raise ValueError(f"hex_elements must have shape (N, 8, 3); got {elems.shape}.")
    n_hex = elems.shape[0]
    if n_hex == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            {},
        )

    flat = elems.reshape(-1, 3)
    rounded = np.round(flat, round_decimals)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    elem_corner_ids = inverse.reshape(-1, 8).astype(np.int32)

    face_count: dict[tuple[int, int, int, int], int] = {}
    for corners in elem_corner_ids:
        for face in _HEX_FACES:
            key = tuple(sorted(int(corners[i]) for i in face))
            face_count[key] = face_count.get(key, 0) + 1

    emit = (
        np.ones(n_hex, dtype=bool)
        if volume_emit_mask is None
        else np.asarray(volume_emit_mask, dtype=bool).ravel()
    )
    if emit.shape[0] != n_hex:
        raise ValueError("volume_emit_mask length must match hex count.")

    node_map: dict[tuple[float, float, float], int] = {}
    nodes_list: list[np.ndarray] = []
    strut_set: set[tuple[int, int]] = set()
    boundary_face_to_node: dict[tuple[int, int], int] = {}

    def _global_index(pt: np.ndarray) -> int:
        key = tuple(np.round(pt, round_decimals).tolist())
        idx = node_map.get(key)
        if idx is None:
            idx = len(nodes_list)
            node_map[key] = idx
            nodes_list.append(np.asarray(pt, dtype=np.float64))
        return idx

    for hi, coords in enumerate(elems):
        face_globals: list[int] = []
        for fi, face in enumerate(_HEX_FACES):
            fc = coords[list(face)].mean(axis=0)
            gid = _global_index(fc)
            face_globals.append(gid)
            key = tuple(sorted(int(elem_corner_ids[hi, i]) for i in face))
            if face_count.get(key, 0) == 1:
                boundary_face_to_node[(int(hi), int(fi))] = gid

        if not emit[hi]:
            continue
        for a, b in _ADJACENT_FACE_PAIRS:
            ga = face_globals[int(a)]
            gb = face_globals[int(b)]
            if ga == gb:
                continue
            if ga > gb:
                ga, gb = gb, ga
            strut_set.add((ga, gb))

    nodes = np.vstack(nodes_list) if nodes_list else np.empty((0, 3), dtype=np.float64)
    struts = (
        np.array(sorted(strut_set), dtype=np.int64)
        if strut_set
        else np.empty((0, 2), dtype=np.int64)
    )
    return nodes, struts, boundary_face_to_node


def filter_struts_drop_exterior_shell_pairs(
    struts: np.ndarray,
    boundary_face_node_ids: set[int] | frozenset[int],
) -> tuple[np.ndarray, int]:
    """
    Remove volume struts whose endpoints are both exterior boundary face centers.

    In two-branch export the skin branch owns the surface dual; keeping these
    struts duplicates the conformal/integer face-center cage on the envelope.
    """
    if struts.size == 0:
        return struts, 0
    bset = boundary_face_node_ids
    kept: list[tuple[int, int]] = []
    dropped = 0
    for a, b in np.asarray(struts, dtype=np.int64):
        ia, ib = int(a), int(b)
        if ia in bset and ib in bset:
            dropped += 1
            continue
        kept.append((ia, ib))
    out = (
        np.array(kept, dtype=np.int64)
        if kept
        else np.empty((0, 2), dtype=np.int64)
    )
    return out, int(dropped)
