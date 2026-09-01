"""
Aristo FEA mesh quality metrics and stress post-processing gates.

Phase 0/1: per-tet volume and aspect ratio, quality masks, diagnostic reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from graphite.aristo.aristo_config import AristoConfig


def element_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Signed tet volumes (mm³); negative if inverted."""
    coords = nodes[elements]
    jacobian = np.stack(
        [
            coords[:, 1] - coords[:, 0],
            coords[:, 2] - coords[:, 0],
            coords[:, 3] - coords[:, 0],
        ],
        axis=1,
    )
    return np.linalg.det(jacobian) / 6.0


def element_max_edge_lengths(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Longest edge length (mm) per tet."""
    coords = nodes[elements]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.stack(
        [np.linalg.norm(coords[:, j] - coords[:, i], axis=1) for i, j in pairs],
        axis=1,
    )
    return lengths.max(axis=1)


def element_aspect_ratios(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Edge-length aspect ratio per tet: max edge / min edge.

    Sliver elements approach large values; equilateral-like tets are O(1).
    """
    coords = nodes[elements]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    lengths = np.stack(
        [np.linalg.norm(coords[:, j] - coords[:, i], axis=1) for i, j in pairs],
        axis=1,
    )
    l_min = lengths.min(axis=1)
    l_max = lengths.max(axis=1)
    return l_max / np.maximum(l_min, 1e-15)


def resolve_min_tet_volume(
    volumes: np.ndarray,
    config: AristoConfig,
) -> float:
    """Absolute minimum tet volume (mm³) from config."""
    if config.min_tet_volume_mm3 is not None:
        return float(config.min_tet_volume_mm3)
    abs_vols = np.abs(volumes)
    positive = abs_vols[abs_vols > 0.0]
    if positive.size == 0:
        return 1e-12
    return float(config.min_tet_volume_fraction) * float(np.median(positive))


def resolve_max_tet_edge(config: AristoConfig) -> float:
    """Upper bound on longest tet edge (mm) for stress post-processing."""
    if config.max_tet_edge_mm is not None:
        return float(config.max_tet_edge_mm)
    return float(config.max_tet_edge_factor) * float(config.fea_mesh_resolution)


def resolve_giant_tet_edge(config: AristoConfig) -> float:
    """
    Edge length (mm) above which an entire mesh is rejected in thorough remesh.

    Separate from the stress gate — lattice meshes legitimately exceed 4×h locally,
    but Netgen can create single tets spanning ~70% of the part diagonal (~2.5 mm).
    """
    if config.giant_tet_edge_mm is not None:
        return float(config.giant_tet_edge_mm)
    h = float(config.fea_mesh_resolution)
    return max(2.2, float(config.giant_tet_edge_factor) * h)


def resolve_max_tet_volume(volumes: np.ndarray, config: AristoConfig) -> float:
    """Upper bound on tet volume (mm³) from median × fraction."""
    abs_vols = np.abs(volumes)
    positive = abs_vols[abs_vols > 0.0]
    if positive.size == 0:
        return float("inf")
    return float(config.max_tet_volume_fraction) * float(np.median(positive))


def build_quality_mask(
    nodes: np.ndarray,
    elements: np.ndarray,
    config: AristoConfig,
    *,
    volumes: np.ndarray | None = None,
    aspects: np.ndarray | None = None,
    max_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(quality_mask, volumes, aspects, max_edges)`` where True = stress-valid.
    """
    if volumes is None:
        volumes = element_volumes(nodes, elements)
    if aspects is None:
        aspects = element_aspect_ratios(nodes, elements)
    if max_edges is None:
        max_edges = element_max_edge_lengths(nodes, elements)

    v_min = resolve_min_tet_volume(volumes, config)
    v_max = resolve_max_tet_volume(volumes, config)
    e_max = resolve_max_tet_edge(config)
    abs_vols = np.abs(volumes)
    mask = (
        (abs_vols >= v_min)
        & (abs_vols <= v_max)
        & (aspects <= config.max_tet_aspect_ratio)
        & (max_edges <= e_max)
    )
    return mask, volumes, aspects, max_edges


def build_quality_gate_breakdown(
    nodes: np.ndarray,
    elements: np.ndarray,
    config: AristoConfig,
    *,
    volumes: np.ndarray | None = None,
    aspects: np.ndarray | None = None,
    max_edges: np.ndarray | None = None,
    quality_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Per-gate failure counts for poor tets (gates may overlap on the same element).

    ``primary_failure`` assigns each poor tet one exclusive reason (first failing
    gate in order: min volume → max volume → aspect ratio → max edge).
    """
    if volumes is None:
        volumes = element_volumes(nodes, elements)
    if aspects is None:
        aspects = element_aspect_ratios(nodes, elements)
    if max_edges is None:
        max_edges = element_max_edge_lengths(nodes, elements)
    if quality_mask is None:
        quality_mask, volumes, aspects, max_edges = build_quality_mask(
            nodes, elements, config, volumes=volumes, aspects=aspects, max_edges=max_edges
        )

    abs_vols = np.abs(volumes)
    v_min = resolve_min_tet_volume(volumes, config)
    v_max = resolve_max_tet_volume(volumes, config)
    e_max = resolve_max_tet_edge(config)
    ar_max = float(config.max_tet_aspect_ratio)

    fail_min_volume = abs_vols < v_min
    fail_max_volume = abs_vols > v_max
    fail_aspect_ratio = aspects > ar_max
    fail_max_edge = max_edges > e_max
    poor = ~quality_mask

    primary = np.full(elements.shape[0], "pass", dtype=object)
    for label, flag in (
        ("min_volume", fail_min_volume),
        ("max_volume", fail_max_volume),
        ("aspect_ratio", fail_aspect_ratio),
        ("max_edge", fail_max_edge),
    ):
        primary[(primary == "pass") & flag] = label

    n_poor = int(poor.sum())
    primary_counts: dict[str, int] = {}
    if n_poor:
        for label in ("min_volume", "max_volume", "aspect_ratio", "max_edge"):
            primary_counts[label] = int(np.sum((primary == label) & poor))

    return {
        "n_elements": int(elements.shape[0]),
        "n_poor_elements": n_poor,
        "poor_fraction": float(n_poor / elements.shape[0]) if elements.shape[0] else 0.0,
        "thresholds": {
            "min_tet_volume_mm3": v_min,
            "max_tet_volume_mm3": v_max,
            "max_tet_aspect_ratio": ar_max,
            "max_tet_edge_mm": e_max,
        },
        "fail_any_gate": {
            "min_volume": int(fail_min_volume.sum()),
            "max_volume": int(fail_max_volume.sum()),
            "aspect_ratio": int(fail_aspect_ratio.sum()),
            "max_edge": int(fail_max_edge.sum()),
        },
        "primary_failure_among_poor": primary_counts,
    }


def mesh_has_giant_elements(
    nodes: np.ndarray,
    elements: np.ndarray,
    config: AristoConfig,
) -> tuple[bool, float, float]:
    """
    Return ``(has_giants, max_edge_mm, max_volume/median_volume)`` for remesh gating.

    Only the max-edge check can reject a full mesh; volume ratios are diagnostic
    (porous lattices always have high vol/median from thin-wall tets).
    """
    volumes = element_volumes(nodes, elements)
    max_edges = element_max_edge_lengths(nodes, elements)
    abs_vols = np.abs(volumes)
    med_vol = float(np.median(abs_vols[abs_vols > 0])) if abs_vols.size else 0.0
    max_edge = float(max_edges.max()) if max_edges.size else 0.0
    vol_ratio = float(abs_vols.max() / med_vol) if med_vol > 0 else 0.0
    edge_cap = resolve_giant_tet_edge(config)
    return max_edge > edge_cap, max_edge, vol_ratio


def stress_percentiles(
    vm: np.ndarray,
    quality_mask: np.ndarray,
) -> dict[str, float]:
    """Stress percentile table on quality-valid elements (falls back to all)."""
    valid = vm[quality_mask] if quality_mask.any() else vm
    if valid.size == 0:
        return {
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "p50": float(np.percentile(valid, 50)),
        "p75": float(np.percentile(valid, 75)),
        "p90": float(np.percentile(valid, 90)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(valid.max()),
    }


def build_mesh_quality_report(
    nodes: np.ndarray,
    elements: np.ndarray,
    config: AristoConfig,
    *,
    vm_raw: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    aspects: np.ndarray | None = None,
    quality_mask: np.ndarray | None = None,
    remesh_attempts: int = 1,
) -> dict[str, Any]:
    """Diagnostic summary for JSON export and logging."""
    max_edges: np.ndarray | None = None
    if quality_mask is None:
        quality_mask, volumes, aspects, max_edges = build_quality_mask(
            nodes, elements, config, volumes=volumes, aspects=aspects
        )
    else:
        if volumes is None:
            volumes = element_volumes(nodes, elements)
        if aspects is None:
            aspects = element_aspect_ratios(nodes, elements)
        if max_edges is None:
            max_edges = element_max_edge_lengths(nodes, elements)

    n = int(elements.shape[0])
    n_poor = int(np.sum(~quality_mask))
    abs_vols = np.abs(volumes)
    v_min = resolve_min_tet_volume(volumes, config)
    v_max = resolve_max_tet_volume(volumes, config)
    e_max = resolve_max_tet_edge(config)
    med_vol = float(np.median(abs_vols[abs_vols > 0])) if n else 0.0

    report: dict[str, Any] = {
        "fea_quality_mode": config.fea_quality_mode,
        "n_elements": n,
        "n_poor_elements": n_poor,
        "poor_fraction": float(n_poor / n) if n else 0.0,
        "min_tet_volume_threshold_mm3": v_min,
        "max_tet_volume_threshold_mm3": v_max,
        "max_tet_edge_threshold_mm": e_max,
        "giant_tet_edge_threshold_mm": resolve_giant_tet_edge(config),
        "max_tet_aspect_ratio": config.max_tet_aspect_ratio,
        "volume_median_mm3": med_vol,
        "volume_p01_mm3": float(np.percentile(abs_vols, 1)) if n else 0.0,
        "max_edge_mm": float(max_edges.max()) if max_edges.size else 0.0,
        "max_edge_median_mm": float(np.median(max_edges)) if max_edges.size else 0.0,
        "max_volume_to_median_ratio": float(abs_vols.max() / med_vol) if med_vol > 0 else 0.0,
        "aspect_ratio_median": float(np.median(aspects)) if n else 0.0,
        "aspect_ratio_p99": float(np.percentile(aspects, 99)) if n else 0.0,
        "remesh_attempts": remesh_attempts,
    }

    if vm_raw is not None and vm_raw.size:
        all_p = stress_percentiles(vm_raw, np.ones(vm_raw.shape[0], dtype=bool))
        valid_p = stress_percentiles(vm_raw, quality_mask)
        p99 = valid_p["p99"]
        report["stress_all_elements"] = all_p
        report["stress_valid_elements"] = valid_p
        report["n_above_2x_p99_valid"] = int(
            np.sum((vm_raw >= 2.0 * p99) & quality_mask)
        ) if quality_mask.any() and p99 > 0 else 0
        if quality_mask.any():
            top_vol = abs_vols[np.argsort(vm_raw)[-1000:]].mean()
            med_vol = float(np.median(abs_vols[quality_mask]))
            report["mean_volume_top_1000_stress_mm3"] = float(top_vol)
            report["median_volume_valid_mm3"] = med_vol

    report["quality_gate_breakdown"] = build_quality_gate_breakdown(
        nodes,
        elements,
        config,
        volumes=volumes,
        aspects=aspects,
        max_edges=max_edges,
        quality_mask=quality_mask,
    )

    return report
