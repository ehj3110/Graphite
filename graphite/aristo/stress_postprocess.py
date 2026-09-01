"""
Nodal stress recovery from P1 tetrahedral element fields (Phase 2).
"""

from __future__ import annotations

import numpy as np

from graphite.aristo.aristo_config import AristoConfig


def element_to_nodal_von_mises(
    elements: np.ndarray,
    element_vm: np.ndarray,
    element_volumes: np.ndarray,
    quality_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Volume-weighted patch average of element von Mises onto nodes.

    Returns ``(nodal_vm, nodal_support)`` where ``nodal_support[i]`` is True if
    node ``i`` received contributions from at least one (quality-valid) element.
    """
    n_nodes = int(elements.max()) + 1
    weights = np.abs(np.asarray(element_volumes, dtype=np.float64))
    vm = np.asarray(element_vm, dtype=np.float64)

    if quality_mask is not None:
        q = np.asarray(quality_mask, dtype=bool)
        weights = np.where(q, weights, 0.0)

    contrib = weights * vm
    numer = np.zeros(n_nodes, dtype=np.float64)
    denom = np.zeros(n_nodes, dtype=np.float64)

    for local_idx in range(4):
        node_ids = elements[:, local_idx]
        np.add.at(numer, node_ids, contrib)
        np.add.at(denom, node_ids, weights)

    nodal = np.zeros(n_nodes, dtype=np.float64)
    support = denom > 0.0
    nodal[support] = numer[support] / denom[support]
    return nodal, support


def nodal_to_element_mean(
    nodal_vm: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """Mean nodal stress on each tet (for element-colored iso solids)."""
    return nodal_vm[elements].mean(axis=1)


def build_stress_fields(
    vm_element_raw: np.ndarray,
    elements: np.ndarray,
    element_volumes: np.ndarray,
    quality_mask: np.ndarray,
    config: AristoConfig,
) -> dict[str, np.ndarray | float | str]:
    """
    Build normalized element/nodal stress fields and peak metrics.

    ``nodal_averaged`` mode normalizes by the **nodal** peak (user preference).
  ``element_raw`` keeps Phase-1 element-peak normalization.
    """
    mode = config.stress_representation
    max_vm_element = float(vm_element_raw.max()) if vm_element_raw.size else 0.0
    vm_valid = vm_element_raw[quality_mask] if quality_mask.any() else vm_element_raw

    if mode == "nodal_averaged":
        nodal_raw, nodal_support = element_to_nodal_von_mises(
            elements, vm_element_raw, element_volumes, quality_mask
        )
        if nodal_support.any():
            max_nodal = float(nodal_raw[nodal_support].max())
            p75 = float(np.quantile(nodal_raw[nodal_support], 0.75))
            hotspot = float(np.mean(nodal_raw[nodal_support] >= p75))
        else:
            max_nodal = 0.0
            p75 = 0.0
            hotspot = 0.0

        denom = max_nodal + 1e-15
        nodal_norm = np.zeros_like(nodal_raw)
        nodal_norm[nodal_support] = nodal_raw[nodal_support] / denom

        vm_norm = np.zeros_like(vm_element_raw, dtype=np.float64)
        if quality_mask.any():
            vm_norm[quality_mask] = vm_element_raw[quality_mask] / denom

        return {
            "stress_field_mode": mode,
            "von_mises": vm_norm,
            "von_mises_nodal_raw": nodal_raw,
            "von_mises_nodal_norm": nodal_norm,
            "max_von_mises_raw": max_nodal,
            "max_von_mises_element_raw": max_vm_element,
            "hotspot_fraction": hotspot,
        }

    max_element = float(vm_valid.max()) if vm_valid.size else 0.0
    denom = max_element + 1e-15
    vm_norm = np.zeros_like(vm_element_raw, dtype=np.float64)
    if quality_mask.any():
        vm_norm[quality_mask] = vm_element_raw[quality_mask] / denom
    p75 = float(np.quantile(vm_valid, 0.75)) if vm_valid.size else 0.0
    hotspot = (
        float(np.mean(vm_element_raw[quality_mask] >= p75))
        if quality_mask.any()
        else 0.0
    )
    n_nodes = int(elements.max()) + 1
    return {
        "stress_field_mode": mode,
        "von_mises": vm_norm,
        "von_mises_nodal_raw": np.zeros(n_nodes, dtype=np.float64),
        "von_mises_nodal_norm": np.zeros(n_nodes, dtype=np.float64),
        "max_von_mises_raw": max_element,
        "max_von_mises_element_raw": max_vm_element,
        "hotspot_fraction": hotspot,
    }
