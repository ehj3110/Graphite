from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.implicit.pore_metrics import (
    compute_cross_sectional_pore_size,
    compute_effective_pore_size,
    compute_max_inscribed_sphere_pore_size,
    compute_pore_metrics_for_z_graded,
)


def test_effective_pore_size_increases_with_void_radius() -> None:
    nx = ny = nz = 48
    axis = np.linspace(-1.0, 1.0, nx)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    r_small = 0.35
    r_large = 0.55
    void_small = (X**2 + Y**2 + Z**2) <= r_small**2
    void_large = (X**2 + Y**2 + Z**2) <= r_large**2
    resolution_mm = 0.05

    small = compute_effective_pore_size(void_small, resolution_mm=resolution_mm, z_bins=6)
    large = compute_effective_pore_size(void_large, resolution_mm=resolution_mm, z_bins=6)

    assert large.effective_global_mm > small.effective_global_mm
    assert large.effective_p50_mm > small.effective_p50_mm


def test_cross_sectional_metrics_monotonic_for_splitp() -> None:
    sf = 0.33
    low = compute_cross_sectional_pore_size(
        lattice_type="Split-P",
        pore_size_mm=0.2,
        solid_fraction=sf,
        unit_cell_resolution_mm=0.01,
    )
    high = compute_cross_sectional_pore_size(
        lattice_type="Split-P",
        pore_size_mm=0.8,
        solid_fraction=sf,
        unit_cell_resolution_mm=0.01,
    )

    assert low.cross_section_2d_mm is not None
    assert low.cross_section_3d_mm is not None
    assert high.cross_section_2d_mm is not None
    assert high.cross_section_3d_mm is not None
    assert high.cross_section_2d_mm > low.cross_section_2d_mm
    assert high.cross_section_3d_mm > low.cross_section_3d_mm


def test_z_graded_profile_and_confidence_guard() -> None:
    z = [0.0, 2.4, 4.8]
    pores = [0.2, 0.5, 0.8]
    result = compute_pore_metrics_for_z_graded(
        lattice_type="Gyroid",
        z_samples_mm=z,
        pore_sizes_mm=pores,
        solid_fraction=0.33,
        unit_cell_resolution_mm=0.01,
    )
    assert result.z_samples_mm == z
    assert len(result.cross_section_2d_mm) == len(z)
    assert len(result.cross_section_3d_mm) == len(z)
    assert result.cross_section_2d_mm[0] < result.cross_section_2d_mm[-1]
    assert result.cross_section_3d_mm[0] < result.cross_section_3d_mm[-1]

    coarse = compute_cross_sectional_pore_size(
        lattice_type="Split-P",
        pore_size_mm=0.2,
        solid_fraction=0.33,
        unit_cell_resolution_mm=0.12,
    )
    assert coarse.confidence in {"low", "medium"}
    assert len(coarse.notes) >= 1


def test_max_inscribed_sphere_boundary_guard_reduces_edge_bias() -> None:
    nx = ny = nz = 30
    void_mask = np.zeros((nx, ny, nz), dtype=bool)
    # Open channel touching boundary (can produce edge-favored centers).
    void_mask[:, 12:18, 12:18] = True
    resolution_mm = 0.05

    raw = compute_max_inscribed_sphere_pore_size(
        void_mask,
        resolution_mm=resolution_mm,
        boundary_guard_mm=0.0,
        require_sphere_within_domain=False,
    )
    guarded = compute_max_inscribed_sphere_pore_size(
        void_mask,
        resolution_mm=resolution_mm,
        boundary_guard_mm=0.1,
        require_sphere_within_domain=True,
    )
    assert raw.max_diameter_mm >= guarded.max_diameter_mm

