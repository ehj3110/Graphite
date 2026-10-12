"""
tests/test_transition_catalog.py

Comprehensive Unit Test Battery for PAM Transition Catalog & Recommendation Engine:
- Validates O(1) instantaneous lookup latency (< 1 ms).
- Verifies golden first-principles transitions (c6tt-d4tet, d4cube-d4tet, c6tt-b8co).
- Verifies same-cell plane-to-plane and rotational transitions.
- Verifies that frustrated/buffered transitions provide non-empty root causes and actionable routes.
- Verifies graph-based shortest path route recommendation.
"""

import time
import pytest
import numpy as np

from graphite.explicit.interlinked import (
    TransitionCatalogEntry,
    get_transition_entry,
    list_feasible_transitions,
    recommend_transition_path,
    is_transition_feasible,
)
from graphite.explicit.interlinked.catalog import _load_catalog


def test_catalog_loads_all_candidates():
    """Verify complete catalog loading across all 5 cores and 3 habit planes."""
    cat = _load_catalog()
    # 30 same-cell combinations + 60 directed inter-cell combinations = 90 total
    assert len(cat) == 90, f"Expected 90 precomputed transitions, found {len(cat)}"

    direct_entries = [e for e in cat.values() if e.classification == "DIRECT"]
    buffered_entries = [e for e in cat.values() if e.classification == "BUFFERED"]
    assert len(direct_entries) == 36, f"Expected 36 DIRECT transitions, found {len(direct_entries)}"
    assert len(buffered_entries) == 54, f"Expected 54 BUFFERED transitions, found {len(buffered_entries)}"


def test_verified_first_principles_heteroepitaxy():
    """Verify the 4 golden first-principles multi-zone heteroepitaxial standards."""
    # 1. D4-CUBE -> D4-TET on {111}
    e1_fwd = get_transition_entry("d4cube", "d4tet", habit_plane="111")
    e1_rev = get_transition_entry("d4tet", "d4cube", habit_plane="111")
    for e in (e1_fwd, e1_rev):
        assert e is not None
        assert e.classification == "DIRECT"
        assert e.is_feasible is True
        assert e.gap_mm == 4.90
        assert e.twist_deg == 0.0
        assert e.min_clearance_um == 135.0
        assert e.boundary_links == 3

    # 2. D4-CUBE -> D4-TET on {110}
    e2_fwd = get_transition_entry("d4cube", "d4tet", habit_plane="110")
    e2_rev = get_transition_entry("d4tet", "d4cube", habit_plane="110")
    for e in (e2_fwd, e2_rev):
        assert e is not None
        assert e.classification == "DIRECT"
        assert e.is_feasible is True
        assert e.gap_mm == 4.00
        assert e.twist_deg == 0.0
        assert e.min_clearance_um == 104.7
        assert e.boundary_links == 5

    # 3. C6-TT -> B8-CO on {100}
    e3_fwd = get_transition_entry("c6tt", "b8co", habit_plane="100")
    e3_rev = get_transition_entry("b8co", "c6tt", habit_plane="100")
    for e in (e3_fwd, e3_rev):
        assert e is not None
        assert e.classification == "DIRECT"
        assert e.is_feasible is True
        assert e.gap_mm == 6.00
        assert e.twist_deg == 0.0
        assert e.min_clearance_um == 183.7
        assert e.boundary_links == 4

    # 4. C6-TT -> D4-TET on {100} (Science 2025 benchmark)
    e4_fwd = get_transition_entry("c6tt", "d4tet", habit_plane="100")
    e4_rev = get_transition_entry("d4tet", "c6tt", habit_plane="100")
    for e in (e4_fwd, e4_rev):
        assert e is not None
        assert e.classification == "DIRECT"
        assert e.is_feasible is True
        assert e.gap_mm == 7.00
        assert e.twist_deg == 45.0
        assert e.min_clearance_um == 191.4
        assert e.boundary_links == 4


def test_same_cell_plane_transitions():
    """Verify homogeneous stacking and plane-to-plane reorientations within each cell."""
    cells = ["c6tt", "d4tet", "d4cube", "b8co", "j4oct"]

    # All 5 cells support homogeneous stacking {100}->{100}, {110}->{110}, {111}->{111}
    for c in cells:
        for p in ("100", "110", "111"):
            entry = get_transition_entry(c, c, plane_a=p, plane_b=p)
            assert entry is not None, f"Missing same-cell entry for {c} {p}->{p}"
            assert entry.classification == "DIRECT"
            assert entry.is_feasible is True
            assert entry.min_clearance_um > 50.0
            assert entry.boundary_links >= 1

    # Cross-plane reorientation tests
    c6tt_100_111 = get_transition_entry("c6tt", "c6tt", plane_a="100", plane_b="111")
    assert c6tt_100_111 is not None and c6tt_100_111.is_feasible is True

    d4cube_100_111 = get_transition_entry("d4cube", "d4cube", plane_a="100", plane_b="111")
    assert d4cube_100_111 is not None and d4cube_100_111.is_feasible is True

    # B8CO {100}->{111} is frustrated/buffered
    b8co_frustrated = get_transition_entry("b8co", "b8co", plane_a="100", plane_b="111")
    assert b8co_frustrated is not None
    assert b8co_frustrated.classification == "BUFFERED"
    assert b8co_frustrated.is_feasible is False
    assert len(b8co_frustrated.recommended_routes) > 0


def test_instantaneous_lookup_latency():
    """Verify that catalog queries complete in well under 1 ms (< 100 microseconds)."""
    # Warmup
    get_transition_entry("d4cube", "d4tet", habit_plane="111")

    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        entry = get_transition_entry("d4cube", "d4tet", habit_plane="111")
        assert entry is not None
    avg_latency_us = (time.perf_counter() - t0) / N * 1e6

    # Must be less than 50 µs
    assert avg_latency_us < 50.0, f"Lookup latency too high: {avg_latency_us:.2f} µs"


def test_frustrated_and_buffered_recommendations():
    """Verify that frustrated combinations return actionable non-frustrated bridge paths."""
    # D4-TET to D4-CUBE on {100}
    entry = get_transition_entry("d4tet", "d4cube", habit_plane="100")
    assert entry is not None
    assert entry.classification == "BUFFERED"
    assert entry.is_feasible is False
    assert len(entry.failure_reason) > 0
    assert len(entry.recommended_routes) >= 1

    # Must suggest path via {111} or {110} or continuous morph
    routes_text = " ".join(entry.recommended_routes)
    assert ("111" in routes_text) or ("110" in routes_text) or ("d4ct" in routes_text)


def test_recommend_transition_path():
    """Verify graph BFS shortest path recommendations between disparate systems."""
    # From D4-CUBE {100} to B8-CO {100}
    paths = recommend_transition_path("d4cube", "b8co", plane_a="100", plane_b="100")
    assert len(paths) >= 1
    p0 = paths[0]
    # Path should traverse through d4tet and c6tt
    assert "d4tet" in p0
    assert "c6tt" in p0
    assert "b8co" in p0

    # Direct case
    direct_path = recommend_transition_path("d4cube", "d4tet", plane_a="111", plane_b="111")
    assert "DIRECT" in direct_path[0]


def test_list_feasible_transitions():
    """Verify filtering and querying of feasible transitions."""
    all_feasible = list_feasible_transitions()
    assert len(all_feasible) == 36

    d4tet_feasible = list_feasible_transitions("d4tet")
    assert len(d4tet_feasible) > 0
    for e in d4tet_feasible:
        assert e.classification == "DIRECT"
        assert e.cell_a == "d4tet" or e.cell_b == "d4tet"


def test_is_transition_feasible_helper():
    """Verify boolean feasibility helper function."""
    assert is_transition_feasible("d4cube", "d4tet", habit_plane="111") is True
    assert is_transition_feasible("d4cube", "d4tet", habit_plane="110") is True
    assert is_transition_feasible("c6tt", "b8co", habit_plane="100") is True
    assert is_transition_feasible("c6tt", "d4tet", habit_plane="100") is True

    # Inadmissible direct cuts
    assert is_transition_feasible("d4cube", "d4tet", habit_plane="100") is False
    assert is_transition_feasible("c6tt", "d4cube", habit_plane="100") is False
    assert is_transition_feasible("c6tt", "j4oct", habit_plane="100") is False
