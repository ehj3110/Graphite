"""
tests/test_interlinked_search.py

Unit and integration tests for the bounded deterministic interface search engine (Phase 4).
Verifies:
- Bounded coarse-to-fine parameter sweep.
- Feasible catenation candidate discovery with positive clearance.
- Direct InterfaceSpec conversion and assembly integration.
- Time budget enforcement.
- Deprecation warnings on legacy Powell solver.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from graphite.explicit.interlinked import (
    ZoneSpec,
    InterfaceSpec,
    assemble,
    search_interface,
    InterfaceCandidate,
    InterfaceSweepResult,
    optimize_interface_phase_and_taper,
)
from graphite.explicit.interlinked.legacy import (
    optimize_interface_phase_and_taper_powell,
)


class TestInterlinkedSearch:
    """Test bounded deterministic interface sweep."""

    def test_search_interface_c6tt_to_d4tet(self):
        """Verify search_interface finds feasible catenated parameters on {100}."""
        a_sc = 9.10
        a_dia = np.sqrt(2.0) * a_sc

        t0 = time.perf_counter()
        res = search_interface(
            zone_a="c6tt",
            zone_b="d4tet",
            pitch_a=a_sc,
            pitch_b=a_dia,
            habit_plane="100",
            min_clearance=0.150,
            budget_s=2.0,
        )
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.5, f"Search took {elapsed:.2f}s, expected < 2.5s"
        assert res.total_evaluated > 0
        assert res.best_candidate is not None
        assert res.is_feasible is True
        assert res.best_candidate.boundary_links >= 1
        assert res.best_candidate.min_clearance >= 0.150

    def test_search_interface_zonespec_input(self):
        """Verify search_interface accepts ZoneSpec objects directly."""
        za = ZoneSpec(cell="c6tt", network="pcu", pitch=9.10, wire_radius=0.35)
        zb = ZoneSpec(cell="d4tet", network="dia", pitch=12.869, wire_radius=0.35)

        res = search_interface(za, zb, habit_plane="100", min_clearance=0.150)
        assert res.is_feasible is True
        assert res.best_candidate is not None
        assert res.best_candidate.boundary_links >= 1

    def test_search_interface_to_interfacespec_assembly(self):
        """Verify search result converts directly to InterfaceSpec and runs assemble()."""
        za = ZoneSpec(cell="c6tt", network="pcu", pitch=9.10, grid_size=(1, 1, 1), wire_radius=0.35)
        zb = ZoneSpec(cell="d4tet", network="dia", pitch=12.869, grid_size=(1, 1, 1), wire_radius=0.35)

        sweep_res = search_interface(za, zb, habit_plane="100", min_clearance=0.150)
        assert sweep_res.is_feasible is True

        iface = sweep_res.to_interface_spec()
        assert isinstance(iface, InterfaceSpec)

        asm_res = assemble(zones=[za, zb], interfaces=[iface], wire_radius=0.35, strict=False)
        assert asm_res.mesh is not None
        assert asm_res.mesh.is_watertight
        assert asm_res.num_rings > 0

    def test_search_budget_enforcement(self):
        """Verify budget_s stops the search early if a tight budget is set."""
        t0 = time.perf_counter()
        res = search_interface(
            "c6tt",
            "d4tet",
            pitch_a=9.10,
            pitch_b=12.87,
            budget_s=0.3,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Search exceeded tight budget: elapsed {elapsed:.2f}s"

    def test_legacy_solver_deprecation_warning(self):
        """Verify legacy Powell solver triggers DeprecationWarning."""
        with pytest.deprecated_call():
            nodes_dummy = np.zeros((4, 3))
            struts_dummy = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
            optimize_interface_phase_and_taper(
                nodes_a=nodes_dummy,
                struts_a=struts_dummy,
                nodes_b=nodes_dummy,
                struts_b=struts_dummy,
            )

        with pytest.deprecated_call():
            optimize_interface_phase_and_taper_powell(
                nodes_a=nodes_dummy,
                struts_a=struts_dummy,
                nodes_b=nodes_dummy,
                struts_b=struts_dummy,
            )
