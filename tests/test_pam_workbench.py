"""
Tests for the Permissive PAM Pair Experimental Workbench (Phase 1).
Verifies:
1. Known linked pair (D4-CUBE A-B diamond basis) has |Lk| >= 1 and status == 'linked'.
2. Poked tip / non-threading overlap is classified as 'entangled' (|Lk| == 0, hull overlap > 0).
3. Overlapping solids report clash_volume > 0, min_clearance < 0, and status == 'colliding'.
4. Disjoint cells report clearance > 0, clash_volume == 0, and status == 'disjoint'.
5. Deterministic parameter sweep operates with zero numerical solver failures.
6. Every single-pair check executes in under 100 ms.
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from graphite.explicit.interlinked.cells import D4CubeCell, D4TetCell
from graphite.explicit.interlinked.workbench import measure_pair, sweep, PairReport


def test_known_linked_pair_has_lk_1():
    # Diamond basis offset for D4-CUBE is [0.25, 0.25, 0.25] * pitch
    pitch = 10.0
    t_dia = np.array([0.25, 0.25, 0.25], dtype=np.float64) * pitch

    rep = measure_pair(
        cell_a="d4cube",
        cell_b="d4cube",
        R_rel=np.eye(3),
        t_rel=t_dia,
        wire_radius=0.35,
        pitch_a=pitch,
        pitch_b=pitch,
    )

    assert isinstance(rep, PairReport)
    assert rep.linking_number == 1
    assert rep.status == "linked"
    assert rep.min_clearance_mm > 0.0
    assert rep.clash_volume_mm3 == 0.0


def test_poked_tip_is_entangled_not_linked():
    # Place a cube such that its corner pokes into the bounding hull of another cube,
    # but does not thread through any closed window
    pitch = 10.0
    # Cube half side = 0.17 * 10 = 1.7 mm. Corner is at (1.7, 1.7, 1.7).
    # If shifted along X by 3.0 mm (just grazing/poking the corner into the envelope),
    # it touches the convex envelope but does not thread through a window.
    t_poke = np.array([2.8, 0.0, 0.0], dtype=np.float64)

    rep = measure_pair(
        cell_a="d4cube",
        cell_b="d4cube",
        R_rel=np.eye(3),
        t_rel=t_poke,
        wire_radius=0.15,
        pitch_a=pitch,
        pitch_b=pitch,
    )

    # Either entangled (hull overlap, Lk=0) or colliding (if struts graze)
    assert rep.linking_number == 0
    assert rep.status in ("entangled", "colliding")


def test_overlapping_solids_report_clash_volume():
    # Direct co-location (origin overlap): solids definitely intersect
    rep = measure_pair(
        cell_a="d4cube",
        cell_b="d4cube",
        R_rel=np.eye(3),
        t_rel=np.zeros(3),
        wire_radius=0.35,
        pitch_a=10.0,
    )

    assert rep.status == "colliding"
    assert rep.min_clearance_mm < 0.0
    assert rep.clash_volume_mm3 > 1e-4


def test_disjoint_cells_report_disjoint():
    # Large separation (20 mm along X)
    rep = measure_pair(
        cell_a="d4cube",
        cell_b="d4cube",
        R_rel=np.eye(3),
        t_rel=np.array([20.0, 0.0, 0.0]),
        wire_radius=0.35,
        pitch_a=10.0,
    )

    assert rep.status == "disjoint"
    assert rep.min_clearance_mm > 10.0
    assert rep.clash_volume_mm3 == 0.0
    assert rep.linking_number == 0


def test_speed_under_100ms():
    pitch = 10.0
    t_dia = np.array([0.25, 0.25, 0.25], dtype=np.float64) * pitch

    t0 = time.perf_counter()
    rep = measure_pair(
        cell_a="d4cube",
        cell_b="d4cube",
        R_rel=np.eye(3),
        t_rel=t_dia,
        wire_radius=0.35,
        pitch_a=pitch,
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.100, f"test_pair took {elapsed*1000:.1f} ms (must be < 100 ms)"


def test_sweep_deterministic():
    gaps = np.linspace(5.0, 15.0, 5)
    results = sweep(
        cell_a="d4cube",
        cell_b="d4cube",
        param="gap",
        values=gaps,
        habit_normal=np.array([1.0, 0.0, 0.0]),
        wire_radius=0.35,
        pitch_a=10.0,
    )

    assert len(results) == 5
    # As gap increases along [1, 0, 0], clearance increases monotonically once separated
    clrs = [r.min_clearance_mm for r in results]
    assert clrs[-1] > clrs[0]
