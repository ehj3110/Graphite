"""Tests for structured woodpile surface meshing."""

from __future__ import annotations

from graphite.case_studies.cube_1mm.generate_woodpile import cube_1mm_woodpile_spec
from graphite.explicit.woodpile_extrude import generate_piecewise_crosshatch_box
from graphite.explicit.woodpile_structured_surface import (
    BarBox,
    enumerate_woodpile_bar_boxes,
    structured_surface_mesh_from_bars,
    structured_surface_mesh_from_spec,
)
from graphite.implicit.woodpile_input import SPLITP_CUBE_MIS_PORE_BOTTOM_MM, SPLITP_CUBE_MIS_PORE_TOP_MM


def test_single_bar_structured_surface_watertight():
    bar = BarBox(0.0, 1.0, 0.0, 0.2, 0.0, 0.2)
    mesh, report = structured_surface_mesh_from_bars([bar], edge_length_mm=0.05)
    assert report["n_bars"] == 1
    assert mesh.is_watertight
    assert mesh.volume > 0
    # Grid edges respect h; face diagonals can reach sqrt(2) * h on square quads.
    assert report["edge_length_mm_max"] <= 0.05 * 1.42 + 1e-6


def test_piecewise_cube_structured_surface_matches_union_volume():
    pores = [SPLITP_CUBE_MIS_PORE_BOTTOM_MM, SPLITP_CUBE_MIS_PORE_TOP_MM]
    union_mesh, _union_report = generate_piecewise_crosshatch_box(
        width_x_mm=1.0,
        depth_y_mm=1.0,
        height_mm=1.0,
        z_breaks_mm=[0.0, 0.5, 1.0],
        pore_mm=pores,
        anchor_mode="center_void",
        alternate_band_orientation=True,
    )
    spec = cube_1mm_woodpile_spec(match_splitp_pores=True, generator="extrude")
    bars = enumerate_woodpile_bar_boxes(spec)
    assert len(bars) >= 20

    surf_mesh, report = structured_surface_mesh_from_spec(spec, edge_length_mm=0.008)
    assert surf_mesh.is_watertight
    assert surf_mesh.volume > 0
    assert report["faces"] > 5_000
    assert report["edge_length_mm_median"] <= 0.008 + 1e-6

    vol_union = float(union_mesh.volume)
    vol_surf = float(surf_mesh.volume)
    assert abs(vol_surf - vol_union) / vol_union < 0.02


def test_structured_surface_edge_lengths_uniform():
    spec = cube_1mm_woodpile_spec(match_splitp_pores=True, generator="extrude")
    mesh, report = structured_surface_mesh_from_spec(spec, edge_length_mm=0.01)
    assert report["edge_length_mm_median"] <= 0.0105
    assert float(mesh.edges_unique_length.max()) <= 0.01 * 1.42 + 1e-6
