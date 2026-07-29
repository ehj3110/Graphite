"""Tests for standalone hex surface dual (boundary quad graph)."""

from __future__ import annotations

import numpy as np
import trimesh

from graphite.explicit.hex_scaffold_module import (
    generate_conformed_hex_scaffold,
    synthesize_vf_gated_hex_volume_and_surface_dual,
)
from graphite.explicit.hex_surface_dual import (
    extract_ordered_boundary_quads,
    extract_ordered_boundary_quads_with_owners,
    generate_hex_surface_dual_cage,
    generate_hex_surface_dual_cage_volume_gated,
    generate_hex_surface_dual_on_surface_paths,
    get_boundary_quad_adjacency,
    hex_node_ids_from_elements,
)
from graphite.explicit.boundary_policy import (
    build_edt_sdf_sampler,
    calculate_hex_volume_fractions,
)


def test_route3_cube_boundary_quads_and_dual_connectivity() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, report = generate_conformed_hex_scaffold(
        box,
        target_element_size=3.0,
        shrink_wrap_relax=False,
    )
    assert len(hexes) > 0
    assert report.get("kept_hexes_after_shrink", 0) or len(hexes) > 0
    assert report.get("inversion_warning_hexes", 999) <= 60, report

    nodes, hex_node_ids = hex_node_ids_from_elements(hexes)
    quads = extract_ordered_boundary_quads(hex_node_ids)
    assert quads.shape[1] == 4
    assert len(quads) > 0

    pairs, _ = get_boundary_quad_adjacency(quads)
    assert len(pairs) > 0

    cage_nodes, struts = generate_hex_surface_dual_cage(
        nodes, quads, target_element_size=3.0
    )
    assert cage_nodes.shape[0] >= len(quads)
    assert len(struts) > 0
    assert np.all(struts < len(cage_nodes))


def test_symmetric_grid_bbox_center_on_centered_cube() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _ = generate_conformed_hex_scaffold(
        box, target_element_size=3.0, grid_anchor="bbox_center"
    )
    nodes, _ = hex_node_ids_from_elements(hexes)
    # Grid centered on origin -> node cloud should be nearly symmetric about 0.
    assert np.max(np.abs(nodes.mean(axis=0))) < 0.5


def test_corner_closure_adds_nodes_at_cube_corners() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _ = generate_conformed_hex_scaffold(
        box, target_element_size=3.0, grid_anchor="bbox_center"
    )
    nodes, hex_node_ids = hex_node_ids_from_elements(hexes)
    quads = extract_ordered_boundary_quads(hex_node_ids)
    cage_nodes, struts = generate_hex_surface_dual_cage(
        nodes,
        quads,
        target_element_size=3.0,
        include_corner_closure=True,
    )
    assert len(cage_nodes) > len(quads)
    assert len(struts) > 0


def test_surface_dual_strut_length_cap() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _ = generate_conformed_hex_scaffold(box, target_element_size=3.0)
    nodes, hex_node_ids = hex_node_ids_from_elements(hexes)
    quads = extract_ordered_boundary_quads(hex_node_ids)

    _, struts = generate_hex_surface_dual_cage(
        nodes, quads, target_element_size=3.0
    )
    centroids, _ = generate_hex_surface_dual_cage(nodes, quads, target_element_size=None)
    lengths = np.linalg.norm(
        centroids[struts[:, 0]] - centroids[struts[:, 1]], axis=1
    )
    assert np.all(lengths <= 1.5 * 3.0 + 1e-6)


def test_volume_gated_surface_dual_element_cage_on_sliver_hexes() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _, dbg = generate_conformed_hex_scaffold(
        box,
        target_element_size=3.0,
        neighbor_stretch=False,
        boundary_stretch_out=False,
        cull_mostly_external_hexes=False,
        laplacian_iterations=0,
        return_debug_payload=True,
    )
    pre = dbg["pre_conform_hex_elements"]
    nodes, hex_node_ids = hex_node_ids_from_elements(hexes)
    quads, owners, _face_idx = extract_ordered_boundary_quads_with_owners(hex_node_ids)
    sample_sdf = build_edt_sdf_sampler(box, 0.5)
    vf = calculate_hex_volume_fractions(pre, sample_sdf)
    has_internal = vf > 0.5

    _, struts, rep = generate_hex_surface_dual_cage_volume_gated(
        nodes,
        quads,
        owners,
        has_internal,
        target_element_size=3.0,
    )
    assert rep["n_hexes_without_internal"] > 0
    assert rep["n_hexes_with_internal"] > 0
    # Adjacent links must include sliver<->neighbor (incl. perpendicular faces at corners).
    assert rep["n_adjacent_struts"] >= 480
    assert len(struts) >= rep["n_adjacent_struts"]


def test_synthesize_vf_gated_merged_lattice() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _, dbg = generate_conformed_hex_scaffold(
        box,
        target_element_size=3.0,
        neighbor_stretch=False,
        boundary_stretch_out=False,
        cull_mostly_external_hexes=False,
        laplacian_iterations=0,
        return_debug_payload=True,
    )
    nodes, struts, rep = synthesize_vf_gated_hex_volume_and_surface_dual(
        hexes,
        box,
        hex_elements_for_vf=dbg["pre_conform_hex_elements"],
        target_element_size=3.0,
        hex_rule="octahedral",
    )
    assert len(nodes) > 0
    assert len(struts) > 0
    assert rep["n_hexes"] == len(hexes)
    assert rep["n_hexes_with_internal_lattice"] < rep["n_hexes"]
    assert rep["merged_struts"] == len(struts)
    assert rep.get("volume_on_all_hexes") is not True
    assert rep.get("merge_mode") == "explicit_hex_face_topology"
    assert rep.get("n_skin_quads_mapped_to_volume", 0) > 0


def test_synthesize_volume_on_all_hexes() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _, dbg = generate_conformed_hex_scaffold(
        box,
        target_element_size=3.0,
        neighbor_stretch=False,
        boundary_stretch_out=False,
        cull_mostly_external_hexes=False,
        laplacian_iterations=0,
        return_debug_payload=True,
    )
    _, _, rep = synthesize_vf_gated_hex_volume_and_surface_dual(
        hexes,
        box,
        hex_elements_for_vf=dbg["pre_conform_hex_elements"],
        volume_on_all_hexes=True,
        hex_rule="octahedral",
    )
    assert rep["volume_on_all_hexes"] is True
    assert rep["n_hexes_with_volume_lattice"] == rep["n_hexes"]


def test_stl_surface_skin_nodes_on_mesh() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    from graphite.explicit.stl_surface_skin import generate_stl_surface_path_skin

    nodes, struts, rep = generate_stl_surface_path_skin(box, project_to_surface=True)
    assert len(nodes) > 0
    assert len(struts) > 0
    assert rep["surface_skin_mode"] == "stl_surface_path"
    assert rep["max_node_surface_offset_mm"] < 0.05


def test_surface_path_dual_more_struts_than_centroid_chords() -> None:
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    hexes, _, dbg = generate_conformed_hex_scaffold(
        box,
        target_element_size=3.0,
        neighbor_stretch=False,
        boundary_stretch_out=False,
        cull_mostly_external_hexes=False,
        laplacian_iterations=0,
        return_debug_payload=True,
    )
    nodes, hex_node_ids = hex_node_ids_from_elements(hexes)
    quads, owners, _face_idx = extract_ordered_boundary_quads_with_owners(hex_node_ids)
    pre = dbg["pre_conform_hex_elements"]
    sample_sdf = build_edt_sdf_sampler(box, 0.5)
    vf = calculate_hex_volume_fractions(pre, sample_sdf)

    path_nodes, path_struts, path_rep = generate_hex_surface_dual_on_surface_paths(
        nodes, quads, owners, vf > 0.5, target_element_size=3.0
    )
    assert path_rep["surface_dual_mode"] == "on_surface_paths"
    pairs, _ = get_boundary_quad_adjacency(quads)
    assert path_rep["n_path_struts"] == 2 * len(pairs)
    assert len(path_nodes) > len(quads)
    assert len(path_struts) > len(quads)
