"""Unit and smoke tests for Graphite Implicit Studio (state, VTK scene, slice, GUI)."""

from __future__ import annotations

import inspect
from itertools import combinations
import os
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
import pytest
import vtk

from graphite.implicit.conformal import generate_conformal_lattice
from graphite.ui.implicit_state import (
    FOUR_WAY_METRICS,
    GRADING_PRESETS,
    PREVIEW_MODES,
    ImplicitStudioState,
    build_or_load_cad_mesh,
    compute_four_way_metrics,
    discover_cad_fixtures,
    estimate_voxel_budget,
    evaluate_grading_profile_1d,
    has_threemf_support,
    recommended_nyquist_resolution_mm,
    repair_cad_mesh_in_memory,
    run_implicit_generation,
    run_recipe_batch,
    run_tpms_calibration,
    solve_four_way_pair,
)
from graphite.ui.implicit_studio import ImplicitStudioWindow
from graphite.ui.implicit_vtk_scene import (
    ImplicitVtkScene,
    _compute_preview_spacing,
    _get_cached_subdivided_surface,
    build_micropillar_preview_polydata,
    build_surface_footprint_polydata,
    compute_2d_slice_image,
    vtk_polydata_to_trimesh,
)

TEST_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "pytest_implicit_studio"


def test_state_recipe_roundtrip_and_presets(tmp_path: Path) -> None:
    sig = inspect.signature(generate_conformal_lattice)
    assert sig.parameters["exact_cad_trim"].default is True

    dflt = ImplicitStudioState()
    assert dflt.size_driver == ""
    assert dflt.density_driver == ""
    assert dflt.unit_cell_size_mm == pytest.approx(1.50, abs=1e-6)
    assert dflt.solid_fraction == pytest.approx(0.33, abs=1e-6)
    assert dflt.wall_thickness_mm == pytest.approx(0.50, abs=1e-6)
    assert dflt.pore_size_mm == pytest.approx(0.80, abs=1e-6)
    assert dflt.exact_cad_trim is True
    assert dflt.export_mode == "core"
    assert PREVIEW_MODES[1] == "3D Flying Edges (Coarse)"

    legacy = ImplicitStudioState.from_recipe_dict(
        {"preview_mode": "3D Flying Edges (Fine)", "exact_cad_trim": False}
    )
    assert legacy.preview_mode == "3D Flying Edges (Coarse)"
    assert legacy.exact_cad_trim is True

    s0 = ImplicitStudioState(
        lattice_type="Split-P",
        pore_size_mm=2.4,
        solid_fraction=0.38,
        show_generated_3d_mesh=False,
    )
    for preset in GRADING_PRESETS:
        s0.apply_preset(preset)
        recipe_file = tmp_path / f"{preset.replace(' ', '_')}.json"
        s0.save_recipe(recipe_file)
        s1 = ImplicitStudioState.load_recipe(recipe_file)
        assert s1.lattice_type == "Split-P"
        assert s1.grading_mode == s0.grading_mode
        assert len(s1.multi_knot_knots) == len(s0.multi_knot_knots)
        assert s1.show_generated_3d_mesh is False


def test_four_way_sizing_solver_and_nyquist() -> None:
    m_pore = compute_four_way_metrics(
        "Gyroid",
        driven_by="pore_size",
        unit_cell_size_mm=5.0,
        pore_size_mm=3.0,
        solid_fraction=0.33,
        wall_thickness_mm=0.5,
    )
    assert m_pore["unit_cell_size_mm"] > m_pore["pore_size_mm"]
    assert m_pore["recommended_resolution_mm"] == pytest.approx(
        recommended_nyquist_resolution_mm(m_pore["wall_thickness_mm"]), abs=1e-4
    )

    m_wall = compute_four_way_metrics(
        "Split-P",
        driven_by="wall_thickness",
        unit_cell_size_mm=4.0,
        pore_size_mm=2.5,
        solid_fraction=0.35,
        wall_thickness_mm=0.45,
        size_driver="unit_cell",
    )
    assert 0.05 < m_wall["solid_fraction"] < 0.90

    m_wood = compute_four_way_metrics(
        "Woodpile",
        driven_by="pore_size",
        unit_cell_size_mm=2.0,
        pore_size_mm=1.2,
        solid_fraction=0.40,
        wall_thickness_mm=0.6,
    )
    assert m_wood["unit_cell_size_mm"] == pytest.approx(1.8, abs=1e-4)

    metric_keys = [k for k, _lbl in FOUR_WAY_METRICS]
    for arch in ("Gyroid", "Woodpile"):
        for d1, d2 in combinations(metric_keys, 2):
            solved = solve_four_way_pair(
                arch,
                d1,
                d2,
                unit_cell_size_mm=2.4,
                pore_size_mm=1.2,
                solid_fraction=0.33,
                wall_thickness_mm=0.50,
            )
            assert solved["unit_cell_size_mm"] >= 0.1
            assert solved["pore_size_mm"] >= 0.05
            assert 0.05 <= solved["solid_fraction"] <= 0.90
            assert solved["wall_thickness_mm"] >= 0.02


def test_cad_build_health_repair_and_budget() -> None:
    s = ImplicitStudioState(primitive_shape="Cube", prim_x_mm=10.0, prim_y_mm=10.0, prim_z_mm=10.0)
    mesh, health = build_or_load_cad_mesh(s)
    assert health["is_watertight"] is True
    assert health["face_count"] > 0

    repaired, rep_health = repair_cad_mesh_in_memory(mesh)
    assert rep_health["is_watertight"] is True
    assert len(repaired.faces) > 0

    bud = estimate_voxel_budget(mesh, 0.25)
    assert bud["voxels"] > 1000
    assert bud["est_ram_mb"] > 0.0

    fixtures = discover_cad_fixtures()
    assert isinstance(fixtures, list)


def test_surface_footprint_and_2d_slice_across_modes() -> None:
    state = ImplicitStudioState(prim_x_mm=8.0, prim_y_mm=8.0, prim_z_mm=8.0, pore_size_mm=2.5)
    cad_mesh, _ = build_or_load_cad_mesh(state)

    for mode in ("Uniform", "Multi-Knot Control Table", "Piecewise Discrete Bands"):
        state.grading_mode = mode
        fp = build_surface_footprint_polydata(state, cad_mesh)
        assert fp.GetNumberOfPoints() > 0

        sl = compute_2d_slice_image(state, cad_mesh, axis="Z", slice_pos_pct=50.0, pixel_size_mm=0.08)
        assert sl.rgb_image.ndim == 3 and sl.rgb_image.shape[2] == 3
        assert 0.0 < sl.solid_area_fraction < 1.0

        prof = evaluate_grading_profile_1d(state, cad_mesh)
        assert len(prof["pos_mm"]) == len(prof["L_mm"]) == len(prof["sf"])


def test_calibration_and_end_to_end_generation() -> None:
    state = ImplicitStudioState(
        primitive_shape="Cube",
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Gyroid",
        size_driver="pore_size",
        density_driver="solid_fraction",
        pore_size_mm=2.5,
        solid_fraction=0.33,
        auto_resolution=False,
        resolution_mm=0.25,
        export_stl=True,
        export_3mf=True,
        write_manifest=True,
        run_post_mis_check=True,
        custom_output_name="pytest_implicit_studio_cube_gyroid",
    )
    cal = run_tpms_calibration(state)
    assert cal.calibrated_L_mm > 0.5
    assert cal.calibrated_tau > 0.05

    cad_mesh, _ = build_or_load_cad_mesh(state)
    res = run_implicit_generation(state, cad_mesh=cad_mesh, output_root=TEST_OUTPUT_ROOT)
    assert len(res.mesh.faces) > 100
    assert res.health["is_watertight"] is True
    assert res.mis_pore_mm is not None and res.mis_pore_mm > 0.2
    suffixes = {p.suffix.lower() for p in res.output_paths}
    assert ".stl" in suffixes
    import importlib.util

    if importlib.util.find_spec("lxml") is not None:
        assert ".3mf" in suffixes
    assert res.recipe_path is not None and res.recipe_path.is_file()
    assert res.manifest_path is not None and res.manifest_path.is_file()


def test_piecewise_bands_generation() -> None:
    state = ImplicitStudioState(
        primitive_shape="Cube",
        prim_x_mm=6.0,
        prim_y_mm=6.0,
        prim_z_mm=6.0,
        lattice_type="Split-P",
        grading_mode="Piecewise Discrete Bands",
        auto_resolution=False,
        resolution_mm=0.25,
        export_stl=True,
        export_3mf=False,
        write_manifest=False,
        custom_output_name="pytest_implicit_studio_piecewise",
    )
    cad_mesh, _ = build_or_load_cad_mesh(state)
    res = run_implicit_generation(state, cad_mesh=cad_mesh, output_root=TEST_OUTPUT_ROOT)
    assert len(res.mesh.faces) > 100
    assert any(p.suffix.lower() == ".stl" for p in res.output_paths)


def test_offscreen_qt_window_wiring() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    win = ImplicitStudioWindow(enable_vtk=False)
    try:
        # Optional 3D generated mesh inspection MUST default to OFF
        assert win.state.show_generated_3d_mesh is False
        assert win._chk_show_gen_mesh_tb.isChecked() is False
        assert win._chk_show_gen_mesh_dock.isChecked() is False

        # Verify unified grading mode selector stacked widget switching
        for idx, _name in enumerate(("Uniform", "Linear / Radial Ramp", "Multi-Knot Control Table", "Piecewise Discrete Bands")):
            win._combo_grading_mode.setCurrentIndex(idx)
            assert win._grade_stack.currentIndex() == idx

        # Verify Generate Skin checkbox gates shell controls and toggles core <-> combined
        assert win._chk_generate_skin.isChecked() is False
        assert win._skin_options_widget.isHidden() is True
        assert win.state.export_mode == "core"
        win._chk_generate_skin.setChecked(True)
        assert win._skin_options_widget.isHidden() is False
        assert win.state.export_mode == "combined"
        win._chk_generate_skin.setChecked(False)
        assert win._skin_options_widget.isHidden() is True
        assert win.state.export_mode == "core"

        # Verify 4-way spinboxes start disabled until 2 distinct drivers are chosen
        assert win._spin_L.value() == pytest.approx(1.50, abs=1e-4)
        assert win._spin_pore.value() == pytest.approx(0.80, abs=1e-4)
        assert win._spin_sf.value() == pytest.approx(0.33, abs=1e-4)
        assert win._spin_wall.value() == pytest.approx(0.50, abs=1e-4)
        for sp in (win._spin_L, win._spin_pore, win._spin_sf, win._spin_wall):
            assert sp.isEnabled() is False

        win._combo_size_drv.setCurrentIndex(win._combo_size_drv.findData("unit_cell"))
        # Still 1 driver chosen -> all 4 remain disabled
        assert win._spin_L.isEnabled() is False
        # Duplicate selection in Driver 2 clears Driver 1 cleanly
        win._combo_density_drv.setCurrentIndex(win._combo_density_drv.findData("unit_cell"))
        assert win._combo_size_drv.currentData() == ""
        # Pick distinct drivers: unit_cell + wall_thickness
        win._combo_size_drv.setCurrentIndex(win._combo_size_drv.findData("unit_cell"))
        win._combo_density_drv.setCurrentIndex(win._combo_density_drv.findData("wall_thickness"))
        assert win._spin_L.isEnabled() is True
        assert win._spin_wall.isEnabled() is True
        assert win._spin_pore.isEnabled() is False
        assert win._spin_sf.isEnabled() is False
        prev_sf = win._spin_sf.value()
        win._spin_wall.setValue(0.35)
        assert win._spin_sf.value() != pytest.approx(prev_sf, abs=1e-4)

        # Apply Osteochondral preset
        win._combo_preset.setCurrentIndex(1)
        assert win._combo_grading_mode.currentText() == "Multi-Knot Control Table"
        assert win._tbl_knots.rowCount() == 3

        # Switch to 2D Slice Inspector tab and verify bitmap + Phase 2 profile chart rendered
        win._tabs.setCurrentIndex(1)
        assert win._lbl_slice_image.pixmap() is not None
        assert "Plane" in win._lbl_slice_metrics.text()
        assert win._profile_chart._profile is not None
        assert "Mode:" in win._lbl_profile_summary.text()

        # Verify center_origin checkbox and smart 3MF gating
        win._chk_center_origin.setChecked(True)
        assert win.state.center_origin is True
        if not has_threemf_support():
            assert win._chk_exp_3mf.isEnabled() is False
            assert win._chk_exp_3mf.isChecked() is False
    finally:
        win.close()


def test_phase2_center_origin_vascular_fillet_and_woodpile() -> None:
    # 1. Case-insensitive fixture discovery & center_origin translation
    fixtures = discover_cad_fixtures()
    assert len(fixtures) > 0
    stl_target = next((p for p in fixtures if "rounded-cube" in p.name.lower()), fixtures[0])
    s_raw = ImplicitStudioState(geometry_source="File", cad_file_path=str(stl_target), center_origin=False)
    m_raw, _ = build_or_load_cad_mesh(s_raw)
    s_ctr = ImplicitStudioState(geometry_source="File", cad_file_path=str(stl_target), center_origin=True)
    m_ctr, _ = build_or_load_cad_mesh(s_ctr)
    assert np.allclose(m_ctr.centroid, [0.0, 0.0, 0.0], atol=1e-2) or np.linalg.norm(
        0.5 * (m_ctr.bounds[0] + m_ctr.bounds[1])
    ) < 1e-4
    assert m_raw.faces.shape == m_ctr.faces.shape

    # 2. Radial Vascular Scaffold + cylindrical gradient + combined fillet + Woodpile
    s_rad = ImplicitStudioState(prim_x_mm=10.0, prim_y_mm=10.0, prim_z_mm=10.0)
    s_rad.apply_preset("Radial Vascular Scaffold")
    s_rad.export_mode = "combined"
    s_rad.shell_thickness_mm = 0.8
    s_rad.blend_radius_mm = 0.4
    cad, _ = build_or_load_cad_mesh(s_rad)
    prof = evaluate_grading_profile_1d(s_rad, cad, n_samples=60)
    assert len(prof["markers_mm"]) >= 1
    fp = build_surface_footprint_polydata(s_rad, cad)
    assert fp.GetNumberOfPoints() > 100
    sl = compute_2d_slice_image(s_rad, cad, axis="Z", slice_pos_pct=50.0, pixel_size_mm=0.08)
    assert 0.0 < sl.solid_area_fraction < 1.0

    # 3. Woodpile preview + slice
    s_wood = ImplicitStudioState(
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Woodpile",
        pore_size_mm=1.2,
        solid_fraction=0.40,
    )
    cad_w, _ = build_or_load_cad_mesh(s_wood)
    fp_w = build_surface_footprint_polydata(s_wood, cad_w)
    assert fp_w.GetNumberOfPoints() > 100
    sl_w = compute_2d_slice_image(s_wood, cad_w, axis="X", slice_pos_pct=50.0, pixel_size_mm=0.06)
    assert 0.05 < sl_w.solid_area_fraction < 0.95

    # 4. Calibration residual/iteration telemetry
    s_cal = ImplicitStudioState(
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Gyroid",
        size_driver="pore_size",
        density_driver="solid_fraction",
        pore_size_mm=2.5,
        solid_fraction=0.33,
    )
    cal = run_tpms_calibration(s_cal)
    assert cal.iterations >= 1
    assert abs(cal.residual_pore_mm) < 0.25


def test_phase3_microfeature_previews_batch_and_env(tmp_path: Path) -> None:
    from scripts.setup_implicit_studio_env import check_environment

    ok, lines = check_environment()
    assert ok is True
    assert any("PySide6" in ln for ln in lines)

    # 1. Surface texture 2D slice modulation (emboss increases solid area fraction vs untextured)
    s_plain = ImplicitStudioState(
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Gyroid",
        pore_size_mm=2.5,
        solid_fraction=0.33,
    )
    cad, _ = build_or_load_cad_mesh(s_plain)
    sl_plain = compute_2d_slice_image(s_plain, cad, axis="X", slice_pos_pct=50.0, pixel_size_mm=0.04)

    s_tex = ImplicitStudioState(
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Gyroid",
        pore_size_mm=2.5,
        solid_fraction=0.33,
        enable_surface_texture=True,
        texture_type="microgrooves",
        texture_amplitude_mm=0.10,
        texture_wavelength_mm=0.40,
        texture_displacement_mode="emboss",
    )
    sl_tex = compute_2d_slice_image(s_tex, cad, axis="X", slice_pos_pct=50.0, pixel_size_mm=0.04)
    assert not np.array_equal(sl_plain.rgb_image, sl_tex.rgb_image)
    assert sl_tex.solid_area_fraction > sl_plain.solid_area_fraction

    # 2. Micropillar glyph preview over Flying Edges footprint
    s_pil = ImplicitStudioState(
        prim_x_mm=8.0,
        prim_y_mm=8.0,
        prim_z_mm=8.0,
        lattice_type="Gyroid",
        pore_size_mm=2.5,
        enable_micropillars=True,
        pillar_diameter_um=140.0,
        pillar_height_um=350.0,
        pillar_spacing_um=450.0,
    )
    fp = build_surface_footprint_polydata(s_pil, cad)
    pil_poly = build_micropillar_preview_polydata(s_pil, cad, footprint_poly=fp, max_pillars=80)
    assert pil_poly.GetNumberOfCells() > 0

    # 3. Headless batch recipe queue
    r1 = tmp_path / "recipe_gyroid.json"
    r2 = tmp_path / "recipe_splitp.json"
    ImplicitStudioState(
        prim_x_mm=6.0,
        prim_y_mm=6.0,
        prim_z_mm=6.0,
        lattice_type="Gyroid",
        auto_resolution=False,
        resolution_mm=0.25,
        export_stl=True,
        export_3mf=False,
        write_manifest=False,
        custom_output_name="batch_gyroid",
    ).save_recipe(r1)
    ImplicitStudioState(
        prim_x_mm=6.0,
        prim_y_mm=6.0,
        prim_z_mm=6.0,
        lattice_type="Split-P",
        auto_resolution=False,
        resolution_mm=0.25,
        export_stl=True,
        export_3mf=False,
        write_manifest=False,
        custom_output_name="batch_splitp",
    ).save_recipe(r2)
    batch_res = run_recipe_batch([r1, r2], output_root=TEST_OUTPUT_ROOT / "batch")
    assert len(batch_res) == 2
    assert all(res.health["is_watertight"] for _p, res in batch_res)


def test_hq_preview_modes_and_cutaway_cap() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None

    s = ImplicitStudioState(
        prim_x_mm=12.0,
        prim_y_mm=12.0,
        prim_z_mm=12.0,
        lattice_type="Gyroid",
        size_driver="pore_size",
        density_driver="solid_fraction",
        pore_size_mm=2.5,
        solid_fraction=0.35,
    )
    assert s.preview_mode == PREVIEW_MODES[0] == "2D Surface (Ultra-Sharp)"

    cad, _ = build_or_load_cad_mesh(s)
    fp_2d = build_surface_footprint_polydata(s, cad, mode="2D Surface (Ultra-Sharp)")
    assert fp_2d.GetNumberOfCells() > 10_000

    fe_pitch = _compute_preview_spacing(s, cad)
    assert fe_pitch >= 0.12
    fp_3d = build_surface_footprint_polydata(s, cad, mode="3D Flying Edges (Coarse)")
    assert fp_3d.GetNumberOfCells() > 5_000
    fe_tm = vtk_polydata_to_trimesh(fp_3d)
    assert fe_tm.is_watertight and float(fe_tm.volume) > 0.0

    scene = ImplicitVtkScene(vtk.vtkRenderer())
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(vtk.vtkRenderWindow())
    scene.attach_interactor(iren)
    assert isinstance(iren.GetInteractorStyle(), vtk.vtkInteractorStyleTrackballCamera)

    assert scene._cad_opacity == pytest.approx(1.0)
    scene.set_cad_mesh(cad, reset_cam=False)
    scene.set_surface_footprint(fp_2d)

    # Interactive drag cutaway uses GPU clip planes without CPU solid-cap rebuild
    scene.set_cutaway(enabled=True, axis="Z", pct=45.0, interactive=True)
    assert scene._cad_mapper.GetNumberOfClippingPlanes() == 1
    assert scene._cad_cap_poly is None

    # Releasing slider builds closed vector solid cap & removes GPU plane
    scene.set_cutaway(enabled=True, axis="Z", pct=50.0, interactive=False)
    scene.set_cutaway_cap(s, cad)
    assert scene._cad_mapper.GetNumberOfClippingPlanes() == 0
    assert scene._cad_cap_poly is not None and scene._cad_cap_poly.GetNumberOfCells() > 0
    assert bool(scene._cut_cap_actor.GetVisibility()) is True

    # Cached-SDF 2D slice should run rapidly on warm cache (< 0.1 s per slice)
    _ = compute_2d_slice_image(s, cad, axis="Z", slice_pos_pct=35.0, pixel_size_mm=0.05)
    t0 = time.perf_counter()
    _ = compute_2d_slice_image(s, cad, axis="Z", slice_pos_pct=65.0, pixel_size_mm=0.05)
    assert (time.perf_counter() - t0) < 0.25

    s.preview_mode = "3D Flying Edges (Coarse)"
    scene.set_surface_footprint(fp_3d)
    scene.set_cutaway_cap(s, cad)
    assert bool(scene._footprint_is_closed_3d) is True
    assert bool(scene._footprint_mapper.GetScalarVisibility()) is True
    assert scene._footprint_mapper.GetNumberOfClippingPlanes() == 0
    assert bool(scene._cut_cap_actor.GetVisibility()) is False

    win = ImplicitStudioWindow(enable_vtk=False)
    try:
        assert win._combo_preview_mode.currentText() == "2D Surface (Ultra-Sharp)"
        assert win._opacity_combo.currentText() == "100% Opaque Gray"
    finally:
        win.close()

    # Conformal selective-edge subdivision must refine spanning flat sidewalls of CAD STLs
    stl = next((p for p in discover_cad_fixtures() if "rounded-cube" in p.name.lower()), None)
    assert stl is not None
    s_rc = ImplicitStudioState(geometry_source="File", cad_file_path=str(stl), center_origin=True)
    m_rc, _ = build_or_load_cad_mesh(s_rc)
    v_sub, f_sub = _get_cached_subdivided_surface(m_rc)
    import trimesh

    max_el = float(trimesh.Trimesh(v_sub, f_sub, process=False).edges_unique_length.max())
    assert max_el <= 0.30


def test_section_help_popups_complete() -> None:
    from graphite.ui.implicit_help import SECTION_HELP_DOCS

    app = QApplication.instance() or QApplication([])
    assert app is not None

    win = ImplicitStudioWindow(enable_vtk=False)
    try:
        expected_sections = {
            "cad_geometry",
            "architecture_shelling",
            "four_way_sizing",
            "functional_grading",
            "micro_features",
            "telemetry",
            "calibration_status",
            "production_synthesis",
            "export_ledger",
        }
        assert set(win._help_boxes.keys()) == expected_sections
        assert set(SECTION_HELP_DOCS.keys()) == expected_sections | {"quick_start"}
        assert win._btn_how_to_use.text() == "How to Use (?)"

        for key in expected_sections | {"quick_start"}:
            if key in win._help_boxes:
                box = win._help_boxes[key]
                assert box.help_btn.text() == "?"
                assert box.help_btn.isVisibleTo(box)
            dlg = win._show_help_popup(key, exec_dialog=False)
            assert dlg.windowTitle() == SECTION_HELP_DOCS[key][0]
            assert len(dlg.browser.toPlainText().strip()) > 40
            dlg.close()
    finally:
        win.close()
