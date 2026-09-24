"""
Graphite Lattice Engine — Interactive Trame + PyVista Application.

Phase 2: Full Production Implicit Integration.
- Custom STL upload and workspace fixtures (test_parts/*.STL) alongside primitives.
- Full 7-TPMS catalog: Gyroid, Diamond, Schwarz-P, Schwarz-Diamond, Neovius, Lidinoid, Split-P.
- Outer solid shelling: Core (lattice only), Skin (solid shell only), Combined (lattice + shell).
- Dual-format export: Native 3MF (.3mf) and STL (.stl) with automated manifold triage.
- Direct in-browser downloading without page reload or websocket disconnect.
- Auto Nyquist voxel resolution: min(250 um, wall / 2) with manual override.
- Blender dark studio theme (#2E3035) with Oceanic Slate Teal accents (#0FA4AF, #024950).
"""
from __future__ import annotations

import base64
import os
import urllib.parse
from datetime import date
from pathlib import Path

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3
from trame_pyvista.ui import plotter_ui

from graphite.ui.cli import SUPPORTED_TPMS_EQUATIONS, run_headless_recipe
from graphite.ui.surface_preview import (
    create_floor_grid,
    generate_surface_tpms_preview,
    generate_explicit_preview,
    generate_interlinked_preview,
    generate_auxetic_preview,
    load_cad_mesh,
)
from graphite.explicit.interlinked.support_recipes import (
    get_available_support_recipes,
    export_seed_cell_stl,
)


def parse_interlinked_cell_key(name: str) -> str:
    """Parse user-friendly cell display label into canonical InterlinkedRegistry key."""
    n = str(name).lower()
    if "c-6-tt" in n or "c6tt" in n:
        return "c6tt"
    if "d-4-tet" in n or "d4tet" in n:
        return "d4tet"
    if "j-4-oct" in n or "j4oct" in n:
        return "j4oct"
    if "euro" in n:
        return "euro_4in1"
    if "kusari" in n:
        return "kusari"
    if "nasa" in n:
        return "nasa_space_fabric"
    return "c6tt"


def parse_auxetic_pattern_key(name: str) -> str:
    """Parse user-friendly auxetic label into canonical generator pattern key."""
    n = str(name).lower()
    if "anti-tetra" in n or "anti_tetra" in n:
        return "anti_tetra_chiral"
    if "anti-tri" in n or "anti_tri" in n:
        return "anti_tri_chiral"
    if "tetra" in n:
        return "tetra_chiral"
    if "tri" in n:
        return "tri_chiral"
    if "re-entrant" in n or "reentrant" in n or "bowtie" in n:
        return "reentrant"
    if "rotating" in n or "square" in n:
        return "rotating_squares"
    if "pentamode" in n:
        return "pentamode"
    return "tetra_chiral"


def parse_auxetic_surface_key(name: str) -> str:
    """Parse user-friendly domain label into canonical surface key ('plate' or 'cylinder')."""
    n = str(name).lower()
    if "cylinder" in n or "tube" in n or "sleeve" in n:
        return "cylinder"
    return "plate"


def create_app(server=None):
    if server is None:
        server = get_server(None, client_type="vue3")

    # Serve generated files directory via HTTP so client downloads stream directly without websocket disconnect
    server.serve["outputs"] = str(Path("outputs").resolve())

    state, ctrl = server.state, server.controller

    # Discovery of workspace fixtures in test_parts/
    test_parts_dir = Path("test_parts")
    if test_parts_dir.is_dir():
        fixtures = sorted([p.name for p in test_parts_dir.glob("*.[sS][tT][lL]")])
    else:
        fixtures = []

    # Options lists in reactive state for proper Vuetify3 binding
    state.modality_options = ["Implicit TPMS", "Explicit Struts", "Interlinked / PAMs", "Auxetics"]
    state.source_options = ["Primitive", "Workspace Fixture", "Custom STL File"]
    state.shape_options = ["Cube", "Cylinder", "Sphere", "Toros", "Tube"]
    state.fixture_options = fixtures
    state.lattice_options = list(SUPPORTED_TPMS_EQUATIONS)
    state.shelling_options = ["Core", "Skin", "Combined"]
    state.grading_axes = ["Z", "X", "Y", "Radial"]
    state.grading_mode_options = ["Cell Size (Chirp)", "Solid Fraction (SF)"]
    state.texture_type_options = ["Microgrooves", "Bumps / Nodules", "Diamond Knurl", "Spinodal"]
    state.texture_direction_options = ["Z-Axial (0,0,1)", "X-Axial (1,0,0)", "Y-Axial (0,1,0)"]
    state.auto_calc_options = ["Wall Thickness", "Solid Fraction", "Unit Cell Size", "Pore Size (MIS)"]

    # Explicit strut options
    state.explicit_type_options = ["A15 (Conformal Kagome)", "SC Hex Modular"]
    state.sc_rule_options = ["octahedral", "cubic", "kelvin"]
    state.explicit_mode_options = ["conformal", "boolean"]

    # Interlinked metamaterial options
    state.interlinked_cell_options = [
        "C-6-TT (Science 2025 Truncated Tetrahedron)",
        "D-4-TET (Diamond Bipartite Dual)",
        "J-4-OCT (Square Planar Octahedron Cross)",
        "European 4-in-1 (Classic Checkerboard Weave)",
        "Japanese Kusari (Orthogonal Flat + Arch Links)",
        "NASA Space Fabric (JPL Hexagonal Weave)",
    ]
    state.interlinked_seeding_options = [
        "Cartesian Grid (Strict Inset Culling)",
        "Cylindrical Tube Wrap",
    ]
    state.interlinked_support_recipe_options = get_available_support_recipes("c6tt")

    # Auxetics metamaterial options
    state.auxetics_surface_options = ["Flat Sheet / Plate", "Cylindrical Sleeve (Tube)"]
    state.auxetics_pattern_options = [
        "Tetra-Chiral Honeycomb (Square Basis, ν ≈ -1)",
        "Tri-Chiral Honeycomb (Hexagonal Basis, ν ≈ -1)",
        "Anti-Tetra-Chiral Honeycomb (Square Basis)",
        "Anti-Tri-Chiral Honeycomb (Hexagonal Basis)",
        "Re-Entrant Auxetic Honeycomb (Bowtie, Chen 2020)",
        "Rotating Rigid Squares (Living Hinges, ν = -1)",
        "Diamond Cubic Pentamode (Extreme Elastic)",
        "Hexagonal Pentamode",
    ]

    # Modality selection & expansion panel management
    state.modality = "Implicit TPMS"
    state.expanded_panels_implicit = [0, 1, 2]
    state.expanded_panels_explicit = [0, 1, 2]
    state.expanded_panels_interlinked = [0, 1, 2]
    state.expanded_panels_auxetics = [0, 1, 2]

    # CAD Health Status (Always-on automatic repair triage)
    state.cad_health_status = "CAD: Watertight (2-Manifold)"
    state.cad_health_color = "success"

    # Multi-Axis Dynamic Cutaway Inspection
    state.show_cutaway = False
    state.cutaway_enabled = False
    state.cutaway_axis = "X"
    state.cutaway_axis_options = ["X", "Y", "Z"]
    state.cutaway_pos = 50.0
    state.cutaway_invert = False

    # Geometry & Source state
    state.geom_source = "Primitive"
    state.prim_shape = "Cube"
    state.prim_size = 20.0
    state.selected_fixture = "BaseRing_1to1.STL" if "BaseRing_1to1.STL" in fixtures else (fixtures[0] if fixtures else "")
    state.custom_file_path = ""
    state.uploaded_file = None

    # Implicit TPMS & Shelling state
    state.lattice_type = "Gyroid"
    state.export_mode = "Core"
    state.shell_thickness = 2.0
    state.woodpile_invert = False

    # Explicit Strut state
    state.explicit_type = "A15 (Conformal Kagome)"
    state.sc_rule = "octahedral"
    state.explicit_mode = "conformal"
    state.strut_radius = 0.40
    state.explicit_solid_fraction = 0.15

    # Interlinked Metamaterials / PAMs state
    state.interlinked_cell = "C-6-TT (Science 2025 Truncated Tetrahedron)"
    state.interlinked_seeding = "Cartesian Grid (Strict Inset Culling)"
    state.interlinked_support_recipe = "None (Unprinted / Free)"
    state.interlinked_pitch = 8.0
    state.interlinked_wire_radius = 0.40
    state.interlinked_min_clearance = 0.35
    state.interlinked_auto_resolve_pitch = False
    state.interlinked_cull_margin = 0.50
    state.interlinked_cylinder_radius = 15.0
    state.interlinked_cylinder_height = 25.0
    state.interlinked_add_perimeter_frame = False
    state.interlinked_frame_wall_thickness = 2.0
    state.interlinked_frame_margin = 0.50
    state.interlinked_frame_shape = "box"
    state.interlinked_calculated_clearance = 0.35
    state.interlinked_clearance_status = "Clearance Valid (Δ ≥ 0.35 mm)"
    state.interlinked_clearance_color = "success"
    state.interlinked_show_cad_boundary = False
    state.interlinked_cad_opacity = 0.15
    state.expanded_panels_interlinked = [0, 1, 2]

    # Auxetics Metamaterials state
    state.auxetics_surface = "Flat Sheet / Plate"
    state.auxetics_pattern = "Tetra-Chiral Honeycomb (Square Basis, ν ≈ -1)"
    state.auxetics_width = 50.0
    state.auxetics_height = 50.0
    state.auxetics_thickness = 2.0
    state.auxetics_r_in = 15.0
    state.auxetics_r_out = 17.5
    state.auxetics_n_circumferential = 6
    state.auxetics_r_node = 2.0
    state.auxetics_strut_w = 1.0
    state.auxetics_square_side = 10.0
    state.auxetics_rotation_angle = 30.0
    state.auxetics_hinge_radius = 0.45
    state.auxetics_pentamode_cell_size = 10.0
    state.auxetics_pentamode_r_min = 0.35
    state.auxetics_pentamode_r_max = 0.90

    # 4-way sizing calculator state
    state.auto_calculate = "Wall Thickness"
    state.disable_unit_cell = False
    state.disable_solid_fraction = False
    state.disable_wall_thickness = True
    state.disable_pore_size = True

    state.unit_cell_size = 5.0
    state.solid_fraction = 0.30
    state.wall_thickness_mm = 0.415
    state.pore_size_mm = 3.275

    # Dual Export Formats & Downloads
    state.export_stl = True
    state.export_3mf = True
    state.download_stl_url = ""
    state.download_stl_name = ""
    state.download_3mf_url = ""
    state.download_3mf_name = ""

    # Grading state
    state.enable_grading = False
    state.grading_axis = "Z"
    state.grading_mode = "Cell Size (Chirp)"
    state.min_solid_fraction = 0.15
    state.max_solid_fraction = 0.45
    state.doubling_interval = 5.0

    # Experimental Conformal Harmonic UVW
    state.conformal_deformation = False
    state.conformal_iterations = 150

    # Surface Micro-Texture & Feature state
    state.enable_surface_texture = False
    state.texture_type = "Microgrooves"
    state.texture_amplitude_um = 25.0
    state.texture_wavelength_um = 50.0
    state.texture_direction = "Z-Axial (0,0,1)"
    state.texture_profile = "Sine"
    state.texture_profile_options = ["Sine", "Triangle", "Square"]
    state.texture_mode = "Centered (±A)"
    state.texture_mode_options = ["Centered (±A)", "Emboss (+A)", "Engrave (-A)"]
    state.texture_triplanar = False

    # Explicit Micropillars / Microfiber Forest state
    state.enable_micropillars = False
    state.micropillar_diameter_um = 50.0
    state.micropillar_height_um = 200.0
    state.micropillar_spacing_um = 250.0
    state.micropillar_location = "All Surfaces"
    state.micropillar_location_options = ["All Surfaces", "Outer Shell Only", "Internal Pores Only"]
    state.micropillar_filter_printable = False

    # FAQ / Architecture Guide Dialog
    state.faq_dialog_open = False

    # Execution & resolution
    state.auto_resolution = True
    state.resolution = 0.208
    state.status_message = "Ready. Select geometry and click 'Update 3D Preview' to refresh."
    state.status_color = "info"
    state.is_generating = False

    def recompute_density():
        """Ensure density variables are synchronized and update Nyquist resolution."""
        # Explicit strut analytical density calculation
        try:
            L_exp = max(float(state.unit_cell_size), 0.1)
            r_exp = max(float(state.strut_radius), 0.01)
            c = 73.36174 if "A15" in str(state.explicit_type) else 12.0
            val = np.pi * c * (r_exp / L_exp) ** 2
            state.explicit_solid_fraction = round(float(np.clip(val, 0.001, 1.0)), 3)
        except Exception:
            pass

        # Implicit TPMS 4-parameter calculator
        mode = str(state.auto_calculate)
        try:
            L = max(float(state.unit_cell_size), 0.1)
            phi = float(np.clip(float(state.solid_fraction), 0.01, 0.95))
            w = max(float(state.wall_thickness_mm), 0.001)
            w_active = w

            if mode == "Wall Thickness":
                state.disable_wall_thickness = True
                state.disable_solid_fraction = False
                state.disable_unit_cell = False
                state.disable_pore_size = False
                tau = phi / 1.15
                calc_w = tau * L / np.pi
                state.wall_thickness_mm = round(calc_w, 3)
                calc_pore = L * (1.0 - 1.15 * phi)
                state.pore_size_mm = round(max(calc_pore, 0.05), 3)
                w_active = calc_w
            elif mode == "Solid Fraction":
                state.disable_solid_fraction = True
                state.disable_wall_thickness = False
                state.disable_unit_cell = False
                state.disable_pore_size = False
                tau = np.pi * w / L
                calc_phi = float(np.clip(tau * 1.15, 0.05, 0.90))
                state.solid_fraction = round(calc_phi, 3)
                calc_pore = L * (1.0 - 1.15 * calc_phi)
                state.pore_size_mm = round(max(calc_pore, 0.05), 3)
                w_active = w
            elif mode == "Unit Cell Size":
                state.disable_unit_cell = True
                state.disable_solid_fraction = False
                state.disable_wall_thickness = False
                state.disable_pore_size = False
                tau = phi / 1.15
                calc_L = np.pi * w / max(tau, 1e-4)
                state.unit_cell_size = round(max(calc_L, 0.5), 2)
                calc_pore = calc_L * (1.0 - 1.15 * phi)
                state.pore_size_mm = round(max(calc_pore, 0.05), 3)
                w_active = w
            elif mode == "Pore Size (MIS)":
                state.disable_pore_size = True
                state.disable_solid_fraction = False
                state.disable_wall_thickness = False
                state.disable_unit_cell = False
                calc_pore = L * (1.0 - 1.15 * phi)
                state.pore_size_mm = round(max(calc_pore, 0.05), 3)
                tau = phi / 1.15
                calc_w = tau * L / np.pi
                state.wall_thickness_mm = round(calc_w, 3)
                w_active = calc_w

            # Set resolution to 250um or calculated wall_thickness / 2, whichever is lower
            if bool(state.auto_resolution):
                recommended_res = round(min(0.250, max(w_active / 2.0, 0.03)), 3)
                state.resolution = recommended_res
        except Exception:
            pass

        # Interlinked Kinematic Clearance Calculator
        try:
            from graphite.explicit.interlinked import InterlinkedRegistry
            c_key = parse_interlinked_cell_key(state.interlinked_cell)
            cell_cls = InterlinkedRegistry.get(c_key)
            cell_inst = cell_cls() if callable(cell_cls) else cell_cls

            wire_r = max(float(state.interlinked_wire_radius), 0.05)
            strut_d = 2.0 * wire_r
            req_clr = max(float(state.interlinked_min_clearance), 0.01)

            if bool(state.interlinked_auto_resolve_pitch):
                resolved_p = cell_inst.resolve_pitch(target_clearance=req_clr, strut_diameter=strut_d)
                state.interlinked_pitch = round(float(resolved_p), 2)
                pred_clr = req_clr
            else:
                p = max(float(state.interlinked_pitch), 0.5)
                pred_clr = cell_inst.forward_clearance(p, strut_d)

            state.interlinked_calculated_clearance = round(float(pred_clr), 3)
            if pred_clr >= req_clr:
                state.interlinked_clearance_status = f"Clearance Valid: {pred_clr:.3f} mm ≥ {req_clr:.2f} mm"
                state.interlinked_clearance_color = "success"
            elif pred_clr > 0.0:
                state.interlinked_clearance_status = f"Clearance Warning: {pred_clr:.3f} mm < {req_clr:.2f} mm requested"
                state.interlinked_clearance_color = "warning"
            else:
                state.interlinked_clearance_status = f"Collision Frustration: {pred_clr:.3f} mm (parts will weld!)"
                state.interlinked_clearance_color = "error"
        except Exception:
            pass

    @state.change("auto_calculate", "solid_fraction", "wall_thickness_mm", "unit_cell_size", "pore_size_mm", "auto_resolution", "modality", "explicit_type", "strut_radius")
    def on_density_change(**kwargs):
        recompute_density()

    @state.change("interlinked_cell", "interlinked_wire_radius", "interlinked_min_clearance", "interlinked_pitch", "interlinked_auto_resolve_pitch")
    def on_interlinked_kinematic_change(**kwargs):
        recompute_density()
        if str(state.modality) == "Interlinked / PAMs":
            update_3d_preview()

    @state.change("interlinked_seeding", "interlinked_cull_margin", "interlinked_cylinder_radius", "interlinked_cylinder_height", "interlinked_show_cad_boundary", "interlinked_cad_opacity", "interlinked_add_perimeter_frame")
    def on_interlinked_display_change(**kwargs):
        if str(state.modality) == "Interlinked / PAMs":
            update_3d_preview()

    @state.change("cutaway_enabled", "cutaway_axis", "cutaway_pos", "cutaway_invert")
    def on_cutaway_change(**kwargs):
        update_3d_preview()

    @state.change("geom_source", "prim_shape", "prim_size", "selected_fixture", "custom_file_path")
    def on_geometry_change(**kwargs):
        update_cad_health()

    @state.change(
        "auxetics_surface", "auxetics_pattern", "auxetics_width", "auxetics_height",
        "auxetics_thickness", "auxetics_r_in", "auxetics_r_out", "auxetics_n_circumferential",
        "auxetics_r_node", "auxetics_strut_w", "auxetics_square_side", "auxetics_rotation_angle",
        "auxetics_hinge_radius", "auxetics_pentamode_cell_size", "auxetics_pentamode_r_min", "auxetics_pentamode_r_max"
    )
    def on_auxetics_param_change(**kwargs):
        if str(state.modality) == "Auxetics":
            update_3d_preview()

    @state.change("uploaded_file")
    def on_file_upload(uploaded_file, **kwargs):
        """Save browser-uploaded STL into outputs/uploads/ and set custom_file_path."""
        if not uploaded_file:
            return
        files = uploaded_file if isinstance(uploaded_file, list) else [uploaded_file]
        if not files:
            return
        f = files[0]
        if isinstance(f, dict):
            fname = f.get("name", "upload.stl")
            content = f.get("content", None)
            if content:
                upload_dir = Path("outputs") / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                dest = upload_dir / fname
                with open(dest, "wb") as fp:
                    fp.write(content)
                state.custom_file_path = str(dest.resolve())
                try:
                    triaged_mesh = load_cad_mesh("file", file_path=dest)
                    status_lbl = "CAD: Watertight (2-Manifold)" if triaged_mesh.is_watertight else "CAD: Auto-Repaired (Watertight)"
                    state.cad_health_status = status_lbl
                    state.cad_health_color = "success" if triaged_mesh.is_watertight else "info"
                    state.status_message = (
                        f"Uploaded '{fname}' ({len(content):,} bytes). "
                        f"CAD health triage: {status_lbl} ({len(triaged_mesh.faces):,} faces)."
                    )
                except Exception as ex:
                    state.status_message = f"Uploaded '{fname}' ({len(content):,} bytes). Auto-triage: {ex}"
                state.status_color = "success"

    def resolve_cad_input() -> tuple[str | None, float | None, Path | None, str]:
        """Resolve current geometry selection into (primitive_shape, size, stl_path, display_name)."""
        src = str(state.geom_source)
        if src == "Workspace Fixture":
            fixture_name = str(state.selected_fixture)
            p = Path("test_parts") / fixture_name
            return None, None, p, fixture_name
        elif src == "Custom STL File":
            p = Path(str(state.custom_file_path).strip())
            return None, None, p, p.name if p.name else "custom.stl"
        else:
            return str(state.prim_shape), float(state.prim_size), None, f"{state.prim_shape} ({state.prim_size}mm)"

    def update_cad_health():
        """Evaluate CAD boundary manifold status and update chip."""
        try:
            shape, size, cad_path, _ = resolve_cad_input()
            if cad_path is not None:
                mesh = load_cad_mesh("file", file_path=cad_path)
            else:
                mesh = load_cad_mesh("primitive", primitive_shape=shape or "Cube", size=size or 20.0)
            if mesh.is_watertight:
                state.cad_health_status = "CAD: Watertight (2-Manifold)"
                state.cad_health_color = "success"
            else:
                state.cad_health_status = "CAD: Auto-Repaired (Watertight)"
                state.cad_health_color = "info"
        except Exception:
            state.cad_health_status = "CAD: Auto-Triage Active"
            state.cad_health_color = "info"

    def export_single_seed_cell():
        """Export single central PAM seed cell for slicer support generation."""
        try:
            c_key = parse_interlinked_cell_key(state.interlinked_cell)
            pitch = max(float(state.interlinked_pitch), 0.5)
            wire_r = max(float(state.interlinked_wire_radius), 0.05)
            seed_dir = Path("outputs") / "seed_cells"
            seed_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"SeedCell_{c_key.upper()}_pitch{pitch:.1f}mm_r{wire_r:.2f}mm.stl"
            out_file = seed_dir / out_name
            export_seed_cell_stl(c_key, pitch, wire_r, out_file)
            state.download_stl_name = out_name
            state.download_stl_url = f"/outputs/seed_cells/{urllib.parse.quote(out_name)}"
            state.status_message = f"Exported single seed cell STL: '{out_name}'. Ready for download or slicer support."
            state.status_color = "success"
        except Exception as ex:
            state.status_message = f"Failed to export seed cell STL: {ex}"
            state.status_color = "error"

    ctrl.export_seed_cell = export_single_seed_cell

    # Initial density resolution & CAD health check
    recompute_density()
    update_cad_health()

    # Setup PyVista plotter with Blender studio dark gray background
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("#2E3035")

    def apply_viewport_clipping(mesh: pv.PolyData | None) -> pv.PolyData | None:
        """Dynamic multi-axis section cutaway clipping for 3D inspection."""
        cutaway_active = bool(state.cutaway_enabled or state.show_cutaway)
        if mesh is None or mesh.n_points == 0 or not cutaway_active:
            return mesh
        axis_str = str(state.cutaway_axis).strip().upper()
        axis_idx = 0 if axis_str == "X" else (1 if axis_str == "Y" else 2)
        b = mesh.bounds
        p_val = b[2 * axis_idx] + (b[2 * axis_idx + 1] - b[2 * axis_idx]) * (float(state.cutaway_pos) / 100.0)
        normal = (1.0, 0.0, 0.0) if axis_str == "X" else ((0.0, 1.0, 0.0) if axis_str == "Y" else (0.0, 0.0, 1.0))
        origin = (p_val, 0.0, 0.0) if axis_str == "X" else ((0.0, p_val, 0.0) if axis_str == "Y" else (0.0, 0.0, p_val))
        try:
            return mesh.clip(normal=normal, origin=origin, invert=bool(state.cutaway_invert))
        except Exception:
            return mesh

    def update_3d_preview():
        """Recompute Carbon-style surface preview and update 3D scene."""
        try:
            recompute_density()

            shape, size, cad_path, display_name = resolve_cad_input()
            cell_size = max(float(state.unit_cell_size), 0.1)

            if str(state.modality) == "Auxetics":
                pat_key = parse_auxetic_pattern_key(state.auxetics_pattern)
                surf_key = parse_auxetic_surface_key(state.auxetics_surface)
                w = max(float(state.auxetics_width), 1.0)
                h = max(float(state.auxetics_height), 1.0)
                t = max(float(state.auxetics_thickness), 0.1)
                r_in = max(float(state.auxetics_r_in), 0.1)
                r_out = max(float(state.auxetics_r_out), r_in + 0.1)
                n_circ = max(int(state.auxetics_n_circumferential), 3)
                r_node = max(float(state.auxetics_r_node), 0.1)
                strut_w = max(float(state.auxetics_strut_w), 0.1)
                sq_side = max(float(state.auxetics_square_side), 1.0)
                rot_deg = float(state.auxetics_rotation_angle)
                hinge_r = max(float(state.auxetics_hinge_radius), 0.05)

                cutaway_active = bool(state.cutaway_enabled or state.show_cutaway)

                pv_auxetic = generate_auxetic_preview(
                    surface=surf_key,
                    pattern=pat_key,
                    width=w,
                    height=h,
                    thickness=t,
                    r_in=r_in,
                    r_out=r_out,
                    n_circumferential=n_circ,
                    r_node=r_node,
                    strut_w=strut_w,
                    square_side=sq_side,
                    rotation_angle_deg=rot_deg,
                    hinge_radius=hinge_r,
                    unit_cell_size=float(state.auxetics_pentamode_cell_size),
                    r_min=float(state.auxetics_pentamode_r_min),
                    r_max=float(state.auxetics_pentamode_r_max),
                    cutaway=cutaway_active,
                    cutaway_axis=str(state.cutaway_axis),
                    cutaway_pos=float(state.cutaway_pos),
                    cutaway_invert=bool(state.cutaway_invert),
                )

                plotter.clear_actors()

                floor = create_floor_grid(pv_auxetic.bounds)
                plotter.add_mesh(
                    floor,
                    name="floor_plate",
                    color="#232528",
                    edge_color="#40444C",
                    show_edges=True,
                    line_width=1.2,
                )

                plotter.add_mesh(
                    pv_auxetic,
                    name="auxetic_mesh",
                    color="#0FA4AF",
                    smooth_shading=True,
                    show_edges=True,
                    edge_color="#003135",
                    line_width=0.8,
                )

                plotter.reset_camera()
                if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
                    try:
                        ctrl.view_update()
                    except Exception:
                        pass

                state.status_message = (
                    f"Auxetics Preview: {state.auxetics_pattern}\n"
                    f"Domain: {surf_key.capitalize()} ({w:.1f}×{h:.1f}mm, t={t:.1f}mm)" +
                    (f" | Living Hinge θ={rot_deg:.1f}°" if pat_key == "rotating_squares" else f" | r_node={r_node:.1f}mm, w={strut_w:.1f}mm")
                )
                state.status_color = "success"
                return

            if str(state.modality) == "Interlinked / PAMs":
                c_key = parse_interlinked_cell_key(state.interlinked_cell)
                seeding_mode = "cylindrical" if "cylindrical" in str(state.interlinked_seeding).lower() else "cartesian"
                wire_r = max(float(state.interlinked_wire_radius), 0.05)
                pitch = max(float(state.interlinked_pitch), 0.5)
                min_clr = max(float(state.interlinked_min_clearance), 0.01)
                cull_m = max(float(state.interlinked_cull_margin), 0.0)
                cutaway = bool(state.show_cutaway or state.cutaway_enabled)
                show_cad = bool(state.interlinked_show_cad_boundary)
                cad_op = float(state.interlinked_cad_opacity)

                pv_cad, pv_seed, pv_connecting = generate_interlinked_preview(
                    shape=shape or "Cube",
                    size=size or 20.0,
                    cad_file=cad_path,
                    cell_type=c_key,
                    seeding_type=seeding_mode,
                    pitch=pitch,
                    wire_radius=wire_r,
                    min_clearance=min_clr,
                    cull_margin=cull_m,
                    cylinder_radius=float(state.interlinked_cylinder_radius),
                    cylinder_height=float(state.interlinked_cylinder_height),
                    cutaway=cutaway,
                    num_segments=16,
                )

                pv_cad = apply_viewport_clipping(pv_cad)
                pv_seed = apply_viewport_clipping(pv_seed)
                pv_connecting = apply_viewport_clipping(pv_connecting)

                plotter.clear_actors()

                floor = create_floor_grid(pv_cad.bounds)
                plotter.add_mesh(
                    floor,
                    name="floor_plate",
                    color="#232528",
                    edge_color="#40444C",
                    show_edges=True,
                    line_width=1.2,
                )

                if show_cad:
                    plotter.add_mesh(
                        pv_cad,
                        name="cad_body",
                        color="#D8DCE3",
                        opacity=cad_op,
                        show_edges=True,
                        edge_color="#78909C",
                        style="surface",
                    )

                has_mesh = False
                if pv_connecting is not None and pv_connecting.n_points > 0:
                    plotter.add_mesh(
                        pv_connecting,
                        name="interlinked_connecting",
                        color="#B0BEC5",
                        opacity=0.10,
                        smooth_shading=True,
                        show_edges=False,
                    )
                    has_mesh = True

                if pv_seed is not None and pv_seed.n_points > 0:
                    plotter.add_mesh(
                        pv_seed,
                        name="interlinked_seed",
                        color="#0FA4AF",
                        opacity=1.0,
                        smooth_shading=True,
                        show_edges=False,
                    )
                    has_mesh = True

                if has_mesh:
                    state.status_message = (
                        f"Interlinked Preview: {display_name}\n"
                        f"Cell: {c_key.upper()} | Pitch={pitch}mm, r={wire_r}mm | Clearance: {state.interlinked_calculated_clearance}mm\n"
                        f"Policy A Inset Culling: Central seed cell solid (100%), connecting cells 90% transparent (gray)"
                    )
                    state.status_color = "success"
                else:
                    state.status_message = (
                        f"Interlinked Preview: {display_name}\n"
                        f"Warning: All particles culled by boundary. Increase part size or decrease cell pitch/cull margin."
                    )
                    state.status_color = "warning"

                plotter.reset_camera()
                if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
                    try:
                        ctrl.view_update()
                    except Exception:
                        pass
                return

            if str(state.modality) == "Explicit Struts":
                exp_type = "A15" if "A15" in str(state.explicit_type) else "SC"
                sc_rule = str(state.sc_rule)
                strut_r = max(float(state.strut_radius), 0.05)
                exp_mode = str(state.explicit_mode)
                cutaway = bool(state.show_cutaway or state.cutaway_enabled)

                pv_cad, pv_struts, pv_boundary = generate_explicit_preview(
                    shape=shape or "Cube",
                    size=size or 20.0,
                    cad_file=cad_path,
                    lattice_type=exp_type,
                    rule_name=sc_rule,
                    cell_size=cell_size,
                    strut_radius=strut_r,
                    mode=exp_mode,
                    cutaway=cutaway,
                    surface_only=True,
                )

                pv_cad = apply_viewport_clipping(pv_cad)
                pv_struts = apply_viewport_clipping(pv_struts)
                pv_boundary = apply_viewport_clipping(pv_boundary)

                plotter.clear_actors()

                # Add floor grid plate beneath CAD part (Blender dark studio style)
                floor = create_floor_grid(pv_cad.bounds)
                plotter.add_mesh(
                    floor,
                    name="floor_plate",
                    color="#232528",
                    edge_color="#40444C",
                    show_edges=True,
                    line_width=1.2,
                )

                # Add opaque solid CAD body (Blender studio titanium gray)
                plotter.add_mesh(
                    pv_cad,
                    name="cad_body",
                    color="#D8DCE3",
                    opacity=1.0,
                    show_edges=False,
                )

                # Add vibrant cyan strut line network (#0FA4AF)
                if pv_struts is not None and pv_struts.n_lines > 0:
                    plotter.add_mesh(
                        pv_struts,
                        name="explicit_struts",
                        color="#0FA4AF",
                        line_width=2.5,
                    )

                # Add warm coral boundary/surface nodes (#E07A5F)
                if pv_boundary is not None and pv_boundary.n_points > 0:
                    plotter.add_mesh(
                        pv_boundary,
                        name="boundary_nodes",
                        color="#E07A5F",
                        point_size=8,
                        render_points_as_spheres=True,
                    )

                plotter.reset_camera()
                if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
                    try:
                        ctrl.view_update()
                    except Exception:
                        pass

                n_s = pv_struts.n_lines if pv_struts else 0
                n_b = pv_boundary.n_points if pv_boundary else 0
                state.status_message = (
                    f"Surface Dual Preview: {display_name}\n"
                    f"Architecture: {exp_type}" + (f" ({sc_rule})" if exp_type == "SC" else "") + f" | L={cell_size}mm, r={strut_r}mm, Mode={exp_mode}\n"
                    f"Boundary Dual: {n_s} surface dual chords, {n_b} boundary nodes"
                )
                state.status_color = "success"
                return

            lattice = str(state.lattice_type)
            sf = float(np.clip(float(state.solid_fraction), 0.01, 0.95))
            wt = max(float(state.wall_thickness_mm), 0.001)
            grading = bool(state.enable_grading)
            axis = str(state.grading_axis)
            doubling = float(state.doubling_interval)
            exp_mode = str(state.export_mode).strip().lower()
            shell_t = float(state.shell_thickness)
            cutaway = bool(state.show_cutaway or state.cutaway_enabled) if exp_mode == "combined" else False

            mode = str(state.auto_calculate)
            if mode == "Solid Fraction":
                resolved_tau = np.pi * wt / cell_size
            else:
                resolved_tau = sf / 1.15

            g_mode = "solid_fraction" if "solid fraction" in str(state.grading_mode).lower() else "cell_size"

            pv_cad, pv_wall = generate_surface_tpms_preview(
                shape=shape or "Cube",
                size=size or 20.0,
                lattice_type=lattice,
                unit_cell_size=cell_size,
                solid_fraction=sf,
                wall_thickness_mm=wt,
                tau=resolved_tau,
                enable_grading=grading,
                grading_axis=axis,
                grading_mode=g_mode,
                doubling_interval_mm=doubling,
                min_solid_fraction=float(state.min_solid_fraction),
                max_solid_fraction=float(state.max_solid_fraction),
                export_mode=exp_mode,
                shell_thickness=shell_t,
                cad_file=cad_path,
                cutaway=cutaway,
            )

            pv_cad = apply_viewport_clipping(pv_cad)
            pv_wall = apply_viewport_clipping(pv_wall)

            # Clear all existing actors to prevent previous shapes from lingering
            plotter.clear_actors()

            # Add floor grid plate beneath CAD part (Blender dark studio style)
            floor = create_floor_grid(pv_cad.bounds)
            plotter.add_mesh(
                floor,
                name="floor_plate",
                color="#232528",
                edge_color="#40444C",
                show_edges=True,
                line_width=1.2,
            )

            # Add opaque solid CAD body (Blender studio titanium gray)
            plotter.add_mesh(
                pv_cad,
                name="cad_body",
                color="#D8DCE3",
                opacity=1.0,
                show_edges=False,
            )

            # Add smooth vibrant cyan TPMS wall footprint (#0FA4AF)
            if pv_wall is not None and pv_wall.n_points > 0:
                plotter.add_mesh(
                    pv_wall,
                    name="tpms_surface",
                    color="#0FA4AF",
                    opacity=1.0,
                    show_edges=False,
                )

            plotter.reset_camera()
            if hasattr(ctrl, "view_update") and callable(ctrl.view_update):
                try:
                    ctrl.view_update()
                except Exception:
                    pass

            state.status_message = (
                f"Preview updated: {display_name}\n"
                f"Lattice: {lattice} (L={cell_size}mm, SF={sf}, w={wt}mm, Mode={state.export_mode})"
            )
            state.status_color = "success"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            state.status_message = f"Preview error: {exc}"
            state.status_color = "error"

    def generate_lattice():
        """Run full lattice generation with dual STL and 3MF export."""
        state.is_generating = True
        state.status_message = "Generating lattice... this may take a moment."
        state.status_color = "info"
        state.flush()

        try:
            recompute_density()

            shape, size, cad_path, display_name = resolve_cad_input()
            cell_size = float(state.unit_cell_size)

            # Resolve requested export formats
            req_formats = []
            if bool(state.export_stl):
                req_formats.append("stl")
            if bool(state.export_3mf):
                req_formats.append("3mf")
            if not req_formats:
                req_formats = ["stl"]

            today_dir = Path("outputs") / "App outputs" / date.today().isoformat()
            today_dir.mkdir(parents=True, exist_ok=True)
            geom_stem = cad_path.stem if cad_path else shape

            if str(state.modality) == "Interlinked / PAMs":
                c_key = parse_interlinked_cell_key(state.interlinked_cell)
                seeding_mode = "cylindrical" if "cylindrical" in str(state.interlinked_seeding).lower() else "cartesian"
                wire_r = max(float(state.interlinked_wire_radius), 0.05)
                pitch = max(float(state.interlinked_pitch), 0.5)
                min_clr = max(float(state.interlinked_min_clearance), 0.01)
                cull_m = max(float(state.interlinked_cull_margin), 0.0)
                out_name = f"Trame_{geom_stem}_{c_key}_pitch{pitch:.1f}mm_clr{min_clr:.2f}mm.stl"
                out_path = today_dir / out_name

                report = run_headless_recipe(
                    modality="interlinked",
                    primitive=shape or "Cube",
                    size=size or 20.0,
                    cad_stl=cad_path,
                    cell_size=pitch,
                    strut_radius=wire_r,
                    formats=tuple(req_formats),
                    output_path=out_path,
                    interlinked_cell=c_key,
                    interlinked_seeding=seeding_mode,
                    wire_radius=wire_r,
                    min_clearance=min_clr,
                    auto_resolve_pitch=bool(state.interlinked_auto_resolve_pitch),
                    cull_margin=cull_m,
                    add_perimeter_frame=bool(state.interlinked_add_perimeter_frame),
                    frame_wall_thickness=float(state.interlinked_frame_wall_thickness),
                    frame_margin=float(state.interlinked_frame_margin),
                    frame_shape=str(state.interlinked_frame_shape),
                    cylinder_radius=float(state.interlinked_cylinder_radius),
                    cylinder_height=float(state.interlinked_cylinder_height),
                    support_recipe=str(state.interlinked_support_recipe),
                )
            elif str(state.modality) == "Auxetics":
                pat_key = parse_auxetic_pattern_key(state.auxetics_pattern)
                surf_key = parse_auxetic_surface_key(state.auxetics_surface)
                w = max(float(state.auxetics_width), 1.0)
                h = max(float(state.auxetics_height), 1.0)
                t = max(float(state.auxetics_thickness), 0.1)
                r_in = max(float(state.auxetics_r_in), 0.1)
                r_out = max(float(state.auxetics_r_out), r_in + 0.1)
                n_circ = max(int(state.auxetics_n_circumferential), 3)
                r_node = max(float(state.auxetics_r_node), 0.1)
                strut_w = max(float(state.auxetics_strut_w), 0.1)
                sq_side = max(float(state.auxetics_square_side), 1.0)
                rot_deg = float(state.auxetics_rotation_angle)
                hinge_r = max(float(state.auxetics_hinge_radius), 0.05)

                out_name = f"Trame_Auxetic_{pat_key}_{surf_key}_{w:.0f}x{h:.0f}mm.stl"
                out_path = today_dir / out_name

                report = run_headless_recipe(
                    modality="auxetics",
                    pattern=pat_key,
                    surface=surf_key,
                    width=w,
                    height=h,
                    thickness=t,
                    r_in=r_in,
                    r_out=r_out,
                    n_circumferential=n_circ,
                    r_node=r_node,
                    strut_w=strut_w,
                    square_side=sq_side,
                    rotation_angle_deg=rot_deg,
                    hinge_radius=hinge_r,
                    unit_cell_size=float(state.auxetics_pentamode_cell_size),
                    r_min=float(state.auxetics_pentamode_r_min),
                    r_max=float(state.auxetics_pentamode_r_max),
                    formats=tuple(req_formats),
                    output_path=out_path,
                )
            elif str(state.modality) == "Explicit Struts":
                exp_type = "A15" if "A15" in str(state.explicit_type) else "SC"
                sc_rule = str(state.sc_rule)
                strut_r = float(state.strut_radius)
                exp_mode = str(state.explicit_mode)
                rule_tag = f"_{sc_rule}" if exp_type == "SC" else ""
                out_name = f"Trame_{geom_stem}_{exp_type}{rule_tag}_{exp_mode}_L{cell_size:.1f}_r{strut_r:.2f}.stl"
                out_path = today_dir / out_name

                report = run_headless_recipe(
                    modality="explicit",
                    explicit_type=exp_type,
                    sc_rule=sc_rule,
                    strut_radius=strut_r,
                    conformal_mode=exp_mode,
                    primitive=shape or "Cube",
                    size=size or 20.0,
                    cad_stl=cad_path,
                    cell_size=cell_size,
                    formats=tuple(req_formats),
                    output_path=out_path,
                )
            else:
                lattice = str(state.lattice_type)
                sf = float(state.solid_fraction)
                wt = float(state.wall_thickness_mm)
                grading = bool(state.enable_grading)
                axis = str(state.grading_axis)
                doubling = float(state.doubling_interval)
                res = float(state.resolution)
                exp_mode = str(state.export_mode).strip().lower()
                shell_t = float(state.shell_thickness)
                mode_tag = f"Graded_{axis}" if grading else "Uniform"
                out_name = f"Trame_{geom_stem}_{lattice}_{exp_mode}_{mode_tag}_L{cell_size:.1f}_SF{sf:.2f}.stl"
                out_path = today_dir / out_name

                g_mode = "solid_fraction" if "solid fraction" in str(state.grading_mode).lower() else "cell_size"
                dir_map = {
                    "Z-Axial (0,0,1)": (0.0, 0.0, 1.0),
                    "X-Axial (1,0,0)": (1.0, 0.0, 0.0),
                    "Y-Axial (0,1,0)": (0.0, 1.0, 0.0),
                }
                tex_dir = dir_map.get(str(state.texture_direction), (0.0, 0.0, 1.0))
                tex_type = str(state.texture_type).lower().split()[0]

                report = run_headless_recipe(
                    primitive=shape or "Cube",
                    size=size or 20.0,
                    cad_stl=cad_path,
                    lattice_type=lattice,
                    cell_size=cell_size,
                    solid_fraction=sf,
                    wall_thickness_mm=wt,
                    export_mode=exp_mode,
                    shell_thickness=shell_t,
                    formats=tuple(req_formats),
                    enable_grading=grading,
                    grading_axis=axis,
                    grading_mode=g_mode,
                    min_solid_fraction=float(state.min_solid_fraction),
                    max_solid_fraction=float(state.max_solid_fraction),
                    doubling_interval_mm=doubling,
                    conformal_deformation=bool(state.conformal_deformation),
                    conformal_iterations=int(state.conformal_iterations),
                    enable_surface_texture=bool(state.enable_surface_texture),
                    texture_type=tex_type,
                    texture_amplitude_mm=float(state.texture_amplitude_um) / 1000.0,
                    texture_wavelength_mm=float(state.texture_wavelength_um) / 1000.0,
                    texture_direction=tex_dir,
                    texture_profile=str(state.texture_profile).lower(),
                    texture_displacement_mode="emboss" if "emboss" in str(state.texture_mode).lower() else ("engrave" if "engrave" in str(state.texture_mode).lower() else "centered"),
                    texture_triplanar=bool(state.texture_triplanar),
                    enable_micropillars=bool(state.enable_micropillars),
                    pillar_diameter_mm=float(state.micropillar_diameter_um) / 1000.0,
                    pillar_height_mm=float(state.micropillar_height_um) / 1000.0,
                    pillar_spacing_mm=float(state.micropillar_spacing_um) / 1000.0,
                    pillar_location="outer_only" if "outer" in str(state.micropillar_location).lower() else ("internal_only" if "internal" in str(state.micropillar_location).lower() else "all"),
                    pillar_filter_printable=bool(state.micropillar_filter_printable),
                    woodpile_invert=bool(state.woodpile_invert),
                    resolution=res,
                    output_path=out_path,
                )

            # Prepare direct HTTP download URLs served by Trame backend
            stem = out_path.stem
            stl_file = out_path.parent / f"{stem}.stl"
            threemf_file = out_path.parent / f"{stem}.3mf"

            if stl_file.is_file() and "stl" in req_formats:
                rel_stl = stl_file.relative_to(Path("outputs"))
                encoded_stl = "/".join(urllib.parse.quote(part) for part in rel_stl.parts)
                state.download_stl_url = f"/outputs/{encoded_stl}"
                state.download_stl_name = stl_file.name
            else:
                state.download_stl_url = ""

            if threemf_file.is_file() and "3mf" in req_formats:
                rel_3mf = threemf_file.relative_to(Path("outputs"))
                encoded_3mf = "/".join(urllib.parse.quote(part) for part in rel_3mf.parts)
                state.download_3mf_url = f"/outputs/{encoded_3mf}"
                state.download_3mf_name = threemf_file.name
            else:
                state.download_3mf_url = ""

            if str(state.modality) == "Interlinked / PAMs":
                state.status_message = (
                    f"Generated: {stem}\n"
                    f"Particles: {report.get('num_particles', 0):,} | Clearance: {report.get('min_clearance_mm', 0.0):.3f} mm (Valid: {report.get('clearance_valid', True)}) | Time: {report.get('generation_time_s', 0.0)}s\n"
                    f"Formats: {', '.join(req_formats).upper()} ready for download below."
                )
            else:
                state.status_message = (
                    f"Generated: {stem}\n"
                    f"Faces: {report['face_count']:,} | Watertight: {report['watertight']} | Time: {report['generation_time_s']}s\n"
                    f"Formats: {', '.join(req_formats).upper()} ready for download below."
                )
            state.status_color = "success"
        except Exception as exc:
            import traceback
            traceback.print_exc()
            state.status_message = f"Generation failed: {exc}"
            state.status_color = "error"
        finally:
            state.is_generating = False
            state.flush()

    ctrl.update_preview = update_3d_preview
    ctrl.generate = generate_lattice

    @state.change("modality")
    def on_modality_change(modality, **kwargs):
        """Automatically refresh 3D viewport when switching between Implicit and Explicit tabs."""
        update_3d_preview()

    @state.change("explicit_type", "sc_rule", "explicit_mode", "strut_radius")
    def on_explicit_settings_change(**kwargs):
        """Automatically refresh 3D viewport when explicit settings change in Explicit Struts tab."""
        if str(state.modality) == "Explicit Struts":
            update_3d_preview()

    def help_bubble(text: str, location: str = "top"):
        with v3.VTooltip(text=text, location=location):
            with v3.Template(v_slot_activator="{ props }"):
                v3.VIcon(
                    "mdi-help-circle-outline",
                    v_bind="props",
                    size="15px",
                    color="#AFDDE5",
                    classes="ml-1 cursor-pointer",
                )

    # --- GUI Layout with Oceanic Slate Teal (#003135 / #024950 / #0FA4AF) & Dark Studio ---
    with SinglePageWithDrawerLayout(server) as layout:
        layout.root.theme = "dark"
        layout.title.set_text("Graphite — Conformal Lattice Studio (Phase 3: Multi-Modality)")
        layout.toolbar.style = "background-color: #002528 !important; color: #AFDDE5 !important; border-bottom: 1px solid #024950 !important;"
        layout.footer.style = "background-color: #001F22 !important; color: #78909C !important; border-top: 1px solid #024950 !important;"

        with layout.toolbar:
            v3.VSpacer()
            with v3.VTooltip(text="Architecture & User Guide (FAQ)", location="bottom"):
                with v3.Template(v_slot_activator="{ props }"):
                    v3.VBtn(
                        icon="mdi-help-circle-outline",
                        v_bind="props",
                        variant="text",
                        color="#AFDDE5",
                        click="faq_dialog_open = true",
                    )

        layout.drawer.width = 460
        layout.drawer.style = "background-color: #002C30 !important; color: #E0F2F1 !important; border-right: 1px solid #024950 !important;"

        with layout.drawer:
            with v3.VContainer(classes="pa-4"):
                v3.VCardTitle(
                    "Production Lattice Studio",
                    style="color: #AFDDE5 !important; font-weight: 700; letter-spacing: 0.5px;",
                    classes="text-h6 px-0 pb-3",
                )

                # 0. Top-Level Modality Tabs
                with v3.VTabs(
                    v_model=("modality", "Implicit TPMS"),
                    color="#0FA4AF",
                    bg_color="#002226",
                    density="compact",
                    grow=True,
                    show_arrows=False,
                    classes="mb-3 rounded",
                    style="border: 1px solid #024950 !important;",
                ):
                    v3.VTab(value="Implicit TPMS", prepend_icon="mdi-waves", text="Implicit", style="min-width: 0 !important; font-size: 0.82rem !important; padding: 0 8px !important;")
                    v3.VTab(value="Explicit Struts", prepend_icon="mdi-vector-polyline", text="Explicit", style="min-width: 0 !important; font-size: 0.82rem !important; padding: 0 8px !important;")
                    v3.VTab(value="Interlinked / PAMs", prepend_icon="mdi-link-variant", text="Interlinked", style="min-width: 0 !important; font-size: 0.82rem !important; padding: 0 8px !important;")
                    v3.VTab(value="Auxetics", prepend_icon="mdi-vector-arrange-below", text="Auxetics", style="min-width: 0 !important; font-size: 0.82rem !important; padding: 0 8px !important;")

                # =====================================================================
                # IMPLICIT TPMS ACCORDION PANELS
                # =====================================================================
                with v3.VContainer(v_show="modality == 'Implicit TPMS'", classes="px-0 py-0"):
                    with v3.VExpansionPanels(
                        multiple=True,
                        v_model=("expanded_panels_implicit", [0, 1, 2]),
                        elevation=0,
                        classes="mb-3",
                    ):
                        # Panel 1: CAD Boundary Geometry
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("1. CAD Boundary Geometry", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                with v3.VRow(align="center", justify="space-between", classes="ma-0 mb-3"):
                                    v3.VChip(
                                        "{{ cad_health_status }}",
                                        color=("cad_health_color", "success"),
                                        density="compact",
                                        variant="tonal",
                                        prepend_icon="mdi-shield-check",
                                        style="font-weight: 600; font-size: 0.78rem;",
                                    )
                                v3.VSelect(
                                    label="Geometry Source",
                                    items=("source_options",),
                                    v_model=("geom_source", "Primitive"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                # Primitive Controls
                                with v3.VContainer(v_show="geom_source == 'Primitive'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Primitive Shape",
                                        items=("shape_options",),
                                        v_model=("prim_shape", "Cube"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Dimension / Size (mm)",
                                        type="number",
                                        v_model=("prim_size", 20.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                # Workspace Fixture Controls
                                with v3.VContainer(v_show="geom_source == 'Workspace Fixture'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Test Fixture (test_parts/)",
                                        items=("fixture_options",),
                                        v_model=("selected_fixture", "BaseRing_1to1.STL"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                # Custom STL Controls
                                with v3.VContainer(v_show="geom_source == 'Custom STL File'", classes="px-0 py-0"):
                                    v3.VFileInput(
                                        label="Upload STL (.stl)",
                                        v_model=("uploaded_file", None),
                                        accept=".stl,.STL",
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        prepend_icon="mdi-upload",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Or Local STL Path",
                                        v_model=("custom_file_path", ""),
                                        hint="e.g. test_parts/BaseRing_1to1.STL or absolute path",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )

                        # Panel 2: Lattice Architecture Catalog
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("2. Architecture Catalog", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Select from 7 canonical triply periodic minimal surfaces or woodpile/cross-hatch rod architectures.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Lattice Type",
                                    items=("lattice_options",),
                                    v_model=("lattice_type", "Gyroid"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )
                                with v3.VContainer(v_show="lattice_type == 'Woodpile' || lattice_type == 'Cross-Hatch'", classes="px-0 py-0"):
                                    v3.VCheckbox(
                                        label="Invert Solids (Beam Complement)",
                                        v_model=("woodpile_invert", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="mb-1",
                                    )
                                with v3.VContainer(v_show="lattice_type != 'Woodpile' && lattice_type != 'Cross-Hatch'", classes="px-0 py-0"):
                                    with v3.VCheckbox(
                                        v_model=("conformal_deformation", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        hide_details=True,
                                        classes="mb-2",
                                    ):
                                        with v3.Template(v_slot_label=True):
                                            v3.VLabel("Harmonic Conformal UVW (Experimental)", style="color: #AFDDE5; font-size: 0.88rem;")
                                            help_bubble("Solves Laplace-Beltrami boundary value problem to warp internal TPMS coordinates so walls and pores flow organically parallel to curved CAD walls rather than cutting abruptly.")

                        # Panel 3: Sizing, Density & Pore Size (4-Way Calculator)
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("3. Sizing & Pore Size Calculator", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Auto-calculate Parameter",
                                    items=("auto_calc_options",),
                                    v_model=("auto_calculate", "Wall Thickness"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    hint="The selected variable is solved from the other three",
                                    persistent_hint=True,
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Unit Cell Size L (mm)",
                                    type="number",
                                    v_model=("unit_cell_size", 5.0),
                                    disabled=("disable_unit_cell",),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Target Solid Fraction (0.05–0.90)",
                                    type="number",
                                    v_model=("solid_fraction", 0.30),
                                    disabled=("disable_solid_fraction",),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Target Wall Thickness (mm)",
                                    type="number",
                                    v_model=("wall_thickness_mm", 0.415),
                                    disabled=("disable_wall_thickness",),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Pore Size MIS (mm)",
                                    type="number",
                                    v_model=("pore_size_mm", 3.275),
                                    disabled=("disable_pore_size",),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    hint="Maximum inscribed sphere void diameter",
                                    persistent_hint=True,
                                    classes="mb-2",
                                )

                        # Panel 4: Functional Grading (Collapsed by default)
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("4. Functional Grading (Optional)", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VCheckbox(
                                    label="Enable Functional Grading",
                                    v_model=("enable_grading", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    classes="mb-2",
                                )
                                with v3.VContainer(v_show="enable_grading", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Grading Mode",
                                        items=("grading_mode_options",),
                                        v_model=("grading_mode", "Cell Size (Chirp)"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VSelect(
                                        label="Grading Direction",
                                        items=("grading_axes",),
                                        v_model=("grading_axis", "Z"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    with v3.VContainer(v_show="grading_mode == 'Solid Fraction (SF)'", classes="px-0 py-0"):
                                        v3.VTextField(
                                            label="Min Solid Fraction",
                                            type="number",
                                            v_model=("min_solid_fraction", 0.15),
                                            color="#0FA4AF",
                                            base_color="#AFDDE5",
                                            density="comfortable",
                                            variant="outlined",
                                            classes="mb-3",
                                        )
                                        v3.VTextField(
                                            label="Max Solid Fraction",
                                            type="number",
                                            v_model=("max_solid_fraction", 0.45),
                                            color="#0FA4AF",
                                            base_color="#AFDDE5",
                                            density="comfortable",
                                            variant="outlined",
                                            classes="mb-2",
                                        )
                                    with v3.VContainer(v_show="grading_mode != 'Solid Fraction (SF)'", classes="px-0 py-0"):
                                        v3.VTextField(
                                            label="Doubling Interval (mm)",
                                            type="number",
                                            v_model=("doubling_interval", 5.0),
                                            hint="Doubles unit cell size every N mm along axis",
                                            persistent_hint=True,
                                            color="#0FA4AF",
                                            base_color="#AFDDE5",
                                            density="comfortable",
                                            variant="outlined",
                                            classes="mb-2",
                                        )

                        # Panel 5: Surface Micro-Textures & Features (Collapsed by default)
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("5. Surface Micro-Textures & Features", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Apply procedural micro-grooves/knurls or synthesize explicit 50 µm micropillar forests via Boolean CSG.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                # Section A: Procedural Displacement
                                with v3.VRow(align="center", classes="ma-0 mb-1"):
                                    v3.VCheckbox(
                                        label="Enable Procedural Surface Texture",
                                        v_model=("enable_surface_texture", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="pa-0 ma-0",
                                    )
                                    help_bubble("High-resolution vertex displacement along surface normals using procedural trigonometric fields.")
                                with v3.VContainer(v_show="enable_surface_texture", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Texture Pattern",
                                        items=("texture_type_options",),
                                        v_model=("texture_type", "Microgrooves"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Amplitude (µm)",
                                        type="number",
                                        v_model=("texture_amplitude_um", 25.0),
                                        hint="Displacement depth (+/- µm)",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Wavelength / Pitch (µm)",
                                        type="number",
                                        v_model=("texture_wavelength_um", 50.0),
                                        hint="Spatial wavelength (µm)",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    with v3.VContainer(v_show="texture_type == 'Microgrooves'", classes="px-0 py-0"):
                                        v3.VSelect(
                                            label="Groove Direction",
                                            items=("texture_direction_options",),
                                            v_model=("texture_direction", "Z-Axial (0,0,1)"),
                                            color="#0FA4AF",
                                            base_color="#AFDDE5",
                                            density="comfortable",
                                            variant="outlined",
                                            classes="mb-3",
                                        )
                                        v3.VSelect(
                                            label="Wave Profile",
                                            items=("texture_profile_options",),
                                            v_model=("texture_profile", "Sine"),
                                            color="#0FA4AF",
                                            base_color="#AFDDE5",
                                            density="comfortable",
                                            variant="outlined",
                                            classes="mb-3",
                                        )
                                    v3.VSelect(
                                        label="Displacement Mode",
                                        items=("texture_mode_options",),
                                        v_model=("texture_mode", "Centered (±A)"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VCheckbox(
                                        label="Normal-Weighted Triplanar Mapping",
                                        v_model=("texture_triplanar", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="mb-2",
                                    )

                                v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                                # Section B: Explicit Micropillar Forest
                                with v3.VRow(align="center", classes="ma-0 mb-1"):
                                    v3.VCheckbox(
                                        label="Enable Micropillar Forest (CSG)",
                                        v_model=("enable_micropillars", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="pa-0 ma-0",
                                    )
                                    help_bubble("Synthesizes high-aspect-ratio cylindrical micro-posts (hairs/cilia) on 3D surfaces via Poisson-disk blue noise distribution, boolean-unioned into a watertight solid via Manifold3D.")
                                with v3.VContainer(v_show="enable_micropillars", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Pillar Diameter (µm)",
                                        type="number",
                                        v_model=("micropillar_diameter_um", 50.0),
                                        hint="Cylindrical post diameter (µm)",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Pillar Height / Length (µm)",
                                        type="number",
                                        v_model=("micropillar_height_um", 200.0),
                                        hint="Post length projecting outward (µm)",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Center-to-Center Spacing (µm)",
                                        type="number",
                                        v_model=("micropillar_spacing_um", 250.0),
                                        hint="Poisson-disk minimum spacing between posts (µm)",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VSelect(
                                        label="Placement Location",
                                        items=("micropillar_location_options",),
                                        v_model=("micropillar_location", "All Surfaces"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VCheckbox(
                                        label="Filter 3D-Printable Overhang Angles",
                                        v_model=("micropillar_filter_printable", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        hint="Discards pillars drooping horizontally to prevent support failure",
                                        persistent_hint=True,
                                        classes="mb-2",
                                    )

                        # Panel 6: Shelling & Resolution (Advanced)
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("6. Shelling & Resolution (Advanced)", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Export Mode",
                                    items=("shelling_options",),
                                    v_model=("export_mode", "Core"),
                                    hint="Core: Lattice only | Skin: Solid shell | Combined: Lattice + Shell",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                with v3.VContainer(v_show="export_mode != 'Core'", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Shell Thickness (mm)",
                                        type="number",
                                        v_model=("shell_thickness", 2.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                with v3.VContainer(v_show="export_mode == 'Combined'", classes="px-0 py-0"):
                                    v3.VCheckbox(
                                        label="Preview Cutaway Section (Inspect Core)",
                                        v_model=("show_cutaway", False),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="mb-3",
                                    )
                                v3.VCheckbox(
                                    label="Auto Nyquist Resolution (min(250 µm, wall / 2))",
                                    v_model=("auto_resolution", True),
                                    color="#0FA4AF",
                                    density="compact",
                                    classes="mb-1",
                                )
                                v3.VTextField(
                                    label="Voxel Resolution (mm)",
                                    type="number",
                                    v_model=("resolution", 0.208),
                                    disabled=("auto_resolution",),
                                    hint="Auto-scaled: 250 µm or wall_thickness/2, whichever is lower",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )

                # =======================================================================
                # EXPLICIT STRUT ACCORDION PANELS
                # =======================================================================
                with v3.VContainer(v_show="modality == 'Explicit Struts'", classes="px-0 py-0"):
                    with v3.VExpansionPanels(
                        multiple=True,
                        v_model=("expanded_panels_explicit", [0, 1, 2]),
                        elevation=0,
                        classes="mb-3",
                    ):
                        # Panel 1: CAD Boundary Geometry
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("1. CAD Boundary Geometry", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                with v3.VRow(align="center", justify="space-between", classes="ma-0 mb-3"):
                                    v3.VChip(
                                        "{{ cad_health_status }}",
                                        color=("cad_health_color", "success"),
                                        density="compact",
                                        variant="tonal",
                                        prepend_icon="mdi-shield-check",
                                        style="font-weight: 600; font-size: 0.78rem;",
                                    )
                                v3.VSelect(
                                    label="Geometry Source",
                                    items=("source_options",),
                                    v_model=("geom_source", "Primitive"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                with v3.VContainer(v_show="geom_source == 'Primitive'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Primitive Shape",
                                        items=("shape_options",),
                                        v_model=("prim_shape", "Cube"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Dimension / Size (mm)",
                                        type="number",
                                        v_model=("prim_size", 20.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="geom_source == 'Workspace Fixture'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Test Fixture (test_parts/)",
                                        items=("fixture_options",),
                                        v_model=("selected_fixture", "BaseRing_1to1.STL"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="geom_source == 'Custom STL File'", classes="px-0 py-0"):
                                    v3.VFileInput(
                                        label="Upload STL (.stl)",
                                        v_model=("uploaded_file", None),
                                        accept=".stl,.STL",
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        prepend_icon="mdi-upload",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Or Local STL Path",
                                        v_model=("custom_file_path", ""),
                                        hint="e.g. test_parts/BaseRing_1to1.STL or absolute path",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )

                        # Panel 2: Explicit Architecture
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("2. Explicit Architecture", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Explicit Architecture",
                                    items=("explicit_type_options",),
                                    v_model=("explicit_type", "A15 (Conformal Kagome)"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                with v3.VContainer(v_show="explicit_type == 'SC Hex Modular'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="SC Hex Topology Rule",
                                        items=("sc_rule_options",),
                                        v_model=("sc_rule", "octahedral"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                v3.VSelect(
                                    label="Conformation Mode",
                                    items=("explicit_mode_options",),
                                    v_model=("explicit_mode", "conformal"),
                                    hint="conformal: valency-ironed boundary | boolean: spatial CAD intersection",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )

                        # Panel 3: Strut Dimensions & Sizing
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            v3.VExpansionPanelTitle("3. Strut Dimensions & Sizing", style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VTextField(
                                    label="Unit Cell Size L (mm)",
                                    type="number",
                                    v_model=("unit_cell_size", 5.0),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Strut Radius r (mm)",
                                    type="number",
                                    v_model=("strut_radius", 0.40),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Analytical Solid Fraction (ϕ)",
                                    type="number",
                                    v_model=("explicit_solid_fraction", 0.15),
                                    readonly=True,
                                    hint="Solved analytically from unit cell strut length",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VCheckbox(
                                    label="Preview Cutaway Section (Inspect Interior)",
                                    v_model=("show_cutaway", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    classes="mb-1",
                                )

                # =======================================================================
                # INTERLINKED METAMATERIALS / PAMS ACCORDION PANELS
                # =======================================================================
                with v3.VContainer(v_show="modality == 'Interlinked / PAMs'", classes="px-0 py-0"):
                    with v3.VExpansionPanels(
                        multiple=True,
                        v_model=("expanded_panels_interlinked", [0, 1, 2]),
                        elevation=0,
                        classes="mb-3",
                    ):
                        # Panel 1: CAD Boundary & Domain Policy
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("1. Boundary Domain & Policy", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Defines the CAD boundary and domain seeding mode. Inset Culling (Policy A) guarantees only 100% whole, uncut particles survive.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                with v3.VRow(align="center", justify="space-between", classes="ma-0 mb-3"):
                                    v3.VChip(
                                        "{{ cad_health_status }}",
                                        color=("cad_health_color", "success"),
                                        density="compact",
                                        variant="tonal",
                                        prepend_icon="mdi-shield-check",
                                        style="font-weight: 600; font-size: 0.78rem;",
                                    )
                                v3.VSelect(
                                    label="Geometry Source",
                                    items=("source_options",),
                                    v_model=("geom_source", "Primitive"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                with v3.VContainer(v_show="geom_source == 'Primitive'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Primitive Shape",
                                        items=("shape_options",),
                                        v_model=("prim_shape", "Cube"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Dimension / Size (mm)",
                                        type="number",
                                        v_model=("prim_size", 20.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="geom_source == 'Workspace Fixture'", classes="px-0 py-0"):
                                    v3.VSelect(
                                        label="Test Fixture (test_parts/)",
                                        items=("fixture_options",),
                                        v_model=("selected_fixture", "BaseRing_1to1.STL"),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="geom_source == 'Custom STL File'", classes="px-0 py-0"):
                                    v3.VFileInput(
                                        label="Upload STL (.stl)",
                                        v_model=("uploaded_file", None),
                                        accept=".stl,.STL",
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        prepend_icon="mdi-upload",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Or Local STL Path",
                                        v_model=("custom_file_path", ""),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                v3.VSelect(
                                    label="Domain Seeding Mode",
                                    items=("interlinked_seeding_options",),
                                    v_model=("interlinked_seeding", "Cartesian Grid (Strict Inset Culling)"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )
                                with v3.VContainer(v_show="interlinked_seeding == 'Cylindrical Tube Wrap'", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Cylinder Target Radius (mm)",
                                        type="number",
                                        v_model=("interlinked_cylinder_radius", 15.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                    v3.VTextField(
                                        label="Cylinder Height (mm)",
                                        type="number",
                                        v_model=("interlinked_cylinder_height", 25.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="interlinked_seeding != 'Cylindrical Tube Wrap'", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Boundary Inset Margin (mm)",
                                        type="number",
                                        v_model=("interlinked_cull_margin", 0.50),
                                        hint="Policy A safety buffer: non-whole cells outside boundary are dropped",
                                        persistent_hint=True,
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )

                        # Panel 2: Kinematic Cell Architecture
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("2. Kinematic Cell Architecture", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Select registered modular kinematic unit cell. C6TT resolves 3D collision frustration; D4TET provides diamond tetrahedral catenation; European 4-in-1 and Kusari provide flexible planar maille.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Kinematic Architecture",
                                    items=("interlinked_cell_options",),
                                    v_model=("interlinked_cell", "C-6-TT (Science 2025 Truncated Tetrahedron)"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )
                                v3.VAlert(
                                    text="• C-6-TT: 3D Truncated Tetrahedron (n=6 coordination, Science 2025).\n"
                                         "• D-4-TET: Diamond Bipartite Dual Cages (n=4 coordination).\n"
                                         "• J-4-OCT: Square Planar Octahedra with 45° twist (n=4 coordination).\n"
                                         "• European 4-in-1: Classic ±28° checkerboard maille (n=4 coordination).\n"
                                         "• Japanese Kusari: Flat rings + linking arch links (n=4 coordination).\n"
                                         "• NASA Space Fabric: 6-fold spiral interlocking arms (n=6 coordination).",
                                    type="info",
                                    density="compact",
                                    variant="tonal",
                                    style="background-color: #002226 !important; color: #AFDDE5 !important; font-size: 0.80rem; line-height: 1.4; border: 1px solid #024950 !important;",
                                    classes="mb-2 pa-2",
                                )

                        # Panel 3: Clearance & Auto-Pitch Solver
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("3. Clearance & Auto-Pitch Solver", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Evaluates surface-to-surface clearance Δ = κ·a₀ - 2r. Inverting pitch solves the required unit cell size to meet your exact clearance.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                with v3.VCheckbox(
                                    v_model=("interlinked_auto_resolve_pitch", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2",
                                ):
                                    with v3.Template(v_slot_label=True):
                                        v3.VLabel("Auto-Resolve Unit Cell Pitch", style="color: #AFDDE5; font-size: 0.88rem;")
                                        help_bubble("Automatically inverts the clearance equation to solve pitch a₀ for the specified target clearance and wire radius.")
                                v3.VTextField(
                                    label="Target Physical Clearance Δ (mm)",
                                    type="number",
                                    v_model=("interlinked_min_clearance", 0.35),
                                    hint="Physical surface gap for non-welded print-in-place articulation",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Unit Cell Pitch a₀ (mm)",
                                    type="number",
                                    v_model=("interlinked_pitch", 10.0),
                                    disabled=("interlinked_auto_resolve_pitch",),
                                    hint="Spatial repeat unit period",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VTextField(
                                    label="Wire / Strut Radius r (mm)",
                                    type="number",
                                    v_model=("interlinked_wire_radius", 0.40),
                                    hint="Cross-sectional strut radius (diameter D = 2r)",
                                    persistent_hint=True,
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                v3.VAlert(
                                    text=("interlinked_clearance_status", "Clearance Valid"),
                                    type=("interlinked_clearance_color", "success"),
                                    density="compact",
                                    variant="tonal",
                                    style="border: 1px solid #024950 !important; font-size: 0.85rem; font-weight: 600;",
                                    classes="mb-2 pa-2",
                                )
                                v3.VDivider(classes="my-3", style="border-color: #024950 !important;")
                                v3.VSelect(
                                    label="Printability Support Recipe",
                                    items=("interlinked_support_recipe_options",),
                                    v_model=("interlinked_support_recipe", "None (Unprinted / Free)"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    hint="Select pre-stored breakaway pin recipe or export seed cell for custom slicer support",
                                    persistent_hint=True,
                                    classes="mb-3",
                                )
                                v3.VBtn(
                                    "Export Single Seed Cell STL",
                                    click=ctrl.export_seed_cell,
                                    style="background-color: #024950 !important; color: #AFDDE5 !important; border: 1px solid #0FA4AF !important; font-weight: 600; font-size: 0.82rem;",
                                    variant="flat",
                                    block=True,
                                    prepend_icon="mdi-cube-outline",
                                    classes="mb-2",
                                )

                        # Panel 4: Perimeter Frame & Visualization
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("4. Boundary Frame & Visualization", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Policy C solid frame welding for mechanical testing, and 3D viewport opacity controls.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                with v3.VCheckbox(
                                    v_model=("interlinked_add_perimeter_frame", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2",
                                ):
                                    with v3.Template(v_slot_label=True):
                                        v3.VLabel("Solid Perimeter Frame (Policy C)", style="color: #AFDDE5; font-size: 0.88rem;")
                                        help_bubble("Synthesizes a solid exterior border that penetrates outermost struts by margin to provide rigid gripping collars for tensile test fixtures.")
                                with v3.VContainer(v_show="interlinked_add_perimeter_frame", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Frame Wall Thickness (mm)",
                                        type="number",
                                        v_model=("interlinked_frame_wall_thickness", 2.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                    v3.VTextField(
                                        label="Frame Strut Penetration Margin (mm)",
                                        type="number",
                                        v_model=("interlinked_frame_margin", 0.50),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                v3.VCheckbox(
                                    label="Show CAD Part Envelope in 3D View",
                                    v_model=("interlinked_show_cad_boundary", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    classes="mb-1",
                                )
                                with v3.VContainer(v_show="interlinked_show_cad_boundary", classes="px-0 py-0"):
                                    v3.VSlider(
                                        label="CAD Envelope Opacity",
                                        min=0.05,
                                        max=1.0,
                                        step=0.05,
                                        v_model=("interlinked_cad_opacity", 0.15),
                                        color="#0FA4AF",
                                        density="compact",
                                        classes="mb-2",
                                    )
                                v3.VCheckbox(
                                    label="Preview Cutaway Section (Inspect Interior)",
                                    v_model=("show_cutaway", False),
                                    color="#0FA4AF",
                                    density="compact",
                                    classes="mb-1",
                                )

                # =======================================================================
                # AUXETICS & METAMATERIALS ACCORDION PANELS
                # =======================================================================
                with v3.VContainer(v_show="modality == 'Auxetics'", classes="px-0 py-0"):
                    with v3.VExpansionPanels(
                        multiple=True,
                        v_model=("expanded_panels_auxetics", [0, 1, 2]),
                        elevation=0,
                        classes="mb-3",
                    ):
                        # Panel 1: Domain Mapping
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("1. Domain Mapping", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Select domain mapping: 2D Flat Sheet / Plate or seamless 3D Cylindrical Sleeve / Tube.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Domain Surface",
                                    items=("auxetics_surface_options",),
                                    v_model=("auxetics_surface", "Flat Sheet / Plate"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-3",
                                )
                                with v3.VContainer(v_show="auxetics_surface == 'Flat Sheet / Plate'", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Sheet Width (mm)",
                                        type="number",
                                        v_model=("auxetics_width", 50.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Sheet Height (mm)",
                                        type="number",
                                        v_model=("auxetics_height", 50.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Plate / Rib Thickness (mm)",
                                        type="number",
                                        v_model=("auxetics_thickness", 2.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                with v3.VContainer(v_show="auxetics_surface == 'Cylindrical Sleeve (Tube)'", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Sleeve Length / Height (mm)",
                                        type="number",
                                        v_model=("auxetics_width", 50.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Inner Radius R_in (mm)",
                                        type="number",
                                        v_model=("auxetics_r_in", 15.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Outer Radius R_out (mm)",
                                        type="number",
                                        v_model=("auxetics_r_out", 17.5),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Circumferential Count (N)",
                                        type="number",
                                        v_model=("auxetics_n_circumferential", 6),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )

                        # Panel 2: Architecture Catalog
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("2. Architecture Catalog", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Auxetics exhibit negative Poisson's ratio (expand laterally when stretched). Default: Tetra-Chiral honeycomb (ν ≈ -1). Also supports Tri-Chiral, Anti-Chiral, Re-Entrant bowtie, Rotating Squares living hinge, and Pentamodes.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                v3.VSelect(
                                    label="Auxetic Architecture",
                                    items=("auxetics_pattern_options",),
                                    v_model=("auxetics_pattern", "Tetra-Chiral Honeycomb (Square Basis, ν ≈ -1)"),
                                    color="#0FA4AF",
                                    base_color="#AFDDE5",
                                    density="comfortable",
                                    variant="outlined",
                                    classes="mb-2",
                                )
                                v3.VAlert(
                                    text="• Tetra-Chiral: Square basis, 4 tangent ligaments per circular node (ν ≈ -1).\n"
                                         "• Tri-Chiral: Hexagonal isotropic basis, 6 tangent ligaments per node.\n"
                                         "• Anti-Chiral: Alternating chirality pairs (prevents out-of-plane saddle warping).\n"
                                         "• Re-Entrant: Classic negative Poisson's ratio bowtie honeycomb (Chen 2020).\n"
                                         "• Rotating Squares: Rigid square mechanism with compliant living hinges (ν = -1).\n"
                                         "• Pentamode: Unclamped extreme-elasticity meta-fluid scaffold.",
                                    type="info",
                                    density="compact",
                                    variant="tonal",
                                    style="background-color: #002226 !important; color: #AFDDE5 !important; font-size: 0.80rem; line-height: 1.4; border: 1px solid #024950 !important;",
                                    classes="mb-2 pa-2",
                                )

                        # Panel 3: Kinematic & Sizing Parameters
                        with v3.VExpansionPanel(
                            bg_color="#002528",
                            style="border: 1px solid #024950 !important; margin-bottom: 6px !important;",
                        ):
                            with v3.VExpansionPanelTitle(style="color: #AFDDE5 !important; font-weight: 600; font-size: 0.88rem;"):
                                v3.VLabel("3. Kinematic & Sizing Parameters", style="color: #AFDDE5; font-weight: 600;")
                                help_bubble("Configure cell size, ligament widths, node radii, or living hinge compliance dimensions.")
                            with v3.VExpansionPanelText(classes="px-2 pt-2 pb-0"):
                                # Parameters for Chiral / Anti-Chiral / Re-entrant
                                with v3.VContainer(v_show="auxetics_pattern.includes('Chiral') || auxetics_pattern.includes('Re-Entrant')", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Circular Node Radius (mm)",
                                        type="number",
                                        v_model=("auxetics_r_node", 2.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Ligament / Strut Width (mm)",
                                        type="number",
                                        v_model=("auxetics_strut_w", 1.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                # Parameters for Rotating Squares
                                with v3.VContainer(v_show="auxetics_pattern.includes('Rotating')", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Square Side Length (mm)",
                                        type="number",
                                        v_model=("auxetics_square_side", 10.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VSlider(
                                        label="Rotation / Deployment Angle (°)",
                                        min=0.0,
                                        max=60.0,
                                        step=1.0,
                                        v_model=("auxetics_rotation_angle", 30.0),
                                        color="#0FA4AF",
                                        density="compact",
                                        thumb_label="always",
                                        classes="mb-2",
                                    )
                                    v3.VTextField(
                                        label="Living Hinge Radius (mm)",
                                        type="number",
                                        v_model=("auxetics_hinge_radius", 0.45),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )
                                # Parameters for Pentamode
                                with v3.VContainer(v_show="auxetics_pattern.includes('Pentamode')", classes="px-0 py-0"):
                                    v3.VTextField(
                                        label="Unit Cell Size (mm)",
                                        type="number",
                                        v_model=("auxetics_pentamode_cell_size", 10.0),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Min Strut Radius r_min (mm)",
                                        type="number",
                                        v_model=("auxetics_pentamode_r_min", 0.35),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-3",
                                    )
                                    v3.VTextField(
                                        label="Max Strut Radius r_max (mm)",
                                        type="number",
                                        v_model=("auxetics_pentamode_r_max", 0.90),
                                        color="#0FA4AF",
                                        base_color="#AFDDE5",
                                        density="comfortable",
                                        variant="outlined",
                                        classes="mb-2",
                                    )

                # Section Cutaway / Interior Inspection Card
                with v3.VCard(
                    style="background-color: #002528 !important; border: 1px solid #024950 !important;",
                    classes="pa-3 mb-3",
                ):
                    with v3.VRow(align="center", classes="ma-0 mb-1"):
                        v3.VCheckbox(
                            label="Dynamic Section Cutaway",
                            v_model=("cutaway_enabled", False),
                            color="#0FA4AF",
                            density="compact",
                            hide_details=True,
                        )
                        help_bubble("Interactively slices the 3D model with a dynamic cutting plane along X, Y, or Z to inspect internal lattice architecture, pore connectivity, and core/shell interfaces.")
                    with v3.VContainer(v_show="cutaway_enabled", classes="px-0 py-0"):
                        with v3.VBtnToggle(
                            v_model=("cutaway_axis", "X"),
                            mandatory=True,
                            density="compact",
                            color="#0FA4AF",
                            classes="mb-2",
                        ):
                            v3.VBtn(value="X", text="Cut X")
                            v3.VBtn(value="Y", text="Cut Y")
                            v3.VBtn(value="Z", text="Cut Z")
                        v3.VSlider(
                            label="Plane Position (%)",
                            min=0.0,
                            max=100.0,
                            step=1.0,
                            v_model=("cutaway_pos", 50.0),
                            color="#0FA4AF",
                            density="compact",
                            thumb_label="always",
                            classes="mb-1",
                        )
                        v3.VCheckbox(
                            label="Invert Cut Direction",
                            v_model=("cutaway_invert", False),
                            color="#0FA4AF",
                            density="compact",
                            hide_details=True,
                        )

                # =====================================================================
                # EXECUTION & DUAL EXPORT (Always visible)
                # =====================================================================
                v3.VCardSubtitle(
                    "Execution & Dual Export",
                    style="color: #0FA4AF !important;",
                    classes="text-subtitle-2 px-0 font-weight-bold mb-2",
                )

                # Export Formats Row
                with v3.VRow(classes="px-2 mb-2"):
                    with v3.VCol(cols=6, classes="py-0 px-1"):
                        v3.VCheckbox(
                            label="Export STL",
                            v_model=("export_stl", True),
                            color="#0FA4AF",
                            density="compact",
                        )
                    with v3.VCol(cols=6, classes="py-0 px-1"):
                        v3.VCheckbox(
                            label="Export 3MF",
                            v_model=("export_3mf", True),
                            color="#0FA4AF",
                            density="compact",
                        )

                # Vibrant Cyan secondary action
                v3.VBtn(
                    "Update 3D Preview",
                    click=ctrl.update_preview,
                    style="background-color: #0FA4AF !important; color: #002528 !important; font-weight: 700;",
                    block=True,
                    classes="mb-3",
                    prepend_icon="mdi-refresh",
                )
                # Terracotta primary call-to-action
                v3.VBtn(
                    "Generate Production Lattice",
                    click=ctrl.generate,
                    loading=("is_generating", False),
                    style="background-color: #964734 !important; color: #FFFFFF !important; font-weight: 700;",
                    block=True,
                    classes="mb-3",
                    prepend_icon="mdi-cube-outline",
                )

                # Dual Download Buttons (Oceanic slate flat buttons with vibrant cyan border)
                v3.VBtn(
                    "Download STL Mesh",
                    href=("download_stl_url", ""),
                    download=("download_stl_name", "lattice.stl"),
                    target="_blank",
                    v_show="download_stl_url",
                    style="background-color: #024950 !important; color: #AFDDE5 !important; border: 1px solid #0FA4AF !important; font-weight: 700;",
                    variant="flat",
                    block=True,
                    prepend_icon="mdi-download",
                    classes="mb-2",
                )
                v3.VBtn(
                    "Download 3MF Mesh",
                    href=("download_3mf_url", ""),
                    download=("download_3mf_name", "lattice.3mf"),
                    target="_blank",
                    v_show="download_3mf_url",
                    style="background-color: #024950 !important; color: #AFDDE5 !important; border: 1px solid #0FA4AF !important; font-weight: 700;",
                    variant="flat",
                    block=True,
                    prepend_icon="mdi-cube-send",
                    classes="mb-3",
                )

                v3.VAlert(
                    text=("status_message", "Ready."),
                    type=("status_color", "info"),
                    density="compact",
                    variant="tonal",
                    style="background-color: #024950 !important; color: #E0F2F1 !important; border: 1px solid #0FA4AF !important; white-space: pre-wrap; word-break: break-word; line-height: 1.5; font-size: 13px;",
                    classes="mt-2 pa-3",
                )

        # Main 3D Viewport
        with layout.content:
            with v3.VContainer(fluid=True, classes="fill-height pa-0 ma-0", style="background-color: #2E3035 !important;"):
                view = plotter_ui(plotter, mode="server")
                ctrl.view_update = view.update

            # Architecture & User Guide FAQ Modal Dialog
            with v3.VDialog(v_model=("faq_dialog_open", False), max_width=860):
                with v3.VCard(style="background-color: #002528 !important; color: #E0F2F1 !important; border: 1px solid #0FA4AF !important;"):
                    with v3.VCardTitle(classes="text-h6 d-flex align-center justify-space-between pa-4", style="color: #AFDDE5 !important; border-bottom: 1px solid #024950 !important;"):
                        with v3.VRow(align="center", classes="ma-0"):
                            v3.VIcon("mdi-book-open-page-variant", color="#0FA4AF", classes="mr-2")
                            v3.VLabel("Graphite Studio — Architecture & User Guide (FAQ)", style="color: #AFDDE5; font-weight: 700; font-size: 1.15rem;")
                        v3.VBtn(icon="mdi-close", variant="text", color="#AFDDE5", click="faq_dialog_open = false")
                    with v3.VCardText(classes="pa-5", style="max-height: 650px; overflow-y: auto; line-height: 1.6;"):
                        # Section 1: Modal Architecture
                        v3.VLabel("1. Modality Comparison: Implicit vs. Explicit vs. Interlinked", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "• Implicit TPMS: Continuous trigonometric level-set surfaces (Gyroid, Diamond, Schwarz-P, Woodpile) without sharp corners or nodal stress concentrations. Best for osseointegration and heat exchangers.\n"
                            "• Explicit Struts: Rigid structural wireframe beam trusses (A15 Conformal Kagome, SC Hex Modular) connected by clean bisector-mitered joints. Best for lightweight load-bearing aerospace scaffolds.\n"
                            "• Interlinked / PAMs: Kinematic polycatenated metamaterials and chainmail textiles with strictly positive physical clearance (Δ > 0) between adjacent closed rings or cages for non-welded print-in-place articulation.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 2: 4-Way Sizing & Pore MIS
                        v3.VLabel("2. Sizing & Pore Size (MIS) Calculator", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "Connects four interdependent parameters in closed-form:\n"
                            "• Unit Cell Size (L): Linear spatial period of the lattice (mm).\n"
                            "• Solid Fraction (ϕ): Volume fraction of solid material (0.05 to 0.90).\n"
                            "• Wall Thickness (w): Physical strut/sheet thickness, solved via calibration curves (w ≈ τ·L / π).\n"
                            "• Minimum Ingrowth Pore Size (MIS, D_pore): Diameter of the largest sphere that can freely pass through internal lumens (D_pore = L·(1 - 1.15ϕ)).\n"
                            "Selecting any variable automatically locks it and recalculates the remaining three.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 3: Conformal Harmonic UVW
                        v3.VLabel("3. What is Harmonic Conformal UVW (Experimental)?", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "Standard implicit lattices evaluate trigonometric equations on a rigid Cartesian (X,Y,Z) grid, which cuts through outer CAD walls arbitrarily.\n"
                            "Harmonic Conformal UVW solves a discrete Laplace-Beltrami Dirichlet boundary value problem (∇²U = 0, ∇²V = 0, ∇²W = 0) inside the voxelized CAD domain.\n"
                            "The resulting (U,V,W) coordinates bend and flow organically parallel to curved CAD walls, eliminating abrupt cuts along organic surfaces.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 4: Micro-Textures & Micropillars
                        v3.VLabel("4. Surface Micro-Textures & Micropillar Forests", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "• Procedural Micro-Textures: Displaces mesh vertices along surface normals using high-frequency sine, triangle, or square waves (Microgrooves), Bumps, Knurling, or Spinodal relief with optional triplanar projection.\n"
                            "• Micropillar Forests: Generates thousands of high-aspect-ratio cylindrical micro-posts (e.g. 50 µm diameter, 200 µm length) distributed via Poisson-disk blue noise sampling, boolean-unioned into a watertight solid with Manifold3D.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 5: Dual Export & Nyquist Resolution
                        v3.VLabel("5. Production Export & Nyquist Resolution", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "• Auto Nyquist Voxel Resolution: Automatically sets voxel pitch to min(250 µm, wall_thickness / 2) to ensure at least 2 voxels across thin walls, preventing pinch defects.\n"
                            "• Direct Browser Downloads: Serves both native multi-body 3MF (.3mf) and STL (.stl) directly over HTTP without disconnecting the live WebSocket session.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 6: Interlinked Metamaterials & Policy A Inset Culling
                        v3.VLabel("6. Interlinked Metamaterials & Policy A Inset Culling", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "• Why Policy A (Strict Inset Culling)? Cutting or planar trimming an interlinked ring or polyhedral cage destroys its closed-loop catenation topology. Inset Culling overlays un-deformed periodic units and drops any particle that exceeds the CAD boundary minus margin, guaranteeing 100% whole, uncut links.\n"
                            "• Cylindrical Tube Wrap: Quantizes unit cell pitch along the circumference (2πR = N_θ·a_θ) to ensure seamless, continuous chainmail loops around tubular geometry.\n"
                            "• Clearance Solver (Δ): Connects physical surface gap, wire radius, and unit cell pitch via closed-form kinematics (Δ = κ·a₀ - 2r), with auto-pitch inversion for zero collision.\n"
                            "• Instanced 3MF Export: Saves each unique cage geometry once in <resources> and references instances with SE(3) transforms in <build>, slashing file size by >98% compared to monolithic STLs.",
                            classes="px-0 pt-0 pb-3",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

                        v3.VDivider(classes="my-3", style="border-color: #024950 !important;")

                        # Section 7: Auxetics & Negative Poisson's Ratio
                        v3.VLabel("7. Auxetic Metamaterials & Negative Poisson's Ratio (ν < 0)", classes="text-subtitle-1 font-weight-bold mb-1", style="color: #0FA4AF;")
                        v3.VCardSubtitle(
                            "• Negative Poisson's Ratio (ν < 0): Unlike conventional materials that thin out when stretched, auxetic metamaterials expand laterally under tension and contract when compressed, providing exceptional energy absorption, synclastic dome curvature, and indentation resistance.\n"
                            "• Chiral Honeycombs: Circular nodes with tangent ligaments that rotate under tension. Tetra-chiral (square basis, ν ≈ -1) provides orthogonal symmetry; tri-chiral provides isotropic in-plane expansion; anti-chiral pairs cancel out-of-plane saddle warping.\n"
                            "• Rotating Rigid Squares: Ideal 2D mechanism consisting of rigid square plates linked at their vertices by compliant living hinges, achieving a theoretical Poisson's ratio ν = -1.\n"
                            "• Re-Entrant Bowties: Classic inverted cellular ribs that flex outward under longitudinal tension (Chen et al. 2020).\n"
                            "• Seamless Cylindrical Wrap: Seamlessly rolls the auxetic network into a closed tubular sleeve without seams or edge dislocations for stents, sleeves, and conformal grips.",
                            classes="px-0 pt-0 pb-1",
                            style="white-space: pre-wrap; color: #AFDDE5; font-size: 0.88rem;"
                        )

        # Protect PyVista's render callback so it never crashes before websocket connection
        existing_cbs = list(plotter._on_render_callbacks)
        plotter._on_render_callbacks.clear()
        for cb in existing_cbs:
            def _wrap_cb(c):
                def _safe_call(*args, **kwargs):
                    try:
                        return c(*args, **kwargs)
                    except Exception:
                        pass
                return _safe_call
            plotter._on_render_callbacks.add(_wrap_cb(cb))

    # Initial preview population with clean actor names
    try:
        plotter.clear_actors()
        shape = str(state.prim_shape)
        size = float(state.prim_size)
        pv_cad, pv_wall = generate_surface_tpms_preview(shape, size, "Gyroid", 5.0, 0.30, tau=0.30 / 1.15)
        floor = create_floor_grid(pv_cad.bounds)
        plotter.add_mesh(floor, name="floor_plate", color="#232528", edge_color="#40444C", show_edges=True, line_width=1.2)
        plotter.add_mesh(pv_cad, name="cad_body", color="#D8DCE3", opacity=1.0, show_edges=False)
        if pv_wall is not None:
            plotter.add_mesh(pv_wall, name="tpms_surface", color="#0FA4AF", opacity=1.0, show_edges=False)
        plotter.reset_camera()
    except Exception:
        pass

    return server
