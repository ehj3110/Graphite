import os
import tempfile
from pathlib import Path

import streamlit as st
import trimesh
from graphite.geometry.primitives import generate_primitive
from graphite.geometry.surface_picking import visualize_surfaces
from graphite.implicit.chirped import generate_chirped_lattice
from graphite.implicit.conformal import generate_conformal_lattice
from graphite.implicit.graded import generate_graded_lattice
from graphite.implicit.osteochondral import generate_osteochondral_lattice


st.set_page_config(page_title="Graphite Lattice Engine", layout="wide")


def try_launch_surface_picker() -> None:
    """Open PyVista surface picker on the uploaded Custom STL (temp file + cleanup)."""
    params = st.session_state.params
    if params.get("geometry_type") != "Custom STL":
        st.warning(
            "Surface Picker only works with **Custom STL**. Choose Custom STL in Step 1 and upload your part."
        )
        return
    if st.session_state.uploaded_file is None:
        st.warning("Please upload a Custom STL in Step 1 first.")
        return
    tmp_path = None
    try:
        feature_angle = float(st.session_state.params.get("feature_angle", 45.0))
        st.session_state.uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
            tmp.write(st.session_state.uploaded_file.getvalue())
            tmp_path = tmp.name
        st.info(
            "Opening PyVista — numbered surfaces will appear. **Close the 3D window** to return to the app."
        )
        visualize_surfaces(Path(tmp_path), feature_angle=feature_angle)
        st.success("Surface picker closed.")
    except Exception as exc:
        st.error(f"Surface picker failed: {exc}")
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if "step" not in st.session_state:
    st.session_state.step = 1
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "modifier_file" not in st.session_state:
    st.session_state.modifier_file = None
if "params" not in st.session_state:
    st.session_state.params = {
        "geometry_type": "Custom STL",
        "engine_type": "Implicit (TPMS)",
        "lattice_type": "Gyroid",
        "explicit_topology": "Kagome Surface Dual",
        "explicit_cell_size": 2.5,
        "explicit_strut_radius": 0.3,
        "resolution": 0.25,
        "solid_fraction": 0.33,
        "pore_size": 5.0,
        "export_mode": "core",
        "shell_thickness": 2.0,
        "center_origin": True,
        "prim_shape": "Cube",
        "prim_size": 20.0,
        "feature_angle": 45.0,
    }


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def step_1():
    st.title("Step 1: Geometry Selection")
    params = st.session_state.params

    geometry_type = st.radio(
        "Geometry Type",
        ["Custom STL", "ASTM Standard", "Primitive"],
        index=["Custom STL", "ASTM Standard", "Primitive"].index(
            params.get("geometry_type", "Custom STL")
        ),
    )
    params["geometry_type"] = geometry_type

    if geometry_type == "Custom STL":
        uploaded_file = st.file_uploader("Upload STL geometry", type=["stl"])
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            params["uploaded_filename"] = uploaded_file.name
            st.success(f"Loaded: {uploaded_file.name}")
        else:
            st.session_state.uploaded_file = None

    elif geometry_type == "ASTM Standard":
        astm_choice = st.selectbox(
            "ASTM Geometry",
            ["ASTM F42 Coupon", "ASTM Compression Cube", "ASTM Dogbone"],
            index=0,
        )
        params["astm_standard"] = astm_choice

    elif geometry_type == "Primitive":
        # Defaults for rapid iteration: cube, 20 mm
        params.setdefault("prim_shape", "Cube")
        params.setdefault("prim_size", 20.0)
        prim_shape = st.selectbox(
            "Primitive Shape",
            ["Cube", "Sphere", "Cylinder"],
            index=["Cube", "Sphere", "Cylinder"].index(
                params.get("prim_shape", "Cube")
            ),
        )
        params["prim_shape"] = prim_shape
        params["prim_size"] = st.number_input(
            "Major Dimension (Size/Diameter) in mm",
            min_value=1.0,
            value=float(params.get("prim_size", 20.0)),
            step=1.0,
        )


def step_2():
    st.title("Step 2: Lattice Selection")
    params = st.session_state.params

    # Migrate legacy `engine` key if present
    if "engine_type" not in params and params.get("engine"):
        legacy = params.get("engine", "Implicit (TPMS)")
        params["engine_type"] = (
            "Explicit - Fast (Delaunay)"
            if legacy == "Explicit (Struts)" or legacy == "Explicit - Fast (Delaunay)"
            else "Implicit (TPMS)"
        )

    _engine_options = ["Implicit (TPMS)", "Explicit - Fast (Delaunay)"]
    _current = params.get("engine_type", "Implicit (TPMS)")
    if _current not in _engine_options:
        _current = "Implicit (TPMS)"
    engine_type = st.radio(
        "Lattice Architecture",
        _engine_options,
        index=_engine_options.index(_current),
    )
    params["engine_type"] = engine_type

    if engine_type == "Implicit (TPMS)":
        params["lattice_type"] = st.selectbox(
            "Lattice Type",
            ["Gyroid", "Diamond", "Schwarz-P", "Neovius", "Split-P"],
            index=["Gyroid", "Diamond", "Schwarz-P", "Neovius", "Split-P"].index(
                params.get("lattice_type", "Gyroid")
            ),
        )
    else:
        params["explicit_topology"] = st.selectbox(
            "Topology Rule",
            ["Standard Tet", "Surface Cage / Dual"],
            index=(
                ["Standard Tet", "Surface Cage / Dual"].index(
                    params.get("explicit_topology", "Standard Tet")
                )
                if params.get("explicit_topology", "Standard Tet")
                in ["Standard Tet", "Surface Cage / Dual"]
                else 0
            ),
        )
        params["explicit_cell_size"] = st.number_input(
            "Target Cell Size (mm)",
            min_value=1.0,
            value=float(params.get("explicit_cell_size", 5.0)),
            step=0.5,
        )
        params["explicit_strut_radius"] = st.number_input(
            "Strut Radius (mm)",
            min_value=0.1,
            value=float(params.get("explicit_strut_radius", 0.5)),
            step=0.1,
        )


def step_3():
    st.title("Step 3: Sizing & Density")
    params = st.session_state.params

    params["solid_fraction"] = st.number_input(
        "Target Solid Fraction",
        min_value=0.01,
        max_value=0.99,
        value=float(params.get("solid_fraction", 0.33)),
        step=0.01,
        format="%.2f",
    )

    size_mode = st.radio(
        "Sizing Mode",
        ["Pore Size (mm)", "Unit Cell Size (mm)"],
        index=0 if params.get("size_mode", "Pore Size (mm)") == "Pore Size (mm)" else 1,
    )
    params["size_mode"] = size_mode

    if size_mode == "Pore Size (mm)":
        params["pore_size"] = st.number_input(
            "Pore Size (mm)",
            min_value=0.1,
            value=float(params.get("pore_size", 5.0)),
            step=0.1,
        )
    else:
        params["unit_cell_size"] = st.number_input(
            "Unit Cell Size (mm)",
            min_value=0.1,
            value=float(params.get("unit_cell_size", 5.0)),
            step=0.1,
        )


def step_4():
    st.title("Step 4: Boundary & Field Control")
    params = st.session_state.params

    params["conformal_masking"] = st.checkbox(
        "Conformal Masking (EDT)",
        value=bool(params.get("conformal_masking", True)),
    )

    _grading_options = [
        "Uniform",
        "Variable Porosity (Thickness)",
        "Variable Pore Size (Chirped)",
        "Osteochondral (Layered Z)",
        "Boundary-Driven (Dual-EDT)",
    ]
    grading_mode = st.selectbox(
        "Functional Grading",
        _grading_options,
        index=_grading_options.index(params.get("grading_mode", "Uniform"))
        if params.get("grading_mode", "Uniform") in _grading_options
        else 0,
    )
    params["grading_mode"] = grading_mode

    if grading_mode == "Osteochondral (Layered Z)":
        st.info("Define control surfaces from the BOTTOM of the part (Z=0) upwards. Separate values with commas.")
        z_str = st.text_input("Z-Heights (mm)", value="0, 5, 10")
        p_str = st.text_input("Pore Sizes (mm)", value="2.0, 2.0, 5.0")
        sf_str = st.text_input("Solid Fractions", value="0.4, 0.4, 0.15")

        try:
            params["osteo_z"] = [float(x.strip()) for x in z_str.split(",")]
            params["osteo_p"] = [float(x.strip()) for x in p_str.split(",")]
            params["osteo_sf"] = [float(x.strip()) for x in sf_str.split(",")]
        except ValueError:
            st.error("Please enter valid comma-separated numbers.")
    elif grading_mode == "Boundary-Driven (Dual-EDT)":
        st.info(
            "Define distances expanding inward from the Start Surface, then define the final "
            "End Surface parameters."
        )

        col_s, col_e = st.columns(2)
        with col_s:
            start_str = st.text_input("Start Surface IDs", value="0, 1")
            d_str = st.text_input("Start Distances (mm)", value="0.0, 5.0")
            p_str = st.text_input("Start Pore Sizes (mm)", value="2.0, 2.0")
            sf_str = st.text_input("Start Solid Fractions", value="0.4, 0.4")
        with col_e:
            end_str = st.text_input("End Surface IDs", value="5")
            end_p = st.number_input("End Pore Size (mm)", value=6.0, step=0.5)
            end_sf = st.number_input("End Solid Fraction", value=0.15, step=0.05)

        st.session_state.params["feature_angle"] = st.slider(
            "Surface Angle Tolerance (Degrees)",
            1.0,
            90.0,
            float(st.session_state.params.get("feature_angle", 45.0)),
            help="Groups adjacent triangles into a single surface if the angle between them is less than this value.",
            key="step4_feature_angle",
        )

        st.button(
            "Launch Surface Picker (Pop-out)",
            key="boundary_surface_picker",
            on_click=try_launch_surface_picker,
            type="secondary",
        )

        try:
            params["bound_start"] = [int(x.strip()) for x in start_str.split(",") if x.strip()]
            params["bound_end"] = [int(x.strip()) for x in end_str.split(",") if x.strip()]
            params["bound_d"] = [float(x.strip()) for x in d_str.split(",") if x.strip()]
            params["bound_p"] = [float(x.strip()) for x in p_str.split(",") if x.strip()]
            params["bound_sf"] = [float(x.strip()) for x in sf_str.split(",") if x.strip()]
            params["bound_end_p"] = end_p
            params["bound_end_sf"] = end_sf
        except ValueError:
            st.error("Please enter valid comma-separated numbers.")
    elif grading_mode in ["Variable Porosity (Thickness)", "Variable Pore Size (Chirped)"]:
        grad_type = st.selectbox(
            "Grading Axis / Method",
            ["X", "Y", "Z", "Radial", "Modifier STL"],
            index=["X", "Y", "Z", "Radial", "Modifier STL"].index(
                params.get("gradient_type", "Z")
            )
            if params.get("gradient_type", "Z") in ["X", "Y", "Z", "Radial", "Modifier STL"]
            else 2,
        )
        params["gradient_type"] = grad_type

        if grad_type == "Modifier STL":
            mod_file = st.file_uploader("Upload Modifier STL", type=["stl"], key="mod_uploader")
            st.session_state.modifier_file = mod_file
            params["transition_width"] = st.slider(
                "Transition Width (mm)",
                min_value=1.0,
                max_value=20.0,
                value=float(params.get("transition_width", 5.0)),
            )
        else:
            st.session_state.modifier_file = None


def step_5():
    st.title("Step 5: Shelling & Export Modes")
    params = st.session_state.params

    export_label = st.selectbox(
        "Export Mode",
        ["Core Only", "Hollow Skin Only", "Combined (Core + Skin)"],
        index={
            "core": 0,
            "skin": 1,
            "combined": 2,
        }.get(params.get("export_mode", "core"), 0),
    )
    params["export_mode"] = {
        "Core Only": "core",
        "Hollow Skin Only": "skin",
        "Combined (Core + Skin)": "combined",
    }[export_label]

    if params["export_mode"] in {"skin", "combined"}:
        params["shell_thickness"] = st.slider(
            "Shell Thickness (mm)",
            min_value=0.1,
            max_value=10.0,
            value=float(params.get("shell_thickness", 2.0)),
            step=0.1,
        )

    st.markdown("**Surface IDs:** Use the picker to label planar facets (e.g. for boundary-driven grading).")
    if params.get("grading_mode") == "Boundary-Driven (Dual-EDT)":
        params["feature_angle"] = st.slider(
            "Surface Angle Tolerance (Degrees)",
            1.0,
            90.0,
            float(params.get("feature_angle", 45.0)),
            help="Groups adjacent triangles into a single surface if the angle between them is less than this value.",
            key="step5_feature_angle",
        )
    if st.button("Launch Surface Picker (Pop-out)", type="secondary"):
        try_launch_surface_picker()


def step_6():
    st.title("Step 6: Execution")
    params = st.session_state.params

    st.session_state.params["resolution"] = st.number_input(
        "Voxel Resolution (mm)",
        min_value=0.01,
        max_value=5.00,
        value=float(params.get("resolution", 0.25)),
        step=0.01,
        format="%.2f",
    )
    params["output_name"] = st.text_input(
        "Output File Name",
        value=params.get("output_name", "Generated_Scaffold.stl"),
    )
    params["center_origin"] = st.checkbox(
        "Center Lattice at Origin (0,0,0)",
        value=bool(params.get("center_origin", True)),
    )

    output_preview = os.path.join(
        str(Path.cwd()),
        "Implicit_Lattice_Exploration",
        "output",
        params["output_name"],
    )
    st.caption(f"Output preview: {output_preview}")

    if st.button("Generate Scaffold", type="primary", use_container_width=True):
        geom_type = params["geometry_type"]
        temp_input_path = None
        mod_path = None
        try:
            engine_type = params.get("engine_type", "Implicit (TPMS)")
            _spin = (
                "Generating explicit strut lattice (GMSH + manifold3d)... This may take a while."
                if engine_type == "Explicit - Fast (Delaunay)"
                else "Generating implicit lattice... This may take a moment."
            )
            with st.spinner(_spin):
                if geom_type == "Primitive":
                    shape = params["prim_shape"]
                    size = params["prim_size"]
                    base_mesh = generate_primitive(shape, size)
                elif geom_type == "Custom STL":
                    if st.session_state.uploaded_file is None:
                        st.warning("Please upload a file in Step 1.")
                        st.stop()
                    st.session_state.uploaded_file.seek(0)
                    base_mesh = trimesh.load(st.session_state.uploaded_file, file_type="stl")
                    if not isinstance(base_mesh, trimesh.Trimesh):
                        base_mesh = base_mesh.dump(concatenate=True)
                else:
                    st.warning("ASTM standard geometry generation is not wired yet. Please use Custom STL or Primitive.")
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_in:
                    base_mesh.export(tmp_in.name)
                    temp_input_path = tmp_in.name

                if "modifier_file" in st.session_state and st.session_state.modifier_file is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_mod:
                        tmp_mod.write(st.session_state.modifier_file.getvalue())
                        mod_path = tmp_mod.name

                output_filename = params.get("output_name", "Generated_Lattice")
                if not output_filename.endswith(".stl"):
                    output_filename += ".stl"

                output_path = Path("Implicit_Lattice_Exploration/output") / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if engine_type == "Explicit - Fast (Delaunay)":
                    from graphite.explicit import (
                        generate_conformal_scaffold,
                        generate_geometry,
                        generate_topology,
                    )

                    st.info("Generating conformal tetrahedral scaffold (GMSH)...")
                    boundary_mesh = trimesh.Trimesh(
                        vertices=base_mesh.vertices,
                        faces=base_mesh.faces,
                        process=True,
                    )
                    cell_size = float(params.get("explicit_cell_size", 2.5))
                    strut_radius = float(params.get("explicit_strut_radius", 0.3))
                    topology_ui = params.get(
                        "explicit_topology", "Kagome Surface Dual"
                    )

                    scaffold = generate_conformal_scaffold(
                        boundary_mesh,
                        target_element_size=cell_size,
                    )

                    st.info("Applying topology rules...")
                    if topology_ui == "Kagome Surface Dual":
                        topo_type = "kagome"
                        include_surface_cage = True
                    elif topology_ui == "Standard Tet":
                        topo_type = "rhombic"
                        include_surface_cage = False
                    elif topology_ui in (
                        "Surface Cage / Dual (Voronoi)",
                        "Surface Cage / Dual",
                    ):
                        topo_type = "voronoi"
                        include_surface_cage = True
                    else:
                        topo_type = "rhombic"
                        include_surface_cage = False

                    nodes_out, struts = generate_topology(
                        scaffold.nodes,
                        scaffold.elements,
                        scaffold.surface_faces,
                        type=topo_type,
                        include_surface_cage=include_surface_cage,
                        target_element_size=cell_size,
                    )

                    st.info("Sweeping strut geometry (manifold3d)...")
                    geom_out = generate_geometry(
                        nodes_out,
                        struts,
                        strut_radius,
                        boundary_mesh=boundary_mesh,
                        crop_to_boundary=True,
                    )
                    explicit_mesh = geom_out[0] if isinstance(geom_out, tuple) else geom_out

                    if params.get("center_origin"):
                        explicit_mesh = explicit_mesh.copy()
                        explicit_mesh.vertices = explicit_mesh.vertices - explicit_mesh.centroid

                    explicit_mesh.export(str(output_path))

                elif engine_type == "Implicit (TPMS)":
                    grading_mode = params.get("grading_mode", "Uniform")
                    if grading_mode == "Uniform":
                        generate_conformal_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            resolution=params["resolution"],
                            pore_size=params.get("pore_size"),
                            solid_fraction=params["solid_fraction"],
                            export_mode=params["export_mode"],
                            shell_thickness=params["shell_thickness"],
                            center_origin=params["center_origin"],
                            output_path=output_path,
                        )
                    elif grading_mode == "Variable Porosity (Thickness)":
                        if params.get("gradient_type") == "Modifier STL" and mod_path is None:
                            st.warning("Modifier STL is required for graded thickness.")
                            st.stop()
                        generate_graded_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            gradient_type=(
                                "modifier"
                                if params.get("gradient_type") == "Modifier STL"
                                else params.get("gradient_type", "Z")
                            ),
                            modifier_path=mod_path,
                            resolution=params["resolution"],
                            pore_size=params.get("pore_size", 5.0),
                            min_solid_fraction=params.get("min_solid_fraction", 0.10),
                            max_solid_fraction=params.get(
                                "max_solid_fraction", params["solid_fraction"]
                            ),
                            transition_width=params.get("transition_width", 5.0),
                            center_origin=params["center_origin"],
                            output_path=output_path,
                        )
                    elif grading_mode == "Variable Pore Size (Chirped)":
                        if params.get("gradient_type") == "Modifier STL" and mod_path is None:
                            st.warning("Modifier STL is required for chirped pore size.")
                            st.stop()
                        generate_chirped_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            gradient_type=(
                                "modifier"
                                if params.get("gradient_type") == "Modifier STL"
                                else params.get("gradient_type", "Z")
                            ),
                            modifier_path=mod_path,
                            resolution=params["resolution"],
                            solid_fraction=params["solid_fraction"],
                            transition_width=params.get("transition_width", 5.0),
                            center_origin=params["center_origin"],
                            output_path=output_path,
                        )
                    elif grading_mode == "Osteochondral (Layered Z)":
                        generate_osteochondral_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            z_heights=params["osteo_z"],
                            pore_sizes=params["osteo_p"],
                            solid_fractions=params["osteo_sf"],
                            resolution=params["resolution"],
                            center_origin=params["center_origin"],
                            output_path=output_path,
                        )
                    elif grading_mode == "Boundary-Driven (Dual-EDT)":
                        from graphite.implicit.boundary_graded import (
                            generate_boundary_graded_lattice,
                        )

                        generate_boundary_graded_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            start_surfaces=params.get("bound_start", [0]),
                            end_surfaces=params.get("bound_end", [1]),
                            start_distances=params.get("bound_d", [0.0, 5.0]),
                            start_pore_sizes=params.get("bound_p", [2.0, 2.0]),
                            start_solid_fractions=params.get("bound_sf", [0.4, 0.4]),
                            end_pore_size=params.get("bound_end_p", 6.0),
                            end_solid_fraction=params.get("bound_end_sf", 0.15),
                            resolution=params["resolution"],
                            feature_angle=st.session_state.params.get("feature_angle", 45.0),
                            center_origin=params["center_origin"],
                            output_path=output_path,
                        )
                    else:
                        raise ValueError(f"Unknown grading mode: {grading_mode}")
                else:
                    raise ValueError(f"Unknown lattice architecture: {engine_type}")

            st.success(f"Success! Lattice saved to {output_path}")

            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download STL",
                    data=file.read(),
                    file_name=output_filename,
                    mime="model/stl",
                )
        except Exception as exc:
            st.error(f"An error occurred: {exc}")
        finally:
            if temp_input_path is not None and os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if mod_path is not None and os.path.exists(mod_path):
                os.remove(mod_path)


st.sidebar.title("Workflow Progress")
st.sidebar.write(f"Step {st.session_state.step} of 6")
st.sidebar.progress(st.session_state.step / 6.0)

# --- MAIN LAYOUT ---
main_col, summary_col = st.columns([3, 1])

with main_col:
    # 1. Render the current step
    if st.session_state.step == 1:
        step_1()
    elif st.session_state.step == 2:
        step_2()
    elif st.session_state.step == 3:
        step_3()
    elif st.session_state.step == 4:
        step_4()
    elif st.session_state.step == 5:
        step_5()
    elif st.session_state.step == 6:
        step_6()

    st.write("---")
    # 2. Render Navigation Buttons inside the main column
    nav1, nav2, nav3 = st.columns([1, 8, 1])
    with nav1:
        if st.session_state.step > 1:
            st.button("Back", on_click=prev_step)
    with nav2:
        st.write("")
    with nav3:
        if st.session_state.step < 6:
            st.button("Next", on_click=next_step)

with summary_col:
    # 3. Render the Persistent Summary Table
    st.subheader("Current Parameters")
    st.write("Review your lattice configuration:")

    # Clean up the dictionary keys for a professional display
    def _summary_val(val):
        if isinstance(val, (list, tuple)):
            return ", ".join(str(x) for x in val)
        return val

    display_dict = {}
    for k, v in st.session_state.params.items():
        if k.startswith("_") or v is None:
            continue
        clean_key = k.replace("_", " ").title()
        display_dict[clean_key] = _summary_val(v)

    st.dataframe([display_dict], use_container_width=True)
