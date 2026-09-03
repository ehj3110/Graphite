import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import trimesh
from graphite.geometry.primitives import generate_primitive
from graphite.geometry.surface_picking import visualize_surfaces
from graphite.implicit.chirped import generate_chirped_lattice
from graphite.implicit.conformal import generate_conformal_lattice
from graphite.implicit.density_control import period_mm_from_sizing
from graphite.implicit.field_driven import (
    COORDINATE_OPTIONS,
    generate_field_driven_lattice,
    generate_field_preview_image,
)
from graphite.implicit.graded import generate_graded_lattice
from graphite.implicit.osteochondral import generate_osteochondral_lattice
from graphite.io.lattice_manifest import build_implicit_output_basename, write_implicit_parameters_manifest
from graphite.io.mesh_export import formats_from_request


st.set_page_config(page_title="Graphite Lattice Engine", layout="wide")

_MAX_CARTESIAN_TABLE_ROWS = 128
_SCALAR_CONTROL_COLUMNS = ["position_mm", "value"]
_CARTESIAN_CONTROL_COLUMNS = [
    "x_position_mm",
    "x_value",
    "y_position_mm",
    "y_value",
    "z_position_mm",
    "z_value",
]


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
        "density_mode": "solid_fraction",
        "solid_fraction": 0.33,
        "wall_thickness_mm": 0.5,
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


def _app_output_dir() -> Path:
    return Path("outputs") / "App outputs" / date.today().isoformat()


def _normalize_output_basename(name: str) -> str:
    path = Path(name)
    if path.suffix.lower() in (".stl", ".step", ".stp"):
        return path.stem
    return name


def _format_mm(value: float) -> str:
    return f"{float(value):.3f}"


def _density_mode(params: dict) -> str:
    mode = str(params.get("density_mode", "solid_fraction")).lower().replace(" ", "_")
    return "wall_thickness" if mode.startswith("wall") else "solid_fraction"


def _is_wall_thickness_mode(params: dict) -> bool:
    return _density_mode(params) == "wall_thickness"


def _base_density_value(params: dict) -> float:
    if _is_wall_thickness_mode(params):
        return float(params.get("wall_thickness_mm", 0.5))
    return float(params.get("solid_fraction", 0.33))


def _step3_period_mm(params: dict) -> float:
    size_mode = params.get("size_mode", "Pore Size (mm)")
    pore = float(params.get("pore_size", 5.0)) if size_mode == "Pore Size (mm)" else None
    uc = float(params.get("unit_cell_size", 5.0)) if size_mode != "Pore Size (mm)" else None
    sf_for_l = float(params.get("solid_fraction", 0.33))
    return period_mm_from_sizing(
        pore_size_mm=pore,
        unit_cell_size_mm=uc,
        solid_fraction_for_pore_mapping=sf_for_l,
    )


def _coerce_finite_float(value: object, fallback: float) -> float:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float(fallback)


def _init_widget_key(key: str, fallback):
    if key not in st.session_state:
        st.session_state[key] = fallback


def _coerce_data_editor_rows(raw: object, *, columns: list[str] | None = None) -> list[dict]:
    """Normalize data_editor return value / widget state to list[dict]."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [dict(r) for r in raw if isinstance(r, dict)]
    if isinstance(raw, pd.DataFrame):
        return [dict(r) for r in raw.to_dict("records")]
    if hasattr(raw, "to_dict"):
        try:
            return [dict(r) for r in raw.to_dict("records")]  # type: ignore[union-attr]
        except Exception:
            pass
    if isinstance(raw, dict):
        if "data" in raw:
            return _coerce_data_editor_rows(raw["data"], columns=columns)
        if raw and all(isinstance(v, dict) for v in raw.values()):
            keys = list(raw.keys())
            try:
                keys = sorted(keys, key=lambda k: int(k))
            except (TypeError, ValueError):
                pass
            return [dict(raw[k]) for k in keys]
    return []


def _publish_table_draft(draft_key: str, rows: list[dict]) -> None:
    st.session_state[draft_key] = [dict(r) for r in rows]


def _invalidate_data_editor_widget(editor_key: str) -> None:
    """Drop widget session state so the next render uses fresh draft rows."""
    if editor_key in st.session_state:
        del st.session_state[editor_key]


def _table_rows_for_data_editor(
    draft_key: str,
    editor_key: str,
    *,
    columns: list[str],
) -> list[dict]:
    """
    Rows passed *into* st.data_editor.

    Always sourced from draft; never pass st.session_state[editor_key] (Streamlit
    overwrites that key with a DataFrame/dict that breaks at large row counts).
    """
    draft_rows = _coerce_data_editor_rows(st.session_state.get(draft_key, []), columns=columns)
    if editor_key in st.session_state:
        widget_rows = _coerce_data_editor_rows(st.session_state[editor_key], columns=columns)
        if widget_rows and len(widget_rows) != len(draft_rows):
            _invalidate_data_editor_widget(editor_key)
    return draft_rows


def _sync_data_editor_to_draft(
    draft_key: str,
    edited_raw: object,
    *,
    columns: list[str] | None = None,
) -> list[dict]:
    rows = _coerce_data_editor_rows(edited_raw, columns=columns)
    _publish_table_draft(draft_key, rows)
    return rows


def _rows_payload_equal(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if set(ra.keys()) != set(rb.keys()):
            return False
        for k in ra.keys():
            va = ra.get(k)
            vb = rb.get(k)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                return False
            try:
                fa = float(va)
                fb = float(vb)
                if np.isnan(fa) and np.isnan(fb):
                    continue
                if not np.isclose(fa, fb, rtol=0.0, atol=1e-12):
                    return False
            except Exception:
                if va != vb:
                    return False
    return True


def _control_points_editor(
    label: str,
    param_key: str,
    default_rows: list[tuple[float, float]],
    *,
    value_label: str,
    min_position: float,
    max_position: float,
) -> list[tuple[float, float]]:
    params = st.session_state.params
    state_key = f"{param_key}_table_rows_committed"
    draft_key = f"{param_key}_table_rows_draft"
    if state_key not in st.session_state:
        seed = params.get(param_key, default_rows)
        rows = [{"position_mm": float(p), "value": float(v)} for p, v in seed]
        if len(rows) < 2:
            rows = [
                {"position_mm": float(min_position), "value": float(default_rows[0][1])},
                {"position_mm": float(max_position), "value": float(default_rows[-1][1])},
            ]
        st.session_state[state_key] = rows
    if draft_key not in st.session_state:
        st.session_state[draft_key] = [dict(r) for r in st.session_state[state_key]]

    editor_key = f"{param_key}_table_editor"
    table_rows = _table_rows_for_data_editor(
        draft_key, editor_key, columns=_SCALAR_CONTROL_COLUMNS
    )

    st.markdown(f"**{label}**")
    st.caption("Edits are staged in the table; click Apply to commit to the lattice.")
    edited = st.data_editor(
        table_rows,
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key,
        column_config={
            "position_mm": st.column_config.NumberColumn("Position (mm)", step=0.1, format="%.3f"),
            "value": st.column_config.NumberColumn(value_label.replace("_", " ").title(), step=0.01, format="%.4f"),
        },
        hide_index=True,
    )
    apply_clicked = st.button(f"Apply {label}", key=f"{param_key}_apply_btn", use_container_width=True)
    current_rows = _sync_data_editor_to_draft(
        draft_key, edited, columns=_SCALAR_CONTROL_COLUMNS
    )

    parsed_valid: list[tuple[float, float]] = []
    parsed_rows: list[dict[str, float | None]] = []
    for row in current_rows:
        try:
            raw_pos = row.get("position_mm")
            raw_val = row.get("value")
            value = _coerce_finite_float(raw_val, float(default_rows[0][1]))
            if raw_pos in (None, ""):
                parsed_rows.append({"position_mm": None, "value": value})
                continue
            pos = float(raw_pos)
            if not np.isfinite(pos):
                parsed_rows.append({"position_mm": None, "value": value})
                continue
            parsed_rows.append({"position_mm": pos, "value": value})
            parsed_valid.append((pos, value))
        except Exception:
            continue
    if len(parsed_valid) < 2:
        parsed_valid = [
            (float(min_position), float(default_rows[0][1])),
            (float(max_position), float(default_rows[-1][1])),
        ]

    parsed_valid.sort(key=lambda item: item[0])
    parsed_valid[0] = (float(min_position), parsed_valid[0][1])
    parsed_valid[-1] = (float(max_position), parsed_valid[-1][1])

    normalized_table = list(parsed_rows)
    if len(normalized_table) < 2:
        normalized_table = [
            {"position_mm": float(min_position), "value": float(default_rows[0][1])},
            {"position_mm": float(max_position), "value": float(default_rows[-1][1])},
        ]
    normalized_table[0]["position_mm"] = float(min_position)
    normalized_table[-1]["position_mm"] = float(max_position)

    prev_table = st.session_state.get(state_key, [])
    wrote_back = False
    if apply_clicked:
        if not _rows_payload_equal(normalized_table, prev_table):
            st.session_state[state_key] = [dict(r) for r in normalized_table]
            wrote_back = True
        _publish_table_draft(draft_key, normalized_table)
        _invalidate_data_editor_widget(editor_key)
    committed_rows = st.session_state.get(state_key, normalized_table)
    committed_pairs: list[tuple[float, float]] = []
    for row in committed_rows:
        try:
            p = _coerce_finite_float(row.get("position_mm"), min_position)
            v = _coerce_finite_float(row.get("value"), default_rows[0][1])
            committed_pairs.append((p, v))
        except Exception:
            continue
    if len(committed_pairs) < 2:
        committed_pairs = [(float(min_position), float(default_rows[0][1])), (float(max_position), float(default_rows[-1][1]))]
    committed_pairs.sort(key=lambda item: item[0])
    committed_pairs[0] = (float(min_position), committed_pairs[0][1])
    committed_pairs[-1] = (float(max_position), committed_pairs[-1][1])
    params[param_key] = committed_pairs
    st.caption("First/last position are auto-locked to min/max coordinate on Apply.")
    return committed_pairs


def _current_geometry_mesh(*, warn: bool = True) -> trimesh.Trimesh | None:
    params = st.session_state.params
    geom_type = params.get("geometry_type", "Custom STL")
    if geom_type == "Primitive":
        return generate_primitive(params.get("prim_shape", "Cube"), float(params.get("prim_size", 20.0)))
    if geom_type == "Custom STL":
        if st.session_state.uploaded_file is None:
            if warn:
                st.warning("Upload a Custom STL in Step 1 before generating a preview.")
            return None
        st.session_state.uploaded_file.seek(0)
        mesh = trimesh.load(st.session_state.uploaded_file, file_type="stl")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        return mesh
    if warn:
        st.warning("Preview is currently available for Custom STL and Primitive geometries.")
    return None


def _mesh_center_of_mass(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        center = np.asarray(mesh.center_mass, dtype=float)
        if np.all(np.isfinite(center)):
            return center
    except Exception:
        pass
    return np.asarray(mesh.centroid, dtype=float)


def _field_origin_for_mesh(mesh: trimesh.Trimesh) -> np.ndarray:
    params = st.session_state.params
    mode = params.get("field_origin_mode", "Center of Mass")
    bounds = np.asarray(mesh.bounds, dtype=float)
    if mode == "Bounding Box Center":
        return (bounds[0] + bounds[1]) * 0.5
    if mode == "Manual":
        default = _mesh_center_of_mass(mesh)
        return np.asarray(
            [
                float(params.get("field_origin_x", default[0])),
                float(params.get("field_origin_y", default[1])),
                float(params.get("field_origin_z", default[2])),
            ],
            dtype=float,
        )
    return _mesh_center_of_mass(mesh)


def _coordinate_position_bounds(coordinate: str) -> tuple[float, float]:
    mesh = _current_geometry_mesh(warn=False)
    if mesh is None:
        fallback = float(st.session_state.params.get("prim_size", 20.0))
        return 0.0, max(fallback, 1e-6)

    vertices = np.asarray(mesh.vertices, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = np.maximum(bounds[1] - bounds[0], 1e-6)
    origin = _field_origin_for_mesh(mesh)

    if coordinate == "Cartesian X":
        return 0.0, float(extents[0])
    if coordinate == "Cartesian Y":
        return 0.0, float(extents[1])
    if coordinate == "Cartesian Z":
        return 0.0, float(extents[2])
    if coordinate == "Cartesian":
        return 0.0, float(np.max(extents))
    if coordinate == "Cylindrical Radius":
        radial = np.sqrt((vertices[:, 0] - origin[0]) ** 2 + (vertices[:, 1] - origin[1]) ** 2)
        return 0.0, float(np.max(radial))
    if coordinate == "Spherical Radius":
        radial = np.linalg.norm(vertices - origin, axis=1)
        return 0.0, float(np.max(radial))
    return 0.0, float(np.max(extents))


def _normalize_coordinate_choice(value: str | None) -> str:
    if value in {"Cartesian X", "Cartesian Y", "Cartesian Z"}:
        return "Cartesian"
    if value in COORDINATE_OPTIONS:
        return str(value)
    return "Cartesian"


def _cartesian_control_points_editor(
    label: str,
    param_prefix: str,
    *,
    base_value: float,
    value_label: str,
) -> dict[str, list[tuple[float, float]]]:
    st.markdown(f"**{label} Cartesian Axes**")
    params = st.session_state.params
    bounds = {
        "x": _coordinate_position_bounds("Cartesian X"),
        "y": _coordinate_position_bounds("Cartesian Y"),
        "z": _coordinate_position_bounds("Cartesian Z"),
    }
    table_key = f"{param_prefix}_cartesian_table_rows_committed"
    draft_key = f"{param_prefix}_cartesian_table_rows_draft"
    existing = params.get(f"{param_prefix}_cartesian_control_points")
    if table_key not in st.session_state:
        if isinstance(existing, dict) and existing:
            max_rows = max(len(existing.get("x", [])), len(existing.get("y", [])), len(existing.get("z", [])), 2)
            seed: list[dict[str, float]] = []
            for i in range(max_rows):
                row: dict[str, float] = {}
                for axis in ("x", "y", "z"):
                    axis_rows = existing.get(axis, [])
                    if i < len(axis_rows):
                        p, v = axis_rows[i]
                    elif i == 0:
                        p, v = bounds[axis][0], float(base_value)
                    elif i == max_rows - 1:
                        p, v = bounds[axis][1], float(base_value)
                    else:
                        p = bounds[axis][0] + (bounds[axis][1] - bounds[axis][0]) * i / max(max_rows - 1, 1)
                        v = float(base_value)
                    row[f"{axis}_position_mm"] = float(p)
                    row[f"{axis}_value"] = float(v)
                seed.append(row)
            st.session_state[table_key] = seed
        else:
            st.session_state[table_key] = [
                {
                    "x_position_mm": float(bounds["x"][0]), "x_value": float(base_value),
                    "y_position_mm": float(bounds["y"][0]), "y_value": float(base_value),
                    "z_position_mm": float(bounds["z"][0]), "z_value": float(base_value),
                },
                {
                    "x_position_mm": float(bounds["x"][1]), "x_value": float(base_value),
                    "y_position_mm": float(bounds["y"][1]), "y_value": float(base_value),
                    "z_position_mm": float(bounds["z"][1]), "z_value": float(base_value),
                },
            ]
    if draft_key not in st.session_state:
        st.session_state[draft_key] = [dict(r) for r in st.session_state[table_key]]

    editor_key = f"{param_prefix}_cartesian_table_editor"
    table_rows = _table_rows_for_data_editor(
        draft_key, editor_key, columns=_CARTESIAN_CONTROL_COLUMNS
    )
    if len(table_rows) > _MAX_CARTESIAN_TABLE_ROWS:
        st.warning(
            f"Cartesian table is limited to {_MAX_CARTESIAN_TABLE_ROWS} rows "
            f"(currently {len(table_rows)}). Remove rows or click Apply before adding more."
        )
        table_rows = table_rows[:_MAX_CARTESIAN_TABLE_ROWS]
        _publish_table_draft(draft_key, table_rows)

    st.caption(
        f"Edits are staged. Use Add/Remove or edit cells, then click Apply to commit. "
        f"Max {_MAX_CARTESIAN_TABLE_ROWS} rows."
    )
    with st.form(f"{param_prefix}_cartesian_table_form", clear_on_submit=False):
        edited = st.data_editor(
            table_rows,
            num_rows="fixed",
            use_container_width=True,
            key=editor_key,
            column_config={
                "x_position_mm": st.column_config.NumberColumn("X Position (mm)", step=0.1, format="%.3f"),
                "x_value": st.column_config.NumberColumn("X Value", step=0.01, format="%.4f"),
                "y_position_mm": st.column_config.NumberColumn("Y Position (mm)", step=0.1, format="%.3f"),
                "y_value": st.column_config.NumberColumn("Y Value", step=0.01, format="%.4f"),
                "z_position_mm": st.column_config.NumberColumn("Z Position (mm)", step=0.1, format="%.3f"),
                "z_value": st.column_config.NumberColumn("Z Value", step=0.01, format="%.4f"),
            },
            hide_index=True,
        )
        row_btn_col, remove_btn_col, apply_btn_col = st.columns(3)
        with row_btn_col:
            add_clicked = st.form_submit_button("Add Cartesian Row", use_container_width=True)
        with remove_btn_col:
            remove_clicked = st.form_submit_button("Remove Last Interior Row", use_container_width=True)
        with apply_btn_col:
            apply_clicked = st.form_submit_button(f"Apply {label}", use_container_width=True)

    current_rows = _sync_data_editor_to_draft(
        draft_key, edited, columns=_CARTESIAN_CONTROL_COLUMNS
    )
    if add_clicked:
        if len(current_rows) >= _MAX_CARTESIAN_TABLE_ROWS:
            st.error(f"Cannot add more than {_MAX_CARTESIAN_TABLE_ROWS} Cartesian rows.")
        else:
            insert_idx = max(len(current_rows) - 1, 1)
            prev_row = current_rows[insert_idx - 1] if insert_idx > 0 else current_rows[0]
            next_row = current_rows[insert_idx] if insert_idx < len(current_rows) else current_rows[-1]
            new_row: dict[str, float | None] = {}
            for axis in ("x", "y", "z"):
                p0 = prev_row.get(f"{axis}_position_mm")
                p1 = next_row.get(f"{axis}_position_mm")
                if p0 is not None and p1 is not None:
                    try:
                        mid = (float(p0) + float(p1)) * 0.5
                    except (TypeError, ValueError):
                        mid = None
                else:
                    mid = None
                v0 = _coerce_finite_float(prev_row.get(f"{axis}_value"), float(base_value))
                v1 = _coerce_finite_float(next_row.get(f"{axis}_value"), float(base_value))
                new_row[f"{axis}_position_mm"] = mid
                new_row[f"{axis}_value"] = (v0 + v1) * 0.5
            current_rows.insert(insert_idx, new_row)
    elif remove_clicked and len(current_rows) > 2:
        current_rows.pop(-2)
    _publish_table_draft(draft_key, current_rows)
    if add_clicked or remove_clicked:
        _invalidate_data_editor_widget(editor_key)

    prev_rows = st.session_state.get(table_key, [])

    parsed_rows: list[dict[str, float | None]] = []
    for ridx, row in enumerate(current_rows):
        try:
            axis_payload: dict[str, float | None] = {}
            for axis in ("x", "y", "z"):
                raw_pos = row.get(f"{axis}_position_mm")
                raw_val = row.get(f"{axis}_value")
                prev_val = base_value
                if ridx < len(prev_rows):
                    prev_val = _coerce_finite_float(
                        prev_rows[ridx].get(f"{axis}_value"),
                        float(base_value),
                    )
                axis_payload[f"{axis}_position_mm"] = (
                    None
                    if raw_pos in (None, "") or not np.isfinite(_coerce_finite_float(raw_pos, np.nan))
                    else float(raw_pos)
                )
                axis_payload[f"{axis}_value"] = _coerce_finite_float(raw_val, prev_val)
            parsed_rows.append(
                axis_payload
            )
        except Exception:
            continue
    if len(parsed_rows) < 2:
        parsed_rows = [
            {
                "x_position_mm": float(bounds["x"][0]), "x_value": float(base_value),
                "y_position_mm": float(bounds["y"][0]), "y_value": float(base_value),
                "z_position_mm": float(bounds["z"][0]), "z_value": float(base_value),
            },
            {
                "x_position_mm": float(bounds["x"][1]), "x_value": float(base_value),
                "y_position_mm": float(bounds["y"][1]), "y_value": float(base_value),
                "z_position_mm": float(bounds["z"][1]), "z_value": float(base_value),
            },
        ]

    rows_by_axis: dict[str, list[tuple[float, float]]] = {"x": [], "y": [], "z": []}
    for axis in ("x", "y", "z"):
        axis_pairs: list[tuple[float, float]] = []
        for r in parsed_rows:
            pos = r.get(f"{axis}_position_mm")
            val = r.get(f"{axis}_value")
            if pos is None:
                continue
            try:
                pos_f = float(pos)
                if not np.isfinite(pos_f):
                    continue
                axis_pairs.append((pos_f, _coerce_finite_float(val, float(base_value))))
            except Exception:
                continue
        if len(axis_pairs) < 2:
            axis_pairs = [
                (float(bounds[axis][0]), float(base_value)),
                (float(bounds[axis][1]), float(base_value)),
            ]
        axis_pairs.sort(key=lambda item: item[0])
        axis_pairs[0] = (float(bounds[axis][0]), axis_pairs[0][1])
        axis_pairs[-1] = (float(bounds[axis][1]), axis_pairs[-1][1])
        rows_by_axis[axis] = axis_pairs

    # Keep user row count/layout; only lock true endpoint rows.
    rebuilt = list(parsed_rows)
    if len(rebuilt) < 2:
        rebuilt = [
            {
                "x_position_mm": float(bounds["x"][0]), "x_value": float(base_value),
                "y_position_mm": float(bounds["y"][0]), "y_value": float(base_value),
                "z_position_mm": float(bounds["z"][0]), "z_value": float(base_value),
            },
            {
                "x_position_mm": float(bounds["x"][1]), "x_value": float(base_value),
                "y_position_mm": float(bounds["y"][1]), "y_value": float(base_value),
                "z_position_mm": float(bounds["z"][1]), "z_value": float(base_value),
            },
        ]
    for axis in ("x", "y", "z"):
        rebuilt[0][f"{axis}_position_mm"] = float(bounds[axis][0])
        rebuilt[-1][f"{axis}_position_mm"] = float(bounds[axis][1])

    prev_table = st.session_state.get(table_key, [])
    wrote_back = False
    if apply_clicked:
        if not _rows_payload_equal(rebuilt, prev_table):
            st.session_state[table_key] = [dict(r) for r in rebuilt]
            wrote_back = True
        _publish_table_draft(draft_key, rebuilt)
        _invalidate_data_editor_widget(editor_key)

    committed_rows = st.session_state.get(table_key, rebuilt)
    committed_by_axis: dict[str, list[tuple[float, float]]] = {"x": [], "y": [], "z": []}
    for axis in ("x", "y", "z"):
        pairs: list[tuple[float, float]] = []
        for r in committed_rows:
            pos = r.get(f"{axis}_position_mm")
            val = r.get(f"{axis}_value")
            try:
                pf = _coerce_finite_float(pos, np.nan)
                if not np.isfinite(pf):
                    continue
                vf = _coerce_finite_float(val, float(base_value))
                pairs.append((pf, vf))
            except Exception:
                continue
        if len(pairs) < 2:
            pairs = [(float(bounds[axis][0]), float(base_value)), (float(bounds[axis][1]), float(base_value))]
        pairs.sort(key=lambda item: item[0])
        pairs[0] = (float(bounds[axis][0]), pairs[0][1])
        pairs[-1] = (float(bounds[axis][1]), pairs[-1][1])
        committed_by_axis[axis] = pairs
    params[f"{param_prefix}_cartesian_control_points"] = committed_by_axis
    st.caption("6-column Cartesian table; first/last positions lock to axis min/max on Apply.")
    return committed_by_axis

def step_1():
    st.title("Step 1: Geometry Selection")
    params = st.session_state.params

    geometry_options = ["Custom STL", "ASTM Standard", "Primitive"]
    geometry_default = params.get("geometry_type", "Custom STL")
    if geometry_default not in geometry_options:
        geometry_default = "Custom STL"
    _init_widget_key("step1_geometry_type_widget", geometry_default)
    geometry_type = st.radio(
        "Geometry Type",
        geometry_options,
        key="step1_geometry_type_widget",
    )
    params["geometry_type"] = geometry_type

    if geometry_type == "Custom STL":
        uploaded_file = st.file_uploader("Upload STL geometry", type=["stl"], key="step1_upload_widget")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            params["uploaded_filename"] = uploaded_file.name
            st.success(f"Loaded: {uploaded_file.name}")

    elif geometry_type == "ASTM Standard":
        astm_options = ["ASTM F42 Coupon", "ASTM Compression Cube", "ASTM Dogbone"]
        _init_widget_key("step1_astm_widget", params.get("astm_standard", astm_options[0]))
        astm_choice = st.selectbox(
            "ASTM Geometry",
            astm_options,
            key="step1_astm_widget",
        )
        params["astm_standard"] = astm_choice

    elif geometry_type == "Primitive":
        # Defaults for rapid iteration: cube, 20 mm
        prim_options = ["Cube", "Sphere", "Cylinder"]
        prim_default = params.get("prim_shape", "Cube")
        if prim_default not in prim_options:
            prim_default = "Cube"
        _init_widget_key("step1_prim_shape_widget", prim_default)
        _init_widget_key("step1_prim_size_widget", float(params.get("prim_size", 20.0)))
        prim_shape = st.selectbox(
            "Primitive Shape",
            prim_options,
            key="step1_prim_shape_widget",
        )
        params["prim_shape"] = prim_shape
        params["prim_size"] = st.number_input(
            "Major Dimension (Size/Diameter) in mm",
            min_value=1.0,
            step=1.0,
            key="step1_prim_size_widget",
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
    _init_widget_key("step2_engine_widget", _current)
    engine_type = st.radio(
        "Lattice Architecture",
        _engine_options,
        key="step2_engine_widget",
    )
    params["engine_type"] = engine_type

    if engine_type == "Implicit (TPMS)":
        implicit_options = ["Gyroid", "Diamond", "Schwarz Diamond", "Schwarz-P", "Neovius", "Split-P"]
        current_lattice = params.get("lattice_type", "Gyroid")
        if current_lattice == "schwarz-diamond":
            current_lattice = "Schwarz Diamond"
        if current_lattice not in implicit_options:
            current_lattice = "Gyroid"
        _init_widget_key("step2_lattice_type_widget", current_lattice)
        params["lattice_type"] = st.selectbox(
            "Lattice Type",
            implicit_options,
            key="step2_lattice_type_widget",
        )
    else:
        topology_options = ["Standard Tet", "Surface Cage / Dual"]
        topology_default = params.get("explicit_topology", "Standard Tet")
        if topology_default not in topology_options:
            topology_default = topology_options[0]
        _init_widget_key("step2_explicit_topology_widget", topology_default)
        _init_widget_key("step2_explicit_cell_size_widget", float(params.get("explicit_cell_size", 5.0)))
        _init_widget_key("step2_explicit_strut_radius_widget", float(params.get("explicit_strut_radius", 0.5)))
        params["explicit_topology"] = st.selectbox(
            "Topology Rule",
            topology_options,
            key="step2_explicit_topology_widget",
        )
        params["explicit_cell_size"] = st.number_input(
            "Target Cell Size (mm)",
            min_value=1.0,
            step=0.5,
            key="step2_explicit_cell_size_widget",
        )
        params["explicit_strut_radius"] = st.number_input(
            "Strut Radius (mm)",
            min_value=0.1,
            step=0.1,
            key="step2_explicit_strut_radius_widget",
        )


def step_3():
    st.title("Step 3: Sizing & Density")
    params = st.session_state.params

    density_options = ["Solid Fraction", "Wall Thickness (mm)"]
    density_default = params.get("density_mode", "solid_fraction")
    if density_default == "wall_thickness":
        density_label = "Wall Thickness (mm)"
    else:
        density_label = "Solid Fraction"
    if density_label not in density_options:
        density_label = density_options[0]
    _init_widget_key("step3_density_mode_widget", density_label)
    density_choice = st.radio(
        "Density Control",
        density_options,
        key="step3_density_mode_widget",
        help=(
            "Solid fraction is a volume-fraction target (optionally calibrated). "
            "Wall thickness sets strut width in mm for a given unit-cell period."
        ),
    )
    params["density_mode"] = (
        "wall_thickness" if density_choice == "Wall Thickness (mm)" else "solid_fraction"
    )

    if _is_wall_thickness_mode(params):
        _init_widget_key("step3_wall_thickness_widget", float(params.get("wall_thickness_mm", 0.5)))
        params["wall_thickness_mm"] = st.number_input(
            "Wall Thickness (mm)",
            min_value=0.05,
            max_value=10.0,
            step=0.05,
            format="%.3f",
            key="step3_wall_thickness_widget",
        )
        st.caption(
            f"At the current cell scale (~{_step3_period_mm(params):.2f} mm period), "
            "wall thickness directly sets strut width; solid fraction is derived at generation time."
        )
    else:
        _init_widget_key("step3_solid_fraction_widget", float(params.get("solid_fraction", 0.33)))
        params["solid_fraction"] = st.number_input(
            "Target Solid Fraction",
            min_value=0.01,
            max_value=0.99,
            step=0.01,
            format="%.2f",
            key="step3_solid_fraction_widget",
        )

    size_options = ["Pore Size (mm)", "Unit Cell Size (mm)"]
    size_default = params.get("size_mode", "Pore Size (mm)")
    if size_default not in size_options:
        size_default = size_options[0]
    _init_widget_key("step3_size_mode_widget", size_default)
    size_mode = st.radio(
        "Sizing Mode",
        size_options,
        key="step3_size_mode_widget",
    )
    params["size_mode"] = size_mode

    if size_mode == "Pore Size (mm)":
        _init_widget_key("step3_pore_size_widget", float(params.get("pore_size", 5.0)))
        params["pore_size"] = st.number_input(
            "Pore Size (mm)",
            min_value=0.1,
            step=0.1,
            key="step3_pore_size_widget",
        )
    else:
        _init_widget_key("step3_unit_cell_size_widget", float(params.get("unit_cell_size", 5.0)))
        params["unit_cell_size"] = st.number_input(
            "Unit Cell Size (mm)",
            min_value=0.1,
            step=0.1,
            key="step3_unit_cell_size_widget",
        )


def step_4():
    st.title("Step 4: Boundary & Field Control")
    params = st.session_state.params

    committed = st.session_state.setdefault(
        "step4_committed",
        {
            "grade_lattice": bool(params.get("grade_lattice", False)),
            "grade_solid_fraction": bool(params.get("grade_solid_fraction", True)),
            "grade_unit_cell": bool(params.get("grade_unit_cell", False)),
            "solid_fraction_coordinate": _normalize_coordinate_choice(params.get("solid_fraction_coordinate", "Cartesian")),
            "unit_cell_coordinate": _normalize_coordinate_choice(params.get("unit_cell_coordinate", "Cartesian")),
            "calibrate_solid_fraction": bool(params.get("calibrate_solid_fraction", True)),
            "field_origin_mode": params.get("field_origin_mode", "Center of Mass"),
        },
    )

    st.session_state.setdefault("step4_grade_lattice_widget", bool(committed.get("grade_lattice", False)))
    grade_lattice = st.checkbox(
        "Grade lattice fields",
        key="step4_grade_lattice_widget",
        help="Enable independent control-point fields for density (solid fraction or wall thickness) and/or unit-cell size.",
    )

    if not grade_lattice:
        committed["grade_lattice"] = False
        params["grade_lattice"] = False
        params["grading_mode"] = "Uniform"
        st.info("Uniform lattice: Step 3 density and size controls will be used.")
        return

    controls_col, preview_col = st.columns([0.62, 0.38], gap="large")

    with controls_col:
        st.markdown("### Field Controls")
        st.caption(
            "Unit-cell size grading always uses integrated phase. Control point endpoints are "
            "locked to the selected coordinate range."
        )

        mesh_for_origin = _current_geometry_mesh(warn=False)
        if mesh_for_origin is not None:
            com = _mesh_center_of_mass(mesh_for_origin)
            bbox_center = np.mean(np.asarray(mesh_for_origin.bounds, dtype=float), axis=0)
            origin_options = ["Center of Mass", "Bounding Box Center", "Manual"]
            current_origin_mode = committed.get("field_origin_mode", "Center of Mass")
            if current_origin_mode not in origin_options:
                current_origin_mode = "Center of Mass"
            st.session_state.setdefault("step4_field_origin_mode_widget", current_origin_mode)
            st.selectbox(
                "Gradient / Phase Origin",
                origin_options,
                index=origin_options.index(st.session_state.get("step4_field_origin_mode_widget", current_origin_mode))
                if st.session_state.get("step4_field_origin_mode_widget", current_origin_mode) in origin_options
                else origin_options.index(current_origin_mode),
                key="step4_field_origin_mode_widget",
                help=(
                    "Anchors radial/cylindrical fields and ungraded TPMS phase axes. "
                    "Center of Mass is the default for balanced samples."
                ),
            )
            active_origin_mode = st.session_state.get("step4_field_origin_mode_widget", current_origin_mode)
            if active_origin_mode == "Manual":
                default_origin = _field_origin_for_mesh(mesh_for_origin)
                params["field_origin_x"] = st.number_input(
                    "Origin X (mm)",
                    value=float(default_origin[0]),
                    step=0.1,
                    format="%.3f",
                )
                params["field_origin_y"] = st.number_input(
                    "Origin Y (mm)",
                    value=float(default_origin[1]),
                    step=0.1,
                    format="%.3f",
                )
                params["field_origin_z"] = st.number_input(
                    "Origin Z (mm)",
                    value=float(default_origin[2]),
                    step=0.1,
                    format="%.3f",
                )
            active_origin = _field_origin_for_mesh(mesh_for_origin)
            st.caption(
                "Active origin: "
                f"({active_origin[0]:.3f}, {active_origin[1]:.3f}, {active_origin[2]:.3f}) mm. "
                f"COM=({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}); "
                f"BBox center=({bbox_center[0]:.3f}, {bbox_center[1]:.3f}, {bbox_center[2]:.3f})."
            )
        else:
            params["field_origin_mode"] = committed.get("field_origin_mode", "Center of Mass")
            st.info("Upload or define geometry to resolve the gradient origin.")

        st.session_state.setdefault(
            "step4_grade_solid_fraction_widget",
            bool(committed.get("grade_solid_fraction", True)),
        )
        density_label = "wall thickness" if _is_wall_thickness_mode(params) else "solid fraction"
        grade_solid_fraction = st.checkbox(
            f"Grade {density_label}",
            key="step4_grade_solid_fraction_widget",
        )
        st.session_state.setdefault(
            "step4_grade_unit_cell_widget",
            bool(committed.get("grade_unit_cell", False)),
        )
        grade_unit_cell = st.checkbox(
            "Grade unit-cell size",
            key="step4_grade_unit_cell_widget",
        )

        committed["grade_lattice"] = bool(grade_lattice)
        committed["grade_solid_fraction"] = bool(grade_solid_fraction)
        committed["grade_unit_cell"] = bool(grade_unit_cell)
        committed["solid_fraction_coordinate"] = _normalize_coordinate_choice(
            st.session_state.get("solid_fraction_coordinate_select", committed.get("solid_fraction_coordinate", "Cartesian"))
        )
        committed["unit_cell_coordinate"] = _normalize_coordinate_choice(
            st.session_state.get("unit_cell_coordinate_select", committed.get("unit_cell_coordinate", "Cartesian"))
        )
        if _is_wall_thickness_mode(params):
            committed["calibrate_solid_fraction"] = False
        else:
            committed["calibrate_solid_fraction"] = bool(
                st.session_state.get(
                    "step4_calibrate_solid_fraction_widget",
                    committed.get("calibrate_solid_fraction", True),
                )
            )
        committed["field_origin_mode"] = st.session_state.get(
            "step4_field_origin_mode_widget", committed.get("field_origin_mode", "Center of Mass")
        )
        params["grade_lattice"] = committed["grade_lattice"]
        params["grade_solid_fraction"] = committed["grade_solid_fraction"]
        params["grade_unit_cell"] = committed["grade_unit_cell"]
        params["solid_fraction_coordinate"] = committed["solid_fraction_coordinate"]
        params["unit_cell_coordinate"] = committed["unit_cell_coordinate"]
        params["calibrate_solid_fraction"] = committed["calibrate_solid_fraction"]
        params["field_origin_mode"] = committed["field_origin_mode"]
        params["grading_mode"] = "Field Controls" if committed["grade_lattice"] else "Uniform"

        if grade_solid_fraction:
            wall_mode = _is_wall_thickness_mode(params)
            field_title = "Wall Thickness Field" if wall_mode else "Solid Fraction Field"
            with st.expander(field_title, expanded=True):
                current = _normalize_coordinate_choice(committed.get("solid_fraction_coordinate", "Cartesian"))
                coord_label = "Density coordinate" if wall_mode else "Solid-fraction coordinate"
                st.selectbox(
                    coord_label,
                    list(COORDINATE_OPTIONS),
                    index=list(COORDINATE_OPTIONS).index(current),
                    key="solid_fraction_coordinate_select",
                )
                active_sf_coordinate = _normalize_coordinate_choice(st.session_state.get("solid_fraction_coordinate_select", current))
                base_density = _base_density_value(params)
                value_label = "wall_thickness_mm" if wall_mode else "solid_fraction"
                if active_sf_coordinate == "Cartesian":
                    _cartesian_control_points_editor(
                        "Wall Thickness" if wall_mode else "Solid Fraction",
                        "solid_fraction",
                        base_value=base_density,
                        value_label=value_label,
                    )
                else:
                    sf_min, sf_max = _coordinate_position_bounds(active_sf_coordinate)
                    default_sf = [
                        (sf_min, base_density),
                        (sf_max, base_density),
                    ]
                    _control_points_editor(
                        "Wall Thickness" if wall_mode else "Solid Fraction",
                        "solid_fraction_control_points",
                        default_sf,
                        value_label=value_label,
                        min_position=sf_min,
                        max_position=sf_max,
                    )
                if not wall_mode:
                    st.session_state.setdefault(
                        "step4_calibrate_solid_fraction_widget",
                        bool(committed.get("calibrate_solid_fraction", True)),
                    )
                    st.checkbox(
                        "Calibrate target solid fraction to TPMS tau",
                        key="step4_calibrate_solid_fraction_widget",
                        help="Uses the TPMS phase distribution to map measured target SF to threshold tau.",
                    )
                else:
                    params["calibrate_solid_fraction"] = False
                    st.caption("Wall thickness maps directly to TPMS threshold using local unit-cell period.")

        if grade_unit_cell:
            with st.expander("Unit-Cell Size Field", expanded=True):
                current = _normalize_coordinate_choice(committed.get("unit_cell_coordinate", "Cartesian"))
                st.selectbox(
                    "Unit-cell coordinate",
                    list(COORDINATE_OPTIONS),
                    index=list(COORDINATE_OPTIONS).index(current),
                    key="unit_cell_coordinate_select",
                )
                active_uc_coordinate = _normalize_coordinate_choice(st.session_state.get("unit_cell_coordinate_select", current))
                base_uc = float(params.get("unit_cell_size", params.get("pore_size", 5.0)))
                if active_uc_coordinate == "Cartesian":
                    _cartesian_control_points_editor(
                        "Unit Cell",
                        "unit_cell",
                        base_value=base_uc,
                        value_label="unit_cell_mm",
                    )
                else:
                    uc_min, uc_max = _coordinate_position_bounds(active_uc_coordinate)
                    default_uc = [(uc_min, base_uc), (uc_max, base_uc)]
                    _control_points_editor(
                        "Unit Cell",
                        "unit_cell_control_points",
                        default_uc,
                        value_label="unit_cell_mm",
                        min_position=uc_min,
                        max_position=uc_max,
                    )
                st.info("Integrated phase is the only production method for unit-cell grading.")

    with preview_col:
        st.markdown("### Preview")
        st.caption("Target preview pixel size: 10 microns (0.01 mm). Large slices are capped for responsiveness.")
        if st.button("Update Preview", type="secondary", use_container_width=True):
            mesh = _current_geometry_mesh()
            if mesh is not None:
                with st.spinner("Rendering center-slice lattice preview..."):
                    base_unit_cell = _step3_period_mm(params)
                    field_origin = _field_origin_for_mesh(mesh)
                    image, metadata = generate_field_preview_image(
                        mesh,
                        lattice_type=params["lattice_type"],
                        preview_resolution=0.01,
                        base_unit_cell_size=base_unit_cell,
                        base_solid_fraction=float(params.get("solid_fraction", 0.33)),
                        grade_unit_cell=bool(params.get("grade_unit_cell", False)),
                        unit_cell_coordinate=params.get("unit_cell_coordinate", "Cartesian"),
                        unit_cell_control_points=params.get("unit_cell_control_points"),
                        unit_cell_cartesian_control_points=params.get(
                            "unit_cell_cartesian_control_points"
                        ),
                        grade_solid_fraction=bool(params.get("grade_solid_fraction", False)),
                        solid_fraction_coordinate=params.get("solid_fraction_coordinate", "Cartesian"),
                        solid_fraction_control_points=params.get("solid_fraction_control_points"),
                        solid_fraction_cartesian_control_points=params.get(
                            "solid_fraction_cartesian_control_points"
                        ),
                        calibrate_solid_fraction=bool(params.get("calibrate_solid_fraction", True)),
                        density_mode=_density_mode(params),
                        base_wall_thickness_mm=float(params.get("wall_thickness_mm", 0.5)),
                        field_origin=field_origin,
                    )
                    st.session_state.field_preview_image = image
                    st.session_state.field_preview_metadata = metadata

        if "field_preview_image" in st.session_state:
            metadata = st.session_state.get("field_preview_metadata", {})
            caption = (
                f"Center slice: {metadata.get('slice_axis', '?')}="
                f"{metadata.get('slice_position_mm', 0.0):.2f} mm, "
                f"pixel={metadata.get('pixel_size_mm', 0.0):.3f} mm"
            )
            st.image(st.session_state.field_preview_image, caption=caption, clamp=True)
        else:
            st.info("Click Update Preview to render the current center slice.")



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
    engine_type = params.get("engine_type", "Implicit (TPMS)")
    if engine_type == "Implicit (TPMS)":
        _init_widget_key("step6_export_format_widget", params.get("export_format", "STL only"))
        params["export_format"] = st.radio(
            "Output format",
            options=["STL only", "STEP only", "STL + STEP"],
            key="step6_export_format_widget",
            help=(
                "STEP is a faceted tessellated solid (Gmsh). It may take longer and "
                "produce large files at fine voxel resolution."
            ),
        )
        preview_mesh = _current_geometry_mesh(warn=False)
        if preview_mesh is not None:
            auto_basename = build_implicit_output_basename(params, preview_mesh)
        else:
            auto_basename = "(set geometry in Step 1 to preview name)"
        st.caption(
            "Auto file name: "
            f"`{auto_basename}` + `.stl` / `.step` "
            "(Geometry_Lattice_SizeXxSizeYxSizeZ_PoreOrUC_WTorSF_Grading)"
        )
        params["use_custom_output_name"] = st.checkbox(
            "Use custom base name instead of auto naming",
            value=bool(params.get("use_custom_output_name", False)),
        )
        if params["use_custom_output_name"]:
            default_name = params.get("output_name", "Generated_Lattice")
            if str(default_name).lower().endswith((".stl", ".step", ".stp", ".txt")):
                default_name = Path(default_name).stem
            params["output_name"] = st.text_input(
                "Custom output base name",
                value=default_name,
            )
        params["export_parameters"] = st.checkbox(
            "Export parameter manifest (.txt)",
            value=bool(params.get("export_parameters", True)),
            help="Writes a text file listing all implicit lattice settings used for this run "
            "(explicit engine fields are omitted).",
        )
        export_formats = formats_from_request(params["export_format"])
        if preview_mesh is not None and not params.get("use_custom_output_name"):
            preview_stem = build_implicit_output_basename(params, preview_mesh)
        elif params.get("use_custom_output_name"):
            preview_stem = _normalize_output_basename(params.get("output_name", "Generated_Lattice"))
        else:
            preview_stem = "…"
        preview_parts = [
            str(Path.cwd() / _app_output_dir() / f"{preview_stem}.{ext}")
            for ext in ("stl", "step")
            if ext in export_formats
        ]
        if params.get("export_parameters"):
            preview_parts.append(str(Path.cwd() / _app_output_dir() / f"{preview_stem}.txt"))
        st.caption("Output preview: " + ", ".join(preview_parts))
    else:
        params["export_format"] = "STL only"
        params["output_name"] = st.text_input(
            "Output File Name",
            value=params.get("output_name", "Generated_Scaffold.stl"),
        )
        export_formats = ("stl",)
        output_preview = str(
            Path.cwd() / _app_output_dir() / _normalize_output_basename(params["output_name"])
        )
        st.caption(f"Output preview: {output_preview}.stl")

    params["center_origin"] = st.checkbox(
        "Center Lattice at Origin (0,0,0)",
        value=bool(params.get("center_origin", True)),
    )

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

                if engine_type == "Implicit (TPMS)":
                    if params.get("use_custom_output_name"):
                        output_basename = _normalize_output_basename(
                            params.get("output_name", "Generated_Lattice")
                        )
                    else:
                        output_basename = build_implicit_output_basename(params, base_mesh)
                    params["output_basename"] = output_basename
                else:
                    output_basename = _normalize_output_basename(
                        params.get("output_name", "Generated_Scaffold")
                    )
                output_path = _app_output_dir() / output_basename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                implicit_export_formats = (
                    formats_from_request(params.get("export_format", "STL only"))
                    if engine_type == "Implicit (TPMS)"
                    else None
                )

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

                    explicit_mesh.export(str(output_path.with_suffix(".stl")))

                elif engine_type == "Implicit (TPMS)":
                    grading_mode = params.get("grading_mode", "Uniform")
                    if grading_mode == "Field Controls":
                        base_unit_cell = _step3_period_mm(params)
                        generate_field_driven_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            resolution=params["resolution"],
                            base_unit_cell_size=base_unit_cell,
                            base_solid_fraction=float(params.get("solid_fraction", 0.33)),
                            grade_unit_cell=bool(params.get("grade_unit_cell", False)),
                            unit_cell_coordinate=params.get(
                                "unit_cell_coordinate", "Cartesian"
                            ),
                            unit_cell_control_points=params.get(
                                "unit_cell_control_points"
                            ),
                            unit_cell_cartesian_control_points=params.get(
                                "unit_cell_cartesian_control_points"
                            ),
                            grade_solid_fraction=bool(
                                params.get("grade_solid_fraction", False)
                            ),
                            solid_fraction_coordinate=params.get(
                                "solid_fraction_coordinate", "Cartesian"
                            ),
                            solid_fraction_control_points=params.get(
                                "solid_fraction_control_points"
                            ),
                            solid_fraction_cartesian_control_points=params.get(
                                "solid_fraction_cartesian_control_points"
                            ),
                            calibrate_solid_fraction=bool(
                                params.get("calibrate_solid_fraction", True)
                            ),
                            density_mode=_density_mode(params),
                            base_wall_thickness_mm=float(params.get("wall_thickness_mm", 0.5)),
                            field_origin=_field_origin_for_mesh(base_mesh),
                            export_mode=params["export_mode"],
                            shell_thickness=params["shell_thickness"],
                            center_origin=params["center_origin"],
                            output_path=output_path,
                            export_formats=implicit_export_formats,
                        )
                    elif grading_mode == "Uniform":
                        size_mode = params.get("size_mode", "Pore Size (mm)")
                        pore = float(params["pore_size"]) if size_mode == "Pore Size (mm)" else None
                        uc = (
                            float(params["unit_cell_size"])
                            if size_mode == "Unit Cell Size (mm)"
                            else None
                        )
                        wall_mm = (
                            float(params["wall_thickness_mm"])
                            if _is_wall_thickness_mode(params)
                            else None
                        )
                        generate_conformal_lattice(
                            stl_path=temp_input_path,
                            lattice_type=params["lattice_type"],
                            resolution=params["resolution"],
                            pore_size=pore,
                            unit_cell_size=uc,
                            solid_fraction=float(params.get("solid_fraction", 0.33)),
                            wall_thickness_mm=wall_mm,
                            export_mode=params["export_mode"],
                            shell_thickness=params["shell_thickness"],
                            center_origin=params["center_origin"],
                            output_path=output_path,
                            export_formats=implicit_export_formats,
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
                            export_formats=implicit_export_formats,
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
                            export_formats=implicit_export_formats,
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
                            export_formats=implicit_export_formats,
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
                            export_formats=implicit_export_formats,
                        )
                    else:
                        raise ValueError(f"Unknown grading mode: {grading_mode}")
                else:
                    raise ValueError(f"Unknown lattice architecture: {engine_type}")

            if engine_type == "Explicit - Fast (Delaunay)":
                written_paths = [output_path.with_suffix(".stl")]
            else:
                written_paths = [
                    output_path.parent / f"{output_basename}.{ext}"
                    for ext in implicit_export_formats
                ]
                if params.get("export_parameters"):
                    manifest_path = write_implicit_parameters_manifest(
                        output_path.parent / f"{output_basename}.txt",
                        params,
                        mesh=base_mesh,
                        output_basename=output_basename,
                    )
                    written_paths.append(manifest_path)

            st.success(
                "Success! Lattice saved to: "
                + ", ".join(str(p) for p in written_paths)
            )

            dl_cols = st.columns(min(len(written_paths), 3))
            for idx, path in enumerate(written_paths):
                if not path.exists():
                    continue
                with dl_cols[idx % len(dl_cols)]:
                    with open(path, "rb") as file:
                        suffix = path.suffix.lower()
                        if suffix in (".step", ".stp"):
                            mime = "application/step"
                            label = "Download STEP"
                        elif suffix == ".txt":
                            mime = "text/plain"
                            label = "Download parameters (.txt)"
                        else:
                            mime = "model/stl"
                            label = "Download STL"
                        st.download_button(
                            label=label,
                            data=file.read(),
                            file_name=path.name,
                            mime=mime,
                            key=f"download_{path.name}",
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
