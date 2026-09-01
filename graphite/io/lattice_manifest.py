"""
Output naming and parameter manifests for lattice generation runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from graphite.implicit.density_control import period_mm_from_sizing

# Keys that belong to the explicit engine only (omit from implicit manifests).
_IMPLICIT_EXCLUDE_KEYS = frozenset(
    {
        "engine_type",
        "explicit_topology",
        "explicit_cell_size",
        "explicit_strut_radius",
        "output_name",
        "output_basename",
        "export_parameters",
        "use_custom_output_name",
    }
)

_GRADING_TOKENS = {
    "Uniform": "Uniform",
    "Field Controls": "FieldControls",
    "Variable Porosity (Thickness)": "GradedPorosity",
    "Variable Pore Size (Chirped)": "Chirped",
    "Osteochondral (Layered Z)": "Osteochondral",
    "Boundary-Driven (Dual-EDT)": "BoundaryDualEDT",
}


def _sanitize_token(text: str, *, max_len: int = 48) -> str:
    raw = str(text).strip()
    if not raw:
        return "Unknown"
    out: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", ".", "/"):
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        return "Unknown"
    return token[:max_len]


def mm_filename_token(value: float) -> str:
    """Format mm values for filenames (e.g. 5.0 -> 5p0)."""
    f = float(value)
    if not np.isfinite(f):
        return "nan"
    text = f"{f:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def mesh_extent_tokens_mm(mesh: trimesh.Trimesh) -> str:
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = np.maximum(bounds[1] - bounds[0], 0.0)
    return "x".join(mm_filename_token(float(e)) for e in extents)


def geometry_type_token(params: dict) -> str:
    geom = str(params.get("geometry_type", "Custom STL"))
    if geom == "Primitive":
        shape = _sanitize_token(str(params.get("prim_shape", "Cube")))
        return f"Primitive_{shape}"
    if geom == "Custom STL":
        uploaded = params.get("uploaded_filename")
        if uploaded:
            stem = _sanitize_token(Path(str(uploaded)).stem, max_len=32)
            return f"CustomSTL_{stem}"
        return "CustomSTL"
    return _sanitize_token(geom.replace(" ", ""))


def lattice_type_token(params: dict) -> str:
    return _sanitize_token(str(params.get("lattice_type", "Gyroid")).replace(" ", ""))


def size_token(params: dict) -> str:
    size_mode = str(params.get("size_mode", "Pore Size (mm)"))
    if "Unit Cell" in size_mode:
        return f"UC{mm_filename_token(float(params.get('unit_cell_size', 5.0)))}"
    return f"P{mm_filename_token(float(params.get('pore_size', 5.0)))}"


def density_token(params: dict) -> str:
    mode = str(params.get("density_mode", "solid_fraction")).lower().replace(" ", "_")
    if mode.startswith("wall"):
        return f"WT{mm_filename_token(float(params.get('wall_thickness_mm', 0.5)))}"
    return f"SF{mm_filename_token(float(params.get('solid_fraction', 0.33)))}"


def grading_type_token(params: dict) -> str:
    mode = str(params.get("grading_mode", "Uniform"))
    return _GRADING_TOKENS.get(mode, _sanitize_token(mode.replace(" ", "")))


def build_implicit_output_basename(params: dict, mesh: trimesh.Trimesh) -> str:
    """
    Build filename stem:

    ``[GeometryType]_[LatticeType]_[SizeXxSizeYxSizeZ]_[Pore|UC]_[WT|SF]_[Grading]``
    """
    parts = [
        geometry_type_token(params),
        lattice_type_token(params),
        mesh_extent_tokens_mm(mesh),
        size_token(params),
        density_token(params),
        grading_type_token(params),
    ]
    stem = "_".join(parts)
    if len(stem) > 200:
        stem = stem[:200].rstrip("_")
    return stem


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.floating, np.integer)):
        f = float(value)
        return None if not np.isfinite(f) else f
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append({"position_mm": _serialize_value(item[0]), "value": _serialize_value(item[1])})
            else:
                out.append(_serialize_value(item))
        return out
    return str(value)


def collect_implicit_run_parameters(
    params: dict,
    *,
    mesh: trimesh.Trimesh | None = None,
    output_basename: str | None = None,
) -> dict[str, Any]:
    """
    Collect implicit-lattice parameters for a manifest file.

    Omits explicit-engine defaults and internal UI-only keys.
    """
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "engine": "Implicit (TPMS)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if output_basename:
        manifest["output_basename"] = output_basename
    if mesh is not None:
        bounds = np.asarray(mesh.bounds, dtype=float)
        extents = bounds[1] - bounds[0]
        manifest["geometry_extents_mm"] = {
            "x": float(extents[0]),
            "y": float(extents[1]),
            "z": float(extents[2]),
        }
        manifest["geometry_extents_token"] = mesh_extent_tokens_mm(mesh)

    size_mode = str(params.get("size_mode", "Pore Size (mm)"))
    pore = float(params.get("pore_size", 5.0)) if "Pore" in size_mode else None
    uc = float(params.get("unit_cell_size", 5.0)) if "Unit Cell" in size_mode else None
    sf_for_l = float(params.get("solid_fraction", 0.33))
    try:
        manifest["computed_period_mm"] = float(
            period_mm_from_sizing(
                pore_size_mm=pore,
                unit_cell_size_mm=uc,
                solid_fraction_for_pore_mapping=sf_for_l,
            )
        )
    except Exception:
        pass

    for key in sorted(params.keys()):
        if key.startswith("_"):
            continue
        if key in _IMPLICIT_EXCLUDE_KEYS:
            continue
        if key.startswith("explicit_"):
            continue
        manifest[key] = _serialize_value(params[key])

    return manifest


def format_parameters_text(manifest: dict[str, Any]) -> str:
    """Human-readable parameter listing."""
    lines = [
        "Graphite Implicit Lattice — Run Parameters",
        "=" * 48,
        f"Generated (UTC): {manifest.get('generated_at_utc', '')}",
        "",
    ]
    skip = {"manifest_version", "generated_at_utc"}
    for key in sorted(manifest.keys()):
        if key in skip:
            continue
        value = manifest[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, indent=2, sort_keys=True)
            lines.append(f"{key}:")
            lines.extend(f"  {line}" for line in rendered.splitlines())
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    return "\n".join(lines)


def write_implicit_parameters_manifest(
    path: str | Path,
    params: dict,
    *,
    mesh: trimesh.Trimesh | None = None,
    output_basename: str | None = None,
) -> Path:
    """Write a ``.txt`` manifest of implicit run parameters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = collect_implicit_run_parameters(
        params, mesh=mesh, output_basename=output_basename
    )
    path.write_text(format_parameters_text(manifest), encoding="utf-8")
    return path
