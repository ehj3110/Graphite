"""Pure-Python state model, closed-form sizing solver, CAD health triage, recipe I/O,
and backend implicit-lattice execution for Graphite Implicit Studio.

This module intentionally has zero Qt or PyVista dependencies so it can be
tested and driven headlessly as well as from the PySide6 desktop GUI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import trimesh

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.geometry.primitives import generate_primitive
from graphite.implicit.blending import blend_lattice_with_skin, blend_multi_knot_tpms_sdfs
from graphite.implicit.calibration import (
    CalibrationConfig,
    calibrate_period_for_pore_at_sf,
    calibrate_tau_at_fixed_period,
    calibrate_tpms_point,
    sample_abs_tpms_phase_distribution,
    tau_from_solid_fraction_quantile,
)
from graphite.implicit.chirped import generate_chirped_lattice
from graphite.implicit.conformal import generate_conformal_lattice
from graphite.implicit.density_control import (
    period_mm_from_sizing,
    tau_from_wall_thickness_mm,
)
from graphite.implicit.field_driven import (
    build_integrated_phase_coordinates,
    generate_field_driven_lattice,
    interpolate_profile,
)
from graphite.implicit.graded import generate_graded_lattice
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
from graphite.implicit.piecewise_bands import (
    cumulative_phase_w_from_l_profile,
    splitp_piecewise_box_single_pass,
    splitp_piecewise_cylinder_single_pass,
)
from graphite.implicit.pore_metrics import compute_max_inscribed_sphere_pore_size
from graphite.implicit.surface_textures import SurfaceTextureConfig, apply_surface_texture
from graphite.implicit.tpms_parameter_lut import lookup_tpms_parameters
from graphite.implicit.uniform_woodpile import generate_uniform_woodpile
from graphite.implicit.woodpile_input import WoodpileLatticeSpec, build_piecewise_woodpile_mesh
from graphite.io.lattice_manifest import (
    build_implicit_output_basename,
    collect_implicit_run_parameters,
    write_implicit_parameters_manifest,
)
from graphite.io.mesh_export import compute_mesh_health, export_mesh
from graphite.math.textures import (
    bump_field,
    knurl_field,
    microgroove_field,
    spinodal_spectral_field,
)
from graphite.math.tpms import evaluate_tpms, evaluate_tpms_phase
from graphite.math.woodpile import evaluate_woodpile
from graphite.repair.repair_suite import repair_trimesh_gentle

TPMS_ARCHITECTURES: tuple[str, ...] = (
    "Gyroid",
    "Diamond",
    "Schwarz-P",
    "Schwarz-Diamond",
    "Neovius",
    "Lidinoid",
    "Split-P",
)
ALL_ARCHITECTURES: tuple[str, ...] = TPMS_ARCHITECTURES + ("Woodpile",)
PRIMITIVE_SHAPES: tuple[str, ...] = ("Cube", "Sphere", "Cylinder", "Torus", "Tube")
EXPORT_MODES: tuple[str, ...] = ("core", "skin", "combined")
BLEND_METHODS: tuple[str, ...] = ("polynomial", "exponential", "chamfer")
PREVIEW_MODES: tuple[str, ...] = ("2D Surface (Ultra-Sharp)", "3D Flying Edges (Coarse)")
FOUR_WAY_METRICS: tuple[tuple[str, str], ...] = (
    ("unit_cell", "Unit Cell L"),
    ("pore_size", "Pore Size (MIS)"),
    ("solid_fraction", "Solid Fraction"),
    ("wall_thickness", "Wall Thickness"),
)
GRADING_MODES: tuple[str, ...] = (
    "Uniform",
    "Linear / Radial Ramp",
    "Multi-Knot Control Table",
    "Piecewise Discrete Bands",
)
MULTI_KNOT_ENGINES: tuple[str, ...] = (
    "Normalized SDF Blend",
    "Jacobian Phase Integration",
)
COORDINATE_AXES: tuple[str, ...] = (
    "Z",
    "X",
    "Y",
    "Cylindrical Radius",
    "Spherical Radius",
)
BLENDING_KERNELS: tuple[str, ...] = ("smoothstep", "smootherstep", "linear")
RAMP_AXES: tuple[str, ...] = ("Z", "X", "Y", "Radial")
RAMP_TARGETS: tuple[str, ...] = ("Solid Fraction", "Cell Size (Chirp)", "Both")
TEXTURE_TYPES: tuple[str, ...] = ("microgrooves", "bumps", "knurling", "spinodal")
TEXTURE_PROFILES: tuple[str, ...] = ("sine", "triangle", "square")
TEXTURE_DISPLACEMENT_MODES: tuple[str, ...] = ("centered", "emboss", "engrave")
PILLAR_DISTRIBUTIONS: tuple[str, ...] = ("poisson_disk", "random")
PILLAR_LOCATIONS: tuple[str, ...] = ("all", "internal_only", "outer_only")

GRADING_PRESETS: tuple[str, ...] = (
    "Osteochondral Z-Loft (3-Zone)",
    "Cube Thirds Stepped",
    "Radial Vascular Scaffold",
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "App outputs"

_TPMS_TAU_CACHE: dict[str, np.ndarray] = {}


def _canonical_tpms_name(name: str) -> str:
    key = str(name).strip().lower()
    mapping = {
        "gyroid": "Gyroid",
        "diamond": "Diamond",
        "schwarz-p": "Schwarz-P",
        "schwarz_p": "Schwarz-P",
        "schwarzp": "Schwarz-P",
        "schwarz-diamond": "Schwarz-Diamond",
        "schwarz_diamond": "Schwarz-Diamond",
        "neovius": "Neovius",
        "lidinoid": "Lidinoid",
        "split-p": "Split-P",
        "split_p": "Split-P",
        "splitp": "Split-P",
        "woodpile": "Woodpile",
    }
    if key in mapping:
        return mapping[key]
    return name if name in TPMS_ARCHITECTURES else "Gyroid"


def _abs_phase_samples(lattice_type: str) -> np.ndarray | None:
    canon = _canonical_tpms_name(lattice_type)
    if canon == "Woodpile":
        return None
    if canon not in _TPMS_TAU_CACHE:
        try:
            _TPMS_TAU_CACHE[canon] = sample_abs_tpms_phase_distribution(canon)
        except Exception:
            return None
    return _TPMS_TAU_CACHE[canon]


def tau_from_solid_fraction_fast(lattice_type: str, solid_fraction: float) -> float:
    """Fast calibrated/quantile mapping from sheet solid fraction to threshold tau."""
    sf = float(np.clip(solid_fraction, 0.02, 0.95))
    samples = _abs_phase_samples(lattice_type)
    if samples is not None:
        return float(tau_from_solid_fraction_quantile(samples, sf))
    return float(1.5 * sf)


def solid_fraction_from_tau_fast(lattice_type: str, tau: float) -> float:
    """Inverse mapping from threshold tau to estimated sheet solid fraction."""
    samples = _abs_phase_samples(lattice_type)
    t = float(max(tau, 1e-5))
    if samples is not None and len(samples) > 0:
        sf = float(np.searchsorted(samples, t, side="right") / len(samples))
        return float(np.clip(sf, 0.02, 0.95))
    return float(np.clip(t / 1.5, 0.02, 0.95))


def wall_thickness_from_tau_L(tau: float, period_L_mm: float) -> float:
    """Approximate sheet wall thickness in mm from tau and unit cell period L."""
    L = max(float(period_L_mm), 0.05)
    t = max(float(tau), 1e-4)
    # Match inverse of tau_from_wall_thickness_mm: tau ~ (pi * w / L) * 1.15
    # Using exact ratio from calibration probe so SF / w stay consistent.
    ref = float(np.asarray(tau_from_wall_thickness_mm(0.5, L)).ravel()[0])
    scale = ref / 0.5 if ref > 1e-9 else (np.pi / L)
    return float(np.clip(t / scale, 0.01, 0.9 * L))


def recommended_nyquist_resolution_mm(
    wall_thickness_mm: float,
    unit_cell_size_mm: float | None = None,
) -> float:
    """Compute fine voxel pitch resolving >= 3 voxels across wall & >= 16 per period."""
    w = max(float(wall_thickness_mm), 0.03)
    target = min(0.200, w / 3.0)
    if unit_cell_size_mm is not None and unit_cell_size_mm > 0:
        target = min(target, float(unit_cell_size_mm) / 16.0)
    return float(np.clip(target, 0.015, 0.200))


def _finish_four_way_result(L: float, pore: float, sf: float, w: float, tau: float) -> dict[str, float]:
    return {
        "unit_cell_size_mm": round(float(L), 4),
        "pore_size_mm": round(float(pore), 4),
        "solid_fraction": round(float(sf), 4),
        "wall_thickness_mm": round(float(w), 4),
        "tau": round(float(tau), 4),
        "recommended_resolution_mm": round(recommended_nyquist_resolution_mm(w, L), 4),
    }


def solve_four_way_pair(
    lattice_type: str,
    driver_a: str,
    driver_b: str,
    *,
    unit_cell_size_mm: float,
    pore_size_mm: float,
    solid_fraction: float,
    wall_thickness_mm: float,
) -> dict[str, float]:
    """Solve all 4 coupled sizing metrics given any distinct pair of driving parameters."""
    canon = _canonical_tpms_name(lattice_type)
    sf = float(np.clip(solid_fraction, 0.05, 0.90))
    L = max(float(unit_cell_size_mm), 0.1)
    pore = max(float(pore_size_mm), 0.05)
    w = max(float(wall_thickness_mm), 0.02)

    pair = frozenset((str(driver_a).strip(), str(driver_b).strip()))
    if len(pair) != 2 or not pair.issubset({"unit_cell", "pore_size", "solid_fraction", "wall_thickness"}):
        raise ValueError(f"Expected 2 distinct 4-way drivers, got ({driver_a!r}, {driver_b!r})")

    if canon == "Woodpile":
        if pair == {"unit_cell", "solid_fraction"}:
            w = float(np.clip(sf * L, 0.02, 0.9 * L))
            pore = max(L - w, 0.05)
        elif pair == {"pore_size", "solid_fraction"}:
            L = pore / max(1.0 - sf, 0.10)
            w = max(L - pore, 0.02)
        elif pair == {"unit_cell", "wall_thickness"}:
            w = float(np.clip(w, 0.02, 0.8 * L))
            sf = float(np.clip(w / L, 0.05, 0.90))
            pore = max(L - w, 0.05)
        elif pair == {"pore_size", "wall_thickness"}:
            L = pore + w
            sf = float(np.clip(w / L, 0.05, 0.90))
        elif pair == {"unit_cell", "pore_size"}:
            w = float(np.clip(L - pore, 0.02, 0.9 * L))
            sf = float(np.clip(w / L, 0.05, 0.90))
        else:  # {"solid_fraction", "wall_thickness"}
            L = max(w / sf, 0.1)
            pore = max(L - w, 0.05)
        tau_arr = tau_from_wall_thickness_mm(w, L)
        tau = float(np.asarray(tau_arr).ravel()[0])
        return _finish_four_way_result(L, pore, sf, w, tau)

    if pair == {"unit_cell", "solid_fraction"}:
        tau = tau_from_solid_fraction_fast(canon, sf)
        w = wall_thickness_from_tau_L(tau, L)
        pore = max(L * max(1.0 - sf, 0.10), 0.05)
    elif pair == {"pore_size", "solid_fraction"}:
        L = float(
            period_mm_from_sizing(
                pore_size_mm=pore,
                unit_cell_size_mm=None,
                solid_fraction_for_pore_mapping=sf,
            )
        )
        tau = tau_from_solid_fraction_fast(canon, sf)
        w = wall_thickness_from_tau_L(tau, L)
    elif pair == {"unit_cell", "wall_thickness"}:
        w = float(np.clip(w, 0.02, 0.45 * L))
        tau = float(np.asarray(tau_from_wall_thickness_mm(w, L)).ravel()[0])
        sf = solid_fraction_from_tau_fast(canon, tau)
        pore = max(L * max(1.0 - sf, 0.10), 0.05)
    elif pair == {"pore_size", "wall_thickness"}:
        L = max(pore + w, 0.15)
        for _ in range(3):
            tau = float(np.asarray(tau_from_wall_thickness_mm(w, L)).ravel()[0])
            sf = solid_fraction_from_tau_fast(canon, tau)
            L = float(
                period_mm_from_sizing(
                    pore_size_mm=pore,
                    unit_cell_size_mm=None,
                    solid_fraction_for_pore_mapping=sf,
                )
            )
        tau = float(np.asarray(tau_from_wall_thickness_mm(w, L)).ravel()[0])
        sf = solid_fraction_from_tau_fast(canon, tau)
    elif pair == {"unit_cell", "pore_size"}:
        sf = float(np.clip((1.0 - pore / L) / 1.15, 0.05, 0.90))
        tau = tau_from_solid_fraction_fast(canon, sf)
        w = wall_thickness_from_tau_L(tau, L)
    else:  # {"solid_fraction", "wall_thickness"}
        tau = tau_from_solid_fraction_fast(canon, sf)
        L = max(w * float(np.pi) / max(tau, 1e-5), 0.15)
        pore = max(L * max(1.0 - sf, 0.10), 0.05)

    return _finish_four_way_result(L, pore, sf, w, tau)


def compute_four_way_metrics(
    lattice_type: str,
    *,
    driven_by: str,
    unit_cell_size_mm: float,
    pore_size_mm: float,
    solid_fraction: float,
    wall_thickness_mm: float,
    size_driver: str = "pore_size",
) -> dict[str, float]:
    """Solve coupled 4-way implicit TPMS sizing metrics analytically.

    Returns ``unit_cell_size_mm``, ``pore_size_mm``, ``solid_fraction``,
    ``wall_thickness_mm``, ``tau``, and ``recommended_resolution_mm``.
    """
    canon = _canonical_tpms_name(lattice_type)
    sf = float(np.clip(solid_fraction, 0.05, 0.90))
    L = max(float(unit_cell_size_mm), 0.1)
    pore = max(float(pore_size_mm), 0.05)
    w = max(float(wall_thickness_mm), 0.02)

    if canon == "Woodpile":
        if driven_by == "unit_cell":
            pore = max(L - w, 0.05)
        elif driven_by == "pore_size":
            L = pore + w
        elif driven_by == "wall_thickness":
            w = float(np.clip(w, 0.02, 0.8 * L))
            sf = float(np.clip(w / L, 0.05, 0.90))
            pore = max(L - w, 0.05)
        else:
            return solve_four_way_pair(
                canon,
                "solid_fraction",
                "unit_cell" if size_driver == "unit_cell" else "pore_size",
                unit_cell_size_mm=L,
                pore_size_mm=pore,
                solid_fraction=sf,
                wall_thickness_mm=w,
            )
        tau = float(np.asarray(tau_from_wall_thickness_mm(w, L)).ravel()[0])
        return _finish_four_way_result(L, pore, sf, w, tau)

    if driven_by == "pore_size":
        pair = ("pore_size", "solid_fraction")
    elif driven_by == "unit_cell":
        pair = ("unit_cell", "solid_fraction")
    elif driven_by == "wall_thickness":
        pair = ("wall_thickness", "pore_size" if size_driver == "pore_size" else "unit_cell")
    else:
        pair = ("solid_fraction", "unit_cell" if size_driver == "unit_cell" else "pore_size")

    return solve_four_way_pair(
        canon,
        pair[0],
        pair[1],
        unit_cell_size_mm=L,
        pore_size_mm=pore,
        solid_fraction=sf,
        wall_thickness_mm=w,
    )


def _default_multi_knots() -> list[dict[str, float]]:
    return [
        {"pos_mm": -5.0, "L_mm": 6.0, "w_mm": 0.6, "sf": 0.30},
        {"pos_mm": 0.0, "L_mm": 4.5, "w_mm": 0.55, "sf": 0.36},
        {"pos_mm": 5.0, "L_mm": 3.0, "w_mm": 0.45, "sf": 0.45},
    ]


def _effective_driver_set(state: "ImplicitStudioState") -> set[str]:
    d1 = str(state.size_driver or "").strip()
    d2 = str(state.density_driver or "").strip()
    if not d1 or not d2 or d1 == d2:
        return {"unit_cell", "solid_fraction"}
    return {d1, d2}


@dataclass
class ImplicitStudioState:
    """Full serializable configuration for Graphite Implicit Studio."""

    # Geometry
    geometry_source: str = "Primitive"
    primitive_shape: str = "Cube"
    prim_x_mm: float = 15.0
    prim_y_mm: float = 15.0
    prim_z_mm: float = 15.0
    cad_file_path: str = ""

    # Architecture & Shelling
    lattice_type: str = "Gyroid"
    woodpile_true_mode: bool = True
    export_mode: str = "core"
    shell_thickness_mm: float = 1.0
    blend_radius_mm: float = 0.0
    blend_method: str = "polynomial"
    exact_cad_trim: bool = True
    trim_dilation_mm: float = 0.0
    center_origin: bool = False
    rotation_x_deg: float = 0.0
    rotation_y_deg: float = 0.0
    rotation_z_deg: float = 0.0

    # 4-Way Sizing & Resolution
    size_driver: str = ""
    density_driver: str = ""
    pore_size_mm: float = 0.80
    unit_cell_size_mm: float = 1.50
    solid_fraction: float = 0.33
    wall_thickness_mm: float = 0.50
    auto_resolution: bool = True
    resolution_mm: float = 0.0938

    # Unified Grading Mode
    grading_mode: str = "Uniform"
    ramp_target: str = "Solid Fraction"
    ramp_axis: str = "Z"
    ramp_sf_min: float = 0.15
    ramp_sf_max: float = 0.50
    ramp_L_min_mm: float = 2.5
    ramp_L_max_mm: float = 6.0
    ramp_transition_width_mm: float = 5.0

    # Multi-Knot Table Mode
    multi_knot_engine: str = "Normalized SDF Blend"
    multi_knot_axis: str = "Z"
    multi_knot_kernel: str = "smoothstep"
    multi_knot_calibrate: bool = True
    multi_knot_knots: list[dict[str, float]] = field(default_factory=_default_multi_knots)

    # Piecewise Discrete Bands Mode
    piecewise_breaks_mm: list[float] = field(default_factory=lambda: [-2.5, 2.5])
    piecewise_L_mm: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.5])
    piecewise_w_mm: list[float] = field(default_factory=lambda: [0.40, 0.50, 0.65])
    piecewise_sf: list[float] = field(default_factory=lambda: [0.45, 0.35, 0.28])
    piecewise_phase_continuous: bool = True

    # Surface Micro-Features
    enable_surface_texture: bool = False
    texture_type: str = "microgrooves"
    texture_amplitude_mm: float = 0.025
    texture_wavelength_mm: float = 0.080
    texture_profile: str = "sine"
    texture_displacement_mode: str = "centered"

    enable_micropillars: bool = False
    pillar_diameter_um: float = 60.0
    pillar_height_um: float = 200.0
    pillar_spacing_um: float = 250.0
    pillar_distribution: str = "poisson_disk"
    pillar_location: str = "all"
    pillar_filter_printable: bool = False

    # Viewport & Export
    preview_mode: str = "2D Surface (Ultra-Sharp)"
    show_generated_3d_mesh: bool = False
    export_stl: bool = True
    export_3mf: bool = True
    write_manifest: bool = True
    run_post_mis_check: bool = False
    custom_output_name: str = ""

    def to_recipe_dict(self) -> dict[str, Any]:
        """Serialize state into a JSON-compatible recipe dictionary."""
        data = asdict(self)
        data["schema_version"] = 1
        data["studio"] = "Graphite Implicit Studio"
        return data

    @classmethod
    def from_recipe_dict(cls, data: dict[str, Any]) -> "ImplicitStudioState":
        """Reconstruct ``ImplicitStudioState`` ignoring unknown keys cleanly."""
        defaults = asdict(cls())
        valid: dict[str, Any] = {}
        for key, default_val in defaults.items():
            if key not in data:
                valid[key] = default_val
            else:
                val = data[key]
                if isinstance(default_val, list):
                    valid[key] = list(val) if isinstance(val, (list, tuple)) else default_val
                else:
                    valid[key] = val
        if valid.get("preview_mode") == "3D Flying Edges (Fine)":
            valid["preview_mode"] = "3D Flying Edges (Coarse)"
        valid["exact_cad_trim"] = True
        return cls(**valid)

    def save_recipe(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_recipe_dict(), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load_recipe(cls, path: str | Path) -> "ImplicitStudioState":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_recipe_dict(raw)

    def to_manifest_params(self) -> dict[str, Any]:
        """Convert state to keys expected by ``graphite.io.lattice_manifest``."""
        drivers = _effective_driver_set(self)
        return {
            "geometry_type": "Primitive" if self.geometry_source == "Primitive" else "Custom STL",
            "prim_shape": self.primitive_shape,
            "uploaded_filename": Path(self.cad_file_path).name if self.cad_file_path else "",
            "lattice_type": self.lattice_type,
            "size_mode": "Pore Size (mm)" if drivers == {"pore_size", "solid_fraction"} else "Unit Cell Size (mm)",
            "unit_cell_size": self.unit_cell_size_mm,
            "pore_size": self.pore_size_mm,
            "density_mode": "wall_thickness" if "wall_thickness" in drivers else "solid_fraction",
            "solid_fraction": self.solid_fraction,
            "wall_thickness_mm": self.wall_thickness_mm,
            "grading_mode": self.grading_mode,
            "resolution_mm": self.resolution_mm,
            "export_mode": self.export_mode,
            "shell_thickness_mm": self.shell_thickness_mm,
            "blend_radius_mm": self.blend_radius_mm,
            "blend_method": self.blend_method,
            "exact_cad_trim": True,
            "multi_knot_engine": self.multi_knot_engine,
            "multi_knot_axis": self.multi_knot_axis,
            "multi_knot_kernel": self.multi_knot_kernel,
            "multi_knot_knots": self.multi_knot_knots,
            "piecewise_breaks_mm": self.piecewise_breaks_mm,
            "piecewise_L_mm": self.piecewise_L_mm,
            "piecewise_w_mm": self.piecewise_w_mm,
            "enable_surface_texture": self.enable_surface_texture,
            "enable_micropillars": self.enable_micropillars,
        }

    def apply_preset(self, preset_name: str, bounds: np.ndarray | None = None) -> None:
        """Populate grading parameters from a curated biological/engineering preset."""
        if bounds is not None and len(bounds) == 2:
            z0, z1 = float(bounds[0][2]), float(bounds[1][2])
            r_max = float(0.5 * max(bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]))
        else:
            hz = 0.5 * self.prim_z_mm
            z0, z1 = -hz, hz
            r_max = 0.5 * max(self.prim_x_mm, self.prim_y_mm)

        z_mid = 0.5 * (z0 + z1)
        z_t1 = z0 + (z1 - z0) / 3.0
        z_t2 = z0 + 2.0 * (z1 - z0) / 3.0

        if preset_name == "Osteochondral Z-Loft (3-Zone)":
            self.grading_mode = "Multi-Knot Control Table"
            self.multi_knot_engine = "Normalized SDF Blend"
            self.multi_knot_axis = "Z"
            self.multi_knot_kernel = "smoothstep"
            self.multi_knot_knots = [
                {"pos_mm": round(z0, 3), "L_mm": 2.4, "w_mm": 0.42, "sf": 0.45},
                {"pos_mm": round(z_mid, 3), "L_mm": 3.5, "w_mm": 0.50, "sf": 0.35},
                {"pos_mm": round(z1, 3), "L_mm": 5.0, "w_mm": 0.40, "sf": 0.22},
            ]
        elif preset_name == "Cube Thirds Stepped":
            self.grading_mode = "Piecewise Discrete Bands"
            self.piecewise_breaks_mm = [round(z_t1, 3), round(z_t2, 3)]
            self.piecewise_L_mm = [2.0, 3.5, 5.0]
            self.piecewise_w_mm = [0.40, 0.50, 0.60]
            self.piecewise_sf = [0.45, 0.35, 0.28]
            self.piecewise_phase_continuous = True
        elif preset_name == "Radial Vascular Scaffold":
            self.grading_mode = "Multi-Knot Control Table"
            self.multi_knot_engine = "Jacobian Phase Integration"
            self.multi_knot_axis = "Cylindrical Radius"
            self.multi_knot_kernel = "smoothstep"
            self.multi_knot_knots = [
                {"pos_mm": 0.0, "L_mm": 5.5, "w_mm": 0.35, "sf": 0.20},
                {"pos_mm": round(0.5 * r_max, 3), "L_mm": 3.8, "w_mm": 0.48, "sf": 0.34},
                {"pos_mm": round(r_max, 3), "L_mm": 2.6, "w_mm": 0.58, "sf": 0.48},
            ]


def has_threemf_support() -> bool:
    """Return True if optional ``lxml`` dependency required for 3MF export is installed."""
    return importlib.util.find_spec("lxml") is not None


def discover_cad_fixtures(root: Path | None = None) -> list[Path]:
    """List read-only CAD fixtures under ``test_parts/`` sorted by name."""
    base = (root or WORKSPACE_ROOT) / "test_parts"
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".stl"),
        key=lambda p: p.name.lower(),
    )


def _health_to_dict(report: Any, mesh: trimesh.Trimesh | None = None) -> dict[str, Any]:
    """Normalize ``MeshHealthReport`` into a plain dictionary for UI/JSON."""
    extents = (
        [round(float(v), 3) for v in (mesh.bounds[1] - mesh.bounds[0])]
        if mesh is not None and mesh.bounds is not None
        else [0.0, 0.0, 0.0]
    )
    if isinstance(report, dict):
        out = dict(report)
        out.setdefault("extents_mm", extents)
        return out
    return {
        "vertex_count": int(getattr(report, "vertex_count", 0)),
        "face_count": int(getattr(report, "face_count", 0)),
        "euler_characteristic": int(getattr(report, "euler_characteristic", 0)),
        "is_watertight": bool(getattr(report, "is_watertight", False)),
        "is_export_ready": bool(getattr(report, "is_export_ready", False)),
        "boundary_edges": int(getattr(report, "boundary_edges", 0)),
        "boundary_loops": int(getattr(report, "boundary_loops", 0)),
        "non_manifold_edges": int(getattr(report, "non_manifold_edges", 0)),
        "non_manifold_vertices": int(getattr(report, "non_manifold_vertices", 0)),
        "surface_area": float(getattr(report, "surface_area", 0.0)),
        "volume": float(getattr(report, "volume", 0.0)),
        "extents_mm": extents,
        "notes": list(getattr(report, "notes", [])),
    }


def build_or_load_cad_mesh(state: ImplicitStudioState) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Build selected primitive or load STL fixture and compute health diagnostics."""
    if state.geometry_source == "Primitive":
        shape = state.primitive_shape
        x, y, z = float(state.prim_x_mm), float(state.prim_y_mm), float(state.prim_z_mm)
        if shape == "Cube":
            if abs(x - y) < 1e-6 and abs(x - z) < 1e-6:
                mesh = generate_primitive("Cube", x)
            else:
                mesh = trimesh.creation.box(extents=[x, y, z])
        elif shape == "Sphere":
            mesh = generate_primitive("Sphere", x)
        elif shape == "Cylinder":
            if abs(x - z) < 1e-6:
                mesh = generate_primitive("Cylinder", x)
            else:
                mesh = trimesh.creation.cylinder(radius=0.5 * x, height=z, sections=64)
        elif shape == "Torus":
            mesh = generate_primitive("Torus", x)
        elif shape == "Tube":
            if abs(y - 0.7 * x) < 1e-6 and abs(x - z) < 1e-6:
                mesh = generate_primitive("Tube", x)
            else:
                import manifold3d as m3d
                from graphite.explicit.geometry_module import _manifold_to_trimesh

                r_out = max(0.5 * x, 1.0)
                r_in = max(min(0.5 * y, 0.85 * r_out), 0.25)
                c_out = m3d.Manifold.cylinder(z, r_out, r_out, circular_segments=48, center=True)
                c_in = m3d.Manifold.cylinder(z * 1.05, r_in, r_in, circular_segments=48, center=True)
                mesh = _manifold_to_trimesh(c_out - c_in)
        else:
            mesh = generate_primitive("Cube", x)
    else:
        path = Path(state.cad_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"CAD file not found: {path}")
        loaded = trimesh.load(str(path), force="mesh")
        mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded

    if state.center_origin and mesh.bounds is not None:
        center = 0.5 * (np.asarray(mesh.bounds[0], float) + np.asarray(mesh.bounds[1], float))
        mesh = mesh.copy()
        mesh.apply_translation(-center)

    health = _health_to_dict(compute_mesh_health(mesh), mesh)
    return mesh, health


def repair_cad_mesh_in_memory(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Run non-destructive Phase 1 gentle repair in memory (never touches test_parts/)."""
    repaired = repair_trimesh_gentle(mesh)
    health = _health_to_dict(compute_mesh_health(repaired), repaired)
    return repaired, health


def estimate_voxel_budget(mesh: trimesh.Trimesh | None, resolution_mm: float) -> dict[str, Any]:
    """Estimate Cartesian voxel count and rough float64 RAM footprint for user feedback."""
    res = max(float(resolution_mm), 0.005)
    if mesh is None or mesh.bounds is None:
        return {"nx": 0, "ny": 0, "nz": 0, "voxels": 0, "est_ram_mb": 0.0, "warning": ""}
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    nx = int(np.ceil(ext[0] / res)) + 5
    ny = int(np.ceil(ext[1] / res)) + 5
    nz = int(np.ceil(ext[2] / res)) + 5
    voxels = nx * ny * nz
    # SDF + TPMS + solid field typically ~ 3 float64/float32 arrays -> ~24 bytes/voxel
    est_ram_mb = (voxels * 24.0) / (1024.0 * 1024.0)
    warning = ""
    if voxels > 80_000_000:
        warning = "Extreme grid (>80M voxels) — increase resolution_mm to avoid RAM pressure."
    elif voxels > 25_000_000:
        warning = "Heavy grid (>25M voxels) — generation may take >30s."
    return {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "voxels": voxels,
        "est_ram_mb": round(est_ram_mb, 1),
        "warning": warning,
    }


def _coord_axis_values(
    axis_name: str,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    origin: tuple[float, float, float],
) -> np.ndarray:
    cx, cy, cz = origin
    name = axis_name.lower()
    if name == "x":
        return X - cx
    if name == "y":
        return Y - cy
    if name == "z":
        return Z - cz
    if "cyl" in name:
        return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)


def _piecewise_zone_lookup(
    z_vals: np.ndarray,
    breaks: list[float],
    values: list[float],
) -> np.ndarray:
    idx = np.digitize(z_vals, np.asarray(breaks, dtype=float), right=False)
    idx = np.clip(idx, 0, max(len(values) - 1, 0))
    arr = np.asarray(values, dtype=float)
    return arr[idx]


def evaluate_grading_profile_1d(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh | None,
    n_samples: int = 120,
) -> dict[str, Any]:
    """Sample L(s), w(s), and SF(s) along the active grading axis for live HUD readouts."""
    if cad_mesh is not None and cad_mesh.bounds is not None:
        b0, b1 = np.asarray(cad_mesh.bounds[0], float), np.asarray(cad_mesh.bounds[1], float)
    else:
        b0 = np.array([-0.5 * state.prim_x_mm, -0.5 * state.prim_y_mm, -0.5 * state.prim_z_mm])
        b1 = -b0
    center = 0.5 * (b0 + b1)

    if state.grading_mode == "Uniform":
        s = np.linspace(b0[2], b1[2], n_samples)
        L = np.full_like(s, state.unit_cell_size_mm)
        w = np.full_like(s, state.wall_thickness_mm)
        sf = np.full_like(s, state.solid_fraction)
        return {
            "axis": "Z",
            "pos_mm": s,
            "L_mm": L,
            "w_mm": w,
            "sf": sf,
            "markers_mm": [],
        }

    if state.grading_mode == "Linear / Radial Ramp":
        ax = state.ramp_axis
        if ax in ("X", "Y", "Z"):
            idx = {"X": 0, "Y": 1, "Z": 2}[ax]
            s = np.linspace(b0[idx], b1[idx], n_samples)
            t = (s - b0[idx]) / max(b1[idx] - b0[idx], 1e-6)
        else:
            r_max = float(np.linalg.norm(0.5 * (b1 - b0)))
            s = np.linspace(0.0, r_max, n_samples)
            t = s / max(r_max, 1e-6)
        sf = (
            state.ramp_sf_min + (state.ramp_sf_max - state.ramp_sf_min) * t
            if state.ramp_target in ("Solid Fraction", "Both")
            else np.full_like(s, state.solid_fraction)
        )
        L = (
            state.ramp_L_min_mm + (state.ramp_L_max_mm - state.ramp_L_min_mm) * t
            if state.ramp_target in ("Cell Size (Chirp)", "Both")
            else np.full_like(s, state.unit_cell_size_mm)
        )
        w = np.array([wall_thickness_from_tau_L(tau_from_solid_fraction_fast(state.lattice_type, f), l) for f, l in zip(sf, L)])
        return {"axis": ax, "pos_mm": s, "L_mm": L, "w_mm": w, "sf": sf, "markers_mm": []}

    if state.grading_mode == "Multi-Knot Control Table":
        knots = sorted(state.multi_knot_knots, key=lambda k: float(k["pos_mm"]))
        pos_k = np.array([float(k["pos_mm"]) for k in knots], dtype=float)
        L_k = np.array([float(k["L_mm"]) for k in knots], dtype=float)
        w_k = np.array([float(k["w_mm"]) for k in knots], dtype=float)
        sf_k = np.array([float(k["sf"]) for k in knots], dtype=float)
        s0, s1 = float(pos_k[0]), float(pos_k[-1])
        s = np.linspace(s0, s1 if s1 > s0 else s0 + 1.0, n_samples)
        L = np.interp(s, pos_k, L_k)
        w = np.interp(s, pos_k, w_k)
        sf = np.interp(s, pos_k, sf_k)
        return {
            "axis": state.multi_knot_axis,
            "pos_mm": s,
            "L_mm": L,
            "w_mm": w,
            "sf": sf,
            "markers_mm": [float(p) for p in pos_k],
        }

    # Piecewise Discrete Bands
    s = np.linspace(b0[2], b1[2], n_samples)
    L = _piecewise_zone_lookup(s, state.piecewise_breaks_mm, state.piecewise_L_mm)
    w = _piecewise_zone_lookup(s, state.piecewise_breaks_mm, state.piecewise_w_mm)
    sf = _piecewise_zone_lookup(s, state.piecewise_breaks_mm, state.piecewise_sf)
    return {
        "axis": "Z",
        "pos_mm": s,
        "L_mm": L,
        "w_mm": w,
        "sf": sf,
        "markers_mm": [float(b) for b in state.piecewise_breaks_mm],
    }


def _field_driven_coord_spec(axis_label: str) -> tuple[str, str]:
    """Map UI coordinate axis label to field_driven (coordinate_name, coord_kind)."""
    key = axis_label.strip().lower()
    if key in ("x", "cartesian x"):
        return "Cartesian X", "x"
    if key in ("y", "cartesian y"):
        return "Cartesian Y", "y"
    if key in ("z", "cartesian z", "cartesian"):
        return "Cartesian Z", "z"
    if "cyl" in key:
        return "Cylindrical Radius", "cylindrical"
    return "Spherical Radius", "spherical"


def _evaluate_jacobian_phase_coordinates(
    axis_label: str,
    unit_cell_control_points: list[tuple[float, float]],
    Xr: np.ndarray,
    Yr: np.ndarray,
    Zr: np.ndarray,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute integrated phase coordinates (U, V, W) and local period L_grid."""
    coord_name, coord_kind = _field_driven_coord_spec(axis_label)
    s_vals = _coord_axis_values(axis_label, Xr, Yr, Zr, origin)
    L_grid = interpolate_profile(s_vals, unit_cell_control_points, min_value=0.1)
    U, V, W = build_integrated_phase_coordinates(
        coord_name,
        coord_kind,
        s_vals,
        L_grid,
        unit_cell_control_points,
        Xr,
        Yr,
        Zr,
        center=np.asarray(origin, dtype=np.float64),
    )
    return U, V, W, L_grid


def _apply_surface_texture_to_field(
    state: ImplicitStudioState,
    field: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    spacing: float,
) -> np.ndarray:
    """Apply optional procedural surface texture modulation when resolved cleanly."""
    if not state.enable_surface_texture or state.texture_amplitude_mm <= 1e-6:
        return field
    wave = max(float(state.texture_wavelength_mm), 0.01)
    # Skip sub-Nyquist static aliasing on coarse preview grids; 2D high-res slices & fine previews resolve cleanly.
    if wave < 1.6 * max(float(spacing), 1e-4):
        return field

    ttype = str(state.texture_type).strip().lower()
    if ttype == "bumps":
        prof = "isolated" if str(state.texture_profile).strip().lower() == "isolated" else "nodule"
        tex = bump_field(X, Y, Z, wavelength_mm=wave, profile=prof)
    elif ttype == "knurling":
        tex = knurl_field(X, Y, Z, wavelength_mm=wave, axis="z")
    elif ttype == "spinodal":
        tex = spinodal_spectral_field(X, Y, Z, wavelength_mm=wave, num_waves=24, seed=42)
    else:
        prof = str(state.texture_profile).strip().lower()
        if prof not in ("sine", "triangle", "square"):
            prof = "sine"
        tex = microgroove_field(
            X,
            Y,
            Z,
            wavelength_mm=wave,
            direction=(0.0, 0.0, 1.0),
            profile=prof,
        )

    disp_mode = str(state.texture_displacement_mode).strip().lower()
    if disp_mode == "emboss":
        shift = 0.5 * (tex + 1.0)
    elif disp_mode == "engrave":
        shift = 0.5 * (tex - 1.0)
    else:
        shift = tex
    return field - float(state.texture_amplitude_mm) * np.asarray(shift, dtype=field.dtype)


def evaluate_implicit_solid_field(
    state: ImplicitStudioState,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cad_mesh: trimesh.Trimesh | None,
    *,
    spacing: float = 0.25,
) -> np.ndarray:
    """Evaluate signed implicit lattice scalar field (``<= 0`` is solid).

    Shared across the fast 3D surface footprint preview, the 2D 10 um slice
    inspector, and Normalized SDF Blend generation so visual previews always
    match production geometry.
    """
    base_field = _evaluate_base_solid_field(
        state, X, Y, Z, cad_mesh, spacing=spacing
    )
    return _apply_surface_texture_to_field(
        state, base_field, X, Y, Z, spacing=spacing
    )


def _evaluate_base_solid_field(
    state: ImplicitStudioState,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cad_mesh: trimesh.Trimesh | None,
    *,
    spacing: float = 0.25,
) -> np.ndarray:
    """Evaluate base untextured signed implicit lattice scalar field."""
    if cad_mesh is not None and cad_mesh.bounds is not None:
        b0 = np.asarray(cad_mesh.bounds[0], dtype=float)
        b1 = np.asarray(cad_mesh.bounds[1], dtype=float)
    else:
        b0 = np.array([-0.5 * state.prim_x_mm, -0.5 * state.prim_y_mm, -0.5 * state.prim_z_mm])
        b1 = -b0
    center = 0.5 * (b0 + b1)
    origin = (float(center[0]), float(center[1]), float(center[2]))

    # Apply optional lattice orientation rotation around part center
    rx, ry, rz = np.deg2rad([state.rotation_x_deg, state.rotation_y_deg, state.rotation_z_deg])
    Xr, Yr, Zr = X, Y, Z
    if abs(rx) > 1e-9 or abs(ry) > 1e-9 or abs(rz) > 1e-9:
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
        dx = X - origin[0]
        dy = Y - origin[1]
        dz = Z - origin[2]
        Xr = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz + origin[0]
        Yr = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz + origin[1]
        Zr = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz + origin[2]

    canon = _canonical_tpms_name(state.lattice_type)

    # Woodpile fast continuous evaluation
    if canon == "Woodpile":
        F = evaluate_woodpile(
            Xr,
            Yr,
            Zr,
            pore_size=float(state.pore_size_mm),
            true_woodpile=state.woodpile_true_mode,
        )
        return np.asarray(F, dtype=float)

    # Multi-Knot Control Table
    if state.grading_mode == "Multi-Knot Control Table":
        knots = sorted(state.multi_knot_knots, key=lambda k: float(k["pos_mm"]))
        axis_knots = [float(k["pos_mm"]) for k in knots]
        period_knots = [float(k["L_mm"]) for k in knots]
        w_knots = [float(k["w_mm"]) for k in knots]

        if state.multi_knot_engine == "Normalized SDF Blend" and state.multi_knot_axis in ("X", "Y", "Z"):
            solid_field, _, _ = blend_multi_knot_tpms_sdfs(
                canon,
                axis_knots,
                period_knots,
                w_knots,
                Xr,
                Yr,
                Zr,
                spacing=max(float(spacing), 0.01),
                transition_axis=state.multi_knot_axis,
                kernel=state.multi_knot_kernel,
                is_sheet=True,
            )
            return solid_field

        # Jacobian integrated phase coordinates for Arbitrary/Cylindrical/Spherical or when chosen
        cp_uc = [(float(k["pos_mm"]), float(k["L_mm"])) for k in knots]
        U, V, W, _ = _evaluate_jacobian_phase_coordinates(
            state.multi_knot_axis,
            cp_uc,
            Xr,
            Yr,
            Zr,
            origin,
        )
        s_vals = _coord_axis_values(state.multi_knot_axis, X, Y, Z, origin)
        L_local = np.interp(s_vals, axis_knots, period_knots)
        if "wall_thickness" in _effective_driver_set(state):
            w_local = np.interp(s_vals, axis_knots, w_knots)
            tau_local = tau_from_wall_thickness_mm(w_local, L_local)
        else:
            sf_knots = [float(k["sf"]) for k in knots]
            sf_local = np.interp(s_vals, axis_knots, sf_knots)
            tau_local = np.asarray([tau_from_solid_fraction_fast(canon, float(f)) for f in sf_local.ravel()]).reshape(sf_local.shape)
        F_phase = evaluate_tpms_phase(canon, U, V, W)
        return np.abs(F_phase) - tau_local

    # Piecewise Discrete Bands (Split-P or other TPMS with optional phase continuity)
    if state.grading_mode == "Piecewise Discrete Bands":
        breaks = sorted(float(b) for b in state.piecewise_breaks_mm)
        L_list = [float(v) for v in state.piecewise_L_mm]
        w_list = [float(v) for v in state.piecewise_w_mm]
        L_local = _piecewise_zone_lookup(Z, breaks, L_list)
        w_local = _piecewise_zone_lookup(Z, breaks, w_list)
        tau_local = tau_from_wall_thickness_mm(w_local, L_local)
        if state.piecewise_phase_continuous:
            height_mm = max(float(b1[2] - b0[2]), 1e-3)
            internal = sorted(float(np.clip(b - b0[2], 0.0, height_mm)) for b in breaks)
            full_breaks = [0.0] + internal[: max(len(L_list) - 1, 0)] + [height_mm]
            z_dense, W_dense = cumulative_phase_w_from_l_profile(
                height_mm, full_breaks, L_list
            )
            W = np.interp(np.asarray(Zr, dtype=float) - float(b0[2]), z_dense, W_dense)
            k_local = (2.0 * np.pi) / np.maximum(L_local, 1e-6)
            U = k_local * (Xr - origin[0])
            V = k_local * (Yr - origin[1])
            F_phase = evaluate_tpms_phase(canon, U, V, W)
        else:
            k_local = (2.0 * np.pi) / np.maximum(L_local, 1e-6)
            F_phase = evaluate_tpms_phase(
                canon,
                k_local * (Xr - origin[0]),
                k_local * (Yr - origin[1]),
                k_local * (Zr - origin[2]),
            )
        return np.abs(F_phase) - tau_local

    # Linear / Radial Ramp
    if state.grading_mode == "Linear / Radial Ramp":
        ax = state.ramp_axis
        if ax in ("X", "Y", "Z"):
            idx = {"X": 0, "Y": 1, "Z": 2}[ax]
            coord = {"X": X, "Y": Y, "Z": Z}[ax]
            t = np.clip((coord - b0[idx]) / max(b1[idx] - b0[idx], 1e-6), 0.0, 1.0)
            s_vals = coord
            s0, s1 = float(b0[idx]), float(b1[idx])
        else:
            r = _coord_axis_values("Spherical Radius", X, Y, Z, origin)
            r_max = max(float(np.linalg.norm(0.5 * (b1 - b0))), 1e-6)
            t = np.clip(r / r_max, 0.0, 1.0)
            s_vals = r
            s0, s1 = 0.0, r_max

        if state.ramp_target in ("Cell Size (Chirp)", "Both"):
            cp_uc = [(s0, state.ramp_L_min_mm), (s1, state.ramp_L_max_mm)]
            coord_sys = "Spherical Radius" if ax == "Radial" else ax
            U, V, W, _ = _evaluate_jacobian_phase_coordinates(
                coord_sys,
                cp_uc,
                Xr,
                Yr,
                Zr,
                origin,
            )
            F_phase = evaluate_tpms_phase(canon, U, V, W)
        else:
            k = (2.0 * np.pi) / max(state.unit_cell_size_mm, 0.1)
            F_phase = evaluate_tpms(canon, k, Xr - origin[0], Yr - origin[1], Zr - origin[2])

        if state.ramp_target in ("Solid Fraction", "Both"):
            sf_vals = state.ramp_sf_min + (state.ramp_sf_max - state.ramp_sf_min) * t
            tau_min = tau_from_solid_fraction_fast(canon, state.ramp_sf_min)
            tau_max = tau_from_solid_fraction_fast(canon, state.ramp_sf_max)
            tau_local = tau_min + (tau_max - tau_min) * t
            return np.abs(F_phase) - tau_local
        tau_u = tau_from_solid_fraction_fast(canon, state.solid_fraction)
        return np.abs(F_phase) - tau_u

    # Uniform default
    k = (2.0 * np.pi) / max(float(state.unit_cell_size_mm), 0.1)
    F = evaluate_tpms(canon, k, Xr - origin[0], Yr - origin[1], Zr - origin[2])
    if "wall_thickness" in _effective_driver_set(state):
        tau = float(np.asarray(tau_from_wall_thickness_mm(state.wall_thickness_mm, state.unit_cell_size_mm)).ravel()[0])
    else:
        tau = tau_from_solid_fraction_fast(canon, state.solid_fraction)
    return np.abs(F) - tau


@dataclass
class CalibrationOutcome:
    """Result summary returned by 1-click calibration."""

    calibrated_L_mm: float
    calibrated_tau: float
    wall_thickness_mm: float
    solid_fraction: float
    measured_pore_mm: float
    converged: bool
    source: str
    iterations: int = 1
    residual_pore_mm: float = 0.0
    updated_knots: list[dict[str, float]] | None = None
    notes: list[str] = field(default_factory=list)


def run_tpms_calibration(
    state: ImplicitStudioState,
    progress_cb: Callable[[str], None] | None = None,
) -> CalibrationOutcome:
    """Run fast LUT seed + Newton calibration for Uniform or Multi-Knot profiles."""
    canon = _canonical_tpms_name(state.lattice_type)
    if canon == "Woodpile":
        return CalibrationOutcome(
            calibrated_L_mm=state.unit_cell_size_mm,
            calibrated_tau=0.0,
            wall_thickness_mm=state.wall_thickness_mm,
            solid_fraction=state.solid_fraction,
            measured_pore_mm=state.pore_size_mm,
            converged=True,
            source="woodpile_analytical",
            notes=["Woodpile uses exact geometric period L = pore + bar width; no TPMS calibration needed."],
        )

    if state.grading_mode == "Multi-Knot Control Table":
        if progress_cb:
            progress_cb(f"Calibrating {len(state.multi_knot_knots)} control knots for {canon}...")
        updated: list[dict[str, float]] = []
        last_res = None
        for idx, knot in enumerate(state.multi_knot_knots):
            res = calibrate_period_for_pore_at_sf(
                canon,
                target_pore_mm=max(float(knot["L_mm"]) * (1.0 - float(knot["sf"])), 0.15),
                target_solid_fraction=float(np.clip(knot["sf"], 0.05, 0.85)),
                config=CalibrationConfig(max_iterations=10),
            )
            last_res = res
            w_cal = wall_thickness_from_tau_L(res.calibrated_tau, res.calibrated_L_mm)
            updated.append(
                {
                    "pos_mm": float(knot["pos_mm"]),
                    "L_mm": round(float(res.calibrated_L_mm), 4),
                    "w_mm": round(float(w_cal), 4),
                    "sf": round(float(res.measured_solid_fraction), 4),
                }
            )
            if progress_cb:
                progress_cb(f"Knot {idx + 1}/{len(state.multi_knot_knots)}: L={res.calibrated_L_mm:.3f} mm, SF={res.measured_solid_fraction:.3f}")
        assert last_res is not None
        return CalibrationOutcome(
            calibrated_L_mm=float(updated[0]["L_mm"]),
            calibrated_tau=float(last_res.calibrated_tau),
            wall_thickness_mm=float(updated[0]["w_mm"]),
            solid_fraction=float(updated[0]["sf"]),
            measured_pore_mm=float(last_res.measured_pore_mm),
            converged=bool(last_res.converged),
            source="multi_knot_period_sf_calibration",
            updated_knots=updated,
            notes=[f"Calibrated {len(updated)} control knots independently."],
        )

    if progress_cb:
        progress_cb(f"Querying LUT & calibrating {canon}...")

    # Try exact LUT lookup first when pore + SF are targets
    lut_hit = None
    try:
        lut_hit = lookup_tpms_parameters(
            lattice_type=canon,
            pore_mm=float(state.pore_size_mm),
            solid_fraction=float(state.solid_fraction),
        )
    except Exception:
        lut_hit = None

    drivers = _effective_driver_set(state)
    if drivers == {"pore_size", "solid_fraction"}:
        init_L = float(lut_hit["L_mm"]) if isinstance(lut_hit, dict) and "L_mm" in lut_hit else None
        init_tau = float(lut_hit["tau"]) if isinstance(lut_hit, dict) and "tau" in lut_hit else None
        res = calibrate_tpms_point(
            canon,
            target_pore_mm=float(state.pore_size_mm),
            target_solid_fraction=float(np.clip(state.solid_fraction, 0.05, 0.90)),
            initial_L_mm=init_L,
            initial_tau=init_tau,
            config=CalibrationConfig(max_iterations=12),
        )
        src = "lut+newton" if lut_hit else res.seed_source
    else:
        res = calibrate_tau_at_fixed_period(
            canon,
            period_mm=float(state.unit_cell_size_mm),
            target_solid_fraction=float(np.clip(state.solid_fraction, 0.05, 0.90)),
        )
        src = "fixed_period_tau_quantile"

    w_cal = wall_thickness_from_tau_L(res.calibrated_tau, res.calibrated_L_mm)
    return CalibrationOutcome(
        calibrated_L_mm=round(float(res.calibrated_L_mm), 4),
        calibrated_tau=round(float(res.calibrated_tau), 4),
        wall_thickness_mm=round(float(w_cal), 4),
        solid_fraction=round(float(res.measured_solid_fraction), 4),
        measured_pore_mm=round(float(res.measured_pore_mm), 4),
        converged=bool(res.converged),
        source=src,
        iterations=int(getattr(res, "iterations", 1)),
        residual_pore_mm=round(float(getattr(res, "residual_pore_mm", 0.0)), 4),
        notes=list(res.notes),
    )


@dataclass
class GenerationResult:
    """Artifacts and diagnostics produced by a production generation run."""

    mesh: trimesh.Trimesh
    health: dict[str, Any]
    output_dir: Path
    output_paths: list[Path]
    recipe_path: Path | None
    manifest_path: Path | None
    mis_pore_mm: float | None
    mis_confidence: str
    elapsed_seconds: float
    stage_notes: list[str] = field(default_factory=list)


def _ensure_dated_output_dir(output_root: Path | None = None) -> Path:
    root = output_root or DEFAULT_OUTPUT_ROOT
    dated = root / datetime.now().strftime("%Y-%m-%d")
    dated.mkdir(parents=True, exist_ok=True)
    return dated


def _generate_normalized_sdf_blend_mesh(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh,
    res_mm: float,
) -> trimesh.Trimesh:
    """Production mesh generator for ``Multi-Knot + Normalized SDF Blend``."""
    X, Y, Z, cad_sdf, padded_min_bound, _, _, _, _ = voxelize_mesh_and_edt(
        cad_mesh, res_mm
    )

    canon = _canonical_tpms_name(state.lattice_type)
    knots = sorted(state.multi_knot_knots, key=lambda k: float(k["pos_mm"]))
    lattice_field, _, _ = blend_multi_knot_tpms_sdfs(
        canon,
        [float(k["pos_mm"]) for k in knots],
        [float(k["L_mm"]) for k in knots],
        [float(k["w_mm"]) for k in knots],
        X,
        Y,
        Z,
        spacing=res_mm,
        transition_axis=state.multi_knot_axis if state.multi_knot_axis in ("X", "Y", "Z") else "Z",
        kernel=state.multi_knot_kernel,
        is_sheet=True,
    )
    if state.export_mode == "skin":
        combined_field = np.maximum(
            cad_sdf, -cad_sdf - float(state.shell_thickness_mm)
        )
    else:
        skin_t = (
            float(state.shell_thickness_mm)
            if state.export_mode == "combined"
            else 0.0
        )
        combined_field = blend_lattice_with_skin(
            lattice_field=lattice_field,
            cad_sdf=cad_sdf,
            skin_thickness=skin_t,
            blend_radius=float(state.blend_radius_mm),
            method=state.blend_method,
        )
    return extract_isosurface(
        combined_field,
        spacing=(res_mm, res_mm, res_mm),
        origin=(
            float(padded_min_bound[0]),
            float(padded_min_bound[1]),
            float(padded_min_bound[2]),
        ),
        backend="auto",
        level=0.0,
        enforce_watertight=False,
    )


def run_implicit_generation(
    state: ImplicitStudioState,
    cad_mesh: trimesh.Trimesh | None = None,
    *,
    output_root: Path | None = None,
    progress_cb: Callable[[int, str], None] | None = None,
) -> GenerationResult:
    """Execute full implicit lattice synthesis, post-processing, and export."""
    t0 = time.perf_counter()
    notes: list[str] = []

    def _emit(pct: int, msg: str) -> None:
        notes.append(msg)
        if progress_cb:
            progress_cb(pct, msg)

    _emit(5, "Preparing CAD geometry...")
    if cad_mesh is None:
        cad_mesh, _ = build_or_load_cad_mesh(state)

    out_dir = _ensure_dated_output_dir(output_root)
    work_dir = out_dir / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_cad_stl = work_dir / f"_cad_input_{int(time.time() * 1000)}.stl"
    cad_mesh.export(str(temp_cad_stl))

    if state.auto_resolution:
        prof = evaluate_grading_profile_1d(state, cad_mesh, n_samples=40)
        min_w = float(np.min(prof["w_mm"])) if len(prof["w_mm"]) else state.wall_thickness_mm
        min_L = float(np.min(prof["L_mm"])) if len(prof["L_mm"]) else state.unit_cell_size_mm
        res_mm = recommended_nyquist_resolution_mm(min_w, min_L)
    else:
        res_mm = max(float(state.resolution_mm), 0.01)
    rot_deg = (
        (state.rotation_x_deg, state.rotation_y_deg, state.rotation_z_deg)
        if any(abs(v) > 1e-6 for v in (state.rotation_x_deg, state.rotation_y_deg, state.rotation_z_deg))
        else None
    )

    try:
        canon = _canonical_tpms_name(state.lattice_type)
        _emit(20, f"Generating {canon} lattice ({state.grading_mode}, res={res_mm:.3f} mm)...")

        if canon == "Woodpile":
            if state.grading_mode == "Piecewise Discrete Bands":
                b0, b1 = np.asarray(cad_mesh.bounds[0], float), np.asarray(cad_mesh.bounds[1], float)
                ext = b1 - b0
                spec = WoodpileLatticeSpec(
                    total_width_x=float(ext[0]),
                    total_depth_y=float(ext[1]),
                    total_height_z=float(ext[2]),
                    z_breaks_mm=[float(b - b0[2]) for b in state.piecewise_breaks_mm],
                    pore_sizes_mm=[float(v) for v in state.piecewise_L_mm],
                    rod_widths_mm=[float(v) for v in state.piecewise_w_mm],
                    resolution_mm=res_mm,
                )
                mesh = build_piecewise_woodpile_mesh(spec)
            else:
                mesh = generate_uniform_woodpile(
                    str(temp_cad_stl),
                    pore_size=float(state.pore_size_mm),
                    true_woodpile=bool(state.woodpile_true_mode),
                    resolution=res_mm,
                    center_origin=bool(state.center_origin),
                )
        elif state.grading_mode == "Uniform":
            drivers = _effective_driver_set(state)
            mesh = generate_conformal_lattice(
                str(temp_cad_stl),
                lattice_type=canon,
                resolution=res_mm,
                pore_size=float(state.pore_size_mm),
                unit_cell_size=None if drivers == {"pore_size", "solid_fraction"} else float(state.unit_cell_size_mm),
                solid_fraction=float(state.solid_fraction),
                wall_thickness_mm=float(state.wall_thickness_mm) if "wall_thickness" in drivers else None,
                export_mode=state.export_mode,
                shell_thickness=float(state.shell_thickness_mm),
                center_origin=bool(state.center_origin),
                exact_cad_trim=True,
                trim_dilation_mm=float(state.trim_dilation_mm) if state.trim_dilation_mm > 0 else None,
                rotation_deg=rot_deg,
                blend_radius=float(state.blend_radius_mm),
                blend_method=state.blend_method,
            )
        elif state.grading_mode == "Linear / Radial Ramp":
            grad_ax = "Radial" if state.ramp_axis == "Radial" else state.ramp_axis
            if state.ramp_target == "Cell Size (Chirp)":
                mesh = generate_chirped_lattice(
                    str(temp_cad_stl),
                    lattice_type=canon,
                    gradient_type=grad_ax,
                    resolution=res_mm,
                    solid_fraction=float(state.solid_fraction),
                    min_pore_size=float(state.ramp_L_min_mm),
                    max_pore_size=float(state.ramp_L_max_mm),
                    transition_width=float(state.ramp_transition_width_mm),
                    export_mode=state.export_mode,
                    shell_thickness=float(state.shell_thickness_mm),
                    center_origin=bool(state.center_origin),
                    blend_radius=float(state.blend_radius_mm),
                    blend_method=state.blend_method,
                )
            elif state.ramp_target == "Solid Fraction":
                mesh = generate_graded_lattice(
                    str(temp_cad_stl),
                    lattice_type=canon,
                    gradient_type=grad_ax,
                    resolution=res_mm,
                    pore_size=float(state.pore_size_mm),
                    min_solid_fraction=float(state.ramp_sf_min),
                    max_solid_fraction=float(state.ramp_sf_max),
                    transition_width=float(state.ramp_transition_width_mm),
                    export_mode=state.export_mode,
                    shell_thickness=float(state.shell_thickness_mm),
                    center_origin=bool(state.center_origin),
                    blend_radius=float(state.blend_radius_mm),
                    blend_method=state.blend_method,
                )
            else:
                b0 = np.asarray(cad_mesh.bounds[0], float)
                b1 = np.asarray(cad_mesh.bounds[1], float)
                if grad_ax in ("X", "Y", "Z"):
                    idx = {"X": 0, "Y": 1, "Z": 2}[grad_ax]
                    s0, s1 = float(b0[idx]), float(b1[idx])
                    coord_sys = _field_driven_coord_spec(grad_ax)[0]
                else:
                    s0, s1 = 0.0, float(np.linalg.norm(0.5 * (b1 - b0)))
                    coord_sys = "Spherical Radius"
                mesh = generate_field_driven_lattice(
                    str(temp_cad_stl),
                    lattice_type=canon,
                    resolution=res_mm,
                    base_unit_cell_size=float(state.unit_cell_size_mm),
                    base_solid_fraction=float(state.solid_fraction),
                    grade_unit_cell=True,
                    unit_cell_coordinate=coord_sys,
                    unit_cell_control_points=[(s0, state.ramp_L_min_mm), (s1, state.ramp_L_max_mm)],
                    grade_solid_fraction=True,
                    solid_fraction_coordinate=coord_sys,
                    solid_fraction_control_points=[(s0, state.ramp_sf_min), (s1, state.ramp_sf_max)],
                    export_mode=state.export_mode,
                    shell_thickness=float(state.shell_thickness_mm),
                    blend_radius=float(state.blend_radius_mm),
                    blend_method=state.blend_method,
                    center_origin=bool(state.center_origin),
                )
        elif state.grading_mode == "Multi-Knot Control Table":
            if state.multi_knot_engine == "Normalized SDF Blend" and state.multi_knot_axis in ("X", "Y", "Z"):
                mesh = _generate_normalized_sdf_blend_mesh(state, cad_mesh, res_mm)
            else:
                knots = sorted(state.multi_knot_knots, key=lambda k: float(k["pos_mm"]))
                uc_cp = [(float(k["pos_mm"]), float(k["L_mm"])) for k in knots]
                if "wall_thickness" in _effective_driver_set(state):
                    sf_cp = [(float(k["pos_mm"]), float(k["w_mm"])) for k in knots]
                    density_mode = "wall_thickness"
                else:
                    sf_cp = [(float(k["pos_mm"]), float(k["sf"])) for k in knots]
                    density_mode = "solid_fraction"
                fd_coord = _field_driven_coord_spec(state.multi_knot_axis)[0]
                mesh = generate_field_driven_lattice(
                    str(temp_cad_stl),
                    lattice_type=canon,
                    resolution=res_mm,
                    base_unit_cell_size=float(state.unit_cell_size_mm),
                    base_solid_fraction=float(state.solid_fraction),
                    base_wall_thickness_mm=float(state.wall_thickness_mm),
                    density_mode=density_mode,
                    grade_unit_cell=True,
                    unit_cell_coordinate=fd_coord,
                    unit_cell_control_points=uc_cp,
                    grade_solid_fraction=True,
                    solid_fraction_coordinate=fd_coord,
                    solid_fraction_control_points=sf_cp,
                    calibrate_solid_fraction=bool(state.multi_knot_calibrate),
                    export_mode=state.export_mode,
                    shell_thickness=float(state.shell_thickness_mm),
                    blend_radius=float(state.blend_radius_mm),
                    blend_method=state.blend_method,
                    center_origin=bool(state.center_origin),
                )
        else:
            # Piecewise Discrete Bands
            b0, b1 = np.asarray(cad_mesh.bounds[0], float), np.asarray(cad_mesh.bounds[1], float)
            ext = b1 - b0
            height_mm = max(float(ext[2]), 1e-3)
            L_vals = [float(v) for v in state.piecewise_L_mm]
            w_vals = [float(v) for v in state.piecewise_w_mm]
            internal = sorted(
                float(np.clip(b - b0[2], 0.0, height_mm))
                for b in state.piecewise_breaks_mm
            )
            full_breaks = [0.0] + internal[: max(len(L_vals) - 1, 0)] + [height_mm]
            taus = [
                float(np.asarray(tau_from_wall_thickness_mm(w_v, L_v)).ravel()[0])
                for w_v, L_v in zip(w_vals, L_vals)
            ]
            if state.geometry_source == "Primitive" and state.primitive_shape == "Cylinder":
                mesh, _pw_stats = splitp_piecewise_cylinder_single_pass(
                    radius_mm=0.5 * float(ext[0]),
                    height_mm=height_mm,
                    resolution_mm=res_mm,
                    z_breaks_mm=full_breaks,
                    L_mm=L_vals,
                    tau=taus,
                    lattice_type=canon.lower(),
                )
            else:
                mesh, _pw_stats = splitp_piecewise_box_single_pass(
                    width_x_mm=float(ext[0]),
                    depth_y_mm=float(ext[1]),
                    height_mm=height_mm,
                    resolution_mm=res_mm,
                    z_breaks_mm=full_breaks,
                    L_mm=L_vals,
                    tau=taus,
                    origin_x=float(b0[0]),
                    origin_y=float(b0[1]),
                    origin_z=0.0,
                    lattice_type=canon.lower(),
                )
                mesh.apply_translation([0.0, 0.0, float(b0[2])])

        if state.enable_surface_texture:
            _emit(65, f"Applying surface texture ({state.texture_type})...")
            tex_cfg = SurfaceTextureConfig(
                texture_type=state.texture_type,
                amplitude_mm=float(state.texture_amplitude_mm),
                wavelength_mm=float(state.texture_wavelength_mm),
                profile=state.texture_profile,
                displacement_mode=state.texture_displacement_mode,
            )
            mesh = apply_surface_texture(mesh, tex_cfg)

        if state.enable_micropillars:
            _emit(78, "Synthesizing explicit micropillar forest...")
            mp_cfg = MicropillarConfig(
                diameter_mm=float(state.pillar_diameter_um) * 1e-3,
                height_mm=float(state.pillar_height_um) * 1e-3,
                spacing_mm=float(state.pillar_spacing_um) * 1e-3,
                distribution=state.pillar_distribution,
                location=state.pillar_location,
                filter_printable=bool(state.pillar_filter_printable),
            )
            mesh = generate_micropillars(mesh, mp_cfg)

        _emit(88, "Inspecting mesh health & exporting files...")
        health = _health_to_dict(compute_mesh_health(mesh), mesh)
        manifest_params = state.to_manifest_params()
        stem = (
            state.custom_output_name.strip()
            if state.custom_output_name.strip()
            else build_implicit_output_basename(manifest_params, cad_mesh)
        )

        formats: list[str] = []
        if state.export_stl:
            formats.append("stl")
        if state.export_3mf:
            formats.append("3mf")
        output_paths: list[Path] = []
        if formats:
            base_out = out_dir / stem
            for fmt in formats:
                try:
                    exported = export_mesh(mesh, str(base_out), formats=(fmt,))
                    if hasattr(exported, "paths_written"):
                        output_paths.extend(Path(p) for p in exported.paths_written)
                    elif isinstance(exported, dict):
                        output_paths.extend(Path(p) for p in exported.values())
                    elif isinstance(exported, (list, tuple)):
                        output_paths.extend(Path(p) for p in exported)
                    else:
                        output_paths.append(Path(exported))
                except ModuleNotFoundError as exc:
                    _emit(90, f"{fmt.upper()} export skipped (missing optional module: {exc.name})")

        recipe_path: Path | None = None
        manifest_path: Path | None = None
        if state.write_manifest:
            recipe_path = state.save_recipe(out_dir / f"{stem}_recipe.json")
            manifest_path = write_implicit_parameters_manifest(
                out_dir / f"{stem}_parameters.txt",
                manifest_params,
                mesh=cad_mesh,
                output_basename=stem,
            )

        mis_pore_mm: float | None = None
        mis_conf = ""
        if state.run_post_mis_check:
            _emit(95, "Running post-mesh MIS pore check...")
            try:
                v_pitch = max(res_mm * 2.0, 0.15)
                vox = mesh.voxelized(pitch=v_pitch)
                void_mask = ~np.asarray(vox.matrix, dtype=bool)
                mis_res = compute_max_inscribed_sphere_pore_size(
                    void_mask=void_mask,
                    resolution_mm=v_pitch,
                    boundary_guard_mm=0.0,
                    require_sphere_within_domain=True,
                )
                mis_pore_mm = float(mis_res.max_diameter_mm)
                mis_conf = str(mis_res.confidence)
            except Exception as exc:
                notes.append(f"MIS post-check skipped: {exc}")

        elapsed = time.perf_counter() - t0
        _emit(100, f"Done in {elapsed:.1f}s ({len(mesh.faces):,} faces).")
        return GenerationResult(
            mesh=mesh,
            health=health,
            output_dir=out_dir,
            output_paths=output_paths,
            recipe_path=recipe_path,
            manifest_path=manifest_path,
            mis_pore_mm=mis_pore_mm,
            mis_confidence=mis_conf,
            elapsed_seconds=elapsed,
            stage_notes=notes,
        )
    finally:
        try:
            temp_cad_stl.unlink(missing_ok=True)
        except Exception:
            pass


def run_recipe_batch(
    recipe_paths: list[str | Path],
    *,
    output_root: str | Path | None = None,
    progress_cb: Callable[[int, int, Path, str], None] | None = None,
) -> list[tuple[Path, GenerationResult]]:
    """Load and execute a queue of saved JSON recipes sequentially."""
    results: list[tuple[Path, GenerationResult]] = []
    paths = [Path(p) for p in recipe_paths]
    total = len(paths)
    for idx, rpath in enumerate(paths):
        if progress_cb:
            progress_cb(idx, total, rpath, f"Loading recipe {rpath.name}...")
        state = ImplicitStudioState.load_recipe(rpath)
        cad_mesh, _ = build_or_load_cad_mesh(state)
        res = run_implicit_generation(
            state,
            cad_mesh=cad_mesh,
            output_root=output_root,
            progress_cb=(
                (lambda _pct, msg, i=idx, p=rpath: progress_cb(i, total, p, msg))
                if progress_cb
                else None
            ),
        )
        results.append((rpath, res))
    if progress_cb:
        progress_cb(total, total, paths[-1] if paths else Path("."), "Batch queue complete.")
    return results
