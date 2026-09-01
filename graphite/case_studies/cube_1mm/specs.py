"""
Paths, stems, and default physics for the 1 mm³ cube case study.

Canonical output root::

    outputs/case_studies/cube_1mm/

Legacy (pre-port)::

    experiments/implicit_to_volume/output/
"""

from __future__ import annotations

from pathlib import Path

from graphite.implicit.woodpile_input import (
    SPLITP_CUBE_MIS_PORE_BOTTOM_MM,
    SPLITP_CUBE_MIS_PORE_TOP_MM,
    woodpile_cube_1mm_stem,
)

# --- Domain / implicit ---
CUBE_SIZE_MM = 1.0
BAND_HEIGHT_MM = 0.5
SPLITP_L_BOTTOM_MM = 0.5
SPLITP_L_TOP_MM = 1.0
SPLITP_SF = 0.33
IMPLICIT_RESOLUTION_MM = 0.015
# Slightly finer MC pitch for case-study TPMS regen with Boolean face trim.
IMPLICIT_RESOLUTION_BOOLEAN_TRIM_MM = 0.010
# Oversize margin (mm) around the design cube before CAD Boolean ∩.
BOOLEAN_TRIM_MARGIN_MM = 0.04
SPLITP_PHASE_ORIGIN_MM = (0.5, 0.5)
SPLITP_PIECEWISE_BOTTOM_BAND_PHASE_SHIFT_X_MM = 0.25 * SPLITP_L_BOTTOM_MM

# --- Aristo defaults ---
SPLITP_ARISTO_H_MM = 0.005
WOODPILE_ARISTO_H_MM = 0.008  # implicit / marching-cubes reference mesh
WOODPILE_EXTRUDE_ARISTO_H_MM = 0.025  # structured-surface extrude path (stable Pardiso solve)
ARISTO_E_MPA = 25.8
ARISTO_FORCE_N = 1.0

# --- Vocal defaults ---
VOCAL_TARGET_N = 128
VOCAL_RE = 5.0
VOCAL_MA = 0.05
VOCAL_STEPS_PREVIEW = 1000
VOCAL_CHECK_INTERVAL = 50
VOCAL_CONTINUE_EXTRA_STEPS = 500
VOCAL_TOLERANCE_CONVERGE = 0.001
VOCAL_MAX_STEPS_CONVERGE = 6000
WOODPILE_VOCAL_CONTINUE_EXTRA_STEPS = 1000

# --- Woodpile pore presets (mm) ---
WOODPILE_PORE_BOTTOM_DEFAULT_MM = 0.2
WOODPILE_PORE_TOP_DEFAULT_MM = 0.4


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_output_dir() -> Path:
    return repo_root() / "outputs" / "case_studies" / "cube_1mm"


def figures_dir(out_dir: Path | None = None) -> Path:
    return (out_dir or default_output_dir()) / "figures"


def geometry_dir(out_dir: Path | None = None) -> Path:
    return (out_dir or default_output_dir()) / "geometry"


def fea_dir(out_dir: Path | None = None) -> Path:
    return (out_dir or default_output_dir()) / "fea"


def fluid_dir(out_dir: Path | None = None) -> Path:
    return (out_dir or default_output_dir()) / "fluid"


def legacy_output_dir() -> Path:
    return repo_root() / "experiments" / "implicit_to_volume" / "output"


def vocal_metrics_dir(out_dir: Path | None = None) -> Path:
    return fluid_dir(out_dir) / "vocal_metrics"


def vocal_cache_dir() -> Path:
    return repo_root() / "outputs" / "vocal" / "cache"


def scripts_dir() -> Path:
    return repo_root() / "scripts"


def torch_python() -> Path:
    candidate = repo_root() / ".venv_torch" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path("python")


def splitp_piecewise_stem(*, solid_fraction: float = SPLITP_SF) -> str:
    sf_pct = int(round(float(solid_fraction) * 100))
    return (
        f"SplitP_Cube1mm_piecewise_L{int(SPLITP_L_BOTTOM_MM * 1000)}umBottom_"
        f"L{int(SPLITP_L_TOP_MM * 1000)}umTop_SF{sf_pct}"
    )


def splitp_piecewise_phase_shift_x_stem(*, solid_fraction: float = SPLITP_SF) -> str:
    """Piecewise cube with bottom-band lateral phase +L_bottom/4 in X."""
    return f"{splitp_piecewise_stem(solid_fraction=solid_fraction)}_phaseTest_qL4_shift_x"


def splitp_raw_fea_stem(raw_stem: str, *, h_mm: float = SPLITP_ARISTO_H_MM) -> str:
    h_token = f"{int(round(h_mm * 1000)):03d}"
    return f"{raw_stem}_h{h_token}_1N_E25p8MPa"


def splitp_piecewise_phase_shift_x_fea_stem(*, h_mm: float = SPLITP_ARISTO_H_MM) -> str:
    return splitp_raw_fea_stem(splitp_piecewise_phase_shift_x_stem(), h_mm=h_mm)


def _vocal_run_id(
    *,
    steps: int | None = None,
    converge: bool = False,
    max_steps: int = VOCAL_MAX_STEPS_CONVERGE,
    tolerance: float = VOCAL_TOLERANCE_CONVERGE,
) -> str:
    if converge:
        return f"conv_ms{int(max_steps)}_tol{tolerance:g}"
    if steps is None:
        raise ValueError("steps required when converge=False")
    return f"steps{int(steps)}"


def vocal_cache_slug(
    stem_cleaned: str,
    *,
    steps: int | None = None,
    converge: bool = False,
    max_steps: int = VOCAL_MAX_STEPS_CONVERGE,
    tolerance: float = VOCAL_TOLERANCE_CONVERGE,
) -> str:
    run_id = _vocal_run_id(
        steps=steps,
        converge=converge,
        max_steps=max_steps,
        tolerance=tolerance,
    )
    return (
        f"{stem_cleaned}_n{VOCAL_TARGET_N}_flow_chamber_"
        f"re{VOCAL_RE:g}_ma{VOCAL_MA:g}_{run_id}"
    )


def splitp_piecewise_phase_shift_x_vocal_cache_slug(
    *,
    steps: int = VOCAL_STEPS_PREVIEW,
    converge: bool = False,
) -> str:
    stem = f"{splitp_piecewise_phase_shift_x_stem()}_cleaned"
    if converge:
        return vocal_cache_slug(stem, converge=True)
    return vocal_cache_slug(stem, steps=steps)


def splitp_linear_graded_vocal_cache_slug(*, converge: bool = True, steps: int = 2000) -> str:
    stem = f"{splitp_linear_graded_stem()}_cleaned"
    if converge:
        return vocal_cache_slug(stem, converge=True)
    return vocal_cache_slug(stem, steps=steps)


def splitp_linear_graded_stem(
    *,
    solid_fraction: float = SPLITP_SF,
    phase_origin_x_mm: float = SPLITP_PHASE_ORIGIN_MM[0],
    phase_origin_y_mm: float = SPLITP_PHASE_ORIGIN_MM[1],
    jacobian: bool = True,
) -> str:
    sf_pct = int(round(float(solid_fraction) * 100))
    stem = (
        f"SplitP_Cube1mm_linearGrad_L{int(SPLITP_L_BOTTOM_MM * 1000)}umBottom_"
        f"L{int(SPLITP_L_TOP_MM * 1000)}umTop_SF{sf_pct}_"
        f"phaseOrigin{int(phase_origin_x_mm * 1000)}um"
        f"{int(phase_origin_y_mm * 1000)}um"
    )
    if jacobian:
        stem += "_JacobianW"
    return stem


def woodpile_stem_default() -> str:
    return woodpile_cube_1mm_stem(
        pore_bottom_mm=WOODPILE_PORE_BOTTOM_DEFAULT_MM,
        pore_top_mm=WOODPILE_PORE_TOP_DEFAULT_MM,
    )


def woodpile_stem_splitp_match() -> str:
    return woodpile_cube_1mm_stem(
        pore_bottom_mm=SPLITP_CUBE_MIS_PORE_BOTTOM_MM,
        pore_top_mm=SPLITP_CUBE_MIS_PORE_TOP_MM,
    )


def resolve_woodpile_stem(
    *,
    match_splitp_pores: bool = False,
    stem: str | None = None,
    generator: str = "extrude",
) -> str:
    if stem:
        base = str(stem)
    elif match_splitp_pores:
        base = woodpile_stem_splitp_match()
    else:
        base = woodpile_stem_default()
    if generator == "extrude" and not base.endswith("_extrude"):
        return f"{base}_extrude"
    return base


def stl_path(out_dir: Path, stem: str, *, cleaned: bool = False) -> Path:
    suffix = "_cleaned" if cleaned else ""
    return out_dir / f"{stem}{suffix}.stl"


def splitp_piecewise_stl(out_dir: Path | None = None, *, cleaned: bool = False) -> Path:
    return stl_path(out_dir or default_output_dir(), splitp_piecewise_stem(), cleaned=cleaned)


def splitp_piecewise_phase_shift_x_stl(out_dir: Path | None = None, *, cleaned: bool = False) -> Path:
    return stl_path(
        out_dir or default_output_dir(),
        splitp_piecewise_phase_shift_x_stem(),
        cleaned=cleaned,
    )


def splitp_linear_graded_stl(out_dir: Path | None = None, *, cleaned: bool = False) -> Path:
    return stl_path(
        out_dir or default_output_dir(),
        splitp_linear_graded_stem(),
        cleaned=cleaned,
    )


def woodpile_stl(out_dir: Path | None, *, match_splitp_pores: bool = False, stem: str | None = None) -> Path:
    root = out_dir or default_output_dir()
    return root / f"{resolve_woodpile_stem(match_splitp_pores=match_splitp_pores, stem=stem)}.stl"


def woodpile_stl_cleaned(
    out_dir: Path | None,
    *,
    match_splitp_pores: bool = False,
    stem: str | None = None,
) -> Path:
    root = out_dir or default_output_dir()
    s = resolve_woodpile_stem(match_splitp_pores=match_splitp_pores, stem=stem)
    return root / f"{s}_cleaned.stl"


def woodpile_fea_stem(raw_stem: str, *, h_mm: float = WOODPILE_ARISTO_H_MM) -> str:
    h_token = str(h_mm).replace(".", "p")
    return f"{raw_stem}_h{h_token}_1N_E25p8MPa"


def woodpile_vocal_cache_slug(
    *,
    match_splitp_pores: bool = True,
    generator: str = "extrude",
    steps: int = 2000,
    converge: bool = False,
    max_steps: int | None = None,
    tolerance: float = VOCAL_TOLERANCE_CONVERGE,
) -> str:
    stem = f"{resolve_woodpile_stem(match_splitp_pores=match_splitp_pores, generator=generator)}_cleaned"
    if converge:
        cap = int(max_steps if max_steps is not None else steps)
        return vocal_cache_slug(stem, converge=True, max_steps=cap, tolerance=tolerance)
    return vocal_cache_slug(stem, steps=steps)


def splitp_fea_stem(variant: str, *, h_mm: float = SPLITP_ARISTO_H_MM) -> str:
    h_token = f"{int(round(h_mm * 1000)):03d}"
    if variant == "piecewise":
        return f"SplitP_Cube1mm_piecewise_h{h_token}_1N_E25p8MPa"
    return f"SplitP_Cube1mm_linearGrad_centerPhase_h{h_token}_1N_E25p8MPa"


def vocal_comparison_png(out_dir: Path | None = None) -> Path:
    return figures_dir(out_dir) / "SplitP_Cube1mm_vocal_cross_section_XZ_midY.png"


def vocal_comparison_summary_json(out_dir: Path | None = None) -> Path:
    return (out_dir or default_output_dir()) / "SplitP_Cube1mm_vocal_comparison_summary.json"
