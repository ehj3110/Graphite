#!/usr/bin/env python
"""
Three-panel Vocal WSS comparison for the 1 mm³ cube case study.

Recomputes wall shear stress from cached ``f.pkl`` (no LBM re-simulation) and
overlays velocity streamlines on the XZ mid-plane slice.

Run from repo root with ``.venv_torch``::

    .venv_torch\\Scripts\\python.exe scripts\\plot_cube_1mm_vocal_wss_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.lbm.vocal_run import plot_wss_comparison

_CACHE = _REPO_ROOT / "outputs" / "vocal" / "cache"
_FIGURES = _REPO_ROOT / "outputs" / "case_studies" / "cube_1mm" / "figures"


_LINEAR_N128 = (
    "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_"
    "phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_"
    "re5_ma0.05_conv_ms6000_tol0.001"
)
_LINEAR_N192 = (
    "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_"
    "phaseOrigin500um500um_JacobianW_cleaned_n192_flow_chamber_"
    "re5_ma0.05_conv_ms6000_tol0.001"
)


def _default_cases(*, linear_n: int = 192) -> tuple[dict, dict, dict]:
    linear_slug = _LINEAR_N192 if int(linear_n) >= 192 else _LINEAR_N128
    linear_label = (
        "Split-P linear grading (n=192)"
        if int(linear_n) >= 192
        else "Split-P linear grading (n=128)"
    )
    return (
        {
            "label": "Cross-hatch woodpile (extrude)",
            "vocal_cache": _CACHE
            / "Woodpile_CrossHatch_Cube1mm_piecewise_P139umBottom_P277umTop_SF50_extrude_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms3000_tol0.001",
        },
        {
            "label": "Split-P piecewise (bottom X +L/4)",
            "vocal_cache": _CACHE
            / "SplitP_Cube1mm_piecewise_L500umBottom_L1000umTop_SF33_phaseTest_qL4_shift_x_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001",
        },
        {
            "label": linear_label,
            "vocal_cache": _CACHE / linear_slug,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_FIGURES / "Cube1mm_Vocal_WSS_3panel_comparison.png",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--shared-color-scale",
        action="store_true",
        help="Use one WSS color scale across all panels (default: per-panel peak).",
    )
    parser.add_argument(
        "--linear-n",
        type=int,
        choices=(128, 192),
        default=192,
        help="Vocal grid resolution for the linear graded Split-P column (default: 192).",
    )
    parser.add_argument(
        "--single-linear-output",
        type=Path,
        default=_FIGURES / "SplitP_Cube1mm_linearGrad_n192_vocal_WSS_XZ_midY.png",
        help="Also write a one-panel WSS figure for linear graded only (default: enabled path).",
    )
    parser.add_argument("--no-single-linear", action="store_true")
    parser.add_argument("--streamline-density", type=float, default=1.15)
    parser.add_argument(
        "--wss-smooth-sigma",
        type=float,
        default=0.0,
        help="Optional Gaussian display smoothing (voxels). Prefer --wss-display-factor.",
    )
    parser.add_argument(
        "--wss-display-factor",
        type=int,
        default=1,
        help=(
            "Upsample the XZ slice for display only (e.g. 4 → 4× finer colormap pixels). "
            "Voxel axes and cached metrics stay at the simulation resolution."
        ),
    )
    parser.add_argument(
        "--wss-display-order",
        type=int,
        choices=(1, 3),
        default=1,
        help="Interpolation order for display upsampling: 1=bilinear, 3=bicubic.",
    )
    parser.add_argument(
        "--stl-cross-section",
        action="store_true",
        help=(
            "Draw smooth walls from the source STL slice and stamp interpolated "
            "Vocal WSS onto that geometry (display only)."
        ),
    )
    args = parser.parse_args()

    cases = _default_cases(linear_n=int(args.linear_n))
    for case in cases:
        if not Path(case["vocal_cache"]).is_dir():
            raise FileNotFoundError(f"Vocal cache not found: {case['vocal_cache']}")

    out = plot_wss_comparison(
        cases,
        output_path=args.output,
        dpi=args.dpi,
        shared_color_scale=bool(args.shared_color_scale),
        streamline_density=float(args.streamline_density),
        device=args.device,
        wss_smooth_sigma=float(args.wss_smooth_sigma),
        wss_display_factor=int(args.wss_display_factor),
        wss_display_order=int(args.wss_display_order),
        use_stl_cross_section=bool(args.stl_cross_section),
    )
    print(f"Wrote {out}")

    if not args.no_single_linear and args.single_linear_output is not None:
        linear_case = cases[2]
        single_out = plot_wss_comparison(
            (linear_case,),
            output_path=args.single_linear_output,
            dpi=args.dpi,
            shared_color_scale=False,
            streamline_density=float(args.streamline_density),
            device=args.device,
            suptitle=f"Vocal WSS — {linear_case['label']} (XZ mid-plane)",
            wss_smooth_sigma=float(args.wss_smooth_sigma),
            wss_display_factor=int(args.wss_display_factor),
            wss_display_order=int(args.wss_display_order),
            use_stl_cross_section=bool(args.stl_cross_section),
        )
        print(f"Wrote {single_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
