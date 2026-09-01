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


def _default_cases() -> tuple[dict, dict, dict]:
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
            "label": "Split-P linear grading",
            "vocal_cache": _CACHE
            / "SplitP_Cube1mm_linearGrad_L500umBottom_L1000umTop_SF33_phaseOrigin500um500um_JacobianW_cleaned_n128_flow_chamber_re5_ma0.05_conv_ms6000_tol0.001",
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
    parser.add_argument("--streamline-density", type=float, default=1.15)
    args = parser.parse_args()

    cases = _default_cases()
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
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
