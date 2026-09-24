"""V2.1 surface-first compare — thin defaults over shared V2 pipeline.

Defaults (see ``docs/SC_CONFORMAL_COMPARE_V2_1_PLAN.md``):
  - ``dual_mode="exposed_corners"``
  - ``stitch_gates="off"`` (nearest dual + valence cap)
  - ``cut_inset_factor=0.0`` (cuts on chord surface crossing)
  - ``valence_cap=3`` for grid corner local degree
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from graphite.explicit.compare_v2_surface_first.generate import generate_compare_v2

DEFAULT_CELL_SIZE: tuple[float, float, float] = (12.0, 12.0, 4.0)
DEFAULT_STRUT_RADIUS: float = 0.6
DEFAULT_VOLUME_FRACTION: float = 0.5
DEFAULT_RULE_NAME: str = "grid"
DEFAULT_CUT_INSET_FACTOR: float = 0.0
DEFAULT_DUAL_MODE: str = "exposed_corners"
DEFAULT_STITCH_GATES: str = "off"
DEFAULT_VALENCE_CAP: int = 3
VERSION_LABEL: str = "v2_1_surface_first"


def generate_compare_v2_1(
    cad_mesh: trimesh.Trimesh | str | Path,
    *,
    cell_size: float | tuple[float, float, float] = DEFAULT_CELL_SIZE,
    volume_fraction_threshold: float = DEFAULT_VOLUME_FRACTION,
    rule_name: str = DEFAULT_RULE_NAME,
    strut_radius: float = DEFAULT_STRUT_RADIUS,
    cut_inset_factor: float = DEFAULT_CUT_INSET_FACTOR,
    dual_mode: str = DEFAULT_DUAL_MODE,
    stitch_gates: str = DEFAULT_STITCH_GATES,
    valence_cap: int | None = DEFAULT_VALENCE_CAP,
    stitch_max_horizontal: float | None = None,
    stitch_max_angle_from_vertical_deg: float = 60.0,
    boolean_trim_volume: bool = True,
    boolean_trim_union: bool = False,
    export_dir: str | Path | None = None,
    stem: str | None = None,
    export_stages: bool = False,
    skip_solidify: bool = False,
    cylinder_segments: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run V2.1 surface-first compare with hardened-cut / simple-stitch defaults."""
    # Grid local corner degree is 3; octahedral face-center nodes are 4.
    if valence_cap is None and str(rule_name).lower() in ("octahedral", "oct"):
        valence_cap = 4
    elif valence_cap is None:
        valence_cap = DEFAULT_VALENCE_CAP

    return generate_compare_v2(
        cad_mesh,
        cell_size=cell_size,
        volume_fraction_threshold=volume_fraction_threshold,
        rule_name=rule_name,
        strut_radius=strut_radius,
        cut_inset_factor=cut_inset_factor,
        dual_mode=dual_mode,
        stitch_gates=stitch_gates,
        valence_cap=valence_cap,
        stitch_max_horizontal=stitch_max_horizontal,
        stitch_max_angle_from_vertical_deg=stitch_max_angle_from_vertical_deg,
        boolean_trim_volume=boolean_trim_volume,
        boolean_trim_union=boolean_trim_union,
        export_dir=export_dir,
        stem=stem or "compare_v2_1_surface_first",
        export_stages=export_stages,
        skip_solidify=skip_solidify,
        cylinder_segments=cylinder_segments,
        version_label=VERSION_LABEL,
        **kwargs,
    )
