"""
graphite/explicit/interlinked/quick.py

User-Friendly High-Level API for Graphite Interlinked & Polycatenated Metamaterials.

Designed for designers, researchers, and engineers who want print-ready interlinked
lattices and chainmail fabrics without needing deep expertise in crystallography,
sublattice symmetries, or numerical optimization.

Key Features:
- Presets: "chainmail", "space_fabric", "shock_absorber", "impact_cushion", "cube_cages", "octahedra".
- Risk Profiles:
    * "low_risk" (default): 1.0 mm struts, 200 um clearance (safe, high reliability).
    * "high_risk": 0.75 mm struts, 150 um clearance (fine features, tight tolerances).
- Dedicated One-Liners: chainmail(), space_fabric(), shock_absorber(), impact_cushion().
- Forgiving Argument Resolution: automatically resolves cell tags, preset names, and colloquial aliases.
- Natural Sizing: accepts scalar, 2D, or 3D bounding envelopes as 'size' or 'bounding_box'.
- Industrial Instanced 3MF Export: one-line save to ultra-compact .3mf or .stl.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Any, Literal
import numpy as np

from .generator import (
    InterlinkedConfig,
    InterlinkedLatticeResult,
    generate_interlinked_lattice,
)
from .cell import InterlinkedRegistry, InterlinkedCell
from .dimensioning import PAMDimensioner, PAMDimensionResult

RISK_PROFILES: dict[str, dict[str, Any]] = {
    "low_risk": {
        "name": "Low Risk",
        "strut_diameter": 1.00,
        "clearance": 0.200,
        "description": "1.0 mm struts, 200 um clearance. Robust, high reliability, easy post-processing.",
    },
    "high_risk": {
        "name": "High Risk",
        "strut_diameter": 0.75,
        "clearance": 0.150,
        "description": "0.75 mm struts, 150 um clearance. Fine feature resolution, lighter weight, requires dialed-in printer calibration.",
    },
}

PROFILE_ALIASES: dict[str, str] = {
    "low_risk": "low_risk",
    "low": "low_risk",
    "safe": "low_risk",
    "robust": "low_risk",
    "high_risk": "high_risk",
    "high": "high_risk",
    "fine": "high_risk",
    "aggressive": "high_risk",
}

PRESETS = {
    "chainmail": {
        "cell": "euro_4in1",
        "description": "Flexible European 4-in-1 maille weave fabric (2D sheet)",
        "is_2d": True,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
    "space_fabric": {
        "cell": "nasa_hexagon",
        "description": "NASA JPL 6-fold spiral interlocking chainmail sheet (2D sheet)",
        "is_2d": True,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
    "shock_absorber": {
        "cell": "d4tet",
        "description": "Diamond regular tetrahedral dual cages for compliant energy absorption (Zhou et al., Science 2025)",
        "is_2d": False,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
    "impact_cushion": {
        "cell": "c6tt",
        "description": "Truncated tetrahedra on Simple Cubic with 6-fold face catenation",
        "is_2d": False,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
    "cube_cages": {
        "cell": "d4cube",
        "description": "Diamond bipartite cubic wireframe cages",
        "is_2d": False,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
    "octahedra": {
        "cell": "j4oct",
        "description": "Square planar octahedra with alternating 45 deg twist",
        "is_2d": False,
        "default_strut_d": 1.00,
        "default_clearance": 0.200,
    },
}

CELL_ALIASES: dict[str, str] = {
    # 2D fabrics
    "chainmail": "euro_4in1",
    "european": "euro_4in1",
    "maille": "euro_4in1",
    "fabric": "euro_4in1",
    "space_fabric": "nasa_hexagon",
    "space": "nasa_hexagon",
    "nasa": "nasa_hexagon",
    "nasa_fabric": "nasa_hexagon",
    "kusari": "kusari",
    "japanese": "kusari",
    # 3D lattices
    "shock_absorber": "d4tet",
    "shock": "d4tet",
    "absorber": "d4tet",
    "tet": "d4tet",
    "tetrahedron": "d4tet",
    "tetrahedral": "d4tet",
    "diamond_tet": "d4tet",
    "impact_cushion": "c6tt",
    "cushion": "c6tt",
    "impact": "c6tt",
    "truncated_tet": "c6tt",
    "cube_cages": "d4cube",
    "cube": "d4cube",
    "cubes": "d4cube",
    "diamond_cube": "d4cube",
    "octahedra": "j4oct",
    "octahedron": "j4oct",
    "rings": "j4ring",
    "ring": "j4ring",
}

PREFERRED_SEEDING: dict[str, str] = {
    "d4tet": "diamond",
    "d4cube": "diamond",
    "d4ct": "diamond",
    "d4ico": "diamond",
    "j4ring": "diamond",
    "b8co": "bcc",
    "c6tt": "cartesian",
    "c6co": "cartesian",
    "j4oct": "cartesian",
    "t6ring": "cartesian",
    "t6hex": "cartesian",
    "euro_4in1": "cartesian",
    "european_4in1": "cartesian",
    "kusari": "cartesian",
    "japanese_kusari": "cartesian",
    "nasa_hexagon": "hexagonal",
    "nasa_space_fabric": "hexagonal",
}


def _resolve_target(name_or_alias: str | None) -> tuple[str, bool, dict[str, Any] | None]:
    """
    Intelligently resolve cell tag, is_2d flag, and preset dict from any user input.
    Returns: (cell_key, is_2d, preset_info_or_none)
    """
    if not name_or_alias:
        return "d4tet", False, PRESETS.get("shock_absorber")

    raw = name_or_alias.lower().strip()

    # 1. Direct preset match
    if raw in PRESETS:
        p_info = PRESETS[raw]
        return p_info["cell"], p_info.get("is_2d", False), p_info

    # 2. Alias resolution
    if raw in CELL_ALIASES:
        resolved = CELL_ALIASES[raw]
        if resolved in PRESETS:
            p_info = PRESETS[resolved]
            return p_info["cell"], p_info.get("is_2d", False), p_info
        is_2d = resolved in ("euro_4in1", "nasa_hexagon", "kusari")
        return resolved, is_2d, None

    # 3. Direct cell key or registered cell
    is_2d = raw in ("euro_4in1", "nasa_hexagon", "kusari")
    return raw, is_2d, None


def quick_lattice(
    cell: str | None = None,
    preset: str | None = None,
    profile: Literal["low_risk", "high_risk", "low", "high"] | str | None = None,
    bounding_box: tuple[float, float, float] | Sequence[float] | None = None,
    size: tuple[float, ...] | Sequence[float] | float | None = None,
    grid_size: tuple[int, int, int] | Sequence[int] | None = None,
    strut_diameter: float | None = None,
    clearance: float | None = None,
    strut: float | None = None,
    gap: float | None = None,
    solid_fraction: float | None = None,
    add_frame: bool = False,
    frame_thickness: float = 2.0,
    frame_margin: float = 0.60,
    export_path: str | Path | None = None,
    save_preview: str | Path | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Generate an explicit interlinked lattice or metamaterial with zero boilerplate.

    Args:
        cell: Name of the cell or preset (e.g. 'd4tet', 'c6tt', 'shock_absorber', 'chainmail').
        preset: Optional explicit preset name ('chainmail', 'space_fabric', 'shock_absorber', 'impact_cushion').
        profile: Risk profile: 'low_risk' (1.0 mm strut, 200 um clearance, default) or
                 'high_risk' (0.75 mm strut, 150 um clearance).
        bounding_box: Physical target envelope (Lx, Ly, Lz) in mm. Automatically solves unit cell pitch
                      and computes integer grid repetitions (nx, ny, nz).
        size: Natural alias for bounding_box. Accepts scalar (cube/square), 2D (Lx, Ly), or 3D (Lx, Ly, Lz).
        grid_size: Explicit lattice repetition counts (nx, ny, nz) if bounding_box/size is not specified.
        strut_diameter: Strut outer diameter in mm (overrides profile if given).
        clearance: Minimum inter-cage surface gap in mm (overrides profile if given).
        strut: Alias for strut_diameter.
        gap: Alias for clearance.
        solid_fraction: Optional target solid volume fraction (e.g. 0.10 for 10%). Solves pitch and strut diameter simultaneously.
        add_frame: If True, generates a rigid solid perimeter border for tensile testing/handling.
        frame_thickness: Perimeter border wall thickness in mm (default: 2.0 mm).
        frame_margin: Gap between outer struts and perimeter border in mm (default: 0.60 mm).
        export_path: Optional destination file path (.3mf or .stl) for instant automatic export.
        save_preview: Optional destination PNG path for automatic scientific matte preview render.
        strict: If True, validates clearance and DfAM pre-flight checks before allocating geometry.

    Returns:
        InterlinkedLatticeResult with watertight mesh, particles, physical clearance, and export methods.
    """
    # 1. Resolve cell and preset
    target_name = preset or cell or "d4tet"
    effective_cell, is_2d, p_info = _resolve_target(target_name)

    # 2. Resolve risk profile vs explicit overrides
    eff_profile = "low_risk"
    if profile is not None:
        p_key = profile.lower().strip()
        if p_key in PROFILE_ALIASES:
            eff_profile = PROFILE_ALIASES[p_key]
        else:
            valid_p = ", ".join(list(RISK_PROFILES.keys()))
            raise ValueError(f"Unknown risk profile '{profile}'. Available profiles: {valid_p}")

    prof_data = RISK_PROFILES[eff_profile]
    base_strut = prof_data["strut_diameter"]
    base_clearance = prof_data["clearance"]

    # Explicit parameter overrides take precedence
    d_wire_arg = strut if strut is not None else strut_diameter
    clr_arg = gap if gap is not None else clearance

    resolved_strut_d = float(d_wire_arg) if d_wire_arg is not None else base_strut
    clr_target = float(clr_arg) if clr_arg is not None else base_clearance

    # 3. Resolve cell class from registry
    cell_cls = InterlinkedRegistry.get(effective_cell)
    cell_inst = cell_cls() if callable(cell_cls) else cell_cls

    # 4. Determine auto-seeding
    seeding_type = PREFERRED_SEEDING.get(effective_cell, getattr(cell_inst, "preferred_seeding_type", "cartesian"))

    # 5. Dimensioning & Pitch Resolution
    if solid_fraction is not None:
        dim_res = PAMDimensioner.solve_from_clearance_and_vf(
            cell=cell_inst,
            clearance=clr_target,
            solid_fraction=float(solid_fraction),
            raise_on_jamming=strict,
        )
        resolved_pitch = dim_res.pitch
        resolved_strut_d = dim_res.wire_diameter
    else:
        resolved_pitch = cell_inst.resolve_pitch(
            target_clearance=clr_target,
            strut_diameter=resolved_strut_d,
        )

    # 6. Resolve Bounding Envelope & Grid Repetitions
    effective_box = bounding_box
    if effective_box is None and size is not None:
        if isinstance(size, (int, float)):
            effective_box = (float(size), float(size), 1.0) if is_2d else (float(size), float(size), float(size))
        elif len(size) == 1:
            val = float(size[0])
            effective_box = (val, val, 1.0) if is_2d else (val, val, val)
        elif len(size) == 2:
            effective_box = (float(size[0]), float(size[1]), 1.0)
        elif len(size) == 3:
            effective_box = (float(size[0]), float(size[1]), float(size[2]))

    if effective_box is not None:
        box = [float(v) for v in effective_box]
        if len(box) != 3:
            raise ValueError(f"bounding_box/size must define 3 dimensions (Lx, Ly, Lz) in mm, got {effective_box}")
        nx = max(1, int(round(box[0] / resolved_pitch)))
        ny = max(1, int(round(box[1] / resolved_pitch)))
        nz = 1 if is_2d else max(1, int(round(box[2] / resolved_pitch)))
        effective_grid = (nx, ny, nz)
    elif grid_size is not None:
        effective_grid = tuple(int(v) for v in grid_size)
    else:
        # Default grid counts
        effective_grid = (8, 8, 1) if is_2d else (3, 3, 3)

    # 7. Configure InterlinkedConfig
    cfg = InterlinkedConfig(
        cell=cell_inst,
        seeding_type=seeding_type,
        pitch=resolved_pitch,
        wire_radius=resolved_strut_d * 0.5,
        min_clearance=clr_target,
        auto_resolve_pitch=False,
        grid_size=effective_grid,
        add_perimeter_frame=add_frame,
        frame_wall_thickness=frame_thickness,
        frame_margin=frame_margin,
        strict=strict,
    )

    # 8. Generate Lattice
    result = generate_interlinked_lattice(cfg, check_clearance=True)

    # 9. Export Deliverables if requested
    if export_path is not None:
        result.export(export_path)

    if save_preview is not None:
        try:
            result.render(
                save_preview,
                title=f"{effective_cell.upper()} Interlinked Lattice",
                subtitle=f"Profile: {prof_data['name']} | Grid: {effective_grid} | Strut: {resolved_strut_d:.2f} mm | Clearance: {result.min_clearance*1000:.1f} um",
            )
        except Exception:
            pass

    return result


def quick_fabric(
    pattern: Literal["chainmail", "space_fabric", "kusari"] = "chainmail",
    width: float = 60.0,
    length: float = 60.0,
    profile: str = "low_risk",
    strut_diameter: float | None = None,
    clearance: float | None = None,
    add_frame: bool = False,
    export_path: str | Path | None = None,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Convenience helper for generating 2D flexible textile and chainmail fabrics.

    Args:
        pattern: 'chainmail' (European 4-in-1), 'space_fabric' (NASA JPL hex), or 'kusari'.
        width: Physical dimension in X (mm).
        length: Physical dimension in Y (mm).
        profile: Risk profile ('low_risk' or 'high_risk').
        strut_diameter: Wire outer diameter in mm (overrides profile).
        clearance: Surface-to-surface gap in mm (overrides profile).
        add_frame: If True, adds a solid handling perimeter frame.
        export_path: Optional destination file path (.3mf or .stl).

    Returns:
        InterlinkedLatticeResult.
    """
    p_map = {
        "chainmail": "euro_4in1",
        "space_fabric": "nasa_hexagon",
        "kusari": "kusari",
    }
    target_cell = p_map.get(pattern.lower().strip(), "euro_4in1")
    return quick_lattice(
        cell=target_cell,
        size=(width, length),
        profile=profile,
        strut_diameter=strut_diameter,
        clearance=clearance,
        add_frame=add_frame,
        export_path=export_path,
        **kwargs,
    )


# =============================================================================
# Dedicated One-Liner Presets
# =============================================================================

def chainmail(
    width: float = 60.0,
    length: float = 60.0,
    profile: Literal["low_risk", "high_risk", "low", "high"] | str = "low_risk",
    strut_diameter: float | None = None,
    clearance: float | None = None,
    add_frame: bool = False,
    export_path: str | Path | None = None,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Generate a flexible European 4-in-1 chainmail fabric sheet in one line.

    Args:
        width: Sheet dimension in X (mm).
        length: Sheet dimension in Y (mm).
        profile: 'low_risk' (1.0 mm strut, 200 um gap) or 'high_risk' (0.75 mm strut, 150 um gap).
        strut_diameter: Explicit wire diameter in mm (overrides profile).
        clearance: Explicit surface gap in mm (overrides profile).
        add_frame: If True, adds a solid handling perimeter border.
        export_path: Destination file path (.3mf or .stl).

    Returns:
        InterlinkedLatticeResult.
    """
    return quick_lattice(
        cell="euro_4in1",
        size=(width, length),
        profile=profile,
        strut_diameter=strut_diameter,
        clearance=clearance,
        add_frame=add_frame,
        export_path=export_path,
        **kwargs,
    )


def space_fabric(
    width: float = 80.0,
    length: float = 80.0,
    profile: Literal["low_risk", "high_risk", "low", "high"] | str = "low_risk",
    strut_diameter: float | None = None,
    clearance: float | None = None,
    add_frame: bool = False,
    export_path: str | Path | None = None,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Generate a NASA JPL 6-fold spiral interlocking space fabric sheet in one line.

    Args:
        width: Sheet dimension in X (mm).
        length: Sheet dimension in Y (mm).
        profile: 'low_risk' (1.0 mm strut, 200 um gap) or 'high_risk' (0.75 mm strut, 150 um gap).
        strut_diameter: Explicit wire diameter in mm (overrides profile).
        clearance: Explicit surface gap in mm (overrides profile).
        add_frame: If True, adds a solid handling perimeter border.
        export_path: Destination file path (.3mf or .stl).

    Returns:
        InterlinkedLatticeResult.
    """
    return quick_lattice(
        cell="nasa_hexagon",
        size=(width, length),
        profile=profile,
        strut_diameter=strut_diameter,
        clearance=clearance,
        add_frame=add_frame,
        export_path=export_path,
        **kwargs,
    )


def shock_absorber(
    size: tuple[float, ...] | Sequence[float] | float = (40.0, 40.0, 40.0),
    profile: Literal["low_risk", "high_risk", "low", "high"] | str = "low_risk",
    strut_diameter: float | None = None,
    clearance: float | None = None,
    add_frame: bool = False,
    export_path: str | Path | None = None,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Generate a 3D compliant shock absorber lattice (Diamond tetrahedral dual cages) in one line.

    Args:
        size: Bounding envelope (Lx, Ly, Lz) in mm, or scalar for a cubic block.
        profile: 'low_risk' (1.0 mm strut, 200 um gap) or 'high_risk' (0.75 mm strut, 150 um gap).
        strut_diameter: Explicit wire diameter in mm (overrides profile).
        clearance: Explicit surface gap in mm (overrides profile).
        add_frame: If True, adds a solid handling perimeter border.
        export_path: Destination file path (.3mf or .stl).

    Returns:
        InterlinkedLatticeResult.
    """
    return quick_lattice(
        cell="d4tet",
        size=size,
        profile=profile,
        strut_diameter=strut_diameter,
        clearance=clearance,
        add_frame=add_frame,
        export_path=export_path,
        **kwargs,
    )


def impact_cushion(
    size: tuple[float, ...] | Sequence[float] | float = (50.0, 50.0, 25.0),
    profile: Literal["low_risk", "high_risk", "low", "high"] | str = "low_risk",
    strut_diameter: float | None = None,
    clearance: float | None = None,
    add_frame: bool = False,
    export_path: str | Path | None = None,
    **kwargs: Any,
) -> InterlinkedLatticeResult:
    """
    Generate a 3D high-clearance impact cushion lattice (C-6-TT truncated tetrahedra) in one line.

    Args:
        size: Bounding envelope (Lx, Ly, Lz) in mm, or scalar for a cubic block.
        profile: 'low_risk' (1.0 mm strut, 200 um gap) or 'high_risk' (0.75 mm strut, 150 um gap).
        strut_diameter: Explicit wire diameter in mm (overrides profile).
        clearance: Explicit surface gap in mm (overrides profile).
        add_frame: If True, adds a solid handling perimeter border.
        export_path: Destination file path (.3mf or .stl).

    Returns:
        InterlinkedLatticeResult.
    """
    return quick_lattice(
        cell="c6tt",
        size=size,
        profile=profile,
        strut_diameter=strut_diameter,
        clearance=clearance,
        add_frame=add_frame,
        export_path=export_path,
        **kwargs,
    )


# =============================================================================
# Discovery & Exploration Helpers
# =============================================================================

def list_presets() -> list[dict[str, Any]]:
    """Return catalog of available high-level presets."""
    summary = []
    for k, v in PRESETS.items():
        summary.append({
            "preset": k,
            "cell": v["cell"],
            "type": "2D Sheet" if v.get("is_2d") else "3D Lattice",
            "default_strut_diameter": v["default_strut_d"],
            "default_clearance": v["default_clearance"],
            "description": v["description"],
        })
    return summary


def list_cells() -> list[dict[str, Any]]:
    """Return catalog of all registered PAM unit cells."""
    return InterlinkedRegistry.list_cells()


def describe_cell(name: str) -> dict[str, Any]:
    """Retrieve detailed metadata, coordination, and scaling metrics for a specific cell."""
    target_tag, _, _ = _resolve_target(name)
    cell_cls = InterlinkedRegistry.get(target_tag)
    inst = cell_cls() if callable(cell_cls) else cell_cls
    kappa = PAMDimensioner.get_kappa(inst)
    c_geom = PAMDimensioner.compute_c_geom(inst)
    vf_max = c_geom * (kappa ** 2)
    return {
        "tag": target_tag,
        "name": getattr(inst, "name", target_tag),
        "family": getattr(inst, "family", "unknown"),
        "parent_network": getattr(inst, "parent_network", "pcu"),
        "coordination_number": getattr(inst, "coordination_number", 0),
        "kappa": kappa,
        "c_geom": c_geom,
        "jamming_limit_vf_percent": vf_max * 100.0,
    }
