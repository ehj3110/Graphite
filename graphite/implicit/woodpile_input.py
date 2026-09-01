"""
CLI / spec helpers for piecewise woodpile generation.

Mirrors ``graphite.aristo.implicit_input`` for the woodpile control-surface path.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import trimesh

from graphite.implicit.piecewise_woodpile import (
    woodpile_piecewise_box_single_pass,
    woodpile_piecewise_cylinder_single_pass,
    woodpile_piecewise_cylinder_union,
)
from graphite.math.woodpile_anchor import WoodpileAnchorMode

DomainKind = Literal["cylinder", "box"]
WoodpileGenerator = Literal["implicit", "extrude"]

# Measured MIS pore diameters from Split-P 1 mm cube @ SF=33 %
# (calibrate_tau_at_fixed_period: L=0.5 / 1.0 mm bands).
SPLITP_CUBE_MIS_PORE_BOTTOM_MM = 0.1386
SPLITP_CUBE_MIS_PORE_TOP_MM = 0.2771


def woodpile_cube_1mm_stem(*, pore_bottom_mm: float, pore_top_mm: float) -> str:
    """Filename stem for 1 mm³ cross-hatch woodpile piecewise variants."""
    pb = int(round(float(pore_bottom_mm) * 1000))
    pt = int(round(float(pore_top_mm) * 1000))
    return (
        "Woodpile_CrossHatch_Cube1mm_piecewise_"
        f"P{pb}umBottom_P{pt}umTop_SF50"
    )


@dataclass
class WoodpileLatticeSpec:
    """Piecewise woodpile / cross-hatch lattice in a cylinder or axis-aligned box."""

    domain: DomainKind = "cylinder"
    diameter_mm: float = 2.0
    width_x_mm: float = 1.0
    depth_y_mm: float = 1.0
    height_mm: float = 1.0
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    origin_z_mm: float = 0.0
    resolution_mm: float = 0.01
    true_woodpile: bool = False
    invert_solids: bool = False
    repair_mesh: bool = True
    z_breaks_mm: list[float] = field(default_factory=lambda: [0.0, 2.23, 2.31, 2.7])
    pore_mm: list[float] = field(default_factory=lambda: [0.8, 0.4, 0.2])
    anchor_mode: WoodpileAnchorMode = "center_void"
    alternate_band_orientation: bool = True
    combine_mode: str = "single-pass"
    generator: WoodpileGenerator = "extrude"
    stem: str | None = None

    def validate(self) -> None:
        if len(self.z_breaks_mm) != len(self.pore_mm) + 1:
            raise ValueError("z_breaks_mm must have len(pore_mm) + 1 entries.")
        if any(b < 0 for b in self.z_breaks_mm):
            raise ValueError("z_breaks_mm must be non-negative.")
        for a, b in zip(self.z_breaks_mm, self.z_breaks_mm[1:]):
            if b <= a:
                raise ValueError(f"Non-increasing z_breaks: {self.z_breaks_mm}")
        if self.domain == "box":
            if self.width_x_mm <= 0 or self.depth_y_mm <= 0 or self.height_mm <= 0:
                raise ValueError("Box dimensions must be positive.")
        if self.generator == "extrude":
            if self.invert_solids:
                raise ValueError("invert_solids is not supported with generator='extrude'.")
            if self.combine_mode == "union":
                raise ValueError("combine_mode='union' is not supported with generator='extrude'.")

    @property
    def z_height_mm(self) -> float:
        return float(self.z_breaks_mm[-1] - self.z_breaks_mm[0])


@dataclass
class WoodpileImplicitSpec(WoodpileLatticeSpec):
    """Deprecated alias for :class:`WoodpileLatticeSpec`."""

    def __post_init__(self) -> None:
        warnings.warn(
            "WoodpileImplicitSpec is deprecated; use WoodpileLatticeSpec",
            DeprecationWarning,
            stacklevel=2,
        )


def default_stem_from_spec(spec: WoodpileLatticeSpec) -> str:
    pores_um = "_".join(str(int(round(p * 1000))) for p in spec.pore_mm)
    mode = "TrueWoodpile" if spec.true_woodpile else "CrossHatch"
    h = spec.z_height_mm
    if spec.domain == "box":
        stem = (
            f"{mode}_Cube{spec.width_x_mm:g}x{spec.depth_y_mm:g}x{h:g}mm_"
            f"piecewise_P{pores_um}um_SF50"
        )
    else:
        stem = (
            f"{mode}_Cylinder{spec.diameter_mm:g}x{h:g}mm_"
            f"piecewise_P{pores_um}um_SF50"
        )
    if spec.generator == "extrude":
        stem = f"{stem}_extrude"
    return stem


def repair_woodpile_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Light trimesh repair after woodpile mesh generation."""
    m = mesh.copy()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(m)
    trimesh.repair.fix_inversion(m)
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    return m


def repair_implicit_woodpile_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Deprecated alias for :func:`repair_woodpile_mesh`."""
    warnings.warn(
        "repair_implicit_woodpile_mesh is deprecated; use repair_woodpile_mesh",
        DeprecationWarning,
        stacklevel=2,
    )
    return repair_woodpile_mesh(mesh)


def _build_extrude_piecewise_woodpile_mesh(
    spec: WoodpileLatticeSpec,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """2D bar union → extrude → manifold union (``graphite.explicit.woodpile_extrude``)."""
    from graphite.explicit.woodpile_extrude import (
        generate_piecewise_crosshatch_box,
        generate_piecewise_crosshatch_cylinder,
    )

    common = dict(
        z_breaks_mm=list(spec.z_breaks_mm),
        pore_mm=list(spec.pore_mm),
        origin_x_mm=float(spec.origin_x_mm),
        origin_y_mm=float(spec.origin_y_mm),
        origin_z_mm=float(spec.origin_z_mm),
        anchor_mode=spec.anchor_mode,
        alternate_band_orientation=bool(spec.alternate_band_orientation),
        true_woodpile=bool(spec.true_woodpile),
    )
    domain = spec.domain.strip().lower()
    if domain == "box":
        mesh, report = generate_piecewise_crosshatch_box(
            width_x_mm=float(spec.width_x_mm),
            depth_y_mm=float(spec.depth_y_mm),
            height_mm=float(spec.height_mm),
            **common,
        )
        report["height_mm"] = float(spec.height_mm)
    elif domain == "cylinder":
        mesh, report = generate_piecewise_crosshatch_cylinder(
            radius_mm=float(spec.diameter_mm) / 2.0,
            height_mm=float(spec.z_height_mm),
            **common,
        )
        report["height_mm"] = float(spec.z_height_mm)
        report["diameter_mm"] = float(spec.diameter_mm)
    else:
        raise ValueError(f"domain must be cylinder or box, got {spec.domain!r}.")
    report["combine_method"] = "extrude"
    report["generator_backend"] = "woodpile_extrude"
    return mesh, report


def build_piecewise_woodpile_mesh(
    spec: WoodpileLatticeSpec,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Evaluate piecewise woodpile → combined trimesh + report."""
    spec.validate()
    domain = spec.domain.strip().lower()

    if spec.generator == "extrude":
        mesh, report = _build_extrude_piecewise_woodpile_mesh(spec)
    elif domain == "box":
        if spec.combine_mode == "union":
            raise ValueError("Per-band union is not supported for box woodpile; use single-pass.")
        mesh, report = woodpile_piecewise_box_single_pass(
            width_x_mm=float(spec.width_x_mm),
            depth_y_mm=float(spec.depth_y_mm),
            height_mm=float(spec.height_mm),
            z_breaks_mm=spec.z_breaks_mm,
            pore_mm=spec.pore_mm,
            resolution_mm=float(spec.resolution_mm),
            true_woodpile=bool(spec.true_woodpile),
            invert_solids=bool(spec.invert_solids),
            anchor_mode=spec.anchor_mode,
            alternate_band_orientation=bool(spec.alternate_band_orientation),
            origin_x=float(spec.origin_x_mm),
            origin_y=float(spec.origin_y_mm),
            origin_z=float(spec.origin_z_mm),
        )
    elif domain == "cylinder":
        common = dict(
            radius_mm=float(spec.diameter_mm) / 2.0,
            z_breaks_mm=spec.z_breaks_mm,
            pore_mm=spec.pore_mm,
            resolution_mm=float(spec.resolution_mm),
            true_woodpile=bool(spec.true_woodpile),
            invert_solids=bool(spec.invert_solids),
            anchor_mode=spec.anchor_mode,
            alternate_band_orientation=bool(spec.alternate_band_orientation),
        )
        if spec.combine_mode == "union":
            mesh, report = woodpile_piecewise_cylinder_union(
                repair_mesh=bool(spec.repair_mesh),
                **common,
            )
        else:
            mesh, report = woodpile_piecewise_cylinder_single_pass(**common)
    else:
        raise ValueError(f"domain must be cylinder or box, got {spec.domain!r}.")

    report["watertight_before_repair"] = bool(mesh.is_watertight)
    if spec.repair_mesh:
        mesh = repair_woodpile_mesh(mesh)
    report["watertight_after_repair"] = bool(mesh.is_watertight)
    report["watertight"] = report["watertight_after_repair"]
    report["faces"] = int(len(mesh.faces))
    report["vertices"] = int(len(mesh.vertices))
    report["volume_mm3"] = float(mesh.volume) if mesh.is_volume else None
    report["generator"] = spec.generator

    lattice_spec = {
        "domain": domain,
        "generator": spec.generator,
        "diameter_mm": float(spec.diameter_mm),
        "width_x_mm": float(spec.width_x_mm),
        "depth_y_mm": float(spec.depth_y_mm),
        "height_mm": float(spec.height_mm),
        "origin_mm": [float(spec.origin_x_mm), float(spec.origin_y_mm), float(spec.origin_z_mm)],
        "resolution_mm": float(spec.resolution_mm),
        "true_woodpile": bool(spec.true_woodpile),
        "combine_mode": spec.combine_mode,
        "anchor_mode": spec.anchor_mode,
        "alternate_band_orientation": bool(spec.alternate_band_orientation),
        "z_breaks_mm": list(spec.z_breaks_mm),
        "pore_mm": list(spec.pore_mm),
        "target_solid_fraction": 0.5,
        "solid_fraction_note": "Woodpile pitch=2*pore; alternating layers ≈50% in-plane.",
    }
    report["lattice_spec"] = lattice_spec
    report["implicit_spec"] = lattice_spec
    report["stem"] = spec.stem or default_stem_from_spec(spec)
    return mesh, report


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def add_piecewise_woodpile_args(parser: argparse.ArgumentParser) -> None:
    """Register piecewise woodpile CLI flags."""
    parser.add_argument(
        "--domain",
        choices=("cylinder", "box"),
        default="cylinder",
        help="Domain shape (default: cylinder).",
    )
    parser.add_argument("--diameter-mm", type=float, default=2.0)
    parser.add_argument("--width-x-mm", type=float, default=1.0)
    parser.add_argument("--depth-y-mm", type=float, default=1.0)
    parser.add_argument("--height-mm", type=float, default=1.0)
    parser.add_argument("--origin-x-mm", type=float, default=0.0)
    parser.add_argument("--origin-y-mm", type=float, default=0.0)
    parser.add_argument("--origin-z-mm", type=float, default=0.0)
    parser.add_argument("--resolution-mm", type=float, default=0.01)
    parser.add_argument(
        "--z-breaks-mm",
        default="0,2.23,2.31,2.7",
        help="Comma-separated Z breakpoints (mm), length = n_bands + 1.",
    )
    parser.add_argument(
        "--pore-mm",
        default="0.8,0.4,0.2",
        help="Comma-separated pore sizes (mm) per band, bottom to top.",
    )
    parser.add_argument(
        "--true-woodpile",
        action="store_true",
        help="Use shifted true woodpile (true_woodpile=True). Default is cross-hatch.",
    )
    parser.add_argument(
        "--no-alternate-band-orientation",
        action="store_true",
        help="Keep X/Y layer parity the same in every Z band (not recommended for piecewise).",
    )
    parser.add_argument("--invert-solids", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument(
        "--anchor-mode",
        choices=("edge_solid", "center_void", "default"),
        default="center_void",
        help="Transverse phase anchor per band (default: center_void = symmetric ±struts).",
    )
    parser.add_argument(
        "--union",
        action="store_true",
        help="(Deprecated) Legacy per-band mesh + manifold union (cylinder only).",
    )
    parser.add_argument(
        "--generator",
        choices=("implicit", "extrude"),
        default="extrude",
        help=(
            "Mesh backend: extrude = 2D bar extrusion (default, sharp square struts); "
            "implicit = marching cubes reference meshes."
        ),
    )


def woodpile_spec_from_args(args: argparse.Namespace) -> WoodpileLatticeSpec:
    z_breaks = args.z_breaks_mm
    pores = args.pore_mm
    if isinstance(z_breaks, str):
        z_breaks = _parse_float_list(z_breaks)
    if isinstance(pores, str):
        pores = _parse_float_list(pores)
    if bool(getattr(args, "union", False)):
        warnings.warn(
            "--union is deprecated; use single-pass (default) or generator=extrude",
            DeprecationWarning,
            stacklevel=2,
        )
    return WoodpileLatticeSpec(
        domain=str(getattr(args, "domain", "cylinder")),  # type: ignore[arg-type]
        diameter_mm=float(args.diameter_mm),
        width_x_mm=float(getattr(args, "width_x_mm", 1.0)),
        depth_y_mm=float(getattr(args, "depth_y_mm", 1.0)),
        height_mm=float(getattr(args, "height_mm", 1.0)),
        origin_x_mm=float(getattr(args, "origin_x_mm", 0.0)),
        origin_y_mm=float(getattr(args, "origin_y_mm", 0.0)),
        origin_z_mm=float(getattr(args, "origin_z_mm", 0.0)),
        resolution_mm=float(args.resolution_mm),
        true_woodpile=bool(getattr(args, "true_woodpile", False)),
        invert_solids=bool(getattr(args, "invert_solids", False)),
        repair_mesh=not bool(getattr(args, "no_repair", False)),
        z_breaks_mm=list(z_breaks),
        pore_mm=list(pores),
        anchor_mode=str(getattr(args, "anchor_mode", "center_void")),  # type: ignore[arg-type]
        alternate_band_orientation=not bool(
            getattr(args, "no_alternate_band_orientation", False)
        ),
        combine_mode="union" if bool(getattr(args, "union", False)) else "single-pass",
        generator=str(getattr(args, "generator", "extrude")),  # type: ignore[arg-type]
        stem=getattr(args, "stem", None),
    )


def woodpile_spec_from_config(params: dict[str, Any]) -> WoodpileLatticeSpec:
    """Build spec from ``generate_lattice.py`` YAML/JSON params."""
    z_breaks = params.get("z_breaks_mm") or params.get("woodpile_z_breaks_mm")
    pores = params.get("pore_mm") or params.get("woodpile_pore_mm")
    if z_breaks is None or pores is None:
        raise ValueError("Woodpile piecewise config requires z_breaks_mm and pore_mm.")

    combine_mode = str(params.get("combine_mode", "single-pass"))
    if combine_mode == "union":
        warnings.warn(
            "combine_mode='union' is deprecated; use single-pass or generator=extrude",
            DeprecationWarning,
            stacklevel=2,
        )

    domain = str(params.get("domain", "cylinder")).strip().lower()
    return WoodpileLatticeSpec(
        domain=domain,  # type: ignore[arg-type]
        diameter_mm=float(params.get("diameter_mm", params.get("cylinder_diameter_mm", 2.0))),
        width_x_mm=float(params.get("width_x_mm", 1.0)),
        depth_y_mm=float(params.get("depth_y_mm", 1.0)),
        height_mm=float(params.get("height_mm", params.get("domain_height_mm", 1.0))),
        origin_x_mm=float(params.get("origin_x_mm", 0.0)),
        origin_y_mm=float(params.get("origin_y_mm", 0.0)),
        origin_z_mm=float(params.get("origin_z_mm", 0.0)),
        resolution_mm=float(params.get("resolution", 0.01)),
        true_woodpile=bool(params.get("true_woodpile", False)),
        invert_solids=bool(params.get("invert_solids", False)),
        repair_mesh=bool(params.get("repair_mesh", True)),
        z_breaks_mm=[float(z) for z in z_breaks],
        pore_mm=[float(p) for p in pores],
        anchor_mode=str(params.get("anchor_mode", "center_void")),  # type: ignore[arg-type]
        alternate_band_orientation=bool(params.get("alternate_band_orientation", True)),
        combine_mode=combine_mode,
        generator=str(params.get("generator", "extrude")),  # type: ignore[arg-type]
    )
