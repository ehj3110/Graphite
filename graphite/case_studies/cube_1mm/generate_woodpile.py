"""Generate piecewise cross-hatch woodpile 1 mm³ cube STLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import trimesh

from graphite.case_studies.cube_1mm.specs import (
    BAND_HEIGHT_MM,
    CUBE_SIZE_MM,
    IMPLICIT_RESOLUTION_MM,
    WOODPILE_PORE_BOTTOM_DEFAULT_MM,
    WOODPILE_PORE_TOP_DEFAULT_MM,
    default_output_dir,
    resolve_woodpile_stem,
)
from graphite.implicit.woodpile_input import (
    SPLITP_CUBE_MIS_PORE_BOTTOM_MM,
    SPLITP_CUBE_MIS_PORE_TOP_MM,
    WoodpileLatticeSpec,
    build_piecewise_woodpile_mesh,
)
from graphite.math.woodpile import evaluate_woodpile
from graphite.math.woodpile_anchor import dominant_strut_axis_at_z


def _parse_pore_mm(text: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--pore-mm expects two comma-separated values (bottom, top).")
    return parts[0], parts[1]


def _preview_mesh_png(mesh: trimesh.Trimesh, path: Path, title: str) -> None:
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, window_size=(960, 720))
    plotter.add_mesh(
        pv.wrap(mesh),
        color="#6b8e75",
        smooth_shading=True,
        specular=0.35,
        specular_power=20,
    )
    plotter.add_text(title, position="upper_left", font_size=11, color="black")
    plotter.view_isometric()
    plotter.camera.zoom(1.15)
    plotter.background_color = "white"
    plotter.screenshot(str(path))
    plotter.close()


def _field_xz_slice_png(
    *,
    spec: WoodpileLatticeSpec,
    report: dict,
    path: Path,
) -> None:
    res = float(spec.resolution_mm)
    wx = float(spec.width_x_mm)
    hz = float(spec.height_mm)
    y_mid = float(spec.origin_y_mm) + float(spec.depth_y_mm) / 2.0
    x = np.arange(spec.origin_x_mm, spec.origin_x_mm + wx + res * 0.5, res)
    z = np.arange(spec.origin_z_mm, spec.origin_z_mm + hz + res * 0.5, res)
    X, Z = np.meshgrid(x, z, indexing="ij")
    Y = np.full_like(X, y_mid)

    field = np.full(X.shape, np.inf, dtype=np.float64)
    for slab in report["slabs"]:
        z0, z1 = float(slab["z0_mm"]), float(slab["z1_mm"])
        mask = (Z >= z0 - 1e-12) & (Z <= z1 + 1e-12)
        band = evaluate_woodpile(
            X,
            Y,
            Z,
            pore_size=float(slab["pore_mm"]),
            true_woodpile=bool(spec.true_woodpile),
            origin_x=float(slab["origin_x_mm"]),
            origin_y=float(slab["origin_y_mm"]),
            flip_layer_parity=bool(slab["flip_layer_parity"]),
            swap_xy=bool(slab["swap_xy"]),
            z_layer_origin=float(slab["z0_mm"]),
            layer_index_offset=int(slab.get("layer_index_offset", 0)),
        )
        field = np.where(mask, band, field)

    solid = field <= 0.0
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.imshow(
        solid.T,
        origin="lower",
        extent=[x.min(), x.max(), z.min(), z.max()],
        cmap="gray_r",
        aspect="equal",
        interpolation="nearest",
    )
    ax.axhline(0.5, color="#e67e22", lw=1.0, ls="--", label="Band interface Z=0.5 mm")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_title(
        "Cross-hatch implicit field (XZ @ mid-Y)\n"
        "dark = solid strut | light = void"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cube_1mm_woodpile_spec(
    *,
    resolution_mm: float = IMPLICIT_RESOLUTION_MM,
    pore_bottom_mm: float | None = None,
    pore_top_mm: float | None = None,
    match_splitp_pores: bool = False,
    generator: str = "extrude",
    stem: str | None = None,
) -> WoodpileLatticeSpec:
    """Build :class:`WoodpileLatticeSpec` for the 1 mm³ piecewise cube case study."""
    band_h = BAND_HEIGHT_MM
    size_mm = CUBE_SIZE_MM
    if match_splitp_pores:
        pb = SPLITP_CUBE_MIS_PORE_BOTTOM_MM
        pt = SPLITP_CUBE_MIS_PORE_TOP_MM
    elif pore_bottom_mm is not None and pore_top_mm is not None:
        pb, pt = float(pore_bottom_mm), float(pore_top_mm)
    else:
        pb = WOODPILE_PORE_BOTTOM_DEFAULT_MM
        pt = WOODPILE_PORE_TOP_DEFAULT_MM

    resolved_stem = stem or resolve_woodpile_stem(
        match_splitp_pores=match_splitp_pores,
        generator=generator,
    )
    return WoodpileLatticeSpec(
        domain="box",
        width_x_mm=size_mm,
        depth_y_mm=size_mm,
        height_mm=size_mm,
        origin_x_mm=0.0,
        origin_y_mm=0.0,
        origin_z_mm=0.0,
        resolution_mm=float(resolution_mm),
        true_woodpile=False,
        z_breaks_mm=[0.0, band_h, size_mm],
        pore_mm=[pb, pt],
        anchor_mode="center_void",
        alternate_band_orientation=True,
        combine_mode="single-pass",
        repair_mesh=True,
        generator=generator,  # type: ignore[arg-type]
        stem=resolved_stem,
    )


def generate_woodpile_cube(
    *,
    out_dir: Path | None = None,
    resolution_mm: float = IMPLICIT_RESOLUTION_MM,
    pore_bottom_mm: float | None = None,
    pore_top_mm: float | None = None,
    match_splitp_pores: bool = False,
    generator: str = "extrude",
) -> dict:
    """Build piecewise cross-hatch woodpile cube STL + QC artifacts."""
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)

    stem = resolve_woodpile_stem(
        match_splitp_pores=match_splitp_pores,
        generator=generator,
    )
    spec = cube_1mm_woodpile_spec(
        resolution_mm=float(resolution_mm),
        pore_bottom_mm=pore_bottom_mm,
        pore_top_mm=pore_top_mm,
        match_splitp_pores=match_splitp_pores,
        generator=generator,
        stem=stem,
    )
    pb, pt = float(spec.pore_mm[0]), float(spec.pore_mm[1])
    size_mm = CUBE_SIZE_MM
    band_h = BAND_HEIGHT_MM

    print("Meshing piecewise cross-hatch woodpile cube ...")
    print(f"  generator: {spec.generator}")
    print(f"  domain: [0,{size_mm}]³ mm  res={spec.resolution_mm} mm")
    print(
        f"  bands: Z=[0,{band_h}] pore={pb*1000:.1f} µm | "
        f"Z=[{band_h},{size_mm}] pore={pt*1000:.1f} µm"
    )
    mesh, report = build_piecewise_woodpile_mesh(spec)

    stem = spec.stem or report["stem"]
    stl_path = out / f"{stem}.stl"
    png_path = out / f"{stem}_preview.png"
    slice_path = out / f"{stem}_field_XZ_midY.png"
    json_path = out / f"{stem}_generation_report.json"

    mesh.export(stl_path)
    _preview_mesh_png(
        mesh,
        png_path,
        "Piecewise cross-hatch woodpile 1 mm³\n"
        f"bottom pore={pb*1000:.0f} µm | top pore={pt*1000:.1f} µm | SF≈50 %",
    )
    _field_xz_slice_png(spec=spec, report=report, path=slice_path)

    report["stl_path"] = str(stl_path)
    report["preview_png"] = str(png_path)
    report["field_slice_png"] = str(slice_path)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Wrote {stl_path} ({len(mesh.faces):,} faces)")
    print(f"  watertight={report.get('watertight_after_repair', report.get('watertight'))}")
    print(f"  volume_mm3={report.get('volume_mm3')}")
    print(f"Wrote {png_path}")
    print(f"Wrote {slice_path}")
    print(f"Wrote {json_path}")

    for slab in report["slabs"]:
        z_mid = 0.5 * (float(slab["z0_mm"]) + float(slab["z1_mm"]))
        axis = dominant_strut_axis_at_z(
            z_mid,
            float(slab["pore_mm"]),
            flip_layer_parity=bool(slab["flip_layer_parity"]),
            swap_xy=bool(slab["swap_xy"]),
            z_layer_origin_mm=float(slab["z0_mm"]),
            layer_index_offset=int(slab.get("layer_index_offset", 0)),
        )
        print(
            f"  band {slab['band_index']}: pore={slab['pore_mm']*1000:.0f} µm "
            f"origin=({slab['origin_x_mm']:.3f}, {slab['origin_y_mm']:.3f}) "
            f"swap_xy={slab['swap_xy']} strut_axis={axis}"
        )
    report["stem"] = stem
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-mm", type=float, default=IMPLICIT_RESOLUTION_MM)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pore-mm", type=_parse_pore_mm, default=None)
    parser.add_argument(
        "--match-splitp-pores",
        action="store_true",
        help=(
            "Use Split-P MIS pore diameters "
            f"({SPLITP_CUBE_MIS_PORE_BOTTOM_MM*1000:.1f} / "
            f"{SPLITP_CUBE_MIS_PORE_TOP_MM*1000:.1f} µm)."
        ),
    )
    parser.add_argument(
        "--generator",
        choices=("implicit", "extrude"),
        default="extrude",
        help="Mesh backend (default: extrude; implicit = marching cubes reference).",
    )
    args = parser.parse_args(argv)
    pore_bottom = pore_top = None
    if args.pore_mm is not None:
        pore_bottom, pore_top = args.pore_mm
    generate_woodpile_cube(
        out_dir=args.out_dir or default_output_dir(),
        resolution_mm=float(args.resolution_mm),
        pore_bottom_mm=pore_bottom,
        pore_top_mm=pore_top,
        match_splitp_pores=bool(args.match_splitp_pores),
        generator=str(args.generator),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
