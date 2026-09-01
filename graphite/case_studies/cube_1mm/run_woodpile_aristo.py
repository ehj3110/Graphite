"""Aristo FEA pipeline for piecewise cross-hatch woodpile 1 mm³ cube."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from graphite.case_studies.cube_1mm.generate_woodpile import (
    cube_1mm_woodpile_spec,
    generate_woodpile_cube,
)
from graphite.explicit.woodpile_structured_surface import (
    structured_surface_mesh_from_spec,
)
from graphite.aristo.volume_mesh_inspect import remove_floating_islands
from graphite.implicit.woodpile_input import WoodpileLatticeSpec
from graphite.case_studies.cube_1mm.specs import (
    ARISTO_E_MPA,
    ARISTO_FORCE_N,
    WOODPILE_ARISTO_H_MM,
    WOODPILE_EXTRUDE_ARISTO_H_MM,
    default_output_dir,
    repo_root,
    resolve_woodpile_stem,
    scripts_dir,
    woodpile_fea_stem,
)

# Structured surface mesh targets h directly; no Loop subdivide pass needed.
WOODPILE_EXTRUDE_TOP_MIN_VERTICES = 10


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], *, desc: str) -> None:
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=repo_root(), check=True)


def _prepare_extrude_stl_for_fea(
    *,
    out_dir: Path,
    stem_raw: str,
    spec: WoodpileLatticeSpec,
    h_mm: float,
) -> Path:
    """Uniform bar-surface tessellation for gmsh single_surface volume fill."""
    mesh, meta = structured_surface_mesh_from_spec(spec, edge_length_mm=float(h_mm))
    mesh, island_meta = remove_floating_islands(mesh, min_volume_mm3=1e-6)
    fea_stl = out_dir / f"{stem_raw}_fea_surface.stl"
    mesh.export(fea_stl)
    print(
        f"  extrude structured surface: {meta['n_bars']} bars, "
        f"{meta['faces']:,} faces, "
        f"edge median={meta.get('edge_length_mm_median', 0)*1000:.1f} µm, "
        f"watertight={meta['watertight']} -> {fea_stl.name}"
    )
    if island_meta["n_islands_removed"]:
        print(
            f"  island cleanup: removed {island_meta['n_islands_removed']} "
            f"component(s)"
        )
    return fea_stl


def _default_h_fea(generator: str) -> float:
    if generator == "extrude":
        return WOODPILE_EXTRUDE_ARISTO_H_MM
    return WOODPILE_ARISTO_H_MM


def run_woodpile_aristo(
    *,
    out_dir: Path | None = None,
    match_splitp_pores: bool = False,
    stem: str | None = None,
    generator: str = "extrude",
    h_fea: float | None = None,
    skip_fea: bool = False,
    skip_plot: bool = False,
    plot_only: bool = False,
    fill_mode: str = "element-slice-soft",
) -> int:
    h_fea = float(h_fea) if h_fea is not None else _default_h_fea(generator)
    out = out_dir or default_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    stem_raw = resolve_woodpile_stem(
        match_splitp_pores=match_splitp_pores,
        stem=stem,
        generator=generator,
    )
    stem_fea = woodpile_fea_stem(stem_raw, h_mm=h_fea)

    stl_raw = out / f"{stem_raw}.stl"
    stl_clean = out / f"{stem_raw}_cleaned.stl"
    vtu = out / f"{stem_fea}_1N_aristo_fea.vtu"
    report = out / f"{stem_fea}_1N_aristo_report.json"
    fill_token = fill_mode.replace("-", "")
    fig_xz = out / f"{stem_raw}_aristo_cross_section_XZ_midY_{fill_token}.png"
    fig_yz = out / f"{stem_raw}_aristo_cross_section_YZ_x300um_{fill_token}.png"
    fig_xy = out / f"{stem_raw}_aristo_cross_section_XY_midZ_{fill_token}.png"
    plot_script = scripts_dir() / "plot_aristo_cross_sections.py"

    if plot_only:
        if not vtu.is_file():
            print(f"VTU not found: {vtu}", file=sys.stderr)
            return 1
        for plane, center, half, fig, label in (
            ("xz", "0.5", "0.012", fig_xz, "Plot XZ cross-section @ mid-Y"),
            ("yz", "0.3", "0.012", fig_yz, "Plot YZ cross-section @ X=0.3 mm"),
        ):
            cmd = [
                _py(),
                str(plot_script),
                "--vtu",
                str(vtu),
                "--label",
                "Cross-hatch woodpile",
                "--plane",
                plane,
                "--slice-center-mm",
                center,
                "--half-thickness-mm",
                half,
                "--fill-mode",
                fill_mode,
                "--clip-unit-cube",
                "--output",
                str(fig),
            ]
            if report.is_file():
                cmd.extend(["--report", str(report)])
            _run(cmd, desc=label)
        if report.is_file():
            _run(
                [
                    _py(),
                    str(plot_script),
                    "--vtu",
                    str(vtu),
                    "--label",
                    "Cross-hatch woodpile",
                    "--report",
                    str(report),
                    "--plane",
                    "xy",
                    "--slice-center-mm",
                    "0.5",
                    "--half-thickness-mm",
                    "0.05",
                    "--fill-mode",
                    fill_mode,
                    "--clip-unit-cube",
                    "--output",
                    str(fig_xy),
                ],
                desc="Plot XY cross-section @ band interface",
            )
        print(f"\nWrote {fig_xz}")
        return 0

    if not stl_raw.is_file():
        generate_woodpile_cube(
            out_dir=out,
            match_splitp_pores=match_splitp_pores,
            generator=generator,
        )

    _run(
        [
            _py(),
            str(scripts_dir() / "aristo_clean_and_mesh_stl.py"),
            "--stl",
            str(stl_raw),
            "--h",
            str(h_fea),
            "--out-dir",
            str(out),
            "--skip-volume-inspect",
        ],
        desc="Clean STL + island removal",
    )

    stl_fea = stl_clean
    if generator == "extrude":
        spec = cube_1mm_woodpile_spec(
            match_splitp_pores=match_splitp_pores,
            generator=generator,
            stem=stem_raw,
        )
        stl_fea = _prepare_extrude_stl_for_fea(
            out_dir=out,
            stem_raw=stem_raw,
            spec=spec,
            h_mm=h_fea,
        )

    if skip_fea:
        print(f"\nSkipped FEA. Cleaned STL: {stl_clean}  FEA STL: {stl_fea}")
        return 0

    fea_cmd = [
        _py(),
        str(scripts_dir() / "run_aristo_fea.py"),
        "--stl",
        str(stl_fea),
        "--h",
        str(h_fea),
        "--force-n",
        str(ARISTO_FORCE_N),
        "--youngs-modulus",
        str(ARISTO_E_MPA),
        "--mesh-mode",
        "single_surface",
        "--solver",
        "pardiso",
        "--bc-load-mode",
        "flat_top_vertex_plane",
        "--top-min-vertices",
        str(
            WOODPILE_EXTRUDE_TOP_MIN_VERTICES
            if generator == "extrude"
            else 100
        ),
        "--stem",
        stem_fea,
        "--out-dir",
        str(out),
        "--no-viz",
    ]
    _run(fea_cmd, desc="Aristo compression FEA (1 N, E=25.8 MPa)")

    if not skip_plot:
        for plane, center, half, fig, label in (
            ("xz", "0.5", "0.012", fig_xz, "Plot XZ cross-section @ mid-Y"),
            ("yz", "0.3", "0.012", fig_yz, "Plot YZ cross-section @ X=0.3 mm"),
        ):
            cmd = [
                _py(),
                str(plot_script),
                "--vtu",
                str(vtu),
                "--label",
                "Cross-hatch woodpile",
                "--plane",
                plane,
                "--slice-center-mm",
                center,
                "--half-thickness-mm",
                half,
                "--fill-mode",
                fill_mode,
                "--clip-unit-cube",
                "--output",
                str(fig),
            ]
            if report.is_file():
                cmd.extend(["--report", str(report)])
            _run(cmd, desc=label)
        if report.is_file():
            _run(
                [
                    _py(),
                    str(plot_script),
                    "--vtu",
                    str(vtu),
                    "--label",
                    "Cross-hatch woodpile",
                    "--report",
                    str(report),
                    "--plane",
                    "xy",
                    "--slice-center-mm",
                    "0.5",
                    "--half-thickness-mm",
                    "0.05",
                    "--fill-mode",
                    fill_mode,
                    "--clip-unit-cube",
                    "--output",
                    str(fig_xy),
                ],
                desc="Plot XY cross-section @ band interface",
            )

    print("\nWoodpile Aristo complete.")
    print(f"  VTU:    {vtu}")
    print(f"  Report: {report}")
    print(f"  Figure (XZ mid-Y): {fig_xz}")
    print(f"  Figure (YZ @ X=0.3 mm): {fig_yz}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fea", action="store_true")
    parser.add_argument(
        "--h",
        type=float,
        default=None,
        help=(
            "FEA mesh size (mm). Default: "
            f"{WOODPILE_EXTRUDE_ARISTO_H_MM} extrude, {WOODPILE_ARISTO_H_MM} implicit."
        ),
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--fill-mode",
        default="element-slice-soft",
        choices=(
            "element-slice",
            "element-slice-soft",
            "voronoi-sharp",
            "voronoi-soft",
            "voronoi-bounded",
            "voronoi-bounded-soft",
            "blur-heatmap",
            "voronoi-disk",
        ),
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--stem", type=str, default=None)
    parser.add_argument("--match-splitp-pores", action="store_true")
    parser.add_argument(
        "--generator",
        choices=("implicit", "extrude"),
        default="extrude",
        help="Woodpile mesh backend (default: extrude).",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_woodpile_aristo(
        out_dir=args.out_dir or default_output_dir(),
        match_splitp_pores=bool(args.match_splitp_pores),
        stem=args.stem,
        generator=str(args.generator),
        h_fea=args.h,
        skip_fea=bool(args.skip_fea),
        skip_plot=bool(args.skip_plot),
        plot_only=bool(args.plot_only),
        fill_mode=args.fill_mode,
    )


if __name__ == "__main__":
    raise SystemExit(main())
