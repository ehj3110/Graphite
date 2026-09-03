"""
Graphite CLI - Generate Textured TPMS Scaffold

Command-line tool to generate macro-scale TPMS scaffolds with micro-scale surface
textures (microgrooves, micropillars/microfibers, or both combined).

Usage Examples
--------------
# Generate 1mm Gyroid unit cell with 50um rectangular microgrooves:
python scripts/generate_textured_tpms.py --size 1.0 --lattice gyroid --pore-size 0.8 \\
    --solid-fraction 0.50 --resolution 0.010 --texture microgrooves \\
    --groove-profile rectangular -o outputs/textures/my_grooved_gyroid.stl

# Generate 1mm Gyroid unit cell with 100x400um micropillars:
python scripts/generate_textured_tpms.py --size 1.0 --lattice gyroid --pore-size 0.8 \\
    --solid-fraction 0.50 --resolution 0.010 --texture micropillars \\
    --pillar-diameter 0.10 --pillar-height 0.40 --pillar-spacing 0.25 \\
    -o outputs/textures/my_pillared_gyroid.stl

# Generate combined microgrooves + micropillars:
python scripts/generate_textured_tpms.py --size 1.0 --lattice gyroid --pore-size 0.8 \\
    --solid-fraction 0.50 --resolution 0.010 --texture both \\
    --groove-profile rectangular --pillar-diameter 0.10 --pillar-height 0.40 \\
    -o outputs/textures/my_combined_gyroid.stl
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from graphite.geometry.masking import axis_aligned_box_grid, axis_aligned_box_sdf, voxelize_mesh_and_edt
from graphite.implicit.density_control import period_mm_from_sizing
from graphite.implicit.meshing_backends import extract_isosurface
from graphite.implicit.micropillars import MicropillarConfig, generate_micropillars
from graphite.implicit.surface_textures import SurfaceTextureConfig, apply_surface_texture
from graphite.math.tpms import evaluate_tpms


def build_scaffold(
    *,
    boundary_stl: str | Path | None = None,
    size_mm: float = 1.0,
    lattice_type: str = "gyroid",
    pore_size_mm: float = 0.80,
    solid_fraction: float = 0.50,
    resolution_mm: float = 0.010,
    texture: str = "none",
    groove_profile: str = "rectangular",
    groove_depth_mm: float = 0.050,
    groove_wavelength_mm: float = 0.050,
    groove_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    pillar_diameter_mm: float = 0.100,
    pillar_height_mm: float = 0.400,
    pillar_spacing_mm: float = 0.300,
    filter_printable_pillars: bool = False,
    output_path: str | Path = "outputs/textures/scaffold.stl",
) -> trimesh.Trimesh:
    """Build and export a textured TPMS scaffold."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    print("=" * 65)
    print(" GRAPHITE TEXTURED TPMS ENGINE ".center(65, "="))
    print(f"Lattice: {lattice_type.capitalize()}, Solid Fraction: {solid_fraction:.2f}")
    print(f"Pore Size: {pore_size_mm*1e3:.0f} um, Resolution: {resolution_mm*1e3:.1f} um")
    print(f"Texture Mode: {texture.upper()}")
    print("=" * 65)

    # 1. Grid & CAD SDF
    if boundary_stl is not None and Path(boundary_stl).exists():
        print(f"[1/4] Voxelizing input boundary mesh: {boundary_stl}...")
        cad_mesh = trimesh.load(str(boundary_stl), force="mesh")
        X, Y, Z, cad_sdf, _, _, _, _, _ = voxelize_mesh_and_edt(cad_mesh, resolution=resolution_mm)
        spacing = (resolution_mm, resolution_mm, resolution_mm)
        origin = np.array([X.min(), Y.min(), Z.min()])
    else:
        half = size_mm / 2.0
        print(f"[1/4] Generating {size_mm:.1f} mm cube grid at {resolution_mm*1e3:.1f} um pitch...")
        X, Y, Z, origin, spacing = axis_aligned_box_grid(
            size_mm, size_mm, size_mm, resolution_mm,
            origin_x=-half, origin_y=-half, origin_z=-half,
            pad_voxels=4,
        )
        cad_sdf = axis_aligned_box_sdf(
            X, Y, Z,
            origin_x=-half, origin_y=-half, origin_z=-half,
            width_x_mm=size_mm, depth_y_mm=size_mm, height_z_mm=size_mm,
        )

    # 2. TPMS Field & Isosurface Extraction
    L = period_mm_from_sizing(pore_size_mm=pore_size_mm, unit_cell_size_mm=None, solid_fraction_for_pore_mapping=solid_fraction)
    k = 2.0 * np.pi / L
    print(f"[2/4] Evaluating {lattice_type} field (L={L:.3f} mm)...")
    F_raw = evaluate_tpms(lattice_type, k, X, Y, Z)
    
    # Quantile tau calibration for exact solid fraction
    tau = float(np.quantile(np.abs(F_raw), solid_fraction))
    solid_field = np.abs(F_raw) - tau
    final_field = np.maximum(solid_field, cad_sdf)

    print(f"[3/4] Meshing base scaffold ({spacing[0]*1e3:.1f} um voxel pitch)...")
    t_m = time.perf_counter()
    iso = extract_isosurface(final_field, spacing=spacing, origin=origin, enforce_watertight=True)
    mesh = iso.mesh
    print(f"      Base mesh: {len(mesh.faces):,} faces, watertight={mesh.is_watertight} in {time.perf_counter()-t_m:.2f}s")

    # 3. Apply Textures
    tex_mode = texture.strip().lower()
    
    # Step A: Apply Microgrooves if requested
    if tex_mode in ("microgrooves", "grooves", "both", "all"):
        print(f"[4/4] Applying microgrooves ({groove_profile}, depth={groove_depth_mm*1e3:.0f}um, pitch={groove_wavelength_mm*1e3:.0f}um)...")
        groove_cfg = SurfaceTextureConfig(
            texture_type="microgrooves",
            amplitude_mm=groove_depth_mm / 2.0, # +/- amplitude = total depth
            wavelength_mm=groove_wavelength_mm,
            direction=groove_direction,
            profile=groove_profile,
            displacement_mode="centered",
            project_transverse_normal=True,
            target_edge_length_mm=resolution_mm,
            max_faces=5_000_000,
        )
        mesh = apply_surface_texture(mesh, groove_cfg)

    # Step B: Apply Micropillars if requested
    if tex_mode in ("micropillars", "pillars", "both", "all"):
        print(f"[4/4] Generating micropillars ({pillar_distribution}, d={pillar_diameter_mm*1e3:.0f}um, h={pillar_height_mm*1e3:.0f}um, spacing={pillar_spacing_mm*1e3:.0f}um)...")
        pillar_cfg = MicropillarConfig(
            diameter_mm=pillar_diameter_mm,
            height_mm=pillar_height_mm,
            spacing_mm=pillar_spacing_mm,
            circular_segments=8,
            embed_depth_mm=0.010,
            distribution=pillar_distribution,
            filter_printable=filter_printable_pillars,
            max_angle_from_horizontal_deg=max_overhang_angle_deg,
            location=pillar_location,
            selected_faces=pillar_faces,
            max_pillars=50_000,
        )
        mesh = generate_micropillars(mesh, pillar_cfg)

    # 4. Export
    mesh.export(str(out_path))
    file_size_mb = out_path.stat().st_size / (1024 ** 2)
    total_time = time.perf_counter() - t_start

    print("=" * 65)
    print(" SCAFFOLD EXPORTED SUCCESSFULLY ".center(65, "="))
    print(f"Output: {out_path}")
    print(f"File Size: {file_size_mb:.1f} MB | Faces: {len(mesh.faces):,} | Watertight: {mesh.is_watertight}")
    print(f"Total Execution Time: {total_time:.2f}s")
    print("=" * 65)

    return mesh


def main():
    parser = argparse.ArgumentParser(description="Graphite CLI - Generate Textured TPMS Scaffold")
    parser.add_argument("boundary_stl", nargs="?", default=None, help="Path to boundary STL (optional)")
    parser.add_argument("--size", type=float, default=1.0, help="Scaffold cube dimension in mm (default: 1.0)")
    parser.add_argument("--lattice", type=str, default="gyroid", help="TPMS lattice type (default: gyroid)")
    parser.add_argument("--pore-size", type=float, default=0.80, help="Pore size in mm (default: 0.80)")
    parser.add_argument("--solid-fraction", type=float, default=0.50, help="Solid volume fraction (default: 0.50)")
    parser.add_argument("--resolution", type=float, default=0.015, help="Voxel resolution in mm (default: 0.015)")
    parser.add_argument("--texture", type=str, default="microgrooves", choices=["none", "microgrooves", "micropillars", "both"], help="Texture type")
    
    # Microgrooves
    parser.add_argument("--groove-profile", type=str, default="rectangular", choices=["rectangular", "triangular", "sine"], help="Groove cross-section profile")
    parser.add_argument("--groove-depth", type=float, default=0.050, help="Groove peak-to-valley depth in mm (default: 0.050)")
    parser.add_argument("--groove-wavelength", type=float, default=0.050, help="Groove wavelength/pitch in mm (default: 0.050)")
    
    # Micropillars
    parser.add_argument("--pillar-diameter", type=float, default=0.050, help="Pillar diameter in mm (default: 0.050)")
    parser.add_argument("--pillar-height", type=float, default=0.200, help="Pillar height in mm (default: 0.200)")
    parser.add_argument("--pillar-spacing", type=float, default=0.200, help="Pillar center-to-center spacing in mm (default: 0.200)")
    parser.add_argument("--pillar-distribution", type=str, default="poisson_disk", choices=["poisson_disk", "random"], help="Pillar distribution mode")
    parser.add_argument("--pillar-location", type=str, default="all", choices=["all", "internal_only", "outer_only"], help="Pillar placement location")
    parser.add_argument("--pillar-faces", nargs="+", default=None, help="Primitive faces to populate (e.g. +z -z, top, sides)")
    parser.add_argument("--filter-printable", action="store_true", help="Filter out non-printable pillar angles (> 30 deg from horizontal)")
    parser.add_argument("--max-overhang-angle", type=float, default=30.0, help="Max angle from horizontal for printability in degrees")
    
    parser.add_argument("-o", "--output", type=str, default="outputs/textures/scaffold.stl", help="Output STL path")
    args = parser.parse_args()

    build_scaffold(
        boundary_stl=args.boundary_stl,
        size_mm=args.size,
        lattice_type=args.lattice,
        pore_size_mm=args.pore_size,
        solid_fraction=args.solid_fraction,
        resolution_mm=args.resolution,
        texture=args.texture,
        groove_profile=args.groove_profile,
        groove_depth_mm=args.groove_depth,
        groove_wavelength_mm=args.groove_wavelength,
        pillar_diameter_mm=args.pillar_diameter,
        pillar_height_mm=args.pillar_height,
        pillar_spacing_mm=args.pillar_spacing,
        pillar_distribution=args.pillar_distribution,
        pillar_location=args.pillar_location,
        pillar_faces=args.pillar_faces,
        filter_printable_pillars=args.filter_printable,
        max_overhang_angle_deg=args.max_overhang_angle,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
