from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def main() -> None:
    radius = 25.0
    height = 10.0
    resolution = 0.15
    pad = 2.0

    axis_xy = np.linspace(
        -radius - pad,
        radius + pad,
        int((radius * 2.0 + pad * 2.0) / resolution) + 1,
    )
    axis_z = np.linspace(
        -height / 2.0 - pad,
        height / 2.0 + pad,
        int((height + pad * 2.0) / resolution) + 1,
    )
    X, Y, Z = np.meshgrid(axis_xy, axis_xy, axis_z, indexing="ij")

    print("=" * 60)
    print("Pure-Math Radial Chirped Gyroid Diagnostic")
    print("=" * 60)
    print(
        f"Radius: {radius} mm, Height: {height} mm, Resolution: {resolution} mm, Pad: {pad} mm"
    )
    print(f"Grid shape: {X.shape}")

    # Pure math cylinder SDF
    R = np.sqrt(X**2 + Y**2)
    cyl_walls = R - radius
    cyl_caps = np.abs(Z) - (height / 2.0)
    cad_sdf = np.maximum(cyl_walls, cyl_caps)

    # Radial chirp parameters
    L_base = 5.0
    k_base = 2.0 * np.pi / L_base
    L_mod = 2.0
    k_mod = 2.0 * np.pi / L_mod
    r_core = 12.5

    out_root = Path(__file__).resolve().parents[2] / "Implicit_Lattice_Exploration" / "output"
    out_root.mkdir(parents=True, exist_ok=True)

    for transition_width in [1.0, 3.0, 6.0]:
        print(f"\n--- transition_width = {transition_width:.1f} mm ---")
        t0 = time.perf_counter()

        dist_to_core = np.clip(R - r_core, 0.0, None)
        W_linear = np.clip(1.0 - (dist_to_core / transition_width), 0.0, 1.0)
        W = 3.0 * W_linear**2 - 2.0 * W_linear**3

        K_grid = k_base * (1.0 - W) + k_mod * W
        F = (
            np.sin(K_grid * X) * np.cos(K_grid * Y)
            + np.sin(K_grid * Y) * np.cos(K_grid * Z)
            + np.sin(K_grid * Z) * np.cos(K_grid * X)
        )
        solid_field = np.abs(F) - 0.33

        final_field = np.maximum(solid_field, cad_sdf)

        t_mc = time.perf_counter()
        verts, faces, _, _ = marching_cubes(
            final_field.astype(np.float32),
            level=0.0,
            spacing=(resolution, resolution, resolution),
        )
        dt_mc = time.perf_counter() - t_mc

        verts[:, 0] = verts[:, 0] * resolution - (radius + pad)
        verts[:, 1] = verts[:, 1] * resolution - (radius + pad)
        verts[:, 2] = verts[:, 2] * resolution - (height / 2.0 + pad)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
        out_path = out_root / f"Radial_Chirp_Smooth_{transition_width}.stl"
        mesh.export(str(out_path))

        dt_total = time.perf_counter() - t0
        print(f"  marching_cubes: {dt_mc:.2f} s")
        print(f"  Total time: {dt_total:.2f} s")
        print(f"  Faces: {len(mesh.faces):,}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
