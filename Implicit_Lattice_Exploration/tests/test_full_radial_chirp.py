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
    print("Pure-Math Full-Radius Linear Chirped Gyroid")
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

    # Full-radius linear chirp
    L_center = 2.5
    k_center = 2.0 * np.pi / L_center
    L_edge = 5.0
    k_edge = 2.0 * np.pi / L_edge

    t0 = time.perf_counter()

    W = np.clip(R / radius, 0.0, 1.0)
    K_grid = k_center * (1.0 - W) + k_edge * W

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
    out_path = (
        Path(__file__).resolve().parents[2]
        / "Implicit_Lattice_Exploration"
        / "output"
        / "Radial_Full_Chirp_Linear.stl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    dt_total = time.perf_counter() - t0
    print(f"marching_cubes: {dt_mc:.2f} s")
    print(f"Total time: {dt_total:.2f} s")
    print(f"Faces: {len(mesh.faces):,}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
