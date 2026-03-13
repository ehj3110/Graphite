"""
Implicit Gyroid Generator - basic SDF + marching cubes export.

Generates a gyroid block using an implicit level-set (signed distance-like field)
with configurable size, resolution, pore size, and solid fraction.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes


def generate_gyroid_block(
    size: float = 20.0,
    resolution: float = 0.25,
    unit_cell_size: float | None = None,
    pore_size: float | None = 5.0,
    solid_fraction: float = 0.33,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a gyroid block and optionally export it to STL.

    Parameters
    ----------
    size : float
        Physical cube size in mm (edge length).
    resolution : float
        Grid spacing in mm (voxel pitch).
    unit_cell_size : float | None
        Direct gyroid unit-cell period L in mm. If provided, overrides
        the pore_size-based fit.
    pore_size : float | None
        Target pore diameter in mm. If provided (and unit_cell_size is None),
        L is derived from the Al-Ketan empirical fit.
    solid_fraction : float
        Target solid volume fraction (0-1).
    output_path : str | Path | None
        If provided, STL is exported to this path.

    Returns
    -------
    mesh : trimesh.Trimesh
        Triangular mesh of the extracted gyroid sheet.
    """
    # Gyroid parameters (legacy Al-Ketan pore fit or direct unit-cell size)
    if pore_size is not None:
        L = pore_size / (1.0 - 1.15 * solid_fraction)
    elif unit_cell_size is not None:
        L = unit_cell_size
    else:
        raise ValueError("Must provide either pore_size or unit_cell_size")

    k = 2.0 * np.pi / L

    # Coordinate grid in physical units, centered at the origin, with padding
    padding = resolution * 3.0
    padded_size = size + padding * 2.0
    n = int(np.round(padded_size / resolution)) + 1
    axis = np.linspace(-padded_size / 2.0, padded_size / 2.0, n, dtype=np.float64)
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")

    # Gyroid field
    F = (
        np.sin(k * X) * np.cos(k * Y)
        + np.sin(k * Y) * np.cos(k * Z)
        + np.sin(k * Z) * np.cos(k * X)
    )

    # Sheet-style field: symmetric shell around F=0, with thickness based on |t|
    t = 1.5 * (2.0 * solid_fraction - 1.0)
    solid_field = np.abs(F) - abs(t)

    # Box SDF to cap the TPMS at the exact cube boundary
    box_sdf = np.maximum(np.abs(X), np.maximum(np.abs(Y), np.abs(Z))) - (size / 2.0)

    # Implicit intersection: TPMS sheet intersected with cube
    final_field = np.maximum(solid_field, box_sdf)

    # Marching cubes on the implicit zero-level set
    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    t_mc = time.perf_counter() - t0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))

    print(
        f"Gyroid block: size={size}mm, res={resolution}mm, pores={pore_size}mm, "
        f"L={L:.3f}mm, SF={solid_fraction:.2f}"
    )
    print(f"  marching_cubes: {t_mc:.2f} s, faces={len(mesh.faces):,}")
    return mesh


if __name__ == "__main__":
    out = Path("Implicit_Lattice_Exploration/output/Basic_Gyroid.stl")
    generate_gyroid_block(output_path=out)
