"""
FEA export utilities — dual-path architecture for Abaqus and CAD workflows.

Path A (exact physics): voxel lattice → structured C3D8R hex mesh → Abaqus ``.inp``
Path B (CAD approximation): parametric Schwarz-P B-Rep assembly → STEP (``build123d``)

Both paths are intended to be validated locally via ``scripts/testbed_scikit_fem_compression.py``
before hand-off to Abaqus.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

__all__ = [
    "export_voxel_to_abaqus_inp",
    "export_parametric_tpms_step",
]

# Abaqus C3D8 / C3D8R corner order (bottom face k, then top face k+1):
# N1–N4: (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k)
# N5–N8: (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1)
_CORNER_I = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
_CORNER_J = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
_CORNER_K = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)


def export_voxel_to_abaqus_inp(
    voxel_array: np.ndarray,
    physical_bounds: tuple[np.ndarray, np.ndarray] | tuple[tuple[float, ...], tuple[float, ...]],
    filepath: str | Path,
    *,
    solid_value: bool | int = True,
    element_type: str = "C3D8R",
) -> Path:
    """
    Export a binary voxel lattice as an Abaqus input deck with C3D8R hexahedra.

    Uses fully vectorized NumPy (no per-voxel Python loops) for node deduplication
    and element connectivity.

    Parameters
    ----------
    voxel_array : ndarray
        3D array of shape ``(nx, ny, nz)``. Solid voxels are ``True`` (or match
        ``solid_value`` when an integer mask is used).
    physical_bounds : tuple
        ``(min_corner, max_corner)`` in mm, each length-3. The voxel grid spans
        the bounding box uniformly: cell size = ``(max - min) / shape`` per axis.
    filepath : path-like
        Output ``.inp`` path.
    solid_value : bool or int, optional
        Value in ``voxel_array`` that denotes solid material, by default ``True``.
    element_type : str, optional
        Abaqus element type label, by default ``"C3D8R"``.

    Returns
    -------
    Path
        Resolved output path.
    """
    voxel_array = np.asarray(voxel_array)
    if voxel_array.ndim != 3:
        raise ValueError(f"voxel_array must be 3D; got shape {voxel_array.shape}.")

    nx, ny, nz = (int(v) for v in voxel_array.shape)
    min_coords = np.asarray(physical_bounds[0], dtype=np.float64).reshape(3)
    max_coords = np.asarray(physical_bounds[1], dtype=np.float64).reshape(3)
    extent = max_coords - min_coords
    spacing = extent / np.array([nx, ny, nz], dtype=np.float64)

    if np.any(spacing <= 0):
        raise ValueError("physical_bounds must define a positive extent on every axis.")

    if voxel_array.dtype == np.bool_:
        solid = voxel_array
    else:
        solid = voxel_array == solid_value

    i, j, k = np.nonzero(solid)
    n_elem = int(i.size)
    if n_elem == 0:
        raise ValueError("No solid voxels to export.")

    # Integer corner indices for all solid hexes: (8 * n_elem, 3)
    ci = i[:, None] + _CORNER_I[None, :]
    cj = j[:, None] + _CORNER_J[None, :]
    ck = k[:, None] + _CORNER_K[None, :]
    corner_idx = np.stack(
        [ci.ravel(), cj.ravel(), ck.ravel()],
        axis=1,
    )

    unique_idx, inv = np.unique(corner_idx, axis=0, return_inverse=True)
    n_nodes = int(unique_idx.shape[0])
    node_ids = np.arange(1, n_nodes + 1, dtype=np.int64)

    # Physical node coordinates (mm)
    node_coords = min_coords + unique_idx * spacing

    # 1-based Abaqus connectivity (n_elem, 8)
    elem_conn = (inv.reshape(n_elem, 8) + 1).astype(np.int64, copy=False)
    elem_ids = np.arange(1, n_elem + 1, dtype=np.int64)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    node_block = np.column_stack([node_ids, node_coords])
    elem_fmt = "%d, %d, %d, %d, %d, %d, %d, %d, %d"
    elem_block = np.column_stack([elem_ids, elem_conn])

    with filepath.open("w", encoding="ascii", newline="\n") as f:
        f.write("** Graphite voxel-to-Abaqus export (C3D8R)\n")
        f.write(f"** grid: {nx} x {ny} x {nz} voxels\n")
        f.write(
            f"** spacing (mm): {spacing[0]:.16g}, {spacing[1]:.16g}, {spacing[2]:.16g}\n"
        )
        f.write(f"** nodes: {n_nodes}, elements: {n_elem}\n")
        f.write("*NODE\n")
        np.savetxt(
            f,
            node_block,
            fmt="%d, %.16g, %.16g, %.16g",
            delimiter=",",
        )
        f.write(f"*ELEMENT, TYPE={element_type}, ELSET=LATTICE_VOLUME\n")
        np.savetxt(f, elem_block, fmt=elem_fmt, delimiter=",")

    return filepath


def export_parametric_tpms_step(
    grid_dims: tuple[int, int, int],
    chirp_function: Callable[[np.ndarray], dict[str, float]],
    filepath: str | Path,
    *,
    lattice_type: str = "schwarz-p",
    unit_cell_size_mm: float = 5.0,
    origin_mm: np.ndarray | None = None,
) -> Path:
    """
    Export a parametric TPMS lattice as a smooth STEP solid (Path B — CAD approximation).

    Parameters
    ----------
    grid_dims : tuple of int
        ``(nx, ny, nz)`` number of unit cells along each axis.
    chirp_function : callable
        ``chirp_function(center_xyz) -> dict`` with keys such as ``period_mm``,
        ``solid_fraction`` or ``wall_thickness_mm`` for the cell centered at ``center_xyz``.
    filepath : path-like
        Output ``.step`` / ``.stp`` path.
    lattice_type : str, optional
        TPMS family to approximate in B-Rep; Phase 1 targets ``"schwarz-p"``, by default
        ``"schwarz-p"``.
    unit_cell_size_mm : float, optional
        Nominal unit cell period when chirp does not override, by default ``5.0`` mm.
    origin_mm : ndarray, optional
        Grid origin (min corner) in mm. Default ``(0, 0, 0)``.

    Returns
    -------
    Path
        Resolved output path (written in a future implementation).
    """
    _ = (grid_dims, chirp_function, lattice_type, unit_cell_size_mm, origin_mm)
    filepath = Path(filepath)
    raise NotImplementedError(
        "export_parametric_tpms_step is a Phase 20 stub. "
        "Install build123d and implement Schwarz-P assembly per docstring."
    )
