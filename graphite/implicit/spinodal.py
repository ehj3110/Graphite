"""
Graphite Implicit Engine - Spinodal GRF Lattice Generator

This module generates conformal Gaussian Random Field (GRF) spinodal metamaterials
bounded by arbitrary CAD geometries using Signed Distance Fields (SDF) and level-set
surface extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.math.boolean import smooth_max
from graphite.math.spinodal import (
    evaluate_spinodal_field,
    threshold_spinodal_field,
)
from graphite.mesh.extraction import extract_isosurface_flying_edges
from graphite.mesh.smoothing import smooth_mesh_taubin


def generate_spinodal_lattice(
    cad_mesh: trimesh.Trimesh | str | Path,
    resolution: float = 0.25,
    wavelength: float = 2.0,
    solid_fraction: float = 0.3,
    is_sheet: bool = False,
    anisotropy: tuple[float, float, float] = (1.0, 1.0, 1.0),
    num_waves: int = 120,
    seed: int | None = 42,
    pad_width: int = 4,
    output_path: str | Path | None = None,
    taubin_iterations: int = 15,
    taubin_lamb: float = 0.5,
    taubin_nu: float = -0.53,
    blend_radius: float = 0.0,
    blend_method: str = "polynomial",
) -> trimesh.Trimesh:
    """
    Generate a conformal Gaussian Random Field (GRF) spinodal lattice inside CAD geometry.

    Follows the conformal implicit pipeline:
    1. Voxelize the input CAD mesh and compute Euclidean Distance Transform (EDT) SDF.
    2. Evaluate the GRF scalar field F(x) via standing cosine wave superposition.
    3. Analytically threshold F(x) to achieve the target solid volume fraction.
    4. Intersect the spinodal lattice field with the CAD SDF:
       final_field = max(solid_field, cad_sdf).
    5. Extract the zero-level-set isosurface via extract_isosurface_flying_edges.
    6. Apply volume-preserving Taubin smoothing to eliminate voxel terracing and
       curvature spikes while maintaining macro volume and pore dimensions.
    7. Clean unreferenced vertices, fix normals, and return the Trimesh representation.

    Parameters
    ----------
    cad_mesh : trimesh.Trimesh or str or Path
        Input CAD boundary mesh, or filesystem path to an STL/OBJ file.
    resolution : float, optional
        Voxel pitch in millimeters, by default 0.25.
    wavelength : float, optional
        Characteristic pore / feature wavelength in millimeters, by default 2.0.
    solid_fraction : float, optional
        Target solid volume fraction in (0, 1), by default 0.3.
    is_sheet : bool, optional
        If True, generates a sheet/lamellar spinodal morphology. If False,
        generates a skeletal/network spinodal morphology. By default False.
    anisotropy : tuple of 3 floats, optional
        Directional anisotropy scaling factors (ax, ay, az), by default (1.0, 1.0, 1.0).
    num_waves : int, optional
        Number of spectral wave components, by default 120.
    seed : int or None, optional
        Random seed for reproducibility, by default 42.
    pad_width : int, optional
        Voxel padding around CAD domain for EDT closure, by default 4.
    output_path : str or Path or None, optional
        Optional filesystem path to export the resulting mesh. By default None.
    taubin_iterations : int, optional
        Number of Taubin smoothing passes, by default 15. Set <= 0 to bypass smoothing.
    taubin_lamb : float, optional
        Taubin positive pass shrinkage factor (0 < lambda < 1), by default 0.5.
    taubin_nu : float, optional
        Taubin negative pass deflation factor (mu < -lambda), by default -0.53.
    blend_radius : float, optional
        Radius of smooth fillet transition at the CAD domain boundary in mm, by default 0.0.
    blend_method : str, optional
        Smooth blending formulation: 'polynomial', 'circular', or 'exponential'.
        Default is "polynomial".

    Returns
    -------
    trimesh.Trimesh
        The meshed and clipped conformal spinodal lattice.

    Raises
    ------
    ValueError
        If parameters are non-positive or outside valid ranges.
    TypeError
        If cad_mesh is not a valid Trimesh or path.
    """
    # Load and normalize CAD geometry
    if isinstance(cad_mesh, (str, Path)):
        loaded = trimesh.load(str(cad_mesh))
        if isinstance(loaded, trimesh.Scene):
            mesh_cad = loaded.dump(concatenate=True)
        else:
            mesh_cad = loaded
    elif isinstance(cad_mesh, trimesh.Scene):
        mesh_cad = cad_mesh.dump(concatenate=True)
    elif isinstance(cad_mesh, trimesh.Trimesh):
        mesh_cad = cad_mesh
    else:
        raise TypeError(
            f"cad_mesh must be a trimesh.Trimesh, Path, or str, got {type(cad_mesh).__name__}"
        )

    if mesh_cad.is_empty or len(mesh_cad.faces) == 0:
        raise ValueError("cad_mesh must contain a non-empty geometry.")

    # Input validations
    if resolution <= 0.0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if wavelength <= 0.0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if not (0.0 < solid_fraction < 1.0):
        raise ValueError(f"solid_fraction must be in (0, 1), got {solid_fraction}")
    if num_waves <= 0:
        raise ValueError(f"num_waves must be > 0, got {num_waves}")
    if pad_width < 0:
        raise ValueError(f"pad_width must be >= 0, got {pad_width}")

    aniso = np.asarray(anisotropy, dtype=np.float64)
    if aniso.shape != (3,):
        raise ValueError(f"anisotropy must be a 3-element tuple or array, got {anisotropy}")
    if np.any(aniso <= 0.0):
        raise ValueError(f"anisotropy components must be strictly positive, got {anisotropy}")
    if blend_radius < 0.0:
        raise ValueError(f"blend_radius must be >= 0, got {blend_radius}")

    # 1. Voxelize input CAD mesh and compute EDT
    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, nx, ny, nz = voxelize_mesh_and_edt(
        mesh_cad, resolution=resolution, pad_width=pad_width
    )

    # 2. Evaluate GRF scalar field F(x)
    F = evaluate_spinodal_field(
        X,
        Y,
        Z,
        wavelength=wavelength,
        num_waves=num_waves,
        anisotropy=anisotropy,
        seed=seed,
    )

    # 3. Compute solid field via analytic thresholding
    solid_field = threshold_spinodal_field(
        F,
        solid_fraction=solid_fraction,
        is_sheet=is_sheet,
    )

    # 4. Intersect with CAD domain: final_field = smax(solid_field, cad_sdf)
    if blend_radius > 0.0:
        final_field = smooth_max(
            solid_field, cad_sdf, r=blend_radius, method=blend_method
        )
    else:
        final_field = np.maximum(solid_field, cad_sdf)

    # 5. Extract isosurface with Flying Edges at level=0.0
    mesh_out = extract_isosurface_flying_edges(
        final_field,
        origin=padded_min_bound,
        spacing=(resolution, resolution, resolution),
        level=0.0,
    )

    # 6. Apply volume-preserving Taubin smoothing to relax curvature
    if taubin_iterations > 0 and not mesh_out.is_empty:
        mesh_out = smooth_mesh_taubin(
            mesh_out,
            iterations=taubin_iterations,
            lamb=taubin_lamb,
            nu=taubin_nu,
            inplace=True,
        )

    # Optional export
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(p))

    return mesh_out
