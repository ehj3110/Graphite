"""
Graphite Implicit Engine - Multi-Zonal Lattices

This module generates conformal TPMS lattices that feature discrete zones 
(layers) stacked along the Z-axis. Each zone can have independent parameters 
(TPMS equation, pore size, solid fraction) stacked progressively.
"""
import numpy as np
import trimesh
from skimage.measure import marching_cubes

from pathlib import Path

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.io.mesh_export import export_mesh
from graphite.math.tpms import evaluate_tpms


def generate_multi_zonal_lattice(
    stl_path,
    zones_data,
    resolution=0.25,
    center_origin=False,
    output_path=None,
    export_formats=None,
):
    """
    Generate a Multi-Zonal TPMS lattice stacking different types and parameters.

    Parameters
    ----------
    stl_path : str or Path
        Path to the target STL boundary mesh.
    zones_data : list of dict
        A list of dictionaries configuring each zone from bottom to top. 
        Expected keys for each dict:
        - "Thickness (mm)": float
        - "Lattice Type": str
        - "Pore Size (mm)": float
        - "Solid Fraction": float
    resolution : float, optional
        Voxel resolution for the evaluation field in mm, by default 0.25.
    center_origin : bool, optional
        If True, translates the final output mesh to center on the origin, by default False.
    output_path : str or Path, optional
        Optional path to write the resulting mesh, by default None.

    Returns
    -------
    trimesh.Trimesh
        The meshed multi-zonal lattice.
    """
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, _nx, _ny, _nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    # Relative Z-height from the bottom of the padded voxel grid
    Z_rel = Z - padded_min_bound[2]

    # Precalculate boundaries
    boundaries = []
    current_z = 0.0
    for zone in zones_data[:-1]:  # The last zone goes to infinity essentially
        current_z += float(zone["Thickness (mm)"])
        boundaries.append(current_z)

    # Calculate weights for each zone
    # TODO: In the future, implement transition blending across boundaries.
    # For now, there is NO transition region (hard boundaries).
    num_zones = len(zones_data)
    weights = []

    last_b = -np.inf
    for i in range(num_zones):
        if i == num_zones - 1:
            next_b = np.inf
        else:
            next_b = boundaries[i]
            
        W_i = np.where((Z_rel >= last_b) & (Z_rel < next_b), 1.0, 0.0)
        weights.append(W_i)
        last_b = next_b

    # Accumulate field
    # Field blending approach: F_sum = sum(W_i * (abs(F_i) - SF_i))
    # where solid is F_sum <= 0.
    final_solid_field = np.zeros_like(Z_rel)

    for i, zone in enumerate(zones_data):
        l_type = zone["Lattice Type"]
        sf = float(zone["Solid Fraction"])
        p_size = float(zone["Pore Size (mm)"])

        v_k = 2.0 * np.pi / max(p_size, 0.001)

        # U, V, W integrated or direct phase? 
        # Using basic spatial evaluation for explicit zone separation.
        # This will blend the fields in the transition_width naturally.
        F_i = evaluate_tpms(l_type, v_k, X, Y, Z)

        # Base threshold solid field (positive is void, negative is solid)
        solid_i = np.abs(F_i) - sf

        final_solid_field += weights[i] * solid_i

    # Inside CAD volume
    # Max intersection since both solid_field and cad_sdf follow inside<=0
    final_field = np.maximum(final_solid_field, cad_sdf)

    verts, faces, _normals, _values = marching_cubes(
        final_field, level=0.0, spacing=(resolution, resolution, resolution)
    )

    verts[:, 0] += padded_min_bound[0]
    verts[:, 1] += padded_min_bound[1]
    verts[:, 2] += padded_min_bound[2]

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if center_origin:
        mesh_out.vertices -= mesh_out.centroid

    if output_path:
        export_mesh(mesh_out, Path(output_path), formats=export_formats)

    return mesh_out
