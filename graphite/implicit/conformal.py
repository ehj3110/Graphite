from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as edt
from skimage.measure import marching_cubes

from graphite.geometry.masking import voxelize_mesh_and_edt
from graphite.math.tpms import evaluate_tpms


def _compute_L_and_k(
    pore_size: float | None, unit_cell_size: float | None, solid_fraction: float
) -> tuple[float, float]:
    if pore_size is not None:
        L = pore_size / (1.0 - 1.15 * solid_fraction)
    elif unit_cell_size is not None:
        L = unit_cell_size
    else:
        raise ValueError("Must provide either pore_size or unit_cell_size")
    k = 2.0 * np.pi / L
    return L, k


def generate_conformal_lattice(
    stl_path: str | Path,
    lattice_type: str = "Gyroid",
    resolution: float = 0.25,
    pore_size: float | None = 5.0,
    solid_fraction: float = 0.33,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    center_origin: bool = False,
    selected_surfaces: list[int] | None = None,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Generate a conformal TPMS lattice inside an input STL using EDT-based CAD SDF."""
    stl_path = Path(stl_path)
    mesh = trimesh.load(str(stl_path))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    X, Y, Z, cad_sdf, padded_min_bound, _padded_max_bound, nx, ny, nz = voxelize_mesh_and_edt(
        mesh, resolution
    )

    L, k = _compute_L_and_k(pore_size, None, solid_fraction)
    F = evaluate_tpms(lattice_type, k, X, Y, Z)

    t = 1.5 * (2.0 * solid_fraction - 1.0)
    solid_field = np.abs(F) - abs(t)

    core_sdf = np.maximum(solid_field, cad_sdf)

    if selected_surfaces is not None and len(selected_surfaces) > 0:
        facets = mesh.facets
        valid_ids = [int(i) for i in selected_surfaces]
        max_id = len(facets) - 1
        invalid_ids = [i for i in valid_ids if i < 0 or i > max_id]
        if invalid_ids:
            raise ValueError(
                f"selected_surfaces contains invalid IDs {invalid_ids}; valid range is 0 to {max_id}"
            )

        selected_face_indices = np.hstack([facets[i] for i in valid_ids])
        sub_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[selected_face_indices],
            process=False,
        )

        num_points = max(10000, int(sub_mesh.area / (resolution**2)) * 2)
        surface_points, _ = trimesh.sample.sample_surface(sub_mesh, num_points)

        ix = np.clip(
            np.round((surface_points[:, 0] - padded_min_bound[0]) / resolution).astype(
                int
            ),
            0,
            nx - 1,
        )
        iy = np.clip(
            np.round((surface_points[:, 1] - padded_min_bound[1]) / resolution).astype(
                int
            ),
            0,
            ny - 1,
        )
        iz = np.clip(
            np.round((surface_points[:, 2] - padded_min_bound[2]) / resolution).astype(
                int
            ),
            0,
            nz - 1,
        )

        surface_mask = np.zeros((nx, ny, nz), dtype=bool)
        surface_mask[ix, iy, iz] = True
        distance_to_surface = edt(~surface_mask) * resolution

        skin_sdf = np.maximum(cad_sdf, distance_to_surface - shell_thickness)
    else:
        skin_sdf = np.maximum(cad_sdf, -cad_sdf - shell_thickness)

    if export_mode == "core":
        final_field = core_sdf
    elif export_mode == "skin":
        final_field = skin_sdf
    elif export_mode == "combined":
        final_field = np.minimum(core_sdf, skin_sdf)
    else:
        raise ValueError("export_mode must be 'core', 'skin', or 'combined'")

    t0 = time.perf_counter()
    verts, faces, _, _ = marching_cubes(
        final_field.astype(np.float32),
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    t_mc = time.perf_counter() - t0

    verts = (verts * resolution) + padded_min_bound

    mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)
    if center_origin:
        verts -= mesh_out.centroid
        mesh_out = trimesh.Trimesh(vertices=verts, faces=faces.astype(np.int64), process=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_out.export(str(output_path))

    print(
        f"Conformal {lattice_type} lattice from '{stl_path.name}': res={resolution}mm, "
        f"SF={solid_fraction:.2f}, mode={export_mode}"
    )
    print(
        f"  L={L:.3f}mm (pore={pore_size}), shell={shell_thickness:.2f}mm, "
        f"marching_cubes: {t_mc:.2f} s, faces={len(mesh_out.faces):,}"
    )
    if center_origin:
        print("  Output mesh centered at origin")
    if selected_surfaces:
        print(f"  Localized shell surfaces: {selected_surfaces}")

    return mesh_out


def generate_conformal_gyroid(
    stl_path: str | Path,
    resolution: float = 0.25,
    pore_size: float | None = 5.0,
    solid_fraction: float = 0.33,
    export_mode: str = "core",
    shell_thickness: float = 2.0,
    selected_surfaces: list[int] | None = None,
    output_path: str | Path | None = None,
) -> trimesh.Trimesh:
    """Backward-compatible wrapper for the original gyroid-only API."""
    return generate_conformal_lattice(
        stl_path=stl_path,
        lattice_type="Gyroid",
        resolution=resolution,
        pore_size=pore_size,
        solid_fraction=solid_fraction,
        export_mode=export_mode,
        shell_thickness=shell_thickness,
        selected_surfaces=selected_surfaces,
        output_path=output_path,
    )

