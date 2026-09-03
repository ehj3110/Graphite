"""
Graphite Implicit Engine - Surface Textures

This module applies high-resolution procedural displacement textures (microgrooves,
bumps, diamond knurling, and spinodal textures) to 3D implicit scaffold meshes.
It uses adaptive subdivision to ensure mesh edge density can faithfully resolve
micro-features, followed by normal-aligned vertex displacement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import trimesh

from graphite.math.textures import (
    bump_field,
    knurl_field,
    microgroove_field,
    spinodal_spectral_field,
    triplanar_map,
)


@dataclass(frozen=True)
class SurfaceTextureConfig:
    """
    Configuration parameters for surface micro-texturing on meshes.

    Attributes
    ----------
    texture_type : str
        Type of surface texture: "microgrooves", "bumps", "knurling", "spinodal", or "none".
    amplitude_mm : float
        Displacement depth / height in millimeters. Default 0.025 mm (+/-25 um = 50 um total depth).
    wavelength_mm : float
        Spatial period / pitch of the texture in millimeters. Default 0.050 mm (50 um).
    direction : tuple of float
        Unit direction vector along which grooves modulate. Defaults to (0.0, 0.0, 1.0) (Z-axial).
    profile : str
        Wave profile for microgrooves: "sine", "triangle", or "square". Defaults to "sine".
    displacement_mode : str
        How displacement is applied:
        - "centered": centered around nominal surface, values in [-amplitude, +amplitude].
        - "emboss": purely outward positive displacement, values in [0, +amplitude].
        - "engrave": purely inward negative displacement, values in [-amplitude, 0].
    use_triplanar : bool
        If True, applies texture via normal-weighted triplanar projection.
        If False, evaluates directly in 3D world space (seamless for axial striations).
    triplanar_sharpness : float
        Exponent for triplanar projection weight blending. Defaults to 2.0.
    target_edge_length_mm : float or None
        Target maximum edge length for adaptive subdivision. If None, defaults to
        wavelength_mm / 2.0 to ensure at least 2 to 4 facets per wave.
    max_faces : int
        Safety cap on maximum face count after subdivision to prevent out-of-memory.
        Defaults to 2,000,000.
    max_subdivisions : int
        Maximum subdivision iterations allowed. Defaults to 6.
    repair_after_displacement : bool
        Whether to run a light normal/manifold repair pass after displacement.
    """
    texture_type: str = "microgrooves"
    amplitude_mm: float = 0.025
    wavelength_mm: float = 0.050
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    profile: str = "sine"
    displacement_mode: str = "centered"
    use_triplanar: bool = False
    triplanar_sharpness: float = 2.0
    target_edge_length_mm: float | None = None
    max_faces: int = 2_000_000
    max_subdivisions: int = 6
    project_transverse_normal: bool = True
    repair_after_displacement: bool = True


def subdivide_for_texture(
    mesh: trimesh.Trimesh,
    target_edge_length_mm: float,
    *,
    max_faces: int = 2_000_000,
    max_subdivisions: int = 6,
) -> trimesh.Trimesh:
    """
    Subdivide a mesh until all unique edges are at most `target_edge_length_mm`.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input triangle mesh.
    target_edge_length_mm : float
        Maximum allowable edge length in millimeters.
    max_faces : int, optional
        Maximum face threshold to avoid excessive memory consumption.
    max_subdivisions : int, optional
        Maximum number of subdivision passes.

    Returns
    -------
    trimesh.Trimesh
        Refined mesh suitable for micro-texturing.
    """
    if target_edge_length_mm <= 0:
        raise ValueError(f"target_edge_length_mm must be > 0, got {target_edge_length_mm}")

    m = mesh.copy()
    for _ in range(int(max_subdivisions)):
        if len(m.faces) >= int(max_faces):
            break
        lengths = m.edges_unique_length
        if len(lengths) == 0:
            break
        if float(np.max(lengths)) <= float(target_edge_length_mm):
            break
        # Guard against exceeding face cap on next step (subdivision quadruples faces)
        if len(m.faces) * 4 > int(max_faces):
            break

        m = m.subdivide()

    return m


def apply_surface_texture(
    mesh: trimesh.Trimesh,
    config: SurfaceTextureConfig | None = None,
) -> trimesh.Trimesh:
    """
    Apply a procedural displacement surface texture to a 3D triangle mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The base (untextured) scaffold mesh.
    config : SurfaceTextureConfig, optional
        Texture configuration parameters. If None, applies default 50 um microgrooves.

    Returns
    -------
    trimesh.Trimesh
        The textured mesh with micro-structures displaced along vertex normals.
    """
    if config is None:
        config = SurfaceTextureConfig()

    t_type = config.texture_type.strip().lower()
    if t_type in ("none", "") or config.amplitude_mm <= 0.0:
        return mesh.copy()

    t0 = time.perf_counter()

    # Determine target edge length for subdivision
    target_edge = config.target_edge_length_mm
    if target_edge is None:
        target_edge = max(float(config.wavelength_mm) / 2.0, 1e-4)

    # Step 1: Subdivide to resolve micro-features
    refined = subdivide_for_texture(
        mesh,
        target_edge_length_mm=target_edge,
        max_faces=config.max_faces,
        max_subdivisions=config.max_subdivisions,
    )

    verts = np.asarray(refined.vertices, dtype=np.float64)
    norms = np.asarray(refined.vertex_normals, dtype=np.float64)

    # Normalize vertex normals
    norm_lens = np.linalg.norm(norms, axis=1, keepdims=True)
    norm_lens = np.where(norm_lens < 1e-12, 1.0, norm_lens)
    norms = norms / norm_lens

    # Step 2: Evaluate procedural texture field
    if t_type == "microgrooves":
        if config.use_triplanar:
            def _planar_groove(u: np.ndarray, v: np.ndarray, wl: float) -> np.ndarray:
                # Modulate along v axis
                return microgroove_field(
                    np.zeros_like(u), np.zeros_like(u), v, wavelength_mm=wl, direction=(0, 0, 1), profile=config.profile
                )
            h = triplanar_map(
                verts, norms, wavelength_mm=config.wavelength_mm, texture_fn=_planar_groove, sharpness=config.triplanar_sharpness
            )
        else:
            h = microgroove_field(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                wavelength_mm=config.wavelength_mm,
                direction=config.direction,
                profile=config.profile,
            )
    elif t_type in ("bumps", "bump", "nodules"):
        h = bump_field(verts[:, 0], verts[:, 1], verts[:, 2], wavelength_mm=config.wavelength_mm, profile="nodule")
    elif t_type in ("knurl", "knurling", "diamond"):
        h = knurl_field(verts[:, 0], verts[:, 1], verts[:, 2], wavelength_mm=config.wavelength_mm, axis="z")
    elif t_type in ("spinodal", "cahn_hilliard"):
        h = spinodal_spectral_field(verts[:, 0], verts[:, 1], verts[:, 2], wavelength_mm=config.wavelength_mm)
    else:
        raise ValueError(
            f"Unknown texture_type: '{config.texture_type}'. Supported: 'microgrooves', 'bumps', 'knurling', 'spinodal', 'none'."
        )

    # Step 3: Apply displacement mode
    amp = float(config.amplitude_mm)
    mode = config.displacement_mode.strip().lower()
    if mode == "centered":
        delta = amp * h
    elif mode == "emboss":
        delta = amp * 0.5 * (h + 1.0)
    elif mode == "engrave":
        delta = amp * 0.5 * (h - 1.0)
    else:
        raise ValueError(f"Unknown displacement_mode: '{config.displacement_mode}'. Must be 'centered', 'emboss', or 'engrave'.")

    # Step 4: Displace vertices along normal vector
    disp_norms = norms.copy()
    if config.project_transverse_normal and t_type == "microgrooves" and not config.use_triplanar:
        d = np.asarray(config.direction, dtype=np.float64)
        d_norm = float(np.linalg.norm(d))
        if d_norm > 1e-12:
            d_unit = d / d_norm
            # Project out the parallel normal component to prevent self-intersection wrinkles
            disp_norms = disp_norms - np.sum(disp_norms * d_unit, axis=1, keepdims=True) * d_unit

    displaced_verts = verts + disp_norms * delta[:, None]

    out_mesh = trimesh.Trimesh(vertices=displaced_verts, faces=refined.faces, process=False)

    if config.repair_after_displacement:
        trimesh.repair.fix_normals(out_mesh)
        trimesh.repair.fill_holes(out_mesh)

    elapsed = time.perf_counter() - t0
    print(
        f"[SurfaceTexture] Applied {t_type} ({config.profile}): "
        f"faces={len(out_mesh.faces):,}, amp={amp*1e3:.1f}um, wl={config.wavelength_mm*1e3:.1f}um, "
        f"watertight={out_mesh.is_watertight} in {elapsed:.2f}s"
    )

    return out_mesh
