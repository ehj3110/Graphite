"""
Graphite I/O Module — Mesh Loading and Verification

This module provides the entry point for loading and validating boundary meshes
(STL) for the conformal lattice generation pipeline. It enforces watertightness
and computes geometric metadata required by downstream modules (scaffold, topology).

Architectural Context:
    - Part of the Graphite conformal lattice generation engine
    - Uses trimesh exclusively (per lattice_generator_spec.md)
    - No external CAD kernels (Blender, FreeCAD, Open3D)

Author: Graphite Project
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import trimesh


# -----------------------------------------------------------------------------
# Return Type: Structured output for load_and_verify_mesh
# -----------------------------------------------------------------------------


class MeshVerificationResult(NamedTuple):
    """
    Structured result from load_and_verify_mesh.

    Attributes:
        mesh: The cleaned, normal-repaired trimesh.Trimesh object.
        bounding_box: (length_x, length_y, length_z) in mesh units.
        volume: Total volumetric capacity (enclosed volume) in cubic units.
    """

    mesh: trimesh.Trimesh
    bounding_box: tuple[float, float, float]
    volume: float


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def load_and_verify_mesh(filepath: str | Path) -> MeshVerificationResult:
    """
    Load an STL mesh from disk, repair normals, verify watertightness,
    and return the cleaned mesh with bounding box and volume.

    This function is the canonical entry point for mesh ingestion in the
    Graphite pipeline. Lattice generation requires a closed, watertight
    volume; non-watertight meshes will raise a descriptive ValueError.

    Args:
        filepath: Path to the STL file (string or pathlib.Path).

    Returns:
        MeshVerificationResult containing:
            - mesh: Repaired trimesh.Trimesh instance
            - bounding_box: (length_x, length_y, length_z)
            - volume: Enclosed volume in cubic units

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the mesh is not watertight after repair.
        trimesh.exchange.load.LoadError: If the file cannot be parsed as mesh.

    Example:
        >>> result = load_and_verify_mesh("boundary.stl")
        >>> print(f"Volume: {result.volume:.2f}, BBox: {result.bounding_box}")
    """
    # --- Resolve and validate file path ---
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path.resolve()}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path.resolve()}")

    # --- Load mesh via trimesh ---
    # trimesh.load() may return Trimesh or Scene; we need a single Trimesh.
    loaded = trimesh.load(str(path), force="mesh")

    # Handle Scene (multi-mesh) by merging into a single mesh
    if isinstance(loaded, trimesh.Scene):
        # Concatenate all geometry in the scene into one mesh
        meshes = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(
                f"Scene at {path} contains no valid mesh geometry."
            )
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
    else:
        raise ValueError(
            f"Unsupported mesh type from {path}: {type(loaded)}. "
            "Expected Trimesh or Scene."
        )

    # --- Repair face normals (in-place) ---
    # Ensures consistent outward-facing normals and correct winding.
    # Critical for correct volume sign and downstream boolean operations.
    trimesh.repair.fix_normals(mesh)

    # --- Enforce watertightness ---
    # Lattice generation fills a closed volume; holes or open boundaries
    # would produce invalid conformal meshes.
    if not mesh.is_watertight:
        raise ValueError(
            "The mesh is not watertight (has holes or non-manifold edges). "
            "Lattice generation requires a closed volume. "
            "Please repair the mesh in a CAD tool (e.g., MeshMixer, Netfabb) "
            "or use trimesh.repair.fill_holes() if appropriate."
        )

    # --- Compute bounding box dimensions ---
    # mesh.extents gives [length_x, length_y, length_z]
    extents = mesh.extents
    bounding_box = (float(extents[0]), float(extents[1]), float(extents[2]))

    # --- Compute enclosed volume ---
    # For watertight meshes, trimesh computes signed volume correctly.
    volume = float(mesh.volume)

    if volume <= 0:
        raise ValueError(
            f"Mesh volume is non-positive ({volume}). "
            "This may indicate inverted normals or degenerate geometry."
        )

    return MeshVerificationResult(
        mesh=mesh,
        bounding_box=bounding_box,
        volume=volume,
    )
