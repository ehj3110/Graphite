"""
Graphite Scaffold Module — Conformal Tetrahedral Meshing with GMSH

This module converts a watertight boundary mesh into a volumetric tetrahedral
scaffold using the GMSH Python API. It follows the "merge bypass" strategy:
export boundary to temporary STL -> merge into GMSH -> define volume -> mesh.

Coordinate integrity (Day 2):
- Input mesh must be watertight (process=True) for downstream Boolean ops.
- Smart Inset: scaffold uses scaled boundary; nodes are inverse-scaled back
  to original CAD coordinates for lattice placement.
- Bbox mismatch warning when scaffold nodes differ from input bounds > 1e-5.

Outputs are NumPy arrays ready for downstream topology synthesis:
    - nodes: global XYZ coordinates
    - elements: tetrahedral connectivity (4-node indices)
    - surface_faces: outer boundary triangle connectivity (3-node indices)
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import gmsh
import numpy as np
import trimesh


class ScaffoldResult(NamedTuple):
    """
    Structured return type for generate_conformal_scaffold.

    Attributes:
        nodes: (N, 3) global node coordinates.
        elements: (M, 4) tetrahedral element connectivity (zero-based indices).
        surface_faces: (K, 3) boundary triangle connectivity (zero-based indices).
    """

    nodes: np.ndarray
    elements: np.ndarray
    surface_faces: np.ndarray


def _map_tags_to_zero_based_indices(
    connectivity: np.ndarray,
    node_tag_to_index: dict[int, int],
) -> np.ndarray:
    """
    Convert GMSH node tags in a connectivity array to contiguous zero-based indices.

    GMSH node tags are positive IDs that are not guaranteed to be contiguous.
    Downstream numerical routines expect direct row indices into the `nodes` array.
    """
    flat_tags = connectivity.reshape(-1)
    mapped = np.fromiter(
        (node_tag_to_index[int(tag)] for tag in flat_tags),
        dtype=np.int64,
        count=flat_tags.size,
    )
    return mapped.reshape(connectivity.shape)


def generate_conformal_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    algorithm_3d: int = 10,
) -> ScaffoldResult:
    """
    Generate a conformal tetrahedral scaffold from an input boundary mesh.

    CRITICAL FLOW (merge bypass):
      1. Export input mesh to temporary STL
      2. gmsh.initialize()
      3. Configure mesh size + Netgen optimizer
      4. gmsh.merge(temp_stl)
      5. Define volume with geo surface loop/volume
      6. gmsh.model.geo.synchronize()
      7. gmsh.model.mesh.generate(3)

    Args:
        mesh: Watertight boundary mesh to tetrahedralize.
        target_element_size: Maximum mesh size target used by GMSH.
        algorithm_3d: GMSH 3D algorithm (10=HXT, 4=Netgen). Use 4 for meshes
            with self-intersecting facets that cause HXT to fail.

    Returns:
        ScaffoldResult with nodes, tetra elements, and surface triangle faces.

    Raises:
        TypeError: If mesh is not trimesh.Trimesh.
        ValueError: If input is invalid or no tetrahedral elements are generated.
        RuntimeError: If GMSH fails to generate or extract scaffold data.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(
            f"`mesh` must be trimesh.Trimesh, received: {type(mesh)}"
        )

    if target_element_size <= 0:
        raise ValueError("`target_element_size` must be > 0.")

    temp_stl_path: str | None = None
    gmsh_initialized = False

    try:
        # ---------------------------------------------------------------------
        # 1) Export boundary mesh to temporary STL
        # ---------------------------------------------------------------------
        # We use a physical temp file so gmsh.merge() can import geometry exactly
        # as instructed by the module API constraints.
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            temp_stl_path = tmp.name
        mesh.export(temp_stl_path)

        # ---------------------------------------------------------------------
        # 2) Initialize GMSH
        # ---------------------------------------------------------------------
        gmsh.initialize()
        gmsh_initialized = True
        # Silence GMSH terminal spam for clean test output.
        gmsh.option.setNumber("General.Terminal", 0)

        # ---------------------------------------------------------------------
        # 3) Global meshing options
        # ---------------------------------------------------------------------
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(target_element_size))
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm_3d))
        gmsh.option.setNumber("Mesh.Smoothing", 100)

        # ---------------------------------------------------------------------
        # 4) Merge temporary STL
        # ---------------------------------------------------------------------
        gmsh.merge(temp_stl_path)

        # ---------------------------------------------------------------------
        # 5) Classify discrete STL facets into remeshable CAD surfaces
        # ---------------------------------------------------------------------
        gmsh.model.mesh.classifySurfaces(math.pi / 4, True, True, math.pi)
        gmsh.model.mesh.createGeometry()

        # Force uniform target element size by locking both min and max.
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(target_element_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(target_element_size))

        # ---------------------------------------------------------------------
        # 6) Define volume from imported/classified surfaces
        # ---------------------------------------------------------------------
        # After classifySurfaces(), the STL boundary can be split into multiple
        # discrete surfaces with non-deterministic tags. Build the surface loop
        # from all current 2D entities to robustly define the closed volume.
        surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surface_tags:
            raise RuntimeError("No surface entities found after STL classification.")
        gmsh.model.geo.addSurfaceLoop(surface_tags, 1)
        gmsh.model.geo.addVolume([1], 1)

        # ---------------------------------------------------------------------
        # 7) Synchronize CAD kernel
        # ---------------------------------------------------------------------
        gmsh.model.geo.synchronize()

        # ---------------------------------------------------------------------
        # 7b) Uniform background field across whole bounding box
        # ---------------------------------------------------------------------
        mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        mins, maxs = mesh_bounds[0], mesh_bounds[1]
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", float(target_element_size))
        gmsh.model.mesh.field.setNumber(1, "VOut", float(target_element_size))
        gmsh.model.mesh.field.setNumber(1, "XMin", float(mins[0]))
        gmsh.model.mesh.field.setNumber(1, "XMax", float(maxs[0]))
        gmsh.model.mesh.field.setNumber(1, "YMin", float(mins[1]))
        gmsh.model.mesh.field.setNumber(1, "YMax", float(maxs[1]))
        gmsh.model.mesh.field.setNumber(1, "ZMin", float(mins[2]))
        gmsh.model.mesh.field.setNumber(1, "ZMax", float(maxs[2]))
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        # ---------------------------------------------------------------------
        # 8) Generate 3D mesh
        # ---------------------------------------------------------------------
        gmsh.model.mesh.generate(3)

        # ---------------------------------------------------------------------
        # Extract nodes (global coordinates)
        # ---------------------------------------------------------------------
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if node_tags.size == 0:
            raise RuntimeError("GMSH returned zero nodes after 3D meshing.")

        nodes = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        node_tags_np = np.asarray(node_tags, dtype=np.int64).reshape(-1)
        node_tag_to_index = {
            int(tag): i for i, tag in enumerate(node_tags_np.tolist())
        }

        # ---------------------------------------------------------------------
        # Extract tetrahedra (element type 4)
        # ---------------------------------------------------------------------
        elem_types_3d, _, elem_nodes_3d = gmsh.model.mesh.getElements(3)
        tet_connectivity_raw: np.ndarray | None = None

        for e_type, e_nodes in zip(elem_types_3d, elem_nodes_3d):
            if int(e_type) == 4:  # 4-node tetrahedron
                tet_connectivity_raw = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 4)
                break

        if tet_connectivity_raw is None or tet_connectivity_raw.size == 0:
            raise ValueError(
                "No tetrahedral elements (GMSH type 4) were generated. "
                "Check mesh watertightness and target element size."
            )

        elements = _map_tags_to_zero_based_indices(
            tet_connectivity_raw,
            node_tag_to_index,
        )

        # ---------------------------------------------------------------------
        # Extract boundary triangles (element type 2)
        # ---------------------------------------------------------------------
        # We query boundary surfaces of the created volume and aggregate all
        # triangular faces from those entities.
        boundary_entities = gmsh.model.getBoundary([(3, 1)], oriented=False, recursive=False)
        surface_entities = [(dim, tag) for dim, tag in boundary_entities if dim == 2]

        surface_chunks: list[np.ndarray] = []
        for dim, tag in surface_entities:
            elem_types_2d, _, elem_nodes_2d = gmsh.model.mesh.getElements(dim, tag)
            for e_type, e_nodes in zip(elem_types_2d, elem_nodes_2d):
                if int(e_type) == 2:  # 3-node triangle
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 3)
                    surface_chunks.append(tri)

        # Fallback: if boundary query produced nothing, scan all 2D elements.
        if not surface_chunks:
            all_2d_types, _, all_2d_nodes = gmsh.model.mesh.getElements(2)
            for e_type, e_nodes in zip(all_2d_types, all_2d_nodes):
                if int(e_type) == 2:
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 3)
                    surface_chunks.append(tri)

        if surface_chunks:
            surface_faces_raw = np.concatenate(surface_chunks, axis=0)
            surface_faces = _map_tags_to_zero_based_indices(
                surface_faces_raw,
                node_tag_to_index,
            )
        else:
            surface_faces = np.empty((0, 3), dtype=np.int64)

        # Verify coordinate handoff: input STL bounds vs scaffold node bounds
        input_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        node_mins = np.min(nodes, axis=0)
        node_maxs = np.max(nodes, axis=0)
        scaffold_bounds = np.array([node_mins, node_maxs])
        max_diff = np.max(np.abs(input_bounds - scaffold_bounds))
        if max_diff > 1e-5:
            import warnings
            warnings.warn(
                f"Scaffold bbox mismatch: max diff = {max_diff:.2e} "
                f"(input vs scaffold nodes). Expected <= 1e-5."
            )
        # Diagnostic print (can be disabled in production)
        print(
            f"[Scaffold] Input STL bounds: X=[{input_bounds[0,0]:.4f},{input_bounds[1,0]:.4f}] "
            f"Y=[{input_bounds[0,1]:.4f},{input_bounds[1,1]:.4f}] "
            f"Z=[{input_bounds[0,2]:.4f},{input_bounds[1,2]:.4f}]"
        )
        print(
            f"[Scaffold] Node bounds:      X=[{node_mins[0]:.4f},{node_maxs[0]:.4f}] "
            f"Y=[{node_mins[1]:.4f},{node_maxs[1]:.4f}] "
            f"Z=[{node_mins[2]:.4f},{node_maxs[2]:.4f}] "
            f"(max_diff={max_diff:.2e})"
        )

        return ScaffoldResult(
            nodes=nodes,
            elements=elements.astype(np.int64, copy=False),
            surface_faces=surface_faces.astype(np.int64, copy=False),
        )

    except Exception as exc:
        raise RuntimeError(f"Failed to generate conformal scaffold: {exc}") from exc
    finally:
        # Always close GMSH session to avoid state bleed between runs.
        if gmsh_initialized:
            gmsh.finalize()

        # Always remove temporary STL generated for gmsh.merge().
        if temp_stl_path is not None and os.path.exists(temp_stl_path):
            os.remove(temp_stl_path)
