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
    - nodes: global XYZ coordinates (includes edge mid-nodes when order=2)
    - elements: tetrahedral connectivity — (M, 4) for linear, (M, 10) for quadratic
    - surface_faces: boundary triangles — (K, 3) linear or (K, 6) quadratic corners/mids
    - element_order: 1 (linear) or 2 (quadratic)
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
        elements: (M, 4) or (M, 10) tet connectivity (zero-based). For order 2,
            use ``elements[:, :4]`` for corner-only topology / masks.
        surface_faces: (K, 3) or (K, 6) boundary triangles (zero-based indices).
        element_order: 1 = linear P1 tets, 2 = quadratic P2 tets (10 nodes).
    """

    nodes: np.ndarray
    elements: np.ndarray
    surface_faces: np.ndarray
    element_order: int = 1


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


def _compute_tet_sicn(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """
    Compute SICN (Scaled Inscribed Circumscribed Ratio) for each tetrahedron.
    SICN = 3 * r_in / R_circum; 1 = perfect equilateral, 0 = degenerate.
    """
    n_tet = elements.shape[0]
    qualities = np.ones(n_tet, dtype=np.float64)
    for i in range(n_tet):
        p = nodes[elements[i]]  # (4, 3)
        v0, v1, v2, v3 = p[0], p[1], p[2], p[3]
        vol = abs(np.linalg.det(np.column_stack([v1 - v0, v2 - v0, v3 - v0]))) / 6.0
        if vol <= 1e-15:
            qualities[i] = 0.0
            continue
        # Face areas (3 edges per face)
        def face_area(a, b, c):
            return 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        s = (
            face_area(v0, v1, v2) + face_area(v0, v1, v3) + face_area(v0, v2, v3)
            + face_area(v1, v2, v3)
        )
        if s <= 1e-15:
            qualities[i] = 0.0
            continue
        r_in = 3.0 * vol / s
        # Circumcenter: solve (p_i - p_0) · (2c - p_0) = |p_i - p_0|^2 for i=1,2,3
        A = np.array([v1 - v0, v2 - v0, v3 - v0])
        b = 0.5 * np.array(
            [
                np.dot(v1, v1) - np.dot(v0, v0),
                np.dot(v2, v2) - np.dot(v0, v0),
                np.dot(v3, v3) - np.dot(v0, v0),
            ]
        )
        try:
            c = np.linalg.solve(A, b)
            R_circum = float(np.linalg.norm(c - v0))
            if R_circum <= 1e-15:
                qualities[i] = 0.0
            else:
                qualities[i] = min(1.0, 3.0 * r_in / R_circum)
        except np.linalg.LinAlgError:
            qualities[i] = 0.0
    return qualities


def _write_quality_vtk(
    nodes: np.ndarray,
    elements: np.ndarray,
    qualities: np.ndarray,
    path: Path,
) -> None:
    """Write Legacy VTK file with SICN quality as cell data."""
    path = Path(path)
    n_nodes, n_cells = nodes.shape[0], elements.shape[0]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nMesh Quality (SICN)\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_nodes} double\n")
        for i in range(n_nodes):
            f.write(f"{nodes[i, 0]:.6e} {nodes[i, 1]:.6e} {nodes[i, 2]:.6e}\n")
        f.write(f"CELLS {n_cells} {5 * n_cells}\n")
        for i in range(n_cells):
            f.write(f"4 {elements[i, 0]} {elements[i, 1]} {elements[i, 2]} {elements[i, 3]}\n")
        f.write(f"CELL_TYPES {n_cells}\n")
        f.write((f"10\n" * n_cells))  # 10 = VTK_TETRA
        f.write(f"CELL_DATA {n_cells}\nSCALARS SICN double 1\nLOOKUP_TABLE default\n")
        for q in qualities:
            f.write(f"{q:.6e}\n")


def _gmsh_discrete_surface_to_cad(
    classify_angle: float = math.pi / 4,
    curve_angle: float = math.pi,
) -> None:
    """
    Build a B-rep from a merged STL so volume meshing can proceed.

    ``createGeometry()`` adds a parametrization; it often fails on organic /
    noisy STLs ("Wrong topology of boundary mesh for parametrization"). In that
    case we fall back to ``createTopology()``, which exports a discrete boundary
    representation without full surface parametrization.
    """
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.model.mesh.removeDuplicateElements()
    gmsh.model.mesh.classifySurfaces(classify_angle, True, True, curve_angle)
    try:
        gmsh.model.mesh.createGeometry()
    except Exception as err:
        print(
            "[Scaffold] createGeometry failed; trying createTopology() "
            f"(discrete BRep). Original: {err}"
        )
        try:
            gmsh.model.mesh.createTopology(True, True)
        except Exception as err2:
            raise RuntimeError(
                "Could not build CAD from STL: both createGeometry() and "
                f"createTopology() failed. Last error: {err2}"
            ) from err2


def generate_conformal_scaffold(
    mesh: trimesh.Trimesh,
    target_element_size: float,
    algorithm_3d: int = 1,
    export_quality_path: str | Path | None = None,
    stl_geometry_tolerance: float | None = None,
    element_order: int = 1,
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
        algorithm_3d: GMSH 3D volume mesher (1=Delaunay, 10=HXT, etc.).
            Default 1 (standard Delaunay); use 10 for HXT when appropriate.
        export_quality_path: If set, compute SICN mesh quality, export to
            `{path}.pos` and `{path}.vtk`, and print average quality.
        stl_geometry_tolerance: If set, passed to GMSH ``Geometry.Tolerance``
            before ``merge()`` (helps some STLs with near-coincident geometry).
            If ``None``, a value scaled to the mesh bounding-box diagonal is used.
        element_order: ``1`` (linear) or ``2`` (quadratic). For ``2``, the mesh is
            generated linearly, then ``gmsh.model.mesh.setOrder(2)`` is applied.
            Strict high-order Jacobian optimization is disabled for stability on
            large/tight-curvature meshes. Returned ``elements`` have shape
            ``(M, 10)`` (GMSH type 11 tetrahedra).

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
    if int(element_order) not in (1, 2):
        raise ValueError("`element_order` must be 1 or 2.")

    target_size = float(target_element_size)
    element_order = int(element_order)
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

        # Geometry tolerance for STL merge / discrete CAD (reduces "overlapping facets")
        try:
            if stl_geometry_tolerance is not None:
                tol = float(stl_geometry_tolerance)
            else:
                lo = np.min(mesh.vertices, axis=0)
                hi = np.max(mesh.vertices, axis=0)
                diag = float(np.linalg.norm(hi - lo))
                tol = max(1e-9, min(1e-2, diag * 1e-5))
            gmsh.option.setNumber("Geometry.Tolerance", tol)
        except Exception:
            pass

        # ---------------------------------------------------------------------
        # 3) Global meshing options — uniform target + aggressive optimization
        # ---------------------------------------------------------------------
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)
        try:
            gmsh.option.setNumber("Mesh.Grading", 1.05)
        except Exception:
            pass
        # Aggressive smoothing / optimizers (Netgen + Laplacian-style passes)
        try:
            gmsh.option.setNumber("Mesh.Optimize", 1)
        except Exception:
            pass
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 100)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (surface)
        gmsh.option.setNumber("Mesh.Algorithm3D", int(algorithm_3d))

        # ---------------------------------------------------------------------
        # 4) Merge temporary STL
        # ---------------------------------------------------------------------
        gmsh.merge(temp_stl_path)

        # ---------------------------------------------------------------------
        # 5) Discrete STL → CAD (parametric or topology fallback)
        # ---------------------------------------------------------------------
        _gmsh_discrete_surface_to_cad()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

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
        # 7b) Re-assert uniform size before volume mesh (geometry sync can reset)
        # ---------------------------------------------------------------------
        # Strictly lock the element size to target and disable adaptive refiners.
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MinimumCurveNodes", 2)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", target_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", target_size)

        # ---------------------------------------------------------------------
        # 8) Generate 3D mesh
        # ---------------------------------------------------------------------
        gmsh.model.mesh.generate(3)

        # ---------------------------------------------------------------------
        # 9) Brute-force post-mesh optimization (Gmsh + Netgen) — linear mesh
        # ---------------------------------------------------------------------
        try:
            for _ in range(3):
                gmsh.model.mesh.optimize("Gmsh")
                gmsh.model.mesh.optimize("Netgen")
        except Exception as e:
            print(f"[Scaffold] Optimization warning: {e}")

        # ---------------------------------------------------------------------
        # 9b) Second-order (quadratic) tets: stable conversion only
        # ---------------------------------------------------------------------
        if element_order == 2:
            # Convert linear tets to quadratic (10-node) tets.
            gmsh.model.mesh.setOrder(2)
            # Turn off strict Jacobian-based high-order optimization to avoid
            # fatal crashes on large elements over tight curves.
            gmsh.option.setNumber("Mesh.HighOrderOptimize", 0)
            # Optional: HighOrderElastic can be attempted later if needed.

        # ---------------------------------------------------------------------
        # Extract nodes (global coordinates) — after setOrder so mid-nodes exist
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
        # Extract tetrahedra: type 4 (P1) or 11 (P2, 10 nodes)
        # ---------------------------------------------------------------------
        elem_types_3d, elem_tags_3d, elem_nodes_3d = gmsh.model.mesh.getElements(3)
        tet_connectivity_raw: np.ndarray | None = None
        tet_elem_tags: np.ndarray | None = None
        tet_gmsh_type: int | None = None

        for e_type, e_tags, e_nodes in zip(elem_types_3d, elem_tags_3d, elem_nodes_3d):
            et = int(e_type)
            if element_order == 1 and et == 4:
                tet_connectivity_raw = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 4)
                tet_elem_tags = np.asarray(e_tags, dtype=np.int64)
                tet_gmsh_type = et
                break
            if element_order == 2 and et == 11:
                tet_connectivity_raw = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 10)
                tet_elem_tags = np.asarray(e_tags, dtype=np.int64)
                tet_gmsh_type = et
                break

        if tet_connectivity_raw is None or tet_connectivity_raw.size == 0:
            raise ValueError(
                "No tetrahedral elements were extracted. "
                f"Expected GMSH type {'11 (10-node)' if element_order == 2 else '4 (4-node)'}; "
                f"got types {list(elem_types_3d)}. Check mesh and element_order."
            )

        elements = _map_tags_to_zero_based_indices(
            tet_connectivity_raw,
            node_tag_to_index,
        )
        if element_order == 2:
            print(
                f"[Scaffold] Quadratic tets: connectivity shape {elements.shape} "
                f"(GMSH type {tet_gmsh_type})"
            )

        # ---------------------------------------------------------------------
        # Extract boundary triangles: type 2 (P1) or 9 (P2, 6-node)
        # ---------------------------------------------------------------------
        boundary_entities = gmsh.model.getBoundary([(3, 1)], oriented=False, recursive=False)
        surface_entities = [(dim, tag) for dim, tag in boundary_entities if dim == 2]

        surface_chunks: list[np.ndarray] = []
        for dim, tag in surface_entities:
            elem_types_2d, _, elem_nodes_2d = gmsh.model.mesh.getElements(dim, tag)
            for e_type, e_nodes in zip(elem_types_2d, elem_nodes_2d):
                et = int(e_type)
                if et == 2:
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 3)
                    surface_chunks.append(tri)
                elif et == 9 and element_order == 2:
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 6)
                    surface_chunks.append(tri)

        if not surface_chunks:
            all_2d_types, _, all_2d_nodes = gmsh.model.mesh.getElements(2)
            for e_type, e_nodes in zip(all_2d_types, all_2d_nodes):
                et = int(e_type)
                if et == 2:
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 3)
                    surface_chunks.append(tri)
                elif et == 9 and element_order == 2:
                    tri = np.asarray(e_nodes, dtype=np.int64).reshape(-1, 6)
                    surface_chunks.append(tri)

        if surface_chunks:
            surface_faces_raw = np.concatenate(surface_chunks, axis=0)
            surface_faces = _map_tags_to_zero_based_indices(
                surface_faces_raw,
                node_tag_to_index,
            )
        else:
            n_surf_cols = 6 if element_order == 2 else 3
            surface_faces = np.empty((0, n_surf_cols), dtype=np.int64)

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

        # ---------------------------------------------------------------------
        # Quality heatmap export (SICN metric)
        # ---------------------------------------------------------------------
        if export_quality_path is not None:
            base = Path(export_quality_path)
            pos_path = base.with_suffix(".pos") if base.suffix != ".pos" else base
            vtk_path = base.parent / (
                base.stem.replace("_Mesh_Quality", "") + "_Quality_Map.vtk"
            )
            corners = elements[:, :4] if elements.shape[1] > 4 else elements
            qualities = _compute_tet_sicn(nodes, corners)
            avg_quality = float(np.mean(qualities))
            print(f"[Scaffold] Average Mesh Quality (SICN): {avg_quality:.6f}")
            try:
                model_name = gmsh.model.getCurrent()
                view_tag = gmsh.view.add("Mesh Quality (SICN)")
                data = [[q] for q in qualities.tolist()]
                gmsh.view.add_model_data(
                    view_tag, 0, model_name, "ElementData",
                    tet_elem_tags.tolist(), data, numComponents=1
                )
                gmsh.view.write(view_tag, str(pos_path))
                gmsh.view.write(view_tag, str(vtk_path))
                print(f"[Scaffold] Quality exported: {pos_path}, {vtk_path}")
            except Exception as qe:
                import warnings
                warnings.warn(f"Quality export failed: {qe}", UserWarning, stacklevel=2)
                # Fallback: write simple VTK with cell data
                _write_quality_vtk(nodes, corners, qualities, vtk_path)

        return ScaffoldResult(
            nodes=nodes,
            elements=elements.astype(np.int64, copy=False),
            surface_faces=surface_faces.astype(np.int64, copy=False),
            element_order=element_order,
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
