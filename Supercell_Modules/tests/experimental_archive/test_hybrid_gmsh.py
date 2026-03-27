"""
Hybrid Core-Shell mesher test using GMSH embedded points.

Pipeline:
  1) Build 40x40x40 cube test STL (centered at origin)
  2) Generate BCC seeds in a bounding box
  3) Keep seeds strictly inside and far from boundary (embedded points)
  4) Import STL in GMSH, create volume, embed points, tetra-mesh (HXT)
  5) Build centroid-dual strut lattice from tetrahedra
  6) Export both full and half-cube cross-section STLs

Run from Graphite root:
    python -m Supercell_Modules.tests.test_hybrid_gmsh
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import manifold3d
import numpy as np
import trimesh

try:
    import gmsh
except ImportError as exc:
    raise ImportError(
        "gmsh is required for this test. Install with `pip install gmsh`."
    ) from exc

# Add Graphite root for geometry_module import
_graphite_root = Path(__file__).resolve().parents[2]
if str(_graphite_root) not in sys.path:
    sys.path.insert(0, str(_graphite_root))

# Add Supercell_Modules for core imports
_supercell_dir = Path(__file__).resolve().parent.parent
if str(_supercell_dir) not in sys.path:
    sys.path.insert(0, str(_supercell_dir))

from graphite.explicit.geometry_module import manifold_to_trimesh
from core.topologies import generate_centroid_dual
from Supercell_Modules.tests.test_triangulation import generate_bcc_seeds


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
CELL_SIZE = 10.0
BOUNDARY_CLEARANCE = CELL_SIZE * 0.5
STRUT_RADIUS = CELL_SIZE * 0.05


def tetra_volumes(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Compute tetra volumes: V = abs(det([b-a, c-a, d-a])) / 6."""
    tets = points[simplices]
    a = tets[:, 0, :]
    b = tets[:, 1, :]
    c = tets[:, 2, :]
    d = tets[:, 3, :]
    m = np.stack((b - a, c - a, d - a), axis=1)
    return np.abs(np.linalg.det(m)) / 6.0


def build_test_surface_stl() -> tuple[trimesh.Trimesh, Path]:
    """Create a 40x40x40 cube (centered) and save STL for GMSH import."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_shape = manifold3d.Manifold.cube([40.0, 40.0, 40.0], center=True)
    surface_mesh = manifold_to_trimesh(test_shape)
    stl_path = OUTPUT_DIR / "Hybrid_Input_Cube.stl"
    surface_mesh.export(str(stl_path))
    return surface_mesh, stl_path


def select_embedded_points(surface_mesh: trimesh.Trimesh, cell_size: float) -> np.ndarray:
    """
    Keep only points strictly inside and at least 0.5*cell_size from boundary.
    """
    bounds = surface_mesh.bounds
    # For [-20, 20] with cell_size=10, grid_size=5 gives exactly 4 macro-cells across.
    grid_size = int(np.round((bounds[1, 0] - bounds[0, 0]) / cell_size)) + 1

    seeds = generate_bcc_seeds(grid_size=grid_size, cell_size=cell_size)
    seeds = seeds + bounds[0]

    inside_mask = surface_mesh.contains(seeds)
    inside_points = seeds[inside_mask]
    if inside_points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    pq = trimesh.proximity.ProximityQuery(surface_mesh)
    signed_dist = pq.signed_distance(inside_points)
    # In trimesh signed_distance, inside points are positive.
    keep = signed_dist > (cell_size * 0.5)
    return inside_points[keep]


def gmsh_tetrahedralize_with_embedded_points(
    stl_path: Path,
    embedded_points: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return mesh nodes (N,3) and tetra simplices (M,4) from GMSH.
    """
    gmsh.initialize()
    try:
        gmsh.model.add("HybridCoreShell")
        gmsh.merge(str(stl_path))

        gmsh.model.mesh.classifySurfaces(
            40.0 * math.pi / 180.0,
            boundary=True,
            forReparametrization=True,
            curveAngle=math.pi,
        )
        gmsh.model.mesh.createGeometry()

        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise RuntimeError("No surfaces recovered from STL import in GMSH.")

        surface_tags = [s[1] for s in surfaces]
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        volume_tag = gmsh.model.geo.addVolume([surface_loop])

        point_tags: list[int] = []
        for xyz in embedded_points:
            tag = gmsh.model.geo.addPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            point_tags.append(tag)

        gmsh.model.geo.synchronize()
        if point_tags:
            gmsh.model.mesh.embed(0, point_tags, 3, volume_tag)

        target_size = float(cell_size) * 0.866
        gmsh.option.setNumber("Mesh.MeshSizeMin", target_size * 0.9)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target_size * 1.1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (2D surfaces)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (3D volume)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3, volume_tag)
        simplices_list: list[list[int]] = []
        for elem_type, nodes_flat in zip(elem_types, elem_node_tags):
            name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            if dim != 3 or num_nodes < 4:
                continue

            arr = np.asarray(nodes_flat, dtype=np.int64).reshape(-1, num_nodes)
            corner_nodes = arr[:, :4]
            for row in corner_nodes:
                simplices_list.append([tag_to_idx[int(t)] for t in row])

        if not simplices_list:
            raise RuntimeError("No tetrahedra extracted from GMSH volume mesh.")

        simplices = np.asarray(simplices_list, dtype=np.int64)
        return points, simplices
    finally:
        gmsh.finalize()


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Build local frame for a cylinder aligned to segment p0->p1.
    Returns (R, length, midpoint), where R columns are local x,y,z axes in world frame.
    """
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise ValueError("Zero-length segment.")

    z_axis = vec / length
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(z_axis, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack((x_axis, y_axis, z_axis))
    midpoint = 0.5 * (p0 + p1)
    return R, length, midpoint


def build_dual_strut_manifold(
    dual_nodes: np.ndarray,
    dual_struts: np.ndarray,
    radius: float,
) -> manifold3d.Manifold:
    """Build centroid-dual as manifold3d cylinder union."""
    cylinders: list[manifold3d.Manifold] = []
    for i, j in dual_struts:
        p0 = dual_nodes[int(i)]
        p1 = dual_nodes[int(j)]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 1e-9:
            continue
        R, seg_len, mid = _frame_from_segment(p0, p1)
        cyl = manifold3d.Manifold.cylinder(
            height=seg_len,
            radius_low=radius,
            radius_high=radius,
            circular_segments=18,
            center=True,
        )
        # 3x4 affine matrix for manifold3d.transform
        affine = [
            [float(R[0, 0]), float(R[0, 1]), float(R[0, 2]), float(mid[0])],
            [float(R[1, 0]), float(R[1, 1]), float(R[1, 2]), float(mid[1])],
            [float(R[2, 0]), float(R[2, 1]), float(R[2, 2]), float(mid[2])],
        ]
        cylinders.append(cyl.transform(affine))

    if not cylinders:
        raise RuntimeError("No valid dual strut cylinders were created.")
    return manifold3d.Manifold.compose(cylinders)


def main() -> None:
    print("=" * 72)
    print("HYBRID CORE-SHELL GMSH TEST")
    print("=" * 72)
    print(f"CELL_SIZE = {CELL_SIZE}")
    print(f"BOUNDARY_CLEARANCE = {BOUNDARY_CLEARANCE}")

    surface_mesh, temp_stl_path = build_test_surface_stl()
    print(f"Input surface STL: {temp_stl_path}")
    print(f"Surface watertight: {surface_mesh.is_watertight}")

    embedded_points = select_embedded_points(surface_mesh, CELL_SIZE)
    print(f"Embedded points kept: {embedded_points.shape[0]}")
    if embedded_points.shape[0] == 0:
        raise RuntimeError("No embedded points survived clearance filter.")

    points, simplices = gmsh_tetrahedralize_with_embedded_points(
        temp_stl_path, embedded_points, CELL_SIZE
    )
    print(f"GMSH tetrahedra (raw): {simplices.shape[0]}")

    volumes = tetra_volumes(points, simplices)
    clean_mask = volumes >= 1e-8
    clean_simplices = simplices[clean_mask]
    clean_vols = volumes[clean_mask]
    print(f"GMSH tetrahedra (clean): {clean_simplices.shape[0]}")
    print(
        "Tet volume stats (clean) "
        f"min={clean_vols.min():.6f}, max={clean_vols.max():.6f}, avg={clean_vols.mean():.6f}"
    )

    dual_nodes, dual_struts = generate_centroid_dual(points, clean_simplices)
    print(f"Dual nodes: {dual_nodes.shape[0]}")
    print(f"Dual struts: {dual_struts.shape[0]}")

    if dual_struts.shape[0] > 0:
        lengths = np.linalg.norm(
            dual_nodes[dual_struts[:, 1]] - dual_nodes[dual_struts[:, 0]],
            axis=1,
        )
        print(
            "Dual strut lengths "
            f"min={lengths.min():.6f}, max={lengths.max():.6f}, avg={lengths.mean():.6f}"
        )

    full_lattice = build_dual_strut_manifold(dual_nodes, dual_struts, STRUT_RADIUS)

    full_stl_path = OUTPUT_DIR / "Hybrid_Kagome_FullCube.stl"
    manifold_to_trimesh(full_lattice).export(str(full_stl_path))
    print(f"Exported full lattice: {full_stl_path}")

    cutter = manifold3d.Manifold.cube([50.0, 50.0, 50.0]).translate([-25.0, 0.0, -25.0])
    half_lattice = full_lattice ^ cutter
    half_stl_path = OUTPUT_DIR / "Hybrid_Kagome_HalfCube.stl"
    manifold_to_trimesh(half_lattice).export(str(half_stl_path))
    print(f"Exported cross-section lattice: {half_stl_path}")
    print("Done.")


if __name__ == "__main__":
    main()
