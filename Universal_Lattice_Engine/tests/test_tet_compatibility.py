"""
Backward Compatibility Split Test — Tetrahedra via Universal Router.

Proves the Universal Router handles 4-node tetrahedral elements correctly by:
  1) Meshing a 20mm box STL with GMSH (pure tetrahedra, recombination OFF)
  2) Extracting Element Type 4 (tetrahedron) into (N, 4, 3) format
  3) Passing to generate_universal_lattice(..., 'kagome')
  4) Rendering with dynamic scaling and exporting the lattice STL

Raw mesh exported as .msh to preserve internal 3D elements for visual inspection.

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_tet_compatibility
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import gmsh
import manifold3d
import numpy as np

_root = Path(__file__).resolve().parents[2]
_ule = Path(__file__).resolve().parent.parent
for _p in (_root, _ule):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Universal_Lattice_Engine.core.topology_module import generate_universal_lattice
from graphite.explicit.geometry_module import manifold_to_trimesh

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
STL_PATH   = _root / "test_parts" / "20mm_cube.stl"   # 20mm box geometry


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _frame_from_segment(
    p0: np.ndarray, p1: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        raise ValueError("Zero-length segment.")
    z = vec / length
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, ref)) > 0.98:
        ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x);   y /= np.linalg.norm(y)
    R = np.column_stack((x, y, z))
    return R, length, 0.5 * (p0 + p1)


def build_manifold(
    nodes: np.ndarray,
    struts: np.ndarray,
    strut_radius: float,
    node_radius: float,
) -> manifold3d.Manifold:
    parts: list[manifold3d.Manifold] = []
    for pt in nodes:
        parts.append(
            manifold3d.Manifold.sphere(node_radius, circular_segments=12)
            .translate(tuple(float(v) for v in pt))
        )
    for a, b in struts:
        p0, p1 = nodes[int(a)], nodes[int(b)]
        try:
            R, length, mid = _frame_from_segment(p0, p1)
        except ValueError:
            continue
        cyl = manifold3d.Manifold.cylinder(
            height=length, radius_low=strut_radius, radius_high=strut_radius,
            circular_segments=10, center=True,
        )
        affine = [
            [float(R[0,0]), float(R[0,1]), float(R[0,2]), float(mid[0])],
            [float(R[1,0]), float(R[1,1]), float(R[1,2]), float(mid[1])],
            [float(R[2,0]), float(R[2,1]), float(R[2,2]), float(mid[2])],
        ]
        parts.append(cyl.transform(affine))
    if not parts:
        raise RuntimeError("No geometry produced.")
    return manifold3d.Manifold.compose(parts)


# ---------------------------------------------------------------------------
# GMSH tetrahedral meshing
# ---------------------------------------------------------------------------

def mesh_box_tets(target_size: float) -> tuple[int, np.ndarray]:
    """
    Mesh the 20mm box STL with pure tetrahedra. Recombination strictly OFF.

    Returns
    -------
    n_tets : int
    tet_coords : ndarray, shape (N, 4, 3)
        Corner coordinates for each tetrahedron (Element Type 4).
    """
    if not STL_PATH.exists():
        raise FileNotFoundError(
            f"STL not found: {STL_PATH}. "
            "Ensure test_parts/20mm_cube.stl exists (20mm box geometry)."
        )

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("tet_box")

    gmsh.merge(str(STL_PATH))

    gmsh.model.mesh.classifySurfaces(
        40.0 * math.pi / 180.0,
        boundary=True,
        forReparametrization=True,
        curveAngle=math.pi,
    )
    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        gmsh.finalize()
        raise RuntimeError("No surfaces after classifySurfaces — check STL is closed.")

    surface_tags = [s[1] for s in surfaces]
    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(surface_tags)])
    gmsh.model.geo.synchronize()

    # Pure tetrahedra — recombination strictly OFF
    gmsh.option.setNumber("Mesh.RecombineAll", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)    # Frontal-Delaunay (2D)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT (3D)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", target_size * 0.9)

    gmsh.model.mesh.generate(3)

    # Export raw mesh as .msh (internal 3D) and surface as .stl (for visualization)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"Raw_Tet_Mesh_{target_size}mm.msh"
    gmsh.write(str(raw_path))
    print(f"  Raw mesh (.msh): {raw_path}")
    surface_path = OUTPUT_DIR / f"Raw_Tet_Mesh_{target_size}mm_Surface.stl"
    gmsh.write(str(surface_path))
    print(f"  Surface (.stl): {surface_path}")

    # Extract Element Type 4 (4-node tetrahedron)
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords_arr = coords_flat.reshape(-1, 3)
    node_coord = {int(t): coords_arr[i] for i, t in enumerate(node_tags)}

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    tet_coords_list: list[np.ndarray] = []

    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 4:   # GMSH type 4 = 4-node tetrahedron
            n_elems = len(en) // 4
            node_matrix = en.reshape(n_elems, 4)
            for row in node_matrix:
                tet_coords_list.append(np.array([node_coord[int(t)] for t in row]))

    gmsh.finalize()

    if not tet_coords_list:
        raise RuntimeError(
            f"GMSH produced 0 type-4 (tetrahedron) elements at target_size={target_size}. "
            "Check that RecombineAll=0 and Algorithm3D=10 are set."
        )

    tet_coords = np.array(tet_coords_list, dtype=np.float64)   # (N, 4, 3)
    return len(tet_coords_list), tet_coords


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_case(target_size: float) -> None:
    print(f"\n{'=' * 60}")
    print(f"Target size: {target_size} mm")
    print("=" * 60)

    n_tets, tet_elements = mesh_box_tets(target_size)
    print(f"  Tetrahedra extracted: {n_tets}")

    nodes, struts = generate_universal_lattice(tet_elements, "kagome")
    print(f"  Router output (kagome): {nodes.shape[0]} nodes, {struts.shape[0]} struts")

    lengths = np.linalg.norm(nodes[struts[:, 1]] - nodes[struts[:, 0]], axis=1)
    print(f"  Strut length  min={lengths.min():.3f}  max={lengths.max():.3f}  "
          f"mean={lengths.mean():.3f} mm")

    # Dynamic rendering scale: thickness = target_size/10, strut_radius = thickness/2,
    # node_radius = strut_radius (sphere diameter = strut thickness)
    thickness = target_size / 10.0
    strut_radius = thickness / 2.0
    node_radius = strut_radius
    print(f"  Dynamic scale: thickness={thickness:.3f}  strut_r={strut_radius:.3f}  "
          f"node_r={node_radius:.3f} mm")

    print("  Building Manifold geometry...")
    geo = build_manifold(nodes, struts, strut_radius, node_radius)

    lattice_path = OUTPUT_DIR / f"Lattice_Tet_Kagome_{target_size}mm_Scaled.stl"
    mesh = manifold_to_trimesh(geo)
    mesh.export(str(lattice_path))
    print(f"  Lattice STL: {lattice_path}  ({len(mesh.faces):,} triangles)")


def main() -> None:
    print("=" * 60)
    print("Backward Compatibility — Tetrahedra via Universal Router")
    print("=" * 60)
    print(f"STL: {STL_PATH}")

    for target_size in [5.0, 2.5]:
        run_case(target_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
