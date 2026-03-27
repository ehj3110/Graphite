"""
Production Pipeline — Rounded Cube (Natural Meshing).

Loads rounded-cube.stl, applies natural meshing (MeshSizeMax=5.0,
RecombineAll=1, Algorithm=8). NO Transfinite — fails on fillets.
Generates 3D volume mesh and extracts hexahedra or tetrahedra.

Curved topology often yields tetrahedra; when so, uses tet rules
(voronoi, kagome, icosahedral, rhombic) instead of hex rules.

Outputs (hex case):
    output/Pipeline_Rounded_Grid.stl, Octahedral, Star, Octet
Outputs (tet case):
    output/Pipeline_Rounded_Voronoi.stl, Kagome, Icosahedral, Rhombic

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_pipeline_rounded
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
STL_PATH   = _root / "test_parts" / "rounded-cube.stl"
MESH_SIZE  = 5.0   # mm
THICKNESS  = 0.5   # mm
HEX_RULES  = ["grid", "octahedral", "star", "octet"]


def _frame_from_segment(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
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


def mesh_rounded_cube_hex() -> tuple[np.ndarray, str]:
    """
    Natural meshing (no Transfinite). Returns (elements, "hex"|"tet").
    Curved topology often yields tets; hex when recombination succeeds.
    """
    if not STL_PATH.exists():
        raise FileNotFoundError(f"STL not found: {STL_PATH}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("pipeline_rounded")

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
        raise RuntimeError("No surfaces after classifySurfaces.")

    surface_tags = [s[1] for s in surfaces]
    gmsh.model.geo.addVolume([gmsh.model.geo.addSurfaceLoop(surface_tags)])
    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.mesh.generate(3)

    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords_arr = coords_flat.reshape(-1, 3)
    node_coord = {int(t): coords_arr[i] for i, t in enumerate(node_tags)}

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    hex_list: list[list] = []
    tet_list: list[list] = []
    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 5:
            n_elems = len(en) // 8
            node_matrix = en.reshape(n_elems, 8)
            for row in node_matrix:
                hex_list.append([node_coord[int(t)] for t in row])
        elif int(et) == 4:
            n_elems = len(en) // 4
            node_matrix = en.reshape(n_elems, 4)
            for row in node_matrix:
                tet_list.append([node_coord[int(t)] for t in row])

    gmsh.finalize()

    if hex_list:
        return np.array(hex_list, dtype=np.float64), "hex"
    if tet_list:
        return np.array(tet_list, dtype=np.float64), "tet"
    raise RuntimeError(
        f"No hex or tet elements. 3D elem types: {list(elem_types)}"
    )


def main() -> None:
    print("=" * 60)
    print("Production Pipeline — Rounded Cube (Natural Meshing)")
    print("=" * 60)
    print(f"STL: {STL_PATH}")
    print(f"MeshSizeMax: {MESH_SIZE} mm  (no Transfinite)")
    print(f"Thickness: {THICKNESS} mm")

    elements, elem_type = mesh_rounded_cube_hex()
    n_elem = elements.shape[0]
    label = "hexahedra" if elem_type == "hex" else "tetrahedra"
    print(f"Extracted {n_elem} {label}")

    strut_r = THICKNESS / 2.0
    node_r = strut_r

    rules = HEX_RULES if elem_type == "hex" else ["voronoi", "kagome", "icosahedral", "rhombic"]
    if elem_type == "tet":
        print("  (Curved topology produced tets; using tet rules)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rule in rules:
        nodes, struts = generate_universal_lattice(elements, rule)
        geo = build_manifold(nodes, struts, strut_r, node_r)
        mesh = manifold_to_trimesh(geo)
        out = OUTPUT_DIR / f"Pipeline_Rounded_{rule.capitalize()}.stl"
        mesh.export(str(out))
        print(f"  {rule}: {nodes.shape[0]} nodes, {struts.shape[0]} struts -> {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
