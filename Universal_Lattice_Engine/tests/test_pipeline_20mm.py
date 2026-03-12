"""
Production Pipeline — Perfect 20mm Cube (Integer Subdivision Transfinite).

Uses OCC box geometry (equivalent to test_parts/20mm_cube.stl), applies strict
integer subdivision (goal_size=5.0mm, 4 nodes per edge), transfinite surfaces
and volume, generates 3D hex mesh, extracts Type 5 Hexahedra, exports all 4 rules.

Outputs:
    output/Pipeline_20mm_Grid.stl
    output/Pipeline_20mm_Octahedral.stl
    output/Pipeline_20mm_Star.stl
    output/Pipeline_20mm_Octet.stl

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_pipeline_20mm
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
from geometry_module import manifold_to_trimesh

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
STL_PATH   = _root / "test_parts" / "20mm_cube.stl"
GOAL_SIZE  = 5.0   # mm — 20/5 = 4 cells per edge, exactly 5 nodes per edge
THICKNESS  = 0.5   # mm for 5mm cell
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


def mesh_20mm_cube_hex() -> np.ndarray:
    """
    Integer subdivision transfinite, extract Type 5 hexahedra. Returns (N, 8, 3).

    Uses OCC box (equivalent to 20mm_cube.stl) for transfinite compatibility —
    STL-derived geometry often fails setTransfiniteVolume due to surface ordering.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("pipeline_20mm")

    box_size = 20.0
    box_tag = gmsh.model.occ.addBox(0.0, 0.0, 0.0, box_size, box_size, box_size)
    gmsh.model.occ.synchronize()

    num_cells = max(1, round(box_size / GOAL_SIZE))
    num_nodes = num_cells + 1

    for _, tag in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(tag, num_nodes)
    for _, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(abs(tag))
    gmsh.model.mesh.setTransfiniteVolume(box_tag)

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
    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 5:   # 8-node hexahedron
            n_elems = len(en) // 8
            node_matrix = en.reshape(n_elems, 8)
            for row in node_matrix:
                hex_list.append([node_coord[int(t)] for t in row])

    gmsh.finalize()

    if not hex_list:
        raise RuntimeError(
            f"No Type 5 hexahedra. 3D elem types: {list(elem_types)}"
        )
    return np.array(hex_list, dtype=np.float64)


def main() -> None:
    print("=" * 60)
    print("Production Pipeline — 20mm Cube (Integer Subdivision)")
    print("=" * 60)
    print(f"Geometry: 20mm OCC box (equiv. to {STL_PATH.name})")
    print(f"Goal size: {GOAL_SIZE} mm  (4 nodes per edge)")
    print(f"Thickness: {THICKNESS} mm")

    hex_elements = mesh_20mm_cube_hex()
    n_hex = hex_elements.shape[0]
    print(f"Extracted {n_hex} hexahedra")

    strut_r = THICKNESS / 2.0
    node_r = strut_r

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for rule in HEX_RULES:
        nodes, struts = generate_universal_lattice(hex_elements, rule)
        geo = build_manifold(nodes, struts, strut_r, node_r)
        mesh = manifold_to_trimesh(geo)
        out = OUTPUT_DIR / f"Pipeline_20mm_{rule.capitalize()}.stl"
        mesh.export(str(out))
        print(f"  {rule}: {nodes.shape[0]} nodes, {struts.shape[0]} struts -> {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
