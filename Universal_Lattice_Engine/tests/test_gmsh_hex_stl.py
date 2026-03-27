"""
GMSH Hex Meshing — Universal Router Test (5 mm and 2.5 mm).

Meshes a 20 mm cube with structured hexahedra (Element Type 5) using GMSH's
Transfinite + RecombineAll pipeline, extracts the (N, 8, 3) node arrays,
and passes them through the universal router with the 'rhombic' rule.

Why OCC geometry (not STL import)?
    GMSH's surface recombination on arbitrary STL-derived BReps rarely
    produces clean type-5 hexahedra; the transfinite approach on an OCC
    box guarantees them.  The geometry is equivalent to test_parts/20mm_cube.stl.

Mesh settings used (per the project specification):
    Mesh.RecombineAll   = 1   force quad faces on all surfaces
    Mesh.Recombine3DAll = 1   promote quad-faced volumes to hexahedra
    Mesh.Algorithm      = 8   Frontal-Delaunay for quads (2-D surfaces)
    Mesh.Algorithm3D    = 1   Delaunay (3-D), prerequisite for 3-D recombination

Outputs:
    output/GMSH_Hex_5mm.stl
    output/GMSH_Hex_2p5mm.stl

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_gmsh_hex_stl
"""

from __future__ import annotations

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

OUTPUT_DIR   = Path(__file__).resolve().parent.parent / "output"
BOX_SIZE     = 20.0   # mm — equivalent to test_parts/20mm_cube.stl
STRUT_RADIUS = 0.3    # mm
NODE_RADIUS  = 0.5    # mm


# ---------------------------------------------------------------------------
# Geometry helpers (shared rendering pattern)
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


def build_manifold(nodes: np.ndarray, struts: np.ndarray) -> manifold3d.Manifold:
    parts: list[manifold3d.Manifold] = []
    for pt in nodes:
        parts.append(
            manifold3d.Manifold.sphere(NODE_RADIUS, circular_segments=12)
            .translate(tuple(float(v) for v in pt))
        )
    for a, b in struts:
        p0, p1 = nodes[int(a)], nodes[int(b)]
        try:
            R, length, mid = _frame_from_segment(p0, p1)
        except ValueError:
            continue
        cyl = manifold3d.Manifold.cylinder(
            height=length, radius_low=STRUT_RADIUS, radius_high=STRUT_RADIUS,
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
# GMSH hex meshing
# ---------------------------------------------------------------------------

def mesh_box_hex(mesh_size: float) -> np.ndarray:
    """
    Use GMSH to mesh a 20 mm box with structured hexahedra.

    Returns
    -------
    ndarray, shape (N, 8, 3)
        Coordinates of the 8 corner nodes for each type-5 hex element,
        in GMSH's standard node ordering (matches hex_rules.py face definitions).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("hex_box")

    # Geometry: 20mm box via OCC (analytically equivalent to test_parts/20mm_cube.stl)
    box_tag = gmsh.model.occ.addBox(0.0, 0.0, 0.0, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    gmsh.model.occ.synchronize()

    # Transfinite structured grid — guarantees axis-aligned hexahedra.
    # n_divs is the number of *nodes* per edge (= intervals + 1).
    n_divs = max(2, int(round(BOX_SIZE / mesh_size)) + 1)

    # Apply transfinite to all 12 edges via getEntities(1) —
    # getBoundary recursive only returns leaf-level points and misses curves.
    for _, tag in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(tag, n_divs)

    # Apply transfinite to all 6 faces
    for _, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(abs(tag))

    # Apply transfinite to the volume
    gmsh.model.mesh.setTransfiniteVolume(box_tag)

    # Recombination settings (as specified)
    gmsh.option.setNumber("Mesh.RecombineAll",   1)   # quad faces on all surfaces
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)   # promote to hexahedra
    gmsh.option.setNumber("Mesh.Algorithm",      8)   # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.Algorithm3D",    1)   # Delaunay 3D
    # Note: MeshSizeMax/Min are intentionally omitted for the structured case;
    # transfinite settings fully control element density and adding size constraints
    # can conflict, producing a different n_divs than requested.

    gmsh.model.mesh.generate(3)

    # Extract all mesh nodes into a lookup dict  {node_tag: (x, y, z)}
    node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
    coords_arr = coords_flat.reshape(-1, 3)
    node_coord = {int(t): coords_arr[i] for i, t in enumerate(node_tags)}

    # Extract type-5 (8-node hexahedron) elements
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

    hex_elements: list[list] = []
    for et, en in zip(elem_types, elem_node_tags):
        if int(et) == 5:
            n_elems = len(en) // 8
            node_matrix = en.reshape(n_elems, 8)
            for row in node_matrix:
                hex_elements.append([node_coord[int(t)] for t in row])

    gmsh.finalize()

    if not hex_elements:
        raise RuntimeError(
            f"GMSH produced 0 type-5 (hex) elements at mesh_size={mesh_size}. "
            "Check that transfinite + RecombineAll settings are working."
        )

    return np.array(hex_elements, dtype=np.float64)   # (N, 8, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_case(label: str, mesh_size: float, out_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{label}  (mesh size = {mesh_size} mm)")
    print("=" * 60)

    hex_elements = mesh_box_hex(mesh_size)
    n_hexes = hex_elements.shape[0]
    expected = int(round(BOX_SIZE / mesh_size)) ** 3
    print(f"Hex elements extracted : {n_hexes}  (expected {expected})")

    nodes, struts = generate_universal_lattice(hex_elements, "star")
    print(f"Router output (star rule):")
    print(f"  Nodes  : {nodes.shape[0]}")
    print(f"  Struts : {struts.shape[0]}")

    lengths = np.linalg.norm(nodes[struts[:, 1]] - nodes[struts[:, 0]], axis=1)
    print(f"  Strut length  min={lengths.min():.3f}  max={lengths.max():.3f}  "
          f"mean={lengths.mean():.3f} mm")

    print("Building Manifold geometry...")
    geo = build_manifold(nodes, struts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / out_name
    mesh = manifold_to_trimesh(geo)
    mesh.export(str(out))
    print(f"Saved: {out}  ({len(mesh.faces):,} triangles)")


def main() -> None:
    print("=" * 60)
    print("GMSH Hex Mesh — Universal Router (Rhombic Rule)")
    print("=" * 60)

    run_case("5 mm mesh",   5.0,  "GMSH_Hex_5mm.stl")
    run_case("2.5 mm mesh", 2.5,  "GMSH_Hex_2p5mm.stl")

    print("\nDone.")


if __name__ == "__main__":
    main()
