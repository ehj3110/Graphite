"""
Force Hex Diagnostics — Push GMSH to its limits on curved topology.

Loads rounded-cube.stl and applies every hex-forcing option to see how much
of the mesh can be recombined into hexahedra vs tets, prisms, and pyramids.

Prints a diagnostic receipt of element type counts and hex percentage.

Run from Graphite root:
    python -m Universal_Lattice_Engine.tests.test_force_hex_diagnostics
"""

from __future__ import annotations

import math
from pathlib import Path

import gmsh

_root = Path(__file__).resolve().parents[2]
STL_PATH = _root / "test_parts" / "rounded-cube.stl"

# GMSH 1st-order 3D element types: type_id -> (name, nodes_per_elem)
ELEM_TYPE_INFO = {
    4: ("Tetrahedron", 4),
    5: ("Hexahedron", 8),
    6: ("Prism/Wedge", 6),
    7: ("Pyramid", 5),
}


def run_diagnostics() -> None:
    if not STL_PATH.exists():
        raise FileNotFoundError(f"STL not found: {STL_PATH}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("force_hex_diagnostics")

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

    # Hex-forcing options
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4.0)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)       # Force 2D Quads
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)     # Force 3D Hexes
    gmsh.option.setNumber("Mesh.Algorithm", 8)          # Frontal-Delaunay for quads
    # Try Algorithm3D 4 (Frontal) first; fallback to 1 (Delaunay) if incompatible with quads
    algo3d_used = 4
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        err = str(e).lower()
        if "frontal" in err and "quadrangles" in err:
            gmsh.model.mesh.clear()
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.model.mesh.generate(3)
            algo3d_used = 1
        else:
            raise

    # Retrieve all 3D elements
    element_types, _, node_tags = gmsh.model.mesh.getElements(dim=3)

    # Tally counts by type
    counts: dict[int, int] = {}
    for et, nt in zip(element_types, node_tags):
        type_id = int(et)
        name, nodes_per_elem = ELEM_TYPE_INFO.get(type_id, (f"Type {type_id}", 1))
        n_elems = len(nt) // nodes_per_elem
        counts[type_id] = n_elems

    gmsh.finalize()

    # Build receipt
    total = sum(counts.values())
    hex_count = counts.get(5, 0)
    hex_pct = (100.0 * hex_count / total) if total else 0.0

    # Print formatted summary
    print()
    print("=" * 60)
    print("  GMSH FORCE-HEX DIAGNOSTIC RECEIPT")
    print("  Rounded Cube (curved topology)")
    print("=" * 60)
    print(f"  STL: {STL_PATH.name}")
    print(f"  MeshSizeMax: 4.0 mm")
    print(f"  RecombineAll: 1  |  Recombine3DAll: 1")
    print(f"  Algorithm: 8 (Frontal-Delaunay)  |  Algorithm3D: {algo3d_used} ({'Frontal' if algo3d_used == 4 else 'Delaunay'})")
    print("-" * 60)
    print("  ELEMENT TYPE COUNTS")
    print("-" * 60)

    for type_id in sorted(ELEM_TYPE_INFO.keys()):
        name, _ = ELEM_TYPE_INFO[type_id]
        n = counts.get(type_id, 0)
        pct = (100.0 * n / total) if total else 0.0
        print(f"    {name:16} (Type {type_id}): {n:6}  ({pct:5.1f}%)")

    # Handle any unknown types
    for type_id in sorted(counts.keys()):
        if type_id not in ELEM_TYPE_INFO:
            n = counts[type_id]
            pct = (100.0 * n / total) if total else 0.0
            print(f"    Type {type_id}:              {n:6}  ({pct:5.1f}%)")

    print("-" * 60)
    print(f"    {'TOTAL':16}           {total:6}  (100.0%)")
    print("-" * 60)
    print(f"  HEXAHEDRAL SUCCESS: {hex_count} / {total}  =  {hex_pct:.1f}%")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run_diagnostics()
