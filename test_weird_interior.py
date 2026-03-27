#!/usr/bin/env python3
"""
Explicit engine on organic STL: scaffold + interior-only topology + PyVista wireframe.

Weird.stl note (GMSH + dirty meshes)
------------------------------------
1. ``createGeometry()`` often fails on organic STLs; ``generate_conformal_scaffold``
   falls back to ``createTopology()`` automatically.

2. If volume meshing then fails with "self-intersecting facets", "HXT 3D mesh failed",
   or "PLC Error", the **surface mesh** needs repair (the STL has intersecting
   triangles even if trimesh reports watertight).

   Optional: ``pip install pymeshfix`` — this script will use it when available to
   run MeshFix before meshing. Alternatively repair Weird.stl in MeshLab / 3D Slicer
   and re-export.

3. The bundled ``test_parts/Weird.stl`` is **not reliably tet-meshable** in GMSH
   even after pymeshfix (MeshFix may print "could not fix everything"). Use
   ``--demo-icosphere`` to run the same pipeline on a smooth closed surface that
   is known to work.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from graphite.explicit import generate_conformal_scaffold, generate_topology

# Weird.stl lives under test_parts/ (repo root)
_ROOT = Path(__file__).resolve().parent
STL_CANDIDATES = (_ROOT / "Weird.stl", _ROOT / "test_parts" / "Weird.stl")


def _find_weird_stl() -> Path:
    for p in STL_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Weird.stl not found. Tried: {', '.join(str(p) for p in STL_CANDIDATES)}"
    )


def _repair_with_pymeshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """MeshFix: close holes / reduce self-intersections when pymeshfix is installed."""
    try:
        import pymeshfix
    except ImportError:
        return mesh

    v = np.asarray(mesh.vertices, dtype=np.float64, order="C")
    f = np.asarray(mesh.faces, dtype=np.int32, order="C")
    v, f = pymeshfix.clean_from_arrays(v, f)
    out = trimesh.Trimesh(v, f, process=True)
    print(
        f"[test_weird_interior] pymeshfix clean_from_arrays: "
        f"{len(mesh.faces)} -> {len(out.faces)} faces"
    )
    return out


def _demo_icosphere_mesh() -> trimesh.Trimesh:
    """Smooth organic-ish closed surface that GMSH can reliably tetrahedralize."""
    m = trimesh.creation.icosphere(subdivisions=4, radius=30.0)
    return trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=True)


def test_weird(demo_icosphere: bool = False) -> None:
    target = 5.0
    if demo_icosphere:
        print("Using --demo-icosphere (smooth sphere; GMSH-friendly).")
        mesh_process = _demo_icosphere_mesh()
    else:
        stl_path = _find_weird_stl()
        print(f"Loading {stl_path} and generating {target}mm scaffold...")
        mesh = trimesh.load(str(stl_path))
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh_process = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)

    # Prefer MeshFix when pymeshfix is installed (handles self-intersections on dirty STLs).
    # Skip for --demo-icosphere (already a clean manifold).
    if (
        not demo_icosphere
        and os.environ.get("GRAPHITE_SKIP_PYMESHFIX", "").lower() not in ("1", "true", "yes")
    ):
        mesh_process = _repair_with_pymeshfix(mesh_process)
    elif not demo_icosphere:
        print("[test_weird_interior] GRAPHITE_SKIP_PYMESHFIX set — skipping pymeshfix.")

    try:
        # HXT (10) tends to work better on repaired organic boundaries than Delaunay (1)
        scaffold = generate_conformal_scaffold(
            mesh_process,
            target_element_size=target,
            algorithm_3d=10,
        )
    except RuntimeError as e:
        err = str(e)
        print(
            "\n[test_weird_interior] Scaffold failed.\n"
            "Common causes for organic STLs:\n"
            "  - Self-intersecting boundary triangles (GMSH HXT/Delaunay both fail)\n"
            "  - Install pymeshfix:  pip install pymeshfix\n"
            "    (this script applies it automatically when importable)\n"
            "  - Or repair Weird.stl externally, then re-run.\n",
            file=sys.stderr,
        )
        raise SystemExit(err) from e

    nodes = scaffold.nodes
    tets = scaffold.elements
    surface_faces = scaffold.surface_faces

    # 2. Isolate Interior Tets
    boundary_nodes = np.unique(surface_faces)
    is_exterior = np.any(np.isin(tets, boundary_nodes), axis=1)
    interior_tets = tets[~is_exterior]

    print(f"Total Tets: {len(tets)} | Interior Tets: {len(interior_tets)}")

    if len(interior_tets) == 0:
        print("Error: The mesh is too thin for the target size to have a pure interior.")
        return

    # 3. Generate Topology (Pure Interior Only)
    print("Extracting Interior Topology...")
    empty_surf = np.empty((0, 3), dtype=np.int64)
    nodes_out, struts = generate_topology(
        nodes,
        interior_tets,
        surface_faces=empty_surf,
        type="rhombic",
        include_surface_cage=False,
        target_element_size=target,
        merge_short_struts=False,
    )

    # 4. Plot Wireframe (VTK line cell format: [2, i, j, 2, i, j, ...])
    print("Plotting...")
    n_lines = len(struts)
    lines = np.empty(n_lines * 3, dtype=np.int64)
    lines[0::3] = 2
    lines[1::3] = struts[:, 0]
    lines[2::3] = struts[:, 1]

    wireframe = pv.PolyData(nodes_out)
    wireframe.lines = lines

    p = pv.Plotter(title="Weird.stl - Pure Interior Lattice")
    p.add_mesh(wireframe, color="black", line_width=2)
    p.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interior-only lattice on organic boundary.")
    ap.add_argument(
        "--demo-icosphere",
        action="store_true",
        help="Use a subdivided icosphere instead of Weird.stl (end-to-end smoke test).",
    )
    args = ap.parse_args()
    test_weird(demo_icosphere=args.demo_icosphere)
