"""
Sweep rack mesh diagnostic — test GMSH 3D tet generation for MariaTubeRack.

Loads MariaTubeRack_Full.STL and attempts the scaffolding (3D meshing) step
for each target_element_size. For 5.0 and 7.5 mm, applies adaptive settings
to see if they prevent failure.

Run from project root: python tools/diagnostics/sweep_rack_mesh.py
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import gmsh
import numpy as np
import trimesh

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STL_PATH = _PROJECT_ROOT / "test_parts" / "MariaTubeRack_Full.STL"
ELEMENT_SIZES = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 25.4]
ADAPTIVE_SIZES = {5.0, 7.5}  # Apply MeshSizeFromCurvature, MeshSizeExtendFromBoundary


def _run_gmsh_scaffold(
    stl_path: Path,
    element_size: float,
    use_adaptive: bool,
) -> tuple[str, str]:
    """
    Initialize GMSH, import STL, attempt 3D tet mesh (scaffolding only).
    Returns (status, detail) where status is "SUCCESS" or "FAIL".
    """
    temp_stl = None
    gmsh_initialized = False

    try:
        mesh = trimesh.load(str(stl_path), force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            temp_stl = tmp.name
        mesh.export(temp_stl)

        gmsh.initialize()
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.option.setNumber("Mesh.MeshSizeMax", float(element_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(element_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(element_size))
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

        if use_adaptive:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

        gmsh.merge(temp_stl)
        gmsh.model.mesh.classifySurfaces(math.pi / 4, True, True, math.pi)
        gmsh.model.mesh.createGeometry()

        surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surface_tags:
            return "FAIL", "No surface entities after STL classification"

        gmsh.model.geo.addSurfaceLoop(surface_tags, 1)
        gmsh.model.geo.addVolume([1], 1)
        gmsh.model.geo.synchronize()

        mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        mins, maxs = mesh_bounds[0], mesh_bounds[1]
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", float(element_size))
        gmsh.model.mesh.field.setNumber(1, "VOut", float(element_size))
        gmsh.model.mesh.field.setNumber(1, "XMin", float(mins[0]))
        gmsh.model.mesh.field.setNumber(1, "XMax", float(maxs[0]))
        gmsh.model.mesh.field.setNumber(1, "YMin", float(mins[1]))
        gmsh.model.mesh.field.setNumber(1, "YMax", float(maxs[1]))
        gmsh.model.mesh.field.setNumber(1, "ZMin", float(mins[2]))
        gmsh.model.mesh.field.setNumber(1, "ZMax", float(maxs[2]))
        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        gmsh.model.mesh.generate(3)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = node_tags.size if node_tags is not None else 0
        return "SUCCESS", f"{n_nodes} nodes"

    except Exception as e:
        return "FAIL", str(e)
    finally:
        if gmsh_initialized:
            gmsh.finalize()
        if temp_stl and os.path.exists(temp_stl):
            os.remove(temp_stl)


def main() -> None:
    if not STL_PATH.exists():
        print(f"Error: {STL_PATH} not found.")
        return

    print("=" * 75)
    print("GMSH 3D Scaffolding Sweep — MariaTubeRack_Full")
    print("=" * 75)
    print(f"STL: {STL_PATH.resolve()}")
    print(f"Element sizes (mm): {ELEMENT_SIZES}")
    print(f"Adaptive settings (5.0, 7.5 mm): MeshSizeFromCurvature=1, MeshSizeExtendFromBoundary=1")
    print()

    results = []
    for size in ELEMENT_SIZES:
        use_adaptive = size in ADAPTIVE_SIZES
        status, detail = _run_gmsh_scaffold(STL_PATH, size, use_adaptive)
        suffix = " (adaptive)" if use_adaptive else ""
        results.append((size, status, detail, use_adaptive))

    # Table
    col_size = 12
    col_status = 10
    col_detail = 45
    print(f"{'Size (mm)':<{col_size}} {'Status':<{col_status}} {'Detail'}")
    print("-" * 75)
    for size, status, detail, use_adaptive in results:
        suffix = " [adaptive]" if use_adaptive else ""
        detail_short = (detail[: col_detail - 3] + "...") if len(detail) > col_detail else detail
        detail_short = detail_short.replace("\n", " ")
        print(f"{size:<{col_size}} {status:<{col_status}} {detail_short}{suffix}")

    print("=" * 75)

    # Summary: which sizes accepted
    accepted = [r[0] for r in results if r[1] == "SUCCESS"]
    failed = [r[0] for r in results if r[1] == "FAIL"]
    print(f"\nAccepted: {accepted if accepted else 'none'}")
    print(f"Failed:   {failed if failed else 'none'}")

    # Full errors for failures
    for size, status, detail, _ in results:
        if status == "FAIL":
            print(f"\n--- FAIL at {size} mm (full error) ---")
            print(detail)


if __name__ == "__main__":
    main()
