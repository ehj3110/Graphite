"""
Sweep box mesh diagnostic — test GMSH 3D tet generation across element sizes.

Loads 20mm_Cube_Check.stl and attempts gmsh.model.mesh.generate(3)
for each target_element_size. Reports SUCCESS or FAIL with error details.

Run from project root: python tools/diagnostics/sweep_box_mesh.py
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
STL_PATH = _PROJECT_ROOT / "results" / "tests" / "20mm_Cube_Check.stl"
if not STL_PATH.exists():
    STL_PATH = _PROJECT_ROOT / "20mm_Cube_Check.stl"
ELEMENT_SIZES = [2.0, 5.0, 10.0, 20.0, 24.0]  # 10% to 120% of 20mm box


def _ensure_cube_stl() -> Path:
    """Create 20mm cube STL if it doesn't exist."""
    if STL_PATH.exists():
        return STL_PATH
    box = trimesh.creation.box(extents=[20.0, 20.0, 20.0])
    box.export(str(STL_PATH))
    print(f"Created {STL_PATH} (20mm cube)")
    return STL_PATH


def _run_gmsh_tet(stl_path: Path, element_size: float) -> tuple[str, str]:
    """
    Initialize GMSH, import STL, attempt 3D tet mesh.
    Returns (status, detail) where status is "SUCCESS" or "FAIL".
    """
    temp_stl = None
    gmsh_initialized = False

    try:
        # Export mesh to temp STL (in case we loaded trimesh)
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
        err_msg = str(e)
        # Extract PLC / segment errors if present
        detail = err_msg
        if "PLC" in err_msg or "segment" in err_msg.lower() or "intersect" in err_msg.lower():
            detail = err_msg
        return "FAIL", detail
    finally:
        if gmsh_initialized:
            gmsh.finalize()
        if temp_stl and os.path.exists(temp_stl):
            os.remove(temp_stl)


def main() -> None:
    stl_path = _ensure_cube_stl()
    if not stl_path.exists():
        print(f"Error: {STL_PATH} not found and could not be created.")
        return

    print("=" * 70)
    print("GMSH 3D Tetrahedral Mesh Sweep — 20mm Cube")
    print("=" * 70)
    print(f"STL: {stl_path.resolve()}")
    print(f"Element sizes (mm): {ELEMENT_SIZES}")
    print()

    results = []
    for size in ELEMENT_SIZES:
        status, detail = _run_gmsh_tet(stl_path, size)
        results.append((size, status, detail))

    # Table
    col_size = 10
    col_detail = 50
    print(f"{'Size (mm)':<{col_size}} {'Status':<10} {'Detail'}")
    print("-" * 70)
    for size, status, detail in results:
        detail_short = (detail[: col_detail - 3] + "...") if len(detail) > col_detail else detail
        detail_short = detail_short.replace("\n", " ")
        print(f"{size:<{col_size}} {status:<10} {detail_short}")

    print("=" * 70)

    # Print full error for any failure
    for size, status, detail in results:
        if status == "FAIL":
            print(f"\n--- FAIL at {size} mm (full error) ---")
            print(detail)


if __name__ == "__main__":
    main()
