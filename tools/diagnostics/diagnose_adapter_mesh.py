"""
Diagnose Part2_Adapter.STL — controlled mesh diagnostic with crash guard and logger.

Tests GMSH 3D tet generation across element sizes. Disables adaptive meshing.
Uses gmsh.logger to capture error details on failure. Saves successful scaffold as VTK.

Run from project root: python tools/diagnostics/diagnose_adapter_mesh.py
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
STL_PATH = _PROJECT_ROOT / "test_parts" / "Part2_Adapter.STL"
if not STL_PATH.exists():
    STL_PATH = _PROJECT_ROOT / "Part2_Adapter.STL"

ELEMENT_SIZES = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 12.5, 25.4]
VTK_OUTPUT = _PROJECT_ROOT / "Adapter_Scaffold_Check.vtk"
LOG_HEAD_TAIL_LINES = 5


def _run_gmsh_scaffold(
    stl_path: Path,
    element_size: float,
) -> tuple[str, str, list[str]]:
    """
    Initialize GMSH, import STL, attempt 3D tet mesh.
    Returns (status, detail, log_lines). Caller must gmsh.finalize() after each run.
    """
    temp_stl = None
    gmsh_initialized = False

    try:
        mesh_obj = trimesh.load(str(stl_path), force="mesh", process=False)
        if isinstance(mesh_obj, trimesh.Scene):
            mesh_obj = mesh_obj.dump(concatenate=True)
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            temp_stl = tmp.name
        mesh_obj.export(temp_stl)

        gmsh.initialize()
        gmsh_initialized = True
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.AbortOnError", 1)

        # Disable adaptive meshing
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        gmsh.option.setNumber("Mesh.MeshSizeMax", float(element_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(element_size))
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(element_size))
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT

        gmsh.merge(temp_stl)
        gmsh.model.mesh.classifySurfaces(math.pi / 4, True, True, math.pi)
        gmsh.model.mesh.createGeometry()

        surface_tags = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surface_tags:
            return "FAIL", "No surface entities after STL classification", []

        gmsh.model.geo.addSurfaceLoop(surface_tags, 1)
        gmsh.model.geo.addVolume([1], 1)
        gmsh.model.geo.synchronize()

        mesh_bounds = np.asarray(mesh_obj.bounds, dtype=np.float64)
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

        gmsh.logger.start()
        try:
            gmsh.model.mesh.generate(3)
        except Exception as e:
            gmsh.logger.stop()
            log_str = gmsh.logger.get()
            log_lines = [s.strip() for s in log_str.split("\n") if s.strip()] if log_str else []
            last_err = gmsh.logger.getLastError()
            detail = str(e)
            if last_err:
                detail = f"{detail} | Logger: {last_err}"
            return "FAIL", detail, log_lines
        gmsh.logger.stop()

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = node_tags.size if node_tags is not None else 0
        return "SUCCESS", f"{n_nodes} nodes", []

    except Exception as e:
        log_lines = []
        if gmsh_initialized:
            try:
                gmsh.logger.stop()
                log_str = gmsh.logger.get()
                log_lines = [s.strip() for s in log_str.split("\n") if s.strip()] if log_str else []
            except Exception:
                pass
        return "FAIL", str(e), log_lines
    finally:
        if temp_stl and os.path.exists(temp_stl):
            os.remove(temp_stl)


def main() -> None:
    if not STL_PATH.exists():
        print(f"Error: {STL_PATH} not found.")
        return

    mesh = trimesh.load(str(STL_PATH), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    bounds = mesh.bounds
    print("=" * 75)
    print("Part2_Adapter.STL — Mesh Diagnostic")
    print("=" * 75)
    print(f"STL: {STL_PATH.resolve()}")
    print(f"Bounding box (mm): X=[{bounds[0][0]:.2f}, {bounds[1][0]:.2f}] "
          f"Y=[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}] "
          f"Z=[{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
    print(f"Extents: {mesh.extents[0]:.2f} x {mesh.extents[1]:.2f} x {mesh.extents[2]:.2f} mm")
    print(f"Element sizes (mm): {ELEMENT_SIZES}")
    print("Adaptive meshing: DISABLED")
    print()

    results = []
    first_success_size = None

    for size in ELEMENT_SIZES:
        status, detail, log_lines = _run_gmsh_scaffold(STL_PATH, size)
        results.append((size, status, detail, log_lines))

        if status == "SUCCESS" and first_success_size is None:
            first_success_size = size
            gmsh.write(str(VTK_OUTPUT))
            print(f"  -> Saved scaffold: {VTK_OUTPUT}")
        try:
            gmsh.finalize()
        except Exception:
            pass

    # Table
    col_size = 12
    col_status = 10
    col_detail = 45
    print(f"{'Size (mm)':<{col_size}} {'Status':<{col_status}} {'Detail'}")
    print("-" * 75)
    for size, status, detail, _ in results:
        detail_short = (detail[: col_detail - 3] + "...") if len(detail) > col_detail else detail
        detail_short = detail_short.replace("\n", " ")
        print(f"{size:<{col_size}} {status:<{col_status}} {detail_short}")

    print("=" * 75)

    # Diagnostic: first/last log lines and PLC/self-intersect for failures
    for size, status, detail, log_lines in results:
        if status == "FAIL":
            print(f"\n--- FAIL at {size} mm ---")
            print(f"Exception: {detail}")
            if log_lines:
                head = log_lines[:LOG_HEAD_TAIL_LINES]
                tail = log_lines[-LOG_HEAD_TAIL_LINES:] if len(log_lines) > LOG_HEAD_TAIL_LINES else []
                print("GMSH log (first lines):")
                for line in head:
                    print(f"  {line}")
                if tail and tail != head:
                    print("GMSH log (last lines):")
                    for line in tail:
                        print(f"  {line}")
                plc = [l for l in log_lines if "PLC" in l or "self-intersect" in l.lower() or "segment" in l.lower()]
                if plc:
                    print("PLC / Self-intersect / Segment mentions:")
                    for line in plc[:10]:
                        print(f"  {line}")

    # Summary
    accepted = [r[0] for r in results if r[1] == "SUCCESS"]
    failed = [r[0] for r in results if r[1] == "FAIL"]
    print(f"\nAccepted: {accepted if accepted else 'none'}")
    print(f"Failed:   {failed if failed else 'none'}")

    if first_success_size is not None:
        print(f"\nSaved successful scaffold ({first_success_size} mm): {VTK_OUTPUT}")


if __name__ == "__main__":
    main()
