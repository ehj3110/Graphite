#!/usr/bin/env python3
"""
scripts/generate_interlinked_review.py

Generates review deliverables for the 'Explicit - Interlinked' package:
  - outputs/interlinked_review/plane_5x5x1_european.stl: 5x5x1 European 4-in-1 flat plane.
  - outputs/interlinked_review/plane_5x5x1_kusari.stl: 5x5x1 Japanese Kusari flat plane.
  - outputs/interlinked_review/cube_2x2x2_interlinked.stl: 2x2x2 cube of connected rings.
  - outputs/interlinked_review/report.json: Summary statistics and clearance metrics.

Usage:
    python scripts/generate_interlinked_review.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Workspace setup
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from graphite.explicit.interlinked import (
    generate_interlinked_lattice,
    InterlinkedConfig,
)


def main() -> int:
    output_dir = WORKSPACE_ROOT / "outputs" / "interlinked_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("Graphite Explicit Interlinked — Review Deliverables Generator")
    print(f"Output Directory: {output_dir}")
    print("================================================================================")

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }

    # -------------------------------------------------------------------------
    # 1. 5x5x1 Flat Plane of European 4-in-1
    # -------------------------------------------------------------------------
    print("\n[1/3] Generating European 4-in-1 Plane (5x5x1)...")
    t0 = time.perf_counter()
    cfg_euro = InterlinkedConfig(
        pattern="european_4in1",
        grid_size=(5, 5, 1),
        pitch=10.0,
        radius_ratio=0.65,
        wire_radius=0.40,
        tilt_angle_deg=28.0,
        min_clearance=0.30,
        num_ring_segments=24,
        add_spheres=True,
    )
    res_euro = generate_interlinked_lattice(cfg_euro)
    stl_euro = output_dir / "plane_5x5x1_european.stl"
    res_euro.mesh.export(str(stl_euro))
    dt_euro = time.perf_counter() - t0

    print(f"  -> Saved: {stl_euro.name} ({dt_euro:.2f}s)")
    print(f"  -> Rings: {res_euro.num_rings}, Nodes: {res_euro.num_nodes}, Struts: {res_euro.num_struts}")
    print(f"  -> Min Clearance: {res_euro.min_clearance:.4f} mm (Valid: {res_euro.clearance_valid})")
    print(f"  -> Watertight: {res_euro.mesh.is_watertight}, Volume: {res_euro.volume:.2f} mm^3")

    report_data["models"]["plane_5x5x1_european"] = {
        "file": str(stl_euro.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "pattern": "european_4in1",
        "grid_size": [5, 5, 1],
        "pitch_mm": 10.0,
        "wire_radius_mm": 0.40,
        "outer_diameter_mm": 2.0 * (res_euro.rings[0].radius + 0.40),
        "num_rings": res_euro.num_rings,
        "num_nodes": res_euro.num_nodes,
        "num_struts": res_euro.num_struts,
        "min_clearance_mm": round(res_euro.min_clearance, 4),
        "clearance_valid": res_euro.clearance_valid,
        "is_watertight": bool(res_euro.mesh.is_watertight),
        "mesh_volume_mm3": round(res_euro.volume, 2),
        "num_faces": len(res_euro.mesh.faces),
        "num_vertices": len(res_euro.mesh.vertices),
        "bounds": res_euro.bounds.tolist(),
        "generation_time_sec": round(dt_euro, 3),
    }

    # -------------------------------------------------------------------------
    # 2. 5x5x1 Flat Plane of Japanese Kusari
    # -------------------------------------------------------------------------
    print("\n[2/3] Generating Japanese Kusari Plane (5x5x1)...")
    t0 = time.perf_counter()
    cfg_kusari = InterlinkedConfig(
        pattern="kusari",
        grid_size=(5, 5, 1),
        pitch=10.0,
        flat_radius=3.6,
        arch_radius=4.1,
        wire_radius=0.35,
        min_clearance=0.30,
        num_ring_segments=24,
        add_spheres=True,
    )
    res_kusari = generate_interlinked_lattice(cfg_kusari)
    stl_kusari = output_dir / "plane_5x5x1_kusari.stl"
    res_kusari.mesh.export(str(stl_kusari))
    dt_kusari = time.perf_counter() - t0

    print(f"  -> Saved: {stl_kusari.name} ({dt_kusari:.2f}s)")
    print(f"  -> Rings: {res_kusari.num_rings}, Nodes: {res_kusari.num_nodes}, Struts: {res_kusari.num_struts}")
    print(f"  -> Min Clearance: {res_kusari.min_clearance:.4f} mm (Valid: {res_kusari.clearance_valid})")
    print(f"  -> Watertight: {res_kusari.mesh.is_watertight}, Volume: {res_kusari.volume:.2f} mm^3")

    report_data["models"]["plane_5x5x1_kusari"] = {
        "file": str(stl_kusari.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "pattern": "kusari",
        "grid_size": [5, 5, 1],
        "pitch_mm": 10.0,
        "wire_radius_mm": 0.35,
        "flat_radius_mm": 3.6,
        "arch_radius_mm": 4.1,
        "num_rings": res_kusari.num_rings,
        "num_nodes": res_kusari.num_nodes,
        "num_struts": res_kusari.num_struts,
        "min_clearance_mm": round(res_kusari.min_clearance, 4),
        "clearance_valid": res_kusari.clearance_valid,
        "is_watertight": bool(res_kusari.mesh.is_watertight),
        "mesh_volume_mm3": round(res_kusari.volume, 2),
        "num_faces": len(res_kusari.mesh.faces),
        "num_vertices": len(res_kusari.mesh.vertices),
        "bounds": res_kusari.bounds.tolist(),
        "generation_time_sec": round(dt_kusari, 3),
    }

    # -------------------------------------------------------------------------
    # 3. 2x2x2 Cube of Interlinked Rings
    # -------------------------------------------------------------------------
    print("\n[3/3] Generating Cube 2x2x2 Interlinked...")
    t0 = time.perf_counter()
    cfg_cube = InterlinkedConfig(
        pattern="cubic_8ring",
        pitch=8.0,
        flat_radius=5.10,
        wire_radius=0.35,
        min_clearance=0.30,
        num_ring_segments=24,
        add_spheres=True,
    )
    res_cube = generate_interlinked_lattice(cfg_cube)
    stl_cube = output_dir / "cube_2x2x2_interlinked.stl"
    res_cube.mesh.export(str(stl_cube))
    dt_cube = time.perf_counter() - t0

    print(f"  -> Saved: {stl_cube.name} ({dt_cube:.2f}s)")
    print(f"  -> Rings: {res_cube.num_rings}, Nodes: {res_cube.num_nodes}, Struts: {res_cube.num_struts}")
    print(f"  -> Min Clearance: {res_cube.min_clearance:.4f} mm (Valid: {res_cube.clearance_valid})")
    print(f"  -> Watertight: {res_cube.mesh.is_watertight}, Volume: {res_cube.volume:.2f} mm^3")

    report_data["models"]["cube_2x2x2_interlinked"] = {
        "file": str(stl_cube.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
        "pattern": "cubic_8ring",
        "pitch_mm": 8.0,
        "ring_radius_mm": 5.10,
        "wire_radius_mm": 0.35,
        "num_rings": res_cube.num_rings,
        "num_nodes": res_cube.num_nodes,
        "num_struts": res_cube.num_struts,
        "min_clearance_mm": round(res_cube.min_clearance, 4),
        "clearance_valid": res_cube.clearance_valid,
        "is_watertight": bool(res_cube.mesh.is_watertight),
        "mesh_volume_mm3": round(res_cube.volume, 2),
        "num_faces": len(res_cube.mesh.faces),
        "num_vertices": len(res_cube.mesh.vertices),
        "bounds": res_cube.bounds.tolist(),
        "generation_time_sec": round(dt_cube, 3),
    }

    # -------------------------------------------------------------------------
    # Save Report JSON
    # -------------------------------------------------------------------------
    report_file = output_dir / "report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Summary report written to: {report_file}")
    print("================================================================================")
    print("Deliverables successfully generated under outputs/interlinked_review/")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
