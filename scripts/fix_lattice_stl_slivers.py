"""
Light STL fix for lattice slabs: collapse collinear sliver triangles.

Detects near-zero-area faces (common on cap/boundary booleans), merges the
middle vertex onto an endpoint, then drops degenerate faces. Does not run
Poisson or aggressive remeshing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _sliver_faces(mesh: trimesh.Trimesh, area_eps: float) -> list[int]:
    areas = mesh.area_faces
    return [int(i) for i in np.where(areas < area_eps)[0]]


def _collapse_sliver_face(mesh: trimesh.Trimesh, face_idx: int) -> tuple[int, int]:
    """Return (from_vertex, to_vertex) merged for one sliver triangle."""
    f = mesh.faces[face_idx]
    pts = mesh.vertices[f]
    # Endpoints = farthest pair; remaining vertex is the collinear middle.
    pairs = ((0, 1), (1, 2), (0, 2))
    i0, i1 = max(pairs, key=lambda ij: float(np.linalg.norm(pts[ij[1]] - pts[ij[0]])))
    mid_local = ({0, 1, 2} - {i0, i1}).pop()
    a, b, mid = int(f[i0]), int(f[i1]), int(f[mid_local])
    d_a = float(np.linalg.norm(mesh.vertices[mid] - mesh.vertices[a]))
    d_b = float(np.linalg.norm(mesh.vertices[mid] - mesh.vertices[b]))
    to_v = a if d_a <= d_b else b
    return mid, to_v


def keep_largest_component(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int]:
    """Drop stray micro-shells; return (main shell, dropped_face_count)."""
    parts = sorted(
        mesh.split(only_watertight=False),
        key=lambda p: len(p.faces),
        reverse=True,
    )
    main = parts[0]
    dropped = int(len(mesh.faces) - len(main.faces))
    return main, dropped


def fix_lattice_stl_slivers(
    mesh: trimesh.Trimesh,
    *,
    area_eps: float = 1e-7,
) -> tuple[trimesh.Trimesh, list[dict]]:
    """
    Collapse sliver triangles and return (fixed_mesh, repair_log).

    Each log entry: face_index, vertices, area, centroid, merged_vertex pair.
    """
    m = mesh.copy()
    log: list[dict] = []

    for fi in _sliver_faces(m, area_eps):
        if fi >= len(m.faces):
            continue
        f = m.faces[fi].copy()
        mid, to_v = _collapse_sliver_face(m, fi)
        if mid == to_v:
            continue
        m.vertices[mid] = m.vertices[to_v]
        m.faces[m.faces == mid] = to_v
        log.append(
            {
                "face_index": int(fi),
                "vertices": [int(x) for x in f],
                "area_mm2": float(m.area_faces[fi]) if fi < len(m.area_faces) else 0.0,
                "centroid_mm": [float(x) for x in m.triangles_center[fi]],
                "merged_vertex": {"from": int(mid), "to": int(to_v)},
            }
        )

    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.update_faces(m.area_faces >= area_eps)
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    return m, log


def main() -> int:
    parser = argparse.ArgumentParser(description="Collapse lattice STL sliver triangles.")
    parser.add_argument("input_stl", type=Path)
    parser.add_argument("-o", "--output-stl", type=Path, default=None)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Write diagnostic / repair log JSON.",
    )
    parser.add_argument("--area-eps", type=float, default=1e-7)
    parser.add_argument(
        "--main-component-only",
        action="store_true",
        help="Keep only the largest connected shell (drops Blender debris).",
    )
    parser.add_argument(
        "--skip-sliver-fix",
        action="store_true",
        help="Only apply --main-component-only (skip sliver collapse; safer on dense remeshes).",
    )
    args = parser.parse_args()

    inp = args.input_stl.resolve()
    out = args.output_stl or inp.with_name(inp.stem + "_fixed.stl")
    report_path = args.report_json or out.with_suffix(".sliver_fix.json")

    mesh = trimesh.load_mesh(str(inp))
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    dropped_faces = 0
    if args.main_component_only:
        mesh, dropped_faces = keep_largest_component(mesh)

    before_slivers = _sliver_faces(mesh, args.area_eps)
    if args.skip_sliver_fix:
        fixed, log = mesh, []
    else:
        fixed, log = fix_lattice_stl_slivers(mesh, area_eps=args.area_eps)
    after_slivers = _sliver_faces(fixed, args.area_eps)

    fixed.export(str(out))

    report = {
        "input_stl": str(inp),
        "output_stl": str(out),
        "area_eps_mm2": args.area_eps,
        "main_component_only": bool(args.main_component_only),
        "dropped_debris_faces": dropped_faces,
        "sliver_faces_before": before_slivers,
        "sliver_faces_after": after_slivers,
        "repairs": log,
        "gmsh_note": (
            "Dense Blender remeshes (~500k+ faces): use single-surface + HXT "
            "(ARISTO_GMSH_ALGORITHM3D=10). TPMS classifySurfaces often fails."
        ),
        "faces_before": int(len(mesh.faces)),
        "faces_after": int(len(fixed.faces)),
        "watertight_after": bool(fixed.is_watertight),
        "volume_mm3_after": float(fixed.volume),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Input:  {inp} ({len(mesh.faces):,} faces, {len(before_slivers)} slivers)")
    print(f"Output: {out} ({len(fixed.faces):,} faces, watertight={fixed.is_watertight})")
    print(f"Report: {report_path}")
    for entry in log:
        c = entry["centroid_mm"]
        print(
            f"  face {entry['face_index']}: area={entry['area_mm2']:.2e} "
            f"centroid=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}) "
            f"merge v{entry['merged_vertex']['from']} -> v{entry['merged_vertex']['to']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
