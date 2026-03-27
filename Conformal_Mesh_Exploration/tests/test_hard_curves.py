"""
Hard Curves Stress Test — Sphere and Torus conformal voxelization.

Runs the conformal voxelizer on highly curved topologies and produces
a verbose Mesh Quality Analysis (aspect ratio) for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh import proximity
from scipy.spatial import cKDTree

_root = Path(__file__).resolve().parents[2]  # Graphite project root
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from Conformal_Mesh_Exploration.core.voxelizer import generate_conformal_hexes
from Conformal_Mesh_Exploration.core.post_processing import discretize_boundary_struts, relax_hex_grid
from Universal_Lattice_Engine.core.hex_rules import apply_hex_grid
from graphite.explicit.geometry_module import export_lattice_to_stl

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEST_PARTS = _root / "test_parts"

_HEX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),   # top
    (0, 4), (1, 5), (2, 6), (3, 7),   # vertical
]


def analyze_mesh_quality(hexes: np.ndarray) -> dict:
    """
    Compute per-element aspect ratio (min_edge / max_edge) for (N, 8, 3) hexes.

    Returns dict with per-element ratios, average, and worst (minimum).
    """
    n = hexes.shape[0]
    ratios = np.empty(n, dtype=np.float64)

    for i in range(n):
        pts = hexes[i]
        lengths = np.array([
            np.linalg.norm(pts[b] - pts[a]) for a, b in _HEX_EDGES
        ])
        min_len = lengths.min()
        max_len = lengths.max()
        ratios[i] = min_len / max_len if max_len > 1e-12 else 0.0

    return {
        "ratios": ratios,
        "average": float(np.mean(ratios)) if n > 0 else 0.0,
        "worst": float(np.min(ratios)) if n > 0 else 0.0,
    }


def _key_from_coords(pt: np.ndarray) -> str:
    return f"{round(float(pt[0]), 4)},{round(float(pt[1]), 4)},{round(float(pt[2]), 4)}"


def _apply_topological_snapping(
    global_nodes: np.ndarray,
    exposed_face_centers: np.ndarray,
    stl_path: str | Path,
) -> np.ndarray:
    if exposed_face_centers.size == 0:
        return global_nodes.copy()

    tree = cKDTree(global_nodes)
    distances, indices = tree.query(exposed_face_centers, k=1)
    distances = np.atleast_1d(distances).ravel()
    indices = np.atleast_1d(indices).ravel()

    mask = distances < 1e-3
    boundary_indices = np.unique(indices[mask])

    if len(boundary_indices) == 0:
        return global_nodes.copy()

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    snapped_points, _, _ = proximity.closest_point(mesh, global_nodes[boundary_indices])
    result = global_nodes.copy()
    result[boundary_indices] = snapped_points
    return result


def process_hard_model(
    stl_path: str | Path,
    target_size: float,
    name: str,
) -> None:
    stl_path = Path(stl_path)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    print()
    print("=" * 60)
    print(f"  Meshing: {name} | Target Size: {target_size:.2f} mm")
    print("=" * 60)

    hexes, exposed_centers = generate_conformal_hexes(stl_path, target_size)

    if hexes.size == 0:
        print("  No hexes produced — skipping.")
        return

    # Masked Laplacian smoothing (relax interior near boundary)
    hexes = relax_hex_grid(hexes, max_depth=2, iterations=15)

    # Quality analysis
    quality = analyze_mesh_quality(hexes)
    ratios = quality["ratios"]
    n_total = len(ratios)

    internal_mask = ratios > 0.99
    boundary_mask = ~internal_mask
    n_internal = int(np.sum(internal_mask))
    n_boundary = int(np.sum(boundary_mask))

    boundary_ratios = ratios[boundary_mask]
    avg_boundary = float(np.mean(boundary_ratios)) if n_boundary > 0 else 1.0
    worst_ratio = quality["worst"]

    print("-" * 60)
    print("  MESH QUALITY REPORT")
    print("-" * 60)
    print(f"  Total Elements:              {n_total}")
    print(f"  Standard Internal (AR>0.99): {n_internal}  ({100.0 * n_internal / n_total:.1f}%)")
    print(f"  Conformal Boundary (AR<=0.99): {n_boundary}  ({100.0 * n_boundary / n_total:.1f}%)")
    print(f"  Average Boundary Skewness:   {avg_boundary:.4f}")
    print(f"  Worst Skewness (min AR):     {worst_ratio:.4f}")
    print("-" * 60)

    # Lattice rule + deduplication
    global_node_map: dict[str, int] = {}
    global_coords: list[list[float]] = []
    global_struts: list[tuple[int, int]] = []

    for hex_coords in hexes:
        nodes, struts = apply_hex_grid(hex_coords)

        local_to_global: dict[int, int] = {}
        for local_idx, pt in enumerate(nodes):
            key = _key_from_coords(pt)
            if key not in global_node_map:
                global_node_map[key] = len(global_coords)
                global_coords.append([float(pt[0]), float(pt[1]), float(pt[2])])
            local_to_global[local_idx] = global_node_map[key]

        for a, b in struts:
            ga = local_to_global[int(a)]
            gb = local_to_global[int(b)]
            if ga != gb:
                global_struts.append((ga, gb))

    global_nodes = np.array(global_coords, dtype=np.float64)
    global_struts_arr = np.array(global_struts, dtype=np.int64)

    # Option 2 KDTree cupping
    global_nodes = _apply_topological_snapping(
        global_nodes, exposed_centers, stl_path
    )

    # Strut discretization DISABLED for raw grid diagnostic
    # mesh = trimesh.load(str(stl_path), force="mesh")
    # if not isinstance(mesh, trimesh.Trimesh):
    #     mesh = mesh.dump().sum()
    # global_nodes, global_struts_arr = discretize_boundary_struts(
    #     global_nodes, global_struts_arr, mesh, target_size, segments=4
    # )

    strut_thickness = target_size * 0.1
    out_path = OUTPUT_DIR / f"Conformal_{name}_Grid.stl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_lattice_to_stl(
        global_nodes,
        global_struts_arr,
        thickness=strut_thickness,
        output_filename=str(out_path),
    )
    print(f"  Exported: {out_path}")


def main() -> None:
    # Sphere — halved resolution
    process_hard_model(
        TEST_PARTS / "Sphere.stl",
        target_size=2.5,
        name="Sphere",
    )

    # Torus — halved dynamic Z-size
    torus_path = TEST_PARTS / "Toros.stl"
    torus_mesh = trimesh.load(str(torus_path), force="mesh")
    if not isinstance(torus_mesh, trimesh.Trimesh):
        torus_mesh = torus_mesh.dump().sum()
    dynamic_size = float(torus_mesh.extents[2]) / 3.0 / 2.0
    print(f"\n  Torus Z-extent: {torus_mesh.extents[2]:.2f} mm -> dynamic size: {dynamic_size:.2f} mm")

    process_hard_model(
        torus_path,
        target_size=dynamic_size,
        name="Torus",
    )

    # Weird — halved to 6.0mm
    process_hard_model(
        TEST_PARTS / "Weird.stl",
        target_size=6.0,
        name="Weird",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
