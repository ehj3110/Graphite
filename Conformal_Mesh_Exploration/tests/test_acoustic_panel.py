"""
Acoustic Panel — Dynamic voxel size for exactly 2 elements in Z.

Loads the acoustic wall panel STL, computes target_size = z_thickness/2
to force 2 voxels through the panel thickness, runs the full conformal
pipeline with octet rule and KDTree lattice cupping.
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
from Universal_Lattice_Engine.core.hex_rules import apply_hex_octet_truss
from geometry_module import export_lattice_to_stl

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TEST_PARTS = _root / "test_parts"
STL_PATH = TEST_PARTS / "acoustic wall panel - single.stl"


def _key_from_coords(pt: np.ndarray) -> str:
    """Round to 4 decimals for deduplication key."""
    return f"{round(float(pt[0]), 4)},{round(float(pt[1]), 4)},{round(float(pt[2]), 4)}"


def _apply_topological_snapping(
    global_nodes: np.ndarray,
    exposed_face_centers: np.ndarray,
    stl_path: str | Path,
) -> np.ndarray:
    """Match lattice nodes to exposed face centers via cKDTree, snap to STL."""
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


def main() -> None:
    print("=" * 60)
    print("Conformal Acoustic Panel (Dynamic Voxel Size)")
    print("=" * 60)

    if not STL_PATH.exists():
        raise FileNotFoundError(f"STL not found: {STL_PATH}")

    mesh = trimesh.load(str(STL_PATH), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    z_thickness = float(mesh.extents[2])
    dynamic_target_size = z_thickness / 2.0

    print(f"  Panel Thickness:       {z_thickness:.2f} mm")
    print(f"  Dynamic Voxel Size:    {dynamic_target_size:.2f} mm")
    print(f"  Strut Thickness:       {dynamic_target_size * 0.1:.2f} mm (10% of voxel)")
    print("-" * 60)

    all_hexes, exposed_face_centers = generate_conformal_hexes(
        STL_PATH, target_size=dynamic_target_size
    )

    if all_hexes.size == 0:
        raise RuntimeError("No hexes produced by voxelizer.")

    global_node_map: dict[str, int] = {}
    global_coords: list[list[float]] = []
    global_struts: list[tuple[int, int]] = []

    for hex_coords in all_hexes:
        nodes, struts = apply_hex_octet_truss(hex_coords)

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

    global_nodes = _apply_topological_snapping(
        global_nodes, exposed_face_centers, STL_PATH
    )

    strut_thickness = dynamic_target_size * 0.1
    out_path = OUTPUT_DIR / "Conformal_Acoustic_Panel_Octet.stl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    export_lattice_to_stl(
        global_nodes,
        global_struts_arr,
        thickness=strut_thickness,
        output_filename=str(out_path),
    )
    print(f"Exported: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
