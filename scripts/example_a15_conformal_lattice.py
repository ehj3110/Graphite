#!/usr/bin/env python
"""
Example script showcasing conformal A15 Kagome lattice generation via direct node snapping.

Methodology:
  1. Base Scaffold: Generate a Cartesian hex grid using a loose bounding shell (centroid SDF <= CELL_SIZE * 0.866)
     to preserve all boundary-straddling supercells.
  2. Strict Sub-Cell Gating: Culls tetrahedra (4-cliques) in apply_hex_a15_kagome. Keeps tetrahedra
     only if all 4 vertices are inside the part (SDF <= 0) or the centroid is deeply inside (SDF <= -0.15 * CELL_SIZE).
  3. Direct Node Snapping: Identifies boundary Kagome nodes based on a combination of:
     - Topological criterion (endpoints of severed struts)
     - Geometric criterion (proximity to surface)
     Then snaps them directly to the zero-isosurface of the target mesh SDF, protected by a distance guard.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphite.explicit.hex_scaffold_module import _generate_bbox_hex_grid
from graphite.explicit.boundary_policy import (
    build_edt_sdf_sampler,
    closest_points_with_fallback,
)
from graphite.explicit.hex_topology_module import generate_hex_topology
from graphite.explicit.geometry_module import generate_geometry


SPHERE_RADIUS = 10.0
CELL_SIZE = 4.0


def make_sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(radius=SPHERE_RADIUS, subdivisions=3)


def snap_boundary_nodes(
    nodes_gated: np.ndarray,
    struts_gated: np.ndarray,
    nodes_raw: np.ndarray,
    struts_raw: np.ndarray,
    target_mesh: trimesh.Trimesh,
    sample_sdf,
) -> np.ndarray:
    """
    Identify and snap boundary nodes of a gated lattice to the target mesh surface.

    Parameters
    ----------
    nodes_gated : ndarray
        (V_gate, 3) node coordinates of the sub-cell gated lattice.
    struts_gated : ndarray
        (E_gate, 2) connectivity of the sub-cell gated lattice.
    nodes_raw : ndarray
        (V_raw, 3) node coordinates of the raw, unculled lattice.
    struts_raw : ndarray
        (E_raw, 2) connectivity of the raw, unculled lattice.
    target_mesh : trimesh.Trimesh
        The target boundary mesh to snap to.
    sample_sdf : callable
        Function mapping (N, 3) coordinates to signed distance values.

    Returns
    -------
    snapped_nodes : ndarray
        (V_gate, 3) node coordinates with boundary nodes projected onto target_mesh.
    """
    # 1. Map gated nodes back to their indices in the raw lattice using rounded coordinates
    rd = 5
    raw_map = {tuple(np.round(coord, rd)): idx for idx, coord in enumerate(nodes_raw)}
    
    gated_to_raw = []
    for coord in nodes_gated:
        key = tuple(np.round(coord, rd))
        if key in raw_map:
            gated_to_raw.append(raw_map[key])
        else:
            # Fallback if rounding limit is reached
            dists = np.linalg.norm(nodes_raw - coord, axis=1)
            gated_to_raw.append(int(np.argmin(dists)))
            
    gated_to_raw = np.array(gated_to_raw, dtype=np.int64)
    gated_raw_indices = set(gated_to_raw)
    
    # 2. Topological Criterion (Severed Struts):
    # Find struts in the raw grid where one endpoint survived culling and the other did not.
    # The surviving endpoint is on the boundary edge of the culling.
    severed_endpoints_raw = set()
    for u, v in struts_raw:
        u_in = u in gated_raw_indices
        v_in = v in gated_raw_indices
        if u_in and not v_in:
            severed_endpoints_raw.add(u)
        elif v_in and not u_in:
            severed_endpoints_raw.add(v)
            
    # Map severed endpoints back to gated lattice index space
    topo_boundary_set = set()
    for gated_idx, raw_idx in enumerate(gated_to_raw):
        if raw_idx in severed_endpoints_raw:
            topo_boundary_set.add(gated_idx)
            
    # 3. Geometric Criterion (Proximity Band):
    # Find nodes that are physically close to the surface boundary.
    vecs = nodes_gated[struts_gated[:, 1]] - nodes_gated[struts_gated[:, 0]]
    avg_strut_len = float(np.mean(np.linalg.norm(vecs, axis=-1)))
    
    sdfs = sample_sdf(nodes_gated)
    geo_band_limit = 0.5 * avg_strut_len
    geo_boundary_indices = np.where(np.abs(sdfs) <= geo_band_limit)[0]
    
    # Combine topological and geometric boundary nodes
    boundary_indices = list(topo_boundary_set.union(geo_boundary_indices))
    print(f"Boundary identification complete:")
    print(f"  - Topological nodes (severed struts): {len(topo_boundary_set)}")
    print(f"  - Geometric nodes (proximity band):   {len(geo_boundary_indices)}")
    print(f"  - Unique union of boundary nodes:    {len(boundary_indices)}")
    
    # 4. Project and snap boundary nodes
    snapped_nodes = nodes_gated.copy()
    if len(boundary_indices) == 0:
        return snapped_nodes
        
    boundary_pts = nodes_gated[boundary_indices]
    closest_pts, _ = closest_points_with_fallback(target_mesh, boundary_pts)
    
    # Enforce a distance guard to prevent extreme stretching/shear of outlier struts
    snap_distances = np.linalg.norm(closest_pts - boundary_pts, axis=1)
    max_snap_dist = 1.2 * avg_strut_len
    valid_snap_mask = snap_distances <= max_snap_dist
    
    snapped_count = 0
    for idx, is_valid, target_pt in zip(boundary_indices, valid_snap_mask, closest_pts):
        if is_valid:
            snapped_nodes[idx] = target_pt
            snapped_count += 1
            
    print(f"Snapping complete: Projected {snapped_count} / {len(boundary_indices)} boundary nodes onto surface.")
    return snapped_nodes


def main() -> None:
    # Set up target mesh and SDF sampler
    sphere = make_sphere()
    sample_sdf = build_edt_sdf_sampler(sphere, resolution=0.5)

    # Export reference sphere for visual check
    ref_path = ROOT / "Reference_Sphere.stl"
    sphere.export(str(ref_path))
    print(f"Exported reference sphere mesh: {ref_path}")

    # Step 1: Base Scaffold Generation using loose bounding shell
    print("\n[Step 1] Building base hex scaffold...")
    hex_all = _generate_bbox_hex_grid(sphere, CELL_SIZE, grid_anchor="bbox_center")
    centroids = np.mean(hex_all, axis=1)
    hex_b1 = hex_all[sample_sdf(centroids) <= (CELL_SIZE * 0.866)]
    print(f"Hex grid: {len(hex_all)} total cells -> {len(hex_b1)} kept within loose bounding shell.")

    # Step 2: Generate raw and sub-cell gated topologies
    print("\n[Step 2] Generating lattice topologies...")
    # Raw unculled topology (required for topological severed strut analysis)
    nodes_raw, struts_raw = generate_hex_topology(hex_b1, rule_name="a15_kagome")
    
    # Gated topology (using the new strict sub-cell culling)
    nodes_gated, struts_gated = generate_hex_topology(
        hex_b1, rule_name="a15_kagome", sdf_sampler=sample_sdf
    )
    print(f"Lattice topology sizes:")
    print(f"  - Raw:   {len(nodes_raw)} nodes, {len(struts_raw)} struts")
    print(f"  - Gated: {len(nodes_gated)} nodes, {len(struts_gated)} struts")

    # Step 3: Perform direct node snapping on gated topology boundary nodes
    print("\n[Step 3] Snapping boundary nodes...")
    nodes_snapped = snap_boundary_nodes(
        nodes_gated, struts_gated,
        nodes_raw, struts_raw,
        sphere, sample_sdf
    )

    # Ensure no node was added or removed during projection
    assert len(nodes_snapped) == len(nodes_gated), "Node count mismatch during snapping!"

    # Step 4: Export final conformed geometry
    print("\n[Step 4] Exporting conformed lattice STL...")
    vecs = nodes_snapped[struts_gated[:, 1]] - nodes_snapped[struts_gated[:, 0]]
    avg_strut_len = float(np.mean(np.linalg.norm(vecs, axis=-1)))
    strut_radius = 0.075 * avg_strut_len
    print(f"Final lattice metrics:")
    print(f"  - Avg strut length: {avg_strut_len:.4f} mm")
    print(f"  - Strut radius:     {strut_radius:.4f} mm")

    mesh_out = generate_geometry(
        nodes_snapped, struts_gated,
        strut_radius=strut_radius,
        boundary_mesh=sphere,
        crop_to_boundary=False,
    )
    if isinstance(mesh_out, tuple):
        mesh_out = mesh_out[0]

    out_path = ROOT / "A15_Conformal_NodeSnapped.stl"
    mesh_out.export(str(out_path))
    print(f"Successfully saved conformed lattice STL to: {out_path}")


if __name__ == "__main__":
    main()
