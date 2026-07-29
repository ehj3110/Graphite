"""
Graphite Explicit Engine - Supercell Scaffold Generation

This module is responsible for generating explicitly tiled crystallographic 
lattices (BCC, FCC, Kagome, etc.) across a Cartesian bounding box, applying 
boundary classification logic, and snapping boundary nodes to the target STL 
surface using Signed Distance Field (SDF) evaluation.

It is heavily used for the 'Oct-Tet Kagome' lattice synthesis.
"""
import numpy as np
from scipy.spatial import cKDTree
from graphite.geometry.masking import voxelize_mesh_and_edt
from dataclasses import dataclass

from graphite.explicit.boundary_policy import (
    apply_tiered_boundary_policy,
    build_edt_sdf_sampler,
    closest_points_with_fallback,
    compress_graph_to_kept_struts,
)


import networkx as nx

from enum import Enum

class ClippingMode(Enum):
    """
    Boundary clipping behavior modes for explicit scaffolding.

    SNAP : Original mode, snaps vertices to the nearest surface.
    STRICT : Deletes any crossing struts, no partial cells.
    OVERFLOW : Keeps full cells that touch the boundary.
    """
    SNAP = "snap"         # Original: snap vertices to surface
    STRICT = "strict"     # NEW: delete any crossing struts (No partials)
    OVERFLOW = "overflow" # Keep full cells touching boundary


@dataclass(frozen=True)
class BoundaryStateAction:
    """Rule payload for a classified boundary-state cell."""
    state_key: str
    apply_conformal: bool
    add_face_diagonal: bool


BOUNDARY_STATE_RULES = {
    "TOP_SHALLOW_CUT": BoundaryStateAction(
        state_key="TOP_SHALLOW_CUT",
        apply_conformal=True,
        add_face_diagonal=False,
    ),
    "SINGLE_CORNER_CUT": BoundaryStateAction(
        state_key="SINGLE_CORNER_CUT",
        apply_conformal=True,
        add_face_diagonal=True,
    ),
    "FACE_HALF_CUT": BoundaryStateAction(
        state_key="FACE_HALF_CUT",
        apply_conformal=True,
        add_face_diagonal=False,
    ),
    "DIAGONAL_FACE_CUT": BoundaryStateAction(
        state_key="DIAGONAL_FACE_CUT",
        apply_conformal=True,
        add_face_diagonal=True,
    ),
}

def generate_cartesian_nodes(bbox_bounds, cell_size, cell_type="Simple Cubic", padding_blocks=1):
    """
    Generate an explicit node array bounded by the specified bounding box.
    
    This function computes the required tiling period (based on the requested 
    lattice type) and generates a regular grid of node seeds. Padding blocks 
    are added to guarantee stable mathematical topologies prior to SDF clipping.

    Parameters
    ----------
    bbox_bounds : tuple
        ((min_x, min_y, min_z), (max_x, max_y, max_z)) defining the domain.
    cell_size : float
        The requested element scale/size.
    cell_type : str, optional
        The type of lattice unit cell to generate, by default "Simple Cubic".
    padding_blocks : int, optional
        Number of extra unit cells to pad around the boundary, by default 1.

    Returns
    -------
    ndarray
        (N, 3) array of generated node coordinates.
    """
    # 0. Apply Internal Scaling (Primary Strut Method)
    # Maps requested element size (d) to tiling period (L)
    # We need to know if we are doing Kagome for the 3x factor
    is_kag = ("Kagome" in cell_type) or ("Kagome" in str(cell_type)) # flexible check
    effective_L = get_effective_tiling_period(cell_size, cell_type, is_kagome=is_kag)

    min_b, max_b = np.array(bbox_bounds[0], dtype=np.float64), np.array(bbox_bounds[1], dtype=np.float64)
    min_b -= effective_L * padding_blocks
    max_b += effective_L * padding_blocks
    
    nx_cells = max(1, int(np.ceil((max_b[0] - min_b[0]) / effective_L)))
    ny_cells = max(1, int(np.ceil((max_b[1] - min_b[1]) / effective_L)))
    nz_cells = max(1, int(np.ceil((max_b[2] - min_b[2]) / effective_L)))

    from graphite.explicit.proven_topologies import (
        generate_simple_cubic_seeds,
        generate_bcc_seeds,
        generate_fcc_seeds,
        generate_a15_seeds,
        generate_truncated_oct_tet_seeds
    )
    
    if cell_type == "Simple Cubic":
        nodes = generate_simple_cubic_seeds(nx_cells, ny_cells, nz_cells, effective_L)
    elif cell_type in ("BCC", "Body-Centered Cubic"):
        nodes = generate_bcc_seeds(nx_cells, ny_cells, nz_cells, effective_L)
    elif cell_type in ("FCC", "Face-Centered Cubic"):
        nodes = generate_fcc_seeds(nx_cells, ny_cells, nz_cells, effective_L)
    elif cell_type == "A15 Frank-Kasper":
        nodes = generate_a15_seeds(nx_cells, ny_cells, nz_cells, effective_L)
    elif cell_type == "Truncated Octa-Tetra Honeycomb":
        nodes = generate_truncated_oct_tet_seeds(nx_cells, ny_cells, nz_cells, effective_L)
    else:
        nodes = generate_simple_cubic_seeds(nx_cells, ny_cells, nz_cells, effective_L)
        
    return nodes + min_b

def get_effective_tiling_period(element_size, cell_type, is_kagome=False):
    """Calculates the grid tiling period L required to achieve target element size d."""
    # Base multipliers from crystallography (Strut = L / factor)
    factors = {
        "Simple Cubic": 1.0,
        "BCC": 1.1547,
        "Body-Centered Cubic": 1.1547,
        "FCC": 1.4142,
        "Face-Centered Cubic": 1.4142,
        "A15 Frank-Kasper": 2.0,
        "Truncated Octa-Tetra Honeycomb": 1.4142,
        "Oct-Tet": 1.4142,
    }
    
    # Map 'Oct-Tet Kagome' to 'Oct-Tet' factor
    base_type = cell_type
    for k in factors:
        if k in cell_type:
            base_type = k
            break
            
    L = element_size * factors.get(base_type, 1.4142)
    
    # Kagome correction: Struts are 1/3 of parent tetrahedron edges.
    if is_kagome:
        L *= 3.0
        
    return L


def calculate_supercell_radius(target_vf_local, element_size, cell_type, high_accuracy=False):
    """
    Calculates the strut radius (r) required to achieve a target Local Volume Fraction (VF_local).
    Local VF is defined as the solid fraction within the tetrahedral sub-cells.
    """
    is_kag = ("Kagome" in cell_type) or ("Kagome" in str(cell_type))
    L = get_effective_tiling_period(element_size, cell_type, is_kagome=is_kag)
    
    if is_kag:
        # Analytical Denominator: Relating Tet Struts to Tet Volume
        # (48 / sqrt(2)) = 33.941125...
        # Denom = 33.941 * pi
        denominator = 33.941125 * np.pi
        
        # Base analytical radius
        radius = np.sqrt((target_vf_local * L**2) / denominator)
        
        return radius
    else:
        # Fallback for standard lattices (Simple Cubic, BCC, FCC)
        # Ratio of Strut Length to Cell Volume (Analytical)
        # SC: 1 edge per cell? No, 3 edges shared by 4 = 3 edges total.
        # But we use the existing generator's behavior.
        return 0.5


def generate_bonds_by_distance(nodes, search_radius):
    """
    Connect all nodes within `search_radius` of each other to form a baseline web.
    """
    # use cKDTree for massive performance over pairwise distance
    tree = cKDTree(nodes)
    # cKDTree query_pairs returns undirected pairs
    pairs = tree.query_pairs(r=search_radius)
    struts = np.array(list(pairs), dtype=np.int32)
    return struts


def extract_fast_faces(struts, num_nodes):
    """
    Extract triangular faces from a graph of struts efficiently.

    Parameters
    ----------
    struts : array_like
        The array of edge pairs (struts) connecting nodes.
    num_nodes : int
        The total number of nodes in the graph.

    Returns
    -------
    list of tuple
        A list of sorted node index tuples representing triangular faces.
    """
    G = nx.Graph()
    G.add_edges_from(struts)
    triangles = []
    visited = set()
    for u in G.nodes():
        neighbors = set(G.neighbors(u))
        visited.add(u)
        for v in neighbors - visited:
            common = neighbors.intersection(G.neighbors(v)) - visited
            for w in common:
                triangles.append((u, v, w))
    return list(set([tuple(sorted(t)) for t in triangles]))

def _apply_greedy_kagome(nodes, element_size):
    """Placeholder for legacy shared-edge Kagome logic."""
    # For now, return empty as this is not the primary path
    return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)

def apply_face_centric_kagome(nodes, element_size, cell_type, target_bbox=None):
    """
    Transforms an explicit FCC base graph into a true 3D Kagome lattice.
    
    Tessellation Strategy:
      - We iterate over a padded range of cells (Target + 1 buffer) to ensure 
        no boundary 'ghost' effects occur during the bridge search.
      - Inter-tet bridges ($L/3$) only form if adjacent tet-nodes both exist.
      - We then filter back to the target volume to remove protrusions.
    """
    if "Oct-Tet" not in cell_type and "Octa-Tetra" not in cell_type and "FCC" not in cell_type:
        return _apply_greedy_kagome(nodes, element_size)

    L = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
    
    # 0. Base Math and Grid Size
    if target_bbox is not None:
        iter_min = np.array(target_bbox[0], dtype=np.float64)
        iter_max = np.array(target_bbox[1], dtype=np.float64)
        nx = int(np.round((iter_max[0] - iter_min[0]) / L))
        ny = int(np.round((iter_max[1] - iter_min[1]) / L))
        nz = int(np.round((iter_max[2] - iter_min[2]) / L))
        grid_min = iter_min
    else:
        # Fallback to empirical bounds of inputted nodes
        grid_min = np.min(nodes, axis=0)
        span = np.max(nodes, axis=0) - grid_min
        nx = max(1, int(np.round(span[0] / L)))
        ny = max(1, int(np.round(span[1] / L)))
        nz = max(1, int(np.round(span[2] / L)))

    all_k_nodes = []
    # Byte-hashing dictionary: raw byte string -> node index
    face_to_node = {} 
    
    active_tet_node_sets = [] # list of lists [n1,n2,n3,n4]
    active_tet_coords_sets = [] # list of lists of shape (4, 3)
    
    def get_k_node(origin, v_offsets):
        pts = origin + np.array(v_offsets) * L
        
        # Sort vertices geometrically (Z, then Y, then X) for stable hashing
        # across boundary-sharing cells
        pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
        
        # Round to 6 decimal places to prevent floating point hash divergence
        pts_rounded = np.round(pts_sorted, 6)
        key = pts_rounded.tobytes()
        
        if key not in face_to_node:
            idx = len(all_k_nodes)
            all_k_nodes.append(np.mean(pts, axis=0))
            face_to_node[key] = idx
            return idx
        return face_to_node[key]

    # 1. Generate Padded Field of Tets
    # We iterate -1 to Count to cover neighbor shells mathematically
    for i in range(-1, nx + 1):
        for j in range(-1, ny + 1):
            for k in range(-1, nz + 1):
                origin = grid_min + np.array([i, j, k]) * L
                
                is_active_cell = (0 <= i < nx) and (0 <= j < ny) and (0 <= k < nz)

                # Standard FCC tet layout
                f = {
                    "xy0": [0.5, 0.5, 0.0], "xy1": [0.5, 0.5, 1.0],
                    "xz0": [0.5, 0.0, 0.5], "xz1": [0.5, 1.0, 0.5],
                    "yz0": [0.0, 0.5, 0.5], "yz1": [1.0, 0.5, 0.5]
                }
                c = [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]
                tet_configs = [
                    [c[0], f["xy0"], f["xz0"], f["yz0"]], [c[1], f["xy0"], f["xz0"], f["yz1"]],
                    [c[2], f["xy0"], f["xz1"], f["yz1"]], [c[3], f["xy0"], f["xz1"], f["yz0"]],
                    [c[4], f["xy1"], f["xz0"], f["yz0"]], [c[5], f["xy1"], f["xz0"], f["yz1"]],
                    [c[6], f["xy1"], f["xz1"], f["yz1"]], [c[7], f["xy1"], f["xz1"], f["yz0"]],
                ]

                for t_verts in tet_configs:
                    fn = [
                        get_k_node(origin, [t_verts[1], t_verts[2], t_verts[3]]),
                        get_k_node(origin, [t_verts[0], t_verts[1], t_verts[2]]),
                        get_k_node(origin, [t_verts[0], t_verts[1], t_verts[3]]),
                        get_k_node(origin, [t_verts[0], t_verts[2], t_verts[3]])
                    ]
                    if is_active_cell:
                        active_tet_node_sets.append(fn)
                        # Store structural coordinates of the Tet for Phase 3 50% Rule bounding
                        active_tet_coords_sets.append([origin + np.array(v) * L for v in t_verts])

    # 3. Global Bridge Search on all nodes (Active + Padding)
    final_nodes_full = np.array(all_k_nodes)
    temp_struts = []
    
    # 3a. Intra-tet struts (Only for active tets)
    for fn in active_tet_node_sets:
        for a in range(4):
            for b in range(a + 1, 4):
                temp_struts.append((fn[a], fn[b]))

    # 3b. Inter-tet bridges (Only between active tet nodes)
    if len(final_nodes_full) > 0:
        kag_tree = cKDTree(final_nodes_full)
        # Radius tailored to L/3 connections
        bridge_radius = L / 3.0
        bridge_pairs = kag_tree.query_pairs(r=bridge_radius + 1e-3)
        
        # We only keep a bridge if BOTH endpoints are part of the 'Active' set 
        # to avoid dangling boundary artifacts.
        active_nodes_mask = np.zeros(len(final_nodes_full), dtype=bool)
        for fn in active_tet_node_sets:
            active_nodes_mask[fn] = True
            
        for s, e in bridge_pairs:
            if active_nodes_mask[s] and active_nodes_mask[e]:
                temp_struts.append((s, e))

    # 4. Final De-duplication and Compression
    if not temp_struts:
        return np.empty((0,3)), np.empty((0,2)), [], face_to_node
        
    s_arr = np.array(temp_struts, dtype=np.int32)
    # Deduplicate
    struts_set = set()
    for s, e in s_arr:
        struts_set.add(tuple(sorted((int(s), int(e)))))
    final_struts_raw = np.array(list(struts_set), dtype=np.int32)

    # 5. Compress to remove unused padding nodes
    used_indices = np.unique(final_struts_raw.ravel())
    final_nodes = final_nodes_full[used_indices]
    mapping = np.full(len(final_nodes_full), -1, dtype=np.int32)
    mapping[used_indices] = np.arange(len(used_indices))
    
    final_struts = mapping[final_struts_raw]
    tet_elements = [mapping[fn].tolist() for fn in active_tet_node_sets]
    active_tet_coords = np.array(active_tet_coords_sets)

    return final_nodes, final_struts, tet_elements, active_tet_coords

def cull_kagome_lattice(kag_nodes, kag_struts, tet_elements, tet_coords, stl_mesh):
    """
    Applies the '50% Rule' at the element (Tet) level to cull a Kagome lattice.
    A tetrahedron is kept if >= 50% of its volume (sampled via 5 points) is inside the mesh.
    
    Returns:
    - new_kag_nodes (M, 3)
    - new_kag_struts (K, 2)
    - kept_tet_elements (List of [n0, n1, n2, n3] relating to new_kag_nodes)
    - kept_tet_coords (N, 4, 3)
    """
    if len(tet_coords) == 0:
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [], np.empty((0, 4, 3))
    
    from graphite.geometry.masking import voxelize_mesh_and_edt
    import trimesh
    
    resolution = min(2.0, max(0.5, np.max(stl_mesh.extents) / 80.0))
    _, _, _, cad_sdf, padded_min, _, nx, ny, nz = voxelize_mesh_and_edt(stl_mesh, resolution)
    
    def get_inside_mask(pts):
        ind = np.round((pts - padded_min) / resolution).astype(int)
        ind[:, 0] = np.clip(ind[:, 0], 0, nx - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, ny - 1)
        ind[:, 2] = np.clip(ind[:, 2], 0, nz - 1)
        return cad_sdf[ind[:, 0], ind[:, 1], ind[:, 2]] <= 0.0

    # tet_coords is shape (N, 4, 3)
    v0 = tet_coords[:, 0, :]
    v1 = tet_coords[:, 1, :]
    v2 = tet_coords[:, 2, :]
    v3 = tet_coords[:, 3, :]
    cnt = (v0 + v1 + v2 + v3) / 4.0
    
    m0 = get_inside_mask(v0)
    m1 = get_inside_mask(v1)
    m2 = get_inside_mask(v2)
    m3 = get_inside_mask(v3)
    mc = get_inside_mask(cnt)
    
    # Vote: 3 or more of 5 points inside = Kept
    votes = m0.astype(int) + m1.astype(int) + m2.astype(int) + m3.astype(int) + mc.astype(int)
    is_kept = votes >= 3
    
    kept_tet_elements_old = [elem for i, elem in enumerate(tet_elements) if is_kept[i]]
    kept_tet_coords = tet_coords[is_kept]
    
    # Prune Kagome Struts
    # A strut is kept if both endpoints exist inside the kept tetrahedra.
    # Actually, a more liberal Kagome boundary is keeping the exact nodes of kept tets,
    # and keeping any bridge struts between kept tets.
    valid_node_mask = np.zeros(len(kag_nodes), dtype=bool)
    for elem in kept_tet_elements_old:
        valid_node_mask[elem] = True
        
    s_starts_valid = valid_node_mask[kag_struts[:, 0]]
    s_ends_valid = valid_node_mask[kag_struts[:, 1]]
    struts_kept = kag_struts[s_starts_valid & s_ends_valid]
    
    # Compress nodes array to remove floating orphaned nodes
    if len(struts_kept) > 0:
        used_nodes_idx = np.unique(struts_kept.ravel())
        new_kag_nodes = kag_nodes[used_nodes_idx]
        
        mapping = np.full(len(kag_nodes), -1, dtype=np.int32)
        mapping[used_nodes_idx] = np.arange(len(used_nodes_idx))
        
        new_kag_struts = mapping[struts_kept]
        
        # Remap elements array
        kept_tet_elements = []
        for elem in kept_tet_elements_old:
            mapped = mapping[elem]
            # Since an element is kept, all 4 of its nodes SHOULD be kept (from intra-tet struts)
            kept_tet_elements.append(mapped.tolist())
    else:
        new_kag_nodes = np.empty((0, 3))
        new_kag_struts = np.empty((0, 2), dtype=np.int32)
        kept_tet_elements = []
        
    return new_kag_nodes, new_kag_struts, kept_tet_elements, kept_tet_coords


def _build_sdf_sampler(stl_mesh):
    """
    Build a reusable SDF lookup closure against the boundary mesh.
    """
    resolution = min(2.0, max(0.5, np.max(stl_mesh.extents) / 80.0))
    _, _, _, cad_sdf, padded_min, _, nx, ny, nz = voxelize_mesh_and_edt(stl_mesh, resolution)

    def sample_sdf(pts):
        ind = np.round((pts - padded_min) / resolution).astype(int)
        ind[:, 0] = np.clip(ind[:, 0], 0, nx - 1)
        ind[:, 1] = np.clip(ind[:, 1], 0, ny - 1)
        ind[:, 2] = np.clip(ind[:, 2], 0, nz - 1)
        return cad_sdf[ind[:, 0], ind[:, 1], ind[:, 2]]

    return sample_sdf


def _classify_boundary_state_cells(tet_coords, is_kept, element_size, cell_type):
    """
    Classify parent supercell boundary states from tet keep/deletion patterns.
    """
    if len(tet_coords) == 0:
        return {}, {}

    L = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
    tet_centroids = np.mean(tet_coords, axis=1)
    origin = np.min(tet_coords.reshape(-1, 3), axis=0)
    cell_idx = np.floor((tet_centroids - origin) / max(L, 1e-9) + 1e-6).astype(np.int32)

    cell_to_tets = {}
    for i, idx3 in enumerate(cell_idx):
        key = tuple(int(v) for v in idx3.tolist())
        cell_to_tets.setdefault(key, []).append(i)

    cell_states = {}
    for key, tet_ids in cell_to_tets.items():
        keep = np.array([bool(is_kept[i]) for i in tet_ids], dtype=bool)
        kept_count = int(np.sum(keep))
        total_count = int(len(tet_ids))

        if kept_count == 0:
            state_key = "FULL_OUTSIDE"
        elif kept_count == total_count:
            state_key = "FULL_INSIDE"
        else:
            removed = ~keep
            local = (tet_centroids[tet_ids] - (origin + np.array(key, dtype=np.float64) * L)) / max(L, 1e-9)
            removed_local = local[removed]
            removed_count = int(np.sum(removed))

            if removed_count == 1 and kept_count >= 4:
                state_key = "SINGLE_CORNER_CUT"
            elif removed_count > 0 and np.all(removed_local[:, 2] > 0.55):
                # "Shallow" top loss means the cut did not penetrate deeply through a full parent period.
                depth_from_top = 1.0 - float(np.min(removed_local[:, 2]))
                state_key = "TOP_SHALLOW_CUT" if depth_from_top < 0.5 else "AMBIGUOUS"
            elif removed_count >= 2:
                face_like = False
                for ax in range(3):
                    high_side = np.sum(removed_local[:, ax] > 0.6)
                    low_side = np.sum(removed_local[:, ax] < 0.4)
                    if max(high_side, low_side) >= max(2, int(np.ceil(0.6 * removed_count))):
                        face_like = True
                        break

                x, y, z = removed_local[:, 0], removed_local[:, 1], removed_local[:, 2]
                diag_xy = np.sum((x + y) > 1.15) >= 2
                diag_xz = np.sum((x + z) > 1.15) >= 2
                diag_yz = np.sum((y + z) > 1.15) >= 2
                diagonal_like = bool(diag_xy or diag_xz or diag_yz)

                if diagonal_like and not face_like:
                    state_key = "DIAGONAL_FACE_CUT"
                elif face_like:
                    state_key = "FACE_HALF_CUT"
                else:
                    state_key = "AMBIGUOUS"
            else:
                state_key = "AMBIGUOUS"

        cell_states[key] = state_key

    return cell_to_tets, cell_states


def cull_kagome_lattice_with_boundary_states(
    kag_nodes,
    kag_struts,
    tet_elements,
    tet_coords,
    stl_mesh,
    element_size,
    cell_type,
    enabled_states=("TOP_SHALLOW_CUT", "SINGLE_CORNER_CUT", "FACE_HALF_CUT", "DIAGONAL_FACE_CUT"),
):
    """
    Experimental boundary-state-aware culling/conforming for tet-oct Kagome.

    Phase behavior:
    - Tet-level cull (50% rule) as base filter.
    - Classify parent cell boundary states from tet occupancy.
    - Apply conformal node projection for selected boundary states.
    - Add one local face-diagonal reinforcement in SINGLE_CORNER_CUT states.

    Returns:
    - new_kag_nodes, new_kag_struts, kept_tet_elements, kept_tet_coords, report
    """
    if len(tet_coords) == 0:
        report = {
            "state_counts": {},
            "states_applied": {},
            "unknown_states": 0,
            "reinforcement_edges_added": 0,
        }
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [], np.empty((0, 4, 3)), report

    import trimesh

    sample_sdf = _build_sdf_sampler(stl_mesh)

    # Tet vote mask (same criterion as cull_kagome_lattice)
    v0 = tet_coords[:, 0, :]
    v1 = tet_coords[:, 1, :]
    v2 = tet_coords[:, 2, :]
    v3 = tet_coords[:, 3, :]
    cnt = (v0 + v1 + v2 + v3) / 4.0
    m0 = sample_sdf(v0) <= 0.0
    m1 = sample_sdf(v1) <= 0.0
    m2 = sample_sdf(v2) <= 0.0
    m3 = sample_sdf(v3) <= 0.0
    mc = sample_sdf(cnt) <= 0.0
    votes = m0.astype(int) + m1.astype(int) + m2.astype(int) + m3.astype(int) + mc.astype(int)
    is_kept = votes >= 3

    cell_to_tets, cell_states = _classify_boundary_state_cells(
        tet_coords=tet_coords,
        is_kept=is_kept,
        element_size=element_size,
        cell_type=cell_type,
    )

    kept_tet_elements_old = [elem for i, elem in enumerate(tet_elements) if is_kept[i]]
    kept_tet_coords = tet_coords[is_kept]

    valid_node_mask = np.zeros(len(kag_nodes), dtype=bool)
    for elem in kept_tet_elements_old:
        valid_node_mask[elem] = True

    s_starts_valid = valid_node_mask[kag_struts[:, 0]]
    s_ends_valid = valid_node_mask[kag_struts[:, 1]]
    struts_kept = kag_struts[s_starts_valid & s_ends_valid]

    if len(struts_kept) == 0:
        report = {
            "state_counts": {},
            "states_applied": {},
            "unknown_states": 0,
            "reinforcement_edges_added": 0,
        }
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [], np.empty((0, 4, 3)), report

    used_nodes_idx = np.unique(struts_kept.ravel())
    new_kag_nodes = kag_nodes[used_nodes_idx].copy()
    old_to_new = np.full(len(kag_nodes), -1, dtype=np.int32)
    old_to_new[used_nodes_idx] = np.arange(len(used_nodes_idx))
    new_kag_struts = old_to_new[struts_kept]

    kept_tet_elements = []
    for elem in kept_tet_elements_old:
        mapped = old_to_new[elem]
        kept_tet_elements.append(mapped.tolist())

    # Boundary-state conform + reinforcement
    L = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
    state_counts = {}
    states_applied = {}
    unknown_states = 0
    reinforcement_edges_added = 0

    try:
        closest_pts, _, _ = trimesh.proximity.closest_point(stl_mesh, new_kag_nodes)
    except Exception:
        tree = cKDTree(stl_mesh.vertices)
        _, vidx = tree.query(new_kag_nodes)
        closest_pts = stl_mesh.vertices[vidx]

    node_sdf = sample_sdf(new_kag_nodes)
    edge_set = {tuple(sorted((int(a), int(b)))) for a, b in np.asarray(new_kag_struts, dtype=np.int32)}

    for cell_key, tet_ids in cell_to_tets.items():
        state_key = cell_states.get(cell_key, "AMBIGUOUS")
        state_counts[state_key] = state_counts.get(state_key, 0) + 1

        if state_key not in enabled_states:
            if state_key not in BOUNDARY_STATE_RULES and state_key not in ("FULL_INSIDE", "FULL_OUTSIDE", "AMBIGUOUS"):
                unknown_states += 1
            continue

        rule = BOUNDARY_STATE_RULES.get(state_key)
        if rule is None:
            unknown_states += 1
            continue

        cell_new_nodes = []
        for tet_id in tet_ids:
            if not is_kept[tet_id]:
                continue
            old_nodes = np.asarray(tet_elements[tet_id], dtype=np.int32)
            mapped = old_to_new[old_nodes]
            mapped = mapped[mapped >= 0]
            if len(mapped) > 0:
                cell_new_nodes.extend(mapped.tolist())
        if not cell_new_nodes:
            continue

        unique_cell_nodes = np.unique(np.asarray(cell_new_nodes, dtype=np.int32))
        near_surface = unique_cell_nodes[np.abs(node_sdf[unique_cell_nodes]) <= (0.35 * L)]
        if len(near_surface) == 0:
            near_surface = unique_cell_nodes[node_sdf[unique_cell_nodes] > 0.0]

        if rule.apply_conformal and len(near_surface) > 0:
            new_kag_nodes[near_surface] = closest_pts[near_surface]
            states_applied[state_key] = states_applied.get(state_key, 0) + 1

        if rule.add_face_diagonal and len(unique_cell_nodes) >= 4:
            coords = new_kag_nodes[unique_cell_nodes]
            pair_i, pair_j = np.triu_indices(len(unique_cell_nodes), k=1)
            best_pair = None
            best_dist = None
            for i, j in zip(pair_i, pair_j):
                ni = int(unique_cell_nodes[i])
                nj = int(unique_cell_nodes[j])
                ekey = (ni, nj) if ni < nj else (nj, ni)
                if ekey in edge_set:
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < 0.45 * L or d > 0.9 * L:
                    continue
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_pair = ekey
            if best_pair is not None:
                edge_set.add(best_pair)
                reinforcement_edges_added += 1

    if reinforcement_edges_added > 0:
        new_kag_struts = np.array(sorted(edge_set), dtype=np.int32)
    else:
        new_kag_struts = np.asarray(new_kag_struts, dtype=np.int32)

    report = {
        "state_counts": state_counts,
        "states_applied": states_applied,
        "unknown_states": int(unknown_states),
        "reinforcement_edges_added": int(reinforcement_edges_added),
    }
    return new_kag_nodes, new_kag_struts, kept_tet_elements, kept_tet_coords, report


def _build_graph_from_kept_tets(kag_nodes, kag_struts, tet_elements, is_kept):
    """
    Build a compressed Kagome graph from a boolean kept-mask over tetra elements.
    """
    kept_tet_elements_old = [elem for i, elem in enumerate(tet_elements) if is_kept[i]]
    valid_node_mask = np.zeros(len(kag_nodes), dtype=bool)
    for elem in kept_tet_elements_old:
        valid_node_mask[np.asarray(elem, dtype=np.int32)] = True

    s_starts_valid = valid_node_mask[kag_struts[:, 0]]
    s_ends_valid = valid_node_mask[kag_struts[:, 1]]
    struts_kept = kag_struts[s_starts_valid & s_ends_valid]
    if len(struts_kept) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            [],
            np.full(len(kag_nodes), -1, dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    used_nodes_idx = np.unique(struts_kept.ravel())
    new_nodes = kag_nodes[used_nodes_idx].copy()
    old_to_new = np.full(len(kag_nodes), -1, dtype=np.int32)
    old_to_new[used_nodes_idx] = np.arange(len(used_nodes_idx))
    new_struts = old_to_new[struts_kept]

    remapped_kept_tets = []
    for elem in kept_tet_elements_old:
        mapped = old_to_new[np.asarray(elem, dtype=np.int32)]
        mapped = mapped[mapped >= 0]
        if len(mapped) == 4:
            remapped_kept_tets.append(mapped.tolist())
    return new_nodes, new_struts, remapped_kept_tets, old_to_new, used_nodes_idx


def cull_kagome_lattice_variants(
    kag_nodes,
    kag_struts,
    tet_elements,
    tet_coords,
    stl_mesh,
    element_size,
    cell_type,
    variant="internal_only_stretch_out",
):
    """
    Additional experimental culling variants for boundary behavior studies.

    Variants:
    - internal_only_stretch_out:
        Keep only fully internal tetra elements (all 5 sample points inside), then
        push boundary graph nodes out to the nearest surface.
    - singleton_surface_recovery:
        Base keep rule (>=3/5 inside). If a parent cell has only one kept tet,
        recover fully exterior tets in that cell and conform them to surface.
    - hybrid_final_candidate:
        Base keep rule (>=3/5), plus sparse-region recovery (cells with 1-2 kept tets
        recover near-boundary tets), then apply a broader boundary conform band.
    """
    if len(tet_coords) == 0:
        report = {"variant": variant, "warning": "No tetra elements provided."}
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32), [], np.empty((0, 4, 3)), report

    import trimesh

    sample_sdf = _build_sdf_sampler(stl_mesh)

    v0 = tet_coords[:, 0, :]
    v1 = tet_coords[:, 1, :]
    v2 = tet_coords[:, 2, :]
    v3 = tet_coords[:, 3, :]
    cnt = (v0 + v1 + v2 + v3) / 4.0
    m0 = sample_sdf(v0) <= 0.0
    m1 = sample_sdf(v1) <= 0.0
    m2 = sample_sdf(v2) <= 0.0
    m3 = sample_sdf(v3) <= 0.0
    mc = sample_sdf(cnt) <= 0.0
    votes = m0.astype(int) + m1.astype(int) + m2.astype(int) + m3.astype(int) + mc.astype(int)

    oct_expand_report = {}
    extra_struts_old: list = []
    oct_tip_detect: dict = {}

    if variant == "internal_only_stretch_out":
        is_kept = votes == 5
        recovered_tets = np.zeros_like(is_kept, dtype=bool)
    elif variant == "singleton_surface_recovery":
        is_kept = votes >= 3
        recovered_tets = np.zeros_like(is_kept, dtype=bool)
        L = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
        tet_centroids = np.mean(tet_coords, axis=1)
        origin = np.min(tet_coords.reshape(-1, 3), axis=0)
        cell_idx = np.floor((tet_centroids - origin) / max(L, 1e-9) + 1e-6).astype(np.int32)
        cell_to_tets = {}
        for i, idx3 in enumerate(cell_idx):
            key = tuple(int(v) for v in idx3.tolist())
            cell_to_tets.setdefault(key, []).append(i)
        for _, tet_ids in cell_to_tets.items():
            keep_mask = np.array([is_kept[i] for i in tet_ids], dtype=bool)
            if int(np.sum(keep_mask)) == 1:
                for tid in tet_ids:
                    if votes[tid] == 0:
                        is_kept[tid] = True
                        recovered_tets[tid] = True
    elif variant == "hybrid_final_candidate":
        is_kept = votes >= 3
        recovered_tets = np.zeros_like(is_kept, dtype=bool)
        L = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
        tet_centroids = np.mean(tet_coords, axis=1)
        origin = np.min(tet_coords.reshape(-1, 3), axis=0)
        cell_idx = np.floor((tet_centroids - origin) / max(L, 1e-9) + 1e-6).astype(np.int32)
        cell_to_tets = {}
        for i, idx3 in enumerate(cell_idx):
            key = tuple(int(v) for v in idx3.tolist())
            cell_to_tets.setdefault(key, []).append(i)
        for _, tet_ids in cell_to_tets.items():
            kept_count = int(np.sum([is_kept[i] for i in tet_ids]))
            # Sparse tip/edge neighborhoods: preserve local connector fabric
            # by recovering near-boundary tets even when mostly outside.
            if kept_count in (1, 2):
                for tid in tet_ids:
                    if not is_kept[tid] and votes[tid] >= 1:
                        is_kept[tid] = True
                        recovered_tets[tid] = True

        from graphite.explicit.supercell_oct_tip import (
            bridge_endpoint_mask_old,
            detect_oct_tip_regions,
            drop_fully_outside_tets_in_tip_cells,
            expand_is_kept_oct_tip,
            map_old_node_mask_to_compressed,
            merge_struts_with_protected_edges,
            tip_region_nodes_old,
        )

        sparse_cells, oct_only_cells, oct_tip_detect = detect_oct_tip_regions(
            tet_coords, votes, is_kept, sample_sdf, L
        )
        is_kept, oct_expand_report = expand_is_kept_oct_tip(
            is_kept,
            votes,
            tet_coords,
            tet_elements,
            kag_struts,
            sample_sdf,
            L,
            sparse_cells,
            oct_only_cells,
        )
        extra_struts_old = oct_expand_report.pop("extra_struts")

        tip_union = sparse_cells | oct_only_cells
        is_kept, n_drop_tip_outside = drop_fully_outside_tets_in_tip_cells(
            is_kept, votes, cell_idx, tip_union
        )
        oct_expand_report["drop_fully_outside_tets_in_tip"] = int(n_drop_tip_outside)

        bridge_ep_old = bridge_endpoint_mask_old(kag_nodes, kag_struts, L)
        tip_node_old = tip_region_nodes_old(
            len(kag_nodes), tet_elements, cell_idx, tip_union
        )
    else:
        raise ValueError(f"Unknown culling variant: {variant}")

    new_nodes, new_struts, kept_tet_elements, old_to_new, _ = _build_graph_from_kept_tets(
        kag_nodes, kag_struts, tet_elements, is_kept
    )

    if variant == "hybrid_final_candidate" and extra_struts_old:
        new_struts = merge_struts_with_protected_edges(
            new_struts, old_to_new, extra_struts_old
        )
    kept_tet_coords = tet_coords[is_kept]
    if len(new_struts) == 0:
        report = {
            "variant": variant,
            "kept_tets": int(np.sum(is_kept)),
            "recovered_exterior_tets": int(np.sum(recovered_tets)),
            "boundary_nodes_conformed": 0,
            "warning": "No struts after filtering",
        }
        return new_nodes, new_struts, kept_tet_elements, kept_tet_coords, report

    closest_pts, _ = closest_points_with_fallback(stl_mesh, new_nodes)

    node_sdf = sample_sdf(new_nodes)
    valences = np.zeros(len(new_nodes), dtype=np.int32)
    for s, e in new_struts:
        valences[s] += 1
        valences[e] += 1
    is_boundary_node = valences < 6

    hybrid_conform_detail: dict = {}

    if variant == "internal_only_stretch_out":
        # Stretch boundary nodes outward from retained interior graph to surface.
        conform_mask = is_boundary_node
    elif variant == "hybrid_final_candidate":
        # Stricter than broad in-band snapping: avoid pulling most interior boundary
        # nodes fully to closest-point (which collapses the lattice inward). Instead:
        # - Full snap only when clearly outside the part SDF.
        # - Barely-outside: partial blend toward surface.
        # - Inside but near boundary: gentle stretch-out blend (not full snap).
        # In oct-tip parent cells: do not flatten intra-tet Kagome nodes onto the skin;
        # only deform nodes that participate in oct-void bridge struts (~L/3), so voids
        # are not collapsed inward.
        tip_node_new = map_old_node_mask_to_compressed(
            tip_node_old, old_to_new, len(new_nodes)
        )
        bridge_ep_new = map_old_node_mask_to_compressed(
            bridge_ep_old, old_to_new, len(new_nodes)
        )
        conform_eligible = (is_boundary_node & ~tip_node_new) | (
            bridge_ep_new & tip_node_new
        )

        Lk = get_effective_tiling_period(element_size, cell_type, is_kagome=True)
        eps_pull = max(1e-5 * Lk, 1e-4)
        band_inner = 0.06 * Lk

        new_nodes, tier_counts = apply_tiered_boundary_policy(
            new_nodes,
            closest_pts,
            node_sdf,
            conform_eligible,
            eps_pull=eps_pull,
            band_inner=band_inner,
            beta_soft=0.45,
            alpha_stretch=0.28,
        )
        n_pull = int(tier_counts["conform_pull_in_full"])
        n_soft = int(tier_counts["conform_pull_in_soft"])
        n_stretch = int(tier_counts["conform_stretch_out"])
        conformed_count = n_pull + n_soft + n_stretch
        hybrid_conform_detail = {
            "conform_pull_in_full": n_pull,
            "conform_pull_in_soft": n_soft,
            "conform_stretch_out": n_stretch,
            "eps_pull_mm": float(eps_pull),
            "band_inner_mm": float(band_inner),
            "tip_nodes_compressed": int(np.sum(tip_node_new)),
            "tip_bridge_endpoints_compressed": int(np.sum(bridge_ep_new & tip_node_new)),
        }
        conform_mask = None
    else:
        # Conform nodes from recovered exterior tets + any boundary nodes outside.
        recovered_nodes_old = set()
        if np.any(recovered_tets):
            for i, elem in enumerate(tet_elements):
                if recovered_tets[i]:
                    recovered_nodes_old.update(int(v) for v in elem)
        recovered_nodes_new = []
        for old_idx in recovered_nodes_old:
            mapped = old_to_new[old_idx]
            if mapped >= 0:
                recovered_nodes_new.append(int(mapped))
        conform_mask = np.zeros(len(new_nodes), dtype=bool)
        if recovered_nodes_new:
            conform_mask[np.asarray(recovered_nodes_new, dtype=np.int32)] = True
        conform_mask |= (is_boundary_node & (node_sdf > 0.0))

    if variant != "hybrid_final_candidate":
        conformed_count = int(np.sum(conform_mask))
        if conformed_count > 0:
            new_nodes[conform_mask] = closest_pts[conform_mask]

    report = {
        "variant": variant,
        "kept_tets": int(np.sum(is_kept)),
        "recovered_exterior_tets": int(np.sum(recovered_tets)),
        "boundary_nodes_conformed": conformed_count,
    }
    if variant == "hybrid_final_candidate":
        report["oct_tip_detection"] = oct_tip_detect
        report["oct_tip_expand"] = {
            k: v for k, v in oct_expand_report.items() if k != "extra_struts"
        }
        report["hybrid_conform"] = hybrid_conform_detail
    return new_nodes, new_struts, kept_tet_elements, kept_tet_coords, report

def extract_boundary_faces(elements):
    """
    Deprecated. Used for older generic tet-mesh extraction.
    """
    return np.empty((0, 3), dtype=np.int32)

def generate_boundary_skin(kag_nodes, kag_struts):
    """
    Generates a Surface Dual / Cage (skin) from a culled Kagome lattice.
    Since Kagome nodes are face centroids, the surface skin is simply the subset 
    of Kagome struts that connect two boundary nodes.
    
    A boundary node is identified topologically as any node with fewer than 6 
    strut connections (missing internal Kagome bridges due to neighbor culling).
    """
    if len(kag_nodes) == 0 or len(kag_struts) == 0:
        return np.empty((0, 2), dtype=np.int32)
        
    # Calculate degree/valence of every Kagome node
    valences = np.zeros(len(kag_nodes), dtype=int)
    for s, e in kag_struts:
        valences[s] += 1
        valences[e] += 1
        
    # Boundary nodes have missing bridge struts (total < 6)
    # Geometrically perfect internal 3D Kagome nodes always have valence 6.
    is_boundary = valences < 6
    
    # The skin consists of all struts where both endpoints lie on the boundary layer
    s_boundary = is_boundary[kag_struts[:, 0]]
    e_boundary = is_boundary[kag_struts[:, 1]]
    skin_struts = kag_struts[skin_mask]
    return skin_struts

def apply_surface_snapping(kag_nodes, kag_struts, stl_mesh, snap_distance=0.0, blend_distance=0.0):
    """
    Deforms the boundary nodes of a topological Kagome lattice so they rest
    exactly on the continuous surface defined by `stl_mesh`.
    
    If snap_distance > 0.0, any node within this Euclidean distance to the surface
    is aggressively flattened onto the boundary, regardless of valence.
    
    If blend_distance > 0.0, any node within (snap_distance + blend_distance)
    is pulled proportionally towards the surface to provide internal strain relief.
    """
    import trimesh
    new_nodes = kag_nodes.copy()
    
    if len(kag_nodes) == 0 or len(kag_struts) == 0:
        return new_nodes
        
    try:
        closest_pts, distances, _ = trimesh.proximity.closest_point(stl_mesh, new_nodes)
    except Exception:
        from scipy.spatial import cKDTree
        mesh_tree = cKDTree(stl_mesh.vertices)
        distances, vertex_indices = mesh_tree.query(new_nodes)
        closest_pts = stl_mesh.vertices[vertex_indices]
        
    valences = np.zeros(len(kag_nodes), dtype=int)
    for s, e in kag_struts:
        valences[s] += 1
        valences[e] += 1
        
    is_topo_boundary = valences < 6
    
    for i in range(len(new_nodes)):
        d = distances[i]
        
        # 1. Topological broken struts ALWAYS snap
        if is_topo_boundary[i]:
            new_nodes[i] = closest_pts[i]
            continue
            
        # 2. Distance-based aggressive flattening (Face morphing)
        if snap_distance > 0.0 and d <= snap_distance:
            new_nodes[i] = closest_pts[i]
            continue
            
        # 3. Non-linear blend stretch relief
        if blend_distance > 0.0 and d <= (snap_distance + blend_distance):
            factor = 1.0 - ((d - snap_distance) / blend_distance)
            ease = 3.0 * (factor ** 2) - 2.0 * (factor ** 3)  # Smoothstep
            new_nodes[i] = new_nodes[i] * (1.0 - ease) + closest_pts[i] * ease
            
    return new_nodes

def apply_supercell_stretching(kag_nodes, kept_elements, kept_tet_coords, stl_mesh):
    """
    Deforms the parent super-cells (FCC mathematical corners) onto the boundary
    and re-averages the Kagome Wigner-Seitz nodes. This structurally prevents 
    element collapse at sharp boundary convergences (like pyramid tips) by 
    anchoring the internal super-cell geometry independently of the skin layer.
    """
    import trimesh
    from scipy.spatial import cKDTree
    
    new_kag_nodes = kag_nodes.copy()
    
    if len(kept_elements) == 0:
        return new_kag_nodes
        
    # 1. Flatten all involved FCC corner coordinates into a single array
    involved_fcc_coords = []
    for coords in kept_tet_coords:
        involved_fcc_coords.extend(coords)
    involved_fcc_coords = np.array(involved_fcc_coords)
    
    # Extract globally unique FCC coordinates using geometric hashing
    pts_rounded = np.round(involved_fcc_coords, 5)
    _, unique_indices, inverse_map = np.unique(pts_rounded, axis=0, return_index=True, return_inverse=True)
    unique_fcc_pts = involved_fcc_coords[unique_indices]
    
    # 2. Determine which unique FCC coordinates fall OUTSIDE the pyramid
    try:
        closest_pts, distances, _ = trimesh.proximity.closest_point(stl_mesh, unique_fcc_pts)
    except Exception:
        mesh_tree = cKDTree(stl_mesh.vertices)
        distances, vertex_indices = mesh_tree.query(unique_fcc_pts)
        closest_pts = stl_mesh.vertices[vertex_indices]
        
    try:
        # If it doesn't contain the point, it is OUTSIDE
        contains_mask = stl_mesh.contains(unique_fcc_pts)
    except Exception:
        # Fallback pseudo-SDF using normal dots, but assumes watertight
        contains_mask = np.ones(len(unique_fcc_pts), dtype=bool) # fallback fail-safe

    is_exterior = ~contains_mask
    
    # Deform only exterior super-cell corners directly onto the CAD profile
    deformed_unique_pts = unique_fcc_pts.copy()
    deformed_unique_pts[is_exterior] = closest_pts[is_exterior]
    
    # 3. Rebuild Kagome centroids per element based on the deformed parent
    for local_idx, k_indices in enumerate(kept_elements):
        # Inverse map offsets for its 4 FCC corners
        base_offset = local_idx * 4
        
        # Deformed coordinates for this tet's 4 corners!
        v0 = deformed_unique_pts[inverse_map[base_offset + 0]]
        v1 = deformed_unique_pts[inverse_map[base_offset + 1]]
        v2 = deformed_unique_pts[inverse_map[base_offset + 2]]
        v3 = deformed_unique_pts[inverse_map[base_offset + 3]]
        
        # Overwrite global Kagome node coordinates dynamically using the true geometric planes
        new_kag_nodes[k_indices[0]] = (v1 + v2 + v3) / 3.0
        new_kag_nodes[k_indices[1]] = (v0 + v1 + v2) / 3.0
        new_kag_nodes[k_indices[2]] = (v0 + v1 + v3) / 3.0
        new_kag_nodes[k_indices[3]] = (v0 + v2 + v3) / 3.0

    return new_kag_nodes

def apply_voronoi_dual(nodes, struts):
    """
    Transforms an explicit base graph into a pure Voronoi Foam network.
    Calculates the exact Wigner-Seitz cells directly from the generated space seeds.
    Because `generate_cartesian_nodes` naturally outputs padded bounds, the central 
    Voronoi cells are definitively closed and perfect prior to the SDF watertight clip.
    """
    if len(nodes) < 4:
        return np.array([]), np.array([])
        
    from scipy.spatial import Voronoi
    vor = Voronoi(nodes)
    
    all_vertices = np.asarray(vor.vertices, dtype=np.float64)
    struts_set = set()
    
    for ridge in vor.ridge_vertices:
        if -1 in ridge: # omit ridges that go to infinity
            continue
        for i in range(len(ridge)):
            a, b = ridge[i], ridge[(i + 1) % len(ridge)]
            if a != b and a >= 0 and b >= 0:
                struts_set.add((min(a, b), max(a, b)))
                
    struts_raw = np.array(sorted(struts_set), dtype=np.int32)
    if struts_raw.shape[0] == 0:
        return all_vertices, np.empty((0, 2), dtype=np.int32)
        
    # Cull unused boundary-infinity vertices
    used_vertices = np.unique(struts_raw.ravel())
    new_nodes = all_vertices[used_vertices]
    
    mapping = np.full(all_vertices.shape[0], -1, dtype=np.int32)
    mapping[used_vertices] = np.arange(len(used_vertices))
    
    new_struts = mapping[struts_raw]
    return new_nodes, new_struts


def apply_strict_clipping(nodes, struts, boundary_mesh):
    """Deletes any struts that cross the boundary mesh using SDF acceleration."""
    if len(struts) == 0: return nodes, struts
    
    from graphite.geometry.masking import voxelize_mesh_and_edt
    # High-speed O(N) containment via SDF
    resolution = min(2.0, max(0.5, np.max(boundary_mesh.extents) / 50.0))
    _, _, _, cad_sdf, padded_min_bound, _, nx, ny, nz = voxelize_mesh_and_edt(boundary_mesh, resolution)
    
    # Calculate sdf for all nodes
    ind = np.round((nodes - padded_min_bound) / resolution).astype(int)
    ind[:, 0] = np.clip(ind[:, 0], 0, nx - 1)
    ind[:, 1] = np.clip(ind[:, 1], 0, ny - 1)
    ind[:, 2] = np.clip(ind[:, 2], 0, nz - 1)
    
    is_inside = cad_sdf[ind[:, 0], ind[:, 1], ind[:, 2]] <= 0.0
    
    # Keep only struts where BOTH endpoints are inside
    keep_mask = is_inside[struts[:, 0]] & is_inside[struts[:, 1]]
    valid_struts = struts[keep_mask]
    if len(valid_struts) > 0:
        used_nodes = np.unique(valid_struts.ravel())
        new_nodes = nodes[used_nodes]
        old_to_new = np.full(len(nodes), -1, dtype=np.int32)
        old_to_new[used_nodes] = np.arange(len(used_nodes))
        return new_nodes, old_to_new[valid_struts]
    return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)

def apply_topological_snapping(nodes, struts, stl_mesh, mode=ClippingMode.SNAP):
    """
    Refines a global lattice grid to conform to an STL boundary.
    Supports different ClippingModes:
    - SNAP: Kept if midpoint is inside. External nodes are snapped to surface.
    - STRICT: Kept only if BOTH nodes are inside. No snapping.
    - OVERFLOW: Kept if at least one node is inside. (TODO)
    """
    # Guard: handle empty or malformed inputs from upstream transforms
    if len(struts) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    nodes = np.atleast_2d(np.asarray(nodes, dtype=np.float64))
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    struts = np.asarray(struts, dtype=np.int32)
    if struts.ndim != 2 or struts.shape[1] != 2 or struts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
    
    resolution = min(2.0, max(0.5, np.max(stl_mesh.extents) / 100.0))
    get_sdfs = build_edt_sdf_sampler(stl_mesh, resolution)

    node_sdfs = get_sdfs(nodes)
    node_is_inside = node_sdfs <= 1e-4 # epsilon for boundary tolerance

    if mode == ClippingMode.STRICT:
        # Kept ONLY if both nodes are inside
        s_starts_inside = node_is_inside[struts[:, 0]]
        s_ends_inside = node_is_inside[struts[:, 1]]
        strut_is_kept = s_starts_inside & s_ends_inside
    else:
        # SNAP Mode (Default): Kept if midpoint is inside
        midpoints = (nodes[struts[:, 0]] + nodes[struts[:, 1]]) / 2.0
        midpoint_sdfs = get_sdfs(midpoints)
        strut_is_kept = midpoint_sdfs <= 0.0
    
    kept_struts_old = struts[strut_is_kept]
    if len(kept_struts_old) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)
        
    used_old_nodes = np.unique(kept_struts_old.ravel())
    
    # Snapping logic for SNAP mode
    if mode == ClippingMode.SNAP:
        nodes_to_snap = []
        for node_idx in used_old_nodes:
            if not node_is_inside[node_idx]:
                nodes_to_snap.append(node_idx)
                
        if len(nodes_to_snap) > 0:
            pts_to_snap = nodes[nodes_to_snap]
            snapped_pts, _ = closest_points_with_fallback(stl_mesh, pts_to_snap)
            for i, idx in enumerate(nodes_to_snap):
                nodes[idx] = snapped_pts[i]
            
    keep_old = np.zeros(len(struts), dtype=bool)
    keep_old[strut_is_kept] = True
    return compress_graph_to_kept_struts(nodes, struts, keep_old)
    
def generate_supercell_surface_skin(elements, face_to_node, target_element_size=None):
    """
    Generates a Surface Dual / Cage (skin) from the boundary of an element set.
    """
    from graphite.explicit.topology_module import generate_surface_dual_cage
    
    # 1. Extract boundary faces
    boundary_faces = extract_boundary_faces(elements)
    if len(boundary_faces) == 0:
        return np.empty((0, 2), dtype=np.int32)
        
    # 2. Build face_to_node_id array
    # We map each triangle (triplet of indices) to its corresponding Kagome node ID
    face_to_node_id = []
    valid_faces = []
    
    for tri in boundary_faces:
        face_key = tuple(sorted(tri.tolist()))
        if face_key in face_to_node:
            face_to_node_id.append(face_to_node[face_key])
            valid_faces.append(tri)
        else:
            # This should not happen if elements and face_to_node are consistent
            pass
            
    if len(valid_faces) == 0:
        return np.empty((0, 2), dtype=np.int32)
        
    valid_faces = np.array(valid_faces, dtype=np.int32)
    face_to_node_id = np.array(face_to_node_id, dtype=np.int32)
    
    # 3. Generate Cage via topology_module
    # generate_surface_dual_cage(surface_faces, face_to_node_id, ...)
    cage_struts = generate_surface_dual_cage(
        surface_faces=valid_faces,
        face_to_node_id=face_to_node_id,
        target_element_size=target_element_size
    )
    
    return cage_struts

def get_supercell_cutoff_distance(element_size, cell_type):
    """
    Returns the nearest neighbor crystallographic cutoff radius for specific cells.
    Uses element_size scaled to the effective tiling period L.
    """
    L = get_effective_tiling_period(element_size, cell_type)
    eps = 1e-4
    if cell_type == "Simple Cubic":
        return L + eps 
    elif cell_type in ("BCC", "Body-Centered Cubic"):
        return L * (3**0.5) / 2.0 + eps
    elif cell_type in ("FCC", "Face-Centered Cubic"):
        return L * (2**0.5) / 2.0 + eps
    elif "Truncated Octa" in cell_type:
        return L * 0.51 + eps # Scaled to catch triangles for Kagome
    elif "A15" in cell_type:
        return L * 0.65 + eps
    elif "Rhombic" in cell_type:
        return L * 0.75 + eps
    elif "Bitruncated" in cell_type:
        return L * 0.65 + eps
    return L + eps
