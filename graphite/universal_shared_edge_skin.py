# -*- coding: utf-8 -*-
"""
universal_shared_edge_skin.py

Universal, lattice-agnostic 2D boundary skin generator using the
Valency-Snapped Kagome Dual approach.

Step 1: Filter tetrahedra (centroids >= target_z or <= target_z).
Step 2: Extract exposed faces (belong to exactly 1 tetrahedron) and centroids.
Step 3: Apply Valency-Gated Planar Snap (Snap band = 0.15 * cell_size).
        - 'default': Inside conformed if valency <= 3, Outside always conformed.
        - 'robust': Inside always conformed, Outside conformed if valency <= 3.
Step 4: Wire conformed nodes into Red (Internal conformed struts) and Cyan (Skin struts).
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Add workspace root to sys.path so we can import packages if run directly
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# ---------------------------------------------------------------------------
# Snapping Helper
# ---------------------------------------------------------------------------

_SNAP_PLANES = np.array([0.0, 0.25, 0.5, 0.75])

def _snap_frac(frac_z: float) -> float:
    """Snap an arbitrary fractional depth to the nearest quarter-cell plane."""
    dists = np.abs(_SNAP_PLANES - frac_z)
    dist_to_one = abs(frac_z - 1.0)
    if dist_to_one < dists.min():
        return 0.0
    return float(_SNAP_PLANES[np.argmin(dists)])

# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

def generate_shared_edge_skin(
    tet_mesh: np.ndarray | tuple[np.ndarray, np.ndarray],
    target_z: float,
    cell_size: float,
    cut_direction: str = 'above',
    caging_mode: str = 'default'
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """
    Generates the universal 2D conformed shared-edge skin for a tetrahedral mesh.

    Parameters
    ----------
    tet_mesh : np.ndarray of shape (M, 4, 3) or tuple (vertices, tets)
        The input tetrahedral mesh representation.
    target_z : float
        The physical Z-height of the desired boundary plane (will be snapped).
    cell_size : float
        The characteristic unit cell size.
    cut_direction : str, default 'above'
        'above' to keep tets with centroids >= target_z.
        'below' to keep tets with centroids <= target_z.
    caging_mode : str, default 'default'
        'default': Inside conformed if valency <= 3, Outside always conformed.
        'robust': Inside always conformed, Outside conformed if valency <= 3.

    Returns
    -------
    red_segments : list of [[x0, y0], [x1, y1]]
        Internal conformed struts where both endpoints are conformed.
    cyan_segments : list of [[x0, y0], [x1, y1]]
        Skin struts representing Weaire-Phelan cell boundaries.
    """
    # Parse input mesh representation
    if isinstance(tet_mesh, tuple):
        vertices, tets = tet_mesh
        tets_phys = vertices[tets]
    else:
        tets_phys = np.asarray(tet_mesh, dtype=np.float64)

    # Calculate fractional height, snap to nearest quarter-plane, and calculate snapped Z
    frac_z = (target_z / cell_size) % 1.0
    snapped_frac = _snap_frac(frac_z)
    snapped_z = snapped_frac * cell_size + (target_z - frac_z * cell_size)

    if abs(snapped_frac - frac_z) > 1e-9:
        print(f"[Router] WARNING: Target Z={target_z:.2f} mm snapped to nearest control plane: Z={snapped_z:.2f} mm")
    else:
        print(f"[Router] Target Z={target_z:.2f} mm is already on a control plane: Z={snapped_z:.2f} mm")

    # Step 1: Filter tetrahedra based on cut direction
    centroids = tets_phys.mean(axis=1)
    if cut_direction == 'above':
        surviving_tets = tets_phys[centroids[:, 2] >= snapped_z - 1e-5]
    elif cut_direction == 'below':
        surviving_tets = tets_phys[centroids[:, 2] <= snapped_z + 1e-5]
    else:
        raise ValueError(f"Unknown cut_direction: {cut_direction}")
        
    print(f"[Skin] Input tets: {len(tets_phys)}, Surviving tets (centroid {cut_direction} {snapped_z:.3f} mm): {len(surviving_tets)}")

    if len(surviving_tets) == 0:
        return [], []

    # Coordinate precision rounding key
    def _pt_key(p):
        return tuple(np.round(p, 5))

    # Step 2: Extract exposed faces and face-centroids
    face_counts = defaultdict(int)
    face_to_centroid = {}
    face_to_verts = {}

    node_coords = []
    coord_to_idx = {}
    strut_set = set()

    def get_or_add(coord):
        key = _pt_key(coord)
        if key not in coord_to_idx:
            coord_to_idx[key] = len(node_coords)
            node_coords.append(coord.copy())
        return coord_to_idx[key]

    FACE_TRIPLETS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    # Generate node positions and struts for surviving tets
    for tet in surviving_tets:
        face_kagome_idx = []
        for fv in FACE_TRIPLETS:
            verts = tet[list(fv)]
            centroid = verts.mean(axis=0)
            n_idx = get_or_add(centroid)
            face_kagome_idx.append(n_idx)

            # Record face representation
            fkey = frozenset(_pt_key(v) for v in verts)
            face_counts[fkey] += 1
            face_to_centroid[fkey] = centroid
            face_to_verts[fkey] = verts

        # Add tetrahedral face connections (Kagome struts)
        for a, b in combinations(range(4), 2):
            u, v = face_kagome_idx[a], face_kagome_idx[b]
            strut_set.add((min(u, v), max(u, v)))

    nodes_3d = np.array(node_coords)
    struts = np.array(sorted(strut_set))

    # Calculate 3D valency (degrees) of Kagome nodes in the surviving network
    degrees = np.zeros(len(nodes_3d), dtype=np.int64)
    for u, v in struts:
        degrees[u] += 1
        degrees[v] += 1

    # Step 3: Valency-Gated Planar Snap (Snap band = 0.15 * cell_size)
    snap_band = 0.15 * cell_size
    conformed_nodes_map = {}  # Map from node index to conformed 2D coordinate

    for idx, pt in enumerate(nodes_3d):
        x, y, z = pt[0], pt[1], pt[2]
        valency = degrees[idx]

        conformed = False
        if caging_mode == 'robust':
            # Config 1: Inside always, Outside if valency <= 3
            if cut_direction == 'above':
                if snapped_z <= z <= snapped_z + snap_band + 1e-5:
                    conformed = True
                elif snapped_z - snap_band - 1e-5 <= z < snapped_z:
                    conformed = (valency <= 3)
            elif cut_direction == 'below':
                if snapped_z - snap_band - 1e-5 <= z <= snapped_z:
                    conformed = True
                elif snapped_z < z <= snapped_z + snap_band + 1e-5:
                    conformed = (valency <= 3)
        elif caging_mode == 'default':
            # Config 3: Inside if valency <= 3, Outside always
            if cut_direction == 'above':
                if snapped_z <= z <= snapped_z + snap_band + 1e-5:
                    conformed = (valency <= 3)
                elif snapped_z - snap_band - 1e-5 <= z < snapped_z:
                    conformed = True
            elif cut_direction == 'below':
                if snapped_z - snap_band - 1e-5 <= z <= snapped_z:
                    conformed = (valency <= 3)
                elif snapped_z < z <= snapped_z + snap_band + 1e-5:
                    conformed = True
        else:
            raise ValueError(f"Unknown caging_mode: {caging_mode}")

        if conformed:
            conformed_nodes_map[idx] = np.array([x, y])

    # Step 4: Network Wiring
    # 4.1 Internal Struts (Red): both endpoints conformed
    red_segments = []
    for u, v in struts:
        if u in conformed_nodes_map and v in conformed_nodes_map:
            red_segments.append([conformed_nodes_map[u].tolist(), conformed_nodes_map[v].tolist()])

    # 4.2 Skin Struts (Cyan): shared-edge dual of conformed exposed faces
    exposed_faces = [fk for fk, cnt in face_counts.items() if cnt == 1]
    
    # Map edges of exposed faces to face keys, only for conformed face centroids
    edge_to_faces = defaultdict(list)
    for fkey in exposed_faces:
        centroid = face_to_centroid[fkey]
        c_idx = coord_to_idx[_pt_key(centroid)]
        
        # Only keep exposed face if its centroid node was conformed
        if c_idx in conformed_nodes_map:
            verts_list = sorted(list(fkey))
            for a, b in combinations(verts_list, 2):
                edge_to_faces[(a, b)].append(fkey)

    # Identify edges shared by exactly two conformed exposed faces
    cage_struts_set = set()
    for edge, faces in edge_to_faces.items():
        if len(faces) == 2:
            fa, fb = faces[0], faces[1]
            # Stable lexicographical sorting of tuples of float-coordinates
            ta = tuple(sorted(list(fa)))
            tb = tuple(sorted(list(fb)))
            if ta < tb:
                cage_struts_set.add((fa, fb))
            else:
                cage_struts_set.add((fb, fa))

    cyan_segments = []
    for fa, fb in cage_struts_set:
        c_a = face_to_centroid[fa]
        c_b = face_to_centroid[fb]
        idx_a = coord_to_idx[_pt_key(c_a)]
        idx_b = coord_to_idx[_pt_key(c_b)]
        
        p_a = conformed_nodes_map[idx_a]
        p_b = conformed_nodes_map[idx_b]
        cyan_segments.append([p_a.tolist(), p_b.tolist()])

    print(f"[Skin] Wiring complete. Internal (Red) struts: {len(red_segments)}, Skin (Cyan) struts: {len(cyan_segments)}")
    return red_segments, cyan_segments

# ---------------------------------------------------------------------------
# Matplotlib Plotting and Torture Test
# ---------------------------------------------------------------------------

# A15 crystallographic basis and cutoff
A15_BASIS = np.array([
    [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
    [0.25, 0.5, 0.0], [0.75, 0.5, 0.0],
    [0.0, 0.25, 0.5], [0.0, 0.75, 0.5],
    [0.5, 0.0, 0.25], [0.5, 0.0, 0.75],
], dtype=np.float64)

BOND_CUTOFF  = 0.62

def main():
    """
    Run the 4-panel torture test on a standard 2x2x1 A15 block (cell_size = 5.0).
    test_heights = [0.0, 1.5, 3.0, 4.375]
    Plots 4 subplots showing the conformed surface duals under 'default' mode.
    """
    from graphite.explicit.a15_kagome import trilinear_warp
    
    cell_size = 5.0
    display_cells = 2
    display_x = display_cells * cell_size
    display_y = display_cells * cell_size

    print("=== Running 4-Panel Torture Test for Universal Shared-Edge Skin ===")
    
    # 1. Generate canonical A15 basis and cliques
    pts_list = []
    for i in (-1, 0, 1, 2):
        for j in (-1, 0, 1, 2):
            for k in (-1, 0, 1, 2):
                for b in A15_BASIS:
                    pts_list.append(b + np.array([i, j, k], dtype=np.float64))
    pts = np.unique(np.round(np.vstack(pts_list), 9), axis=0)

    dist = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    adj = {i: set() for i in range(len(pts))}
    for u, v in zip(*np.where((dist > 1e-9) & (dist <= BOND_CUTOFF))):
        if u < v:
            adj[u].add(v); adj[v].add(u)

    cliques = []
    for u in range(len(pts)):
        for v in adj[u]:
            if v <= u: continue
            uv = adj[u] & adj[v]
            for w in uv:
                if w <= v: continue
                for x in (uv & adj[w]):
                    if x > w:
                        cliques.append((u, v, w, x))

    arr = np.array(cliques, dtype=np.int64)
    centroids = pts[arr].mean(axis=1)
    inside = np.all((centroids >= -1e-9) & (centroids <= 1.0 + 1e-9), axis=1)
    cliques_inside = arr[inside]

    # Generate cell coordinates (2x2x2 elements surrounding the cut plane)
    hex_corners = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=np.float64)

    seen_tets = set()
    unique_tets = []
    def _pt_key(p): return tuple(np.round(p, 4))
    def _tet_key(t): return tuple(sorted([_pt_key(p) for p in t]))

    for ix in range(-1, display_cells + 1):
        for iy in range(-1, display_cells + 1):
            for iz in range(-1, 3):
                off = np.array([ix, iy, iz], dtype=np.float64) * cell_size
                corners = off + cell_size * hex_corners
                phys = trilinear_warp(pts, corners)
                for cl in cliques_inside:
                    tet_phys = phys[cl]
                    tk = _tet_key(tet_phys)
                    if tk not in seen_tets:
                        seen_tets.add(tk)
                        unique_tets.append(tet_phys)

    # 2. Setup plotting layout (1x4 grid)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)
    fig.patch.set_facecolor("#0f0f1a")

    test_heights = [0.0, 1.5, 3.0, 4.375]

    for ax, tz in zip(axes, test_heights):
        # Calculate snapped physical Z-height to show in title
        frac_z = (tz / cell_size) % 1.0
        snapped_frac = _snap_frac(frac_z)
        snapped_z = snapped_frac * cell_size + (tz - frac_z * cell_size)

        print(f"\n--- Testing Z = {tz:.3f} mm (snaps to {snapped_z:.3f} mm) ---")
        
        red_segments, cyan_segments = generate_shared_edge_skin(
            tet_mesh=unique_tets,
            target_z=tz,
            cell_size=cell_size,
            cut_direction='above',
            caging_mode='default'
        )

        # Deduplicate conformed coordinates for node plotting
        conformed_coords = set()
        for seg in red_segments:
            conformed_coords.add(tuple(np.round(seg[0], 4)))
            conformed_coords.add(tuple(np.round(seg[1], 4)))
        for seg in cyan_segments:
            conformed_coords.add(tuple(np.round(seg[0], 4)))
            conformed_coords.add(tuple(np.round(seg[1], 4)))

        display_nodes = []
        for coord in conformed_coords:
            if (-1e-4 <= coord[0] <= display_x + 1e-4) and (-1e-4 <= coord[1] <= display_y + 1e-4):
                display_nodes.append(coord)
        display_nodes = np.array(display_nodes)

        # Filter segments for plotting inside bounds
        def filter_seg(seg):
            p0, p1 = seg[0], seg[1]
            def in_box(p): return (-1e-4 <= p[0] <= display_x + 1e-4) and (-1e-4 <= p[1] <= display_y + 1e-4)
            return in_box(p0) or in_box(p1)

        red_display = [s for s in red_segments if filter_seg(s)]
        cyan_display = [s for s in cyan_segments if filter_seg(s)]

        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

        # Draw conformed struts
        lc_nat = LineCollection(red_display, colors="#e57373", linewidths=1.8, alpha=0.9, zorder=2)
        ax.add_collection(lc_nat)
        
        lc_cage = LineCollection(cyan_display, colors="#4fc3f7", linewidths=1.8, alpha=0.9, zorder=3)
        ax.add_collection(lc_cage)

        # Draw conformed nodes
        if len(display_nodes):
            ax.scatter(
                display_nodes[:, 0], display_nodes[:, 1],
                s=20, c="#e57373", edgecolors="#883b3b",
                linewidths=0.6, zorder=4, alpha=0.95
            )

        # Cell grid lines
        for x in np.arange(0, display_x + 1e-6, cell_size):
            ax.axvline(x, color="#555577", linewidth=0.6, linestyle=":", zorder=1)
        for y in np.arange(0, display_y + 1e-6, cell_size):
            ax.axhline(y, color="#555577", linewidth=0.6, linestyle=":", zorder=1)

        ax.set_xlim(-0.5, display_x + 0.5)
        ax.set_ylim(-0.5, display_y + 0.5)
        
        ax.set_title(
            f"Input Z = {tz:.3f} mm\n"
            f"Snapped Z = {snapped_z:.3f} mm\n"
            f"(frac={snapped_frac:.2f})",
            color="white", fontsize=10, pad=8
        )

    plt.suptitle(
        "Torture Test -- Universal Shared-Edge Skin Generator (Default Mode)\n"
        "Lattice-Agnostic Snap-Routing at Quarter-Cell Control Heights (A15)",
        color="white", fontsize=14, y=1.03
    )

    plt.tight_layout()

    # Save to workspace and brain directory
    BRAIN_DIR = Path("C:/Users/ehunt/.gemini/antigravity/brain/d4e04f7b-5149-4db6-8e8d-c0fff73a1834")
    fname = "Torture_Test_Universal_Skin.png"
    plt.savefig(str(WORKSPACE_ROOT / fname), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.savefig(str(BRAIN_DIR / fname), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"\nSaved torture test plot {fname} successfully.")

if __name__ == "__main__":
    main()
