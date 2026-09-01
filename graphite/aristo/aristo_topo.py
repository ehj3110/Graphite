"""
Basic Topology Optimization (Single-Pass Thresholding) for Aristo FEA.

Extracts a printable, smoothed 3D mesh (STL) consisting only of regions 
above a certain stress percentile limit, while preserving constrained/loaded non-design spaces.
"""
from __future__ import annotations
import numpy as np
import pyvista as pv
from graphite.aristo.aristo_solver import AristoResult
import os

def extract_topology(
    result: AristoResult, 
    output_path: str,
    percentile: float = 50.0,
    smoothing_iters: int = 50,
    pass_band: float = 0.05
) -> str:
    """
    Extract a basic topology-optimized shape from a static FEA result.
    
    Parameters
    ----------
    result : AristoResult
        The direct result output from `run_aristo`.
    output_path : str
        Target file path for the STL export.
    percentile : float
        The stress percentile threshold. Only material carrying stress above 
        this limit is kept (unless it is a mandatory boundary condition).
    smoothing_iters : int
        Number of Taubin smoothing iterations to apply to the jagged cut volume.
    pass_band : float
        Taubin pass-band factor for geometric preservation.
        
    Returns
    -------
    str : The output filepath of the generated `.stl`.
    """
    print(f"[Aristo TO] Extracting topology at {percentile}th percentile...")
    
    # 1. Reconstruct PyVista UnstructuredGrid
    nodes = result.fea_nodes
    elements = result.fea_elements
    m, n_node_per_elem = elements.shape
    
    cells = np.column_stack([np.full(m, n_node_per_elem), elements]).ravel()
    cell_types = np.full(m, 10, dtype=np.uint8) # 10 = VTK_TETRA
    
    grid = pv.UnstructuredGrid(cells, cell_types, nodes)
    
    # 2. Nodal or element-normalized stress on grid points
    scalar_title = "Normalized Von Mises Stress"
    if (
        result.stress_field_mode == "nodal_averaged"
        and result.von_mises_nodal_norm.size == nodes.shape[0]
    ):
        grid.point_data[scalar_title] = result.von_mises_nodal_norm
    else:
        grid.cell_data[scalar_title] = result.von_mises
        grid = grid.cell_data_to_point_data()
    
    # 3. Non-Design Space Preservation
    # Find the nodes that MUST be preserved
    protected_nodes = np.unique(np.concatenate([result.fixed_nodes, result.load_nodes]))
    
    # We artificially boost the stress of protected nodes to infinity.
    # This guarantees the threshold filter will never drop elements connected to them.
    point_scalars = grid.point_data[scalar_title].copy()
    point_scalars[protected_nodes] = np.inf
    grid.point_data[scalar_title] = point_scalars
    
    # 4. Volumetric Thresholding
    # Calculate the exact cut value based on the ORIGINAL data (ignoring the infs)
    raw_point_data = grid.point_data[scalar_title]
    safe_data = raw_point_data[~np.isinf(raw_point_data)]
    cut_value = float(np.percentile(safe_data, percentile))
    
    print(f"[Aristo TO] Thresholding > {cut_value:.4f}")
    # filter drops any cell where ALL points are below `value` (if all_scalars=False, which is default)
    # wait, PyVista threshold usually keeps cells if ANY point is in range (depending on all_scalars).
    # all_scalars=False ensures that if a protected node is attached to a cell, that cell is fully saved.
    topo_vol = grid.threshold(value=cut_value, scalars=scalar_title, all_scalars=False)
    
    if topo_vol.n_points == 0 or topo_vol.n_cells == 0:
         raise RuntimeError("Topology extraction resulted in completely empty mesh!")
         
    # 5. Island Connectivity Filtering
    print("[Aristo TO] Running spatial connectivity analysis...")
    topo_vol = topo_vol.connectivity("largest")
    
    # 6. Extract Outer Surface and Smooth
    print("[Aristo TO] Extracting and organo-smoothing geometric skin...")
    topo_surf = topo_vol.extract_surface()
    
    if topo_surf.n_faces > 0:
        topo_surf = topo_surf.smooth_taubin(n_iter=smoothing_iters, pass_band=pass_band)
        # Ensure normals are consistent for STL export
        topo_surf.compute_normals(inplace=True, consistent_normals=True)
    
    # 7. Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    topo_surf.save(output_path)
    print(f"[Aristo TO] Topology optimized STL saved to: {output_path}")
    
    return output_path
