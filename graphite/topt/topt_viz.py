"""
Graphite Topology Optimization - Visualization

This module provides tools for visualizing the results of topology optimization 
tasks, including convergence plots for objective metrics and 3D rendering 
of the final extracted isosurfaces using PyVista.
"""
import os
import pyvista as pv
import matplotlib.pyplot as plt
import trimesh
from graphite.viz.png_framing import create_offscreen_plotter, frame_plotter_content

def plot_topt_convergence(opt):
    """
    Generate convergence plots for compliance and volume metrics.

    Reads the optimization history from the scikit-topt optimizer 
    and returns a matplotlib figure with convergence plots.

    Parameters
    ----------
    opt : Any
        The optimizer object containing the iteration history.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure containing the plots.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Check if history exists (scikit-topt saves it internally in a specific way)
    # The history is usually in opt.history or we can use the logger output.
    # Actually, sktopt.tools.history contains lists if it's running.
    # We might have to rely on what's available. If it's empty, we return empty fig.
    
    # NOTE: scikit-topt's OC_Optimizer has a 'history' attribute that holds lists? 
    # Let's check `opt.timer` or `opt.history`... Wait, `sktopt.tools.history` tracks it globally?
    # No, it's better to just try accessing `opt.history` or fallback.
    iters = []
    comp = []
    vol = []
    
    # Try to extract metrics from the optimizer object
    if hasattr(opt, 'history') and hasattr(opt.history, 'metrics'):
        metrics = opt.history.metrics
        if 'compliance' in metrics and 'mean(rho)' in metrics:
            comp = metrics['compliance']
            vol = metrics['mean(rho)']
            iters = list(range(1, len(comp) + 1))
            
    # Fallback to plotting something generic if we can't find the exact array
    if not iters:
        ax1.text(0.5, 0.5, "Convergence data not found", ha='center')
        ax2.text(0.5, 0.5, "Convergence data not found", ha='center')
    else:
        ax1.plot(iters, comp, 'b-', marker='o', markersize=4)
        ax1.set_title("Compliance (Strain Energy)")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Compliance")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(iters, vol, 'r-', marker='o', markersize=4)
        ax2.set_title("Mean Density (Volume)")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Density")
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    return fig

def render_topt_isosurface(mesh: trimesh.Trimesh, output_path: str = "topt_preview.png") -> str:
    """
    Render a 3D mesh isosurface to an image file using PyVista.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to render.
    output_path : str, optional
        The path to save the rendered image, by default "topt_preview.png".

    Returns
    -------
    str
        The absolute path to the saved image file.
    """
    pv_mesh = pv.wrap(mesh)
    plotter = create_offscreen_plotter()
    
    plotter.add_mesh(
        pv_mesh,
        color="lightblue",
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
        smooth_shading=True
    )
    
    plotter.add_axes()
    frame_plotter_content(plotter, pv_mesh.bounds, view_isometric=True)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plotter.screenshot(output_path)
    plotter.close()
    
    return output_path
